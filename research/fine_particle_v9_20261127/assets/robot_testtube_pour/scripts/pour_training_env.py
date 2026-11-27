"""Live PBF environment for goal-conditioned pouring policy experiments.

This module does not create Kit, USD, a robot, a renderer, or a video clock.
The training action drives the same kinematic tube collider used by the GUI.
Every action is followed by the unmodified liquid solver at physical 60 Hz.
Reported mL are particle-mass simulation truth, not a camera measurement.
"""

from contextlib import nullcontext
import gzip
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import time

import numpy as np


DIRECTORY = Path(__file__).resolve().parent


def _local_module(name):
    path = DIRECTORY / (name + ".py")
    spec = importlib.util.spec_from_file_location("pour_training_" + name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), module.__dict__)
    return module


def nearest_attainable(target_ml, particle_volume_ml):
    """Nearest integer-particle target; never change the requested goal."""
    target, quantum = float(target_ml), float(particle_volume_ml)
    if not math.isfinite(target) or not math.isfinite(quantum) or target < 0 or quantum <= 0:
        raise ValueError("Target must be nonnegative and particle volume positive.")
    count = int(math.floor(target / quantum + 0.5))
    volume = count * quantum
    return {"nearest_particle_count": count, "nearest_attainable_ml": volume,
            "quantization_error_ml": volume - target,
            "minimum_abs_quantization_error_ml": abs(volume - target)}


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value).__name__)


def _write_line(stream, value):
    if stream is not None:
        stream.write(json.dumps(value, separators=(",", ":"), ensure_ascii=False,
                                allow_nan=False, default=_json_default) + "\n")


class PourTrainingEnv:
    """One reusable fixed-particle environment with independently reset episodes.

    Omitted modules are loaded from this directory. A supplied ``geometry`` is
    already on its intended physical motion time scale and is never modified.
    The default private geometry uses the GUI's physical time scale of 2.0.
    ``progress`` in run_episode is a callback accepting one event dictionary.
    ``record_path`` accepts JSONL or gzip-compressed ``.jsonl.gz``.
    """

    def __init__(self, geometry=None, liquid=None, control=None, *,
                 spacing=0.0035, motion_time_scale=2.0, density=1000.0,
                 viscosity=0.001, max_seconds=35.0, tolerance_ml=0.25,
                 surface_voxel=0.0025):
        self.geometry = geometry if geometry is not None else _local_module("geometry")
        self.liquid_module = liquid if liquid is not None else _local_module("liquid")
        self.control_module = control if control is not None else _local_module("target_control")
        self.spacing, self.density = float(spacing), float(density)
        self.base_viscosity = float(viscosity)
        self.max_seconds, self.tolerance_ml = float(max_seconds), float(tolerance_ml)
        self.surface_voxel = float(surface_voxel)
        if not all(math.isfinite(v) and v > 0 for v in
                   (self.spacing, self.density, self.max_seconds, self.tolerance_ml,
                    self.surface_voxel, float(motion_time_scale))):
            raise ValueError("Environment dimensions, time and tolerance must be positive.")
        if not math.isfinite(self.base_viscosity) or self.base_viscosity < 0:
            raise ValueError("Viscosity must be finite and nonnegative.")
        if geometry is None:
            self.geometry.DURATION *= float(motion_time_scale)
            self.geometry.MOTION_KNOTS = tuple(
                (float(t) * float(motion_time_scale), point, angle)
                for t, point, angle in self.geometry.MOTION_KNOTS)
        self.dt = 1.0 / 60.0
        self.clean_points, self.labels = self.geometry.initial_water_points(self.spacing)
        self.clean_points = np.asarray(self.clean_points, dtype=np.float32).copy()
        self.labels = np.asarray(self.labels, dtype=np.int32).copy()
        self.particle_volume_ml = self.spacing ** 3 * 1.0e6
        self.source_count = int(np.count_nonzero(self.labels == 1))
        self.fluid = None
        self.last_controller = None
        self.last_result = None
        self.last_initial_points = None
        self.episodes_run = 0

    def _reset(self, target_ml, seed, policy, randomize):
        rng = np.random.default_rng(int(seed))
        jitter = (rng.uniform(-0.03, 0.03, self.clean_points.shape) * self.spacing
                  if randomize else np.zeros_like(self.clean_points))
        initial = (self.clean_points.astype(np.float64) + jitter).astype(np.float32)
        viscosity_factor = float(rng.uniform(0.9, 1.1)) if randomize else 1.0
        controller = self.control_module.TargetPourController(
            self.geometry, target_ml=float(target_ml), spacing=self.spacing, policy=policy)
        if self.fluid is None:
            self.fluid = self.liquid_module.PourFluid(
                initial, self.labels, self.geometry.rounded_testtube_mesh(),
                self.geometry.closed_beaker_mesh(), controller.tube_pose,
                spacing=self.spacing, voxel=2.0 * self.spacing, dt=self.dt, substeps=4,
                density=self.density, viscosity=self.base_viscosity * viscosity_factor,
                surface_voxel=self.surface_voxel)
        else:
            # Deliberately replace the reset position snapshot with this episode's
            # independently sampled initial condition, not the previous rollout.
            # Other physical state is cleared by the solver's normal reset path.
            self.fluid._rest["particle_q"].assign(initial)
            self.fluid._rest["particle_qd"].fill_(0.0)
            self.fluid.pose_fn = controller.tube_pose
        self.fluid._kinematic_viscosity = self.base_viscosity * viscosity_factor / self.density
        self.fluid.model.mpm.viscosity.fill_(self.base_viscosity * viscosity_factor)
        self.fluid.reset()
        self.last_controller = controller
        self.last_initial_points = initial.copy()
        self.episodes_run += 1
        return controller, initial, {"initial_jitter_max_spacing": 0.03 if randomize else 0.0,
            "initial_jitter_realized_max_m": float(np.max(np.abs(initial - self.clean_points))),
            "viscosity_factor": viscosity_factor,
            "dynamic_viscosity_pa_s": self.base_viscosity * viscosity_factor,
            "kinematic_viscosity_m2_s": self.fluid._kinematic_viscosity}

    def _measurement(self, controller, positions):
        if positions.shape != self.clean_points.shape or not np.isfinite(positions).all():
            raise RuntimeError("Particle positions changed shape or became nonfinite.")
        source = controller.source_time(self.fluid.time)
        metrics = self.geometry.classify_water(positions, self.fluid.labels, source, self.spacing)
        source_partition = sum(int(metrics[key]) for key in
                               ("tube_to_beaker", "tube_remaining", "tube_airborne", "tube_spilled"))
        total_partition = sum(int(metrics[key]) for key in ("beaker", "tube", "airborne", "spilled"))
        if source_partition != self.source_count or total_partition != len(self.clean_points):
            raise RuntimeError("Liquid containment partitions overlap or lose particles.")
        masses = np.asarray(self.fluid.model.particle_mass.numpy(), dtype=np.float64)
        if masses.shape != (len(self.clean_points),) or not np.isfinite(masses).all() or np.any(masses <= 0):
            raise RuntimeError("Particle mass array is invalid.")
        metrics.update({"physical_time_s": self.fluid.time, "source_time": source,
                        "received_ml": metrics["tube_to_beaker"] * self.particle_volume_ml,
                        "remaining_ml": metrics["tube_remaining"] * self.particle_volume_ml,
                        "inflight_ml": metrics["tube_airborne"] * self.particle_volume_ml,
                        "spilled_ml": metrics["tube_spilled"] * self.particle_volume_ml,
                        "total_mass_kg": float(np.sum(masses, dtype=np.float64)),
                        "source_partition_count": source_partition,
                        "total_partition_count": total_partition,
                        "tilt_degrees": abs(math.degrees(self.geometry.mouth_pose(source)[1]))})
        return metrics, masses

    def _record_step(self, stream, step, target_ml, metrics, status):
        observation_keys = ("target_ml", "received_ml", "inflight_ml", "flow_ml_s",
                            "tilt_deg", "released_ml", "tube_remaining_ml", "phase")
        command_keys = ("command_goal_source", "command_max_rate", "command_acceleration",
                        "command_source_velocity", "command_next_source_velocity")
        _write_line(stream, {"type": "step", "step": step,
            "physical_time_s": self.fluid.time, "physics_dt_s": self.dt,
            "source_time": metrics["source_time"],
            "observation": {key: status.get(key) for key in observation_keys},
            "action": {"policy": status.get("policy_action"),
                       **{key: status.get(key) for key in command_keys}},
            "truth": {"target_ml": target_ml, "received_ml": metrics["received_ml"],
                      "inflight_ml": metrics["inflight_ml"], "remaining_ml": metrics["remaining_ml"],
                      "spilled_ml": metrics["spilled_ml"], "mass_kg": metrics["total_mass_kg"],
                      "source_partition_count": metrics["source_partition_count"],
                      "below_table_particles": metrics["below_table"]},
            "solver_substeps": int(self.fluid.last_substeps)})

    def run_episode(self, target_ml, seed, policy=None, randomize=True,
                    record_path=None, progress=None):
        """Run live physics to settled-upright completion, failure or timeout.

        A policy receives current state through TargetPourController.act; no
        renderer, recorded particle trajectory or video speed participates.
        Candidate instability returns a failed result so a search can continue.
        Construction/configuration errors propagate rather than looking like a
        successfully evaluated but low-reward policy.
        """
        target_ml = float(target_ml)
        if target_ml not in (20.0, 30.0, 40.0):
            raise ValueError("Training targets are 20, 30 or 40 mL.")
        if int(seed) != seed or isinstance(seed, bool):
            raise ValueError("seed must be an integer.")
        started = time.perf_counter()
        controller, initial, variation = self._reset(target_ml, int(seed), policy, bool(randomize))
        positions = self.fluid.positions()
        reset_error = float(np.max(np.abs(positions - initial)))
        status = controller.update(0.0, positions, self.fluid.labels,
                                   self.fluid.state_0.particle_qd.numpy())
        metrics, initial_masses = self._measurement(controller, positions)
        initial_mass = metrics["total_mass_kg"]
        source_mass = float(initial_masses[self.labels == 1].sum(dtype=np.float64))
        expected_particle_mass = self.density * self.spacing ** 3
        mass_unit_consistent = bool(np.allclose(initial_masses, expected_particle_mass, rtol=2e-6, atol=1e-12))
        max_mass_drift = 0.0
        mass_array_unchanged = True
        max_below_table = int(metrics["below_table"])
        max_spilled_ml = float(metrics["spilled_ml"])
        max_source_rate = 0.0
        max_source_acceleration = 0.0
        max_substeps = 0
        min_substeps = 10**9
        all_finite, partition_conserved = True, True
        previous_source = metrics["source_time"]
        error_message = None
        termination_reason = "physical_time_limit"
        policy_name = status.get("policy_name", "baseline_feedback" if policy is None else type(policy).__name__)
        metadata = {"type": "episode", "target_ml": target_ml, "seed": int(seed),
            "randomized": bool(randomize), "variation": variation,
            "policy_name": policy_name, "policy_params": status.get("policy_params"),
            "physics_dt_s": self.dt, "minimum_substeps": 4,
            "particle_spacing_m": self.spacing, "particle_count": len(self.clean_points),
            "particle_volume_ml": self.particle_volume_ml, "density_kg_m3": self.density,
            "motion_knots_physical_seconds": [float(k[0]) for k in self.geometry.MOTION_KNOTS],
            "fluid_mode": "live_PBF_action_conditioned", "rendered_training": False,
            "mass_measurement": "model.particle_mass; fixed initial source labels",
            "volume_measurement": "fixed_particle_mass_divided_by_density_inside_beaker",
            "contact_force_grasp_simulated": False,
            "source_sha256": {name: hashlib.sha256((DIRECTORY / name).read_bytes()).hexdigest()
                              for name in ("liquid.py", "geometry.py", "target_control.py")}}
        path = Path(record_path) if record_path is not None else None
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            opener = gzip.open(path, "wt", encoding="utf-8") if path.suffix == ".gz" else path.open("w", encoding="utf-8")
        else:
            opener = nullcontext(None)
        with opener as stream:
            _write_line(stream, metadata)
            self._record_step(stream, 0, target_ml, metrics, status)
            if progress is not None:
                progress({"event": "start", "seed": int(seed), "target_ml": target_ml})
            for step in range(1, int(math.ceil(self.max_seconds / self.dt)) + 1):
                try:
                    self.fluid.step_frame()
                    positions = self.fluid.positions()
                    velocities = self.fluid.state_0.particle_qd.numpy()
                    if not np.isfinite(positions).all() or not np.isfinite(velocities).all():
                        all_finite = False
                        raise RuntimeError("Nonfinite particle state.")
                    status = controller.update(self.fluid.time, positions, self.fluid.labels, velocities)
                    metrics, masses = self._measurement(controller, positions)
                    max_mass_drift = max(max_mass_drift, abs(metrics["total_mass_kg"] - initial_mass))
                    if not np.array_equal(masses, initial_masses):
                        mass_array_unchanged = False
                        raise RuntimeError("The actual particle mass array changed.")
                    max_below_table = max(max_below_table, int(metrics["below_table"]))
                    max_spilled_ml = max(max_spilled_ml, float(metrics["spilled_ml"]))
                    rate = abs(metrics["source_time"] - previous_source) / self.dt
                    max_source_rate = max(max_source_rate, rate,
                        abs(float(status.get("command_source_velocity", 0.0))),
                        abs(float(status.get("command_next_source_velocity", 0.0))))
                    max_source_acceleration = max(max_source_acceleration,
                        abs(float(status.get("command_acceleration", 0.0))))
                    previous_source = metrics["source_time"]
                    max_substeps = max(max_substeps, int(self.fluid.last_substeps))
                    min_substeps = min(min_substeps, int(self.fluid.last_substeps))
                    self._record_step(stream, step, target_ml, metrics, status)
                    if status.get("done", False):
                        termination_reason = "settled_upright"
                        break
                except (RuntimeError, ValueError, FloatingPointError) as error:
                    error_message = str(error)
                    partition_conserved = "partition" not in error_message.lower()
                    termination_reason = "invalid_physics_state"
                    _write_line(stream, {"type": "failure", "step": step,
                                        "physical_time_s": self.fluid.time, "error": error_message})
                    break
                if progress is not None and step % 60 == 0:
                    progress({"event": "step", "seed": int(seed), "target_ml": target_ml,
                              "physical_time_s": self.fluid.time,
                              "received_ml": metrics["received_ml"], "phase": status.get("phase")})
            received = float(metrics["received_ml"])
            error_ml = received - target_ml
            mass_conserved = max_mass_drift <= 1e-12 and mass_unit_consistent and mass_array_unchanged
            upright = metrics["tilt_degrees"] < 0.1
            settled = metrics["inflight_ml"] <= 0.1
            valid = (error_message is None and all_finite and partition_conserved and mass_conserved
                     and reset_error < 1e-7 and max_source_rate <= 1.00001
                     and max_source_acceleration <= 3.00001 and min_substeps >= 4)
            result = {"target_ml": target_ml, "received_ml": received, "error_ml": error_ml,
                "absolute_error_ml": abs(error_ml), "tolerance_ml": self.tolerance_ml,
                "success": bool(valid and status.get("done", False) and upright and settled
                                and max_below_table == 0 and max_spilled_ml == 0.0
                                and abs(error_ml) <= self.tolerance_ml),
                "valid_physics": bool(valid), "complete": bool(status.get("done", False)),
                "termination_reason": termination_reason, "error": error_message,
                "seed": int(seed), "randomized": bool(randomize), "variation": variation,
                "policy_name": policy_name, "policy_params": status.get("policy_params"),
                "spilled_ml": metrics["spilled_ml"], "max_spilled_ml": max_spilled_ml,
                "inflight_ml": metrics["inflight_ml"], "remaining_ml": metrics["remaining_ml"],
                "final_tilt_degrees": metrics["tilt_degrees"], "upright": bool(upright),
                "physical_time_s": self.fluid.time, "simulation_seconds": self.fluid.time,
                "physical_steps": int(self.fluid._frame_index), "physics_dt_s": self.dt,
                "wall_seconds": time.perf_counter() - started,
                "particle_count": len(self.clean_points), "source_particle_count": self.source_count,
                "particle_volume_ml": self.particle_volume_ml,
                "initial_source_ml": self.source_count * self.particle_volume_ml,
                "actual_initial_source_mass_kg": source_mass,
                "actual_initial_total_mass_kg": initial_mass,
                "actual_final_total_mass_kg": metrics["total_mass_kg"],
                "maximum_mass_drift_kg": max_mass_drift, "mass_conserved": bool(mass_conserved),
                "particle_mass_array_unchanged": bool(mass_array_unchanged),
                "mass_volume_units_consistent": mass_unit_consistent,
                "all_states_finite": bool(all_finite), "source_partition_conserved": bool(partition_conserved),
                "max_below_table_particles": max_below_table,
                "max_source_rate": max_source_rate, "max_source_acceleration": max_source_acceleration,
                "minimum_substeps": 0 if min_substeps == 10**9 else min_substeps,
                "maximum_substeps": max_substeps, "average_substeps": self.fluid.average_substeps,
                "reset_max_position_error_m": reset_error,
                "fluid_mode": "live_PBF_action_conditioned", "rendered_training": False,
                "volume_measurement": "particle-mass simulation truth, not a camera or real scale",
                "contact_force_grasp_simulated": False,
                **nearest_attainable(target_ml, self.particle_volume_ml)}
            self.last_result = result
            _write_line(stream, {"type": "result", **result})
            if stream is not None:
                stream.flush()
        if progress is not None:
            progress({"event": "end", **result})
        return result


def run_episode(target_ml, seed, policy=None, randomize=True, record_path=None,
                progress=None, *, env=None):
    """Convenience entry point; pass ``env=PourTrainingEnv()`` to reuse it."""
    environment = env if env is not None else PourTrainingEnv()
    return environment.run_episode(target_ml, seed, policy, randomize, record_path, progress)
