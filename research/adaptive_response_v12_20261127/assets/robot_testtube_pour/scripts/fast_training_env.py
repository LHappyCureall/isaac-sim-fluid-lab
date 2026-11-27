"""Five-second full-cycle collector using unchanged live PBF and fixed mass.

This V3 changes robot path timing and feedback, not the liquid solver. The
V3 collector writes its own result footer using the explicitly declared new
path-coordinate bounds; original result schema and liquid measurements remain.
No GUI, video or liquid replay is involved.
"""
from contextlib import nullcontext
from datetime import datetime, timezone
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
spec = importlib.util.spec_from_file_location('fast_original_training_env', DIRECTORY / 'pour_training_env.py')
base = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = base
spec.loader.exec_module(base)
_write_line = base._write_line
nearest_attainable = base.nearest_attainable


class FastTrainingEnv(base.PourTrainingEnv):
    """The complete action must finish within 300 original physical steps.

    URDF joint speed limits are checked from the generated joint path. The
    25 rad/s² acceleration cap is an explicitly declared simulation envelope,
    not a Franka hardware rating. All raw finite differences remain in logs.
    """

    JOINT_ACCELERATION_CAP_RAD_S2 = 25.0
    PATH_ACCELERATION_CAP = 3.0

    def __init__(self, **kwargs):
        if kwargs.get('geometry') is None or kwargs.get('control') is None:
            raise ValueError('Pass the explicit V3 geometry and controller modules.')
        kwargs.update(max_seconds=5.0, tolerance_ml=.1)
        super().__init__(**kwargs)
        limits = np.asarray(self.geometry.JOINT_VELOCITY_LIMITS_RAD_S, dtype=float)
        lower = np.asarray(self.geometry.JOINT_LOWER_LIMITS_RAD, dtype=float)
        upper = np.asarray(self.geometry.JOINT_UPPER_LIMITS_RAD, dtype=float)
        if (any(v.shape != (7,) or not np.isfinite(v).all() for v in (limits, lower, upper))
                or np.any(limits <= 0) or np.any(lower >= upper)):
            raise ValueError('Explicit finite URDF joint bounds are required.')
        self.fast_metrics = {}

    def _reset(self, *args, **kwargs):
        self.reset_started_utc = datetime.now(timezone.utc).isoformat()
        self._last_q = None
        self._last_qd = np.zeros(7)
        self._last_measurement_time = None
        self._last_tcp_position = None
        self._last_dense_q = None
        self._last_dense_qd = np.zeros(7)
        self._last_dense_time = None
        self._last_tilt = None
        self._current_kinematics = {}
        self._last_liquid_metrics = None
        self._last_liquid_masses = None
        self.fast_metrics = {'maximum_joint_speed_limit_ratio': 0.0,
                             'maximum_joint_acceleration_rad_s2': 0.0,
                             'maximum_tcp_speed_m_s': 0.0, 'maximum_tilt_speed_deg_s': 0.0,
                             'maximum_hand_fk_position_error_m': 0.0,
                             'maximum_hand_fk_orientation_error_rad': 0.0,
                             'physical_duration_limit_s': 5.0,
                             'joint_acceleration_simulation_envelope_rad_s2': self.JOINT_ACCELERATION_CAP_RAD_S2}
        return super()._reset(*args, **kwargs)

    def _measurement(self, controller, positions):
        metrics, masses = super()._measurement(controller, positions)
        # Retain this frame's liquid truth before a later robot-path check can
        # fail. A failed motion must not pair the previous dose with a new time.
        self._last_liquid_metrics = dict(metrics)
        self._last_liquid_masses = masses.copy()
        q = np.asarray(self.geometry.joint_positions(metrics['source_time']), dtype=float)
        limits = np.asarray(self.geometry.JOINT_VELOCITY_LIMITS_RAD_S, dtype=float)
        if q.shape != (7,) or limits.shape != (7,) or not np.isfinite(q).all():
            raise RuntimeError('Invalid generated robot joint path.')
        if (np.any(q < self.geometry.JOINT_LOWER_LIMITS_RAD - 1e-6)
                or np.any(q > self.geometry.JOINT_UPPER_LIMITS_RAD + 1e-6)):
            raise RuntimeError('Actual robot joint position limit exceeded.')
        time_s = float(self.fluid.time)
        hand_position, hand_quat = self.geometry.hand_pose(metrics['source_time'])
        fk_position, fk_rotation = self.geometry.hand_fk(metrics['source_time'])
        if not all(np.isfinite(value).all() for value in
                   (np.asarray(hand_position), np.asarray(hand_quat), fk_position, fk_rotation)):
            raise RuntimeError('Nonfinite prescribed or URDF FK hand pose.')
        w, x, y, z = np.asarray(hand_quat, dtype=float)
        expected_rotation = np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
            [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
            [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
        position_error = float(np.linalg.norm(fk_position - hand_position))
        orientation_error = float(np.arccos(np.clip((np.trace(fk_rotation.T @ expected_rotation)-1)/2, -1, 1)))
        self.fast_metrics['maximum_hand_fk_position_error_m'] = max(self.fast_metrics['maximum_hand_fk_position_error_m'], position_error)
        self.fast_metrics['maximum_hand_fk_orientation_error_rad'] = max(self.fast_metrics['maximum_hand_fk_orientation_error_rad'], orientation_error)
        if position_error > .003 or orientation_error > .02:
            raise RuntimeError('Independent URDF hand FK does not match the prescribed tube grasp.')
        tcp_position = np.asarray(hand_position) + expected_rotation @ np.array(
            [0., 0., self.geometry.PANDA_HAND_TO_TCP])
        dense = []
        if self._last_dense_q is None:
            self._last_dense_q, self._last_dense_time = q.copy(), time_s
        elif time_s > self._last_dense_time + 1e-10:
            # Four independent kinematic samples per physical frame catch path
            # corners that a single 60 Hz average could hide. No fluid substeps
            # or physical clock are replaced by these CPU-only observations.
            sample_start = self._last_dense_time
            for sample_time in np.linspace(sample_start, time_s, 5)[1:]:
                sample_q = np.asarray(self.geometry.joint_positions(controller.source_time(sample_time)))
                sample_dt = float(sample_time - self._last_dense_time)
                sample_qd = (sample_q - self._last_dense_q) / sample_dt
                sample_qdd = (sample_qd - self._last_dense_qd) / sample_dt
                sample_ratio = float(np.max(np.abs(sample_qd) / limits))
                sample_acceleration = float(np.max(np.abs(sample_qdd)))
                if not (np.isfinite(sample_q).all() and np.isfinite(sample_qd).all() and np.isfinite(sample_qdd).all()):
                    raise RuntimeError('Nonfinite dense joint path state.')
                self.fast_metrics['maximum_joint_speed_limit_ratio'] = max(self.fast_metrics['maximum_joint_speed_limit_ratio'], sample_ratio)
                self.fast_metrics['maximum_joint_acceleration_rad_s2'] = max(self.fast_metrics['maximum_joint_acceleration_rad_s2'], sample_acceleration)
                dense.append({'time_s': float(sample_time), 'q_rad': sample_q.tolist(),
                              'dq_rad_s': sample_qd.tolist(), 'ddq_rad_s2': sample_qdd.tolist(),
                              'speed_limit_ratio': sample_ratio})
                self._last_dense_q, self._last_dense_qd, self._last_dense_time = sample_q.copy(), sample_qd.copy(), float(sample_time)
        qd, qdd = np.zeros(7), np.zeros(7)
        if self._last_q is not None and time_s > self._last_measurement_time + 1e-10:
            dt = time_s - self._last_measurement_time
            qd = (q - self._last_q) / dt
            qdd = (qd - self._last_qd) / dt
            self.fast_metrics['maximum_tcp_speed_m_s'] = max(self.fast_metrics['maximum_tcp_speed_m_s'],
                float(np.linalg.norm(tcp_position - self._last_tcp_position) / dt))
            self.fast_metrics['maximum_tilt_speed_deg_s'] = max(self.fast_metrics['maximum_tilt_speed_deg_s'],
                abs(metrics['tilt_degrees'] - self._last_tilt) / dt)
        elif self._last_q is not None:
            qd = self._last_qd.copy()
        ratio = float(np.max(np.abs(qd) / limits))
        acceleration = float(np.max(np.abs(qdd)))
        self.fast_metrics['maximum_joint_speed_limit_ratio'] = max(self.fast_metrics['maximum_joint_speed_limit_ratio'], ratio)
        self.fast_metrics['maximum_joint_acceleration_rad_s2'] = max(self.fast_metrics['maximum_joint_acceleration_rad_s2'], acceleration)
        self._current_kinematics = {'joint_position_rad': q.tolist(), 'joint_velocity_rad_s': qd.tolist(),
            'joint_acceleration_rad_s2': qdd.tolist(), 'joint_speed_limit_ratio': ratio,
            'joint_velocity_limits_rad_s': limits.tolist(), 'dense_240_hz_samples': dense,
            'tcp_position_m': tcp_position.tolist(), 'hand_fk_position_error_m': position_error,
            'hand_fk_orientation_error_rad': orientation_error}
        self._last_q, self._last_qd, self._last_measurement_time = q.copy(), qd.copy(), time_s
        self._last_tcp_position, self._last_tilt = tcp_position.copy(), metrics['tilt_degrees']
        peak_ratio = self.fast_metrics['maximum_joint_speed_limit_ratio']
        peak_acceleration = self.fast_metrics['maximum_joint_acceleration_rad_s2']
        if peak_ratio > 1.00001:
            raise RuntimeError(f'Actual sampled URDF joint speed limit exceeded: {peak_ratio:.6f}.')
        if peak_acceleration > self.JOINT_ACCELERATION_CAP_RAD_S2 + 1e-4:
            raise RuntimeError(f'Declared simulation joint acceleration cap exceeded: {peak_acceleration:.6f}.')
        return metrics, masses

    def _record_step(self, stream, step, target_ml, metrics, status):
        if step == 0:
            effective = {str(Path(module.__file__).name): hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
                         for module in (self.geometry, self.control_module)}
            base._write_line(stream, {'type': 'effective_fast_sources', 'reset_started_utc': self.reset_started_utc,
                'source_sha256': effective, 'cache_sha256': self.geometry.CACHE_SHA256,
                'controller_class': type(self.last_controller).__qualname__,
                'completion_profile': 'pregrasp_lift_transfer_dose_upright_return_home_stable',
                'deadline_physical_seconds': 5.0, 'physics_solver_changed': False,
                'source_coordinate_units': 'URDF_joint_velocity_normalized_path_seconds',
                'legacy_header_motion_knots_field_is_path_coordinate_not_physical_timing': True})
        super()._record_step(stream, step, target_ml, metrics, status)
        base._write_line(stream, {'type': 'fast_step_evidence', 'step': step, 'physical_time_s': self.fluid.time,
            'kinematics': self._current_kinematics, 'controller_status': status,
            'partitions': {k: int(metrics[k]) for k in ('total', 'beaker', 'tube', 'airborne', 'spilled',
                'below_table', 'initial_tube', 'tube_to_beaker', 'tube_remaining', 'tube_airborne', 'tube_spilled')},
            'fast_metrics': self.fast_metrics})

    def post_observe(self, path, seconds=2.0):
        """Late-drop check; never turns a failed 5-second action into success."""
        controller = self.last_controller
        start = self.fluid.time
        initial_received = self.last_result['received_ml']
        initial_masses = self.fluid.model.particle_mass.numpy().copy()
        maximum_change = 0.0
        finite = conserved = True
        deadline_received = self.last_result['received_ml'] if abs(start - 5.0) < 1e-8 else None
        end_time = max(7.0, start + float(seconds))
        with gzip.open(path, 'wt', encoding='utf-8') as stream:
            for step in range(1, int(round((end_time - start) / self.dt)) + 1):
                self.fluid.step_frame()
                positions = self.fluid.positions()
                velocities = self.fluid.state_0.particle_qd.numpy()
                finite = finite and bool(np.isfinite(positions).all() and np.isfinite(velocities).all())
                status = controller.update(self.fluid.time, positions, self.fluid.labels, velocities)
                metrics, masses = self._measurement(controller, positions)
                conserved = conserved and bool(np.array_equal(masses, initial_masses))
                maximum_change = max(maximum_change, abs(metrics['received_ml'] - initial_received))
                if abs(self.fluid.time - 5.0) < 1e-8:
                    deadline_received = metrics['received_ml']
                base._write_line(stream, {'type': 'post_observation', 'step': step,
                    'physical_time_s': self.fluid.time, 'measurement': metrics,
                    'controller_status': status, 'kinematics': self._current_kinematics,
                    'solver_substeps': int(self.fluid.last_substeps),
                    'all_finite': finite, 'mass_array_unchanged': conserved})
        return {'purpose': 'late_drop_check_not_action_time_or_deadline_rescue',
                'start_physical_time_s': start, 'duration_s': self.fluid.time - start,
                'end_physical_time_s': self.fluid.time, 'deadline_received_ml': deadline_received,
                'maximum_received_change_ml': maximum_change, 'all_states_finite': finite,
                'mass_array_unchanged': conserved,
                'passed': bool(maximum_change == 0 and finite and conserved and metrics['spilled_ml'] == 0),
                'final_received_ml': metrics['received_ml'], 'final_spilled_ml': metrics['spilled_ml']}


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
            "contact_force_grasp_simulated": False, "declared_path_acceleration_cap": self.PATH_ACCELERATION_CAP,
            "source_sha256": {name: hashlib.sha256((DIRECTORY / name).read_bytes()).hexdigest()
                              for name in ("liquid.py", "geometry.py", "target_control.py", "fast_geometry.py", "fast_control.py", "fast_training_env.py")}}
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
                failure_stage = "fluid_step"
                try:
                    self.fluid.step_frame()
                    failure_stage = "controller_observation"
                    positions = self.fluid.positions()
                    velocities = self.fluid.state_0.particle_qd.numpy()
                    if not np.isfinite(positions).all() or not np.isfinite(velocities).all():
                        all_finite = False
                        raise RuntimeError("Nonfinite particle state.")
                    status = controller.update(self.fluid.time, positions, self.fluid.labels, velocities)
                    failure_stage = "liquid_and_motion_measurement"
                    metrics, masses = self._measurement(controller, positions)
                    failure_stage = "integrity_and_record"
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
                    if (failure_stage == "liquid_and_motion_measurement"
                            and self._last_liquid_metrics is not None
                            and abs(self._last_liquid_metrics['physical_time_s'] - self.fluid.time) < 1e-10):
                        metrics = dict(self._last_liquid_metrics)
                        masses = self._last_liquid_masses.copy()
                        max_mass_drift = max(max_mass_drift, abs(metrics['total_mass_kg'] - initial_mass))
                        mass_array_unchanged = mass_array_unchanged and bool(np.array_equal(masses, initial_masses))
                        max_below_table = max(max_below_table, int(metrics['below_table']))
                        max_spilled_ml = max(max_spilled_ml, float(metrics['spilled_ml']))
                    partition_conserved = "partition" not in error_message.lower()
                    termination_reason = "invalid_physics_state"
                    _write_line(stream, {"type": "failure", "step": step,
                        "physical_time_s": self.fluid.time, "error": error_message,
                        "failure_stage": failure_stage,
                        "measurement_physical_time_s": metrics['physical_time_s'],
                        "last_available_liquid_measurement": metrics,
                        "partial_unlogged_fluid_frame_possible": failure_stage == 'fluid_step',
                        "kinematic_evidence_time_s": self._last_measurement_time,
                        "last_motion_evidence": self._current_kinematics})
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
                     and max_source_acceleration <= self.PATH_ACCELERATION_CAP + 1e-5 and min_substeps >= 4)
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
                "measurement_physical_time_s": metrics['physical_time_s'],
                "measurement_basis": ("last_complete_measurement_before_partial_fluid_failure"
                    if error_message is not None and failure_stage == 'fluid_step'
                    else "liquid_measurement_at_reported_sample_time"),
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
                "contact_force_grasp_simulated": False, "declared_path_acceleration_cap": self.PATH_ACCELERATION_CAP,
                **nearest_attainable(target_ml, self.particle_volume_ml)}
            self.last_result = result
            _write_line(stream, {"type": "result", **result})
            if stream is not None:
                stream.flush()
        if progress is not None:
            progress({"event": "end", **result})
        return result
