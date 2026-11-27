"""Full-R direct-path experiments with the unchanged live liquid solver.

The geometric clock embeds C2 robot acceleration. Coordinate-rate switches at
zero physical derivatives are not joint acceleration impulses. Actual URDF
joint speed, 240Hz joint acceleration and grasp FK remain the release gates.
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
spec = importlib.util.spec_from_file_location('direct_frozen_fast_env', DIRECTORY / 'fast_training_env.py')
frozen = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = frozen
spec.loader.exec_module(frozen)
_write_line = frozen.base._write_line
nearest_attainable = frozen.base.nearest_attainable


class DirectTrainingEnv(frozen.FastTrainingEnv):
    # A path-coordinate acceleration cap has no physical meaning across
    # embedded C2 clock joins. Every physical joint is still limited to the
    # unchanged URDF speed and the declared 25rad/s² simulation envelope.
    PATH_ACCELERATION_CAP = None

    def _reset(self, *args, **kwargs):
        result = super()._reset(*args, **kwargs)
        self.fast_metrics.update(maximum_actual_body_position_error_m=0.0,
                                 maximum_actual_body_orientation_error_rad=0.0,
                                 maximum_actual_grasp_position_error_m=0.0)
        return result

    def _observe_actual_tube(self, controller):
        pose = np.asarray(self.fluid.state_0.body_q.numpy()[self.fluid.tube_body], dtype=float)
        if pose.shape != (7,) or not np.isfinite(pose).all():
            raise RuntimeError('Nonfinite actual tube collider transform.')
        if hasattr(controller, 'set_observed_tube_pose'):
            controller.set_observed_tube_pose(pose.copy())
        return pose

    @staticmethod
    def _rotation_xyzw(quaternion):
        quaternion = np.asarray(quaternion, dtype=float)
        length = np.linalg.norm(quaternion)
        if not np.isfinite(length) or length < 1e-12:
            raise RuntimeError('Invalid actual tube orientation.')
        x, y, z, w = quaternion / length
        return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
            [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
            [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])

    def _measurement(self, controller, positions):
        metrics, masses = super()._measurement(controller, positions)
        actual = self._observe_actual_tube(controller)
        rotation = self._rotation_xyzw(actual[3:])
        desired_position, desired_quaternion = self.geometry.tube_pose(metrics['source_time'])
        desired_rotation = self._rotation_xyzw(desired_quaternion)
        position_error = float(np.linalg.norm(actual[:3] - desired_position))
        angle_error = float(np.arccos(np.clip((np.trace(rotation.T @ desired_rotation)-1)/2, -1, 1)))
        hand_position, hand_rotation = self.geometry.hand_fk(metrics['source_time'])
        hand_tcp = hand_position + hand_rotation @ np.array([0., 0., self.geometry.PANDA_HAND_TO_TCP])
        tube_tcp = actual[:3] + rotation @ np.array([0., 0., self.geometry.TUBE_LENGTH-self.geometry.TCP_BELOW_MOUTH])
        grasp_error = float(np.linalg.norm(hand_tcp - tube_tcp))
        for key, value in (('maximum_actual_body_position_error_m', position_error),
                           ('maximum_actual_body_orientation_error_rad', angle_error),
                           ('maximum_actual_grasp_position_error_m', grasp_error)):
            self.fast_metrics[key] = max(self.fast_metrics.get(key, 0.0), value)
        self._current_kinematics.update(actual_tube_pose_xyzw=actual.tolist(),
            actual_body_position_error_m=position_error, actual_body_orientation_error_rad=angle_error,
            actual_grasp_position_error_m=grasp_error)
        if position_error > 1e-5 or angle_error > 1e-4 or grasp_error > .003:
            raise RuntimeError('Actual physical tube collider differs from the verified rigid grasp.')
        # Membership uses the actual float32 solver body pose, with the full
        # rotation matrix. Rendering/nominal pitch never determines dose truth.
        g = self.geometry
        points = np.asarray(positions, dtype=float)
        labels = np.asarray(self.fluid.labels)
        delta = points - np.asarray(g.BEAKER_CENTER)
        beaker = (np.sum(delta[:, :2]**2, axis=1) <= g.BEAKER_RADIUS**2) & (delta[:, 2] >= 0) & (delta[:, 2] <= g.BEAKER_HEIGHT)
        tube = g._cavity_mask((points-actual[:3]) @ rotation)
        free = ~(tube | beaker)
        airborne = free & (points[:, 2] > g.TABLE_TOP_Z+self.spacing)
        spilled = free & ~airborne
        source = labels == 1
        counts = {'total': len(points), 'beaker': int(np.count_nonzero(beaker)),
            'tube': int(np.count_nonzero(tube)), 'airborne': int(np.count_nonzero(airborne)),
            'spilled': int(np.count_nonzero(spilled)),
            'below_table': int(np.count_nonzero(free & (points[:, 2] < g.TABLE_TOP_Z-self.spacing))),
            'initial_tube': int(np.count_nonzero(source)),
            'tube_to_beaker': int(np.count_nonzero(source & beaker)),
            'tube_remaining': int(np.count_nonzero(source & tube)),
            'tube_airborne': int(np.count_nonzero(source & airborne)),
            'tube_spilled': int(np.count_nonzero(source & spilled))}
        if (sum(counts[k] for k in ('beaker', 'tube', 'airborne', 'spilled')) != counts['total']
                or sum(counts[k] for k in ('tube_to_beaker', 'tube_remaining', 'tube_airborne', 'tube_spilled')) != counts['initial_tube']):
            raise RuntimeError('Full-R actual-body particle membership overlaps or loses identities.')
        metrics.update(counts)
        metrics.update(received_ml=counts['tube_to_beaker']*self.particle_volume_ml,
            remaining_ml=counts['tube_remaining']*self.particle_volume_ml,
            inflight_ml=counts['tube_airborne']*self.particle_volume_ml,
            spilled_ml=counts['tube_spilled']*self.particle_volume_ml,
            tube_transfer_fraction=counts['tube_to_beaker']/counts['initial_tube'],
            beaker_volume_ml=counts['beaker']*self.particle_volume_ml,
            tube_volume_ml=counts['tube']*self.particle_volume_ml,
            volume_transform_basis='actual_solver_body_q_full_R')
        self._last_liquid_metrics, self._last_liquid_masses = dict(metrics), masses.copy()
        return metrics, masses

    def _record_step(self, stream, step, target_ml, metrics, status):
        if step == 0:
            effective = {Path(module.__file__).name: hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
                         for module in (self.geometry, self.control_module)}
            _write_line(stream, {'type': 'effective_direct_sources', 'reset_started_utc': self.reset_started_utc,
                'source_sha256': effective, 'cache_sha256': self.geometry.CACHE_SHA256,
                'controller_class': type(self.last_controller).__qualname__,
                'completion_profile': 'pregrasp_transfer_dose_upright_return_original_pose_stable',
                'deadline_physical_seconds': 5.0, 'physics_solver_changed': False,
                'source_coordinate_units': 'embedded_C2_reference_robot_clock_seconds',
                'coordinate_acceleration_is_not_a_physical_release_gate': True,
                'tube_containment': 'full_rotation_matrix_not_a_pitch_projection',
                'legacy_header_motion_knots_field_is_path_coordinate_not_elapsed_time': True})
        frozen.base.PourTrainingEnv._record_step(self, stream, step, target_ml, metrics, status)
        _write_line(stream, {'type': 'direct_step_evidence', 'step': step, 'physical_time_s': self.fluid.time,
            'kinematics': self._current_kinematics, 'controller_status': status,
            'partitions': {k: int(metrics[k]) for k in ('total', 'beaker', 'tube', 'airborne', 'spilled',
                'below_table', 'initial_tube', 'tube_to_beaker', 'tube_remaining', 'tube_airborne', 'tube_spilled')},
            'direct_metrics': self.fast_metrics})


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
        # The tube holds about 51.4 mL. The same controller compares the requested
        # volume with the beaker, so a target is not limited to the 20/30/40 exam.
        if not (10.0 <= target_ml <= 48.0):
            raise ValueError("Pour targets must be between 10 and 48 mL.")
        if int(seed) != seed or isinstance(seed, bool):
            raise ValueError("seed must be an integer.")
        started = time.perf_counter()
        controller, initial, variation = self._reset(target_ml, int(seed), policy, bool(randomize))
        positions = self.fluid.positions()
        reset_error = float(np.max(np.abs(positions - initial)))
        self._observe_actual_tube(controller)
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
                              for name in ("liquid.py", "geometry.py", "target_control.py", "fast_geometry.py", "fast_control.py", "fast_training_env.py", "direct_fast_geometry.py", "direct_fast_control.py", "direct_training_env.py")}}
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
                    self._observe_actual_tube(controller)
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
                     and (self.PATH_ACCELERATION_CAP is None or max_source_acceleration <= self.PATH_ACCELERATION_CAP + 1e-5) and min_substeps >= 4)
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
                self._observe_actual_tube(controller)
                status = controller.update(self.fluid.time, positions, self.fluid.labels, velocities)
                metrics, masses = self._measurement(controller, positions)
                conserved = conserved and bool(np.array_equal(masses, initial_masses))
                maximum_change = max(maximum_change, abs(metrics['received_ml'] - initial_received))
                if abs(self.fluid.time - 5.0) < 1e-8:
                    deadline_received = metrics['received_ml']
                _write_line(stream, {'type': 'post_observation', 'step': step,
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
