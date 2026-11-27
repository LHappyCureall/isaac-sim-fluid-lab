"""24-second fine-particle evidence on the existing live PBF collector.

Only the experiment header and all-label spill accounting change here. The
original action footer remains untouched and is evaluated by a separate gate.
"""
import gzip
import hashlib
import importlib.util
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "assets/robot_testtube_pour/scripts"
spec = importlib.util.spec_from_file_location("adaptive_response_direct_env", SCRIPTS / "direct_training_env.py")
parent = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = parent
spec.loader.exec_module(parent)
write_line = parent._write_line


class ResponseEnv(parent.DirectTrainingEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.max_seconds = 24.0

    def _reset(self, *args, **kwargs):
        self.max_all_label_spilled_particles = 0
        self.max_below_table_particles = 0
        result = super()._reset(*args, **kwargs)
        self.fast_metrics["physical_duration_limit_s"] = 24.0
        return result

    def _measurement(self, controller, positions):
        metrics, masses = super()._measurement(controller, positions)
        self.max_all_label_spilled_particles = max(self.max_all_label_spilled_particles,
                                                   int(metrics["spilled"]))
        self.max_below_table_particles = max(self.max_below_table_particles, int(metrics["below_table"]))
        return metrics, masses

    def _record_step(self, stream, step, target_ml, metrics, status):
        self.max_all_label_spilled_particles = max(self.max_all_label_spilled_particles,
                                                   int(metrics["spilled"]))
        if step == 0:
            actual_module = sys.modules[type(self.last_controller).__module__]
            actual_path = Path(actual_module.__file__).resolve()
            write_line(stream, {"type": "effective_direct_sources", "run_kind": "adaptive_response_24s_v11",
                "reset_started_utc": self.reset_started_utc,
                "source_sha256": {Path(m.__file__).name: hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest()
                                  for m in (self.geometry, self.control_module)},
                "controller_module": type(self.last_controller).__module__,
                "controller_class": type(self.last_controller).__qualname__,
                "actual_controller_file": actual_path.relative_to(ROOT).as_posix(),
                "actual_controller_sha256": hashlib.sha256(actual_path.read_bytes()).hexdigest(),
                "cache_sha256": self.geometry.CACHE_SHA256,
                "completion_profile": "pregrasp_full_return_and_12_home_quiet_frames",
                "deadline_physical_seconds": 24.0, "relative_volume_tolerance": 0.001,
                "maximum_all_label_spill_particles": 1,
                "physics_solver_changed": False, "cuda_graph": False,
                "tube_containment": "actual_solver_body_q_full_R",
                "legacy_motion_knots_are_path_coordinates_not_elapsed_seconds": True})
        parent.frozen.base.PourTrainingEnv._record_step(self, stream, step, target_ml, metrics, status)
        write_line(stream, {"type": "direct_step_evidence", "step": step,
            "physical_time_s": self.fluid.time, "kinematics": self._current_kinematics,
            "controller_status": status,
            "partitions": {k: int(metrics[k]) for k in ("total", "beaker", "tube", "airborne", "spilled",
                "below_table", "initial_tube", "tube_to_beaker", "tube_remaining", "tube_airborne", "tube_spilled")},
            "maximum_all_label_spilled_particles": self.max_all_label_spilled_particles,
            "direct_metrics": self.fast_metrics})

    def post_observe(self, path, seconds=2.0):
        """Observe late water separately; extra time never rescues the action."""
        start = float(self.fluid.time)
        received = float(self.last_result["received_ml"])
        masses = self.fluid.model.particle_mass.numpy().copy()
        maximum_change = 0.0
        with gzip.open(path, "wt", encoding="utf8") as stream:
            for step in range(1, int(round(seconds / self.dt)) + 1):
                self.fluid.step_frame()
                positions = self.fluid.positions()
                velocities = self.fluid.state_0.particle_qd.numpy()
                if not np.isfinite(positions).all() or not np.isfinite(velocities).all():
                    raise RuntimeError("Nonfinite post-observation state")
                self._observe_actual_tube(self.last_controller)
                status = self.last_controller.update(self.fluid.time, positions, self.fluid.labels, velocities)
                metrics, current_mass = self._measurement(self.last_controller, positions)
                if not np.array_equal(masses, current_mass):
                    raise RuntimeError("Mass changed during post-observation")
                self.max_all_label_spilled_particles = max(self.max_all_label_spilled_particles,
                                                           int(metrics["spilled"]))
                maximum_change = max(maximum_change, abs(metrics["received_ml"] - received))
                write_line(stream, {"type": "post_observation", "step": step,
                    "physical_time_s": self.fluid.time, "measurement": metrics,
                    "controller_status": status, "kinematics": self._current_kinematics})
        return {"purpose": "late_drop_only_no_deadline_rescue", "start_physical_time_s": start,
                "end_physical_time_s": self.fluid.time, "maximum_received_change_ml": maximum_change,
                "physical_steps":int(round(seconds/self.dt)), "duration_s":self.fluid.time-start,
                "mass_array_unchanged": True, "all_states_finite": True,
                "maximum_below_table_particles": self.max_below_table_particles}
