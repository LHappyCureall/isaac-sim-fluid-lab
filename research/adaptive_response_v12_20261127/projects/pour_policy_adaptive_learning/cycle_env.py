"""Read-only current-frame all-label spill feedback for cycle decisions.

The unchanged PBF collector already observes the actual collider before every
controller update. This subclass adds an independent membership read there;
it never writes particles, solver state, labels or the requested target.
"""

import hashlib
import importlib.util
from pathlib import Path
import sys

import numpy as np


HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("cycle_response_evidence_parent", HERE / "response_env.py")
response = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = response
spec.loader.exec_module(response)


def current_all_label_spill(points, actual_pose, geometry, spacing):
    """Count table spills using the same actual full-R physical containment.

    Labels intentionally do not enter this all-label count. Beaker liquid
    spilled by a new action is as relevant as liquid initially in the tube.
    """
    points = np.asarray(points, dtype=float)
    pose = np.asarray(actual_pose, dtype=float)
    if (points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all()
            or pose.shape != (7,) or not np.isfinite(pose).all()
            or not np.isfinite(spacing) or spacing <= 0):
        raise RuntimeError("Invalid actual state for current-frame spill observation")
    length = float(np.linalg.norm(pose[3:]))
    if length < 1e-12:
        raise RuntimeError("Invalid actual collider quaternion for spill observation")
    x, y, z, w = pose[3:] / length
    rotation = np.array(((1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)),
                         (2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)),
                         (2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y))))
    local = (points - pose[:3]) @ rotation
    tube = geometry._cavity_mask(local)
    delta = points - np.asarray(geometry.BEAKER_CENTER)
    beaker = ((np.sum(delta[:, :2] ** 2, axis=1) <= geometry.BEAKER_RADIUS ** 2)
              & (delta[:, 2] >= 0) & (delta[:, 2] <= geometry.BEAKER_HEIGHT))
    if np.any(tube & beaker):
        raise RuntimeError("Current physical tube/beaker memberships overlap")
    free = ~(tube | beaker)
    airborne = free & (points[:, 2] > geometry.TABLE_TOP_Z + spacing)
    return {"current_all_label_spilled_particles": int(np.count_nonzero(free & ~airborne)),
        "current_below_table_particles": int(np.count_nonzero(free & (points[:, 2] < geometry.TABLE_TOP_Z - spacing)))}


class CycleEnv(response.ResponseEnv):
    def _reset(self, *args, **kwargs):
        self._last_cycle_spill_observation = None
        return super()._reset(*args, **kwargs)

    def _observe_actual_tube(self, controller):
        pose = super()._observe_actual_tube(controller)
        observed = current_all_label_spill(self.fluid.positions(), pose, self.geometry, self.spacing)
        self.max_all_label_spilled_particles = max(self.max_all_label_spilled_particles,
            observed["current_all_label_spilled_particles"])
        self.max_below_table_particles = max(self.max_below_table_particles,
            observed["current_below_table_particles"])
        self._last_cycle_spill_observation = {"physical_time_s": float(self.fluid.time),
            **observed, "maximum_all_label_spilled_particles": self.max_all_label_spilled_particles,
            "same_frame_read_only_observation": True,
            "basis": "current_particle_positions_and_actual_solver_full_R_all_labels"}
        if not hasattr(controller, "set_observed_peak_spill_particles"):
            raise RuntimeError("Cycle controller lacks actual all-label spill observation hook")
        controller.set_observed_peak_spill_particles(self.max_all_label_spilled_particles)
        return pose

    def _record_step(self, stream, step, target_ml, metrics, status):
        super()._record_step(stream, step, target_ml, metrics, status)
        observation = self._last_cycle_spill_observation
        if observation is None or abs(observation["physical_time_s"] - self.fluid.time) > 1e-9:
            raise RuntimeError("Cycle choice has no same-frame actual spill observation")
        response.write_line(stream, {"type": "cycle_spill_observation", "step": step,
            **observation, "observer_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
