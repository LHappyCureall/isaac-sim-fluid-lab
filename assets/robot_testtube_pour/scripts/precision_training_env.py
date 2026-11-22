"""Provenance and complete count logging for live precision-pouring trials.

The physical environment, reset data, timestep, solver and episode result are
inherited unchanged. Inject the desired controller module through ``control``
or set ``control_module`` before a new episode. Reuse the same environment to
keep one fluid model/hash grid when comparing controllers.
"""

from datetime import datetime, timezone
import hashlib
import inspect
from pathlib import Path
import sys

from pour_training_env import PourTrainingEnv, _write_line


def _now():
    return datetime.now(timezone.utc).isoformat()


def _file_digest(path):
    path = Path(path).resolve()
    if not path.is_file():
        raise RuntimeError(f"Cannot record controller provenance: missing source file {path}")
    return str(path), hashlib.sha256(path.read_bytes()).hexdigest()


def _class_source_path(cls):
    try:
        source = inspect.getsourcefile(cls)
    except (OSError, TypeError):
        source = None
    if source is None:
        source = getattr(sys.modules.get(cls.__module__), "__file__", None)
    if source is None:
        try:
            source = inspect.getsourcefile(cls.__init__)
        except (OSError, TypeError):
            source = None
    if source is None:
        raise RuntimeError(f"Cannot identify the actual controller class source: {cls.__module__}.{cls.__qualname__}")
    return Path(source)


def _class_provenance(cls):
    path, digest = _file_digest(_class_source_path(cls))
    try:
        source = inspect.getsource(cls).replace("\r\n", "\n").replace("\r", "\n")
    except (OSError, TypeError):
        source = None
    return {"class": cls.__qualname__, "module": cls.__module__, "file": path,
            "file_sha256": digest,
            "class_source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest() if source is not None else None}


class PrecisionTrainingEnv(PourTrainingEnv):
    """Logging-only subclass; the base run_episode/footer are not overridden."""

    def _reset(self, *args, **kwargs):
        # Capture this event before the base reset/initialization and before any
        # physical step, rather than infer a start time from a completed trace.
        self._precision_wall_clock_start_utc = _now()
        return super()._reset(*args, **kwargs)

    def _implementation_record(self):
        controller = self.last_controller
        if controller is None:
            raise RuntimeError("No active controller exists for provenance logging.")
        actual_class = type(controller)
        active = _class_provenance(actual_class)
        mro = [_class_provenance(cls) for cls in actual_class.__mro__ if cls is not object]
        wrapper_file, wrapper_sha = _file_digest(__file__)
        base_file, base_sha = _file_digest(inspect.getsourcefile(PourTrainingEnv))
        return {"type": "controller_implementation",
                "wall_clock_start_utc": self._precision_wall_clock_start_utc,
                "wall_clock_start_event": "precision environment _reset entry, before base reset and physics",
                "metadata_recorded_at_utc": _now(),
                "physical_time_s": self.fluid.time,
                "controller_class": active["class"], "controller_module": active["module"],
                "controller_file": active["file"], "controller_file_sha256": active["file_sha256"],
                "controller_class_source_sha256": active["class_source_sha256"],
                "controller_mro_sources": mro,
                "precision_wrapper_file": wrapper_file, "precision_wrapper_sha256": wrapper_sha,
                "base_environment_file": base_file, "base_environment_sha256": base_sha,
                "legacy_header_source_sha256_semantics":
                    "The inherited episode header lists the base collector's legacy dependency references. "
                    "The actual injected controller class/file and inherited implementation sources are recorded here.",
                "physical_solver_changed_by_wrapper": False,
                "episode_result_changed_by_wrapper": False}

    def _record_step(self, stream, step, target_ml, metrics, status):
        if stream is None:
            return
        if step == 0:
            _write_line(stream, self._implementation_record())
        observation_keys = ("target_ml", "received_ml", "inflight_ml", "flow_ml_s",
                            "tilt_deg", "released_ml", "tube_remaining_ml", "phase")
        command_keys = ("command_goal_source", "command_max_rate", "command_acceleration",
                        "command_source_velocity", "command_next_source_velocity")
        policy_action = status.get("policy_action")
        if isinstance(policy_action, dict):
            policy_action = dict(policy_action)
        # Preserve the base collector's fields and add directly auditable total
        # partition counts. Controller-specific optional fields remain complete
        # in controller_status, without inventing baseline pulse semantics.
        _write_line(stream, {"type": "step", "step": step,
            "physical_time_s": self.fluid.time, "physics_dt_s": self.dt,
            "source_time": metrics["source_time"],
            "observation": {key: status.get(key) for key in observation_keys},
            "action": {"policy": policy_action,
                       **{key: status.get(key) for key in command_keys}},
            "truth": {"target_ml": target_ml, "received_ml": metrics["received_ml"],
                      "inflight_ml": metrics["inflight_ml"], "remaining_ml": metrics["remaining_ml"],
                      "spilled_ml": metrics["spilled_ml"], "mass_kg": metrics["total_mass_kg"],
                      "source_partition_count": metrics["source_partition_count"],
                      "below_table_particles": metrics["below_table"],
                      "total_partition_count": metrics["total_partition_count"],
                      "total_particle_count": metrics["total"],
                      "source_particle_count": metrics["initial_tube"],
                      "beaker_particles": metrics["beaker"],
                      "tube_particles": metrics["tube"],
                      "airborne_particles": metrics["airborne"],
                      "spilled_particles": metrics["spilled"]},
            "solver_substeps": int(self.fluid.last_substeps),
            "precision_phase": status.get("precision_phase"),
            "pulse_count": status.get("pulse_count"),
            "controller_status": dict(status)})
