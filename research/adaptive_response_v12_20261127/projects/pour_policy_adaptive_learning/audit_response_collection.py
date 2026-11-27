"""Post-hoc independent CPU audit of a completed live response collection.

This imports only the pinned CPU audit geometry implementation, never Kit,
Warp, a controller or a solver. Collection task failures stay failures. The
derived response data are cross-bound to the actual action trace endpoints.
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

from response_samples import samples_from_events, strict_assessment


ROOT = Path(__file__).resolve().parents[2]
OLD_AUDIT = "projects/pour_policy_training_v9/audit_fine_run.py"
OLD_AUDIT_SHA = "cadf9e35089db98a1d971e1476229e856342a802628f95c414ceb244f27f5c83"
HERE = "projects/pour_policy_adaptive_learning/"
DT, QUANTUM = 1 / 60, .019683
GOAL_KIND = "adaptive_frozen_response_goal_evaluation_v12"
CYCLE_KINDS = ("adaptive_real_cycle_response_collection_v12", GOAL_KIND)
COLLECTION_SCOPE = "record_integrity_and_independent_physical_motion_mass_response_audit_not_task_success"
GOAL_SCOPE = "record_integrity_and_independent_physical_motion_mass_goal_audit_not_task_success"
ARTIFACTS = ("protocol.json", "summary.json", "job_state.json", "episodes.jsonl",
             "responses.jsonl", "matrix.json", "versions.json")


class AuditError(ValueError):
    pass


def need(condition, message):
    if not condition:
        raise AuditError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    def reject(value):
        raise AuditError("Nonfinite JSON constant: " + value)
    return json.loads(Path(path).read_text(encoding="utf-8-sig"), parse_constant=reject)


def close(value, expected, message, tolerance=1e-9):
    need(type(value) in (int, float) and math.isfinite(value)
         and abs(value - expected) <= tolerance, message)


def utc(value):
    need(isinstance(value, str), "Explicit frozen/reset UTC timestamp required")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    need(parsed.tzinfo is not None and parsed.utcoffset() is not None,
         "Frozen/reset timestamps need an explicit time zone")
    return parsed.astimezone(timezone.utc)


def bounded(root, name):
    root = Path(root).resolve()
    path = Path(name).expanduser()
    path = (path if path.is_absolute() else root / path).resolve()
    need(path.is_relative_to(root), "Goal evidence path escapes its explicit workspace")
    return path


def required_runtime_sources(kind):
    common = (HERE + "response_control.py", HERE + "response_env.py", HERE + "response_samples.py")
    if kind == "adaptive_real_small_action_response_collection_v11":
        return (*common, HERE + "collect_responses.py")
    if kind in CYCLE_KINDS:
        cycle = (*common, HERE + "cycle_control.py", HERE + "cycle_env.py")
        if kind == GOAL_KIND:
            return (*cycle, HERE + "evaluate_cycle_policy.py", HERE + "cycle_response_model.py")
        return (*cycle, HERE + "collect_cycles.py")
    raise AuditError("Unknown runtime owner")


def goal_model_binding(run, root, plan, summary, ref):
    """Independently prove original fitting identity and allowed harness change.

    Execution provenance differs because its runner, model, and input evidence
    differ. The original model fingerprint is preserved, never rewritten to
    make the evaluation look like the fitting collection.
    """
    binding = plan["evaluation_model_binding"]
    need(isinstance(binding, dict) and summary["evaluation_model_binding"] == binding,
         "Goal summary changed its model compatibility receipt")
    checkpoint = bounded(root, binding["model_checkpoint"])
    model_sha = sha(checkpoint)
    need(model_sha == binding["model_raw_sha256"]
         and plan["input_sha256"].get(checkpoint.relative_to(root).as_posix()) == model_sha,
         "Frozen goal checkpoint differs from the protocol input bytes")
    model = read(checkpoint)
    ref.finite(model)
    need(model.get("content_sha256") == ref.signature({k: v for k, v in model.items() if k != "content_sha256"})
         and model["content_sha256"] == binding["model_content_sha256"], "Model content seal differs")
    need(model.get("kind") == "empirical_anchor_approach_micro_cutoff_cycle_v1"
         and model.get("schema_version") == 1
         and model.get("selection_kind") == binding["selection_kind"] == "validation_only",
         "Goal evaluation requires the frozen validation-only whole-cycle candidate")
    model_source = HERE + "cycle_response_model.py"
    need(model["source_sha256"] == binding["model_source_sha256"] == plan["source_sha256"][model_source],
         "Loaded goal model implementation differs from the fitted source")
    created = utc(model["created_at_utc"])
    need(model["created_at_utc"] == binding["model_created_at_utc"]
         and created < utc(plan["created_at_utc"]), "Model was not frozen before evaluation protocol creation")
    collection = bounded(root, binding["collection_run"])
    original = read(collection / "protocol.json")
    original_summary = read(collection / "summary.json")
    original_job = read(collection / "job_state.json")
    need(original["kind"] == "adaptive_real_cycle_response_collection_v12"
         and original_summary["complete"] is True and original_job["state"] == "complete"
         and original_job["pid"] is None and not (collection / "worker.lock").exists(),
         "Fitting collection is not finalized")
    need(ref.signature({k: v for k, v in original.items() if k != "protocol_id"}) == original["protocol_id"],
         "Original collection protocol identity changed")
    provenance = model["provenance"]
    need(original["protocol_id"] == binding["collection_protocol_id"] == provenance["collection_protocol_id"]
         and provenance["collection_artifact_sha256"] == binding["collection_artifact_sha256"],
         "Checkpoint provenance does not identify this original collection")
    hashes = binding["collection_artifact_sha256"]
    need(set(ARTIFACTS) <= set(hashes), "Original collection artifact bindings are incomplete")
    for name in ARTIFACTS:
        path = bounded(collection, name)
        need(sha(path) == hashes[name]
             and plan["input_sha256"].get(path.relative_to(root).as_posix()) == hashes[name],
             "Original collection artifact differs from frozen goal inputs: " + name)
    need(original["source_sha256"] == binding["collection_source_sha256"]
         and original["input_sha256"] == binding["collection_input_sha256"]
         and original["gui_sha256"] == binding["collection_gui_sha256"] == plan["gui_sha256"],
         "Model compatibility receipt rewrote original source/input/GUI identities")
    # The model=None collection did not execute the post-hoc response-model
    # implementation. Evaluation adds that now-loaded implementation plus its
    # factory; both are frozen in evaluation. Core runtime sources stay exact.
    added = {HERE + "evaluate_cycle_policy.py", HERE + "cycle_response_model.py"}
    need(set(plan["source_sha256"]) == set(original["source_sha256"]) | added,
         "Evaluation added or removed an undeclared implementation role")
    for category, folder in (("source_sha256", "source_snapshot"), ("input_sha256", "input_snapshot")):
        for name, digest in original[category].items():
            need(plan[category].get(name) == digest, "Core fitting/evaluation source/input changed: " + name)
            need(sha(ref.inside(collection / folder, name)) == digest,
                 "Original fitting source/input snapshot changed: " + name)
    fingerprint = hashlib.sha256(json.dumps({"sources": original["source_sha256"],
        "inputs": original["input_sha256"], "spacing": .0027}, sort_keys=True).encode()).hexdigest()
    need(model["physics_fingerprint"] == fingerprint == binding["fitted_collection_physics_fingerprint"]
         and binding["execution_harness_fingerprint_is_separately_recorded"] is True,
         "Original fitting fingerprint was rebound to another execution")
    matrix = original["matrix"]
    need(read(collection / "matrix.json") == matrix, "Original fitting schedule changed")
    collection_seeds = {case["seed"] for case in matrix["cases"]}
    used_seeds = set(model["training_seeds"] + model["validation_seeds"])
    need(used_seeds <= collection_seeds and not set(model["training_seeds"]) & set(model["validation_seeds"]),
         "Model fitting/validation seed identities differ")
    cases = plan["matrix"]["cases"]
    splits, seeds = {case["split"] for case in cases}, {case["seed"] for case in cases}
    need(len(splits) == 1 and splits <= {"development", "test"}
         and not seeds & collection_seeds and binding["evaluation_split"] == next(iter(splits))
         and binding["evaluation_seeds"] == sorted(seeds), "Goal case splits/seeds overlap fitting data")
    need(binding["historical_seed_matches"] == [], "Goal protocol admitted previously used initial-state seeds")
    normalized_profiles = []
    for case in cases:
        policy = case["policy"]
        need(bounded(root, policy["model_checkpoint"]) == checkpoint and policy["model_sha256"] == model_sha,
             "Goal case changed the frozen selected checkpoint")
        normalized = dict(policy)
        normalized["model_checkpoint"] = checkpoint.relative_to(root).as_posix()
        normalized_profiles.append(normalized)
    need(all(profile == normalized_profiles[0] for profile in normalized_profiles)
         and binding["normalized_declared_policy"] == normalized_profiles[0]
         and binding["declared_policy_sha256"] == ref.signature(normalized_profiles[0]),
         "Goal policy profile/menu differs from its frozen nominal compatibility receipt")
    need(read(run / "versions.json") == read(collection / "versions.json"),
         "Evaluation runtime versions differ from fitting physics")
    return {"checkpoint_sha256": model_sha, "checkpoint_content_sha256": model["content_sha256"],
        "model_created_at_utc": model["created_at_utc"], "model_source_sha256": model["source_sha256"],
        "original_collection_protocol_id": original["protocol_id"],
        "fitted_collection_physics_fingerprint": fingerprint, "evaluation_split": next(iter(splits)),
        "core_physics_and_controller_hashes_unchanged": True, "original_fingerprint_preserved": True,
        "declared_policy_sha256": binding["declared_policy_sha256"],
        "evaluation_model_binding": binding}


def goal_row_binding(row, effective, model):
    need(row["stage"] == "goal_evaluation" and row["role"] == "candidate"
         and row["split"] == model["evaluation_split"], "Goal row stage/role/split differs")
    need(row["frozen_model_raw_sha256"] == model["checkpoint_sha256"]
         and row["declared_policy"]["model_sha256"] == model["checkpoint_sha256"]
         and row["status"]["policy_params"]["model_checkpoint_sha256"] == model["checkpoint_sha256"],
         "Actual goal policy checkpoint differs from its freeze")
    need(row["model_created_at_utc"] == model["model_created_at_utc"]
         and row["actual_reset_started_at_utc"] == effective["reset_started_utc"]
         and row["model_frozen_before_actual_reset"] is True
         and utc(model["model_created_at_utc"]) < utc(effective["reset_started_utc"]),
         "Model freeze did not precede the actual physical reset")
    need(row["fitted_collection_physics_fingerprint"] == model["fitted_collection_physics_fingerprint"],
         "Goal row changed original fitted provenance fingerprint")


def load_cpu_reference(root):
    path = root / OLD_AUDIT
    need(sha(path) == OLD_AUDIT_SHA, "Pinned independent audit implementation changed")
    spec = importlib.util.spec_from_file_location("response_independent_v9_audit", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    def commanded(status, u, start, time_s, geometry):
        segment = status.get("response_motion_segment")
        need(isinstance(segment, list) and len(segment) == 7,
             "Every action state requires an explicit committed motion segment")
        t0, u0, velocity, acceleration, duration, goal, kind = segment
        need(all(type(x) in (int, float) and math.isfinite(x)
                 for x in segment[:6]), "Nonfinite committed motion")
        if kind == "v10_C2_source_step":
            need(duration > 0, "C2 step needs positive physical duration")
            x = min(1., max(0., (time_s - t0) / duration))
            return u0 + (goal - u0) * x ** 3 * (10 - 15 * x + 6 * x * x)
        if kind == "v10_stationary_hold":
            return u0
        elapsed = max(0., time_s - t0)
        need(duration >= 0, "Negative committed horizon")
        if kind in ("prepare_clock", "reverse_prepare_clock"):
            return goal if elapsed >= duration else u0 + velocity * elapsed
        horizon = min(elapsed, duration)
        source = u0 + velocity * horizon + .5 * acceleration * horizon * horizon
        terminal_velocity = velocity + acceleration * horizon
        if elapsed > duration:
            extra = min(elapsed - duration, abs(terminal_velocity) / 1.5)
            brake = -math.copysign(1.5, terminal_velocity) if terminal_velocity else 0.
            source += terminal_velocity * extra + .5 * brake * extra * extra
        if kind == "C1_retract_to_78_commit" and source <= goal:
            return goal
        return min(geometry.maximum, max(0., source))

    module.commanded_u = commanded
    return module


def observe_endpoint(event_observation, raw_step, raw_detail, label):
    """Response state is a raw same-time full-R measurement, not a model label."""
    for key, expected in (("received_ml", raw_step["truth"]["received_ml"]),
            ("tube_remaining_ml", raw_step["truth"]["remaining_ml"]),
            ("tilt_deg", raw_detail["controller_status"]["tilt_deg"]),
            ("quiet_measured_seconds", raw_detail["controller_status"]["quiet_measured_seconds"])):
        close(event_observation[key], expected, label + " actual " + key)
    need(event_observation["source_airborne_particles"] == raw_detail["partitions"]["tube_airborne"],
         label + " airborne count differs")
    if "settled_measurement" in event_observation:
        need(event_observation["settled_measurement"] == raw_detail["controller_status"]["settled_measurement"],
             label + " actual settled observation differs")
    if "net_outside_particles" in event_observation:
        need(event_observation["net_outside_particles"] == 2880 - raw_detail["partitions"]["tube_remaining"],
             label + " actual net outside count differs")


def response_bindings(row, steps, direct):
    """Bind every retained event prefix and complete sample to raw evidence."""
    events = row["response_events"]
    need(isinstance(events, list), "Event array required")
    indices = [event.get("event_id") for event in events]
    need(indices == list(range(len(events))), "Unique ordered controller event identities required")
    by_time = {round(step["physical_time_s"] / DT): (step, detail)
               for step, detail in zip(steps, direct)}
    observed = []
    for detail in direct:
        recorded = detail["controller_status"].get("response_events", [])
        need(events[:len(recorded)] == recorded, "Response event history was rewritten")
        need(len(recorded) >= len(observed) and recorded[:len(observed)] == observed,
             "Per-frame response events lost their immutable prefix")
        observed = recorded
    need(observed == events, "Completed controller event list differs from action-end trace")
    canonical = samples_from_events(row)
    need(row["response_samples"] == canonical, "Derived response sample differs from actual event projection")
    for sample in canonical:
        for prefix in ("pre", "post"):
            t = sample[prefix + "_physical_time_s"]
            frame = int(round(t / DT))
            need(frame in by_time, "Response endpoint is not a recorded real action frame")
            step, detail = by_time[frame]
            close(t, step["physical_time_s"], "Response endpoint time", 1e-8)
            observe_endpoint(sample[prefix + "_observation"], step, detail, prefix)
    return canonical


def cycle_bindings(row, steps, direct):
    """Independently bind explicit whole cycles, including approach water."""
    frames = {int(round(step["physical_time_s"] / DT)): (step, detail)
              for step, detail in zip(steps, direct)}
    pending, micro_started, reports = None, False, []
    for event in row["response_events"]:
        kind = event["type"]
        if kind == "response_cycle_start":
            need(pending is None, "Previous whole cycle has no measured endpoint")
            pending, micro_started = event, False
            frame = int(round(event["physical_time_s"] / DT))
            need(frame in frames, "Whole cycle start is not an actual physical frame")
            step, detail = frames[frame]
            close(step["physical_time_s"], event["physical_time_s"], "Cycle actual start frame", 1e-8)
            observe_endpoint(event["observation"], step, detail, "cycle_start")
            close(event["observation"]["tilt_deg"], 78., "Cycle actual start78anchor", 1e-6)
            close(event["approach_tilt_deg"], event["observation"]["tilt_deg"]
                + event["action"]["approach_delta_deg"], "Whole cycle declares full approach geometry")
        elif kind == "response_action_start" and pending is not None:
            micro_started = True
        elif kind in ("response_cycle_complete", "response_cycle_unsettled"):
            need(pending is not None, "Whole cycle endpoint has no actual quiet anchor")
            need(event["pre_event_id"] == pending["event_id"]
                 and event["pre_observation"] == pending["observation"]
                 and event["action"] == pending["action"], "Whole-cycle start/action was rewritten")
            pre, post = event["pre_observation"], event["post_observation"]
            before = pending["physical_time_s"]
            after = event["physical_time_s"]
            need(0 <= before < after <= min(24., row["result"]["physical_time_s"]),
                 "Whole cycle response must end before original action, not during post")
            close(event["pre_physical_time_s"], before, "Cycle actual start time")
            close(event["post_physical_time_s"], after, "Cycle actual endpoint time")
            for label, t, obs in (("cycle_pre", before, pre), ("cycle_post", after, post)):
                frame = int(round(t / DT))
                need(frame in frames, "Whole cycle endpoint is missing an actual physical frame")
                step, detail = frames[frame]
                close(step["physical_time_s"], t, "Whole cycle real frame time", 1e-8)
                observe_endpoint(obs, step, detail, label)
                close(obs["tilt_deg"], 78., "Whole cycle actual78-degree anchor", 1e-6)
            delta = post["received_ml"] - pre["received_ml"]
            close(event["outcome"]["settled_delta_ml"], delta, "Whole-cycle actual dose includes approach")
            close(event["outcome"]["duration_s"], after - before, "Whole-cycle original physical duration")
            need(event["executed_micro"] == micro_started
                 and event["action_execution_complete"] == bool(event["settled"] and micro_started),
                 "Whole cycle invented execution of a skipped micro branch")
            if event["settled"]:
                need(pre["settled_measurement"] is True and post["settled_measurement"] is True
                     and pre["source_airborne_particles"] == post["source_airborne_particles"] == 0
                     and min(pre["quiet_measured_seconds"], post["quiet_measured_seconds"]) >= .2 - 1e-9,
                     "Settled whole-cycle endpoint lacks actual quiet measurements")
            reports.append({"source_start_event_id": pending["event_id"],
                "source_end_event_id": event["event_id"], "settled_delta_ml": delta,
                "duration_s": after - before, "settled": event["settled"],
                "executed_micro": micro_started, "includes_approach_water": True})
            pending = None
    return {"measured_cycles": reports, "partial_cycle_starts": 0 if pending is None else 1}


def audit_episode(run, row, plan, geometry, ref):
    name, result = row["episode_id"], row["result"]
    ref.finite(row)
    need(ref.read(ref.inside(run, "completed/" + name + ".json")) == row, "Sidecar/journal differs")
    cases = {case["case_id"]: case for case in plan["matrix"]["cases"]}
    case = cases[name]
    need(row["target_ml"] == case["target_ml"] == result["target_ml"]
         and row["seed"] == case["seed"] == result["seed"] and row["split"] == case["split"]
         and row["declared_policy"] == case["policy"], "Scheduled conditions differ")
    action_name = "traces/" + name + ".steps.jsonl.gz"
    post_name = "traces/" + name + ".post.jsonl.gz"
    need(action_name in row["artifacts"] and set(row["artifacts"]) <= {action_name, post_name},
         "Unexpected raw artifact set")
    for relative, digest in row["artifacts"].items():
        need(sha(ref.inside(run, relative)) == digest, "Artifact bytes changed: " + relative)
    action, _ = ref.gz(ref.inside(run, action_name))
    header = action[0]
    need(header["type"] == "episode" and header["fluid_mode"] == "live_PBF_action_conditioned",
         "Original live physics header missing")
    need(header["randomized"] is True and result["randomized"] is True, "Randomized physical prefix missing")
    for key in ("target_ml", "seed", "policy_name", "policy_params", "randomized"):
        ref.same(header[key], result[key], "Original header/footer " + key)
    close(header["physics_dt_s"], DT, "Original physical dt")
    close(header["particle_spacing_m"], .0027, "Unchanged spacing")
    close(header["particle_volume_ml"], QUANTUM, "Unchanged particle quantum")
    close(header["density_kg_m3"], 1000., "Unchanged density")
    need(header["particle_count"] == result["particle_count"] == 11625
         and result["source_particle_count"] == 2880, "Fixed identities changed")
    close(result["initial_source_ml"], 56.68704, "Source mass volume")
    close(result["physical_time_s"], result["physical_steps"] * DT, "Action clock step/60", 1e-8)
    need(result["physical_steps"] <= 1440 and result["minimum_substeps"] >= 4,
         "Deadline or adaptive minimum changed")
    close(result["error_ml"], result["received_ml"] - result["target_ml"], "Signed actual dose error")
    effective = [item for item in action if item["type"] == "effective_direct_sources"]
    need(len(effective) == 1, "One effective physical-source header required")
    effective = effective[0]
    expected_controller = (HERE + "cycle_control.py" if plan["kind"] in CYCLE_KINDS
                           else HERE + "response_control.py")
    expected_collector = (HERE + "evaluate_cycle_policy.py" if plan["kind"] == GOAL_KIND else
        HERE + "collect_cycles.py" if plan["kind"] in CYCLE_KINDS else HERE + "collect_responses.py")
    if plan["kind"] in CYCLE_KINDS:
        need(plan["controller_source"] == expected_controller
             and plan["collector_source"] == expected_collector, "Declared V12 source ownership differs")
    need(effective["actual_controller_file"] == expected_controller
         and effective["actual_controller_sha256"] == plan["source_sha256"][expected_controller],
         "Actual controller differs from protocol")
    expected_bindings = {"mouth_pivot_lift_geometry.py": plan["source_sha256"][
        "assets/robot_testtube_pour/scripts/mouth_pivot_lift_geometry.py"],
        Path(expected_collector).name: plan["source_sha256"][expected_collector]}
    need(effective["source_sha256"] == expected_bindings, "Actual geometry/factory source binding differs")
    selected_cache = "projects/pour_policy_training_v3/mouth_pivot_lift025_z022_cache.npz"
    need(effective["cache_sha256"] == plan["input_sha256"][selected_cache], "Actual path cache binding differs")
    need(effective["cuda_graph"] is False and effective["physics_solver_changed"] is False
         and effective["tube_containment"] == "actual_solver_body_q_full_R"
         and effective["deadline_physical_seconds"] == 24.
         and effective["relative_volume_tolerance"] == .001
         and effective["maximum_all_label_spill_particles"] == 1, "Actual task/physics contract changed")
    steps = [item for item in action if item["type"] == "step"]
    direct = [item for item in action if item["type"] == "direct_step_evidence"]
    spill_observations = [item for item in action if item["type"] == "cycle_spill_observation"]
    footers = [item for item in action if item["type"] == "result"]
    failures = [item for item in action if item["type"] == "failure"]
    need(len(footers) == 1 and action[-1] is footers[0]
         and {k: v for k, v in footers[0].items() if k != "type"} == result,
         "Original result footer changed")
    need(len(steps) == len(direct) and steps, "Every real step requires raw state and direct evidence")
    cycle_mode = plan["kind"] in CYCLE_KINDS
    if cycle_mode:
        need(len(spill_observations) == len(steps), "Every cycle frame needs same-frame all-label spill feedback")
    motion = ref.Motion(geometry)
    peak = below = 0
    drift = 0.
    initial_mass = result["actual_initial_total_mass_kg"]
    for index, (step, detail) in enumerate(zip(steps, direct)):
        need(step["step"] == detail["step"] == index and step["solver_substeps"] >= 4,
             "Real step/substep sequence differs")
        close(step["physical_time_s"], index * DT, "60Hz actual frame time", 1e-8)
        close(detail["physical_time_s"], step["physical_time_s"], "Paired physical evidence time", 1e-8)
        ref.partitions(detail["partitions"], step["truth"])
        peak = max(peak, detail["partitions"]["spilled"])
        below = max(below, detail["partitions"]["below_table"])
        need(detail["maximum_all_label_spilled_particles"] == peak, "All-label peak tally differs")
        if cycle_mode:
            observed = spill_observations[index]
            need(observed["step"] == index and observed["same_frame_read_only_observation"] is True,
                 "Same-frame cycle observer sequence differs")
            close(observed["physical_time_s"], step["physical_time_s"], "Cycle spill observation clock", 1e-8)
            need(observed["current_all_label_spilled_particles"] == detail["partitions"]["spilled"]
                 and observed["current_below_table_particles"] == detail["partitions"]["below_table"]
                 and observed["maximum_all_label_spilled_particles"] == peak,
                 "Cycle spill feedback differs from current actual full-R membership")
            need(observed["observer_source_sha256"] == plan["source_sha256"][HERE + "cycle_env.py"],
                 "Actual cycle observer source differs")
            need(detail["controller_status"]["response_cycle_observed_peak_spill_particles"] == peak,
                 "Cycle choice used missing or delayed all-label peak spill feedback")
        drift = max(drift, abs(step["truth"]["mass_kg"] - initial_mass))
        motion.row(detail["physical_time_s"], step["source_time"], detail["kinematics"], detail["controller_status"])
    action_rows = motion.rows
    for failure in failures:
        need(not result["valid_physics"], "Partial physics action relabelled valid")
        motion.dense(failure.get("last_motion_evidence", {}).get("dense_240_hz_samples", []), terminal=True)
    gap = result["physical_steps"] - steps[-1]["step"]
    need(gap in (0, 1) and (gap == 0 or failures), "Unexplained terminal frame gap")
    if gap == 0:
        close(result["received_ml"], steps[-1]["truth"]["received_ml"], "Footer actual received amount")
    if result["complete"]:
        need(row["status"] == direct[-1]["controller_status"], "Action endpoint status differs")
        need(len(direct) >= 13 and all(item["controller_status"]["home_static"] for item in direct[-13:]),
             "Full physical home and12quiet interval missing")
    canonical = response_bindings(row, steps, direct)
    cycles = (cycle_bindings(row, steps, direct)
              if plan["kind"] in CYCLE_KINDS else None)
    return {"episode_id": name, "action_rows": action_rows, "unlogged_terminal_action_frames": gap,
        "failures": failures, "canonical_samples": canonical, "motion": motion,
        "peak": peak, "below": below, "drift": drift, "initial_mass": initial_mass,
        "effective": effective, "post_name": post_name, "cycles": cycles, "cycle_mode": cycle_mode}


def audit_post(run, row, report, ref):
    result, post = row["result"], row["post_observation"]
    valid = (isinstance(post, dict) and post.get("all_states_finite") is True
        and post.get("mass_array_unchanged") is True and not post.get("error"))
    name = report["post_name"]
    post_rows, parse_error = (ref.gz(ref.inside(run, name), allow_partial=not valid)
        if name in row["artifacts"] else ([], None))
    maximum_change, last_time = 0., result["physical_time_s"]
    for index, item in enumerate(post_rows, 1):
        need(item["type"] == "post_observation" and item["step"] == index, "Post sequence differs")
        close(item["physical_time_s"], result["physical_time_s"] + index * DT,
              "Two-second actual post clock", 1e-8)
        measurement = item["measurement"]
        close(measurement["physical_time_s"], item["physical_time_s"], "Post measurement time", 1e-8)
        need(measurement["volume_transform_basis"] == "actual_solver_body_q_full_R", "Post full-R basis differs")
        ref.partitions(measurement, measurement)
        report["peak"] = max(report["peak"], measurement["spilled"])
        if report["cycle_mode"]:
            need(item["controller_status"]["response_cycle_observed_peak_spill_particles"] == report["peak"],
                 "Post controller used missing or delayed current all-label spill observation")
        report["below"] = max(report["below"], measurement["below_table"])
        report["drift"] = max(report["drift"], abs(measurement["total_mass_kg"] - report["initial_mass"]))
        maximum_change = max(maximum_change, abs(measurement["received_ml"] - result["received_ml"]))
        report["motion"].row(item["physical_time_s"], measurement["source_time"],
                             item["kinematics"], item["controller_status"])
        last_time = item["physical_time_s"]
    if valid:
        need(len(post_rows) == post["physical_steps"] == 120, "Exactly120 actual post frames required")
        close(post["start_physical_time_s"], result["physical_time_s"], "Actual post start", 1e-8)
        close(post["end_physical_time_s"], last_time, "Actual post endpoint", 1e-8)
        close(last_time - result["physical_time_s"], 2., "Two raw physical seconds", 1e-8)
        close(post["maximum_received_change_ml"], maximum_change, "Actual late-water change")
        need(post["maximum_below_table_particles"] == report["below"], "Post below-table maximum differs")
    else:
        need(row["assessment"]["valid_evidence"] is False, "Invalid post provided valid task evidence")
    need(row["assessment"]["maximum_all_label_spilled_particles"] == report["peak"],
         "All-label peak does not include full action and post")
    expected_legacy = ref.assess(result, row["status"], report["peak"], post)
    need(row["assessment"] == expected_legacy, "Original legacy assessment rewritten")
    expected = strict_assessment(result, row["status"], report["peak"], post)
    need(row["strict_assessment"] == expected, "New closed-interval assessment differs")
    if expected["valid_evidence"]:
        need(report["drift"] <= 1e-12 and result["particle_mass_array_unchanged"] is True
             and result["mass_volume_units_consistent"] is True, "Actual mass evidence differs")
        limits = {"speed_limit_ratio": 1.00001, "acceleration_rad_s2": 25.0001,
            "position_violation_rad": 1e-6, "body_position_error_m": 1e-5,
            "body_orientation_error_rad": 1e-4, "grasp_error_m": .003}
        need(all(report["motion"].maximum[key] <= bound for key, bound in limits.items()),
             "Independent q/FK/rigid-grasp bounds contradict valid evidence")
    return {"episode_id": row["episode_id"], "target_ml": row["target_ml"], "seed": row["seed"],
        "split": row["split"], "physical_s": result["physical_time_s"], "received_ml": result["received_ml"],
        "error_ml": result["error_ml"], "wall_s": row["wall_seconds"], "strict_assessment": expected,
        "action_frames": report["action_rows"], "post_frames": len(post_rows),
        "maximum_motion": report["motion"].maximum,
        "all_dense_rows": report["motion"].dense_rows,
        "maximum_mass_drift_kg": report["drift"], "peak_all_label_spill_particles": report["peak"],
        "maximum_below_table_particles": report["below"], "post_change_ml": maximum_change,
        "post_partial_parse_error": parse_error,
        "unlogged_terminal_action_frames": report["unlogged_terminal_action_frames"],
        "failure_records": report["failures"], "response_samples": len(report["canonical_samples"]),
        "whole_cycle_evidence": report["cycles"],
        "actual_reset_started_utc": report["effective"]["reset_started_utc"],
        "old_footer_success_used": False}


def audit(run, root=ROOT):
    run, root = Path(run).resolve(), Path(root).resolve()
    summary = read(run / "summary.json")
    job = read(run / "job_state.json")
    need(summary.get("complete") is True and job.get("state") == "complete" and job.get("pid") is None,
         "Complete immutable collection required before journal/sample inspection")
    plan = read(run / "protocol.json")
    need(plan.get("kind") in ("adaptive_real_small_action_response_collection_v11", *CYCLE_KINDS),
         "Unexpected collection/evaluation protocol")
    goal_mode = plan["kind"] == GOAL_KIND
    ref = load_cpu_reference(root)
    protocol_id = plan["protocol_id"]
    need(ref.signature({k: v for k, v in plan.items() if k != "protocol_id"}) == protocol_id
         and summary["protocol_id"] == protocol_id, "Canonical protocol identity differs")
    need(plan["matrix"]["draft"] is False and read(run / "matrix.json") == plan["matrix"],
         "Finalized matrix snapshot differs")
    criteria = {"deadline_s": 24., "relative_tolerance": .001, "peak_all_label_spill_limit": 1,
        "spacing_m": .0027, "particle_ml": .0027 ** 3 * 1e6, "total_particles": 11625,
        "source_particles": 2880, "initial_source_ml": 56.68704, "post_seconds": 2.}
    for key, value in criteria.items():
        close(plan[key], value, "Fixed protocol criterion " + key)
    need(plan["solver"] == "unchanged_live_PBF_60Hz_pressure6_min4_graph_off"
         and plan["collection_task_success_not_required"] is (not goal_mode)
         and plan["phase_training_time_not_hidden"] is True
         and plan["no_saved_liquid_replay"] is True, "Collection physics claims differ")
    for key, folder in (("source_sha256", "source_snapshot"), ("input_sha256", "input_snapshot")):
        need(isinstance(plan[key], dict) and bool(plan[key]), "Empty source/input freeze")
        for name, digest in plan[key].items():
            need(sha(ref.inside(run / folder, name)) == digest, "Frozen snapshot changed: " + name)
            need(sha(ref.inside(root, name)) == digest, "Active published experiment source/input changed: " + name)
    for name, digest in plan["gui_sha256"].items():
        need(sha(ref.inside(root, name)) == digest, "Old GUI weight changed: " + name)
    required = required_runtime_sources(plan["kind"])
    need(all(name in plan["source_sha256"] for name in required), "New runtime source not frozen")
    expected_physics = hashlib.sha256(json.dumps({"sources": plan["source_sha256"],
        "inputs": plan["input_sha256"], "spacing": .0027}, sort_keys=True).encode()).hexdigest()
    rows = [json.loads(line, parse_constant=ref.reject_constant)
            for line in (run / "episodes.jsonl").read_text(encoding="utf8").splitlines() if line.strip()]
    cases = plan["matrix"]["cases"]
    ids = [case["case_id"] for case in cases]
    need(len(set(ids)) == len(ids) and [row["episode_id"] for row in rows] == ids
         and len(rows) == summary["episodes"], "Unique full scheduled case sequence differs")
    need({p.stem for p in (run / "completed").glob("*.json")} == set(ids), "Unjournaled completed case exists")
    groups = {}
    for case in cases:
        seed, split = case["seed"], case["split"]
        need(seed not in groups or groups[seed] == split, "Same initial-condition seed leaked between splits")
        groups[seed] = split
    geometry = ref.Geometry(run, root, plan)
    model_binding = goal_model_binding(run, root, plan, summary, ref) if goal_mode else None
    reports, all_samples = [], []
    for row in rows:
        need(row["protocol_id"] == protocol_id and row["physics_fingerprint"] == expected_physics,
             "Row source/physics identity differs")
        report = audit_episode(run, row, plan, geometry, ref)
        if goal_mode:
            goal_row_binding(row, report["effective"], model_binding)
        reports.append(audit_post(run, row, report, ref))
        all_samples.extend(report["canonical_samples"])
    samples = [json.loads(line, parse_constant=ref.reject_constant)
               for line in (run / "responses.jsonl").read_text(encoding="utf8").splitlines() if line.strip()]
    need(samples == all_samples and summary["response_samples"] == len(samples)
         and summary["responses_sha256"] == sha(run / "responses.jsonl"), "Derived response artifact changed")
    need(len({sample["row_id"] for sample in samples}) == len(samples), "Duplicate response row identity")
    cycle_path = root / HERE / "cycle_samples.py"
    cycle_projection = None
    cycle_source_sha = None
    if plan["kind"] in CYCLE_KINDS:
        need(cycle_path.is_file(), "Explicit whole-cycle post-hoc projection tool missing")
        cycle_source_sha = sha(cycle_path)
        spec = importlib.util.spec_from_file_location("response_independent_cycle_projection", cycle_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        cycle_projection = {"samples": [], "partial_cycles": []}
        for row in rows:
            projected = module.cycles_from_episode(row)
            need(projected["explicit_cycle_events"] is True
                 or not (projected["samples"] or projected["partial_cycles"]),
                 "V12 response samples require explicit measured cycle events")
            cycle_projection["samples"].extend(projected["samples"])
            cycle_projection["partial_cycles"].extend(projected["partial_cycles"])
        need(sha(cycle_path) == cycle_source_sha, "Cycle projection source changed during audit")
    passes = sum(report["strict_assessment"]["feasible"] for report in reports)
    need(summary["strict_task_passes"] == passes and summary["new_goal_policy_trained"] is False
         and summary["gui_deployment"] is False, "Collection summary fabricated a task/policy success")
    if goal_mode:
        need(summary["frozen_response_model_used"] is True, "Goal summary omitted its frozen-model role")
    artifacts = ARTIFACTS
    return {"complete": True, "pass": True,
        "kind": plan["kind"], "scope": GOAL_SCOPE if goal_mode else COLLECTION_SCOPE,
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "protocol_id": protocol_id,
        "episodes_audited": len(rows), "response_samples": len(samples),
        "responses_sha256": sha(run / "responses.jsonl"),
        "input_artifact_sha256": {name: sha(run / name) for name in artifacts},
        "audit_tool_sha256": sha(__file__), "independent_geometry_audit_sha256": OLD_AUDIT_SHA,
        "strict_task_passes": passes, "all_strict_task_cases_passed": passes == len(reports),
        "cycle_events_crossbound": plan["kind"] in CYCLE_KINDS,
        "goal_evaluation": goal_mode,
        "model_checkpoint_sha256": None if model_binding is None else model_binding["checkpoint_sha256"],
        "model_compatibility_audit": model_binding,
        "whole_cycles_audited": sum(len(report["whole_cycle_evidence"]["measured_cycles"])
            for report in reports if report["whole_cycle_evidence"] is not None),
        "cycle_projection_sha256": None if cycle_projection is None else ref.signature(cycle_projection),
        "cycle_samples_source_sha256": cycle_source_sha,
        "partial_cycles_retained": 0 if cycle_projection is None else len(cycle_projection["partial_cycles"]),
        "eligible_samples": sum(sample["safe"] and sample["settled"] and sample["peak_spill_particles"] <= 1
                                and sample["outcome"]["settled_delta_ml"] >= 0 for sample in samples),
        "samples_by_split": dict(Counter(sample["split"] for sample in samples)),
        "episode_reports": reports}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit(args.run, args.workspace)
    if args.output:
        with args.output.open("x", encoding="utf8") as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
            stream.write("\n")
    print(json.dumps({key: value for key, value in result.items() if key != "episode_reports"}, indent=2))


if __name__ == "__main__":
    main()
