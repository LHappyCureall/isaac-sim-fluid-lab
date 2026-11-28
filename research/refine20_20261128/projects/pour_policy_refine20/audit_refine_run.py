"""Independent full evidence audit for the fixed20mL +/-0.5mL settled-correction curriculum.

The pinned original physical audit is reused unchanged. Only the experiment
factory ownership is parameterized. Original legacy/0.1% assessments remain
intact; this new criterion and its train/validation selection are recomputed.
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
import types


ROOT = Path(__file__).resolve().parents[2]
HERE = "projects/pour_policy_refine20/"
PARENT = "projects/pour_policy_adaptive_learning/audit_prefix_calibration.py"
PARENT_SHA = "d0e6acc33e93baa57edb6e8b689f79d7b5d251821bc20a8167b192ee49d9aec6"
KIND = "refine20_settled_single_cycle_angle_v1"
SCOPE = "record_integrity_and_independent_physical_fixed20_refine_task_audit_not_real_robot_certification"
CONTROLLER = HERE + "refine_control.py"
CONTROLLER_SHA = "cb0644d3dade50cfe150964ddf2a2c4a487ab4b1bf6d2431b82975df4900672c"
COLLECTOR = HERE + "run_refine_search.py"
ENGINEERING = (195001,)
TRAIN, VAL, TEST = (195101, 195102), (195201, 195202), tuple(range(195301, 195306))
ANGLES = (84., 84.1, 84.2)
COUNTS = {"engineering": 3, "train": 6, "validation": 4, "test": 10}
PLAN_SHA = "204f5c94c95555da719cae1678e9f0d932f1881c6f288b1ce9fb8a3c44dc27c3"
ARTIFACTS = ("protocol.json", "summary.json", "job_state.json", "episodes.jsonl",
             "responses.jsonl", "matrix.json", "versions.json")
COARSE = {"prepare_fraction": 1., "dose_tilt_deg_s": 25.6, "return_fraction": 1.,
          "tail_lead_s": .26, "stop_bias_ml": 10., "max_dose_u_accel": 6.}
MOTION = {"approach_tilt_deg_s": 6., "action_delta_deg": .03,
          "action_tilt_deg_s": .6, "hold_seconds": .1, "backoff_delta_deg": .5,
          "backoff_tilt_deg_s": 4., "anchor_return_tilt_deg_s": 6.,
          "quiet_seconds": .25, "maximum_measurement_hold_s": 1., "maximum_cycles": 1}


class AuditError(ValueError):
    pass


def need(value, message):
    if not value:
        raise AuditError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
        allow_nan=False).encode()).hexdigest()


def read(path):
    def reject(value):
        raise AuditError("Nonfinite JSON constant: " + value)
    return json.loads(Path(path).read_text(encoding="utf8"), parse_constant=reject)


def utc(value):
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    need(parsed.tzinfo is not None and parsed.utcoffset() is not None, "Explicit UTC freeze/reset time required")
    return parsed.astimezone(timezone.utc)


def load_physical_audit(root=ROOT):
    sys.path.insert(0, str(root / "projects/pour_policy_adaptive_learning"))
    path = root / PARENT
    need(sha(path) == PARENT_SHA, "Pinned original physical audit changed")
    spec = importlib.util.spec_from_file_location("refine20_pinned_prefix_audit", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    # Reuse the identical compiled numerical audit. Only ownership and the
    # original whole-cycle endpoint binder differ from the prefix-only wrapper.
    namespace = dict(module.audit_episode.__globals__)
    namespace.update(COLLECTOR_SOURCE=COLLECTOR, CONTROLLER_SOURCE=CONTROLLER, CYCLE_KINDS=(KIND,),
                     cycle_bindings=module.parent.cycle_bindings)
    episode = types.FunctionType(module.audit_episode.__code__, namespace, "audit_refine_physical_episode",
                                 module.audit_episode.__defaults__, module.audit_episode.__closure__)
    return module, episode


def tolerance_assessment(result, strict, tolerance):
    target, received = result["target_ml"], result["received_ml"]
    need(type(target) in (int, float) and target == 20.
         and type(received) in (int, float) and math.isfinite(received), "Actual fixed20mL dose required")
    need(tolerance in (.5, 1.), "Only declared primary or historical easy tolerances may be used")
    dose = 20. - tolerance <= received <= 20. + tolerance
    task = ("fixed20ml_absolute05ml_24s_onegrain_v1" if tolerance == .5
            else "fixed20ml_absolute1ml_24s_onegrain_v1")
    answer = {**strict, "dose_passed": dose, "tolerance_ml": tolerance, "task_kind": task}
    answer["feasible"] = bool(strict["valid_evidence"] and dose and strict["full_home_within_deadline"]
        and strict["spill_passed"] and strict["no_late_dose_change"])
    return answer


def score(rows):
    need(rows, "Empty candidate group cannot become an elite")
    if not all(row["refine_assessment"]["valid_evidence"] for row in rows):
        return None
    if all(row["refine_assessment"]["feasible"] for row in rows):
        return (0, sum(row["result"]["physical_time_s"] for row in rows) / len(rows))
    return (1, sum(abs(row["result"]["received_ml"] - 20.) for row in rows) / len(rows),
            sum(row["refine_assessment"]["maximum_all_label_spilled_particles"] for row in rows))


def choose_angle(rows):
    choices = []
    for angle in ANGLES:
        group = [row for row in rows if row["declared_policy"]["approach_tilt_deg"] == angle]
        need(len(group) == 2 and {row["seed"] for row in group} == set(TRAIN)
             and all(row["declared_policy"]["enable_correction"] is True for row in group),
             "Training angle lacks both declared initial-state conditions")
        key = score(group)
        if key is not None:
            choices.append((key, angle))
    need(choices, "No valid training candidate may be selected")
    return min(choices)[1]


def choose_validation_role(rows):
    choices = []
    for index, role in enumerate(("baseline", "candidate")):
        group = [row for row in rows if row["role"] == role]
        need(len(group) == 2 and {row["seed"] for row in group} == set(VAL),
             "Validation role lacks the two new conditions")
        key = score(group)
        if key is not None:
            choices.append((key, index, role))
    need(choices, "No valid validation role may be recommended")
    return min(choices)[2]


def validate_schedule(cases, stage, candidate_angle=None):
    need(stage in COUNTS and isinstance(cases, list) and len(cases) == COUNTS[stage],
         "Stage must preserve the complete3/6/4/10 bounded budget")
    identities, conditions = set(), []
    split = "development" if stage == "engineering" else stage
    for case in cases:
        identifier = case["case_id"]
        need(isinstance(identifier, str) and identifier and identifier not in identities
             and all(c in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in identifier),
             "Unique safe case identities required")
        identities.add(identifier)
        need(case["target_ml"] == 20. and type(case["seed"]) is int and case["split"] == split
             and set(case["policy"]) == {"enable_correction", "approach_tilt_deg"},
             "Stage changed goal/split/action scope")
        enabled, angle = case["policy"]["enable_correction"], case["policy"]["approach_tilt_deg"]
        need(type(enabled) is bool and type(angle) in (int, float) and angle in ANGLES,
             "Unknown correction flag or approach angle")
        conditions.append((case["role"], enabled, float(angle), case["seed"]))
    if stage in ("engineering", "train"):
        seeds = ENGINEERING if stage == "engineering" else TRAIN
        expected = [("angle_" + format(angle, "g"), True, angle, seed) for angle in ANGLES for seed in seeds]
    else:
        need(candidate_angle in ANGLES, "Training candidate must freeze before later schedules")
        expected = [(role, role == "candidate", 84. if role == "baseline" else candidate_angle, seed)
                    for role in ("baseline", "candidate") for seed in (VAL if stage == "validation" else TEST)]
    need(Counter(conditions) == Counter(expected), "Stage does not contain its full paired declared conditions")


def validate_refine_status(status, policy, controller_sha):
    """Independent fixed-profile and unchanged target checks; no imports of the controller."""
    need(status["refine_controller_source_sha256"] == controller_sha
         and status["refine_enable_correction"] is policy["enable_correction"]
         and status["enable_correction"] is policy["enable_correction"]
         and status["refine_configured_approach_tilt_deg"] == policy["approach_tilt_deg"]
         and status["refine_maximum_cycles"] == 1,
         "Actual correction controller/profile ownership differs")
    need(status["actual_target_ml"] == status["control_target_ml"] == status["target_ml"] == 20.
         and status["control_target_offset_ml"] == status["prefix_target_offset_ml"] == 0.
         and status["actual_measurement_target_unchanged"] is True,
         "Curriculum relabelled its actual goal or secretly shifted its coarse reference")
    params = status["policy_params"]
    for key, value in {**COARSE, **MOTION}.items():
        need(type(params[key]) in (int, float) and params[key] == value,
             "Fixed motion parameter changed: " + key)
    need(params["approach_tilt_deg"] == policy["approach_tilt_deg"]
         and params["enable_correction"] is policy["enable_correction"]
         and params["all_motion_speeds_fixed"] is True
         and params["model_checkpoint_sha256"] is None
         and params["model_action_menu"] is None and status["response_model_used"] is False,
         "Single-angle calibration unexpectedly loaded/changed a learned policy or speed")
    count = status["response_cycle_count"]
    need(type(count) is int and 0 <= count <= 1, "One-cycle physical budget changed")
    if not policy["enable_correction"]:
        need(count == 0 and status["response_events"] == [], "Original coarse baseline executed a correction")
    received = status["received_ml"]
    need(status["refine_actual_dose_in_closed_band"] is (19.5 <= received <= 20.5)
         and status["original_strict_actual_dose_in_closed_band"] is (19.98 <= received <= 20.02),
         "Actual primary and original strict bands were conflated")
    action = status["policy_action"]["coarse_action"]
    predicted = received + status["inflight_ml"] + max(0., status["flow_ml_s"]) * .26
    need(abs(action["predicted_final_ml"] - predicted) <= 1e-8
         and abs(action["remaining_committed_ml"] - (20. - received - status["inflight_ml"])) <= 1e-8
         and action["requested_tilt_deg_s"] == 25.6
         and action["stop"] is (predicted + 10. >= 20.),
         "Original coarse stop law changed in the supposed paired baseline/candidate")


def refine_bindings(row, report, plan, parent):
    """Bind retained first quiet78 anchor and the new actual-volume home decision."""
    policy, controller_sha = row["declared_policy"], plan["source_sha256"][CONTROLLER]
    recorded, prior_decision = None, False
    for step, detail in zip(report["raw_steps"], report["raw_direct"]):
        status = detail["controller_status"]
        validate_refine_status(status, policy, controller_sha)
        anchor = status.get("prefix_anchor_observation")
        if anchor is not None:
            if recorded is None:
                parent.close(anchor["physical_time_s"], detail["physical_time_s"],
                             "First quiet anchor recorded at its real frame", 1e-8)
                parent.observe_endpoint(anchor, step, detail, "refine_coarse_anchor")
                parent.close(anchor["tilt_deg"], 78., "Actual first anchor tilt", 1e-6)
                need(anchor["settled_measurement"] is True and anchor["source_airborne_particles"] == 0
                     and anchor["quiet_measured_seconds"] >= .25 - 1e-9,
                     "Refinement started from an unsettled coarse measurement")
            else:
                need(anchor == recorded, "First coarse anchor observation was rewritten")
            recorded = anchor
        else:
            need(recorded is None, "Retained first anchor disappeared")
        decision = status["refine_actual_settled_home_decision"]
        need(type(decision) is bool and (not prior_decision or decision), "Settled home decision was rewritten")
        if decision and not prior_decision:
            need(status["settled_measurement"] is True and 19.5 <= status["received_ml"] <= 20.5,
                 "New home decision did not use an actual settled in-band volume")
        prior_decision = decision
    need(recorded == row["status"].get("prefix_anchor_observation"), "Final retained anchor differs from raw evidence")
    if row["result"]["complete"]:
        need(isinstance(recorded, dict), "Completed curriculum lacks actual first quiet78 anchor")
    starts = [event for event in row["response_events"] if event["type"] == "response_cycle_start"]
    need(len(starts) <= 1 and (policy["enable_correction"] or not starts), "Extra or baseline correction cycle")
    for start in starts:
        expected = {"approach_delta_deg": policy["approach_tilt_deg"] - 78.,
            "approach_speed_deg_s": 6., "delta_deg": .03, "tilt_speed_deg_s": .6,
            "hold_s": .1, "retract_deg": .5, "retract_speed_deg_s": 4.,
            "cutoff_tilt_deg": 78., "cutoff_speed_deg_s": 6., "quiet_seconds": .25,
            "maximum_measurement_hold_s": 1.}
        need(set(start["action"]) == set(expected), "Whole-cycle action scope changed")
        for key, value in expected.items():
            parent.close(start["action"][key], value, "Actual fixed whole-cycle action: " + key, 1e-8)
        need(start["approach_tilt_deg"] == policy["approach_tilt_deg"], "Actual approach differs from trained angle")
    return recorded


def audit_stage(run, stage, candidate_angle=None, *, root=ROOT, parent_plan_sha=None):
    run, root = Path(run).resolve(), Path(root).resolve()
    summary, job = read(run / "summary.json"), read(run / "job_state.json")
    need(summary["complete"] is True and summary["episodes"] == COUNTS[stage]
         and summary["stage"] == stage and job["state"] == "complete" and job["pid"] is None
         and not (run / "worker.lock").exists(), "Complete immutable stage required before raw-journal audit")
    physical, audit_episode = load_physical_audit(root)
    ref = physical.load_cpu_reference(root)
    plan = read(run / "protocol.json")
    need(plan["kind"] == KIND and summary["kind"] == KIND
         and plan["controller_source"] == CONTROLLER and plan["collector_source"] == COLLECTOR,
         "New experiment/controller/factory ownership differs")
    need(signature({key: value for key, value in plan.items() if key != "protocol_id"}) == plan["protocol_id"]
         == summary["protocol_id"], "Stage canonical protocol identity differs")
    matrix = read(run / "matrix.json")
    need(matrix == plan["matrix"] and matrix["draft"] is False and matrix["stage"] == stage,
         "Actual stage matrix differs from frozen protocol")
    if parent_plan_sha is not None:
        need(matrix["parent_plan_sha256"] == parent_plan_sha, "Stage no longer identifies preregistered total experiment")
    validate_schedule(matrix["cases"], stage, candidate_angle)
    expected = {"deadline_s": 24., "relative_tolerance": .001, "refine_absolute_tolerance_ml": .5, "easy_absolute_tolerance_ml": 1.,
        "peak_all_label_spill_limit": 1, "spacing_m": .0027, "particle_ml": .019683,
        "total_particles": 11625, "source_particles": 2880, "initial_source_ml": 56.68704, "post_seconds": 2.}
    for key, value in expected.items():
        physical.close(plan[key], value, "Fixed easy20 task/physics " + key)
    need(plan["refine_task_kind"] == "fixed20ml_absolute05ml_24s_onegrain_v1"
         and plan["original_strict_standard_unchanged"] is True
         and plan["solver"] == "unchanged_live_PBF_60Hz_pressure6_min4_graph_off"
         and plan["no_saved_liquid_replay"] is True and plan["phase_training_time_not_hidden"] is True,
         "Easy tolerance was confused with a change to original strict evidence or physics")
    need(len(plan["source_sha256"]) == 28 and len(plan["input_sha256"]) == 6
         and len(plan["gui_sha256"]) == 3 and plan["source_sha256"][CONTROLLER] == CONTROLLER_SHA,
         "Effective source/input/old-GUI count or controller changed")
    for group, folder in (("source_sha256", "source_snapshot"), ("input_sha256", "input_snapshot")):
        for name, expected_sha in plan[group].items():
            need(sha(ref.inside(root, name)) == sha(ref.inside(run / folder, name)) == expected_sha,
                 "Current/snapshot source/input changed: " + name)
    for name, expected_sha in plan["gui_sha256"].items():
        need(sha(ref.inside(root, name)) == expected_sha, "Original GUI model changed")
    need(COLLECTOR in plan["source_sha256"] and HERE + "task_score.py" in plan["source_sha256"],
         "New runner/task scorer absent from source freeze")
    rows = [json.loads(line, parse_constant=ref.reject_constant)
            for line in (run / "episodes.jsonl").read_text(encoding="utf8").splitlines() if line.strip()]
    ids = [case["case_id"] for case in matrix["cases"]]
    need([row["episode_id"] for row in rows] == ids and len(set(ids)) == len(ids)
         and {path.stem for path in (run / "completed").glob("*.json")} == set(ids),
         "Unique journal/matrix/sidecar case set differs")
    fingerprint = hashlib.sha256(json.dumps({"sources": plan["source_sha256"],
        "inputs": plan["input_sha256"], "spacing": .0027}, sort_keys=True).encode()).hexdigest()
    geometry = ref.Geometry(run, root, plan)
    reports, expected_artifacts = [], set()
    for row, case in zip(rows, matrix["cases"]):
        need(row["stage"] == stage and row["split"] == ("development" if stage == "engineering" else stage) and row["role"] == case["role"]
             and row["protocol_id"] == plan["protocol_id"] and row["physics_fingerprint"] == fingerprint,
             "Actual stage/role/source/physics differs")
        report = audit_episode(run, row, plan, geometry, ref)
        anchor = refine_bindings(row, report, plan, physical.parent)
        audited = physical.audit_post(run, row, report, ref)
        for item in ref.gz(ref.inside(run, report["post_name"]))[0] if report["post_name"] in row["artifacts"] else []:
            validate_refine_status(item["controller_status"], case["policy"], plan["source_sha256"][CONTROLLER])
            need(item["controller_status"].get("prefix_anchor_observation")
                 == row["status"].get("prefix_anchor_observation"), "Actual initial anchor changed during post")
        primary = tolerance_assessment(row["result"], audited["strict_assessment"], .5)
        easy = tolerance_assessment(row["result"], audited["strict_assessment"], 1.)
        need(primary == row["refine_assessment"] and easy == row["easy_assessment"],
             "Primary/easy assessments differ from actual unchanged original-target evidence")
        audited.update(stage=stage, role=row["role"], refine_assessment=primary, easy_assessment=easy,
            declared_policy=case["policy"], actual_refine_coarse_anchor=anchor, artifacts=row["artifacts"])
        reports.append(audited)
        expected_artifacts.update(row["artifacts"])
    need({path.relative_to(run).as_posix() for path in (run / "traces").glob("*.jsonl.gz")} == expected_artifacts,
         "An extra or unjournaled physical attempt exists")
    data = [sample for row in rows for sample in row["response_samples"]]
    records = [json.loads(line, parse_constant=ref.reject_constant)
               for line in (run / "responses.jsonl").read_text(encoding="utf8").splitlines() if line.strip()]
    need(data == records and summary["response_samples"] == len(data)
         and summary["responses_sha256"] == sha(run / "responses.jsonl")
         and summary["new_goal_policy_trained"] is False and summary["gui_deployment"] is False,
         "Cycle response records or non-deployment statement differ")
    for key in ("strict", "refine", "easy"):
        need(summary[key + "_task_passes"] == sum(row[key + "_assessment"]["feasible"] for row in rows),
             "Original/primary/secondary task counts were conflated")
    safety = all(report["strict_assessment"]["valid_evidence"]
        and report["strict_assessment"]["full_home_within_deadline"]
        and report["strict_assessment"]["spill_passed"]
        and report["strict_assessment"]["no_late_dose_change"] for report in reports)
    return {"stage": stage, "protocol_id": plan["protocol_id"], "episodes_audited": len(rows),
        "all_non_dose_safety_passed": safety, "engineering_dose_did_not_select_candidate": True,
        "refine_task_passes": summary["refine_task_passes"], "easy_task_passes": summary["easy_task_passes"], "original_strict_task_passes": summary["strict_task_passes"],
        "source_sha256": plan["source_sha256"], "input_sha256": plan["input_sha256"],
        "gui_sha256": plan["gui_sha256"], "versions": read(run / "versions.json"),
        "input_artifact_sha256": {name: sha(run / name) for name in ARTIFACTS},
        "episode_reports": reports, "rows": rows}


def validate_experiment_plan(experiment):
    path = experiment / "experiment_plan.json"
    need(sha(path) == PLAN_SHA, "Preregistered total experiment bytes changed")
    plan = read(path)
    need(plan["draft"] is False and plan["budget"] == 23 and plan["target_ml"] == 20.
         and plan["absolute_tolerance_ml"] == .5 and plan["deadline_s"] == 24.
         and plan["peak_all_label_spill_limit"] == 1 and plan["post_seconds"] == 2.
         and plan["budgets"] == COUNTS and tuple(plan["approach_angles_deg"]) == ANGLES
         and tuple(plan["train_seeds"]) == TRAIN and (plan["engineering_seed"],) == ENGINEERING
         and tuple(plan["validation_seeds"]) == VAL and tuple(plan["test_seeds"]) == TEST
         and plan["test_may_not_select_or_update_model"] is True
         and plan["original_strict_target_relative_tolerance"] == .001
         and plan["default_model_deployment"] is False, "Preregistered task/total budget changed")
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", "--experiment", dest="experiment", type=Path, required=True)
    parser.add_argument("--stage", choices=tuple(COUNTS), required=True)
    parser.add_argument("--workspace", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    experiment = args.experiment.resolve()
    validate_experiment_plan(experiment)
    offset = None if args.stage in ("engineering", "train") else read(experiment / "candidate.json")["policy"]["approach_tilt_deg"]
    result = audit_stage(experiment / args.stage, args.stage, offset,
                         root=args.workspace, parent_plan_sha=PLAN_SHA)
    result.pop("rows")
    result.update(complete=True, **{"pass": True}, kind=KIND, scope=SCOPE + "_stage_only",
        complete_experiment=False, experiment_plan_sha256=PLAN_SHA,
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        auditor_source_file=HERE + "audit_refine_run.py", auditor_source_sha256=sha(__file__),
        pinned_parent_audit_sha256=PARENT_SHA)
    snapshot_dir = args.output.parent / "audit_source_snapshot"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    raw = Path(__file__).read_bytes()
    snapshot = snapshot_dir / ("audit_refine_run_" + hashlib.sha256(raw).hexdigest() + ".py")
    if snapshot.exists():
        need(snapshot.read_bytes() == raw, "Audit implementation snapshot changed")
    else:
        with snapshot.open("xb") as stream:
            stream.write(raw)
    with args.output.open("x", encoding="utf8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({key: value for key, value in result.items()
                      if key not in ("episode_reports", "stage_reports")}, indent=2))


if __name__ == "__main__":
    main()
