"""Independent full evidence audit for the fixed20mL +/-1mL benchmark.

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
HERE = "projects/pour_policy_easy20/"
PARENT = "projects/pour_policy_adaptive_learning/audit_prefix_calibration.py"
PARENT_SHA = "d0e6acc33e93baa57edb6e8b689f79d7b5d251821bc20a8167b192ee49d9aec6"
KIND = "easy20_fixed_target_offset_search_v1"
SCOPE = "record_integrity_and_independent_physical_fixed20_easy_task_audit_not_real_robot_certification"
CONTROLLER = "projects/pour_policy_adaptive_learning/prefix_control.py"
CONTROLLER_SHA = "7fd23b7d35bc1de8b181a8a2784231b0c6812a6a426211a8a000cb0f83dea699"
COLLECTOR = HERE + "run_easy_search.py"
TRAIN, VAL, TEST = (194101, 194102), (194201, 194202), tuple(range(194301, 194306))
OFFSETS = (0., .5, 1.)
COUNTS = {"train": 6, "validation": 4, "test": 10}
PLAN_SHA = "db124b685beba509ffa82991ffd92d4cac0b3bc4522334726773e1506130b275"
ARTIFACTS = ("protocol.json", "summary.json", "job_state.json", "episodes.jsonl",
             "responses.jsonl", "matrix.json", "versions.json")


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
    spec = importlib.util.spec_from_file_location("easy20_pinned_prefix_audit", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    # Reuse the identical compiled numerical audit. The new experiment has a
    # different entry point and kind; its physical controller stays the same.
    namespace = dict(module.audit_episode.__globals__)
    namespace.update(COLLECTOR_SOURCE=COLLECTOR, CONTROLLER_SOURCE=CONTROLLER, CYCLE_KINDS=(KIND,))
    episode = types.FunctionType(module.audit_episode.__code__, namespace, "audit_easy_physical_episode",
                                 module.audit_episode.__defaults__, module.audit_episode.__closure__)
    return module, episode


def easy_assessment(result, strict):
    target, received = result["target_ml"], result["received_ml"]
    need(type(target) in (int, float) and target == 20.
         and type(received) in (int, float) and math.isfinite(received), "Actual fixed20mL dose required")
    dose = 19. <= received <= 21.
    answer = {**strict, "dose_passed": dose, "tolerance_ml": 1.,
              "task_kind": "fixed20ml_absolute1ml_24s_onegrain_v1"}
    answer["feasible"] = bool(strict["valid_evidence"] and dose and strict["full_home_within_deadline"]
        and strict["spill_passed"] and strict["no_late_dose_change"])
    return answer


def score(rows):
    need(rows, "Empty candidate group cannot become an elite")
    if not all(row["easy_assessment"]["valid_evidence"] for row in rows):
        return None
    if all(row["easy_assessment"]["feasible"] for row in rows):
        return (0, sum(row["result"]["physical_time_s"] for row in rows) / len(rows))
    return (1, sum(abs(row["result"]["received_ml"] - 20.) for row in rows) / len(rows),
            sum(row["easy_assessment"]["maximum_all_label_spilled_particles"] for row in rows))


def choose_offset(rows):
    candidates = []
    for offset in OFFSETS:
        group = [row for row in rows if row["declared_policy"]["control_target_offset_ml"] == offset]
        need(len(group) == 2 and {row["seed"] for row in group} == set(TRAIN),
             "Training candidate lacks both declared initial-state conditions")
        key = score(group)
        if key is not None:
            candidates.append((key, offset))
    need(candidates, "No valid training candidate may be selected")
    return min(candidates)[1]


def choose_validation_role(rows):
    choices = []
    # A tie preserves the original baseline, rather than using test outcomes.
    for index, role in enumerate(("baseline", "candidate")):
        group = [row for row in rows if row["role"] == role]
        need(len(group) == 2 and {row["seed"] for row in group} == set(VAL),
             "Validation role lacks the two new conditions")
        key = score(group)
        if key is not None:
            choices.append((key, index, role))
    need(choices, "No valid validation role may be recommended")
    return min(choices)[2]


def validate_schedule(cases, stage, candidate_offset=None):
    need(stage in COUNTS and isinstance(cases, list) and len(cases) == COUNTS[stage],
         "Stage must preserve its6/4/10 complete case budget")
    identities, conditions = set(), []
    for case in cases:
        identifier = case["case_id"]
        need(isinstance(identifier, str) and identifier and identifier not in identities
             and all(c in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in identifier),
             "Unique safe case identities required")
        identities.add(identifier)
        need(case["target_ml"] == 20. and type(case["seed"]) is int and case["split"] == stage
             and set(case["policy"]) == {"control_target_offset_ml"}, "Stage changed goal/split/action scope")
        offset = case["policy"]["control_target_offset_ml"]
        need(type(offset) in (int, float) and offset in OFFSETS, "Unknown internal stop offset")
        conditions.append((case["role"], float(offset), case["seed"]))
    if stage == "train":
        expected = [("offset_" + format(offset, "g"), offset, seed) for offset in OFFSETS for seed in TRAIN]
    else:
        need(candidate_offset in OFFSETS, "Training candidate must freeze before later schedules")
        expected = [(role, 0. if role == "baseline" else candidate_offset, seed)
                    for role in ("baseline", "candidate") for seed in (VAL if stage == "validation" else TEST)]
    need(Counter(conditions) == Counter(expected), "Stage does not contain its full paired declared conditions")


def audit_stage(run, stage, candidate_offset=None, *, root=ROOT, parent_plan_sha=None):
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
    validate_schedule(matrix["cases"], stage, candidate_offset)
    expected = {"deadline_s": 24., "relative_tolerance": .001, "easy_absolute_tolerance_ml": 1.,
        "peak_all_label_spill_limit": 1, "spacing_m": .0027, "particle_ml": .019683,
        "total_particles": 11625, "source_particles": 2880, "initial_source_ml": 56.68704, "post_seconds": 2.}
    for key, value in expected.items():
        physical.close(plan[key], value, "Fixed easy20 task/physics " + key)
    need(plan["easy_task_kind"] == "fixed20ml_absolute1ml_24s_onegrain_v1"
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
        need(row["stage"] == row["split"] == stage and row["role"] == case["role"]
             and row["protocol_id"] == plan["protocol_id"] and row["physics_fingerprint"] == fingerprint,
             "Actual stage/role/source/physics differs")
        report = audit_episode(run, row, plan, geometry, ref)
        anchor = physical.prefix_bindings(row, report, plan)
        audited = physical.audit_post(run, row, report, ref)
        for item in ref.gz(ref.inside(run, report["post_name"]))[0] if report["post_name"] in row["artifacts"] else []:
            physical.validate_prefix_status(item["controller_status"], 20., case["policy"]["control_target_offset_ml"], CONTROLLER_SHA)
            need(item["controller_status"].get("prefix_anchor_observation") == row["status"].get("prefix_anchor_observation"),
                 "Actual anchor changed during independent post observation")
        easy = easy_assessment(row["result"], audited["strict_assessment"])
        need(easy == row["easy_assessment"], "New +/-1mL assessment differs from actual original-target evidence")
        audited.update(stage=stage, role=row["role"], easy_assessment=easy,
            control_target_offset_ml=case["policy"]["control_target_offset_ml"], actual_prefix_anchor=anchor,
            artifacts=row["artifacts"])
        reports.append(audited)
        expected_artifacts.update(row["artifacts"])
    need({path.relative_to(run).as_posix() for path in (run / "traces").glob("*.jsonl.gz")} == expected_artifacts,
         "An extra or unjournaled physical attempt exists")
    need((run / "responses.jsonl").read_bytes() == b"" and summary["response_samples"] == 0
         and summary["responses_sha256"] == sha(run / "responses.jsonl")
         and summary["new_goal_policy_trained"] is False and summary["gui_deployment"] is False,
         "Prefix-only calibration fabricated response/model/default evidence")
    need(summary["strict_task_passes"] == sum(row["strict_assessment"]["feasible"] for row in rows)
         and summary["easy_task_passes"] == sum(row["easy_assessment"]["feasible"] for row in rows),
         "Original strict and new easy task counts were conflated")
    return {"stage": stage, "protocol_id": plan["protocol_id"], "episodes_audited": len(rows),
        "easy_task_passes": summary["easy_task_passes"], "original_strict_task_passes": summary["strict_task_passes"],
        "source_sha256": plan["source_sha256"], "input_sha256": plan["input_sha256"],
        "gui_sha256": plan["gui_sha256"], "versions": read(run / "versions.json"),
        "input_artifact_sha256": {name: sha(run / name) for name in ARTIFACTS},
        "episode_reports": reports, "rows": rows}


def validate_experiment_plan(experiment):
    path = experiment / "experiment_plan.json"
    need(sha(path) == PLAN_SHA, "Preregistered total experiment bytes changed")
    plan = read(path)
    need(plan["draft"] is False and plan["budget"] == 20 and plan["target_ml"] == 20.
         and plan["absolute_tolerance_ml"] == 1. and plan["deadline_s"] == 24.
         and plan["peak_all_label_spill_limit"] == 1 and plan["post_seconds"] == 2.
         and tuple(plan["train_offsets_ml"]) == OFFSETS and tuple(plan["train_seeds"]) == TRAIN
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
    offset = None if args.stage == "train" else read(experiment / "candidate.json")["policy"]["control_target_offset_ml"]
    result = audit_stage(experiment / args.stage, args.stage, offset,
                         root=args.workspace, parent_plan_sha=PLAN_SHA)
    result.pop("rows")
    result.update(complete=True, **{"pass": True}, kind=KIND, scope=SCOPE + "_stage_only",
        complete_experiment=False, experiment_plan_sha256=PLAN_SHA,
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        auditor_source_file=HERE + "audit_easy_run.py", auditor_source_sha256=sha(__file__),
        pinned_parent_audit_sha256=PARENT_SHA)
    snapshot_dir = args.output.parent / "audit_source_snapshot"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    raw = Path(__file__).read_bytes()
    snapshot = snapshot_dir / ("audit_easy_run_" + hashlib.sha256(raw).hexdigest() + ".py")
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
