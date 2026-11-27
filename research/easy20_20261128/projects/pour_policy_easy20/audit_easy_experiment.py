"""Final independent6/4/10 audit and freeze/selection proof for easy20.

No GPU or controller imports. The separate pinned stage auditor is rerun on
all original physical evidence; no held-out row enters model selection.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
HERE = "projects/pour_policy_easy20/"
STAGE_SHA = "3ba8464416df2d7d02e9b533c11dd0f599dc76da7de5c809db791d3020709855"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def need(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    def reject(value):
        raise ValueError("Nonfinite JSON value: " + value)
    return json.loads(Path(path).read_text(encoding="utf8"), parse_constant=reject)


def load_stage(root=ROOT):
    path = root / HERE / "audit_easy_run.py"
    need(sha(path) == STAGE_SHA, "Pinned finished stage auditor changed")
    spec = importlib.util.spec_from_file_location("overall_easy20_pinned_stage_audit", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def complete_gate(experiment, stage):
    """Before any checkpoint/test/journal read, every stage must be complete."""
    for name, count in stage.COUNTS.items():
        run = experiment / name
        need((run / "summary.json").is_file() and (run / "job_state.json").is_file(),
             "All6/4/10 stages must complete before final audit")
        summary, job = read(run / "summary.json"), read(run / "job_state.json")
        need(summary.get("complete") is True and summary.get("episodes") == count
             and summary.get("stage") == name and job.get("state") == "complete" and job.get("pid") is None
             and not (run / "worker.lock").exists(), "All6/4/10 stages must complete before final audit")


def checkpoint(path, role, offset, train, experiment, stage, root=ROOT):
    model = read(path)
    need(model["kind"] == "easy20_internal_stop_offset_v1" and model["role"] == role
         and model["task_kind"] == "fixed20ml_absolute1ml_24s_onegrain_v1"
         and model["policy"] == {"control_target_offset_ml": offset}
         and model["is_neural_policy"] is False and model["default_gui_deployment"] is False,
         "Fixed20 scalar calibration checkpoint kind/units/policy differs")
    need(model["training_protocol_id"] == train["protocol_id"]
         and model["training_journal_sha256"] == train["input_artifact_sha256"]["episodes.jsonl"]
         and model["training_audit_sha256"] == sha(experiment / "train/audit.json")
         and model["source_sha256"] == train["source_sha256"], "Checkpoint no longer binds only training evidence")
    selected = "offset_" + format(offset, "g") if role == "candidate" else "offset_0"
    need(model["selected_group"] == selected, "Checkpoint selection group changed")
    selector = root / HERE / "select_stage.py"
    need(model["selection_tool_sha256"] == sha(selector), "Checkpoint selection implementation changed")
    stage.utc(model["created_at_utc"])
    return model


def expected_scores(rows, stage, roles):
    result = {}
    for role in roles:
        key = stage.score([row for row in rows if row["role"] == role])
        result[role] = [2] if key is None else list(key)
    return result


def verify_freeze_times(models, recommended, end_train, end_validation, first_validation, first_test, parse):
    for model in models.values():
        need(end_train < parse(model["created_at_utc"]) < first_validation < first_test,
             "Candidate/baseline freeze did not precede actual validation/test reset")
    need(end_validation < parse(recommended["created_at_utc"]) < first_test,
         "Recommendation freeze did not precede actual test reset")


def audit(experiment, root=ROOT):
    experiment, root = Path(experiment).resolve(), Path(root).resolve()
    stage = load_stage(root)
    stage.validate_experiment_plan(experiment)
    complete_gate(experiment, stage)
    # Train rows alone select the finite stop offset. All raw numerical and
    # physical gates are recomputed by the pinned stage auditor first.
    train = stage.audit_stage(experiment / "train", "train", root=root, parent_plan_sha=stage.PLAN_SHA)
    offset = stage.choose_offset(train["rows"])
    models = {role: checkpoint(experiment / (role + ".json"), role,
        offset if role == "candidate" else 0., train, experiment, stage, root) for role in ("baseline", "candidate")}
    roles = ("offset_0", "offset_0.5", "offset_1")
    scores = expected_scores(train["rows"], stage, roles)
    need(all(model["selection_scores"] == scores for model in models.values()),
         "Training checkpoint rank differs from independent all-case ranking")
    stages = {"train": train,
        "validation": stage.audit_stage(experiment / "validation", "validation", offset, root=root, parent_plan_sha=stage.PLAN_SHA),
        "test": stage.audit_stage(experiment / "test", "test", offset, root=root, parent_plan_sha=stage.PLAN_SHA)}
    for name, report in stages.items():
        cached = read(experiment / name / "audit.json")
        need(cached["complete"] is True and cached["pass"] is True and cached["stage"] == name
             and cached["auditor_source_sha256"] == STAGE_SHA, "Stage audit metadata/implementation differs")
        for key, expected in report.items():
            if key != "rows":
                need(cached[key] == expected, "Recomputed original stage evidence differs from prior audit: " + name + "." + key)
        snapshot = experiment / name / "audit_source_snapshot" / ("audit_easy_run_" + STAGE_SHA + ".py")
        need(sha(snapshot) == STAGE_SHA, "Stage audit source snapshot changed")
        need(report["source_sha256"] == train["source_sha256"]
             and report["input_sha256"] == train["input_sha256"]
             and report["gui_sha256"] == train["gui_sha256"] and report["versions"] == train["versions"],
             "Physical implementation/inputs/GUI/runtime changed between stages")
    recommended_role = stage.choose_validation_role(stages["validation"]["rows"])
    recommended = read(experiment / "recommended.json")
    need(recommended["kind"] == "easy20_validation_recommendation_v1"
         and recommended["task_kind"] == "fixed20ml_absolute1ml_24s_onegrain_v1"
         and recommended["recommended_role"] == recommended_role
         and recommended["policy"] == models[recommended_role]["policy"]
         and recommended["checkpoint_sha256"] == sha(experiment / (recommended_role + ".json"))
         and recommended["test_used_for_selection"] is False, "Recommendation was not validation-only")
    validation = stages["validation"]
    need(recommended["validation_protocol_id"] == validation["protocol_id"]
         and recommended["validation_journal_sha256"] == validation["input_artifact_sha256"]["episodes.jsonl"]
         and recommended["validation_audit_sha256"] == sha(experiment / "validation/audit.json")
         and recommended["selection_scores"] == expected_scores(validation["rows"], stage, ("baseline", "candidate"))
         and recommended["selection_tool_sha256"] == sha(root / HERE / "select_stage.py"),
         "Recommendation evidence/score/tool no longer matches completed validation")
    all_reports = [report for name in stage.COUNTS for report in stages[name]["episode_reports"]]
    need(len(all_reports) == len({report["episode_id"] for report in all_reports}) == 20,
         "Total physical case budget is not exactly twenty unique attempts")
    checkpoint_meta = {role: {"file": role + ".json", "sha256": sha(experiment / (role + ".json")),
        "created_at_utc": model["created_at_utc"], "policy": model["policy"]} for role, model in
        {**models, "recommended": recommended}.items()}
    # Both roles are frozen after training and before the first real validation
    # reset. The recommendation is frozen only after validation, before test.
    end_train = max(stage.utc(row["completed_at_utc"]) for row in train["rows"])
    end_val = max(stage.utc(row["completed_at_utc"]) for row in validation["rows"])
    first_val = min(stage.utc(report["actual_reset_started_utc"]) for report in validation["episode_reports"])
    first_test = min(stage.utc(report["actual_reset_started_utc"]) for report in stages["test"]["episode_reports"])
    verify_freeze_times(models, recommended, end_train, end_val, first_val, first_test, stage.utc)
    for name in ("validation", "test"):
        matrix = read(experiment / name / "matrix.json")
        need(matrix["profile_sha256"] == {role: checkpoint_meta[role]["sha256"] for role in models}
             and matrix["selection_tool_sha256"] == sha(root / HERE / "select_stage.py"),
             "Validation/test matrix did not freeze the same two profiles")
        if name == "test":
            need(matrix["recommendation_sha256"] == checkpoint_meta["recommended"]["sha256"],
                 "Test matrix changed recommendation after model freeze")
    selected_test = [report for report in stages["test"]["episode_reports"] if report["role"] == recommended_role]
    return {"complete": True, "pass": True, "kind": stage.KIND, "scope": stage.SCOPE,
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "experiment_plan_sha256": stage.PLAN_SHA,
        "budgets": {**stage.COUNTS, "total": 20}, "checkpoints": checkpoint_meta,
        "selection": {"candidate_offset_ml": offset, "recommended_role": recommended_role,
            "validation_only": True, "test_used_for_selection": False,
            "candidate_equals_baseline": models["candidate"]["policy"] == models["baseline"]["policy"],
            "training_scores": scores, "validation_scores": recommended["selection_scores"]},
        "all_physical_sources_inputs_gui_unchanged": True,
        "stage_artifact_sha256": {name: {**report["input_artifact_sha256"], "audit.json": sha(experiment / name / "audit.json")}
            for name, report in stages.items()},
        "stage_reports": {name: {key: value for key, value in report.items() if key != "rows"}
            for name, report in stages.items()}, "episode_reports": all_reports,
        "recommended_test_passes": sum(report["easy_assessment"]["feasible"] for report in selected_test),
        "recommended_test_cases": 5, "all_recommended_test_cases_passed": all(report["easy_assessment"]["feasible"] for report in selected_test),
        "original_strict_task_was_not_relaxed_or_relabelled": True,
        "default_gui_deployment": False, "all_continuous_goals_guaranteed": False,
        "auditor_source_file": HERE + "audit_easy_experiment.py", "auditor_source_sha256": sha(__file__),
        "pinned_stage_audit_sha256": STAGE_SHA}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", "--experiment", dest="experiment", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.experiment, args.workspace)
    with args.output.open("x", encoding="utf8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({key: value for key, value in result.items() if key not in ("stage_reports", "episode_reports")}, indent=2))


if __name__ == "__main__":
    main()
