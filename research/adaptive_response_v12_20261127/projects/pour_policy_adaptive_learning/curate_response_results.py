"""Curate completed audited response identification, without publishing raw logs.

Collection fit metrics are separate from terminal goal-policy acceptance. This
post-hoc CPU reader never advances physics, changes a model, or rewrites a run.
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

import cycle_response_model
import fit_cycle_model_cli


GOAL_AUDIT_SCOPE = "record_integrity_and_independent_physical_motion_mass_goal_audit_not_task_success"


def need(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    value = json.loads(Path(path).read_text(encoding="utf8"))
    json.dumps(value, allow_nan=False)
    return value


def jsonl(path):
    values = [json.loads(line) for line in Path(path).read_text(encoding="utf8").splitlines() if line.strip()]
    json.dumps(values, allow_nan=False)
    return values


def finite(value, name):
    need(type(value) in (int, float) and math.isfinite(value), "Finite number required: " + name)
    return float(value)


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def compact_episode(row):
    result, gate = row["result"], row["strict_assessment"]
    target, received = finite(row["target_ml"], "target"), finite(result["received_ml"], "received")
    need(target == result["target_ml"] and 20 <= target <= 40, "Original target changed")
    tolerance = .001 * target
    dose = target - tolerance <= received <= target + tolerance
    need(gate["dose_passed"] is dose, "Recorded strict dose gate differs from closed original target window")
    feasible = bool(gate["valid_evidence"] and dose and gate["full_home_within_deadline"]
                    and gate["spill_passed"] and gate["no_late_dose_change"])
    need(gate["feasible"] is feasible, "Recorded joint acceptance differs")
    decisions = row.get("status", {}).get("response_cycle_model_decisions", [])
    abstentions = [decision for decision in decisions if decision.get("mode") == "abstain"]
    return {"episode_id": row["episode_id"], "split": row["split"], "seed": row["seed"],
            "target_ml": target, "received_ml": received,
            "signed_error_ml": received - target, "absolute_error_ml": abs(received - target),
            "signed_error_percent": (received - target) / target * 100., "tolerance_ml": tolerance,
            "physical_action_seconds": finite(result["physical_time_s"], "physical action time"),
            "wall_seconds_including_post": finite(row["wall_seconds"], "compute wall time"),
            "post_physical_seconds": finite(row["post_observation"]["duration_s"], "post duration"),
            "peak_all_label_spill_particles": gate["maximum_all_label_spilled_particles"],
            "valid_evidence": gate["valid_evidence"], "dose_passed": dose,
            "full_home_within_deadline": gate["full_home_within_deadline"],
            "spill_passed": gate["spill_passed"], "no_late_dose_change": gate["no_late_dose_change"],
            "strict_task_feasible": feasible, "model_decision_count": len(decisions),
            "model_abstention_count": len(abstentions),
            "model_abstention_reasons": dict(Counter(decision.get("reason", "unspecified") for decision in abstentions))}


def episode_metrics(rows):
    need(rows, "A completed study cannot have no cases")
    result = {"cases": len(rows), "joint_successes": sum(row["strict_task_feasible"] for row in rows),
              "mae_ml": statistics.mean(row["absolute_error_ml"] for row in rows),
              "worst_absolute_error_ml": max(row["absolute_error_ml"] for row in rows),
              "mean_physical_action_seconds": statistics.mean(row["physical_action_seconds"] for row in rows),
              "max_physical_action_seconds": max(row["physical_action_seconds"] for row in rows),
              "mean_wall_seconds_including_post": statistics.mean(row["wall_seconds_including_post"] for row in rows),
              "max_peak_all_label_spill_particles": max(row["peak_all_label_spill_particles"] for row in rows),
              "model_abstention_cases": sum(row["model_abstention_count"] > 0 for row in rows),
              "model_abstention_decisions": sum(row["model_abstention_count"] for row in rows)}
    result["failure_counts_overlapping"] = {name: sum(not row[name] for row in rows) for name in
        ("valid_evidence", "dose_passed", "full_home_within_deadline", "spill_passed", "no_late_dose_change")}
    return result


def compact_cycle(row):
    return {key: row[key] for key in
        ("row_id", "episode_id", "seed", "split", "target_ml", "safe", "settled",
         "action_execution_complete", "peak_spill_particles", "state", "action", "outcome",
         "pre_physical_time_s", "post_physical_time_s", "includes_approach_water",
         "episode_safety_vetoes", "home_additional_received_ml", "episode_no_late_dose_change")}


def validation_points(model, validation):
    selected = model["configuration_selection"][model["selected_config_index"]]["validation"]
    actual = {row["row_id"]: row for row in validation}
    need(set(actual) == {row["row_id"] for row in selected["rows"]}, "Validation records differ from selected model")
    points = []
    for record in selected["rows"]:
        row, prediction = actual[record["row_id"]], record["prediction"]
        point = {"row_id": row["row_id"], "seed": row["seed"], "target_ml": row["target_ml"],
                 "actual_delta_ml": row["outcome"]["settled_delta_ml"],
                 "actual_duration_s": row["outcome"]["duration_s"],
                 "safe": row["safe"], "settled": row["settled"],
                 "action_execution_complete": row["action_execution_complete"],
                 "supported": prediction["supported"], "reason": prediction["reason"]}
        if prediction["supported"]:
            point.update(mean_predicted_delta_ml=prediction["mean_delta_ml"],
                         predicted_lower_delta_ml=prediction["lower_delta_ml"],
                         predicted_upper_delta_ml=prediction["upper_delta_ml"],
                         signed_prediction_error_ml=prediction["mean_delta_ml"] - point["actual_delta_ml"],
                         absolute_prediction_error_ml=record["absolute_delta_error_ml"],
                         independent_support_seeds=prediction["independent_seed_groups"])
        points.append(point)
    return points


def load_study(run, audit_path, audit_sha256, fit):
    """Reject incomplete/unbound runs before producing a final publication."""
    validated = fit_cycle_model_cli.validate_audited_collection(run, audit_path, audit_sha256)
    fit = Path(fit).resolve()
    manifest, candidate_path = read(fit / "fit_manifest.json"), fit / "candidate.json"
    need(manifest.get("complete") is True
         and manifest.get("kind") == "audited_whole_cycle_response_model_fit_v1", "Complete response fit required")
    need(manifest.get("independent_audit_raw_sha256") == audit_sha256
         and manifest.get("collection_protocol_id") == validated["protocol"]["protocol_id"]
         and manifest.get("collection_artifact_sha256") == validated["artifact_sha256"],
         "Frozen fit does not bind the current audited collection")
    model = cycle_response_model.load_response_model(candidate_path, manifest["candidate_raw_sha256"])
    need(model["content_sha256"] == manifest["model_content_sha256"]
         and model["source_sha256"] == sha(cycle_response_model.__file__), "Model implementation/content differs")
    need(model["provenance"]["cycle_projection_sha256"] == validated["cycle_projection_sha256"]
         == manifest["cycle_projection_sha256"], "Whole-cycle data projection changed")
    need(sha(fit / "cycle_responses.jsonl") == manifest["cycle_responses_raw_sha256"]
         and sha(fit / "partial_cycles.json") == manifest["partial_cycles_raw_sha256"], "Fit dataset artifacts changed")
    need(jsonl(fit / "cycle_responses.jsonl") == validated["cycle_projection"]["samples"]
         and read(fit / "partial_cycles.json") == validated["cycle_projection"]["partial_cycles"],
         "Frozen fit dataset differs from independently audited events")
    need(model["selection_kind"] == "validation_only" and manifest["test_data_used"] is False
         and manifest["dataset_splits_reassigned"] is False, "Fit selection or dataset use differs")
    episodes = jsonl(validated["run"] / "episodes.jsonl")
    return {**validated, "fit": fit, "fit_manifest": manifest, "model": model,
            "episodes": episodes, "fit_manifest_raw_sha256": sha(fit / "fit_manifest.json")}


def load_goal_evaluation(directory, audit_path, audit_sha256, study):
    """Optional independent frozen-model task evaluation; never infer missing acceptance."""
    run = Path(directory).resolve()
    summary = read(run / "summary.json")
    need(summary.get("complete") is True and summary.get("frozen_response_model_used") is True,
         "Goal evaluation is not complete")
    need(sha(audit_path) == audit_sha256, "Goal audit raw SHA mismatch")
    audit = read(audit_path)
    need(audit.get("complete") is True and audit.get("pass") is True
         and audit.get("scope") == GOAL_AUDIT_SCOPE, "Independent goal evaluation audit required")
    bindings = audit.get("input_artifact_sha256", {})
    for name in ("protocol.json", "summary.json", "job_state.json", "episodes.jsonl", "matrix.json", "versions.json"):
        need(bindings.get(name) == sha(run / name), "Goal evaluation artifact changed: " + name)
    protocol, job, matrix = read(run / "protocol.json"), read(run / "job_state.json"), read(run / "matrix.json")
    need(protocol.get("kind") == "adaptive_frozen_response_goal_evaluation_v12"
         and protocol["protocol_id"] == summary["protocol_id"] == audit["protocol_id"],
         "Unexpected frozen goal evaluation protocol")
    need(job.get("state") == "complete" and job.get("pid") is None and not (run / "worker.lock").exists(),
         "Goal evaluation worker is not finalized")
    need(protocol.get("deadline_s") == 24. and protocol.get("relative_tolerance") == .001
         and protocol.get("peak_all_label_spill_limit") == 1, "Goal acceptance contract changed")
    rows = jsonl(run / "episodes.jsonl")
    need(len(rows) == summary["episodes"] == audit["episodes_audited"] == len(matrix["cases"]),
         "Goal evaluation case budget is incomplete")
    need([row["episode_id"] for row in rows] == [case["case_id"] for case in matrix["cases"]]
         and len({row["episode_id"] for row in rows}) == len(rows), "Goal case identities differ")
    candidate_sha = study["fit_manifest"]["candidate_raw_sha256"]
    need(audit.get("model_checkpoint_sha256") == candidate_sha, "Goal audit bound another candidate")
    fitted_seeds = set(study["model"]["training_seeds"] + study["model"]["validation_seeds"])
    need(not fitted_seeds & {row["seed"] for row in rows}, "Goal evaluation seeds leaked into fitting")
    freeze = datetime.fromisoformat(study["model"]["created_at_utc"])
    for row in rows:
        need(row["declared_policy"]["model_sha256"] == candidate_sha,
             "Goal case used another response model")
        need(datetime.fromisoformat(row["started_at_utc"]) > freeze,
             "Goal case predates the frozen response model")
    compact = [compact_episode(row) for row in rows]
    integer_targets = set(range(20, 41))
    decimal_targets = {20.375, 22.625, 26.125, 29.875, 34.375, 38.625}
    by_target = {target: [row for row in compact if row["target_ml"] == target]
                 for target in {row["target_ml"] for row in compact}}
    coverage = {"all_21_integer_targets": integer_targets <= set(by_target),
                "all_6_predeclared_decimal_targets": decimal_targets <= set(by_target),
                "exact_54_heldout_cases": len(compact) == 54,
                "all_cases_explicit_test_split": all(row["split"] == "test" for row in compact),
                "exactly_two_distinct_seeds_per_target": all(len(group) == 2
                    and len({row["seed"] for row in group}) == 2 for group in by_target.values())}
    coverage["full_predeclared_range_suite"] = all(coverage.values())
    return {"complete": True, "protocol_id": protocol["protocol_id"],
            "audit_raw_sha256": audit_sha256, "candidate_raw_sha256": candidate_sha,
            "metrics": episode_metrics(compact), "cases": compact,
            "per_goal": [{"target_ml": target, **episode_metrics([row for row in compact if row["target_ml"] == target])}
                         for target in sorted({row["target_ml"] for row in compact})],
            "all_tested_cases_jointly_passed": all(row["strict_task_feasible"] for row in compact),
            "coverage": coverage,
            "all_continuous_goals_guaranteed": False}


def build_public_summary(study, goal_evaluation=None):
    """Select compact, path-free actual results; preserve every collection case."""
    protocol, model, manifest = study["protocol"], study["model"], study["fit_manifest"]
    rows = [compact_episode(row) for row in study["episodes"]]
    samples = [compact_cycle(row) for row in study["cycle_projection"]["samples"]]
    partials = [{key: row[key] for key in
                ("row_id", "episode_id", "seed", "split", "state", "action", "pre_physical_time_s",
                 "settled_outcome_available", "reason")} for row in study["cycle_projection"]["partial_cycles"]]
    targets_by_episode = {row["episode_id"]: row["target_ml"] for row in rows}
    for partial in partials:
        partial["target_ml"] = targets_by_episode[partial["episode_id"]]
    for row in rows:
        row["complete_response_cycles"] = sum(sample["episode_id"] == row["episode_id"] for sample in samples)
        row["partial_response_cycles"] = sum(partial["episode_id"] == row["episode_id"] for partial in partials)
        row["no_response_cycle_started"] = row["complete_response_cycles"] == row["partial_response_cycles"] == 0
    completed = [row["completed_at_utc"] for row in study["episodes"]]
    selected = model["configuration_selection"][model["selected_config_index"]]["validation"]
    output = {"complete": True, "kind": "audited_real_whole_cycle_response_identification_v12",
              "record_created_at_utc": datetime.now(timezone.utc).isoformat(),
              "actual_collection_started_at_utc": min(row["started_at_utc"] for row in study["episodes"]),
              "actual_collection_completed_at_utc": max(completed),
              "protocol_id": protocol["protocol_id"],
              "independent_audit_raw_sha256": manifest["independent_audit_raw_sha256"],
              "collection_artifact_sha256": study["artifact_sha256"],
              "source_sha256": protocol["source_sha256"],
              "input_fingerprints_sha256": sorted(protocol["input_sha256"].values()),
              "gui_policy_sha256_unchanged": protocol["gui_sha256"],
              "task_contract": {"deadline_physical_s": 24., "relative_dose_tolerance": .001,
                  "max_all_label_peak_spill_particles": 1, "full_home_quiet_frames": 12,
                  "post_observation_physical_s": 2., "post_can_rescue_deadline": False},
              "physics": {"solver": protocol["solver"], "measurement": protocol["measurement"],
                  "particle_volume_ml": protocol["particle_ml"], "total_particles": protocol["total_particles"],
                  "source_particles": protocol["source_particles"], "initial_source_ml": protocol["initial_source_ml"],
                  "cuda_graph": False, "robot_grasp": "kinematic_pregrasp_not_dynamic_force_grasp"},
              "collection_metrics": episode_metrics(rows), "cases": rows,
              "response_samples": samples, "partial_cycles": partials,
              "response_sample_counts": {"total_completed_endpoints": len(samples), "partial": len(partials),
                  "cases_without_any_cycle_started": sum(row["no_response_cycle_started"] for row in rows),
                  "fit_eligible": sum(row["safe"] and row["settled"] and row["action_execution_complete"]
                      and row["outcome"]["settled_delta_ml"] >= 0 for row in samples),
                  "zero_dose": sum(row["outcome"]["settled_delta_ml"] == 0 for row in samples),
                  "unsafe": sum(not row["safe"] for row in samples),
                  "unsettled": sum(not row["settled"] for row in samples),
                  "skipped_or_incomplete_branch": sum(not row["action_execution_complete"] for row in samples),
                  "negative_dose": sum(row["outcome"]["settled_delta_ml"] < 0 for row in samples)},
              "response_model": {"kind": model["kind"], "candidate_raw_sha256": manifest["candidate_raw_sha256"],
                  "model_content_sha256": model["content_sha256"], "model_source_sha256": model["source_sha256"],
                  "fit_manifest_raw_sha256": study["fit_manifest_raw_sha256"],
                  "cycle_projection_sha256": manifest["cycle_projection_sha256"],
                  "selection_kind": model["selection_kind"], "configuration": model["config"],
                  "training_rows": model["training_row_count"], "validation_rows": model["validation_row_count"],
                  "partial_training_cycles": model["partial_training_cycles"],
                  "partial_validation_cycles": model["partial_validation_cycles"],
                  "training_seeds": model["training_seeds"], "validation_seeds": model["validation_seeds"],
                  "validation_complete_local_support": model["validation_complete_local_support"],
                  "validation_all_responses_valid": model["validation_all_responses_valid"],
                  "validation_predicted_rows": selected["predicted_rows"],
                  "validation_abstained_rows": selected["abstained_rows"],
                  "validation_mae_ml_on_supported_rows": selected["mean_absolute_delta_error_ml"],
                  "validation_worst_ml_on_supported_rows": selected["maximum_absolute_delta_error_ml"],
                  "validation_estimates_are_before_residual_envelope_calibration": True,
                  "validation_points": validation_points(model, study["validation"])},
              "goal_evaluation": goal_evaluation,
              "terminal_goal_policy_acceptance_proven": bool(goal_evaluation
                  and goal_evaluation["all_tested_cases_jointly_passed"]
                  and goal_evaluation["coverage"]["full_predeclared_range_suite"]),
              "default_gui_deployment": False, "all_continuous_goals_guaranteed": False,
              "limitations": ["Response system identification is not terminal-policy acceptance.",
                  "Every small cycle includes approach drainage and cutoff tail, measured after actual quiet.",
                  "Response envelopes are empirical, not certified bounds; unsupported states abstain.",
                  "Privileged particle truth and kinematic grasp do not prove scale/torque/real-robot performance.",
                  "Finite tests cannot guarantee every continuous target or initial condition."]}
    reject_private_paths(output)
    return output


def reject_private_paths(value):
    if isinstance(value, str):
        lowered = value.lower().replace("\\", "/")
        need(not any(token in lowered for token in ("c:/", "d:/", "/users/", "/home/", "file://")),
             "A private absolute path would enter the public summary")
    elif isinstance(value, dict):
        for key, item in value.items():
            reject_private_paths(key)
            reject_private_paths(item)
    elif isinstance(value, list):
        for item in value:
            reject_private_paths(item)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--audit-sha256", required=True)
    parser.add_argument("--fit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--goal-run", type=Path)
    parser.add_argument("--goal-audit", type=Path)
    parser.add_argument("--goal-audit-sha256")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args(argv)
    study = load_study(args.run, args.audit, args.audit_sha256, args.fit)
    goal = None
    if any(value is not None for value in (args.goal_run, args.goal_audit, args.goal_audit_sha256)):
        need(all(value is not None for value in (args.goal_run, args.goal_audit, args.goal_audit_sha256)),
             "All three optional goal-evaluation arguments are required")
        goal = load_goal_evaluation(args.goal_run, args.goal_audit, args.goal_audit_sha256, study)
    result = build_public_summary(study, goal)
    if not args.check_only:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf8") as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
            stream.write("\n")
    print(json.dumps({"complete": True, "check_only": args.check_only,
        "collection_cases": result["collection_metrics"]["cases"],
        "response_samples": result["response_sample_counts"],
        "terminal_goal_policy_acceptance_proven": result["terminal_goal_policy_acceptance_proven"]}, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, TypeError) as error:
        print("Response curation refused: " + str(error), file=sys.stderr)
        raise SystemExit(1)
