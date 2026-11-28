"""Curate the audited fixed-policy pilot and ten new paired retry tests.

No training, validation, test-based selection, physics or GUI work occurs here.
The known previous test seed used for the pilot is explicitly development data.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path


KIND = "retry20_conditional_second_cycle_v1"
COUNTS = {"pilot": 1, "test": 20}
SEEDS = {"pilot": {195301}, "test": set(range(196301, 196311))}
TOP = ("protocol.json", "summary.json", "job_state.json", "episodes.jsonl",
       "responses.jsonl", "matrix.json", "versions.json", "audit.json")


def need(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    def reject(value):
        raise ValueError("Nonfinite JSON: " + value)
    return json.loads(Path(path).read_text(encoding="utf8"), parse_constant=reject)


def safe_path(root, relative):
    need(isinstance(relative, str) and relative, "Artifact needs a relative name")
    destination = (root / relative).resolve()
    need(destination.is_relative_to(root.resolve()), "Artifact escapes the experiment")
    return destination


def close(left, right, message, tolerance=1e-9):
    need(type(left) in (int, float) and type(right) in (int, float)
         and math.isfinite(left) and math.isfinite(right)
         and abs(left - right) <= tolerance, message)


def metrics(cases):
    need(cases, "Empty metric group")
    return {"cases": len(cases), "valid_evidence_cases": sum(case["valid_evidence"] for case in cases),
        "joint_passes": sum(case["refine_feasible"] for case in cases),
        "easy_joint_passes": sum(case["easy_feasible"] for case in cases),
        "original_strict_joint_passes": sum(case["original_strict_feasible"] for case in cases),
        "mean_absolute_error_ml": sum(case["absolute_error_ml"] for case in cases) / len(cases),
        "maximum_absolute_error_ml": max(case["absolute_error_ml"] for case in cases),
        "mean_signed_error_ml": sum(case["error_ml"] for case in cases) / len(cases),
        "mean_physical_action_seconds": sum(case["physical_action_seconds"] for case in cases) / len(cases),
        "maximum_physical_action_seconds": max(case["physical_action_seconds"] for case in cases),
        "mean_compute_wall_seconds_including_post": sum(case["compute_wall_seconds_including_post"] for case in cases) / len(cases),
        "mean_action_compute_wall_seconds": sum(case["action_compute_wall_seconds"] for case in cases) / len(cases),
        "maximum_peak_all_label_spill_particles": max(case["peak_all_label_spill_particles"] for case in cases),
        "full_home_within_deadline_cases": sum(case["full_home_within_deadline"] for case in cases),
        "no_late_dose_change_cases": sum(case["no_late_dose_change"] for case in cases),
        "actual_second_cycle_cases": sum(case["actual_second_cycle_started"] for case in cases),
        "retry_decision_cases": sum(bool(case["retry_decisions"]) for case in cases),
        "retry_allowed_cases": sum(case["retry_allowed"] for case in cases)}


def compact_case(row, stage, report):
    result, status = row["result"], row["status"]
    refine, easy, strict = row["refine_assessment"], row["easy_assessment"], row["strict_assessment"]
    need(row["stage"] == stage and row["split"] == ("development" if stage == "pilot" else "test"),
         "Pilot development and new test labels differ")
    close(row["target_ml"], 20., "Actual target changed")
    close(result["target_ml"], 20., "Result target changed")
    received, error = result["received_ml"], result["received_ml"] - 20.
    close(result["error_ml"], error, "Actual signed error differs")
    close(result["absolute_error_ml"], abs(error), "Actual absolute error differs")
    for name, assessment, tolerance in (("refine", refine, .5), ("easy", easy, 1.), ("strict", strict, .02)):
        close(assessment["tolerance_ml"], tolerance, "Task criterion changed: " + name)
        need(type(assessment["valid_evidence"]) is bool, "Physical evidence validity must be explicit")
        dose = 20. - tolerance <= received <= 20. + tolerance
        need(assessment["dose_passed"] is dose, "Closed actual dose gate differs: " + name)
        expected = bool(assessment["valid_evidence"] and dose and assessment["full_home_within_deadline"]
                        and assessment["spill_passed"] and assessment["no_late_dose_change"])
        need(assessment["feasible"] is expected, "Joint acceptance differs: " + name)
        need(report[name + "_assessment"] == assessment, "Independent assessment differs: " + name)
    need(report["episode_id"] == row["episode_id"], "Audited case identity differs")
    for key in ("stage", "role", "seed"):
        need(report[key] == (stage if key == "stage" else row[key]), "Audited case label differs: " + key)
    for key in ("received_ml", "error_ml"):
        close(report[key], result[key], "Audited measurement differs: " + key)
    if "physical_s" in report:
        close(report["physical_s"], result["physical_time_s"], "Audited physical time differs")
    peak = refine["maximum_all_label_spilled_particles"]
    need(type(peak) is int and peak >= 0 and refine["spill_passed"] is (peak <= 1), "Invalid all-label spill count")
    close(result["physical_time_s"], result["physical_steps"] / 60., "Physical clock differs from step/60")
    post = row["post_observation"]
    close(post["duration_s"], 2., "Post observation is not two physical seconds", 1e-8)
    need(post["physical_steps"] == 120, "Post observation must retain120 live steps")
    policy = row["declared_policy"]
    need(set(policy) == {"enable_retry"} and type(policy["enable_retry"]) is bool,
         "Only the declared conditional-retry flag may vary")
    enabled = policy["enable_retry"]
    need(status["retry_enabled"] is enabled and status["retry_maximum_cycles"] == (2 if enabled else 1),
         "Executed retry mode or cycle budget differs")
    close(status["retry_first_approach_tilt_deg"], 84., "First angle changed")
    close(status["retry_second_approach_tilt_deg"], 84.1, "Second angle changed")
    starts = [event for event in row["response_events"] if event["type"] == "response_cycle_start"]
    ends = [event for event in row["response_events"] if event["type"] in ("response_cycle_complete", "response_cycle_unsettled")]
    need(len(starts) <= (2 if enabled else 1) and len(ends) <= len(starts)
         and status["response_cycle_count"] == len(starts), "Fixed first/conditional-second cycle budget changed")
    decisions = status["retry_decisions"]
    need(isinstance(decisions, list) and len(decisions) <= 1 and (enabled or not decisions),
         "Retry decision budget or baseline mode differs")
    allowed = bool(decisions and decisions[0]["retry_allowed"])
    for decision in decisions:
        close(decision["actual_deficit_ml"], 20. - decision["actual_received_ml"], "Actual retry deficit differs")
        close(decision["declared_retry_approach_tilt_deg"], 84.1, "Retry angle changed at decision")
        observed_peak = decision["observed_all_label_peak_spill_particles"]
        safe_spill = type(observed_peak) is int and 0 <= observed_peak <= 1
        need(decision["spill_guard_passed"] is safe_spill, "Actual retry spill guard differs")
        t, budget = decision["physical_time_s"], decision["conservative_full_home_budget_s"]
        close(decision["deadline_physical_seconds"], 24., "Retry physical deadline changed")
        close(decision["remaining_deadline_physical_seconds"], 24. - t, "Retry remaining physical time differs")
        fits = type(budget) in (int, float) and math.isfinite(budget) and budget >= 0. and t + budget <= 24.
        need(decision["full_home_budget_fits"] is fits, "Actual retry full-home time guard differs")
        expected = bool(decision["actual_settled_measurement"]
            and decision["actual_received_ml"] < 19.5 and decision["spill_guard_passed"]
            and decision["full_home_budget_fits"])
        need(decision["retry_allowed"] is expected, "Retry must use actual settled undershoot, spill and full-home guards")
        need(decision["completed_cycles_before_decision"] == 1
             and decision["deficit_exceeds_half_ml"] is (decision["actual_received_ml"] < 19.5)
             and decision["budget_is_robot_motion_not_predicted_water_dose"] is True,
             "Second-cycle decision did not follow one actual cycle")
    need(len(starts) < 2 or allowed, "Second cycle lacks an allowed measured-feedback decision")
    anchor = status["prefix_anchor_observation"]
    close(anchor["actual_target_ml"], 20., "Actual anchor target changed")
    close(anchor["control_target_ml"], 20., "Coarse target was offset")
    audited_retry = report["actual_retry_evidence"]
    completed = sum(event["type"] == "response_cycle_complete" for event in ends)
    need(audited_retry["first_coarse_anchor"] == anchor
         and audited_retry["conditional_decisions"] == decisions
         and audited_retry["retry_intent_allowed"] is allowed
         and audited_retry["actual_cycles_started"] == len(starts)
         and audited_retry["actual_second_cycle_executed"] is (len(starts) == 2)
         and audited_retry["actual_cycles_completed"] == completed
         and audited_retry["actual_second_cycle_complete"] is (completed == 2),
         "Independent actual retry/cycle evidence differs")
    cycles = []
    for index, start in enumerate(starts):
        expected_angle = 84. if index == 0 else 84.1
        close(start["approach_tilt_deg"], expected_angle, "Actual approach angle sequence changed")
        cycle = {"cycle_number": index + 1, "approach_tilt_deg": expected_angle,
            "start_physical_s": start["physical_time_s"],
            "actual_start_received_ml": start["observation"]["received_ml"],
            "has_recorded_endpoint": index < len(ends)}
        if index < len(ends):
            end = ends[index]
            cycle.update(end_physical_s=end["physical_time_s"],
                actual_end_received_ml=end["post_observation"]["received_ml"], outcome=end["outcome"],
                settled=end["settled"], action_execution_complete=end["action_execution_complete"],
                completion_event_type=end["type"], executed_micro=end["executed_micro"],
                micro_skipped=not end["executed_micro"], completion_reason=end.get("completion_reason"))
        cycles.append(cycle)
    known = None
    if stage == "pilot":
        old = audited_retry["seen_case_first_cycle_reference"]
        need(old["seen_development_case_not_independent"] is True,
             "Seen pilot reference cannot become independent test evidence")
        known = {key: old[key] for key in ("original_failed_received_ml", "first_cycle_end_physical_s",
            "first_cycle_frames", "first_cycle_water_motion_partitions_exactly_equal", "first_cycle_original_events_exactly_equal",
            "all_completed_frame_substep_counts_exactly_equal", "zero_time_inherited_last_substeps")}
        known["reference_artifact_sha256"] = sorted(old["source_reference_sha256"].values())
    return {"episode_id": row["episode_id"], "stage": stage, "split": row["split"], "role": row["role"],
        "seed": row["seed"], "target_ml": 20., "retry_enabled": enabled,
        "known_previous_test_seed_used_as_development": stage == "pilot",
        "actual_correction_cycles_started": len(starts), "actual_correction_cycles_ended": len(ends),
        "actual_second_cycle_started": len(starts) == 2, "retry_allowed": allowed, "retry_decisions": decisions,
        "known_pilot_reference": known,
        "retry_actual_settled_home_decision": status["retry_actual_settled_home_decision"],
        "correction_cycle_completion_reason": status.get("response_cycle_terminal_reason"),
        "actual_coarse_cut_tilt_degrees": anchor["coarse_cut_tilt_deg"],
        "first_quiet_anchor_received_ml": anchor["received_ml"],
        "first_quiet_anchor_physical_seconds": anchor["physical_time_s"],
        "first_quiet_anchor_source_rms_m_s": anchor["source_inside_velocity_rms_m_s"],
        "actual_cycles": cycles, "received_ml": received, "error_ml": error, "absolute_error_ml": abs(error),
        "refine_feasible": refine["feasible"], "easy_feasible": easy["feasible"],
        "original_strict_feasible": strict["feasible"], "original_strict_dose_passed": strict["dose_passed"],
        "valid_evidence": refine["valid_evidence"], "physical_action_seconds": result["physical_time_s"],
        "action_compute_wall_seconds": result["wall_seconds"], "compute_wall_seconds_including_post": row["wall_seconds"],
        "post_physical_seconds": post["duration_s"], "peak_all_label_spill_particles": peak,
        "full_home_within_deadline": refine["full_home_within_deadline"], "no_late_dose_change": refine["no_late_dose_change"],
        "mass_conserved": result["mass_conserved"], "all_states_finite": result["all_states_finite"],
        "started_at_utc": row["started_at_utc"], "completed_at_utc": row["completed_at_utc"]}


def pair_equivalence(records, cases, counter_explanation):
    """Expose audited pair diagnostics without claiming unavailable comparisons."""
    need(len(records) == 10 and {row["seed"] for row in records} == SEEDS["test"],
         "All ten independent first-cycle comparisons required")
    output = []
    for row in records:
        seed = row["seed"]
        group = {case["role"]: case for case in cases if case["stage"] == "test" and case["seed"] == seed}
        need(set(group) == {"baseline", "candidate"}, "First-cycle diagnostic lost a paired case")
        item = {"seed": seed, "verified": row["verified"],
            "endpoint_unavailable": row.get("endpoint_unavailable", False)}
        need(type(item["verified"]) is bool and type(item["endpoint_unavailable"]) is bool,
             "Pair diagnostic status must be explicit")
        if item["endpoint_unavailable"]:
            need(item["verified"] is False, "Missing endpoint cannot prove first-cycle equivalence")
            need(any(not case["actual_cycles"] or not case["actual_cycles"][0]["has_recorded_endpoint"]
                     for case in group.values()), "Endpoint-unavailable diagnostic contradicts actual endpoints")
            item["all_completed_frame_substep_counts_exactly_equal"] = None
        else:
            for role in ("baseline", "candidate"):
                need(row[role + "_episode_id"] == group[role]["episode_id"], "Paired diagnostic identity differs")
                need(group[role]["actual_cycles"] and group[role]["actual_cycles"][0]["has_recorded_endpoint"],
                     "Audited first endpoint is absent from selected actual cycles")
                endpoint = group[role]["actual_cycles"][0]
                if row["verified"]:
                    close(row["first_cycle_end_physical_s"], endpoint["end_physical_s"],
                          "Verified first-cycle time differs")
                item[role + "_actual_first_cycle_end_received_ml"] = endpoint["actual_end_received_ml"]
            for key in ("baseline_episode_id", "candidate_episode_id", "first_cycle_end_physical_s",
                        "first_cycle_frames", "original_first_cycle_events_exactly_equal",
                        "actual_water_motion_partitions_exactly_equal"):
                item[key] = row[key]
            need(row["verified"] is bool(row["original_first_cycle_events_exactly_equal"]
                 and row["actual_water_motion_partitions_exactly_equal"]), "Pair equivalence conjunction differs")
            if row["verified"]:
                close(item["baseline_actual_first_cycle_end_received_ml"],
                      item["candidate_actual_first_cycle_end_received_ml"], "Verified first-cycle doses differ")
            # The audited physical-equality conjunction compares solver_substeps
            # on every completed frame (step>0). A false conjunction cannot say
            # which component differed, so the substep-only result is unknown.
            item["all_completed_frame_substep_counts_exactly_equal"] = (
                True if row["actual_water_motion_partitions_exactly_equal"] else None)
            item["completed_frame_substep_equality_basis"] = "included_in_independent_physical_equality_conjunction"
            counter = dict(row["zero_time_inherited_last_substeps"])
            need(counter["is_actual_completed_frame"] is False and counter["excluded_from_equivalence_only"] is True,
                 "Pair t0 diagnostic exception may not hide completed physical frames")
            for key in ("source_file", "source_sha256", "reason"):
                counter[key] = counter_explanation[key]
            counter["explanation_uses_same_audited_frozen_solver"] = True
            item["zero_time_inherited_last_substeps"] = counter
        output.append(item)
    return output


def curate(run, audit_path, audit_digest):
    run, audit_path = Path(run).resolve(), Path(audit_path).resolve()
    need(audit_path == run / "overall_audit.json", "Use this run's complete overall audit")
    need(sha(audit_path) == audit_digest, "Independent overall audit SHA differs")
    audit = read(audit_path)
    need(audit.get("complete") is True and audit.get("pass") is True,
         "Complete passing independent evidence audit required before reading journals")
    need(audit["kind"] == KIND and audit["budgets"] == {**COUNTS, "total": 21}, "Complete21-case retry study required")
    auditor = audit["auditor_source_file"]
    need(auditor == "projects/pour_policy_retry20/audit_retry_experiment.py", "Use the independent completed-experiment auditor")
    need(sha(Path(__file__).resolve().parents[2] / auditor) == audit["auditor_source_sha256"], "Independent auditor source changed")
    need(audit["all_physical_sources_inputs_gui_unchanged"] is True
         and audit["fixed_policy_not_selected_from_test"] is True
         and audit["pilot_uses_known_previous_test_seed"] is True
         and audit["no_data_fitting_or_model_selection"] is True
         and audit["test_used_for_selection"] is False
         and audit["fixed_strategy_frozen_before_actual_test_reset"] is True,
         "Fixed-policy, source or known-pilot provenance differs")
    plan_path = run / "experiment_plan.json"
    need(sha(plan_path) == audit["experiment_plan_sha256"], "Frozen experiment plan differs")
    plan = read(plan_path)
    need(plan["budget"] == 21 and plan["draft"] is False, "Final21-case plan required")
    need(plan["budgets"] == COUNTS and plan["pilot_seed"] == 195301
         and set(plan["test_seeds"]) == SEEDS["test"] and len(plan["test_seeds"]) == 10
         and plan["pilot_is_seen_development_revisit"] is True
         and plan["data_fitting_or_model_selection"] is False
         and plan["test_may_not_select_or_update_model"] is True,
         "Fixed pilot/test scope or no-selection contract changed")
    close(plan["absolute_tolerance_ml"], .5, "Wrong primary retry criterion")
    close(plan["easy_absolute_tolerance_ml"], 1., "Wrong secondary beginner criterion")
    close(plan["original_strict_target_relative_tolerance"], .001, "Original strict criterion changed")
    for stage in COUNTS:
        hashes = audit["stage_artifact_sha256"][stage]
        need(set(hashes) == set(TOP), "All seven stage artifacts plus independent stage audit must be bound")
        for name, digest in hashes.items():
            need(sha(run / stage / name) == digest, "Audited stage artifact changed: " + stage + "/" + name)
        need(read(run / stage / "summary.json").get("complete") is True, "Incomplete stage")
        job = read(run / stage / "job_state.json")
        need(job["state"] == "complete" and job["pid"] is None and not (run / stage / "worker.lock").exists(), "Stage still active")
    strategy = audit["fixed_strategy"]
    need(strategy["file"] == "fixed_strategy.json" and sha(safe_path(run, strategy["file"])) == strategy["sha256"],
         "Fixed strategy checkpoint changed")
    frozen = read(run / strategy["file"])
    need(frozen["kind"] == "fixed20_conditional_second_cycle_strategy_v1"
         and frozen["task_kind"] == "fixed20ml_absolute05ml_24s_onegrain_v1"
         and frozen["profiles"] == {"baseline": {"enable_retry": False}, "candidate": {"enable_retry": True}}
         and frozen["no_data_fitting_or_model_selection"] is True
         and frozen["test_used_for_selection"] is False
         and frozen["pilot_uses_seen_previous_test_seed"] is True
         and frozen["pilot_joint_task_passed"] is True and frozen["default_gui_deployment"] is False,
         "Frozen strategy was changed into learned, selected or default-deployed policy")
    reports = audit["episode_reports"]
    need(len(reports) == 21 and len({row["episode_id"] for row in reports}) == 21, "Audit must include21 unique cases")
    reports = {row["episode_id"]: row for row in reports}
    cases = []
    for stage, count in COUNTS.items():
        rows = [json.loads(line) for line in (run / stage / "episodes.jsonl").read_text(encoding="utf8").splitlines() if line]
        need(len(rows) == count and {row["seed"] for row in rows} == SEEDS[stage], "Stage count or seeds differ")
        for row in rows:
            need(row["episode_id"] in reports, "Case absent from independent audit")
            need(read(run / stage / "completed" / (row["episode_id"] + ".json")) == row, "Completed sidecar differs from journal")
            for relative, digest in row["artifacts"].items():
                need(sha(safe_path(run / stage, relative)) == digest, "Raw action/post evidence changed")
            cases.append(compact_case(row, stage, reports[row["episode_id"]]))
    need(len({case["episode_id"] for case in cases}) == 21, "Duplicate case names")
    grouped = {stage: {role: metrics([case for case in cases if case["stage"] == stage and case["role"] == role])
        for role in sorted({case["role"] for case in cases if case["stage"] == stage})} for stage in COUNTS}
    paired = []
    for seed in sorted(SEEDS["test"]):
        group = {case["role"]: case for case in cases if case["stage"] == "test" and case["seed"] == seed}
        need(set(group) == {"baseline", "candidate"}, "Ten tests must retain both roles for every seed")
        need(not group["baseline"]["retry_enabled"] and group["candidate"]["retry_enabled"], "Paired fixed retry modes differ")
        paired.append({"seed": seed, **group})
    known_pilot = next(case for case in cases if case["stage"] == "pilot")
    equivalence = pair_equivalence(audit["paired_first_cycle_equivalence"], cases,
        known_pilot["known_pilot_reference"]["zero_time_inherited_last_substeps"])
    need(audit["all_paired_first_cycle_physical_states_equal"] is all(row["verified"] for row in equivalence),
         "Overall paired first-cycle diagnostic differs")
    return {"complete": True, "kind": KIND, "record_created_at_utc": datetime.now(timezone.utc).isoformat(),
        "actual_started_at_utc": min(case["started_at_utc"] for case in cases),
        "actual_completed_at_utc": max(case["completed_at_utc"] for case in cases),
        "independent_audit_raw_sha256": audit_digest, "experiment_plan_sha256": audit["experiment_plan_sha256"],
        "stage_artifact_sha256": audit["stage_artifact_sha256"], "fixed_strategy": strategy,
        "task_contract": {"target_ml": 20., "primary_absolute_tolerance_ml": .5, "secondary_easy_absolute_tolerance_ml": 1.,
            "deadline_physical_seconds": 24., "maximum_all_label_peak_spill_particles": 1,
            "home_quiet_frames": 12, "post_observation_physical_seconds": 2., "original_strict_relative_tolerance": .001,
            "original_strict_absolute_tolerance_ml": .02, "original_strict_standard_unchanged": True},
        "physics": {"solver": "live_PBF_action_conditioned", "physics_hz": 60, "pressure_iterations": 6,
            "minimum_substeps": 4, "cuda_graph": False, "spacing_m": .0027, "particle_volume_ml": .019683,
            "total_particles": 11625, "source_particles": 2880,
            "volume_measurement": "fixed-identity particle-mass simulation truth", "contact_force_grasp_simulated": False},
        "budget": {**COUNTS, "total": 21}, "training_episodes": 0, "validation_episodes": 0,
        "test_used_for_selection": False, "known_pilot_is_not_independent_test": True,
        "all_motion_speeds_fixed": True, "fixed_first_approach_tilt_deg": 84., "fixed_retry_approach_tilt_deg": 84.1,
        "stage_metrics": grouped, "test_pairs": paired, "cases": cases,
        "paired_first_cycle_equivalence": equivalence,
        "all_paired_first_cycle_physical_states_equal": audit["all_paired_first_cycle_physical_states_equal"],
        "fresh_test_seeds_that_used_second_cycle": [case["seed"] for case in cases
            if case["stage"] == "test" and case["role"] == "candidate" and case["actual_second_cycle_started"]],
        "default_gui_deployment": False, "all_continuous_goals_guaranteed": False,
        "limitations": ["The known previous failed test condition is development pilot data, not new independent evidence.",
            "This is a fixed conditional two-cycle rule, with no training, validation, learned response model or test-based policy selection.",
            "Only the actual settled undershoot and declared spill/full-home budget guards may trigger the second cycle.",
            "Ten narrow-randomization paired seeds do not guarantee arbitrary20--40mL or hardware accuracy.",
            "Kinematic pregrasp and privileged particle truth do not validate contact grasp, a real scale, or hardware."]}


build_summary = curate


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--audit-sha256", required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args()


def new_output(path):
    need(path is not None, "A new output directory is required")
    path = Path(path).resolve()
    need(not path.exists(), "Output directory must be new; do not overwrite selected evidence")
    path.mkdir(parents=True)
    return path


def main():
    args = arguments()
    summary = curate(args.run, args.audit, args.audit_sha256)
    if args.check_only:
        print(json.dumps({"complete": True, "cases": len(summary["cases"]), "fresh_pairs": len(summary["test_pairs"])})); return
    output = new_output(args.output)
    (output / "retry20_results.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf8")
    print("Wrote " + str(output / "retry20_results.json"))


if __name__ == "__main__":
    main()
