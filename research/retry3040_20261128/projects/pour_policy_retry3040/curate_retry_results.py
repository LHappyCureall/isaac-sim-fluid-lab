"""Curate a completed independently audited30/40mL fixed-retry expansion.

All development episodes remain separate from the new matched tests. A failed
bounded development gate may finalize with no independent test claims.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

KIND = "retry3040_conditional_second_cycle_v1"
GOALS = (30., 40.)
TEST_SEEDS = set(range(197301, 197311))
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
    target = row["target_ml"]
    need(target in (30., 40.), "Only the two declared30/40mL goals are accepted")
    refine, easy, strict = row["refine_assessment"], row["easy_assessment"], row["strict_assessment"]
    need(row["stage"] == stage and row["split"] == ("development" if stage != "test" else "test"),
         "Pilot development and new test labels differ")
    need(report["declared_policy"] == row["declared_policy"], "Audited goal profile differs")
    close(result["target_ml"], target, "Result target changed")
    received, error = result["received_ml"], result["received_ml"] - target
    close(result["error_ml"], error, "Actual signed error differs")
    close(result["absolute_error_ml"], abs(error), "Actual absolute error differs")
    for name, assessment, tolerance in (("refine", refine, .5), ("easy", easy, 1.), ("strict", strict, .001 * target)):
        close(assessment["tolerance_ml"], tolerance, "Task criterion changed: " + name)
        need(type(assessment["valid_evidence"]) is bool, "Physical evidence validity must be explicit")
        dose = target - tolerance <= received <= target + tolerance
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
    post = row.get("post_observation")
    post_available = isinstance(post, dict)
    post_data = post if post_available else {}
    post_complete = bool(post_available and not post_data.get("error")
        and post_data.get("all_states_finite") is True and post_data.get("mass_array_unchanged") is True
        and post_data.get("physical_steps") == 120
        and type(post_data.get("duration_s")) in (int, float)
        and abs(post_data["duration_s"] - 2.) <= 1e-8 and report.get("post_frames") == 120)
    if refine["valid_evidence"]:
        need(post_complete, "Valid acceptance needs all120 actual post frames and two physical seconds")
    policy = row["declared_policy"]
    need(set(policy) == {"enable_retry", "first_approach_tilt_deg", "retry_approach_tilt_deg"} and type(policy["enable_retry"]) is bool,
         "Explicit retry flag and per-goal first/retry angles are required")
    need(all(type(policy[key]) in (int, float) and math.isfinite(policy[key])
             for key in ("first_approach_tilt_deg", "retry_approach_tilt_deg")), "Explicit finite goal angles required")
    close(policy["retry_approach_tilt_deg"] - policy["first_approach_tilt_deg"], .1,
          "Declared goal profile must retain the0.1degree retry gap", 1e-8)
    enabled = policy["enable_retry"]
    need(status["retry_profile_key"] == format(target, "g")
         and isinstance(status["retry_profile_id"], str)
         and isinstance(status["retry_profile_sha256"], str) and len(status["retry_profile_sha256"]) == 64,
         "Executed target-specific profile provenance differs")
    need(status["retry_enabled"] is enabled and status["retry_maximum_cycles"] == (2 if enabled else 1),
         "Executed retry mode or cycle budget differs")
    close(status["retry_first_approach_tilt_deg"], policy["first_approach_tilt_deg"], "First angle changed")
    close(status["retry_second_approach_tilt_deg"], policy["retry_approach_tilt_deg"], "Second angle changed")
    starts = [event for event in row["response_events"] if event["type"] == "response_cycle_start"]
    ends = [event for event in row["response_events"] if event["type"] in ("response_cycle_complete", "response_cycle_unsettled")]
    need(len(starts) <= (2 if enabled else 1) and len(ends) <= len(starts)
         and status["response_cycle_count"] == len(starts), "Fixed first/conditional-second cycle budget changed")
    decisions = status["retry_decisions"]
    need(isinstance(decisions, list) and len(decisions) <= 1 and (enabled or not decisions),
         "Retry decision budget or baseline mode differs")
    allowed = bool(decisions and decisions[0]["retry_allowed"])
    for decision in decisions:
        close(decision["declared_target_ml"], target, "Retry decision changed its actual goal")
        need(decision["target_profile_key"] == status["retry_profile_key"]
             and decision["target_profile_id"] == status["retry_profile_id"]
             and decision["target_profile_sha256"] == status["retry_profile_sha256"], "Retry decision profile changed")
        close(decision["actual_deficit_ml"], target - decision["actual_received_ml"], "Actual retry deficit differs")
        close(decision["declared_retry_approach_tilt_deg"], policy["retry_approach_tilt_deg"], "Retry angle changed at decision")
        observed_peak = decision["observed_all_label_peak_spill_particles"]
        safe_spill = type(observed_peak) is int and 0 <= observed_peak <= 1
        need(decision["spill_guard_passed"] is safe_spill, "Actual retry spill guard differs")
        t, budget = decision["physical_time_s"], decision["conservative_full_home_budget_s"]
        close(decision["deadline_physical_seconds"], 24., "Retry physical deadline changed")
        close(decision["remaining_deadline_physical_seconds"], 24. - t, "Retry remaining physical time differs")
        fits = type(budget) in (int, float) and math.isfinite(budget) and budget >= 0. and t + budget <= 24.
        need(decision["full_home_budget_fits"] is fits, "Actual retry full-home time guard differs")
        expected = bool(decision["actual_settled_measurement"]
            and decision["actual_received_ml"] < target - .5 and decision["spill_guard_passed"]
            and decision["full_home_budget_fits"])
        need(decision["retry_allowed"] is expected, "Retry must use actual settled undershoot, spill and full-home guards")
        need(decision["completed_cycles_before_decision"] == 1
             and decision["deficit_exceeds_half_ml"] is (decision["actual_received_ml"] < target - .5)
             and decision["budget_is_robot_motion_not_predicted_water_dose"] is True,
             "Second-cycle decision did not follow one actual cycle")
    need(len(starts) < 2 or allowed, "Second cycle lacks an allowed measured-feedback decision")
    anchor = status.get("prefix_anchor_observation")
    if anchor is None:
        need(refine["valid_evidence"] is False and result.get("complete") is False,
             "A valid/completed action cannot lack its actual first anchor")
    else:
        need(isinstance(anchor, dict), "Actual first anchor must be an observation or null")
        close(anchor["actual_target_ml"], target, "Actual anchor target changed")
        close(anchor["control_target_ml"], target, "Coarse target was offset")
    anchor_data = anchor if isinstance(anchor, dict) else {}
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
        expected_angle = policy["first_approach_tilt_deg"] if index == 0 else policy["retry_approach_tilt_deg"]
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
    if len(starts) == 2:
        retry_reason = "second_cycle_actually_started"
    elif not enabled:
        retry_reason = "baseline_declares_one_cycle_only"
    elif allowed:
        retry_reason = "retry_allowed_intent_but_second_start_not_recorded"
    elif not cycles:
        if anchor_data.get("settled_measurement") is not True:
            retry_reason = "actual_coarse_anchor_unavailable_or_not_settled"
        elif target - .5 <= anchor_data["received_ml"] <= target + .5:
            retry_reason = "actual_coarse_settled_band"
        elif anchor_data["received_ml"] > target + .5:
            retry_reason = "actual_coarse_settled_overdose_cannot_be_removed_by_retry"
        else:
            retry_reason = "actual_coarse_settled_undershoot_without_first_cycle"
    elif not cycles[0]["has_recorded_endpoint"]:
        retry_reason = "actual_first_cycle_endpoint_unavailable"
    elif cycles[0]["settled"] is not True:
        retry_reason = "actual_first_cycle_endpoint_unsettled"
    else:
        after_first = cycles[0]["actual_end_received_ml"]
        if after_first > target + .5:
            retry_reason = "actual_first_cycle_overdose_cannot_be_removed_by_retry"
        elif target - .5 <= after_first <= target + .5:
            retry_reason = "actual_first_cycle_already_in_primary_band"
        elif decisions and not decisions[0]["actual_settled_measurement"]:
            retry_reason = "actual_remeasurement_not_quiet"
        elif decisions and not decisions[0]["spill_guard_passed"]:
            retry_reason = "actual_current_spill_guard_failed"
        elif decisions and not decisions[0]["full_home_budget_fits"]:
            retry_reason = "insufficient_conservative_full_home_time_budget"
        else:
            retry_reason = "measured_undershoot_without_recorded_retry_permission"
    failures = []
    for failure in report.get("failure_records", []):
        record = {key: failure[key] for key in ("type", "step", "physical_time_s", "failure_stage",
            "measurement_physical_time_s", "partial_unlogged_fluid_frame_possible", "kinematic_evidence_time_s")
            if key in failure}
        if "error" in failure:
            record["error_message_sha256"] = hashlib.sha256(str(failure["error"]).encode("utf8")).hexdigest()
        failures.append(record)
    return {"episode_id": row["episode_id"], "stage": stage, "split": row["split"], "role": row["role"],
        "seed": row["seed"], "target_ml": target, "retry_enabled": enabled,
        "first_approach_tilt_deg": policy["first_approach_tilt_deg"],
        "retry_approach_tilt_deg": policy["retry_approach_tilt_deg"],
        "profile_key": status["retry_profile_key"], "profile_id": status["retry_profile_id"],
        "profile_sha256": status["retry_profile_sha256"],
        "development_not_independent_test": stage != "test",
        "actual_correction_cycles_started": len(starts), "actual_correction_cycles_ended": len(ends),
        "actual_second_cycle_started": len(starts) == 2, "retry_allowed": allowed, "retry_decisions": decisions,
        "second_cycle_execution_or_skip_reason": retry_reason,
        "retry_actual_settled_home_decision": status["retry_actual_settled_home_decision"],
        "correction_cycle_completion_reason": status.get("response_cycle_terminal_reason"),
        "first_quiet_anchor_available": anchor is not None,
        "first_quiet_anchor_settled_measurement": anchor_data.get("settled_measurement"),
        "actual_coarse_cut_tilt_degrees": anchor_data.get("coarse_cut_tilt_deg"),
        "first_quiet_anchor_received_ml": anchor_data.get("received_ml"),
        "first_quiet_anchor_physical_seconds": anchor_data.get("physical_time_s"),
        "first_quiet_anchor_source_rms_m_s": anchor_data.get("source_inside_velocity_rms_m_s"),
        "actual_cycles": cycles, "received_ml": received, "error_ml": error, "absolute_error_ml": abs(error),
        "refine_feasible": refine["feasible"], "easy_feasible": easy["feasible"],
        "original_strict_feasible": strict["feasible"], "original_strict_dose_passed": strict["dose_passed"],
        "valid_evidence": refine["valid_evidence"], "physical_action_seconds": result["physical_time_s"],
        "action_compute_wall_seconds": result["wall_seconds"], "compute_wall_seconds_including_post": row["wall_seconds"],
        "action_complete": result.get("complete"), "valid_physics": result.get("valid_physics"),
        "action_termination_reason": result.get("termination_reason"),
        "received_measurement_physical_time_s": result.get("measurement_physical_time_s"),
        "received_measurement_basis": result.get("measurement_basis"),
        "unlogged_terminal_action_frames": report.get("unlogged_terminal_action_frames"),
        "failure_records": failures,
        "post_observation_available": post_available, "post_observation_valid_complete": post_complete,
        "post_logged_frames": report.get("post_frames"), "post_physical_steps_reported": post_data.get("physical_steps"),
        "post_physical_seconds": post_data.get("duration_s"),
        "post_start_physical_time_s": post_data.get("start_physical_time_s"),
        "post_end_physical_time_s": post_data.get("end_physical_time_s"),
        "post_measurement_basis": post_data.get("measurement_basis"),
        "post_error_message_sha256": (hashlib.sha256(str(post_data["error"]).encode("utf8")).hexdigest()
            if post_data.get("error") is not None else None),
        "peak_all_label_spill_particles": peak,
        "full_home_within_deadline": refine["full_home_within_deadline"], "no_late_dose_change": refine["no_late_dose_change"],
        "mass_conserved": result["mass_conserved"], "all_states_finite": result["all_states_finite"],
        "started_at_utc": row["started_at_utc"], "completed_at_utc": row["completed_at_utc"]}


def summarize_cases(cases, stages, development_stop):
    """Keep all bounded development outcomes; never synthesize unrun tests."""
    need(type(development_stop) is bool, "Development-stop status must be explicit")
    development = [stage for stage in stages if stage != "test"]
    need(development in (["pilot_001"], ["pilot_001", "pilot_002"]), "Unknown bounded development stages")
    need(stages == development + ([] if development_stop else ["test"]), "Actual phase order differs")
    need(len({case["episode_id"] for case in cases}) == len(cases), "Duplicate physical case names")
    for stage in development:
        group = [case for case in cases if case["stage"] == stage]
        need(len(group) == 4 and {(row["target_ml"], row["role"], row["seed"]) for row in group}
             == {(goal, role, 197101) for goal in GOALS for role in ("baseline", "candidate")},
             "Development must retain both goals and both roles at the declared initial state")
    test = [case for case in cases if case["stage"] == "test"]
    need(len(cases) == 4 * len(development) + (0 if development_stop else 40), "Actual bounded case count differs")
    if development_stop:
        need(not test, "Development-stop cannot carry independent test outcomes")
    else:
        need(len(test) == 40 and {(row["target_ml"], row["role"], row["seed"]) for row in test}
             == {(goal, role, seed) for goal in GOALS for role in ("baseline", "candidate") for seed in TEST_SEEDS},
             "Every one of the forty matched new tests is required")
    grouped = {stage: {role: metrics([case for case in cases if case["stage"] == stage and case["role"] == role])
        for role in ("baseline", "candidate")} for stage in stages}
    per_goal = {format(goal, "g"): {stage: {role: metrics([case for case in cases
        if case["stage"] == stage and case["target_ml"] == goal and case["role"] == role])
        for role in ("baseline", "candidate")} for stage in stages} for goal in GOALS}
    pairs, matching = [], {}
    for goal in GOALS:
        goal_pairs = []
        for seed in sorted(TEST_SEEDS) if not development_stop else []:
            group = {row["role"]: row for row in test if row["seed"] == seed and row["target_ml"] == goal}
            base, candidate = group["baseline"], group["candidate"]
            need(base["retry_enabled"] is False and candidate["retry_enabled"] is True,
                 "Paired fixed retry modes changed")
            pair = {"target_ml": goal, "seed": seed, **group,
                "final_dose_exactly_equal": base["received_ml"] == candidate["received_ml"],
                "physical_duration_exactly_equal": base["physical_action_seconds"] == candidate["physical_action_seconds"]}
            pairs.append(pair); goal_pairs.append(pair)
        no_retry = [pair for pair in goal_pairs if not pair["candidate"]["actual_second_cycle_started"]]
        matching[format(goal, "g")] = {"fresh_pairs": len(goal_pairs), "no_retry_pairs": len(no_retry),
            "no_retry_dose_and_time_exact_pairs": sum(pair["final_dose_exactly_equal"]
                and pair["physical_duration_exactly_equal"] for pair in no_retry),
            "actual_second_cycle_seeds": [pair["seed"] for pair in goal_pairs if pair["candidate"]["actual_second_cycle_started"]],
            "baseline_failure_candidate_pass_pairs": sum(not pair["baseline"]["refine_feasible"]
                and pair["candidate"]["refine_feasible"] for pair in goal_pairs),
            "baseline_pass_candidate_failure_pairs": sum(pair["baseline"]["refine_feasible"]
                and not pair["candidate"]["refine_feasible"] for pair in goal_pairs)}
    return {"actual_stages": stages, "development_stop": development_stop,
        "independent_test_not_run": development_stop, "budget": {"development": 4 * len(development),
            "test": len(test), "total": len(cases), "maximum_development": 8, "maximum_total": 48},
        "stage_metrics": grouped, "per_goal_metrics": per_goal, "test_pairs": pairs,
        "paired_no_retry_matching": matching}


def compact_pair_proofs(records, cases, stages, solver_sha):
    expected = {(stage, goal, seed) for stage in stages for goal in GOALS
        for seed in (TEST_SEEDS if stage == "test" else {197101})}
    need(len(records) == len(expected) and {(row["stage"], row["target_ml"], row["seed"]) for row in records} == expected,
         "All actually audited goal/stage/seed first-cycle comparisons required")
    output = []
    for row in records:
        group = {case["role"]: case for case in cases if case["stage"] == row["stage"]
            and case["target_ml"] == row["target_ml"] and case["seed"] == row["seed"]}
        need(set(group) == {"baseline", "candidate"}, "Pair proof lost an actual role")
        item = {key: row[key] for key in ("stage", "target_ml", "seed", "verified")}
        item["endpoint_unavailable"] = row.get("endpoint_unavailable", False)
        if item["endpoint_unavailable"]:
            need(row["verified"] is False and any(not case["actual_cycles"]
                 or not case["actual_cycles"][0]["has_recorded_endpoint"] for case in group.values()),
                 "Missing endpoints cannot prove first-cycle equality")
            item["all_completed_frame_substep_counts_exactly_equal"] = None
        else:
            endpoint_kind = row.get("comparison_endpoint_kind", "actual_first_complete_or_unsettled_whole_cycle")
            need(endpoint_kind in ("actual_first_complete_or_unsettled_whole_cycle", "actual_quiet_coarse_anchor_no_cycle_needed"),
                 "Unknown actual pair comparison endpoint")
            item["comparison_endpoint_kind"] = endpoint_kind
            for role in ("baseline", "candidate"):
                need(row[role + "_episode_id"] == group[role]["episode_id"], "Pair proof identity differs")
                if endpoint_kind == "actual_quiet_coarse_anchor_no_cycle_needed":
                    need(not group[role]["actual_cycles"], "Anchor comparison may not replace an actual cycle")
                    physical = group[role]["first_quiet_anchor_physical_seconds"]
                    amount = group[role]["first_quiet_anchor_received_ml"]
                else:
                    need(group[role]["actual_cycles"] and group[role]["actual_cycles"][0]["has_recorded_endpoint"],
                         "Audited first endpoint missing from compact cases")
                    endpoint = group[role]["actual_cycles"][0]
                    physical, amount = endpoint["end_physical_s"], endpoint["actual_end_received_ml"]
                    item[role + "_actual_first_cycle_end_received_ml"] = amount
                item[role + "_actual_comparison_received_ml"] = amount
                if row["verified"]:
                    close(row["first_cycle_end_physical_s"], physical, "Verified pair endpoint time differs")
            for key in ("baseline_episode_id", "candidate_episode_id", "first_cycle_end_physical_s", "first_cycle_frames",
                        "original_first_cycle_events_exactly_equal", "actual_water_motion_partitions_exactly_equal"):
                item[key] = row[key]
            need(row["verified"] is bool(row["original_first_cycle_events_exactly_equal"]
                 and row["actual_water_motion_partitions_exactly_equal"]), "First-cycle conjunction differs")
            if row["verified"]:
                close(item["baseline_actual_comparison_received_ml"], item["candidate_actual_comparison_received_ml"],
                      "Verified first-cycle doses differ")
            item["all_completed_frame_substep_counts_exactly_equal"] = (
                True if row["actual_water_motion_partitions_exactly_equal"] else None)
            item["completed_frame_substep_equality_basis"] = "included_in_independent_physical_equality_conjunction"
            counter = dict(row["zero_time_inherited_last_substeps"])
            need(counter["is_actual_completed_frame"] is False and counter["excluded_from_equivalence_only"] is True,
                 "Inherited t0 counter exception may not hide a completed physical frame")
            counter.update(source_file="assets/robot_testtube_pour/scripts/liquid.py", source_sha256=solver_sha,
                reason="Frozen reset retains last_substeps at t0; each actual step_frame recomputes it from physical state.")
            item["zero_time_inherited_last_substeps"] = counter
        output.append(item)
    return output


def curate(run, audit_path, audit_digest):
    run, audit_path = Path(run).resolve(), Path(audit_path).resolve()
    need(audit_path == run / "overall_audit.json", "Use this experiment's final overall audit")
    need(sha(audit_path) == audit_digest, "Independent overall audit SHA differs")
    audit = read(audit_path)
    need(audit.get("complete") is True and audit.get("pass") is True,
         "Complete passing independent audit required before reading journals")
    need(audit["kind"] == KIND, "Wrong completed experiment kind")
    need(audit["all_physical_sources_geometry_inputs_gui_unchanged"] is True
         and audit["only_declared_goal_profile_inputs_may_differ"] is True,
         "Declared profile changes may not hide physical source/geometry/default changes")
    auditor = "projects/pour_policy_retry3040/audit_retry_experiment.py"
    workspace = Path(__file__).resolve().parents[2]
    need(audit["auditor_source_file"] == auditor and sha(workspace / auditor) == audit["auditor_source_sha256"],
         "Independent finalized audit implementation changed")
    stopped = audit["development_stop"]
    need(type(stopped) is bool and audit["outcome"] == ("development_gate_failed" if stopped else "fresh_test_completed"),
         "Final development/test outcome is ambiguous")
    development = audit["actual_development_stages"]
    need(development in (["pilot_001"], ["pilot_001", "pilot_002"]), "Unknown development stage set")
    stages = development + ([] if stopped else ["test"])
    need(set(audit["stage_artifact_sha256"]) == set(stages), "Every actual stage must be retained")
    need(sha(run / "experiment_plan.json") == audit["experiment_plan_sha256"], "Preregistered plan changed")
    plan = read(run / "experiment_plan.json")
    need(plan["draft"] is False and plan["targets_ml"] == [30., 40.]
         and plan["development"]["maximum_cases"] == 8 and plan["test"]["cases"] == 40
         and set(plan["test"]["seeds"]) == TEST_SEEDS and plan["test"]["test_used_for_selection"] is False,
         "Declared target, bounded budget or no-test-selection contract changed")
    task = plan["task"]
    need(task == {"absolute_tolerance_ml": .5, "deadline_physical_seconds": 24., "home_quiet_frames": 12,
        "peak_all_label_spill_particles": 1, "post_physical_seconds": 2., "original_strict_relative_tolerance": .001},
        "Primary/strict or physical-time criteria changed")
    protocols = {}
    for stage in stages:
        hashes = audit["stage_artifact_sha256"][stage]
        need(set(hashes) == set(TOP), "Each stage needs seven artifacts plus its independent audit")
        for name, digest in hashes.items():
            need(sha(run / stage / name) == digest, "Audited stage artifact changed: " + stage + "/" + name)
        summary, job = read(run / stage / "summary.json"), read(run / stage / "job_state.json")
        need(summary["complete"] is True and summary["episodes"] == (40 if stage == "test" else 4)
             and job["state"] == "complete" and job["pid"] is None and not (run / stage / "worker.lock").exists(),
             "All actual phases must finish before curation")
        protocol = protocols[stage] = read(run / stage / "protocol.json")
        need(len(protocol["source_sha256"]) == 28 and len(protocol["input_sha256"]) == 7
             and len(protocol["gui_sha256"]) == 3, "Frozen runtime/input/GUI inventory differs")
        for key, folder in (("source_sha256", "source_snapshot"), ("input_sha256", "input_snapshot")):
            for relative, digest in protocol[key].items():
                need(sha(safe_path(workspace, relative)) == sha(safe_path(run / stage / folder, relative)) == digest,
                     "Current/frozen source or input changed: " + relative)
        for relative, digest in protocol["gui_sha256"].items():
            need(sha(safe_path(workspace, relative)) == digest, "Old GUI model changed")
    first = protocols[development[0]]
    def core_inputs(protocol):
        profile = protocol["matrix"]["goal_profile_file"]
        need(profile in protocol["input_sha256"], "Explicit goal-profile input is missing")
        return {key: value for key, value in protocol["input_sha256"].items() if key != profile}
    for protocol in protocols.values():
        need(protocol["source_sha256"] == first["source_sha256"] and protocol["gui_sha256"] == first["gui_sha256"]
             and core_inputs(protocol) == core_inputs(first), "Physical sources/geometry/defaults changed across phases")
        need(not any("/pour_policy_retry20/" in name for name in protocol["source_sha256"]),
             "Earlier20mL study became an undeclared runtime dependency")
    adjustment = audit["development_adjustment"]
    need((adjustment is None) is (len(development) == 1), "Signed second-stage adjustment receipt differs")
    if adjustment is not None:
        need(sha(safe_path(run, adjustment["file"])) == adjustment["sha256"], "Bounded development adjustment receipt changed")
    if stopped:
        need(audit["fixed_strategy"] is None and audit["finalization"] is not None
             and not (run / "test").exists(), "Development-stop may not hide actual tests or a selected strategy")
        finalization = audit["finalization"]
        need(sha(safe_path(run, finalization["file"])) == finalization["sha256"], "Final development-stop receipt changed")
    else:
        need(audit["finalization"] is None and audit["fixed_strategy"] is not None, "Completed tests need a frozen fixed strategy")
        fixed = audit["fixed_strategy"]
        need(fixed["file"] == "fixed_strategy.json" and sha(run / fixed["file"]) == fixed["sha256"], "Fixed strategy changed")
    reports = audit["episode_reports"]
    expected_count = 4 * len(development) + (0 if stopped else 40)
    need(len(reports) == len({row["episode_id"] for row in reports}) == expected_count, "Audit lost an actual physical attempt")
    reports = {row["episode_id"]: row for row in reports}
    cases = []
    for stage in stages:
        rows = [json.loads(line) for line in (run / stage / "episodes.jsonl").read_text(encoding="utf8").splitlines() if line.strip()]
        need(len(rows) == (40 if stage == "test" else 4), "Journal case count differs")
        for row in rows:
            need(row["episode_id"] in reports, "Physical case absent from independent audit")
            need(read(run / stage / "completed" / (row["episode_id"] + ".json")) == row, "Completed sidecar differs from journal")
            for relative, digest in row["artifacts"].items():
                need(sha(safe_path(run / stage, relative)) == digest, "Raw action/post evidence changed")
            cases.append(compact_case(row, stage, reports[row["episode_id"]]))
    body = summarize_cases(cases, stages, stopped)
    need(all(audit["budgets"][key] == body["budget"][key] for key in ("development", "test", "total")),
         "Independent actual budget differs")
    proof = compact_pair_proofs(audit["paired_first_cycle_equivalence"], cases, stages,
        first["source_sha256"]["assets/robot_testtube_pour/scripts/liquid.py"])
    return {"complete": True, "kind": KIND, "outcome": audit["outcome"],
        "record_created_at_utc": datetime.now(timezone.utc).isoformat(),
        "actual_started_at_utc": min(case["started_at_utc"] for case in cases),
        "actual_completed_at_utc": max(case["completed_at_utc"] for case in cases),
        "independent_audit_raw_sha256": audit_digest, "experiment_plan_sha256": audit["experiment_plan_sha256"],
        "stage_artifact_sha256": audit["stage_artifact_sha256"], "fixed_strategy": audit["fixed_strategy"],
        "finalization": audit["finalization"], "development_adjustment": adjustment,
        "task_contract": {"targets_ml": [30., 40.], **task,
            "secondary_easy_absolute_tolerance_ml": 1., "original_strict_absolute_tolerance_ml": {"30": .03, "40": .04}},
        "physics": {"solver": "live_PBF_action_conditioned", "physics_hz": 60, "pressure_iterations": 6,
            "minimum_substeps": 4, "cuda_graph": False, "spacing_m": .0027, "particle_volume_ml": .019683,
            "total_particles": 11625, "source_particles": 2880,
            "volume_measurement": "fixed-identity particle-mass simulation truth", "contact_force_grasp_simulated": False},
        "cases": cases, **body, "paired_first_cycle_equivalence": proof,
        "actual_development_stages": development,
        "development_gate_passed": audit["development_gate_passed"],
        "stage_goal_profiles": {stage: protocol["matrix"]["goal_profiles"] for stage, protocol in protocols.items()},
        "stage_goal_profile_inputs": {stage: {"file": protocol["matrix"]["goal_profile_file"],
            "sha256": protocol["input_sha256"][protocol["matrix"]["goal_profile_file"]]} for stage, protocol in protocols.items()},
        "all_physical_sources_geometry_inputs_gui_unchanged": True,
        "only_declared_goal_profile_inputs_may_differ": True,
        "bounded_development_angle_calibration_performed": adjustment is not None,
        "test_used_for_profile_selection": False, "development_is_not_independent_test": True,
        "all_motion_speeds_fixed": True, "old20_not_imported_by_new_runtime": True,
        "default_gui_deployment": False, "all_continuous_goals_guaranteed": False,
        "limitations": ["At most two paired development stages permit one preregistered signed0.1degree profile adjustment.",
            "That prescribed development adjustment is parameter calibration; it is retained separately from the frozen fresh tests.",
            "Every attempted development case is retained; development data are not independent tests.",
            "Invalid or partial cases retain the raw footer measurement basis; reported last-available doses are not valid terminal measurements.",
            "Fresh tests, when authorized by the development gate, never select or retune the frozen profile.",
            "No new PPO or neural policy is trained; this is a bounded measured-feedback engineering expansion.",
            "Two fixed targets and narrow-randomization pairs do not prove arbitrary20--40mL or hardware precision.",
            "Kinematic pregrasp and privileged particle truth are not a real scale or contact-grasp validation."]}


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
    need(not path.exists(), "Output directory must be new; preserve previous records")
    path.mkdir(parents=True)
    return path


def main():
    args = arguments()
    summary = curate(args.run, args.audit, args.audit_sha256)
    if args.check_only:
        print(json.dumps({"complete": True, "cases": len(summary["cases"]), "development_stop": summary["development_stop"]})); return
    output = new_output(args.output)
    (output / "retry3040_results.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf8")
    print("Wrote " + str(output / "retry3040_results.json"))


if __name__ == "__main__":
    main()


