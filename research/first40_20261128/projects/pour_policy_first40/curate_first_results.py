"""Curate independently finalized first-angle-only40mL calibration and evaluation.

Only the first approach angle is calibrated. Both roles retain the same
conditional retry policy. Known-case calibration, fresh validation and new
paired tests remain separate, including every failed or partial attempt.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

KIND = "first40_first_angle_calibration_v1"
TARGET_ML = 40.
CALIBRATION_SEED = 197304
VALIDATION_SEEDS = {198101, 198102}
TEST_SEEDS = set(range(198301, 198311))
CANDIDATE_ORDER = ("first_8700", "first_8705", "first_8695")
ANGLES = {"baseline": 87.1, "first_8700": 87.0, "first_8705": 87.05, "first_8695": 86.95}
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
    need(target == 40., "Only the declared40mL goal is accepted")
    refine, easy, strict = row["refine_assessment"], row["easy_assessment"], row["strict_assessment"]
    need(row["stage"] == stage and row["split"] == ("test" if stage == "test" else "development"),
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
    need(policy["enable_retry"] is True and 86.9 <= policy["first_approach_tilt_deg"] <= 87.1
         and policy["retry_approach_tilt_deg"] == 87.2,
         "Both roles keep retry enabled and exactly87.2degrees; only first angle may vary")
    enabled = policy["enable_retry"]
    need(status["first40_condition_seed"] == row["seed"]
         and status["condition_seed_provided_by_runner"] is True, "Actual condition seed metadata differs")
    need(row["candidate_id"] == row["profile_key"] and row["candidate_id"] in ANGLES,
         "Declared candidate identity/profile lookup differs")
    close(policy["first_approach_tilt_deg"], ANGLES[row["candidate_id"]], "Candidate label does not match actual declared angle")
    if "candidate_id" in report or "profile_key" in report:
        need(report["candidate_id"] == row["candidate_id"] and report["profile_key"] == row["profile_key"],
             "Audited candidate identity/profile lookup differs")
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
        "first40_condition_seed": status["first40_condition_seed"],
        "condition_seed_provided_by_runner": status["condition_seed_provided_by_runner"],
        "first_approach_tilt_deg": policy["first_approach_tilt_deg"],
        "retry_approach_tilt_deg": policy["retry_approach_tilt_deg"],
        "profile_namespace": status["retry_profile_key"], "profile_key": row["profile_key"],
        "candidate_id": row["candidate_id"], "profile_id": status["retry_profile_id"],
        "profile_sha256": status["retry_profile_sha256"],
        "known_seen_development_case": stage == "calibration",
        "validation_is_not_final_holdout": stage == "validation",
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


def calibration_key(case):
    if not case["valid_evidence"]:
        return None
    if case["refine_feasible"]:
        return (0, case["physical_action_seconds"])
    return (1, case["absolute_error_ml"], case["peak_all_label_spill_particles"])


def choose_calibration(cases):
    """Recompute the preregistered feasible-time rank; no validation/test input."""
    calibration = [case for case in cases if case["stage"] == "calibration"]
    need(len(calibration) == 4 and {case["candidate_id"] for case in calibration} == set(ANGLES)
         and {case["seed"] for case in calibration} == {CALIBRATION_SEED}, "Complete known-case baseline and three candidates required")
    options = []
    for index, candidate_id in enumerate(CANDIDATE_ORDER):
        group = [case for case in calibration if case["candidate_id"] == candidate_id]
        need(len(group) == 1 and group[0]["role"] == "candidate", "Each declared candidate must be retained once")
        case = group[0]
        close(case["first_approach_tilt_deg"], ANGLES[candidate_id], "Calibration angle differs")
        key = calibration_key(case)
        if key is not None:
            options.append((key, index, candidate_id))
    return min(options)[2] if options else None


def summarize_cases(cases, stages, stopped):
    need(stages in (["calibration"], ["calibration", "validation"], ["calibration", "validation", "test"]),
         "Unknown bounded calibration/validation/test phases")
    need(type(stopped) is bool and stopped is ("test" not in stages), "Stopped/completed-test outcome differs")
    need(len({case["episode_id"] for case in cases}) == len(cases), "Duplicate physical attempt identities")
    expected_count = sum({"calibration": 4, "validation": 4, "test": 20}[stage] for stage in stages)
    need(len(cases) == expected_count, "Every actual bounded attempt must remain in the report")
    chosen = choose_calibration(cases)
    for stage in stages:
        group = [case for case in cases if case["stage"] == stage]
        if stage == "calibration":
            continue
        seeds = VALIDATION_SEEDS if stage == "validation" else TEST_SEEDS
        need(len(group) == 2 * len(seeds) and {(case["role"], case["seed"]) for case in group}
             == {(role, seed) for role in ("baseline", "candidate") for seed in seeds}, "Complete paired phase conditions required")
        for case in group:
            need(case["candidate_id"] == ("baseline" if case["role"] == "baseline" else chosen),
                 "Calibration candidate changed after freeze")
            close(case["first_approach_tilt_deg"], ANGLES[case["candidate_id"]], "Frozen phase angle changed")
    grouped = {stage: {role: metrics([case for case in cases if case["stage"] == stage and case["role"] == role])
        for role in ("baseline", "candidate")} for stage in stages}
    calibration = {candidate_id: metrics([case for case in cases if case["stage"] == "calibration"
        and case["candidate_id"] == candidate_id]) for candidate_id in ANGLES}
    pairs = {}
    for stage in stages:
        if stage == "calibration":
            continue
        seeds = VALIDATION_SEEDS if stage == "validation" else TEST_SEEDS
        pairs[stage] = []
        for seed in sorted(seeds):
            group = {case["role"]: case for case in cases if case["stage"] == stage and case["seed"] == seed}
            base, candidate = group["baseline"], group["candidate"]
            pairs[stage].append({"seed": seed, **group,
                "candidate_minus_baseline_absolute_error_ml": candidate["absolute_error_ml"] - base["absolute_error_ml"],
                "candidate_minus_baseline_physical_seconds": candidate["physical_action_seconds"] - base["physical_action_seconds"],
                "dose_exactly_equal": candidate["received_ml"] == base["received_ml"],
                "physical_duration_exactly_equal": candidate["physical_action_seconds"] == base["physical_action_seconds"]})
    return {"actual_stages": stages, "stopped_before_independent_test": stopped,
        "independent_test_not_run": stopped, "budget": {stage: sum(case["stage"] == stage for case in cases)
            for stage in ("calibration", "validation", "test")}, "actual_cases": len(cases), "maximum_actual_cases": 28,
        "calibration_selected_candidate_id_recomputed": chosen,
        "calibration_metrics_by_candidate": calibration, "stage_metrics": grouped, "paired_comparisons": pairs,
        "test_pairs": pairs.get("test", [])}


def compact_known_reference(reference):
    result = {key: reference.get(key) for key in ("verified", "endpoint_unavailable",
        "seen_development_case_not_independent", "original_failed_received_ml", "first_cycle_end_physical_s",
        "first_cycle_frames", "all_completed_frame_substep_counts_exactly_equal",
        "first_cycle_water_motion_partitions_exactly_equal", "first_cycle_original_events_exactly_equal",
        "actual_water_motion_partitions_exactly_equal", "zero_time_inherited_last_substeps")}
    result["reference_artifact_sha256"] = sorted(reference.get("source_reference_sha256", {}).values())
    return result


def compact_prefix_proofs(records, cases, stages, selected, solver_sha=None):
    """The comparison ends before the changed first angle, never after it."""
    expected = {("calibration", CALIBRATION_SEED, candidate) for candidate in CANDIDATE_ORDER}
    for stage in stages:
        if stage != "calibration":
            expected.update((stage, seed, selected) for seed in (VALIDATION_SEEDS if stage == "validation" else TEST_SEEDS))
    need(len(records) == len(expected) and {(row["stage"], row["seed"], row["candidate_id"]) for row in records} == expected,
         "Every actual candidate/coarse-prefix comparison must remain in the report")
    result = []
    for row in records:
        stage, seed, candidate_id = row["stage"], row["seed"], row["candidate_id"]
        base = next(case for case in cases if case["stage"] == stage and case["seed"] == seed and case["candidate_id"] == "baseline")
        candidate = next(case for case in cases if case["stage"] == stage and case["seed"] == seed and case["candidate_id"] == candidate_id)
        need(row["baseline_episode_id"] == base["episode_id"] and row["candidate_episode_id"] == candidate["episode_id"],
             "Coarse-prefix comparison identities differ")
        item = json.loads(json.dumps(row, allow_nan=False))
        need(type(row["verified"]) is bool, "Coarse-prefix proof validity must be explicit")
        # These audited diagnostics contain only compact frame/equality/counter
        # metadata; raw trajectories are deliberately absent from this record.
        for key in item:
            need(not key.startswith("raw_") and key not in ("steps", "truth_rows", "direct_rows"),
                 "Raw trajectories may not enter the selected coarse-prefix proof")
        if row["verified"]:
            need(base["first_quiet_anchor_available"] and candidate["first_quiet_anchor_available"],
                 "Verified prefix lacks an actual anchor")
            close(base["first_quiet_anchor_physical_seconds"], candidate["first_quiet_anchor_physical_seconds"],
                  "Verified coarse-prefix anchor times differ")
            close(base["first_quiet_anchor_received_ml"], candidate["first_quiet_anchor_received_ml"],
                  "Verified coarse-prefix anchor water differs")
            close(row["coarse_anchor_physical_s"], base["first_quiet_anchor_physical_seconds"],
                  "Verified coarse-prefix clock differs")
        if not row.get("endpoint_unavailable", False):
            need(row["whole_first_cycle_equality_is_not_required"] is True
                 and row["verified"] is row["actual_coarse_water_motion_partitions_exactly_equal"],
                 "Coarse-prefix control proof was confused with changed-angle first-cycle equality")
            item["all_completed_frame_substep_counts_exactly_equal"] = True if row["verified"] else None
            counter = item["zero_time_inherited_last_substeps"]
            need(counter["is_actual_completed_frame"] is False and counter["excluded_from_equivalence_only"] is True,
                 "Inherited t0 counter exception may not hide a completed physical frame")
            if solver_sha is not None:
                counter.update(source_file="assets/robot_testtube_pour/scripts/liquid.py", source_sha256=solver_sha,
                    reason="Frozen reset retains diagnostic last_substeps at t0; every completed physical frame is compared.")
        item.update(baseline_actual_coarse_anchor_received_ml=base["first_quiet_anchor_received_ml"],
            candidate_actual_coarse_anchor_received_ml=candidate["first_quiet_anchor_received_ml"],
            baseline_actual_coarse_anchor_physical_s=base["first_quiet_anchor_physical_seconds"],
            candidate_actual_coarse_anchor_physical_s=candidate["first_quiet_anchor_physical_seconds"])
        result.append(item)
    return result


def curate(run, audit_path, audit_digest):
    run, audit_path = Path(run).resolve(), Path(audit_path).resolve()
    need(audit_path == run / "overall_audit.json", "Use this experiment's final overall audit")
    need(sha(audit_path) == audit_digest, "Independent overall audit SHA differs")
    audit = read(audit_path)
    need(audit.get("complete") is True and audit.get("pass") is True,
         "Complete passing independent audit required before reading journals")
    need(audit["kind"] == KIND, "Wrong first-angle-only experiment kind")
    need(audit["first_angle_parameter_calibration_performed"] is True
         and audit["calibration_grid_selection_performed"] is True
         and audit["validation_used_for_retuning_or_fallback"] is False and audit["test_used_for_selection"] is False
         and audit["only_first_angle_varies"] is True and audit["fixed_retry_angle_deg"] == 87.2
         and audit["all_physical_sources_inputs_gui_unchanged"] is True,
         "Finite calibration scope, no-retuning rule or fixed physical implementation differs")
    workspace = Path(__file__).resolve().parents[2]
    auditor = "projects/pour_policy_first40/audit_first_experiment.py"
    need(audit["auditor_source_file"] == auditor and sha(workspace / auditor) == audit["auditor_source_sha256"],
         "Independent overall auditor changed")
    stages = audit["actual_stages"]
    need(stages in (["calibration"], ["calibration", "validation"], ["calibration", "validation", "test"]), "Unknown actual phase set")
    stopped = audit["development_stop"]
    need(stopped is ("test" not in stages) and audit["outcome"] == ("fresh_test_completed" if not stopped
         else "calibration_gate_failed" if stages == ["calibration"] else "validation_gate_failed"), "Final phase outcome differs")
    need(set(audit["stage_artifact_sha256"]) == set(stages), "Actual phase proof is missing")
    need(sha(run / "experiment_plan.json") == audit["experiment_plan_sha256"], "Preregistered plan changed")
    plan = read(run / "experiment_plan.json")
    need(plan["draft"] is False and plan["target_ml"] == 40. and plan["maximum_actual_cases"] == 28
         and plan["stages"]["calibration"]["candidate_angles_in_tie_order"] == [87., 87.05, 86.95]
         and set(plan["stages"]["validation"]["seeds"]) == VALIDATION_SEEDS
         and set(plan["stages"]["test"]["seeds"]) == TEST_SEEDS and plan["retry_angle_fixed_deg"] == 87.2
         and plan["enable_retry_for_both_roles"] is True, "Task, fixed retry or candidate/seed scope changed")
    need(plan["task"] == {"absolute_tolerance_ml": .5, "deadline_physical_seconds": 24., "home_quiet_frames": 12,
        "peak_all_label_spill_particles": 1, "post_physical_seconds": 2., "original_strict_relative_tolerance": .001},
        "Task standards changed")
    protected_path = safe_path(workspace, plan["protected_versions_file"])
    need(sha(protected_path) == plan["protected_versions_raw_sha256"], "Protected old-version manifest changed")
    protected = read(protected_path)["files_sha256"]
    need(len(protected) == 42 and protected == audit["protected_file_sha256"]
         and audit["old20_and30_versions_unchanged"] is True, "All42 old20/30 version bindings required")
    for relative, digest in protected.items():
        need(sha(safe_path(workspace, relative)) == digest, "A protected earlier version changed")
    protocols = {}
    for stage in stages:
        hashes = audit["stage_artifact_sha256"][stage]
        need(set(hashes) == set(TOP), "Each phase needs seven artifacts plus its independent audit")
        for name, digest in hashes.items():
            need(sha(run / stage / name) == digest, "Audited phase artifact changed: " + stage + "/" + name)
        summary, job = read(run / stage / "summary.json"), read(run / stage / "job_state.json")
        need(summary["complete"] is True and summary["episodes"] == (20 if stage == "test" else 4)
             and job["state"] == "complete" and job["pid"] is None and not (run / stage / "worker.lock").exists(), "Actual phases must be finished")
        protocol = protocols[stage] = read(run / stage / "protocol.json")
        need(len(protocol["source_sha256"]) == 28 and len(protocol["input_sha256"]) == 8
             and len(protocol["gui_sha256"]) == 3 and protocol["protected_file_sha256"] == protected,
             "Frozen source/input/protected inventories differ")
        for key, folder in (("source_sha256", "source_snapshot"), ("input_sha256", "input_snapshot")):
            for relative, digest in protocol[key].items():
                need(sha(safe_path(workspace, relative)) == sha(safe_path(run / stage / folder, relative)) == digest,
                     "Current/frozen first-only runtime or input changed")
        for relative, digest in protocol["gui_sha256"].items():
            need(sha(safe_path(workspace, relative)) == digest, "Original GUI weights changed")
    first = protocols["calibration"]
    for protocol in protocols.values():
        need(all(protocol[key] == first[key] for key in ("source_sha256", "input_sha256", "gui_sha256", "protected_file_sha256")),
             "Sources, all eight static inputs or old versions changed between phases")
    reports = audit["episode_reports"]
    count = sum(20 if stage == "test" else 4 for stage in stages)
    need(len(reports) == len({row["episode_id"] for row in reports}) == count, "An actual failed/partial attempt was lost")
    reports = {row["episode_id"]: row for row in reports}
    cases = []
    for stage in stages:
        rows = [json.loads(line) for line in (run / stage / "episodes.jsonl").read_text(encoding="utf8").splitlines() if line.strip()]
        need(len(rows) == (20 if stage == "test" else 4), "Actual journal count differs")
        for row in rows:
            need(row["episode_id"] in reports and read(run / stage / "completed" / (row["episode_id"] + ".json")) == row,
                 "Completed sidecar/audit differs from journal")
            for relative, digest in row["artifacts"].items():
                need(sha(safe_path(run / stage, relative)) == digest, "Raw action/post evidence changed")
            cases.append(compact_case(row, stage, reports[row["episode_id"]]))
    body = summarize_cases(cases, stages, stopped)
    need(all(audit["budgets"][phase] == body["budget"][phase] for phase in ("calibration", "validation", "test"))
         and audit["budgets"]["total"] == body["actual_cases"], "Independently audited bounded budget differs")
    selected = body["calibration_selected_candidate_id_recomputed"]
    selection, strategy = audit["calibration_selection"], audit["fixed_strategy"]
    need((selection is None) is (stages == ["calibration"]), "Calibration selection checkpoint phase differs")
    for metadata, filename in ((selection, "calibration_selection.json"), (strategy, "fixed_strategy.json")):
        if metadata is not None:
            need(metadata["file"] == filename and sha(run / filename) == metadata["sha256"], "Frozen selection/strategy changed")
            model = read(run / filename)
            need(model["selected_profile_key"] == selected and model["selected_first_approach_tilt_deg"] == ANGLES[selected]
                 and model["retry_approach_tilt_deg"] == 87.2 and model["both_roles_enable_retry"] is True
                 and model["test_used_for_selection"] is False and model["default_gui_deployment"] is False,
                 "Frozen profile was retuned or selected from test outcomes")
    need((strategy is None) is stopped, "Only a validation-passed frozen strategy may authorize tests")
    finalization = audit["finalization"]
    need((finalization is not None) is stopped, "A stopped study needs its finalization receipt")
    if finalization is not None:
        need(sha(safe_path(run, finalization["file"])) == finalization["sha256"] and not (run / "test").exists(), "Stopped study hid actual tests or changed receipt")
    need(audit["selected_profile_key"] == selected, "Independent calibration ranking differs")
    prefixes = compact_prefix_proofs(audit["coarse_prefix_equivalence"], cases, stages, selected,
        first["source_sha256"]["assets/robot_testtube_pour/scripts/liquid.py"])
    need(audit["all_coarse_prefix_physical_states_equal"] is all(row["verified"] for row in prefixes),
         "Aggregate coarse-prefix control proof differs")
    return {"complete": True, "kind": KIND, "outcome": audit["outcome"],
        "record_created_at_utc": datetime.now(timezone.utc).isoformat(),
        "actual_started_at_utc": min(case["started_at_utc"] for case in cases),
        "actual_completed_at_utc": max(case["completed_at_utc"] for case in cases),
        "independent_audit_raw_sha256": audit_digest, "experiment_plan_sha256": audit["experiment_plan_sha256"],
        "stage_artifact_sha256": audit["stage_artifact_sha256"], "calibration_selection": selection,
        "fixed_strategy": strategy, "finalization": finalization, "cases": cases, **body,
        "task_contract": {"target_ml": 40., **plan["task"], "secondary_easy_absolute_tolerance_ml": 1.,
            "original_strict_absolute_tolerance_ml": .04},
        "physics": {"solver": "live_PBF_action_conditioned", "physics_hz": 60, "pressure_iterations": 6,
            "minimum_substeps": 4, "cuda_graph": False, "spacing_m": .0027, "particle_volume_ml": .019683,
            "total_particles": 11625, "source_particles": 2880,
            "volume_measurement": "fixed-identity particle-mass simulation truth", "contact_force_grasp_simulated": False},
        "known_baseline_equivalence": compact_known_reference(audit["known_baseline_equivalence"]),
        "coarse_prefix_equivalence": prefixes,
        "all_coarse_prefix_physical_states_equal": audit["all_coarse_prefix_physical_states_equal"],
        "protected_manifest_raw_sha256": plan["protected_versions_raw_sha256"], "protected_file_sha256": protected,
        "old20_and30_versions_unchanged": True, "bounded_first_angle_calibration": True,
        "feasible_calibration_rank_uses_only_physical_time_then_declared_order": True,
        "selected_from_calibration_not_validation_or_test": True,
        "collector_no_data_fit_or_model_selection_header_scope": "collector_only_not_the_finite_calibration_experiment",
        "only_first_angle_varies": True, "retry_angle_fixed_deg": 87.2, "both_roles_enable_retry": True,
        "whole_first_cycle_equality_across_changed_angles_is_not_expected": True,
        "default_gui_deployment": False, "all_continuous_goals_guaranteed": False,
        "limitations": ["Known-case four-run finite angle calibration is separate from two fresh validation conditions and ten new test pairs.",
            "Feasible calibration candidates rank only by original physical duration; exact ties follow the declared grid order.",
            "Validation may gate release; it may not retune the candidate or select a baseline fallback.",
            "Only the first angle changes; both roles retain the fixed87.2degree conditional retry and all other actions/physics.",
            "Invalid/partial attempts retain their actual last-available measurement basis and do not become valid terminal measurements.",
            "This finite calibration is not CEM, PPO or a fitted neural policy; collector-only no-fit metadata does not erase the calibration selection.",
            "Fixed40mL and narrow randomization do not establish arbitrary target or hardware precision.",
            "Kinematic pregrasp and privileged particle truth are not contact-grasp or real-scale validation."]}


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
    need(not path.exists(), "Output directory must be new; preserve previous selected results")
    path.mkdir(parents=True)
    return path


def main():
    args = arguments()
    summary = curate(args.run, args.audit, args.audit_sha256)
    if args.check_only:
        print(json.dumps({"complete": True, "actual_cases": summary["actual_cases"], "outcome": summary["outcome"]})); return
    output = new_output(args.output)
    (output / "first40_results.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf8")
    print("Wrote " + str(output / "first40_results.json"))


if __name__ == "__main__":
    main()


