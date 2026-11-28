"""CPU-only coarse-speed calibration guards; no fixture represents physics."""

import copy
import json
from pathlib import Path
import tempfile
import unittest

import curate_coarse_results as curate

TEST_TEMP = Path(__file__).resolve().parent / "_cpu_test_temp"
TEST_TEMP.mkdir(exist_ok=True)

def fixture(candidate_id="coarse_252", *, second=False, stage="test", seed=198401, received=None, elapsed=None, peak=0):
    goal, enabled = 40., True
    received = goal + .05 if received is None else received
    first, retry = 87., 87.2
    speed = curate.SPEEDS[candidate_id]
    policy = {"enable_retry": enabled, "first_approach_tilt_deg": first, "retry_approach_tilt_deg": retry, "coarse_dose_tilt_deg_s": speed}
    anchor = {"coarse_cut_tilt_deg": 94., "received_ml": goal - .75, "physical_time_s": 3.,
        "actual_target_ml": goal, "control_target_ml": goal, "source_inside_velocity_rms_m_s": .08,
        "settled_measurement": True}
    first_dose = goal - .7 if second else received
    events = [{"type": "response_cycle_start", "physical_time_s": 3., "approach_tilt_deg": first,
        "observation": {"received_ml": goal - .75}},
        {"type": "response_cycle_complete", "physical_time_s": 11., "post_observation": {"received_ml": first_dose},
        "outcome": {"settled_delta_ml": first_dose - (goal - .75)}, "settled": True,
        "action_execution_complete": True, "executed_micro": True}]
    decisions = []
    if second:
        decisions = [{"physical_time_s": 11., "declared_target_ml": goal, "target_profile_key": format(goal, "g"),
            "target_profile_id": "synthetic_fixture", "target_profile_sha256": "0" * 64,
            "completed_cycles_before_decision": 1, "actual_received_ml": goal - .7,
            "actual_remaining_ml": 20., "actual_deficit_ml": .7, "actual_settled_measurement": True,
            "deficit_exceeds_half_ml": True, "observed_all_label_peak_spill_particles": 0,
            "spill_guard_passed": True, "declared_retry_approach_tilt_deg": retry, "declared_coarse_dose_tilt_deg_s": speed,
            "conservative_full_home_budget_s": 11.1, "deadline_physical_seconds": 24.,
            "remaining_deadline_physical_seconds": 13., "full_home_budget_fits": True, "retry_allowed": True,
            "budget_is_robot_motion_not_predicted_water_dose": True}]
        events.extend([{"type": "response_cycle_start", "physical_time_s": 11., "approach_tilt_deg": retry,
            "observation": {"received_ml": goal - .7}},
            {"type": "response_cycle_complete", "physical_time_s": 20., "post_observation": {"received_ml": received},
            "outcome": {"settled_delta_ml": received - (goal - .7)}, "settled": True,
            "action_execution_complete": False, "executed_micro": False}])
    elapsed = (21.5 if second else 14.2) if elapsed is None else elapsed
    row = {"episode_id": f"synthetic_{stage}_{candidate_id}_{seed}", "stage": stage,
        "split": "development" if stage != "test" else "test", "role": "baseline" if candidate_id == "baseline" else "candidate",
        "candidate_id": candidate_id, "profile_key": candidate_id,
        "seed": seed, "target_ml": goal, "declared_policy": policy,
        "status": {"coarse40_condition_seed": seed, "coarse_dose_tilt_deg_s": speed,
            "policy_params": {"dose_tilt_deg_s": speed, "coarse_dose_tilt_deg_s": speed},
            "only_coarse_dose_speed_varies": True, "only_first_angle_varies": False, "condition_seed_provided_by_runner": True,
            "retry_enabled": enabled, "retry_maximum_cycles": 2 if enabled else 1,
            "retry_profile_key": format(goal, "g"), "retry_profile_id": "synthetic_fixture", "retry_profile_sha256": "0" * 64,
            "retry_first_approach_tilt_deg": first, "retry_second_approach_tilt_deg": retry,
            "response_cycle_count": 2 if second else 1, "retry_decisions": decisions,
            "retry_actual_settled_home_decision": abs(received - goal) <= .5, "prefix_anchor_observation": anchor},
        "result": {"target_ml": goal, "received_ml": received, "error_ml": received - goal,
            "absolute_error_ml": abs(received - goal), "physical_time_s": elapsed, "physical_steps": round(elapsed * 60),
            "wall_seconds": 50., "mass_conserved": True, "all_states_finite": True, "complete": True,
            "valid_physics": True, "measurement_physical_time_s": elapsed,
            "measurement_basis": "liquid_measurement_at_reported_sample_time"},
        "response_events": events, "post_observation": {"duration_s": 2., "physical_steps": 120,
            "all_states_finite": True, "mass_array_unchanged": True}, "wall_seconds": 60.,
        "started_at_utc": "2026-11-28T01:00:00+00:00", "completed_at_utc": "2026-11-28T01:01:00+00:00"}
    for name, tolerance in (("refine", .5), ("easy", 1.), ("strict", goal * .001)):
        dose = goal - tolerance <= received <= goal + tolerance
        row[name + "_assessment"] = {"valid_evidence": True, "dose_passed": dose,
            "full_home_within_deadline": True, "maximum_all_label_spilled_particles": peak,
            "spill_passed": peak <= 1, "no_late_dose_change": True, "tolerance_ml": tolerance, "feasible": bool(dose and peak <= 1)}
    report = {key: copy.deepcopy(row[key]) for key in ("episode_id", "stage", "role", "seed", "candidate_id", "profile_key", "declared_policy",
        "refine_assessment", "easy_assessment", "strict_assessment")}
    report.update(received_ml=received, error_ml=received - goal, physical_s=elapsed,
        post_frames=120, unlogged_terminal_action_frames=0, failure_records=[],
        actual_retry_evidence={"first_coarse_anchor": copy.deepcopy(anchor), "conditional_decisions": copy.deepcopy(decisions),
            "retry_intent_allowed": second, "actual_cycles_started": 2 if second else 1,
            "actual_second_cycle_executed": second, "actual_cycles_completed": 2 if second else 1,
            "actual_second_cycle_complete": second})
    return row, report


def summary_fixture(stopped=False, validation=True):
    cases = []
    for seed in sorted(curate.KNOWN_SEEDS):
        for candidate_id, dose, elapsed in (("baseline", 40.2, 14.2),
                ("coarse_252", 40.2, 14.4), ("coarse_248", 40., 14.5)):
            row, report = fixture(candidate_id, stage="calibration", seed=seed,
                received=dose, elapsed=elapsed, peak=2 if candidate_id == "baseline" and seed == 198304 else 0)
            cases.append(curate.compact_case(row, "calibration", report))
    stages = ["calibration"]
    if validation:
        stages.append("validation")
        for seed in sorted(curate.VALIDATION_SEEDS):
            for candidate_id in ("baseline", "coarse_252"):
                row, report = fixture(candidate_id, stage="validation", seed=seed,
                    received=40.55 if candidate_id == "baseline" else 40.1)
                cases.append(curate.compact_case(row, "validation", report))
    if not stopped:
        stages.append("test")
        for index, seed in enumerate(sorted(curate.TEST_SEEDS)):
            for candidate_id in ("baseline", "coarse_252"):
                row, report = fixture(candidate_id, seed=seed,
                    received=(40.55 if index == 0 else 40.35) if candidate_id == "baseline" else 40.1)
                cases.append(curate.compact_case(row, "test", report))
    return {"cases": cases, **curate.summarize_cases(cases, stages, stopped)}


def prepare_records(summary):
    result = []
    for stage in summary["actual_stages"]:
        seeds = curate.KNOWN_SEEDS if stage == "calibration" else (
            curate.VALIDATION_SEEDS if stage == "validation" else curate.TEST_SEEDS)
        keys = curate.CANDIDATE_ORDER if stage == "calibration" else ("coarse_252",)
        for seed in sorted(seeds):
            for candidate_id in keys:
                base = next(case for case in summary["cases"] if case["stage"] == stage
                    and case["seed"] == seed and case["candidate_id"] == "baseline")
                candidate = next(case for case in summary["cases"] if case["stage"] == stage
                    and case["seed"] == seed and case["candidate_id"] == candidate_id)
                result.append({"stage": stage, "seed": seed, "candidate_id": candidate_id,
                    "baseline_episode_id": base["episode_id"], "candidate_episode_id": candidate["episode_id"],
                    "verified": True, "endpoint_unavailable": False,
                    "comparison_endpoint_kind": "actual_prepared78_before_coarse_dose_speed_treatment",
                    "prepare_endpoint_physical_s": 1.5, "prepare_prefix_frames": 91,
                    "maximum_compared_dense_time_s": [1.5, 1.5], "no_future_kinematics_included": True,
                    "actual_prepare_water_motion_partitions_exactly_equal": True,
                    "post_coarse_anchor_equality_is_not_required": True,
                    "whole_first_cycle_equality_is_not_required": True,
                    "zero_time_inherited_last_substeps": {"baseline": 4, "candidate": 6,
                        "is_actual_completed_frame": False, "excluded_from_equivalence_only": True}})
    return result


class CurationGuards(unittest.TestCase):
    def test_feasible_group_ranks_only_mean_physical_time_then_declared_order(self):
        cases = summary_fixture()["cases"]
        self.assertEqual(curate.choose_calibration(cases), "coarse_252")
        for case in cases:
            if case["stage"] == "calibration" and case["candidate_id"] == "coarse_248":
                case["physical_action_seconds"] = 14.4
        self.assertEqual(curate.choose_calibration(cases), "coarse_252")
        # Smaller error never beats a faster all-joint-feasible group.
        self.assertEqual(curate.calibration_group_key([case for case in cases
            if case["stage"] == "calibration" and case["candidate_id"] == "coarse_248"]), (0, 14.4))

    def test_one_failed_condition_disqualifies_group_from_feasible_tier(self):
        cases = summary_fixture()["cases"]
        chosen = next(case for case in cases if case["stage"] == "calibration"
            and case["candidate_id"] == "coarse_252" and case["seed"] == 198304)
        chosen.update(refine_feasible=False, peak_all_label_spill_particles=2)
        self.assertEqual(curate.choose_calibration(cases), "coarse_248")
        chosen["valid_evidence"] = False
        self.assertEqual(curate.choose_calibration(cases), "coarse_248")
        for case in cases:
            if case["stage"] == "calibration" and case["role"] == "candidate":
                case["valid_evidence"] = False
        self.assertIsNone(curate.choose_calibration(cases))

    def test_infeasible_groups_rank_mae_before_total_spill_without_time(self):
        cases = summary_fixture()["cases"]
        groups = {candidate: [case for case in cases if case["stage"] == "calibration"
            and case["candidate_id"] == candidate] for candidate in curate.CANDIDATE_ORDER}
        for candidate, group in groups.items():
            for case in group:
                case.update(refine_feasible=False, absolute_error_ml=.6 if candidate == "coarse_252" else 10.,
                    peak_all_label_spill_particles=2 if candidate == "coarse_252" else 0,
                    physical_action_seconds=23. if candidate == "coarse_252" else 3.)
        self.assertEqual(curate.choose_calibration(cases), "coarse_252")
        for case in groups["coarse_248"]:
            case["absolute_error_ml"] = .6
        self.assertEqual(curate.choose_calibration(cases), "coarse_248")
        self.assertEqual(curate.calibration_group_key(groups["coarse_248"]), (1, .6, 0))

    def test_spill_two_baseline_is_valid_negative_not_deleted_or_selected(self):
        summary = summary_fixture()
        baseline = next(case for case in summary["cases"] if case["stage"] == "calibration"
            and case["candidate_id"] == "baseline" and case["seed"] == 198304)
        self.assertTrue(baseline["valid_evidence"])
        self.assertFalse(baseline["refine_feasible"])
        self.assertEqual(baseline["peak_all_label_spill_particles"], 2)
        self.assertEqual(summary["calibration_metrics_by_candidate"]["baseline"]["joint_passes"], 1)
        self.assertNotEqual(summary["calibration_selected_candidate_id_recomputed"], "baseline")

    def test_both_retry_policies_keep_first_and_retry_angles_fixed(self):
        for candidate_id, speed in curate.SPEEDS.items():
            row, report = fixture(candidate_id)
            case = curate.compact_case(row, "test", report)
            self.assertTrue(case["retry_enabled"])
            self.assertEqual(case["first_approach_tilt_deg"], 87.)
            self.assertEqual(case["retry_approach_tilt_deg"], 87.2)
            self.assertEqual(case["coarse_dose_tilt_deg_s"], speed)
        row, report = fixture()
        row["declared_policy"]["first_approach_tilt_deg"] = 87.05
        report["declared_policy"] = copy.deepcopy(row["declared_policy"])
        with self.assertRaisesRegex(ValueError, "Both roles keep first87"):
            curate.compact_case(row, "test", report)

    def test_actual_speed_alias_and_condition_seed_are_bound(self):
        row, report = fixture()
        row["status"]["policy_params"]["dose_tilt_deg_s"] = 25.6
        with self.assertRaisesRegex(ValueError, "Underlying coarse policy speed"):
            curate.compact_case(row, "test", report)
        row, report = fixture()
        row["status"]["coarse40_condition_seed"] += 1
        with self.assertRaisesRegex(ValueError, "condition seed metadata"):
            curate.compact_case(row, "test", report)

    def test_original_strict_and_inclusive_primary_bands_remain_separate(self):
        row, report = fixture(received=40.04)
        self.assertTrue(curate.compact_case(row, "test", report)["original_strict_feasible"])
        row, report = fixture(received=40.50000000000001)
        self.assertFalse(curate.compact_case(row, "test", report)["refine_feasible"])
        row["refine_assessment"].update(dose_passed=True, feasible=True)
        report["refine_assessment"].update(dose_passed=True, feasible=True)
        with self.assertRaisesRegex(ValueError, "Closed actual dose"):
            curate.compact_case(row, "test", report)

    def test_all_six_thirty_and_stop_budgets_preserve_failures(self):
        summary = summary_fixture()
        self.assertEqual(summary["actual_cases"], 30)
        self.assertEqual(summary["budget"], {"calibration": 6, "validation": 4, "test": 20})
        self.assertEqual(len(summary["test_pairs"]), 10)
        self.assertLess(summary["test_pairs"][0]["candidate_minus_baseline_absolute_error_ml"], 0.)
        with self.assertRaisesRegex(ValueError, "An actual failed/partial case was lost"):
            curate.summarize_cases(summary["cases"][:-1], summary["actual_stages"], False)
        for validation, expected in ((False, 6), (True, 10)):
            stopped = summary_fixture(stopped=True, validation=validation)
            self.assertEqual(stopped["actual_cases"], expected)
            self.assertEqual(stopped["test_pairs"], [])
            self.assertNotIn("test", stopped["stage_metrics"])

    def test_partial_failure_preserves_null_anchor_post_and_unlogged_basis(self):
        row, report = fixture(received=0.)
        row["response_events"] = []
        row["status"].update(prefix_anchor_observation=None, response_cycle_count=0)
        row["post_observation"] = None
        row["result"].update(physical_time_s=1. / 60., physical_steps=1, complete=False, valid_physics=False,
            measurement_physical_time_s=0., measurement_basis="last_complete_measurement_before_partial_fluid_failure")
        report.update(physical_s=1. / 60., post_frames=0, unlogged_terminal_action_frames=1,
            failure_records=[{"type": "failure", "step": 1, "failure_stage": "fluid_step",
                "error": "Synthetic failure at <synthetic-private-path>/person/source.py",
                "partial_unlogged_fluid_frame_possible": True}])
        report["actual_retry_evidence"].update(first_coarse_anchor=None, actual_cycles_started=0, actual_cycles_completed=0)
        for name in ("refine", "easy", "strict"):
            row[name + "_assessment"].update(valid_evidence=False, full_home_within_deadline=False,
                no_late_dose_change=False, feasible=False)
            report[name + "_assessment"] = copy.deepcopy(row[name + "_assessment"])
        case = curate.compact_case(row, "test", report)
        self.assertFalse(case["refine_feasible"])
        self.assertIsNone(case["first_quiet_anchor_received_ml"])
        self.assertIsNone(case["post_physical_seconds"])
        self.assertEqual(case["received_measurement_physical_time_s"], 0.)
        self.assertEqual(case["unlogged_terminal_action_frames"], 1)
        self.assertNotIn("<synthetic-private-path>", json.dumps(case))

    def test_complete_independent_audit_precedes_journal_reads(self):
        with tempfile.TemporaryDirectory(dir=TEST_TEMP) as temporary:
            run = Path(temporary); audit = run / "overall_audit.json"
            audit.write_text(json.dumps({"complete": False, "pass": True}), encoding="utf8")
            with self.assertRaisesRegex(ValueError, "Complete passing independent audit"):
                curate.curate(run, audit, curate.sha(audit))
            with self.assertRaisesRegex(ValueError, "audit SHA"):
                curate.curate(run, audit, "0" * 64)

    def test_prepare_only_control_allows_changed_post_coarse_outcomes(self):
        summary = summary_fixture()
        rows = prepare_records(summary)
        for case in summary["cases"]:
            if case["candidate_id"] != "baseline":
                case["first_quiet_anchor_received_ml"] += .5
                case["first_quiet_anchor_physical_seconds"] += .1
        proofs = curate.compact_prepare_proofs(rows, summary["cases"], summary["actual_stages"], "coarse_252")
        self.assertEqual(len(proofs), 16)
        self.assertNotEqual(proofs[0]["baseline_post_coarse_anchor_received_ml"], proofs[0]["candidate_post_coarse_anchor_received_ml"])
        rows[0]["maximum_compared_dense_time_s"] = [1.5, 1.50001]
        with self.assertRaisesRegex(ValueError, "future treatment kinematics"):
            curate.compact_prepare_proofs(rows, summary["cases"], summary["actual_stages"], "coarse_252")

    def test_prepare_proof_cannot_move_endpoint_after_treatment_or_hide_frame(self):
        summary = summary_fixture(stopped=True, validation=False)
        rows = prepare_records(summary)
        rows[0]["comparison_endpoint_kind"] = "post_coarse_quiet78"
        with self.assertRaisesRegex(ValueError, "Speed-treatment effects"):
            curate.compact_prepare_proofs(rows, summary["cases"], summary["actual_stages"], "coarse_252")
        rows = prepare_records(summary)
        rows[0]["zero_time_inherited_last_substeps"]["is_actual_completed_frame"] = True
        with self.assertRaisesRegex(ValueError, "t0 counter exception"):
            curate.compact_prepare_proofs(rows, summary["cases"], summary["actual_stages"], "coarse_252")

    def test_known_reference_retains_positive_negative_and_removes_private_paths(self):
        for feasible, peak in ((True, 1), (False, 2)):
            raw = {"verified": True, "original_reference_primary_feasible": feasible,
                "original_reference_received_ml": 39.9, "original_reference_peak_spill_particles": peak,
                "source_reference_sha256": {"<synthetic-private-path>/old.steps.jsonl.gz": "a" * 64}}
            result = curate.compact_known_reference(raw)
            self.assertIs(result["original_reference_primary_feasible"], feasible)
            self.assertEqual(result["original_reference_peak_spill_particles"], peak)
            self.assertEqual(result["reference_artifact_sha256"], ["a" * 64])
            self.assertNotIn("<synthetic-private-path>", json.dumps(result))

    def test_whole_second_cycle_counts_independently_of_micro_execution(self):
        row, report = fixture(second=True)
        case = curate.compact_case(row, "test", report)
        self.assertTrue(case["actual_second_cycle_started"])
        self.assertFalse(case["actual_cycles"][1]["executed_micro"])
        self.assertTrue(case["actual_cycles"][1]["micro_skipped"])
        row, report = fixture()
        row["response_events"][-1]["settled"] = False
        row["response_events"][-1]["type"] = "response_cycle_unsettled"
        report["actual_retry_evidence"]["actual_cycles_completed"] = 0
        self.assertEqual(curate.compact_case(row, "test", report)["second_cycle_execution_or_skip_reason"],
            "actual_first_cycle_endpoint_unsettled")

    def test_synthetic_figures_cover_full_and_two_actual_gate_stop_branches(self):
        import plot_coarse_results as plot
        for stopped, validation in ((False, True), (True, True), (True, False)):
            summary = summary_fixture(stopped=stopped, validation=validation)
            figure = plot.render(summary, synthetic=True)
            self.assertEqual(len(figure.axes), 4 if stopped else 9)
            self.assertTrue(any("SYNTHETIC CPU FIXTURE" in item.get_text() for item in figure.texts))
            figure.savefig(TEST_TEMP / f"synthetic_stop{stopped}_validation{validation}.png", dpi=80)
            plot.plotting().close(figure)


if __name__ == "__main__":
    unittest.main()


