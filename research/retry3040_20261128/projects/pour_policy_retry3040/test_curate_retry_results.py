"""CPU counterexamples only; all fixtures explicitly are not physical results."""

import copy
import json
from pathlib import Path
import tempfile
import unittest

import curate_retry_results as curate

TEST_TEMP = Path(__file__).resolve().parent / "_cpu_test_temp"
TEST_TEMP.mkdir(exist_ok=True)


def fixture(goal, *, enabled=False, second=False, stage="test", seed=197301, received=None):
    received = goal + .05 if received is None else received
    first, retry = ((85.5, 85.6) if goal == 30. else (87.2, 87.3))
    policy = {"enable_retry": enabled, "first_approach_tilt_deg": first, "retry_approach_tilt_deg": retry}
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
            "spill_guard_passed": True, "declared_retry_approach_tilt_deg": retry,
            "conservative_full_home_budget_s": 11.1, "deadline_physical_seconds": 24.,
            "remaining_deadline_physical_seconds": 13., "full_home_budget_fits": True, "retry_allowed": True,
            "budget_is_robot_motion_not_predicted_water_dose": True}]
        events.extend([{"type": "response_cycle_start", "physical_time_s": 11., "approach_tilt_deg": retry,
            "observation": {"received_ml": goal - .7}},
            {"type": "response_cycle_complete", "physical_time_s": 20., "post_observation": {"received_ml": received},
            "outcome": {"settled_delta_ml": received - (goal - .7)}, "settled": True,
            "action_execution_complete": False, "executed_micro": False}])
    elapsed = 21.5 if second else 11.8
    row = {"episode_id": f"synthetic_{stage}_{goal}_{enabled}_{seed}", "stage": stage,
        "split": "development" if stage != "test" else "test", "role": "candidate" if enabled else "baseline",
        "seed": seed, "target_ml": goal, "declared_policy": policy,
        "status": {"retry_enabled": enabled, "retry_maximum_cycles": 2 if enabled else 1,
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
            "full_home_within_deadline": True, "maximum_all_label_spilled_particles": 0,
            "spill_passed": True, "no_late_dose_change": True, "tolerance_ml": tolerance, "feasible": dose}
    report = {key: copy.deepcopy(row[key]) for key in ("episode_id", "stage", "role", "seed", "declared_policy",
        "refine_assessment", "easy_assessment", "strict_assessment")}
    report.update(received_ml=received, error_ml=received - goal, physical_s=elapsed,
        post_frames=120, unlogged_terminal_action_frames=0, failure_records=[],
        actual_retry_evidence={"first_coarse_anchor": copy.deepcopy(anchor), "conditional_decisions": copy.deepcopy(decisions),
            "retry_intent_allowed": second, "actual_cycles_started": 2 if second else 1,
            "actual_second_cycle_executed": second, "actual_cycles_completed": 2 if second else 1,
            "actual_second_cycle_complete": second})
    return row, report


def summary_fixture(development_stop=False, two_stages=False):
    stages = ["pilot_001"] + (["pilot_002"] if two_stages else [])
    cases = []
    for stage in stages:
        for goal in curate.GOALS:
            for enabled in (False, True):
                row, report = fixture(goal, enabled=enabled, stage=stage, seed=197101,
                    received=goal - .7 if development_stop else goal + .05)
                cases.append(curate.compact_case(row, stage, report))
    if not development_stop:
        stages.append("test")
        for goal in curate.GOALS:
            for index, seed in enumerate(sorted(curate.TEST_SEEDS)):
                for enabled in (False, True):
                    second = enabled and index < 5
                    dose = goal + .05 if enabled or index >= 5 else goal - .7
                    row, report = fixture(goal, enabled=enabled, second=second, seed=seed, received=dose)
                    cases.append(curate.compact_case(row, "test", report))
    return {"cases": cases, **curate.summarize_cases(cases, stages, development_stop)}


class CurationGuards(unittest.TestCase):
    def test_each_goal_preserves_original_relative_strict_secondary(self):
        for goal, tolerance in ((30., .03), (40., .04)):
            row, report = fixture(goal, received=goal + tolerance)
            case = curate.compact_case(row, "test", report)
            self.assertTrue(case["original_strict_feasible"])
            self.assertTrue(case["refine_feasible"])
            row["strict_assessment"]["tolerance_ml"] = .02
            report["strict_assessment"]["tolerance_ml"] = .02
            with self.assertRaisesRegex(ValueError, "Task criterion changed"):
                curate.compact_case(row, "test", report)

    def test_retry_guard_uses_actual_goal_and_explicit_profile(self):
        row, report = fixture(40., enabled=True, second=True)
        case = curate.compact_case(row, "test", report)
        self.assertTrue(case["actual_second_cycle_started"])
        self.assertTrue(case["actual_cycles"][1]["micro_skipped"])
        row["status"]["retry_decisions"][0]["declared_target_ml"] = 20.
        with self.assertRaisesRegex(ValueError, "changed its actual goal"):
            curate.compact_case(row, "test", report)

    def test_exact_primary_band_is_not_widened_to_one_grain(self):
        row, report = fixture(30., received=30.500000000000004)
        self.assertFalse(curate.compact_case(row, "test", report)["refine_feasible"])
        row["refine_assessment"].update(dose_passed=True, feasible=True)
        report["refine_assessment"].update(dose_passed=True, feasible=True)
        with self.assertRaisesRegex(ValueError, "Closed actual dose gate"):
            curate.compact_case(row, "test", report)

    def test_development_stop_keeps_all_failures_and_invents_no_fresh_metrics(self):
        summary = summary_fixture(development_stop=True, two_stages=True)
        self.assertEqual(summary["budget"], {"development": 8, "test": 0, "total": 8,
            "maximum_development": 8, "maximum_total": 48})
        self.assertEqual(summary["test_pairs"], [])
        self.assertTrue(summary["independent_test_not_run"])
        for goal in ("30", "40"):
            self.assertNotIn("test", summary["per_goal_metrics"][goal])
            self.assertEqual(summary["per_goal_metrics"][goal]["pilot_002"]["candidate"]["joint_passes"], 0)

    def test_full_report_requires_all_forty_goal_role_seed_conditions(self):
        summary = summary_fixture()
        self.assertEqual(summary["budget"]["total"], 44)
        self.assertEqual(len(summary["test_pairs"]), 20)
        for goal in ("30", "40"):
            self.assertEqual(summary["per_goal_metrics"][goal]["test"]["candidate"]["cases"], 10)
            self.assertEqual(summary["paired_no_retry_matching"][goal]["no_retry_dose_and_time_exact_pairs"], 5)
            self.assertEqual(summary["paired_no_retry_matching"][goal]["baseline_failure_candidate_pass_pairs"], 5)
        with self.assertRaisesRegex(ValueError, "Actual bounded case count"):
            curate.summarize_cases(summary["cases"][:-1], summary["actual_stages"], False)

    def test_partial_second_cycle_has_no_invented_outcome(self):
        row, report = fixture(30., enabled=True, second=True)
        row["response_events"].pop()
        report["actual_retry_evidence"].update(actual_cycles_completed=1, actual_second_cycle_complete=False)
        case = curate.compact_case(row, "test", report)
        self.assertFalse(case["actual_cycles"][1]["has_recorded_endpoint"])
        self.assertNotIn("outcome", case["actual_cycles"][1])

    def test_no_second_cycle_reason_separates_overdose_in_band_and_time_guard(self):
        row, report = fixture(40., enabled=True, received=40.9)
        self.assertEqual(curate.compact_case(row, "test", report)["second_cycle_execution_or_skip_reason"],
                         "actual_first_cycle_overdose_cannot_be_removed_by_retry")
        row, report = fixture(30., enabled=True, received=30.1)
        self.assertEqual(curate.compact_case(row, "test", report)["second_cycle_execution_or_skip_reason"],
                         "actual_first_cycle_already_in_primary_band")
        row, report = fixture(40., enabled=True, received=39.3)
        source, _ = fixture(40., enabled=True, second=True)
        decision = source["status"]["retry_decisions"][0]
        decision.update(conservative_full_home_budget_s=14., full_home_budget_fits=False, retry_allowed=False)
        row["status"]["retry_decisions"] = [decision]
        report["actual_retry_evidence"].update(conditional_decisions=copy.deepcopy([decision]), retry_intent_allowed=False)
        case = curate.compact_case(row, "test", report)
        self.assertFalse(case["actual_second_cycle_started"])
        self.assertEqual(case["second_cycle_execution_or_skip_reason"], "insufficient_conservative_full_home_time_budget")

    def test_valid_zero_cycle_coarse_band_does_not_invent_a_first_cycle_endpoint(self):
        row, report = fixture(30., enabled=True, received=30.1)
        row["response_events"] = []
        row["status"]["response_cycle_count"] = 0
        row["status"]["prefix_anchor_observation"]["received_ml"] = 30.1
        report["actual_retry_evidence"].update(first_coarse_anchor=copy.deepcopy(row["status"]["prefix_anchor_observation"]),
            actual_cycles_started=0, actual_cycles_completed=0)
        case = curate.compact_case(row, "test", report)
        self.assertTrue(case["refine_feasible"])
        self.assertEqual(case["second_cycle_execution_or_skip_reason"], "actual_coarse_settled_band")
        self.assertEqual(case["actual_cycles"], [])
        self.assertEqual(case["actual_correction_cycles_started"], 0)

    def test_unsettled_in_band_endpoint_is_not_called_settled_success(self):
        row, report = fixture(40., enabled=True, received=40.1)
        row["response_events"][-1].update(type="response_cycle_unsettled", settled=False)
        row["result"]["complete"] = False
        report["actual_retry_evidence"]["actual_cycles_completed"] = 0
        for name in ("refine", "easy", "strict"):
            row[name + "_assessment"].update(full_home_within_deadline=False, feasible=False)
            report[name + "_assessment"] = copy.deepcopy(row[name + "_assessment"])
        case = curate.compact_case(row, "test", report)
        self.assertFalse(case["refine_feasible"])
        self.assertEqual(case["second_cycle_execution_or_skip_reason"], "actual_first_cycle_endpoint_unsettled")
        self.assertFalse(case["actual_cycles"][0]["settled"])

    def test_early_partial_keeps_missing_anchor_post_and_last_measurement_basis(self):
        row, report = fixture(30., enabled=True, received=0.)
        row["response_events"] = []
        row["status"].update(prefix_anchor_observation=None, response_cycle_count=0)
        row["post_observation"] = None
        row["result"].update(physical_time_s=1. / 60., physical_steps=1, complete=False, valid_physics=False,
            termination_reason="invalid_physics_state", measurement_physical_time_s=0.,
            measurement_basis="last_complete_measurement_before_partial_fluid_failure")
        failure = {"type": "failure", "step": 1, "physical_time_s": 1. / 60., "failure_stage": "fluid_step",
            "measurement_physical_time_s": 0., "partial_unlogged_fluid_frame_possible": True,
            "kinematic_evidence_time_s": 0., "error": "Synthetic failure in <synthetic-private-path>/person/source.py"}
        report.update(physical_s=1. / 60., post_frames=0, unlogged_terminal_action_frames=1, failure_records=[failure])
        report["actual_retry_evidence"].update(first_coarse_anchor=None, actual_cycles_started=0, actual_cycles_completed=0)
        for name in ("refine", "easy", "strict"):
            row[name + "_assessment"].update(valid_evidence=False, full_home_within_deadline=False,
                no_late_dose_change=False, feasible=False)
            report[name + "_assessment"] = copy.deepcopy(row[name + "_assessment"])
        case = curate.compact_case(row, "test", report)
        self.assertFalse(case["valid_evidence"])
        self.assertFalse(case["action_complete"])
        self.assertIsNone(case["first_quiet_anchor_received_ml"])
        self.assertIsNone(case["post_physical_seconds"])
        self.assertIsNone(case["post_physical_steps_reported"])
        self.assertEqual(case["post_logged_frames"], 0)
        self.assertEqual(case["received_measurement_physical_time_s"], 0.)
        self.assertEqual(case["received_measurement_basis"], "last_complete_measurement_before_partial_fluid_failure")
        self.assertEqual(case["unlogged_terminal_action_frames"], 1)
        self.assertEqual(len(case["failure_records"]), 1)
        self.assertEqual(len(case["failure_records"][0]["error_message_sha256"]), 64)
        self.assertNotIn("<synthetic-private-path>", json.dumps(case))

    def test_incomplete_overall_audit_is_rejected_before_journal_reads(self):
        with tempfile.TemporaryDirectory(dir=TEST_TEMP) as temporary:
            run = Path(temporary); audit = run / "overall_audit.json"
            audit.write_text(json.dumps({"complete": False, "pass": True}), encoding="utf8")
            with self.assertRaisesRegex(ValueError, "Complete passing independent audit"):
                curate.curate(run, audit, curate.sha(audit))
            with self.assertRaisesRegex(ValueError, "audit SHA"):
                curate.curate(run, audit, "0" * 64)

    def test_pair_proofs_cover_every_actual_goal_and_reject_endpoint_tampering(self):
        summary = summary_fixture()
        records = []
        for stage in summary["actual_stages"]:
            for goal in curate.GOALS:
                for seed in (sorted(curate.TEST_SEEDS) if stage == "test" else [197101]):
                    group = {case["role"]: case for case in summary["cases"] if case["stage"] == stage
                        and case["target_ml"] == goal and case["seed"] == seed}
                    records.append({"stage": stage, "target_ml": goal, "seed": seed, "verified": True,
                        "baseline_episode_id": group["baseline"]["episode_id"],
                        "candidate_episode_id": group["candidate"]["episode_id"],
                        "first_cycle_end_physical_s": 11., "first_cycle_frames": 661,
                        "original_first_cycle_events_exactly_equal": True, "actual_water_motion_partitions_exactly_equal": True,
                        "zero_time_inherited_last_substeps": {"baseline": 4, "candidate": 6,
                            "is_actual_completed_frame": False, "excluded_from_equivalence_only": True}})
        result = curate.compact_pair_proofs(records, summary["cases"], summary["actual_stages"], "0" * 64)
        self.assertEqual(len(result), 22)
        self.assertTrue(all(row["all_completed_frame_substep_counts_exactly_equal"] for row in result))
        records[0]["first_cycle_end_physical_s"] += 1. / 60.
        with self.assertRaisesRegex(ValueError, "endpoint time differs"):
            curate.compact_pair_proofs(records, summary["cases"], summary["actual_stages"], "0" * 64)

    def test_no_cycle_anchor_comparison_does_not_invent_a_first_cycle(self):
        summary = summary_fixture(development_stop=True)
        records = []
        for goal in curate.GOALS:
            group = [case for case in summary["cases"] if case["target_ml"] == goal]
            for case in group:
                case["actual_cycles"] = []
                case["actual_correction_cycles_started"] = case["actual_correction_cycles_ended"] = 0
            ids = {case["role"]: case["episode_id"] for case in group}
            records.append({"stage": "pilot_001", "target_ml": goal, "seed": 197101, "verified": True,
                "baseline_episode_id": ids["baseline"], "candidate_episode_id": ids["candidate"],
                "first_cycle_end_physical_s": 3., "first_cycle_frames": 181,
                "comparison_endpoint_kind": "actual_quiet_coarse_anchor_no_cycle_needed",
                "original_first_cycle_events_exactly_equal": True, "actual_water_motion_partitions_exactly_equal": True,
                "zero_time_inherited_last_substeps": {"baseline": 4, "candidate": 6,
                    "is_actual_completed_frame": False, "excluded_from_equivalence_only": True}})
        compact = curate.compact_pair_proofs(records, summary["cases"], summary["actual_stages"], "0" * 64)
        self.assertEqual(len(compact), 2)
        self.assertNotIn("baseline_actual_first_cycle_end_received_ml", compact[0])
        self.assertEqual(compact[0]["baseline_actual_comparison_received_ml"], 29.25)

    def test_synthetic_both_plot_branches_are_clearly_labeled(self):
        import plot_retry_results as plot
        for stopped in (False, True):
            summary = summary_fixture(development_stop=stopped)
            figure = plot.render(summary, synthetic=True)
            self.assertEqual(len(figure.axes), 4 if stopped else 10)
            self.assertTrue(any("SYNTHETIC CPU FIXTURE" in item.get_text() for item in figure.texts))
            figure.savefig(TEST_TEMP / ("development_stop_fixture.png" if stopped else "full_fixture.png"), dpi=80)
            plot.plotting().close(figure)


if __name__ == "__main__":
    unittest.main()
