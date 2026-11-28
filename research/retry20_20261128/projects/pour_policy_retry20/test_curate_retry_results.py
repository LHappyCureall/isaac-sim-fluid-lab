"""CPU counterexamples only; fixtures are not real liquid simulation results."""

import copy
import json
from pathlib import Path
import tempfile
import unittest

import curate_retry_results as curate


TEST_TEMP = Path(__file__).resolve().parent / "_cpu_test_temp"
TEST_TEMP.mkdir(exist_ok=True)


def bind_retry_report(row, report):
    starts = [event for event in row["response_events"] if event["type"] == "response_cycle_start"]
    ends = [event for event in row["response_events"] if event["type"] == "response_cycle_complete"]
    decisions = row["status"]["retry_decisions"]
    retry = {"first_coarse_anchor": copy.deepcopy(row["status"]["prefix_anchor_observation"]),
        "conditional_decisions": copy.deepcopy(decisions), "retry_intent_allowed": bool(decisions and decisions[0]["retry_allowed"]),
        "actual_cycles_started": len(starts), "actual_second_cycle_executed": len(starts) == 2,
        "actual_cycles_completed": len(ends), "actual_second_cycle_complete": len(ends) == 2}
    if row["stage"] == "pilot":
        retry["seen_case_first_cycle_reference"] = {"seen_development_case_not_independent": True,
            "original_failed_received_ml": 19.3, "first_cycle_end_physical_s": 11., "first_cycle_frames": 660,
            "first_cycle_water_motion_partitions_exactly_equal": True, "first_cycle_original_events_exactly_equal": True,
            "all_completed_frame_substep_counts_exactly_equal": True,
            "zero_time_inherited_last_substeps": {"reference": 6, "pilot": 4, "is_actual_completed_frame": False,
                "excluded_from_equivalence_only": True, "source_file": "synthetic_fixture_not_physics.py",
                "source_sha256": "0" * 64, "reason": "synthetic fixture, never a physical observation"},
            "source_reference_sha256": {str(index): str(index) * 64 for index in range(4)}}
    report["actual_retry_evidence"] = retry


def fixture(*, enabled=False, second=False, stage="test", seed=196301, received=None):
    received = (20.05 if second else 19.3) if received is None else received
    assessments = {}
    for name, tolerance in (("refine", .5), ("easy", 1.), ("strict", .02)):
        dose = 20. - tolerance <= received <= 20. + tolerance
        assessments[name] = {"valid_evidence": True, "dose_passed": dose,
            "full_home_within_deadline": True, "maximum_all_label_spilled_particles": 0,
            "spill_passed": True, "no_late_dose_change": True,
            "tolerance_ml": tolerance, "feasible": dose}
    anchor = {"coarse_cut_tilt_deg": 90., "received_ml": 19.25,
        "physical_time_s": 3., "actual_target_ml": 20., "control_target_ml": 20.,
        "source_inside_velocity_rms_m_s": .08}
    first_end = 19.3 if second else received
    events = [{"type": "response_cycle_start", "physical_time_s": 3., "approach_tilt_deg": 84.,
        "observation": {"received_ml": 19.25}},
        {"type": "response_cycle_complete", "physical_time_s": 11.,
        "post_observation": {"received_ml": first_end},
        "outcome": {"settled_delta_ml": first_end - 19.25}, "settled": True, "action_execution_complete": True,
        "executed_micro": True}]
    decisions = []
    if second:
        decisions = [{"physical_time_s": 11., "completed_cycles_before_decision": 1,
            "actual_received_ml": 19.3, "actual_remaining_ml": 37., "actual_deficit_ml": .7,
            "actual_settled_measurement": True, "deficit_exceeds_half_ml": True,
            "observed_all_label_peak_spill_particles": 0, "spill_guard_passed": True,
            "declared_retry_approach_tilt_deg": 84.1, "conservative_full_home_budget_s": 11.1,
            "deadline_physical_seconds": 24., "remaining_deadline_physical_seconds": 13.,
            "full_home_budget_fits": True, "retry_allowed": True,
            "reason": "actual_quiet_undershoot_retry_allowed", "budget_is_robot_motion_not_predicted_water_dose": True}]
        events.extend([{"type": "response_cycle_start", "physical_time_s": 11., "approach_tilt_deg": 84.1,
            "observation": {"received_ml": 19.3}},
            {"type": "response_cycle_complete", "physical_time_s": 20.,
            "post_observation": {"received_ml": received},
            "outcome": {"settled_delta_ml": received - 19.3}, "settled": True, "action_execution_complete": False,
            "executed_micro": False}])
    elapsed = 21.5 if second else 11.8
    row = {"episode_id": "synthetic_not_physical", "stage": stage,
        "split": "development" if stage == "pilot" else "test", "seed": seed,
        "role": "candidate" if enabled else "baseline", "target_ml": 20.,
        "declared_policy": {"enable_retry": enabled},
        "status": {"retry_enabled": enabled, "retry_maximum_cycles": 2 if enabled else 1,
            "retry_first_approach_tilt_deg": 84., "retry_second_approach_tilt_deg": 84.1,
            "response_cycle_count": 2 if second else 1, "retry_decisions": decisions,
            "retry_actual_settled_home_decision": received >= 19.5,
            "prefix_anchor_observation": anchor},
        "result": {"target_ml": 20., "received_ml": received, "error_ml": received - 20.,
            "absolute_error_ml": abs(received - 20.), "physical_time_s": elapsed,
            "physical_steps": round(elapsed * 60), "wall_seconds": 50.,
            "mass_conserved": True, "all_states_finite": True},
        "response_events": events, "post_observation": {"duration_s": 2., "physical_steps": 120},
        "wall_seconds": 60., "started_at_utc": "2026-11-28T01:00:00+00:00",
        "completed_at_utc": "2026-11-28T01:01:00+00:00"}
    for name, value in assessments.items():
        row[name + "_assessment"] = value
    report = {key: row[key] for key in ("episode_id", "stage", "role", "seed")}
    report.update(received_ml=received, error_ml=received - 20., physical_s=elapsed)
    for name, value in assessments.items():
        report[name + "_assessment"] = copy.deepcopy(value)
    bind_retry_report(row, report)
    return row, report


def synthetic_summary():
    row, report = fixture(enabled=True, second=True, stage="pilot", seed=195301)
    cases = [curate.compact_case(row, "pilot", report)]
    for index, seed in enumerate(sorted(curate.SEEDS["test"])):
        for enabled in (False, True):
            needs_second = index < 5
            row, report = fixture(enabled=enabled, second=enabled and needs_second,
                seed=seed, received=20.05 if enabled or not needs_second else 19.3)
            row["episode_id"] = report["episode_id"] = f"synthetic_{seed}_{enabled}"
            cases.append(curate.compact_case(row, "test", report))
    return {"cases": cases, "stage_metrics": {stage: {role: curate.metrics([case for case in cases
        if case["stage"] == stage and case["role"] == role])
        for role in {case["role"] for case in cases if case["stage"] == stage}} for stage in curate.COUNTS}}


def synthetic_pair_diagnostics(cases):
    result = []
    for seed in sorted(curate.SEEDS["test"]):
        group = {case["role"]: case for case in cases if case["stage"] == "test" and case["seed"] == seed}
        result.append({"seed": seed, "verified": True, "baseline_episode_id": group["baseline"]["episode_id"],
            "candidate_episode_id": group["candidate"]["episode_id"], "first_cycle_end_physical_s": 11.,
            "first_cycle_frames": 661, "original_first_cycle_events_exactly_equal": True,
            "actual_water_motion_partitions_exactly_equal": True,
            "zero_time_inherited_last_substeps": {"baseline": 4, "candidate": 6,
                "is_actual_completed_frame": False, "excluded_from_equivalence_only": True}})
    return result


class CurationGuards(unittest.TestCase):
    def test_retry_is_actual_second_cycle_not_only_intent(self):
        row, report = fixture(enabled=True, second=True)
        case = curate.compact_case(row, "test", report)
        self.assertTrue(case["refine_feasible"])
        self.assertFalse(case["original_strict_feasible"])
        self.assertTrue(case["actual_second_cycle_started"])
        self.assertEqual(case["actual_correction_cycles_started"], 2)
        self.assertTrue(case["actual_cycles"][1]["micro_skipped"])
        self.assertFalse(case["actual_cycles"][1]["action_execution_complete"])
        row["response_events"] = row["response_events"][:2]
        row["status"]["response_cycle_count"] = 1
        bind_retry_report(row, report)
        case = curate.compact_case(row, "test", report)
        self.assertTrue(case["retry_allowed"])
        self.assertFalse(case["actual_second_cycle_started"])

    def test_second_cycle_requires_all_actual_guards(self):
        for field, value in (("actual_settled_measurement", False), ("actual_received_ml", 19.5)):
            row, report = fixture(enabled=True, second=True)
            row["status"]["retry_decisions"][0][field] = value
            row["status"]["retry_decisions"][0]["actual_deficit_ml"] = 20. - row["status"]["retry_decisions"][0]["actual_received_ml"]
            with self.assertRaisesRegex(ValueError, "Retry must use actual settled"):
                curate.compact_case(row, "test", report)
        row, report = fixture(enabled=True, second=True)
        row["status"]["retry_decisions"][0]["observed_all_label_peak_spill_particles"] = 2
        with self.assertRaisesRegex(ValueError, "spill guard"):
            curate.compact_case(row, "test", report)
        row, report = fixture(enabled=True, second=True)
        row["status"]["retry_decisions"][0]["conservative_full_home_budget_s"] = 13.00000001
        with self.assertRaisesRegex(ValueError, "full-home time guard"):
            curate.compact_case(row, "test", report)

    def test_baseline_cannot_hide_second_cycle(self):
        row, report = fixture(enabled=False, second=True)
        with self.assertRaisesRegex(ValueError, "cycle budget changed"):
            curate.compact_case(row, "test", report)

    def test_partial_cycle_remains_failure_without_invented_endpoint(self):
        row, report = fixture(enabled=True, second=True)
        row["response_events"].pop()
        for name in ("refine", "easy", "strict"):
            row[name + "_assessment"].update(valid_evidence=False, feasible=False)
            report[name + "_assessment"].update(valid_evidence=False, feasible=False)
        bind_retry_report(row, report)
        case = curate.compact_case(row, "test", report)
        self.assertFalse(case["refine_feasible"])
        self.assertEqual(case["actual_correction_cycles_ended"], 1)
        self.assertFalse(case["actual_cycles"][1]["has_recorded_endpoint"])
        self.assertNotIn("outcome", case["actual_cycles"][1])

    def test_primary_band_is_closed_without_particle_widening(self):
        for volume in (19.5, 20.5):
            row, report = fixture(received=volume)
            self.assertTrue(curate.compact_case(row, "test", report)["refine_feasible"])
        row, report = fixture(received=20.500000000000004)
        row["refine_assessment"].update(dose_passed=True, feasible=True)
        report["refine_assessment"].update(dose_passed=True, feasible=True)
        with self.assertRaisesRegex(ValueError, "Closed actual dose gate"):
            curate.compact_case(row, "test", report)

    def test_old_seed_pilot_is_development_not_fresh_test(self):
        row, report = fixture(enabled=True, second=True, stage="pilot", seed=195301)
        case = curate.compact_case(row, "pilot", report)
        self.assertTrue(case["known_previous_test_seed_used_as_development"])
        row["split"] = "test"
        with self.assertRaisesRegex(ValueError, "Pilot development"):
            curate.compact_case(row, "pilot", report)

    def test_audited_measurement_and_post_clock_cannot_be_changed(self):
        row, report = fixture()
        report["received_ml"] += .019683
        with self.assertRaisesRegex(ValueError, "Audited measurement"):
            curate.compact_case(row, "test", report)
        row, report = fixture()
        row["post_observation"]["physical_steps"] = 119
        with self.assertRaisesRegex(ValueError, "120 live steps"):
            curate.compact_case(row, "test", report)
        row, report = fixture()
        row["result"]["physical_steps"] += 1
        with self.assertRaisesRegex(ValueError, "step/60"):
            curate.compact_case(row, "test", report)

    def test_incomplete_audit_rejected_before_test_journal_read(self):
        with tempfile.TemporaryDirectory(dir=TEST_TEMP) as temporary:
            run = Path(temporary); audit = run / "overall_audit.json"
            audit.write_text(json.dumps({"complete": False, "pass": True}), encoding="utf8")
            with self.assertRaisesRegex(ValueError, "Complete passing independent"):
                curate.curate(run, audit, curate.sha(audit))
            with self.assertRaisesRegex(ValueError, "audit SHA"):
                curate.curate(run, audit, "0" * 64)

    def test_paired_raw_proof_is_bound_to_actual_ids_endpoints_and_counter_scope(self):
        cases = synthetic_summary()["cases"]
        records = synthetic_pair_diagnostics(cases)
        counter = cases[0]["known_pilot_reference"]["zero_time_inherited_last_substeps"]
        compact = curate.pair_equivalence(records, cases, counter)
        self.assertEqual(len(compact), 10)
        self.assertTrue(all(row["all_completed_frame_substep_counts_exactly_equal"] for row in compact))
        records[0]["first_cycle_end_physical_s"] += 1. / 60.
        with self.assertRaisesRegex(ValueError, "Verified first-cycle time"):
            curate.pair_equivalence(records, cases, counter)
        records = synthetic_pair_diagnostics(cases)
        records[0]["zero_time_inherited_last_substeps"]["is_actual_completed_frame"] = True
        with self.assertRaisesRegex(ValueError, "may not hide completed physical frames"):
            curate.pair_equivalence(records, cases, counter)

    def test_unavailable_or_unequal_pair_cannot_acquire_unproven_substep_equality(self):
        cases = synthetic_summary()["cases"]
        records = synthetic_pair_diagnostics(cases)
        counter = cases[0]["known_pilot_reference"]["zero_time_inherited_last_substeps"]
        records[0] = {"seed": 196301, "verified": False, "endpoint_unavailable": True}
        case = next(row for row in cases if row["stage"] == "test" and row["seed"] == 196301 and row["role"] == "baseline")
        case["actual_cycles"] = []
        records[1]["verified"] = False
        records[1]["actual_water_motion_partitions_exactly_equal"] = False
        compact = curate.pair_equivalence(records, cases, counter)
        self.assertIsNone(compact[0]["all_completed_frame_substep_counts_exactly_equal"])
        self.assertIsNone(compact[1]["all_completed_frame_substep_counts_exactly_equal"])
        self.assertFalse(compact[0]["verified"])
        self.assertNotIn("zero_time_inherited_last_substeps", compact[0])

    def test_synthetic_plot_keeps_all_pairs_and_separate_criteria(self):
        import plot_retry_results as plot
        summary = synthetic_summary()
        self.assertEqual(len(summary["cases"]), 21)
        self.assertEqual(summary["stage_metrics"]["test"]["candidate"]["actual_second_cycle_cases"], 5)
        figure = plot.render(summary, synthetic=True)
        self.assertEqual(len(figure.axes), 7)
        self.assertEqual([bar.get_height() for bar in figure.axes[3].patches], [5, 10, 0, 10, 10, 0])
        self.assertTrue(any("SYNTHETIC CPU FIXTURE" in item.get_text() for item in figure.texts))
        figure.savefig(TEST_TEMP / "retry_plot_fixture.png", dpi=80)
        plot.plotting().close(figure)


if __name__ == "__main__":
    unittest.main()
