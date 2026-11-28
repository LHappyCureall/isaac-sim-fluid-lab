"""CPU-only first-angle calibration guards; no fixture represents physics."""

import copy
import json
from pathlib import Path
import tempfile
import unittest

import curate_first_results as curate

TEST_TEMP = Path(__file__).resolve().parent / "_cpu_test_temp"
TEST_TEMP.mkdir(exist_ok=True)

def fixture(candidate_id="first_8700", *, second=False, stage="test", seed=198301, received=None, elapsed=None):
    goal, enabled = 40., True
    received = goal + .05 if received is None else received
    first, retry = curate.ANGLES[candidate_id], 87.2
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
    elapsed = (21.5 if second else 14.2) if elapsed is None else elapsed
    row = {"episode_id": f"synthetic_{stage}_{candidate_id}_{seed}", "stage": stage,
        "split": "development" if stage != "test" else "test", "role": "baseline" if candidate_id == "baseline" else "candidate",
        "candidate_id": candidate_id, "profile_key": candidate_id,
        "seed": seed, "target_ml": goal, "declared_policy": policy,
        "status": {"first40_condition_seed": seed, "condition_seed_provided_by_runner": True,
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
            "full_home_within_deadline": True, "maximum_all_label_spilled_particles": 0,
            "spill_passed": True, "no_late_dose_change": True, "tolerance_ml": tolerance, "feasible": dose}
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
    values = {"baseline": (40.527297, 14.4), "first_8700": (40.2, 14.2),
              "first_8705": (40., 14.3), "first_8695": (39.9, 14.2)}
    for candidate_id, (dose, time_s) in values.items():
        row, report = fixture(candidate_id, stage="calibration", seed=197304, received=dose, elapsed=time_s)
        cases.append(curate.compact_case(row, "calibration", report))
    stages = ["calibration"]
    if validation:
        stages.append("validation")
        for seed in sorted(curate.VALIDATION_SEEDS):
            for candidate_id in ("baseline", "first_8700"):
                row, report = fixture(candidate_id, stage="validation", seed=seed,
                    received=40.55 if candidate_id == "baseline" else 40.1)
                cases.append(curate.compact_case(row, "validation", report))
    if not stopped:
        stages.append("test")
        for index, seed in enumerate(sorted(curate.TEST_SEEDS)):
            for candidate_id in ("baseline", "first_8700"):
                row, report = fixture(candidate_id, seed=seed,
                    received=(40.55 if index == 0 else 40.35) if candidate_id == "baseline" else 40.1)
                cases.append(curate.compact_case(row, "test", report))
    return {"cases": cases, **curate.summarize_cases(cases, stages, stopped)}


class CurationGuards(unittest.TestCase):
    def test_feasible_rank_uses_time_then_declared_order_not_error_or_angle(self):
        summary = summary_fixture()
        #87.05 has zero error but is slower;86.95 ties87.0 yet occurs last.
        self.assertEqual(curate.choose_calibration(summary["cases"]), "first_8700")
        cases = copy.deepcopy(summary["cases"])
        chosen = next(case for case in cases if case["stage"] == "calibration" and case["candidate_id"] == "first_8700")
        chosen["valid_evidence"] = False
        self.assertEqual(curate.choose_calibration(cases), "first_8695")
        for case in cases:
            if case["stage"] == "calibration" and case["role"] == "candidate":
                case["valid_evidence"] = False
        self.assertIsNone(curate.choose_calibration(cases))

    def test_both_roles_keep_retry_enabled_without_fixed_first_retry_gap(self):
        for candidate_id in curate.ANGLES:
            row, report = fixture(candidate_id)
            case = curate.compact_case(row, "test", report)
            self.assertTrue(case["retry_enabled"])
            self.assertEqual(case["retry_approach_tilt_deg"], 87.2)
            self.assertEqual(case["first_approach_tilt_deg"], curate.ANGLES[candidate_id])
        row, report = fixture("baseline")
        row["declared_policy"]["enable_retry"] = False
        report["declared_policy"]["enable_retry"] = False
        with self.assertRaisesRegex(ValueError, "Both roles keep retry enabled"):
            curate.compact_case(row, "test", report)
        row, report = fixture()
        row["declared_policy"]["retry_approach_tilt_deg"] = 87.25
        report["declared_policy"]["retry_approach_tilt_deg"] = 87.25
        with self.assertRaisesRegex(ValueError, "exactly87.2degrees"):
            curate.compact_case(row, "test", report)

    def test_candidate_identity_and_condition_seed_are_audited(self):
        row, report = fixture()
        row["candidate_id"] = row["profile_key"] = "first_8705"
        report["candidate_id"] = report["profile_key"] = "first_8705"
        with self.assertRaisesRegex(ValueError, "label does not match"):
            curate.compact_case(row, "test", report)
        row, report = fixture()
        row["status"]["first40_condition_seed"] += 1
        with self.assertRaisesRegex(ValueError, "condition seed metadata"):
            curate.compact_case(row, "test", report)

    def test_original_strict_and_exact_primary_bands_remain_separate(self):
        row, report = fixture(received=40.04)
        self.assertTrue(curate.compact_case(row, "test", report)["original_strict_feasible"])
        row, report = fixture(received=40.50000000000001)
        self.assertFalse(curate.compact_case(row, "test", report)["refine_feasible"])
        row["refine_assessment"].update(dose_passed=True, feasible=True)
        report["refine_assessment"].update(dose_passed=True, feasible=True)
        with self.assertRaisesRegex(ValueError, "Closed actual dose"):
            curate.compact_case(row, "test", report)

    def test_all_phases_and_paired_treatment_differences_are_retained(self):
        summary = summary_fixture()
        self.assertEqual(summary["actual_cases"], 28)
        self.assertEqual(summary["budget"], {"calibration": 4, "validation": 4, "test": 20})
        self.assertEqual(len(summary["test_pairs"]), 10)
        self.assertFalse(summary["test_pairs"][0]["dose_exactly_equal"])
        self.assertLess(summary["test_pairs"][0]["candidate_minus_baseline_absolute_error_ml"], 0.)
        self.assertEqual(summary["stage_metrics"]["test"]["candidate"]["cases"], 10)
        with self.assertRaisesRegex(ValueError, "Every actual bounded attempt"):
            curate.summarize_cases(summary["cases"][:-1], summary["actual_stages"], False)

    def test_gate_stop_does_not_invent_validation_or_test_evidence(self):
        summary = summary_fixture(stopped=True, validation=False)
        self.assertEqual(summary["actual_cases"], 4)
        self.assertEqual(summary["budget"], {"calibration": 4, "validation": 0, "test": 0})
        self.assertEqual(summary["test_pairs"], [])
        self.assertNotIn("validation", summary["stage_metrics"])
        summary = summary_fixture(stopped=True, validation=True)
        self.assertEqual(summary["actual_cases"], 8)
        self.assertEqual(len(summary["paired_comparisons"]["validation"]), 2)

    def test_partial_failure_retains_null_anchor_post_and_raw_measurement_basis(self):
        row, report = fixture(received=0.)
        row["response_events"] = []
        row["status"].update(prefix_anchor_observation=None, response_cycle_count=0)
        row["post_observation"] = None
        row["result"].update(physical_time_s=1. / 60., physical_steps=1, complete=False, valid_physics=False,
            measurement_physical_time_s=0., measurement_basis="last_complete_measurement_before_partial_fluid_failure")
        report.update(physical_s=1. / 60., post_frames=0, unlogged_terminal_action_frames=1,
            failure_records=[{"type": "failure", "step": 1, "failure_stage": "fluid_step",
                "error": "Synthetic failure at <synthetic-private-path>", "partial_unlogged_fluid_frame_possible": True}])
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
        self.assertEqual(len(case["failure_records"]), 1)
        self.assertNotIn("<synthetic-private-path>", json.dumps(case))

    def test_complete_independent_audit_precedes_journal_and_model_reads(self):
        with tempfile.TemporaryDirectory(dir=TEST_TEMP) as temporary:
            run = Path(temporary); audit = run / "overall_audit.json"
            audit.write_text(json.dumps({"complete": False, "pass": True}), encoding="utf8")
            with self.assertRaisesRegex(ValueError, "Complete passing independent audit"):
                curate.curate(run, audit, curate.sha(audit))
            with self.assertRaisesRegex(ValueError, "audit SHA"):
                curate.curate(run, audit, "0" * 64)

    def test_coarse_prefix_proof_allows_intended_first_cycle_dose_difference(self):
        summary = summary_fixture()
        cases = summary["cases"]
        records = []
        for stage in summary["actual_stages"]:
            seeds = [197304] if stage == "calibration" else sorted(curate.VALIDATION_SEEDS if stage == "validation" else curate.TEST_SEEDS)
            keys = curate.CANDIDATE_ORDER if stage == "calibration" else ("first_8700",)
            for seed in seeds:
                for candidate_id in keys:
                    base = next(case for case in cases if case["stage"] == stage and case["seed"] == seed and case["candidate_id"] == "baseline")
                    candidate = next(case for case in cases if case["stage"] == stage and case["seed"] == seed and case["candidate_id"] == candidate_id)
                    records.append({"stage": stage, "seed": seed, "candidate_id": candidate_id,
                        "baseline_episode_id": base["episode_id"], "candidate_episode_id": candidate["episode_id"],
                        "coarse_anchor_physical_s": 3., "coarse_prefix_frames": 181,
                        "actual_coarse_water_motion_partitions_exactly_equal": True, "verified": True,
                        "whole_first_cycle_equality_is_not_required": True,
                        "zero_time_inherited_last_substeps": {"baseline": 4, "candidate": 6,
                            "is_actual_completed_frame": False, "excluded_from_equivalence_only": True}})
        proof = curate.compact_prefix_proofs(records, cases, summary["actual_stages"], "first_8700")
        self.assertEqual(len(proof), 15)
        self.assertNotEqual(cases[0]["received_ml"], cases[1]["received_ml"])
        candidate = next(case for case in cases if case["stage"] == "calibration" and case["candidate_id"] == "first_8700")
        candidate["first_quiet_anchor_received_ml"] += .019683
        with self.assertRaisesRegex(ValueError, "coarse-prefix anchor water differs"):
            curate.compact_prefix_proofs(records, cases, summary["actual_stages"], "first_8700")

    def test_known_reference_is_compact_without_private_reference_paths(self):
        raw = {"verified": False, "endpoint_unavailable": True,
            "source_reference_sha256": {"<synthetic-private-path>/old.steps.jsonl.gz": "a" * 64}}
        result = curate.compact_known_reference(raw)
        self.assertFalse(result["verified"])
        self.assertIsNone(result["first_cycle_end_physical_s"])
        self.assertEqual(result["reference_artifact_sha256"], ["a" * 64])
        self.assertNotIn("<synthetic-private-path>", json.dumps(result))

    def test_synthetic_figures_cover_full_and_gate_stop_without_fake_tests(self):
        import plot_first_results as plot
        for stopped in (False, True):
            summary = summary_fixture(stopped=stopped)
            figure = plot.render(summary, synthetic=True)
            self.assertEqual(len(figure.axes), 4 if stopped else 9)
            self.assertTrue(any("SYNTHETIC CPU FIXTURE" in item.get_text() for item in figure.texts))
            figure.savefig(TEST_TEMP / ("stopped_fixture.png" if stopped else "full_fixture.png"), dpi=80)
            plot.plotting().close(figure)


if __name__ == "__main__":
    unittest.main()


