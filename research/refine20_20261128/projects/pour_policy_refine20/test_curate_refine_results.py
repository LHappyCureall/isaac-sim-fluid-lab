"""CPU counterexamples; every fixture is explicitly synthetic, never physics evidence."""

import copy
import json
from pathlib import Path
import tempfile
import unittest

import curate_refine_results as curate


TEST_TEMP = Path(__file__).resolve().parent / "_cpu_test_temp"
TEST_TEMP.mkdir(exist_ok=True)


def fixture(received=19.75, *, stage="test", enabled=False, seed=195301, angle=84.):
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
    row = {"episode_id": "synthetic_not_physical", "stage": stage,
        "split": "development" if stage == "engineering" else stage,
        "seed": seed, "role": "candidate" if enabled else "baseline", "target_ml": 20.,
        "declared_policy": {"enable_correction": enabled, "approach_tilt_deg": angle},
        "status": {"refine_configured_approach_tilt_deg": angle,
            "refine_enable_correction": enabled, "refine_maximum_cycles": 1,
            "response_cycle_count": 0, "refine_actual_settled_home_decision": enabled,
            "prefix_anchor_observation": anchor},
        "result": {"target_ml": 20., "received_ml": received, "error_ml": received - 20.,
            "absolute_error_ml": abs(received - 20.), "physical_time_s": 5., "physical_steps": 300,
            "wall_seconds": 20., "mass_conserved": True, "all_states_finite": True},
        "response_events": [], "post_observation": {"duration_s": 2., "physical_steps": 120},
        "wall_seconds": 25., "started_at_utc": "2026-11-28T01:00:00+00:00",
        "completed_at_utc": "2026-11-28T01:00:25+00:00"}
    for name, value in assessments.items():
        row[name + "_assessment"] = value
    report = {key: row[key] for key in ("episode_id", "stage", "role", "seed")}
    report.update(received_ml=received, error_ml=received - 20., physical_s=5.,
        actual_refine_coarse_anchor=copy.deepcopy(anchor))
    for name, value in assessments.items():
        report[name + "_assessment"] = copy.deepcopy(value)
    return row, report


def synthetic_summary():
    cases = []
    for stage in ("engineering", "train"):
        for angle, dose in ((84., 19.8), (84.1, 20.1), (84.2, 20.6)):
            for seed in sorted(curate.SEEDS[stage]):
                row, report = fixture(dose, stage=stage, enabled=True, seed=seed, angle=angle)
                row["role"] = report["role"] = "angle_" + format(angle, "g")
                row["episode_id"] = report["episode_id"] = f"synthetic_{stage}_{angle}_{seed}"
                cases.append(curate.compact_case(row, stage, report))
    for stage in ("validation", "test"):
        for index, seed in enumerate(sorted(curate.SEEDS[stage])):
            for enabled in (False, True):
                dose = 20. + .01 * index if enabled else 19.1 + .02 * index
                row, report = fixture(dose, stage=stage, enabled=enabled, seed=seed,
                                      angle=84.1 if enabled else 84.)
                row["episode_id"] = report["episode_id"] = f"synthetic_{stage}_{enabled}_{seed}"
                cases.append(curate.compact_case(row, stage, report))
    return {"cases": cases,
        "selection": {"candidate_approach_tilt_deg": 84.1, "recommended_role": "candidate"},
        "stage_metrics": {stage: {role: curate.metrics([case for case in cases
            if case["stage"] == stage and case["role"] == role])
            for role in {case["role"] for case in cases if case["stage"] == stage}}
            for stage in curate.COUNTS}}


class CurationGuards(unittest.TestCase):
    def test_primary_easy_and_original_strict_are_separate(self):
        row, report = fixture()
        case = curate.compact_case(row, "test", report)
        self.assertTrue(case["refine_feasible"])
        self.assertTrue(case["easy_feasible"])
        self.assertFalse(case["original_strict_feasible"])
        self.assertEqual(case["physical_action_seconds"], 5.)
        self.assertEqual(case["compute_wall_seconds_including_post"], 25.)

    def test_closed_half_ml_band_has_no_grain_or_float_widening(self):
        for volume in (19.5, 20.5):
            row, report = fixture(volume)
            self.assertTrue(curate.compact_case(row, "test", report)["refine_feasible"])
        row, report = fixture(20.500000000000004)
        row["refine_assessment"].update(dose_passed=True, feasible=True)
        report["refine_assessment"].update(dose_passed=True, feasible=True)
        with self.assertRaisesRegex(ValueError, "Closed actual dose gate"):
            curate.compact_case(row, "test", report)

    def test_invalid_or_partial_case_is_retained_not_selected_as_success(self):
        row, report = fixture(19.25, enabled=True)
        for name in ("refine", "easy", "strict"):
            row[name + "_assessment"].update(valid_evidence=False, feasible=False)
            report[name + "_assessment"].update(valid_evidence=False, feasible=False)
        row["response_events"] = [{"type": "response_cycle_start", "physical_time_s": 3.,
            "observation": {"received_ml": 19.25}}]
        row["status"]["response_cycle_count"] = 1
        case = curate.compact_case(row, "test", report)
        self.assertFalse(case["valid_evidence"])
        self.assertFalse(case["refine_feasible"])
        self.assertEqual(case["actual_correction_cycles_started"], 1)
        self.assertEqual(case["actual_correction_cycles_ended"], 0)
        self.assertFalse(case["actual_cycles"][0]["has_recorded_endpoint"])
        self.assertNotIn("outcome", case["actual_cycles"][0])

    def test_audited_dose_and_anchor_tampering_are_rejected(self):
        row, report = fixture()
        report["received_ml"] += .019683
        with self.assertRaisesRegex(ValueError, "Audited measurement"):
            curate.compact_case(row, "test", report)
        row, report = fixture()
        report["actual_refine_coarse_anchor"]["received_ml"] += .019683
        with self.assertRaisesRegex(ValueError, "Independent first quiet anchor"):
            curate.compact_case(row, "test", report)

    def test_post_clock_and_control_target_cannot_be_hidden(self):
        row, report = fixture()
        row["post_observation"]["physical_steps"] = 119
        with self.assertRaisesRegex(ValueError, "120 live steps"):
            curate.compact_case(row, "test", report)
        row, report = fixture()
        row["result"]["physical_steps"] = 301
        with self.assertRaisesRegex(ValueError, "step/60"):
            curate.compact_case(row, "test", report)
        row, report = fixture()
        row["status"]["prefix_anchor_observation"]["control_target_ml"] += .5
        with self.assertRaisesRegex(ValueError, "Coarse target was offset"):
            curate.compact_case(row, "test", report)

    def test_baseline_cannot_hide_an_extra_cycle(self):
        row, report = fixture(enabled=False)
        row["response_events"] = [{"type": "response_cycle_start", "physical_time_s": 3.,
            "observation": {"received_ml": 19.25}}]
        row["status"]["response_cycle_count"] = 1
        with self.assertRaisesRegex(ValueError, "baseline unexpectedly"):
            curate.compact_case(row, "test", report)

    def test_engineering_stays_development_not_training(self):
        row, report = fixture(stage="engineering", enabled=True, seed=195001)
        self.assertEqual(curate.compact_case(row, "engineering", report)["split"], "development")
        row["split"] = "train"
        with self.assertRaisesRegex(ValueError, "Stage/split"):
            curate.compact_case(row, "engineering", report)

    def test_complete_independent_audit_and_raw_sha_precede_journal_reads(self):
        with tempfile.TemporaryDirectory(dir=TEST_TEMP) as temporary:
            run = Path(temporary)
            audit = run / "overall_audit.json"
            audit.write_text(json.dumps({"complete": False, "pass": True}), encoding="utf8")
            with self.assertRaisesRegex(ValueError, "Complete passing independent"):
                curate.curate(run, audit, curate.sha(audit))
            with self.assertRaisesRegex(ValueError, "audit SHA"):
                curate.curate(run, audit, "0" * 64)

    def test_synthetic_plot_preserves_four_stages_and_separate_acceptance_counts(self):
        import plot_refine_results as plot
        summary = synthetic_summary()
        self.assertEqual(len(summary["cases"]), 23)
        self.assertEqual(summary["stage_metrics"]["test"]["baseline"]["joint_passes"], 0)
        self.assertEqual(summary["stage_metrics"]["test"]["baseline"]["easy_joint_passes"], 5)
        figure = plot.render(summary, synthetic=True)
        self.assertEqual(len(figure.axes), 9)  # eight panels plus compute-wall secondary axis
        self.assertTrue(any("SYNTHETIC CPU FIXTURE" in item.get_text() for item in figure.texts))
        self.assertEqual([bar.get_height() for bar in figure.axes[6].patches], [0, 5, 0, 5, 5, 3])
        figure.savefig(TEST_TEMP / "refine_plot_fixture.png", dpi=80)
        plot.plotting().close(figure)


if __name__ == "__main__":
    unittest.main()
