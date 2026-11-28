"""Synthetic CPU three-condition counterexamples; no liquid evidence is made."""

from datetime import datetime, timezone
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import select_and_freeze as select

TEMP_ROOT = Path(__file__).resolve().parent / "_cpu_freeze_test_temp"
TEMP_ROOT.mkdir(exist_ok=True)
PLAN = Path(__file__).resolve().parent / "experiment_plan_001.json"


def row(key, seed, *, amount=40.1, elapsed=14., peak=0, valid=True, home=True, late=False):
    feasible = bool(valid and 39.5 <= amount <= 40.5 and peak <= 1 and home and not late and elapsed <= 24.)
    assessment = {"maximum_all_label_spilled_particles": peak, "valid_evidence": valid,
        "tolerance_ml": .5, "dose_passed": 39.5 <= amount <= 40.5,
        "spill_passed": peak <= 1, "full_home_within_deadline": home,
        "no_late_dose_change": not late, "feasible": feasible}
    return {"episode_id": key + "_" + str(seed), "profile_key": key,
        "role": "baseline" if key == "baseline" else "candidate", "seed": seed, "target_ml": 40.,
        "declared_policy": {"enable_retry": True, "first_approach_tilt_deg": select.ANGLES[key],
            "retry_approach_tilt_deg": 87.2, "coarse_dose_tilt_deg_s": 24.8},
        "result": {"target_ml": 40., "received_ml": amount, "physical_time_s": elapsed,
            "valid_physics": valid, "mass_conserved": True, "all_states_finite": True},
        "refine_assessment": assessment, "strict_assessment": {"valid_evidence": valid}}


def fixture():
    return [row(key, seed, peak=2 if key == "baseline" else 0)
            for key in select.ANGLES for seed in select.KNOWN_SEEDS]


def fake_stage(name, rows):
    profiles = {key: fixture()[index * 3]["declared_policy"] for index, key in enumerate(select.ANGLES)}
    chosen, _, scores, _ = select.choose_candidate(fixture())
    references = select.read(PLAN)["known_reference_sha256"]
    known = {str(seed): {"verified": True, "actual_water_motion_partitions_exactly_equal": True,
        "first_cycle_original_events_exactly_equal": True, "source_reference_sha256": references[str(seed)]}
        for seed in select.KNOWN_SEEDS}
    maps = {key: {key + "_synthetic": "a" * 64} for key in select.MAPS}
    return {"rows": rows, "audit": {"all_baseline_valid_comparators": True,
            "all_non_dose_safety_passed": False, "calibration_gate_passed": True, "validation_gate_passed": True,
            "selected_profile_key": chosen, "candidate_scores": scores, "known_baseline_equivalence": known},
        "record": {"stage": name, "audit_sha256": "c" * 64, "created_at_utc": "2000-01-01T00:00:00+00:00"},
        "protocol": {**maps, "profile_raw_sha256": "b" * 64}, "matrix": {"profile_file": select.HERE + "profiles_calibration.json"},
        "profile": {"profile_id": "synthetic_profiles", "profiles": profiles}, "versions": {"CPU_fixture_only": True}}


class Contract(unittest.TestCase):
    def test_negative_baselines_are_valid_comparators_without_task_success(self):
        rows = fixture()
        for baseline in rows[:3]:
            self.assertTrue(select.valid_baseline_comparator(baseline))
            self.assertFalse(select.actual_case(baseline)["primary_passed"])
        baseline = row("baseline", 198403, amount=39.464415, home=False, late=True)
        self.assertTrue(select.valid_baseline_comparator(baseline))
        self.assertFalse(select.actual_case(baseline)["primary_passed"])
        self.assertEqual(select.choose_candidate(rows)[0], "first_8705")

    def test_third_case_failure_cannot_hide_behind_two_successes(self):
        rows = fixture(); rows[5] = row("first_8705", 198403, amount=39.49)
        chosen, _, scores, _ = select.choose_candidate(rows)
        self.assertEqual(chosen, "first_8710")
        self.assertEqual(scores["first_8705"][0], 1)

    def test_feasible_group_compares_only_mean_physical_time(self):
        rows = fixture()
        rows[3:6] = [row("first_8705", seed, amount=40.49, elapsed=13.9) for seed in select.KNOWN_SEEDS]
        rows[6:9] = [row("first_8710", seed, amount=40.000001, elapsed=14.) for seed in select.KNOWN_SEEDS]
        self.assertEqual(select.choose_candidate(rows)[0], "first_8705")

    def test_exact_mean_time_tie_uses_declared_angle_order(self):
        rows = fixture()
        self.assertEqual(select.choose_candidate(rows)[0], "first_8705")
        rows[5]["result"]["physical_time_s"] = 14.000000000000014
        self.assertEqual(select.choose_candidate(rows)[0], "first_8710")

    def test_one_invalid_case_rejects_the_whole_three_case_group(self):
        rows = fixture(); rows[4] = row("first_8705", 198304, amount=40., elapsed=1., valid=False)
        chosen, _, scores, _ = select.choose_candidate(rows)
        self.assertEqual(chosen, "first_8710")
        self.assertIsNone(scores["first_8705"])

    def test_infeasible_score_is_mae_then_total_peak_not_fewer_failures_or_time(self):
        under = [row("first_8705", seed, amount=39.4, peak=0, elapsed=1.) for seed in select.KNOWN_SEEDS]
        closer = [row("first_8710", seed, amount=40.51, peak=2, elapsed=23.) for seed in select.KNOWN_SEEDS]
        self.assertLess(select.score_group(closer), select.score_group(under))
        self.assertEqual(select.score_group(closer)[2], 6)

    def test_closed_task_standards_are_not_relaxed(self):
        for kwargs in ({"amount": 40.50000000000001}, {"peak": 2}, {"elapsed": 24.000001}, {"home": False}, {"late": True}):
            self.assertFalse(select.actual_case(row("first_8705", 198403, **kwargs))["primary_passed"])
        rows = [row(key, seed, amount=39.4) for key in select.ANGLES for seed in select.KNOWN_SEEDS]
        with self.assertRaisesRegex(ValueError, "No candidate passes all three"):
            select.choose_candidate(rows)

    def test_missing_third_condition_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "nine three-seed"):
            select.choose_candidate(fixture()[:-1])
        with self.assertRaisesRegex(ValueError, "all three unique"):
            select.score_group(fixture()[3:5])

    def test_pending_auditor_cannot_write_any_model(self):
        with patch.object(select, "AUDITOR_SHA256", "pending"):
            with self.assertRaisesRegex(ValueError, "not frozen"):
                select.freeze("unused", "selection")

    def test_fresh_matrices_keep_fixed_coarse_and_retry(self):
        profiles = fake_stage("calibration", fixture())["profile"]["profiles"]
        chosen = {"profile_file": "profiles.json", "profiles": profiles, "selected_profile_key": "first_8705"}
        validation = select.make_matrix("validation", chosen, "a" * 64)
        test = select.make_matrix("test", chosen, "a" * 64, "b" * 64)
        self.assertEqual({case["seed"] for case in validation["cases"]}, {199101, 199102})
        self.assertEqual(len(test["cases"]), 20)
        self.assertEqual({case["seed"] for case in test["cases"]}, set(range(199301, 199311)))
        for case in test["cases"]:
            self.assertIs(case["policy"]["enable_retry"], True)
            self.assertEqual(case["policy"]["coarse_dose_tilt_deg_s"], 24.8)
            self.assertEqual(case["policy"]["retry_approach_tilt_deg"], 87.2)
            self.assertEqual(case["policy"]["first_approach_tilt_deg"], 87. if case["role"] == "baseline" else 87.05)

    def workspace_fixture(self, temporary):
        workspace = Path(temporary).resolve()
        self.assertTrue(workspace.is_relative_to(TEMP_ROOT.resolve()))
        experiment = workspace / "droplet_out/pour_policy_balanced40/synthetic_001"
        experiment.mkdir(parents=True);(experiment / "experiment_plan.json").write_bytes(PLAN.read_bytes())
        for patcher in (patch.object(select, "AUDITOR_SHA256", "a" * 64),
                        patch.object(select, "verify_prior_references", return_value=None)):
            patcher.start();self.addCleanup(patcher.stop)
        return workspace, experiment

    def test_selection_retains_all_nine_and_valid_negative_baselines(self):
        with tempfile.TemporaryDirectory(dir=TEMP_ROOT) as temporary:
            workspace, experiment = self.workspace_fixture(temporary)
            calibrated = fake_stage("calibration", fixture())
            with patch.object(select, "inspect_stage", return_value=calibrated):
                outcome = select.freeze(experiment, "selection", workspace)
            model = select.read(experiment / "calibration_selection.json")
            self.assertEqual(outcome["selected_profile_key"], "first_8705")
            self.assertEqual(len(model["all_calibration_cases"]), 9)
            self.assertEqual(model["fixed_coarse_dose_tilt_deg_s"], 24.8)
            self.assertEqual(model["fixed_parameters"]["dose_tilt_deg_s"], 24.8)
            self.assertTrue(model["only_first_angle_varies"])
            self.assertFalse(model["only_coarse_dose_speed_varies"])

    def test_candidate_validation_failure_blocks_tests_without_fallback(self):
        with tempfile.TemporaryDirectory(dir=TEMP_ROOT) as temporary:
            workspace, experiment = self.workspace_fixture(temporary)
            calibrated = fake_stage("calibration", fixture())
            with patch.object(select, "inspect_stage", return_value=calibrated):
                select.freeze(experiment, "selection", workspace)
            rows = [row("baseline" if role == "baseline" else "first_8705", seed, peak=2)
                    for role in ("baseline", "candidate") for seed in select.VALIDATION_SEEDS]
            now = datetime.now(timezone.utc).isoformat()
            for r in rows:r.update(started_at_utc=now, completed_at_utc=now)
            validated = fake_stage("validation", rows);validated["matrix"] = select.read(experiment / "validation_matrix.json")
            with patch.object(select, "inspect_stage", side_effect=[calibrated, validated]):
                with self.assertRaisesRegex(ValueError, "Fresh validation gate failed"):
                    select.freeze(experiment, "strategy", workspace)
            self.assertFalse((experiment / "fixed_strategy.json").exists())
            self.assertFalse((experiment / "test_matrix.json").exists())

    def test_validated_candidate_can_advance_with_negative_baselines(self):
        with tempfile.TemporaryDirectory(dir=TEMP_ROOT) as temporary:
            workspace, experiment = self.workspace_fixture(temporary)
            calibrated = fake_stage("calibration", fixture())
            with patch.object(select, "inspect_stage", return_value=calibrated):
                select.freeze(experiment, "selection", workspace)
            rows = [row("baseline" if role == "baseline" else "first_8705", seed,
                        peak=2 if role == "baseline" else 1) for role in ("baseline", "candidate") for seed in select.VALIDATION_SEEDS]
            now = datetime.now(timezone.utc).isoformat()
            for r in rows:r.update(started_at_utc=now, completed_at_utc=now)
            validated = fake_stage("validation", rows);validated["matrix"] = select.read(experiment / "validation_matrix.json")
            with patch.object(select, "inspect_stage", side_effect=[calibrated, validated]):
                select.freeze(experiment, "strategy", workspace)
            strategy = select.read(experiment / "fixed_strategy.json")
            self.assertEqual(strategy["kind"], "balanced40_frozen_strategy_v1")
            self.assertEqual(strategy["selected_first_approach_tilt_deg"], 87.05)
            self.assertFalse(strategy["test_used_for_selection"])
            self.assertEqual(len(select.read(experiment / "test_matrix.json")["cases"]), 20)


if __name__ == "__main__": unittest.main()
