"""Synthetic CPU group-score counterexamples; no liquid evidence is generated."""

import copy
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
        "declared_policy": {"enable_retry": True, "first_approach_tilt_deg": 87.,
            "retry_approach_tilt_deg": 87.2, "coarse_dose_tilt_deg_s": select.SPEEDS[key]},
        "result": {"target_ml": 40., "received_ml": amount, "physical_time_s": elapsed,
            "valid_physics": valid, "mass_conserved": True, "all_states_finite": True},
        "refine_assessment": assessment, "strict_assessment": {"valid_evidence": valid}}


def fixture():
    return [row(key, seed, peak=2 if key == "baseline" else 0)
            for key in select.SPEEDS for seed in select.KNOWN_SEEDS]


def fake_stage(name, rows):
    profiles = {key: fixture()[index * 2]["declared_policy"] for index, key in enumerate(select.SPEEDS)}
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
    def test_spill_two_baseline_is_valid_negative_not_candidate_relaxation(self):
        rows = fixture()
        for baseline in rows[:2]:
            self.assertTrue(select.valid_baseline_comparator(baseline))
            self.assertFalse(select.actual_case(baseline)["primary_passed"])
        self.assertEqual(select.choose_candidate(rows)[0], "coarse_252")

    def test_baseline_home_late_dose_failures_are_valid_negative_evidence(self):
        baseline = row("baseline", 197304, amount=42., peak=2, home=False, late=True)
        self.assertTrue(select.valid_baseline_comparator(baseline))
        self.assertFalse(select.actual_case(baseline)["primary_passed"])
        baseline["strict_assessment"]["valid_evidence"] = False
        self.assertFalse(select.valid_baseline_comparator(baseline))

    def test_candidate_requires_both_cases_not_mean_error_or_success_count(self):
        rows = fixture(); rows[3] = row("coarse_252", 198304, amount=40.6)
        chosen, _, scores, _ = select.choose_candidate(rows)
        self.assertEqual(chosen, "coarse_248")
        self.assertEqual(scores["coarse_252"][0], 1)

    def test_feasible_group_ranks_only_mean_physical_time(self):
        rows = fixture()
        rows[2] = row("coarse_252", 197304, amount=40.49, elapsed=13.)
        rows[3] = row("coarse_252", 198304, amount=40.49, elapsed=14.)
        rows[4] = row("coarse_248", 197304, amount=40.000001, elapsed=14.)
        rows[5] = row("coarse_248", 198304, amount=40.000001, elapsed=14.)
        self.assertEqual(select.choose_candidate(rows)[0], "coarse_252")

    def test_exact_mean_tie_uses_preregistered_order(self):
        rows = fixture()
        self.assertEqual(select.choose_candidate(rows)[0], "coarse_252")
        rows[3]["result"]["physical_time_s"] = 14.00000000000001
        self.assertEqual(select.choose_candidate(rows)[0], "coarse_248")

    def test_one_invalid_case_rejects_whole_group(self):
        rows = fixture(); rows[2] = row("coarse_252", 197304, amount=40., elapsed=1., valid=False)
        chosen, _, scores, _ = select.choose_candidate(rows)
        self.assertEqual(chosen, "coarse_248")
        self.assertIsNone(scores["coarse_252"])

    def test_infeasible_group_error_precedes_spill_and_time(self):
        under = [row("coarse_252", seed, amount=39.4, peak=0, elapsed=1.) for seed in select.KNOWN_SEEDS]
        closer = [row("coarse_248", seed, amount=40.51, peak=2, elapsed=23.) for seed in select.KNOWN_SEEDS]
        self.assertLess(select.score_group(closer), select.score_group(under))
        self.assertEqual(select.score_group(closer)[2], 4)

    def test_closed_band_spill_and_deadline_remain_hard(self):
        for kwargs in ({"amount": 40.50000000000001}, {"peak": 2}, {"elapsed": 24.000001}, {"home": False}, {"late": True}):
            self.assertFalse(select.actual_case(row("coarse_252", 197304, **kwargs))["primary_passed"])
        rows = [row(key, seed, amount=39.4) for key in select.SPEEDS for seed in select.KNOWN_SEEDS]
        with self.assertRaisesRegex(ValueError, "No candidate passes both"):
            select.choose_candidate(rows)

    def test_no_checkpoint_or_validation_without_final_plan_and_auditor(self):
        with patch.object(select, "AUDITOR_SHA256", "pending"):
            with self.assertRaisesRegex(ValueError, "not frozen"):
                select.freeze("unused", "selection")

    def test_fresh_matrices_keep_first_retry_and_both_retry_flags(self):
        profiles = {key: fixture()[index * 2]["declared_policy"] for index, key in enumerate(select.SPEEDS)}
        selection = {"profile_file": "profiles.json", "profiles": profiles, "selected_profile_key": "coarse_252"}
        validation = select.make_matrix("validation", selection, "a" * 64)
        test = select.make_matrix("test", selection, "a" * 64, "b" * 64)
        self.assertEqual(len(validation["cases"]), 4)
        self.assertEqual({case["seed"] for case in validation["cases"]}, {198201, 198202})
        self.assertEqual(len(test["cases"]), 20)
        self.assertEqual({case["seed"] for case in test["cases"]}, set(range(198401, 198411)))
        self.assertTrue(all(case["policy"]["first_approach_tilt_deg"] == 87.
            and case["policy"]["retry_approach_tilt_deg"] == 87.2
            and case["policy"]["enable_retry"] is True for case in test["cases"]))

    def workspace_fixture(self, temporary):
        workspace = Path(temporary).resolve()
        self.assertTrue(workspace.is_relative_to(TEMP_ROOT.resolve()))
        experiment = workspace / "droplet_out/pour_policy_coarse40/synthetic_001"
        experiment.mkdir(parents=True)
        (experiment / "experiment_plan.json").write_bytes(PLAN.read_bytes())
        for patcher in (patch.object(select, "AUDITOR_SHA256", "a" * 64),
                        patch.object(select, "verify_prior_references", return_value=None)):
            patcher.start(); self.addCleanup(patcher.stop)
        return workspace, experiment

    def test_selection_can_keep_spill_two_baselines_without_relaxing_candidate(self):
        with tempfile.TemporaryDirectory(dir=TEMP_ROOT) as temporary:
            workspace, experiment = self.workspace_fixture(temporary)
            calibrated = fake_stage("calibration", fixture())
            with patch.object(select, "inspect_stage", return_value=calibrated):
                outcome = select.freeze(experiment, "selection", workspace)
            self.assertEqual(outcome["selected_profile_key"], "coarse_252")
            chosen = select.read(experiment / "calibration_selection.json")
            self.assertEqual(len(chosen["all_calibration_cases"]), 6)
            self.assertEqual(sum(not r["primary_passed"] for r in chosen["all_calibration_cases"]), 2)
            self.assertTrue(chosen["baseline_comparator_does_not_require_task_success"])

    def test_validation_candidate_spill_failure_prevents_strategy_and_test(self):
        with tempfile.TemporaryDirectory(dir=TEMP_ROOT) as temporary:
            workspace, experiment = self.workspace_fixture(temporary)
            calibrated = fake_stage("calibration", fixture())
            with patch.object(select, "inspect_stage", return_value=calibrated):
                select.freeze(experiment, "selection", workspace)
            rows = [row("baseline" if role == "baseline" else "coarse_252", seed,
                        peak=2, valid=True) for role in ("baseline", "candidate") for seed in select.VALIDATION_SEEDS]
            now = datetime.now(timezone.utc).isoformat()
            for r in rows:r.update(started_at_utc=now, completed_at_utc=now)
            validated = fake_stage("validation", rows);validated["matrix"] = select.read(experiment / "validation_matrix.json")
            with patch.object(select, "inspect_stage", side_effect=[calibrated, validated]):
                with self.assertRaisesRegex(ValueError, "Fresh validation gate failed"):
                    select.freeze(experiment, "strategy", workspace)
            self.assertFalse((experiment / "fixed_strategy.json").exists())
            self.assertFalse((experiment / "test_matrix.json").exists())

    def test_validated_candidate_can_advance_with_valid_negative_baselines(self):
        with tempfile.TemporaryDirectory(dir=TEMP_ROOT) as temporary:
            workspace, experiment = self.workspace_fixture(temporary)
            calibrated = fake_stage("calibration", fixture())
            with patch.object(select, "inspect_stage", return_value=calibrated):
                select.freeze(experiment, "selection", workspace)
            rows = [row("baseline" if role == "baseline" else "coarse_252", seed,
                        peak=2 if role == "baseline" else 1) for role in ("baseline", "candidate") for seed in select.VALIDATION_SEEDS]
            now = datetime.now(timezone.utc).isoformat()
            for r in rows:r.update(started_at_utc=now, completed_at_utc=now)
            validated = fake_stage("validation", rows);validated["matrix"] = select.read(experiment / "validation_matrix.json")
            with patch.object(select, "inspect_stage", side_effect=[calibrated, validated]):
                select.freeze(experiment, "strategy", workspace)
            strategy = select.read(experiment / "fixed_strategy.json")
            self.assertEqual(strategy["kind"], "coarse40_frozen_speed_strategy_v1")
            self.assertEqual(strategy["selected_coarse_dose_tilt_deg_s"], 25.2)
            self.assertFalse(strategy["test_used_for_selection"])
            self.assertEqual(len(select.read(experiment / "test_matrix.json")["cases"]), 20)


if __name__ == "__main__": unittest.main()
