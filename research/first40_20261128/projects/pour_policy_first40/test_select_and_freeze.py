"""Pure CPU counterexamples; synthetic fixtures are not physical evidence."""

import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import select_and_freeze as freeze

TEMP_ROOT = Path(__file__).resolve().parent / "_cpu_freeze_test_temp"
TEMP_ROOT.mkdir(exist_ok=True)
PLAN = Path(__file__).resolve().parent / "experiment_plan_001.json"


def row(key, amount=40.1, seconds=14., *, valid=True, stage="calibration", seed=197304):
    role = "baseline" if key == "baseline" else "candidate"
    feasible = bool(valid and 39.5 <= amount <= 40.5)
    now = datetime.now(timezone.utc).isoformat()
    return {"episode_id": stage + "_" + key + "_" + str(seed), "profile_key": key, "role": role,
        "target_ml": 40., "seed": seed, "started_at_utc": now, "completed_at_utc": now,
        "refine_assessment": {"maximum_all_label_spilled_particles": 0, "valid_evidence": valid,
            "tolerance_ml": .5, "dose_passed": 39.5 <= amount <= 40.5, "spill_passed": True,
            "full_home_within_deadline": True, "no_late_dose_change": True, "feasible": feasible},
        "result": {"target_ml": 40., "received_ml": amount, "physical_time_s": seconds,
            "mass_conserved": True, "all_states_finite": True, "complete": True, "valid_physics": True},
        "status": {"home_static": True, "stable_frames": 12, "timed_out": False},
        "post_observation": {"physical_steps": 120, "duration_s": 2., "maximum_received_change_ml": 0.,
            "mass_array_unchanged": True, "all_states_finite": True}}


def calibration():
    return [row("baseline", 40.527297), row("first_8700", 40.1, 14.),
            row("first_8705", 40.2, 14.), row("first_8695", 39.8, 14.1)]


def fake_stage(name, rows):
    profiles = {key: {"enable_retry": True, "first_approach_tilt_deg": angle,
                     "retry_approach_tilt_deg": 87.2} for key, angle in freeze.ANGLES.items()}
    maps = {key: {key + "_synthetic": "a" * 64} for key in freeze.MAPS}
    selected, cases, _ = freeze.choose_candidate(calibration())
    scores = {case["profile_key"]: [0, case["physical_time_s"]] for case in cases if case["profile_key"] in freeze.ORDER}
    audit = {"all_non_dose_safety_passed": True, "calibration_gate_passed": True,
        "validation_gate_passed": True, "selected_profile_key": selected, "candidate_scores": scores,
        "known_baseline_equivalence": {"verified": True, "actual_water_motion_partitions_exactly_equal": True,
            "first_cycle_original_events_exactly_equal": True,
            "source_reference_sha256": freeze.read(PLAN)["known_reference_sha256"]}}
    return {"rows": rows, "audit": audit, "record": {"stage": name, "audit_sha256": "c" * 64,
            "created_at_utc": "2000-01-01T00:00:00+00:00"},
        "protocol": {**maps, "profile_raw_sha256": "b" * 64},
        "matrix": {"profile_file": freeze.HERE + "profiles_calibration.json"},
        "profile": {"profile_id": "synthetic_profiles", "profiles": profiles}, "versions": {"CPU_fixture_only": True}}


class Contracts(unittest.TestCase):
    def test_feasible_compares_physical_time_not_smaller_error(self):
        rows = calibration(); rows[1] = row("first_8700", 40.49, 13.9)
        rows[2] = row("first_8705", 40.000001, 14.)
        chosen, cases, ranks = freeze.choose_candidate(rows)
        self.assertEqual(chosen, "first_8700")
        self.assertEqual(len(cases), 4)
        self.assertEqual(ranks[0]["rank"], [0, 13.9, 0])

    def test_only_exact_time_ties_use_declared_order(self):
        rows = calibration()
        self.assertEqual(freeze.choose_candidate(rows)[0], "first_8700")
        rows[1]["result"]["physical_time_s"] = 14.000000000000002
        self.assertEqual(freeze.choose_candidate(rows)[0], "first_8705")

    def test_invalid_cannot_be_elite_even_if_fast_and_accurate(self):
        rows = calibration(); rows[1] = row("first_8700", 40., 1., valid=False)
        self.assertEqual(freeze.choose_candidate(rows)[0], "first_8705")

    def test_no_feasible_candidate_stops_and_closed_boundary_is_exact(self):
        rows = [row(key, 40.50000000000001) for key in freeze.ANGLES]
        with self.assertRaisesRegex(ValueError, "No primary-feasible"):
            freeze.choose_candidate(rows)
        self.assertTrue(freeze.actual_case(row("first_8700", 40.5))["primary_passed"])
        self.assertFalse(freeze.actual_case(row("first_8700", 40.50000000000001))["primary_passed"])

    def test_infeasible_ranks_error_before_spill_and_never_uses_time(self):
        rows = calibration(); rows[1] = row("first_8700", 38., 1.)
        rows[2] = row("first_8705", 39., 23.)
        rows[3] = row("first_8695", 40.2, 14.)
        _, _, ranks = freeze.choose_candidate(rows)
        self.assertEqual([entry["profile_key"] for entry in ranks], ["first_8695", "first_8705", "first_8700"])
        self.assertEqual(ranks[1]["rank"], [1, 1., 0, 1])

    def test_matrix_preserves_static_profiles_retry_and_fresh_grids(self):
        selected = {"profile_file": "profiles.json", "profiles": fake_stage("calibration", calibration())["profile"]["profiles"],
                    "selected_profile_key": "first_8700"}
        validation = freeze.make_matrix("validation", selected, "a" * 64)
        test = freeze.make_matrix("test", selected, "a" * 64, "b" * 64)
        self.assertEqual(len(validation["cases"]), 4)
        self.assertEqual(len(test["cases"]), 20)
        self.assertEqual({r["seed"] for r in test["cases"]}, set(range(198301, 198311)))
        self.assertTrue(all(r["policy"]["enable_retry"] is True and r["policy"]["retry_approach_tilt_deg"] == 87.2 for r in test["cases"]))
        self.assertEqual(test["profiles"], selected["profiles"])

    def fixture(self, temporary):
        workspace = Path(temporary).resolve()
        self.assertTrue(workspace.is_relative_to(TEMP_ROOT.resolve()))
        experiment = workspace / "droplet_out/pour_policy_first40/synthetic_001"
        experiment.mkdir(parents=True)
        (experiment / "experiment_plan.json").write_bytes(PLAN.read_bytes())
        references = patch.object(freeze, "verify_prior_references", return_value=None)
        references.start()
        self.addCleanup(references.stop)
        return workspace, experiment

    def test_selection_freezes_before_validation_and_cannot_overwrite(self):
        with tempfile.TemporaryDirectory(dir=TEMP_ROOT) as temporary:
            workspace, experiment = self.fixture(temporary)
            stage = fake_stage("calibration", calibration())
            with patch.object(freeze, "inspect_stage", return_value=stage):
                outcome = freeze.freeze(experiment, "selection", workspace)
                self.assertEqual(outcome["selected_profile_key"], "first_8700")
                selection = freeze.read(experiment / "calibration_selection.json")
                self.assertEqual(len(selection["all_calibration_cases"]), 4)
                self.assertEqual(len(freeze.read(experiment / "validation_matrix.json")["cases"]), 4)
                with self.assertRaisesRegex(ValueError, "Never overwrite"):
                    freeze.freeze(experiment, "selection", workspace)

    def test_early_directories_reject_before_reading_plan_or_journal(self):
        for stage in ("test", "validation"):
            with tempfile.TemporaryDirectory(dir=TEMP_ROOT) as temporary:
                workspace, experiment = self.fixture(temporary)
                (experiment / stage).mkdir()
                with patch.object(freeze, "read", side_effect=AssertionError("No reads allowed")):
                    with self.assertRaises(ValueError): freeze.freeze(experiment, "selection", workspace)

    def test_validation_failure_creates_no_final_strategy_or_tests(self):
        with tempfile.TemporaryDirectory(dir=TEMP_ROOT) as temporary:
            workspace, experiment = self.fixture(temporary)
            calibrated = fake_stage("calibration", calibration())
            with patch.object(freeze, "inspect_stage", return_value=calibrated):
                freeze.freeze(experiment, "selection", workspace)
            selected = freeze.read(experiment / "calibration_selection.json")
            rows = [row("baseline" if role == "baseline" else "first_8700", 40.8 if role == "candidate" else 40.1,
                        stage="validation", seed=seed) for role in ("baseline", "candidate") for seed in (198101, 198102)]
            validated = fake_stage("validation", rows)
            validated["matrix"] = freeze.read(experiment / "validation_matrix.json")
            with patch.object(freeze, "inspect_stage", side_effect=[calibrated, validated]):
                with self.assertRaisesRegex(ValueError, "Fresh validation gate failed"):
                    freeze.freeze(experiment, "strategy", workspace)
            self.assertFalse((experiment / "fixed_strategy.json").exists())
            self.assertFalse((experiment / "test_matrix.json").exists())
            self.assertFalse((experiment / "test").exists())

    def test_validated_strategy_binds_selection_and_does_not_retune(self):
        with tempfile.TemporaryDirectory(dir=TEMP_ROOT) as temporary:
            workspace, experiment = self.fixture(temporary)
            calibrated = fake_stage("calibration", calibration())
            with patch.object(freeze, "inspect_stage", return_value=calibrated):
                freeze.freeze(experiment, "selection", workspace)
            rows = [row("baseline" if role == "baseline" else "first_8700", 40.1,
                        stage="validation", seed=seed) for role in ("baseline", "candidate") for seed in (198101, 198102)]
            validated = fake_stage("validation", rows)
            validated["matrix"] = freeze.read(experiment / "validation_matrix.json")
            with patch.object(freeze, "inspect_stage", side_effect=[calibrated, validated]):
                result = freeze.freeze(experiment, "strategy", workspace)
            strategy = freeze.read(experiment / "fixed_strategy.json")
            self.assertEqual(result["fresh_test_cases"], 20)
            self.assertEqual(strategy["selected_profile_key"], "first_8700")
            self.assertFalse(strategy["test_used_for_selection"])
            self.assertEqual(strategy["calibration_selection_sha256"], freeze.sha(experiment / "calibration_selection.json"))


if __name__ == "__main__": unittest.main()
