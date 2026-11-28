"""Pure CPU gates; never start liquid or read an active test journal."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import freeze_strategy as freeze

TEMP_ROOT = Path(__file__).resolve().parent / "_cpu_freeze_temp"
TEMP_ROOT.mkdir(exist_ok=True)


def profiles():
    return {"30": {"first_approach_tilt_deg": 85.5, "retry_approach_tilt_deg": 85.6},
            "40": {"first_approach_tilt_deg": 87.2, "retry_approach_tilt_deg": 87.3}}


class Contract(unittest.TestCase):
    def test_allowed_adjustment_keeps_passing_goal(self):
        current = profiles()
        current["40"] = {"first_approach_tilt_deg": 87.1, "retry_approach_tilt_deg": 87.2}
        results = {"30": {"primary_passed": True, "error_ml": -.2},
                   "40": {"primary_passed": False, "error_ml": .7}}
        freeze.validate_adjustment(profiles(), current, results)
        current["30"]["first_approach_tilt_deg"] = 85.6
        with self.assertRaisesRegex(ValueError, "retain passing goals"):
            freeze.validate_adjustment(profiles(), current, results)

    def test_undershoot_requires_positive_point_one_and_no_extra_stage_after_pass(self):
        results = {"30": {"primary_passed": True, "error_ml": 0.},
                   "40": {"primary_passed": False, "error_ml": -.7}}
        current = profiles(); current["40"] = {"first_approach_tilt_deg": 87.3, "retry_approach_tilt_deg": 87.4}
        freeze.validate_adjustment(profiles(), current, results)
        with self.assertRaises(ValueError): freeze.validate_adjustment(profiles(), profiles(), results)
        results["40"] = {"primary_passed": True, "error_ml": .3}
        with self.assertRaisesRegex(ValueError, "forbidden"):
            freeze.validate_adjustment(profiles(), profiles(), results)

    def test_adjustment_receipt_is_bound_to_actual_previous_audit(self):
        first = {"audit_sha256": "audit1", "goal_profile_raw_sha256": "profile1",
            "candidate_by_goal": {"30": {"target_ml": 30., "primary_passed": True, "error_ml": .2},
                                  "40": {"target_ml": 40., "primary_passed": False, "error_ml": .9}}}
        selected = {"goal_profile_raw_sha256": "profile2"}
        receipt = {"kind": "retry3040_preregistered_development_adjustment_v1", "pilot001_audit_sha256": "audit1",
            "old_profile_sha256": "profile1", "new_profile_sha256": "profile2", "no_code_or_speed_or_criterion_change": True,
            "test_data_read_or_used": False, "changes": [{"target_ml": 40., "actual_prior_error_ml": .9, "delta_degrees": -.1}]}
        freeze.validate_adjustment_receipt(receipt, first, selected)
        receipt["changes"][0]["delta_degrees"] = .1
        with self.assertRaisesRegex(ValueError, "signed rule"):
            freeze.validate_adjustment_receipt(receipt, first, selected)

    def test_existing_test_directory_blocks_before_any_read(self):
        with tempfile.TemporaryDirectory(dir=TEMP_ROOT) as temporary:
            workspace = Path(temporary)
            experiment = workspace / "droplet_out/pour_policy_retry3040/cpu_fixture"
            (experiment / "test").mkdir(parents=True)
            with patch.object(freeze, "read", side_effect=AssertionError("read before test gate")):
                with self.assertRaisesRegex(ValueError, "before any test directory"):
                    freeze.freeze(experiment, workspace)

    def test_existing_checkpoint_is_never_overwritten(self):
        with tempfile.TemporaryDirectory(dir=TEMP_ROOT) as temporary:
            workspace = Path(temporary)
            experiment = workspace / "droplet_out/pour_policy_retry3040/cpu_fixture"
            experiment.mkdir(parents=True)
            checkpoint = experiment / "fixed_strategy.json"; checkpoint.write_text("immutable", encoding="utf8")
            with self.assertRaisesRegex(ValueError, "Never overwrite"):
                freeze.freeze(experiment, workspace)
            self.assertEqual(checkpoint.read_text(), "immutable")

    def test_complete_role_target_seed_order_and_exact_policy_keys(self):
        plan = {"test": {"seeds": list(range(197301, 197311))}}
        selected = {"matrix": {"goal_profile_file": "profiles.json"}, "profile": {"goal_profiles": profiles()}}
        matrix = freeze.make_test_matrix(plan, selected, "a" * 64)
        self.assertEqual(len(matrix["cases"]), 40)
        self.assertEqual(matrix["cases"][0]["case_id"], "test_baseline_t30_s197301")
        self.assertEqual(matrix["cases"][10]["case_id"], "test_baseline_t40_s197301")
        self.assertEqual(matrix["cases"][20]["case_id"], "test_candidate_t30_s197301")
        self.assertEqual(len({row["case_id"] for row in matrix["cases"]}), 40)
        self.assertTrue(all(set(row["policy"]) == {"enable_retry", "first_approach_tilt_deg", "retry_approach_tilt_deg"}
                            for row in matrix["cases"]))

    def test_candidate_gate_uses_actual_target_relative_amount(self):
        rows = [{"role": "candidate", "target_ml": target, "result": {"received_ml": target + error}}
                for target, error in ((30., .5), (40., -.500001))]
        result = freeze.candidate_results(rows)
        self.assertTrue(result["30"]["primary_passed"])
        self.assertFalse(result["40"]["primary_passed"])
        self.assertEqual(result["40"]["target_ml"], 40.)


if __name__ == "__main__": unittest.main()
