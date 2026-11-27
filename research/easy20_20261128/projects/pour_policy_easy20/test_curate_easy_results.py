"""Meaningful CPU guards; fixtures never masquerade as physical results."""

import copy
import json
from pathlib import Path
import tempfile
import unittest

import curate_easy_results as curate

TEST_TEMP = Path(__file__).resolve().parent / "_cpu_test_temp"
TEST_TEMP.mkdir(exist_ok=True)


def fixture():
    strict = {"valid_evidence": True, "dose_passed": False,
        "full_home_within_deadline": True, "maximum_all_label_spilled_particles": 0,
        "spill_passed": True, "no_late_dose_change": True, "tolerance_ml": .02, "feasible": False}
    easy = {**strict, "dose_passed": True, "tolerance_ml": 1., "feasible": True}
    row = {"episode_id": "synthetic_not_physical", "stage": "test", "split": "test",
        "seed": 194301, "role": "baseline", "target_ml": 20.,
        "declared_policy": {"control_target_offset_ml": 0.},
        "status": {"prefix_anchor_observation": {"settled_measurement": True,
            "coarse_cut_tilt_deg": 90., "received_ml": 19.25, "physical_time_s": 3.,
            "tube_remaining_ml": 37.437, "actual_target_ml": 20., "control_target_ml": 20.,
            "control_target_offset_ml": 0.}},
        "easy_assessment": easy, "strict_assessment": strict,
        "result": {"target_ml": 20., "received_ml": 19.25, "error_ml": -.75,
            "absolute_error_ml": .75, "physical_time_s": 5., "physical_steps": 300,
            "wall_seconds": 20., "policy_params": {"control_target_offset_ml": 0.},
            "mass_conserved": True, "all_states_finite": True},
        "post_observation": {"duration_s": 2., "physical_steps": 120},
        "wall_seconds": 25., "started_at_utc": "2026-11-28T01:00:00+00:00",
        "completed_at_utc": "2026-11-28T01:00:25+00:00"}
    report = {key: row[key] for key in ("episode_id", "stage", "role", "seed")}
    report.update(received_ml=19.25, error_ml=-.75, physical_s=5., control_target_offset_ml=0.,
                  easy_assessment=copy.deepcopy(easy), strict_assessment=copy.deepcopy(strict),
                  actual_prefix_anchor={"physical_time_s": 3., "received_ml": 19.25,
                      "remaining_ml": 37.437, "actual_target_ml": 20., "control_target_ml": 20.,
                      "control_target_offset_ml": 0.})
    return row, report


class CurationGuards(unittest.TestCase):
    def test_easy_pass_does_not_become_original_strict_pass(self):
        row, report = fixture()
        result = curate.compact_case(row, "test", report)
        self.assertTrue(result["easy_feasible"])
        self.assertFalse(result["original_strict_feasible"])
        self.assertEqual(result["compute_wall_seconds_including_post"], 25.)
        self.assertEqual(result["physical_action_seconds"], 5.)

    def test_audited_dose_tampering_is_rejected(self):
        row, report = fixture()
        report["received_ml"] += .019683
        with self.assertRaisesRegex(ValueError, "Audited measurement"):
            curate.compact_case(row, "test", report)

    def test_post_and_clock_may_not_be_hidden(self):
        row, report = fixture()
        row["post_observation"]["physical_steps"] = 119
        with self.assertRaisesRegex(ValueError, "120 live steps"):
            curate.compact_case(row, "test", report)
        row, report = fixture()
        row["result"]["physical_steps"] = 301
        with self.assertRaisesRegex(ValueError, "step/60"):
            curate.compact_case(row, "test", report)

    def test_false_easy_pass_rejected_even_when_audit_repeats_it(self):
        row, report = fixture()
        row["result"].update(received_ml=18., error_ml=-2., absolute_error_ml=2.)
        report.update(received_ml=18., error_ml=-2.)
        with self.assertRaisesRegex(ValueError, "Beginner dose gate"):
            curate.compact_case(row, "test", report)

    def test_incomplete_audit_rejected_before_journal_read(self):
        with tempfile.TemporaryDirectory(dir=TEST_TEMP) as temporary:
            run = Path(temporary)
            audit = run / "overall_audit.json"
            audit.write_text(json.dumps({"complete": False, "pass": True}), encoding="utf8")
            with self.assertRaisesRegex(ValueError, "Complete passing independent"):
                curate.curate(run, audit, curate.sha(audit))

    def test_explicit_audit_sha_required(self):
        with tempfile.TemporaryDirectory(dir=TEST_TEMP) as temporary:
            run = Path(temporary)
            audit = run / "overall_audit.json"
            audit.write_text("{}", encoding="utf8")
            with self.assertRaisesRegex(ValueError, "audit SHA"):
                curate.curate(run, audit, "0" * 64)


if __name__ == "__main__":
    unittest.main()
