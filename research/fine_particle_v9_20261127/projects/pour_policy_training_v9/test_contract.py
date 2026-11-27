"""Pure CPU counterexamples for the V9 task gate and selection ordering.

No NumPy, Warp, Newton, Kit, geometry cache, or liquid simulation is loaded.
These tests exercise the public task assessment rather than legacy success.
"""

import copy
import gzip
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("v9_contract_runner", HERE / "run_fine_search.py")
RUNNER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = RUNNER
SPEC.loader.exec_module(RUNNER)


def evidence(target=20.0, error=0.0, time_s=8.0):
    result = {
        "target_ml": target,
        "received_ml": target + error,
        "error_ml": error,
        "physical_time_s": time_s,
        "measurement_physical_time_s": time_s,
        "physics_dt_s": 1.0 / 60.0,
        "valid_physics": True,
        "mass_conserved": True,
        "all_states_finite": True,
        "source_partition_conserved": True,
        "max_below_table_particles": 0,
        "complete": True,
        "success": False,
    }
    status = {
        "done": True,
        "home_static": True,
        "stable_frames": 13,
        "source_airborne_particles": 0,
        "timed_out": False,
        "completed_by_deadline": True,
    }
    post = {
        "start_physical_time_s": time_s,
        "end_physical_time_s": time_s + 2.0,
        "physical_steps": 120,
        "duration_s": 2.0,
        "maximum_received_change_ml": 0.0,
        "all_states_finite": True,
        "mass_array_unchanged": True,
        "maximum_below_table_particles": 0,
    }
    return result, status, post


def row(error=0.0, spill=0, time_s=8.0, target=20.0, **result_changes):
    result, status, post = evidence(target, error, time_s)
    result.update(result_changes)
    return {
        "result": result,
        "assessment": RUNNER.assessment(result, status, spill, post),
        "wall_seconds": 1.0,
    }


class AssessmentContract(unittest.TestCase):
    def assess(self, result_changes=None, status_changes=None, post_changes=None, spill=0):
        result, status, post = evidence()
        result.update(result_changes or {})
        if result_changes and "error_ml" in result_changes:
            result["received_ml"] = result["target_ml"] + result["error_ml"]
        status.update(status_changes or {})
        post.update(post_changes or {})
        return RUNNER.assessment(result, status, spill, post)

    def test_zero_and_one_all_label_spill_are_allowed(self):
        self.assertTrue(self.assess(spill=0)["feasible"])
        self.assertTrue(self.assess(spill=1)["feasible"])

    def test_two_all_label_spills_are_rejected(self):
        self.assertFalse(self.assess(spill=2)["feasible"])

    def test_relative_error_thresholds_on_both_sides(self):
        for target in (20.0, 30.0, 40.0):
            for sign in (-1, 1):
                with self.subTest(target=target, sign=sign):
                    self.assertTrue(row(sign * target * .001, target=target)["assessment"]["dose_passed"])
                    self.assertFalse(row(sign * (target * .001 + 1e-9), target=target)["assessment"]["dose_passed"])

    def test_old_point_one_ml_success_is_not_percent_acceptance(self):
        self.assertFalse(self.assess({"error_ml": .05, "success": True})["feasible"])
        self.assertTrue(self.assess({"error_ml": .01, "success": False})["feasible"])

    def test_deadline_boundary_is_raw_physical_time(self):
        self.assertTrue(row(time_s=24.0)["assessment"]["feasible"])
        self.assertFalse(row(time_s=24.0 + 1e-9)["assessment"]["feasible"])

    def test_incomplete_or_nonhome_or_short_quiet_is_rejected(self):
        self.assertFalse(self.assess({"complete": False})["feasible"])
        for changes in ({"done": False}, {"home_static": False}, {"stable_frames": 11},
                        {"source_airborne_particles": 1}, {"timed_out": True}):
            with self.subTest(changes=changes):
                self.assertFalse(self.assess(status_changes=changes)["feasible"])

    def test_physics_mass_finite_and_partition_failures_are_invalid(self):
        for field in ("valid_physics", "mass_conserved", "all_states_finite", "source_partition_conserved"):
            with self.subTest(field=field):
                self.assertFalse(self.assess({field: False})["valid_evidence"])

    def test_stale_measurement_is_invalid(self):
        self.assertFalse(self.assess({"measurement_physical_time_s": 7.9})["valid_evidence"])

    def test_below_table_action_or_post_is_invalid(self):
        self.assertFalse(self.assess({"max_below_table_particles": 1})["valid_evidence"])
        self.assertFalse(self.assess(post_changes={"maximum_below_table_particles": 1})["valid_evidence"])

    def test_late_dose_change_cannot_rescue_action(self):
        self.assertFalse(self.assess(post_changes={"maximum_received_change_ml": .019683})["feasible"])

    def test_missing_or_failed_post_is_invalid(self):
        result, status, _ = evidence()
        self.assertFalse(RUNNER.assessment(result, status, 0, None)["valid_evidence"])
        self.assertFalse(self.assess(post_changes={"all_states_finite": False})["valid_evidence"])
        self.assertFalse(self.assess(post_changes={"mass_array_unchanged": False})["valid_evidence"])

    def test_zero_or_short_post_duration_is_rejected(self):
        for duration in (0.0, 1.0, 2.0 - 1e-6):
            with self.subTest(duration=duration):
                self.assertFalse(self.assess(post_changes={"end_physical_time_s": 8.0 + duration})["valid_evidence"])

    def test_post_must_start_at_action_measurement(self):
        self.assertFalse(self.assess(post_changes={"start_physical_time_s": 7.0})["valid_evidence"])


class RankingContract(unittest.TestCase):
    def test_infeasible_volume_precedes_spill(self):
        self.assertLess(RUNNER.key([row(error=.04, spill=10)]), RUNNER.key([row(error=.08, spill=0)]))

    def test_infeasible_equal_volume_then_spill(self):
        self.assertLess(RUNNER.key([row(error=.04, spill=0)]), RUNNER.key([row(error=.04, spill=1)]))

    def test_infeasible_time_does_not_change_key(self):
        self.assertEqual(RUNNER.key([row(error=.04, time_s=5)]), RUNNER.key([row(error=.04, time_s=20)]))

    def test_all_cases_must_be_feasible_before_time_ranking(self):
        self.assertLess(RUNNER.key([row(time_s=20)]), RUNNER.key([row(error=.04, time_s=5)]))
        self.assertLess(RUNNER.key([row(time_s=5)]), RUNNER.key([row(time_s=20)]))

    def test_invalid_evidence_never_beats_valid_infeasible(self):
        self.assertEqual(RUNNER.key([row(error=0, valid_physics=False)])[0], 2)
        self.assertLess(RUNNER.key([row(error=1)]), RUNNER.key([row(error=0, valid_physics=False)]))

    def test_import_is_cpu_only(self):
        self.assertNotIn("warp", sys.modules)
        self.assertNotIn("newton", sys.modules)


class ResumePostEvidenceContract(unittest.TestCase):
    """Synthetic JSONL evidence only; no formal output or solver is accessed."""

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="contract_", dir=HERE)
        self.output = Path(self.temporary.name)
        self.previous_output = RUNNER.OUT
        RUNNER.OUT = self.output
        (self.output / "completed").mkdir()
        (self.output / "traces").mkdir()
        self.episode_id = "train_candidate_g0c0_t20_s171101"
        self.action_name = f"traces/{self.episode_id}.steps.jsonl.gz"
        self.post_name = f"traces/{self.episode_id}.post.jsonl.gz"
        result, status, post = evidence()
        self.case = {
            "episode_id": self.episode_id, "seed": 171101, "target_ml": 20.0,
            "result": result, "status": status, "post_observation": post,
            "assessment": RUNNER.assessment(result, status, 0, post), "artifacts": {},
        }
        self.post_rows = [{"type": "post_observation", "step": step,
            "physical_time_s": 8.0 + step / 60.0,
            "measurement": {"physical_time_s": 8.0 + step / 60.0,
                "received_ml": 20.0, "below_table": 0, "spilled": 0}}
            for step in range(1, 121)]
        self.write_action()
        self.write_post()
        self.persist()

    def tearDown(self):
        RUNNER.OUT = self.previous_output
        self.temporary.cleanup()

    def write_action(self):
        items = [
            {"type": "episode", "seed": self.case["seed"], "target_ml": 20.0},
            {"type": "direct_step_evidence", "controller_status": self.case["status"],
             "maximum_all_label_spilled_particles": 0, "partitions": {"spilled": 0}},
            {"type": "result", **self.case["result"]},
        ]
        with gzip.open(self.output / self.action_name, "wt", encoding="utf8") as stream:
            for item in items:
                stream.write(json.dumps(item) + "\n")

    def write_post(self):
        with gzip.open(self.output / self.post_name, "wt", encoding="utf8") as stream:
            for item in self.post_rows:
                stream.write(json.dumps(item) + "\n")

    def persist(self, recompute_assessment=True):
        self.case["artifacts"] = {name: RUNNER.sha(self.output / name)
            for name in (self.action_name, self.post_name) if (self.output / name).is_file()}
        if recompute_assessment:
            self.case["assessment"] = RUNNER.assessment(self.case["result"], self.case["status"],
                self.case["assessment"]["maximum_all_label_spilled_particles"], self.case["post_observation"])
        self.persist_sidecar()

    def persist_sidecar(self):
        (self.output / "completed" / (self.episode_id + ".json")).write_text(
            json.dumps(self.case), encoding="utf8")

    def assert_rejected(self):
        with self.assertRaises(RuntimeError):
            RUNNER.validate_row(self.case)

    def failed_post(self):
        self.case["post_observation"] = {"error": "synthetic failed observation",
            "all_states_finite": False, "mass_array_unchanged": False,
            "maximum_received_change_ml": None, "end_physical_time_s": 8.1}

    def test_valid_120_frame_post_is_accepted(self):
        RUNNER.validate_row(self.case)

    def test_valid_post_without_file_is_rejected(self):
        (self.output / self.post_name).unlink()
        self.persist()
        self.assert_rejected()

    def test_post_file_without_sha_entry_is_rejected(self):
        del self.case["artifacts"][self.post_name]
        self.persist_sidecar()
        self.assert_rejected()

    def test_extra_post_artifact_is_rejected(self):
        extra = "traces/unrelated.post.jsonl.gz"
        (self.output / extra).write_bytes((self.output / self.post_name).read_bytes())
        self.case["artifacts"][extra] = RUNNER.sha(self.output / extra)
        self.persist_sidecar()
        self.assert_rejected()

    def test_truncated_valid_post_is_rejected(self):
        path = self.output / self.post_name
        path.write_bytes(path.read_bytes()[:-8])
        self.persist()
        self.assert_rejected()

    def test_short_post_cannot_claim_120_frames(self):
        self.post_rows.pop()
        self.write_post()
        self.persist()
        self.assert_rejected()

    def test_repeated_or_boolean_step_is_rejected(self):
        self.post_rows[1]["step"] = 1
        self.write_post()
        self.persist()
        self.assert_rejected()
        self.post_rows[1]["step"] = 2
        self.post_rows[0]["step"] = True
        self.write_post()
        self.persist()
        self.assert_rejected()

    def test_post_clock_gap_is_rejected(self):
        self.post_rows[1]["physical_time_s"] += 1.0 / 60.0
        self.write_post()
        self.persist()
        self.assert_rejected()

    def test_measurement_clock_must_match_post_clock(self):
        self.post_rows[1]["measurement"]["physical_time_s"] += .1
        self.write_post()
        self.persist()
        self.assert_rejected()

    def test_nonfinite_post_dose_is_rejected(self):
        self.post_rows[0]["measurement"]["received_ml"] = float("nan")
        self.write_post()
        self.persist()
        self.assert_rejected()

    def test_raw_dose_change_must_match_metadata(self):
        self.post_rows[-1]["measurement"]["received_ml"] += .019683
        self.write_post()
        self.persist()
        self.assert_rejected()

    def test_truthful_late_change_is_retained_as_task_failure(self):
        self.post_rows[-1]["measurement"]["received_ml"] += .019683
        self.case["post_observation"]["maximum_received_change_ml"] = abs(
            self.post_rows[-1]["measurement"]["received_ml"] - 20.0)
        self.write_post()
        self.persist()
        RUNNER.validate_row(self.case)
        self.assertFalse(self.case["assessment"]["feasible"])

    def test_metadata_steps_duration_and_endpoint_are_bound(self):
        original = copy.deepcopy(self.case["post_observation"])
        for field, value in (("physical_steps", 121), ("physical_steps", 120.0),
                             ("duration_s", 1.0), ("end_physical_time_s", 10.1)):
            with self.subTest(field=field, value=value):
                self.case["post_observation"] = {**original, field: value}
                self.persist()
                self.assert_rejected()

    def test_partial_failed_post_is_preserved_without_acceptance(self):
        self.failed_post()
        self.post_rows = self.post_rows[:3]
        self.write_post()
        self.persist()
        RUNNER.validate_row(self.case)
        self.assertFalse(self.case["assessment"]["valid_evidence"])
        self.assertFalse(self.case["assessment"]["feasible"])

    def test_truncated_failed_post_is_preserved_without_acceptance(self):
        self.failed_post()
        path = self.output / self.post_name
        path.write_bytes(path.read_bytes()[:-8])
        self.persist()
        RUNNER.validate_row(self.case)
        self.assertFalse(self.case["assessment"]["valid_evidence"])

    def test_failed_post_before_file_creation_is_retained(self):
        self.failed_post()
        (self.output / self.post_name).unlink()
        self.persist()
        RUNNER.validate_row(self.case)
        self.assertFalse(self.case["assessment"]["feasible"])

    def test_failed_post_cannot_claim_valid_evidence(self):
        self.failed_post()
        self.persist()
        self.case["assessment"]["valid_evidence"] = True
        self.persist_sidecar()
        self.assert_rejected()


if __name__ == "__main__":
    unittest.main()
