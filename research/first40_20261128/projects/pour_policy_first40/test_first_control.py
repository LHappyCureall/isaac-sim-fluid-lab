"""CPU contracts for the one-factor40mL angle study; never liquid validation."""

import math
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
for path in list((ROOT / "extscache").glob("omni.kit.pip_archive-*/pip_prebundle")) + [ROOT / "exts/omni.pip.compute/pip_prebundle"]:
    if path.is_dir(): sys.path.insert(0, str(path))
import first_control as control


def fixture(first=87., *, count=1, retry=True, peak=0, received=39., quiet=True, budget=10.):
    instance = object.__new__(control.TargetPourController)
    instance.profile = control.CorrectionPolicy(first_approach_tilt_deg=first, enable_retry=retry,
        profile_id="cpu_fixture", profile_sha256="a" * 64, condition_seed=197304)
    instance.target_ml = 40.
    instance._cycle_count, instance._cycle_observed_peak_spill = count, peak
    instance._retry_in_band_home, instance._retry_blocked_reason = False, None
    instance._retry_decisions, instance.response_events = [], []
    instance._retry_selected_approach_deg = None
    instance.deadline = 24.
    instance.source_time = lambda t: 1.
    instance._planned_budget = lambda u, angle: budget
    instance._observation = lambda t, status: {"tube_remaining_ml": 16.}
    instance.logged_intents = []
    instance._event = lambda name, t, status, **extra: instance.logged_intents.append((name, t, extra))
    return instance, {"received_ml": received, "settled_measurement": quiet, "tilt_deg": 78.}


class Contract(unittest.TestCase):
    def test_exact40ml_only_and_original_particle_clock(self):
        self.assertEqual(control.CorrectionPolicy().target_ml, 40.)
        for target in (20., 30., 35., 40.01, True, float("nan")):
            with self.assertRaises(ValueError): control.CorrectionPolicy(target_ml=target)
        for deadline in (5., 23.9, 24.1, True):
            with self.assertRaises(ValueError): control.CorrectionPolicy(deadline_s=deadline)

    def test_first_angle_closed_bounds_retry_fixed_independently(self):
        for first in (86.9, 86.95, 87., 87.05, 87.1):
            profile = control.CorrectionPolicy(first_approach_tilt_deg=first)
            self.assertEqual(profile.first_approach_tilt_deg, first)
            self.assertEqual(profile.retry_approach_tilt_deg, 87.2)
        for first in (86.899999, 87.100001, None, True, float("nan")):
            with self.assertRaises(ValueError): control.CorrectionPolicy(first_approach_tilt_deg=first)
        for second in (87.1, 87.3, True, None, float("inf")):
            with self.assertRaises(ValueError): control.CorrectionPolicy(retry_approach_tilt_deg=second)

    def test_one_factor_changes_no_motion_or_retry_parameter(self):
        baseline = control.CorrectionPolicy(first_approach_tilt_deg=87.1, condition_seed=197304)
        candidate = control.CorrectionPolicy(first_approach_tilt_deg=86.95, condition_seed=197304)
        difference = {key for key in baseline.params if baseline.params[key] != candidate.params[key]}
        self.assertEqual(difference, {"approach_tilt_deg", "first_approach_tilt_deg"})
        self.assertTrue(baseline.enable_retry and candidate.enable_retry)
        self.assertEqual(baseline.maximum_cycles, 2)
        self.assertEqual(baseline.coarse, control.FIXED_COARSE)
        expected = {"approach_tilt_deg_s": 6., "action_delta_deg": .03, "action_tilt_deg_s": .6,
            "hold_seconds": .1, "backoff_delta_deg": .5, "backoff_tilt_deg_s": 4.,
            "anchor_return_tilt_deg_s": 6., "quiet_seconds": .25, "maximum_measurement_hold_s": 1.}
        for key, value in expected.items(): self.assertEqual(candidate.params[key], value)
        altered = dict(control.FIXED_COARSE); altered["dose_tilt_deg_s"] = 25.5
        with self.assertRaises(ValueError): control.CorrectionPolicy(coarse=altered)

    def test_seed_and_external_profile_provenance_are_explicit(self):
        profile = control.CorrectionPolicy(condition_seed=197304, profile_id="known_failure",
            profile_sha256="b" * 64)
        self.assertEqual(profile.condition_seed, 197304)
        self.assertEqual(profile.params["target_profile_sha256"], "b" * 64)
        for seed in (True, -1, 1.5, 2 ** 31):
            with self.assertRaises(ValueError): control.CorrectionPolicy(condition_seed=seed)
        with self.assertRaises(ValueError): control.CorrectionPolicy(profile_id="missing_digest")
        instance, _ = fixture()
        metadata = instance._metadata()
        self.assertEqual(metadata["first40_condition_seed"], 197304)
        self.assertTrue(metadata["condition_seed_provided_by_runner"])
        self.assertEqual(metadata["first40_controller_source_sha256"], control._SOURCE_SHA256)
        self.assertNotIn("retry3040_controller_source_sha256", metadata)

    def test_actual_closed_primary_and_original_strict_are_separate(self):
        for amount in (39.5, 40.5):
            self.assertTrue(control.in_task_band(40., amount))
            self.assertFalse(control.cycle.actual_dose_in_band(40., amount))
        self.assertFalse(control.in_task_band(40., math.nextafter(40.5, math.inf)))
        self.assertFalse(control.in_task_band(40., math.nextafter(39.5, -math.inf)))

    def test_first_action_varies_second_action_remains87point2(self):
        for angle in (86.95, 87., 87.05):
            instance, status = fixture(first=angle, count=0)
            first = instance._choose_cycle_action(3., status)
            self.assertAlmostEqual(first["approach_delta_deg"] + 78., angle)
            instance._cycle_count = 1
            second = instance._choose_cycle_action(10., status)
            self.assertAlmostEqual(second["approach_delta_deg"] + 78., 87.2)
            self.assertEqual(instance._retry_decisions[0]["condition_seed"], 197304)
            self.assertEqual(instance._retry_decisions[0]["actual_remaining_ml"], 16.)
            self.assertNotIn("tube_remaining_ml", status)

    def test_in_band_actual_measurement_and_overdose_skip_retry(self):
        for amount in (39.5, 40.5, 40.527297):
            instance, status = fixture(received=amount)
            self.assertIsNone(instance._choose_cycle_action(11., status))
            self.assertEqual(instance._retry_decisions, [])
        instance, status = fixture(received=39.4, quiet=False)
        self.assertIsNone(instance._choose_cycle_action(11., status))

    def test_current_spill_and_full_home_time_guards_are_unchanged(self):
        for peak in (None, 2):
            instance, status = fixture(peak=peak)
            self.assertIsNone(instance._choose_cycle_action(11., status))
            self.assertFalse(instance._retry_decisions[0]["spill_guard_passed"])
        instance, status = fixture(budget=13.7)
        self.assertIsNone(instance._choose_cycle_action(12.7, status))
        self.assertFalse(instance._retry_decisions[0]["full_home_budget_fits"])
        instance, status = fixture(budget=10.)
        self.assertIsNotNone(instance._choose_cycle_action(14., status))

    def test_decision_intent_is_not_an_actual_second_cycle(self):
        instance, status = fixture()
        action = instance._choose_cycle_action(11., status)
        self.assertIsNotNone(action)
        self.assertEqual(instance._cycle_count, 1)
        self.assertEqual(instance.response_events, [])
        self.assertIsNone(instance._metadata()["retry_actual_approach_tilt_deg"])
        decision = instance._retry_decisions[0]
        self.assertTrue(decision["decision_is_intent_not_execution"])
        self.assertTrue(decision["budget_is_robot_motion_not_predicted_water_dose"])
        self.assertEqual(instance.logged_intents[0][0], "first40_conditional_second_cycle_decision")

    def test_no_more_than_two_cycles_or_disabled_retry(self):
        for values in ({"retry": False}, {"count": 2}):
            instance, status = fixture(**values)
            self.assertIsNone(instance._choose_cycle_action(11., status))


if __name__ == "__main__": unittest.main()
