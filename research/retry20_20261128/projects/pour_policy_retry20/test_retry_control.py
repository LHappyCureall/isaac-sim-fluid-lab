"""CPU contracts for a fixed conditional retry; not liquid validation."""

from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
for path in list((ROOT / "extscache").glob("omni.kit.pip_archive-*/pip_prebundle")) + [
        ROOT / "exts/omni.pip.compute/pip_prebundle"]:
    if path.is_dir():
        sys.path.insert(0, str(path))
import retry_control as control


def fixture(count=1, retry=True, peak=0, received=19.3, quiet=True):
    instance = object.__new__(control.TargetPourController)
    instance.profile = control.CorrectionPolicy(enable_retry=retry)
    instance._cycle_count = count
    instance._cycle_observed_peak_spill = peak
    instance._retry_in_band_home = False
    instance._retry_blocked_reason = None
    instance._retry_decisions = []
    instance._retry_selected_actual_approach_deg = None
    instance.deadline = 24.
    instance.source_time = lambda t: 1.
    instance._planned_budget = lambda u, angle: 9.
    instance._fixed_cycle_action = lambda status: {"approach_delta_deg": 6., "delta_deg": .03}
    instance._observation = lambda t, status: {"physical_time_s": t,
        "received_ml": status["received_ml"], "tube_remaining_ml": 37.,
        "settled_measurement": status["settled_measurement"]}
    instance._event = lambda *args, **kwargs: None
    return instance, {"received_ml": received, "settled_measurement": quiet,
                      "tilt_deg": 78.}


class Contract(unittest.TestCase):
    def test_only_retry_budget_differs_between_profiles(self):
        baseline, retry = control.CorrectionPolicy(enable_retry=False), control.CorrectionPolicy(enable_retry=True)
        differences = {key for key in baseline.params if baseline.params[key] != retry.params[key]}
        self.assertEqual(differences, {"enable_retry", "maximum_cycles", "one_declared_small_action_per_episode"})
        self.assertEqual((baseline.maximum_cycles, retry.maximum_cycles), (1, 2))
        with self.assertRaises(ValueError): control.CorrectionPolicy(enable_retry=1)
        with self.assertRaises(ValueError): control.CorrectionPolicy(deadline_s=25.)
        with self.assertRaises(ValueError): control.CorrectionPolicy({**control.FIXED_COARSE, "dose_tilt_deg_s": 25.})

    def test_exact_half_ml_band_and_original_strict_unchanged(self):
        for amount in (19.5, 20.5):
            self.assertTrue(control.in_task_band(amount))
            self.assertFalse(control.cycle.actual_dose_in_band(20., amount))
        self.assertFalse(control.in_task_band(19.499999))

    def test_first_cycle_uses84_and_second_uses841(self):
        instance, status = fixture(count=0)
        self.assertEqual(instance._choose_cycle_action(3., status)["approach_delta_deg"], 6.)
        self.assertEqual(instance._retry_selected_actual_approach_deg, 84.)
        self.assertEqual(instance._retry_decisions, [])
        instance._cycle_count = 1
        action = instance._choose_cycle_action(11., status)
        self.assertAlmostEqual(action["approach_delta_deg"], 6.1)
        self.assertTrue(instance._retry_decisions[0]["retry_allowed"])
        self.assertEqual(instance._retry_decisions[0]["conservative_full_home_budget_s"], 9.)

    def test_actual_settled_band_skips_retry(self):
        instance, status = fixture(received=19.5)
        self.assertIsNone(instance._choose_cycle_action(11., status))
        self.assertTrue(instance._retry_in_band_home)
        self.assertEqual(instance._retry_decisions, [])

    def test_second_cycle_requires_observed_spill_guard(self):
        for peak in (None, 2):
            instance, status = fixture(peak=peak)
            self.assertIsNone(instance._choose_cycle_action(11., status))
            self.assertFalse(instance._retry_decisions[0]["retry_allowed"])
            self.assertFalse(instance._retry_decisions[0]["spill_guard_passed"])
        instance, status = fixture(peak=1)
        self.assertIsNotNone(instance._choose_cycle_action(11., status))

    def test_remaining_water_comes_from_actual_observation_not_status(self):
        instance, status = fixture()
        self.assertNotIn("tube_remaining_ml", status)
        instance._observation = lambda t, value: {"tube_remaining_ml": 36.9}
        self.assertIsNotNone(instance._choose_cycle_action(11., status))
        self.assertEqual(instance._retry_decisions[0]["actual_remaining_ml"], 36.9)

    def test_second_cycle_requires_full_home_budget(self):
        instance, status = fixture()
        self.assertIsNone(instance._choose_cycle_action(15.01, status))
        self.assertFalse(instance._retry_decisions[0]["full_home_budget_fits"])
        instance, status = fixture()
        self.assertIsNotNone(instance._choose_cycle_action(15., status))

    def test_baseline_and_completed_budget_cannot_retry(self):
        for count, retry in ((1, False), (2, True)):
            instance, status = fixture(count=count, retry=retry)
            self.assertIsNone(instance._choose_cycle_action(11., status))
            self.assertEqual(instance._retry_decisions, [])

    def test_no_action_from_unsettled_or_overdose(self):
        for kwargs in ({"quiet": False}, {"received": 20.6}):
            instance, status = fixture(**kwargs)
            self.assertIsNone(instance._choose_cycle_action(11., status))
            self.assertEqual(instance._retry_decisions, [])


if __name__ == "__main__":
    unittest.main()
