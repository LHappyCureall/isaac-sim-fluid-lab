"""CPU checks of declared30/40mL feedback; never liquid validation."""

from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
for path in list((ROOT / "extscache").glob("omni.kit.pip_archive-*/pip_prebundle")) + [ROOT / "exts/omni.pip.compute/pip_prebundle"]:
    if path.is_dir(): sys.path.insert(0, str(path))
import retry_control as control


def fixture(target=30., count=1, retry=True, peak=0, received=None, quiet=True, budget=10.):
    instance = object.__new__(control.TargetPourController)
    instance.profile = control.CorrectionPolicy(target_ml=target, enable_retry=retry,
        profile_id="cpu_test", profile_sha256="a" * 64)
    instance.target_ml = target
    instance._cycle_count, instance._cycle_observed_peak_spill = count, peak
    instance._retry_in_band_home, instance._retry_blocked_reason = False, None
    instance._retry_decisions, instance.response_events = [], []
    instance._retry_selected_approach_deg = None
    instance.deadline = 24.
    instance.source_time = lambda t: 1.
    instance._planned_budget = lambda u, angle: budget
    instance._observation = lambda t, status: {"tube_remaining_ml": 26.}
    instance._event = lambda *args, **kwargs: None
    return instance, {"received_ml": target - 1. if received is None else received,
                      "settled_measurement": quiet, "tilt_deg": 78.}


class Contract(unittest.TestCase):
    def test_declared_targets_and_default_map(self):
        for target, first, second in ((30., 85.5, 85.6), (40., 87.2, 87.3)):
            policy = control.CorrectionPolicy(target_ml=target)
            self.assertEqual((policy.first_approach_tilt_deg, policy.retry_approach_tilt_deg), (first, second))
        for target in (20., 35., True, float("nan")):
            with self.assertRaises(ValueError): control.CorrectionPolicy(target_ml=target)

    def test_explicit_profile_does_not_change_code_or_speeds(self):
        policy = control.CorrectionPolicy(target_ml=40., first_approach_tilt_deg=87.1,
            retry_approach_tilt_deg=87.2, profile_id="pilot_002", profile_sha256="b" * 64)
        self.assertEqual(policy.approach_tilt_deg, 87.1)
        self.assertEqual(policy.params["target_profile_sha256"], "b" * 64)
        self.assertEqual(policy.coarse, control.FIXED_COARSE)
        self.assertEqual(policy.params["approach_tilt_deg_s"], 6.)
        for kwargs in ({"first_approach_tilt_deg": 87.},
                       {"first_approach_tilt_deg": 87., "retry_approach_tilt_deg": 87.2},
                       {"first_approach_tilt_deg": True, "retry_approach_tilt_deg": 78.1},
                       {"profile_sha256": "a" * 64}):
            with self.assertRaises(ValueError): control.CorrectionPolicy(target_ml=40., **kwargs)

    def test_actual_closed_half_ml_band_and_strict_separate(self):
        for target in (30., 40.):
            for amount in (target - .5, target + .5):
                self.assertTrue(control.in_task_band(target, amount))
                self.assertFalse(control.cycle.actual_dose_in_band(target, amount))
            self.assertFalse(control.in_task_band(target, target - .500001))

    def test_one_cycle_and_retry_only_differ_in_budget_flags(self):
        for target in (30., 40.):
            baseline = control.CorrectionPolicy(target_ml=target, enable_retry=False)
            retry = control.CorrectionPolicy(target_ml=target, enable_retry=True)
            difference = {key for key in baseline.params if baseline.params[key] != retry.params[key]}
            self.assertEqual(difference, {"enable_retry", "maximum_cycles", "one_declared_small_action_per_episode"})

    def test_first_and_second_angles_use_declared_goal_profile(self):
        for target in (30., 40.):
            instance, status = fixture(target=target, count=0)
            first = instance._choose_cycle_action(3., status)
            self.assertAlmostEqual(first["approach_delta_deg"] + 78., instance.profile.first_approach_tilt_deg)
            instance._cycle_count = 1
            second = instance._choose_cycle_action(11., status)
            self.assertAlmostEqual(second["approach_delta_deg"] + 78., instance.profile.retry_approach_tilt_deg)
            decision = instance._retry_decisions[0]
            self.assertEqual(decision["declared_target_ml"], target)
            self.assertEqual(decision["actual_deficit_ml"], 1.)
            self.assertEqual(decision["actual_remaining_ml"], 26.)
            self.assertNotIn("tube_remaining_ml", status)

    def test_half_ml_actual_quiet_can_skip_second(self):
        for target in (30., 40.):
            instance, status = fixture(target=target, received=target - .5)
            self.assertIsNone(instance._choose_cycle_action(11., status))
            self.assertEqual(instance._retry_decisions, [])

    def test_spill_and_time_guards_cannot_be_relaxed_for40(self):
        for peak in (None, 2):
            instance, status = fixture(target=40., peak=peak)
            self.assertIsNone(instance._choose_cycle_action(11., status))
            self.assertFalse(instance._retry_decisions[0]["spill_guard_passed"])
        instance, status = fixture(target=40., budget=13.7)
        self.assertIsNone(instance._choose_cycle_action(12.7, status))
        self.assertFalse(instance._retry_decisions[0]["full_home_budget_fits"])
        self.assertIsNone(instance._metadata()["retry_actual_approach_tilt_deg"])
        instance, status = fixture(target=30., budget=10.)
        self.assertIsNotNone(instance._choose_cycle_action(14., status))

    def test_no_retry_when_disabled_completed_or_not_quiet(self):
        for values in ({"retry": False}, {"count": 2}, {"quiet": False}, {"received": 40.6}):
            instance, status = fixture(target=40., **values)
            self.assertIsNone(instance._choose_cycle_action(11., status))


if __name__ == "__main__": unittest.main()
