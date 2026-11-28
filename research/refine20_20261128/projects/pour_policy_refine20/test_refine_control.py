"""CPU-only contract checks; none of these tests claim a liquid result."""

import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
for path in list((ROOT / "extscache").glob("omni.kit.pip_archive-*/pip_prebundle")) + [
        ROOT / "exts/omni.pip.compute/pip_prebundle"]:
    if path.is_dir():
        sys.path.insert(0, str(path))

import refine_control as control


class Contract(unittest.TestCase):
    def test_only_one_physical_parameter_varies(self):
        profiles = [control.CorrectionPolicy(approach_tilt_deg=angle) for angle in control.APPROACH_ANGLES_DEG]
        common = [{key: value for key, value in profile.params.items() if key != "approach_tilt_deg"}
                  for profile in profiles]
        self.assertEqual(common[0], common[1])
        self.assertEqual(common[1], common[2])
        self.assertEqual([profile.approach_tilt_deg for profile in profiles], [84., 84.1, 84.2])

    def test_unlisted_angle_and_other_profile_changes_rejected(self):
        for angle in (83.9, 84.3, True, float("nan")):
            with self.assertRaises(ValueError):
                control.CorrectionPolicy(approach_tilt_deg=angle)
        with self.assertRaises(ValueError):
            control.CorrectionPolicy({**control.FIXED_COARSE, "dose_tilt_deg_s": 26.})
        with self.assertRaises(ValueError):
            control.CorrectionPolicy(deadline_s=25.)
        with self.assertRaises(ValueError):
            control.CorrectionPolicy(enable_correction=1)

    def test_actual_closed_band_is_distinct_from_original_strict(self):
        for dose in (19.5, 20.5):
            self.assertTrue(control.in_refine_band(dose))
            self.assertFalse(control.cycle.actual_dose_in_band(20., dose))
        for dose in (19.499999, 20.500001):
            self.assertFalse(control.in_refine_band(dose))

    def test_measured_band_can_skip_a_correction(self):
        instance = object.__new__(control.TargetPourController)
        instance._refine_in_band_home = False
        self.assertIsNone(instance._choose_cycle_action(3., {"settled_measurement": True, "received_ml": 19.75}))
        self.assertTrue(instance._refine_in_band_home)
        with patch.object(control.cycle.TargetPourController, "_choose_cycle_action", return_value="declared"):
            self.assertEqual(instance._choose_cycle_action(3., {"settled_measurement": False, "received_ml": 19.75}), "declared")
            self.assertEqual(instance._choose_cycle_action(3., {"settled_measurement": True, "received_ml": 19.25}), "declared")

    def test_baseline_home_bypasses_cycle_interceptors(self):
        instance = object.__new__(control.TargetPourController)
        instance.profile = control.CorrectionPolicy(enable_correction=False)
        instance._refine_anchor_requested_at_s = None
        with patch.object(control.cycle.response.parent.TargetPourController, "_start_home", return_value="realhome") as real:
            with patch.object(control.cycle.TargetPourController, "_start_home", side_effect=AssertionError("intercepted")):
                self.assertEqual(instance._start_home(3., 1.), "realhome")
                real.assert_called_once_with(instance, 3., 1.)
        self.assertEqual(instance._refine_anchor_requested_at_s, 3.)

    def test_new_home_reason_does_not_change_parent_strict_predicate(self):
        instance = object.__new__(control.TargetPourController)
        instance._refine_in_band_home = False
        status = {"settled_measurement": True, "received_ml": 20.25}
        with patch.object(control.cycle.TargetPourController, "_real_home") as home:
            instance._real_home(5., 1., status, "actual_overdose_cannot_be_removed")
            self.assertEqual(home.call_args.args[-1], "actual_settled_refine_half_ml_closed_interval")
        self.assertTrue(instance._refine_in_band_home)
        self.assertFalse(control.cycle.actual_dose_in_band(20., 20.25))


if __name__ == "__main__":
    unittest.main()
