"""CPU contracts only; no test runs liquid physics or certifies a spill fix."""

import hashlib
import importlib.util
import math
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
for directory in list((ROOT / "extscache").glob("omni.kit.pip_archive-*/pip_prebundle")) + [ROOT / "exts/omni.pip.compute/pip_prebundle"]:
    if directory.is_dir():
        sys.path.insert(0, str(directory))
import numpy as np
import coarse_control as control


def fixture(speed=25.6, *, count=1, peak=0, received=39., quiet=True, budget=10.):
    instance = object.__new__(control.TargetPourController)
    instance.profile = control.CorrectionPolicy(coarse_dose_tilt_deg_s=speed,
        profile_id="cpu_fixture", profile_sha256="a" * 64, condition_seed=198304)
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


def load_geometry():
    filename = ROOT / "assets/robot_testtube_pour/scripts/mouth_pivot_lift_geometry.py"
    expected = "6f5ee0f694958ede3f111b1d27134c6571c4d7ef0ce5dae96425c4829638bfa9"
    if hashlib.sha256(filename.read_bytes()).hexdigest() != expected:
        raise ValueError("CPU constructor proof requires the frozen actual geometry")
    spec = importlib.util.spec_from_file_location("coarse40_cpu_frozen_geometry", filename)
    geometry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(geometry)
    return geometry.load_geometry(geometry.DEFAULT_CACHE_FILE, geometry.DEFAULT_CACHE_SHA256)


class Contract(unittest.TestCase):
    def test_exact40ml_original_clock_and_enabled_two_cycle_policy(self):
        profile = control.CorrectionPolicy()
        self.assertEqual(profile.target_ml, 40.)
        self.assertTrue(profile.enable_retry)
        self.assertEqual(profile.maximum_cycles, 2)
        for target in (20., 30., 40.01, True, float("nan")):
            with self.assertRaises(ValueError):
                control.CorrectionPolicy(target_ml=target)
        for deadline in (5., 23.9, 24.1, True):
            with self.assertRaises(ValueError):
                control.CorrectionPolicy(deadline_s=deadline)
        for enabled in (False, 1, None):
            with self.assertRaises(ValueError):
                control.CorrectionPolicy(enable_retry=enabled)

    def test_only_three_finite_coarse_speeds_are_accepted(self):
        for speed in (25.6, 25.2, 24.8):
            profile = control.CorrectionPolicy(coarse_dose_tilt_deg_s=speed)
            self.assertEqual(profile.coarse["dose_tilt_deg_s"], speed)
            self.assertEqual(profile.params["coarse_dose_tilt_deg_s"], speed)
        for speed in (25.4, 25.5, 24., 25.600000001, True, None, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                control.CorrectionPolicy(coarse_dose_tilt_deg_s=speed)

    def test_full_coarse_dict_must_match_scalar_and_other_five_constants(self):
        values = {**control.FIXED_COARSE, "dose_tilt_deg_s": 25.2}
        profile = control.CorrectionPolicy(coarse=values, coarse_dose_tilt_deg_s=25.2)
        values["stop_bias_ml"] = 9.
        self.assertEqual(profile.coarse["stop_bias_ml"], 10.)
        with self.assertRaises(ValueError):
            control.CorrectionPolicy(coarse={**control.FIXED_COARSE, "dose_tilt_deg_s": 25.2})
        for key in set(control.FIXED_COARSE) - {"dose_tilt_deg_s"}:
            changed = {**control.FIXED_COARSE, key: control.FIXED_COARSE[key] + .01}
            with self.assertRaises(ValueError):
                control.CorrectionPolicy(coarse=changed)
        changed = {**control.FIXED_COARSE, "undeclared": 1.}
        with self.assertRaises(ValueError):
            control.CorrectionPolicy(coarse=changed)

    def test_parameter_difference_is_only_actual_coarse_speed_and_alias(self):
        baseline = control.CorrectionPolicy(condition_seed=198304)
        for speed in (25.2, 24.8):
            candidate = control.CorrectionPolicy(coarse_dose_tilt_deg_s=speed, condition_seed=198304)
            self.assertEqual({key for key in baseline.coarse if baseline.coarse[key] != candidate.coarse[key]},
                             {"dose_tilt_deg_s"})
            self.assertEqual({key for key in baseline.params if baseline.params[key] != candidate.params[key]},
                             {"dose_tilt_deg_s", "coarse_dose_tilt_deg_s"})

    def test_first_retry_and_every_fine_motion_parameter_stay_fixed(self):
        expected = {"approach_tilt_deg": 87., "first_approach_tilt_deg": 87., "retry_approach_tilt_deg": 87.2,
            "approach_tilt_deg_s": 6., "action_delta_deg": .03, "action_tilt_deg_s": .6,
            "hold_seconds": .1, "backoff_delta_deg": .5, "backoff_tilt_deg_s": 4.,
            "anchor_return_tilt_deg_s": 6., "quiet_seconds": .25, "maximum_measurement_hold_s": 1.}
        for speed in control.COARSE_DOSE_SPEEDS_DEG_S:
            profile = control.CorrectionPolicy(coarse_dose_tilt_deg_s=speed)
            for key, value in expected.items():
                self.assertEqual(profile.params[key], value)
        for first in (86.95, 87.05, 87.1, True, None):
            with self.assertRaises(ValueError):
                control.CorrectionPolicy(first_approach_tilt_deg=first)
        for retry in (87., 87.1, 87.3, True, None):
            with self.assertRaises(ValueError):
                control.CorrectionPolicy(retry_approach_tilt_deg=retry)

    def test_source_seed_profile_and_one_factor_flags_are_explicit(self):
        instance, _ = fixture(25.2)
        metadata = instance._metadata()
        self.assertEqual(metadata["coarse40_condition_seed"], 198304)
        self.assertTrue(metadata["condition_seed_provided_by_runner"])
        self.assertEqual(metadata["coarse40_controller_source_sha256"], control._SOURCE_SHA256)
        self.assertTrue(metadata["only_coarse_dose_speed_varies"])
        self.assertFalse(metadata["only_first_angle_varies"])
        self.assertFalse(metadata["all_motion_speeds_fixed"])
        self.assertTrue(metadata["all_other_motion_speeds_fixed"])
        self.assertEqual(metadata["coarse_dose_tilt_deg_s"], 25.2)
        self.assertNotIn("first40_controller_source_sha256", metadata)
        for seed in (True, -1, 1.5, 2 ** 31):
            with self.assertRaises(ValueError):
                control.CorrectionPolicy(condition_seed=seed)
        with self.assertRaises(ValueError):
            control.CorrectionPolicy(profile_id="missing_raw_sha")

    def test_actual_closed_dose_and_original_strict_criterion_are_separate(self):
        for volume in (39.5, 40.5):
            self.assertTrue(control.in_task_band(40., volume))
            self.assertFalse(control.cycle.actual_dose_in_band(40., volume))
        self.assertFalse(control.in_task_band(40., math.nextafter(40.5, math.inf)))
        self.assertFalse(control.in_task_band(40., math.nextafter(39.5, -math.inf)))

    def test_actual_first_and_retry_fine_actions_do_not_depend_on_coarse_speed(self):
        actions = []
        for speed in control.COARSE_DOSE_SPEEDS_DEG_S:
            instance, status = fixture(speed, count=0)
            first = instance._choose_cycle_action(3., status)
            self.assertAlmostEqual(first["approach_delta_deg"] + 78., 87.)
            instance._cycle_count = 1
            second = instance._choose_cycle_action(10., status)
            self.assertAlmostEqual(second["approach_delta_deg"] + 78., 87.2)
            self.assertEqual(instance._retry_decisions[0]["declared_coarse_dose_tilt_deg_s"], speed)
            actions.append((first, second))
        self.assertEqual(actions[0], actions[1])
        self.assertEqual(actions[0], actions[2])

    def test_actual_in_band_overdose_and_unsettled_measurements_skip_retry(self):
        for volume in (39.5, 40.5, 40.527297):
            instance, status = fixture(received=volume)
            self.assertIsNone(instance._choose_cycle_action(11., status))
            self.assertEqual(instance._retry_decisions, [])
        instance, status = fixture(received=39.4, quiet=False)
        self.assertIsNone(instance._choose_cycle_action(11., status))

    def test_current_spill_and_full_home_physical_time_guards_are_unchanged(self):
        for peak in (None, 2):
            instance, status = fixture(peak=peak)
            self.assertIsNone(instance._choose_cycle_action(11., status))
            self.assertFalse(instance._retry_decisions[0]["spill_guard_passed"])
        instance, status = fixture(budget=13.7)
        self.assertIsNone(instance._choose_cycle_action(12.7, status))
        self.assertFalse(instance._retry_decisions[0]["full_home_budget_fits"])
        instance, status = fixture(budget=10., peak=1)
        self.assertIsNotNone(instance._choose_cycle_action(14., status))

    def test_retry_intent_is_not_execution_and_cycle_limit_stays_two(self):
        instance, status = fixture()
        self.assertIsNotNone(instance._choose_cycle_action(11., status))
        self.assertEqual(instance._cycle_count, 1)
        self.assertEqual(instance.response_events, [])
        self.assertEqual(instance.logged_intents[0][0], "coarse40_conditional_second_cycle_decision")
        self.assertTrue(instance._retry_decisions[0]["decision_is_intent_not_execution"])
        self.assertTrue(instance._retry_decisions[0]["budget_is_robot_motion_not_predicted_water_dose"])
        instance, status = fixture(count=2)
        self.assertIsNone(instance._choose_cycle_action(11., status))

    def test_actual_geometry_constructor_cpu_proof_and_requested_coarse_rate(self):
        geometry = load_geometry()
        home = geometry.joint_positions(0.)
        observation = {"target_ml": 40., "received_ml": 28., "inflight_ml": .6, "flow_ml_s": 3.}
        action_reference = None
        for speed in control.COARSE_DOSE_SPEEDS_DEG_S:
            profile = control.CorrectionPolicy(coarse_dose_tilt_deg_s=speed, condition_seed=198304)
            controller = control.TargetPourController(geometry, target_ml=40., spacing=.0027, policy=profile)
            self.assertEqual(controller.status["actual_target_ml"], 40.)
            self.assertEqual(controller.status["coarse_dose_tilt_deg_s"], speed)
            self.assertEqual(controller.source_time(0.), 0.)
            self.assertTrue(np.array_equal(geometry.joint_positions(controller.source_time(0.)), home))
            self.assertTrue(np.isfinite(home).all())
            action = controller.policy.act(observation)
            self.assertEqual(action["requested_tilt_deg_s"], speed)
            same = {key: value for key, value in action.items() if key != "requested_tilt_deg_s"}
            if action_reference is None:
                action_reference = same
            self.assertEqual(same, action_reference)
        with self.assertRaises(ValueError):
            control.TargetPourController(geometry, spacing=.0035, policy=control.CorrectionPolicy())

    def test_baseline_cpu_action_matches_frozen_v1_first87_profile(self):
        filename = ROOT / "projects/pour_policy_first40/first_control.py"
        self.assertEqual(hashlib.sha256(filename.read_bytes()).hexdigest(),
            "3a9f84bec0a31e77fa034d3ea95d6dda28e253827a5d88d4153213bb0f1b9299")
        spec = importlib.util.spec_from_file_location("coarse40_cpu_original_v1", filename)
        old = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(old)
        geometry = load_geometry()
        baseline = control.TargetPourController(geometry, policy=control.CorrectionPolicy(condition_seed=198304))
        prior = old.TargetPourController(geometry, policy=old.CorrectionPolicy(first_approach_tilt_deg=87., condition_seed=198304))
        observation = {"target_ml": 40., "received_ml": 29., "inflight_ml": 1., "flow_ml_s": 2.}
        self.assertEqual(baseline.policy.act(observation), prior.policy.act(observation))
        self.assertEqual(baseline.profile.coarse, prior.profile.coarse)
        for key in ("approach_tilt_deg", "approach_tilt_deg_s", "action_delta_deg", "action_tilt_deg_s",
                    "hold_seconds", "backoff_delta_deg", "backoff_tilt_deg_s", "anchor_return_tilt_deg_s",
                    "quiet_seconds", "maximum_measurement_hold_s"):
            self.assertEqual(baseline.profile.params[key], prior.profile.params[key])


if __name__ == "__main__":
    unittest.main()
