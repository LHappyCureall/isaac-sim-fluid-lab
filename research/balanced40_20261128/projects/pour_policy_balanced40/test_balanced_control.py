"""CPU-only balanced40 action contracts; no liquid steps or spill claims."""

import copy
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
import balanced_control as control
import run_balanced_test as runner
import task_score


def fixture(first=87., *, count=1, peak=0, received=39., quiet=True, budget=10.):
    instance = object.__new__(control.TargetPourController)
    instance.profile = control.CorrectionPolicy(first_approach_tilt_deg=first,
        profile_id="cpu_fixture", profile_sha256="a" * 64, condition_seed=198403)
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
    spec = importlib.util.spec_from_file_location("balanced40_cpu_frozen_geometry", filename)
    geometry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(geometry)
    return geometry.load_geometry(geometry.DEFAULT_CACHE_FILE, geometry.DEFAULT_CACHE_SHA256)


def matrix_fixture(stage):
    angles = {"baseline": 87., "first_8705": 87.05, "first_8710": 87.1}
    profiles = {key: {"enable_retry": True, "first_approach_tilt_deg": angle,
        "retry_approach_tilt_deg": 87.2, "coarse_dose_tilt_deg_s": 24.8} for key, angle in angles.items()}
    seeds = (197304, 198304, 198403) if stage == "calibration" else (
        (199101, 199102) if stage == "validation" else tuple(range(199301, 199311)))
    keys = tuple(profiles) if stage == "calibration" else ("baseline", "first_8705")
    rows = [{"case_id": f"synthetic_{stage}_{key}_{seed}", "target_ml": 40., "seed": seed,
        "profile_key": key, "candidate_id": key, "role": "baseline" if key == "baseline" else "candidate",
        "split": "test" if stage == "test" else "development", "policy": copy.deepcopy(profiles[key])}
        for seed in seeds for key in keys]
    return {"draft": False, "stage": stage, "profiles": profiles, "cases": rows}


class ControllerContract(unittest.TestCase):
    def test_exact40_target_original_clock_and_enabled_two_cycle_policy(self):
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

    def test_only_three_declared_first_angles_and_fixed_retry_are_accepted(self):
        for first in (87., 87.05, 87.1):
            self.assertEqual(control.CorrectionPolicy(first_approach_tilt_deg=first).first_approach_tilt_deg, first)
        for first in (86.95, 87.00001, 87.15, True, None, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                control.CorrectionPolicy(first_approach_tilt_deg=first)
        for retry in (87., 87.1, 87.3, True, None):
            with self.assertRaises(ValueError):
                control.CorrectionPolicy(retry_approach_tilt_deg=retry)

    def test_all_six_coarse_parameters_are_fixed_and_copied(self):
        original = dict(control.FIXED_COARSE)
        profile = control.CorrectionPolicy(coarse=original)
        original["stop_bias_ml"] = 9.
        self.assertEqual(profile.coarse, control.FIXED_COARSE)
        self.assertEqual(profile.coarse["dose_tilt_deg_s"], 24.8)
        for speed in (25.6, 25.2, 24.800001, True, None, float("nan")):
            with self.assertRaises(ValueError):
                control.CorrectionPolicy(coarse_dose_tilt_deg_s=speed)
        for key, value in control.FIXED_COARSE.items():
            with self.assertRaises(ValueError):
                control.CorrectionPolicy(coarse={**control.FIXED_COARSE, key: value + .01})

    def test_only_first_angle_parameters_differ_across_profiles(self):
        baseline = control.CorrectionPolicy(condition_seed=198403)
        for angle in (87.05, 87.1):
            candidate = control.CorrectionPolicy(first_approach_tilt_deg=angle, condition_seed=198403)
            self.assertEqual(baseline.coarse, candidate.coarse)
            self.assertEqual({key for key in baseline.params if baseline.params[key] != candidate.params[key]},
                {"approach_tilt_deg", "first_approach_tilt_deg"})

    def test_fine_motion_measurement_and_safety_parameters_stay_fixed(self):
        expected = {"retry_approach_tilt_deg": 87.2, "approach_tilt_deg_s": 6., "action_delta_deg": .03,
            "action_tilt_deg_s": .6, "hold_seconds": .1, "backoff_delta_deg": .5,
            "backoff_tilt_deg_s": 4., "anchor_return_tilt_deg_s": 6., "quiet_seconds": .25,
            "maximum_measurement_hold_s": 1., "maximum_cycles": 2}
        for first in control.FIRST_APPROACH_TILT_DEGREES:
            profile = control.CorrectionPolicy(first_approach_tilt_deg=first)
            for key, value in expected.items():
                self.assertEqual(profile.params[key], value)

    def test_source_seed_profile_and_first_angle_only_flags_are_explicit(self):
        instance, _ = fixture(87.05)
        metadata = instance._metadata()
        self.assertEqual(metadata["balanced40_condition_seed"], 198403)
        self.assertTrue(metadata["condition_seed_provided_by_runner"])
        self.assertEqual(metadata["balanced40_controller_source_sha256"], control._SOURCE_SHA256)
        self.assertTrue(metadata["only_first_angle_varies"])
        self.assertFalse(metadata["only_coarse_dose_speed_varies"])
        self.assertTrue(metadata["all_motion_speeds_fixed"])
        self.assertEqual(metadata["fixed_coarse_dose_tilt_deg_s"], 24.8)
        self.assertNotIn("coarse40_controller_source_sha256", metadata)
        for seed in (True, -1, 1.5, 2 ** 31):
            with self.assertRaises(ValueError):
                control.CorrectionPolicy(condition_seed=seed)
        with self.assertRaises(ValueError):
            control.CorrectionPolicy(profile_id="missing_raw_sha")

    def test_actual_first_angle_changes_but_retry_action_is_identical(self):
        first_actions, retry_actions = [], []
        for first in control.FIRST_APPROACH_TILT_DEGREES:
            instance, status = fixture(first, count=0)
            action = instance._choose_cycle_action(3., status)
            self.assertAlmostEqual(action["approach_delta_deg"] + 78., first)
            first_actions.append(action)
            instance._cycle_count = 1
            retry_actions.append(instance._choose_cycle_action(10., status))
        self.assertEqual(retry_actions[0], retry_actions[1])
        self.assertEqual(retry_actions[0], retry_actions[2])
        for action in first_actions:
            self.assertEqual({key: value for key, value in action.items() if key != "approach_delta_deg"},
                {key: value for key, value in first_actions[0].items() if key != "approach_delta_deg"})

    def test_closed_actual_dose_and_original_strict_bands_are_separate(self):
        for volume in (39.5, 40.5):
            self.assertTrue(control.in_task_band(40., volume))
            self.assertFalse(control.cycle.actual_dose_in_band(40., volume))
        self.assertFalse(control.in_task_band(40., math.nextafter(40.5, math.inf)))
        self.assertFalse(control.in_task_band(40., math.nextafter(39.5, -math.inf)))

    def test_in_band_overdose_and_unsettled_measurement_never_retry(self):
        for volume in (39.5, 40.5, 40.527297):
            instance, status = fixture(received=volume)
            self.assertIsNone(instance._choose_cycle_action(11., status))
            self.assertEqual(instance._retry_decisions, [])
        instance, status = fixture(received=39.4, quiet=False)
        self.assertIsNone(instance._choose_cycle_action(11., status))

    def test_current_spill_and_full_home_physical_budget_guards_are_unchanged(self):
        for peak in (None, 2):
            instance, status = fixture(peak=peak)
            self.assertIsNone(instance._choose_cycle_action(11., status))
            self.assertFalse(instance._retry_decisions[0]["spill_guard_passed"])
        instance, status = fixture(budget=13.8581)
        self.assertIsNone(instance._choose_cycle_action(12.716667, status))
        self.assertFalse(instance._retry_decisions[0]["full_home_budget_fits"])
        instance, status = fixture(budget=10., peak=1)
        self.assertIsNotNone(instance._choose_cycle_action(14., status))

    def test_retry_intent_is_not_execution_and_cycle_limit_is_two(self):
        instance, status = fixture()
        self.assertIsNotNone(instance._choose_cycle_action(11., status))
        self.assertEqual(instance._cycle_count, 1)
        self.assertEqual(instance.response_events, [])
        self.assertEqual(instance.logged_intents[0][0], "balanced40_conditional_second_cycle_decision")
        self.assertTrue(instance._retry_decisions[0]["decision_is_intent_not_execution"])
        self.assertEqual(instance._retry_decisions[0]["declared_coarse_dose_tilt_deg_s"], 24.8)
        instance, status = fixture(count=2)
        self.assertIsNone(instance._choose_cycle_action(11., status))

    def test_actual_geometry_constructor_and_requested_coarse_action_cpu_only(self):
        geometry = load_geometry()
        home = geometry.joint_positions(0.)
        observation = {"target_ml": 40., "received_ml": 28., "inflight_ml": .6, "flow_ml_s": 3.}
        actions = []
        for angle in control.FIRST_APPROACH_TILT_DEGREES:
            profile = control.CorrectionPolicy(first_approach_tilt_deg=angle, condition_seed=198403)
            controller = control.TargetPourController(geometry, policy=profile)
            self.assertEqual(controller.status["actual_target_ml"], 40.)
            self.assertEqual(controller.status["coarse_dose_tilt_deg_s"], 24.8)
            self.assertEqual(controller.source_time(0.), 0.)
            self.assertTrue(np.array_equal(geometry.joint_positions(controller.source_time(0.)), home))
            self.assertTrue(np.isfinite(home).all())
            action = controller.policy.act(observation)
            self.assertEqual(action["requested_tilt_deg_s"], 24.8)
            actions.append(action)
        self.assertEqual(actions[0], actions[1])
        self.assertEqual(actions[0], actions[2])
        with self.assertRaises(ValueError):
            control.TargetPourController(geometry, spacing=.0035, policy=control.CorrectionPolicy())

    def test_baseline_cpu_action_matches_frozen_coarse40_candidate24_8(self):
        filename = ROOT / "projects/pour_policy_coarse40/coarse_control.py"
        self.assertEqual(hashlib.sha256(filename.read_bytes()).hexdigest(),
            "67a2b35a363c757e6e30224701e527959557e480264638287bba86a6e8a8ad07")
        spec = importlib.util.spec_from_file_location("balanced40_cpu_old_coarse24_8", filename)
        old = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(old)
        geometry = load_geometry()
        baseline = control.TargetPourController(geometry, policy=control.CorrectionPolicy(condition_seed=198403))
        prior = old.TargetPourController(geometry,
            policy=old.CorrectionPolicy(coarse_dose_tilt_deg_s=24.8, condition_seed=198403))
        observation = {"target_ml": 40., "received_ml": 29., "inflight_ml": 1., "flow_ml_s": 2.}
        self.assertEqual(baseline.policy.act(observation), prior.policy.act(observation))
        self.assertEqual(baseline.profile.coarse, prior.profile.coarse)
        for key in ("approach_tilt_deg", "approach_tilt_deg_s", "action_delta_deg", "action_tilt_deg_s", "hold_seconds",
                "backoff_delta_deg", "backoff_tilt_deg_s", "anchor_return_tilt_deg_s", "quiet_seconds", "maximum_measurement_hold_s"):
            self.assertEqual(baseline.profile.params[key], prior.profile.params[key])


class RunnerAndScoreContract(unittest.TestCase):
    def test_exact_nine_four_twenty_conditions_and_four_key_profiles(self):
        for stage, count in (("calibration", 9), ("validation", 4), ("test", 20)):
            matrix = matrix_fixture(stage)
            self.assertEqual(len(runner.validate_matrix(matrix)), count)
            self.assertTrue(all(len(row["policy"]) == 4 for row in matrix["cases"]))
        for mutate in (lambda row: row["cases"].pop(),
                lambda row: row["cases"].__setitem__(1, copy.deepcopy(row["cases"][0])),
                lambda row: row["cases"][0].update(seed=198401),
                lambda row: row["cases"][0].update(split="test")):
            matrix = matrix_fixture("calibration"); mutate(matrix)
            with self.assertRaises(ValueError):
                runner.validate_matrix(matrix)

    def test_runner_rejects_speed_changes_retry_disabled_and_wrong_angle(self):
        for key, value in (("coarse_dose_tilt_deg_s", 25.6), ("enable_retry", False),
                ("first_approach_tilt_deg", 87.15), ("retry_approach_tilt_deg", 87.3)):
            matrix = matrix_fixture("calibration")
            matrix["profiles"]["first_8705"][key] = value
            with self.assertRaises(ValueError):
                runner.validate_matrix(matrix)

    def test_runner_factory_uses_actual_new_controller_and_matching_seed(self):
        runner.CONTROL = control
        runner.PROFILE = {"profile_id": "cpu_fixture"}; runner.PROFILE_SHA256 = "a" * 64
        runner.CURRENT_SEED = 198403
        geometry = load_geometry()
        policy = matrix_fixture("calibration")["profiles"]["first_8710"]
        controller = runner.TargetPourController(geometry, 40., .0027, policy)
        self.assertEqual(controller.profile.first_approach_tilt_deg, 87.1)
        self.assertEqual(controller.profile.coarse["dose_tilt_deg_s"], 24.8)
        self.assertEqual(controller.status["balanced40_condition_seed"], 198403)
        self.assertEqual(Path(sys.modules[type(controller).__module__].__file__).resolve(),
            Path(control.__file__).resolve())

    def test_unchanged_primary_easy_and_original_strict_gates_are_separate(self):
        strict = {"valid_evidence": True, "dose_passed": False, "full_home_within_deadline": True,
            "spill_passed": True, "no_late_dose_change": True, "tolerance_ml": .04, "feasible": False}
        result = {"target_ml": 40., "received_ml": 39.5}
        self.assertTrue(task_score.assess(result, strict)["feasible"])
        self.assertEqual(strict["tolerance_ml"], .04)
        self.assertFalse(strict["feasible"])
        result["received_ml"] = math.nextafter(39.5, -math.inf)
        self.assertFalse(task_score.assess(result, strict)["feasible"])
        self.assertTrue(task_score.easy_assess(result, strict)["feasible"])
        result["received_ml"] = 40.
        for key in ("valid_evidence", "full_home_within_deadline", "spill_passed", "no_late_dose_change"):
            self.assertFalse(task_score.assess(result, {**strict, key: False})["feasible"])


if __name__ == "__main__":
    unittest.main()
