"""Synthetic CPU observation/motion contracts; not liquid training results."""

import copy
import importlib.util
import math
from pathlib import Path
import sys
import unittest

import test_response_control as fixture


HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("prefix_calibration_cpu_control", HERE / "prefix_control.py")
control = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = control
spec.loader.exec_module(control)
np, g, COARSE = fixture.np, fixture.g, fixture.COARSE


def new(target=25., offset=-.5):
    return control.TargetPourController(g, target, .0027,
        control.CorrectionPolicy(COARSE, control_target_offset_ml=offset))


class PrefixControlTests(unittest.TestCase):
    def test_finite_offsets_and_deadline_bounds(self):
        for offset in (-2., 0., 2.):
            self.assertEqual(new(offset=offset).profile.control_target_offset_ml, offset)
        for offset in (-2.001, 2.001, True, float("nan"), float("inf")):
            with self.subTest(offset=offset), self.assertRaises(ValueError):
                new(offset=offset)
        for deadline in (23., 25., True):
            with self.subTest(deadline=deadline), self.assertRaises(ValueError):
                control.CorrectionPolicy(COARSE, deadline_s=deadline)

    def test_all_six_original_coarse_parameters_are_fixed(self):
        for key in COARSE:
            with self.subTest(key=key), self.assertRaises(ValueError):
                control.CorrectionPolicy({**COARSE, key: COARSE[key] + .001})
        with self.assertRaises(ValueError):
            control.CorrectionPolicy({**COARSE, "unknown": 1.})
        policy = control.CorrectionPolicy(COARSE, control_target_offset_ml=1.)
        self.assertEqual(policy.coarse, COARSE)
        self.assertFalse(policy.enable_topup)
        self.assertIsNone(policy.model)

    def test_coarse_target_offset_does_not_change_original_observation(self):
        observation = {"target_ml": 25., "received_ml": 14., "inflight_ml": 0., "flow_ml_s": 0.}
        before = copy.deepcopy(observation)
        early = control.PrefixCoarsePolicy(COARSE, -1.).act(observation)
        unchanged = control.PrefixCoarsePolicy(COARSE, 0.).act(observation)
        self.assertTrue(early["stop"])
        self.assertFalse(unchanged["stop"])
        self.assertEqual(observation, before)
        self.assertEqual(early["actual_target_ml"], 25.)
        self.assertEqual(early["control_target_ml"], 24.)
        self.assertEqual(early["predicted_final_ml"], unchanged["predicted_final_ml"])

    def test_actual_measurement_goal_and_source_arrays_remain_unchanged(self):
        controller = new(target=25., offset=-1.)
        fixture.place_anchor(controller)
        status = fixture.observe(controller, 0., received=1200)
        self.assertEqual(controller.target_ml, 25.)
        self.assertEqual(status["target_ml"], 25.)
        self.assertEqual(status["actual_target_ml"], 25.)
        self.assertEqual(status["control_target_ml"], 24.)
        self.assertAlmostEqual(status["received_ml"], 1200 * controller.particle_ml)
        self.assertAlmostEqual(status["error_ml"], 1200 * controller.particle_ml - 25.)
        self.assertEqual(controller.policy.params, COARSE)
        self.assertEqual(status["policy_action"]["coarse_action"]["actual_target_ml"], 25.)

    def test_first_quiet_anchor_goes_directly_to_exact_home_without_cycles(self):
        controller = new(target=25., offset=-.5)
        fixture.place_anchor(controller)
        phases = []
        for frame in range(300):
            status = fixture.observe(controller, frame / 60, received=1200)
            phases.append(status["phase"])
            if status["done"]:
                break
        self.assertTrue(status["done"])
        self.assertTrue(status["home_static"])
        self.assertGreaterEqual(status["stable_frames"], 12)
        self.assertLessEqual(status["completed_physical_time_s"], 24.)
        self.assertEqual(controller.response_events, [])
        self.assertEqual(status["response_cycle_count"], 0)
        self.assertFalse(any("response" in phase or "fine" in phase for phase in phases))
        anchor = controller.prefix_anchor_observation
        self.assertTrue(anchor["settled_measurement"])
        self.assertAlmostEqual(anchor["tilt_deg"], 78.)
        self.assertEqual(anchor["actual_target_ml"], 25.)
        self.assertEqual(anchor["control_target_ml"], 24.5)
        self.assertEqual(anchor["source_velocity_measurement_basis"], "provided_actual_particle_qd")
        np.testing.assert_allclose(g.joint_positions(controller.source_time(frame / 60)), g.joint_positions(0.), atol=1e-10)
        self.assertEqual(len(status["response_motion_segment"]), 7)
        self.assertFalse(status["response_model_used"])

    def test_full_pregrasp_clock_and_readonly_spill_hook(self):
        controller = new(target=25., offset=-.5)
        controller.set_observed_peak_spill_particles(1)
        controller.set_observed_peak_spill_particles(0)
        self.assertEqual(controller._cycle_observed_peak_spill, 1)
        for frame in range(1441):
            status = fixture.observe(controller, frame / 60, received=1200)
            if status["done"]:
                break
        self.assertTrue(status["done"])
        self.assertEqual(status["actual_target_ml"], 25.)
        self.assertEqual(status["control_target_ml"], 24.5)
        self.assertEqual(controller.response_events, [])
        self.assertGreater(status["completed_physical_time_s"], 2.)
        self.assertGreaterEqual(status["stable_frames"], 12)

    def test_reset_restores_actual_goal_and_offset_adapter(self):
        controller = new(target=25., offset=1.5)
        controller.reset(40.)
        self.assertEqual(controller.target_ml, 40.)
        self.assertEqual(controller.status["actual_target_ml"], 40.)
        self.assertEqual(controller.status["control_target_ml"], 41.5)
        self.assertEqual(controller.policy.control_target_offset_ml, 1.5)
        self.assertIsNone(controller.prefix_anchor_observation)


if __name__ == "__main__":
    unittest.main()
