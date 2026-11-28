"""CPU counterexamples for a separate task band and unchanged physical gates."""
import copy
from datetime import datetime, timezone
import math
from pathlib import Path
import tempfile
import unittest

import audit_refine_run as audit
import audit_refine_experiment as overall


def row(seed, dose=20., seconds=5., angle=84., role="baseline", spill=0, valid=True):
    strict = {"valid_evidence": valid, "dose_passed": False, "full_home_within_deadline": seconds <= 24,
        "maximum_all_label_spilled_particles": spill, "spill_passed": spill <= 1,
        "no_late_dose_change": True, "tolerance_ml": .02, "feasible": False}
    result = {"target_ml": 20., "received_ml": dose, "error_ml": dose - 20., "physical_time_s": seconds}
    return {"seed": seed, "role": role, "result": result, "strict_assessment": strict,
        "refine_assessment": audit.tolerance_assessment(result, strict, .5),
        "easy_assessment": audit.tolerance_assessment(result, strict, 1.),
        "declared_policy": {"enable_correction": role != "baseline", "approach_tilt_deg": angle}}


def status(enabled=True, angle=84.):
    return {"refine_controller_source_sha256": audit.CONTROLLER_SHA, "refine_enable_correction": enabled,
        "enable_correction": enabled, "refine_configured_approach_tilt_deg": angle, "refine_maximum_cycles": 1,
        "actual_target_ml": 20., "control_target_ml": 20., "target_ml": 20.,
        "control_target_offset_ml": 0., "prefix_target_offset_ml": 0., "actual_measurement_target_unchanged": True,
        "policy_params": {**audit.COARSE, **audit.MOTION, "approach_tilt_deg": angle,
            "enable_correction": enabled, "all_motion_speeds_fixed": True,
            "model_checkpoint_sha256": None, "model_action_menu": None},
        "response_model_used": False, "response_cycle_count": 0, "response_events": [],
        "received_ml": 19.2, "inflight_ml": 0., "flow_ml_s": 0.,
        "refine_actual_dose_in_closed_band": False, "original_strict_actual_dose_in_closed_band": False,
        "policy_action": {"coarse_action": {"predicted_final_ml": 19.2, "remaining_committed_ml": .8,
            "requested_tilt_deg_s": 25.6, "stop": True}}}


class Contracts(unittest.TestCase):
    def test_primary_is_closed_half_ml_without_relabeling_strict_or_easy(self):
        for dose in (19.5, 20.5):
            data = row(1, dose)
            self.assertTrue(data["refine_assessment"]["feasible"])
            self.assertFalse(data["strict_assessment"]["feasible"])
            self.assertEqual(data["strict_assessment"]["tolerance_ml"], .02)
        for dose in (math.nextafter(19.5, -math.inf), math.nextafter(20.5, math.inf)):
            data = row(1, dose)
            self.assertFalse(data["refine_assessment"]["dose_passed"])
            self.assertTrue(data["easy_assessment"]["dose_passed"])

    def test_tolerance_cannot_rescue_safety_or_missing_evidence(self):
        self.assertIsNone(audit.score([row(1, valid=False)]))
        self.assertFalse(row(1, spill=2)["refine_assessment"]["feasible"])
        self.assertFalse(row(1, seconds=24.01)["refine_assessment"]["feasible"])

    def test_all_invalid_and_less_spill_cannot_hide_ten_ml_shortfall(self):
        missing = [row(seed, 10.) for seed in audit.TRAIN]
        near = [row(seed, 19.4, spill=1) for seed in audit.TRAIN]
        self.assertLess(audit.score(near), audit.score(missing))
        data = [row(seed, valid=False, angle=angle, role="angle_" + format(angle, "g"))
                for angle in audit.ANGLES for seed in audit.TRAIN]
        with self.assertRaises(audit.AuditError):
            audit.choose_angle(data)

    def test_training_and_validation_ties_keep_preregistered_order(self):
        data = [row(seed, 19.8, angle=angle, role="angle_" + format(angle, "g"))
                for angle in audit.ANGLES for seed in audit.TRAIN]
        self.assertEqual(audit.choose_angle(data), 84.)
        roles = [row(seed, 19.8, role=role) for role in ("baseline", "candidate") for seed in audit.VAL]
        self.assertEqual(audit.choose_validation_role(roles), "baseline")

    def test_schedule_preserves_engineering_and_holds_out_every_test_seed(self):
        cases = [{"case_id": f"eng_{index}", "target_ml": 20., "seed": 195001, "split": "development",
            "role": "angle_" + format(angle, "g"),
            "policy": {"enable_correction": True, "approach_tilt_deg": angle}}
            for index, angle in enumerate(audit.ANGLES)]
        audit.validate_schedule(cases, "engineering")
        cases[-1]["seed"] = audit.TEST[0]
        with self.assertRaises(audit.AuditError):
            audit.validate_schedule(cases, "engineering")

    def test_physical_compiled_math_reused_and_original_whole_cycles_reenabled(self):
        parent, implementation = audit.load_physical_audit()
        self.assertIs(implementation.__code__, parent.audit_episode.__code__)
        changed = {key for key in parent.audit_episode.__globals__
                   if implementation.__globals__.get(key) != parent.audit_episode.__globals__[key]}
        self.assertEqual(changed, {"COLLECTOR_SOURCE", "CONTROLLER_SOURCE", "CYCLE_KINDS", "cycle_bindings"})
        self.assertIs(implementation.__globals__["cycle_bindings"], parent.parent.cycle_bindings)

    def test_new_profile_cannot_change_speed_offset_model_or_execute_baseline_cycle(self):
        policy = {"enable_correction": True, "approach_tilt_deg": 84.}
        raw = status()
        audit.validate_refine_status(raw, policy, audit.CONTROLLER_SHA)
        for mutate in (lambda s: s["policy_params"].update(action_tilt_deg_s=1.),
                       lambda s: s.update(control_target_ml=20.5),
                       lambda s: s.update(response_model_used=True)):
            changed = copy.deepcopy(raw)
            mutate(changed)
            with self.assertRaises(audit.AuditError):
                audit.validate_refine_status(changed, policy, audit.CONTROLLER_SHA)
        baseline = status(enabled=False)
        baseline["response_cycle_count"] = 1
        with self.assertRaises(audit.AuditError):
            audit.validate_refine_status(baseline, {**policy, "enable_correction": False}, audit.CONTROLLER_SHA)

    def test_preregistered_plan_unchanged_and_incomplete_cannot_expose_test(self):
        original = audit.ROOT / "projects/pour_policy_refine20/experiment_plan_001.json"
        self.assertEqual(audit.sha(original), audit.PLAN_SHA)
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            target = Path(directory)
            (target / "experiment_plan.json").write_bytes(original.read_bytes())
            audit.validate_experiment_plan(target)
            with self.assertRaisesRegex(ValueError, "before final audit"):
                overall.complete_gate(target, audit)

    def test_actual_reset_must_follow_model_freeze_without_equality_relaxation(self):
        parse = datetime.fromisoformat
        time = lambda second: datetime(2026, 11, 28, 0, 0, second, tzinfo=timezone.utc)
        models = {role: {"created_at_utc": time(2).isoformat()} for role in ("baseline", "candidate")}
        recommended = {"created_at_utc": time(5).isoformat()}
        overall.verify_freeze_times(models, recommended, time(1), time(4), time(3), time(6), parse)
        recommended["created_at_utc"] = time(6).isoformat()
        with self.assertRaisesRegex(ValueError, "precede actual test"):
            overall.verify_freeze_times(models, recommended, time(1), time(4), time(3), time(6), parse)


if __name__ == "__main__":
    unittest.main()
