"""CPU evidence counterexamples; never imports a solver or claims liquid success."""
import ast
import copy
import inspect
import json
import math
from pathlib import Path
from types import SimpleNamespace
import unittest
import tempfile
from unittest.mock import patch

import audit_retry_run as audit
import audit_retry_experiment as overall


PROFILES = {"30": {"first_approach_tilt_deg": 85.5, "retry_approach_tilt_deg": 85.6},
            "40": {"first_approach_tilt_deg": 87.2, "retry_approach_tilt_deg": 87.3}}


def guarded_frame(target=30., received=29., time_s=10., budget=12.):
    angle = PROFILES[format(target, "g")]["retry_approach_tilt_deg"]
    observed = {"received_ml": received, "tube_remaining_ml": 56.68704 - received,
        "tilt_deg": 78., "quiet_measured_seconds": .25, "settled_measurement": True,
        "source_airborne_particles": 0}
    step = {"physical_time_s": time_s,
            "truth": {"received_ml": received, "remaining_ml": 56.68704 - received}}
    detail = {"controller_status": dict(observed), "maximum_all_label_spilled_particles": 0,
              "partitions": {"tube_airborne": 0}}
    decision = {"physical_time_s": time_s, "completed_cycles_before_decision": 1,
        "actual_received_ml": received, "actual_remaining_ml": 56.68704 - received,
        "declared_target_ml": target, "target_profile_key": format(target, "g"),
        "target_profile_id": "CPU_fixture", "target_profile_sha256": "a" * 64,
        "actual_deficit_ml": target - received, "actual_settled_measurement": True,
        "deficit_exceeds_half_ml": True, "observed_all_label_peak_spill_particles": 0,
        "spill_guard_passed": True, "declared_retry_approach_tilt_deg": angle,
        "conservative_full_home_budget_s": budget, "deadline_physical_seconds": 24.,
        "remaining_deadline_physical_seconds": 24. - time_s, "full_home_budget_fits": time_s + budget <= 24.,
        "retry_allowed": time_s + budget <= 24.,
        "reason": "actual_quiet_target_undershoot_retry_allowed" if time_s + budget <= 24.
            else "retry3040_full_home_motion_budget_exceeds_deadline",
        "decision_is_intent_not_execution": True, "actual_execution_evidence_event_type": "response_cycle_start",
        "budget_is_robot_motion_not_predicted_water_dose": True}
    return decision, {"observation": observed}, step, detail, budget, target, angle


class Contracts(unittest.TestCase):
    def test_primary_boundary_is_actual_goal_relative_not_fixed20_or_percent(self):
        strict = {"valid_evidence": True, "full_home_within_deadline": True, "spill_passed": True,
                  "no_late_dose_change": True, "dose_passed": False, "tolerance_ml": .03}
        for target in audit.TARGETS:
            for received in (target - .5, target + .5):
                result = {"target_ml": target, "received_ml": received}
                self.assertTrue(audit.tolerance_assessment(result, strict, .5)["feasible"])
            result["received_ml"] = math.nextafter(target + .5, math.inf)
            self.assertFalse(audit.tolerance_assessment(result, strict, .5)["dose_passed"])
            self.assertTrue(audit.tolerance_assessment(result, strict, 1.)["dose_passed"])
        self.assertFalse(strict["dose_passed"])
        with self.assertRaises(audit.AuditError):
            audit.tolerance_assessment({"target_ml": 20., "received_ml": 20.}, strict, .5)

    def test_complete_schedule_includes_both_goals_roles_and_ten_new_seeds(self):
        cases = [{"case_id": f"test_{int(target)}_{role}_{seed}", "target_ml": target, "seed": seed,
            "split": "test", "role": role,
            "policy": {"enable_retry": role == "candidate", **PROFILES[format(target, "g")]}}
            for target in audit.TARGETS for role in ("baseline", "candidate") for seed in audit.TEST]
        audit.validate_schedule(cases, "test", PROFILES)
        with self.assertRaises(audit.AuditError):
            audit.validate_schedule(cases[:-1], "test", PROFILES)
        cases[-1]["seed"] = audit.PILOT_SEED
        with self.assertRaises(audit.AuditError):
            audit.validate_schedule(cases, "test", PROFILES)

    def test_goal_profile_cannot_silently_change_speed_gap_or_target_set(self):
        audit.validate_goal_profiles(PROFILES)
        changed = copy.deepcopy(PROFILES)
        changed["40"]["retry_approach_tilt_deg"] += .1
        with self.assertRaises(audit.AuditError):
            audit.validate_goal_profiles(changed)
        changed = copy.deepcopy(PROFILES)
        changed["30"]["speed_deg_s"] = 10.
        with self.assertRaises(audit.AuditError):
            audit.validate_goal_profiles(changed)

    def test_physical_math_and_original_cycle_endpoints_stay_bytecode_identical(self):
        parent, implementation = audit.load_physical_audit()
        self.assertIs(implementation.__code__, parent.audit_episode.__code__)
        changed = {key for key in parent.audit_episode.__globals__
                   if implementation.__globals__.get(key) != parent.audit_episode.__globals__[key]}
        self.assertEqual(changed, {"CONTROLLER_SOURCE", "COLLECTOR_SOURCE", "CYCLE_KINDS", "cycle_bindings"})
        self.assertIs(implementation.__globals__["cycle_bindings"], parent.parent.cycle_bindings)

    def test_fullhome_budget_is_inclusive_and_cannot_be_a_water_prediction(self):
        parent, _ = audit.load_physical_audit()
        for time_s, allowed in ((12., True), (math.nextafter(24., math.inf) - 12., False)):
            decision, event, step, detail, budget, target, angle = guarded_frame(time_s=time_s)
            result = audit.check_decision(decision, event, step, detail, budget, parent.parent,
                                          target, angle, "CPU_fixture", "a" * 64)
            self.assertIs(result, allowed)

    def test_profile_hash_and_target30_cannot_be_reused_to_license40_retry(self):
        parent, _ = audit.load_physical_audit()
        decision, event, step, detail, budget, target, angle = guarded_frame()
        for key, value in (("target_profile_sha256", "b" * 64), ("declared_target_ml", 40.),
                           ("actual_deficit_ml", .4), ("conservative_full_home_budget_s", 1.)):
            changed = {**decision, key: value}
            with self.assertRaises(ValueError):
                audit.check_decision(changed, event, step, detail, budget, parent.parent,
                                     target, angle, "CPU_fixture", "a" * 64)

    def test_primary_cannot_rescue_unsafe_or_unsettled_retry(self):
        parent, _ = audit.load_physical_audit()
        decision, event, step, detail, budget, target, angle = guarded_frame()
        changed = copy.deepcopy(detail)
        changed["maximum_all_label_spilled_particles"] = 2
        with self.assertRaises(ValueError):
            audit.check_decision(decision, event, step, changed, budget, parent.parent,
                                 target, angle, "CPU_fixture", "a" * 64)

    def test_bounded_plan_and_initial_external_profile_remain_unchanged(self):
        here = Path(__file__).resolve().parent
        plan = audit.read(here / "experiment_plan_001.json")
        self.assertEqual(audit.sha(here / "experiment_plan_001.json"), audit.PLAN_SHA)
        self.assertEqual(audit.sha(audit.ROOT / plan["initial_goal_profile_file"]), plan["initial_goal_profile_raw_sha256"])

    def test_optional_adjustment_only_changes_failed_goal_in_declared_direction(self):
        first = {"all_non_dose_safety_passed": True, "development_gate_passed": False,
            "goal_profiles": copy.deepcopy(PROFILES), "episode_reports": [
                {"target_ml": 30., "role": "candidate", "error_ml": -.16,
                 "refine_assessment": {"dose_passed": True}},
                {"target_ml": 40., "role": "candidate", "error_ml": .920957,
                 "refine_assessment": {"dose_passed": False}}]}
        adjusted = overall.adjustment_profiles(first)
        self.assertEqual(adjusted["30"], PROFILES["30"])
        self.assertEqual(adjusted["40"], {"first_approach_tilt_deg": 87.1, "retry_approach_tilt_deg": 87.2})
        self.assertEqual(first["goal_profiles"], PROFILES)
        first["all_non_dose_safety_passed"] = False
        with self.assertRaises(ValueError):
            overall.adjustment_profiles(first)

    def test_development_stop_is_explicit_and_cannot_hide_test_or_strategy(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as folder:
            root = Path(folder)
            receipt = {"kind": "retry3040_finalization_v1", "outcome": "development_gate_failed",
                "actual_development_stages": ["pilot_001", "pilot_002"], "independent_test_cases": 0,
                "bounds_exhausted_or_unsafe": True}
            (root / "finalization.json").write_text(json.dumps(receipt), encoding="utf8")
            self.assertEqual(overall.stop_receipt(root, receipt["actual_development_stages"], False)["outcome"],
                             "development_gate_failed")
            with self.assertRaises(ValueError):
                overall.stop_receipt(root, receipt["actual_development_stages"], True)
            (root / "test").mkdir()
            with self.assertRaises(ValueError):
                overall.stop_receipt(root, receipt["actual_development_stages"], False)

    def test_incomplete_required_stage_cannot_open_fresh_test_models_or_journals(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as folder:
            with self.assertRaisesRegex(ValueError, "before final audit"):
                overall.complete_gate(Path(folder), audit, ["pilot_001", "test"])

    def test_old_and_new_budget_math_are_identical(self):
        old_path = audit.ROOT / "projects/pour_policy_retry20/audit_retry_run.py"
        self.assertEqual(audit.sha(old_path), "f10e0d87cb2e56d18af7c727a4967e25090d8e4ef3bb06fcb966c2ddf8a916fc")
        original = ast.parse(old_path.read_text(encoding="utf8"))
        old_class = next(node for node in original.body if isinstance(node, ast.ClassDef) and node.name == "ConservativeBudget")
        new_class = ast.parse(inspect.getsource(audit.ConservativeBudget)).body[0]
        self.assertEqual(ast.dump(old_class, include_attributes=False), ast.dump(new_class, include_attributes=False))

    def test_actual_frozen_strategy_ledger_matches_and_fixed_motion_tamper_is_rejected(self):
        experiment = audit.ROOT / "droplet_out/pour_policy_retry3040/expansion_001"
        names = ["pilot_001", "pilot_002"]
        reports, cached = {}, {}
        for name in names:
            cached[name] = overall.read(experiment / name / "audit.json")
            reports[name] = dict(cached[name])
            reports[name]["rows"] = [json.loads(line) for line in
                (experiment / name / "episodes.jsonl").read_text(encoding="utf8").splitlines()]
        plan = audit.validate_experiment_plan(experiment)
        strategy = overall.verify_strategy(experiment, names, reports, cached, plan, audit, audit.ROOT)
        self.assertEqual(strategy["goal_profiles"], reports["pilot_002"]["goal_profiles"])
        original_read = overall.read
        altered = copy.deepcopy(strategy)
        altered["fixed_parameters"]["action_delta_deg"] = .5
        def changed_read(path):
            return altered if Path(path) == experiment / "fixed_strategy.json" else original_read(path)
        with patch.object(overall, "read", side_effect=changed_read):
            with self.assertRaisesRegex(ValueError, "Fixed motion"):
                overall.verify_strategy(experiment, names, reports, cached, plan, audit, audit.ROOT)


if __name__ == "__main__":
    unittest.main()
