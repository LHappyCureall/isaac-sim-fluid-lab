"""CPU-only counterexamples for a single first-angle treatment and known replay."""
import ast
import copy
import inspect
import json
import math
from pathlib import Path
import unittest
import tempfile
from unittest.mock import patch

import audit_first_run as audit
import audit_first_experiment as overall


def score_row(name, dose=40., seconds=14., valid=True, spill=0):
    result = {"target_ml": 40., "received_ml": dose, "physical_time_s": seconds}
    strict = {"valid_evidence": valid, "full_home_within_deadline": seconds <= 24.,
        "spill_passed": spill <= 1, "no_late_dose_change": True, "tolerance_ml": .04,
        "maximum_all_label_spilled_particles": spill, "dose_passed": False, "feasible": False}
    return {"candidate_id": name, "profile_key": name, "seed": audit.KNOWN_SEED,
        "role": "candidate", "result": result,
        "refine_assessment": audit.tolerance_assessment(result, strict, .5)}


class Contracts(unittest.TestCase):
    def test_half_ml_closed_actual_goal_does_not_change_original_strict_band(self):
        old = {"valid_evidence": True, "full_home_within_deadline": True,
               "spill_passed": True, "no_late_dose_change": True, "tolerance_ml": .04}
        for amount in (39.5, 40.5):
            self.assertTrue(audit.tolerance_assessment({"target_ml": 40., "received_ml": amount}, old, .5)["feasible"])
        self.assertFalse(audit.tolerance_assessment({"target_ml": 40., "received_ml": math.nextafter(40.5, math.inf)}, old, .5)["dose_passed"])
        self.assertEqual(old["tolerance_ml"], .04)

    def test_feasible_ranks_time_only_then_declared_grid_order(self):
        rows = [score_row(name, dose=40.01 if index == 1 else 40.4) for index, name in enumerate(audit.CANDIDATES)]
        self.assertEqual(audit.choose_candidate(rows), "first_8700")
        rows[2]["result"]["physical_time_s"] = 13.5
        self.assertEqual(audit.choose_candidate(rows), "first_8695")

    def test_infeasible_dose_error_precedes_spill_and_invalid_is_not_an_elite(self):
        rows = [score_row("first_8700", dose=30., spill=0), score_row("first_8705", dose=39.4, spill=1),
                score_row("first_8695", valid=False)]
        self.assertEqual(audit.choose_candidate(rows), "first_8705")
        for row in rows:
            row["refine_assessment"]["valid_evidence"] = False
        with self.assertRaises(audit.AuditError):
            audit.choose_candidate(rows)

    def test_baseline_and_selected_known_case_are_not_independent_conditions(self):
        profiles = {key: {"enable_retry": True, "first_approach_tilt_deg": angle, "retry_approach_tilt_deg": 87.2}
                    for key, angle in audit.FIRST_ANGLES.items()}
        cases = [{"case_id": name, "target_ml": 40., "seed": audit.KNOWN_SEED, "split": "development",
            "role": "baseline" if name == "baseline" else "candidate", "candidate_id": name,
            "profile_key": name, "policy": profile} for name, profile in profiles.items()]
        audit.validate_schedule(cases, "calibration", profiles)
        self.assertNotEqual(profiles["first_8695"]["retry_approach_tilt_deg"] - profiles["first_8695"]["first_approach_tilt_deg"], .1)
        changed = copy.deepcopy(cases)
        changed[1]["profile_key"] = "first_8695"
        with self.assertRaises(audit.AuditError):
            audit.validate_schedule(changed, "calibration", profiles)
        changed = copy.deepcopy(profiles)
        changed["baseline"]["enable_retry"] = False
        with self.assertRaises(audit.AuditError):
            audit.validate_profiles(changed)

    def test_numerical_physics_and_endpoint_binders_remain_pinned(self):
        parent, implementation = audit.load_physical_audit()
        self.assertIs(implementation.__code__, parent.audit_episode.__code__)
        changed = {key for key in parent.audit_episode.__globals__
                   if implementation.__globals__.get(key) != parent.audit_episode.__globals__[key]}
        self.assertEqual(changed, {"CONTROLLER_SOURCE", "COLLECTOR_SOURCE", "CYCLE_KINDS", "cycle_bindings"})
        self.assertIs(implementation.__globals__["cycle_bindings"], parent.parent.cycle_bindings)

    def test_budget_math_is_identical_to_the_previous_auditor(self):
        path = audit.ROOT / "projects/pour_policy_retry20/audit_retry_run.py"
        self.assertEqual(audit.sha(path), "f10e0d87cb2e56d18af7c727a4967e25090d8e4ef3bb06fcb966c2ddf8a916fc")
        prior = ast.parse(path.read_text(encoding="utf8"))
        before = next(node for node in prior.body if isinstance(node, ast.ClassDef) and node.name == "ConservativeBudget")
        after = ast.parse(inspect.getsource(audit.ConservativeBudget)).body[0]
        self.assertEqual(ast.dump(before, include_attributes=False), ast.dump(after, include_attributes=False))

    def test_actual_known_baseline_and_protected42_are_byte_bound(self):
        run = audit.ROOT / "droplet_out/pour_policy_first40/first40_001/calibration"
        if not (run / "completed/calibration_baseline_s197304.json").exists():
            self.skipTest("The real known-case baseline has not completed")
        parent, episode = audit.load_physical_audit()
        ref = parent.load_cpu_reference(audit.ROOT)
        plan = audit.read(run / "protocol.json")
        contract = audit.validate_experiment_plan(run.parent)
        old = audit.protected_history(run, audit.ROOT, plan, contract, ref)
        self.assertEqual(len(old), 42)
        row = audit.read(run / "completed/calibration_baseline_s197304.json")
        geometry = ref.Geometry(run, audit.ROOT, plan)
        evidence = episode(run, row, plan, geometry, ref)
        proof = audit.reference_equivalence(row, evidence, audit.ROOT, ref, contract)
        self.assertTrue(proof["verified"])
        evidence["raw_steps"][1]["solver_substeps"] += 1
        changed = audit.reference_equivalence(row, evidence, audit.ROOT, ref, contract)
        self.assertFalse(changed["verified"])

    def test_case_seed_is_actual_runner_context_not_an_implicit_angle_selector(self):
        path = audit.ROOT / "droplet_out/pour_policy_first40/first40_001/calibration/completed/calibration_baseline_s197304.json"
        if not path.exists():
            self.skipTest("No completed actual baseline yet")
        row = audit.read(path)
        status = copy.deepcopy(row["status"])
        audit.validate_retry_status(status, row["declared_policy"], audit.CONTROLLER_SHA, 40.,
            status["retry_profile_id"], status["retry_profile_sha256"], row["seed"])
        status["first40_condition_seed"] += 1
        with self.assertRaises(audit.AuditError):
            audit.validate_retry_status(status, row["declared_policy"], audit.CONTROLLER_SHA, 40.,
                status["retry_profile_id"], status["retry_profile_sha256"], row["seed"])

    def test_incomplete_gate_precedes_checkpoint_or_fresh_test_journal_reads(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as folder:
            with self.assertRaisesRegex(ValueError, "before final audit"):
                overall.complete_gate(Path(folder), ["calibration", "validation", "test"], audit)

    def test_development_stop_cannot_claim_a_passing_gate_or_hide_actual_tests(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as folder:
            path = Path(folder)
            names = ["calibration"]
            receipt = {"kind": "first40_finalization_v1", "outcome": "calibration_gate_failed",
                "actual_stages": names, "independent_test_cases": 0, "no_retune_or_baseline_fallback": True}
            (path / "finalization.json").write_text(json.dumps(receipt), encoding="utf8")
            self.assertEqual(overall.stop_receipt(path, names, False)["outcome"], "calibration_gate_failed")
            with self.assertRaises(ValueError):
                overall.stop_receipt(path, names, True)
            (path / "test").mkdir()
            with self.assertRaises(ValueError):
                overall.stop_receipt(path, names, False)

    def test_actual_selection_is_calibration_time_ranking_and_fixed_retry_only(self):
        experiment = audit.ROOT / "droplet_out/pour_policy_first40/first40_001"
        if not (experiment / "calibration_selection.json").exists():
            self.skipTest("Actual selection has not been frozen yet")
        cached = overall.read(experiment / "calibration/audit.json")
        report = dict(cached)
        report["rows"] = [json.loads(line) for line in (experiment / "calibration/episodes.jsonl").read_text(encoding="utf8").splitlines()]
        selected = overall.verify_selection(experiment, report, cached, audit, audit.ROOT)
        self.assertEqual(selected["selected_profile_key"], "first_8700")
        altered = copy.deepcopy(selected)
        altered["retry_approach_tilt_deg"] = 87.1
        original_read = overall.read
        def changed_read(path):
            return altered if Path(path) == experiment / "calibration_selection.json" else original_read(path)
        with patch.object(overall, "read", side_effect=changed_read):
            with self.assertRaises(ValueError):
                overall.verify_selection(experiment, report, cached, audit, audit.ROOT)

    def test_coarse_prefix_comparison_stops_before_the_deliberate_first_angle_difference(self):
        experiment = audit.ROOT / "droplet_out/pour_policy_first40/first40_001"
        if not (experiment / "calibration/audit.json").exists():
            self.skipTest("Actual calibration has not completed")
        rows = [json.loads(line) for line in (experiment / "calibration/episodes.jsonl").read_text(encoding="utf8").splitlines()]
        physical, _ = audit.load_physical_audit()
        ref = physical.load_cpu_reference(audit.ROOT)
        baseline = next(row for row in rows if row["role"] == "baseline")
        for candidate in (row for row in rows if row["role"] == "candidate"):
            proof = overall.coarse_prefix_pair(baseline, candidate, experiment / "calibration", ref)
            self.assertTrue(proof["verified"])
            self.assertTrue(proof["whole_first_cycle_equality_is_not_required"])
            self.assertNotEqual(candidate["result"]["received_ml"], baseline["result"]["received_ml"])


if __name__ == "__main__":
    unittest.main()
