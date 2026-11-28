"""CPU counterexamples for the independent one-factor coarse-speed audit."""

import ast
import copy
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import audit_coarse_run as audit
import audit_coarse_experiment as overall


def row(candidate, seed, *, dose=40., elapsed=15., peak=0, valid=True, feasible=True):
    return {"candidate_id": candidate, "seed": seed,
        "refine_assessment": {"valid_evidence": valid, "feasible": feasible,
            "maximum_all_label_spilled_particles": peak},
        "result": {"received_ml": dose, "physical_time_s": elapsed}}


def profiles():
    return {key: {"enable_retry": True, "first_approach_tilt_deg": 87.,
        "retry_approach_tilt_deg": 87.2, "coarse_dose_tilt_deg_s": speed}
        for key, speed in audit.SPEEDS.items()}


class CoarseAuditTests(unittest.TestCase):
    def test_candidate_group_requires_both_seen_conditions(self):
        a = [row("coarse_252", seed) for seed in audit.KNOWN_SEEDS]
        self.assertEqual(audit.score_group(a), (0, 15.))
        with self.assertRaises(ValueError):
            audit.score_group(a[:1])
        with self.assertRaises(ValueError):
            audit.score_group([a[0], a[0]])

    def test_infeasible_water_error_precedes_reduced_spill(self):
        inaccurate = [row("coarse_252", seed, dose=30., peak=0, feasible=False)
                      for seed in audit.KNOWN_SEEDS]
        accurate = [row("coarse_248", seed, dose=39.8, peak=2, feasible=False)
                    for seed in audit.KNOWN_SEEDS]
        self.assertEqual(audit.choose_candidate(inaccurate + accurate), "coarse_248")

    def test_feasible_group_compares_only_mean_physical_time_and_declared_tie(self):
        a = [row("coarse_252", seed, dose=39.51, elapsed=15.) for seed in audit.KNOWN_SEEDS]
        b = [row("coarse_248", seed, dose=40., elapsed=16.) for seed in audit.KNOWN_SEEDS]
        self.assertEqual(audit.choose_candidate(a + b), "coarse_252")
        for item in b:
            item["result"]["physical_time_s"] = 15.
        self.assertEqual(audit.choose_candidate(list(reversed(a + b))), "coarse_252")

    def test_one_invalid_case_rejects_whole_candidate_but_preserves_no_elite_result(self):
        a = [row("coarse_252", seed) for seed in audit.KNOWN_SEEDS]
        b = [row("coarse_248", seed) for seed in audit.KNOWN_SEEDS]
        a[0]["refine_assessment"]["valid_evidence"] = False
        self.assertIsNone(audit.score_group(a))
        self.assertEqual(audit.choose_candidate(a + b), "coarse_248")
        b[0]["refine_assessment"]["valid_evidence"] = False
        self.assertIsNone(audit.choose_candidate(a + b))

    def test_nonfinite_values_never_enter_elite_ranking(self):
        for key in ("received_ml", "physical_time_s"):
            group = [row("coarse_252", seed) for seed in audit.KNOWN_SEEDS]
            group[0]["result"][key] = float("nan")
            self.assertIsNone(audit.score_group(group))

    def test_baseline_task_failure_is_not_invalid_physics(self):
        report = {"strict_assessment": {"valid_evidence": True,
            "spill_passed": False, "full_home_within_deadline": False,
            "no_late_dose_change": False, "feasible": False}}
        self.assertTrue(audit.baseline_valid_comparator(report))
        report["strict_assessment"]["valid_evidence"] = False
        self.assertFalse(audit.baseline_valid_comparator(report))

    def test_closed_primary_and_original_strict_are_distinct(self):
        strict = {"valid_evidence": True, "full_home_within_deadline": True,
            "spill_passed": True, "no_late_dose_change": True, "dose_passed": False}
        for value in (39.5, 40.5):
            self.assertTrue(audit.tolerance_assessment({"target_ml": 40., "received_ml": value}, strict, .5)["feasible"])
        for value in (39.5 - 1e-10, 40.5 + 1e-10):
            self.assertFalse(audit.tolerance_assessment({"target_ml": 40., "received_ml": value}, strict, .5)["feasible"])
        self.assertFalse(strict["dose_passed"])

    def test_speed_profiles_do_not_change_fine_angles_or_retry_mode(self):
        values = profiles()
        audit.validate_profiles(values)
        for key, replacement in (("first_approach_tilt_deg", 86.9),
                                 ("retry_approach_tilt_deg", 87.1), ("enable_retry", False)):
            bad = copy.deepcopy(values)
            bad["coarse_252"][key] = replacement
            with self.assertRaises(ValueError):
                audit.validate_profiles(bad)

    def test_schedule_needs_all_six_original_cases_and_exact_profile_lookup(self):
        values = profiles()
        cases = [{"case_id": f"calibration_{key}_{seed}", "target_ml": 40., "seed": seed,
            "split": "development", "profile_key": key, "candidate_id": key,
            "role": "baseline" if key == "baseline" else "candidate", "policy": value}
            for key, value in values.items() for seed in audit.KNOWN_SEEDS]
        audit.validate_schedule(cases, "calibration", values)
        changed = copy.deepcopy(cases)
        changed[-1]["seed"] = 198401
        with self.assertRaises(ValueError):
            audit.validate_schedule(changed, "calibration", values)

    def test_physical_audit_compiled_math_is_not_reimplemented(self):
        parent, wrapper = audit.load_physical_audit()
        self.assertIs(wrapper.__code__, parent.audit_episode.__code__)
        self.assertEqual(wrapper.__globals__["CONTROLLER_SOURCE"], audit.CONTROLLER)
        self.assertEqual(wrapper.__globals__["COLLECTOR_SOURCE"], audit.COLLECTOR)
        self.assertIs(wrapper.__globals__["cycle_bindings"], parent.parent.cycle_bindings)

    def test_robot_budget_formula_matches_frozen_prior_audit(self):
        prior = audit.ROOT / "projects/pour_policy_first40/audit_first_run.py"
        def body(path):
            tree = ast.parse(Path(path).read_text())
            return ast.dump(next(node for node in tree.body if isinstance(node, ast.ClassDef)
                                 and node.name == "ConservativeBudget"), include_attributes=False)
        self.assertEqual(body(prior), body(audit.__file__))

    def test_actual_controller_t0_schema_matches_independent_audit_all_speeds(self):
        # This uses the real frozen geometry and particle initial array, with
        # no fluid object, fluid step or GPU process. It catches ownership and
        # policy/status field drift before the first measured physical case.
        from test_coarse_control import control, load_geometry, np
        geometry = load_geometry()
        points, labels = geometry.initial_water_points(.0027)
        for speed in (25.6, 25.2, 24.8):
            policy = control.CorrectionPolicy(coarse_dose_tilt_deg_s=speed,
                condition_seed=198304, profile_id="cpu_audit_schema", profile_sha256="a" * 64)
            controller = control.TargetPourController(geometry, policy=policy)
            controller.set_observed_peak_spill_particles(0)
            status = controller.update(0., points, labels, np.zeros_like(points))
            audit.validate_retry_status(status, {"enable_retry": True, "first_approach_tilt_deg": 87.,
                "retry_approach_tilt_deg": 87.2, "coarse_dose_tilt_deg_s": speed},
                audit.CONTROLLER_SHA, 40., "cpu_audit_schema", "a" * 64, 198304)

    def test_incomplete_phases_are_rejected_before_private_journal_reads(self):
        with tempfile.TemporaryDirectory(dir=audit.ROOT / "_cache") as temporary:
            with patch.object(overall, "read", side_effect=AssertionError("No journal may be read")):
                with self.assertRaises(ValueError):
                    overall.complete_gate(Path(temporary), ["calibration", "validation", "test"], audit)

    def test_prepare_prefix_ends_before_speed_treatment_and_rejects_future_dense_time(self):
        def records(after_value):
            output = []
            for step, t in enumerate((0., 1.5, 1.5166666666666666)):
                q = 0. if t <= 1.5 else after_value
                output.append({"type": "step", "step": step, "physical_time_s": t,
                    "physics_dt_s": 1. / 60., "source_time": q,
                    "truth": {"received_ml": q}, "solver_substeps": 4})
                output.append({"type": "direct_step_evidence", "step": step, "physical_time_s": t,
                    "controller_status": {"phase": "prepare" if t == 0. else "coarse_dose", "tilt_deg": 78.},
                    "kinematics": {"q": q, "dense_240_hz_samples": [] if t == 0. else [{"time_s": t}]},
                    "partitions": {"tube": 2880}, "maximum_all_label_spilled_particles": 0})
            return output
        values = {"baseline": records(25.6), "candidate": records(25.2)}
        ref = types.SimpleNamespace(gz=lambda path: (values["baseline" if "baseline" in path.name else "candidate"], None))
        base = {"episode_id": "baseline", "seed": 197304}
        candidate = {"episode_id": "candidate", "seed": 197304, "candidate_id": "coarse_252"}
        proof = overall.prepare_prefix_pair(base, candidate, Path("unused"), ref)
        self.assertTrue(proof["verified"])
        self.assertEqual(proof["prepare_endpoint_physical_s"], 1.5)
        self.assertTrue(proof["post_coarse_anchor_equality_is_not_required"])
        values["candidate"][3]["kinematics"]["dense_240_hz_samples"][0]["time_s"] = 1.50001
        with self.assertRaises(ValueError):
            overall.prepare_prefix_pair(base, candidate, Path("unused"), ref)


if __name__ == "__main__":
    unittest.main()
