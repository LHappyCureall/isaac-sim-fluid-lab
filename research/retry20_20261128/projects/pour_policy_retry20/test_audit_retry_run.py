"""CPU-only counterexamples for evidence, fixed retry guards and disjoint holdout."""
import copy
from datetime import datetime, timezone
import math
from pathlib import Path
from types import SimpleNamespace
import unittest
import tempfile

import audit_retry_run as audit
import audit_retry_experiment as overall


def old_physical_frame():
    parent, _ = audit.load_physical_audit()
    reference = parent.load_cpu_reference(audit.ROOT)
    path = audit.ROOT / "droplet_out/pour_policy_refine20/refine20_001/test"
    row = audit.read(path / "completed/test_candidate_s195301.json")
    endpoint = next(event for event in row["response_events"] if event["type"] == "response_cycle_complete")
    records, _ = reference.gz(path / "traces/test_candidate_s195301.steps.jsonl.gz")
    time_s = endpoint["physical_time_s"]
    step = next(item for item in records if item["type"] == "step" and item["physical_time_s"] == time_s)
    detail = next(item for item in records if item["type"] == "direct_step_evidence" and item["physical_time_s"] == time_s)
    return parent, reference, path, row, endpoint, step, detail


def decision_fixture():
    parent, reference, path, row, event, step, detail = old_physical_frame()
    time_s, received = event["physical_time_s"], event["post_observation"]["received_ml"]
    budget = 11.137543798633804
    decision = {"physical_time_s": time_s, "completed_cycles_before_decision": 1,
        "actual_received_ml": received, "actual_remaining_ml": step["truth"]["remaining_ml"],
        "actual_deficit_ml": 20. - received, "actual_settled_measurement": True,
        "deficit_exceeds_half_ml": True, "observed_all_label_peak_spill_particles": 0,
        "spill_guard_passed": True, "declared_retry_approach_tilt_deg": 84.1,
        "conservative_full_home_budget_s": budget, "deadline_physical_seconds": 24.,
        "remaining_deadline_physical_seconds": 24. - time_s, "full_home_budget_fits": True,
        "retry_allowed": True, "reason": "actual_quiet_undershoot_retry_allowed",
        "decision_is_intent_not_execution": True, "actual_execution_evidence_event_type": "response_cycle_start",
        "budget_is_robot_motion_not_predicted_water_dose": True}
    observed = {"observation": event["post_observation"]}
    return parent, decision, observed, step, detail, budget


class Contracts(unittest.TestCase):
    def test_half_ml_is_closed_original_bands_unchanged(self):
        strict = {"valid_evidence": True, "dose_passed": False, "tolerance_ml": .02,
            "full_home_within_deadline": True, "spill_passed": True, "no_late_dose_change": True}
        for dose in (19.5, 20.5):
            result = {"target_ml": 20., "received_ml": dose}
            self.assertTrue(audit.tolerance_assessment(result, strict, .5)["feasible"])
        result["received_ml"] = math.nextafter(20.5, math.inf)
        self.assertFalse(audit.tolerance_assessment(result, strict, .5)["dose_passed"])
        self.assertEqual(strict["tolerance_ml"], .02)
        self.assertFalse(strict["dose_passed"])

    def test_pilot_is_seen_development_and_test_has_ten_distinct_new_paired_seeds(self):
        cases = [{"case_id": f"test_{role}_{seed}", "target_ml": 20., "seed": seed, "split": "test",
            "role": role, "policy": {"enable_retry": role == "candidate"}}
            for role in ("baseline", "candidate") for seed in audit.TEST]
        audit.validate_schedule(cases, "test")
        with self.assertRaises(audit.AuditError):
            audit.validate_schedule(cases[:-1], "test")
        cases[-1]["seed"] = audit.PILOT_SEED
        with self.assertRaises(audit.AuditError):
            audit.validate_schedule(cases, "test")

    def test_new_ownership_reuses_exact_frozen_physical_bytecode_and_cycle_binder(self):
        parent, implementation = audit.load_physical_audit()
        self.assertIs(implementation.__code__, parent.audit_episode.__code__)
        changed = {key for key in parent.audit_episode.__globals__
                   if implementation.__globals__.get(key) != parent.audit_episode.__globals__[key]}
        self.assertEqual(changed, {"CONTROLLER_SOURCE", "COLLECTOR_SOURCE", "CYCLE_KINDS", "cycle_bindings"})
        self.assertIs(implementation.__globals__["cycle_bindings"], parent.parent.cycle_bindings)

    def test_budget_independently_reconstructed_from_actual_frozen_cache(self):
        parent, reference, run, _, _, _, _ = old_physical_frame()
        geometry = reference.Geometry(run, audit.ROOT, audit.read(run / "protocol.json"))
        planner = audit.ConservativeBudget(geometry)
        self.assertAlmostEqual(planner.planned(84.), 11.043103996503062, places=10)
        self.assertAlmostEqual(planner.planned(84.1), 11.137543798633809, places=10)

    def test_seen_failed_actual_frame_allows_only_its_explicit_guarded_intent(self):
        parent, decision, event, step, detail, budget = decision_fixture()
        self.assertTrue(audit.check_decision(decision, event, step, detail, budget, parent.parent))
        for key, value in (("actual_deficit_ml", .4), ("observed_all_label_peak_spill_particles", 0.5),
                           ("declared_retry_approach_tilt_deg", 84.2), ("conservative_full_home_budget_s", 1.)):
            changed = {**decision, key: value}
            with self.assertRaises(parent.AuditError):
                audit.check_decision(changed, event, step, detail, budget, parent.parent)

    def test_unsettled_or_ten_particle_spill_cannot_use_valid_dose_as_a_retry_guard(self):
        parent, decision, event, step, detail, budget = decision_fixture()
        for mutate in (lambda changed: changed["controller_status"].update(settled_measurement=False),
                       lambda changed: changed.update(maximum_all_label_spilled_particles=10)):
            changed = copy.deepcopy(detail)
            mutate(changed)
            with self.assertRaises(parent.AuditError):
                audit.check_decision(decision, event, step, changed, budget, parent.parent)

    def test_plan_and_reference_bytes_are_explicitly_bound(self):
        path = audit.ROOT / "projects/pour_policy_retry20/experiment_plan_001.json"
        self.assertEqual(audit.sha(path), audit.PLAN_SHA)
        plan = audit.read(path)
        for name, digest in plan["pilot_reference_sha256"].items():
            self.assertEqual(audit.sha(audit.ROOT / name), digest)

    def test_only_zero_time_inherited_counter_is_excluded_not_a_real_step(self):
        parent, reference, _, _, _, _, _ = old_physical_frame()
        run = audit.ROOT / "droplet_out/pour_policy_retry20/retry20_001/pilot"
        row = audit.read(run / "completed/pilot_known_s195301.json")
        records, _ = reference.gz(run / "traces/pilot_known_s195301.steps.jsonl.gz")
        report = {"raw_steps": [item for item in records if item["type"] == "step"],
                  "raw_direct": [item for item in records if item["type"] == "direct_step_evidence"]}
        plan = audit.read(run.parent / "experiment_plan.json")
        actual = audit.reference_equivalence(row, report, audit.ROOT, reference, plan)
        self.assertTrue(actual["first_cycle_water_motion_partitions_exactly_equal"])
        self.assertEqual(actual["zero_time_inherited_last_substeps"]["reference"], 6)
        self.assertEqual(actual["zero_time_inherited_last_substeps"]["pilot"], 4)
        report["raw_steps"][1]["solver_substeps"] += 1
        changed = audit.reference_equivalence(row, report, audit.ROOT, reference, plan)
        self.assertFalse(changed["first_cycle_water_motion_partitions_exactly_equal"])

    def test_incomplete_experiment_rejects_before_fresh_journals_or_policy_are_opened(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            with self.assertRaisesRegex(ValueError, "before final audit"):
                overall.complete_gate(Path(directory), audit)

    def test_fixed_strategy_must_freeze_after_pilot_audit_before_actual_test_reset(self):
        time = lambda second: datetime(2026, 11, 28, 0, 0, second, tzinfo=timezone.utc)
        overall.verify_freeze_times(time(1), time(2), time(3), time(4))
        for times in ((1, 2, 4, 4), (1, 3, 2, 4), (2, 2, 3, 4)):
            with self.assertRaises(ValueError):
                overall.verify_freeze_times(*(time(value) for value in times))


if __name__ == "__main__":
    unittest.main()
