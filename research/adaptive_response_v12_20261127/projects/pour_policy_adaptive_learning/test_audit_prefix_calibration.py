"""CPU tamper/ownership contracts only, never new physics acceptance."""

import ast
import copy
import inspect
import json
from pathlib import Path
import tempfile
import unittest

import audit_prefix_calibration as audit


def schedule():
    return [{"case_id": f"prefix_{index}_{seed}", "target_ml": target, "seed": seed,
             "split": "development", "policy": {"control_target_offset_ml": offset}}
            for index, (target, offset) in enumerate(audit.CONDITIONS) for seed in audit.SEEDS]


def status_fixture(target=25., offset=-.5):
    metadata = {"actual_target_ml": target, "control_target_ml": target + offset,
                "prefix_target_offset_ml": offset, "control_target_offset_ml": offset}
    status = {**metadata, "target_ml": target, "prefix_controller_source_sha256": "a" * 64,
              "policy_params": dict(audit.COARSE), "response_events": [],
              "response_cycle_count": 0, "response_model_used": False,
              "received_ml": 24., "inflight_ml": .1, "flow_ml_s": 1.,
              "tilt_deg": 78., "quiet_measured_seconds": .25, "settled_measurement": True}
    predicted = status["received_ml"] + status["inflight_ml"] + .26 * status["flow_ml_s"]
    action = {**metadata, "stop": predicted + 10. >= target + offset,
              "predicted_final_ml": predicted, "remaining_committed_ml": target + offset - 24.1,
              "requested_tilt_deg_s": 25.6}
    status["policy_action"] = {**metadata, "coarse_action": action}
    return status


def anchor_fixture():
    observation = {"physical_time_s": 4., "received_ml": 24., "tube_remaining_ml": 32.68704,
                   "tilt_deg": 78., "quiet_measured_seconds": .25,
                   "source_airborne_particles": 0, "settled_measurement": True}
    status = status_fixture()
    status["prefix_anchor_observation"] = observation
    row = {"target_ml": 25., "declared_policy": {"control_target_offset_ml": -.5},
           "result": {"complete": True}, "status": status}
    step = {"physical_time_s": 4., "truth": {"received_ml": 24., "remaining_ml": 32.68704}}
    detail = {"physical_time_s": 4., "controller_status": status,
              "partitions": {"tube_airborne": 0}}
    report = {"raw_steps": [step], "raw_direct": [detail]}
    plan = {"source_sha256": {audit.CONTROLLER_SOURCE: "a" * 64}}
    return row, report, plan


class PrefixAuditContracts(unittest.TestCase):
    def test_exact_twelve_development_conditions(self):
        audit.validate_schedule(schedule())
        self.assertEqual(len(schedule()), 12)

    def test_missing_condition_duplicate_identity_and_test_relabel_rejected(self):
        for mutate in (lambda rows: rows.pop(),
                       lambda rows: rows[-1].update(case_id=rows[0]["case_id"]),
                       lambda rows: rows[0].update(split="test"),
                       lambda rows: rows[0].update(seed=193103)):
            cases = schedule()
            mutate(cases)
            with self.assertRaises(audit.AuditError):
                audit.validate_schedule(cases)

    def test_offset_and_motion_parameter_expansion_rejected(self):
        cases = schedule()
        cases[0]["policy"]["control_target_offset_ml"] = -.75
        with self.assertRaises(audit.AuditError):
            audit.validate_schedule(cases)
        cases = schedule()
        cases[0]["policy"]["dose_tilt_deg_s"] = 40.
        with self.assertRaises(audit.AuditError):
            audit.validate_schedule(cases)
        cases = schedule()
        cases[0]["policy"]["control_target_offset_ml"] = True
        with self.assertRaises(audit.AuditError):
            audit.validate_schedule(cases)

    def test_internal_target_shift_preserves_actual_goal_and_original_formula(self):
        for target, offset in audit.CONDITIONS:
            status = status_fixture(target, offset)
            audit.validate_prefix_status(status, target, offset, "a" * 64)
            self.assertEqual(status["target_ml"], target)
            self.assertEqual(status["control_target_ml"], target + offset)

    def test_measurement_goal_or_control_alias_tamper_rejected(self):
        for mutate in (lambda s: s.update(target_ml=24.5),
                       lambda s: s.update(actual_target_ml=24.5),
                       lambda s: s.update(control_target_ml=25.),
                       lambda s: s["policy_action"].update(control_target_offset_ml=0.),
                       lambda s: s["policy_action"]["coarse_action"].update(actual_target_ml=24.5)):
            status = status_fixture()
            mutate(status)
            with self.assertRaises(audit.AuditError):
                audit.validate_prefix_status(status, 25., -.5, "a" * 64)

    def test_any_original_coarse_parameter_change_rejected(self):
        for key in audit.COARSE:
            status = status_fixture()
            status["policy_params"][key] += .01
            with self.subTest(key=key), self.assertRaises(audit.AuditError):
                audit.validate_prefix_status(status, 25., -.5, "a" * 64)

    def test_prefix_stop_and_prediction_are_independently_recomputed(self):
        for key, value in (("stop", False), ("predicted_final_ml", 24.),
                           ("remaining_committed_ml", 1.), ("requested_tilt_deg_s", 40.)):
            status = status_fixture()
            status["policy_action"]["coarse_action"][key] = value
            with self.subTest(key=key), self.assertRaises(audit.AuditError):
                audit.validate_prefix_status(status, 25., -.5, "a" * 64)

    def test_prefix_response_or_model_cycle_is_not_allowed(self):
        for change in ({"response_events": [{"type": "response_cycle_start"}]},
                       {"response_cycle_count": 1}, {"response_model_used": True}):
            status = {**status_fixture(), **change}
            with self.assertRaises(audit.AuditError):
                audit.validate_prefix_status(status, 25., -.5, "a" * 64)
        with self.assertRaises(audit.AuditError):
            audit.cycle_bindings({"response_events": [], "response_samples": [{}]}, [], [])

    def test_actual_anchor_crossbinds_raw_received_volume(self):
        row, report, plan = anchor_fixture()
        bound = audit.prefix_bindings(row, report, plan)
        self.assertEqual(bound["received_ml"], 24.)
        self.assertEqual(bound["actual_target_ml"], 25.)
        self.assertEqual(bound["control_target_ml"], 24.5)
        report["raw_steps"][0]["truth"]["received_ml"] = 25.
        with self.assertRaises(audit.AuditError):
            audit.prefix_bindings(row, report, plan)

    def test_anchor_must_be_actual_quiet_not_backdated_or_rewritten(self):
        for field, value in (("physical_time_s", 3.9), ("tilt_deg", 79.),
                             ("settled_measurement", False), ("source_airborne_particles", 1),
                             ("quiet_measured_seconds", .1)):
            row, report, plan = anchor_fixture()
            row["status"]["prefix_anchor_observation"][field] = value
            with self.subTest(field=field), self.assertRaises(audit.AuditError):
                audit.prefix_bindings(row, report, plan)
        row, report, plan = anchor_fixture()
        second = copy.deepcopy(report["raw_direct"][0])
        second["physical_time_s"] = 4.1
        second["controller_status"]["prefix_anchor_observation"]["received_ml"] += .019683
        report["raw_direct"].append(second)
        report["raw_steps"].append(copy.deepcopy(report["raw_steps"][0]))
        with self.assertRaises(audit.AuditError):
            audit.prefix_bindings(row, report, plan)

    def test_complete_gate_precedes_journal_read(self):
        with tempfile.TemporaryDirectory(prefix="_prefix_audit_contract_", dir=Path(__file__).resolve().parent) as folder:
            root = Path(folder)
            (root / "summary.json").write_text('{"complete":false}', encoding="utf8")
            (root / "job_state.json").write_text('{"state":"running","pid":17}', encoding="utf8")
            with self.assertRaisesRegex(audit.AuditError, "before journal"):
                audit.audit(root)

    def test_owned_sources_do_not_require_unloaded_learning_or_old_factory(self):
        names = set(audit.required_runtime_sources())
        self.assertIn(audit.CONTROLLER_SOURCE, names)
        self.assertIn(audit.COLLECTOR_SOURCE, names)
        self.assertNotIn(audit.HERE + "cycle_response_model.py", names)
        self.assertNotIn(audit.HERE + "collect_responses.py", names)

    def test_step_physics_mathematics_exactly_matches_pinned_parent(self):
        def normalized(function):
            tree = ast.parse(inspect.getsource(function))
            node = tree.body[0]
            node.body = [child for child in node.body if not (isinstance(child, ast.Assign)
                and any(isinstance(target, ast.Name) and target.id in
                        ("expected_controller", "expected_collector") for target in child.targets))]
            for child in ast.walk(node):
                if isinstance(child, ast.Dict):
                    indices = [i for i, key in enumerate(child.keys) if isinstance(key, ast.Constant)
                               and key.value in ("raw_steps", "raw_direct")]
                    for i in reversed(indices):
                        child.keys.pop(i)
                        child.values.pop(i)
            return ast.dump(node, include_attributes=False)
        self.assertEqual(normalized(audit.audit_episode), normalized(audit.parent.audit_episode))
        self.assertIs(audit.audit_post, audit.parent.audit_post)

    def test_independent_motion_clock_never_imports_controller_or_cuda(self):
        ref = audit.load_cpu_reference(audit.ROOT)
        status = {"response_motion_segment": [2., 1., 0., 0., 2., 3., "v10_C2_source_step"]}
        self.assertEqual(ref.commanded_u(status, 99., -99., 3., None), 2.)
        self.assertEqual(ref.commanded_u(status, 99., -99., 4., None), 3.)


if __name__ == "__main__":
    unittest.main()
