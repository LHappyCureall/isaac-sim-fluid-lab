"""Independent CPU counterexamples; no solver run or task acceptance claim."""

from datetime import datetime, timezone
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import audit_easy_run as stage
import audit_easy_experiment as overall


def row(seed, dose=20., seconds=5., role="baseline", offset=0., spill=0, valid=True):
    old = {"valid_evidence": valid, "dose_passed": False, "full_home_within_deadline": seconds <= 24,
        "maximum_all_label_spilled_particles": spill, "spill_passed": spill <= 1,
        "no_late_dose_change": True, "tolerance_ml": .02, "feasible": False}
    result = {"target_ml": 20., "received_ml": dose, "physical_time_s": seconds, "error_ml": dose - 20}
    return {"seed": seed, "role": role, "result": result, "strict_assessment": old,
        "easy_assessment": stage.easy_assessment(result, old),
        "declared_policy": {"control_target_offset_ml": offset}}


class Contracts(unittest.TestCase):
    def test_easy_endpoints_without_altering_original_strict_gate(self):
        for dose in (19., 21.):
            record = row(1, dose)
            self.assertTrue(record["easy_assessment"]["feasible"])
            self.assertFalse(record["strict_assessment"]["feasible"])
            self.assertEqual(record["strict_assessment"]["tolerance_ml"], .02)
        for dose in (math.nextafter(19., -math.inf), math.nextafter(21., math.inf)):
            self.assertFalse(row(1, dose)["easy_assessment"]["dose_passed"])

    def test_easy_cannot_rescue_invalid_physics_spill_or_deadline(self):
        self.assertIsNone(stage.score([row(1, valid=False)]))
        self.assertFalse(row(1, spill=2)["easy_assessment"]["feasible"])
        self.assertFalse(row(1, seconds=24.01)["easy_assessment"]["feasible"])

    def test_training_tie_order_is_frozen_not_test_selected(self):
        data = [row(seed, dose=19.3, role="offset_" + format(offset, "g"), offset=offset)
                for offset in stage.OFFSETS for seed in stage.TRAIN]
        self.assertEqual(stage.choose_offset(data), 0.)
        data[-2]["result"]["physical_time_s"] = 4.
        data[-1]["result"]["physical_time_s"] = 4.
        self.assertEqual(stage.choose_offset(data), 1.)

    def test_validation_tie_preserves_baseline(self):
        data = [row(seed, role=role) for role in ("baseline", "candidate") for seed in stage.VAL]
        self.assertEqual(stage.choose_validation_role(data), "baseline")

    def test_fewer_spills_cannot_hide_large_dose_failure(self):
        missing = [row(1, dose=10., spill=0), row(2, dose=10., spill=0)]
        closer = [row(1, dose=18.9, spill=1), row(2, dose=18.9, spill=1)]
        self.assertLess(stage.score(closer), stage.score(missing))

    def test_schedule_rejects_deletion_or_seed_reuse(self):
        cases = [{"case_id": f"c{index}_{seed}", "target_ml": 20., "seed": seed,
            "split": "train", "role": "offset_" + format(offset, "g"),
            "policy": {"control_target_offset_ml": offset}}
            for index, offset in enumerate(stage.OFFSETS) for seed in stage.TRAIN]
        stage.validate_schedule(cases, "train")
        with self.assertRaises(stage.AuditError):
            stage.validate_schedule(cases[:-1], "train")
        cases[-1]["seed"] = stage.VAL[0]
        with self.assertRaises(stage.AuditError):
            stage.validate_schedule(cases, "train")

    def test_exact_pinned_physical_math_is_reused(self):
        original, implementation = stage.load_physical_audit()
        self.assertIs(implementation.__code__, original.audit_episode.__code__)
        changed = {key for key in original.audit_episode.__globals__
                   if implementation.__globals__.get(key) != original.audit_episode.__globals__[key]}
        self.assertEqual(changed, {"COLLECTOR_SOURCE", "CYCLE_KINDS"})

    def test_final_gate_rejects_incomplete_before_test_or_model_reads(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            with self.assertRaisesRegex(ValueError, "before final audit"):
                overall.complete_gate(Path(directory), stage)

    def test_actual_reset_must_be_after_all_frozen_models(self):
        parse = lambda value: datetime.fromisoformat(value)
        t = lambda seconds: datetime(2026, 11, 28, 0, 0, seconds, tzinfo=timezone.utc)
        models = {role: {"created_at_utc": t(2).isoformat()} for role in ("candidate", "baseline")}
        recommended = {"created_at_utc": t(5).isoformat()}
        overall.verify_freeze_times(models, recommended, t(1), t(4), t(3), t(6), parse)
        recommended["created_at_utc"] = t(6).isoformat()
        with self.assertRaisesRegex(ValueError, "precede actual test"):
            overall.verify_freeze_times(models, recommended, t(1), t(4), t(3), t(6), parse)


if __name__ == "__main__":
    unittest.main()
