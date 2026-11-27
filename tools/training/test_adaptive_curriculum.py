"""Meaningful pure-CPU contracts; no solver, model, GPU or experiment output."""

import copy
import json
import math
import sys
import unittest

sys.dont_write_bytecode = True
from adaptive_curriculum import (
    AdaptiveResetSampler, canonical_sha256, dose_accuracy, normalize_target,
)


def resign(value):
    value = copy.deepcopy(value)
    value.pop("sha256", None)
    return {**value, "sha256": canonical_sha256(value)}


class TargetContracts(unittest.TestCase):
    def test_fractional_goal_coarse_and_fine_features(self):
        value = normalize_target(23.25, 56.68704, 23.0)
        self.assertAlmostEqual(value["model"]["target_fraction"], 23.25 / 56.68704)
        self.assertAlmostEqual(value["model"]["coarse_shortfall_fraction"], .25 / 56.68704)
        self.assertAlmostEqual(value["actual"]["fine_shortfall_unclipped"], .25 / .02325)
        self.assertEqual(value["model"]["fine_shortfall_clipped"], 10.)
        self.assertFalse(value["actual"]["dose_passed"])

    def test_clipping_never_changes_strict_dose_success(self):
        value = normalize_target(20, 56.68704, 20.03, fine_error_clip=.1)
        self.assertEqual(value["model"]["fine_shortfall_clipped"], -.1)
        self.assertLess(value["actual"]["fine_shortfall_unclipped"], -1.)
        self.assertFalse(value["actual"]["dose_passed"])
        self.assertEqual(value["actual"]["tolerance_ml"], .020)
        self.assertTrue(dose_accuracy(20., 20.)["dose_passed"])
        # No epsilon or one-grain band is added at the strict boundary.
        self.assertFalse(dose_accuracy(20., math.nextafter(20.02, math.inf))["dose_passed"])
        self.assertFalse(dose_accuracy(20., math.nextafter(19.98, -math.inf))["dose_passed"])

    def test_target_endpoints_and_invalid_inputs(self):
        for target in (20., 20.5, 23.25, 39.5, 40.):
            self.assertTrue(normalize_target(target, 56.68704, target)["actual"]["dose_passed"])
        for target in (19.999, 40.001, True, "30", None, math.nan, math.inf, -math.inf):
            with self.subTest(target=target), self.assertRaises(ValueError):
                normalize_target(target, 56.68704, 20.)
        for source in (0., -1., 19., math.nan, math.inf, True):
            with self.subTest(source=source), self.assertRaises(ValueError):
                normalize_target(20., source, 20.)
        for received in (-1., math.nan, math.inf, True):
            with self.subTest(received=received), self.assertRaises(ValueError):
                normalize_target(20., 56.68704, received)
        for clip in (0., -1., math.nan, math.inf, True):
            with self.subTest(clip=clip), self.assertRaises(ValueError):
                normalize_target(20., 56.68704, 20., fine_error_clip=clip)

    def test_strict_closed_interval_endpoints_without_tolerance_widening(self):
        for target in (20., 23.25, 27.75, 30., 31.5, 39.5, 40.):
            tolerance = target * .001
            lower, upper = target - tolerance, target + tolerance
            with self.subTest(target=target):
                self.assertTrue(dose_accuracy(target, lower)["dose_passed"])
                self.assertTrue(dose_accuracy(target, upper)["dose_passed"])
                self.assertFalse(dose_accuracy(target, math.nextafter(lower, -math.inf))["dose_passed"])
                self.assertFalse(dose_accuracy(target, math.nextafter(upper, math.inf))["dose_passed"])
                self.assertEqual(dose_accuracy(target, upper)["signed_error_ml"], upper - target)


class SamplerContracts(unittest.TestCase):
    def test_unknown_and_half_progress_frontier_keep_zero_one_epsilon(self):
        sampler = AdaptiveResetSampler(["unknown", "zero", "half", "one"], epsilon=.02)
        for index in range(50):
            for row, progress in (("zero", False), ("half", index % 2 == 0), ("one", True)):
                sampler.record_outcome(row, local_progress=progress, safe=True)
        unknown = sampler.row_statistics("unknown")
        self.assertTrue(unknown["unknown"])
        self.assertIsNone(unknown["safe_local_progress_probability"])
        self.assertEqual(unknown["history_count"], 0)
        self.assertEqual(unknown["priority"], .52)
        self.assertEqual(sampler.row_statistics("half")["priority"], .52)
        for row in ("zero", "one"):
            self.assertEqual(sampler.row_statistics(row)["priority"], .02)
            self.assertFalse(sampler.row_statistics(row)["unknown"])

    def test_one_hundred_outcomes_keep_only_latest_fifty(self):
        sampler = AdaptiveResetSampler(["row"])
        for index in range(100):
            sampler.record_outcome("row", local_progress=index >= 50, safe=True)
        stats = sampler.row_statistics("row")
        self.assertEqual(stats["history_count"], 50)
        self.assertEqual(stats["safe_local_progress_probability"], 1.)
        self.assertEqual(sampler.state_dict()["histories"]["row"], [True] * 50)

    def test_unsafe_milestone_is_retained_as_false(self):
        sampler = AdaptiveResetSampler(["row"])
        response = sampler.record_outcome("row", local_progress=True, safe=False, task_success=True)
        self.assertFalse(response["recorded_local_progress"])
        self.assertEqual(sampler.state_dict()["histories"]["row"], [False])
        self.assertEqual(sampler.row_statistics("row")["safe_local_progress_probability"], 0.)

    def test_local_progress_and_terminal_success_are_independent(self):
        left = AdaptiveResetSampler(["a", "b"], seed=7)
        right = AdaptiveResetSampler(["a", "b"], seed=7)
        for sampler, terminal in ((left, False), (right, True)):
            sampler.record_outcome("a", local_progress=True, safe=True, task_success=terminal)
            sampler.record_outcome("b", local_progress=False, safe=True, task_success=not terminal)
        self.assertEqual(left.state_dict(), right.state_dict())
        self.assertEqual(left.row_statistics("a")["safe_local_progress_probability"], 1.)
        self.assertEqual(left.row_statistics("b")["safe_local_progress_probability"], 0.)

    def test_exact_mix_and_cyclic_coverage_even_for_low_priority_rows(self):
        ids = [f"r{index}" for index in range(7)]
        sampler = AdaptiveResetSampler(ids, seed=20261127)
        for row in ids:
            sampler.record_outcome(row, local_progress=False, safe=True)
        first = sampler.sample_many(14)
        second = sampler.sample_many(14)
        for samples in (first, second):
            self.assertEqual([s["stream"] for s in samples], ["cyclic", "frontier"] * 7)
            self.assertEqual({s["row_id"] for s in samples if s["stream"] == "cyclic"}, set(ids))
        self.assertEqual(sampler.state_dict()["draws_by_stream"], {"cyclic": 14, "frontier": 14})

    def test_state_roundtrip_and_continued_feedback_are_deterministic(self):
        ids = [f"row_{index}" for index in range(9)]
        original = AdaptiveResetSampler(ids, seed=8842)
        for index in range(100):
            original.record_outcome(ids[index % len(ids)], local_progress=index % 3 == 0, safe=index % 7 != 0)
        original.sample_many(17)
        text = original.to_json()
        restored = AdaptiveResetSampler.from_json(text, expected_row_ids=ids)
        restored_dict = AdaptiveResetSampler.from_state_dict(original.state_dict(), expected_row_ids=ids)
        self.assertEqual(restored.to_json(), text)
        self.assertEqual(restored_dict.to_json(), text)
        for index in range(120):
            expected = original.sample()
            self.assertEqual(restored.sample(), expected)
            self.assertEqual(restored_dict.sample(), expected)
            for sampler in (original, restored, restored_dict):
                sampler.record_outcome(expected["row_id"], local_progress=index % 4 == 0,
                                       safe=index % 11 != 0, task_success=index % 5 == 0)
        self.assertEqual(original.state_dict(), restored.state_dict())
        self.assertEqual(original.state_dict(), restored_dict.state_dict())

    def test_tamper_and_changed_artifact_ids_are_rejected(self):
        sampler = AdaptiveResetSampler(["a", "b"], seed=4)
        state = sampler.state_dict()
        state["histories"]["a"].append(True)
        with self.assertRaisesRegex(ValueError, "SHA mismatch"):
            AdaptiveResetSampler.from_state_dict(state)
        with self.assertRaisesRegex(ValueError, "expected artifact"):
            AdaptiveResetSampler.from_json(sampler.to_json(), expected_row_ids=["a", "unknown"])
        state = sampler.state_dict()
        state["histories"]["unknown"] = []
        with self.assertRaisesRegex(ValueError, "unknown history"):
            AdaptiveResetSampler.from_state_dict(resign(state))

    def test_rehashed_structural_corruption_is_also_rejected(self):
        original = AdaptiveResetSampler(["a", "b"]).state_dict()
        mutations = [
            {"cycle_order": ["a", "a"]}, {"cycle_cursor": 1}, {"next_stream": "frontier"},
            {"draws_by_stream": {"cyclic": 5, "frontier": 0}}, {"history_capacity": 51},
            {"histories": {"a": [True] * 51, "b": []}},
            {"histories": {"a": [1], "b": []}}, {"row_ids": ["a", "a"]},
            {"rng_state": [3, [0] * 624 + [625], None]},
            {"random_compatibility": {"engine": "unknown"}},
        ]
        for mutation in mutations:
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                AdaptiveResetSampler.from_state_dict(resign({**original, **mutation}))

    def test_unknown_ids_and_non_boolean_outcomes_are_rejected(self):
        sampler = AdaptiveResetSampler(["a"])
        with self.assertRaises(ValueError): sampler.row_statistics("unknown")
        with self.assertRaises(ValueError): sampler.record_outcome("unknown", local_progress=True, safe=True)
        for kwargs in ({"local_progress": 1, "safe": True},
                       {"local_progress": True, "safe": 1},
                       {"local_progress": True, "safe": True, "task_success": 1}):
            with self.assertRaises(ValueError): sampler.record_outcome("a", **kwargs)
        self.assertEqual(sampler.row_statistics("a")["history_count"], 0)

    def test_constructor_and_json_reject_nonfinite_or_ambiguous_states(self):
        for ids in ([], ["a", "a"], [""], [1], "row"):
            with self.assertRaises(ValueError): AdaptiveResetSampler(ids)
        for epsilon in (0., -1., math.nan, math.inf, True):
            with self.assertRaises(ValueError): AdaptiveResetSampler(["a"], epsilon=epsilon)
        with self.assertRaises(ValueError): AdaptiveResetSampler(["a"], seed=True)
        for text in ('{"schema_version":1,"schema_version":2}', '{"epsilon":NaN}', '[]'):
            with self.assertRaises(ValueError): AdaptiveResetSampler.from_json(text)
        with self.assertRaises(ValueError): canonical_sha256({"bad": math.nan})


if __name__ == "__main__":
    unittest.main(verbosity=2)
