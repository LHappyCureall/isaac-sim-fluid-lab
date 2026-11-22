"""Small counterexamples for the user's V5 volume-first task definition."""

from dataclasses import replace
import math
import unittest

from task_score import CaseOutcome, aggregate_cases, assess_case


def case(name="a", error=0., seconds=4., spill=0, **overrides):
    outcome = CaseOutcome(name, error, seconds, True, spill, True, True, True)
    return replace(outcome, **overrides)


def candidate(*outcomes):
    return aggregate_cases(outcomes, expected_case_ids=tuple(item.case_id for item in outcomes))


class TaskScoreTests(unittest.TestCase):
    def test_tolerance_and_deadline_include_exact_boundaries(self):
        for error in (-.2, .2):
            result = assess_case(case(error=error, seconds=5.))
            self.assertTrue(result.feasible)
            self.assertEqual(result.outcome.signed_error_ml, error)
        self.assertFalse(assess_case(case(error=math.nextafter(.2, math.inf))).feasible)
        self.assertFalse(assess_case(case(seconds=math.nextafter(5., math.inf))).feasible)

    def test_accurate_small_spill_beats_ten_ml_underfill_without_spill(self):
        under = assess_case(case(error=-10., spill=0))
        accurate = assess_case(case(error=.05, spill=2))
        self.assertFalse(under.feasible)
        self.assertFalse(accurate.feasible)
        self.assertLess(accurate.score, under.score)

    def test_any_feasible_case_beats_any_valid_infeasible_case(self):
        good = assess_case(case(error=.2, seconds=5.))
        exact_spill = assess_case(case(error=0., seconds=1., spill=1))
        self.assertLess(good.score, exact_spill.score)

    def test_feasible_cases_compare_time_only(self):
        fast = assess_case(case(error=.19, seconds=3.))
        precise_slow = assess_case(case(error=0., seconds=4.))
        self.assertLess(fast.score, precise_slow.score)
        self.assertEqual(fast.score, assess_case(case(error=-.01, seconds=3.)).score)

    def test_equal_failed_error_then_compare_total_spill(self):
        self.assertLess(assess_case(case(error=-.5, spill=1)).score,
                        assess_case(case(error=.5, spill=2, seconds=1.)).score)

    def test_timeout_missing_home_and_either_label_spill_fail(self):
        for value in (case(seconds=5.01), case(full_home=False), case(spill=1)):
            result = assess_case(value)
            self.assertTrue(result.eligible_for_elite)
            self.assertFalse(result.feasible)

    def test_invalid_evidence_is_excluded_not_ranked_as_failed(self):
        values = (case(error=float("nan")), case(seconds=float("inf")), case(mass_conserved=False),
                  case(physical_valid=False), case(measurement_valid=False),
                  case(measurement_basis="flow_tail_prediction"), case(time_basis="accelerated_video"))
        for value in values:
            with self.subTest(value=value):
                result = assess_case(value)
                self.assertFalse(result.eligible_for_elite)
                self.assertIsNone(result.score)
                group = candidate(value, case(name="b"))
                self.assertFalse(group.eligible_for_elite)
                self.assertIsNone(group.score)

    def test_fewer_failed_cases_cannot_hide_severe_underfill(self):
        one_failure = candidate(case(error=-10.), case(name="b"))
        two_small_spills = candidate(case(error=.01, spill=1), case(name="b", error=-.01, spill=1))
        self.assertEqual(one_failure.infeasible_case_count, 1)
        self.assertEqual(two_small_spills.infeasible_case_count, 2)
        self.assertLess(two_small_spills.score, one_failure.score)
        self.assertEqual(two_small_spills.score, (1, .01, 2))

    def test_only_all_case_feasible_candidates_enter_the_fast_tier(self):
        all_feasible = candidate(case(error=.2, seconds=5.), case(name="b", error=-.2, seconds=5.))
        one_small_spill = candidate(case(error=0., seconds=1., spill=1), case(name="b", seconds=1.))
        self.assertTrue(all_feasible.feasible)
        self.assertFalse(one_small_spill.feasible)
        self.assertLess(all_feasible.score, one_small_spill.score)

    def test_equal_failure_count_volume_precedes_spill(self):
        underfill = candidate(case(error=-10., spill=0), case(name="b"))
        near_dose = candidate(case(error=.05, spill=2), case(name="b"))
        self.assertLess(near_dose.score, underfill.score)
        self.assertEqual(underfill.cases[0].outcome.signed_error_ml, -10.)

    def test_candidate_equal_error_then_spill_and_feasible_only_time(self):
        fewer = candidate(case(error=-.4, spill=1), case(name="b"))
        more = candidate(case(error=.4, spill=2, seconds=1.), case(name="b"))
        self.assertLess(fewer.score, more.score)
        faster = candidate(case(error=.19, seconds=2.), case(name="b", error=-.19, seconds=4.))
        slower = candidate(case(error=0., seconds=3.5), case(name="b", seconds=3.5))
        self.assertTrue(faster.feasible and slower.feasible)
        self.assertLess(faster.score, slower.score)
        self.assertEqual(faster.score, (0, 3.))

    def test_omitted_duplicate_or_foreign_cases_cannot_enter_elites(self):
        for values in ((case(),), (case(), case()), (case(), case(name="c"))):
            result = aggregate_cases(values, expected_case_ids=("a", "b"))
            self.assertFalse(result.eligible_for_elite)
            self.assertIsNone(result.score)
        self.assertFalse(aggregate_cases([], expected_case_ids=()).eligible_for_elite)

    def test_mapping_preserves_signed_particle_error_and_rejects_old_keys(self):
        original = case(error=-.042875)
        reconstructed = CaseOutcome.from_mapping(vars(original))
        self.assertEqual(assess_case(reconstructed).outcome.signed_error_ml, -.042875)
        with self.assertRaises(TypeError):
            CaseOutcome.from_mapping({**vars(original), "source_only_spill": 0})

    def test_infeasible_cases_do_not_compete_on_time(self):
        slow = case(error=.05, spill=1, seconds=8.)
        rushed = case(error=.05, spill=1, seconds=1., full_home=False)
        self.assertEqual(assess_case(slow).score, assess_case(rushed).score)
        self.assertEqual(candidate(slow).score, candidate(rushed).score)
        # A real settled late measurement is still valid evidence of a
        # time-limit failure, rather than a record to drop from the dataset.
        late = assess_case(case(error=.01, seconds=6.))
        self.assertTrue(late.eligible_for_elite)
        self.assertFalse(late.feasible)
        self.assertIn("full_action_exceeds_5s", late.infeasible_reasons)


if __name__ == "__main__":
    unittest.main()
