"""Pure V5 task scoring, independent of every active solver and controller.

A valid episode is feasible only when measured absolute error is <=0.2 mL,
all-label spill is zero, and full return home finishes in <=5 raw physical
seconds. Error is the signed settled-particle-volume measurement, never a
tail-flow prediction. A feasible case is ranked only by action time.

Across the SAME declared cases, candidates with any invalid physical/mass/
measurement evidence are ineligible for elites. A candidate is feasible only
when every declared case is feasible, and then only mean action time ranks.
Every other valid candidate ranks by mean absolute volume error, then total
all-label spill. Failure count is diagnostic only; time affects ranking only
after every case is feasible.
"""

from dataclasses import dataclass
import math
from numbers import Integral, Real
from typing import Iterable, Mapping, Sequence


ERROR_TOLERANCE_ML = .2
ACTION_DEADLINE_SECONDS = 5.
MEASUREMENT_BASIS = "settled_particle_volume"
TIME_BASIS = "raw_physics"


@dataclass(frozen=True)
class CaseOutcome:
    """Collector evidence, with total spill covering BOTH particle labels.

    full_home means upright at the original complete home pose, with the
    required physical quiet interval already included in action time.
    measurement_valid means a real settled remeasurement establishes the
    dose; later observation may verify it but must not rescue a late action.
    The caller remains responsible for those physical evidence assertions.
    """

    case_id: str
    signed_error_ml: float
    action_physical_seconds: float
    full_home: bool
    total_spilled_particles: int
    physical_valid: bool
    measurement_valid: bool
    mass_conserved: bool
    measurement_basis: str = MEASUREMENT_BASIS
    time_basis: str = TIME_BASIS

    @classmethod
    def from_mapping(cls, value: Mapping):
        """Reject unknown fields instead of silently accepting old schemas."""
        return cls(**dict(value))


@dataclass(frozen=True)
class CaseAssessment:
    outcome: CaseOutcome
    eligible_for_elite: bool
    feasible: bool
    absolute_error_ml: float | None
    score: tuple | None
    invalid_reasons: tuple[str, ...]
    infeasible_reasons: tuple[str, ...]


@dataclass(frozen=True)
class CandidateAssessment:
    eligible_for_elite: bool
    feasible: bool
    score: tuple | None
    cases: tuple[CaseAssessment, ...]
    infeasible_case_count: int
    sum_absolute_error_ml: float | None
    mean_absolute_error_ml: float | None
    total_spilled_particles: int | None
    mean_action_physical_seconds: float | None
    invalid_reasons: tuple[str, ...]


def _finite_real(value):
    return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(float(value))


def assess_case(outcome: CaseOutcome) -> CaseAssessment:
    """Assess a case without modifying, clipping or quantizing its record.

    score=None means EXCLUDE from elite selection. It is not an infinite
    score to mix into ordinary infeasible ranking. Bounds are inclusive,
    with no hidden epsilon that accepts errors above0.2 or actions above5s.
    """
    if not isinstance(outcome, CaseOutcome):
        raise TypeError("assess_case requires a CaseOutcome or CaseOutcome.from_mapping(...).")
    invalid = []
    if not isinstance(outcome.case_id, str) or not outcome.case_id:
        invalid.append("invalid_case_id")
    if not _finite_real(outcome.signed_error_ml):
        invalid.append("nonfinite_or_nonnumeric_error")
    if not _finite_real(outcome.action_physical_seconds) or outcome.action_physical_seconds < 0:
        invalid.append("invalid_raw_physical_time")
    if type(outcome.full_home) is not bool:
        invalid.append("invalid_full_home_flag")
    if (not isinstance(outcome.total_spilled_particles, Integral)
            or isinstance(outcome.total_spilled_particles, bool) or outcome.total_spilled_particles < 0):
        invalid.append("invalid_total_spill_count")
    for name in ("physical_valid", "measurement_valid", "mass_conserved"):
        if getattr(outcome, name) is not True:
            invalid.append(name + "_required")
    if outcome.measurement_basis != MEASUREMENT_BASIS:
        invalid.append("settled_particle_measurement_required")
    if outcome.time_basis != TIME_BASIS:
        invalid.append("raw_physical_time_required")
    if invalid:
        return CaseAssessment(outcome, False, False, None, None, tuple(invalid), ())

    error = abs(float(outcome.signed_error_ml))
    elapsed = float(outcome.action_physical_seconds)
    spill = int(outcome.total_spilled_particles)
    failures = []
    if error > ERROR_TOLERANCE_ML:
        failures.append("volume_error_exceeds_0_2_ml")
    if spill != 0:
        failures.append("nonzero_total_spill")
    if elapsed > ACTION_DEADLINE_SECONDS:
        failures.append("full_action_exceeds_5s")
    if not outcome.full_home:
        failures.append("full_home_not_reached")
    feasible = not failures
    # Only feasible cases compete on speed. Failed cases cannot gain a
    # tie-break advantage by rushing an incomplete action.
    score = (0, elapsed) if feasible else (1, error, spill)
    return CaseAssessment(outcome, True, feasible, error, score, (), tuple(failures))


def aggregate_cases(outcomes: Iterable[CaseOutcome], *, expected_case_ids: Sequence[str]) -> CandidateAssessment:
    """Rank one candidate over an exact declared set of paired conditions.

    Required coverage prevents a candidate gaining by omitting difficult
    cases. Invalid evidence rejects the WHOLE candidate, including otherwise
    feasible cases. All signed errors remain present in cases[].outcome.
    """
    assessments = tuple(assess_case(item) for item in outcomes)
    expected = tuple(expected_case_ids)
    invalid = []
    if (not expected or any(not isinstance(name, str) or not name for name in expected)
            or len(set(expected)) != len(expected)):
        invalid.append("invalid_expected_case_set")
    actual = tuple(item.outcome.case_id for item in assessments)
    # Do not set() malformed IDs; assess_case already flags their schema.
    if all(isinstance(name, str) for name in actual):
        if len(set(actual)) != len(actual):
            invalid.append("duplicate_case_ids")
        if len(actual) != len(expected) or set(actual) != set(expected):
            invalid.append("incomplete_or_foreign_case_coverage")
    else:
        invalid.append("invalid_case_coverage")
    for index, item in enumerate(assessments):
        if not item.eligible_for_elite:
            invalid.extend(f"case[{index}]:{reason}" for reason in item.invalid_reasons)
    failed = sum(not item.feasible for item in assessments)
    if invalid:
        return CandidateAssessment(False, False, None, assessments, failed, None, None, None, None, tuple(invalid))

    total_error = math.fsum(item.absolute_error_ml for item in assessments)
    mean_error = total_error / len(assessments)
    total_spill = sum(int(item.outcome.total_spilled_particles) for item in assessments)
    mean_time = math.fsum(float(item.outcome.action_physical_seconds) for item in assessments) / len(assessments)
    feasible = failed == 0
    score = ((0, mean_time) if feasible else
             (1, mean_error, total_spill))
    return CandidateAssessment(True, feasible, score, assessments, failed,
                               total_error, mean_error, total_spill, mean_time, ())
