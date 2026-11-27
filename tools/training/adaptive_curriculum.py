"""Independent CPU target features and reset-row sampling, without physics I/O.

Local progress trains only the sampler. It never replaces terminal task success.
The serialized SHA detects accidental edits; it is not an authentication key.
"""

from collections import deque
import hashlib
import json
import math
import platform
import random
import sys


HISTORY_CAPACITY = 50
RELATIVE_DOSE_TOLERANCE = 0.001
STATE_KIND = "safe_local_progress_reset_sampler_v1"


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _number(value, name):
    _require(type(value) in (int, float), name + " must be a finite number, not a Boolean.")
    try:
        result = float(value)
    except OverflowError as error:
        raise ValueError(name + " is outside finite float range.") from error
    _require(math.isfinite(result), name + " must be finite.")
    return result


def _target(value):
    value = _number(value, "target_ml")
    _require(20.0 <= value <= 40.0, "target_ml must lie in [20, 40] mL.")
    return value


def dose_accuracy(target_ml, received_ml):
    """Strict dose-only check on actual, unclipped values; no one-grain widening.

    This does not assess home pose, elapsed time, spills, mass or late droplets.
    A full experiment must evaluate those independent task gates separately.
    """
    target = _target(target_ml)
    received = _number(received_ml, "received_ml")
    _require(received >= 0.0, "received_ml cannot be negative.")
    signed_error = received - target
    tolerance = RELATIVE_DOSE_TOLERANCE * target
    lower, upper = target - tolerance, target + tolerance
    # Compare the measured quantity with the closed interval directly. For
    # example, abs(30.03 - 30) rounds just above 0.03 although 30.03 is the
    # representable upper endpoint; no epsilon or particle allowance is added.
    return {"target_ml": target, "received_ml": received,
            "signed_error_ml": signed_error, "absolute_error_ml": abs(signed_error),
            "tolerance_ml": tolerance, "lower_bound_ml": lower, "upper_bound_ml": upper,
            "dose_passed": lower <= received <= upper}


def normalize_target(target_ml, initial_source_ml, received_ml, *, fine_error_clip=10.0):
    """Return model features separately from the strict actual dose assessment.

    Positive shortfall means underfill. Only the fine model feature is clipped.
    Particle quantization and measurement uncertainty do not widen the criterion.
    """
    actual = dose_accuracy(target_ml, received_ml)
    initial = _number(initial_source_ml, "initial_source_ml")
    clip = _number(fine_error_clip, "fine_error_clip")
    _require(initial >= actual["target_ml"], "Initial source volume must cover the requested target.")
    _require(clip > 0.0, "fine_error_clip must be positive.")
    shortfall = -actual["signed_error_ml"]
    fine = shortfall / actual["tolerance_ml"]
    coarse = shortfall / initial
    _require(math.isfinite(fine) and math.isfinite(coarse), "Scaled errors must remain finite.")
    return {"model": {"target_fraction": actual["target_ml"] / initial,
                      "coarse_shortfall_fraction": coarse,
                      "fine_shortfall_clipped": max(-clip, min(clip, fine))},
            "actual": {**actual, "initial_source_ml": initial,
                       "shortfall_ml": shortfall, "fine_shortfall_unclipped": fine},
            "fine_error_clip": clip,
            "success_scope": "dose_only_full_task_gates_are_separate"}


def _canonical_bytes(value):
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=False, allow_nan=False).encode("utf-8")
    except (ValueError, TypeError) as error:
        raise ValueError("State must contain finite JSON values.") from error


def canonical_sha256(value):
    """SHA of canonical UTF-8 JSON, without a trailing newline."""
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _row_ids(values):
    _require(type(values) in (list, tuple) and len(values) > 0, "Provide a nonempty ordered row-ID sequence.")
    _require(all(type(value) is str and value and not value.isspace() for value in values),
             "Every row ID must be a nonempty string.")
    _require(len(set(values)) == len(values), "Reset row IDs must be unique.")
    return tuple(values)


def _random_compatibility():
    return {"implementation": platform.python_implementation(),
            "major_minor": list(sys.version_info[:2]), "engine": "random.Random/MT19937/state-v3"}


def _no_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        _require(key not in result, "Duplicate JSON state key: " + key)
        result[key] = value
    return result


class AdaptiveResetSampler:
    """50% shuffled cyclic coverage and 50% weighted frontier, in alternation.

    Histories hold the last 50 safety-qualified local-progress Booleans per row.
    Empty histories are explicitly unknown and receive priority 0.5 + epsilon.
    Alternation gives an exact 50/50 split over every even number of draws.
    """

    def __init__(self, row_ids, *, seed=0, epsilon=0.05):
        self.row_ids = _row_ids(row_ids)
        _require(type(seed) is int, "seed must be an integer, not a Boolean.")
        self.epsilon = _number(epsilon, "epsilon")
        _require(0.0 < self.epsilon <= 1.0, "epsilon must lie in (0, 1].")
        self._rng = random.Random(seed)
        self._histories = {row: deque(maxlen=HISTORY_CAPACITY) for row in self.row_ids}
        self._cycle_order = list(self.row_ids)
        self._rng.shuffle(self._cycle_order)
        self._cycle_cursor = 0
        self._next_stream = "cyclic"
        self._draws = {"cyclic": 0, "frontier": 0}

    def _history(self, row_id):
        _require(type(row_id) is str and row_id in self._histories, "Unknown reset row ID.")
        return self._histories[row_id]

    def row_statistics(self, row_id):
        history = self._history(row_id)
        probability = sum(history) / len(history) if history else None
        priority = (math.sqrt(probability * (1.0 - probability))
                    if probability is not None else 0.5) + self.epsilon
        return {"row_id": row_id, "history_count": len(history), "unknown": probability is None,
                "safe_local_progress_probability": probability, "priority": priority}

    def record_outcome(self, row_id, *, local_progress, safe, task_success=None):
        """Record sampler evidence; terminal task success cannot change its label.

        Unsafe trials are retained as False even when a milestone was reached.
        ``task_success`` is optional reporting context, deliberately not stored in
        the sampler history or used in its priorities.
        """
        history = self._history(row_id)
        _require(type(local_progress) is bool and type(safe) is bool,
                 "local_progress and safe must be Boolean.")
        _require(task_success is None or type(task_success) is bool, "task_success must be Boolean or None.")
        label = safe and local_progress
        history.append(label)
        return {"row_id": row_id, "recorded_local_progress": label, "safe": safe,
                "reported_task_success": task_success, "task_success_used_for_sampling": False}

    def sample(self):
        stream = self._next_stream
        if stream == "cyclic":
            if self._cycle_cursor == len(self.row_ids):
                self._cycle_order = list(self.row_ids)
                self._rng.shuffle(self._cycle_order)
                self._cycle_cursor = 0
            row = self._cycle_order[self._cycle_cursor]
            self._cycle_cursor += 1
        else:
            weights = [self.row_statistics(row)["priority"] for row in self.row_ids]
            threshold = self._rng.random() * sum(weights)
            cumulative = 0.0
            row = self.row_ids[-1]
            for candidate, weight in zip(self.row_ids, weights):
                cumulative += weight
                if threshold < cumulative:
                    row = candidate
                    break
        self._draws[stream] += 1
        self._next_stream = "frontier" if stream == "cyclic" else "cyclic"
        return {**self.row_statistics(row), "stream": stream}

    def sample_many(self, count):
        _require(type(count) is int and count >= 0, "count must be a nonnegative integer.")
        return [self.sample() for _ in range(count)]

    def state_dict(self):
        payload = {"schema_version": 1, "kind": STATE_KIND,
                   "row_ids": list(self.row_ids), "epsilon": self.epsilon,
                   "history_capacity": HISTORY_CAPACITY,
                   "histories": {row: list(self._histories[row]) for row in self.row_ids},
                   "cycle_order": list(self._cycle_order), "cycle_cursor": self._cycle_cursor,
                   "next_stream": self._next_stream, "draws_by_stream": dict(self._draws),
                   "random_compatibility": _random_compatibility(),
                   "rng_state": self._rng.getstate()}
        # A JSON round-trip makes tuples lists, so the public representation is
        # exactly the same before and after writing it to disk.
        payload = json.loads(_canonical_bytes(payload))
        return {**payload, "sha256": canonical_sha256(payload)}

    def to_json(self):
        return _canonical_bytes(self.state_dict()).decode("utf-8")

    @classmethod
    def from_json(cls, text, *, expected_row_ids=None):
        try:
            value = json.loads(text, object_pairs_hook=_no_duplicate_keys)
        except (ValueError, TypeError, UnicodeError) as error:
            raise ValueError("Invalid serialized sampler JSON.") from error
        return cls.from_state_dict(value, expected_row_ids=expected_row_ids)

    @classmethod
    def from_state_dict(cls, state, *, expected_row_ids=None):
        expected_keys = {"schema_version", "kind", "row_ids", "epsilon", "history_capacity", "histories",
                         "cycle_order", "cycle_cursor", "next_stream", "draws_by_stream",
                         "random_compatibility", "rng_state", "sha256"}
        _require(type(state) is dict and set(state) == expected_keys, "Unexpected sampler state fields.")
        payload = {key: value for key, value in state.items() if key != "sha256"}
        _require(type(state["sha256"]) is str and canonical_sha256(payload) == state["sha256"],
                 "Sampler state SHA mismatch.")
        _require(type(state["schema_version"]) is int and state["schema_version"] == 1
                 and state["kind"] == STATE_KIND and type(state["history_capacity"]) is int
                 and state["history_capacity"] == HISTORY_CAPACITY, "Unsupported sampler state schema.")
        _require(state["random_compatibility"] == _random_compatibility(),
                 "Python random compatibility differs; deterministic reload is not promised across runtimes.")
        row_ids = _row_ids(state["row_ids"])
        if expected_row_ids is not None:
            _require(row_ids == _row_ids(expected_row_ids), "Restored reset row IDs differ from the expected artifact.")
        histories = state["histories"]
        _require(type(histories) is dict and set(histories) == set(row_ids), "State contains missing or unknown history row IDs.")
        for history in histories.values():
            _require(type(history) is list and len(history) <= HISTORY_CAPACITY
                     and all(type(value) is bool for value in history), "Invalid local-progress history.")
        cycle = state["cycle_order"]
        _require(type(cycle) is list and len(cycle) == len(row_ids)
                 and all(type(value) is str for value in cycle) and set(cycle) == set(row_ids),
                 "Cyclic queue must be an exact permutation of known rows.")
        draws = state["draws_by_stream"]
        _require(type(draws) is dict and set(draws) == {"cyclic", "frontier"}
                 and all(type(value) is int and value >= 0 for value in draws.values()), "Invalid draw counts.")
        _require(draws["cyclic"] - draws["frontier"] in (0, 1), "Draw counts violate the 50/50 schedule.")
        next_stream = "cyclic" if draws["cyclic"] == draws["frontier"] else "frontier"
        cursor = 0 if draws["cyclic"] == 0 else (draws["cyclic"] - 1) % len(row_ids) + 1
        _require(type(state["cycle_cursor"]) is int and state["cycle_cursor"] == cursor
                 and state["next_stream"] == next_stream, "Cycle cursor or next stream is inconsistent.")
        rng_state = state["rng_state"]
        _require(type(rng_state) is list and len(rng_state) == 3
                 and type(rng_state[0]) is int and rng_state[0] == 3
                 and type(rng_state[1]) is list and len(rng_state[1]) == 625,
                 "Unsupported random state.")
        _require(all(type(value) is int and 0 <= value <= 0xFFFFFFFF for value in rng_state[1][:-1])
                 and type(rng_state[1][-1]) is int and 0 <= rng_state[1][-1] <= 624,
                 "Invalid MT19937 random state.")
        if rng_state[2] is not None:
            _number(rng_state[2], "cached_gaussian")
        sampler = cls(row_ids, seed=0, epsilon=state["epsilon"])
        sampler._rng.setstate((rng_state[0], tuple(rng_state[1]), rng_state[2]))
        sampler._histories = {row: deque(histories[row], maxlen=HISTORY_CAPACITY) for row in row_ids}
        sampler._cycle_order = list(cycle)
        sampler._cycle_cursor = cursor
        sampler._next_stream = next_stream
        sampler._draws = dict(draws)
        return sampler
