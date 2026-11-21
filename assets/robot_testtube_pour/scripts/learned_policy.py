"""Small, interpretable feedback policy trained by episodic CEM search.

This is a three-parameter policy for stopping a live fluid pour. It is not
PPO, a neural network, or a six-joint grasp controller. The observations are
privileged simulator measurements. No particle positions or physical time
are modified here; every candidate must be evaluated by the live solver.
"""

import json
import math
import os
from pathlib import Path
import tempfile

import numpy as np


PARAMETER_NAMES = ("bias_ml", "lead_seconds", "target_slope_ml")
DEFAULT_PARAMS = {"bias_ml": 0.20, "lead_seconds": 0.20, "target_slope_ml": 0.0}
PARAMETER_BOUNDS = {
    "bias_ml": (-0.8, 0.8),
    "lead_seconds": (0.0, 0.5),
    "target_slope_ml": (-0.5, 0.5),
}
POLICY_KIND = "cem_linear_tail_compensation_v1"


def _vector(parameters):
    if isinstance(parameters, dict):
        if set(parameters) != set(PARAMETER_NAMES):
            raise ValueError(f"Expected exactly these parameters: {PARAMETER_NAMES}")
        values = [parameters[name] for name in PARAMETER_NAMES]
    else:
        values = parameters
    result = np.asarray(values, dtype=np.float64)
    if result.shape != (3,) or not np.isfinite(result).all():
        raise ValueError("Policy parameters must contain three finite values.")
    return result


def _parameters(values):
    return {name: float(value) for name, value in zip(PARAMETER_NAMES, values)}


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         delete=False, suffix=".tmp") as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


class LearnedPourPolicy:
    """Memoryless feedback policy; default decisions match the old baseline.

    The stop decision is
      received + catchable_inflight + lead_seconds * flow
        + bias_ml + target_slope_ml * ((target_ml - 30) / 10) >= target_ml.
    Pour speed retains the existing bounded heuristic. Only the three stop
    parameters are learned; prepare/return/settle remain controller stages.
    """

    name = POLICY_KIND

    def __init__(self, params=None):
        values = _vector(DEFAULT_PARAMS if params is None else params)
        low = np.asarray([PARAMETER_BOUNDS[name][0] for name in PARAMETER_NAMES])
        high = np.asarray([PARAMETER_BOUNDS[name][1] for name in PARAMETER_NAMES])
        if np.any(values < low) or np.any(values > high):
            raise ValueError("Policy parameters exceed the reviewed search bounds.")
        self.params = _parameters(values)

    def act(self, observation):
        keys = ("target_ml", "received_ml", "inflight_ml", "flow_ml_s", "tilt_deg", "released_ml")
        values = {name: float(observation[name]) for name in keys}
        if not all(math.isfinite(value) for value in values.values()):
            raise ValueError("Policy observations must be finite.")
        target = values["target_ml"]
        if not 0.0 < target <= 45.0:
            raise ValueError("target_ml must be in (0, 45].")
        received, inflight = values["received_ml"], values["inflight_ml"]
        flow = max(0.0, values["flow_ml_s"])
        p = self.params
        margin = p["bias_ml"] + p["target_slope_ml"] * ((target - 30.0) / 10.0)
        predicted = received + inflight + p["lead_seconds"] * flow
        # Compare in the baseline's original arithmetic order so the default
        # policy also preserves decisions exactly at a floating-point boundary.
        stop = predicted >= target - margin and values["released_ml"] > 1.0
        rate = 0.05 if values["tilt_deg"] < 80.0 else 0.025
        if max(0.0, target - (received + inflight)) < 8.0:
            rate = min(rate, 0.012)
        return {"stop": bool(stop), "rate": float(rate),
                "predicted_final_ml": float(predicted),
                "stop_margin_ml": float(margin),
                "decision": "upright" if stop else "continue_pour"}

    def save(self, path, metadata=None):
        checkpoint = {"schema_version": 1, "policy_kind": POLICY_KIND,
                      "parameters": dict(self.params),
                      "observation_units": {"volume": "mL", "flow": "mL/s", "tilt": "degrees"},
                      "target_feature": "(target_ml - 30.0) / 10.0",
                      "metadata": {} if metadata is None else metadata}
        _atomic_json(path, checkpoint)
        return checkpoint

    @classmethod
    def load(cls, path):
        checkpoint = json.loads(Path(path).read_text(encoding="utf-8"))
        if checkpoint.get("schema_version") != 1 or checkpoint.get("policy_kind") != POLICY_KIND:
            raise ValueError("Unsupported pouring-policy checkpoint.")
        return cls(checkpoint["parameters"])


class CEMSearch:
    """Seeded diagonal cross-entropy policy search; lower score is better.

    ask() includes the best evaluated parameters and the current mean, then
    bounded antithetic Gaussian samples. tell() retains the best training
    score ever seen. Test episodes must never be passed to tell().
    """

    def __init__(self, seed=0, population=6, elite_count=2, mean=None, std=None,
                 bounds=None, smoothing=0.7):
        if int(population) != population or population < 3:
            raise ValueError("population must be an integer of at least three.")
        if int(elite_count) != elite_count or not 1 <= elite_count < population:
            raise ValueError("elite_count must be in [1, population).")
        if not 0.0 < float(smoothing) <= 1.0:
            raise ValueError("smoothing must be in (0, 1].")
        selected_bounds = PARAMETER_BOUNDS if bounds is None else bounds
        if set(selected_bounds) != set(PARAMETER_NAMES):
            raise ValueError("bounds must specify the three policy parameters.")
        pairs = np.asarray([selected_bounds[name] for name in PARAMETER_NAMES], dtype=np.float64)
        if pairs.shape != (3, 2) or not np.isfinite(pairs).all() or np.any(pairs[:, 0] >= pairs[:, 1]):
            raise ValueError("bounds must be finite increasing pairs.")
        reviewed = np.asarray([PARAMETER_BOUNDS[name] for name in PARAMETER_NAMES])
        if np.any(pairs[:, 0] < reviewed[:, 0]) or np.any(pairs[:, 1] > reviewed[:, 1]):
            raise ValueError("Search bounds cannot exceed reviewed policy bounds.")
        self.low, self.high = pairs[:, 0], pairs[:, 1]
        self.mean = _vector(DEFAULT_PARAMS if mean is None else mean)
        if np.any(self.mean < self.low) or np.any(self.mean > self.high):
            raise ValueError("Initial mean must lie within the search bounds.")
        self.std = _vector([0.25, 0.08, 0.15] if std is None else std)
        if np.any(self.std <= 0):
            raise ValueError("Search standard deviations must be positive.")
        self.minimum_std = np.minimum(np.asarray([0.015, 0.008, 0.015]), (self.high - self.low) / 10)
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self.population, self.elite_count = int(population), int(elite_count)
        self.smoothing = float(smoothing)
        self.generation = 0
        self.best_params, self.best_score = None, None
        self.history = []
        self._pending = None

    def ask(self):
        if self._pending is not None:
            raise RuntimeError("Score the pending population with tell() before asking again.")
        values = []
        if self.best_params is not None:
            values.append(_vector(self.best_params))
        if not values or not np.array_equal(values[0], self.mean):
            values.append(self.mean.copy())
        while len(values) < self.population:
            delta = self.rng.standard_normal(3) * self.std
            for sign in (1.0, -1.0):
                if len(values) < self.population:
                    values.append(np.clip(self.mean + sign * delta, self.low, self.high))
        self._pending = [_parameters(value) for value in values]
        return [dict(parameters) for parameters in self._pending]

    def tell(self, scores, candidates=None):
        evaluated = self._pending if candidates is None else candidates
        if evaluated is None:
            raise RuntimeError("Call ask() before tell(), or supply evaluated candidates.")
        matrix = np.asarray([_vector(parameters) for parameters in evaluated])
        scores = np.asarray(scores, dtype=np.float64)
        if matrix.shape != (self.population, 3) or scores.shape != (self.population,):
            raise ValueError("One score is required per population member.")
        if not np.isfinite(scores).all() or np.any(matrix < self.low) or np.any(matrix > self.high):
            raise ValueError("Scores must be finite and candidates must lie within bounds.")
        order = np.argsort(scores, kind="stable")
        winner = int(order[0])
        if self.best_score is None or float(scores[winner]) < self.best_score:
            self.best_score = float(scores[winner])
            self.best_params = _parameters(matrix[winner])
        elite = matrix[order[:self.elite_count]]
        alpha = self.smoothing
        self.mean = np.clip((1.0 - alpha) * self.mean + alpha * elite.mean(axis=0), self.low, self.high)
        self.std = np.maximum(self.minimum_std, (1.0 - alpha) * self.std + alpha * elite.std(axis=0))
        summary = {"generation": self.generation, "scores": scores.tolist(),
                   "candidates": [_parameters(value) for value in matrix],
                   "elite_indices": order[:self.elite_count].tolist(),
                   "best_score": self.best_score, "best_params": dict(self.best_params),
                   "next_mean": _parameters(self.mean), "next_std": _parameters(self.std)}
        self.history.append(summary)
        self.generation += 1
        self._pending = None
        return summary

    def state_dict(self):
        return {"schema_version": 1, "seed": self.seed, "generation": self.generation,
                "population": self.population, "elite_count": self.elite_count,
                "mean": _parameters(self.mean), "std": _parameters(self.std),
                "bounds": {name: [float(a), float(b)] for name, a, b in
                           zip(PARAMETER_NAMES, self.low, self.high)},
                "best_params": self.best_params, "best_score": self.best_score,
                "rng_state": self.rng.bit_generator.state, "history": self.history,
                "pending": self._pending}
