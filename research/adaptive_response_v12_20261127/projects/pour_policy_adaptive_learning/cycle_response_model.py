"""Conservative local model of *measured whole-cycle settled* pouring responses.

This is supervised system identification, not PPO and not a liquid solver.
Every response includes approach from the anchor, micro-action, retreat, cutoff and its actual
settled remeasurement.  The model's empirical envelope is not a statistical
safety certificate.  Terminal dose/home/spill acceptance remains external.
"""

from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics


MODEL_KIND = "empirical_anchor_approach_micro_cutoff_cycle_v1"
STATE_FEATURES = ("received_ml", "tube_remaining_ml", "tilt_deg",
                  "source_inside_velocity_rms_m_s", "source_inside_centroid_local_z_m")
ACTION_FEATURES = ("approach_delta_deg", "approach_speed_deg_s", "delta_deg",
                   "tilt_speed_deg_s", "hold_s", "retract_deg", "retract_speed_deg_s",
                   "cutoff_tilt_deg", "cutoff_speed_deg_s", "quiet_seconds",
                   "maximum_measurement_hold_s")
FEATURES = STATE_FEATURES + ACTION_FEATURES
DEFAULT_CONFIG = {
    "scales": {"received_ml": 1.0, "tube_remaining_ml": 1.0, "tilt_deg": .02,
               "source_inside_velocity_rms_m_s": .04,
               "source_inside_centroid_local_z_m": .005,
               "approach_delta_deg": .35, "approach_speed_deg_s": .001,
               "delta_deg": .005, "tilt_speed_deg_s": .001,
               "hold_s": .001, "retract_deg": .001,
               "retract_speed_deg_s": .001, "cutoff_tilt_deg": .001,
               "cutoff_speed_deg_s": .001, "quiet_seconds": .001,
               "maximum_measurement_hold_s": .001},
    "radius": 1.0, "min_neighbors": 3, "min_seed_groups": 2,
    "max_neighbors": 12, "uncertainty_floor_ml": .019683,
}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _finite(value, name):
    _require(type(value) in (int, float), name + " must be a number, not Boolean")
    try:
        result = float(value)
    except OverflowError as error:
        raise ValueError(name + " is not finite") from error
    _require(math.isfinite(result), name + " must be finite")
    return result


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("utf8")


def _digest(value):
    return hashlib.sha256(_canonical(value)).hexdigest()


def _config(value=None):
    config = json.loads(json.dumps(DEFAULT_CONFIG))
    if value is not None:
        _require(isinstance(value, dict), "config must be an object")
        _require(not set(value) - set(config), "Unknown model configuration field")
        config.update(value)
    _require(set(config["scales"]) == set(FEATURES), "Every feature needs an explicit scale")
    for name in FEATURES:
        config["scales"][name] = _finite(config["scales"][name], "scale." + name)
        _require(config["scales"][name] > 0, "Feature scales must be positive")
    for name in ("radius", "uncertainty_floor_ml"):
        config[name] = _finite(config[name], name)
        _require(config[name] > 0, name + " must be positive")
    for name in ("min_neighbors", "min_seed_groups", "max_neighbors"):
        _require(type(config[name]) is int and config[name] > 0,
                 name + " must be a positive integer")
    _require(config["max_neighbors"] >= config["min_neighbors"],
             "max_neighbors cannot be smaller than min_neighbors")
    return config


def _features(state, action):
    _require(isinstance(state, dict) and isinstance(action, dict),
             "State and action must be mappings")
    values = {name: _finite(state[name] if name in STATE_FEATURES else action[name], name)
              for name in FEATURES}
    _require(values["received_ml"] >= 0 and values["tube_remaining_ml"] >= 0,
             "Actual received and remaining volume must be nonnegative")
    _require(0 <= values["tilt_deg"] <= 180, "Tilt must lie in [0,180] degrees")
    _require(values["hold_s"] >= 0 and values["retract_deg"] >= 0
             and values["retract_speed_deg_s"] > 0 and values["tilt_speed_deg_s"] > 0
             and values["cutoff_speed_deg_s"] > 0
             and 0 <= values["cutoff_tilt_deg"] <= 180,
             "Invalid whole-cycle action timing/retraction")
    _require(values["source_inside_velocity_rms_m_s"] >= 0, "Source RMS velocity cannot be negative")
    _require(values["approach_speed_deg_s"] > 0 and values["quiet_seconds"] >= .2
             and values["maximum_measurement_hold_s"] >= values["quiet_seconds"],
             "Invalid whole-cycle approach/quiet measurement profile")
    _require(0 <= values["tilt_deg"] + values["approach_delta_deg"] <= 180,
             "Whole-cycle approach angle is outside physical angular range")
    return values


def _row(value, expected_split):
    _require(isinstance(value, dict), "Response row must be an object")
    for name in ("row_id", "group_id", "physics_fingerprint"):
        _require(isinstance(value.get(name), str) and value[name], name + " is required")
    _require(type(value.get("seed")) is int and 0 <= value["seed"] < 2 ** 32,
             "A simulation seed is required to guard split leakage")
    _require(value.get("split") == expected_split,
             "Fitting cannot use test rows or relabel another split")
    for name in ("safe", "settled", "action_execution_complete"):
        _require(type(value.get(name)) is bool, name + " must be an actual Boolean")
    spill = value.get("peak_spill_particles")
    _require(type(spill) is int and spill >= 0, "Peak spill must be an integer count")
    features = _features(value["state"], value["action"])
    outcome = value["outcome"]
    delta = _finite(outcome["settled_delta_ml"], "settled_delta_ml")
    duration = _finite(outcome["duration_s"], "duration_s")
    _require(duration > 0, "Response must consume positive physical time")
    # Negative actual changes (e.g. re-entry/loss) remain recorded as hazards.
    # Do not clip them into successful zero-volume examples.
    eligible = (value["safe"] and value["settled"] and value["action_execution_complete"]
                and spill <= 1 and delta >= 0)
    normalized = {name: value[name] for name in
                  ("row_id", "group_id", "seed", "split", "physics_fingerprint",
                   "safe", "settled", "action_execution_complete", "peak_spill_particles")}
    normalized.update(features=features, outcome={"settled_delta_ml": delta, "duration_s": duration},
                      fit_eligible=eligible)
    for name in ("episode_safety_vetoes", "episode_no_late_dose_change",
                 "home_additional_received_ml", "home_endpoint_comparison_available"):
        if name in value:
            normalized[name] = value[name]
    return normalized


def _partial_row(value, expected_split):
    _require(isinstance(value, dict) and value.get("split") == expected_split,
             "Partial response split cannot be reassigned")
    for name in ("row_id", "group_id", "physics_fingerprint"):
        _require(isinstance(value.get(name), str) and value[name], name + " is required")
    _require(type(value.get("seed")) is int and 0 <= value["seed"] < 2 ** 32,
             "Partial response requires its original simulation seed")
    _require(value.get("settled_outcome_available") is False and "outcome" not in value,
             "An unfinished cycle cannot invent a settled endpoint")
    return {**{name: value[name] for name in ("row_id", "group_id", "seed", "split", "physics_fingerprint")},
            "features": _features(value["state"], value["action"]),
            "reason": value.get("reason", "unfinished_cycle"), "fit_eligible": False}


def _distance(left, right, config):
    # A maximum feature distance prevents many close coordinates from hiding
    # a large unsupported jump in one crucial angular/action coordinate.
    return max(abs(left[name] - right[name]) / config["scales"][name] for name in FEATURES)


def _seal(model):
    model = json.loads(json.dumps(model, allow_nan=False))
    model.pop("content_sha256", None)
    model["content_sha256"] = _digest(model)
    return model


def verify_model(model):
    _require(isinstance(model, dict) and model.get("kind") == MODEL_KIND,
             "Unknown response model kind")
    payload = dict(model)
    digest = payload.pop("content_sha256", None)
    _require(digest == _digest(payload), "Response model checksum mismatch")
    _config(model["config"])
    return True


def predict_response(model, state, action):
    """Return an empirical dose/time envelope, or explicitly abstain.

    Unsafe, unsettled and negative-volume training records veto the same local
    region. Zero-dose records participate normally and are never filtered out.
    """
    verify_model(model)
    query = _features(state, action)
    config = model["config"]
    nearby = sorted(((_distance(query, row["features"], config), row)
                     for row in model["training_rows"]), key=lambda item: (item[0], item[1]["row_id"]))
    nearby = [(distance, row) for distance, row in nearby if distance <= config["radius"]]
    rejected = [row["row_id"] for _, row in nearby if not row["fit_eligible"]]
    rejected.extend(row["row_id"] for row in model.get("partial_training_rows", [])
                    if _distance(query, row["features"], config) <= config["radius"])
    if rejected:
        return {"supported": False, "reason": "local_unsafe_unsettled_or_negative_response",
                "veto_row_ids": rejected}
    neighbors = nearby[:config["max_neighbors"]]
    seeds = {row["seed"] for _, row in neighbors}
    groups = {row["group_id"] for _, row in neighbors}
    if (len(neighbors) < config["min_neighbors"]
            or len(seeds) < config["min_seed_groups"] or len(groups) < config["min_seed_groups"]):
        return {"supported": False, "reason": "insufficient_local_independent_support",
                "neighbors": len(neighbors), "seed_groups": len(seeds)}
    distances = [distance for distance, _ in neighbors]
    doses = [row["outcome"]["settled_delta_ml"] for _, row in neighbors]
    durations = [row["outcome"]["duration_s"] for _, row in neighbors]
    weights = [1 / (1 + distance) for distance in distances]
    total_weight = sum(weights)
    mean = sum(weight * dose for weight, dose in zip(weights, doses)) / total_weight
    spread = statistics.pstdev(doses)
    calibration = model["validation_calibration"]
    margin = max(config["uncertainty_floor_ml"], spread, calibration["maximum_absolute_residual_ml"])
    return {"supported": True, "reason": "local_empirical_support_only",
            "mean_delta_ml": mean, "lower_delta_ml": max(0., min(doses) - margin),
            "upper_delta_ml": max(doses) + margin, "empirical_std_ml": spread,
            "mean_duration_s": statistics.mean(durations),
            "upper_duration_s": max(durations) + calibration["maximum_duration_residual_s"],
            "upper_peak_spill_particles": max(row["peak_spill_particles"] for _, row in neighbors),
            "support_row_ids": [row["row_id"] for _, row in neighbors],
            "independent_seed_groups": len(seeds), "maximum_feature_distance": max(distances),
            "uncertainty_is_empirical_not_a_guarantee": True}


def _public_state_action(row):
    features = row["features"]
    return ({name: features[name] for name in STATE_FEATURES},
            {name: features[name] for name in ACTION_FEATURES})


def _evaluate_rows(model, rows):
    predictions, errors, duration_errors = [], [], []
    for row in rows:
        state, action = _public_state_action(row)
        prediction = predict_response(model, state, action)
        record = {"row_id": row["row_id"], "group_id": row["group_id"],
                  "seed": row["seed"], "safe": row["safe"], "settled": row["settled"],
                  "peak_spill_particles": row["peak_spill_particles"],
                  "actual_delta_ml": row["outcome"]["settled_delta_ml"],
                  "actual_duration_s": row["outcome"]["duration_s"], "prediction": prediction}
        if prediction["supported"]:
            error = abs(prediction["mean_delta_ml"] - row["outcome"]["settled_delta_ml"])
            duration_error = abs(prediction["mean_duration_s"] - row["outcome"]["duration_s"])
            errors.append(error)
            duration_errors.append(duration_error)
            record.update(absolute_delta_error_ml=error, absolute_duration_error_s=duration_error)
        predictions.append(record)
    return {"row_count": len(rows), "predicted_rows": len(errors),
            "abstained_rows": len(rows) - len(errors),
            "invalid_or_hazard_rows": sum(not row["fit_eligible"] for row in rows),
            "mean_absolute_delta_error_ml": statistics.mean(errors) if errors else None,
            "maximum_absolute_delta_error_ml": max(errors) if errors else None,
            "maximum_duration_error_s": max(duration_errors) if duration_errors else None,
            "rows": predictions}


def fit_response_model(training_rows, validation_rows=None, *, configs=None,
                       freeze_path=None, provenance=None, partial_training_rows=None,
                       partial_validation_rows=None):
    """Fit local empirical responses, selecting configuration on validation only.

    Entire trajectory groups AND seeds must be disjoint across splits. A model
    fitted without validation is explicitly engineering/unselected. Test rows
    can only be evaluated through :func:`evaluate_response_model`, never fit.
    Configs are declared before examining validation, not supplied from test.
    """
    training = [_row(row, "train") for row in training_rows]
    validation = [_row(row, "validation") for row in (validation_rows or [])]
    partial_train = [_partial_row(row, "train") for row in (partial_training_rows or [])]
    partial_val = [_partial_row(row, "validation") for row in (partial_validation_rows or [])]
    _require(training, "At least one physical training response is required")
    combined = training + validation + partial_train + partial_val
    training_scope, validation_scope = training + partial_train, validation + partial_val
    _require(len({row["row_id"] for row in combined}) == len(combined), "Duplicate response row ID")
    _require(not ({row["group_id"] for row in training_scope} & {row["group_id"] for row in validation_scope}),
             "Training and validation trajectory groups overlap")
    _require(not ({row["seed"] for row in training_scope} & {row["seed"] for row in validation_scope}),
             "Training and validation simulation seeds overlap")
    fingerprints = {row["physics_fingerprint"] for row in combined}
    _require(len(fingerprints) == 1, "Cannot pool different physics/measurement protocols")
    declared_configs = [_config(value) for value in (configs or [DEFAULT_CONFIG])]
    _require(validation or len(declared_configs) == 1,
             "Multiple configurations require independent validation")
    summaries, candidates = [], []
    for index, config in enumerate(declared_configs):
        model = {"kind": MODEL_KIND, "schema_version": 1, "config": config,
                 "physics_fingerprint": next(iter(fingerprints)), "training_rows": training,
                 "partial_training_rows": partial_train,
                 "validation_calibration": {"maximum_absolute_residual_ml": 0.,
                                            "maximum_duration_residual_s": 0.},
                 "training_row_ids": [row["row_id"] for row in training_scope],
                 "validation_row_ids": [row["row_id"] for row in validation_scope],
                 "training_group_ids": sorted({row["group_id"] for row in training_scope}),
                 "validation_group_ids": sorted({row["group_id"] for row in validation_scope}),
                 "training_seeds": sorted({row["seed"] for row in training_scope}),
                 "validation_seeds": sorted({row["seed"] for row in validation_scope})}
        model = _seal(model)
        evaluation = _evaluate_rows(model, validation)
        evaluation["partial_validation_cycles"] = len(partial_val)
        # Do not let abstention on difficult rows artificially lower MAE.
        rank = (evaluation["abstained_rows"] + len(partial_val),
                evaluation["mean_absolute_delta_error_ml"] or 0.,
                evaluation["maximum_absolute_delta_error_ml"] or 0., index)
        summaries.append({"config_index": index, "config": config,
                          "validation": evaluation, "selection_rank": list(rank)})
        candidates.append(model)
    selected_index = min(range(len(candidates)), key=lambda index: summaries[index]["selection_rank"])
    model = candidates[selected_index]
    evaluation = summaries[selected_index]["validation"]
    model["validation_calibration"] = {
        "maximum_absolute_residual_ml": evaluation["maximum_absolute_delta_error_ml"] or 0.,
        "maximum_duration_residual_s": evaluation["maximum_duration_error_s"] or 0.}
    model.update(
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        selection_kind="validation_only" if validation else "engineering_unselected",
        selected_config_index=selected_index, configuration_selection=summaries,
        training_data_sha256=_digest(training), validation_data_sha256=_digest(validation),
        training_row_count=len(training), validation_row_count=len(validation),
        training_eligible_rows=sum(row["fit_eligible"] for row in training),
        training_hazard_rows=sum(not row["fit_eligible"] for row in training),
        partial_training_cycles=len(partial_train), partial_validation_cycles=len(partial_val),
        training_zero_dose_rows=sum(row["outcome"]["settled_delta_ml"] == 0 for row in training),
        validation_complete_local_support=bool(validation) and not partial_val and evaluation["abstained_rows"] == 0,
        validation_all_responses_valid=bool(validation) and not partial_val and evaluation["invalid_or_hazard_rows"] == 0,
        terminal_task_acceptance_proven=False,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        provenance=provenance or {},
        limitations=["Empirical local support is not a certified physical uncertainty bound.",
                     "A response model does not prove 24s/full-home/dose/spill task success.",
                     "Actual settled remeasurement must decide final dose; no future-flow stop substitution."])
    # Only physically declared training actions form this convenience menu.
    # Validation/test actions never expand the trained controller action set.
    menu = {_digest({name: row["features"][name] for name in ACTION_FEATURES}):
            {name: row["features"][name] for name in ACTION_FEATURES} for row in training}
    model["action_menu"] = [menu[key] for key in sorted(menu)]
    model = _seal(model)
    if freeze_path is not None:
        save_response_model(freeze_path, model)
    return model


def evaluate_response_model(model, test_rows):
    """Evaluate a frozen model on unseen test groups/seeds without updating it."""
    verify_model(model)
    rows = [_row(row, "test") for row in test_rows]
    _require(len({row["row_id"] for row in rows}) == len(rows), "Duplicate test row ID")
    used_ids = set(model["training_row_ids"] + model["validation_row_ids"])
    used_groups = set(model["training_group_ids"] + model["validation_group_ids"])
    used_seeds = set(model["training_seeds"] + model["validation_seeds"])
    _require(not used_ids & {row["row_id"] for row in rows}, "Test response row was used in fitting")
    _require(not used_groups & {row["group_id"] for row in rows}, "Test trajectory group was used in fitting")
    _require(not used_seeds & {row["seed"] for row in rows}, "Test seed was used in fitting")
    _require(all(row["physics_fingerprint"] == model["physics_fingerprint"] for row in rows),
             "Test physics/measurement protocol differs from fitted model")
    return _evaluate_rows(model, rows)


def save_response_model(path, model):
    """Create one immutable checkpoint; report raw SHA for external manifests."""
    verify_model(model)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(model, indent=2, sort_keys=True, allow_nan=False).encode("utf8") + b"\n"
    with path.open("xb") as stream:
        stream.write(content)
    return {"path": str(path), "raw_sha256": hashlib.sha256(content).hexdigest(),
            "content_sha256": model["content_sha256"]}


def load_response_model(path, expected_raw_sha256=None):
    content = Path(path).read_bytes()
    if expected_raw_sha256 is not None:
        _require(hashlib.sha256(content).hexdigest() == expected_raw_sha256,
                 "Frozen raw checkpoint SHA mismatch")
    model = json.loads(content)
    verify_model(model)
    return model


def choose_action(model, state, actions, *, target_ml, remaining_time_s,
                  home_reserve_s, current_peak_spill_particles=0):
    """Choose one empirically supported action, or abstain/request actual home.

    The candidate envelope cannot cross the user's upper dose limit. A below-
    target lower envelope is allowed because a later action may be considered.
    Predicted success is never substituted for actual settled measurement.
    """
    verify_model(model)
    target = _finite(target_ml, "target_ml")
    remaining = _finite(remaining_time_s, "remaining_time_s")
    reserve = _finite(home_reserve_s, "home_reserve_s")
    _require(20 <= target <= 40 and remaining >= 0 and reserve >= 0,
             "Invalid target or remaining/home time budget")
    _require(type(current_peak_spill_particles) is int and current_peak_spill_particles >= 0,
             "Current spill must be an actual nonnegative particle count")
    received = _finite(state["received_ml"], "received_ml")
    tolerance = .001 * target
    low, high = target - tolerance, target + tolerance
    if low <= received <= high:
        return {"mode": "request_home_and_terminal_remeasurement", "action": None,
                "actual_dose_within_window_now": True, "terminal_task_success": False}
    if received > high or current_peak_spill_particles > 1:
        return {"mode": "abstain", "action": None, "reason": "actual_irreversible_task_violation"}
    considered, feasible = [], []
    for index, action in enumerate(actions):
        prediction = predict_response(model, state, action)
        record = {"index": index, "action": action, "prediction": prediction}
        reason = None
        if not prediction["supported"]:
            reason = prediction["reason"]
        elif prediction["upper_peak_spill_particles"] > 1:
            reason = "predicted_spill"
        elif prediction["upper_duration_s"] + reserve > remaining:
            reason = "insufficient_home_time"
        elif received + prediction["upper_delta_ml"] > high:
            reason = "dose_envelope_crosses_upper_limit"
        elif prediction["mean_delta_ml"] <= 0:
            reason = "no_measured_positive_response"
        record["rejection_reason"] = reason
        considered.append(record)
        if reason is None:
            # Rank actual target-conditioned dose error, then uncertainty/time.
            rank = (abs(target - received - prediction["mean_delta_ml"]),
                    prediction["upper_delta_ml"] - prediction["lower_delta_ml"],
                    prediction["upper_duration_s"], index)
            feasible.append((rank, record))
    if not feasible:
        return {"mode": "abstain", "action": None, "reason": "no_supported_safe_action",
                "considered": considered, "terminal_task_success": False}
    _, selected = min(feasible, key=lambda item: item[0])
    return {"mode": "execute_then_actual_settled_remeasure", "action": selected["action"],
            "prediction": selected["prediction"], "considered": considered,
            "terminal_task_success": False}
