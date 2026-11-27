"""Post-hoc whole-cycle response projection from actual controller events.

An action begins at the quiet anchor before approaching and ends after cutoff
and actual quiet. Approach water is included. A skipped micro branch remains a
complete observed but ineligible/hazard response; an unfinished cycle has no
invented dose/time endpoint and remains in ``partial_cycles``.
"""

import copy
import math


STATE_KEYS = ("received_ml", "tube_remaining_ml", "tilt_deg",
              "source_inside_velocity_rms_m_s", "source_inside_centroid_local_z_m")
ACTION_KEYS = ("approach_delta_deg", "approach_speed_deg_s", "delta_deg",
               "tilt_speed_deg_s", "hold_s", "retract_deg", "retract_speed_deg_s",
               "cutoff_tilt_deg", "cutoff_speed_deg_s", "quiet_seconds",
               "maximum_measurement_hold_s")


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _number(value, name):
    _require(type(value) in (int, float) and math.isfinite(value), "Finite numeric value required: " + name)
    return float(value)


def state_from_observation(observation):
    """Extract actual state features without inferring motionless liquid."""
    _require(isinstance(observation, dict), "Actual observation must be a mapping")
    centroid = observation.get("source_inside_centroid_local_m")
    if "source_inside_centroid_local_z_m" in observation:
        z = _number(observation["source_inside_centroid_local_z_m"], "source centroid local Z")
    else:
        _require(isinstance(centroid, list) and len(centroid) == 3,
                 "An actual source centroid is required")
        z = _number(centroid[2], "source centroid local Z")
    values = {key: _number(observation[key], key) for key in STATE_KEYS[:-1]}
    values[STATE_KEYS[-1]] = z
    _require(values["source_inside_velocity_rms_m_s"] >= 0, "Actual source RMS cannot be negative")
    _require(observation.get("source_velocity_measurement_basis") == "provided_actual_particle_qd",
             "Model state requires actual particle qd, not missing-velocity zeros")
    return values


def _quiet(observation):
    return (observation.get("settled_measurement") is True
            and observation.get("source_airborne_particles") == 0
            and _number(observation.get("quiet_measured_seconds"), "quiet_measured_seconds") >= .2 - 1e-9)


def _action_from_start(start, episode):
    raw = start.get("action", start.get("declared_action"))
    _require(isinstance(raw, dict), "Cycle must declare its complete action")
    action = copy.deepcopy(raw)
    pre = start["observation"]
    if "approach_delta_deg" not in action:
        action["approach_delta_deg"] = (_number(start["approach_tilt_deg"], "approach_tilt_deg")
                                         - _number(pre["tilt_deg"], "anchor tilt"))
    if "approach_speed_deg_s" not in action:
        action["approach_speed_deg_s"] = _number(start["approach_speed_deg_s"], "approach speed")
    policy = episode.get("declared_policy", {})
    for key in ("quiet_seconds", "maximum_measurement_hold_s"):
        if key not in action:
            action[key] = _number(policy[key], key)
    return {key: _number(action[key], key) for key in ACTION_KEYS}


def cycles_from_episode(episode):
    """Return samples plus explicit partials; preserve original split/seed/IDs.

    Original engineering controller events are supported for diagnostics only.
    A dataset's declaration determines whether they may train; this extractor
    never upgrades development rows to train or test rows to validation.
    """
    _require(isinstance(episode, dict), "Episode must be an object")
    identifier, seed, split = episode["episode_id"], episode["seed"], episode["split"]
    _require(isinstance(identifier, str) and identifier and type(seed) is int,
             "Actual episode and simulation seed identities are required")
    _require(split in ("development", "train", "validation", "test"), "Unknown dataset split")
    fingerprint = episode.get("physics_fingerprint")
    _require(isinstance(fingerprint, str) and fingerprint, "Actual physics fingerprint required")
    events = episode["response_events"]
    _require(isinstance(events, list), "Actual event array is required")
    explicit = any(event.get("type") == "response_cycle_start" for event in events)
    starts = {"response_cycle_start"} if explicit else {"response_approach_start"}
    ends = ({"response_cycle_complete", "response_cycle_unsettled"} if explicit else
            {"response_action_complete", "response_action_unsettled", "response_not_executed"})
    pending, intermediate_micro_started, samples, partials = None, False, [], []
    last_time, event_ids = -1., set()
    action_endpoint = _number(episode["result"]["physical_time_s"], "episode physical endpoint")
    for event in events:
        _require(isinstance(event, dict), "Actual events must be objects")
        t = _number(event["physical_time_s"], "event physical time")
        _require(t >= last_time and 0 <= t <= min(24., action_endpoint) + 1e-9,
                 "Events must follow original action time, not post-observation")
        last_time = t
        eid = event["event_id"]
        _require(eid not in event_ids, "Duplicate controller event identity")
        event_ids.add(eid)
        kind = event.get("type")
        if kind in starts:
            _require(pending is None, "A previous whole cycle has no measured endpoint")
            pending, intermediate_micro_started = event, False
        elif kind == "response_action_start" and pending is not None:
            intermediate_micro_started = True
        elif kind in ends and pending is not None:
            pre, post = pending["observation"], event.get("post_observation", event["observation"])
            before, after = state_from_observation(pre), state_from_observation(post)
            t0, t1 = _number(pending["physical_time_s"], "cycle start"), t
            _require(t1 > t0, "Whole cycle must consume physical time")
            action = _action_from_start(pending, episode)
            if explicit:
                if "pre_observation" in event:
                    _require(event["pre_observation"] == pre, "Cycle endpoint rewrote its actual anchor state")
                link = event.get("pre_event_id", event.get("cycle_start_event_id"))
                _require(link == pending["event_id"], "Cycle endpoint references another start")
                if "action" in event:
                    endpoint_action = {key: _number(event["action"][key], key) for key in ACTION_KEYS}
                    _require(endpoint_action == action, "Cycle endpoint changed its declared action")
            executed = event.get("action_execution_complete", event.get("executed_micro", intermediate_micro_started))
            _require(type(executed) is bool, "Actual cycle branch completion must be Boolean")
            delta, duration = after["received_ml"] - before["received_ml"], t1 - t0
            if explicit and "outcome" in event:
                _require(abs(_number(event["outcome"]["settled_delta_ml"], "cycle measured delta") - delta) < 1e-9
                         and abs(_number(event["outcome"]["duration_s"], "cycle duration") - duration) < 1e-9,
                         "Whole-cycle outcome differs from actual anchor/endpoint")
            assessment = episode["strict_assessment"]
            peak = assessment["maximum_all_label_spilled_particles"]
            _require(type(peak) is int and peak >= 0, "Actual all-label peak spill required")
            settled = bool(event.get("settled") is True and _quiet(pre) and _quiet(post))
            row_id = identifier + ":whole_cycle_" + str(len(samples))
            samples.append({"row_id": row_id, "group_id": str(seed), "episode_id": identifier,
                "seed": seed, "split": split, "physics_fingerprint": fingerprint,
                "target_ml": _number(episode["target_ml"], "target_ml"),
                "safe": bool(assessment["valid_evidence"] is True and peak <= 1),
                "settled": settled, "action_execution_complete": executed,
                "peak_spill_particles": peak, "state": before, "action": action,
                "outcome": {"settled_delta_ml": delta, "duration_s": duration},
                "pre_physical_time_s": t0, "post_physical_time_s": t1,
                "pre_observation": copy.deepcopy(pre), "post_observation": copy.deepcopy(post),
                "source_start_event_id": pending["event_id"], "source_end_event_id": event["event_id"],
                "includes_approach_water": True,
                "unexecuted_or_unsettled_is_hazard_not_zero_success": True,
                "task_feasible": bool(assessment["feasible"])})
            pending, intermediate_micro_started = None, False
    if pending is not None:
        partials.append({"row_id": identifier + ":partial_cycle_" + str(len(samples)),
                         "group_id": str(seed), "physics_fingerprint": fingerprint,
                         "episode_id": identifier, "seed": seed, "split": split,
                         "source_start_event_id": pending["event_id"],
                         "pre_physical_time_s": pending["physical_time_s"],
                         "pre_observation": copy.deepcopy(pending["observation"]),
                         "state": state_from_observation(pending["observation"]),
                         "action": _action_from_start(pending, episode),
                         "declared_action": _action_from_start(pending, episode),
                         "settled_outcome_available": False,
                         "reason": "no_recorded_whole_cycle_endpoint"})
    vetoes = []
    no_late = episode["strict_assessment"].get("no_late_dose_change") is True
    if not no_late:
        vetoes.append("whole_episode_post_observation_dose_changed_or_unverified")
    # Compare the final measured cycle endpoint with the actual action footer.
    # Comparing every intermediate endpoint would misclassify a legitimate
    # second cycle's dose as water released during the return-home leg.
    home_delta, home_endpoint_available = None, bool(samples) and not partials
    if home_endpoint_available:
        actual_final = _number(episode["result"]["received_ml"], "actual footer received_ml")
        home_delta = actual_final - samples[-1]["post_observation"]["received_ml"]
        if home_delta != 0.:
            vetoes.append("received_volume_changed_after_last_cycle_endpoint")
    elif partials:
        vetoes.append("unfinished_cycle_has_no_home_leg_measurement_boundary")
    for sample in samples:
        sample.update(episode_safety_vetoes=list(vetoes),
                      episode_no_late_dose_change=no_late,
                      home_endpoint_comparison_available=home_endpoint_available,
                      home_additional_received_ml=home_delta)
        sample["safe"] = sample["safe"] and not vetoes
    return {"samples": samples, "partial_cycles": partials,
            "explicit_cycle_events": explicit, "original_split_preserved": True,
            "episode_safety": {"vetoes": vetoes, "home_additional_received_ml": home_delta,
                               "home_endpoint_comparison_available": home_endpoint_available}}
