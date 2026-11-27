"""Pure evidence projection for the new live response experiment.

No solver, controller or GPU imports.  Original action footer/legacy gate stay
untouched; this module derives an explicit strict task gate and response rows.
Completed unsafe, zero and negative responses are retained with eligibility
flags. A partial action has no invented settled outcome.
"""

import copy
import math


ACTION_KEYS = ("delta_deg", "tilt_speed_deg_s", "hold_s", "retract_deg",
               "retract_speed_deg_s", "cutoff_tilt_deg", "cutoff_speed_deg_s")
STATE_KEYS = ("received_ml", "tube_remaining_ml", "tilt_deg")


def number(value, name):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError("Finite numeric value required: " + name)
    return float(value)


def strict_assessment(result, status, spill, post):
    """24 s / closed original-target +/-0.1% / <=1 grain, no late rescue.

    The closed interval avoids subtractive floating-point cancellation at a
    representable endpoint. It adds neither epsilon nor particle allowance.
    """
    target = number(result["target_ml"], "target_ml")
    received = number(result["received_ml"], "received_ml")
    if not 20 <= target <= 40:
        raise ValueError("Goal must be within20--40mL")
    if type(spill) is not int or spill < 0:
        raise ValueError("Spill must be a nonnegative count")
    elapsed = number(result["physical_time_s"], "physical_time_s")
    measured = number(result["measurement_physical_time_s"], "measurement_physical_time_s")
    post_valid = (isinstance(post, dict) and post.get("all_states_finite") is True
        and post.get("mass_array_unchanged") is True and not post.get("error")
        and post.get("maximum_below_table_particles", 0) == 0
        and abs(post.get("start_physical_time_s", -1) - elapsed) < 1e-8
        and type(post.get("physical_steps")) is int and post["physical_steps"] >= 120
        and post.get("end_physical_time_s", -1) - post.get("start_physical_time_s", 0) >= 2 - 1e-8)
    valid = bool(result.get("valid_physics") is True and result.get("mass_conserved") is True
        and result.get("all_states_finite") is True and result.get("source_partition_conserved") is True
        and result.get("max_below_table_particles") == 0 and abs(measured - elapsed) < 1e-8
        and post_valid)
    home = bool(result.get("complete") is True and status.get("done") is True
        and status.get("home_static") is True and status.get("stable_frames", 0) >= 12
        and status.get("source_airborne_particles") == 0 and not status.get("timed_out")
        and elapsed <= 24.)
    tolerance = .001 * target
    dose = target - tolerance <= received <= target + tolerance
    no_late = isinstance(post, dict) and post.get("maximum_received_change_ml") == 0.
    return {"valid_evidence": valid, "dose_passed": dose,
        "full_home_within_deadline": home, "maximum_all_label_spilled_particles": spill,
        "spill_passed": spill <= 1, "no_late_dose_change": no_late, "tolerance_ml": tolerance,
        "feasible": bool(valid and dose and home and spill <= 1 and no_late)}


def event_kind(event):
    return event.get("type", event.get("event", event.get("kind")))


def observation(value):
    if not isinstance(value, dict):
        raise ValueError("Actual observation must be an object")
    for key in STATE_KEYS:
        number(value[key], key)
    if (type(value.get("source_airborne_particles")) is not int
            or value["source_airborne_particles"] < 0):
        raise ValueError("Actual source airborne count required")
    number(value["quiet_measured_seconds"], "quiet_measured_seconds")
    return copy.deepcopy(value)


def samples_from_events(row):
    """Flatten complete start/end pairs; never use task success as a label.

    A seed forms one group across every action/target, so downstream splitting
    cannot put the same initial-condition seed in both fitting and validation.
    Failed and incomplete action records remain in row.response_events; only a
    complete pair can define a measured response training row.
    """
    events = row.get("response_events", [])
    if not isinstance(events, list):
        raise ValueError("Response event array required")
    seed = row["seed"]
    if type(seed) is not int or seed < 0:
        raise ValueError("Integer seed required for grouped evidence")
    split = row.get("split", "development")
    if split not in ("train", "validation", "development", "test"):
        raise ValueError("Explicit supported dataset split required")
    episode = row["episode_id"]
    if not isinstance(episode, str) or not episode or any(c in episode for c in "/\\:"):
        raise ValueError("Safe episode identity required")
    action_end = number(row["result"]["physical_time_s"], "action endpoint time")
    if not isinstance(row.get("physics_fingerprint"), str) or not row["physics_fingerprint"]:
        raise ValueError("A nonempty physical environment fingerprint is required")
    pending, samples = None, []
    for event in events:
        if not isinstance(event, dict):
            raise ValueError("Response events must be objects")
        kind = event_kind(event)
        if kind == "response_action_start":
            if pending is not None:
                raise ValueError("Nested response start has no measured completion")
            pending = event
        elif kind in ("response_action_complete", "response_action_unsettled"):
            if pending is None:
                raise ValueError("Response completion has no actual start")
            if event.get("pre_event_id") != pending.get("event_id"):
                raise ValueError("Response completion references a different start")
            pre = observation(event["pre_observation"])
            post = observation(event["post_observation"])
            if pre != pending["observation"]:
                raise ValueError("Response completion rewrote its start observation")
            t0 = number(event["pre_physical_time_s"], "pre_physical_time_s")
            t1 = number(event["post_physical_time_s"], "post_physical_time_s")
            if (t0 != number(pending["physical_time_s"], "start physical_time_s")
                    or not 0 <= t0 < t1 <= min(24., action_end)):
                raise ValueError("Response must use ordered original action time")
            action = pending["action"]
            if not isinstance(action, dict):
                raise ValueError("Declared action required")
            for key in ACTION_KEYS:
                number(action[key], key)
            if event.get("action", action) != action:
                raise ValueError("Response action changed before measurement")
            if (action["hold_s"] < 0 or action["retract_deg"] < 0
                    or action["retract_speed_deg_s"] <= 0 or action["tilt_speed_deg_s"] <= 0
                    or action["cutoff_speed_deg_s"] <= 0):
                raise ValueError("Response action units/bounds are invalid")
            delta = post["received_ml"] - pre["received_ml"]
            duration = t1 - t0
            outcome = event["outcome"]
            if (abs(number(outcome["settled_delta_ml"], "settled_delta_ml") - delta) > 1e-9
                    or abs(number(outcome["duration_s"], "duration_s") - duration) > 1e-9):
                raise ValueError("Outcome differs from actual endpoint measurements")
            peak = row["strict_assessment"]["maximum_all_label_spilled_particles"]
            if type(peak) is not int or peak < 0:
                raise ValueError("All-label peak spill is required")
            settled = bool(event.get("settled") is True
                and pre["source_airborne_particles"] == post["source_airborne_particles"] == 0
                and pre["quiet_measured_seconds"] >= .2 - 1e-9
                and post["quiet_measured_seconds"] >= .2 - 1e-9)
            safe = bool(row["strict_assessment"]["valid_evidence"] is True and peak <= 1)
            event_id = episode + ":response_" + str(len(samples))
            samples.append({"event_id": event_id, "row_id": event_id,
                "episode_id": episode, "seed": seed, "group_id": str(seed), "split": split,
                "physics_fingerprint": copy.deepcopy(row["physics_fingerprint"]),
                "target_ml": number(row["target_ml"], "target_ml"),
                "safe": safe, "settled": settled, "peak_spill_particles": peak,
                "state": {key: number(pre[key], key) for key in STATE_KEYS},
                "action": {key: number(action[key], key) for key in ACTION_KEYS},
                "outcome": {"settled_delta_ml": float(delta), "duration_s": float(duration)},
                "pre_physical_time_s": t0, "post_physical_time_s": t1,
                "pre_observation": pre, "post_observation": post,
                "declared_action": copy.deepcopy(action),
                "actual_action": copy.deepcopy(event.get("actual_action")),
                "task_feasible": bool(row["strict_assessment"]["feasible"])})
            pending = None
    return samples
