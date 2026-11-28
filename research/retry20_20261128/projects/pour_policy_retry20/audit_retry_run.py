"""Independent full evidence audit for the fixed20mL +/-0.5mL settled-correction curriculum.

The pinned original physical audit is reused unchanged. Only the experiment
factory ownership is parameterized. Original legacy/0.1% assessments remain
intact; this new criterion and its train/validation selection are recomputed.
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import types


ROOT = Path(__file__).resolve().parents[2]
HERE = "projects/pour_policy_retry20/"
PARENT = "projects/pour_policy_adaptive_learning/audit_prefix_calibration.py"
PARENT_SHA = "d0e6acc33e93baa57edb6e8b689f79d7b5d251821bc20a8167b192ee49d9aec6"
KIND = "retry20_conditional_second_cycle_v1"
SCOPE = "record_integrity_and_independent_physical_fixed20_retry_task_audit_not_real_robot_certification"
CONTROLLER = HERE + "retry_control.py"
CONTROLLER_SHA = "3110775a203c538b51c563603409cc035fb547c24faf7f5de4bc7ada3b60255b"
COLLECTOR = HERE + "run_retry_test.py"
PILOT_SEED = 195301
TEST = tuple(range(196301, 196311))
COUNTS = {"pilot": 1, "test": 20}
PLAN_SHA = "fb91ad34f86e1298885fefc3b00c18b8077342916d53169906ddf98cf35db3b2"
ARTIFACTS = ("protocol.json", "summary.json", "job_state.json", "episodes.jsonl",
             "responses.jsonl", "matrix.json", "versions.json")
COARSE = {"prepare_fraction": 1., "dose_tilt_deg_s": 25.6, "return_fraction": 1.,
          "tail_lead_s": .26, "stop_bias_ml": 10., "max_dose_u_accel": 6.}
MOTION = {"approach_tilt_deg_s": 6., "action_delta_deg": .03,
          "action_tilt_deg_s": .6, "hold_seconds": .1, "backoff_delta_deg": .5,
          "backoff_tilt_deg_s": 4., "anchor_return_tilt_deg_s": 6.,
          "quiet_seconds": .25, "maximum_measurement_hold_s": 1., }


class AuditError(ValueError):
    pass


def need(value, message):
    if not value:
        raise AuditError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
        allow_nan=False).encode()).hexdigest()


def read(path):
    def reject(value):
        raise AuditError("Nonfinite JSON constant: " + value)
    return json.loads(Path(path).read_text(encoding="utf8"), parse_constant=reject)


def utc(value):
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    need(parsed.tzinfo is not None and parsed.utcoffset() is not None, "Explicit UTC freeze/reset time required")
    return parsed.astimezone(timezone.utc)


def load_physical_audit(root=ROOT):
    sys.path.insert(0, str(root / "projects/pour_policy_adaptive_learning"))
    path = root / PARENT
    need(sha(path) == PARENT_SHA, "Pinned original physical audit changed")
    spec = importlib.util.spec_from_file_location("refine20_pinned_prefix_audit", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    # Reuse the identical compiled numerical audit. Only ownership and the
    # original whole-cycle endpoint binder differ from the prefix-only wrapper.
    namespace = dict(module.audit_episode.__globals__)
    namespace.update(COLLECTOR_SOURCE=COLLECTOR, CONTROLLER_SOURCE=CONTROLLER, CYCLE_KINDS=(KIND,),
                     cycle_bindings=module.parent.cycle_bindings)
    episode = types.FunctionType(module.audit_episode.__code__, namespace, "audit_refine_physical_episode",
                                 module.audit_episode.__defaults__, module.audit_episode.__closure__)
    return module, episode


def tolerance_assessment(result, strict, tolerance):
    target, received = result["target_ml"], result["received_ml"]
    need(type(target) in (int, float) and target == 20.
         and type(received) in (int, float) and math.isfinite(received), "Actual fixed20mL dose required")
    need(tolerance in (.5, 1.), "Only declared primary or historical easy tolerances may be used")
    dose = 20. - tolerance <= received <= 20. + tolerance
    task = ("fixed20ml_absolute05ml_24s_onegrain_v1" if tolerance == .5
            else "fixed20ml_absolute1ml_24s_onegrain_v1")
    answer = {**strict, "dose_passed": dose, "tolerance_ml": tolerance, "task_kind": task}
    answer["feasible"] = bool(strict["valid_evidence"] and dose and strict["full_home_within_deadline"]
        and strict["spill_passed"] and strict["no_late_dose_change"])
    return answer


class ConservativeBudget:
    """Independent cache calculus and dense C2 joint-limit planning; no runtime imports."""

    def __init__(self, geometry):
        self.g, self.np = geometry, geometry.np
        self.cache = {}

    def progress(self, time_s, duration):
        t, r = min(duration, max(0., time_s)), self.g.ramp
        peak = 1. / (duration - r)
        if t <= r:
            return (self.g.progress(t, duration), .5 * peak * (1. - math.cos(math.pi * t / r)),
                    .5 * peak * math.pi / r * math.sin(math.pi * t / r))
        if t < duration - r:
            return self.g.progress(t, duration), peak, 0.
        left = duration - t
        return (self.g.progress(t, duration), .5 * peak * (1. - math.cos(math.pi * left / r)),
                -.5 * peak * math.pi / r * math.sin(math.pi * left / r))

    def polynomial(self, grid, coefficients, value, derivative):
        index = int(self.np.clip(self.np.searchsorted(grid, value, side="right") - 1, 0, len(grid) - 2))
        h = float(grid[index + 1] - grid[index])
        x = (value - grid[index]) / h
        return sum(coefficients[index, power] * math.factorial(power) / math.factorial(power - derivative)
                   * x ** (power - derivative) / h ** derivative for power in range(derivative, 6))

    def derivatives(self, u):
        g, d = self.g, self.g.data
        if u <= g.lift:
            p, v, a = self.progress(u, g.lift)
            first = self.polynomial(d["lift_progress_grid"], d["lift_joint_coefficients"], p, 1)
            second = self.polynomial(d["lift_progress_grid"], d["lift_joint_coefficients"], p, 2)
            return first * v, second * v * v + first * a
        if u <= g.prepare:
            _, v, a = self.progress(u - g.lift, g.prepare - g.lift)
            delta = d["theta_joint_q"][0] - d["q_lift_end"]
            return delta * v, delta * a
        p, v, a = self.progress(u - g.prepare, g.maximum - g.prepare)
        angles = d["theta_rad"]
        delta = float(angles[-1] - angles[0])
        theta, velocity, acceleration = float(angles[0]) + delta * p, delta * v, delta * a
        first = self.polynomial(angles, d["theta_joint_coefficients"], theta, 1)
        second = self.polynomial(angles, d["theta_joint_coefficients"], theta, 2)
        return first * velocity, second * velocity * velocity + first * acceleration

    def tilt(self, u):
        tube = self.g.hand(self.g.q(u)) @ self.g.data["hand_to_tube"]
        return math.degrees(math.acos(float(self.np.clip(tube[2, 2], -1., 1.))))

    def coordinate(self, angle):
        if angle == 78.:
            return self.g.prepare
        low, high = 0., self.g.maximum
        for _ in range(48):
            middle = .5 * (low + high)
            if self.tilt(middle) < angle:
                low = middle
            else:
                high = middle
        return .5 * (low + high)

    def step(self, start, goal, rate):
        delta_angle = abs(self.tilt(goal) - self.tilt(start))
        duration = max(.1, 1.875 * delta_angle / rate, 1.875 * abs(goal - start))
        for _ in range(30):
            speed = acceleration = clock_speed = tilt_speed = 0.
            for time_s in self.np.linspace(0., duration, max(81, int(math.ceil(duration * 240.)) + 1)):
                x = min(1., max(0., float(time_s / duration)))
                smooth = x ** 3 * (10. - 15. * x + 6. * x * x)
                first_clock = 30. * x * x * (1. - x) ** 2 / duration
                second_clock = 60. * x * (1. - x) * (1. - 2. * x) / duration ** 2
                u = start + (goal - start) * smooth
                velocity, change = (goal - start) * first_clock, (goal - start) * second_clock
                first, second = self.derivatives(u)
                speed = max(speed, float(self.np.max(self.np.abs(first * velocity) / self.g.limits)))
                acceleration = max(acceleration,
                    float(self.np.max(self.np.abs(second * velocity * velocity + first * change))))
                clock_speed = max(clock_speed, abs(velocity))
                if u > self.g.prepare:
                    _, theta_rate, _ = self.progress(u - self.g.prepare, self.g.maximum - self.g.prepare)
                    angle_rate = math.degrees(float(self.g.data["theta_rad"][-1] - self.g.data["theta_rad"][0]) * theta_rate)
                else:
                    low, high = max(0., u - 1e-5), min(self.g.prepare, u + 1e-5)
                    angle_rate = (self.tilt(high) - self.tilt(low)) / (high - low)
                tilt_speed = max(tilt_speed, abs(angle_rate * velocity))
            if speed <= .97 and acceleration <= 23. and clock_speed <= 1. and tilt_speed <= rate * (1. + 1e-10):
                return duration
            duration *= max(1.05, speed / .97, math.sqrt(acceleration / 23.), clock_speed, tilt_speed / rate)
        raise AuditError("Independent planner found no bounded C2 step")

    def planned(self, angle):
        if angle not in self.cache:
            u, total = self.g.prepare, 0.
            for target, rate in zip((angle, angle + .03, angle + .03 - .5, 78.), (6., .6, 4., 6.)):
                goal = self.coordinate(target)
                total += self.step(u, goal, rate)
                u = goal
            self.cache[angle] = total + .1 + 3. * 1. + self.g.prepare + .45 + .15
        return self.cache[angle]


def validate_schedule(cases, stage):
    need(stage in COUNTS and isinstance(cases, list) and len(cases) == COUNTS[stage],
         "The declared1+20 bounded fixed-policy budget must remain complete")
    seen, conditions = set(), []
    split = "development" if stage == "pilot" else "test"
    for case in cases:
        name = case["case_id"]
        need(isinstance(name, str) and name and name not in seen
             and all(c in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in name),
             "Unique safe case identities required")
        seen.add(name)
        need(case["target_ml"] == 20. and type(case["seed"]) is int and case["split"] == split
             and set(case["policy"]) == {"enable_retry"} and type(case["policy"]["enable_retry"]) is bool,
             "Stage changed actual goal/split or the one declared fixed-policy flag")
        conditions.append((case["role"], case["policy"]["enable_retry"], case["seed"]))
    expected = ([("candidate", True, PILOT_SEED)] if stage == "pilot" else
        [(role, role == "candidate", seed) for role in ("baseline", "candidate") for seed in TEST])
    need(Counter(conditions) == Counter(expected), "Not the complete declared pilot or paired new held-out conditions")


def validate_retry_status(status, policy, controller_sha):
    maximum = 2 if policy["enable_retry"] else 1
    need(status["retry_controller_source_sha256"] == controller_sha
         and status["enable_retry"] is policy["enable_retry"]
         and status["retry_enabled"] is policy["enable_retry"]
         and status["retry_first_approach_tilt_deg"] == 84.
         and status["retry_second_approach_tilt_deg"] == 84.1
         and status["retry_maximum_cycles"] == maximum,
         "Actual fixed retry ownership/profile differs")
    need(status["actual_target_ml"] == status["control_target_ml"] == status["target_ml"] == 20.
         and status["control_target_offset_ml"] == status["prefix_target_offset_ml"] == 0.
         and status["actual_measurement_target_unchanged"] is True,
         "Retry relabelled actual water or changed its paired first coarse prefix")
    parameters = status["policy_params"]
    for key, value in {**COARSE, **MOTION}.items():
        need(type(parameters[key]) in (int, float) and parameters[key] == value,
             "Fixed speed/micro-motion changed: " + key)
    need(parameters["maximum_cycles"] == maximum and parameters["enable_retry"] is policy["enable_retry"]
         and parameters["enable_correction"] is True and parameters["first_approach_tilt_deg"] == 84.
         and parameters["retry_approach_tilt_deg"] == 84.1 and parameters["approach_tilt_deg"] == 84.
         and parameters["retry_actual_deficit_threshold_ml"] == .5
         and parameters["retry_maximum_all_label_peak_spill_particles"] == 1
         and parameters["model_checkpoint_sha256"] is None and parameters["model_action_menu"] is None
         and parameters["no_angle_training_or_selection"] is True and status["response_model_used"] is False,
         "Fixed strategy became a speed/model search or used a different retry threshold")
    count = status["response_cycle_count"]
    need(type(count) is int and 0 <= count <= maximum, "Actual first/retry cycle budget changed")
    starts = [event for event in status["response_events"] if event["type"] == "response_cycle_start"]
    angles = [float(event["approach_tilt_deg"]) for event in starts]
    need(angles == [84., 84.1][:len(angles)] and len(angles) == count
         and status["retry_actual_cycle_approach_tilt_degrees"] == angles
         and status["retry_actual_approach_tilt_deg"] == (angles[-1] if angles else None),
         "Actual execution was inferred from an intent or the fixed second angle changed")
    received = status["received_ml"]
    need(status["retry_actual_dose_in_closed_band"] is (19.5 <= received <= 20.5)
         and status["original_strict_actual_dose_in_closed_band"] is (19.98 <= received <= 20.02),
         "Primary and original strict measurements were conflated")
    action = status["policy_action"]["coarse_action"]
    predicted = received + status["inflight_ml"] + max(0., status["flow_ml_s"]) * .26
    need(abs(action["predicted_final_ml"] - predicted) <= 1e-8
         and abs(action["remaining_committed_ml"] - (20. - received - status["inflight_ml"])) <= 1e-8
         and action["requested_tilt_deg_s"] == 25.6 and action["stop"] is (predicted + 10. >= 20.),
         "Original coarse stop law changed in the fixed first-cycle comparison")


def check_decision(decision, event, step, detail, budget, parent):
    """Intent is checked against same-frame truth and independent cache planning."""
    time_s, received = step["physical_time_s"], step["truth"]["received_ml"]
    parent.observe_endpoint(event["observation"], step, detail, "retry_decision")
    status, peak = detail["controller_status"], detail["maximum_all_label_spilled_particles"]
    quiet = bool(status["settled_measurement"])
    deficient = received < 19.5
    fits = time_s + budget <= 24.
    allowed = quiet and deficient and peak <= 1 and fits
    reason = ("actual_quiet_undershoot_retry_allowed" if allowed else
        "retry20_current_spill_evidence_missing_or_excess" if peak > 1 else
        "retry20_full_home_motion_budget_exceeds_deadline")
    expected = {"physical_time_s": time_s, "completed_cycles_before_decision": 1,
        "actual_received_ml": received, "actual_remaining_ml": step["truth"]["remaining_ml"],
        "actual_deficit_ml": 20. - received, "actual_settled_measurement": quiet,
        "deficit_exceeds_half_ml": deficient, "observed_all_label_peak_spill_particles": peak,
        "spill_guard_passed": peak <= 1, "declared_retry_approach_tilt_deg": 84.1,
        "deadline_physical_seconds": 24., "remaining_deadline_physical_seconds": 24. - time_s,
        "full_home_budget_fits": fits, "retry_allowed": allowed, "reason": reason,
        "decision_is_intent_not_execution": True, "actual_execution_evidence_event_type": "response_cycle_start",
        "budget_is_robot_motion_not_predicted_water_dose": True}
    need(set(decision) == {*expected, "conservative_full_home_budget_s"}, "Retry decision scope changed")
    for key, value in expected.items():
        if type(value) in (int, float):
            parent.close(decision[key], value, "Actual retry condition: " + key, 1e-8)
        else:
            need(type(decision[key]) is type(value) and decision[key] == value,
                 "Actual retry condition differs: " + key)
    parent.close(decision["conservative_full_home_budget_s"], budget,
                 "Independent conservative full-home budget", 1e-7)
    need(quiet and deficient and event["observation"]["source_airborne_particles"] == 0
         and event["observation"]["quiet_measured_seconds"] >= .25 - 1e-9,
         "Conditional decision lacked an actual quiet deficit")
    parent.close(event["observation"]["tilt_deg"], 78., "Retry decision real78 anchor", 1e-6)
    return allowed


def retry_bindings(row, report, plan, parent, geometry):
    policy, controller_sha = row["declared_policy"], plan["source_sha256"][CONTROLLER]
    recorded, previous_decisions, prior_home = None, [], False
    frames = {int(round(step["physical_time_s"] * 60.)): (step, detail)
              for step, detail in zip(report["raw_steps"], report["raw_direct"])}
    for step, detail in zip(report["raw_steps"], report["raw_direct"]):
        status = detail["controller_status"]
        validate_retry_status(status, policy, controller_sha)
        anchor = status.get("prefix_anchor_observation")
        if anchor is not None:
            if recorded is None:
                parent.close(anchor["physical_time_s"], step["physical_time_s"], "First actual coarse anchor", 1e-8)
                parent.observe_endpoint(anchor, step, detail, "retry_coarse_anchor")
                parent.close(anchor["tilt_deg"], 78., "Actual first78 anchor", 1e-6)
                need(anchor["settled_measurement"] is True and anchor["source_airborne_particles"] == 0
                     and anchor["quiet_measured_seconds"] >= .25 - 1e-9, "First anchor lacks quiet measurement")
            else:
                need(anchor == recorded, "Actual first anchor was rewritten")
            recorded = anchor
        else:
            need(recorded is None, "Actual coarse anchor disappeared")
        decisions = status["retry_decisions"]
        need(isinstance(decisions, list) and len(decisions) <= 1
             and decisions[:len(previous_decisions)] == previous_decisions,
             "Conditional decision history was rewritten")
        previous_decisions = decisions
        home = status["retry_actual_settled_home_decision"]
        need(type(home) is bool and (not prior_home or home), "Actual settled home decision was rewritten")
        if home and not prior_home:
            need(status["settled_measurement"] is True and 19.5 <= status["received_ml"] <= 20.5,
                 "Home completion claimed by a predicted or unsettled dose")
        prior_home = home
    need(recorded == row["status"].get("prefix_anchor_observation"), "Final retained coarse anchor differs")
    if row["result"]["complete"]:
        need(isinstance(recorded, dict), "Completed action lacks its actual first quiet anchor")
    starts = [event for event in row["response_events"] if event["type"] == "response_cycle_start"]
    decisions = [event for event in row["response_events"] if event["type"] == "retry20_conditional_second_cycle_decision"]
    need(len(starts) <= (2 if policy["enable_retry"] else 1) and len(decisions) <= 1,
         "Fixed one/conditional-two-cycle budget changed")
    for index, start in enumerate(starts):
        angle = 84. if index == 0 else 84.1
        need(start["cycle_index"] == index + 1 and start["approach_tilt_deg"] == angle,
             "Actual first or second angle was selected after observing a test result")
        expected = {"approach_delta_deg": angle - 78., "approach_speed_deg_s": 6.,
            "delta_deg": .03, "tilt_speed_deg_s": .6, "hold_s": .1, "retract_deg": .5,
            "retract_speed_deg_s": 4., "cutoff_tilt_deg": 78., "cutoff_speed_deg_s": 6.,
            "quiet_seconds": .25, "maximum_measurement_hold_s": 1.}
        need(set(start["action"]) == set(expected), "Actual whole-cycle action scope changed")
        for key, value in expected.items():
            parent.close(start["action"][key], value, "Fixed executed cycle action: " + key, 1e-8)
    allowed, budget = False, None
    if decisions:
        need(policy["enable_retry"] is True, "One-cycle baseline attempted conditional retry")
        event = decisions[0]
        earlier = [item for item in row["response_events"] if item["type"] == "response_cycle_complete"
                   and item["event_id"] < event["event_id"]]
        need(len(earlier) == 1 and earlier[0]["settled"] is True,
             "Retry was not preceded by exactly one actually quiet completed cycle")
        parent.close(earlier[0]["physical_time_s"], event["physical_time_s"], "Same-time actual remeasurement", 1e-8)
        frame = int(round(event["physical_time_s"] * 60.))
        need(frame in frames, "Retry decision has no actual physical frame")
        budget = ConservativeBudget(geometry).planned(84.1)
        allowed = check_decision(event["retry_decision"], event, *frames[frame], budget, parent)
        need(previous_decisions == [event["retry_decision"]], "Event and retained retry decision differ")
        if allowed:
            need(len(starts) == 2 and starts[1]["event_id"] > event["event_id"],
                 "Allowed intent was falsely presented as an actually executed second cycle")
            parent.close(starts[1]["physical_time_s"], event["physical_time_s"], "Actual second-cycle commit", 1e-8)
        else:
            need(len(starts) == 1, "Rejected retry still executed a second cycle")
    else:
        need(previous_decisions == [] and len(starts) <= 1, "Actual second cycle lacks conditional decision evidence")
    completed = [event for event in row["response_events"] if event["type"] == "response_cycle_complete"]
    if (policy["enable_retry"] and len(completed) == 1 and len(starts) == 1
            and completed[0]["post_observation"]["received_ml"] < 19.5 and row["result"]["complete"]):
        need(decisions, "Completed underfill bypassed its declared actual-volume retry decision")
    return {"first_coarse_anchor": recorded, "conditional_decisions": [item["retry_decision"] for item in decisions],
        "retry_intent_allowed": allowed, "actual_cycles_started": len(starts),
        "actual_second_cycle_executed": len(starts) == 2,
        "actual_cycles_completed": len(completed), "independent_retry_full_home_budget_s": budget,
        "actual_second_cycle_complete": len(completed) == 2}


def reference_equivalence(row, report, root, ref, experiment_plan):
    """The seen pilot is a repeat of one failed case, never a new hold-out result."""
    bindings = experiment_plan["pilot_reference_sha256"]
    need(isinstance(bindings, dict) and len(bindings) == 4, "Four original seen-case artifact bindings required")
    for name, digest in bindings.items():
        need(sha(ref.inside(root, name)) == digest, "Original seen failed-case bytes changed: " + name)
    old_rows = [name for name in bindings if "/completed/" in name]
    old_steps = [name for name in bindings if name.endswith(".steps.jsonl.gz")]
    need(len(old_rows) == len(old_steps) == 1, "Ambiguous old failed-case journal or physical trace")
    old = read(ref.inside(root, old_rows[0]))
    old_protocol_paths = [name for name in bindings if name.endswith("/protocol.json")]
    need(len(old_protocol_paths) == 1, "One frozen original protocol required")
    old_protocol = read(ref.inside(root, old_protocol_paths[0]))
    liquid_name = "assets/robot_testtube_pour/scripts/liquid.py"
    counter_source_sha = old_protocol["source_sha256"][liquid_name]
    need(sha(ref.inside(root, liquid_name)) == counter_source_sha,
         "Frozen source for the zero-time inherited counter explanation changed")
    need(old["seed"] == row["seed"] == PILOT_SEED and old["target_ml"] == row["target_ml"] == 20.
         and old["declared_policy"] == {"enable_correction": True, "approach_tilt_deg": 84.}
         and old["refine_assessment"]["feasible"] is False,
         "Known development revisit was relabelled as a fresh test or successful reference")
    old_events = old["response_events"]
    old_end = next((event for event in old_events if event["type"] == "response_cycle_complete"), None)
    new_end = next((event for event in row["response_events"] if event["type"] == "response_cycle_complete"), None)
    if old_end is None or new_end is None:
        return {"seen_development_case_not_independent": True, "source_reference_sha256": bindings,
            "first_cycle_water_motion_partitions_exactly_equal": False,
            "first_cycle_original_events_exactly_equal": False, "endpoint_unavailable": True}
    events_equal = old_end == new_end and old_events[:old_end["event_id"] + 1] == row["response_events"][:new_end["event_id"] + 1]
    terminal = old_end["physical_time_s"]
    original, _ = ref.gz(ref.inside(root, old_steps[0]))
    old_truth = [item for item in original if item["type"] == "step" and item["physical_time_s"] <= terminal]
    old_direct = [item for item in original if item["type"] == "direct_step_evidence"
                  and item["physical_time_s"] <= terminal]
    new_truth = [item for item in report["raw_steps"] if item["physical_time_s"] <= terminal]
    new_direct = [item for item in report["raw_direct"] if item["physical_time_s"] <= terminal]
    physical_equal = len(old_truth) == len(old_direct) == len(new_truth) == len(new_direct)
    for one, two in zip(old_truth, new_truth):
        for key in ("step", "physical_time_s", "physics_dt_s", "source_time", "truth"):
            physical_equal = physical_equal and one[key] == two[key]
        # last_substeps is not reset in the frozen liquid.reset implementation.
        # At time0 it describes the previous episode/constructor, before any
        # actual step. Preserve both values while comparing every real frame.
        if one["step"] != 0:
            physical_equal = physical_equal and one["solver_substeps"] == two["solver_substeps"]
        else:
            need(one["physical_time_s"] == two["physical_time_s"] == 0.
                 and one["source_time"] == two["source_time"] == 0.
                 and one["truth"]["received_ml"] == two["truth"]["received_ml"] == 0.
                 and one["solver_substeps"] >= 4 and two["solver_substeps"] >= 4,
                 "Zero-time inherited substep counter was confused with an actual physical frame")
    for one, two in zip(old_direct, new_direct):
        for key in ("step", "physical_time_s", "kinematics", "partitions", "maximum_all_label_spilled_particles"):
            physical_equal = physical_equal and one[key] == two[key]
    return {"seen_development_case_not_independent": True, "source_reference_sha256": bindings,
        "original_failed_received_ml": old["result"]["received_ml"],
        "first_cycle_end_physical_s": terminal, "first_cycle_frames": len(old_truth),
        "zero_time_inherited_last_substeps": {"reference": old_truth[0]["solver_substeps"],
            "pilot": new_truth[0]["solver_substeps"], "is_actual_completed_frame": False,
            "excluded_from_equivalence_only": True, "source_file": liquid_name,
            "source_sha256": counter_source_sha,
            "reason": "frozen reset clears frame count but retains last_substeps; step_frame recomputes each real frame from actual velocity"},
        "all_completed_frame_substep_counts_exactly_equal": all(one["solver_substeps"] == two["solver_substeps"]
            for one, two in zip(old_truth[1:], new_truth[1:])),
        "first_cycle_water_motion_partitions_exactly_equal": physical_equal,
        "first_cycle_original_events_exactly_equal": events_equal}


def audit_stage(run, stage, *, root=ROOT, parent_plan_sha=None):
    run, root = Path(run).resolve(), Path(root).resolve()
    summary, job = read(run / "summary.json"), read(run / "job_state.json")
    need(summary["complete"] is True and summary["episodes"] == COUNTS[stage]
         and summary["stage"] == stage and job["state"] == "complete" and job["pid"] is None
         and not (run / "worker.lock").exists(), "Complete immutable stage required before raw-journal audit")
    physical, audit_episode = load_physical_audit(root)
    ref = physical.load_cpu_reference(root)
    plan = read(run / "protocol.json")
    need(plan["kind"] == KIND and summary["kind"] == KIND
         and plan["controller_source"] == CONTROLLER and plan["collector_source"] == COLLECTOR,
         "New experiment/controller/factory ownership differs")
    need(signature({key: value for key, value in plan.items() if key != "protocol_id"}) == plan["protocol_id"]
         == summary["protocol_id"], "Stage canonical protocol identity differs")
    matrix = read(run / "matrix.json")
    need(matrix == plan["matrix"] and matrix["draft"] is False and matrix["stage"] == stage,
         "Actual stage matrix differs from frozen protocol")
    if parent_plan_sha is not None:
        need(matrix["parent_plan_sha256"] == parent_plan_sha, "Stage no longer identifies preregistered total experiment")
    validate_schedule(matrix["cases"], stage)
    expected = {"deadline_s": 24., "relative_tolerance": .001, "refine_absolute_tolerance_ml": .5, "easy_absolute_tolerance_ml": 1.,
        "peak_all_label_spill_limit": 1, "spacing_m": .0027, "particle_ml": .019683,
        "total_particles": 11625, "source_particles": 2880, "initial_source_ml": 56.68704, "post_seconds": 2.}
    for key, value in expected.items():
        physical.close(plan[key], value, "Fixed easy20 task/physics " + key)
    need(plan["refine_task_kind"] == "fixed20ml_absolute05ml_24s_onegrain_v1"
         and plan["original_strict_standard_unchanged"] is True
         and plan["solver"] == "unchanged_live_PBF_60Hz_pressure6_min4_graph_off"
         and plan["no_saved_liquid_replay"] is True and plan["phase_training_time_not_hidden"] is True,
         "Easy tolerance was confused with a change to original strict evidence or physics")
    need(len(plan["source_sha256"]) == 28 and len(plan["input_sha256"]) == 6
         and len(plan["gui_sha256"]) == 3 and plan["source_sha256"][CONTROLLER] == CONTROLLER_SHA,
         "Effective source/input/old-GUI count or controller changed")
    for group, folder in (("source_sha256", "source_snapshot"), ("input_sha256", "input_snapshot")):
        for name, expected_sha in plan[group].items():
            need(sha(ref.inside(root, name)) == sha(ref.inside(run / folder, name)) == expected_sha,
                 "Current/snapshot source/input changed: " + name)
    for name, expected_sha in plan["gui_sha256"].items():
        need(sha(ref.inside(root, name)) == expected_sha, "Original GUI model changed")
    need(COLLECTOR in plan["source_sha256"] and HERE + "task_score.py" in plan["source_sha256"],
         "New runner/task scorer absent from source freeze")
    rows = [json.loads(line, parse_constant=ref.reject_constant)
            for line in (run / "episodes.jsonl").read_text(encoding="utf8").splitlines() if line.strip()]
    ids = [case["case_id"] for case in matrix["cases"]]
    need([row["episode_id"] for row in rows] == ids and len(set(ids)) == len(ids)
         and {path.stem for path in (run / "completed").glob("*.json")} == set(ids),
         "Unique journal/matrix/sidecar case set differs")
    fingerprint = hashlib.sha256(json.dumps({"sources": plan["source_sha256"],
        "inputs": plan["input_sha256"], "spacing": .0027}, sort_keys=True).encode()).hexdigest()
    geometry = ref.Geometry(run, root, plan)
    reports, expected_artifacts = [], set()
    experiment_plan = validate_experiment_plan(run.parent)
    for row, case in zip(rows, matrix["cases"]):
        need(row["stage"] == stage and row["split"] == ("development" if stage == "pilot" else stage) and row["role"] == case["role"]
             and row["protocol_id"] == plan["protocol_id"] and row["physics_fingerprint"] == fingerprint,
             "Actual stage/role/source/physics differs")
        report = audit_episode(run, row, plan, geometry, ref)
        retry = retry_bindings(row, report, plan, physical.parent, geometry)
        if stage == "pilot":
            retry["seen_case_first_cycle_reference"] = reference_equivalence(row, report, root, ref, experiment_plan)
        audited = physical.audit_post(run, row, report, ref)
        for item in ref.gz(ref.inside(run, report["post_name"]))[0] if report["post_name"] in row["artifacts"] else []:
            validate_retry_status(item["controller_status"], case["policy"], plan["source_sha256"][CONTROLLER])
            need(item["controller_status"].get("prefix_anchor_observation")
                 == row["status"].get("prefix_anchor_observation"), "Actual initial anchor changed during post")
        primary = tolerance_assessment(row["result"], audited["strict_assessment"], .5)
        easy = tolerance_assessment(row["result"], audited["strict_assessment"], 1.)
        need(primary == row["refine_assessment"] and easy == row["easy_assessment"],
             "Primary/easy assessments differ from actual unchanged original-target evidence")
        audited.update(stage=stage, role=row["role"], refine_assessment=primary, easy_assessment=easy,
            declared_policy=case["policy"], actual_retry_evidence=retry, artifacts=row["artifacts"])
        reports.append(audited)
        expected_artifacts.update(row["artifacts"])
    need({path.relative_to(run).as_posix() for path in (run / "traces").glob("*.jsonl.gz")} == expected_artifacts,
         "An extra or unjournaled physical attempt exists")
    data = [sample for row in rows for sample in row["response_samples"]]
    records = [json.loads(line, parse_constant=ref.reject_constant)
               for line in (run / "responses.jsonl").read_text(encoding="utf8").splitlines() if line.strip()]
    need(data == records and summary["response_samples"] == len(data)
         and summary["responses_sha256"] == sha(run / "responses.jsonl")
         and summary["new_goal_policy_trained"] is False and summary["gui_deployment"] is False,
         "Cycle response records or non-deployment statement differ")
    for key in ("strict", "refine", "easy"):
        need(summary[key + "_task_passes"] == sum(row[key + "_assessment"]["feasible"] for row in rows),
             "Original/primary/secondary task counts were conflated")
    safety = all(report["strict_assessment"]["valid_evidence"]
        and report["strict_assessment"]["full_home_within_deadline"]
        and report["strict_assessment"]["spill_passed"]
        and report["strict_assessment"]["no_late_dose_change"] for report in reports)
    pilot_gate = None
    if stage == "pilot":
        actual = reports[0]["actual_retry_evidence"]
        reference = actual["seen_case_first_cycle_reference"]
        pilot_gate = bool(reports[0]["refine_assessment"]["feasible"]
            and actual["actual_second_cycle_executed"] and actual["actual_second_cycle_complete"]
            and actual["retry_intent_allowed"] and reference["first_cycle_water_motion_partitions_exactly_equal"]
            and reference["first_cycle_original_events_exactly_equal"])
    return {"stage": stage, "protocol_id": plan["protocol_id"], "episodes_audited": len(rows),
        "pilot_gate_passed": pilot_gate,
        "all_non_dose_safety_passed": safety, "fixed_policy_not_selected_from_test": True,
        "refine_task_passes": summary["refine_task_passes"], "easy_task_passes": summary["easy_task_passes"], "original_strict_task_passes": summary["strict_task_passes"],
        "source_sha256": plan["source_sha256"], "input_sha256": plan["input_sha256"],
        "gui_sha256": plan["gui_sha256"], "versions": read(run / "versions.json"),
        "input_artifact_sha256": {name: sha(run / name) for name in ARTIFACTS},
        "episode_reports": reports, "rows": rows}


def validate_experiment_plan(experiment):
    path = experiment / "experiment_plan.json"
    need(sha(path) == PLAN_SHA, "Preregistered fixed strategy bytes changed")
    plan = read(path)
    need(plan["kind"] == "retry20_fixed_policy_preregistration_v1" and plan["draft"] is False
         and plan["budget"] == 21 and plan["budgets"] == COUNTS and plan["target_ml"] == 20.
         and plan["absolute_tolerance_ml"] == .5 and plan["easy_absolute_tolerance_ml"] == 1.
         and plan["deadline_s"] == 24. and plan["peak_all_label_spill_limit"] == 1 and plan["post_seconds"] == 2.
         and plan["pilot_seed"] == PILOT_SEED and plan["pilot_is_seen_development_revisit"] is True
         and tuple(plan["test_seeds"]) == TEST and plan["profiles"] == {
             "baseline": {"enable_retry": False}, "candidate": {"enable_retry": True}}
         and plan["first_approach_tilt_deg"] == 84. and plan["retry_approach_tilt_deg"] == 84.1
         and plan["maximum_cycles_baseline"] == 1 and plan["maximum_cycles_candidate"] == 2
         and plan["freeze_fixed_strategy_before_fresh_tests"] is True
         and plan["data_fitting_or_model_selection"] is False
         and plan["test_may_not_select_or_update_model"] is True
         and plan["original_strict_target_relative_tolerance"] == .001
         and plan["default_model_deployment"] is False,
         "Fixed new strategy, pilot provenance or declared twenty-case test coverage changed")
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", "--experiment", dest="experiment", type=Path, required=True)
    parser.add_argument("--stage", choices=tuple(COUNTS), required=True)
    parser.add_argument("--workspace", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    experiment = args.experiment.resolve()
    validate_experiment_plan(experiment)
    result = audit_stage(experiment / args.stage, args.stage,
                         root=args.workspace, parent_plan_sha=PLAN_SHA)
    result.pop("rows")
    result.update(complete=True, **{"pass": True}, kind=KIND, scope=SCOPE + "_stage_only",
        complete_experiment=False, experiment_plan_sha256=PLAN_SHA,
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        auditor_source_file=HERE + "audit_retry_run.py", auditor_source_sha256=sha(__file__),
        pinned_parent_audit_sha256=PARENT_SHA)
    snapshot_dir = args.output.parent / "audit_source_snapshot"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    raw = Path(__file__).read_bytes()
    snapshot = snapshot_dir / ("audit_retry_run_" + hashlib.sha256(raw).hexdigest() + ".py")
    if snapshot.exists():
        need(snapshot.read_bytes() == raw, "Audit implementation snapshot changed")
    else:
        with snapshot.open("xb") as stream:
            stream.write(raw)
    with args.output.open("x", encoding="utf8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({key: value for key, value in result.items()
                      if key not in ("episode_reports", "stage_reports")}, indent=2))


if __name__ == "__main__":
    main()
