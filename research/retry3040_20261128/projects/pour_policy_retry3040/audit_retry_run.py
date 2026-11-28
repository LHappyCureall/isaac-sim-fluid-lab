"""Independent full evidence audit for the30/40mL conditional-retry extension.

The pinned original physical audit is reused unchanged. Only the experiment
factory ownership and goal-profile bindings are parameterized. Original
legacy/0.1% assessments stay intact; this task adds a separate half-mL gate.
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
HERE = "projects/pour_policy_retry3040/"
PARENT = "projects/pour_policy_adaptive_learning/audit_prefix_calibration.py"
PARENT_SHA = "d0e6acc33e93baa57edb6e8b689f79d7b5d251821bc20a8167b192ee49d9aec6"
KIND = "retry3040_conditional_second_cycle_v1"
SCOPE = "record_integrity_and_independent_physical_fixed3040_retry_task_audit_not_real_robot_certification"
CONTROLLER = HERE + "retry_control.py"
CONTROLLER_SHA = "c71402904507995fbd8e08ad19991629ac085561725114adc72196aeaf61c0a2"
COLLECTOR = HERE + "run_retry_test.py"
PILOT_SEED = 197101
TEST = tuple(range(197301, 197311))
TARGETS = (30., 40.)
COUNTS = {"pilot_001": 4, "pilot_002": 4, "test": 40}
PLAN_SHA = "dc4fa45c82dcb30011b3f682cafb3b2506babc50803e97d32f1bd74bbcd79add"
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
    need(type(target) in (int, float) and target in TARGETS
         and type(received) in (int, float) and math.isfinite(received), "Actual30/40mL target required")
    need(tolerance in (.5, 1.), "Only the declared primary or historical easy tolerance may be used")
    task = ("fixed30_40ml_absolute05ml_24s_onegrain_v1" if tolerance == .5
            else "fixed30_40ml_absolute1ml_24s_onegrain_v1")
    output = {**strict, "dose_passed": target - tolerance <= received <= target + tolerance,
              "tolerance_ml": tolerance, "task_kind": task}
    output["feasible"] = bool(strict["valid_evidence"] and output["dose_passed"]
        and strict["full_home_within_deadline"] and strict["spill_passed"] and strict["no_late_dose_change"])
    return output


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


def validate_goal_profiles(profiles):
    need(isinstance(profiles, dict) and set(profiles) == {"30", "40"}, "Exactly two explicit goal profiles required")
    for goal, profile in profiles.items():
        need(isinstance(profile, dict) and set(profile) == {"first_approach_tilt_deg", "retry_approach_tilt_deg"},
             "Goal-specific calibration may change only the two approach angles")
        first, retry = profile["first_approach_tilt_deg"], profile["retry_approach_tilt_deg"]
        need(type(first) in (int, float) and type(retry) in (int, float)
             and math.isfinite(first) and math.isfinite(retry) and 78. <= first <= retry <= 99.97
             and abs(retry - first - .1) <= 1e-10, "Invalid declared first/retry angle geometry")


def validate_schedule(cases, stage, profiles):
    validate_goal_profiles(profiles)
    need(stage in COUNTS and isinstance(cases, list) and len(cases) == COUNTS[stage],
         "Complete paired four-case development or forty-case test budget required")
    seen, conditions = set(), []
    split = "test" if stage == "test" else "development"
    for case in cases:
        name = case["case_id"]
        need(isinstance(name, str) and name and name not in seen
             and all(c in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in name),
             "Unique safe case identities required")
        seen.add(name)
        target, policy = case["target_ml"], case["policy"]
        need(type(target) in (int, float) and target in TARGETS and type(case["seed"]) is int
             and case["split"] == split and isinstance(policy, dict)
             and set(policy) == {"enable_retry", "first_approach_tilt_deg", "retry_approach_tilt_deg"}
             and type(policy["enable_retry"]) is bool, "Actual target/split/action scope changed")
        need({key: value for key, value in policy.items() if key != "enable_retry"}
             == profiles[format(target, "g")], "Per-case declared action differs from the frozen goal profile")
        conditions.append((target, case["role"], policy["enable_retry"], case["seed"]))
    seeds = TEST if stage == "test" else (PILOT_SEED,)
    expected = [(target, role, role == "candidate", seed)
                for target in TARGETS for role in ("baseline", "candidate") for seed in seeds]
    need(Counter(conditions) == Counter(expected), "Deleted/reused or unmatched paired goal conditions")


def validate_retry_status(status, policy, controller_sha, target, profile_id, profile_sha):
    maximum = 2 if policy["enable_retry"] else 1
    need(status["retry3040_controller_source_sha256"] == controller_sha
         and status["enable_retry"] is policy["enable_retry"]
         and status["retry_enabled"] is policy["enable_retry"]
         and status["retry_first_approach_tilt_deg"] == policy["first_approach_tilt_deg"]
         and status["retry_second_approach_tilt_deg"] == policy["retry_approach_tilt_deg"]
         and status["retry_maximum_cycles"] == maximum
         and status["retry_profile_id"] == profile_id and status["retry_profile_sha256"] == profile_sha,
         "Actual fixed retry ownership/profile differs")
    need(status["actual_target_ml"] == status["control_target_ml"] == status["target_ml"] == target
         and status["control_target_offset_ml"] == status["prefix_target_offset_ml"] == 0.
         and status["actual_measurement_target_unchanged"] is True,
         "Retry relabelled actual water or changed its paired first coarse prefix")
    parameters = status["policy_params"]
    for key, value in {**COARSE, **MOTION}.items():
        need(type(parameters[key]) in (int, float) and parameters[key] == value,
             "Fixed speed/micro-motion changed: " + key)
    need(parameters["maximum_cycles"] == maximum and parameters["enable_retry"] is policy["enable_retry"]
         and parameters["enable_correction"] is True and parameters["first_approach_tilt_deg"] == policy["first_approach_tilt_deg"]
         and parameters["retry_approach_tilt_deg"] == policy["retry_approach_tilt_deg"]
         and parameters["approach_tilt_deg"] == policy["first_approach_tilt_deg"]
         and parameters["retry_actual_deficit_threshold_ml"] == .5
         and parameters["retry_maximum_all_label_peak_spill_particles"] == 1
         and parameters["model_checkpoint_sha256"] is None and parameters["model_action_menu"] is None
         and parameters["no_test_based_angle_or_strategy_selection"] is True
         and parameters["requested_target_ml"] == target
         and parameters["target_profile_key"] == format(target, "g")
         and parameters["target_profile_id"] == profile_id and parameters["target_profile_sha256"] == profile_sha
         and status["response_model_used"] is False,
         "Fixed strategy became a speed/model search or used a different retry threshold")
    count = status["response_cycle_count"]
    need(type(count) is int and 0 <= count <= maximum, "Actual first/retry cycle budget changed")
    starts = [event for event in status["response_events"] if event["type"] == "response_cycle_start"]
    angles = [float(event["approach_tilt_deg"]) for event in starts]
    need(angles == [policy["first_approach_tilt_deg"], policy["retry_approach_tilt_deg"]][:len(angles)] and len(angles) == count
         and status["retry_actual_cycle_approach_tilt_degrees"] == angles
         and status["retry_actual_approach_tilt_deg"] == (angles[-1] if angles else None),
         "Actual execution was inferred from an intent or the fixed second angle changed")
    received = status["received_ml"]
    need(status["retry_actual_dose_in_closed_band"] is (target - .5 <= received <= target + .5)
         and status["original_strict_actual_dose_in_closed_band"] is (target - .001 * target <= received <= target + .001 * target),
         "Primary and original strict measurements were conflated")
    action = status["policy_action"]["coarse_action"]
    predicted = received + status["inflight_ml"] + max(0., status["flow_ml_s"]) * .26
    need(abs(action["predicted_final_ml"] - predicted) <= 1e-8
         and abs(action["remaining_committed_ml"] - (target - received - status["inflight_ml"])) <= 1e-8
         and action["requested_tilt_deg_s"] == 25.6 and action["stop"] is (predicted + 10. >= target),
         "Original coarse stop law changed in the fixed first-cycle comparison")


def check_decision(decision, event, step, detail, budget, parent, target, angle, profile_id, profile_sha):
    """Intent is checked against same-frame truth and independent cache planning."""
    time_s, received = step["physical_time_s"], step["truth"]["received_ml"]
    parent.observe_endpoint(event["observation"], step, detail, "retry_decision")
    status, peak = detail["controller_status"], detail["maximum_all_label_spilled_particles"]
    quiet = bool(status["settled_measurement"])
    deficient = received < target - .5
    fits = time_s + budget <= 24.
    allowed = quiet and deficient and peak <= 1 and fits
    reason = ("actual_quiet_target_undershoot_retry_allowed" if allowed else
        "retry3040_current_spill_evidence_missing_or_excess" if peak > 1 else
        "retry3040_full_home_motion_budget_exceeds_deadline")
    expected = {"physical_time_s": time_s, "completed_cycles_before_decision": 1,
        "actual_received_ml": received, "actual_remaining_ml": step["truth"]["remaining_ml"],
        "declared_target_ml": target, "target_profile_key": format(target, "g"),
        "target_profile_id": profile_id, "target_profile_sha256": profile_sha,
        "actual_deficit_ml": target - received, "actual_settled_measurement": quiet,
        "deficit_exceeds_half_ml": deficient, "observed_all_label_peak_spill_particles": peak,
        "spill_guard_passed": peak <= 1, "declared_retry_approach_tilt_deg": angle,
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


def retry_bindings(row, report, plan, parent, geometry, profile):
    policy, controller_sha = row["declared_policy"], plan["source_sha256"][CONTROLLER]
    target = row["target_ml"]
    recorded, previous_decisions, prior_home = None, [], False
    frames = {int(round(step["physical_time_s"] * 60.)): (step, detail)
              for step, detail in zip(report["raw_steps"], report["raw_direct"])}
    for step, detail in zip(report["raw_steps"], report["raw_direct"]):
        status = detail["controller_status"]
        validate_retry_status(status, policy, controller_sha, target, profile["profile_id"], plan["goal_profile_raw_sha256"])
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
            need(status["settled_measurement"] is True and target - .5 <= status["received_ml"] <= target + .5,
                 "Home completion claimed by a predicted or unsettled dose")
        prior_home = home
    need(recorded == row["status"].get("prefix_anchor_observation"), "Final retained coarse anchor differs")
    if row["result"]["complete"]:
        need(isinstance(recorded, dict), "Completed action lacks its actual first quiet anchor")
    starts = [event for event in row["response_events"] if event["type"] == "response_cycle_start"]
    decisions = [event for event in row["response_events"] if event["type"] == "retry3040_conditional_second_cycle_decision"]
    need(len(starts) <= (2 if policy["enable_retry"] else 1) and len(decisions) <= 1,
         "Fixed one/conditional-two-cycle budget changed")
    for index, start in enumerate(starts):
        angle = policy["first_approach_tilt_deg"] if index == 0 else policy["retry_approach_tilt_deg"]
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
        budget = ConservativeBudget(geometry).planned(policy["retry_approach_tilt_deg"])
        allowed = check_decision(event["retry_decision"], event, *frames[frame], budget, parent, target,
                                 policy["retry_approach_tilt_deg"], profile["profile_id"], plan["goal_profile_raw_sha256"])
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
            and completed[0]["post_observation"]["received_ml"] < target - .5 and row["result"]["complete"]):
        need(decisions, "Completed underfill bypassed its declared actual-volume retry decision")
    return {"first_coarse_anchor": recorded, "conditional_decisions": [item["retry_decision"] for item in decisions],
        "retry_intent_allowed": allowed, "actual_cycles_started": len(starts),
        "actual_second_cycle_executed": len(starts) == 2,
        "actual_cycles_completed": len(completed), "independent_retry_full_home_budget_s": budget,
        "actual_second_cycle_complete": len(completed) == 2}


def bound_profile(run, root, plan, matrix, ref):
    name = matrix["goal_profile_file"]
    need(isinstance(name, str) and name.startswith(HERE) and name.endswith(".json"),
         "Goal profile must be an explicit JSON input in the new project")
    current = ref.inside(root, name)
    need(name in plan["input_sha256"] and sha(current) == sha(ref.inside(run / "input_snapshot", name))
         == plan["input_sha256"][name] == plan["goal_profile_raw_sha256"],
         "External goal profile is not byte-bound to current/snapshot/protocol input")
    profile = read(current)
    need(profile["kind"] == "retry3040_target_angle_profiles_v1"
         and isinstance(profile["profile_id"], str) and profile["profile_id"]
         and profile["goal_profiles"] == matrix["goal_profiles"] == plan["goal_profiles"]
         and profile["all_motion_speeds_fixed"] is True,
         "Actual runtime goal profiles differ from the declared external input")
    validate_goal_profiles(profile["goal_profiles"])
    return profile


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
    need(matrix["experiment_plan_sha256"] == plan["experiment_plan_sha256"] == PLAN_SHA,
         "Runtime stage lost its preregistered bounded-study identity")
    if parent_plan_sha is not None:
        need(matrix["experiment_plan_sha256"] == parent_plan_sha, "Stage no longer identifies preregistered total experiment")
    validate_schedule(matrix["cases"], stage, matrix["goal_profiles"])
    profile = bound_profile(run, root, plan, matrix, ref)
    expected = {"deadline_s": 24., "relative_tolerance": .001, "refine_absolute_tolerance_ml": .5, "easy_absolute_tolerance_ml": 1.,
        "peak_all_label_spill_limit": 1, "spacing_m": .0027, "particle_ml": .019683,
        "total_particles": 11625, "source_particles": 2880, "initial_source_ml": 56.68704, "post_seconds": 2.}
    for key, value in expected.items():
        physical.close(plan[key], value, "Fixed easy20 task/physics " + key)
    need(plan["refine_task_kind"] == "fixed30_40ml_absolute05ml_24s_onegrain_v1"
         and plan["original_strict_standard_unchanged"] is True
         and plan["solver"] == "unchanged_live_PBF_60Hz_pressure6_min4_graph_off"
         and plan["no_saved_liquid_replay"] is True and plan["phase_training_time_not_hidden"] is True,
         "Easy tolerance was confused with a change to original strict evidence or physics")
    need(len(plan["source_sha256"]) == 28 and len(plan["input_sha256"]) == 7
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
    for row, case in zip(rows, matrix["cases"]):
        need(row["stage"] == stage and row["split"] == ("test" if stage == "test" else "development") and row["role"] == case["role"]
             and row["protocol_id"] == plan["protocol_id"] and row["physics_fingerprint"] == fingerprint,
             "Actual stage/role/source/physics differs")
        report = audit_episode(run, row, plan, geometry, ref)
        retry = retry_bindings(row, report, plan, physical.parent, geometry, profile)
        audited = physical.audit_post(run, row, report, ref)
        for item in ref.gz(ref.inside(run, report["post_name"]))[0] if report["post_name"] in row["artifacts"] else []:
            validate_retry_status(item["controller_status"], case["policy"], plan["source_sha256"][CONTROLLER],
                                  row["target_ml"], profile["profile_id"], plan["goal_profile_raw_sha256"])
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
    pilot_gate = None if stage == "test" else bool(safety and all(report["refine_assessment"]["feasible"]
        for report in reports if report["role"] == "candidate"))
    candidate_by_goal = {format(goal, "g"): [{"episode_id": report["episode_id"],
        "seed": report["seed"], "primary_passed": report["refine_assessment"]["feasible"],
        "actual_received_ml": report["received_ml"], "physical_time_s": report["physical_s"],
        "actual_cycles_started": report["actual_retry_evidence"]["actual_cycles_started"],
        "actual_second_cycle_executed": report["actual_retry_evidence"]["actual_second_cycle_executed"]}
        for report in reports if report["role"] == "candidate" and report["target_ml"] == goal]
        for goal in TARGETS}
    return {"stage": stage, "protocol_id": plan["protocol_id"], "episodes_audited": len(rows),
        "pilot_gate_passed": pilot_gate, "development_gate_passed": pilot_gate,
        "candidate_by_goal": candidate_by_goal,
        "all_non_dose_safety_passed": safety, "fixed_policy_not_selected_from_test": True, "goal_profiles": matrix["goal_profiles"],
        "goal_profile_id": profile["profile_id"], "goal_profile_raw_sha256": plan["goal_profile_raw_sha256"],
        "refine_task_passes": summary["refine_task_passes"], "easy_task_passes": summary["easy_task_passes"], "original_strict_task_passes": summary["strict_task_passes"],
        "source_sha256": plan["source_sha256"], "input_sha256": plan["input_sha256"],
        "gui_sha256": plan["gui_sha256"], "versions": read(run / "versions.json"),
        "input_artifact_sha256": {name: sha(run / name) for name in ARTIFACTS},
        "episode_reports": reports, "rows": rows}


def validate_experiment_plan(experiment):
    path = experiment / "experiment_plan.json"
    need(sha(path) == PLAN_SHA, "Preregistered bounded goal extension bytes changed")
    plan = read(path)
    need(plan["kind"] == "retry3040_bounded_expansion_preregistration_v1" and plan["draft"] is False
         and tuple(plan["targets_ml"]) == TARGETS and plan["task"] == {
             "absolute_tolerance_ml": .5, "deadline_physical_seconds": 24., "home_quiet_frames": 12,
             "peak_all_label_spill_particles": 1, "post_physical_seconds": 2., "original_strict_relative_tolerance": .001}
         and plan["development"]["first_stage"] == "pilot_001"
         and plan["development"]["cases_per_stage"] == 4 and plan["development"]["maximum_stages"] == 2
         and plan["development"]["maximum_cases"] == 8 and plan["development"]["seed"] == PILOT_SEED
         and plan["development"]["paired_roles"] == ["baseline", "candidate"]
         and plan["test"]["stage"] == "test" and plan["test"]["cases"] == 40
         and tuple(plan["test"]["seeds"]) == TEST and tuple(plan["test"]["targets_ml"]) == TARGETS
         and plan["test"]["roles"] == ["baseline", "candidate"] and plan["test"]["paired_conditions"] == 20
         and plan["test"]["fixed_strategy_frozen_before_every_reset"] is True
         and plan["test"]["test_used_for_selection"] is False and plan["actual_angle_gap_degrees"] == .1
         and plan["no_new_training_search_or_neural_policy"] is True
         and plan["old20_implementation_and_results_immutable"] is True
         and plan["no_default_gui_deployment"] is True and plan["no_continuous_goal_guarantee"] is True,
         "Bounded development/fresh-test task or lifecycle changed")
    initial = ROOT / plan["initial_goal_profile_file"]
    need(sha(initial) == plan["initial_goal_profile_raw_sha256"], "Original proposal profile was overwritten")
    scan_file = ROOT / HERE / "fresh_seed_scan.json"
    need(sha(scan_file) == plan["fresh_seed_scan_sha256"], "Prelaunch seed-scan proof changed")
    scan = read(scan_file)
    need(scan["seeds"] == [PILOT_SEED, *TEST] and scan["collisions"] == []
         and scan["scanned_prior_protocol_plan_matrix_journals"] >= 1, "Fresh-seed declaration contains old conditions")
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
