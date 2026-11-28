"""Freeze the bounded30/40mL profile only after independent development gates.

All actual development stages remain bound by their audits, sidecars and raw
trace SHA. The helper never opens test results, fits a model or chooses from
test outcomes. Optional second-stage changes follow the preregistered rule.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
RELATIVE = "projects/pour_policy_retry3040/"
PLAN_SHA256 = "dc4fa45c82dcb30011b3f682cafb3b2506babc50803e97d32f1bd74bbcd79add"
AUDITOR_SHA256 = "f658538d89e404eaef4b9ba83d813c64e19d375983dfa0bdcf0fc61c4c1a1d75"
KIND = "retry3040_conditional_second_cycle_v1"
TARGETS = (30., 40.)
STAGES = ("pilot_001", "pilot_002")
ARTIFACTS = ("protocol.json", "summary.json", "job_state.json", "episodes.jsonl",
             "responses.jsonl", "matrix.json", "versions.json")
PARAMETERS = ("prepare_fraction", "dose_tilt_deg_s", "return_fraction", "tail_lead_s", "stop_bias_ml",
    "max_dose_u_accel", "approach_tilt_deg_s", "action_delta_deg", "action_tilt_deg_s", "hold_seconds",
    "backoff_delta_deg", "backoff_tilt_deg_s", "anchor_return_tilt_deg_s", "quiet_seconds", "maximum_measurement_hold_s")


def need(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    def reject(value):
        raise ValueError("Nonfinite JSON: " + value)
    return json.loads(Path(path).read_text(encoding="utf8"), parse_constant=reject)


def inside(root, relative):
    need(isinstance(relative, str) and relative, "A named relative artifact is required")
    path = (root / relative).resolve()
    need(path.is_relative_to(root.resolve()), "Artifact escapes its declared root")
    return path


def profile_for(row):
    return {name: row["declared_policy"][name]
            for name in ("first_approach_tilt_deg", "retry_approach_tilt_deg")}


def candidate_results(rows):
    selected = {}
    for target in TARGETS:
        group = [row for row in rows if row["role"] == "candidate" and row["target_ml"] == target]
        need(len(group) == 1, "Each development goal needs exactly one candidate result")
        result = group[0]["result"]
        amount = result["received_ml"]
        need(type(amount) in (int, float) and math.isfinite(amount), "Finite actual development water required")
        selected[str(int(target))] = {"target_ml": target, "actual_received_ml": amount,
            "error_ml": amount - target, "primary_passed": target - .5 <= amount <= target + .5}
    return selected


def validate_adjustment(previous_profiles, current_profiles, previous_results):
    need(set(previous_profiles) == set(current_profiles) == {"30", "40"}, "Both development profiles required")
    need(not all(row["primary_passed"] for row in previous_results.values()),
         "An optional second stage is forbidden after both candidate goals pass")
    for key in ("30", "40"):
        old, new, observed = previous_profiles[key], current_profiles[key], previous_results[key]
        shift = (0. if observed["primary_passed"] else .1 if observed["error_ml"] < -.5 else -.1)
        for name in ("first_approach_tilt_deg", "retry_approach_tilt_deg"):
            need(type(new[name]) in (int, float) and math.isfinite(new[name])
                 and abs(new[name] - (old[name] + shift)) <= 1e-9,
                 "Second development stage must retain passing goals and shift only failed goals by the declared signed0.1degree")


def validate_adjustment_receipt(receipt, first, selected):
    need(receipt["kind"] == "retry3040_preregistered_development_adjustment_v1"
         and receipt["pilot001_audit_sha256"] == first["audit_sha256"]
         and receipt["old_profile_sha256"] == first["goal_profile_raw_sha256"]
         and receipt["new_profile_sha256"] == selected["goal_profile_raw_sha256"]
         and receipt["no_code_or_speed_or_criterion_change"] is True
         and receipt["test_data_read_or_used"] is False, "Prescribed development adjustment receipt differs")
    expected = {row["target_ml"]: row for row in first["candidate_by_goal"].values() if not row["primary_passed"]}
    changes = receipt["changes"]
    need(len(changes) == len(expected) and {row["target_ml"] for row in changes} == set(expected),
         "Adjustment receipt must include exactly the previously failed candidate goals")
    for change in changes:
        observed = expected[change["target_ml"]]
        shift = .1 if observed["error_ml"] < -.5 else -.1
        need(abs(change["actual_prior_error_ml"] - observed["error_ml"]) < 1e-9
             and change["delta_degrees"] == shift, "Adjustment receipt changed the measured error or signed rule")


def non_dose_safe(row):
    assessment, result, status, post = row["refine_assessment"], row["result"], row["status"], row["post_observation"]
    return bool(assessment["valid_evidence"] and assessment["full_home_within_deadline"]
        and assessment["spill_passed"] and assessment["no_late_dose_change"]
        and result["complete"] and result["valid_physics"] and result["mass_conserved"]
        and result["all_states_finite"] and result["physical_time_s"] <= 24.
        and status["home_static"] and status["stable_frames"] >= 12 and not status["timed_out"]
        and post["physical_steps"] == 120 and abs(post["duration_s"] - 2.) < 1e-8
        and post["maximum_received_change_ml"] == 0. and post["mass_array_unchanged"]
        and post["all_states_finite"])


def inspect_stage(experiment, name, workspace):
    run = experiment / name
    summary, job, audit = (read(run / item) for item in ("summary.json", "job_state.json", "audit.json"))
    need(summary.get("complete") is True and summary.get("episodes") == 4 and summary.get("stage") == name
         and job.get("state") == "complete" and job.get("pid") is None and not (run / "worker.lock").exists(),
         "Every actual development stage must finish before freeze")
    need(audit.get("complete") is True and audit.get("pass") is True and audit.get("stage") == name
         and audit.get("kind") == KIND and audit.get("experiment_plan_sha256") == PLAN_SHA256
         and audit.get("auditor_source_file") == RELATIVE + "audit_retry_run.py"
         and audit.get("auditor_source_sha256") == AUDITOR_SHA256, "Reviewed independent stage audit required")
    need(sha(workspace / RELATIVE / "audit_retry_run.py") == AUDITOR_SHA256, "Stage auditor changed")
    need(sha(run / "audit_source_snapshot" / ("audit_retry_run_" + AUDITOR_SHA256 + ".py")) == AUDITOR_SHA256,
         "Exact independently executed audit snapshot is missing or changed")
    need(set(audit["input_artifact_sha256"]) == set(ARTIFACTS), "All seven audited stage artifacts required")
    for relative, digest in audit["input_artifact_sha256"].items():
        need(sha(run / relative) == digest, "Audited development evidence changed: " + name + "/" + relative)
    protocol, matrix = read(run / "protocol.json"), read(run / "matrix.json")
    need(protocol["matrix"] == matrix and matrix["stage"] == name
         and matrix["experiment_plan_sha256"] == PLAN_SHA256, "Stage does not bind the unchanged plan/matrix")
    need(len(protocol["source_sha256"]) == 28 and len(protocol["input_sha256"]) == 7
         and len(protocol["gui_sha256"]) == 3, "Expected28source/7input/3oldGUI freeze")
    for category, snapshot in (("source_sha256", "source_snapshot"), ("input_sha256", "input_snapshot")):
        for relative, digest in protocol[category].items():
            need(sha(inside(workspace, relative)) == digest
                 and sha(inside(run / snapshot, relative)) == digest, "Frozen workspace/snapshot changed: " + relative)
    for relative, digest in protocol["gui_sha256"].items():
        need(sha(inside(workspace, relative)) == digest, "An old GUI/default model changed")
    profile_name = matrix["goal_profile_file"]
    profile_path = inside(workspace, profile_name)
    need(profile_path.is_relative_to((workspace / RELATIVE).resolve()), "External profiles must belong to this new project")
    profile = read(profile_path)
    need(sha(profile_path) == protocol["goal_profile_raw_sha256"] == protocol["input_sha256"][profile_name]
         and profile["goal_profiles"] == matrix["goal_profiles"] == audit["goal_profiles"]
         and profile["profile_id"] == audit["goal_profile_id"]
         and audit["goal_profile_raw_sha256"] == protocol["goal_profile_raw_sha256"], "Stage profile identity/map/rawSHA differs")
    rows = [json.loads(line) for line in (run / "episodes.jsonl").read_text(encoding="utf8").splitlines() if line]
    expected = {(target, role, 197101) for target in TARGETS for role in ("baseline", "candidate")}
    need(len(rows) == 4 and len({row["episode_id"] for row in rows}) == 4
         and {(row["target_ml"], row["role"], row["seed"]) for row in rows} == expected,
         "Keep all four development cases, including dose failures")
    sidecars, raw = {}, {}
    for row in rows:
        need(row["stage"] == name and row["split"] == "development"
             and row["result"]["target_ml"] == row["target_ml"]
             and row["declared_policy"]["enable_retry"] is (row["role"] == "candidate")
             and profile_for(row) == profile["goal_profiles"][str(int(row["target_ml"]))], "Actual development profile/role differs")
        completed = run / "completed" / (row["episode_id"] + ".json")
        need(read(completed) == row, "Completed development sidecar differs from journal")
        sidecars[row["episode_id"]] = sha(completed)
        for relative, digest in row["artifacts"].items():
            need(sha(inside(run, relative)) == digest, "Original action/post gzip changed")
            raw[relative] = digest
        need(non_dose_safe(row), "A numerical/home/time/spill/late development failure may not be hidden")
        target, amount = row["target_ml"], row["result"]["received_ml"]
        need(row["refine_assessment"]["dose_passed"] is (target - .5 <= amount <= target + .5)
             and row["refine_assessment"]["feasible"] is (target - .5 <= amount <= target + .5), "Actual primary dose gate differs")
    actual_raw = {path.relative_to(run).as_posix() for path in (run / "traces").glob("*.jsonl.gz")}
    need(actual_raw == set(raw), "Extra or unjournaled physical attempts exist")
    candidate = candidate_results(rows)
    gate = all(result["primary_passed"] for result in candidate.values())
    need(audit["all_non_dose_safety_passed"] is True
         and audit["development_gate_passed"] is gate and audit["pilot_gate_passed"] is gate,
         "Independent development gate differs from actual candidate water and all-case safety")
    record = {"stage": name, "protocol_id": protocol["protocol_id"], "audit_sha256": sha(run / "audit.json"),
        "auditor_source_sha256": AUDITOR_SHA256, "input_artifact_sha256": audit["input_artifact_sha256"],
        "completed_sidecar_sha256": sidecars, "raw_artifact_sha256": raw,
        "goal_profile_file": profile_name, "goal_profile_id": profile["profile_id"],
        "goal_profile_raw_sha256": protocol["goal_profile_raw_sha256"], "goal_profiles": profile["goal_profiles"],
        "candidate_by_goal": candidate, "all_non_dose_safety_passed": True, "development_gate_passed": gate}
    return {"record": record, "rows": rows, "protocol": protocol, "matrix": matrix,
            "profile": profile, "versions": read(run / "versions.json")}


def make_test_matrix(plan, selected, strategy_sha256):
    return {"draft": False, "stage": "test", "experiment_plan_sha256": PLAN_SHA256,
        "goal_profile_file": selected["matrix"]["goal_profile_file"],
        "goal_profiles": selected["profile"]["goal_profiles"], "fixed_strategy_sha256": strategy_sha256,
        "cases": [{"case_id": f"test_{role}_t{int(target)}_s{seed}", "target_ml": target,
            "seed": seed, "split": "test", "role": role,
            "policy": {"enable_retry": role == "candidate", **selected["profile"]["goal_profiles"][str(int(target))]}}
            for role in ("baseline", "candidate") for target in TARGETS for seed in plan["test"]["seeds"]]}


def freeze(experiment, workspace=ROOT):
    experiment, workspace = Path(experiment).resolve(), Path(workspace).resolve()
    need(experiment.is_relative_to(workspace / "droplet_out/pour_policy_retry3040")
         and experiment != workspace / "droplet_out/pour_policy_retry3040", "Use a named new experiment root")
    # This gate precedes any plan, journal or strategy read.
    need(not (experiment / "test").exists(), "Fixed strategy must freeze before any test directory exists")
    need(not (experiment / "fixed_strategy.json").exists() and not (experiment / "test_matrix.json").exists(),
         "Never overwrite an existing frozen strategy or its matrix")
    need(sha(experiment / "experiment_plan.json") == PLAN_SHA256, "The predeclared bounded plan changed")
    plan = read(experiment / "experiment_plan.json")
    need(plan["draft"] is False and plan["targets_ml"] == list(TARGETS)
         and plan["test"]["seeds"] == list(range(197301, 197311)) and plan["test"]["cases"] == 40,
         "Declared targets/fresh-test coverage differs")
    names = [name for name in STAGES if (experiment / name).exists()]
    need(names in (["pilot_001"], ["pilot_001", "pilot_002"]), "Keep one or both actual bounded development stages")
    need(not any(path.is_dir() and path.name.startswith("pilot_") and path.name not in STAGES
                 for path in experiment.iterdir()), "Unexpected extra development attempts exist")
    stages = [inspect_stage(experiment, name, workspace) for name in names]
    first, selected = stages[0], stages[-1]
    adjustment = None
    if len(stages) == 2:
        validate_adjustment(first["profile"]["goal_profiles"], selected["profile"]["goal_profiles"], first["record"]["candidate_by_goal"])
        receipt = experiment / "development_adjustment_001.json"
        validate_adjustment_receipt(read(receipt), first["record"], selected["record"])
        adjustment = {"file": receipt.name, "sha256": sha(receipt)}
        for key in ("source_sha256", "gui_sha256"):
            need(first["protocol"][key] == selected["protocol"][key], "Runtime/GUI changed between development stages")
        old_inputs = {k: v for k, v in first["protocol"]["input_sha256"].items() if k != first["matrix"]["goal_profile_file"]}
        new_inputs = {k: v for k, v in selected["protocol"]["input_sha256"].items() if k != selected["matrix"]["goal_profile_file"]}
        need(old_inputs == new_inputs and first["versions"] == selected["versions"], "Geometry/runtime versions changed")
    need(selected["record"]["development_gate_passed"] is True, "Both candidate goals must pass before any fresh tests")
    parameters = selected["rows"][0]["result"]["policy_params"]
    fixed = {name: parameters[name] for name in PARAMETERS}
    for stage in stages:
        for row in stage["rows"]:
            need({name: row["result"]["policy_params"][name] for name in PARAMETERS} == fixed,
                 "Fixed coarse/micro speeds or action parameters changed")
    strategy = {"kind": "fixed3040_conditional_second_cycle_strategy_v1",
        "task_kind": "fixed30_40ml_absolute05ml_24s_onegrain_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "experiment_plan_sha256": PLAN_SHA256,
        "targets_ml": list(TARGETS), "completed_development_stages": names,
        "development_stages": [stage["record"] for stage in stages],
        "development_artifact_sha256": {stage["record"]["stage"]: {
            **stage["record"]["input_artifact_sha256"], "audit.json": stage["record"]["audit_sha256"]}
            for stage in stages}, "development_adjustment": adjustment,
        "selected_development_stage": names[-1], "development_cases": 4 * len(stages),
        "development_gate_passed": True, "goal_profile_file": selected["matrix"]["goal_profile_file"],
        "goal_profile_id": selected["profile"]["profile_id"],
        "goal_profile_raw_sha256": selected["record"]["goal_profile_raw_sha256"],
        "goal_profiles": selected["profile"]["goal_profiles"], "fixed_parameters": fixed,
        "maximum_cycles_baseline": 1, "maximum_cycles_candidate": 2,
        "test_cases": 40, "test_goal_seed_conditions": 20,
        "no_data_fitting_or_model_selection": True, "no_new_training_search_or_neural_policy": True,
        "test_used_for_selection": False, "versions": selected["versions"],
        "default_gui_deployment": False, "no_continuous_goal_guarantee": True,
        "freeze_helper_sha256": sha(__file__), "stage_auditor_source_sha256": AUDITOR_SHA256,
        **{key: selected["protocol"][key] for key in ("source_sha256", "input_sha256", "gui_sha256")}}
    with (experiment / "fixed_strategy.json").open("x", encoding="utf8") as stream:
        json.dump(strategy, stream, indent=2, allow_nan=False); stream.write("\n")
    matrix = make_test_matrix(plan, selected, sha(experiment / "fixed_strategy.json"))
    with (experiment / "test_matrix.json").open("x", encoding="utf8") as stream:
        json.dump(matrix, stream, indent=2, allow_nan=False); stream.write("\n")
    return {"fixed_strategy_sha256": sha(experiment / "fixed_strategy.json"),
            "completed_development_stages": names, "fresh_test_cases": 40, "data_fitting": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(freeze(args.experiment), indent=2))


if __name__ == "__main__": main()
