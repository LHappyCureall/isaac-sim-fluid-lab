"""CPU selection for grouped upstream40mL coarse-speed calibration.

Only the declared coarse dosing speed varies. A negative baseline may fail
dose, spill, home or late-water task gates while remaining a valid measured
comparator. Every selected candidate case must satisfy the full primary task.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = "projects/pour_policy_coarse40/"
PLAN_SHA256 = "604819128b25be9856e228a118a9567c34cc8bfe4e38441096015cb4dc7e05c7"
AUDITOR_SHA256 = "f5f9b277f828d7ad913115965c6f8e4e71ddbdc2481a4f0c9481d56ca74e82b4"
KIND = "coarse40_upstream_speed_calibration_v1"
ORDER = ("coarse_252", "coarse_248")
SPEEDS = {"baseline": 25.6, "coarse_252": 25.2, "coarse_248": 24.8}
KNOWN_SEEDS = (197304, 198304)
VALIDATION_SEEDS = (198201, 198202)
TEST_SEEDS = tuple(range(198401, 198411))
MAPS = ("source_sha256", "input_sha256", "gui_sha256", "protected_file_sha256")
FIXED_PARAMETERS = {"prepare_fraction": 1., "return_fraction": 1., "tail_lead_s": .26,
    "stop_bias_ml": 10., "max_dose_u_accel": 6., "approach_tilt_deg_s": 6.,
    "action_delta_deg": .03, "action_tilt_deg_s": .6, "hold_seconds": .1,
    "backoff_delta_deg": .5, "backoff_tilt_deg_s": 4., "anchor_return_tilt_deg_s": 6.,
    "quiet_seconds": .25, "maximum_measurement_hold_s": 1.}
ARTIFACTS = ("protocol.json", "summary.json", "job_state.json", "episodes.jsonl",
             "responses.jsonl", "matrix.json", "versions.json")


def need(value, message):
    if not value: raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    def reject(value):
        raise ValueError("Nonfinite JSON: " + value)
    return json.loads(Path(path).read_text(encoding="utf8"), parse_constant=reject)


def inside(root, relative):
    need(isinstance(relative, str) and relative, "A named relative artifact is required")
    path = (root / relative).resolve()
    need(path.is_relative_to(root.resolve()), "Artifact escaped its declared root")
    return path


def timestamp(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def verify_prior_references(workspace, plan):
    references = plan["known_reference_sha256"]
    need(set(references) == {str(seed) for seed in KNOWN_SEEDS}, "Both actual seen references must remain")
    for bindings in references.values():
        need(len(bindings) == 4, "Four original artifacts per seen condition are required")
        for relative, expected in bindings.items():
            need(sha(inside(workspace, relative)) == expected, "An original seen reference changed")
    need(sha(workspace / HERE / "fresh_seed_scan.json") == plan["fresh_seed_scan_sha256"],
         "The fresh-condition scan changed")


def actual_case(row):
    result, assessment = row["result"], row["refine_assessment"]
    amount, elapsed = result["received_ml"], result["physical_time_s"]
    need(row["target_ml"] == result["target_ml"] == 40.
         and all(type(v) in (int, float) and math.isfinite(v) for v in (amount, elapsed))
         and elapsed >= 0., "Finite actual40mL measurement and physical action time are required")
    peak = assessment["maximum_all_label_spilled_particles"]
    need(type(peak) is int and peak >= 0 and type(assessment["valid_evidence"]) is bool,
         "Actual all-label spill count and evidence validity must be explicit")
    valid = bool(assessment["valid_evidence"] and result["mass_conserved"] and result["all_states_finite"])
    dose = 39.5 <= amount <= 40.5
    feasible = bool(valid and dose and assessment["full_home_within_deadline"] and elapsed <= 24.
                    and peak <= 1 and assessment["no_late_dose_change"])
    need(assessment["tolerance_ml"] == .5 and assessment["dose_passed"] is dose
         and assessment["spill_passed"] is (peak <= 1) and assessment["feasible"] is feasible,
         "The primary closed task gate differs from actual unchanged measurements")
    return {"episode_id": row["episode_id"], "profile_key": row["profile_key"], "role": row["role"],
        "seed": row["seed"], "coarse_dose_tilt_deg_s": row["declared_policy"]["coarse_dose_tilt_deg_s"],
        "actual_received_ml": amount, "error_ml": amount - 40., "absolute_error_ml": abs(amount - 40.),
        "physical_time_s": elapsed, "all_label_peak_spill_particles": peak,
        "valid_evidence": valid, "primary_passed": feasible}


def valid_baseline_comparator(row):
    """Evidence validity alone; negative task outcomes are retained explicitly."""
    assessment, result = row["strict_assessment"], row["result"]
    return bool(assessment["valid_evidence"] and result["valid_physics"]
                and result["mass_conserved"] and result["all_states_finite"])


def score_group(rows):
    need(len(rows) == 2 and {row["seed"] for row in rows} == set(KNOWN_SEEDS)
         and len({row["profile_key"] for row in rows}) == 1, "Each declared group needs both unique seen conditions")
    group = [actual_case(row) for row in rows]
    if not all(case["valid_evidence"] for case in group): return None
    if all(case["primary_passed"] for case in group):
        return [0, sum(case["physical_time_s"] for case in group) / 2.]
    return [1, sum(case["absolute_error_ml"] for case in group) / 2.,
            sum(case["all_label_peak_spill_particles"] for case in group)]


def choose_candidate(rows):
    need(len(rows) == 6 and len({row["episode_id"] for row in rows}) == 6
         and {(row["profile_key"], row["seed"]) for row in rows}
             == {(key, seed) for key in SPEEDS for seed in KNOWN_SEEDS},
         "Retain all six two-seed calibration cases, including negative baselines")
    actual, scores, ranking = [actual_case(row) for row in rows], {}, []
    for index, key in enumerate(ORDER):
        score = score_group([row for row in rows if row["profile_key"] == key])
        rank = (2, index) if score is None else (*score, index)
        scores[key] = score
        ranking.append((rank, key))
    ranking.sort()
    chosen = ranking[0][1]
    need(scores[chosen] is not None and scores[chosen][0] == 0,
         "No candidate passes both known cases; preserve records without validation or retuning")
    return chosen, actual, scores, [{"profile_key": key, "rank": list(rank)} for rank, key in ranking]


def make_matrix(stage, selection, selection_sha, strategy_sha=None):
    need(stage in ("validation", "test"), "Only a new validation or test matrix can be created")
    seeds = VALIDATION_SEEDS if stage == "validation" else TEST_SEEDS
    result = {"draft": False, "stage": stage, "experiment_plan_sha256": PLAN_SHA256,
        "profile_file": selection["profile_file"], "profiles": selection["profiles"],
        "calibration_selection_sha256": selection_sha,
        "cases": [{"case_id": f"{stage}_{role}_s{seed}", "target_ml": 40., "seed": seed,
            "split": "test" if stage == "test" else "development", "role": role,
            "candidate_id": "baseline" if role == "baseline" else selection["selected_profile_key"],
            "profile_key": "baseline" if role == "baseline" else selection["selected_profile_key"],
            "policy": dict(selection["profiles"]["baseline" if role == "baseline" else selection["selected_profile_key"]])}
            for role in ("baseline", "candidate") for seed in seeds]}
    if stage == "test":
        need(isinstance(strategy_sha, str) and len(strategy_sha) == 64, "A frozen final strategy SHA is required")
        result["fixed_strategy_sha256"] = strategy_sha
    return result


def inspect_stage(experiment, name, workspace, plan):
    run = experiment / name
    count = 6 if name == "calibration" else 4
    summary, job, audit = (read(run / item) for item in ("summary.json", "job_state.json", "audit.json"))
    need(summary.get("complete") is True and summary.get("episodes") == count and summary.get("stage") == name
         and job.get("state") == "complete" and job.get("pid") is None and not (run / "worker.lock").exists(),
         "A complete declared complete stage with an exited worker is required")
    need(len(AUDITOR_SHA256) == 64 and sha(workspace / HERE / "audit_coarse_run.py") == AUDITOR_SHA256,
         "Final reviewed independent stage auditor must be pinned")
    need(audit.get("complete") is True and audit.get("pass") is True and audit.get("kind") == KIND
         and audit.get("stage") == name and audit.get("experiment_plan_sha256") == PLAN_SHA256
         and audit.get("auditor_source_file") == HERE + "audit_coarse_run.py"
         and audit.get("auditor_source_sha256") == AUDITOR_SHA256,
         "Complete reviewed independent stage audit required")
    need(sha(run / "audit_source_snapshot" / ("audit_coarse_run_" + AUDITOR_SHA256 + ".py")) == AUDITOR_SHA256,
         "Independently executed stage audit snapshot changed")
    hashes = audit["input_artifact_sha256"]
    need(set(hashes) == set(ARTIFACTS), "Every independently audited stage artifact is required")
    for relative, digest in hashes.items():
        need(sha(run / relative) == digest, "Audited stage artifact changed: " + name + "/" + relative)
    protocol, matrix = read(run / "protocol.json"), read(run / "matrix.json")
    need(protocol["matrix"] == matrix and matrix["stage"] == name
         and matrix["experiment_plan_sha256"] == PLAN_SHA256, "Stage plan/matrix binding differs")
    for category, expected_count in zip(MAPS, (28, 8, 3, plan["protected_file_count"])):
        need(len(protocol[category]) == expected_count and audit[category] == protocol[category],
             "Frozen source/input/GUI/protected inventory differs: " + category)
        for relative, digest in protocol[category].items():
            need(sha(inside(workspace, relative)) == digest, "A frozen current file changed: " + relative)
            if category in ("source_sha256", "input_sha256"):
                folder = "source_snapshot" if category == "source_sha256" else "input_snapshot"
                need(sha(inside(run / folder, relative)) == digest, "A frozen snapshot changed")
            elif category == "protected_file_sha256" and not any(relative in protocol[key] for key in MAPS[:3]):
                need(sha(inside(run / "protected_snapshot", relative)) == digest, "A protected old version snapshot changed")
    protected_name = plan["protected_versions_file"]
    protected = read(inside(workspace, protected_name))
    need(sha(inside(workspace, protected_name)) == plan["protected_versions_raw_sha256"]
         == protocol["protected_manifest_raw_sha256"] == protocol["input_sha256"][protected_name]
         and protected["files_sha256"] == protocol["protected_file_sha256"], "Protected old-version manifest changed")
    profile_name = matrix["profile_file"]
    profile = read(inside(workspace, profile_name))
    expected_profiles = {key: {"enable_retry": True, "first_approach_tilt_deg": 87.,
        "retry_approach_tilt_deg": 87.2, "coarse_dose_tilt_deg_s": speed} for key, speed in SPEEDS.items()}
    need(profile["kind"] == "coarse40_explicit_profiles_v1" and profile["profiles"] == expected_profiles
         == matrix["profiles"] == protocol["profiles"] == audit["profiles"]
         and sha(inside(workspace, profile_name)) == protocol["profile_raw_sha256"]
         == protocol["input_sha256"][profile_name] == audit["profile_raw_sha256"]
         and profile["profile_id"] == audit["profile_id"], "Static full three-profile input changed")
    rows = [json.loads(line) for line in (run / "episodes.jsonl").read_text(encoding="utf8").splitlines() if line]
    cases = {case["case_id"]: case for case in matrix["cases"]}
    need(len(rows) == len(cases) == len({r["episode_id"] for r in rows}) == count
         and {r["episode_id"] for r in rows} == set(cases), "Keep every actual stage attempt exactly once")
    sidecars, raw = {}, {}
    reports = {report["episode_id"]: report for report in audit["episode_reports"]}
    for row in rows:
        case = cases[row["episode_id"]]
        need(row["stage"] == name and row["split"] == "development" and row["target_ml"] == 40.
             and row["profile_key"] == row["candidate_id"] == case["profile_key"]
             and row["role"] == case["role"] == ("baseline" if row["profile_key"] == "baseline" else "candidate")
             and row["seed"] == case["seed"]
             and row["declared_policy"] == case["policy"] == expected_profiles[row["profile_key"]],
             "Actual known/validation condition or fixed retry profile differs")
        need(all(row["status"]["policy_params"][key] == value for key, value in FIXED_PARAMETERS.items())
             and row["status"]["policy_params"]["maximum_cycles"] == 2
             and row["status"]["policy_params"]["dose_tilt_deg_s"] == row["declared_policy"]["coarse_dose_tilt_deg_s"],
             "Actual coarse/micro speeds, waits or retry budget changed")
        report = reports[row["episode_id"]]
        need(report["refine_assessment"] == row["refine_assessment"]
             and report["declared_policy"] == row["declared_policy"], "Independent actual task/profile differs")
        actual_case(row)
        completed = run / "completed" / (row["episode_id"] + ".json")
        need(read(completed) == row, "Completed sidecar differs from actual journal")
        sidecars[row["episode_id"]] = sha(completed)
        for relative, digest in row["artifacts"].items():
            need(sha(inside(run, relative)) == digest, "Original action/post evidence changed")
            raw[relative] = digest
    need({p.relative_to(run).as_posix() for p in (run / "traces").glob("*.jsonl.gz")} == set(raw),
         "Extra or unjournaled physical attempts exist")
    expected_conditions = ({(seed, key) for seed in KNOWN_SEEDS for key in SPEEDS} if name == "calibration" else
                           {(seed, role) for seed in VALIDATION_SEEDS for role in ("baseline", "candidate")})
    actual_conditions = {(r["seed"], r["profile_key"] if name == "calibration" else r["role"]) for r in rows}
    need(actual_conditions == expected_conditions, "Known calibration or fresh validation seed grid changed")
    need(timestamp(audit["created_at_utc"]) > max(timestamp(r["completed_at_utc"]) for r in rows),
         "Independent audit must follow completion of all physical cases")
    record = {"stage": name, "protocol_id": protocol["protocol_id"], "audit_sha256": sha(run / "audit.json"),
        "auditor_source_sha256": AUDITOR_SHA256, "input_artifact_sha256": hashes,
        "completed_sidecar_sha256": sidecars, "raw_artifact_sha256": raw,
        "created_at_utc": audit["created_at_utc"], "actual_cases": [actual_case(r) for r in rows]}
    return {"record": record, "rows": rows, "protocol": protocol, "matrix": matrix,
            "audit": audit, "profile": profile, "versions": read(run / "versions.json")}


def freeze(experiment, phase, workspace=ROOT):
    need(len(PLAN_SHA256) == len(AUDITOR_SHA256) == 64,
         "New coarse40 plan and independent auditor are not frozen; no checkpoint may be written")
    experiment, workspace = Path(experiment).resolve(), Path(workspace).resolve()
    need(experiment.is_relative_to(workspace / "droplet_out/pour_policy_coarse40")
         and experiment != workspace / "droplet_out/pour_policy_coarse40", "Use a named new coarse40 experiment")
    need(phase in ("selection", "strategy"), "Unknown freeze phase")
    # Refuse later physical phases or overwrite before reading any evidence.
    need(not (experiment / "test").exists(), "Freeze must precede any test directory")
    outputs = ("calibration_selection.json", "validation_matrix.json") if phase == "selection" else ("fixed_strategy.json", "test_matrix.json")
    need(not any((experiment / name).exists() for name in outputs), "Never overwrite a frozen output")
    if phase == "selection": need(not (experiment / "validation").exists(), "Choose speed before any validation directory")
    need(sha(experiment / "experiment_plan.json") == PLAN_SHA256, "Preregistered grouped calibration plan changed")
    plan = read(experiment / "experiment_plan.json")
    need(plan["draft"] is False and plan["target_ml"] == 40. and plan["maximum_actual_cases"] == 30
         and plan["stages"]["calibration"]["candidate_speeds_in_tie_order_deg_s"] == [25.2, 24.8]
         and plan["stages"]["calibration"]["seeds"] == list(KNOWN_SEEDS)
         and plan["stages"]["validation"]["seeds"] == list(VALIDATION_SEEDS)
         and plan["stages"]["test"]["seeds"] == list(TEST_SEEDS), "Grouped budget, candidate order or fresh seeds changed")
    verify_prior_references(workspace, plan)
    calibration = inspect_stage(experiment, "calibration", workspace, plan)
    selected_key, actual, scores, ranks = choose_candidate(calibration["rows"])
    known = calibration["audit"]["known_baseline_equivalence"]
    need(set(known) == {str(seed) for seed in KNOWN_SEEDS}, "Both measured seen-reference proofs are required")
    for seed in KNOWN_SEEDS:
        proof = known[str(seed)]
        need(proof["verified"] is True and proof["actual_water_motion_partitions_exactly_equal"] is True
             and proof["first_cycle_original_events_exactly_equal"] is True
             and proof["source_reference_sha256"] == plan["known_reference_sha256"][str(seed)],
             "A known baseline did not reproduce its original complete first-cycle physics")
    need(all(valid_baseline_comparator(row) for row in calibration["rows"] if row["role"] == "baseline")
         and calibration["audit"]["all_baseline_valid_comparators"] is True
         and calibration["audit"]["calibration_gate_passed"] is True
         and calibration["audit"]["selected_profile_key"] == selected_key
         and calibration["audit"]["candidate_scores"] == scores,
         "Calibration needs valid baseline evidence and one fully feasible candidate group")
    common = {"task_kind": "fixed40ml_absolute05ml_24s_onegrain_v1", "experiment_plan_sha256": PLAN_SHA256,
        "profile_file": calibration["matrix"]["profile_file"], "profile_id": calibration["profile"]["profile_id"],
        "profile_raw_sha256": calibration["protocol"]["profile_raw_sha256"], "profiles": calibration["profile"]["profiles"],
        "selected_profile_key": selected_key, "selected_coarse_dose_tilt_deg_s": SPEEDS[selected_key],
        "first_approach_tilt_deg": 87., "retry_approach_tilt_deg": 87.2,
        "baseline_profile_key": "baseline", "baseline_coarse_dose_tilt_deg_s": 25.6,
        "both_roles_enable_retry": True, "maximum_cycles": 2, "only_coarse_dose_speed_varies": True,
        "only_first_angle_varies": False, "fixed_parameters": FIXED_PARAMETERS,
        "baseline_comparator_does_not_require_task_success": True,
        "candidate_group_requires_all_joint_primary": True,
        "calibration_proof": calibration["record"], "known_baseline_equivalence": known,
        "selector_source_sha256": sha(__file__), "auditor_source_sha256": AUDITOR_SHA256,
        "versions": calibration["versions"], **{key: calibration["protocol"][key] for key in MAPS}}
    if phase == "selection":
        selection = {"kind": "coarse40_calibration_selection_v1", "created_at_utc": datetime.now(timezone.utc).isoformat(),
            **common, "calibration_gate_passed": True, "calibration_cases": 6,
            "all_calibration_cases": actual, "candidate_scores": scores, "candidate_ranking": ranks,
            "candidate_tie_order": list(ORDER), "known_case_is_seen_development": True,
            "grouped_calibration_seeds": list(KNOWN_SEEDS), "validation_used_for_selection": False,
            "test_used_for_selection": False, "feasible_rank_uses_only_physical_time_then_exact_declared_order": True,
            "feasible_group_rank_uses_mean_physical_time": True,
            "infeasible_group_rank_uses_mae_then_total_peak_spill": True,
            "validation_may_gate_but_not_retune_or_fallback": True,
            "default_gui_deployment": False, "all_continuous_goals_guaranteed": False}
        need(timestamp(selection["created_at_utc"]) > timestamp(calibration["record"]["created_at_utc"]),
             "Selection must follow the independently audited complete six-case calibration")
        write_new(experiment / outputs[0], selection)
        write_new(experiment / outputs[1], make_matrix("validation", selection, sha(experiment / outputs[0])))
        return {"phase": phase, "selected_profile_key": selected_key,
                "calibration_selection_sha256": sha(experiment / outputs[0]), "validation_cases": 4}
    selection_path = experiment / "calibration_selection.json"
    selection = read(selection_path)
    need(selection["kind"] == "coarse40_calibration_selection_v1" and selection["calibration_gate_passed"] is True
         and all(selection[key] == value for key, value in common.items())
         and selection["all_calibration_cases"] == actual and selection["candidate_scores"] == scores
         and selection["candidate_ranking"] == ranks and selection["validation_used_for_selection"] is False
         and selection["test_used_for_selection"] is False, "Frozen grouped calibration choice changed")
    validation = inspect_stage(experiment, "validation", workspace, plan)
    need(all(validation["protocol"][key] == calibration["protocol"][key] == selection[key] for key in MAPS),
         "Sources, static inputs or prior protected versions changed during validation")
    expected_matrix = make_matrix("validation", selection, sha(selection_path))
    need(validation["matrix"] == expected_matrix and read(experiment / "validation_matrix.json") == expected_matrix,
         "Validation changed the previously selected coarse speed")
    need(all(timestamp(selection["created_at_utc"]) < timestamp(row["started_at_utc"]) for row in validation["rows"]),
         "The selected group did not freeze before actual fresh validation")
    need(all(valid_baseline_comparator(row) for row in validation["rows"] if row["role"] == "baseline")
         and all(actual_case(row)["primary_passed"] for row in validation["rows"] if row["role"] == "candidate")
         and validation["audit"]["all_baseline_valid_comparators"] is True
         and validation["audit"]["validation_gate_passed"] is True
         and validation["audit"]["selected_profile_key"] == selected_key,
         "Fresh validation gate failed; preserve negative baselines and stop without retuning or fallback")
    strategy = {"kind": "coarse40_frozen_speed_strategy_v1", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        **common, "calibration_selection_sha256": sha(selection_path),
        "calibration_selection": {"file": selection_path.name, "sha256": sha(selection_path), "created_at_utc": selection["created_at_utc"]},
        "validation_proof": validation["record"], "validation_gate_passed": True,
        "calibration_cases": 6, "validation_cases": 4, "test_cases": 20,
        "candidate_not_retuned_or_replaced_by_validation": True, "test_used_for_selection": False,
        "default_gui_deployment": False, "all_continuous_goals_guaranteed": False}
    need(timestamp(strategy["created_at_utc"]) > timestamp(validation["record"]["created_at_utc"]),
         "The final speed strategy must freeze after the independent complete validation audit")
    write_new(experiment / outputs[0], strategy)
    write_new(experiment / outputs[1], make_matrix("test", selection, sha(selection_path), sha(experiment / outputs[0])))
    return {"phase": phase, "selected_profile_key": selected_key,
            "fixed_strategy_sha256": sha(experiment / outputs[0]), "fresh_test_cases": 20}


def write_new(path, value):
    with path.open("x", encoding="utf8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False); stream.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("selection", "strategy"), required=True)
    parser.add_argument("--experiment", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(freeze(args.experiment, args.phase), indent=2))


if __name__ == "__main__": main()
