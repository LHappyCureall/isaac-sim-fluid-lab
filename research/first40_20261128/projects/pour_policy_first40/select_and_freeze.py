"""Freeze a known-case first-angle choice, then gate a separate fresh test.

This CPU helper never runs liquid or reads test results. Calibration chooses
among three preregistered first angles only. Validation can authorize a fixed
test or stop the study; it cannot retune or fall back to the baseline.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = "projects/pour_policy_first40/"
PLAN_SHA256 = "e65b69757b9129940c0ff0786b90c38d83018d83e9022a524a0c8679f671b857"
AUDITOR_SHA256 = "5ccd055256c5147430ff83db4c576dd17d32863ab1a4ad2cfc45e9aaf2ae654e"
KIND = "first40_first_angle_calibration_v1"
ORDER = ("first_8700", "first_8705", "first_8695")
ANGLES = {"baseline": 87.1, "first_8700": 87., "first_8705": 87.05, "first_8695": 86.95}
ARTIFACTS = ("protocol.json", "summary.json", "job_state.json", "episodes.jsonl",
             "responses.jsonl", "matrix.json", "versions.json")
MAPS = ("source_sha256", "input_sha256", "gui_sha256", "protected_file_sha256")
FIXED_PARAMETERS = {"prepare_fraction": 1., "dose_tilt_deg_s": 25.6, "return_fraction": 1.,
    "tail_lead_s": .26, "stop_bias_ml": 10., "max_dose_u_accel": 6., "approach_tilt_deg_s": 6.,
    "action_delta_deg": .03, "action_tilt_deg_s": .6, "hold_seconds": .1,
    "backoff_delta_deg": .5, "backoff_tilt_deg_s": 4., "anchor_return_tilt_deg_s": 6.,
    "quiet_seconds": .25, "maximum_measurement_hold_s": 1.}


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
    need(path.is_relative_to(root.resolve()), "Artifact escaped its declared root")
    return path


def timestamp(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def verify_prior_references(workspace, plan):
    for relative, expected in plan["known_reference_sha256"].items():
        need(sha(inside(workspace, relative)) == expected, "Original known-case reference changed")
    need(sha(workspace / HERE / "fresh_seed_scan.json") == plan["fresh_seed_scan_sha256"],
         "Predeclared fresh-condition scan changed")


def actual_case(row):
    """Independently recompute dose and task rank; never widen the closed band."""
    assessment, result = row["refine_assessment"], row["result"]
    amount, elapsed = result["received_ml"], result["physical_time_s"]
    need(row["target_ml"] == result["target_ml"] == 40.
         and all(type(v) in (int, float) and math.isfinite(v) for v in (amount, elapsed))
         and elapsed >= 0., "Finite actual40mL measurement and physical time required")
    peak = assessment["maximum_all_label_spilled_particles"]
    need(type(peak) is int and peak >= 0 and type(assessment["valid_evidence"]) is bool,
         "Actual all-label spill count and evidence validity must be explicit")
    dose = 39.5 <= amount <= 40.5
    valid = bool(assessment["valid_evidence"] and result["mass_conserved"] and result["all_states_finite"])
    feasible = bool(valid and dose and assessment["full_home_within_deadline"] and elapsed <= 24.
                    and peak <= 1 and assessment["no_late_dose_change"])
    need(assessment["tolerance_ml"] == .5 and assessment["dose_passed"] is dose
         and assessment["spill_passed"] is (peak <= 1) and assessment["feasible"] is feasible,
         "Primary task assessment differs from unchanged actual measurement")
    return {"episode_id": row["episode_id"], "profile_key": row["profile_key"], "role": row["role"],
        "seed": row["seed"], "actual_received_ml": amount, "error_ml": amount - 40.,
        "absolute_error_ml": abs(amount - 40.), "physical_time_s": elapsed,
        "all_label_peak_spill_particles": peak, "valid_evidence": valid, "primary_passed": feasible}


def choose_candidate(rows):
    """Feasible: physical time then exact declared tie order; invalid: no elite."""
    need(len(rows) == 4 and {row["profile_key"] for row in rows} == set(ANGLES),
         "Retain baseline and all three actual calibration candidates")
    cases = {row["profile_key"]: actual_case(row) for row in rows}
    ranks = []
    for index, key in enumerate(ORDER):
        case = cases[key]
        if not case["valid_evidence"]:
            rank = (2, index)
        elif case["primary_passed"]:
            rank = (0, case["physical_time_s"], index)
        else:
            rank = (1, case["absolute_error_ml"], case["all_label_peak_spill_particles"], index)
        ranks.append((rank, key))
    ranks.sort()
    chosen = ranks[0][1]
    need(cases[chosen]["primary_passed"] is True,
         "No primary-feasible calibration candidate; do not freeze or start validation")
    return chosen, list(cases.values()), [{"profile_key": key, "rank": list(rank)} for rank, key in ranks]


def non_dose_safe(row):
    assessment, result, status = row["refine_assessment"], row["result"], row["status"]
    post = row.get("post_observation")
    return bool(assessment["valid_evidence"] and assessment["full_home_within_deadline"]
        and assessment["spill_passed"] and assessment["no_late_dose_change"]
        and result["complete"] and result["valid_physics"] and result["mass_conserved"]
        and result["all_states_finite"] and result["physical_time_s"] <= 24.
        and status["home_static"] and status["stable_frames"] >= 12 and not status["timed_out"]
        and isinstance(post, dict) and post.get("physical_steps") == 120
        and type(post.get("duration_s")) in (int, float) and abs(post["duration_s"] - 2.) < 1e-8
        and post.get("maximum_received_change_ml") == 0.
        and post.get("mass_array_unchanged") is True and post.get("all_states_finite") is True)


def inspect_stage(experiment, name, workspace, plan):
    run = experiment / name
    count = 4
    summary, job, audit = (read(run / item) for item in ("summary.json", "job_state.json", "audit.json"))
    need(summary.get("complete") is True and summary.get("episodes") == count and summary.get("stage") == name
         and job.get("state") == "complete" and job.get("pid") is None and not (run / "worker.lock").exists(),
         "A complete four-case stage with an exited worker is required")
    need(len(AUDITOR_SHA256) == 64 and sha(workspace / HERE / "audit_first_run.py") == AUDITOR_SHA256,
         "Final reviewed independent stage auditor must be pinned")
    need(audit.get("complete") is True and audit.get("pass") is True and audit.get("kind") == KIND
         and audit.get("stage") == name and audit.get("experiment_plan_sha256") == PLAN_SHA256
         and audit.get("auditor_source_file") == HERE + "audit_first_run.py"
         and audit.get("auditor_source_sha256") == AUDITOR_SHA256,
         "Complete reviewed independent stage audit required")
    need(sha(run / "audit_source_snapshot" / ("audit_first_run_" + AUDITOR_SHA256 + ".py")) == AUDITOR_SHA256,
         "Independently executed stage audit snapshot changed")
    hashes = audit["input_artifact_sha256"]
    need(set(hashes) == set(ARTIFACTS), "Every independently audited stage artifact is required")
    for relative, digest in hashes.items():
        need(sha(run / relative) == digest, "Audited stage artifact changed: " + name + "/" + relative)
    protocol, matrix = read(run / "protocol.json"), read(run / "matrix.json")
    need(protocol["matrix"] == matrix and matrix["stage"] == name
         and matrix["experiment_plan_sha256"] == PLAN_SHA256, "Stage plan/matrix binding differs")
    for category, count in zip(MAPS, (28, 8, 3, 42)):
        need(len(protocol[category]) == count and audit[category] == protocol[category],
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
         and protected["files_sha256"] == protocol["protected_file_sha256"], "Protected42-version manifest changed")
    profile_name = matrix["profile_file"]
    profile = read(inside(workspace, profile_name))
    expected_profiles = {key: {"enable_retry": True, "first_approach_tilt_deg": angle,
                              "retry_approach_tilt_deg": 87.2} for key, angle in ANGLES.items()}
    need(profile["kind"] == "first40_explicit_profiles_v1" and profile["profiles"] == expected_profiles
         == matrix["profiles"] == protocol["profiles"] == audit["profiles"]
         and sha(inside(workspace, profile_name)) == protocol["profile_raw_sha256"]
         == protocol["input_sha256"][profile_name] == audit["profile_raw_sha256"]
         and profile["profile_id"] == audit["profile_id"], "Static full four-profile input changed")
    rows = [json.loads(line) for line in (run / "episodes.jsonl").read_text(encoding="utf8").splitlines() if line]
    cases = {case["case_id"]: case for case in matrix["cases"]}
    need(len(rows) == len(cases) == len({r["episode_id"] for r in rows}) == 4
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
             and row["status"]["policy_params"]["maximum_cycles"] == 2,
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
    expected_conditions = ({(197304, key) for key in ANGLES} if name == "calibration" else
                           {(seed, role) for seed in (198101, 198102) for role in ("baseline", "candidate")})
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


def make_matrix(stage, selection, selection_sha, strategy_sha=None):
    need(stage in ("validation", "test"), "Only a new validation or test matrix may be created")
    seeds = (198101, 198102) if stage == "validation" else tuple(range(198301, 198311))
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
        need(isinstance(strategy_sha, str) and len(strategy_sha) == 64, "A frozen strategy SHA is required for tests")
        result["fixed_strategy_sha256"] = strategy_sha
    return result


def write_new(path, value):
    with path.open("x", encoding="utf8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False); stream.write("\n")


def freeze(experiment, phase, workspace=ROOT):
    experiment, workspace = Path(experiment).resolve(), Path(workspace).resolve()
    need(experiment.is_relative_to(workspace / "droplet_out/pour_policy_first40")
         and experiment != workspace / "droplet_out/pour_policy_first40", "Use a named new first40 experiment")
    need(phase in ("selection", "strategy"), "Unknown freeze phase")
    # These gates precede any plan, journal, existing checkpoint or test read.
    need(not (experiment / "test").exists(), "Freeze must precede any test directory")
    outputs = ("calibration_selection.json", "validation_matrix.json") if phase == "selection" else ("fixed_strategy.json", "test_matrix.json")
    need(not any((experiment / name).exists() for name in outputs), "Never overwrite frozen outputs")
    if phase == "selection":
        need(not (experiment / "validation").exists(), "Choose the first angle before any validation directory")
    need(sha(experiment / "experiment_plan.json") == PLAN_SHA256, "Preregistered bounded plan changed")
    plan = read(experiment / "experiment_plan.json")
    need(plan["draft"] is False and plan["target_ml"] == 40. and plan["maximum_actual_cases"] == 28
         and plan["stages"]["calibration"]["candidate_angles_in_tie_order"] == [87., 87.05, 86.95]
         and plan["stages"]["validation"]["seeds"] == [198101, 198102]
         and plan["stages"]["test"]["seeds"] == list(range(198301, 198311)), "Task, candidate order or fresh seeds changed")
    verify_prior_references(workspace, plan)
    calibration = inspect_stage(experiment, "calibration", workspace, plan)
    selected_key, actual, ranks = choose_candidate(calibration["rows"])
    known = calibration["audit"]["known_baseline_equivalence"]
    need(known["verified"] is True and known["actual_water_motion_partitions_exactly_equal"] is True
         and known["first_cycle_original_events_exactly_equal"] is True
         and known["source_reference_sha256"] == plan["known_reference_sha256"],
         "Known baseline must reproduce the old first-cycle physical evidence")
    need(all(non_dose_safe(row) for row in calibration["rows"])
         and calibration["audit"]["all_non_dose_safety_passed"] is True
         and calibration["audit"]["calibration_gate_passed"] is True,
         "Complete known-case calibration safety gate failed; preserve without new validation")
    candidate_scores = {case["profile_key"]: (None if not case["valid_evidence"] else
        [0, case["physical_time_s"]] if case["primary_passed"] else
        [1, case["absolute_error_ml"], case["all_label_peak_spill_particles"]])
        for case in actual if case["profile_key"] in ORDER}
    need(calibration["audit"]["selected_profile_key"] == selected_key
         and calibration["audit"]["candidate_scores"] == candidate_scores,
         "Independent calibration-only choice or ranking differs")
    common = {"task_kind": "fixed40ml_absolute05ml_24s_onegrain_v1", "experiment_plan_sha256": PLAN_SHA256,
        "profile_file": calibration["matrix"]["profile_file"], "profile_id": calibration["profile"]["profile_id"],
        "profile_raw_sha256": calibration["protocol"]["profile_raw_sha256"], "profiles": calibration["profile"]["profiles"],
        "selected_profile_key": selected_key, "selected_first_approach_tilt_deg": ANGLES[selected_key],
        "retry_approach_tilt_deg": 87.2, "baseline_profile_key": "baseline", "baseline_first_approach_tilt_deg": 87.1,
        "both_roles_enable_retry": True, "maximum_cycles": 2, "only_first_angle_varies": True,
        "fixed_parameters": FIXED_PARAMETERS,
        "calibration_proof": calibration["record"], "known_baseline_equivalence": known,
        "selector_source_sha256": sha(__file__), "auditor_source_sha256": AUDITOR_SHA256,
        "versions": calibration["versions"], **{key: calibration["protocol"][key] for key in MAPS}}
    if phase == "selection":
        selection = {"kind": "first40_calibration_selection_v1", "created_at_utc": datetime.now(timezone.utc).isoformat(),
            **common, "calibration_gate_passed": True, "calibration_cases": 4,
            "all_calibration_cases": actual, "candidate_ranking": ranks,
            "candidate_scores": candidate_scores,
            "candidate_tie_order": list(ORDER), "known_case_is_seen_development": True,
            "validation_used_for_selection": False, "test_used_for_selection": False,
            "feasible_rank_uses_only_physical_time_then_exact_declared_order": True,
            "validation_may_gate_but_not_retune_or_fallback": True,
            "default_gui_deployment": False, "all_continuous_goals_guaranteed": False}
        need(timestamp(selection["created_at_utc"]) > timestamp(calibration["record"]["created_at_utc"]),
             "Selection must freeze after the independent completed calibration audit")
        write_new(experiment / outputs[0], selection)
        matrix = make_matrix("validation", selection, sha(experiment / outputs[0]))
        write_new(experiment / outputs[1], matrix)
        return {"phase": phase, "selected_profile_key": selected_key,
                "calibration_selection_sha256": sha(experiment / outputs[0]), "validation_cases": 4}
    selection_path = experiment / "calibration_selection.json"
    selection = read(selection_path)
    need(selection["kind"] == "first40_calibration_selection_v1" and selection["calibration_gate_passed"] is True
         and selection["selector_source_sha256"] == sha(__file__) and selection["selected_profile_key"] == selected_key
         and selection["calibration_proof"] == calibration["record"] and selection["known_baseline_equivalence"] == known
         and selection["all_calibration_cases"] == actual and selection["candidate_ranking"] == ranks
         and selection["candidate_scores"] == candidate_scores
         and selection["validation_used_for_selection"] is False and selection["test_used_for_selection"] is False,
         "Pre-validation calibration selection changed")
    need(all(selection[key] == value for key, value in common.items()), "Frozen calibration metadata changed")
    validation = inspect_stage(experiment, "validation", workspace, plan)
    for key in MAPS:
        need(validation["protocol"][key] == calibration["protocol"][key] == selection[key],
             "Sources, static inputs or protected old versions changed during validation")
    expected_matrix = make_matrix("validation", selection, sha(selection_path))
    need(validation["matrix"] == expected_matrix and read(experiment / "validation_matrix.json") == expected_matrix,
         "Validation changed the previously selected profile or frozen matrix")
    need(all(timestamp(selection["created_at_utc"]) < timestamp(row["started_at_utc"]) for row in validation["rows"]),
         "Calibration choice did not freeze before every actual validation reset")
    need(all(non_dose_safe(row) for row in validation["rows"])
         and all(actual_case(row)["primary_passed"] for row in validation["rows"] if row["role"] == "candidate")
         and validation["audit"]["all_non_dose_safety_passed"] is True
         and validation["audit"]["validation_gate_passed"] is True
         and validation["audit"]["selected_profile_key"] == selected_key,
         "Fresh validation gate failed; preserve results without retuning, fallback or tests")
    strategy = {"kind": "first40_frozen_first_angle_strategy_v1", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        **common, "calibration_selection_sha256": sha(selection_path),
        "calibration_selection": {"file": selection_path.name, "sha256": sha(selection_path),
                                  "created_at_utc": selection["created_at_utc"]},
        "validation_proof": validation["record"], "validation_gate_passed": True,
        "calibration_cases": 4, "validation_cases": 4, "test_cases": 20,
        "candidate_not_retuned_or_replaced_by_validation": True, "test_used_for_selection": False,
        "default_gui_deployment": False, "all_continuous_goals_guaranteed": False}
    need(timestamp(strategy["created_at_utc"]) > timestamp(validation["record"]["created_at_utc"]),
         "Final strategy must freeze after the independent completed validation audit")
    write_new(experiment / outputs[0], strategy)
    matrix = make_matrix("test", selection, sha(selection_path), sha(experiment / outputs[0]))
    write_new(experiment / outputs[1], matrix)
    return {"phase": phase, "selected_profile_key": selected_key,
            "fixed_strategy_sha256": sha(experiment / outputs[0]), "fresh_test_cases": 20}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("selection", "strategy"), required=True)
    parser.add_argument("--experiment", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(freeze(args.experiment, args.phase), indent=2))


if __name__ == "__main__": main()
