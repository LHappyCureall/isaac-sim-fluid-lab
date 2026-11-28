"""Curate the completed fixed20mL single-correction curriculum after independent raw audit.

This CPU-only tool never runs physics or chooses a model from test outcomes.
The original strict assessment remains a separately named secondary result.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path


KIND = "refine20_settled_single_cycle_angle_v1"
COUNTS = {"engineering": 3, "train": 6, "validation": 4, "test": 10}
SEEDS = {"engineering": {195001}, "train": {195101, 195102}, "validation": {195201, 195202},
         "test": set(range(195301, 195306))}
TOP = ("protocol.json", "summary.json", "job_state.json", "episodes.jsonl",
       "responses.jsonl", "matrix.json", "versions.json", "audit.json")


def need(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    def reject(value):
        raise ValueError("Nonfinite JSON: " + value)
    return json.loads(Path(path).read_text(encoding="utf8"), parse_constant=reject)


def safe_path(root, relative):
    need(isinstance(relative, str) and relative, "Artifact needs a relative name")
    destination = (root / relative).resolve()
    need(destination.is_relative_to(root.resolve()), "Artifact escapes the experiment")
    return destination


def close(left, right, message, tolerance=1e-9):
    need(type(left) in (int, float) and type(right) in (int, float)
         and math.isfinite(left) and math.isfinite(right)
         and abs(left - right) <= tolerance, message)


def metrics(cases):
    need(cases, "Empty metric group")
    return {"cases": len(cases),
        "valid_evidence_cases": sum(case["valid_evidence"] for case in cases),
        "joint_passes": sum(case["refine_feasible"] for case in cases),
        "easy_joint_passes": sum(case["easy_feasible"] for case in cases),
        "original_strict_joint_passes": sum(case["original_strict_feasible"] for case in cases),
        "mean_absolute_error_ml": sum(case["absolute_error_ml"] for case in cases) / len(cases),
        "maximum_absolute_error_ml": max(case["absolute_error_ml"] for case in cases),
        "mean_signed_error_ml": sum(case["error_ml"] for case in cases) / len(cases),
        "mean_physical_action_seconds": sum(case["physical_action_seconds"] for case in cases) / len(cases),
        "maximum_physical_action_seconds": max(case["physical_action_seconds"] for case in cases),
        "mean_compute_wall_seconds_including_post": sum(case["compute_wall_seconds_including_post"] for case in cases) / len(cases),
        "mean_action_compute_wall_seconds": sum(case["action_compute_wall_seconds"] for case in cases) / len(cases),
        "maximum_peak_all_label_spill_particles": max(case["peak_all_label_spill_particles"] for case in cases),
        "full_home_within_deadline_cases": sum(case["full_home_within_deadline"] for case in cases),
        "no_late_dose_change_cases": sum(case["no_late_dose_change"] for case in cases)}


def compact_case(row, stage, report):
    result, status = row["result"], row["status"]
    refine, easy, strict = row["refine_assessment"], row["easy_assessment"], row["strict_assessment"]
    need(row["split"] == ("development" if stage == "engineering" else stage)
         and row["stage"] == stage, "Stage/split label differs")
    close(row["target_ml"], 20., "Actual target changed")
    close(result["target_ml"], 20., "Result target changed")
    received, error = result["received_ml"], result["received_ml"] - 20.
    close(result["error_ml"], error, "Actual signed error differs")
    close(result["absolute_error_ml"], abs(error), "Actual absolute error differs")
    for name, assessment, tolerance in (("refine", refine, .5), ("easy", easy, 1.), ("strict", strict, .02)):
        close(assessment["tolerance_ml"], tolerance, "Task criterion changed: " + name)
        need(type(assessment["valid_evidence"]) is bool, "Physical evidence validity must be explicit")
        dose = 20. - tolerance <= received <= 20. + tolerance
        need(assessment["dose_passed"] is dose, "Closed actual dose gate differs: " + name)
        expected = bool(assessment["valid_evidence"] and dose and assessment["full_home_within_deadline"]
                        and assessment["spill_passed"] and assessment["no_late_dose_change"])
        need(assessment["feasible"] is expected, "Joint acceptance differs: " + name)
        need(report[name + "_assessment"] == assessment, "Independent assessment differs: " + name)
    need(report["episode_id"] == row["episode_id"], "Audited case identity differs")
    for key in ("stage", "role", "seed"):
        need(report[key] == (stage if key == "stage" else row[key]), "Audited case label differs: " + key)
    for key in ("received_ml", "error_ml"):
        close(report[key], result[key], "Audited measurement differs: " + key)
    if "physical_s" in report:
        close(report["physical_s"], result["physical_time_s"], "Audited physical time differs")
    peak = refine["maximum_all_label_spilled_particles"]
    need(type(peak) is int and peak >= 0 and refine["spill_passed"] is (peak <= 1), "Invalid all-label spill count")
    close(result["physical_time_s"], result["physical_steps"] / 60., "Physical clock differs fromstep/60")
    post = row["post_observation"]
    close(post["duration_s"], 2., "Post observation is not two physical seconds", 1e-8)
    need(post["physical_steps"] == 120, "Post observation must retain120 live steps")
    policy = row["declared_policy"]
    need(set(policy) == {"enable_correction", "approach_tilt_deg"}
         and type(policy["enable_correction"]) is bool and policy["approach_tilt_deg"] in (84., 84.1, 84.2),
         "Actual correction profile differs from the finite declared action set")
    close(status["refine_configured_approach_tilt_deg"], policy["approach_tilt_deg"], "Executed angle differs")
    need(status["refine_enable_correction"] is policy["enable_correction"]
         and status["refine_maximum_cycles"] == 1, "Executed correction mode differs")
    starts = [event for event in row["response_events"] if event["type"] == "response_cycle_start"]
    ends = [event for event in row["response_events"] if event["type"] in ("response_cycle_complete", "response_cycle_unsettled")]
    need(len(starts) <= 1 and len(ends) <= 1 and status["response_cycle_count"] == len(starts),
         "Single-cycle budget changed")
    need(policy["enable_correction"] or not row["response_events"], "Coarse baseline unexpectedly executed a correction")
    anchor = status["prefix_anchor_observation"]
    close(anchor["actual_target_ml"], 20., "Actual anchor target changed")
    close(anchor["control_target_ml"], 20., "Coarse target was offset")
    need(report["actual_refine_coarse_anchor"] == anchor, "Independent first quiet anchor differs")
    cycles = []
    for index, start in enumerate(starts):
        cycle = {"start_physical_s": start["physical_time_s"],
            "actual_start_received_ml": start["observation"]["received_ml"],
            "has_recorded_endpoint": index < len(ends)}
        if index < len(ends):
            end = ends[index]
            cycle.update(end_physical_s=end["physical_time_s"],
                actual_end_received_ml=end["post_observation"]["received_ml"],
                outcome=end["outcome"], settled=end["settled"],
                action_execution_complete=end["action_execution_complete"])
        cycles.append(cycle)
    return {"episode_id": row["episode_id"], "stage": stage, "split": row["split"], "role": row["role"],
        "seed": row["seed"], "target_ml": 20., "approach_tilt_deg": policy["approach_tilt_deg"],
        "correction_enabled": policy["enable_correction"], "actual_correction_cycles_started": len(starts),
        "actual_correction_cycles_ended": len(ends),
        "refine_actual_settled_home_decision": status["refine_actual_settled_home_decision"],
        "correction_cycle_completion_reason": status.get("response_cycle_terminal_reason"),
        "actual_coarse_cut_tilt_degrees": anchor["coarse_cut_tilt_deg"],
        "first_quiet_anchor_received_ml": anchor["received_ml"],
        "first_quiet_anchor_physical_seconds": anchor["physical_time_s"],
        "first_quiet_anchor_source_rms_m_s": anchor["source_inside_velocity_rms_m_s"],
        "actual_cycles": cycles,
        "received_ml": received, "error_ml": error, "absolute_error_ml": abs(error),
        "refine_feasible": refine["feasible"], "easy_feasible": easy["feasible"],
        "original_strict_feasible": strict["feasible"], "original_strict_dose_passed": strict["dose_passed"],
        "valid_evidence": refine["valid_evidence"],
        "physical_action_seconds": result["physical_time_s"], "action_compute_wall_seconds": result["wall_seconds"],
        "compute_wall_seconds_including_post": row["wall_seconds"], "post_physical_seconds": post["duration_s"],
        "peak_all_label_spill_particles": peak, "full_home_within_deadline": refine["full_home_within_deadline"],
        "no_late_dose_change": refine["no_late_dose_change"], "mass_conserved": result["mass_conserved"],
        "all_states_finite": result["all_states_finite"],
        "started_at_utc": row["started_at_utc"], "completed_at_utc": row["completed_at_utc"]}


def curate(run, audit_path, audit_digest):
    run, audit_path = Path(run).resolve(), Path(audit_path).resolve()
    need(audit_path == run / "overall_audit.json", "Use this run's complete overall audit")
    need(sha(audit_path) == audit_digest, "Independent overall audit SHA differs")
    audit = read(audit_path)
    need(audit.get("complete") is True and audit.get("pass") is True,
         "Complete passing independent evidence audit required before reading journals")
    need(audit["kind"] == KIND and audit["budgets"] == {**COUNTS, "total": 23},
         "The complete23-case settled-correction study is required")
    auditor = audit["auditor_source_file"]
    need(auditor == "projects/pour_policy_refine20/audit_refine_experiment.py",
         "Use the independent completed-experiment auditor")
    need(sha(Path(__file__).resolve().parents[2] / auditor) == audit["auditor_source_sha256"],
         "Independent auditor source changed after audit")
    need(audit["selection"]["validation_only"] is True
         and audit["selection"]["test_used_for_selection"] is False
         and audit["selection"]["engineering_used_for_selection"] is False,
         "Engineering/test cases cannot select the recommendation")
    need(audit["all_physical_sources_inputs_gui_unchanged"] is True
         and audit["engineering_safety_audited_before_formal_training"] is True
         and audit["original_strict_task_was_not_relaxed_or_relabelled"] is True,
         "Source/physics/GUI, engineering safety order or original task proof differs")
    plan_path = run / "experiment_plan.json"
    need(sha(plan_path) == audit["experiment_plan_sha256"], "Frozen experiment plan differs")
    plan = read(plan_path)
    need(plan["budget"] == 23 and plan["draft"] is False, "Final23-case plan required")
    close(plan["absolute_tolerance_ml"], .5, "Wrong primary refinement criterion")
    close(plan["easy_absolute_tolerance_ml"], 1., "Wrong secondary beginner criterion")
    close(plan["original_strict_target_relative_tolerance"], .001, "Original strict criterion changed")
    need(plan["engineering_is_not_training_or_selection"] is True,
         "Engineering must remain a separate safety stage")
    for stage in COUNTS:
        hashes = audit["stage_artifact_sha256"][stage]
        need(set(hashes) == set(TOP), "All seven stage artifacts plus independent stage audit must be bound")
        for name, digest in hashes.items():
            need(sha(run / stage / name) == digest, "Audited stage artifact changed: " + stage + "/" + name)
        need(read(run / stage / "summary.json").get("complete") is True, "Incomplete stage")
        job = read(run / stage / "job_state.json")
        need(job["state"] == "complete" and job["pid"] is None
             and not (run / stage / "worker.lock").exists(), "Stage still active")
    checkpoints = audit["checkpoints"]
    need(set(checkpoints) == {"candidate", "baseline", "recommended"}, "Three frozen checkpoints required")
    for role, evidence in checkpoints.items():
        need(sha(safe_path(run, evidence["file"])) == evidence["sha256"], "Frozen checkpoint changed: " + role)
    reports = audit["episode_reports"]
    need(len(reports) == 23 and len({row["episode_id"] for row in reports}) == 23,
         "Audit must include23 unique cases")
    reports = {row["episode_id"]: row for row in reports}
    cases = []
    for stage, count in COUNTS.items():
        rows = [json.loads(line) for line in (run / stage / "episodes.jsonl").read_text(encoding="utf8").splitlines() if line]
        need(len(rows) == count, "Stage episode count differs")
        need({row["seed"] for row in rows} == SEEDS[stage], "Unexpected stage seeds")
        for row in rows:
            need(row["episode_id"] in reports, "Case absent from independent audit")
            sidecar = run / stage / "completed" / (row["episode_id"] + ".json")
            need(read(sidecar) == row, "Completed sidecar differs from journal")
            for relative, digest in row["artifacts"].items():
                need(sha(safe_path(run / stage, relative)) == digest, "Raw action/post evidence changed")
            cases.append(compact_case(row, stage, reports[row["episode_id"]]))
    need(len({case["episode_id"] for case in cases}) == 23, "Duplicate case names")
    grouped = {}
    for stage in COUNTS:
        roles = sorted({case["role"] for case in cases if case["stage"] == stage})
        grouped[stage] = {role: metrics([case for case in cases if case["stage"] == stage and case["role"] == role])
                          for role in roles}
    paired = []
    for seed in sorted(SEEDS["test"]):
        group = {case["role"]: case for case in cases if case["stage"] == "test" and case["seed"] == seed}
        need(set(group) == {"baseline", "candidate"}, "Test must retain both roles for every seed")
        paired.append({"seed": seed, "baseline": group["baseline"], "candidate": group["candidate"]})
    selection = audit["selection"]
    need(selection["recommended_role"] in ("baseline", "candidate"), "Unknown validation recommendation")
    candidate_angle = selection["candidate_approach_tilt_deg"]
    for case in cases:
        if case["stage"] in ("validation", "test"):
            need(case["correction_enabled"] is (case["role"] == "candidate"),
                 "Frozen correction enablement differs")
            close(case["approach_tilt_deg"], 84. if case["role"] == "baseline" else candidate_angle,
                  "Frozen correction angle differs")

    return {"complete": True, "kind": KIND,
        "record_created_at_utc": datetime.now(timezone.utc).isoformat(),
        "actual_started_at_utc": min(case["started_at_utc"] for case in cases),
        "actual_completed_at_utc": max(case["completed_at_utc"] for case in cases),
        "independent_audit_raw_sha256": audit_digest,
        "experiment_plan_sha256": audit["experiment_plan_sha256"],
        "stage_artifact_sha256": audit["stage_artifact_sha256"], "checkpoints": checkpoints,
        "task_contract": {"target_ml": 20., "primary_absolute_tolerance_ml": .5, "secondary_easy_absolute_tolerance_ml": 1.,
            "deadline_physical_seconds": 24., "maximum_all_label_peak_spill_particles": 1,
            "home_quiet_frames": 12, "post_observation_physical_seconds": 2.,
            "original_strict_relative_tolerance": .001, "original_strict_absolute_tolerance_ml": .02,
            "original_strict_standard_unchanged": True},
        "physics": {"solver": "live_PBF_action_conditioned", "physics_hz": 60,
            "pressure_iterations": 6, "minimum_substeps": 4, "cuda_graph": False,
            "spacing_m": .0027, "particle_volume_ml": .019683,
            "total_particles": 11625, "source_particles": 2880,
            "volume_measurement": "fixed-identity particle-mass simulation truth",
            "contact_force_grasp_simulated": False},
        "budget": {**COUNTS, "total": 23}, "selection": selection,
        "baseline_is_coarse_only_zero_offset": True, "candidate_is_single_settled_correction": True,
        "engineering_used_for_model_selection": False, "all_motion_speeds_fixed": True,
        "stage_metrics": grouped, "test_pairs": paired, "cases": cases,
        "recommended_test_metrics": grouped["test"][selection["recommended_role"]],
        "default_gui_deployment": False, "all_continuous_goals_guaranteed": False,
        "limitations": ["Separate fixed20mL +/-0.5mL primary curriculum; +/-1mL and original +/-0.020mL remain separate secondary criteria.",
            "Three engineering probes establish safety only and never choose the trained angle.",
            "Five narrow-randomization paired test seeds do not establish arbitrary20--40mL performance.",
            "One bounded settled correction and finite approach-angle calibration are not PPO or a neural policy.",
            "The measured-feedback principle resembles V2 but this is not the old V2 controller or a paired comparison with its historical results.",
            "Kinematic pregrasp and privileged particle truth do not validate contact grasp, a real scale, or hardware."]}



build_summary = curate


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--audit-sha256", required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args()


def new_output(path):
    need(path is not None, "A new output directory is required")
    path = Path(path).resolve()
    need(not path.exists(), "Output directory must be new; do not overwrite selected evidence")
    path.mkdir(parents=True)
    return path


def main():
    args = arguments()
    summary = curate(args.run, args.audit, args.audit_sha256)
    if args.check_only:
        print(json.dumps({"complete": True, "cases": len(summary["cases"]),
                          "recommended_role": summary["selection"]["recommended_role"]})); return
    output = new_output(args.output)
    (output / "refine20_results.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf8")
    print("Wrote " + str(output / "refine20_results.json"))


if __name__ == "__main__":
    main()
