"""Curate the completed fixed20mL beginner study after independent raw audit.

This CPU-only tool never runs physics or chooses a model from test outcomes.
The original strict assessment remains a separately named secondary result.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path


KIND = "easy20_fixed_target_offset_search_v1"
COUNTS = {"train": 6, "validation": 4, "test": 10}
SEEDS = {"train": {194101, 194102}, "validation": {194201, 194202},
         "test": set(range(194301, 194306))}
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
        "joint_passes": sum(case["easy_feasible"] for case in cases),
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
    result, easy, strict = row["result"], row["easy_assessment"], row["strict_assessment"]
    need(row["split"] == stage and row["stage"] == stage, "Stage label differs")
    close(row["target_ml"], 20., "Actual target changed")
    close(result["target_ml"], 20., "Result target changed")
    error = result["received_ml"] - 20.
    close(result["error_ml"], error, "Actual signed error differs")
    close(result["absolute_error_ml"], abs(error), "Actual absolute error differs")
    close(easy["tolerance_ml"], 1., "Beginner tolerance changed")
    close(strict["tolerance_ml"], .02, "Original strict tolerance changed")
    for assessment in (easy, strict):
        need(assessment["valid_evidence"] is True, "Independent physical evidence is invalid")
    need(easy["dose_passed"] == (19. <= result["received_ml"] <= 21.), "Beginner dose gate differs")
    need(strict["dose_passed"] == (19.98 <= result["received_ml"] <= 20.02), "Original strict dose gate differs")
    for assessment, tolerance in ((easy, 1.), (strict, .02)):
        expected = bool(assessment["valid_evidence"] and 20. - tolerance <= result["received_ml"] <= 20. + tolerance
            and assessment["full_home_within_deadline"] and assessment["spill_passed"]
            and assessment["no_late_dose_change"])
        need(assessment["feasible"] == expected, "Joint acceptance differs")
    need(report["episode_id"] == row["episode_id"], "Audited case identity differs")
    for key in ("stage", "role", "seed"):
        need(report[key] == (stage if key == "stage" else row[key]), "Audited case label differs: " + key)
    for key in ("received_ml", "error_ml"):
        close(report[key], result[key], "Audited measurement differs: " + key)
    if "physical_s" in report:
        close(report["physical_s"], result["physical_time_s"], "Audited physical time differs")
    for key in ("valid_evidence", "dose_passed", "full_home_within_deadline",
                "maximum_all_label_spilled_particles", "spill_passed", "no_late_dose_change", "feasible"):
        need(report["easy_assessment"][key] == easy[key], "Audited beginner assessment differs: " + key)
        need(report["strict_assessment"][key] == strict[key], "Audited strict assessment differs: " + key)
    peak = easy["maximum_all_label_spilled_particles"]
    need(type(peak) is int and peak >= 0, "All-label peak must be a particle count")
    need(easy["spill_passed"] == (peak <= 1), "Spill acceptance differs")
    close(result["physical_time_s"], result["physical_steps"] / 60., "Physical clock is not step/60")
    post = row["post_observation"]
    close(post["duration_s"], 2., "Post observation is not two physical seconds", 1e-8)
    need(post["physical_steps"] == 120, "Post observation must retain120 live steps")
    offset = row["declared_policy"]["control_target_offset_ml"]
    close(result["policy_params"]["control_target_offset_ml"], offset, "Executed offset differs")
    close(report["control_target_offset_ml"], offset, "Audited offset differs")
    anchor = row["status"]["prefix_anchor_observation"]
    anchor_fields = {"physical_time_s": "physical_time_s", "received_ml": "received_ml",
        "remaining_ml": "tube_remaining_ml", "actual_target_ml": "actual_target_ml",
        "control_target_ml": "control_target_ml", "control_target_offset_ml": "control_target_offset_ml"}
    for audited_key, actual_key in anchor_fields.items():
        close(report["actual_prefix_anchor"][audited_key], anchor[actual_key],
              "Audited prefix anchor differs: " + audited_key)
    need(anchor["settled_measurement"] is True, "Prefix anchor must be an actual settled measurement")
    for value in (anchor["coarse_cut_tilt_deg"], anchor["received_ml"], anchor["physical_time_s"]):
        need(type(value) in (int, float) and math.isfinite(value), "Prefix anchor observation is nonfinite")
    return {"episode_id": row["episode_id"], "stage": stage, "role": row["role"],
        "seed": row["seed"], "target_ml": 20., "control_target_offset_ml": offset,
        "actual_coarse_cut_tilt_degrees": anchor["coarse_cut_tilt_deg"],
        "first_quiet_anchor_received_ml": anchor["received_ml"],
        "first_quiet_anchor_physical_seconds": anchor["physical_time_s"],
        "received_ml": result["received_ml"], "error_ml": error, "absolute_error_ml": abs(error),
        "easy_feasible": easy["feasible"], "original_strict_feasible": strict["feasible"],
        "original_strict_dose_passed": strict["dose_passed"],
        "physical_action_seconds": result["physical_time_s"],
        "action_compute_wall_seconds": result["wall_seconds"],
        "compute_wall_seconds_including_post": row["wall_seconds"],
        "post_physical_seconds": post["duration_s"],
        "peak_all_label_spill_particles": peak,
        "full_home_within_deadline": easy["full_home_within_deadline"],
        "no_late_dose_change": easy["no_late_dose_change"],
        "mass_conserved": result["mass_conserved"],
        "all_states_finite": result["all_states_finite"],
        "started_at_utc": row["started_at_utc"], "completed_at_utc": row["completed_at_utc"]}


def curate(run, audit_path, audit_digest):
    run, audit_path = Path(run).resolve(), Path(audit_path).resolve()
    need(audit_path == run / "overall_audit.json", "Use this run's complete overall audit")
    need(sha(audit_path) == audit_digest, "Independent overall audit SHA differs")
    audit = read(audit_path)
    need(audit.get("complete") is True and audit.get("pass") is True,
         "Complete passing independent evidence audit required before reading journals")
    need(audit["kind"] == KIND and audit["budgets"] == {**COUNTS, "total": 20},
         "The complete20-case beginner study is required")
    auditor = audit["auditor_source_file"]
    need(auditor == "projects/pour_policy_easy20/audit_easy_experiment.py",
         "Use the independent completed-experiment auditor")
    need(sha(Path(__file__).resolve().parents[2] / auditor) == audit["auditor_source_sha256"],
         "Independent auditor source changed after audit")
    need(audit["selection"]["validation_only"] is True
         and audit["selection"]["test_used_for_selection"] is False, "Test cannot select the recommendation")
    plan_path = run / "experiment_plan.json"
    need(sha(plan_path) == audit["experiment_plan_sha256"], "Frozen experiment plan differs")
    plan = read(plan_path)
    need(plan["budget"] == 20 and plan["draft"] is False, "Final20-case plan required")
    close(plan["absolute_tolerance_ml"], 1., "Wrong beginner criterion")
    close(plan["original_strict_target_relative_tolerance"], .001, "Original strict criterion changed")
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
    need(len(reports) == 20 and len({row["episode_id"] for row in reports}) == 20,
         "Audit must include20 unique cases")
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
    need(len({case["episode_id"] for case in cases}) == 20, "Duplicate case names")
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
    candidate_offset = selection["candidate_offset_ml"]
    for case in cases:
        if case["stage"] in ("validation", "test"):
            close(case["control_target_offset_ml"], 0. if case["role"] == "baseline" else candidate_offset,
                  "Executed validation/test policy differs from frozen selection")
    return {"complete": True, "kind": KIND,
        "record_created_at_utc": datetime.now(timezone.utc).isoformat(),
        "actual_started_at_utc": min(case["started_at_utc"] for case in cases),
        "actual_completed_at_utc": max(case["completed_at_utc"] for case in cases),
        "independent_audit_raw_sha256": audit_digest,
        "experiment_plan_sha256": audit["experiment_plan_sha256"],
        "stage_artifact_sha256": audit["stage_artifact_sha256"], "checkpoints": checkpoints,
        "task_contract": {"target_ml": 20., "beginner_absolute_tolerance_ml": 1.,
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
        "budget": {**COUNTS, "total": 20}, "selection": selection,
        "candidate_and_baseline_offsets_identical": candidate_offset == 0.,
        "stage_metrics": grouped, "test_pairs": paired, "cases": cases,
        "recommended_test_metrics": grouped["test"][selection["recommended_role"]],
        "default_gui_deployment": False, "all_continuous_goals_guaranteed": False,
        "limitations": ["Separate fixed20mL +/-1mL beginner task; original +/-0.1% task remains unchanged.",
            "Five narrow-randomization paired test seeds do not establish arbitrary20--40mL performance.",
            "One internal coarse stop offset is finite calibration, not PPO or a neural policy.",
            "Kinematic pregrasp and privileged particle truth do not validate contact grasp, a real scale, or hardware.",
            "A baseline recommendation or identical offsets must not be described as a learned improvement."]}


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
    (output / "easy20_results.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf8")
    print("Wrote " + str(output / "easy20_results.json"))


if __name__ == "__main__":
    main()
