"""Compact publication of one audited 12-case coarse-prefix calibration.

No physics execution or fitted model. Original goals, strict dose gates and all
failures remain visible. Only the internal coarse control target was offset.
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys


ROOT = Path(__file__).resolve().parents[2]
AUDIT_SOURCE = "projects/pour_policy_adaptive_learning/audit_prefix_calibration.py"
AUDIT_SOURCE_SHA256 = "d0e6acc33e93baa57edb6e8b689f79d7b5d251821bc20a8167b192ee49d9aec6"
CONTROL_SOURCE = "projects/pour_policy_adaptive_learning/prefix_control.py"
CONTROL_SOURCE_SHA256 = "7fd23b7d35bc1de8b181a8a2784231b0c6812a6a426211a8a000cb0f83dea699"
KIND = "adaptive_coarse_prefix_calibration_v13"
SCOPE = "record_integrity_and_independent_physical_motion_mass_prefix_audit_not_task_success"
ARTIFACTS = ("protocol.json", "summary.json", "job_state.json", "episodes.jsonl", "responses.jsonl", "matrix.json", "versions.json")
FIXED_COARSE = {"prepare_fraction": 1., "dose_tilt_deg_s": 25.6, "return_fraction": 1.,
                "tail_lead_s": .26, "stop_bias_ml": 10., "max_dose_u_accel": 6.}
CONDITIONS = {25.: (-.5, -1.), 35.: (0., .5), 40.: (1., 1.5)}
SEEDS = (193101, 193102)


def need(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    value = json.loads(Path(path).read_text(encoding="utf8"))
    json.dumps(value, allow_nan=False)
    return value


def inside(root, name):
    root = Path(root).resolve()
    path = (root / name).resolve()
    need(not Path(name).is_absolute() and path.is_relative_to(root) and path.is_file(), "Unsafe/missing bound artifact")
    return path


def strict_gate(row):
    result, status, post = row["result"], row["status"], row["post_observation"]
    target, received, elapsed = row["target_ml"], result["received_ml"], result["physical_time_s"]
    need(all(type(value) in (int, float) and math.isfinite(value) for value in (target, received, elapsed)),
         "Nonfinite actual target/dose/time")
    need(result["target_ml"] == target and target in CONDITIONS, "Original physical target changed")
    peak = row["strict_assessment"]["maximum_all_label_spilled_particles"]
    need(type(peak) is int and peak >= 0, "Actual all-label peak spill count required")
    valid = bool(result.get("valid_physics") is True and result.get("mass_conserved") is True
        and result.get("all_states_finite") is True and result.get("source_partition_conserved") is True
        and result.get("max_below_table_particles") == 0
        and abs(result["measurement_physical_time_s"] - elapsed) < 1e-8
        and post.get("all_states_finite") is True and post.get("mass_array_unchanged") is True
        and not post.get("error") and post.get("maximum_below_table_particles", 0) == 0
        and post.get("physical_steps") == 120
        and abs(post["start_physical_time_s"] - elapsed) < 1e-8
        and post["end_physical_time_s"] - post["start_physical_time_s"] >= 2 - 1e-8)
    home = bool(result.get("complete") is True and status.get("done") is True
        and status.get("home_static") is True and status.get("stable_frames", 0) >= 12
        and status.get("source_airborne_particles") == 0 and not status.get("timed_out") and elapsed <= 24.)
    tolerance = .001 * target
    dose, late = target - tolerance <= received <= target + tolerance, post.get("maximum_received_change_ml") == 0.
    actual = {"valid_evidence": valid, "dose_passed": dose, "full_home_within_deadline": home,
              "maximum_all_label_spilled_particles": peak, "spill_passed": peak <= 1,
              "no_late_dose_change": late, "tolerance_ml": tolerance,
              "feasible": bool(valid and dose and home and peak <= 1 and late)}
    need(actual == row["strict_assessment"], "Strict gate differs from original actual dose/time evidence")
    return actual


def curate(run, audit_path, expected_audit_sha256, workspace=ROOT):
    run, workspace = Path(run).resolve(), Path(workspace).resolve()
    need(sha(audit_path) == expected_audit_sha256, "Pinned independent audit raw SHA mismatch")
    audit = read(audit_path)
    need(audit.get("complete") is True and audit.get("pass") is True and audit.get("kind") == KIND
         and audit.get("scope") == SCOPE and audit.get("audit_tool_sha256") == AUDIT_SOURCE_SHA256
         and sha(inside(workspace, AUDIT_SOURCE)) == AUDIT_SOURCE_SHA256,
         "Expected complete pinned independent prefix audit")
    bindings = audit["input_artifact_sha256"]
    for name in ARTIFACTS:
        need(bindings.get(name) == sha(inside(run, name)), "Top-level artifact changed: " + name)
    plan, summary, job = read(run / "protocol.json"), read(run / "summary.json"), read(run / "job_state.json")
    matrix = read(run / "matrix.json")
    canonical = hashlib.sha256(json.dumps({key: value for key, value in plan.items() if key != "protocol_id"},
        sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    need(plan["kind"] == summary["kind"] == KIND and plan["protocol_id"] == canonical
         == summary["protocol_id"] == audit["protocol_id"], "Prefix protocol identity differs")
    need(summary["complete"] is True and job["state"] == "complete" and job["pid"] is None
         and not (run / "worker.lock").exists(), "Prefix worker is not finalized")
    need(plan["matrix"] == matrix and matrix["draft"] is False, "Frozen matrix changed")
    need(plan["deadline_s"] == 24. and plan["relative_tolerance"] == .001
         and plan["peak_all_label_spill_limit"] == 1 and plan["total_particles"] == 11625
         and plan["source_particles"] == 2880 and plan["spacing_m"] == .0027,
         "Fixed physical task contract changed")
    for category, folder in (("source_sha256", "source_snapshot"), ("input_sha256", "input_snapshot")):
        for name, digest in plan[category].items():
            need(sha(inside(workspace, name)) == sha(inside(run / folder, name)) == digest,
                 "Frozen current/snapshot source or input changed: " + name)
    for name, digest in plan["gui_sha256"].items():
        need(sha(inside(workspace, name)) == digest, "GUI weight changed")
    need(plan["source_sha256"][CONTROL_SOURCE] == CONTROL_SOURCE_SHA256, "Prefix control implementation differs")
    rows = [json.loads(line) for line in (run / "episodes.jsonl").read_text(encoding="utf8").splitlines() if line.strip()]
    cases, reports = matrix["cases"], audit["episode_reports"]
    need(len(rows) == len(cases) == len(reports) == audit["episodes_audited"] == summary["episodes"] == 12,
         "Exactly all twelve calibrated cases are required")
    ids = [case["case_id"] for case in cases]
    need([row["episode_id"] for row in rows] == [report["episode_id"] for report in reports] == ids
         and len(set(ids)) == 12 and {path.stem for path in (run / "completed").glob("*.json")} == set(ids),
         "Sidecar/journal/audit/matrix case identity differs")
    expected_conditions = {(target, offset, seed) for target, offsets in CONDITIONS.items() for offset in offsets for seed in SEEDS}
    need({(case["target_ml"], case["policy"]["control_target_offset_ml"], case["seed"]) for case in cases} == expected_conditions,
         "The six paired conditions/two development seeds changed")
    compact, sidecars, raw_artifacts = [], {}, {}
    for row, case, report in zip(rows, cases, reports):
        need(row["protocol_id"] == plan["protocol_id"] and row["target_ml"] == case["target_ml"]
             and row["seed"] == case["seed"] and row["split"] == case["split"] == "development"
             and row["declared_policy"] == case["policy"], "Actual goal/offset/seed/split differs from matrix")
        sidecar = inside(run / "completed", row["episode_id"] + ".json")
        need(read(sidecar) == row, "Completed sidecar differs from immutable journal")
        serialized = json.dumps(row, indent=2, allow_nan=False) + "\n"
        need(sidecar.read_bytes() in (serialized.encode(), serialized.replace("\n", "\r\n").encode()),
             "Sidecar bytes differ from original writer serialization")
        sidecars[row["episode_id"]] = sha(sidecar)
        names = {"traces/" + row["episode_id"] + ending for ending in (".steps.jsonl.gz", ".post.jsonl.gz")}
        need(set(row["artifacts"]) == names, "Missing or extra raw action/post artifact")
        for name, digest in row["artifacts"].items():
            need(sha(inside(run, name)) == digest, "Raw action/post SHA changed")
            raw_artifacts[name] = digest
        gate, result, status = strict_gate(row), row["result"], row["status"]
        need(gate == report["strict_assessment"] and row["response_events"] == row["response_samples"] == [],
             "Audit dose gate or prefix-only action scope differs")
        offset, target = case["policy"]["control_target_offset_ml"], case["target_ml"]
        need(status["actual_target_ml"] == status["target_ml"] == target
             and status["control_target_ml"] == target + offset
             and status["prefix_target_offset_ml"] == status["control_target_offset_ml"] == offset,
             "Actual target was shifted instead of the internal control signal")
        need(all(result["policy_params"][key] == status["policy_params"][key] == value for key, value in FIXED_COARSE.items()),
             "Coarse speed or another fixed parameter changed")
        anchor = status["prefix_anchor_observation"]
        compact.append({"episode_id": row["episode_id"], "seed": row["seed"], "split": row["split"],
            "target_ml": target, "internal_control_target_ml": target + offset, "control_target_offset_ml": offset,
            "received_ml": result["received_ml"], "signed_error_ml": result["received_ml"] - target,
            "shortfall_ml": target - result["received_ml"], "physical_action_seconds": result["physical_time_s"],
            "wall_seconds_including_post": row["wall_seconds"], "strict_assessment": gate,
            "actual_anchor": {key: anchor[key] for key in ("physical_time_s", "received_ml", "tube_remaining_ml",
                "tilt_deg", "quiet_measured_seconds", "source_airborne_particles", "source_inside_velocity_rms_m_s")},
            "home_additional_received_ml": result["received_ml"] - anchor["received_ml"]})
    need(len(raw_artifacts) == 24 and len(sidecars) == 12 and summary["response_samples"] == 0
         and (run / "responses.jsonl").read_bytes() == b"", "Unexpected response labels or missing attempts")
    passes = sum(row["strict_assessment"]["feasible"] for row in compact)
    need(passes == audit["strict_task_passes"] == summary["strict_task_passes"], "Failure counts changed")
    conditions = []
    for target, offsets in CONDITIONS.items():
        for offset in offsets:
            group = [row for row in compact if row["target_ml"] == target and row["control_target_offset_ml"] == offset]
            doses = [row["received_ml"] for row in group]
            conditions.append({"target_ml": target, "control_target_offset_ml": offset, "internal_control_target_ml": target + offset,
                "cases": 2, "seeds": [row["seed"] for row in group], "received_ml": doses,
                "minimum_received_ml": min(doses), "maximum_received_ml": max(doses),
                "shortfalls_ml": [row["shortfall_ml"] for row in group],
                "mean_physical_action_seconds": statistics.mean(row["physical_action_seconds"] for row in group),
                "mean_wall_seconds_including_post": statistics.mean(row["wall_seconds_including_post"] for row in group),
                "peak_spill_particles": max(row["strict_assessment"]["maximum_all_label_spilled_particles"] for row in group),
                "both_actual_anchor_doses_below_goal": all(row["actual_anchor"]["received_ml"] < target for row in group)})
    return {"complete": True, "kind": KIND, "record_created_at_utc": datetime.now(timezone.utc).isoformat(),
        "actual_started_at_utc": min(row["started_at_utc"] for row in rows),
        "actual_completed_at_utc": max(row["completed_at_utc"] for row in rows), "protocol_id": plan["protocol_id"],
        "independent_audit_raw_sha256": expected_audit_sha256, "audit_tool_sha256": AUDIT_SOURCE_SHA256,
        "top_artifact_sha256": bindings, "completed_sidecar_sha256": sidecars, "raw_artifact_sha256": raw_artifacts,
        "source_sha256": plan["source_sha256"], "input_fingerprints_sha256": sorted(plan["input_sha256"].values()),
        "gui_policy_sha256_unchanged": plan["gui_sha256"], "fixed_coarse_parameters": FIXED_COARSE,
        "task_contract": {"deadline_physical_s": 24., "relative_tolerance": .001, "max_peak_spill_particles": 1,
                          "full_original_home_quiet_frames": 12, "post_observation_s": 2.},
        "cases": compact, "per_condition": conditions, "strict_task_passes": passes,
        "all_cases_finite_mass_home_post_valid": all(row["strict_assessment"]["valid_evidence"]
            and row["strict_assessment"]["full_home_within_deadline"] and row["strict_assessment"]["no_late_dose_change"] for row in compact),
        "prefix_only_no_fine_dosing": True, "new_goal_policy_trained": False, "default_gui_deployment": False,
        "all_continuous_goals_guaranteed": False,
        "limitation": "Paired developmental prefix measurements only; dose failures retained, not precision training or held-out acceptance."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--audit-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, default=ROOT)
    args = parser.parse_args()
    result = curate(args.run, args.audit, args.audit_sha256, args.workspace)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"complete": True, "cases": 12, "strict_task_passes": result["strict_task_passes"],
                      "new_goal_policy_trained": False}, indent=2))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, TypeError) as error:
        print("Prefix curation refused: " + str(error), file=sys.stderr)
        raise SystemExit(1)
