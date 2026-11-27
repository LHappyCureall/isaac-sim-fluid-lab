"""Read-only integrity summary of the four completed V10 development pilots.

This checks records and provenance, not independently recorded particle
arrays or a new physical run. No model is trained, selected or deployed.
"""
import argparse
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

REFERENCE = "projects/pour_policy_training_v9/run_fine_search.py"
REFERENCE_SHA256 = "8a9f10e4d2448ff67ab44e5e6bf58c01ecaffd2d8b3a4441249403025dcdb251"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf8"),
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError("Nonfinite JSON")))


def summarize(workspace):
    workspace = workspace.resolve()
    path = workspace / REFERENCE
    if sha(path) != REFERENCE_SHA256:
        raise ValueError("Reference evidence validator changed")
    spec = importlib.util.spec_from_file_location("engineering_evidence_validator", path)
    helper = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = helper
    exec(compile(path.read_bytes(), str(path), "exec"), helper.__dict__)
    runs = []
    for number in range(1, 5):
        name = f"pilot_{number:03d}"
        run = workspace / "droplet_out/pour_policy_range_v10" / name
        plan, summary = read(run / "protocol.json"), read(run / "summary.json")
        identity = hashlib.sha256(json.dumps({k:v for k,v in plan.items() if k != "protocol_id"},
            sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        if (not summary["complete"] or summary["episodes"] != 3
                or identity != plan["protocol_id"] or identity != summary["protocol_id"]):
            raise ValueError("Incomplete or changed development protocol")
        if len(plan["source_sha256"]) != 22 or len(plan["input_sha256"]) != 6:
            raise ValueError("Expected22 source and6 geometry inputs")
        for key, folder in (("source_sha256", "source_snapshot"), ("input_sha256", "input_snapshot")):
            for relative, expected in plan[key].items():
                file = (run / folder / relative).resolve()
                if not file.is_relative_to((run / folder).resolve()) or sha(file) != expected:
                    raise ValueError("Archived snapshot changed")
        for relative, expected in plan["gui_sha256"].items():
            if sha(workspace / relative) != expected:
                raise ValueError("Protected GUI model changed")
        rows = [json.loads(line) for line in (run / "episodes.jsonl").read_text().splitlines() if line.strip()]
        if len(rows) != 3 or {r["target_ml"] for r in rows} != {20., 30., 40.}:
            raise ValueError("Development target coverage changed")
        if len({r["episode_id"] for r in rows}) != 3:
            raise ValueError("Duplicate development identity")
        helper.OUT = run
        cases = []
        for row in rows:
            helper.validate_row(row)
            result = row["result"]
            actual = None
            with gzip.open(run / "traces" / (row["episode_id"] + ".steps.jsonl.gz"), "rt", encoding="utf8") as stream:
                for line in stream:
                    item = json.loads(line)
                    if item.get("type") == "effective_direct_sources":
                        actual = item
                        break
            controller = "projects/pour_policy_training_v10/range_control.py"
            if (actual is None or actual["actual_controller_sha256"] != plan["source_sha256"][controller]
                    or actual["deadline_physical_seconds"] != 24.):
                raise ValueError("Actual controller evidence differs")
            if result["particle_count"] != 11625 or result["source_particle_count"] != 2880:
                raise ValueError("Particle identity counts changed")
            if abs(result["error_ml"] - (result["received_ml"] - row["target_ml"])) > 1e-10:
                raise ValueError("Signed dose error differs")
            cases.append({"episode_id": row["episode_id"], "target_ml": row["target_ml"], "seed": row["seed"],
                "received_ml": result["received_ml"], "signed_error_ml": result["error_ml"],
                "action_physical_s": result["physical_time_s"], "case_wall_s": row["wall_seconds"],
                "assessment": row["assessment"], "mass_conserved_reported": result["mass_conserved"],
                "finite_reported": result["all_states_finite"], "complete": result["complete"],
                "measurement_basis": result["measurement_basis"], "artifacts_sha256": row["artifacts"]})
        runs.append({"run_id": name, "protocol_id": identity, "protocol_sha256": sha(run / "protocol.json"),
            "journal_sha256": sha(run / "episodes.jsonl"), "controller_sha256": plan["source_sha256"][controller],
            "sources": plan["source_sha256"], "inputs": plan["input_sha256"], "cases": cases})
    return {"schema_version": 1, "kind": "V10_completed_development_integrity_summary",
        "actual_case_count": 12, "independent_physics_audit": False, "new_simulation_run": False,
        "formal_training_180_started": False, "trained_or_deployed_model": False,
        "source_measurement": "fixed_particle_mass_simulation_truth_not_scale_or_torque",
        "criterion": {"deadline_physical_s":24., "relative_error":.001, "max_peak_all_label_spill_particles":1},
        "reference_validator_sha256": REFERENCE_SHA256, "runs": runs,
        "limitations": ["Same development seed across four different controller revisions; not independent evaluation.",
                        "Raw logs and geometry caches stay local; hashes bind their original bytes.",
                        "Numerical flags are the original run reports, not a new audit of complete GPU state arrays."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Refuse to overwrite a previous selected summary")
    value = summarize(args.workspace)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf8")
    print(json.dumps({"cases":12, "integrity_checked":True, "independent_physics_audit":False,
                      "strict_success_count":sum(c["assessment"]["feasible"] for r in value["runs"] for c in r["cases"])}))


if __name__ == "__main__":
    main()
