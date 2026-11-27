"""Twelve real coarse-prefix calibration cases with unchanged actual goals.

Use SDK Python. Only the coarse actuator's declared internal control-target
offset varies. Full lift, actual78anchor quiet, home and2s post are measured.
No micro correction, learned-policy test or held-out goal is performed here.
The matrix is frozen before CUDA initialization. No completed case is re-poured.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
CONTROL = None
SPACING = .0027
COARSE = {"prepare_fraction": 1., "dose_tilt_deg_s": 25.6,
          "return_fraction": 1., "tail_lead_s": .26,
          "stop_bias_ml": 10., "max_dose_u_accel": 6.}


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf8")
    os.replace(temporary, path)


def TargetPourController(geometry, target_ml, spacing, policy):
    profile = CONTROL.CorrectionPolicy(dict(COARSE), deadline_s=24., **policy)
    controller = CONTROL.TargetPourController(geometry, target_ml=target_ml,
                                              spacing=spacing, policy=profile)
    actual = Path(sys.modules[type(controller).__module__].__file__).resolve()
    if actual != Path(CONTROL.__file__).resolve():
        raise RuntimeError("Actual controller differs from response experiment")
    return controller


def validate_matrix(matrix):
    if not isinstance(matrix, dict) or matrix.get("draft") is not False:
        raise ValueError("A finalized explicit matrix is required")
    cases = matrix.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("Matrix has no cases")
    ids = set()
    seed_splits = {}
    for row in cases:
        identifier, target, seed = row["case_id"], row["target_ml"], row["seed"]
        if (not isinstance(identifier, str) or not identifier or len(identifier) > 96
                or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in identifier)
                or identifier in ids):
            raise ValueError("Case identities must be unique safe strings")
        ids.add(identifier)
        if (isinstance(target, bool) or not isinstance(target, (int, float))
                or not math.isfinite(target) or not 20 <= target <= 40
                or type(seed) is not int or not 0 <= seed < 2**32
                or row.get("split") not in ("development", "train", "validation")):
            raise ValueError("Invalid collection target/seed/split")
        if seed in seed_splits and seed_splits[seed] != row["split"]:
            raise ValueError("One seed cannot occur in two collection splits")
        seed_splits[seed] = row["split"]
        if not isinstance(row.get("policy"), dict):
            raise ValueError("Every case declares its full action profile")
        json.dumps(row["policy"], allow_nan=False)
        if set(row["policy"]) & {"coarse", "deadline_s", "target_ml", "spacing"}:
            raise ValueError("Collection actions cannot alter the task or physics")
    return cases


def main():
    global CONTROL
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--matrix", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    matrix = json.loads(args.matrix.read_text(encoding="utf8"))
    cases = validate_matrix(matrix)
    output = args.output.resolve()
    permitted = (ROOT / "droplet_out/pour_policy_adaptive_response").resolve()
    if not output.is_relative_to(permitted) or output == permitted:
        raise ValueError("Use a named new adaptive_response output directory")
    if output.exists() and any(output.iterdir()):
        raise RuntimeError("Existing evidence is immutable; never restart by overwriting")
    for path in list((ROOT / "extscache").glob("omni.kit.pip_archive-*/pip_prebundle")) + [
            ROOT / "exts/omni.pip.compute/pip_prebundle"]:
        if path.is_dir():
            sys.path.insert(0, str(path))
    evidence = load("response_v9_original_evidence", ROOT / "projects/pour_policy_training_v9/run_fine_search.py")
    evidence.OUT = output
    geometry_module = load("response_lift_geometry", ROOT / "assets/robot_testtube_pour/scripts/mouth_pivot_lift_geometry.py")
    geometry = geometry_module.load_geometry(geometry_module.DEFAULT_CACHE_FILE,
                                            geometry_module.DEFAULT_CACHE_SHA256)
    CONTROL = load("response_controller", HERE / "prefix_control.py")
    samples = load("response_sample_schema", HERE / "response_samples.py")
    for case in cases:
        TargetPourController(geometry, case["target_ml"], SPACING, case["policy"])
    source_names = tuple(dict.fromkeys((*evidence.SOURCE_NAMES,
        "projects/pour_policy_training_v10/range_control.py",
        *("projects/pour_policy_adaptive_learning/" + name for name in
          ("response_control.py", "cycle_control.py", "prefix_control.py", "response_samples.py", "response_env.py", "cycle_env.py", "run_prefix_calibration.py")))))
    sources = {name: sha(ROOT / name) for name in source_names}
    inputs = dict(geometry.INPUT_SHA256)
    inputs[Path(geometry.CACHE_FILE).relative_to(ROOT).as_posix()] = geometry.CACHE_SHA256
    ancestor = ROOT / "projects/pour_policy_training_v3/fast_path_smooth_cache.npz"
    inputs[ancestor.relative_to(ROOT).as_posix()] = sha(ancestor)
    guis = {name: sha(ROOT / name) for name in (
        "assets/robot_testtube_pour/policy/precision_v2.json",
        "assets/robot_testtube_pour/policy/target_volume_policy.json",
        "assets/robot_testtube_pour/policy/target_volume_candidate.json")}
    points, labels = geometry.initial_water_points(SPACING)
    if len(points) != 11625 or int(sum(labels == 1)) != 2880:
        raise RuntimeError("Initial particle identities changed")
    plan = {"kind": "adaptive_coarse_prefix_calibration_v13",
        "controller_source": "projects/pour_policy_adaptive_learning/prefix_control.py", "collector_source": "projects/pour_policy_adaptive_learning/run_prefix_calibration.py", "created_at_utc": now(), "matrix": matrix, "matrix_raw_sha256": sha(args.matrix),
        "deadline_s": 24., "relative_tolerance": .001, "peak_all_label_spill_limit": 1,
        "spacing_m": SPACING, "particle_ml": SPACING**3 * 1e6,
        "total_particles": 11625, "source_particles": 2880,
        "initial_source_ml": 56.68704, "post_seconds": 2.,
        "solver": "unchanged_live_PBF_60Hz_pressure6_min4_graph_off",
        "measurement": "actual_full_R_particle_mass_not_scale_or_torque",
        "source_sha256": sources, "input_sha256": inputs, "gui_sha256": guis,
        "collection_task_success_not_required": True,
        "phase_training_time_not_hidden": True, "no_saved_liquid_replay": True}
    plan["protocol_id"] = hashlib.sha256(json.dumps(plan, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    if args.dry_run:
        print(json.dumps(plan, indent=2)); return
    for name, digest in {**sources, **inputs, **guis}.items():
        if sha(ROOT / name) != digest:
            raise RuntimeError("Input changed before launch: " + name)
    output.mkdir(parents=True, exist_ok=True)
    lock = output / "worker.lock"
    with lock.open("x", encoding="utf8") as stream:
        json.dump({"pid": os.getpid(), "command": sys.argv, "started_at_utc": now()}, stream)
    rows = []
    try:
        write(output / "protocol.json", plan)
        for key, folder in (("source_sha256", "source_snapshot"), ("input_sha256", "input_snapshot")):
            for name, digest in plan[key].items():
                destination = output / folder / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ROOT / name, destination)
                if sha(destination) != digest:
                    raise RuntimeError("Snapshot changed: " + name)
        write(output / "matrix.json", matrix)
        write(output / "job_state.json", {"state": "starting", "pid": os.getpid(), "started_at_utc": now()})
        sdk = load("response_sdk", ROOT / "train_target_volume_policy.py")
        write(output / "versions.json", sdk.configure_sdk())
        environment = load("response_environment", HERE / "cycle_env.py")
        env = environment.CycleEnv(geometry=geometry, control=sys.modules[__name__], spacing=SPACING)
        for case in cases:
            for name, digest in {**sources, **inputs, **guis}.items():
                if sha(ROOT / name) != digest:
                    raise RuntimeError("Frozen runtime changed: " + name)
            identifier = case["case_id"]
            trace = output / "traces" / (identifier + ".steps.jsonl.gz")
            post_path = trace.with_name(identifier + ".post.jsonl.gz")
            write(output / "job_state.json", {"state": "running", "pid": os.getpid(), "current_episode": identifier})
            write(output / "progress.json", {"state": "running", "pid": os.getpid(),
                "completed": len(rows), "budget": len(cases), "current_episode": identifier})
            print("START " + identifier, flush=True)
            started, wall = now(), time.perf_counter()
            result = env.run_episode(case["target_ml"], case["seed"], policy=case["policy"],
                                     randomize=True, record_path=trace)
            status = dict(env.last_controller.status)
            action_events = json.loads(json.dumps(env.last_controller.response_events, allow_nan=False))
            post = None
            if result["valid_physics"]:
                try:
                    post = env.post_observe(post_path)
                except (RuntimeError, ValueError, FloatingPointError) as error:
                    post = {"error": repr(error), "all_states_finite": False,
                        "mass_array_unchanged": False, "maximum_received_change_ml": None,
                        "end_physical_time_s": float(env.fluid.time),
                        "measurement_basis": "failed_post_not_acceptance"}
            peak = env.max_all_label_spilled_particles
            row = {"episode_id": identifier, "stage": "prefix_calibration", "role": "candidate",
                "split": case["split"], "group_id": "seed_" + str(case["seed"]),
                "target_ml": case["target_ml"], "seed": case["seed"],
                "declared_policy": case["policy"], "protocol_id": plan["protocol_id"],
                "started_at_utc": started, "completed_at_utc": now(), "result": result,
                "status": status, "assessment": evidence.assessment(result, status, peak, post),
                "strict_assessment": samples.strict_assessment(result, status, peak, post),
                "post_observation": post, "motion_metrics": dict(env.fast_metrics),
                "wall_seconds": time.perf_counter() - wall,
                "response_events": action_events,
                "physics_fingerprint": hashlib.sha256(json.dumps({"sources": sources,
                    "inputs": inputs, "spacing": SPACING}, sort_keys=True).encode()).hexdigest(),
                "artifacts": {p.relative_to(output).as_posix(): sha(p)
                              for p in (trace, post_path) if p.is_file()}}
            row["response_samples"] = samples.samples_from_events(row)
            write(output / "completed" / (identifier + ".json"), row)
            with (output / "episodes.jsonl").open("a", encoding="utf8") as stream:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
            rows.append(row)
            evidence.validate_row(row)
            print(json.dumps({"id": identifier, "dose_ml": result["received_ml"],
                "physical_s": result["physical_time_s"], "responses": len(row["response_samples"]),
                "strict_task": row["strict_assessment"]}, allow_nan=False), flush=True)
            if not row["assessment"]["valid_evidence"]:
                raise RuntimeError("Invalid numerical/measurement evidence; stop and preserve case")
        for name, digest in {**sources, **inputs, **guis}.items():
            if sha(ROOT / name) != digest:
                raise RuntimeError("Completed runtime changed: " + name)
        data = [sample for row in rows for sample in row["response_samples"]]
        with (output / "responses.jsonl").open("x", encoding="utf8") as stream:
            for sample in data:
                stream.write(json.dumps(sample, allow_nan=False) + "\n")
        write(output / "summary.json", {"complete": True, "kind": plan["kind"],
            "protocol_id": plan["protocol_id"], "episodes": len(rows), "response_samples": len(data),
            "responses_sha256": sha(output / "responses.jsonl"),
            "strict_task_passes": sum(row["strict_assessment"]["feasible"] for row in rows),
            "new_goal_policy_trained": False, "gui_deployment": False})
        write(output / "progress.json", {"state": "complete", "completed": len(rows), "budget": len(cases)})
        write(output / "job_state.json", {"state": "complete", "pid": None, "completed_at_utc": now()})
    except BaseException as error:
        write(output / "job_state.json", {"state": "failed", "pid": None,
            "completed": len(rows), "error": repr(error), "at_utc": now()})
        raise
    finally:
        lock.unlink()


if __name__ == "__main__":
    main()
