"""One frozen stage of40mL coarse-dose-speed-only liquid calibration.

Actual PBF, physical time and original strict raw evidence remain unchanged.
Actual settled deficiency and a conservative full-home budget gate the retry.
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
PROFILE = None
PROFILE_SHA256 = None
CURRENT_SEED = None
ADAPTIVE = ROOT / "projects/pour_policy_adaptive_learning"
import task_score
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
    actual_coarse = dict(COARSE)
    actual_coarse["dose_tilt_deg_s"] = policy["coarse_dose_tilt_deg_s"]
    profile = CONTROL.CorrectionPolicy(actual_coarse, target_ml=target_ml, deadline_s=24.,
        profile_id=PROFILE["profile_id"], profile_sha256=PROFILE_SHA256, condition_seed=CURRENT_SEED, **policy)
    controller = CONTROL.TargetPourController(geometry, target_ml=target_ml,
                                              spacing=spacing, policy=profile)
    actual = Path(sys.modules[type(controller).__module__].__file__).resolve()
    if actual != Path(CONTROL.__file__).resolve():
        raise RuntimeError("Actual controller differs from response experiment")
    return controller


def validate_matrix(matrix):
    if not isinstance(matrix, dict) or matrix.get("draft") is not False:
        raise ValueError("A finalized exact coarse40 stage matrix is required")
    stage = matrix.get("stage")
    if stage not in ("calibration", "validation", "test"):
        raise ValueError("Undeclared stage")
    speeds = {"baseline": 25.6, "coarse_252": 25.2, "coarse_248": 24.8}
    if set(matrix.get("profiles", {})) != set(speeds):
        raise ValueError("All three declared coarse speed profiles must be retained")
    for key, speed in speeds.items():
        if matrix["profiles"][key] != {"enable_retry": True, "first_approach_tilt_deg": 87.,
                "retry_approach_tilt_deg": 87.2, "coarse_dose_tilt_deg_s": speed}:
            raise ValueError("Only the coarse dose speed may differ")
    cases = matrix.get("cases")
    if not isinstance(cases, list) or len(cases) != {"calibration": 6, "validation": 4, "test": 20}[stage]:
        raise ValueError("Complete exact stage budget required")
    ids, conditions = set(), set()
    for row in cases:
        name, target, seed, key = row["case_id"], row["target_ml"], row["seed"], row["profile_key"]
        if (not isinstance(name, str) or not name or len(name) > 100 or name in ids
                or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in name)):
            raise ValueError("Unique safe case identity required")
        ids.add(name)
        if type(target) not in (int, float) or target != 40. or type(seed) is not int:
            raise ValueError("Actual40mL and integer initial-state seed required")
        if key not in speeds or row["policy"] != matrix["profiles"][key]:
            raise ValueError("Every action must match the explicit finite profile")
        if (type(row["policy"]["enable_retry"]) is not bool or row["policy"]["enable_retry"] is not True
                or row["candidate_id"] != key or row["role"] != ("baseline" if key == "baseline" else "candidate")):
            raise ValueError("Both roles must keep existing retry enabled")
        if row["split"] != ("test" if stage == "test" else "development"):
            raise ValueError("Seen development cannot be renamed independent test")
        condition = (seed, key) if stage == "calibration" else (seed, row["role"])
        if condition in conditions:
            raise ValueError("Repeated paired condition")
        conditions.add(condition)
    if stage == "calibration":
        expected = {(seed, key) for seed in (197304, 198304) for key in speeds}
    else:
        seeds = (198201, 198202) if stage == "validation" else tuple(range(198401, 198411))
        expected = {(seed, role) for seed in seeds for role in ("baseline", "candidate")}
    if conditions != expected:
        raise ValueError("Predeclared conditions cannot be omitted or repeated")
    return cases


def main():
    global CONTROL, PROFILE, PROFILE_SHA256, CURRENT_SEED
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--matrix", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    matrix = json.loads(args.matrix.read_text(encoding="utf8"))
    cases = validate_matrix(matrix)
    profile_file = (ROOT / matrix["profile_file"]).resolve()
    if not profile_file.is_relative_to(HERE.resolve()):
        raise ValueError("Profiles must be a named input under the new experiment project")
    PROFILE = json.loads(profile_file.read_text(encoding="utf8"))
    PROFILE_SHA256 = sha(profile_file)
    if (PROFILE.get("kind") != "coarse40_explicit_profiles_v1"
            or PROFILE.get("profiles") != matrix["profiles"]):
        raise ValueError("Declared goal profiles differ from the external frozen input")
    experiment_plan = HERE / "experiment_plan_001.json"
    if matrix.get("experiment_plan_sha256") != sha(experiment_plan):
        raise ValueError("Matrix must bind the predeclared bounded experiment")
    plan_record = json.loads(experiment_plan.read_text(encoding="utf8"))
    protected_file = ROOT / plan_record["protected_versions_file"]
    if sha(protected_file) != plan_record["protected_versions_raw_sha256"]:
        raise ValueError("Protected20/30 version manifest changed")
    protected = json.loads(protected_file.read_text(encoding="utf8"))["files_sha256"]
    for name, digest in protected.items():
        if sha(ROOT / name) != digest:
            raise ValueError("An existing20/30 version changed: " + name)
    strategy = selection = None
    if matrix["stage"] != "calibration":
        selection_file = args.output.resolve().parent / "calibration_selection.json"
        if matrix.get("calibration_selection_sha256") != sha(selection_file):
            raise ValueError("Validation/test must bind the pre-validation calibration selection")
        selection = json.loads(selection_file.read_text(encoding="utf8"))
        key = selection["selected_profile_key"]
        if (selection.get("kind") != "coarse40_calibration_selection_v1"
                or selection.get("calibration_gate_passed") is not True
                or selection["profiles"] != matrix["profiles"] or key == "baseline"
                or any(row["profile_key"] != ("baseline" if row["role"] == "baseline" else key) for row in cases)):
            raise ValueError("Calibration-selected coarse speed may not be changed by validation/test")
    if matrix["stage"] == "test":
        strategy_file = args.output.resolve().parent / "fixed_strategy.json"
        if matrix.get("fixed_strategy_sha256") != sha(strategy_file):
            raise ValueError("Test must bind its frozen post-validation strategy")
        strategy = json.loads(strategy_file.read_text(encoding="utf8"))
        if (strategy.get("kind") != "coarse40_frozen_speed_strategy_v1"
                or strategy.get("validation_gate_passed") is not True
                or strategy["profiles"] != matrix["profiles"]
                or strategy["selected_profile_key"] != selection["selected_profile_key"]
                or strategy["calibration_selection_sha256"] != sha(selection_file)
                or strategy.get("test_used_for_selection") is not False):
            raise ValueError("Final strategy or validation gate mismatch")
    output = args.output.resolve()
    permitted = (ROOT / "droplet_out/pour_policy_coarse40").resolve()
    if not output.is_relative_to(permitted) or output == permitted:
        raise ValueError("Use a named new coarse40 output directory")
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
    CONTROL = load("response_controller", HERE / "coarse_control.py")
    samples = load("response_sample_schema", ADAPTIVE / "response_samples.py")
    for case in cases:
        CURRENT_SEED = case["seed"]
        TargetPourController(geometry, case["target_ml"], SPACING, case["policy"])
    source_names = tuple(dict.fromkeys((*evidence.SOURCE_NAMES,
        "projects/pour_policy_training_v10/range_control.py",
        *("projects/pour_policy_adaptive_learning/" + name for name in
          ("response_control.py", "cycle_control.py", "response_samples.py", "response_env.py", "cycle_env.py")),
        "projects/pour_policy_coarse40/run_coarse_test.py", "projects/pour_policy_coarse40/task_score.py",
        "projects/pour_policy_coarse40/coarse_control.py")))
    sources = {name: sha(ROOT / name) for name in source_names}
    inputs = dict(geometry.INPUT_SHA256)
    inputs[Path(geometry.CACHE_FILE).relative_to(ROOT).as_posix()] = geometry.CACHE_SHA256
    ancestor = ROOT / "projects/pour_policy_training_v3/fast_path_smooth_cache.npz"
    inputs[ancestor.relative_to(ROOT).as_posix()] = sha(ancestor)
    inputs[profile_file.relative_to(ROOT).as_posix()] = PROFILE_SHA256
    inputs[protected_file.relative_to(ROOT).as_posix()] = sha(protected_file)
    guis = {name: sha(ROOT / name) for name in (
        "assets/robot_testtube_pour/policy/precision_v2.json",
        "assets/robot_testtube_pour/policy/target_volume_policy.json",
        "assets/robot_testtube_pour/policy/target_volume_candidate.json")}
    points, labels = geometry.initial_water_points(SPACING)
    if len(points) != 11625 or int(sum(labels == 1)) != 2880:
        raise RuntimeError("Initial particle identities changed")
    plan = {"kind": "coarse40_upstream_speed_calibration_v1",
        "controller_source": "projects/pour_policy_coarse40/coarse_control.py", "collector_source": "projects/pour_policy_coarse40/run_coarse_test.py", "created_at_utc": now(), "matrix": matrix, "matrix_raw_sha256": sha(args.matrix),
        "deadline_s": 24., "relative_tolerance": .001,
        "refine_task_kind": task_score.TASK_KIND, "refine_absolute_tolerance_ml": .5,
        "easy_absolute_tolerance_ml": 1., "v2_feedback_principle_not_original_controller": True,
        "profiles": matrix["profiles"], "profile_raw_sha256": PROFILE_SHA256,
        "experiment_plan_sha256": sha(experiment_plan),
        "protected_file_sha256": protected, "protected_manifest_raw_sha256": sha(protected_file),
        "retry_only_after_actual_settled_deficiency": True, "collector_fits_or_selects_policy": False,
        "only_coarse_dose_speed_varies": True, "fixed_first_approach_tilt_deg": 87., "fixed_retry_approach_tilt_deg": 87.2,
        "original_strict_standard_unchanged": True, "peak_all_label_spill_limit": 1,
        "spacing_m": SPACING, "particle_ml": SPACING**3 * 1e6,
        "total_particles": 11625, "source_particles": 2880,
        "initial_source_ml": 56.68704, "post_seconds": 2.,
        "solver": "unchanged_live_PBF_60Hz_pressure6_min4_graph_off",
        "measurement": "actual_full_R_particle_mass_not_scale_or_torque",
        "source_sha256": sources, "input_sha256": inputs, "gui_sha256": guis,
        "collection_task_success_not_required": True,
        "phase_training_time_not_hidden": True, "no_saved_liquid_replay": True}
    if selection is not None:
        for key, value in (("source_sha256", sources), ("input_sha256", inputs), ("gui_sha256", guis), ("protected_file_sha256", protected)):
            if selection.get(key) != value:
                raise ValueError("Runtime/old versions changed after calibration selection: " + key)
    if strategy is not None:
        for key, value in (("source_sha256", sources), ("input_sha256", inputs), ("gui_sha256", guis), ("protected_file_sha256", protected)):
            if strategy.get(key) != value:
                raise ValueError("Runtime or inputs changed after strategy freeze: " + key)
    plan["protocol_id"] = hashlib.sha256(json.dumps(plan, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    if args.dry_run:
        print(json.dumps(plan, indent=2)); return
    for name, digest in {**sources, **inputs, **guis, **protected}.items():
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
        for name, digest in protected.items():
            if name in sources or name in inputs or name in guis:
                continue
            destination = output / "protected_snapshot" / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, destination)
            if sha(destination) != digest:
                raise RuntimeError("Protected version snapshot changed: " + name)
        write(output / "matrix.json", matrix)
        write(output / "job_state.json", {"state": "starting", "pid": os.getpid(), "started_at_utc": now()})
        sdk = load("response_sdk", ROOT / "train_target_volume_policy.py")
        write(output / "versions.json", sdk.configure_sdk())
        environment = load("response_environment", ADAPTIVE / "cycle_env.py")
        env = environment.CycleEnv(geometry=geometry, control=sys.modules[__name__], spacing=SPACING)
        for case in cases:
            for name, digest in {**sources, **inputs, **guis, **protected}.items():
                if sha(ROOT / name) != digest:
                    raise RuntimeError("Frozen runtime changed: " + name)
            CURRENT_SEED = case["seed"]
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
            row = {"episode_id": identifier, "stage": matrix["stage"], "role": case["role"],
                "candidate_id": case["candidate_id"], "profile_key": case["profile_key"],
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
            row["refine_assessment"] = task_score.assess(result, row["strict_assessment"])
            row["easy_assessment"] = task_score.easy_assess(result, row["strict_assessment"])
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
        for name, digest in {**sources, **inputs, **guis, **protected}.items():
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
            "easy_task_passes": sum(row["easy_assessment"]["feasible"] for row in rows),
            "refine_task_passes": sum(row["refine_assessment"]["feasible"] for row in rows),
            "stage": matrix["stage"],
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
