"""Evaluate one frozen whole-cycle policy on explicitly new target conditions.

Use SDK Python. Each matrix row is a real episode, including lift, coarse
prefix, approach, micro action, cutoff, settled return, and2s post observation.
The matrix and checkpoint are frozen before CUDA initialization.  Held-out
tests require ten independently audited passing nominal development cases.
No fitting, source rebinding, or completed-case re-pouring occurs here.
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


def require(condition, message):
    if not condition:
        raise ValueError(message)


def bounded_path(value, root=ROOT):
    """Resolve and bound a path before any content read, including symlinks."""
    root = Path(root).resolve()
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = root / path
    path = path.resolve()
    require(path.is_relative_to(root), "Evaluation inputs must stay inside the explicit workspace")
    return path


def read_json(path):
    def reject(value):
        raise ValueError("Nonfinite JSON constant: " + value)
    return json.loads(Path(path).read_text(encoding="utf8"), parse_constant=reject)


def read_jsonl(path):
    lines = Path(path).read_text(encoding="utf8").splitlines()
    require(lines and all(line.strip() for line in lines), "Missing or blank journal records")
    rows = [json.loads(line) for line in lines]
    require(all(isinstance(row, dict) for row in rows), "Journal records must be objects")
    json.dumps(rows, allow_nan=False)
    return rows


def utc_timestamp(value):
    require(isinstance(value, str), "Frozen model requires an explicit UTC creation time")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    require(parsed.tzinfo is not None and parsed.utcoffset() is not None,
            "Model/reset timestamps require a time zone")
    return parsed.astimezone(timezone.utc)


def valid_sha(value):
    return (isinstance(value, str) and len(value) == 64
            and all(char in "0123456789abcdef" for char in value))


def normalized_declared_policy(policy, root):
    """Normalize only the checkpoint path; keep all declared control choices."""
    require(isinstance(policy, dict), "A declared evaluation policy is required")
    normalized = json.loads(json.dumps(policy, sort_keys=True, allow_nan=False))
    require(isinstance(normalized.get("model_checkpoint"), str),
            "Declared evaluation policy lacks its checkpoint path")
    normalized["model_checkpoint"] = bounded_path(normalized["model_checkpoint"], root).relative_to(root).as_posix()
    return normalized


def declared_policy_sha256(policy, root):
    value = normalized_declared_policy(policy, root)
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def verified_artifacts(run, bindings, required):
    require(isinstance(bindings, dict) and set(required) <= set(bindings),
            "Model/audit lacks complete artifact SHA bindings")
    for name in required:
        path = bounded_path(run / name, run)
        require(valid_sha(bindings[name]) and path.is_file() and sha(path) == bindings[name],
                "Frozen collection/evaluation artifact changed: " + name)


COLLECTION_ARTIFACTS = ("protocol.json", "summary.json", "job_state.json", "episodes.jsonl",
                        "responses.jsonl", "matrix.json", "versions.json")


def validate_engineering_gate(report_path, expected_sha, model_sha, root, collection_seeds,
                              expected_policy):
    """Require five fit-context targets at two seeds and an independent audit."""
    require(report_path is not None and valid_sha(expected_sha),
            "Held-out evaluation requires --engineering-report and --engineering-sha")
    report_path = bounded_path(report_path, root)
    require(report_path.is_file() and sha(report_path) == expected_sha,
            "Nominal engineering audit raw SHA mismatch")
    report, run = read_json(report_path), report_path.parent
    require(report.get("complete") is True and report.get("pass") is True
            and report.get("scope") == "record_integrity_and_independent_physical_motion_mass_goal_audit_not_task_success",
            "Nominal report must be a complete independent physical/evidence audit")
    verified_artifacts(run, report.get("input_artifact_sha256"), COLLECTION_ARTIFACTS)
    protocol, summary, job = (read_json(run / name) for name in
                              ("protocol.json", "summary.json", "job_state.json"))
    require(protocol.get("kind") == "adaptive_frozen_response_goal_evaluation_v12"
            and summary.get("complete") is True and job.get("state") == "complete"
            and job.get("pid") is None and not (run / "worker.lock").exists(),
            "Nominal development evaluation is not finalized")
    require(protocol.get("protocol_id") == report.get("protocol_id") == summary.get("protocol_id"),
            "Nominal report protocol binding differs")
    require(protocol.get("evaluation_model_binding", {}).get("model_raw_sha256") == model_sha,
            "Nominal report used a different frozen response checkpoint")
    expected_policy_sha = declared_policy_sha256(expected_policy, root)
    require(protocol.get("evaluation_model_binding", {}).get("declared_policy_sha256") == expected_policy_sha,
            "Nominal report used a different declared control profile")
    rows = read_jsonl(run / "episodes.jsonl")
    require(len(rows) == summary.get("episodes") == report.get("episodes_audited") == 10,
            "Nominal gate requires exactly ten completed cases")
    require(report.get("strict_task_passes") == 10 and report.get("all_strict_task_cases_passed") is True,
            "Nominal cases did not all pass strict task acceptance")
    seeds = {row.get("seed") for row in rows}
    require(len(seeds) == 2 and all(type(seed) is int for seed in seeds)
            and not seeds & collection_seeds, "Nominal gate requires two new independent seeds")
    require({(row.get("target_ml"), row.get("seed")) for row in rows}
            == {(target, seed) for target in (20., 25., 30., 35., 40.) for seed in seeds},
            "Nominal gate must cover20/25/30/35/40mL at both development seeds")
    require(len({row.get("episode_id") for row in rows}) == 10, "Duplicate nominal episode identity")
    audits = {row.get("episode_id"): row for row in report.get("episode_reports", [])}
    require(set(audits) == {row["episode_id"] for row in rows},
            "Nominal independent audit does not cover all ten identities")
    for row in rows:
        require(row.get("stage") == "goal_evaluation" and row.get("split") == "development"
                and row.get("role") == "candidate"
                and row.get("declared_policy", {}).get("model_sha256") == model_sha
                and row.get("strict_assessment", {}).get("feasible") is True
                and audits[row["episode_id"]].get("strict_assessment", {}).get("feasible") is True,
                "Nominal gate has an unpassed or different-policy case")
        require(declared_policy_sha256(row["declared_policy"], root) == expected_policy_sha,
                "Nominal case control profile differs from requested held-out policy")
        require(isinstance(row.get("artifacts"), dict) and len(row["artifacts"]) >= 2,
                "Nominal gate lacks original action/post trace bindings")
        require(any(name.endswith(".steps.jsonl.gz") for name in row["artifacts"])
                and any(name.endswith(".post.jsonl.gz") for name in row["artifacts"]),
                "Nominal action or post-observation trace is absent")
        verified_artifacts(run, row["artifacts"], tuple(row["artifacts"]))
    require(report.get("audit_tool_sha256") == sha(bounded_path(
        root / "projects/pour_policy_adaptive_learning/audit_response_collection.py", root)),
        "Nominal audit implementation differs from its declared source")
    return {"audit_raw_sha256": expected_sha, "protocol_id": protocol["protocol_id"],
            "development_seeds": sorted(seeds), "all_ten_strict_task_cases_passed": True,
            "declared_policy_sha256": expected_policy_sha,
            "artifact_sha256": report["input_artifact_sha256"]}


def validate_evaluation_models(cases, *, collection_run, workspace=ROOT,
                               engineering_report=None, engineering_sha256=None,
                               scan_history=True):
    """Pure CPU, pre-GPU checkpoint/provenance/split/freshness contract."""
    root = Path(workspace).resolve()
    run = bounded_path(collection_run, root)
    require(run.is_dir() and not (run / "worker.lock").exists(),
            "Fitting collection is missing or still active")
    split = {case["split"] for case in cases}
    require(len(split) == 1 and split <= {"development", "test"},
            "Development and held-out tests need separate evaluation protocols")
    model_paths, model_shas = set(), set()
    for case in cases:
        policy = case["policy"]
        require(isinstance(policy.get("model_checkpoint"), str) and valid_sha(policy.get("model_sha256")),
                "Every evaluated goal requires one explicit checkpoint and lowercase raw SHA256")
        model_paths.add(bounded_path(policy["model_checkpoint"], root))
        model_shas.add(policy["model_sha256"])
    require(len(model_paths) == len(model_shas) == 1,
            "One evaluation protocol must use the same frozen model for every goal")
    policy_shas = {declared_policy_sha256(case["policy"], root) for case in cases}
    require(len(policy_shas) == 1,
            "One evaluation protocol must use one identical declared control profile for every goal")
    policy_sha = next(iter(policy_shas))
    checkpoint, model_sha = next(iter(model_paths)), next(iter(model_shas))
    require(checkpoint.is_file() and sha(checkpoint) == model_sha,
            "Evaluation checkpoint differs from the requested frozen bytes")
    # Verify all source/input paths before importing even the pure model helper.
    protocol = read_json(bounded_path(run / "protocol.json", run))
    require(protocol.get("kind") == "adaptive_real_cycle_response_collection_v12",
            "Response checkpoint must come from the declared whole-cycle collection")
    maps = {name: protocol.get(name) for name in ("source_sha256", "input_sha256", "gui_sha256")}
    require(all(isinstance(value, dict) for value in maps.values())
            and len(maps["source_sha256"]) == 26 and len(maps["input_sha256"]) == 6
            and len(maps["gui_sha256"]) == 3, "Expected original26sources/6inputs/3GUI bindings")
    for category, values in maps.items():
        for name, digest in values.items():
            require(isinstance(name, str) and not Path(name).is_absolute()
                    and valid_sha(digest), "Malformed original source/input binding")
            current = bounded_path(root / name, root)
            require(current.is_file() and sha(current) == digest,
                    "Original collection runtime/input/GUI changed: " + name)
            if category != "gui_sha256":
                folder = "source_snapshot" if category == "source_sha256" else "input_snapshot"
                snapshot = bounded_path(run / folder / name, run)
                require(snapshot.is_file() and sha(snapshot) == digest,
                        "Original collection snapshot changed: " + name)
    module_file = bounded_path(root / "projects/pour_policy_adaptive_learning/cycle_response_model.py", root)
    # model=None collection never imported this post-hoc learner.  Do not
    # fabricate original runtime closure by adding it to the old26 sources.
    # The fitted checkpoint binds it here; the new evaluation then freezes
    # this actual model runtime alongside its separate goal harness.
    checkpoint_header = read_json(checkpoint)
    require(isinstance(checkpoint_header, dict) and module_file.is_file()
            and valid_sha(checkpoint_header.get("source_sha256"))
            and checkpoint_header["source_sha256"] == sha(module_file),
            "Fitted checkpoint model source differs from the current post-hoc implementation")
    spec = importlib.util.spec_from_file_location("goal_evaluation_verified_cycle_model", module_file)
    model_module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = model_module
    spec.loader.exec_module(model_module)
    model = model_module.load_response_model(checkpoint, expected_raw_sha256=model_sha)
    for case in cases:
        menu = case["policy"].get("model_action_menu")
        require(menu is None or json.dumps(menu, sort_keys=True, separators=(",", ":"), allow_nan=False)
                == json.dumps(model.get("action_menu"), sort_keys=True, separators=(",", ":"), allow_nan=False),
                "Custom action menu differs from the exact frozen training action menu")
    require(model.get("source_sha256") == sha(module_file),
            "Fitted checkpoint model source differs from the current frozen implementation")
    require(model.get("selection_kind") == "validation_only",
            "Goal evaluation requires a candidate chosen only by the response validation split")
    created = utc_timestamp(model.get("created_at_utc"))
    require(created <= datetime.now(timezone.utc), "Frozen checkpoint creation time is in the future")
    provenance = model.get("provenance", {})
    verified_artifacts(run, provenance.get("collection_artifact_sha256"), COLLECTION_ARTIFACTS)
    require(provenance.get("collection_artifact_sha256", {}).get("protocol.json") == sha(run / "protocol.json")
            and provenance.get("collection_protocol_id") == protocol.get("protocol_id"),
            "Checkpoint does not identify these exact original collection bytes")
    summary, job, matrix = (read_json(run / name) for name in
                             ("summary.json", "job_state.json", "matrix.json"))
    require(summary.get("complete") is True and job.get("state") == "complete"
            and job.get("pid") is None and summary.get("protocol_id") == protocol.get("protocol_id"),
            "Original response collection is not complete")
    require(protocol.get("matrix") == matrix and matrix.get("draft") is False,
            "Original collection matrix differs from frozen protocol")
    episodes = read_jsonl(run / "episodes.jsonl")
    require(len(episodes) == summary.get("episodes") == len(matrix.get("cases", [])),
            "Original collection completion count differs from its matrix/journal")
    used_seeds = set(model.get("training_seeds", []) + model.get("validation_seeds", []))
    collection_seeds = {case["seed"] for case in matrix["cases"]}
    require(used_seeds <= collection_seeds and all(type(seed) is int for seed in used_seeds),
            "Checkpoint fitting seeds differ from its declared collection")
    evaluation_seeds = {case["seed"] for case in cases}
    require(not evaluation_seeds & collection_seeds,
            "Evaluation initial-state seeds overlap response training/validation collection")
    source_fingerprint = hashlib.sha256(json.dumps({"sources": protocol["source_sha256"],
        "inputs": protocol["input_sha256"], "spacing": SPACING}, sort_keys=True).encode()).hexdigest()
    require(model.get("physics_fingerprint") == source_fingerprint,
            "Checkpoint physics/measurement fingerprint differs from original collection")
    historical = []
    if scan_history:
        for journal in sorted((root / "droplet_out").glob("pour_policy*/*/episodes.jsonl")):
            journal = bounded_path(journal, root)
            for row in read_jsonl(journal):
                seed = row.get("seed", row.get("conditions", {}).get("seed", row.get("result", {}).get("seed")))
                if seed in evaluation_seeds:
                    historical.append({"journal": journal.relative_to(root).as_posix(), "seed": seed})
        require(not historical, "Evaluation seeds already occur in previous physical journals")
    engineering = None
    if "test" in split:
        engineering = validate_engineering_gate(engineering_report, engineering_sha256,
                                                model_sha, root, collection_seeds, cases[0]["policy"])
    elif engineering_report is not None or engineering_sha256 is not None:
        require(False, "Development evaluation does not borrow a held-out engineering gate")
    return {"model_checkpoint": checkpoint.relative_to(root).as_posix(), "model_raw_sha256": model_sha,
            "declared_policy_sha256": policy_sha,
            "normalized_declared_policy": normalized_declared_policy(cases[0]["policy"], root),
            "model_content_sha256": model["content_sha256"], "model_created_at_utc": model["created_at_utc"],
            "model_source_sha256": model["source_sha256"], "selection_kind": model["selection_kind"],
            "collection_run": run.relative_to(root).as_posix(),
            "collection_protocol_id": protocol["protocol_id"],
            "collection_artifact_sha256": provenance["collection_artifact_sha256"],
            "collection_source_sha256": maps["source_sha256"],
            "collection_input_sha256": maps["input_sha256"], "collection_gui_sha256": maps["gui_sha256"],
            "fitted_collection_physics_fingerprint": source_fingerprint,
            "execution_harness_fingerprint_is_separately_recorded": True,
            "evaluation_split": next(iter(split)), "evaluation_seeds": sorted(evaluation_seeds),
            "historical_seed_matches": historical, "engineering_gate": engineering}


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
    resolved_policy = dict(policy)
    resolved_policy["model_checkpoint"] = str(bounded_path(policy["model_checkpoint"]))
    profile = CONTROL.CorrectionPolicy(dict(COARSE), deadline_s=24., **resolved_policy)
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
        require(isinstance(row, dict) and {"case_id", "target_ml", "seed"} <= set(row),
                "Matrix cases require identity, target, and seed fields")
        identifier, target, seed = row["case_id"], row["target_ml"], row["seed"]
        if (not isinstance(identifier, str) or not identifier or len(identifier) > 96
                or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in identifier)
                or identifier in ids):
            raise ValueError("Case identities must be unique safe strings")
        ids.add(identifier)
        if (isinstance(target, bool) or not isinstance(target, (int, float))
                or not math.isfinite(target) or not 20 <= target <= 40
                or type(seed) is not int or not 0 <= seed < 2**32
                or row.get("split") not in ("development", "test")):
            raise ValueError("Invalid collection target/seed/split")
        if seed in seed_splits and seed_splits[seed] != row["split"]:
            raise ValueError("One seed cannot occur in two collection splits")
        seed_splits[seed] = row["split"]
        if not isinstance(row.get("policy"), dict):
            raise ValueError("Every case declares its full action profile")
        json.dumps(row["policy"], allow_nan=False)
        if set(row["policy"]) & {"coarse", "deadline_s", "target_ml", "spacing"}:
            raise ValueError("Collection actions cannot alter the task or physics")
    require(len(set(seed_splits.values())) == 1,
            "Development and held-out cases require separate protocols")
    return cases


def main(argv=None):
    global CONTROL
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--matrix", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--collection-run", type=Path, required=True)
    ap.add_argument("--engineering-report", type=Path)
    ap.add_argument("--engineering-sha")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    args.matrix = bounded_path(args.matrix)
    args.collection_run = bounded_path(args.collection_run)
    if args.engineering_report is not None:
        args.engineering_report = bounded_path(args.engineering_report)
    matrix = read_json(args.matrix)
    cases = validate_matrix(matrix)
    output = args.output.resolve()
    permitted = (ROOT / "droplet_out/pour_policy_adaptive_response").resolve()
    if not output.is_relative_to(permitted) or output == permitted:
        raise ValueError("Use a named new adaptive_response output directory")
    if output.exists() and any(output.iterdir()):
        raise RuntimeError("Existing evidence is immutable; never restart by overwriting")
    binding = validate_evaluation_models(cases, collection_run=args.collection_run,
        engineering_report=args.engineering_report, engineering_sha256=args.engineering_sha)
    for path in list((ROOT / "extscache").glob("omni.kit.pip_archive-*/pip_prebundle")) + [
            ROOT / "exts/omni.pip.compute/pip_prebundle"]:
        if path.is_dir():
            sys.path.insert(0, str(path))
    evidence = load("response_v9_original_evidence", ROOT / "projects/pour_policy_training_v9/run_fine_search.py")
    evidence.OUT = output
    geometry_module = load("response_lift_geometry", ROOT / "assets/robot_testtube_pour/scripts/mouth_pivot_lift_geometry.py")
    geometry = geometry_module.load_geometry(geometry_module.DEFAULT_CACHE_FILE,
                                            geometry_module.DEFAULT_CACHE_SHA256)
    CONTROL = load("response_controller", HERE / "cycle_control.py")
    samples = load("response_sample_schema", HERE / "response_samples.py")
    for case in cases:
        TargetPourController(geometry, case["target_ml"], SPACING, case["policy"])
    source_names = tuple(dict.fromkeys((*evidence.SOURCE_NAMES,
        "projects/pour_policy_training_v10/range_control.py",
        *("projects/pour_policy_adaptive_learning/" + name for name in
          ("response_control.py", "cycle_control.py", "cycle_response_model.py", "response_samples.py", "response_env.py", "cycle_env.py", "evaluate_cycle_policy.py")))))
    sources = {**binding["collection_source_sha256"],
               **{name: sha(ROOT / name) for name in source_names}}
    inputs = dict(geometry.INPUT_SHA256)
    inputs[Path(geometry.CACHE_FILE).relative_to(ROOT).as_posix()] = geometry.CACHE_SHA256
    ancestor = ROOT / "projects/pour_policy_training_v3/fast_path_smooth_cache.npz"
    inputs[ancestor.relative_to(ROOT).as_posix()] = sha(ancestor)
    for case in cases:
        checkpoint = bounded_path(case["policy"]["model_checkpoint"])
        if sha(checkpoint) != case["policy"]["model_sha256"]:
            raise ValueError("Evaluation checkpoint differs from the frozen requested bytes")
        inputs[checkpoint.relative_to(ROOT).as_posix()] = sha(checkpoint)
    inputs[args.matrix.relative_to(ROOT).as_posix()] = sha(args.matrix)
    for name, digest in binding["collection_artifact_sha256"].items():
        inputs[(args.collection_run / name).relative_to(ROOT).as_posix()] = digest
    if args.engineering_report is not None:
        inputs[args.engineering_report.relative_to(ROOT).as_posix()] = args.engineering_sha
    guis = {name: sha(ROOT / name) for name in (
        "assets/robot_testtube_pour/policy/precision_v2.json",
        "assets/robot_testtube_pour/policy/target_volume_policy.json",
        "assets/robot_testtube_pour/policy/target_volume_candidate.json")}
    points, labels = geometry.initial_water_points(SPACING)
    if len(points) != 11625 or int(sum(labels == 1)) != 2880:
        raise RuntimeError("Initial particle identities changed")
    plan = {"kind": "adaptive_frozen_response_goal_evaluation_v12",
        "controller_source": "projects/pour_policy_adaptive_learning/cycle_control.py", "collector_source": "projects/pour_policy_adaptive_learning/evaluate_cycle_policy.py", "created_at_utc": now(), "matrix": matrix, "matrix_raw_sha256": sha(args.matrix),
        "deadline_s": 24., "relative_tolerance": .001, "peak_all_label_spill_limit": 1,
        "spacing_m": SPACING, "particle_ml": SPACING**3 * 1e6,
        "total_particles": 11625, "source_particles": 2880,
        "initial_source_ml": 56.68704, "post_seconds": 2.,
        "solver": "unchanged_live_PBF_60Hz_pressure6_min4_graph_off",
        "measurement": "actual_full_R_particle_mass_not_scale_or_torque",
        "source_sha256": sources, "input_sha256": inputs, "gui_sha256": guis,
        "collection_task_success_not_required": False, "evaluation_model_binding": binding,
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
            freeze_before_reset = (utc_timestamp(binding["model_created_at_utc"])
                                   < utc_timestamp(env.reset_started_utc))
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
            row = {"episode_id": identifier, "stage": "goal_evaluation", "role": "candidate",
                "split": case["split"], "group_id": "seed_" + str(case["seed"]),
                "target_ml": case["target_ml"], "seed": case["seed"],
                "declared_policy": case["policy"], "protocol_id": plan["protocol_id"],
                "started_at_utc": started, "completed_at_utc": now(), "result": result,
                "status": status, "assessment": evidence.assessment(result, status, peak, post),
                "strict_assessment": samples.strict_assessment(result, status, peak, post),
                "post_observation": post, "motion_metrics": dict(env.fast_metrics),
                "wall_seconds": time.perf_counter() - wall,
                "response_events": action_events,
                "frozen_model_raw_sha256": binding["model_raw_sha256"],
                "model_created_at_utc": binding["model_created_at_utc"],
                "actual_reset_started_at_utc": env.reset_started_utc,
                "model_frozen_before_actual_reset": freeze_before_reset,
                "fitted_collection_physics_fingerprint": binding["fitted_collection_physics_fingerprint"],
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
            if not freeze_before_reset:
                raise RuntimeError("Model freeze was not earlier than actual reset; preserve invalid attempt")
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
            "frozen_response_model_used": True, "new_goal_policy_trained": False,
            "evaluation_model_binding": binding, "gui_deployment": False})
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
