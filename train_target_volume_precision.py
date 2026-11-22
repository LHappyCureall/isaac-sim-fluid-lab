"""48 live episodes for predeclared local calibration of precision pouring.

This is a coarse-to-fine hand-designed feedback controller with finite local
parameter calibration. It is not PPO, neural-network training or an end-to-end
robot policy. Every candidate action is evaluated by live physical-time PBF.
No GUI model is deployed by this runner.
"""

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path

from lab_paths import PROJECT_ROOT
import platform
import re
import sys
import time


ROOT = PROJECT_ROOT
SCRIPTS = ROOT / "assets/robot_testtube_pour/scripts"
TARGETS = (20, 30, 40)
TRAIN_SEEDS = (41101, 41102)
VALIDATION_SEEDS = (42101, 42102)
TEST_SEEDS = (51101, 51102, 51103)
ALGORITHM = "hand_designed_coarse_to_fine_feedback_plus_predeclared_finite_local_calibration"
GUI_MODELS = tuple(ROOT / "assets/robot_testtube_pour/policy" / name for name in
                   ("target_volume_policy.json", "target_volume_candidate.json"))


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    exec(compile(Path(path).read_text(encoding="utf-8"), str(path), "exec"), module.__dict__)
    return module


shared = load_module("precision_calibration_shared", ROOT / "train_target_volume_policy.py")


def now():
    return datetime.now(timezone.utc).isoformat()


def normalize_json(value):
    return json.loads(json.dumps(value, allow_nan=False,
                     default=lambda item: item.item() if hasattr(item, "item") else item.tolist()))


def read_configs(path):
    raw = Path(path).read_bytes()
    configs = json.loads(raw.decode("utf-8-sig"))
    if not isinstance(configs, list) or len(configs) != 3:
        raise ValueError("--configs must contain exactly three {name, params} objects.")
    names = set()
    for config in configs:
        if not isinstance(config, dict) or set(config) != {"name", "params"}:
            raise ValueError("Each predeclared configuration must have exactly name and params.")
        name = config["name"]
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,39}", name) or name in names:
            raise ValueError("Configuration names must be unique short letters/digits/underscore/hyphen identifiers.")
        if not isinstance(config["params"], dict):
            raise ValueError("Configuration params must be a dictionary; the active PrecisionPolicy validates its keys.")
        json.dumps(config["params"], allow_nan=False)
        names.add(name)
    return configs, raw, hashlib.sha256(raw).hexdigest()


def sources():
    old_names = ("train_target_volume_policy.py", "train_target_volume_policy_full.py")
    paths = {name: ROOT / name for name in old_names}
    for name in ("geometry.py", "liquid.py", "target_control.py", "learned_policy.py", "pour_training_env.py", "pbf_graph.py"):
        paths["assets/robot_testtube_pour/scripts/" + name] = SCRIPTS / name
    old = {name: shared._sha(path) for name, path in paths.items()}
    paths[Path(__file__).name] = Path(__file__)
    for name in ("precision_control.py", "precision_training_env.py"):
        paths["assets/robot_testtube_pour/scripts/" + name] = SCRIPTS / name
    return {name: shared._sha(path) for name, path in paths.items()}, old


def gui_hashes():
    return {str(path.relative_to(ROOT)).replace("\\", "/"): shared._sha(path) if path.is_file() else None
            for path in GUI_MODELS}


def require_unchanged(expected, gui_expected):
    actual, _ = sources()
    if actual != expected:
        raise RuntimeError("Frozen precision-calibration source changed: " + ", ".join(
            name for name in set(actual) | set(expected) if actual.get(name) != expected.get(name)))
    if gui_hashes() != gui_expected:
        raise RuntimeError("A protected GUI checkpoint changed; this experiment never deploys models.")


def prepare_inputs(args):
    """Freeze exact input bytes and current source versions before GPU setup."""
    output = Path(args.output).resolve()
    if output == ROOT or output.is_relative_to(ROOT / "assets"):
        raise ValueError("Use a dedicated run directory outside GUI assets.")
    configs, raw, config_sha = read_configs(args.configs)
    if not math.isfinite(float(args.max_seconds)) or float(args.max_seconds) <= 0:
        raise ValueError("--max-seconds must be finite and positive.")
    all_sources, old_sources = sources()
    old_protocol_path = ROOT / "droplet_out/pour_policy_training/run_20261122_full/protocol.json"
    historical_manifest_verified = False
    historical_manifest_sha256 = None
    if old_protocol_path.is_file():
        protected = json.loads(old_protocol_path.read_text(encoding="utf-8"))["protocol"]["source_sha256"]
        if old_sources != protected:
            raise ValueError("The previous eight frozen sources no longer match their completed experiment.")
        historical_manifest_verified = True
        historical_manifest_sha256 = shared._sha(old_protocol_path)
    declaration = {"schema_version": 1, "algorithm": ALGORITHM, "neural_policy": False,
        "config_input_sha256": config_sha, "configurations_raw": configs,
        "budget": {"calibration": 18, "validation": 12, "test": 18, "total_live_episodes": 48},
        "train_initial_seeds": list(TRAIN_SEEDS), "validation_seeds": list(VALIDATION_SEEDS),
        "test_seeds": list(TEST_SEEDS), "targets_ml": list(TARGETS),
        "max_seconds": float(args.max_seconds), "tolerance_ml": .1, "cuda_graph": bool(args.graph),
        "source_sha256": all_sources, "protected_legacy_source_sha256": old_sources,
        "historical_manifest_verified": historical_manifest_verified,
        "historical_manifest_sha256": historical_manifest_sha256,
        "gui_checkpoint_sha256_before": gui_hashes(), "automatic_gui_deployment": False}
    identity = shared._signature(declaration)
    output.mkdir(parents=True, exist_ok=True)
    input_path = output / "input_freeze.json"
    if input_path.exists():
        if not args.resume:
            raise FileExistsError("Inputs are already frozen; use --resume or a new run directory.")
        previous = json.loads(input_path.read_text(encoding="utf-8"))
        if previous.get("input_id") != identity or previous.get("declaration") != declaration:
            raise ValueError("Resume rejected: raw configs, source, budget, physics options or GUI reference changed.")
        if shared._sha(output / "configs_frozen.json") != config_sha:
            raise ValueError("The frozen configuration input changed.")
    else:
        if any((output / name).exists() for name in ("protocol.json", "episodes.jsonl", "summary.json", "freeze.json")):
            raise FileExistsError("An unrelated/partial experiment exists without its input freeze.")
        frozen_config = output / "configs_frozen.json"
        if frozen_config.exists() and frozen_config.read_bytes() != raw:
            raise ValueError("Existing frozen config bytes differ.")
        frozen_config.write_bytes(raw)
        for name, digest in all_sources.items():
            destination = output / "source_snapshot" / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists() and shared._sha(destination) != digest:
                raise ValueError("Existing source snapshot differs: " + name)
            destination.write_bytes((ROOT / name).read_bytes())
            if shared._sha(destination) != digest:
                raise RuntimeError("Source changed while snapshotting: " + name)
        shared.write_json(input_path, {"frozen_at_utc": now(), "input_id": identity,
                                     "config_input_original_path": str(Path(args.configs).resolve()),
                                     "declaration": declaration})
    for name, digest in all_sources.items():
        if shared._sha(output / "source_snapshot" / name) != digest:
            raise ValueError("Frozen source snapshot changed: " + name)
    require_unchanged(all_sources, declaration["gui_checkpoint_sha256_before"])
    return output, configs, declaration, identity


def safe(result):
    return bool(result["valid_physics"] and result["mass_conserved"] and result["complete"]
                and result["upright"] and result["max_spilled_ml"] == 0
                and result["max_below_table_particles"] == 0 and result["inflight_ml"] <= .1)


def summarize(results):
    errors = [abs(float(result["error_ml"])) for result in results]
    n = len(results)
    return {"episodes": n, "mae_ml": sum(errors) / n, "worst_absolute_error_ml": max(errors),
        "safe_completion_rate": sum(safe(result) for result in results) / n,
        "within_0_1_ml_rate": sum(safe(result) and error <= .1 for result, error in zip(results, errors)) / n,
        "within_0_2_ml_rate": sum(safe(result) and error <= .2 for result, error in zip(results, errors)) / n,
        "within_0_5_ml_rate": sum(safe(result) and error <= .5 for result, error in zip(results, errors)) / n,
        "all_mass_conserved": all(result["mass_conserved"] for result in results),
        "physical_steps": sum(result["physical_steps"] for result in results),
        "physical_seconds": sum(result["physical_time_s"] for result in results),
        "total_spilled_ml": sum(result["max_spilled_ml"] for result in results),
        "invalid_physics_episodes": sum(not result["valid_physics"] for result in results),
        "unsafe_or_incomplete_episodes": sum(not safe(result) for result in results)}


def group_score(results):
    m = summarize(results)
    return (m["mae_ml"] + .5 * m["worst_absolute_error_ml"]
            + 10 * m["total_spilled_ml"] / m["episodes"]
            + 1000 * m["invalid_physics_episodes"] / m["episodes"]
            + 100 * m["unsafe_or_incomplete_episodes"] / m["episodes"]
            + 1e-5 * m["physical_seconds"] / m["episodes"])


def release_metrics(metrics):
    return bool(metrics["mae_ml"] <= .1 and metrics["worst_absolute_error_ml"] <= .2
                and metrics["safe_completion_rate"] == 1 and metrics["all_mass_conserved"])


def default_runtime(args):
    versions = shared.configure_sdk()
    baseline = load_module("precision_run_baseline", SCRIPTS / "target_control.py")
    precision = load_module("precision_run_controller", SCRIPTS / "precision_control.py")
    from precision_training_env import PrecisionTrainingEnv
    from learned_policy import LearnedPourPolicy, DEFAULT_PARAMS
    return {"versions": versions, "baseline_control": baseline, "precision_control": precision,
        "policy_factory": precision.PrecisionPolicy,
        "precision_default_params": dict(precision.DEFAULT_PARAMS),
        "baseline_policy_factory": LearnedPourPolicy, "baseline_params": dict(DEFAULT_PARAMS),
        "environment_factory": lambda: PrecisionTrainingEnv(control=precision,
            max_seconds=float(args.max_seconds), tolerance_ml=.1)}


def verify_trace(path, result, expected_controller_sha):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        header = json.loads(next(stream))
        implementation = json.loads(next(stream))
        final = None
        for line in stream:
            final = line
    if header.get("type") != "episode" or implementation.get("type") != "controller_implementation":
        raise ValueError("Precision trace lacks its actual controller/UTC implementation record.")
    if implementation.get("controller_file_sha256") != expected_controller_sha:
        raise ValueError("Recorded active controller differs from the frozen controller source.")
    if not implementation.get("wall_clock_start_utc"):
        raise ValueError("The actual pre-physics UTC start event was not recorded.")
    footer = json.loads(final) if final is not None else {}
    if footer.get("type") != "result" or {key: value for key, value in footer.items() if key != "type"} != result:
        raise ValueError("Base episode footer and returned result differ; do not post-edit physical results.")
    return implementation


def run_precision(args, *, runtime=None):
    output, raw_configs, declaration, input_id = prepare_inputs(args)
    runtime = default_runtime(args) if runtime is None else runtime
    require_unchanged(declaration["source_sha256"], declaration["gui_checkpoint_sha256_before"])
    configs = [{"name": config["name"], "params": normalize_json(runtime["policy_factory"](config["params"]).params)}
               for config in raw_configs]
    software = {**runtime["versions"], "python": sys.version, "platform": platform.platform()}
    protocol = {"schema_version": 1, "kind": "precision_local_calibration_48_live_episodes",
        "algorithm": ALGORITHM, "neural_policy": False, "ppo": False, "input_freeze_id": input_id,
        "config_input_sha256": declaration["config_input_sha256"], "configs": configs,
        "targets_ml": list(TARGETS), "train_seeds": list(TRAIN_SEEDS),
        "validation_seeds": list(VALIDATION_SEEDS), "test_seeds": list(TEST_SEEDS),
        "calibration_episodes": 18, "validation_episodes": 12, "test_episodes": 18, "total_live_episodes": 48,
        "distinct_calibration_target_seed_conditions": 6, "distinct_validation_target_seed_conditions": 6,
        "distinct_test_target_seed_conditions": 9,
        "baseline_params": runtime["baseline_params"], "precision_defaults_at_freeze": runtime["precision_default_params"],
        "physics": {"mode": "live_PBF_action_conditioned", "dt_s": 1/60, "spacing_m": .0035,
                    "minimum_substeps": 4, "pressure_iterations": 6, "motion_time_scale": 2,
                    "max_seconds": float(args.max_seconds), "tolerance_ml": .1, "cuda_graph": bool(args.graph)},
        "randomization": {"initial_jitter_max_spacing": .03, "viscosity_factor": [.9, 1.1]},
        "group_score": "MAE + 0.5*worst_abs_error + 10*mean_spilled_ml + 1000*invalid_physics_rate + 100*unsafe_or_incomplete_rate + 0.00001*mean_physical_seconds",
        "calibration_selection": "lowest group score over six calibration conditions; ties choose first predeclared config",
        "validation_selection": "candidate only if lower validation group score; ties recommend baseline",
        "test_use": "release acceptance/descriptive paired comparison only; never refit or change validation selection",
        "release_criteria": {"mae_ml_at_most": .1, "worst_absolute_error_ml_at_most": .2, "all_safe_mass_conserved": True},
        "source_sha256": declaration["source_sha256"],
        "protected_legacy_source_sha256": declaration["protected_legacy_source_sha256"],
        "historical_manifest_verified": declaration["historical_manifest_verified"],
        "historical_manifest_sha256": declaration["historical_manifest_sha256"],
        "gui_checkpoint_sha256_before": declaration["gui_checkpoint_sha256_before"],
        "software": software, "automatic_gui_deployment": False}
    protocol_id = shared._signature(protocol)
    protocol_path = output / "protocol.json"
    if protocol_path.exists():
        previous = json.loads(protocol_path.read_text(encoding="utf-8"))
        if not args.resume or previous.get("protocol_id") != protocol_id or previous.get("protocol") != protocol:
            raise ValueError("Resume source/config/versions/protocol differ.")
    else:
        shared.write_json(protocol_path, {"created_at_utc": now(), "protocol_id": protocol_id, "protocol": protocol})
    records = shared._load_journal(output / "episodes.jsonl", protocol_id) if args.resume else {}
    expected_ids = {f"calibration_{config['name']}_t{target}_s{seed}" for config in configs for seed in TRAIN_SEEDS for target in TARGETS}
    expected_ids |= {f"{stage}_{role}_t{target}_s{seed}" for stage, seeds in
                    (("validation", VALIDATION_SEEDS), ("test", TEST_SEEDS))
                    for role in ("candidate", "baseline") for seed in seeds for target in TARGETS}
    if not set(records).issubset(expected_ids):
        raise ValueError("Journal contains an unexpected episode.")
    environment = None
    started, newly_run = time.perf_counter(), 0

    def persist_rewards():
        rows = [{"episode_id": record["episode_id"], **record["conditions"],
                 **{key: record["result"][key] for key in ("received_ml", "error_ml", "physical_steps", "physical_time_s", "wall_seconds")}}
                for record in records.values()]
        fields = ("episode_id", "stage", "policy_role", "config_name", "target_ml", "seed", "received_ml", "error_ml", "physical_steps", "physical_time_s", "wall_seconds")
        shared._write_csv(output / "episodes.csv", rows, fields)

    def episode(episode_id, stage, role, target, seed, config=None):
        nonlocal environment, newly_run
        precision_role = role != "baseline"
        relative = "assets/robot_testtube_pour/scripts/" + ("precision_control.py" if precision_role else "target_control.py")
        controller_sha = declaration["source_sha256"][relative]
        conditions = {"stage": stage, "policy_role": role, "config_name": None if config is None else config["name"],
            "target_ml": float(target), "seed": int(seed), "randomized": True,
            "policy_params": None if config is None else config["params"], "controller_source_sha256": controller_sha}
        require_unchanged(declaration["source_sha256"], declaration["gui_checkpoint_sha256_before"])
        if episode_id in records:
            record = records[episode_id]
            if record["conditions"] != conditions or record["conditions_sha256"] != shared._signature(conditions):
                raise ValueError("Resumed episode conditions differ: " + episode_id)
            return record["result"]
        if environment is None:
            environment = runtime["environment_factory"]()
            if args.graph:
                shared._enable_graph(environment)
        environment.control_module = runtime["precision_control"] if precision_role else runtime["baseline_control"]
        policy = runtime["policy_factory"](config["params"]) if config is not None else None
        raw_path = output / "traces" / (episode_id + ".steps.jsonl.gz")
        def progress(event):
            shared.write_json(output / "progress.json", {"state": "running", "protocol_id": protocol_id,
                "updated_at_utc": now(), "episode_id": episode_id, "stage": stage, "conditions": conditions,
                "completed_episodes": len(records), "total_planned_episodes": 48,
                "wall_seconds_this_invocation": time.perf_counter()-started, **normalize_json(event)})
        print(f"PRECISION EPISODE {len(records)+1}/48: {episode_id}", flush=True)
        result = normalize_json(environment.run_episode(target, seed, policy=policy, randomize=True,
                                                       record_path=raw_path, progress=progress))
        controller_summary = normalize_json(dict(environment.last_controller.status))
        implementation = verify_trace(raw_path, result, controller_sha)
        if result["target_ml"] != target or result["seed"] != seed:
            raise ValueError("Returned physical conditions differ.")
        record = {"episode_id": episode_id, "protocol_id": protocol_id,
            "conditions": conditions, "conditions_sha256": shared._signature(conditions),
            "record_path": raw_path.relative_to(output).as_posix(), "record_sha256": shared._sha(raw_path),
            "completed_at_utc": now(), "episode_wall_clock_start_utc": implementation["wall_clock_start_utc"],
            "controller_implementation": implementation, "controller_summary": controller_summary,
            "result": result}
        with (output / "episodes.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, allow_nan=False, separators=(",", ":")) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        records[episode_id] = record
        newly_run += 1
        persist_rewards()
        progress({"event": "episode_recorded", "received_ml": result["received_ml"], "error_ml": result["error_ml"]})
        print(f"PRECISION RESULT {episode_id}: {result['received_ml']:.6f} mL, error={result['error_ml']:+.6f}, safe={safe(result)}", flush=True)
        return result

    try:
        calibration = []
        for index, config in enumerate(configs):
            values = [episode(f"calibration_{config['name']}_t{target}_s{seed}", "calibration", "candidate", target, seed, config)
                      for seed in TRAIN_SEEDS for target in TARGETS]
            calibration.append({"index": index, "name": config["name"], "params": config["params"],
                                "group_score": group_score(values), "metrics": summarize(values), "results": values})
        best = min(calibration, key=lambda entry: (entry["group_score"], entry["index"]))
        best_config = {"name": best["name"], "params": best["params"]}
        shared.write_json(output / "calibration.json", {"protocol_id": protocol_id,
                          "method": ALGORITHM, "configs": calibration, "selected_config": best_config})
        shared._write_csv(output / "calibration_scores.csv", [{"configuration": entry["name"], "group_score": entry["group_score"], **entry["metrics"]} for entry in calibration],
                          ("configuration", "group_score", "episodes", "mae_ml", "worst_absolute_error_ml", "safe_completion_rate", "physical_seconds"))
        validation = {role: [] for role in ("candidate", "baseline")}
        for seed in VALIDATION_SEEDS:
            for target in TARGETS:
                for role in validation:
                    validation[role].append(episode(f"validation_{role}_t{target}_s{seed}", "validation", role,
                                                    target, seed, best_config if role == "candidate" else None))
        validation_scores = {role: group_score(values) for role, values in validation.items()}
        recommended_role = "candidate" if validation_scores["candidate"] < validation_scores["baseline"] else "baseline"
        validation_report = {"protocol_id": protocol_id, "group_score": validation_scores,
            "metrics": {role: summarize(values) for role, values in validation.items()},
            "results": validation, "recommended_role": recommended_role, "test_used_for_selection": False}
        shared.write_json(output / "validation.json", validation_report)
        candidate_path, recommended_path = output / "candidate_frozen.json", output / "recommended_frozen.json"
        freeze_path = output / "freeze.json"
        changed = best_config["params"] != normalize_json(runtime["precision_default_params"])
        recommended_params = best_config["params"] if recommended_role == "candidate" else runtime["baseline_params"]
        if freeze_path.exists():
            freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
            if (not args.resume or freeze["protocol_id"] != protocol_id or freeze["candidate_sha256"] != shared._sha(candidate_path)
                    or freeze["recommended_sha256"] != shared._sha(recommended_path)
                    or freeze["candidate_config"] != best_config or freeze["recommended_role"] != recommended_role):
                raise ValueError("Pre-test model freeze differs from calibration/validation.")
            if (json.loads(candidate_path.read_text())["parameters"] != best_config["params"]
                    or json.loads(recommended_path.read_text())["parameters"] != recommended_params):
                raise ValueError("Frozen model parameters differ.")
        else:
            if any(record["conditions"]["stage"] == "test" for record in records.values()):
                raise ValueError("Test records exist without their pre-test model freeze.")
            if candidate_path.exists() or recommended_path.exists():
                raise ValueError("Partial pre-test freeze exists; review rather than overwrite it.")
            frozen_at = now()
            metadata = {"frozen_at_utc": frozen_at, "protocol_id": protocol_id, "algorithm": ALGORITHM,
                "neural_policy": False, "ppo": False, "calibration_configurations": 3,
                "calibration_episodes": 18, "selected_config_name": best_config["name"],
                "validation_scores": validation_scores, "test_used_for_selection": False,
                "source_sha256": declaration["source_sha256"], "config_input_sha256": declaration["config_input_sha256"],
                "automatic_gui_deployment": False}
            runtime["policy_factory"](best_config["params"]).save(candidate_path, metadata={**metadata,
                "role": "candidate", "calibration": True, "parameter_changed_from_precision_default": changed,
                "controller_changed_from_legacy_baseline": True})
            factory = runtime["policy_factory"] if recommended_role == "candidate" else runtime["baseline_policy_factory"]
            factory(recommended_params).save(recommended_path, metadata={**metadata,
                "role": "recommended", "recommended_role": recommended_role,
                "calibration": recommended_role == "candidate", "parameter_changed_from_precision_default": changed if recommended_role == "candidate" else False,
                "controller_changed_from_legacy_baseline": recommended_role == "candidate"})
            freeze = {"protocol_id": protocol_id, "frozen_at_utc": frozen_at, "candidate_config": best_config,
                "candidate_sha256": shared._sha(candidate_path), "recommended_sha256": shared._sha(recommended_path),
                "recommended_role": recommended_role, "baseline_params": runtime["baseline_params"],
                "automatic_gui_deployment": False, "test_used_for_selection": False}
            shared.write_json(freeze_path, freeze)
            shared._atomic_text(output / "models_frozen.sha256", freeze["candidate_sha256"] + "  candidate_frozen.json\n"
                               + freeze["recommended_sha256"] + "  recommended_frozen.json\n")
        pairs = []
        for seed in TEST_SEEDS:
            for target in TARGETS:
                candidate = episode(f"test_candidate_t{target}_s{seed}", "test", "candidate", target, seed, best_config)
                baseline = episode(f"test_baseline_t{target}_s{seed}", "test", "baseline", target, seed)
                pairs.append({"target_ml": target, "seed": seed, "candidate": candidate, "baseline": baseline,
                              "recommended_result_role": recommended_role})
        require_unchanged(declaration["source_sha256"], declaration["gui_checkpoint_sha256_before"])
        if (set(records) != expected_ids or len(records) != 48 or shared._sha(candidate_path) != freeze["candidate_sha256"]
                or shared._sha(recommended_path) != freeze["recommended_sha256"]):
            raise RuntimeError("Final episode set/model hashes differ.")
        first_test_start = min(record["episode_wall_clock_start_utc"] for record in records.values() if record["conditions"]["stage"] == "test")
        if datetime.fromisoformat(first_test_start) <= datetime.fromisoformat(freeze["frozen_at_utc"]):
            raise RuntimeError("A test episode began before both models were frozen.")
        candidate_metrics = summarize([pair["candidate"] for pair in pairs])
        baseline_metrics = summarize([pair["baseline"] for pair in pairs])
        recommended_metrics = candidate_metrics if recommended_role == "candidate" else baseline_metrics
        summary = {"complete": True, "completed_at_utc": now(), "protocol_id": protocol_id, "protocol": protocol,
            "algorithm": ALGORITHM, "neural_policy": False, "calibration": {"episodes": 18,
                "configurations": 3, "selected_config": best_config, "configs": calibration},
            "validation": validation_report, "models": {"candidate": {"file": candidate_path.name,
                "sha256": freeze["candidate_sha256"], "parameters": best_config["params"]},
                "recommended": {"file": recommended_path.name, "sha256": freeze["recommended_sha256"], "parameters": recommended_params}},
            "recommendation": {"role": recommended_role, "selected_on_validation_only": True,
                "test_result_is_reference": True, "additional_test_rollouts": 0},
            "test": {"paired_results": pairs, "candidate_metrics": candidate_metrics, "baseline_metrics": baseline_metrics,
                "recommended_metrics": recommended_metrics, "distinct_target_seed_conditions": 9,
                "independent_initial_state_seeds": 3, "executed_live_episodes": 18,
                "candidate_mae_better_than_baseline": candidate_metrics["mae_ml"] < baseline_metrics["mae_ml"],
                "candidate_worst_error_not_worse": candidate_metrics["worst_absolute_error_ml"] <= baseline_metrics["worst_absolute_error_ml"]},
            "release_acceptance": {"criteria": protocol["release_criteria"],
                "candidate_passed": release_metrics(candidate_metrics), "baseline_passed": release_metrics(baseline_metrics),
                "recommended_passed": release_metrics(recommended_metrics), "does_not_reselect_model": True,
                "automatic_gui_deployment": False},
            "all_rollouts": summarize([record["result"] for record in records.values()]),
            "total_recorded_episodes": 48, "new_episodes_this_invocation": newly_run,
            "physical_log_result_footer_modified": False, "test_used_for_training_or_selection": False,
            "source_sha256_unchanged_after_run": True, "protected_legacy_eight_unchanged": True,
            "historical_manifest_verified": declaration["historical_manifest_verified"],
            "gui_checkpoint_sha256_before": declaration["gui_checkpoint_sha256_before"],
            "gui_checkpoint_sha256_after": gui_hashes(), "gui_checkpoint_unchanged": True,
            "wall_seconds_this_invocation": time.perf_counter()-started,
            "total_rollout_wall_seconds": sum(record["result"]["wall_seconds"] for record in records.values()),
            "limitations": ["Hand-designed feedback structure plus three predeclared local configurations, not neural/RL policy learning.",
                "Six calibration and six validation target/seed conditions each use two initial-state seeds; test uses nine conditions and three fresh seeds.",
                "Volumes are privileged particle-mass truth at 0.042875mL quantum, not vision or a calibrated hardware scale.",
                "Kinematic pre-grasp and the existing bounded path; no friction grasp or fluid torque feedback.",
                "All precision failures, timeouts and inaccurate doses remain in traces/journal; test cannot change parameters or recommendation."]}
        shared.write_json(output / "summary.json", summary)
        write_report(output / "REPORT.md", summary)
        shared.write_json(output / "progress.json", {"state": "complete", "updated_at_utc": now(),
            "protocol_id": protocol_id, "completed_episodes": 48, "recommended_role": recommended_role,
            "recommended_release_passed": release_metrics(recommended_metrics), "automatic_gui_deployment": False})
        return summary
    except BaseException as error:
        shared.write_json(output / "progress.json", {"state": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
            "updated_at_utc": now(), "protocol_id": protocol_id, "completed_episodes": len(records),
            "error_type": type(error).__name__, "error": str(error)})
        raise


def write_report(path, summary):
    candidate, baseline = summary["test"]["candidate_metrics"], summary["test"]["baseline_metrics"]
    lines = ["# 精量倒水局部标定记录", "", f"完成（UTC）：{summary['completed_at_utc']}", "",
        "算法：粗到细手工反馈控制器＋预声明的三组参数有限局部标定；不是 PPO 或神经网络训练。",
        "18 次标定、12 次验证、18 次冻结后测试，共 48 次实时 PBF；不使用视频回放或压缩物理时间。",
        f"标定选中：{summary['calibration']['selected_config']['name']}；验证推荐：{summary['recommendation']['role']}。测试不重新选模型，GUI 权重不自动更新。", "",
        "| 新测试（各9条件、3种子） | MAE mL | 最大绝对误差 mL | 安全且±0.1mL |",
        "| --- | ---: | ---: | ---: |"]
    for label, data in (("冻结精量候选", candidate), ("原反馈基准", baseline)):
        lines.append(f"| {label} | {data['mae_ml']:.6f} | {data['worst_absolute_error_ml']:.6f} | {data['within_0_1_ml_rate']:.1%} |")
    lines += ["", f"推荐策略测试验收（MAE≤0.1、最大≤0.2mL且全部安全）：{summary['release_acceptance']['recommended_passed']}。",
        "体积为粒子质量真值，量化 0.042875mL，不代表视觉估水或真机计量精度。",
        "原始失败记录、完整控制器来源/开始UTC、动作、物理时间及1198/5330分区均保留；候选是否更准确以新测试实测为准。", ""]
    shared._atomic_text(path, "\n".join(lines))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", type=Path, default=PROJECT_ROOT / "configs/precision_v2_calibration.json")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "droplet_out/pour_policy_precision_v2/run_20261122_calibrated")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--graph", action="store_true")
    parser.add_argument("--max-seconds", type=float, default=95.0)
    args = parser.parse_args(argv)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    lock = output / "precision_runner.lock"
    with lock.open("x", encoding="utf-8") as stream:
        json.dump({"pid": os.getpid(), "started_at_utc": now()}, stream)
    try:
        summary = run_precision(args)
        print("PRECISION COMPLETE " + json.dumps({"episodes": 48, "recommended_role": summary["recommendation"]["role"],
            "candidate_mae_ml": summary["test"]["candidate_metrics"]["mae_ml"],
            "baseline_mae_ml": summary["test"]["baseline_metrics"]["mae_ml"],
            "recommended_release_passed": summary["release_acceptance"]["recommended_passed"]}), flush=True)
        return 0
    finally:
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
