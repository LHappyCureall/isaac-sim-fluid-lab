"""Independent, CPU-only audit of a completed 48-rollout precision experiment.

This reads the declared experiment and raw gzip logs without importing any
simulator/controller module. It never selects parameters, deploys a checkpoint,
or changes the original result/physics flags. A partial run is refused before
its journal, models or traces are opened. The report is private experiment data.
"""

import argparse
import ast
from collections import Counter
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path
import struct
import sys


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN = ROOT / "droplet_out/pour_policy_precision_v2/run_20261122_calibrated"
SCRIPTS = "assets/robot_testtube_pour/scripts/"
ALGORITHM = "hand_designed_coarse_to_fine_feedback_plus_predeclared_finite_local_calibration"
DT = 1.0 / 60.0
QUANTUM_ML = .0035 ** 3 * 1e6
PARTICLES = 5330
SOURCE_PARTICLES = 1198
FLOAT32_MASS = struct.unpack("f", struct.pack("f", 1000 * .0035 ** 3))[0]
TOTAL_MASS = PARTICLES * FLOAT32_MASS
SOURCE_MASS = SOURCE_PARTICLES * FLOAT32_MASS


class AuditError(ValueError):
    """A required artifact/record does not support the declared experiment."""


def require(condition, message):
    if not condition:
        raise AuditError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode("utf-8")).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def finite(value, context):
    if isinstance(value, float):
        require(math.isfinite(value), context + ": nonfinite number")
    elif isinstance(value, dict):
        for key, child in value.items():
            finite(child, context + "." + str(key))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            finite(child, f"{context}[{index}]")


def close(actual, expected, context, tolerance=1e-10):
    require(isinstance(actual, (int, float)) and not isinstance(actual, bool)
            and math.isfinite(actual) and abs(actual - expected) <= tolerance,
            f"{context}: {actual!r} != {expected!r} (tolerance {tolerance:g})")


def equivalent(actual, expected, context):
    """Recomputed metrics tolerate only floating reduction roundoff."""
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and set(actual) == set(expected), context + ": keys differ")
        for key in expected:
            equivalent(actual[key], expected[key], context + "." + str(key))
    elif isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected), context + ": length differs")
        for index, (one, two) in enumerate(zip(actual, expected)):
            equivalent(one, two, f"{context}[{index}]")
    elif isinstance(expected, float):
        close(actual, expected, context, max(1e-10, abs(expected) * 2e-13))
    else:
        require(actual == expected, context + ": value differs")


def utc(value):
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    require(result.tzinfo is not None and result.utcoffset().total_seconds() == 0,
            "Timestamp is not explicit UTC: " + str(value))
    return result


def normalize_paths(mapping):
    result = {str(key).replace("\\", "/"): value for key, value in mapping.items()}
    require(len(result) == len(mapping), "Duplicate normalized manifest paths")
    return result


def contained(base, relative):
    candidate = (Path(base) / relative).resolve()
    require(candidate.is_relative_to(Path(base).resolve()), "Artifact escapes its root: " + str(relative))
    return candidate


def safe(result):
    return bool(result["valid_physics"] and result["mass_conserved"] and result["complete"]
                and result["upright"] and result["max_spilled_ml"] == 0
                and result["max_below_table_particles"] == 0 and result["inflight_ml"] <= .1)


def summarize(results):
    errors = [abs(float(result["error_ml"])) for result in results]
    n = len(results)
    require(n > 0, "Empty metric group")
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


def release(metrics):
    return bool(metrics["mae_ml"] <= .1 and metrics["worst_absolute_error_ml"] <= .2
                and metrics["safe_completion_rate"] == 1 and metrics["all_mass_conserved"])


def source_class_hash(path, name):
    source = Path(path).read_text(encoding="utf-8-sig").replace("\r\n", "\n").replace("\r", "\n")
    # The logged classes are top-level classes, so AST spans match inspect's
    # class source without importing the GPU-dependent implementation.
    node = next((node for node in ast.parse(source).body
                 if isinstance(node, ast.ClassDef) and node.name == name), None)
    require(node is not None, "Recorded class is missing from its frozen source: " + name)
    start = min([node.lineno] + [item.lineno for item in node.decorator_list]) - 1
    text = "".join(source.splitlines(keepends=True)[start:node.end_lineno])
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def controller_limits(snapshot):
    tree = ast.parse(Path(snapshot).read_text(encoding="utf-8-sig"))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "TargetPourController")
    values = {}
    for node in cls.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in ("MAX_PULSES", "RETURN_BY_SECONDS"):
                    values[target.id] = ast.literal_eval(node.value)
    require(set(values) == {"MAX_PULSES", "RETURN_BY_SECONDS"}, "Missing frozen precision limits")
    return values


def source_literal(path, name):
    for node in ast.parse(Path(path).read_text(encoding="utf-8-sig")).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise AuditError("Missing frozen source literal: " + name)


def initial_jitter_roundoff(snapshot):
    """Bound the final float32 cast in the frozen reset, in world meters."""
    tree = ast.parse(Path(snapshot).read_text(encoding="utf-8-sig"))
    assignments = {target.id: node.value for node in tree.body if isinstance(node, ast.Assign)
                   for target in node.targets if isinstance(target, ast.Name)}
    value = lambda name: ast.literal_eval(assignments[name])
    center = value("BEAKER_CENTER")
    mouth = ast.literal_eval(assignments["MOTION_KNOTS"].elts[0])[1]
    # A conservative bound suffices: both cavities at reset lie within these
    # extents. No simulated coordinates or outcome are used to set tolerance.
    extent = max(abs(center[0]) + value("BEAKER_RADIUS"), abs(center[1]) + value("BEAKER_RADIUS"),
                 abs(center[2]) + value("BEAKER_WATER_DEPTH"),
                 abs(mouth[0]) + value("TUBE_RADIUS"), abs(mouth[1]) + value("TUBE_RADIUS"),
                 abs(mouth[2]) + value("TUBE_LENGTH") + value("TUBE_FILL_HEIGHT")) + .03 * .0035
    return 2.0 ** (math.floor(math.log2(extent)) - 24)


def validate_manifests(run, root, summary):
    document = read_json(run / "protocol.json")
    p, pid = document["protocol"], document["protocol_id"]
    require(signature(p) == pid == summary["protocol_id"], "Protocol signature mismatch")
    require(summary["protocol"] == p, "Summary changes the original protocol")
    finite(p, "protocol")
    require(p["algorithm"] == ALGORITHM and p["neural_policy"] is False and p["ppo"] is False,
            "Algorithm labeling differs from the declared local calibration")
    require([p[key] for key in ("calibration_episodes", "validation_episodes", "test_episodes", "total_live_episodes")]
            == [18, 12, 18, 48], "Unexpected 48-episode budget")
    require([p[key] for key in ("distinct_calibration_target_seed_conditions", "distinct_validation_target_seed_conditions",
                              "distinct_test_target_seed_conditions")] == [6, 6, 9], "Unexpected condition budget")
    require(p["targets_ml"] == [20, 30, 40] and p["train_seeds"] == [41101, 41102]
            and p["validation_seeds"] == [42101, 42102] and p["test_seeds"] == [51101, 51102, 51103],
            "Targets/seeds differ from the preregistered fresh splits")
    require(not (set(p["train_seeds"]) & set(p["validation_seeds"]))
            and not (set(p["train_seeds"]) & set(p["test_seeds"]))
            and not (set(p["validation_seeds"]) & set(p["test_seeds"])), "Seeds leak across splits")
    require(len(p["configs"]) == 3 and len({c["name"] for c in p["configs"]}) == 3, "Config count/name mismatch")
    physics = p["physics"]
    for key, expected in {"dt_s": DT, "spacing_m": .0035, "minimum_substeps": 4,
                          "pressure_iterations": 6, "motion_time_scale": 2, "tolerance_ml": .1}.items():
        close(physics[key], expected, "protocol.physics." + key)
    require(physics["mode"] == "live_PBF_action_conditioned", "Protocol is not live liquid physics")
    require(p["randomization"] == {"initial_jitter_max_spacing": .03, "viscosity_factor": [.9, 1.1]},
            "Randomization bounds changed")
    require(p["release_criteria"] == {"mae_ml_at_most": .1, "worst_absolute_error_ml_at_most": .2,
                                      "all_safe_mass_conserved": True}, "Release gate changed")
    require(p["automatic_gui_deployment"] is False, "Unexpected automatic deployment")
    freeze = read_json(run / "input_freeze.json")
    d = freeze["declaration"]
    require(signature(d) == freeze["input_id"] == p["input_freeze_id"], "Input declaration signature mismatch")
    require(d["algorithm"] == p["algorithm"] and d["neural_policy"] is False
            and d["automatic_gui_deployment"] is False, "Input algorithm/deployment declaration mismatch")
    require(p["group_score"] == "MAE + 0.5*worst_abs_error + 10*mean_spilled_ml + 1000*invalid_physics_rate + 100*unsafe_or_incomplete_rate + 0.00001*mean_physical_seconds",
            "Predeclared group score formula changed")
    require(p["calibration_selection"] == "lowest group score over six calibration conditions; ties choose first predeclared config"
            and p["validation_selection"] == "candidate only if lower validation group score; ties recommend baseline"
            and p["test_use"] == "release acceptance/descriptive paired comparison only; never refit or change validation selection",
            "Predeclared selection/test boundary changed")
    require(d["budget"] == {"calibration": 18, "validation": 12, "test": 18, "total_live_episodes": 48},
            "Input budget mismatch")
    require(d["train_initial_seeds"] == p["train_seeds"] and d["validation_seeds"] == p["validation_seeds"]
            and d["test_seeds"] == p["test_seeds"] and d["targets_ml"] == p["targets_ml"], "Input split mismatch")
    close(d["max_seconds"], physics["max_seconds"], "input max time")
    close(d["tolerance_ml"], .1, "input tolerance")
    require(d["cuda_graph"] == physics["cuda_graph"], "Graph declaration mismatch")
    raw_path = run / "configs_frozen.json"
    require(digest(raw_path) == d["config_input_sha256"] == p["config_input_sha256"], "Raw config byte hash mismatch")
    raw = read_json(raw_path)
    require(raw == d["configurations_raw"], "Frozen raw JSON differs from the input declaration")
    require(len(raw) == 3, "Raw configuration count differs")
    for original, effective in zip(raw, p["configs"]):
        require(original["name"] == effective["name"], "Effective configuration order/name changed")
        expected_params = {**p["precision_defaults_at_freeze"], **original["params"]}
        require(expected_params == effective["params"], "Effective params differ from raw params plus declared defaults")
    sources = normalize_paths(p["source_sha256"])
    old = normalize_paths(p["protected_legacy_source_sha256"])
    require(len(sources) == 11 and len(old) == 8 and all(sources.get(k) == v for k, v in old.items()),
            "Expected eleven sources including the eight protected legacy files")
    require(sources == normalize_paths(d["source_sha256"]) and old == normalize_paths(d["protected_legacy_source_sha256"]),
            "Input/source manifests disagree")
    for name, expected in sources.items():
        require(digest(contained(root, name)) == expected, "Current frozen source changed: " + name)
        require(digest(contained(run / "source_snapshot", name)) == expected, "Frozen source snapshot changed: " + name)
    historical = root / "droplet_out/pour_policy_training/run_20261122_full/protocol.json"
    verified = p["historical_manifest_verified"]
    require(verified == d["historical_manifest_verified"] == summary["historical_manifest_verified"],
            "Historical verification label mismatch")
    require(d["historical_manifest_sha256"] == p["historical_manifest_sha256"], "Historical manifest digest mismatch")
    if verified:
        require(historical.is_file() and digest(historical) == p["historical_manifest_sha256"],
                "Verified historical protocol missing/changed")
        previous = read_json(historical)["protocol"]["source_sha256"]
        require(normalize_paths(previous) == old, "Protected old eight do not match their historical manifest")
    else:
        require(p["historical_manifest_sha256"] is None, "Unverified history incorrectly claims a digest")
    gui = normalize_paths(p["gui_checkpoint_sha256_before"])
    require(len(gui) == 2 and gui == normalize_paths(d["gui_checkpoint_sha256_before"])
            == normalize_paths(summary["gui_checkpoint_sha256_before"])
            == normalize_paths(summary["gui_checkpoint_sha256_after"]), "Protected GUI model manifests differ")
    for name, expected in gui.items():
        path = contained(root, name)
        require((digest(path) if path.is_file() else None) == expected, "Protected GUI checkpoint changed: " + name)
    job_path = run / "job_state.json"
    job_diagnostics = {"present": job_path.is_file()}
    if job_path.is_file():
        job = read_json(job_path)
        require(normalize_paths(job["source_sha256"]) == sources and job["config_sha256"] == p["config_input_sha256"],
                "Supervisor input hashes disagree")
        require(normalize_paths(job["gui_models_before_sha256"]) == gui, "Supervisor GUI before hashes disagree")
        if "gui_models_after_sha256" in job:
            require(normalize_paths(job["gui_models_after_sha256"]) == gui, "Supervisor GUI after hashes disagree")
        # summary may be committed just before supervisor commits its final
        # state; report that narrow race without inventing a completed job.
        job_diagnostics.update(state=job["state"], exit_code=job.get("exit_code"),
                               sources_unchanged=job.get("sources_unchanged"))
    require(utc(freeze["frozen_at_utc"]) <= utc(document["created_at_utc"]), "Input was frozen after protocol creation")
    return p, pid, sources, gui, job_diagnostics


def expected_conditions(p, selected, stage, role, target, seed, config=None):
    if role == "candidate" and config is None:
        config = selected
    return {"stage": stage, "policy_role": role, "config_name": config["name"] if config else None,
            "target_ml": float(target), "seed": seed, "randomized": True,
            "policy_params": config["params"] if config else None,
            "controller_source_sha256": p["source_sha256"][SCRIPTS + ("precision_control.py" if role == "candidate" else "target_control.py")]}


def episode_specs(p, selected):
    specs = {}
    for config in p["configs"]:
        for seed in p["train_seeds"]:
            for target in p["targets_ml"]:
                name = f"calibration_{config['name']}_t{target}_s{seed}"
                specs[name] = expected_conditions(p, selected, "calibration", "candidate", target, seed, config)
    for stage, seeds in (("validation", p["validation_seeds"]), ("test", p["test_seeds"])):
        for seed in seeds:
            for target in p["targets_ml"]:
                for role in ("candidate", "baseline"):
                    name = f"{stage}_{role}_t{target}_s{seed}"
                    specs[name] = expected_conditions(p, selected, stage, role, target, seed)
    return specs


def audit_implementation(implementation, record, run, root, sources):
    require(implementation["type"] == "controller_implementation", "Missing actual controller metadata")
    require(implementation == record["controller_implementation"], "Journal controller provenance differs from trace")
    require(implementation["controller_file_sha256"] == record["conditions"]["controller_source_sha256"],
            "Actual controller source differs from the role's frozen source")
    controller_name = "precision_control.py" if record["conditions"]["policy_role"] == "candidate" else "target_control.py"
    require(Path(implementation["controller_file"]).resolve() == (root / SCRIPTS / controller_name).resolve(),
            "Actual controller path differs from its role")
    require(implementation["wall_clock_start_utc"] == record["episode_wall_clock_start_utc"], "Reset UTC differs in journal")
    require(utc(implementation["metadata_recorded_at_utc"]) >= utc(implementation["wall_clock_start_utc"]),
            "Implementation metadata predates reset")
    require(utc(record["completed_at_utc"]) >= utc(implementation["metadata_recorded_at_utc"]), "Completion predates metadata")
    close(implementation["physical_time_s"], 0, "Implementation physical start")
    require(implementation["physical_solver_changed_by_wrapper"] is False
            and implementation["episode_result_changed_by_wrapper"] is False, "Wrapper claims changed physics/results")
    for prefix, relative in (("precision_wrapper", SCRIPTS + "precision_training_env.py"),
                             ("base_environment", SCRIPTS + "pour_training_env.py")):
        require(implementation[prefix + "_sha256"] == sources[relative]
                and Path(implementation[prefix + "_file"]).resolve() == (root / relative).resolve(),
                "Environment provenance differs: " + prefix)
    require(len(implementation["controller_mro_sources"]) == (2 if controller_name == "precision_control.py" else 1),
            "Unexpected controller inheritance chain")
    for index, entry in enumerate(implementation["controller_mro_sources"]):
        relative = Path(entry["file"]).resolve().relative_to(root.resolve()).as_posix()
        require(relative in sources and entry["file_sha256"] == sources[relative], "MRO file is not a frozen source")
        if entry["class_source_sha256"] is not None:
            require(source_class_hash(run / "source_snapshot" / relative, entry["class"]) == entry["class_source_sha256"],
                    "MRO class source digest differs")
        if index == 0:
            require(entry["class"] == implementation["controller_class"] and entry["module"] == implementation["controller_module"]
                    and entry["file_sha256"] == implementation["controller_file_sha256"]
                    and entry["class_source_sha256"] == implementation["controller_class_source_sha256"],
                    "Actual class provenance disagrees with its MRO entry")


def particle_number(volume, context):
    count = round(volume / QUANTUM_ML)
    require(count >= 0, context + ": negative particle count")
    close(volume, count * QUANTUM_ML, context, 1e-9)
    return count


def audit_trace(path, record, run, root, sources, p, limits):
    """Stream a complete trace; diagnosed failed rollouts remain audit inputs."""
    name = record["episode_id"]
    require(digest(path) == record["record_sha256"], name + ": gzip bytes SHA mismatch")
    result = record["result"]
    finite(record, name)
    role = record["conditions"]["policy_role"]
    maximum = Counter()
    previous = previous_previous = last = None
    step_rows = 0
    failures = []
    footer = None
    pulse_events = []
    motion_samples = []
    summary_mismatch = False
    min_substeps = math.inf
    max_substeps = 0
    substeps_sum = 0
    max_below = max_spilled = 0
    max_logged_command_rate = 0.0
    max_logged_command_acceleration = 0.0
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        header = json.loads(next(stream))
        implementation = json.loads(next(stream))
        require(header["type"] == "episode", name + ": missing episode header")
        audit_implementation(implementation, record, run, root, sources)
        for key in ("target_ml", "seed", "randomized"):
            require(header[key] == record["conditions"][key], name + ": header condition mismatch " + key)
        require(header["variation"] == result["variation"], name + ": initial randomization differs")
        require(header["policy_name"] == result["policy_name"]
                and header["policy_params"] == result["policy_params"] == record["conditions"]["policy_params"],
                name + ": policy identity/parameters differ from the declared role")
        require(header["particle_count"] == PARTICLES and header["minimum_substeps"] == 4,
                name + ": header solver/count differs")
        for key, expected in (("physics_dt_s", DT), ("particle_volume_ml", QUANTUM_ML),
                              ("particle_spacing_m", .0035), ("density_kg_m3", 1000)):
            close(header[key], expected, name + ": header " + key)
        require(header["fluid_mode"] == "live_PBF_action_conditioned" and header["rendered_training"] is False,
                name + ": trace is not live headless physics")
        for short in ("liquid.py", "geometry.py", "target_control.py"):
            require(header["source_sha256"][short] == sources[SCRIPTS + short], name + ": inherited source header differs")
        for line_number, line in enumerate(stream, 3):
            row = json.loads(line)
            finite(row, f"{name}:line{line_number}")
            require(footer is None, name + ": data exists after the result footer")
            if row.get("type") == "result":
                footer = dict(row)
                footer.pop("type")
                continue
            if row.get("type") == "failure":
                failures.append(row)
                continue
            require(row.get("type") == "step" and not failures, name + ": unexpected event/step after a failure")
            step = row["step"]
            require(step == step_rows, name + ": missing/duplicate/nonmonotonic physical step")
            close(row["physics_dt_s"], DT, name + ": step dt", 1e-14)
            close(row["physical_time_s"], step * DT, name + ": raw physical clock", 1e-12)
            t = row["truth"]
            a = row["action"]
            s = row["controller_status"]
            o = row["observation"]
            require(s["policy_params"] == record["conditions"]["policy_params"]
                    and s["policy_name"] == result["policy_name"], name + ": step policy changed")
            close(t["target_ml"], result["target_ml"], name + ": step target")
            require(t["source_partition_count"] == t["source_particle_count"] == SOURCE_PARTICLES,
                    name + ": source labels do not partition 1198 particles")
            require(t["total_partition_count"] == t["total_particle_count"] == PARTICLES,
                    name + ": total labels do not partition 5330 particles")
            counts = [t[key] for key in ("beaker_particles", "tube_particles", "airborne_particles", "spilled_particles")]
            require(all(isinstance(count, int) and not isinstance(count, bool) and count >= 0 for count in counts)
                    and sum(counts) == PARTICLES, name + ": invalid total classification")
            source_counts = [particle_number(t[key], name + ": " + key)
                             for key in ("received_ml", "remaining_ml", "inflight_ml", "spilled_ml")]
            require(sum(source_counts) == SOURCE_PARTICLES and all(x <= y for x, y in zip(source_counts, counts)),
                    name + ": source volume/classification mismatch")
            require(0 <= t["below_table_particles"] <= t["spilled_particles"], name + ": invalid below-table count")
            close(t["mass_kg"], TOTAL_MASS, name + ": actual particle mass sum", 1e-12)
            maximum["mass_drift_kg"] = max(maximum["mass_drift_kg"], abs(t["mass_kg"] - TOTAL_MASS))
            max_below = max(max_below, t["below_table_particles"])
            max_spilled = max(max_spilled, t["spilled_ml"])
            require(o == {key: s.get(key) for key in o}, name + ": observation differs from actual controller status")
            require(a["policy"] == s.get("policy_action"), name + ": policy action differs from actual controller status")
            require(row["precision_phase"] == s.get("precision_phase"), name + ": optional precision phase differs from actual status")
            for key in ("command_goal_source", "command_max_rate", "command_acceleration", "command_source_velocity", "command_next_source_velocity"):
                require(a[key] == s.get(key), name + ": command differs from actual status " + key)
            close(row["source_time"], s["source_time"], name + ": measured/commanded source position", 1e-9)
            require(-1e-10 <= row["source_time"] <= 8.4 + 1e-10, name + ": source path leaves its domain")
            for key in ("received_ml", "tube_remaining_ml"):
                truth_key = "remaining_ml" if key == "tube_remaining_ml" else key
                close(s[key], t[truth_key], name + ": status " + key, 1e-9)
            require(-1e-12 <= s["inflight_ml"] <= t["inflight_ml"] + 1e-9,
                    name + ": catch-predicted inflight exceeds all airborne source water")
            for key in ("command_source_velocity", "command_next_source_velocity"):
                maximum["command_speed"] = max(maximum["command_speed"], abs(a[key]))
            maximum["command_acceleration"] = max(maximum["command_acceleration"], abs(a["command_acceleration"]))
            if step > 0:
                max_logged_command_rate = max(max_logged_command_rate, abs(a["command_source_velocity"]), abs(a["command_next_source_velocity"]))
                max_logged_command_acceleration = max(max_logged_command_acceleration, abs(a["command_acceleration"]))
            if role == "candidate":
                pulse = row["pulse_count"]
                require(isinstance(pulse, int) and 0 <= pulse <= limits["MAX_PULSES"]
                        and pulse == s["pulse_count"] == a["policy"]["pulse_count"], name + ": invalid pulse count")
                require(step != 0 or pulse == 0, name + ": initial pulse count is not reset")
                require(pulse >= (previous["pulse_count"] if previous else 0), name + ": pulse count decreased")
                if previous and pulse != previous["pulse_count"]:
                    require(pulse == previous["pulse_count"] + 1 and a["policy"]["event_this_frame"]
                            and a["policy"]["event"] == "start_fine_pulse", name + ": pulse lacks its real start event")
                    pulse_events.append({"step": step, "physical_time_s": row["physical_time_s"],
                                         "pulse_count": pulse, "source_time": row["source_time"]})
                require(a["policy"]["decision"] == s["phase"], name + ": pulse action phase mismatch")
                require(s["nearest_target_count"] == int(math.floor(result["target_ml"] / QUANTUM_ML + .5)),
                        name + ": target particle conversion mismatch")
            else:
                require(row["pulse_count"] is None, name + ": baseline incorrectly claims precision pulses")
            substeps = row["solver_substeps"]
            require(isinstance(substeps, int) and substeps >= (4 if step else 0), name + ": invalid/reduced solver substeps")
            if step:
                min_substeps = min(min_substeps, substeps)
                max_substeps = max(max_substeps, substeps)
                substeps_sum += substeps
            if previous:
                speed = abs(row["source_time"] - previous["source_time"]) / DT
                maximum["finite_difference_speed"] = max(maximum["finite_difference_speed"], speed)
                predicted = (previous["source_time"] + previous["action"]["command_source_velocity"] * DT
                             + .5 * previous["action"]["command_acceleration"] * DT * DT)
                residual = abs(row["source_time"] - predicted)
                velocity_jump = abs(a["command_source_velocity"] - previous["action"]["command_next_source_velocity"])
                maximum["position_integration_residual"] = max(maximum["position_integration_residual"], residual)
                maximum["command_velocity_boundary_jump"] = max(maximum["command_velocity_boundary_jump"], velocity_jump)
                if residual > 1e-9 or velocity_jump > 1e-8:
                    if len(motion_samples) < 8:
                        motion_samples.append({"step": step, "source_position_integration_residual": residual,
                                               "command_velocity_boundary_jump": velocity_jump})
            if previous_previous:
                acceleration = abs(row["source_time"] - 2 * previous["source_time"] + previous_previous["source_time"]) / DT ** 2
                maximum["finite_difference_acceleration"] = max(maximum["finite_difference_acceleration"], acceleration)
            previous_previous, previous, last = previous, row, row
            step_rows += 1
    require(footer == result, name + ": result footer differs from the journal result (including failures)")
    require(step_rows > 0 and last is not None, name + ": no initial physical measurement")
    last_step = last["step"]
    terminal_gap = result["physical_steps"] - last_step
    require(terminal_gap in (0, 1), name + ": unaccounted physical-step coverage gap")
    if failures:
        require(len(failures) == 1 and result["valid_physics"] is False
                and result["termination_reason"] == "invalid_physics_state"
                and failures[0]["error"] == result["error"], name + ": failure event/result mismatch")
        require(failures[0]["step"] == last_step + 1, name + ": failure event step mismatch")
        close(failures[0]["physical_time_s"], result["physical_time_s"], name + ": failure clock", 1e-12)
    else:
        require(terminal_gap == 0 and result["error"] is None, name + ": unexplained missing terminal state")
    close(result["physical_time_s"], result["physical_steps"] * DT, name + ": result physical clock", 1e-12)
    close(result["simulation_seconds"], result["physical_time_s"], name + ": simulation clock")
    close(result["physics_dt_s"], DT, name + ": result dt", 1e-14)
    require(result["physical_time_s"] <= p["physics"]["max_seconds"] + DT, name + ": exceeds declared physical budget")
    require(result["particle_count"] == PARTICLES and result["source_particle_count"] == SOURCE_PARTICLES,
            name + ": result particle counts differ")
    for key, expected in (("particle_volume_ml", QUANTUM_ML), ("initial_source_ml", QUANTUM_ML * SOURCE_PARTICLES),
                          ("actual_initial_source_mass_kg", SOURCE_MASS), ("actual_initial_total_mass_kg", TOTAL_MASS)):
        close(result[key], expected, name + ": result " + key, 1e-10 if "ml" in key else 1e-12)
    close(result["error_ml"], result["received_ml"] - result["target_ml"], name + ": signed error")
    close(result["absolute_error_ml"], abs(result["error_ml"]), name + ": absolute error")
    close(result["tolerance_ml"], .1, name + ": tolerance")
    nearest_count = int(math.floor(result["target_ml"] / QUANTUM_ML + .5))
    require(result["nearest_particle_count"] == nearest_count, name + ": nearest particle count")
    quantization = nearest_count * QUANTUM_ML - result["target_ml"]
    close(result["nearest_attainable_ml"], nearest_count * QUANTUM_ML, name + ": nearest attainable volume")
    close(result["quantization_error_ml"], quantization, name + ": signed quantization floor")
    close(result["minimum_abs_quantization_error_ml"], abs(quantization), name + ": absolute quantization floor")
    if not failures:
        close(result["actual_final_total_mass_kg"], last["truth"]["mass_kg"], name + ": final actual mass", 1e-12)
        for key in ("received_ml", "inflight_ml", "remaining_ml", "spilled_ml"):
            close(result[key], last["truth"][key], name + ": final " + key, 1e-9)
        close(result["max_spilled_ml"], max_spilled, name + ": maximum spill", 1e-9)
        require(result["max_below_table_particles"] == max_below, name + ": maximum below table")
        close(result["maximum_mass_drift_kg"], maximum["mass_drift_kg"], name + ": maximum mass drift", 1e-12)
        require(record["controller_summary"] == last["controller_status"], name + ": final controller status differs")
        close(result["minimum_substeps"], min_substeps, name + ": minimum substeps")
        close(result["maximum_substeps"], max_substeps, name + ": maximum substeps")
        close(result["average_substeps"], substeps_sum / result["physical_steps"], name + ": average substeps", 1e-10)
        close(result["max_source_acceleration"], max_logged_command_acceleration, name + ": declared command acceleration", 1e-9)
        close(result["max_source_rate"], max(maximum["finite_difference_speed"], max_logged_command_rate), name + ": declared source speed", 1e-9)
        require(result["complete"] == last["controller_status"]["done"], name + ": completion flag differs from the controller")
        close(result["final_tilt_degrees"], abs(last["controller_status"]["tilt_deg"]), name + ": final tilt", 1e-8)
    else:
        summary_mismatch = record["controller_summary"] != last["controller_status"]
        # A caught terminal exception may occur after a new measurement but
        # before its step is serialized. Its footer is still preserved exactly;
        # do not claim an unlogged failing particle state was independently seen.
        require(result["max_spilled_ml"] >= max_spilled
                and result["max_below_table_particles"] >= max_below
                and result["maximum_mass_drift_kg"] >= maximum["mass_drift_kg"],
                name + ": failure result discards an already recorded safety maximum")
    require(result["mass_conserved"] == bool(result["maximum_mass_drift_kg"] <= 1e-12
            and result["mass_volume_units_consistent"] and result["particle_mass_array_unchanged"]),
            name + ": mass conservation flag differs from its recorded checks")
    expected_valid = bool(result["error"] is None and result["all_states_finite"]
                          and result["source_partition_conserved"] and result["mass_conserved"]
                          and result["reset_max_position_error_m"] < 1e-7
                          and result["max_source_rate"] <= 1.00001
                          and result["max_source_acceleration"] <= 3.00001
                          and result["minimum_substeps"] >= 4)
    require(result["valid_physics"] == expected_valid, name + ": original physical validity definition differs")
    require(result["upright"] == (result["final_tilt_degrees"] < .1), name + ": upright flag mismatch")
    expected_success = bool(safe(result) and result["absolute_error_ml"] <= .1)
    require(result["success"] == expected_success, name + ": base success flag differs from its original definition")
    variation = result["variation"]
    close(variation["initial_jitter_max_spacing"], .03, name + ": jitter bound")
    jitter_roundoff = initial_jitter_roundoff(run / "source_snapshot" / SCRIPTS / "geometry.py")
    require(0 <= variation["initial_jitter_realized_max_m"] <= .03 * .0035 + jitter_roundoff + 1e-12
            and .9 <= variation["viscosity_factor"] <= 1.1, name + ": randomization leaves declared bounds")
    close(variation["dynamic_viscosity_pa_s"], .001 * variation["viscosity_factor"], name + ": actual dynamic viscosity", 1e-12)
    close(variation["kinematic_viscosity_m2_s"], variation["dynamic_viscosity_pa_s"] / 1000,
          name + ": actual kinematic viscosity", 1e-14)
    independent_motion_passed = (maximum["finite_difference_speed"] <= 1.00001
                                 and maximum["command_speed"] <= 1.00001
                                 and maximum["command_acceleration"] <= 3.00001)
    if role == "candidate":
        independent_motion_passed = bool(independent_motion_passed
                                          and maximum["finite_difference_acceleration"] <= 3.00001
                                          and maximum["position_integration_residual"] <= 1e-9
                                          and maximum["command_velocity_boundary_jump"] <= 1e-8)
    return {"episode_id": name, "role": role, "stage": record["conditions"]["stage"],
            "target_ml": result["target_ml"], "seed": result["seed"],
            "record_sha256": record["record_sha256"], "raw_step_rows_including_initial": step_rows,
            "physical_steps": result["physical_steps"], "terminal_unlogged_failure_steps": terminal_gap,
            "failure_events": failures, "controller_summary_changed_after_last_logged_step": summary_mismatch,
            "original_valid_physics": result["valid_physics"], "safe_under_predeclared_protocol": safe(result),
            "termination_reason": result["termination_reason"], "error_ml": result["error_ml"],
            "within_0_1_ml": safe(result) and abs(result["error_ml"]) <= .1,
            "within_0_2_ml": safe(result) and abs(result["error_ml"]) <= .2,
            "final_pulse_count": record["controller_summary"].get("pulse_count"), "pulse_start_events": pulse_events,
            "precision_failure": record["controller_summary"].get("precision_failure"),
            "precision_success": record["controller_summary"].get("precision_success"),
            "observed_maxima": dict(maximum), "motion_discontinuity_examples": motion_samples,
            "independent_motion_checks_passed": independent_motion_passed,
            "actual_source_fd_speed_bound_passed": maximum["finite_difference_speed"] <= 1.00001,
            "actual_source_fd_acceleration_bound_passed": maximum["finite_difference_acceleration"] <= 3.00001,
            "baseline_goal_snap_is_diagnostic_only": role == "baseline",
            "actual_controller_sha256": implementation["controller_file_sha256"],
            "initial_jitter_float32_roundoff_allowance_m": jitter_roundoff,
            "wall_clock_start_utc": implementation["wall_clock_start_utc"]}


def audit_run(run=DEFAULT_RUN, *, root=ROOT):
    run, root = Path(run).resolve(), Path(root).resolve()
    # This gate must precede all result/journal/model reads. In particular it
    # cannot peek at a still-running holdout to choose or alter parameters.
    summary_path = run / "summary.json"
    require(summary_path.is_file(), "Refusing partial experiment: final summary.json does not exist")
    summary = read_json(summary_path)
    require(summary.get("complete") is True and summary.get("total_recorded_episodes") == 48,
            "Refusing partial experiment: summary must explicitly complete all 48 episodes")
    finite(summary, "summary")
    p, pid, sources, gui, job = validate_manifests(run, root, summary)
    calibration_doc = read_json(run / "calibration.json")
    selected = calibration_doc["selected_config"]
    require(selected in p["configs"], "Selected candidate is outside predeclared input configurations")
    specs = episode_specs(p, selected)
    records = {}
    journal_order = []
    with (run / "episodes.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            name = record["episode_id"]
            require(name not in records and name in specs, "Duplicate/unexpected journal episode: " + name)
            require(record["protocol_id"] == pid and record["conditions"] == specs[name]
                    and record["conditions_sha256"] == signature(specs[name]), "Journal condition mismatch: " + name)
            require(record["record_path"] == "traces/" + name + ".steps.jsonl.gz", "Unexpected trace path: " + name)
            require(record["result"]["target_ml"] == specs[name]["target_ml"]
                    and record["result"]["seed"] == specs[name]["seed"]
                    and record["result"]["randomized"] is True, "Journal result condition mismatch: " + name)
            records[name] = record
            journal_order.append(name)
    require(len(records) == 48 and set(records) == set(specs), "Journal does not contain exactly all 48 unique episodes")
    require(Counter(r["conditions"]["stage"] for r in records.values())
            == {"calibration": 18, "validation": 12, "test": 18}, "Journal stage budget mismatch")
    expected_trace_names = {record["record_path"] for record in records.values()}
    actual_trace_names = {path.relative_to(run).as_posix() for path in (run / "traces").glob("*.gz")}
    require(actual_trace_names == expected_trace_names, "Unjournaled/extra/missing gzip traces exist")
    with (run / "episodes.csv").open(encoding="utf-8-sig", newline="") as stream:
        csv_rows = list(csv.DictReader(stream))
    require(len(csv_rows) == 48 and {row["episode_id"] for row in csv_rows} == set(records),
            "Episode CSV omits/duplicates raw journal episodes")
    for row in csv_rows:
        rec = records[row["episode_id"]]
        for key in ("stage", "policy_role"):
            require(row[key] == rec["conditions"][key], "CSV episode condition differs: " + key)
        for key in ("received_ml", "error_ml", "physical_steps", "physical_time_s", "wall_seconds"):
            close(float(row[key]), rec["result"][key], "CSV episode result " + key, 1e-9)
    limits = controller_limits(run / "source_snapshot" / SCRIPTS / "precision_control.py")
    traces = [audit_trace(contained(run, records[name]["record_path"]), records[name], run, root, sources, p, limits)
              for name in journal_order]
    calibration = []
    for index, config in enumerate(p["configs"]):
        values = [records[f"calibration_{config['name']}_t{target}_s{seed}"]["result"]
                  for seed in p["train_seeds"] for target in p["targets_ml"]]
        calibration.append({"index": index, **config, "group_score": group_score(values),
                            "metrics": summarize(values), "results": values})
    best = min(calibration, key=lambda row: (row["group_score"], row["index"]))
    require(selected == {"name": best["name"], "params": best["params"]}, "Calibration selection does not minimize the declared score")
    expected_calibration = {"protocol_id": pid, "method": ALGORITHM, "configs": calibration, "selected_config": selected}
    equivalent(calibration_doc, expected_calibration, "calibration.json")
    equivalent(summary["calibration"], {"episodes": 18, "configurations": 3,
                                      "selected_config": selected, "configs": calibration}, "summary.calibration")
    validation = {role: [records[f"validation_{role}_t{target}_s{seed}"]["result"]
                         for seed in p["validation_seeds"] for target in p["targets_ml"]]
                  for role in ("candidate", "baseline")}
    validation_scores = {role: group_score(values) for role, values in validation.items()}
    recommended_role = "candidate" if validation_scores["candidate"] < validation_scores["baseline"] else "baseline"
    validation_report = {"protocol_id": pid, "group_score": validation_scores,
                         "metrics": {role: summarize(values) for role, values in validation.items()},
                         "results": validation, "recommended_role": recommended_role, "test_used_for_selection": False}
    equivalent(read_json(run / "validation.json"), validation_report, "validation.json")
    equivalent(summary["validation"], validation_report, "summary.validation")
    freeze = read_json(run / "freeze.json")
    require(freeze["protocol_id"] == pid and freeze["candidate_config"] == selected
            and freeze["recommended_role"] == recommended_role and freeze["baseline_params"] == p["baseline_params"]
            and freeze["automatic_gui_deployment"] is False and freeze["test_used_for_selection"] is False,
            "Pre-test freeze differs from calibration/validation recommendation")
    expected_model_lines = []
    changed = selected["params"] != p["precision_defaults_at_freeze"]
    for role in ("candidate", "recommended"):
        path = run / (role + "_frozen.json")
        expected_digest = freeze[role + "_sha256"]
        require(digest(path) == expected_digest, role + ": checkpoint digest differs from freeze")
        model = read_json(path)
        is_precision = role == "candidate" or recommended_role == "candidate"
        policy_file = "precision_control.py" if is_precision else "learned_policy.py"
        require(model["schema_version"] == 1 and model["policy_kind"] == source_literal(run / "source_snapshot" / SCRIPTS / policy_file, "POLICY_KIND"),
                role + ": checkpoint policy kind differs from its frozen implementation")
        params = selected["params"] if is_precision else p["baseline_params"]
        require(model["parameters"] == params, role + ": frozen parameters differ")
        m = model["metadata"]
        expected_metadata = {"frozen_at_utc": freeze["frozen_at_utc"], "protocol_id": pid, "algorithm": ALGORITHM,
                             "neural_policy": False, "ppo": False, "calibration_configurations": 3,
                             "calibration_episodes": 18, "selected_config_name": selected["name"],
                             "validation_scores": validation_scores, "test_used_for_selection": False,
                             "source_sha256": p["source_sha256"], "config_input_sha256": p["config_input_sha256"],
                             "automatic_gui_deployment": False, "role": role, "calibration": is_precision,
                             "parameter_changed_from_precision_default": changed if is_precision else False,
                             "controller_changed_from_legacy_baseline": is_precision}
        if role == "recommended":
            expected_metadata["recommended_role"] = recommended_role
        equivalent(m, expected_metadata, role + ": model metadata")
        require("is_trained" not in m, role + ": ambiguous is_trained label replaces the declared calibration metadata")
        equivalent(summary["models"][role], {"file": path.name, "sha256": expected_digest, "parameters": params},
                   "summary.models." + role)
        expected_model_lines.append(expected_digest + "  " + path.name)
    require((run / "models_frozen.sha256").read_text(encoding="utf-8").splitlines() == expected_model_lines,
            "Two-model checksum file differs")
    freeze_time = utc(freeze["frozen_at_utc"])
    test_starts = [utc(record["episode_wall_clock_start_utc"]) for record in records.values()
                   if record["conditions"]["stage"] == "test"]
    require(freeze_time < min(test_starts), "A holdout reset started before both models were frozen")
    require(max(utc(record["completed_at_utc"]) for record in records.values()
                if record["conditions"]["stage"] != "test") <= freeze_time,
            "Calibration or validation was completed after model freezing")
    pairs = [{"target_ml": target, "seed": seed,
              **{role: records[f"test_{role}_t{target}_s{seed}"]["result"] for role in ("candidate", "baseline")},
              "recommended_result_role": recommended_role}
             for seed in p["test_seeds"] for target in p["targets_ml"]]
    test_metrics = {role: summarize([pair[role] for pair in pairs]) for role in ("candidate", "baseline")}
    expected_test = {"paired_results": pairs, "candidate_metrics": test_metrics["candidate"],
                     "baseline_metrics": test_metrics["baseline"], "recommended_metrics": test_metrics[recommended_role],
                     "distinct_target_seed_conditions": 9, "independent_initial_state_seeds": 3,
                     "executed_live_episodes": 18,
                     "candidate_mae_better_than_baseline": test_metrics["candidate"]["mae_ml"] < test_metrics["baseline"]["mae_ml"],
                     "candidate_worst_error_not_worse": test_metrics["candidate"]["worst_absolute_error_ml"] <= test_metrics["baseline"]["worst_absolute_error_ml"]}
    equivalent(summary["test"], expected_test, "summary.test")
    expected_release = {"criteria": p["release_criteria"], **{role + "_passed": release(test_metrics[role]) for role in test_metrics},
                        "recommended_passed": release(test_metrics[recommended_role]), "does_not_reselect_model": True,
                        "automatic_gui_deployment": False}
    equivalent(summary["release_acceptance"], expected_release, "summary.release_acceptance")
    equivalent(summary["recommendation"], {"role": recommended_role, "selected_on_validation_only": True,
                                          "test_result_is_reference": True, "additional_test_rollouts": 0}, "summary.recommendation")
    equivalent(summary["all_rollouts"], summarize([records[name]["result"] for name in journal_order]), "summary.all_rollouts")
    for key, expected in {"physical_log_result_footer_modified": False, "test_used_for_training_or_selection": False,
                          "source_sha256_unchanged_after_run": True, "protected_legacy_eight_unchanged": True,
                          "gui_checkpoint_unchanged": True, "neural_policy": False, "algorithm": ALGORITHM}.items():
        require(summary[key] == expected, "Summary declaration mismatch: " + key)
    require(utc(summary["completed_at_utc"]) >= max(utc(r["completed_at_utc"]) for r in records.values()),
            "Summary predates completed rollouts")
    diagnostics = [row for row in traces if not row["within_0_1_ml"] or row["precision_failure"]
                   or not row["independent_motion_checks_passed"] or row["failure_events"]]
    return {"audit_schema_version": 1, "audited_at_utc": datetime.now(timezone.utc).isoformat(),
            "audit_tool_sha256": digest(__file__), "run_directory": str(run), "protocol_id": pid,
            "complete_run_only": True, "artifact_integrity_verified": True,
            "summary_sha256": digest(summary_path), "journal_sha256": digest(run / "episodes.jsonl"),
            "algorithm": ALGORITHM, "neural_policy": False, "ppo": False,
            "budgets": {"calibration": 18, "validation": 12, "test": 18, "total": 48},
            "source_files_and_snapshots_verified": 11, "legacy_sources_protected": 8,
            "historical_manifest_verified": p["historical_manifest_verified"],
            "protected_gui_checkpoints": gui, "supervisor": job,
            "raw_gzip_traces_verified": len(traces), "result_footers_equal_journal": True,
            "raw_step_rows_including_initial": sum(row["raw_step_rows_including_initial"] for row in traces),
            "terminal_unlogged_failure_steps": sum(row["terminal_unlogged_failure_steps"] for row in traces),
            "physics_dt_s": DT, "particle_volume_ml": QUANTUM_ML,
            "nominal_source_ml": QUANTUM_ML * SOURCE_PARTICLES, "actual_float32_source_mass_kg": SOURCE_MASS,
            "actual_float32_total_mass_kg": TOTAL_MASS,
            "calibration": [{key: row[key] for key in ("name", "params", "group_score", "metrics")} for row in calibration],
            "candidate_configuration": selected, "parameter_changed_from_precision_default": changed,
            "validation_scores": validation_scores, "recommended_role_from_validation_only": recommended_role,
            "model_freeze_utc": freeze["frozen_at_utc"], "first_holdout_reset_utc": min(test_starts).isoformat(),
            "freeze_to_first_test_reset_seconds": (min(test_starts) - freeze_time).total_seconds(),
            "freeze_order_evidence": "Explicit trace UTC is recorded on _reset entry before initialization/physics; not inferred by subtracting episode wall duration.",
            "test_metrics": test_metrics, "release_acceptance": expected_release,
            "test_did_not_change_parameters_or_recommendation": True, "automatic_gui_deployment": False,
            "paired_test_errors": [{"target_ml": pair["target_ml"], "seed": pair["seed"],
                                    **{role + "_error_ml": pair[role]["error_ml"] for role in test_metrics}}
                                   for pair in pairs],
            "all_rollouts": summarize([records[name]["result"] for name in journal_order]),
            "independent_motion_checks_all_passed": all(row["independent_motion_checks_passed"] for row in traces),
            "actual_source_fd_acceleration_bound_passed_all_roles": all(row["actual_source_fd_acceleration_bound_passed"] for row in traces),
            "candidate_valid_flag_with_independent_motion_discrepancy": [row["episode_id"] for row in traces
                if row["role"] == "candidate" and row["original_valid_physics"] and not row["independent_motion_checks_passed"]],
            "episodes_requiring_diagnostic_attention": diagnostics, "all_episode_audits": traces,
            "limitations": [
                "Recorded aggregate mass, partitions, timing, commands, action dictionaries and provenance are independently checked; raw particle positions/velocities are not serialized, so full-state finite checks remain the frozen solver collector's assertions.",
                "Volume is privileged particle-label geometry with 0.042875mL nominal quantum and float32 mass consistency, not calibrated camera/scale accuracy or physical hardware validation.",
                "Reset jitter is sampled within 0.03 spacing before a float32 position cast; the realized coordinate difference can exceed that nominal bound by the geometry-derived half-ULP rounding allowance recorded per trace.",
                "Three hand-designed parameter configurations and fresh initial-state jitter/viscosity seeds provide local calibration evidence, not neural/RL policy learning or broad sim-to-real generalization.",
                "Baseline goal-clamping can cause finite-difference acceleration/velocity discontinuities; these are reported without retroactively changing its predeclared validity or release criteria.",
                "Failures, timeouts, overdoses and controller precision-failure statuses remain in every recomputed group metric; no outliers are removed.",
                "UTC/checksum provenance establishes the recorded ordering and immutable artifact consistency; it is not an external trusted timestamp or independent hardware measurement."
            ]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--output", type=Path, help="Default: RUN/precision_run_audit.json; only written after complete audit")
    parser.add_argument("--root", type=Path, default=ROOT, help="Isaac project root for immutable-source verification")
    args = parser.parse_args()
    try:
        report = audit_run(args.run, root=args.root)
        output = (args.output or args.run / "precision_run_audit.json").resolve()
        if output.exists():
            prior = read_json(output)
            require(prior.get("audit_schema_version") == 1 and "audit_tool_sha256" in prior,
                    "Refusing to overwrite an existing non-audit artifact: " + str(output))
    except (AuditError, KeyError, OSError, StopIteration, TypeError, json.JSONDecodeError) as error:
        print("PRECISION AUDIT REFUSED: " + str(error), file=sys.stderr)
        return 2
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(json.dumps({"artifact_integrity_verified": report["artifact_integrity_verified"],
                      "episodes": 48, "recommended_role": report["recommended_role_from_validation_only"],
                      "release_acceptance": report["release_acceptance"], "output": str(output)}, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
