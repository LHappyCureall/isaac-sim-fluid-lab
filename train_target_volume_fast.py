"""Formal V3 speed/stop CEM: 108 train + 12 validation + 18 holdout episodes.

The complete action has a hard five-second physical deadline, followed by live
late-drop observation through physical time seven seconds and for at least two
seconds. FastTrainingEnv generates the V3 action result with its declared path
validity predicate; its returned raw footer is retained exactly. Additional V3
motion/deadline assessments are separate data.
No checkpoint is deployed to the GUI. --help and --dry-run require no GPU.
Engineering prototypes and the final parameter bounds must be settled first.
"""

import argparse
import copy
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path

from lab_paths import PROJECT_ROOT, require_isaac_sim_root
import platform
import random
import sys
import time


ROOT = PROJECT_ROOT
SCRIPTS = ROOT / "assets/robot_testtube_pour/scripts"
TRAIN_SEEDS = (61101, 61102)
VALIDATION_SEEDS = (62101, 62102)
TEST_SEEDS = (71101, 71102, 71103)
TARGETS = (20, 30, 40)
GENERATIONS, POPULATION, ELITES = 3, 6, 2
CEM_SEED = 2026112203
TOTAL_EPISODES = 138
ALGORITHM = "feasibility_first_CEM_speed_and_stop_feedback_v3"
DEADLINE, POST_SECONDS, DT = 5.0, 2.0, 1 / 60
QUANTUM = .0035 ** 3 * 1e6
PREFIX = "assets/robot_testtube_pour/scripts/"
LEGACY_SOURCES = ("train_target_volume_policy.py", "train_target_volume_policy_full.py",
                  "train_target_volume_precision.py", *[PREFIX + name for name in (
                      "geometry.py", "liquid.py", "target_control.py", "learned_policy.py",
                      "pour_training_env.py", "pbf_graph.py", "precision_control.py", "precision_training_env.py")])
NEW_SOURCES = ("train_target_volume_fast.py", *[PREFIX + name for name in
               ("fast_geometry.py", "fast_control.py", "fast_training_env.py", "robot.py")],
               "tools/training/prepare_fast_path.py",
               "tools/training/prepare_smooth_fast_path.py",
               "tools/training/verify_fast_path.py",
               "tools/training/export_owned_kinematic_bank.py")


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode("utf-8")).hexdigest()


def normalized(value):
    return json.loads(json.dumps(value, allow_nan=False,
                                 default=lambda x: x.tolist() if hasattr(x, "tolist") else x.item()))


def read_json(path):
    def reject(value):
        raise ValueError("Nonfinite JSON token: " + value)
    return json.loads(Path(path).read_text(encoding="utf-8-sig"), parse_constant=reject)


def atomic_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def write_json(path, value):
    atomic_text(path, json.dumps(normalized(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def append_json(path, value):
    with Path(path).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(normalized(value), allow_nan=False, separators=(",", ":")) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def write_csv(path, rows, fields):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def load_module(name, path):
    path = Path(path)
    raw = path.read_bytes()
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    exec(compile(raw.decode("utf-8-sig"), str(path), "exec"), module.__dict__)
    module._runner_loaded_sha256 = hashlib.sha256(raw).hexdigest()
    return module


def finite(value):
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Nonfinite recorded number")
    if isinstance(value, dict):
        for child in value.values():
            finite(child)
    if isinstance(value, list):
        for child in value:
            finite(child)


def numeric(value, default=math.inf):
    return float(value) if type(value) in (int, float) and math.isfinite(value) else default


def paths_and_sources():
    return {name: sha(ROOT / name) for name in (*LEGACY_SOURCES, *NEW_SOURCES)}


def gui_hashes():
    return {path.relative_to(ROOT).as_posix(): sha(path)
            for path in sorted((ROOT / "assets/robot_testtube_pour/policy").glob("*.json"))}


def require_unchanged(declaration):
    if paths_and_sources() != declaration["source_sha256"]:
        raise RuntimeError("Frozen V3/legacy source changed; preserve this run and start a new protocol")
    if gui_hashes() != declaration["gui_checkpoint_sha256_before"]:
        raise RuntimeError("A protected GUI checkpoint changed; this runner never deploys models")
    for label, entry in declaration["input_files"].items():
        if sha(entry["path"]) != entry["sha256"]:
            raise RuntimeError("Frozen V3 raw input changed: " + label)


def declared_seeds(value):
    result = set()
    def numbers(child):
        if type(child) is int:
            result.add(child)
        elif isinstance(child, list):
            for one in child:
                numbers(one)
        elif isinstance(child, dict):
            for one in child.values():
                numbers(one)
    def walk(child):
        if isinstance(child, dict):
            for key, one in child.items():
                if "seed" in key.lower():
                    numbers(one)
                walk(one)
        elif isinstance(child, list):
            for one in child:
                walk(one)
    walk(value)
    return result


def historical_seeds(output):
    records, used = {}, set()
    for path in sorted((ROOT / "droplet_out").rglob("*protocol*.json")):
        if path.resolve().is_relative_to(output) or "source_snapshot" in path.parts:
            continue
        document = read_json(path)
        used.update(declared_seeds(document))
        records[path.relative_to(ROOT).as_posix()] = sha(path)
    proposed = set(TRAIN_SEEDS + VALIDATION_SEEDS + TEST_SEEDS)
    if used & proposed:
        raise ValueError("Formal V3 seeds overlap a previous declared run: " + str(sorted(used & proposed)))
    return records


class LexCEM:
    """Deterministic bounded Gaussian CEM; elites/best use the original tuple."""

    def __init__(self, names, bounds, initial, *, seed=CEM_SEED, initial_std_fraction=.2):
        self.names = tuple(names)
        self.bounds = {key: tuple(bounds[key]) for key in self.names}
        self.mean = dict(initial)
        self.std = {key: (bounds[key][1] - bounds[key][0]) * initial_std_fraction for key in self.names}
        self.rng = random.Random(seed)
        self.best_params, self.best_key = None, None
        self.generation = 0

    def ask(self):
        candidates = [dict(self.mean)]
        if self.best_params is not None and self.best_params != self.mean:
            candidates.append(dict(self.best_params))
        while len(candidates) < POPULATION:
            candidates.append({key: min(self.bounds[key][1], max(self.bounds[key][0],
                self.rng.gauss(self.mean[key], self.std[key]))) for key in self.names})
        return candidates

    def tell(self, candidates, keys):
        if len(candidates) != POPULATION or len(keys) != POPULATION:
            raise ValueError("CEM generation budget differs")
        indices = sorted(range(POPULATION), key=lambda index: (tuple(keys[index]), index))
        for index in indices:
            key = tuple(keys[index])
            if self.best_key is None or key < self.best_key:
                self.best_key, self.best_params = key, dict(candidates[index])
        elites = [candidates[index] for index in indices[:ELITES]]
        for name in self.names:
            mean = sum(item[name] for item in elites) / ELITES
            std = math.sqrt(sum((item[name] - mean) ** 2 for item in elites) / ELITES)
            self.mean[name] = .65 * self.mean[name] + .35 * mean
            self.std[name] = max(.02 * (self.bounds[name][1] - self.bounds[name][0]),
                                 .65 * self.std[name] + .35 * std)
        self.generation += 1
        return {"elite_indices": indices[:ELITES], "generation_best_key": list(keys[indices[0]]),
                "best_key": list(self.best_key), "best_params": dict(self.best_params),
                "next_mean": dict(self.mean), "next_std": dict(self.std)}


def default_runtime(args):
    # Preflight is CPU-only even for an actual run. Solver initialization is
    # deferred until engineering/input/source guards have all succeeded.
    sdk = require_isaac_sim_root()
    candidates = list((sdk / "extscache").glob("omni.kit.pip_archive-*/pip_prebundle"))
    candidates += [sdk / "exts/omni.pip.compute/pip_prebundle", SCRIPTS]
    for path in candidates:
        if path.is_dir() and str(path) not in sys.path:
            sys.path.insert(0, str(path))
    def initialize_solver():
        shared = load_module("fast_formal_shared", ROOT / "train_target_volume_policy.py")
        return shared.configure_sdk()
    geometry = load_module("fast_formal_geometry", SCRIPTS / "fast_geometry.py")
    control = load_module("fast_formal_control", SCRIPTS / "fast_control.py")
    collector = load_module("fast_formal_collector", SCRIPTS / "fast_training_env.py")
    names = tuple(control.PARAMETER_NAMES)
    units = getattr(control, "PARAMETER_UNITS", None)
    if not isinstance(units, dict) or set(units) != set(names):
        raise ValueError("fast_control must declare PARAMETER_UNITS for every searched parameter")
    reference = control.TargetPourController(geometry, target_ml=30.0, policy=control.FastPolicy())
    return {"versions": {"mode": "CPU_declaration_preview_no_GPU"}, "initialize_solver": initialize_solver,
            "geometry": geometry, "control": control, "collector": collector,
            "policy_factory": control.FastPolicy, "parameter_names": names,
            "parameter_bounds": normalized(control.PARAMETER_BOUNDS), "parameter_units": dict(units),
            "baseline_params": dict(control.FastPolicy().params),
            "environment_factory": lambda: collector.FastTrainingEnv(geometry=geometry, control=control),
            "joint_acceleration_cap_rad_s2": float(collector.FastTrainingEnv.JOINT_ACCELERATION_CAP_RAD_S2),
            "collector_path_acceleration_cap": float(collector.FastTrainingEnv.PATH_ACCELERATION_CAP),
            "controller_max_u_rate": float(reference.max_speed),
            "controller_max_u_acceleration": float(reference.max_acceleration)}


def validate_engineering(report_path, runtime, baseline, cache_sha, *, allow_spill=False):
    """Verify motion/numerics; explicit opt-in can admit existing task spill."""
    report_path = Path(report_path)
    report = read_json(report_path)
    protocol_path = report_path.parent / "protocol.json"
    if report.get("complete") is not True or not protocol_path.is_file():
        raise ValueError("A complete engineering report and its adjacent protocol.json are required")
    protocol = read_json(protocol_path)
    if (protocol.get("purpose") != "engineering_feasibility_not_heldout"
            or protocol.get("cache_sha256") != cache_sha or protocol.get("policy_parameters") != baseline
            or protocol.get("deadline_s") != DEADLINE or protocol.get("post_observation_s") != POST_SECONDS
            or protocol.get("physical_dt_s") != DT or protocol.get("cuda_graph") is not False
            or protocol.get("pressure_iterations") != 6 or protocol.get("minimum_substeps") != 4):
        raise ValueError("Engineering protocol does not validate the current initial V3 policy/physics/cache")
    required = {"train_target_volume_policy.py", *[PREFIX + name for name in (
        "geometry.py", "liquid.py", "target_control.py", "pour_training_env.py", "pbf_graph.py",
        "fast_geometry.py", "fast_control.py", "fast_training_env.py")]}
    sources = protocol.get("sources", {})
    if not required.issubset(sources):
        raise ValueError("Engineering protocol omits current controller/geometry/env/solver sources")
    for name, expected in sources.items():
        path = (ROOT / name).resolve()
        if not path.is_relative_to(ROOT) or not path.is_file() or sha(path) != expected:
            raise ValueError("Engineering source differs from actual start: " + str(name))
        if sha(report_path.parent / "source_snapshot" / name) != expected:
            raise ValueError("Engineering source snapshot changed: " + str(name))
    rows = report.get("results", [])
    if len(rows) != 3 or {row["result"]["target_ml"] for row in rows} != set(TARGETS):
        raise ValueError("Engineering pilot must cover exactly 20/30/40mL for the initial V3 strategy")
    post_trace_evidence, action_trace_evidence, spill_defects = {}, {}, []
    trace_declaration = {"source_sha256": paths_and_sources(),
        "robot_envelope": {"joint_lower_limits_rad": normalized(runtime["geometry"].JOINT_LOWER_LIMITS_RAD),
                           "joint_upper_limits_rad": normalized(runtime["geometry"].JOINT_UPPER_LIMITS_RAD)},
        "input_files": {"path_cache": {"sha256": cache_sha}}}
    for row in rows:
        result, status, motion = row["result"], row["controller"], row["motion_metrics"]
        post = row.get("post_observation")
        if not (result["valid_physics"] and result["mass_conserved"] and result["complete"]
                and result["physical_time_s"] <= DEADLINE + 1e-12 and result["physical_steps"] <= 300
                and numeric(result.get("max_spilled_ml")) >= numeric(result.get("spilled_ml")) >= 0
                and result["max_below_table_particles"] == 0
                and status.get("completed_by_deadline") is True and status.get("timed_out") is False
                and status.get("home_static") is True and status.get("done") is True
                and numeric(motion.get("maximum_joint_speed_limit_ratio")) <= 1.00001
                and numeric(motion.get("maximum_joint_acceleration_rad_s2")) <= runtime["joint_acceleration_cap_rad_s2"] + 1e-4
                and numeric(motion.get("maximum_hand_fk_position_error_m")) <= .003
                and numeric(motion.get("maximum_hand_fk_orientation_error_rad")) <= .02
                and isinstance(post, dict) and post.get("all_states_finite") is True
                and post.get("mass_array_unchanged") is True and not post.get("error")
                and post.get("maximum_received_change_ml") == 0):
            raise ValueError("Engineering initial V3 strategy has not proved a numerically/kinematically feasible <=5s home cycle for every target")
        start, end, duration, dose = (numeric(post.get(key)) for key in
            ("start_physical_time_s", "end_physical_time_s", "duration_s", "deadline_received_ml"))
        if not (all(math.isfinite(x) for x in (start, end, duration, dose))
                and abs(start - result["physical_time_s"]) <= 1e-10
                and end >= DEADLINE + POST_SECONDS - 1e-10 and duration >= POST_SECONDS - 1e-10
                and abs(end - start - duration) <= 1e-10):
            raise ValueError("Engineering post observation lacks the actual start/end>=7/duration>=2/exact-5s dose sample")
        trace_path = report_path.parent / f"target_{result['target_ml']:g}.post.jsonl.gz"
        action_path = report_path.parent / f"target_{result['target_ml']:g}.steps.jsonl.gz"
        if not trace_path.is_file():
            raise ValueError("Engineering post-observation raw trace is missing: " + trace_path.name)
        if not action_path.is_file():
            raise ValueError("Engineering action raw trace is missing: " + action_path.name)
        action_checked = verify_action_trace(action_path, result, trace_declaration)
        end_partitions = action_checked["last_recorded_partitions"]
        source_spill = end_partitions["tube_spilled"]
        total_spill = end_partitions["spilled"]
        if (action_checked["maximum_below_table_particles"] != 0
                or abs(source_spill * QUANTUM - result["spilled_ml"]) > 1e-10):
            raise ValueError("Engineering action spill/below-table truth differs from its raw evidence")
        checked = verify_post_trace(trace_path, post, result)
        if (checked["partial_after_exception"]
                or checked["all_logged_states_finite"] is not True
                or checked["all_logged_mass_arrays_unchanged"] is not True):
            raise ValueError("Engineering raw post trace fails its complete late-drop safety check")
        if (checked["maximum_total_spilled_particles"] != total_spill
                or checked["minimum_total_spilled_particles"] != total_spill
                or checked["maximum_source_spilled_particles"] != source_spill
                or checked["minimum_source_spilled_particles"] != source_spill
                or checked["maximum_below_table_particles"] != 0
                or abs(numeric(post.get("final_spilled_ml")) - result["spilled_ml"]) > 1e-10):
            raise ValueError("Engineering post observation adds/changes spill counts or has below-table particles")
        # With zero late-dose change and finite/conserved observations, the
        # frozen collector's original post.passed can be false only for the
        # already-recorded source spill. This does not change that task flag.
        if post.get("passed") is not (source_spill == 0):
            raise ValueError("Engineering post.passed has an unexplained failure or inconsistent spill semantics")
        has_spill = result["max_spilled_ml"] > 0 or action_checked["maximum_total_spilled_particles"] > 0
        if has_spill and not allow_spill:
            raise ValueError("Engineering initial strategy spills; explicit --allow-engineering-spill is required to optimize this defect")
        if not allow_spill and (post.get("passed") is not True or total_spill != 0):
            raise ValueError("Strict engineering gate requires zero spill")
        if has_spill:
            spill_defects.append({"target_ml": result["target_ml"], "action_end_source_spilled_ml": result["spilled_ml"],
                "action_max_source_spilled_ml": result["max_spilled_ml"],
                "action_end_source_spilled_particles": source_spill, "action_end_total_spilled_particles": total_spill,
                "action_max_total_spilled_particles": action_checked["maximum_total_spilled_particles"],
                "deadline_received_ml": dose, "deadline_dose_error_ml": dose - result["target_ml"],
                "original_post_passed": post["passed"], "post_spill_counts_unchanged": True,
                "task_zero_spill_requirement_passed": False})
        action_trace_evidence[str(int(result["target_ml"]))] = {"file": action_path.name,
            "sha256": sha(action_path), "verification": action_checked}
        post_trace_evidence[str(int(result["target_ml"]))] = {"file": trace_path.name,
            "sha256": sha(trace_path), "verification": checked}
    return {"protocol_sha256": sha(protocol_path), "results_sha256": sha(report_path),
            "all_three_initial_cycles_numerically_and_kinematically_feasible_with_home_within_5s": True,
            "allow_existing_engineering_spill_for_optimization": bool(allow_spill),
            "initial_task_spill_defects": spill_defects,
            "engineering_gate_does_not_certify_task_success": True,
            "CEM_elite_and_release_zero_spill_requirements_relaxed": False,
            "dose_precision_not_required_for_engineering_gate": True,
            "raw_post_trace_evidence": post_trace_evidence,
            "raw_action_trace_evidence": action_trace_evidence,
            "late_drop_outcomes": [row["post_observation"].get("passed") for row in rows]}


def make_declaration(args, runtime):
    output = Path(args.output).resolve()
    if output == ROOT or output.is_relative_to(ROOT / "assets"):
        raise ValueError("Use a dedicated run directory outside GUI assets")
    if args.graph:
        raise ValueError("Formal V3 uses original PBF dispatch; graph requires a separately verified protocol")
    names = tuple(runtime["parameter_names"])
    if len(names) not in (5, 6) or len(set(names)) != len(names):
        raise ValueError("V3 expects five or six explicitly named speed/stop parameters")
    bounds = normalized(runtime["parameter_bounds"])
    baseline = normalized(runtime["policy_factory"](runtime["baseline_params"]).params)
    config = read_json(args.search_config) if args.search_config else {}
    if set(config) - {"bounds", "initial_params", "initial_std_fraction", "notes"}:
        raise ValueError("Unknown search configuration fields")
    bounds = config.get("bounds", bounds)
    if set(bounds) != set(names):
        raise ValueError("Search bounds differ from FastPolicy parameter names")
    for key in names:
        lo, hi = bounds[key]
        outer_lo, outer_hi = runtime["parameter_bounds"][key]
        if (type(lo) not in (int, float) or type(hi) not in (int, float)
                or not math.isfinite(lo) or not math.isfinite(hi) or not outer_lo <= lo < hi <= outer_hi):
            raise ValueError("Bounds must be finite nonempty subsets of FastPolicy bounds: " + key)
    initial = normalized(runtime["policy_factory"](config.get("initial_params", baseline)).params)
    for key in names:
        if not bounds[key][0] <= initial[key] <= bounds[key][1]:
            raise ValueError("Initial CEM mean is outside declared bounds: " + key)
    std_fraction = config.get("initial_std_fraction", .2)
    if type(std_fraction) not in (int, float) or not 0 < std_fraction <= 1:
        raise ValueError("initial_std_fraction must lie in (0,1]")
    inputs = {"path_cache": Path(args.path_cache), "path_manifest": Path(args.path_manifest),
              "joint_bank": Path(args.joint_bank), "bank_manifest": Path(args.bank_manifest), "urdf": Path(args.urdf)}
    for key in ("prototype_cache", "prototype_manifest", "path_report"):
        path = getattr(args, key, None)
        if path is not None:
            inputs[key] = Path(path)
    if args.search_config:
        inputs["search_config"] = Path(args.search_config)
    engineering_check = None
    if args.engineering_report:
        inputs["engineering_report"] = Path(args.engineering_report)
        engineering = read_json(args.engineering_report)
        if engineering.get("complete") is not True:
            raise ValueError("The engineering pilot report must be complete")
        sibling = Path(args.engineering_report).parent / "protocol.json"
        if sibling.is_file():
            inputs["engineering_protocol"] = sibling
        for target in TARGETS:
            inputs[f"engineering_post_t{target}"] = Path(args.engineering_report).parent / f"target_{target}.post.jsonl.gz"
            inputs[f"engineering_action_t{target}"] = Path(args.engineering_report).parent / f"target_{target}.steps.jsonl.gz"
    elif not args.dry_run:
        raise ValueError("Pass --engineering-report for the completed pre-training feasibility pilot")
    manifest = read_json(inputs["path_manifest"])
    geometry = runtime["geometry"]
    # The smoothed archive inherits its raw provenance scalars from the
    # prototype archive. Its public JSON compares clock candidates, so the
    # embedded immutable metadata is authoritative for input byte identities.
    payload = getattr(geometry, "_CACHE", {})
    cache_inputs = {key: str(payload[key]) for key in ("legacy_bank_sha256", "geometry_sha256", "urdf_sha256") if key in payload}
    if not cache_inputs:
        cache_inputs = manifest.get("inputs", {})
    if (sha(inputs["path_cache"]) != manifest["cache_sha256"]
            or sha(inputs["joint_bank"]) != cache_inputs["legacy_bank_sha256"]
            or sha(inputs["urdf"]) != cache_inputs["urdf_sha256"]
            or sha(SCRIPTS / "geometry.py") != cache_inputs["geometry_sha256"]):
        raise ValueError("Cache/bank/URDF/original geometry raw hashes disagree")
    if (Path(geometry.CACHE_FILE).resolve() != inputs["path_cache"].resolve()
            or geometry.CACHE_SHA256 != manifest["cache_sha256"]):
        raise ValueError("Effective fast_geometry loads a different cache")
    if "path_report" in inputs:
        report = read_json(inputs["path_report"])
        if report["cache_sha256"] != geometry.CACHE_SHA256 or report["geometry_sha256"] != sha(SCRIPTS / "fast_geometry.py"):
            raise ValueError("CPU path report describes a different final geometry/cache")
    if args.engineering_report:
        engineering_check = validate_engineering(args.engineering_report, runtime, baseline, manifest["cache_sha256"],
                                               allow_spill=bool(getattr(args, "allow_engineering_spill", False)))
        for kind in ("post", "action"):
            for target, entry in engineering_check[f"raw_{kind}_trace_evidence"].items():
                if sha(inputs[f"engineering_{kind}_t" + target]) != entry["sha256"]:
                    raise ValueError("Engineering raw trace changed during startup validation")
    sources = paths_and_sources()
    for relative, module in ((PREFIX + "fast_geometry.py", geometry), (PREFIX + "fast_control.py", runtime.get("control")),
                             (PREFIX + "fast_training_env.py", runtime.get("collector"))):
        if module is not None and getattr(module, "_runner_loaded_sha256", sources[relative]) != sources[relative]:
            raise ValueError("A module changed between import and source freezing: " + relative)
    legacy = {name: sources[name] for name in LEGACY_SOURCES}
    old_manifest = ROOT / "droplet_out/pour_policy_precision_v2/run_20261122_calibrated/protocol.json"
    historical_verified = old_manifest.is_file()
    if historical_verified and read_json(old_manifest)["protocol"]["source_sha256"] != legacy:
        raise ValueError("Protected V1/V2 sources differ from the completed precision experiment")
    historical = historical_seeds(output)
    return {"schema_version": 1, "algorithm": ALGORITHM, "neural_policy": False, "ppo": False,
            "targets_ml": list(TARGETS), "train_seeds": list(TRAIN_SEEDS),
            "validation_seeds": list(VALIDATION_SEEDS), "test_seeds": list(TEST_SEEDS),
            "budget": {"generations": GENERATIONS, "population": POPULATION, "elites": ELITES,
                       "training": 108, "validation": 12, "test": 18, "total": TOTAL_EPISODES},
            "cem": {"seed": CEM_SEED, "parameter_names": list(names), "bounds": bounds,
                    "initial_params": initial, "baseline_params": baseline,
                    "initial_std_fraction": std_fraction, "update_fraction": .35, "std_floor_range_fraction": .02},
            "parameter_units": normalized(runtime["parameter_units"]),
            "physics": {"mode": "live_PBF_action_conditioned", "dt_s": DT, "spacing_m": .0035,
                        "minimum_substeps": 4, "pressure_iterations": 6, "cuda_graph": False,
                        "action_deadline_s": DEADLINE, "post_observation_s": POST_SECONDS,
                        "post_observation_until_physical_time_s": DEADLINE + POST_SECONDS,
                        "dose_error_basis": "direct physical-5s measurement; invalid/unavailable samples explicitly fail and retain terminal fallback for descriptive loss",
                        "tolerance_ml": .1, "maximum_absolute_error_ml": .2},
            "completion": {"profile": "pregrasp_lift_transfer_dose_upright_return_home_stable",
                           "stable_frames": 12, "home_position_tolerance_m": .001,
                           "home_orientation_tolerance_rad": math.radians(.1), "home_joint_tolerance_rad": .01,
                           "late_drop_maximum_received_change_ml": 0.0},
            "robot_envelope": {"joint_lower_limits_rad": normalized(geometry.JOINT_LOWER_LIMITS_RAD),
                               "joint_upper_limits_rad": normalized(geometry.JOINT_UPPER_LIMITS_RAD),
                               "joint_velocity_limits_rad_s": normalized(geometry.JOINT_VELOCITY_LIMITS_RAD_S),
                               "joint_acceleration_simulation_cap_rad_s2": runtime["joint_acceleration_cap_rad_s2"],
                               "hand_fk_position_tolerance_m": .003, "hand_fk_orientation_tolerance_rad": .02,
                               "joint_derivative_sampling_hz": 240,
                               "hand_fk_and_tcp_sampling_hz": 60,
                               "acceleration_is_manufacturer_rating": False},
            "path_coordinate": {"units": "URDF_joint_velocity_normalized_path_seconds", "u_max": float(geometry.U_MAX),
                                "controller_max_u_rate_per_physical_s": runtime["controller_max_u_rate"],
                                "controller_max_u_acceleration_per_physical_s2": runtime["controller_max_u_acceleration"],
                                "collector_path_acceleration_validity_cap": runtime["collector_path_acceleration_cap"]},
            "randomization": {"initial_jitter_max_spacing": .03, "viscosity_factor": [.9, 1.1]},
            "selection_tuple": ["unsafe_count", "deadline_or_home_or_late_drop_failures", "precision_violation_ml",
                                "MAE_plus_half_worst_ml", "mean_action_physical_seconds"],
            "precision_violation_definition": "max(0,MAE-.1)+max(0,worst-.2), summed overall and per target",
            "test_use": "release/descriptive paired comparison only; cannot refit or reselect",
            "release": {"all_conditions_within_5s_and_home_stable_and_no_late_drop": True,
                        "all_safe_mass_conserved": True, "mae_ml_at_most": .1,
                        "worst_absolute_error_ml_at_most": .2, "per_target_thresholds_also_apply": True},
            "source_sha256": sources, "protected_legacy_source_sha256": legacy,
            "historical_manifest_verified": historical_verified,
            "historical_manifest_sha256": sha(old_manifest) if historical_verified else None,
            "historical_seed_protocol_sha256": historical,
            "input_files": {key: {"path": str(path.resolve()), "sha256": sha(path)} for key, path in inputs.items()},
            "cache_input_sha256": cache_inputs, "raw_cache_payload_snapshotted_without_trimming": True,
            "engineering_gate": engineering_check,
            "allow_engineering_spill": bool(getattr(args, "allow_engineering_spill", False)),
            "gui_checkpoint_sha256_before": gui_hashes(), "automatic_gui_deployment": False}


def freeze_inputs(output, declaration, resume):
    output.mkdir(parents=True, exist_ok=True)
    path = output / "input_freeze.json"
    identity = signature(declaration)
    if path.exists():
        previous = read_json(path)
        if not resume or previous["input_id"] != identity or previous["declaration"] != declaration:
            raise ValueError("Resume rejected: source/cache/bounds/inputs/GUI/seed history changed")
    else:
        if any((output / name).exists() for name in ("protocol.json", "episodes.jsonl", "summary.json", "freeze.json")):
            raise FileExistsError("Existing unrelated run has no intact input freeze")
        for relative, expected in declaration["source_sha256"].items():
            destination = output / "source_snapshot" / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes((ROOT / relative).read_bytes())
            if sha(destination) != expected:
                raise RuntimeError("Source changed while snapshotting: " + relative)
        for key, entry in declaration["input_files"].items():
            destination = output / "input_snapshot" / (key + Path(entry["path"]).suffix)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(Path(entry["path"]).read_bytes())
            if sha(destination) != entry["sha256"]:
                raise RuntimeError("Raw input changed while snapshotting: " + key)
        write_json(path, {"input_id": identity, "frozen_at_utc": now(), "declaration": declaration})
    for relative, expected in declaration["source_sha256"].items():
        if sha(output / "source_snapshot" / relative) != expected:
            raise ValueError("Frozen source snapshot changed: " + relative)
    for key, entry in declaration["input_files"].items():
        if sha(output / "input_snapshot" / (key + Path(entry["path"]).suffix)) != entry["sha256"]:
            raise ValueError("Frozen raw input snapshot changed: " + key)
    require_unchanged(declaration)
    return identity


def verify_action_trace(path, result, declaration):
    footer = effective = None
    steps, evidence = [], []
    failure = []
    lower = declaration["robot_envelope"]["joint_lower_limits_rad"]
    upper = declaration["robot_envelope"]["joint_upper_limits_rad"]
    maximum_position_violation = 0.0
    maximum_total_spilled = maximum_below_table = 0
    last_partitions = None
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        header = json.loads(next(stream))
        if header.get("type") != "episode" or (header["target_ml"], header["seed"]) != (result["target_ml"], result["seed"]):
            raise ValueError("Raw action header differs from returned conditions")
        for line in stream:
            row = json.loads(line)
            finite(row)
            if footer is not None:
                raise ValueError("Raw action contains data after its result footer")
            kind = row.get("type")
            if kind == "effective_fast_sources":
                if effective is not None:
                    raise ValueError("Duplicate effective source record")
                effective = row
            elif kind == "step":
                if row["step"] != len(steps) or abs(row["physical_time_s"] - row["step"] * DT) > 1e-12:
                    raise ValueError("Raw action step/physical time differs from 60Hz")
                steps.append(row["step"])
            elif kind == "fast_step_evidence":
                if row["step"] != len(evidence) or abs(row["physical_time_s"] - row["step"] * DT) > 1e-12:
                    raise ValueError("Fast joint evidence step/clock differs")
                partitions = row["partitions"]
                if (partitions["total"] != 5330 or partitions["initial_tube"] != 1198
                        or sum(partitions[k] for k in ("beaker", "tube", "airborne", "spilled")) != 5330
                        or sum(partitions[k] for k in ("tube_to_beaker", "tube_remaining", "tube_airborne", "tube_spilled")) != 1198):
                    raise ValueError("Explicit fast evidence loses/overlaps particle partitions")
                maximum_total_spilled = max(maximum_total_spilled, partitions["spilled"])
                maximum_below_table = max(maximum_below_table, partitions["below_table"])
                last_partitions = dict(partitions)
                q = row["kinematics"]["joint_position_rad"]
                if len(q) != 7:
                    raise ValueError("Fast evidence must contain all seven actual joint positions")
                maximum_position_violation = max(maximum_position_violation,
                    max(max(lo - one, one - hi, 0.0) for one, lo, hi in zip(q, lower, upper)))
                evidence.append(row["step"])
            elif kind == "failure":
                failure.append(row)
            elif kind == "result":
                footer = {key: value for key, value in row.items() if key != "type"}
            else:
                raise ValueError("Unexpected action trace event: " + str(kind))
    if footer != result or effective is None or steps != evidence or not steps:
        raise ValueError("Raw action result footer/effective source/fast evidence coverage mismatch")
    expected = {name: declaration["source_sha256"][PREFIX + name] for name in ("fast_geometry.py", "fast_control.py")}
    if (effective["source_sha256"] != expected
            or effective["cache_sha256"] != declaration["input_files"]["path_cache"]["sha256"]):
        raise ValueError("Actual fast controller/geometry/cache differs from frozen source")
    utc_start = datetime.fromisoformat(effective["reset_started_utc"])
    if utc_start.tzinfo is None:
        raise ValueError("Missing explicit pre-physics reset UTC")
    gap = result["physical_steps"] - steps[-1]
    if gap not in (0, 1) or (gap and not failure) or (failure and result["valid_physics"]):
        raise ValueError("Unexplained terminal raw-state coverage gap")
    if abs(result["physical_time_s"] - result["physical_steps"] * DT) > 1e-12 or result["physical_steps"] > 300:
        raise ValueError("Action advanced beyond 300 original physical steps")
    return {"reset_started_utc": effective["reset_started_utc"], "raw_step_rows": len(steps),
            "unlogged_terminal_failure_steps": gap, "failure_events": failure,
            "maximum_joint_position_violation_rad": maximum_position_violation,
            "maximum_total_spilled_particles": maximum_total_spilled,
            "maximum_below_table_particles": maximum_below_table,
            "last_recorded_partitions": last_partitions,
            "footer_equals_original_returned_result": True}


def verify_post_trace(path, post, result):
    """Check the independent observation clock and exact deadline sample."""
    if path is None:
        if isinstance(post, dict) and post.get("error"):
            return {"raw_rows": 0, "partial_after_exception": True,
                    "exception_before_first_observation": True, "physical_seconds_logged": 0.0,
                    "deadline_received_ml": result["received_ml"] if abs(result["physical_time_s"] - DEADLINE) < 1e-10 else None,
                    "maximum_total_spilled_particles": 0, "all_logged_states_finite": None,
                    "all_logged_mass_arrays_unchanged": None}
        if post is not None:
            raise ValueError("Post summary exists without its raw observation trace")
        return None
    count = 0
    received_at_deadline = result["received_ml"] if abs(result["physical_time_s"] - DEADLINE) < 1e-10 else None
    maximum_change = 0.0
    maximum_spilled = 0
    minimum_spilled = minimum_source_spilled = None
    maximum_source_spilled = maximum_below_table = 0
    all_finite = all_mass_unchanged = True
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            finite(row)
            count += 1
            if (row.get("type") != "post_observation" or row["step"] != count
                    or abs(row["physical_time_s"] - result["physical_time_s"] - count * DT) > 1e-12):
                raise ValueError("Post trace does not use the original physical clock")
            m = row["measurement"]
            if (m["total"] != 5330 or m["initial_tube"] != 1198
                    or sum(m[k] for k in ("beaker", "tube", "airborne", "spilled")) != 5330
                    or sum(m[k] for k in ("tube_to_beaker", "tube_remaining", "tube_airborne", "tube_spilled")) != 1198
                    or row["solver_substeps"] < 4):
                raise ValueError("Post trace particle partitions or substeps changed")
            maximum_spilled = max(maximum_spilled, m["spilled"])
            minimum_spilled = m["spilled"] if minimum_spilled is None else min(minimum_spilled, m["spilled"])
            maximum_source_spilled = max(maximum_source_spilled, m["tube_spilled"])
            minimum_source_spilled = m["tube_spilled"] if minimum_source_spilled is None else min(minimum_source_spilled, m["tube_spilled"])
            maximum_below_table = max(maximum_below_table, m["below_table"])
            maximum_change = max(maximum_change, abs(m["received_ml"] - result["received_ml"]))
            all_finite = all_finite and row["all_finite"] is True
            all_mass_unchanged = all_mass_unchanged and row["mass_array_unchanged"] is True
            if abs(row["physical_time_s"] - DEADLINE) < 1e-10:
                received_at_deadline = m["received_ml"]
    failed = isinstance(post, dict) and bool(post.get("error"))
    expected_end = max(DEADLINE + POST_SECONDS, result["physical_time_s"] + POST_SECONDS)
    if not failed:
        if (not isinstance(post, dict) or count != round((expected_end - result["physical_time_s"]) / DT)
                or abs(post["duration_s"] - count * DT) > 1e-10
                or abs(post["end_physical_time_s"] - expected_end) > 1e-10
                or received_at_deadline != post["deadline_received_ml"]
                or abs(maximum_change - post["maximum_received_change_ml"]) > 1e-10
                or post["all_states_finite"] != all_finite or post["mass_array_unchanged"] != all_mass_unchanged):
            raise ValueError("Post summary differs from its raw elapsed time/deadline/dose observations")
    return {"raw_rows": count, "partial_after_exception": failed,
            "physical_seconds_logged": count * DT, "deadline_received_ml": received_at_deadline,
            "maximum_total_spilled_particles": maximum_spilled,
            "minimum_total_spilled_particles": minimum_spilled,
            "maximum_source_spilled_particles": maximum_source_spilled,
            "minimum_source_spilled_particles": minimum_source_spilled,
            "maximum_below_table_particles": maximum_below_table,
            "all_logged_states_finite": all_finite, "all_logged_mass_arrays_unchanged": all_mass_unchanged}


def assessment(result, controller, motion, post, trace_check, declaration, *, post_motion=None, post_check=None):
    envelope = declaration["robot_envelope"]
    post_motion = motion if post_motion is None else post_motion
    def motion_safe(values):
        return bool(numeric(values.get("maximum_joint_speed_limit_ratio")) <= 1.00001
            and numeric(values.get("maximum_joint_acceleration_rad_s2")) <= envelope["joint_acceleration_simulation_cap_rad_s2"] + 1e-4
            and numeric(values.get("maximum_hand_fk_position_error_m")) <= .003
            and numeric(values.get("maximum_hand_fk_orientation_error_rad")) <= .02)
    physical_safe = bool(result["valid_physics"] and result["mass_conserved"] and result["all_states_finite"]
        and result["source_partition_conserved"] and result["max_spilled_ml"] == 0
        and result["max_below_table_particles"] == 0
        and motion_safe(motion) and motion_safe(post_motion)
        and trace_check["maximum_joint_position_violation_rad"] <= 1e-6
        and trace_check["maximum_total_spilled_particles"] == trace_check["maximum_below_table_particles"] == 0
        and (post_check is None or (post_check["maximum_total_spilled_particles"] == 0
            and post_check["all_logged_states_finite"] and post_check["all_logged_mass_arrays_unchanged"]))
        and not (isinstance(post, dict) and post.get("error")))
    home = bool(controller.get("home_static") is True and numeric(controller.get("home_position_error_m")) <= .001
        and numeric(controller.get("home_orientation_error_rad")) <= math.radians(.1)
        and numeric(controller.get("home_joint_error_rad")) <= .01
        and abs(numeric(controller.get("u"))) <= 1e-7
        and controller.get("source_airborne_particles") == 0
        and numeric(controller.get("stable_frames"), -1) >= 12)
    on_time = bool(result["complete"] and result["physical_time_s"] <= DEADLINE + 1e-12
        and result["physical_steps"] <= 300 and controller.get("completed_by_deadline") is True
        and controller.get("timed_out") is False and controller.get("done") is True and home)
    deadline_sample = post.get("deadline_received_ml") if isinstance(post, dict) else None
    measured_at_deadline = type(deadline_sample) in (int, float) and math.isfinite(deadline_sample)
    late_ok = bool(isinstance(post, dict) and post.get("passed") is True and measured_at_deadline
        and numeric(post.get("duration_s"), -1) >= POST_SECONDS - 1e-10
        and numeric(post.get("end_physical_time_s"), -1) >= DEADLINE + POST_SECONDS - 1e-10
        and abs(numeric(post.get("start_physical_time_s")) - result["physical_time_s"]) < 1e-10
        and post.get("maximum_received_change_ml") == 0
        and post.get("all_states_finite") is True and post.get("mass_array_unchanged") is True)
    error = deadline_sample - result["target_ml"] if measured_at_deadline else result["error_ml"]
    return {"physical_safe": physical_safe, "home_stable": home, "completed_by_deadline": on_time,
            "late_drop_passed": late_ok, "constraint_feasible": physical_safe and on_time and late_ok,
            "dose_error_ml": error, "absolute_error_ml": abs(error), "deadline_dose_measurement_available": measured_at_deadline,
            "dose_measurement_basis": "physical_5s" if measured_at_deadline else "invalid_terminal_fallback_not_a_5s_measurement",
            "fallback_measurement_physical_time_s": result.get("measurement_physical_time_s") if not measured_at_deadline else None,
            "fallback_measurement_basis": result.get("measurement_basis") if not measured_at_deadline else None,
            "action_physical_seconds": result["physical_time_s"],
            "success_within_0_1_ml": bool(physical_safe and on_time and late_ok and abs(error) <= .1),
            "action_result_was_not_modified": True,
            "action_footer_success_alone_is_not_V3_release_success": True}


def group_metrics(records):
    if not records:
        raise ValueError("Empty group")
    n = len(records)
    errors = [r["assessment"]["absolute_error_ml"] for r in records]
    return {"episodes": n, "mae_ml": sum(errors) / n, "worst_absolute_error_ml": max(errors),
            "unsafe_count": sum(not r["assessment"]["physical_safe"] for r in records),
            "deadline_or_home_or_late_drop_failures": sum(not (r["assessment"]["completed_by_deadline"]
                and r["assessment"]["late_drop_passed"]) for r in records),
            "feasible_count": sum(r["assessment"]["constraint_feasible"] for r in records),
            "missing_deadline_measurements": sum(not r["assessment"]["deadline_dose_measurement_available"] for r in records),
            "within_0_1_ml_count": sum(r["assessment"]["success_within_0_1_ml"] for r in records),
            "within_0_2_ml_count": sum(r["assessment"]["constraint_feasible"] and e <= .2 for r, e in zip(records, errors)),
            "all_mass_conserved": all(r["result"]["mass_conserved"] for r in records),
            "mean_action_physical_seconds": sum(r["result"]["physical_time_s"] for r in records) / n,
            "physical_steps": sum(r["result"]["physical_steps"] for r in records),
            "action_physical_seconds": sum(r["result"]["physical_time_s"] for r in records),
            "post_observation_physical_seconds": sum(r["post_observation"].get("duration_s", 0)
                for r in records if isinstance(r["post_observation"], dict))}


def precision_violation(records):
    groups = [records] + [[r for r in records if r["conditions"]["target_ml"] == target] for target in TARGETS]
    return sum(max(0.0, group_metrics(group)["mae_ml"] - .1)
               + max(0.0, group_metrics(group)["worst_absolute_error_ml"] - .2) for group in groups if group)


def selection_key(records):
    m = group_metrics(records)
    return (m["unsafe_count"], m["deadline_or_home_or_late_drop_failures"], precision_violation(records),
            m["mae_ml"] + .5 * m["worst_absolute_error_ml"], m["mean_action_physical_seconds"])


def release_passed(records):
    groups = [records] + [[r for r in records if r["conditions"]["target_ml"] == target] for target in TARGETS]
    return bool(all(group for group in groups) and all(group_metrics(group)["unsafe_count"] == 0
        and group_metrics(group)["deadline_or_home_or_late_drop_failures"] == 0
        and group_metrics(group)["mae_ml"] <= .1 and group_metrics(group)["worst_absolute_error_ml"] <= .2
        and group_metrics(group)["all_mass_conserved"] for group in groups))


def preserve_uncommitted(output, paths):
    for path in paths:
        if path.is_file():
            if not path.resolve().is_relative_to(output):
                raise ValueError("Incomplete artifact escapes output directory")
            digest = sha(path)
            destination = output / "interrupted_artifacts" / (path.name + "." + str(time.time_ns()) + "." + digest[:12])
            destination.parent.mkdir(parents=True, exist_ok=True)
            path.replace(destination)
            append_json(output / "interrupted_artifacts.jsonl", {"preserved_at_utc": now(),
                "original_path": path.relative_to(output).as_posix(),
                "preserved_path": destination.relative_to(output).as_posix(), "sha256": digest,
                "reason": "uncommitted trace from interruption; retained before deterministic retry"})


def load_journal(output, pid, declaration, resume):
    records = {}
    path = output / "episodes.jsonl"
    if not path.exists():
        return records
    if not resume:
        raise FileExistsError("Journal already exists; use --resume")
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    good = []
    for index, line in enumerate(lines):
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            if index != len(lines) - 1 or line.endswith("\n"):
                raise ValueError("Corrupt committed journal line")
            archive = output / "interrupted_artifacts" / ("journal_tail." + str(time.time_ns()) + ".txt")
            atomic_text(archive, line)
            atomic_text(path, "".join(good))
            break
        name = record["episode_id"]
        if record["protocol_id"] != pid or name in records or record["conditions_sha256"] != signature(record["conditions"]):
            raise ValueError("Duplicate/foreign/damaged journal record")
        for key in ("record_path", "motion_record_path", "post_record_path"):
            relative = record.get(key)
            if relative is not None:
                artifact = (output / relative).resolve()
                if not artifact.is_relative_to(output) or sha(artifact) != record[key.replace("_path", "_sha256")]:
                    raise ValueError("Resumed raw/sidecar SHA differs: " + str(relative))
        trace_check = verify_action_trace(output / record["record_path"], record["result"], declaration)
        post_check = verify_post_trace(output / record["post_record_path"] if record["post_record_path"] else None,
                                       record["post_observation"], record["result"])
        if post_check != record["post_trace_check"]:
            raise ValueError("Resumed post observation trace differs")
        if trace_check != record["trace_check"] or read_json(output / record["motion_record_path"]) != record["motion_evidence"]:
            raise ValueError("Resumed trace/footer/motion sidecar differs")
        if assessment(record["result"], record["controller_summary"], record["motion_metrics"],
                      record["post_observation"], trace_check, declaration,
                      post_motion=record["motion_evidence"]["post_motion_metrics"], post_check=post_check) != record["assessment"]:
            raise ValueError("Resumed V3 assessment differs")
        records[name] = record
        good.append(line if line.endswith("\n") else line + "\n")
    if good and not lines[-1].endswith("\n") and len(good) == len(lines):
        atomic_text(path, "".join(good))
    return records


def run_fast(args, *, runtime=None):
    if not args.dry_run and not args.engineering_report:
        raise ValueError("Complete the engineering feasibility pilot and pass --engineering-report before starting")
    runtime = default_runtime(args) if runtime is None else runtime
    output = Path(args.output).resolve()
    declaration = make_declaration(args, runtime)
    if args.dry_run:
        return {"dry_run": True, "no_gpu_or_episodes": True, "proposed_declaration": declaration}
    iid = freeze_inputs(output, declaration, args.resume)
    if "initialize_solver" in runtime:
        runtime["versions"] = runtime["initialize_solver"]()
        require_unchanged(declaration)
    protocol = {"schema_version": 1, "kind": "formal_fast_speed_stop_CEM_138_live_episodes", "input_id": iid,
                **declaration, "software": {**runtime["versions"], "python": sys.version, "platform": platform.platform()},
                "action_footer_semantics": "FastTrainingEnv generates V3 action validity using its declared path cap; its returned footer is retained unchanged and assessment/motion/post evidence are separate"}
    pid = signature(protocol)
    protocol_path = output / "protocol.json"
    if protocol_path.is_file():
        previous = read_json(protocol_path)
        if not args.resume or previous["protocol_id"] != pid or previous["protocol"] != protocol:
            raise ValueError("Resume protocol/software differs")
    else:
        write_json(protocol_path, {"created_at_utc": now(), "protocol_id": pid, "protocol": protocol})
    records = load_journal(output, pid, declaration, args.resume)
    expected = {f"train_g{g:03d}_c{c:03d}_t{t}_s{s}" for g in range(GENERATIONS)
                for c in range(POPULATION) for t in TARGETS for s in TRAIN_SEEDS}
    expected |= {f"{stage}_{role}_t{t}_s{s}" for stage, seeds in (("validation", VALIDATION_SEEDS), ("test", TEST_SEEDS))
                 for role in ("candidate", "baseline") for t in TARGETS for s in seeds}
    if not set(records).issubset(expected):
        raise ValueError("Unexpected journal episode ID")
    search_config = declaration["cem"]
    search = LexCEM(search_config["parameter_names"], search_config["bounds"], search_config["initial_params"],
                    seed=search_config["seed"], initial_std_fraction=search_config["initial_std_fraction"])
    env = None
    started, new_episodes = time.perf_counter(), 0

    def persist_episode_csv():
        rows = [{"episode_id": r["episode_id"], "stage": r["conditions"]["stage"],
                 "policy_role": r["conditions"]["policy_role"], "target_ml": r["conditions"]["target_ml"],
                 "seed": r["conditions"]["seed"], "dose_error_at_deadline_ml": r["assessment"]["dose_error_ml"],
                 "deadline_measurement_available": r["assessment"]["deadline_dose_measurement_available"],
                 "raw_action_footer_error_ml": r["result"]["error_ml"],
                 "action_physical_seconds": r["result"]["physical_time_s"],
                 "constraint_feasible": r["assessment"]["constraint_feasible"],
                 "physical_safe": r["assessment"]["physical_safe"],
                 "completed_by_deadline": r["assessment"]["completed_by_deadline"],
                 "late_drop_passed": r["assessment"]["late_drop_passed"]} for r in records.values()]
        if rows:
            write_csv(output / "episodes.csv", rows, tuple(rows[0]))

    def episode(name, stage, role, target, seed, params, generation=None, candidate=None):
        nonlocal env, new_episodes
        params = normalized(runtime["policy_factory"](params).params)
        conditions = {"stage": stage, "policy_role": role, "target_ml": float(target), "seed": seed,
                      "policy_params": params, "randomized": True, "generation": generation, "candidate": candidate}
        require_unchanged(declaration)
        if name in records:
            if records[name]["conditions"] != conditions:
                raise ValueError("Resumed sampled parameters/conditions differ: " + name)
            return records[name]
        action_path = output / "traces" / (name + ".steps.jsonl.gz")
        post_path = output / "traces" / (name + ".post.jsonl.gz")
        motion_path = output / "motion" / (name + ".json")
        preserve_uncommitted(output, (action_path, post_path, motion_path))
        if env is None:
            env = runtime["environment_factory"]()
        def progress(event):
            write_json(output / "progress.json", {"state": "running", "updated_at_utc": now(), "protocol_id": pid,
                "episode_id": name, "conditions": conditions, "completed_episodes": len(records),
                "total_planned_episodes": TOTAL_EPISODES, **normalized(event)})
        print(f"FAST EPISODE {len(records)+1}/{TOTAL_EPISODES}: {name}", flush=True)
        episode_wall_start = time.perf_counter()
        result = normalized(env.run_episode(target, seed, policy=runtime["policy_factory"](params), randomize=True,
                                            record_path=action_path, progress=progress))
        finite(result)
        controller = normalized(copy.deepcopy(env.last_controller.status))
        motion = normalized(copy.deepcopy(env.fast_metrics))
        action_terminal_kinematics = normalized(copy.deepcopy(getattr(env, "_current_kinematics", {})))
        trace_check = verify_action_trace(action_path, result, declaration)
        if (result["target_ml"], result["seed"]) != (target, seed):
            raise ValueError("Live returned different episode conditions")
        post = None
        if result["valid_physics"]:
            try:
                post = normalized(env.post_observe(post_path, seconds=POST_SECONDS))
            except (RuntimeError, ValueError, FloatingPointError) as error:
                post = {"passed": False, "error": str(error), "error_type": type(error).__name__,
                        "purpose": "failed_late_drop_observation_retained_without_rewriting_action_result"}
        post_check = verify_post_trace(post_path if post_path.is_file() else None, post, result)
        evidence = {"episode_id": name, "protocol_id": pid, "action_motion_metrics": motion,
                    "post_motion_metrics": normalized(copy.deepcopy(env.fast_metrics)), "controller_at_action_end": controller,
                    "action_terminal_kinematics": action_terminal_kinematics,
                    "post_terminal_kinematics": normalized(copy.deepcopy(getattr(env, "_current_kinematics", {}))),
                    "action_exception": result.get("error"),
                    "post_exception": post.get("error") if isinstance(post, dict) else None,
                    "post_observation": post, "trace_check": trace_check, "post_trace_check": post_check,
                    "action_result_modified": False}
        write_json(motion_path, evidence)
        record = {"episode_id": name, "protocol_id": pid, "conditions": conditions,
                  "conditions_sha256": signature(conditions), "completed_at_utc": now(),
                  "episode_wall_clock_start_utc": trace_check["reset_started_utc"],
                  "record_path": action_path.relative_to(output).as_posix(), "record_sha256": sha(action_path),
                  "motion_record_path": motion_path.relative_to(output).as_posix(), "motion_record_sha256": sha(motion_path),
                  "post_record_path": post_path.relative_to(output).as_posix() if post_path.is_file() else None,
                  "post_record_sha256": sha(post_path) if post_path.is_file() else None,
                  "result": result, "controller_summary": controller, "motion_metrics": motion, "motion_evidence": evidence,
                  "post_observation": post, "trace_check": trace_check,
                  "post_trace_check": post_check,
                  "assessment": assessment(result, controller, motion, post, trace_check, declaration,
                                           post_motion=evidence["post_motion_metrics"], post_check=post_check),
                  "total_action_plus_post_wall_seconds": time.perf_counter() - episode_wall_start}
        append_json(output / "episodes.jsonl", record)
        records[name] = record
        new_episodes += 1
        persist_episode_csv()
        progress({"event": "episode_recorded", "assessment": record["assessment"]})
        return record

    try:
        generations, training_records, curve = [], [], []
        for generation in range(GENERATIONS):
            params = [normalized(runtime["policy_factory"](candidate).params) for candidate in search.ask()]
            keys, groups = [], []
            for index, candidate in enumerate(params):
                group = [episode(f"train_g{generation:03d}_c{index:03d}_t{target}_s{seed}", "train", "candidate",
                                 target, seed, candidate, generation, index) for target in TARGETS for seed in TRAIN_SEEDS]
                groups.append(group)
                keys.append(selection_key(group))
                training_records.extend(group)
            update = search.tell(params, keys)
            entry = {"generation": generation, "candidates": [{"index": index, "params": candidate,
                "selection_key": list(key), "metrics": group_metrics(group),
                "episode_ids": [record["episode_id"] for record in group]}
                for index, (candidate, key, group) in enumerate(zip(params, keys, groups))],
                "completed_training_episodes": len(training_records), **update}
            generations.append(entry)
            write_json(output / f"generation_{generation:03d}.json", {"protocol_id": pid, **entry})
            write_json(output / "search_state.json", {"protocol_id": pid, "generations_completed": generation + 1, **update})
            curve.append({"generation": generation, "completed_training_episodes": len(training_records),
                **{"mean_" + name: sum(key[index] for key in keys) / len(keys)
                   for index, name in enumerate(declaration["selection_tuple"])},
                **{"best_" + name: search.best_key[index] for index, name in enumerate(declaration["selection_tuple"])}})
            write_csv(output / "training_curve.csv", curve, tuple(curve[0]))
        candidate_params = dict(search.best_params)
        baseline_params = dict(search_config["baseline_params"])
        validation = {role: [episode(f"validation_{role}_t{target}_s{seed}", "validation", role, target, seed, params)
            for seed in VALIDATION_SEEDS for target in TARGETS]
            for role, params in (("candidate", candidate_params), ("baseline", baseline_params))}
        scores = {role: selection_key(group) for role, group in validation.items()}
        changed = candidate_params != baseline_params
        recommended_role = "candidate" if changed and scores["candidate"] < scores["baseline"] else "baseline"
        recommendation_params = candidate_params if recommended_role == "candidate" else baseline_params
        validation_report = {"protocol_id": pid, "selection_keys": {role: list(key) for role, key in scores.items()},
            "metrics": {role: group_metrics(group) for role, group in validation.items()},
            "episode_ids": {role: [record["episode_id"] for record in group] for role, group in validation.items()},
            "recommended_role": recommended_role, "test_used_for_selection": False}
        write_json(output / "validation.json", validation_report)
        freeze_path = output / "freeze.json"
        if freeze_path.is_file():
            freeze = read_json(freeze_path)
            if (not args.resume or freeze["protocol_id"] != pid or freeze["recommended_role"] != recommended_role
                    or freeze["candidate_params"] != candidate_params or freeze["recommended_params"] != recommendation_params):
                raise ValueError("Pre-test frozen recommendation differs")
        else:
            if any(record["conditions"]["stage"] == "test" for record in records.values()):
                raise ValueError("Holdout records exist without a pre-test model freeze")
            if any((output / name).exists() for name in ("candidate_frozen.json", "recommended_frozen.json")):
                raise ValueError("Partial two-model freeze; preserve and inspect before resuming")
            frozen_at = now()
            metadata = {"protocol_id": pid, "frozen_at_utc": frozen_at, "algorithm": ALGORITHM,
                        "neural_policy": False, "ppo": False, "training_episodes": 108,
                        "parameters_changed_from_initial_baseline": changed, "test_used_for_selection": False,
                        "source_sha256": declaration["source_sha256"], "input_id": iid,
                        "validation_selection_keys": {role: list(key) for role, key in scores.items()},
                        "automatic_gui_deployment": False}
            for role, params in (("candidate", candidate_params), ("recommended", recommendation_params)):
                runtime["policy_factory"](params).save(output / (role + "_frozen.json"),
                    metadata={**metadata, "role": role, "recommended_role": recommended_role,
                              "parameter_optimization_changed_model": changed if role == "candidate" else recommended_role == "candidate"})
            freeze = {"protocol_id": pid, "frozen_at_utc": frozen_at, "recommended_role": recommended_role,
                      "candidate_params": candidate_params, "recommended_params": recommendation_params,
                      "candidate_sha256": sha(output / "candidate_frozen.json"),
                      "recommended_sha256": sha(output / "recommended_frozen.json"),
                      "test_used_for_selection": False, "automatic_gui_deployment": False}
            write_json(freeze_path, freeze)
            atomic_text(output / "models_frozen.sha256", "".join(freeze[role + "_sha256"] + "  " + role + "_frozen.json\n"
                                                                  for role in ("candidate", "recommended")))
        for role, params in (("candidate", candidate_params), ("recommended", recommendation_params)):
            path = output / (role + "_frozen.json")
            if sha(path) != freeze[role + "_sha256"] or read_json(path)["parameters"] != params:
                raise ValueError("Frozen model bytes/parameters changed before holdout")
        pairs = []
        test_records = {"candidate": [], "baseline": []}
        for seed in TEST_SEEDS:
            for target in TARGETS:
                pair = {"target_ml": target, "seed": seed, "recommended_result_role": recommended_role}
                for role, params in (("candidate", candidate_params), ("baseline", baseline_params)):
                    record = episode(f"test_{role}_t{target}_s{seed}", "test", role, target, seed, params)
                    if datetime.fromisoformat(record["episode_wall_clock_start_utc"]) <= datetime.fromisoformat(freeze["frozen_at_utc"]):
                        raise ValueError("Holdout reset preceded the two-model freeze")
                    pair[role] = record
                    test_records[role].append(record)
                pairs.append(pair)
        require_unchanged(declaration)
        if set(records) != expected or len(records) != TOTAL_EPISODES:
            raise ValueError("Final journal does not contain the complete 138 declared episodes")
        for role in ("candidate", "recommended"):
            if sha(output / (role + "_frozen.json")) != freeze[role + "_sha256"]:
                raise ValueError("Frozen model changed during holdout")
        summary = {"schema_version": 1, "complete": True, "completed_at_utc": now(), "protocol_id": pid,
            "protocol": protocol, "training": {"generations": generations, "metrics": group_metrics(training_records),
                "best_key": list(search.best_key), "best_params": candidate_params,
                "parameters_changed_from_initial_baseline": changed,
                "distinct_parameter_vectors": len({signature(row["params"]) for gen in generations for row in gen["candidates"]})},
            "validation": validation_report,
            "models": {role: {"file": role + "_frozen.json", "sha256": freeze[role + "_sha256"], "parameters": params}
                       for role, params in (("candidate", candidate_params), ("recommended", recommendation_params))},
            "recommendation": {"role": recommended_role, "selected_on_validation_only": True, "additional_test_rollouts": 0},
            "test": {"paired_results": pairs, "metrics": {role: group_metrics(group) for role, group in test_records.items()},
                     "distinct_target_seed_conditions": 9, "executed_live_episodes": 18,
                     "recommended_metrics": group_metrics(test_records[recommended_role])},
            "release_acceptance": {role + "_passed": release_passed(group) for role, group in
                (*test_records.items(), ("recommended", test_records[recommended_role]))},
            "all_rollouts": group_metrics(list(records.values())), "total_recorded_episodes": len(records),
            "new_episodes_this_invocation": new_episodes, "wall_seconds_this_invocation": time.perf_counter() - started,
            "total_action_plus_post_wall_seconds": sum(record["total_action_plus_post_wall_seconds"] for record in records.values()),
            "all_episode_ids": list(records), "test_used_for_training_or_selection": False,
            "raw_action_result_footers_modified": False, "source_sha256_unchanged_after_run": True,
            "gui_checkpoint_sha256_after": gui_hashes(), "automatic_gui_deployment": False,
            "limitations": ["CEM tunes speed/stop feedback on a fixed kinematic robot path, not neural or joint-torque learning.",
                "The baseline is the initial V3 five-second controller; historical V2 accuracy is not a paired baseline here.",
                "All failures remain in raw action/post logs and every aggregate; post observation cannot rescue a five-second miss.",
                "Volumes are privileged simulation particle truth, quantum 0.042875mL, not hardware/vision accuracy.",
                "Joint acceleration uses a declared simulation envelope; friction grasp, torque feasibility and full collision safety are not established by kinematic FK alone."]}
        summary["release_acceptance"].update(does_not_reselect_model=True, automatic_gui_deployment=False)
        write_json(output / "summary.json", summary)
        write_json(output / "progress.json", {"state": "complete", "updated_at_utc": now(), "protocol_id": pid,
            "completed_episodes": len(records), "recommended_role": recommended_role,
            "recommended_release_passed": summary["release_acceptance"]["recommended_passed"]})
        return summary
    except BaseException as error:
        write_json(output / "progress.json", {"state": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
            "updated_at_utc": now(), "protocol_id": pid, "completed_episodes": len(records),
            "error_type": type(error).__name__, "error": str(error)})
        raise


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=ROOT / "droplet_out/pour_policy_fast_v3/run_20261122_formal")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="CPU-only declaration preview; no run artifacts or episodes")
    p.add_argument("--graph", action="store_true", help="Rejected until a separate V3 graph-equivalence protocol exists")
    p.add_argument("--search-config", type=Path, help="Pilot-derived bounds/initial_params/initial_std_fraction JSON")
    p.add_argument("--engineering-report", type=Path, help="Required for actual start: completed feasibility pilot results.json")
    p.add_argument("--allow-engineering-spill", action="store_true",
                   help="Explicitly admit existing pilot spill as an optimization defect; never relaxes formal CEM/release criteria")
    home = ROOT / "_cache/pour_training/fast_v3"
    p.add_argument("--path-cache", type=Path, default=home / "fast_path_smooth_cache.npz")
    p.add_argument("--path-manifest", type=Path, default=home / "fast_path_smooth_cache.json")
    p.add_argument("--prototype-cache", type=Path, default=home / "fast_path_cache.npz")
    p.add_argument("--prototype-manifest", type=Path, default=home / "fast_path_cache.json")
    p.add_argument("--path-report", type=Path, default=home / "fast_path_cpu_report.json")
    p.add_argument("--joint-bank", type=Path, default=home / "legacy_kinematic_bank.npz")
    p.add_argument("--bank-manifest", type=Path, default=home / "legacy_kinematic_bank.json")
    sdk = require_isaac_sim_root()
    p.add_argument("--urdf", type=Path, default=sdk / "exts/isaacsim.asset.importer.urdf/data/urdf/robots/franka_description/robots/panda_arm_hand.urdf")
    return p


def main(argv=None):
    argument_parser = parser()
    args = argument_parser.parse_args(argv)
    if args.dry_run:
        print(json.dumps(run_fast(args), indent=2, allow_nan=False))
        return 0
    if not args.engineering_report:
        argument_parser.error("--engineering-report is required before GPU setup or creation of a formal run")
    output = Path(args.output).resolve()
    if output == ROOT or output.is_relative_to(ROOT / "assets"):
        argument_parser.error("Use a dedicated run output directory outside GUI assets")
    output.mkdir(parents=True, exist_ok=True)
    lock = output / "fast_runner.lock"
    with lock.open("x", encoding="utf-8") as stream:
        json.dump({"pid": os.getpid(), "started_at_utc": now()}, stream)
    try:
        summary = run_fast(args)
        print("FAST FORMAL COMPLETE " + json.dumps({"episodes": summary["total_recorded_episodes"],
            "recommended_role": summary["recommendation"]["role"], "release": summary["release_acceptance"]}), flush=True)
        return 0
    finally:
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
