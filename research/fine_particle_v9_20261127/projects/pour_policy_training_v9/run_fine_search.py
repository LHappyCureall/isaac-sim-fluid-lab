"""42 live episodes: coarse-speed CEM, paired validation and frozen tests.

The user selected 24 raw physical seconds, 0.1 percent dose error and at most
one spilled particle. No old journal, test seed or GUI default is overwritten.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.util
import gzip
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = ROOT / "droplet_out/pour_policy_fine_v9/run_20261127"
ASSETS = ROOT / "assets/robot_testtube_pour/scripts"
TRAIN_SEEDS, VAL_SEEDS, TEST_SEEDS = (171101,), (172101,), (173101, 173102, 173103)
TARGETS = (20, 30, 40)
SPACING, DEADLINE, SEARCH_SEED = .0027, 24.0, 2026112709
COARSE = {"prepare_fraction": 1., "dose_tilt_deg_s": 25.6, "return_fraction": 1.,
          "tail_lead_s": .26, "stop_bias_ml": 10., "max_dose_u_accel": 6.}
SOURCE_NAMES = ("train_target_volume_policy.py", *["assets/robot_testtube_pour/scripts/" + n for n in (
    "geometry.py", "liquid.py", "target_control.py", "robot.py", "pour_training_env.py",
    "fast_geometry.py", "fast_control.py", "fast_training_env.py", "direct_fast_geometry.py",
    "direct_fast_control.py", "direct_training_env.py", "mouth_pivot_fast_geometry.py",
    "mouth_pivot_lift_geometry.py", "mouth_pivot_lift_control_v2.py")],
    *["projects/pour_policy_training_v9/" + n for n in
      ("legacy_control.py", "fine_control.py", "fine_env.py", "run_fine_search.py")])
FINE = LEGACY = None


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf8")
    os.replace(temp, path)


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class TrialPolicy:
    def __init__(self, role, multiplier):
        self.role, self.multiplier = role, float(multiplier)


def TargetPourController(geometry, target_ml, spacing, policy):
    module = LEGACY if policy.role == "baseline" else FINE
    coarse = dict(COARSE)
    coarse["dose_tilt_deg_s"] *= policy.multiplier
    calibrated = module.CorrectionPolicy(coarse, deadline_s=DEADLINE,
        topup_tilt_deg_s=2.2, topup_lead_s=.12, enable_topup=True, particle_trim=True)
    controller = module.TargetPourController(geometry, target_ml=target_ml, spacing=spacing, policy=calibrated)
    actual_file = Path(sys.modules[type(controller).__module__].__file__).resolve()
    if actual_file != Path(module.__file__).resolve():
        raise RuntimeError("Actual controller class differs from the declared role")
    return controller


def assessment(result, status, spill, post):
    target = result["target_ml"]
    measurement_valid = (result["valid_physics"] and result["mass_conserved"]
        and result["all_states_finite"] and result["source_partition_conserved"]
        and result["max_below_table_particles"] == 0
        and abs(result["measurement_physical_time_s"] - result["physical_time_s"]) < 1e-8
        and post is not None and post["all_states_finite"] and post["mass_array_unchanged"]
        and post.get("maximum_below_table_particles", 0) == 0
        and abs(post.get("start_physical_time_s", -1) - result["physical_time_s"]) < 1e-8
        and post.get("physical_steps", 0) >= 120
        and post.get("end_physical_time_s", -1) - post.get("start_physical_time_s", 0) >= 2.-1e-8)
    home = bool(result["complete"] and status.get("done") and status.get("home_static")
        and status.get("stable_frames", 0) >= 12 and status.get("source_airborne_particles") == 0
        and not status.get("timed_out") and result["physical_time_s"] <= DEADLINE)
    dose = abs(result["error_ml"]) <= .001 * target
    no_late = post is not None and post["maximum_received_change_ml"] == 0.0
    return {"valid_evidence": bool(measurement_valid), "dose_passed": dose,
        "full_home_within_deadline": home, "maximum_all_label_spilled_particles": spill,
        "spill_passed": spill <= 1, "no_late_dose_change": no_late,
        "tolerance_ml": .001 * target,
        "feasible": bool(measurement_valid and dose and home and spill <= 1 and no_late)}


def metrics(rows):
    errors = [abs(r["result"]["error_ml"]) for r in rows]
    return {"n": len(rows), "mae_ml": sum(errors)/len(errors), "worst_ml": max(errors),
        "feasible": sum(r["assessment"]["feasible"] for r in rows),
        "valid": sum(r["assessment"]["valid_evidence"] for r in rows),
        "dose_passed": sum(r["assessment"]["dose_passed"] for r in rows),
        "spill_particles": sum(r["assessment"]["maximum_all_label_spilled_particles"] for r in rows),
        "mean_physical_s": sum(r["result"]["physical_time_s"] for r in rows)/len(rows),
        "mean_wall_s": sum(r["wall_seconds"] for r in rows)/len(rows)}


def key(rows):
    m = metrics(rows)
    if m["valid"] != m["n"]:
        return (2, m["n"] - m["valid"], m["mae_ml"])
    if m["feasible"] == m["n"]:
        return (0, m["mean_physical_s"])
    return (1, m["mae_ml"], m["spill_particles"])


def validate_row(row):
    """Bind completed journals to sidecars and actual action evidence on resume."""
    episode_id = row["episode_id"]
    if any(c in episode_id for c in ("/", "\\", ":")):
        raise RuntimeError("Unsafe episode identity")
    if json.loads((OUT/"completed"/(episode_id+".json")).read_text()) != row:
        raise RuntimeError("Journal differs from completed sidecar: "+episode_id)
    action_name = f"traces/{episode_id}.steps.jsonl.gz"
    post_name = f"traces/{episode_id}.post.jsonl.gz"
    post_path = OUT/post_name
    post = row["post_observation"]
    post_valid = (isinstance(post, dict) and post.get("all_states_finite") is True
                  and post.get("mass_array_unchanged") is True and not post.get("error"))
    expected_artifacts = {action_name}
    if post is not None and post_path.is_file():
        expected_artifacts.add(post_name)
    elif post_valid:
        raise RuntimeError("Valid post-observation has no corresponding trace: "+episode_id)
    elif post is None and post_path.exists():
        raise RuntimeError("Unclaimed post-observation trace: "+episode_id)
    if not isinstance(row["artifacts"], dict) or set(row["artifacts"]) != expected_artifacts:
        raise RuntimeError("Completed action/post artifact set differs: "+episode_id)
    for name, expected in row["artifacts"].items():
        path = (OUT/name).resolve()
        if not path.is_relative_to(OUT.resolve()) or not path.is_file() or sha(path) != expected:
            raise RuntimeError("Completed evidence changed: "+name)
    trace = OUT/action_name
    header = footer = status = None
    action_spill_peak = 0
    with gzip.open(trace, "rt", encoding="utf8") as stream:
        for line in stream:
            item = json.loads(line)
            if item["type"] == "episode": header = item
            if item["type"] == "result": footer = {k:v for k,v in item.items() if k != "type"}
            if item["type"] == "direct_step_evidence":
                status = item["controller_status"]
                action_spill_peak = max(action_spill_peak,
                    int(item.get("maximum_all_label_spilled_particles", 0)),
                    int(item.get("partitions", {}).get("spilled", 0)))
    if footer != row["result"] or header is None or header["seed"] != row["seed"] or header["target_ml"] != row["target_ml"]:
        raise RuntimeError("Journal does not match original action header/footer")
    if row["result"]["complete"] and status != row["status"]:
        raise RuntimeError("Completed controller evidence changed")
    if post is not None:
        if not post_valid and (row["assessment"]["valid_evidence"] or row["assessment"]["feasible"]):
            raise RuntimeError("Failed post-observation cannot provide valid evidence")
        count = 0
        maximum_change = 0.0
        maximum_below = int(footer["max_below_table_particles"])
        maximum_spill = action_spill_peak
        dt = float(footer["physics_dt_s"])
        start = float(footer["physical_time_s"])
        received = float(footer["received_ml"])
        last_time = start
        if not all(math.isfinite(v) for v in (dt, start, received)) or abs(dt-1/60) > 1e-12:
            raise RuntimeError("Post-observation has an invalid physical clock")
        if post_path.is_file():
            try:
                with gzip.open(post_path, "rt", encoding="utf8") as stream:
                    for line in stream:
                        item = json.loads(line)
                        count += 1
                        if (item.get("type") != "post_observation"
                                or type(item.get("step")) is not int or item["step"] != count):
                            raise RuntimeError("Post-observation step sequence changed")
                        measurement = item["measurement"]
                        t = float(item["physical_time_s"])
                        measurement_time = float(measurement["physical_time_s"])
                        dose = float(measurement["received_ml"])
                        if (not all(math.isfinite(v) for v in (t, measurement_time, dose))
                                or abs(t-(start+count*dt)) > 1e-8
                                or abs(measurement_time-t) > 1e-8):
                            raise RuntimeError("Post-observation time or dose is invalid")
                        last_time = t
                        maximum_change = max(maximum_change, abs(dose-received))
                        maximum_below = max(maximum_below, int(measurement["below_table"]))
                        maximum_spill = max(maximum_spill, int(measurement["spilled"]))
            except (EOFError, OSError, UnicodeError, json.JSONDecodeError) as error:
                if post_valid:
                    raise RuntimeError("Valid post-observation trace is incomplete") from error
                # A failed post may end mid-write. Its bytes remain SHA-bound;
                # neither its prefix nor its metadata may rescue the action.
        if post_valid:
            metadata_values = (post.get("start_physical_time_s"), post.get("end_physical_time_s"),
                               post.get("duration_s"), post.get("maximum_received_change_ml"))
            if any(isinstance(v, bool) or not isinstance(v, (int, float))
                   or not math.isfinite(v) for v in metadata_values):
                raise RuntimeError("Valid post-observation metadata must be finite")
            if (type(post.get("physical_steps")) is not int
                    or post.get("physical_steps") != count or count < 120
                    or abs(post["start_physical_time_s"]-start) > 1e-8
                    or abs(post["end_physical_time_s"]-last_time) > 1e-8
                    or abs(post["duration_s"]-(last_time-start)) > 1e-8
                    or abs((last_time-start)-count*dt) > 1e-8
                    or last_time-start < 2.-1e-8
                    or abs(post["maximum_received_change_ml"]-maximum_change) > 1e-10
                    or post.get("maximum_below_table_particles") != maximum_below
                    or row["assessment"]["maximum_all_label_spilled_particles"] != maximum_spill):
                raise RuntimeError("Post-observation trace differs from its metadata")
    if row["assessment"] != assessment(row["result"], row["status"],
            row["assessment"]["maximum_all_label_spilled_particles"], row["post_observation"]):
        raise RuntimeError("Recorded task assessment changed")


def protocol(geometry):
    sources = {name: sha(ROOT / name) for name in SOURCE_NAMES}
    inputs = dict(geometry.INPUT_SHA256)
    inputs[Path(geometry.CACHE_FILE).relative_to(ROOT).as_posix()] = geometry.CACHE_SHA256
    for name in ("fast_path_smooth_cache.npz",):
        path = ROOT / "projects/pour_policy_training_v3" / name
        inputs[path.relative_to(ROOT).as_posix()] = sha(path)
    # All referenced geometry input identities, including the selected cache,
    # are checked before any SDK/GPU setup. Keep a local snapshot of each.
    for name, expected in inputs.items():
        if sha(ROOT/name) != expected:
            raise RuntimeError("Geometry input changed: " + name)
    history = []
    for group in (ROOT/"droplet_out").glob("pour_policy*/*/episodes.jsonl"):
        for line in group.read_text(encoding="utf8").splitlines():
            if line.strip():
                row = json.loads(line)
                seed = row.get("seed", row.get("conditions", {}).get("seed",row.get("result",{}).get("seed")))
                if seed in (*TRAIN_SEEDS, *VAL_SEEDS, *TEST_SEEDS) and group.parent != OUT:
                    history.append(str(group.relative_to(ROOT)))
    if history:
        raise RuntimeError("Proposed seed was already used: " + str(set(history)))
    points, labels = geometry.initial_water_points(SPACING)
    value = {"kind": "fine_particle_24s_coarse_speed_CEM_v9", "created_at_utc": now(),
        "user_criterion": {"deadline_s": 24., "relative_error": .001, "max_all_label_spill_particles": 1,
                           "full_original_home_and_12_quiet_frames": True},
        "budget": {"train": 18, "validation": 6, "test": 18, "total": 42},
        "search": {"generations": 2, "population": 3, "elites": 2, "seed": SEARCH_SEED,
            "parameter": "coarse_tilt_speed_multiplier", "bounds": [.92, 1.02],
            "center": 1., "initial_std": .025, "actual_deg_s": [25.6*.92, 25.6*1.02]},
        "train_seeds": TRAIN_SEEDS, "validation_seeds": VAL_SEEDS, "test_seeds": TEST_SEEDS,
        "targets_ml": TARGETS, "spacing_m": SPACING, "particle_ml": SPACING**3*1e6,
        "initial_particle_count": len(points), "initial_source_particles": int(sum(labels == 1)),
        "initial_source_ml": int(sum(labels == 1))*SPACING**3*1e6,
        "solver": "unchanged_live_PBF_60Hz_pressure6_adaptive_min4_graph_off",
        "fixed_engineering_changes": ["micro_retract_6deg_s", "settled_holds_at_least_0.2s",
                                      "main_topup_margin_0.15mL"],
        "baseline": "unchanged_V8_fine_trim_controller_at_multiplier1",
        "observation": "privileged_particle_volume_not_real_scale_or_torque",
        "selection": "train_selects_candidate_validation_selects_recommendation_test_accepts_only",
        "initial_evidence_gate": "first_three_training_cases_must_have_valid_physics_and_complete_evidence_not_dose_success",
        "source_sha256": sources, "input_sha256": inputs,
        "gui_sha256": {n: sha(ROOT/n) for n in (
            "assets/robot_testtube_pour/policy/precision_v2.json",
            "assets/robot_testtube_pour/policy/target_volume_policy.json",
            "assets/robot_testtube_pour/policy/target_volume_candidate.json")}}
    value["protocol_id"] = hashlib.sha256(json.dumps(value, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    return value


def main():
    global FINE, LEGACY
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    # Load SDK NumPy for the CPU-only geometry/preflight; do not initialize
    # Warp, CUDA or a Kit application until the protocol has been frozen.
    dependencies = list((ROOT/"extscache").glob("omni.kit.pip_archive-*/pip_prebundle"))
    dependencies += [ROOT/"exts/omni.pip.compute/pip_prebundle"]
    for path in dependencies:
        if path.is_dir(): sys.path.insert(0, str(path))
    geometry_module = load("fine_v9_geometry", ASSETS / "mouth_pivot_lift_geometry.py")
    geometry = geometry_module.load_geometry(geometry_module.DEFAULT_CACHE_FILE,
                                            geometry_module.DEFAULT_CACHE_SHA256)
    plan = protocol(geometry)
    if args.dry_run:
        print(json.dumps(plan, indent=2)); return
    OUT.mkdir(parents=True, exist_ok=True)
    lock = OUT/"worker.lock"
    with lock.open("x", encoding="utf8") as f:
        json.dump({"pid": os.getpid(), "started_utc": now(), "command": sys.argv}, f)
    journal = OUT/"episodes.jsonl"
    try:
        if (OUT/"protocol.json").exists():
            if not args.resume:
                raise RuntimeError("Existing protocol requires explicit, inspected --resume")
            old = json.loads((OUT/"protocol.json").read_text())
            for field in ("source_sha256", "input_sha256", "gui_sha256", "budget", "search", "user_criterion"):
                if old[field] != plan[field]: raise RuntimeError("Resume input changed: " + field)
            identity = hashlib.sha256(json.dumps({k:v for k,v in old.items() if k != "protocol_id"},
                sort_keys=True,separators=(",", ":"),allow_nan=False).encode()).hexdigest()
            if identity != old["protocol_id"]: raise RuntimeError("Frozen protocol identity changed")
            plan = old
        else:
            write_json(OUT/"protocol.json", plan)
            for name in plan["source_sha256"]:
                destination = OUT/"source_snapshot"/name
                destination.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(ROOT/name, destination)
            for name in plan["input_sha256"]:
                destination = OUT/"input_snapshot"/name
                destination.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(ROOT/name, destination)
        for group, folder in (("source_sha256", "source_snapshot"), ("input_sha256", "input_snapshot")):
            for name, expected in plan[group].items():
                if sha(OUT/folder/name) != expected:
                    raise RuntimeError("Frozen snapshot changed: " + name)
        done = {}
        if journal.exists():
            for line in journal.read_text().splitlines():
                row = json.loads(line)
                if row["episode_id"] in done: raise RuntimeError("Duplicate episode")
                validate_row(row)
                done[row["episode_id"]] = row
        write_json(OUT/"job_state.json", {"state": "starting", "pid": os.getpid(), "started_at_utc": now()})
        sdk = load("fine_v9_sdk", ROOT/"train_target_volume_policy.py")
        versions = sdk.configure_sdk()
        versions_path = OUT/"versions.json"
        if versions_path.exists():
            if json.loads(versions_path.read_text(encoding="utf8")) != versions:
                raise RuntimeError("SDK versions changed on resume")
        else:
            write_json(versions_path, versions)
        FINE, LEGACY = load("fine_v9_control", HERE/"fine_control.py"), load("fine_v9_legacy", HERE/"legacy_control.py")
        environment = load("fine_v9_environment", HERE/"fine_env.py")
        env = environment.FineEnv(geometry=geometry, control=sys.modules[__name__], spacing=SPACING)
        if len(env.clean_points) != plan["initial_particle_count"] or env.source_count != plan["initial_source_particles"]:
            raise RuntimeError("Initial particle identity count changed")

        def run(stage, role, seeds, multiplier, candidate=""):
            rows = []
            for seed in seeds:
                for target in TARGETS:
                    episode_id = f"{stage}_{role}_{candidate}_t{target}_s{seed}"
                    if episode_id in done:
                        row = done[episode_id]
                        expected = (stage,role,target,seed,multiplier)
                        if tuple(row[k] for k in ("stage","role","target_ml","seed","multiplier")) != expected:
                            raise RuntimeError("Completed condition or policy changed: "+episode_id)
                        rows.append(row); continue
                    for name, expected in plan["source_sha256"].items():
                        if sha(ROOT/name) != expected: raise RuntimeError("Active source changed: " + name)
                    trace = OUT/"traces"/(episode_id+".steps.jsonl.gz")
                    if trace.exists(): raise RuntimeError("Unjournaled attempt exists; inspect and recover without re-pouring: " + episode_id)
                    wall = time.perf_counter(); started = now()
                    if stage == "test":
                        frozen_models = json.loads((OUT/"freeze.json").read_text())
                        if datetime.fromisoformat(started) <= datetime.fromisoformat(frozen_models["frozen_at_utc"]):
                            raise RuntimeError("Test must start after model freeze")
                        for filename, expected in frozen_models["models_sha256"].items():
                            if sha(OUT/filename) != expected: raise RuntimeError("Frozen model changed")
                    write_json(OUT/"progress.json", {"state": "running", "pid": os.getpid(),
                        "completed": len(done), "budget": 42, "current_episode": episode_id,
                        "started_at_utc": started})
                    print("START " + episode_id, flush=True)
                    result = env.run_episode(target, seed, policy=TrialPolicy(role, multiplier),
                        randomize=True, record_path=trace)
                    status = dict(env.last_controller.status)
                    post_path = trace.with_name(episode_id+".post.jsonl.gz")
                    post = None
                    if result["valid_physics"]:
                        try:
                            post = env.post_observe(post_path)
                        except (RuntimeError, ValueError, FloatingPointError) as error:
                            post = {"error":repr(error), "all_states_finite":False,
                                "mass_array_unchanged":False, "maximum_received_change_ml":None,
                                "end_physical_time_s":float(env.fluid.time),
                                "measurement_basis":"partial_failed_post_observation_not_acceptance",
                                "last_available_measurement":dict(env._last_liquid_metrics or {})}
                    a = assessment(result, status, env.max_all_label_spilled_particles, post)
                    paths = [trace] + ([post_path] if post_path.is_file() else [])
                    row = {"episode_id": episode_id, "stage": stage, "role": role,
                        "target_ml": target, "seed": seed, "multiplier": multiplier,
                        "started_at_utc": started, "completed_at_utc": now(), "result": result,
                        "status": status, "assessment": a, "post_observation": post,
                        "motion_metrics": dict(env.fast_metrics), "wall_seconds": time.perf_counter()-wall,
                        "artifacts": {p.relative_to(OUT).as_posix(): sha(p) for p in paths}}
                    write_json(OUT/"completed"/(episode_id+".json"), row)
                    with journal.open("a", encoding="utf8") as f: f.write(json.dumps(row, allow_nan=False)+"\n")
                    done[episode_id] = row; rows.append(row)
                    print(json.dumps({"id": episode_id, "dose": result["received_ml"], "seconds": result["physical_time_s"], "assessment": a}), flush=True)
            return rows

        rng = random.Random(SEARCH_SEED)
        mean, std, best = 1., .025, None
        generations = []
        for generation in range(2):
            proposals = [mean]
            if best and best["multiplier"] != mean: proposals.append(best["multiplier"])
            while len(proposals) < 3: proposals.append(min(1.02, max(.92, rng.gauss(mean, std))))
            candidates = []
            for index, value in enumerate(proposals):
                rows = run("train", "candidate", TRAIN_SEEDS, value, f"g{generation}c{index}")
                candidate = {"generation": generation, "index": index, "multiplier": value,
                             "key": key(rows), "metrics": metrics(rows)}
                candidates.append(candidate)
                if generation == 0 and index == 0:
                    initial_gate = {"checked_at_utc":now(), "episodes":[r["episode_id"] for r in rows],
                        "passed":all(r["assessment"]["valid_evidence"] for r in rows),
                        "dose_is_not_a_gate_for_continuing_search":True}
                    write_json(OUT/"initial_evidence_gate.json", initial_gate)
                    if not initial_gate["passed"]:
                        raise RuntimeError("First three training cases failed the physical/evidence gate")
                if candidate["key"][0] < 2 and (best is None or candidate["key"] < best["key"]): best = candidate
            eligible = sorted((c for c in candidates if c["key"][0] < 2), key=lambda c: c["key"])[:2]
            if not eligible: raise RuntimeError("All candidate evidence invalid; stop instead of selecting invalid physics")
            elite_mean = sum(c["multiplier"] for c in eligible)/len(eligible)
            elite_std = math.sqrt(sum((c["multiplier"]-elite_mean)**2 for c in eligible)/len(eligible))
            mean, std = .65*mean+.35*elite_mean, max(.005, .65*std+.35*elite_std)
            generations.append({"generation": generation, "candidates": candidates, "next_mean": mean, "next_std": std})
            write_json(OUT/f"generation_{generation}.json", generations[-1])
        val_candidate = run("validation", "candidate", VAL_SEEDS, best["multiplier"])
        val_baseline = run("validation", "baseline", VAL_SEEDS, 1.)
        recommendation = "candidate" if key(val_candidate) <= key(val_baseline) else "baseline"
        model_definitions = {}
        for model_role, implementation in (("candidate", "candidate"), ("recommended", recommendation)):
            model = {"schema_version": 1, "role": model_role, "implementation": implementation,
                "policy_kind": "fine_particle_settled_micro_correction_v9" if implementation == "candidate" else LEGACY.POLICY_KIND,
                "coarse_speed_multiplier": best["multiplier"] if implementation == "candidate" else 1.,
                "coarse_base_parameters": COARSE, "deadline_s": DEADLINE, "spacing_m": SPACING,
                "particle_trim": True, "topup_tilt_deg_s":2.2, "topup_lead_s":.12,
                "fixed_main_topup_margin_ml": .15 if implementation == "candidate" else 0.,
                "fixed_micro_backoff_deg_s":6. if implementation == "candidate" else 2.,
                "source_sha256":plan["source_sha256"], "protocol_id":plan["protocol_id"],
                "test_used_for_selection":False, "gui_deployment":False}
            filename = model_role+"_frozen.json"
            if (OUT/filename).exists():
                if json.loads((OUT/filename).read_text()) != model: raise RuntimeError("Model changed on resume")
            else: write_json(OUT/filename, model)
            model_definitions[filename] = sha(OUT/filename)
        frozen = {"frozen_at_utc": now(), "protocol_id":plan["protocol_id"], "recommendation": recommendation,
            "candidate_multiplier": best["multiplier"], "baseline_multiplier": 1.,
            "source_sha256": plan["source_sha256"], "validation_only_selection": True,
            "test_used_for_selection": False, "gui_deployment": False, "models_sha256":model_definitions}
        if (OUT/"freeze.json").exists():
            previous = json.loads((OUT/"freeze.json").read_text())
            if any(previous[k] != frozen[k] for k in frozen if k != "frozen_at_utc"):
                raise RuntimeError("Frozen model changed on resume")
            frozen = previous
        else: write_json(OUT/"freeze.json", frozen)
        candidate_test = run("test", "candidate", TEST_SEEDS, best["multiplier"])
        baseline_test = run("test", "baseline", TEST_SEEDS, 1.)
        if len(done) != 42 or Counter(r["stage"] for r in done.values()) != {"train":18,"validation":6,"test":18}:
            raise RuntimeError("Completed budget differs from frozen protocol")
        summary = {"complete": True, "completed_at_utc": now(), "counts": {"train":18,"validation":6,"test":18},
            "versions": versions, "generations": generations, "freeze": frozen,
            "validation": {"candidate": metrics(val_candidate), "baseline": metrics(val_baseline)},
            "test": {"candidate": metrics(candidate_test), "baseline": metrics(baseline_test)},
            "recommended_role": recommendation,
            "task_passed": all(r["assessment"]["feasible"] for r in
                               (candidate_test if recommendation == "candidate" else baseline_test)),
            "gui_deployment": False}
        for n, h in plan["gui_sha256"].items():
            if sha(ROOT/n) != h: raise RuntimeError("GUI model changed during this run")
        for n, h in {**plan["source_sha256"], **plan["input_sha256"]}.items():
            if sha(ROOT/n) != h: raise RuntimeError("Input or source changed during this run: "+n)
        write_json(OUT/"summary.json", summary)
        write_json(OUT/"progress.json", {"state":"complete", "completed":42})
        write_json(OUT/"job_state.json", {"state":"complete", "pid":None, "completed_at_utc":now()})
    except BaseException as error:
        write_json(OUT/"job_state.json", {"state":"failed", "pid":None, "at_utc":now(),
            "error":repr(error)})
        raise
    finally:
        lock.unlink()


if __name__ == "__main__":
    main()
