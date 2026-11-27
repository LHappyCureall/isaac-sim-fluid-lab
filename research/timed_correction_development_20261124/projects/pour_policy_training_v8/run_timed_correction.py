"""10-second closed-loop top-up. Escalate the same policy to 12 seconds if time runs out.

The coarse pour is the known under-pouring lift point. After the stream settles,
one or two slow tilts add the measured shortfall. Validation selects against the
same coarse pour with top-up disabled. Test seeds stay unused until both policies
are frozen. Archived five-second logs are not rewritten.
"""

import importlib.util
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "droplet_out" / "pour_policy_timed_correction" / "run_20261123"
CACHE = ROOT / "projects" / "pour_policy_training_v3" / "mouth_pivot_lift025_z022_cache.npz"
CACHE_SHA = "106706110581ef85ac87137aafaf5f513491b37f9f6ea16c2f427f7685c06a1f"
COARSE = {"prepare_fraction": 1.0, "dose_tilt_deg_s": 25.6, "return_fraction": 1.0,
          "tail_lead_s": 0.23, "stop_bias_ml": 10.0, "max_dose_u_accel": 6.0}
PILOT_SEEDS = (111101, 111102)
VAL_SEEDS = (112101, 112102)
TEST_SEEDS = (113101, 113102, 113103)
TARGETS = (20, 30, 40)


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def summarize(rows, deadline):
    errors = [row["result"]["error_ml"] for row in rows]
    absolute = [abs(value) for value in errors]
    return {
        "n": len(rows),
        "mean_absolute_error_ml": sum(absolute) / len(rows),
        "worst_absolute_error_ml": max(absolute),
        "within_0_2_ml": sum(value <= 0.2 for value in absolute),
        "spill_cases": sum(row["spill_particles"] > 0 or row["result"]["max_spilled_ml"] > 0 for row in rows),
        "home_cases": sum(row["full_home"] and row["result"]["physical_time_s"] <= deadline + 1e-9 for row in rows),
        "mean_time_s": sum(row["result"]["physical_time_s"] for row in rows) / len(rows),
        "signed_errors_ml": errors,
    }


def succeeded(rows, deadline):
    stats = summarize(rows, deadline)
    return stats["within_0_2_ml"] == stats["n"] and stats["spill_cases"] == 0 and stats["home_cases"] == stats["n"]


def needs_more_time(rows, deadline):
    for row in rows:
        result = row["result"]
        late = (not row["full_home"]) or result["physical_time_s"] > deadline - 0.05 or row["controller"].get("timed_out")
        short = result["error_ml"] < -0.2
        if late and short:
            return True
        if row["controller"].get("timed_out") and abs(result["error_ml"]) > 0.2:
            return True
    return False


def main():
    shared = load_module("correction_sdk", ROOT / "train_target_volume_policy.py")
    versions = shared.configure_sdk()
    control = load_module("correction_control", ROOT / "projects/pour_policy_training_v8/timed_correction_control.py")
    geometry_module = load_module("correction_geometry", ROOT / "assets/robot_testtube_pour/scripts/mouth_pivot_lift_geometry.py")
    collector = load_module("correction_env", ROOT / "assets/robot_testtube_pour/scripts/direct_training_env.py")
    geometry = geometry_module.load_geometry(CACHE, CACHE_SHA)
    OUT.mkdir(parents=True, exist_ok=True)
    journal = OUT / "episodes.jsonl"
    done = {}
    if journal.exists():
        for line in journal.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                done[row["episode_id"]] = row
    env = collector.DirectTrainingEnv(geometry=geometry, control=control)

    def run_block(stage, role, seeds, spec):
        env.max_seconds = float(spec["deadline_s"]) + 1 / 60
        rows = []
        for seed in seeds:
            for target in TARGETS:
                episode_id = f"{stage}_{role}_d{int(spec['deadline_s'])}_t{target}_s{seed}"
                if episode_id in done:
                    rows.append(done[episode_id])
                    continue
                policy = control.CorrectionPolicy(
                    COARSE if spec.get("coarse") is None else spec["coarse"],
                    deadline_s=spec["deadline_s"], topup_tilt_deg_s=spec["topup_tilt_deg_s"],
                    topup_lead_s=spec["topup_lead_s"], enable_topup=spec["enable_topup"])
                print(f"CORR {episode_id} deadline={spec['deadline_s']} topup={spec['topup_tilt_deg_s']} "
                      f"enabled={spec['enable_topup']}", flush=True)
                started = time.perf_counter()
                result = env.run_episode(target, seed, policy=policy, randomize=True,
                                         record_path=OUT / "traces" / f"{episode_id}.steps.jsonl.gz")
                post = env.post_observe(OUT / "traces" / f"{episode_id}.post.jsonl.gz", seconds=2.0) if result["valid_physics"] else None
                status = dict(env.last_controller.status)
                metrics = dict(env._last_liquid_metrics)
                full_home = bool(status.get("done") and status.get("home_static") and status.get("completed_by_deadline")
                                 and status.get("source_airborne_particles") == 0 and not status.get("timed_out"))
                row = {"episode_id": episode_id, "stage": stage, "role": role, "target_ml": target, "seed": seed,
                       "spec": spec, "result": result, "controller": status,
                       "spill_particles": int(metrics["spilled"]), "full_home": full_home,
                       "post_observation": post, "wall_seconds": time.perf_counter() - started,
                       "recorded_at_utc": now()}
                with journal.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(row, allow_nan=False) + "\n")
                done[episode_id] = row
                print(json.dumps({"id": episode_id, "received_ml": result["received_ml"], "error_ml": result["error_ml"],
                                  "time_s": result["physical_time_s"], "spill": row["spill_particles"],
                                  "home": full_home, "topups": status.get("correction_count"),
                                  "phase": status.get("phase")}, allow_nan=False), flush=True)
                rows.append(row)
        return rows

    deadline = 10.0
    spec = {"deadline_s": deadline, "topup_tilt_deg_s": 2.5, "topup_lead_s": 0.15, "enable_topup": True, "coarse": dict(COARSE)}
    pilot = run_block("pilot", "candidate", PILOT_SEEDS, spec)
    note = "First 10 s slow top-up."
    if not succeeded(pilot, deadline):
        errors_20 = [row["result"]["error_ml"] for row in pilot if row["target_ml"] == 20]
        topped = any(row["controller"].get("correction_count", 0) > 0 for row in pilot if row["target_ml"] == 20)
        if sum(errors_20) / len(errors_20) > 0.2 and topped:
            spec = {**spec, "topup_tilt_deg_s": 1.5, "topup_lead_s": 0.2}
            note = "The top-up released a gulp, so the next pilot tilts more slowly."
            pilot = run_block("pilot_slow", "candidate", PILOT_SEEDS, spec)
        elif sum(errors_20) / len(errors_20) > 0.2:
            spec = {"deadline_s": 10.0, "topup_tilt_deg_s": 8.0, "topup_lead_s": 0.08, "enable_topup": True,
                    "coarse": {**COARSE, "dose_tilt_deg_s": 24.8, "tail_lead_s": 0.27}}
            note = "20 mL coarse pour was already over, so the next pilot stops earlier. Deadline stays 10 s."
            pilot = run_block("pilot2", "candidate", PILOT_SEEDS, spec)
        elif needs_more_time(pilot, deadline) or summarize(pilot, deadline)["within_0_2_ml"] < 6:
            short = any(row["result"]["error_ml"] < -0.2 for row in pilot)
            if short and not needs_more_time(pilot, deadline):
                spec = {**spec, "topup_tilt_deg_s": 4.0, "topup_lead_s": 0.1}
                note = "10 s finished early while still short, so the top-up is a little faster but still slow."
                pilot = run_block("pilot2", "candidate", PILOT_SEEDS, spec)
            if not succeeded(pilot, spec["deadline_s"]):
                spec = {**spec, "deadline_s": 12.0}
                note = "10 s did not meet ±0.2 mL with a full home, so the same correction runs to 12 s."
                pilot = run_block("pilot12", "candidate", PILOT_SEEDS, spec)
    if not succeeded(pilot, spec["deadline_s"]) and spec["deadline_s"] < 12.0:
        spec = {**spec, "deadline_s": 12.0}
        note = "10 s did not meet ±0.2 mL, so the same correction runs to 12 s."
        pilot = run_block("pilot12", "candidate", PILOT_SEEDS, spec)
    chosen = dict(spec)
    deadline = chosen["deadline_s"]
    decision = {"recorded_at_utc": now(), "versions": versions, "note": note, "chosen": chosen,
                "pilot_summary": summarize(pilot, deadline), "pilot_succeeded": succeeded(pilot, deadline)}
    write_json(OUT / "pilot_decision.json", decision)
    print("PILOT DECISION " + json.dumps(decision, ensure_ascii=False), flush=True)
    baseline = {**chosen, "enable_topup": False, "topup_tilt_deg_s": 0.0}
    validation = {"candidate": run_block("validation", "candidate", VAL_SEEDS, chosen),
                  "baseline": run_block("validation", "baseline", VAL_SEEDS, baseline)}
    val_scores = {role: summarize(rows, deadline) for role, rows in validation.items()}
    def rank_key(stats):
        return (stats["n"] - stats["home_cases"], stats["spill_cases"], stats["mean_absolute_error_ml"])

    recommended = "candidate" if rank_key(val_scores["candidate"]) <= rank_key(val_scores["baseline"]) else "baseline"
    frozen = chosen if recommended == "candidate" else baseline
    write_json(OUT / "candidate_frozen.json", {"policy_kind": control.POLICY_KIND, "parameters": chosen, "role": "candidate"})
    write_json(OUT / "recommended_frozen.json", {"policy_kind": control.POLICY_KIND, "parameters": frozen,
                                                "role": recommended, "test_used_for_selection": False,
                                                "automatic_gui_deployment": False})
    tests = {"candidate": run_block("test", "candidate", TEST_SEEDS, chosen),
             "baseline": run_block("test", "baseline", TEST_SEEDS, baseline)}
    summary = {"complete": True, "recorded_at_utc": now(), "versions": versions, "deadline_s": deadline,
               "pilot_decision": decision, "chosen": chosen, "recommended_role": recommended,
               "validation": val_scores,
               "test": {role: summarize(rows, deadline) for role, rows in tests.items()},
               "test_used_for_selection": False, "automatic_gui_deployment": False}
    write_json(OUT / "summary.json", summary)
    print("CORR SUMMARY " + json.dumps(summary["test"], allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
