"""Leave a measured reserve, then fill it inside 12 seconds.

The previous top-up could not remove liquid once 20 mL had already been passed.
Each attempt stops the coarse pour reserve_ml early, settles, and fills the
hole. Up to three reserves are tried on the same two pilot seeds. Validation
then compares the chosen reserve with and without the fill. Test seeds stay
unused until both policies are frozen.
"""

import importlib.util
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "droplet_out" / "pour_policy_timed_correction" / "run_20261123_reserve"
CACHE = ROOT / "projects" / "pour_policy_training_v3" / "mouth_pivot_lift025_z022_cache.npz"
CACHE_SHA = "106706110581ef85ac87137aafaf5f513491b37f9f6ea16c2f427f7685c06a1f"
COARSE = {"prepare_fraction": 1.0, "dose_tilt_deg_s": 25.6, "return_fraction": 1.0,
          "tail_lead_s": 0.23, "stop_bias_ml": 10.0, "max_dose_u_accel": 6.0}
DEADLINE = 12.0
PILOT_SEEDS = (114101, 114102)
VAL_SEEDS = (115101, 115102)
TEST_SEEDS = (116101, 116102, 116103)
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


def summarize(rows):
    errors = [row["result"]["error_ml"] for row in rows]
    absolute = [abs(value) for value in errors]
    return {
        "n": len(rows),
        "mean_absolute_error_ml": sum(absolute) / len(rows),
        "worst_absolute_error_ml": max(absolute),
        "within_0_2_ml": sum(value <= 0.2 for value in absolute),
        "spill_cases": sum(row["spill_particles"] > 0 or row["result"]["max_spilled_ml"] > 0 for row in rows),
        "home_cases": sum(row["full_home"] and row["result"]["physical_time_s"] <= DEADLINE + 1e-9 for row in rows),
        "mean_time_s": sum(row["result"]["physical_time_s"] for row in rows) / len(rows),
        "signed_errors_ml": errors,
    }


def succeeded(rows):
    stats = summarize(rows)
    return stats["within_0_2_ml"] == stats["n"] and stats["spill_cases"] == 0 and stats["home_cases"] == stats["n"]


def rank_key(stats):
    return (stats["n"] - stats["home_cases"], stats["spill_cases"], stats["n"] - stats["within_0_2_ml"],
            stats["mean_absolute_error_ml"])


def main():
    shared = load_module("reserve_sdk", ROOT / "train_target_volume_policy.py")
    versions = shared.configure_sdk()
    control = load_module("reserve_control", ROOT / "projects/pour_policy_training_v8/timed_correction_control.py")
    geometry_module = load_module("reserve_geometry", ROOT / "assets/robot_testtube_pour/scripts/mouth_pivot_lift_geometry.py")
    collector = load_module("reserve_env", ROOT / "assets/robot_testtube_pour/scripts/direct_training_env.py")
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
    env.max_seconds = DEADLINE + 1 / 60

    def run_block(stage, role, seeds, spec):
        rows = []
        for seed in seeds:
            for target in TARGETS:
                episode_id = f"{stage}_{role}_r{spec['reserve_ml']:.1f}_t{target}_s{seed}"
                if episode_id in done:
                    rows.append(done[episode_id])
                    continue
                policy = control.CorrectionPolicy(
                    COARSE, deadline_s=DEADLINE, topup_tilt_deg_s=spec["topup_tilt_deg_s"],
                    topup_lead_s=spec["topup_lead_s"], enable_topup=spec["enable_topup"],
                    reserve_ml=spec["reserve_ml"])
                print(f"RESERVE {episode_id} reserve={spec['reserve_ml']} topup={spec['enable_topup']}", flush=True)
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

    attempts = []
    reserve = 2.0
    note = "Stop 2 mL early, then fill."
    for index in range(3):
        spec = {"reserve_ml": reserve, "topup_tilt_deg_s": 4.0, "topup_lead_s": 0.12, "enable_topup": True}
        rows = run_block(f"pilot{index}", "candidate", PILOT_SEEDS, spec)
        stats = summarize(rows)
        attempts.append({"spec": spec, "summary": stats, "succeeded": succeeded(rows)})
        print("ATTEMPT " + json.dumps(attempts[-1], ensure_ascii=False), flush=True)
        if succeeded(rows):
            note = f"Reserve {reserve:.1f} mL met ±0.2 mL on both pilot seeds."
            break
        over = max(row["result"]["error_ml"] for row in rows)
        under = min(row["result"]["error_ml"] for row in rows)
        if over > 0.2:
            reserve = round(min(4.5, reserve + max(0.8, over)), 1)
            note = "A case was still over, so the next reserve stops earlier."
        elif under < -0.45 and stats["home_cases"] == stats["n"]:
            reserve = round(max(0.8, reserve - 0.6), 1)
            note = "Every case finished at home while still short, so the next reserve leaves less."
        else:
            note = "The reserve was not met inside 12 s. One faster fill is tried."
            spec = {**spec, "topup_tilt_deg_s": 6.0}
            rows = run_block(f"pilot{index}b", "candidate", PILOT_SEEDS, spec)
            stats = summarize(rows)
            attempts.append({"spec": spec, "summary": stats, "succeeded": succeeded(rows)})
            print("ATTEMPT " + json.dumps(attempts[-1], ensure_ascii=False), flush=True)
            break
    chosen_attempt = min(attempts, key=lambda item: rank_key(item["summary"]))
    chosen = dict(chosen_attempt["spec"])
    decision = {"recorded_at_utc": now(), "versions": versions, "note": note, "attempts": attempts, "chosen": chosen}
    write_json(OUT / "pilot_decision.json", decision)
    print("PILOT DECISION " + json.dumps({"note": note, "chosen": chosen, "succeeded": chosen_attempt["succeeded"]},
                                         ensure_ascii=False), flush=True)
    baseline = {**chosen, "enable_topup": False}
    validation = {"candidate": run_block("validation", "candidate", VAL_SEEDS, chosen),
                  "baseline": run_block("validation", "baseline", VAL_SEEDS, baseline)}
    val_scores = {role: summarize(rows) for role, rows in validation.items()}
    recommended = "candidate" if rank_key(val_scores["candidate"]) <= rank_key(val_scores["baseline"]) else "baseline"
    frozen = chosen if recommended == "candidate" else baseline
    write_json(OUT / "candidate_frozen.json", {"policy_kind": control.POLICY_KIND, "parameters": chosen, "role": "candidate"})
    write_json(OUT / "recommended_frozen.json", {"policy_kind": control.POLICY_KIND, "parameters": frozen, "role": recommended,
                                                "test_used_for_selection": False, "automatic_gui_deployment": False})
    tests = {"candidate": run_block("test", "candidate", TEST_SEEDS, chosen),
             "baseline": run_block("test", "baseline", TEST_SEEDS, baseline)}
    summary = {"complete": True, "recorded_at_utc": now(), "versions": versions, "deadline_s": DEADLINE,
               "pilot_decision": decision, "chosen": chosen, "recommended_role": recommended,
               "validation": val_scores, "test": {role: summarize(rows) for role, rows in tests.items()},
               "test_used_for_selection": False, "automatic_gui_deployment": False}
    write_json(OUT / "summary.json", summary)
    print("RESERVE SUMMARY " + json.dumps(summary["test"], allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
