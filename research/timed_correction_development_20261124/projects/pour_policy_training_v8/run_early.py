"""Stop the coarse stream earlier, then use the slow fill that already hit 20.02 mL.

A 2-4 mL volume reserve did not move the stop: the stream jumps across that
whole band in one fifth of a second. A longer tail prediction fires the existing
stop before that jump. The fill stays at 2.2 deg/s, which added 0.43 mL and
landed on 20.02 mL. Deadline stays 12 s. Up to three tail settings, then
validation and a fresh test.
"""

import importlib.util
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "droplet_out" / "pour_policy_timed_correction" / "run_20261123_early"
CACHE = ROOT / "projects" / "pour_policy_training_v3" / "mouth_pivot_lift025_z022_cache.npz"
CACHE_SHA = "106706110581ef85ac87137aafaf5f513491b37f9f6ea16c2f427f7685c06a1f"
DEADLINE = 12.0
PILOT_SEEDS = (117101, 117102)
VAL_SEEDS = (118101, 118102)
TEST_SEEDS = (119101, 119102, 119103)
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


def coarse(lead):
    return {"prepare_fraction": 1.0, "dose_tilt_deg_s": 25.6, "return_fraction": 1.0,
            "tail_lead_s": float(lead), "stop_bias_ml": 10.0, "max_dose_u_accel": 6.0}


def summarize(rows):
    errors = [row["result"]["error_ml"] for row in rows]
    absolute = [abs(value) for value in errors]
    return {"n": len(rows), "mean_absolute_error_ml": sum(absolute) / len(rows),
            "worst_absolute_error_ml": max(absolute),
            "within_0_2_ml": sum(value <= 0.2 for value in absolute),
            "spill_cases": sum(row["spill_particles"] > 0 or row["result"]["max_spilled_ml"] > 0 for row in rows),
            "home_cases": sum(row["full_home"] and row["result"]["physical_time_s"] <= DEADLINE + 1e-9 for row in rows),
            "mean_time_s": sum(row["result"]["physical_time_s"] for row in rows) / len(rows),
            "signed_errors_ml": errors}


def succeeded(rows):
    stats = summarize(rows)
    return stats["within_0_2_ml"] == stats["n"] and stats["spill_cases"] == 0 and stats["home_cases"] == stats["n"]


def rank_key(stats):
    return (stats["n"] - stats["home_cases"], stats["spill_cases"], stats["n"] - stats["within_0_2_ml"],
            stats["mean_absolute_error_ml"])


def main():
    shared = load_module("early_sdk", ROOT / "train_target_volume_policy.py")
    versions = shared.configure_sdk()
    control = load_module("early_control", ROOT / "projects/pour_policy_training_v8/timed_correction_control.py")
    geometry_module = load_module("early_geometry", ROOT / "assets/robot_testtube_pour/scripts/mouth_pivot_lift_geometry.py")
    collector = load_module("early_env", ROOT / "assets/robot_testtube_pour/scripts/direct_training_env.py")
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

    def run_block(stage, role, seeds, lead, enable):
        rows = []
        for seed in seeds:
            for target in TARGETS:
                episode_id = f"{stage}_{role}_lead{lead:.2f}_t{target}_s{seed}"
                if episode_id in done:
                    rows.append(done[episode_id])
                    continue
                policy = control.CorrectionPolicy(
                    coarse(lead), deadline_s=DEADLINE, topup_tilt_deg_s=2.2, topup_lead_s=0.12,
                    enable_topup=enable, reserve_ml=0.0)
                print(f"EARLY {episode_id} lead={lead:.2f} topup={enable}", flush=True)
                started = time.perf_counter()
                result = env.run_episode(target, seed, policy=policy, randomize=True,
                                         record_path=OUT / "traces" / f"{episode_id}.steps.jsonl.gz")
                post = env.post_observe(OUT / "traces" / f"{episode_id}.post.jsonl.gz", seconds=2.0) if result["valid_physics"] else None
                status = dict(env.last_controller.status)
                metrics = dict(env._last_liquid_metrics)
                full_home = bool(status.get("done") and status.get("home_static") and status.get("completed_by_deadline")
                                 and status.get("source_airborne_particles") == 0 and not status.get("timed_out"))
                row = {"episode_id": episode_id, "stage": stage, "role": role, "target_ml": target, "seed": seed,
                       "lead_s": lead, "enable_topup": enable, "result": result, "controller": status,
                       "spill_particles": int(metrics["spilled"]), "full_home": full_home,
                       "post_observation": post, "wall_seconds": time.perf_counter() - started,
                       "recorded_at_utc": now()}
                with journal.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(row, allow_nan=False) + "\n")
                done[episode_id] = row
                print(json.dumps({"id": episode_id, "received_ml": result["received_ml"], "error_ml": result["error_ml"],
                                  "time_s": result["physical_time_s"], "spill": row["spill_particles"],
                                  "home": full_home, "topups": status.get("correction_count")}, allow_nan=False), flush=True)
                rows.append(row)
        return rows

    lead = 0.50
    attempts = []
    note = "Tail 0.50 s, then the slow fill."
    for index in range(3):
        rows = run_block(f"pilot{index}", "candidate", PILOT_SEEDS, lead, True)
        stats = summarize(rows)
        attempts.append({"lead_s": lead, "summary": stats, "succeeded": succeeded(rows)})
        print("ATTEMPT " + json.dumps(attempts[-1], ensure_ascii=False), flush=True)
        if succeeded(rows):
            note = f"Tail {lead:.2f} s met ±0.2 mL on both pilot seeds."
            break
        over = max(row["result"]["error_ml"] for row in rows)
        under = min(row["result"]["error_ml"] for row in rows)
        if over > 0.2:
            lead = round(min(0.75, lead + 0.12), 2)
            note = "Still over on a coarse pour, so the next tail stops earlier."
        elif under < -0.45 and stats["home_cases"] == stats["n"]:
            lead = round(max(0.30, lead - 0.10), 2)
            note = "Home was reached while still short, so the next tail leaves more liquid."
        else:
            note = "The 12 s window closed before the fill finished."
            break
    best = min(attempts, key=lambda item: rank_key(item["summary"]))
    chosen_lead = best["lead_s"]
    decision = {"recorded_at_utc": now(), "versions": versions, "note": note, "attempts": attempts,
                "chosen_lead_s": chosen_lead}
    write_json(OUT / "pilot_decision.json", decision)
    print("PILOT DECISION " + json.dumps({"note": note, "lead": chosen_lead}, ensure_ascii=False), flush=True)
    validation = {"candidate": run_block("validation", "candidate", VAL_SEEDS, chosen_lead, True),
                  "baseline": run_block("validation", "baseline", VAL_SEEDS, chosen_lead, False)}
    val_scores = {role: summarize(rows) for role, rows in validation.items()}
    recommended = "candidate" if rank_key(val_scores["candidate"]) <= rank_key(val_scores["baseline"]) else "baseline"
    write_json(OUT / "candidate_frozen.json", {"policy_kind": control.POLICY_KIND, "lead_s": chosen_lead,
                                              "topup": True, "role": "candidate"})
    write_json(OUT / "recommended_frozen.json", {"policy_kind": control.POLICY_KIND, "lead_s": chosen_lead,
                                                "topup": recommended == "candidate", "role": recommended,
                                                "test_used_for_selection": False, "automatic_gui_deployment": False})
    tests = {"candidate": run_block("test", "candidate", TEST_SEEDS, chosen_lead, True),
             "baseline": run_block("test", "baseline", TEST_SEEDS, chosen_lead, False)}
    summary = {"complete": True, "recorded_at_utc": now(), "versions": versions, "deadline_s": DEADLINE,
               "pilot_decision": decision, "chosen_lead_s": chosen_lead, "recommended_role": recommended,
               "validation": val_scores, "test": {role: summarize(rows) for role, rows in tests.items()},
               "test_used_for_selection": False, "automatic_gui_deployment": False}
    write_json(OUT / "summary.json", summary)
    print("EARLY SUMMARY " + json.dumps(summary["test"], allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
