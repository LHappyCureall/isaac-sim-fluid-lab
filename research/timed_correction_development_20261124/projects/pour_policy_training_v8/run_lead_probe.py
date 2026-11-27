"""Measure the coarse gulp at tail leads between the published 0.23 s and 0.50 s.

Lead 0.23 s still finishes some 20 mL pours over the target. Lead 0.50 s stops
the command while only 2.6 mL has left, and the mouth still delivers about 14 mL.
Top-up is off, so these rows are the coarse landing only. Deadline stays 12 s.
"""

import importlib.util
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "droplet_out" / "pour_policy_timed_correction" / "run_20261123_lead_probe"
CACHE = ROOT / "projects" / "pour_policy_training_v3" / "mouth_pivot_lift025_z022_cache.npz"
CACHE_SHA = "106706110581ef85ac87137aafaf5f513491b37f9f6ea16c2f427f7685c06a1f"
DEADLINE = 12.0
SEED = 120101
FIRST_LEADS = (0.26, 0.30, 0.34, 0.38)


def now():
    return datetime.now(timezone.utc).isoformat()


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def coarse(lead):
    return {"prepare_fraction": 1.0, "dose_tilt_deg_s": 25.6, "return_fraction": 1.0,
            "tail_lead_s": float(lead), "stop_bias_ml": 10.0, "max_dose_u_accel": 6.0}


def main():
    shared = load_module("probe_sdk", ROOT / "train_target_volume_policy.py")
    shared.configure_sdk()
    control = load_module("probe_control", ROOT / "projects/pour_policy_training_v8/timed_correction_control.py")
    geometry_module = load_module("probe_geometry", ROOT / "assets/robot_testtube_pour/scripts/mouth_pivot_lift_geometry.py")
    collector = load_module("probe_env", ROOT / "assets/robot_testtube_pour/scripts/direct_training_env.py")
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

    def run_one(lead, target, seed):
        episode_id = f"probe_lead{lead:.2f}_t{target}_s{seed}"
        if episode_id in done:
            row = done[episode_id]
            result = row["result"]
            print(json.dumps({"id": episode_id, "received_ml": result["received_ml"],
                              "error_ml": result["error_ml"], "spill": row["spill_particles"],
                              "home": row["full_home"], "cached": True}), flush=True)
            return row
        policy = control.CorrectionPolicy(
            coarse(lead), deadline_s=DEADLINE, topup_tilt_deg_s=2.2, topup_lead_s=0.12,
            enable_topup=False, reserve_ml=0.0, lip_pulses=False)
        print(f"PROBE {episode_id}", flush=True)
        started = time.perf_counter()
        result = env.run_episode(target, seed, policy=policy, randomize=True,
                                 record_path=OUT / "traces" / f"{episode_id}.steps.jsonl.gz")
        status = dict(env.last_controller.status)
        metrics = dict(env._last_liquid_metrics)
        full_home = bool(status.get("done") and status.get("home_static") and status.get("completed_by_deadline")
                         and status.get("source_airborne_particles") == 0 and not status.get("timed_out"))
        row = {"episode_id": episode_id, "target_ml": target, "seed": seed, "lead_s": lead,
               "result": result, "spill_particles": int(metrics["spilled"]), "full_home": full_home,
               "wall_seconds": time.perf_counter() - started, "recorded_at_utc": now()}
        with journal.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
        done[episode_id] = row
        print(json.dumps({"id": episode_id, "received_ml": result["received_ml"], "error_ml": result["error_ml"],
                          "time_s": result["physical_time_s"], "spill": row["spill_particles"],
                          "home": full_home}), flush=True)
        return row

    rows = [run_one(lead, 20, SEED) for lead in FIRST_LEADS]
    errors = [row["result"]["error_ml"] for row in rows]
    extra = []
    if min(errors) > 0.15:
        extra = [0.42, 0.46]
    elif max(errors) < -1.5:
        extra = [0.24]
    rows.extend(run_one(lead, 20, SEED) for lead in extra)
    ranked = sorted(rows, key=lambda row: (abs(row["result"]["error_ml"] + 0.55), abs(row["result"]["error_ml"])))
    chosen = ranked[0]["lead_s"]
    fillable = [row for row in rows if -1.5 <= row["result"]["error_ml"] <= -0.15]
    if fillable:
        chosen = sorted(fillable, key=lambda row: abs(row["result"]["error_ml"] + 0.55))[0]["lead_s"]
        follow = [run_one(chosen, target, SEED) for target in (30, 40)]
    else:
        follow = []
    summary = {
        "seed": SEED,
        "twenty_ml": [{"lead_s": row["lead_s"], "error_ml": row["result"]["error_ml"],
                       "received_ml": row["result"]["received_ml"], "home": row["full_home"],
                       "spill": row["spill_particles"]} for row in rows],
        "chosen_lead_s": chosen if fillable else None,
        "follow": [{"target_ml": row["target_ml"], "error_ml": row["result"]["error_ml"],
                    "received_ml": row["result"]["received_ml"]} for row in follow],
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print("PROBE SUMMARY " + json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
