"""Retrain the 12 s fill with particles small enough for a 0.1 percent band.

Spacing 0.0027 m makes one particle about 0.0197 mL, under 0.1 percent of
20 mL. The solver cell stays two spacings wide. The controller is the same
0.26 s lead and 2.2 deg/s fill. A pass is |error| <= 0.1 percent of the target,
at most one spilled particle, and home within 12 s. Validation and the fresh
test run only if both pilot seeds pass.
"""

import importlib.util
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "droplet_out" / "pour_policy_timed_correction" / "run_20261124_fine_particle"
CACHE = ROOT / "projects" / "pour_policy_training_v3" / "mouth_pivot_lift025_z022_cache.npz"
CACHE_SHA = "106706110581ef85ac87137aafaf5f513491b37f9f6ea16c2f427f7685c06a1f"
DEADLINE = 12.0
LEAD = 0.26
SPACING = 0.0027
PILOT_SEEDS = (165101, 165102)
VAL_SEEDS = (166101, 166102)
TEST_SEEDS = (167101, 167102, 167103)
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


def coarse():
    return {"prepare_fraction": 1.0, "dose_tilt_deg_s": 25.6, "return_fraction": 1.0,
            "tail_lead_s": LEAD, "stop_bias_ml": 10.0, "max_dose_u_accel": 6.0}


def summarize(rows):
    errors = [row["result"]["error_ml"] for row in rows]
    absolute = [abs(value) for value in errors]
    percents = [100.0 * abs(row["result"]["error_ml"]) / row["target_ml"] for row in rows]
    return {"n": len(rows), "mean_absolute_error_ml": sum(absolute) / len(rows),
            "worst_absolute_error_ml": max(absolute), "worst_percent": max(percents),
            "within_0_1_percent": sum(row["within_band"] for row in rows),
            "spill_over_one": sum(row["spill_particles"] > 1 for row in rows),
            "home_cases": sum(row["full_home"] and row["result"]["physical_time_s"] <= DEADLINE + 1e-9 for row in rows),
            "mean_time_s": sum(row["result"]["physical_time_s"] for row in rows) / len(rows),
            "signed_errors_ml": errors, "signed_percent": percents,
            "targets_ml": [row["target_ml"] for row in rows]}


def succeeded(rows):
    stats = summarize(rows)
    return stats["within_0_1_percent"] == stats["n"] and stats["spill_over_one"] == 0 and stats["home_cases"] == stats["n"]


def main():
    shared = load_module("fine_sdk", ROOT / "train_target_volume_policy.py")
    versions = shared.configure_sdk()
    control = load_module("fine_control", ROOT / "projects/pour_policy_training_v8/timed_correction_control.py")
    geometry_module = load_module("fine_geometry", ROOT / "assets/robot_testtube_pour/scripts/mouth_pivot_lift_geometry.py")
    collector = load_module("fine_env", ROOT / "assets/robot_testtube_pour/scripts/direct_training_env.py")
    geometry = geometry_module.load_geometry(CACHE, CACHE_SHA)
    OUT.mkdir(parents=True, exist_ok=True)
    journal = OUT / "episodes.jsonl"
    done = {}
    if journal.exists():
        for line in journal.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                done[row["episode_id"]] = row
    env = collector.DirectTrainingEnv(geometry=geometry, control=control, spacing=SPACING)
    env.max_seconds = DEADLINE + 1 / 60
    particle_ml = env.spacing ** 3 * 1.0e6
    print(json.dumps({"spacing_m": env.spacing, "particle_ml": particle_ml,
                      "particle_count": int(len(env.clean_points)),
                      "source_particles": int(env.source_count),
                      "source_ml": env.source_count * particle_ml,
                      "limit_20_ml": 0.001 * 20}), flush=True)

    def run_block(stage, seeds):
        rows = []
        for seed in seeds:
            for target in TARGETS:
                episode_id = f"{stage}_candidate_t{target}_s{seed}"
                if episode_id in done:
                    rows.append(done[episode_id])
                    continue
                policy = control.CorrectionPolicy(
                    coarse(), deadline_s=DEADLINE, topup_tilt_deg_s=2.2, topup_lead_s=0.12,
                    enable_topup=True, reserve_ml=0.0, lip_pulses=False, fine_percent=False)
                print(f"FINE {episode_id}", flush=True)
                started = time.perf_counter()
                result = env.run_episode(target, seed, policy=policy, randomize=True,
                                         record_path=OUT / "traces" / f"{episode_id}.steps.jsonl.gz")
                status = dict(env.last_controller.status)
                metrics = dict(env._last_liquid_metrics)
                limit = 0.001 * float(target)
                within = abs(result["error_ml"]) <= limit + 1e-9
                full_home = bool(status.get("done") and status.get("home_static") and status.get("completed_by_deadline")
                                 and status.get("source_airborne_particles") == 0 and not status.get("timed_out"))
                row = {"episode_id": episode_id, "stage": stage, "target_ml": target, "seed": seed,
                       "spacing_m": SPACING, "particle_ml": particle_ml, "limit_ml": limit,
                       "within_band": within, "corrections": status.get("correction_count"),
                       "result": result, "spill_particles": int(metrics["spilled"]), "full_home": full_home,
                       "wall_seconds": time.perf_counter() - started, "recorded_at_utc": now()}
                with journal.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(row, allow_nan=False) + "\n")
                done[episode_id] = row
                print(json.dumps({"id": episode_id, "received_ml": result["received_ml"],
                                  "error_ml": result["error_ml"],
                                  "percent": 100.0 * result["error_ml"] / target,
                                  "limit_ml": limit, "within": within,
                                  "time_s": result["physical_time_s"], "spill": row["spill_particles"],
                                  "home": full_home, "corrections": status.get("correction_count"),
                                  "valid": result.get("valid_physics")}), flush=True)
                rows.append(row)
        return rows

    pilot = run_block("pilot", PILOT_SEEDS)
    stats = summarize(pilot)
    print("FINE PILOT " + json.dumps(stats), flush=True)
    summary = {"complete": succeeded(pilot), "recorded_at_utc": now(), "versions": versions,
               "spacing_m": SPACING, "particle_ml": particle_ml, "deadline_s": DEADLINE,
               "pass_rule": "absolute error <= 0.1 percent of target; spilled particles <= 1",
               "pilot": stats, "test_used_for_selection": False, "automatic_gui_deployment": False}
    if not succeeded(pilot):
        summary["note"] = "Pilot missed 0.1 percent, spilled more than one particle, or did not go home."
        write_json(OUT / "summary.json", summary)
        print("FINE STOP " + json.dumps({"within": stats["within_0_1_percent"], "spill_over_one": stats["spill_over_one"],
                                         "home": stats["home_cases"], "percent": stats["signed_percent"]}), flush=True)
        return
    validation = run_block("validation", VAL_SEEDS)
    tests = run_block("test", TEST_SEEDS)
    summary["validation"] = summarize(validation)
    summary["test"] = summarize(tests)
    summary["complete"] = succeeded(validation) and succeeded(tests)
    write_json(OUT / "summary.json", summary)
    print("FINE SUMMARY " + json.dumps(summary["test"]), flush=True)


if __name__ == "__main__":
    main()
