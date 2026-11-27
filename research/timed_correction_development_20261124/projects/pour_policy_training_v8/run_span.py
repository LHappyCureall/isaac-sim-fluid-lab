"""Same 0.26 s fill on volumes that were not used to choose it.

The coarse stop and the slow fill both compare the requested millilitres with
the beaker. Nothing in the rule is special to 20, 30 or 40. This run freezes
tilt 25.6 deg/s, bias 10 mL, lead 0.26 s and the 2.2 deg/s fill, then pours
20, 25, 30, 35 and 40 mL. The high-flow retract margin is 0.20 mL. Deadline
stays 12 s. Validation and the fresh test run only if both pilot seeds meet
the dose, spill and home bar. The test is not used to change the margin.
"""

import importlib.util
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "droplet_out" / "pour_policy_timed_correction" / "run_20261123_span_b"
CACHE = ROOT / "projects" / "pour_policy_training_v3" / "mouth_pivot_lift025_z022_cache.npz"
CACHE_SHA = "106706110581ef85ac87137aafaf5f513491b37f9f6ea16c2f427f7685c06a1f"
DEADLINE = 12.0
LEAD = 0.26
PILOT_SEEDS = (159101, 159102)
VAL_SEEDS = (160101, 160102)
TEST_SEEDS = (161101, 161102, 161103)
TARGETS = (20, 25, 30, 35, 40)


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
    return {"n": len(rows), "mean_absolute_error_ml": sum(absolute) / len(rows),
            "worst_absolute_error_ml": max(absolute),
            "within_0_2_ml": sum(value <= 0.2 for value in absolute),
            "spill_cases": sum(row["spill_particles"] > 0 or row["result"]["max_spilled_ml"] > 0 for row in rows),
            "home_cases": sum(row["full_home"] and row["result"]["physical_time_s"] <= DEADLINE + 1e-9 for row in rows),
            "mean_time_s": sum(row["result"]["physical_time_s"] for row in rows) / len(rows),
            "signed_errors_ml": errors,
            "targets_ml": [row["target_ml"] for row in rows]}


def succeeded(rows):
    stats = summarize(rows)
    return stats["within_0_2_ml"] == stats["n"] and stats["spill_cases"] == 0 and stats["home_cases"] == stats["n"]


def main():
    shared = load_module("span_sdk", ROOT / "train_target_volume_policy.py")
    versions = shared.configure_sdk()
    control = load_module("span_control", ROOT / "projects/pour_policy_training_v8/timed_correction_control.py")
    geometry_module = load_module("span_geometry", ROOT / "assets/robot_testtube_pour/scripts/mouth_pivot_lift_geometry.py")
    collector = load_module("span_env", ROOT / "assets/robot_testtube_pour/scripts/direct_training_env.py")
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

    def run_block(stage, role, seeds, enable):
        rows = []
        for seed in seeds:
            for target in TARGETS:
                episode_id = f"{stage}_{role}_t{target}_s{seed}"
                if episode_id in done:
                    rows.append(done[episode_id])
                    continue
                policy = control.CorrectionPolicy(
                    coarse(), deadline_s=DEADLINE, topup_tilt_deg_s=2.2, topup_lead_s=0.12,
                    enable_topup=enable, reserve_ml=0.0, lip_pulses=False)
                print(f"SPAN {episode_id} topup={enable}", flush=True)
                started = time.perf_counter()
                result = env.run_episode(target, seed, policy=policy, randomize=True,
                                         record_path=OUT / "traces" / f"{episode_id}.steps.jsonl.gz")
                status = dict(env.last_controller.status)
                metrics = dict(env._last_liquid_metrics)
                full_home = bool(status.get("done") and status.get("home_static") and status.get("completed_by_deadline")
                                 and status.get("source_airborne_particles") == 0 and not status.get("timed_out"))
                row = {"episode_id": episode_id, "stage": stage, "role": role, "target_ml": target, "seed": seed,
                       "lead_s": LEAD, "enable_topup": enable, "corrections": status.get("correction_count"),
                       "result": result, "spill_particles": int(metrics["spilled"]), "full_home": full_home,
                       "wall_seconds": time.perf_counter() - started, "recorded_at_utc": now()}
                with journal.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(row, allow_nan=False) + "\n")
                done[episode_id] = row
                print(json.dumps({"id": episode_id, "received_ml": result["received_ml"], "error_ml": result["error_ml"],
                                  "time_s": result["physical_time_s"], "spill": row["spill_particles"],
                                  "home": full_home, "corrections": status.get("correction_count"),
                                  "valid": result.get("valid_physics")}), flush=True)
                rows.append(row)
        return rows

    pilot = run_block("pilot", "candidate", PILOT_SEEDS, True)
    stats = summarize(pilot)
    print("SPAN PILOT " + json.dumps(stats), flush=True)
    summary = {"complete": succeeded(pilot), "recorded_at_utc": now(), "versions": versions,
               "deadline_s": DEADLINE, "lead_s": LEAD, "targets_ml": list(TARGETS),
               "parameters_fit_on_ml": [20, 30, 40], "pilot": stats,
               "test_used_for_selection": False, "automatic_gui_deployment": False}
    if not succeeded(pilot):
        summary["note"] = "Pilot did not meet ±0.2 mL, zero spill and full home on 25 and 35 mL."
        write_json(OUT / "summary.json", summary)
        print("SPAN STOP " + json.dumps({"within": stats["within_0_2_ml"], "spill": stats["spill_cases"],
                                         "home": stats["home_cases"], "errors": stats["signed_errors_ml"]}), flush=True)
        return
    validation = {"candidate": run_block("validation", "candidate", VAL_SEEDS, True)}
    tests = {"candidate": run_block("test", "candidate", TEST_SEEDS, True)}
    summary["validation"] = {role: summarize(rows) for role, rows in validation.items()}
    summary["test"] = {role: summarize(rows) for role, rows in tests.items()}
    summary["complete"] = succeeded(tests["candidate"]) and succeeded(validation["candidate"])
    write_json(OUT / "summary.json", summary)
    print("SPAN SUMMARY " + json.dumps(summary["test"]), flush=True)


if __name__ == "__main__":
    main()
