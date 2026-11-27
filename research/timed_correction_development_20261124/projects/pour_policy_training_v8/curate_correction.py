"""Curate the finished 12-second correction run. Does not rescore or retrain."""

import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "droplet_out" / "pour_policy_timed_correction" / "run_20261123"
PUBLIC = ROOT / "projects" / "isaac-sim-fluid-lab"
DEPS = ROOT / "_cache" / "pour_training" / "plot_deps"
DATE = "20261123"


def sha256(path):
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def compact(row):
    result, post = row["result"], row.get("post_observation") or {}
    return {
        "episode_id": row["episode_id"], "stage": row["stage"], "role": row["role"],
        "target_ml": row["target_ml"], "seed": row["seed"], "deadline_s": row["spec"]["deadline_s"],
        "enable_topup": row["spec"]["enable_topup"], "topup_tilt_deg_s": row["spec"]["topup_tilt_deg_s"],
        "received_ml": result["received_ml"], "error_ml": result["error_ml"],
        "physical_time_s": result["physical_time_s"], "spill_particles": row["spill_particles"],
        "full_home": row["full_home"], "correction_count": row["controller"].get("correction_count", 0),
        "valid_physics": result["valid_physics"], "mass_conserved": result["mass_conserved"],
        "post_received_change_ml": post.get("maximum_received_change_ml"),
        "wall_seconds": row["wall_seconds"],
    }


def metrics(rows):
    errors = [abs(row["error_ml"]) for row in rows]
    signed = [row["error_ml"] for row in rows]
    return {
        "n": len(rows),
        "mean_absolute_error_ml": sum(errors) / len(rows),
        "worst_absolute_error_ml": max(errors),
        "mean_signed_error_ml": sum(signed) / len(rows),
        "within_0_2_ml": sum(value <= 0.2 for value in errors),
        "spill_cases": sum(row["spill_particles"] > 0 for row in rows),
        "full_home_cases": sum(bool(row["full_home"]) for row in rows),
        "mean_action_physical_seconds": sum(row["physical_time_s"] for row in rows) / len(rows),
    }


def main():
    summary = json.loads((RUN / "summary.json").read_text(encoding="utf-8"))
    rows = [compact(json.loads(line)) for line in (RUN / "episodes.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != 42:
        raise SystemExit(f"Expected 42 episodes, found {len(rows)}")
    test = {role: [row for row in rows if row["stage"] == "test" and row["role"] == role] for role in ("candidate", "baseline")}
    policies = {}
    for role in ("candidate", "recommended"):
        source = RUN / f"{role}_frozen.json"
        dest = PUBLIC / "assets/robot_testtube_pour/policy" / f"correction_12s_{DATE}_{role}.json"
        shutil.copyfile(source, dest)
        policies[role] = {"file": dest.relative_to(PUBLIC).as_posix(), "sha256": sha256(dest)}
    if DEPS.is_dir():
        sys.path.insert(0, str(DEPS))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    images = PUBLIC / "docs/images"
    images.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    width = 0.36
    for offset, role, color in ((-width / 2, "candidate", "#137fbd"), (width / 2, "baseline", "#da7b25")):
        vals = []
        for target in (20, 30, 40):
            group = [row for row in test[role] if row["target_ml"] == target]
            vals.append(sum(abs(row["error_ml"]) for row in group) / len(group))
        ax.bar([index + offset for index in range(3)], vals, width=width, color=color, label=role)
    ax.axhline(0.2, color="#666666", lw=1, ls="--")
    ax.set(title="12 s correction, held-out MAE", ylabel="Mean absolute error (mL)",
           xticks=[0, 1, 2], xticklabels=["20 mL", "30 mL", "40 mL"])
    ax.legend()
    fig.tight_layout()
    path = images / f"correction_12s_{DATE}_targets.png"
    fig.savefig(path, facecolor="white")
    report = {
        "schema_version": 1,
        "kind": "curated_timed_correction_12s_results",
        "complete": True,
        "neural_policy": False,
        "ppo": False,
        "automatic_gui_deployment": False,
        "test_used_for_selection": False,
        "recommended_role": summary["recommended_role"],
        "deadline_s": 12.0,
        "counts": {"pilot_10s": 6, "pilot_12s": 6, "validation": 12, "test": 18, "total": 42},
        "protocol_summary": {
            "targets_ml": [20, 30, 40],
            "pilot_seeds": [111101, 111102],
            "validation_seeds": [112101, 112102],
            "test_seeds": [113101, 113102, 113103],
            "coarse": summary["chosen"]["coarse"],
            "topup_tilt_deg_s": 2.5,
            "topup_lead_s": 0.15,
            "method": "Fast-path coarse stop, settle at 78 degrees, then a slow measured top-up. "
                      "Same structure as precision V2, on the five-second lift path, with a 12 s deadline.",
            "ten_second_pilot_passed": False,
        },
        "pilot_decision": summary["pilot_decision"],
        "validation": summary["validation"],
        "test": {role: {"metrics": metrics(group), "cases": group} for role, group in test.items()},
        "policies": policies,
        "images": {"targets": {"file": path.relative_to(PUBLIC).as_posix(), "sha256": sha256(path)}},
        "limitations": [
            "A top-up can add liquid only. A coarse pour that already passed 20 mL cannot be corrected.",
            "Received volume is privileged particle mass, quantum 0.042875 mL.",
            "Not a GUI default. The 51 s precision V2 remains the run that hit every held-out target.",
        ],
    }
    out = PUBLIC / "results" / f"correction_12s_{DATE}.json"
    out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(out)
    print(json.dumps(report["test"]["candidate"]["metrics"]))


if __name__ == "__main__":
    main()
