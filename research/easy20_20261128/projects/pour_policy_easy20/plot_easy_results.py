"""One Matplotlib comparison figure for the fully audited20-case study."""

import json
import os
from pathlib import Path
import sys

import curate_easy_results as curate


ROOT = Path(__file__).resolve().parents[2]
COLORS = {"baseline": "#246eb9", "candidate": "#d97706"}


def plotting():
    dependencies = ROOT / "_cache/pour_training/plot_deps"
    if dependencies.is_dir():
        sys.path.insert(0, str(dependencies))
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "_cache/pour_training/mplconfig"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def pairs(summary, stage):
    rows = [row for row in summary["cases"] if row["stage"] == stage]
    seeds = sorted({row["seed"] for row in rows})
    return seeds, {role: [next(row for row in rows if row["seed"] == seed and row["role"] == role)
                         for seed in seeds] for role in ("baseline", "candidate")}


def paired_panel(axis, seeds, groups, value, ylabel, title):
    x = list(range(len(seeds)))
    for index in x:
        axis.plot([index - .07, index + .07],
                  [groups["baseline"][index][value], groups["candidate"][index][value]],
                  color=".75", lw=1, zorder=1)
    for role, shift, marker in (("baseline", -.07, "o"), ("candidate", .07, "s")):
        axis.scatter([value_x + shift for value_x in x], [row[value] for row in groups[role]],
                     color=COLORS[role], marker=marker, s=44, label=role, zorder=3)
    axis.set_xticks(x, [str(seed) for seed in seeds], rotation=20, fontsize=8)
    axis.set_ylabel(ylabel)
    axis.set_title(title, fontsize=11)
    axis.grid(axis="y", alpha=.2)
    axis.legend(fontsize=8)


def render(summary, synthetic=False):
    plt = plotting()
    figure, axes = plt.subplots(3, 2, figsize=(14.5, 13.8))
    figure.subplots_adjust(top=.88, bottom=.09, hspace=.64, wspace=.28)
    recommendation = summary["selection"]["recommended_role"]
    identical = summary["candidate_and_baseline_offsets_identical"]
    figure.suptitle("Fixed 20 mL beginner pouring: real-fluid offset calibration\n"
        "Easy task +/-1 mL | 24 physical seconds | peak spill <=1 particle | original strict +/-0.020 mL separate",
        fontsize=14, y=.975)
    figure.text(.5, .91, f"6 train + 4 validation + 10 paired test episodes | validation recommendation: {recommendation}"
        + (" | candidate and baseline offsets identical" if identical else ""), ha="center", fontsize=10)
    train = [row for row in summary["cases"] if row["stage"] == "train"]
    axis = axes[0, 0]
    axis.axhspan(19., 21., color="#70ad47", alpha=.13, label="beginner acceptance: 19--21 mL")
    axis.axhline(20., color="black", lw=.8)
    for index, seed in enumerate(sorted({row["seed"] for row in train})):
        rows = sorted([row for row in train if row["seed"] == seed], key=lambda row: row["control_target_offset_ml"])
        axis.plot([row["control_target_offset_ml"] for row in rows], [row["received_ml"] for row in rows],
                  marker=("o", "s")[index], label=f"train seed {seed}")
    axis.set(xlabel="Internal stop-target offset (mL)", ylabel="Actual settled received volume (mL)",
             title="Training response: all 6 actual episodes")
    axis.set_xticks([0., .5, 1.])
    axis.grid(alpha=.2)
    axis.legend(fontsize=8)
    val_seeds, val = pairs(summary, "validation")
    test_seeds, test = pairs(summary, "test")
    for axis, seeds, data, title in ((axes[0, 1], val_seeds, val, "Validation: same 2 conditions per role"),
                                     (axes[1, 0], test_seeds, test, "Independent test: same 5 conditions per role")):
        axis.axhspan(-1., 1., color="#70ad47", alpha=.13)
        axis.axhline(0., color="black", lw=.8)
        paired_panel(axis, seeds, data, "error_ml", "Actual dose minus 20 mL (mL)", title)
        low = min(-1.15, min(row["error_ml"] for rows in data.values() for row in rows) - .15)
        high = max(1.15, max(row["error_ml"] for rows in data.values() for row in rows) + .15)
        axis.set_ylim(low, high)
    axis = axes[1, 1]
    paired_panel(axis, test_seeds, test, "physical_action_seconds", "Raw physical action time (s)",
                 "Test full-home time; dashed = compute wall incl.post")
    axis.axhline(24., color="crimson", lw=1, ls=":", label="24 s deadline")
    axis.set_ylim(0., 25.)
    wall_axis = axis.twinx()
    for role in ("baseline", "candidate"):
        wall_axis.plot(range(len(test_seeds)), [row["compute_wall_seconds_including_post"] for row in test[role]],
                       color=COLORS[role], lw=1, ls="--", alpha=.7)
    wall_axis.set_ylabel("Compute wall time incl. 2 s post (s)")
    axis = axes[2, 0]
    paired_panel(axis, test_seeds, test, "peak_all_label_spill_particles", "All-label peak spill (particles)",
                 "Test spill: action and post observation included")
    axis.axhline(1., color="crimson", ls=":", lw=1)
    axis.set_ylim(-.1, max(1.3, max(row["peak_all_label_spill_particles"] for rows in test.values() for row in rows) + .2))
    axis.set_yticks(range(int(axis.get_ylim()[1]) + 1))
    axis = axes[2, 1]
    axis.axhspan(-.02, .02, color="purple", alpha=.2, label="original strict +/-0.020 mL")
    axis.axhline(0., color="black", lw=.8)
    paired_panel(axis, test_seeds, test, "error_ml", "Actual dose minus 20 mL (mL)",
                 "Original strict criterion: separately retained")
    stats = summary["stage_metrics"]["test"]
    detail = "\n".join(f"{role}: easy {stats[role]['joint_passes']}/5; strict {stats[role]['original_strict_joint_passes']}/5"
                         for role in ("baseline", "candidate"))
    axis.text(.02, .98, detail, transform=axis.transAxes, va="top", fontsize=8,
              bbox={"facecolor": "white", "alpha": .85, "edgecolor": ".8"})
    figure.text(.5, .039, "Real live PBF | fixed-identity particle-mass truth | kinematic pregrasp | narrow initial-state randomization\n"
        "This easier fixed-goal benchmark does not certify +/-0.1%, arbitrary 20--40 mL, contact grasp, or hardware performance.",
        ha="center", fontsize=9)
    if synthetic:
        figure.text(.5, .005, "SYNTHETIC CPU FIXTURE - NOT PHYSICAL RESULTS", color="crimson", ha="center", weight="bold")
    return figure


def main():
    args = curate.arguments()
    summary = curate.curate(args.run, args.audit, args.audit_sha256)
    if args.check_only:
        print(json.dumps({"complete": True, "cases": len(summary["cases"]), "figure_ready": True})); return
    output = curate.new_output(args.output)
    figure = render(summary)
    figure.savefig(output / "easy20_results.png", dpi=150)
    (output / "easy20_results.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf8")
    provenance = {"complete": True, "kind": "easy20_audited_plot_provenance_v1",
        "independent_audit_raw_sha256": args.audit_sha256,
        "curator_source_sha256": curate.sha(Path(curate.__file__)),
        "plot_source_sha256": curate.sha(Path(__file__)),
        "stage_artifact_sha256": summary["stage_artifact_sha256"],
        "files_sha256": {name: curate.sha(output / name)
                         for name in ("easy20_results.json", "easy20_results.png")},
        "synthetic_fixture": False, "new_physical_simulation_run": False}
    (output / "plot_provenance.json").write_text(json.dumps(provenance, indent=2, allow_nan=False) + "\n", encoding="utf8")
    print("Wrote audited JSON and one figure to " + str(output))


if __name__ == "__main__":
    main()
