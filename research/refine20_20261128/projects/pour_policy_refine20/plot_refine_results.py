"""One Matplotlib comparison figure for the fully audited23-case refinement study."""

import json
import os
from pathlib import Path
import sys

import curate_refine_results as curate


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
    figure, axes = plt.subplots(4, 2, figsize=(14.5, 17.5))
    figure.subplots_adjust(top=.90, bottom=.075, hspace=.66, wspace=.3)
    recommendation = summary["selection"]["recommended_role"]
    angle = summary["selection"]["candidate_approach_tilt_deg"]
    figure.suptitle("Fixed 20 mL: one actual settled correction\n"
        "Primary +/-0.5 mL | secondary +/-1 mL and original +/-0.020 mL kept separate",
        fontsize=15, y=.98)
    figure.text(.5, .93, f"3 engineering (not selected) + 6 train + 4 validation + 10 test episodes | "
        + f"trained angle: {angle:g} deg | validation recommendation: {recommendation}", ha="center", fontsize=10)
    for axis, stage, title in ((axes[0, 0], "engineering", "Engineering safety probes: not model selection"),
                               (axes[0, 1], "train", "Training: actual response to approach angle")):
        group = [row for row in summary["cases"] if row["stage"] == stage]
        axis.axhspan(19.5, 20.5, color="#70ad47", alpha=.18, label="primary acceptance: 19.5--20.5 mL")
        axis.axhline(20., color="black", lw=.8)
        for index, seed in enumerate(sorted({row["seed"] for row in group})):
            rows = sorted([row for row in group if row["seed"] == seed], key=lambda row: row["approach_tilt_deg"])
            axis.plot([row["approach_tilt_deg"] for row in rows], [row["received_ml"] for row in rows],
                      marker=("o", "s")[index % 2], label=f"seed {seed}")
        axis.set(xlabel="Configured absolute approach angle (degrees)", ylabel="Actual final received volume (mL)", title=title)
        axis.set_xticks([84., 84.1, 84.2])
        axis.grid(alpha=.2)
        axis.legend(fontsize=8)
    val_seeds, val = pairs(summary, "validation")
    test_seeds, test = pairs(summary, "test")
    for axis, seeds, rows, title in ((axes[1, 0], val_seeds, val, "Validation: candidate versus coarse reference"),
                                    (axes[1, 1], test_seeds, test, "Independent test: all five paired initial states")):
        axis.axhspan(-1., 1., color="#b8d3ed", alpha=.18, label="secondary +/-1 mL")
        axis.axhspan(-.5, .5, color="#70ad47", alpha=.16, label="primary +/-0.5 mL")
        axis.axhline(0., color="black", lw=.8)
        paired_panel(axis, seeds, rows, "error_ml", "Actual received minus20 mL (mL)", title)
        values = [row["error_ml"] for items in rows.values() for row in items]
        axis.set_ylim(min(-1.1, min(values) - .1), max(1.1, max(values) + .1))
    axis = axes[2, 0]
    paired_panel(axis, test_seeds, test, "physical_action_seconds", "Full-home physical action time (s)",
                 "Test timing: dashed = compute wall including post")
    axis.axhline(24., color="crimson", ls=":", label="24s raw physical limit")
    axis.set_ylim(0., 25.)
    second = axis.twinx()
    for role in ("baseline", "candidate"):
        second.plot(range(len(test_seeds)), [row["compute_wall_seconds_including_post"] for row in test[role]],
                    color=COLORS[role], ls="--", lw=1, alpha=.7)
    second.set_ylabel("Actual compute wall incl.2s post (s)")
    axis = axes[2, 1]
    paired_panel(axis, test_seeds, test, "peak_all_label_spill_particles", "All-label peak spill (particles)",
                 "All action and post-observation spills retained")
    axis.axhline(1., color="crimson", ls=":")
    axis.set_ylim(-.1, max(1.3, max(row["peak_all_label_spill_particles"] for items in test.values() for row in items) + .2))
    axis.set_yticks(range(int(axis.get_ylim()[1]) + 1))
    axis = axes[3, 0]
    fields = ("joint_passes", "easy_joint_passes", "original_strict_joint_passes")
    for role, shift in (("baseline", -.18), ("candidate", .18)):
        metrics = summary["stage_metrics"]["test"][role]
        axis.bar([index + shift for index in range(3)], [metrics[field] for field in fields],
                 width=.32, color=COLORS[role], label=role)
    axis.set_xticks(range(3), ["Primary +/-0.5mL", "Secondary +/-1mL", "Original +/-0.020mL"], fontsize=9)
    axis.set(ylabel="Jointly accepted test cases /5", title="Dose, home, time, safety and late-drop gates combined", ylim=(0., 5.8))
    axis.set_yticks(range(6))
    axis.grid(axis="y", alpha=.2)
    axis.legend(fontsize=8)
    axis = axes[3, 1]
    axis.axis("off")
    details = ["Actual execution, separate from configured enablement"]
    for role in ("baseline", "candidate"):
        group = test[role]
        started = sum(row["actual_correction_cycles_started"] for row in group)
        skipped = sum(row["correction_enabled"] and row["actual_correction_cycles_started"] == 0 for row in group)
        details.extend((f"{role}: {started} actual cycles, {skipped} enabled but skipped",
                        f"  MAE {summary['stage_metrics']['test'][role]['mean_absolute_error_ml']:.6f} mL"))
    details.extend(("Skip/home uses actual settled primary-band measurement.",
                    "One correction maximum; every speed and coarse offset fixed.",
                    "No PPO / neural policy / historical V2 paired comparison.",
                    "Five narrow test seeds do not prove arbitrary target coverage."))
    axis.text(.02, .93, "\n".join(details), va="top", fontsize=10, linespacing=1.6)
    figure.text(.5, .035, "Real live PBF | measured whole-cycle flow, including approach and cutoff | kinematic pregrasp\n"
        "Particle-mass simulation truth is not wrist weighing, real contact grasp or hardware validation.", ha="center", fontsize=9)
    if synthetic:
        figure.text(.5, .006, "SYNTHETIC CPU FIXTURE - NOT PHYSICAL RESULTS", color="crimson", ha="center", weight="bold")
    return figure


def main():
    args = curate.arguments()
    summary = curate.curate(args.run, args.audit, args.audit_sha256)
    if args.check_only:
        print(json.dumps({"complete": True, "cases": len(summary["cases"]), "figure_ready": True})); return
    output = curate.new_output(args.output)
    figure = render(summary)
    figure.savefig(output / "refine20_results.png", dpi=150)
    (output / "refine20_results.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf8")
    provenance = {"complete": True, "synthetic_fixture": False,
        "independent_audit_raw_sha256": args.audit_sha256,
        "plot_source_sha256": curate.sha(__file__), "curator_source_sha256": curate.sha(curate.__file__),
        "files_sha256": {name: curate.sha(output / name) for name in ("refine20_results.json", "refine20_results.png")}}
    (output / "plot_provenance.json").write_text(json.dumps(provenance, indent=2, allow_nan=False) + "\n", encoding="utf8")
    print("Wrote independently audited refinement JSON and one figure to " + str(output))


if __name__ == "__main__":
    main()


