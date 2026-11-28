"""Plot only a completed independently audited21-case fixed retry study."""

import json
import os
from pathlib import Path
import sys

import curate_retry_results as curate


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


def pairs(summary):
    rows = [row for row in summary["cases"] if row["stage"] == "test"]
    seeds = sorted({row["seed"] for row in rows})
    return seeds, {role: [next(row for row in rows if row["seed"] == seed and row["role"] == role)
        for seed in seeds] for role in ("baseline", "candidate")}


def paired_panel(axis, seeds, groups, field, ylabel, title):
    x = list(range(len(seeds)))
    for index in x:
        axis.plot([index - .07, index + .07],
            [groups["baseline"][index][field], groups["candidate"][index][field]], color=".75", lw=1, zorder=1)
    for role, shift, marker in (("baseline", -.07, "o"), ("candidate", .07, "s")):
        axis.scatter([index + shift for index in x], [row[field] for row in groups[role]],
            color=COLORS[role], marker=marker, s=38, label=role, zorder=3)
        invalid = [(index + shift, row[field]) for index, row in enumerate(groups[role]) if not row["valid_evidence"]]
        if invalid:
            axis.scatter([item[0] for item in invalid], [item[1] for item in invalid], marker="x",
                color="crimson", s=65, label=role + " invalid evidence", zorder=4)
    axis.set_xticks(x, [str(seed) for seed in seeds], rotation=35, fontsize=8)
    axis.set(ylabel=ylabel, title=title)
    axis.grid(axis="y", alpha=.2)
    axis.legend(fontsize=8)


def render(summary, synthetic=False):
    plt = plotting()
    figure, axes = plt.subplots(3, 2, figsize=(15., 13.5))
    figure.subplots_adjust(top=.89, bottom=.09, hspace=.52, wspace=.31)
    seeds, groups = pairs(summary)
    figure.suptitle("Fixed 20 mL: one cycle versus an actual-undershoot conditional second cycle\n"
        "Primary +/-0.5 mL | secondary +/-1 mL and original +/-0.020 mL kept separate", fontsize=15, y=.98)
    figure.text(.5, .925, "1 known-seed development pilot + 20 fresh tests (10 paired initial states) | "
        "fixed angles 84.0 /84.1 deg; no training, validation or test-based selection", ha="center", fontsize=10)
    axis = axes[0, 0]
    axis.axhspan(-1., 1., color="#b8d3ed", alpha=.18, label="secondary +/-1mL")
    axis.axhspan(-.5, .5, color="#70ad47", alpha=.18, label="primary +/-0.5mL")
    axis.axhline(0., color="black", lw=.8)
    paired_panel(axis, seeds, groups, "error_ml", "Actual received minus20 mL (mL)", "Fresh paired tests: final measured error")
    values = [row["error_ml"] for group in groups.values() for row in group]
    axis.set_ylim(min(-1.1, min(values) - .1), max(1.1, max(values) + .1))
    axis = axes[0, 1]
    paired_panel(axis, seeds, groups, "physical_action_seconds", "Full-home physical action time (s)",
                 "Time cost: dashed = actual compute wall including post")
    axis.axhline(24., color="crimson", ls=":")
    axis.set_ylim(0., max(25., max(row["physical_action_seconds"] for group in groups.values() for row in group) + 1.))
    second = axis.twinx()
    for role in ("baseline", "candidate"):
        second.plot(range(len(seeds)), [row["compute_wall_seconds_including_post"] for row in groups[role]],
            color=COLORS[role], ls="--", lw=1, alpha=.7)
    second.set_ylabel("Actual compute wall incl.2s post (s)")
    axis = axes[1, 0]
    paired_panel(axis, seeds, groups, "peak_all_label_spill_particles", "All-label peak spill (particles)",
                 "Action and post spills retained; maximum1 particle")
    axis.axhline(1., color="crimson", ls=":")
    axis.set_ylim(-.1, max(1.3, max(row["peak_all_label_spill_particles"] for group in groups.values() for row in group) + .2))
    axis.set_yticks(range(int(axis.get_ylim()[1]) + 1))
    axis = axes[1, 1]
    fields = ("joint_passes", "easy_joint_passes", "original_strict_joint_passes")
    for role, shift in (("baseline", -.18), ("candidate", .18)):
        metric = summary["stage_metrics"]["test"][role]
        axis.bar([index + shift for index in range(3)], [metric[field] for field in fields],
            width=.32, color=COLORS[role], label=role)
    axis.set_xticks(range(3), ["Primary +/-0.5mL", "Secondary +/-1mL", "Original +/-0.020mL"], fontsize=9)
    axis.set(ylabel="Jointly accepted test cases /10", title="Dose, home, time, safety and late-drop gates combined", ylim=(0., 11.5))
    axis.set_yticks(range(0, 11, 2)); axis.grid(axis="y", alpha=.2); axis.legend(fontsize=8)
    axis = axes[2, 0]
    axis.axhspan(19.5, 20.5, color="#70ad47", alpha=.18, label="primary 19.5--20.5mL")
    first = [row["actual_cycles"][0].get("actual_end_received_ml", float("nan"))
             if row["actual_cycles"] else float("nan") for row in groups["candidate"]]
    axis.scatter(range(len(seeds)), first, marker="o", color=".4", label="actual endpoint after first cycle")
    axis.scatter(range(len(seeds)), [row["received_ml"] for row in groups["candidate"]],
        marker="s", color=COLORS["candidate"], label="actual final candidate volume")
    axis.set_xticks(range(len(seeds)), [str(seed) for seed in seeds], rotation=35, fontsize=8)
    axis.set(ylabel="Actual received water (mL)", title="Candidate: first measured cycle versus actual final volume")
    axis.grid(axis="y", alpha=.2); axis.legend(fontsize=8)
    axis = axes[2, 1]; axis.axis("off")
    candidate = summary["stage_metrics"]["test"]["candidate"]
    baseline = summary["stage_metrics"]["test"]["baseline"]
    repaired = sum(not base["refine_feasible"] and retry["refine_feasible"]
                   for base, retry in zip(groups["baseline"], groups["candidate"]))
    regressed = sum(base["refine_feasible"] and not retry["refine_feasible"]
                    for base, retry in zip(groups["baseline"], groups["candidate"]))
    pilot = next(row for row in summary["cases"] if row["stage"] == "pilot")
    partial = sum(not cycle["has_recorded_endpoint"] for row in groups["candidate"] for cycle in row["actual_cycles"])
    skipped = sum(cycle.get("micro_skipped", False) for row in groups["candidate"] for cycle in row["actual_cycles"])
    details = ["Fixed strategy; candidate is not selected from these tests.",
        f"Actual second-cycle use: {candidate['actual_second_cycle_cases']} /10 fresh candidate cases",
        f"Baseline failure -> candidate pass: {repaired}; reverse: {regressed}",
        f"MAE baseline /candidate: {baseline['mean_absolute_error_ml']:.6f} /{candidate['mean_absolute_error_ml']:.6f} mL",
        f"Mean physical time: {baseline['mean_physical_action_seconds']:.3f} /{candidate['mean_physical_action_seconds']:.3f} s",
        f"Known-seed pilot: {pilot['received_ml']:.6f} mL, {pilot['physical_action_seconds']:.3f} s",
        "Pilot is seen development data, never a fresh test.",
        f"Candidate whole-cycle endpoints missing: {partial}; micro branches skipped: {skipped}",
        "Retry requires actual quiet undershoot, spill and home-time guards.",
        "No neural/PPO model; no GUI/default or arbitrary-goal claim."]
    axis.text(.01, .98, "\n".join(details), va="top", fontsize=10, linespacing=1.65)
    figure.text(.5, .038, "Real live PBF | whole-cycle flow includes approach and cutoff | final acceptance uses actual measured volume\n"
        "Kinematic pregrasp and particle-mass truth are not wrist weighing, contact grasp or hardware validation.", ha="center", fontsize=9)
    if synthetic:
        figure.text(.5, .008, "SYNTHETIC CPU FIXTURE - NOT PHYSICAL RESULTS", color="crimson", ha="center", weight="bold")
    return figure


def main():
    args = curate.arguments()
    summary = curate.curate(args.run, args.audit, args.audit_sha256)
    if args.check_only:
        print(json.dumps({"complete": True, "cases": len(summary["cases"]), "figure_ready": True})); return
    output = curate.new_output(args.output)
    figure = render(summary)
    figure.savefig(output / "retry20_results.png", dpi=150)
    (output / "retry20_results.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf8")
    provenance = {"complete": True, "synthetic_fixture": False,
        "independent_audit_raw_sha256": args.audit_sha256,
        "plot_source_sha256": curate.sha(__file__), "curator_source_sha256": curate.sha(curate.__file__),
        "files_sha256": {name: curate.sha(output / name) for name in ("retry20_results.json", "retry20_results.png")}}
    (output / "plot_provenance.json").write_text(json.dumps(provenance, indent=2, allow_nan=False) + "\n", encoding="utf8")
    print("Wrote independently audited fixed-retry JSON and one figure to " + str(output))


if __name__ == "__main__":
    main()
