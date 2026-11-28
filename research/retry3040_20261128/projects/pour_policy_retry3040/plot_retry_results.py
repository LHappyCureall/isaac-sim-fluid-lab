"""Plot only the independently finalized30/40mL bounded retry experiment."""

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


def paired_panel(axis, pairs, field, ylabel, title):
    for index, pair in enumerate(pairs):
        axis.plot([index - .07, index + .07], [pair[role][field] for role in ("baseline", "candidate")],
            color=".75", lw=1, zorder=1)
    for role, shift, marker in (("baseline", -.07, "o"), ("candidate", .07, "s")):
        axis.scatter([index + shift for index in range(len(pairs))], [pair[role][field] for pair in pairs],
            color=COLORS[role], marker=marker, s=36, label=role, zorder=3)
        invalid = [(index + shift, pair[role][field]) for index, pair in enumerate(pairs) if not pair[role]["valid_evidence"]]
        if invalid:
            axis.scatter([row[0] for row in invalid], [row[1] for row in invalid], marker="x", color="crimson",
                label=role + " invalid evidence", zorder=4)
    axis.set_xticks(range(len(pairs)), [str(pair["seed"]) for pair in pairs], rotation=35, fontsize=8)
    axis.set(ylabel=ylabel, title=title)
    axis.grid(axis="y", alpha=.2); axis.legend(fontsize=8)


def full_figure(summary):
    plt = plotting()
    figure, axes = plt.subplots(4, 2, figsize=(15., 17.))
    figure.subplots_adjust(top=.91, bottom=.11, hspace=.57, wspace=.31)
    figure.suptitle("30 /40 mL fixed target expansion: measured conditional second cycle\n"
        "Primary +/-0.5 mL | original strict secondary +/-0.030 /0.040 mL kept separate", fontsize=15, y=.98)
    figure.text(.5, .94, f"{summary['budget']['development']} development episodes +40 fresh tests (20 matched pairs) | "
        "fixed strategy before test; no test-based profile selection", ha="center", fontsize=10)
    for column, goal in enumerate(curate.GOALS):
        pairs = [pair for pair in summary["test_pairs"] if pair["target_ml"] == goal]
        axis = axes[0, column]
        axis.axhspan(-1., 1., color="#b8d3ed", alpha=.15, label="secondary +/-1mL")
        axis.axhspan(-.5, .5, color="#70ad47", alpha=.18, label="primary +/-0.5mL")
        axis.axhline(0., color="black", lw=.8)
        paired_panel(axis, pairs, "error_ml", f"Actual received minus{goal:g} mL (mL)", f"{goal:g}mL: all ten new paired dose errors")
        errors = [pair[role]["error_ml"] for pair in pairs for role in ("baseline", "candidate")]
        axis.set_ylim(min(-1.1, min(errors) - .1), max(1.1, max(errors) + .1))
        axis = axes[1, column]
        paired_panel(axis, pairs, "physical_action_seconds", "Full-home physical action time (s)",
            f"{goal:g}mL: solid points = physics; dashed = compute wall")
        axis.axhline(24., color="crimson", ls=":")
        axis.set_ylim(0., max(25., max(pair[role]["physical_action_seconds"] for pair in pairs for role in COLORS) + 1.))
        second = axis.twinx()
        for role in COLORS:
            second.plot(range(len(pairs)), [pair[role]["compute_wall_seconds_including_post"] for pair in pairs],
                color=COLORS[role], ls="--", lw=1, alpha=.7)
        second.set_ylabel("Compute wall incl.2s post (s)")
        axis = axes[2, column]
        for role, shift in (("baseline", -.18), ("candidate", .18)):
            metric = summary["per_goal_metrics"][format(goal, "g")]["test"][role]
            axis.bar([index + shift for index in range(3)],
                [metric[key] for key in ("joint_passes", "easy_joint_passes", "original_strict_joint_passes")],
                width=.32, color=COLORS[role], label=role)
        axis.set_xticks(range(3), ["Primary +/-0.5mL", "Secondary +/-1mL", f"Original +/-{goal*.001:.3f}mL"], fontsize=9)
        axis.set(ylabel="Joint accepted cases /10", title=f"{goal:g}mL: dose + home + time + spill + no late water", ylim=(0., 11.))
        axis.set_yticks(range(0, 11, 2)); axis.grid(axis="y", alpha=.2); axis.legend(fontsize=8)
        axis = axes[3, column]
        candidate = [pair["candidate"] for pair in pairs]
        first = [case["actual_cycles"][0].get("actual_end_received_ml", float("nan"))
            if case["actual_cycles"] else float("nan") for case in candidate]
        axis.axhspan(goal - .5, goal + .5, color="#70ad47", alpha=.18, label="primary actual-dose band")
        axis.scatter(range(10), first, marker="o", color=".4", label="first actual cycle endpoint")
        axis.scatter(range(10), [case["received_ml"] for case in candidate], color=COLORS["candidate"], marker="s",
            label="actual final candidate dose")
        axis.set_xticks(range(10), [str(pair["seed"]) for pair in pairs], rotation=35, fontsize=8)
        matching = summary["paired_no_retry_matching"][format(goal, "g")]
        peaks = [summary["per_goal_metrics"][format(goal, "g")]["test"][role]["maximum_peak_all_label_spill_particles"] for role in COLORS]
        axis.set(ylabel="Actual received water (mL)", title=f"{goal:g}mL: actual retries{len(matching['actual_second_cycle_seeds'])}/10; "
            + f"no-retry exact dose/time{matching['no_retry_dose_and_time_exact_pairs']}/{matching['no_retry_pairs']}; peak spill{peaks[0]}/{peaks[1]}")
        axis.grid(axis="y", alpha=.2); axis.legend(fontsize=8)
    return figure


def development_figure(summary):
    plt = plotting()
    figure, axes = plt.subplots(2, 2, figsize=(14., 10.5))
    figure.subplots_adjust(top=.85, bottom=.12, hspace=.5, wspace=.3)
    figure.suptitle("Development gate stopped:30 /40mL conditional-retry expansion\n"
        "ZERO independent tests were run; development results are not held-out evidence", fontsize=15, y=.97)
    stages = summary["actual_stages"]
    for column, goal in enumerate(curate.GOALS):
        pairs = [{"seed": stage, **{role: next(case for case in summary["cases"]
            if case["stage"] == stage and case["target_ml"] == goal and case["role"] == role)
            for role in COLORS}} for stage in stages]
        axis = axes[0, column]
        axis.axhspan(-.5, .5, color="#70ad47", alpha=.18, label="primary +/-0.5mL")
        axis.axhline(0., color="black", lw=.8)
        paired_panel(axis, pairs, "error_ml", f"Actual received minus{goal:g}mL (mL)", f"{goal:g}mL: all bounded development attempts")
        axis = axes[1, column]
        paired_panel(axis, pairs, "physical_action_seconds", "Full-home physical action time (s)",
            f"{goal:g}mL: actual time, including every settling/retry phase")
        axis.axhline(24., color="crimson", ls=":")
        axis.set_ylim(0., max(25., max(pair[role]["physical_action_seconds"] for pair in pairs for role in COLORS) + 1.))
    return figure


def render(summary, synthetic=False):
    figure = development_figure(summary) if summary["development_stop"] else full_figure(summary)
    figure.text(.5, .04, "Live action-conditioned PBF | all failed/partial attempts retained | final dose is measured after actual settling\n"
        "Kinematic pregrasp and particle-mass simulation truth are not real weighing, contact grasp or hardware validation.",
        ha="center", fontsize=9)
    if synthetic:
        figure.text(.5, .008, "SYNTHETIC CPU FIXTURE - NOT PHYSICAL RESULTS", color="crimson", ha="center", weight="bold")
    return figure


def main():
    args = curate.arguments()
    summary = curate.curate(args.run, args.audit, args.audit_sha256)
    if args.check_only:
        print(json.dumps({"complete": True, "cases": len(summary["cases"]),
            "development_stop": summary["development_stop"], "figure_ready": True})); return
    output = curate.new_output(args.output)
    figure = render(summary)
    figure.savefig(output / "retry3040_results.png", dpi=150)
    (output / "retry3040_results.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf8")
    provenance = {"complete": True, "synthetic_fixture": False,
        "independent_audit_raw_sha256": args.audit_sha256,
        "plot_source_sha256": curate.sha(__file__), "curator_source_sha256": curate.sha(curate.__file__),
        "files_sha256": {name: curate.sha(output / name) for name in ("retry3040_results.json", "retry3040_results.png")}}
    (output / "plot_provenance.json").write_text(json.dumps(provenance, indent=2, allow_nan=False) + "\n", encoding="utf8")
    print("Wrote finalized bounded30/40mL report and one figure to " + str(output))


if __name__ == "__main__":
    main()
