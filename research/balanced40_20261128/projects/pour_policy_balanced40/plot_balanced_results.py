"""Plot an audited first-angle40mL study with fixed24.8deg/s coarse dosing, including gate stops."""

import json
import os
from pathlib import Path
import sys

import curate_balanced_results as curate

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


def calibration_panels(summary, dose_axis, time_axis):
    cases = [row for row in summary["cases"] if row["stage"] == "calibration"]
    chosen = summary["calibration_selected_candidate_id_recomputed"]
    dose_axis.axhspan(39.5, 40.5, color="#70ad47", alpha=.18, label="primary39.5--40.5mL")
    dose_axis.axhline(40., color="black", lw=.8)
    for index, seed in enumerate(sorted(curate.KNOWN_SEEDS)):
        rows = sorted([case for case in cases if case["seed"] == seed], key=lambda row: row["first_approach_tilt_deg"])
        dose_axis.plot([row["first_approach_tilt_deg"] for row in rows], [row["received_ml"] for row in rows],
            color=".65", ls=("-", "--", ":")[index], lw=1., label=f"seen seed{seed}")
        for case in rows:
            dose_axis.scatter(case["first_approach_tilt_deg"], case["received_ml"],
                marker=("o", "s", "^")[index], color="#70ad47" if case["refine_feasible"] else "#bb5555", s=48)
            if case["peak_all_label_spill_particles"] > 1:
                dose_axis.annotate(f"spill{case['peak_all_label_spill_particles']}",
                    (case["first_approach_tilt_deg"], case["received_ml"]), xytext=(3, 5), textcoords="offset points", fontsize=8)
    if chosen is not None:
        dose_axis.axvline(curate.ANGLES[chosen], color=".4", ls=":", lw=1.)
    dose_axis.set(xlabel="FIRST approach angle (degrees)", ylabel="Actual final received volume (mL)",
        title="Three seen conditions per first angle: green =joint pass, red =failure")
    dose_axis.set_xticks([87., 87.05, 87.1]); dose_axis.grid(alpha=.2); dose_axis.legend(fontsize=8)
    order = ("baseline",) + curate.CANDIDATE_ORDER
    groups = [[case for case in cases if case["candidate_id"] == candidate_id] for candidate_id in order]
    time_axis.bar(range(3), [sum(case["physical_action_seconds"] for case in group) / len(group) for group in groups],
        color=["#70ad47" if all(case["refine_feasible"] for case in group) else "#bb5555" for group in groups])
    for index, group in enumerate(groups):
        time_axis.text(index, max(case["physical_action_seconds"] for case in group) + .4,
            f"{sum(case['refine_feasible'] for case in group)}/3 joint", ha="center", fontsize=9)
    time_axis.axhline(24., color="crimson", ls=":")
    time_axis.set_xticks(range(3), [f"{curate.ANGLES[key]:g} deg" for key in order])
    time_axis.set(ylabel="Mean original full-home physical time (s)",
        title="Candidate elite needs ALL THREE passes\nBaseline remains a comparator",
        ylim=(0., max(25., max(case["physical_action_seconds"] for case in cases) + 1.)))
    time_axis.grid(axis="y", alpha=.2)


def paired_panel(axis, pairs, field, ylabel, title):
    for index, pair in enumerate(pairs):
        axis.plot([index - .07, index + .07], [pair[role][field] for role in ("baseline", "candidate")], color=".75", lw=1.)
    for role, shift, marker in (("baseline", -.07, "o"), ("candidate", .07, "s")):
        axis.scatter([index + shift for index in range(len(pairs))], [pair[role][field] for pair in pairs],
            color=COLORS[role], marker=marker, s=38, label=role, zorder=3)
        bad = [(index + shift, pair[role][field]) for index, pair in enumerate(pairs) if not pair[role]["valid_evidence"]]
        if bad:
            axis.scatter([row[0] for row in bad], [row[1] for row in bad], marker="x", color="crimson", s=65,
                label=role + " invalid evidence", zorder=4)
    axis.set_xticks(range(len(pairs)), [str(pair["seed"]) for pair in pairs], rotation=35, fontsize=8)
    axis.set(ylabel=ylabel, title=title); axis.grid(axis="y", alpha=.2); axis.legend(fontsize=8)


def dose_panel(axis, pairs, title):
    axis.axhspan(-1., 1., color="#b8d3ed", alpha=.15, label="secondary +/-1mL")
    axis.axhspan(-.5, .5, color="#70ad47", alpha=.18, label="primary +/-0.5mL")
    axis.axhline(0., color="black", lw=.8)
    paired_panel(axis, pairs, "error_ml", "Actual received minus40mL (mL)", title)
    errors = [pair[role]["error_ml"] for pair in pairs for role in COLORS]
    axis.set_ylim(min(-1.1, min(errors) - .1), max(1.1, max(errors) + .1))


def render(summary, synthetic=False):
    plt = plotting()
    stopped = summary["stopped_before_independent_test"]
    figure, axes = plt.subplots(2 if stopped else 4, 2, figsize=(15., 10.5 if stopped else 17.))
    figure.subplots_adjust(top=.87 if stopped else .91, bottom=.12 if stopped else .10, hspace=.57, wspace=.31)
    chosen = summary["calibration_selected_candidate_id_recomputed"]
    figure.suptitle("40mL first-angle calibration | coarse24.8deg/s /retry87.2 fixed\n"
        + ("Gate stopped; zero independent tests were run" if stopped
           else "Known calibration9 +fresh validation4 +new paired tests20; primary +/-0.5mL"), fontsize=15, y=.98)
    figure.text(.5, .91 if stopped else .94, f"Calibration rank choice: {chosen} | both roles retain the same conditional retry; "
        "no angle retuning from validation or test", ha="center", fontsize=10)
    calibration_panels(summary, axes[0, 0], axes[0, 1])
    validation = summary["paired_comparisons"].get("validation", [])
    if validation:
        dose_panel(axes[1, 0], validation, "Fresh validation: release gate, no angle retuning")
    else:
        axes[1, 0].axis("off"); axes[1, 0].text(.05, .7, "Validation was not run.\nKnown calibration does not replace it.", fontsize=12)
    if stopped:
        axis = axes[1, 1]; axis.axis("off")
        axis.text(.03, .88, f"Actual retained episodes: {summary['actual_cases']}\n"
            "Independent test cases: ZERO\n"
            "All calibration/validation failures remain in the report.\n"
            "No test precision or acceptance claim is available.\n"
            "20/30mL source and result versions stay separate.\n"
            "This is finite parameter calibration, not neural/PPO training.", va="top", fontsize=11, linespacing=1.8)
    else:
        pairs = summary["test_pairs"]
        dose_panel(axes[1, 1], pairs, "Independent paired test: all ten new initial states")
        axis = axes[2, 0]
        paired_panel(axis, pairs, "physical_action_seconds", "Full-home physical action time (s)",
            "Solid points: physical time; dashed: compute wall including post")
        axis.axhline(24., color="crimson", ls=":")
        axis.set_ylim(0., max(25., max(pair[role]["physical_action_seconds"] for pair in pairs for role in COLORS) + 1.))
        second = axis.twinx()
        for role in COLORS:
            second.plot(range(len(pairs)), [pair[role]["compute_wall_seconds_including_post"] for pair in pairs],
                color=COLORS[role], ls="--", lw=1, alpha=.7)
        second.set_ylabel("Compute wall incl.2s post (s)")
        axis = axes[2, 1]
        paired_panel(axis, pairs, "peak_all_label_spill_particles", "Peak all-label spill (particles)", "Every action and post spill retained")
        axis.axhline(1., color="crimson", ls=":")
        axis.set_ylim(-.1, max(1.3, max(pair[role]["peak_all_label_spill_particles"] for pair in pairs for role in COLORS) + .2))
        axis.set_yticks(range(int(axis.get_ylim()[1]) + 1))
        axis = axes[3, 0]
        for role, shift in (("baseline", -.18), ("candidate", .18)):
            metric = summary["stage_metrics"]["test"][role]
            axis.bar([index + shift for index in range(3)], [metric[key] for key in
                ("joint_passes", "easy_joint_passes", "original_strict_joint_passes")], width=.32, color=COLORS[role], label=role)
        axis.set_xticks(range(3), ["Primary +/-0.5mL", "Secondary +/-1mL", "Original +/-0.040mL"], fontsize=9)
        axis.set(ylabel="Joint accepted cases /10", title="Dose +home +time +spill +no late water", ylim=(0., 11.))
        axis.set_yticks(range(0, 11, 2)); axis.grid(axis="y", alpha=.2); axis.legend(fontsize=8)
        axis = axes[3, 1]; axis.axis("off")
        text = ["Only the FIRST approach angle differs between roles.", "The quiet78 coarse prefix before FIRST-angle treatment is a control."]
        for role in COLORS:
            metric = summary["stage_metrics"]["test"][role]
            text.extend((f"{role}: MAE {metric['mean_absolute_error_ml']:.6f}mL; worst {metric['maximum_absolute_error_ml']:.6f}mL",
                f"  actual second cycles {metric['actual_second_cycle_cases']}/10; mean time {metric['mean_physical_action_seconds']:.3f}s"))
        text.extend(("Calibration is finite, physically measured and kept separate.",
            "Both validation and test use frozen first-angle profiles.", "No arbitrary-target, default-GUI or hardware claim."))
        axis.text(.01, .95, "\n".join(text), va="top", fontsize=10, linespacing=1.65)
    figure.text(.5, .04, "Live action-conditioned PBF | final dose uses actual settling | failed/partial attempts retain their measurement basis\n"
        "Kinematic pregrasp and privileged particle mass are not real weighing, contact grasp or hardware validation.", ha="center", fontsize=9)
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
    figure.savefig(output / "balanced40_results.png", dpi=150)
    (output / "balanced40_results.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf8")
    provenance = {"complete": True, "synthetic_fixture": False,
        "independent_audit_raw_sha256": args.audit_sha256, "plot_source_sha256": curate.sha(__file__),
        "curator_source_sha256": curate.sha(curate.__file__),
        "files_sha256": {name: curate.sha(output / name) for name in ("balanced40_results.json", "balanced40_results.png")}}
    (output / "plot_provenance.json").write_text(json.dumps(provenance, indent=2, allow_nan=False) + "\n", encoding="utf8")
    print("Wrote independently finalized balanced first-angle40mL report and figure to " + str(output))


if __name__ == "__main__":
    main()
