"""Matplotlib figures for an audited whole-cycle response study and fit.

The normal CLI requires a completed independent collection audit and a bound
frozen fit. Synthetic CPU fixtures are watermarked and never written to a real
run. Plotting does not run liquid physics or imply terminal-policy success.
"""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

import curate_response_results as curate


ROOT = Path(__file__).resolve().parents[2]


def plotting():
    dependency = ROOT / "_cache/pour_training/plot_deps"
    if dependency.is_dir():
        sys.path.insert(0, str(dependency))
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "_cache/pour_training/mplconfig"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.lines import Line2D
    return plt, Normalize, Line2D


def title_footer(figure, summary, synthetic):
    if synthetic:
        figure.text(.5, .01, "SYNTHETIC CPU FIXTURE - NOT PHYSICAL RESULTS", ha="center", color="crimson", weight="bold")
    else:
        identifier = summary["protocol_id"][:12]
        figure.text(.5, .01, "Real live PBF; actual settled particle volume | protocol " + identifier
                    + " | response identification, not terminal task acceptance", ha="center", fontsize=8)


def response_curves(summary, synthetic=False):
    plt, Normalize, Line2D = plotting()
    rows, partials = summary["response_samples"], summary["partial_cycles"]
    targets = sorted({row["target_ml"] for row in rows + partials})
    if not targets:
        targets = [None]
    columns = 3
    figure, axes = plt.subplots((len(targets) + 1 + columns - 1) // columns, columns,
                               figsize=(18, 5.2 * ((len(targets) + 1 + columns - 1) // columns)), squeeze=False)
    flat = list(axes.flat)
    rms = [row["state"]["source_inside_velocity_rms_m_s"] for row in rows]
    normalizer = Normalize(vmin=min(rms) if rms else 0., vmax=max(rms) + 1e-9 if rms else 1.)
    cmap, last_scatter = plt.get_cmap("viridis"), None
    for axis, target in zip(flat, targets):
        group = [row for row in rows if row["target_ml"] == target]
        for row in group:
            state, action = row["state"], row["action"]
            angle = state["tilt_deg"] + action["approach_delta_deg"]
            dose = row["outcome"]["settled_delta_ml"]
            eligible = row["safe"] and row["settled"] and row["action_execution_complete"] and dose >= 0
            color = cmap(normalizer(state["source_inside_velocity_rms_m_s"]))
            if not eligible:
                axis.scatter(angle, dose, color="crimson", marker="x", s=80, zorder=4)
            else:
                marker = "s" if row["split"] == "validation" else "o"
                if dose == 0:
                    axis.scatter(angle, dose, facecolors="none", marker=marker,
                                 s=48, linewidths=1.1, edgecolors=color)
                else:
                    last_scatter = axis.scatter(angle, dose, c=[state["source_inside_velocity_rms_m_s"]],
                        cmap=cmap, norm=normalizer, marker=marker, s=48, linewidths=1.1, edgecolors=color)
        for partial in partials:
            # A partial cycle has no dose. Its cross is on the axis, explicitly
            # marked as an unmeasured endpoint, never plotted as measured zero.
            if partial["target_ml"] != target:
                continue
            angle = partial["state"]["tilt_deg"] + partial["action"]["approach_delta_deg"]
            axis.plot(angle, 0, marker="|", color="black", markersize=15)
            axis.annotate("partial (no dose)", (angle, 0), xytext=(0, 7), textcoords="offset points", fontsize=7)
        received = [row["state"]["received_ml"] for row in group]
        remaining = [row["state"]["tube_remaining_ml"] for row in group]
        axis.set_title("Prefix goal " + str(target) + " mL\n"
            + (f"received {min(received):.3f}-{max(received):.3f} mL\n"
               f"remaining {min(remaining):.3f}-{max(remaining):.3f} mL"
               if group else "No complete response endpoints"), fontsize=9)
        axis.set_xlabel("Absolute approach angle (degrees)")
        axis.set_ylabel("Actual whole-cycle settled increment (mL)")
        axis.axhline(0., color="gray", lw=.7)
        axis.grid(alpha=.2)
    legend_axis = flat[len(targets)]
    legend_axis.axis("off")
    handles = [Line2D([], [], marker="o", ls="none", color="#336b8e", label="Training response"),
               Line2D([], [], marker="s", ls="none", color="#336b8e", label="Independent validation response"),
               Line2D([], [], marker="o", markerfacecolor="none", ls="none", color="#336b8e", label="Measured zero dose retained"),
               Line2D([], [], marker="x", ls="none", color="crimson", label="Unsafe / unsettled /\nskipped / negative"),
               Line2D([], [], marker="|", ls="none", color="black", label="Partial endpoint: no measured dose")]
    legend_axis.legend(handles=handles, loc="upper left", frameon=False, fontsize=9)
    counts = summary["response_sample_counts"]
    legend_axis.text(0., .45, f"Completed endpoints: {counts['total_completed_endpoints']}\n"
        f"Eligible responses: {counts['fit_eligible']}\nMeasured zeros: {counts['zero_dose']}\n"
        f"Unfinished cycles: {counts['partial']}\nAll collection cases: {summary['collection_metrics']['cases']}\n"
        f"Cases without any cycle: {counts['cases_without_any_cycle_started']}\n"
        "Approach + micro-action + cutoff included.\nRMS colour records internal motion.", fontsize=9, va="top")
    for axis in flat[len(targets) + 1:]:
        axis.axis("off")
    if last_scatter is not None:
        colorbar_axis = figure.add_axes([.90, .18, .015, .66])
        figure.colorbar(last_scatter, cax=colorbar_axis, label="Actual anchor source RMS velocity (m/s)")
    figure.suptitle("Measured whole-cycle pouring response", fontsize=15)
    figure.subplots_adjust(left=.06, right=.86, top=.87, bottom=.09, hspace=.65, wspace=.38)
    title_footer(figure, summary, synthetic)
    return figure


def validation_figure(summary, synthetic=False):
    plt, _, _ = plotting()
    points = summary["response_model"]["validation_points"]
    figure, axes = plt.subplots(1, 3, figsize=(16, 5.2))
    supported = [point for point in points if point["supported"]]
    if supported:
        actual = [point["actual_delta_ml"] for point in supported]
        predicted = [point["mean_predicted_delta_ml"] for point in supported]
        axes[0].scatter(actual, predicted, c=[point["target_ml"] for point in supported], cmap="viridis", edgecolors="black")
        lower = min([0.] + actual + predicted)
        upper = max([1e-3] + actual + predicted)
        axes[0].plot([lower, upper], [lower, upper], "k--", lw=1)
    else:
        axes[0].text(.5, .5, "No supported validation predictions", ha="center", transform=axes[0].transAxes)
    axes[0].set(xlabel="Actual whole-cycle increment (mL)", ylabel="Predicted increment (mL)", title="Independent validation")
    for index, point in enumerate(points):
        if point["supported"]:
            axes[1].bar(index, point["signed_prediction_error_ml"], color="#277c8e")
        else:
            axes[1].plot(index, 0, marker="x", color="crimson")
            axes[1].annotate("abstain", (index, 0), xytext=(0, 8), textcoords="offset points", rotation=90, fontsize=7, color="crimson")
    axes[1].axhline(0., color="black", lw=.8)
    axes[1].set(xlabel="Validation response index (every row retained)", ylabel="Prediction - actual (mL)", title="Errors; no residual invented for abstention")
    counts = {"Supported": len(supported), "Abstained": len(points) - len(supported),
              "Partial": summary["response_model"]["partial_validation_cycles"]}
    axes[2].bar(counts.keys(), counts.values(), color=["#277c8e", "#c44846", "#888888"])
    axes[2].set(ylabel="Responses", title="Coverage, separate from prediction error")
    for axis in axes:
        axis.grid(axis="y", alpha=.2)
    figure.suptitle("Frozen response fit: validation-only selection; no terminal task claim", fontsize=14)
    figure.subplots_adjust(top=.82, bottom=.18, wspace=.35)
    title_footer(figure, summary, synthetic)
    return figure


def goal_figure(summary, synthetic=False):
    plt, _, _ = plotting()
    goal = summary["goal_evaluation"]
    if goal is None:
        return None
    cases, groups = goal["cases"], goal["per_goal"]
    figure, axes = plt.subplots(2, 2, figsize=(14, 9))
    for row in cases:
        color = "#277c8e" if row["strict_task_feasible"] else "crimson"
        axes[0, 0].scatter(row["target_ml"], row["signed_error_percent"], color=color, marker="x" if row["model_abstention_count"] else "o")
        axes[0, 1].scatter(row["target_ml"], row["physical_action_seconds"], color=color)
        axes[1, 0].scatter(row["target_ml"], row["wall_seconds_including_post"], color=color)
        axes[1, 1].scatter(row["target_ml"], row["peak_all_label_spill_particles"], color=color, marker="x" if row["model_abstention_count"] else "o")
    axes[0, 0].axhspan(-.1, .1, color="#c3e5c8", alpha=.5)
    axes[0, 0].axhline(-.1, color="gray", ls="--")
    axes[0, 0].axhline(.1, color="gray", ls="--")
    axes[0, 0].set(ylabel="Actual signed dose error (%)", title="Strict +/-0.1%; red = joint failure")
    axes[0, 1].axhline(24., color="crimson", ls="--")
    axes[0, 1].set(ylabel="Full action physical seconds", title="Pregrasp through original home + quiet")
    axes[1, 0].set(ylabel="Actual compute wall seconds, including post", title="Compute time is separate from physical action time")
    axes[1, 1].axhline(1., color="crimson", ls="--")
    axes[1, 1].set(ylabel="Peak all-label spill particles", title="Crosses denote model abstention; not omitted")
    for axis in axes.flat:
        axis.set_xlabel("Target (mL)")
        axis.grid(alpha=.2)
    figure.suptitle(f"Frozen goal controller: {goal['metrics']['joint_successes']}/{len(cases)} jointly accepted; "
                   + f"{len(groups)} tested goals", fontsize=14)
    figure.subplots_adjust(top=.90, bottom=.09, hspace=.32, wspace=.28)
    title_footer(figure, summary, synthetic)
    return figure


def write_figures(summary, output, *, synthetic=False, provenance=None):
    output = Path(output)
    curate.need(not output.exists(), "Plot output already exists; use a new named directory")
    plt, _, _ = plotting()
    output.mkdir(parents=True, exist_ok=False)
    prefix = "SYNTHETIC_" if synthetic else ""
    figures = [("response_curves.png", response_curves(summary, synthetic)),
               ("validation_response_fit.png", validation_figure(summary, synthetic))]
    optional = goal_figure(summary, synthetic)
    if optional is not None:
        figures.append(("goal_policy_evaluation.png", optional))
    artifacts = {}
    for name, figure in figures:
        path = output / (prefix + name)
        figure.savefig(path, dpi=160, facecolor="white")
        plt.close(figure)
        artifacts[path.name] = curate.sha(path)
    report = {"complete": True, "synthetic_fixture": synthetic,
              "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "protocol_id": summary["protocol_id"], "plot_source_sha256": curate.sha(__file__),
              "curator_source_sha256": curate.sha(curate.__file__), "files_sha256": artifacts,
              "terminal_goal_policy_acceptance_proven": summary["terminal_goal_policy_acceptance_proven"],
              "input_provenance": provenance or {}}
    with (output / (prefix + "plot_provenance.json")).open("x", encoding="utf8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--audit-sha256", required=True)
    parser.add_argument("--fit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--goal-run", type=Path)
    parser.add_argument("--goal-audit", type=Path)
    parser.add_argument("--goal-audit-sha256")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args(argv)
    study = curate.load_study(args.run, args.audit, args.audit_sha256, args.fit)
    goal = None
    if any(value is not None for value in (args.goal_run, args.goal_audit, args.goal_audit_sha256)):
        curate.need(all(value is not None for value in (args.goal_run, args.goal_audit, args.goal_audit_sha256)),
                    "All optional goal-evaluation arguments are required")
        goal = curate.load_goal_evaluation(args.goal_run, args.goal_audit, args.goal_audit_sha256, study)
    summary = curate.build_public_summary(study, goal)
    if args.check_only:
        print(json.dumps({"complete": True, "check_only": True, "collection_cases": len(summary["cases"]),
                          "terminal_goal_policy_acceptance_proven": summary["terminal_goal_policy_acceptance_proven"]}))
        return 0
    report = write_figures(summary, args.output, provenance={"audit_raw_sha256": args.audit_sha256,
        "candidate_raw_sha256": study["fit_manifest"]["candidate_raw_sha256"],
        "fit_manifest_raw_sha256": study["fit_manifest_raw_sha256"]})
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, TypeError) as error:
        print("Response plotting refused: " + str(error), file=sys.stderr)
        raise SystemExit(1)
