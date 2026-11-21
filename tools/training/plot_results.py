"""Plot only a completed pouring-training run, using its recorded results.

Example (no Kit or GPU):
  SDK_PYTHON tools/training/plot_results.py RUN_DIRECTORY

The two input files are summary.json and training_curve.csv. No smoothing,
inferred generations, synthetic rewards, or removal of failed test cases is
performed. Optional plotting dependencies live in a separate local cache.
"""

import argparse
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN = ROOT / "droplet_out" / "pour_policy_training" / "run_20261121_quick"
DEFAULT_DEPS = ROOT / "_cache" / "pour_training" / "plot_deps"


class IncompleteRun(ValueError):
    pass


def _finite(value, name):
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"Nonfinite recorded value: {name}")
    return number


def _integer(value, name):
    number = _finite(value, name)
    if int(number) != number:
        raise ValueError(f"Expected integer {name}, found {value!r}")
    return int(number)


def _safe_result(result):
    return (result["valid_physics"] is True and result["complete"] is True
            and result["upright"] is True
            and _finite(result["max_spilled_ml"], "max_spilled_ml") == 0
            and _finite(result["inflight_ml"], "inflight_ml") <= 0.1)


def load_completed_run(run):
    """Reject partial or mismatched data before creating any final artifact."""
    run = Path(run).resolve()
    summary_path, curve_path = run / "summary.json", run / "training_curve.csv"
    if not summary_path.is_file():
        raise IncompleteRun("summary.json is not ready; the run must finish before final plots are created.")
    summary_bytes = summary_path.read_bytes()
    summary = json.loads(summary_bytes.decode("utf-8"))
    if summary.get("complete") is not True:
        raise IncompleteRun("The training summary is incomplete; final plots are disabled.")
    if not curve_path.is_file():
        raise IncompleteRun("Completed summary exists but training_curve.csv is missing.")
    curve_bytes = curve_path.read_bytes()
    rows = list(csv.DictReader(io.StringIO(curve_bytes.decode("utf-8-sig"))))
    protocol, training = summary["protocol"], summary["training"]
    generations = _integer(training["generations"], "training.generations")
    population = _integer(training["population"], "training.population")
    episodes = _integer(training["episodes"], "training.episodes")
    targets = [_finite(target, "target") for target in protocol["targets_ml"]]
    if len(set(targets)) != len(targets) or not targets:
        raise ValueError("Targets must be unique and nonempty.")
    if generations < 1 or population < 1 or len(rows) != generations:
        raise IncompleteRun("The CSV does not contain exactly the completed number of CEM generations.")
    if protocol["generations"] != generations or protocol["population"] != population:
        raise ValueError("Protocol and summary training budget differ.")
    if episodes != generations * population * len(targets):
        raise ValueError("Reported training episode count does not match the actual CEM budget.")
    curve = []
    previous_best = math.inf
    for expected_generation, row in enumerate(rows):
        generation = _integer(row["generation"], "generation")
        count = _integer(row["completed_training_episodes"], "completed_training_episodes")
        if generation != expected_generation or count != (generation + 1) * population * len(targets):
            raise IncompleteRun("CSV generation indices or completed episode counts are inconsistent.")
        mean = _finite(row["mean_score"], "mean_score")
        best = _finite(row["generation_best_score"], "generation_best_score")
        best_so_far = _finite(row["best_training_score"], "best_training_score")
        if best > mean + 1e-10 or best_so_far > best + 1e-10 or best_so_far > previous_best + 1e-10:
            raise ValueError("Best/mean CEM scores are inconsistent.")
        if "mean_reward" in row and abs(_finite(row["mean_reward"], "mean_reward") + mean) > 1e-8:
            raise ValueError("Recorded reward is inconsistent with the score.")
        previous_best = best_so_far
        curve.append({"generation": generation, "mean_score": mean,
                      "generation_best_score": best, "best_training_score": best_so_far,
                      "completed_training_episodes": count})
    if abs(curve[-1]["best_training_score"] - _finite(training["best_score"], "training.best_score")) > 1e-8:
        raise ValueError("CSV best score and completed summary differ.")
    if summary.get("test_used_for_training_or_selection") is not False:
        raise ValueError("The summary must explicitly confirm that test data was not used for selection.")
    seeds = [_integer(seed, "test_seed") for seed in protocol["test_seeds"]]
    train_seeds = {_integer(seed, "train_seed") for seed in protocol["train_common_seed_by_target"].values()}
    validation_seeds = {_integer(seed, "validation_seed") for seed in protocol["validation_seeds"]}
    if not seeds or len(set(seeds)) != len(seeds) or set(seeds) & (train_seeds | validation_seeds):
        raise ValueError("Held-out test seeds are missing, duplicated, or overlap fitting/selection seeds.")
    pairs = summary["test"]["paired_results"]
    expected = {(target, seed) for target in targets for seed in seeds}
    observed = set()
    for pair in pairs:
        key = (_finite(pair["target_ml"], "pair.target_ml"), _integer(pair["seed"], "pair.seed"))
        if key not in expected or key in observed:
            raise ValueError("Held-out target/seed pair is unexpected or duplicated.")
        observed.add(key)
        for label in ("baseline", "selected"):
            result = pair[label]
            if (_finite(result["target_ml"], "result.target_ml"), _integer(result["seed"], "result.seed")) != key:
                raise ValueError("A paired result uses a different target or seed.")
            if result["randomized"] is not True:
                raise ValueError("Expected held-out randomized episodes, not nominal/replayed data.")
            received = _finite(result["received_ml"], "received_ml")
            error = _finite(result["error_ml"], "error_ml")
            if abs(received - key[0] - error) > 1e-8:
                raise ValueError("Received water and signed error disagree.")
            _safe_result(result)  # Validate status and numeric fields; retain failures.
    if observed != expected:
        raise IncompleteRun("Not all paired held-out results are present.")
    for label, metric_key in (("baseline", "baseline_metrics"), ("selected", "selected_metrics")):
        actual_mae = sum(abs(_finite(pair[label]["error_ml"], "error_ml")) for pair in pairs) / len(pairs)
        if abs(actual_mae - _finite(summary["test"][metric_key]["mae_ml"], "summary mae_ml")) > 1e-8:
            raise ValueError("Summary MAE disagrees with the plotted held-out observations.")
    provenance = {"protocol_id": summary["protocol_id"], "completed_at_utc": summary["completed_at_utc"],
                  "summary_sha256": hashlib.sha256(summary_bytes).hexdigest(),
                  "training_curve_sha256": hashlib.sha256(curve_bytes).hexdigest()}
    return summary, curve, provenance


def _matplotlib(deps):
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "_cache" / "pour_training" / "matplotlib"))
    try:
        import matplotlib
    except ModuleNotFoundError:
        if not Path(deps).is_dir():
            raise RuntimeError("Matplotlib is unavailable. Prepare the optional plot_deps cache first.") from None
        if sys.version_info[:2] != (3, 12) and Path(deps).resolve() == DEFAULT_DEPS.resolve():
            raise RuntimeError("The optional local plotting cache is built for Python 3.12. "
                               "Run this tool with Isaac Sim kit/python/python.exe.") from None
        sys.path.insert(0, str(Path(deps).resolve()))
        import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titlesize": 12, "figure.facecolor": "white",
                         "axes.axisbelow": True, "savefig.facecolor": "white"})
    return matplotlib, plt


def load_completed_candidate_audit(path, summary, provenance):
    path = Path(path)
    if path.is_dir():
        path = path / "candidate_audit_summary.json"
    if not path.is_file():
        raise IncompleteRun("The supplementary candidate audit has not finished.")
    audit = json.loads(path.read_text(encoding="utf-8"))
    if audit.get("complete") is not True:
        raise IncompleteRun("The supplementary candidate audit is incomplete.")
    if (audit["source_protocol_id"] != summary["protocol_id"]
            or audit["protocol"]["source_summary_sha256"] != provenance["summary_sha256"]):
        raise ValueError("Candidate audit belongs to a different original run.")
    if audit["candidate_parameters"] != summary["training"]["best_params"]:
        raise ValueError("Candidate audit does not use the original training-best policy.")
    if audit.get("deployment_changed") is not False or audit.get("test_used_for_training_or_deployment_selection") is not False:
        raise ValueError("Supplementary test must not alter the original validation deployment.")
    original = {(float(pair["target_ml"]), int(pair["seed"])): pair
                for pair in summary["test"]["paired_results"]}
    seen = set()
    for pair in audit["paired_results"]:
        key = (float(pair["target_ml"]), int(pair["seed"]))
        if key in seen or key not in original:
            raise ValueError("Candidate-audit target/seed pair is duplicated or unexpected.")
        seen.add(key)
        if pair["baseline"] != original[key]["baseline"] or pair["original_selected"] != original[key]["selected"]:
            raise ValueError("Candidate audit does not reuse the matching original baseline/default results.")
        candidate = pair["candidate"]
        if (float(candidate["target_ml"]), int(candidate["seed"])) != key or candidate["randomized"] is not True:
            raise ValueError("Candidate and reused baseline conditions differ.")
        error = _finite(candidate["error_ml"], "candidate.error_ml")
        if abs(_finite(candidate["received_ml"], "candidate.received_ml") - key[0] - error) > 1e-8:
            raise ValueError("Candidate volume and signed error disagree.")
        _safe_result(candidate)
    if seen != set(original) or audit["additional_evaluation_episodes"] != len(original):
        raise IncompleteRun("Supplementary candidate test pairs are incomplete.")
    actual_mae = sum(abs(float(pair["candidate"]["error_ml"])) for pair in audit["paired_results"]) / len(original)
    if abs(actual_mae - _finite(audit["candidate_metrics"]["mae_ml"], "candidate MAE")) > 1e-8:
        raise ValueError("Candidate summary MAE differs from its plotted results.")
    return audit


def training_figure(plt, summary, curve):
    training = summary["training"]
    x = [row["generation"] + 1 for row in curve]
    means = [row["mean_score"] for row in curve]
    bests = [row["generation_best_score"] for row in curve]
    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    fig.subplots_adjust(top=0.75, bottom=0.20, left=0.10, right=0.96)
    fig.suptitle(f"CEM training: {len(curve)} generations, {training['episodes']} live episodes",
                 x=0.10, y=0.97, ha="left", fontsize=17, weight="bold")
    fig.text(0.10, 0.90, "Three-parameter stopping policy  |  lower score is better",
             color="#4A5568", fontsize=11)
    ax.plot(x, means, "o-", color="#2463A6", linewidth=2, markersize=7,
            label="Generation mean (all candidates)")
    ax.plot(x, bests, "s-", color="#D66B1B", linewidth=2, markersize=7,
            label="Generation best candidate")
    ax.set_xticks(x)
    ax.set_xlim(min(x) - 0.15, max(x) + 0.15)
    ax.set_xlabel("Generation (one-based)")
    ax.set_ylabel("Candidate score, averaged over 20/30/40 mL")
    ax.grid(axis="y", color="#E2E8F0", linewidth=0.8)
    ax.legend(frameon=False, loc="best")
    top = ax.secondary_xaxis("top")
    top.set_xticks(x, [str(row["completed_training_episodes"]) for row in curve])
    top.set_xlabel("Completed training episodes", labelpad=8)
    fig.text(0.10, 0.095, "Markers show recorded generations only; no smoothing or discarded failed rollouts.",
             fontsize=9, color="#4A5568")
    fig.text(0.10, 0.05, f"Physical steps: {training['physical_steps']:,}  |  "
             f"Physical time: {training['physical_seconds']:.2f} s  |  "
             f"Final selected model: {summary['selected']['selection']}", fontsize=9, color="#4A5568")
    return fig


def heldout_figure(plt, summary, candidate_audit=None):
    targets = [float(target) for target in summary["protocol"]["targets_ml"]]
    seeds = [int(seed) for seed in summary["protocol"]["test_seeds"]]
    pairs = {(float(pair["target_ml"]), int(pair["seed"])): pair
             for pair in summary["test"]["paired_results"]}
    if candidate_audit is not None:
        candidate_pairs = {(float(pair["target_ml"]), int(pair["seed"])): pair
                           for pair in candidate_audit["paired_results"]}
        pairs = {key: {**pair, "candidate": candidate_pairs[key]["candidate"]}
                 for key, pair in pairs.items()}
    count = len(pairs)
    frozen_label = "Frozen learned policy" if summary["selected"]["is_trained"] else "Frozen baseline fallback"
    series = ((-0.12, "baseline", "o", "#2463A6", "Original feedback baseline"),
              (0.12, "selected", "s", "#D66B1B", frozen_label))
    if candidate_audit is not None:
        series = ((-0.20, "baseline", "o", "#2463A6", "Original feedback baseline"),
                  (0.00, "selected", "D", "#64748B", "Frozen default (validation-selected)"),
                  (0.20, "candidate", "s", "#D66B1B", "Trained candidate (not default)"))
    largest = max(abs(float(pair[label]["error_ml"]))
                  for pair in pairs.values() for _, label, _, _, _ in series)
    extent = max(0.28, 1.25 * largest)
    fig, axes = plt.subplots(1, len(targets), figsize=(12, 5.7), sharey=True, squeeze=False)
    fig.subplots_adjust(left=0.08, right=0.97, top=0.72, bottom=0.26, wspace=0.13)
    title = ("Supplementary candidate audit: deployment unchanged" if candidate_audit is not None
             else "Held-out paired test: baseline vs frozen policy")
    fig.suptitle(title, x=0.08, y=0.97,
                 ha="left", fontsize=17, weight="bold")
    baseline_mae = summary["test"]["baseline_metrics"]["mae_ml"]
    frozen_mae = summary["test"]["selected_metrics"]["mae_ml"]
    subtitle = (f"{count} paired target/seed cases ({2 * count} live episodes)  |  "
                f"MAE: baseline {baseline_mae:.4f} mL; frozen {frozen_mae:.4f} mL")
    if candidate_audit is not None:
        subtitle = (f"{count} matched cases  |  MAE: baseline {baseline_mae:.4f}; "
                    f"default {frozen_mae:.4f}; candidate {candidate_audit['candidate_metrics']['mae_ml']:.4f} mL")
    fig.text(0.08, 0.90, subtitle, fontsize=11, color="#4A5568")
    failed = 0
    for target, ax in zip(targets, axes[0]):
        ax.axhspan(-0.2, 0.2, color="#56B583", alpha=0.14, zorder=0)
        ax.axhline(0.2, color="#56A783", linestyle="--", linewidth=0.9)
        ax.axhline(-0.2, color="#56A783", linestyle="--", linewidth=0.9)
        ax.axhline(0.0, color="#64748B", linewidth=0.8)
        for index, seed in enumerate(seeds):
            pair = pairs[(target, seed)]
            ax.plot([index + offset for offset, _, _, _, _ in series],
                    [float(pair[label]["error_ml"]) for _, label, _, _, _ in series],
                    color="#CBD5E1", linewidth=1.5, zorder=1)
            for offset, label, marker, color, legend in series:
                value = float(pair[label]["error_ml"])
                ax.scatter(index + offset, value, marker=marker, s=62, color=color,
                           label=legend if index == 0 else None, zorder=3)
                if not _safe_result(pair[label]):
                    failed += 1
                    ax.scatter(index + offset, value, marker="x", s=115,
                               color="#B91C1C", linewidths=2, zorder=4)
        ax.set_title(f"Target {target:g} mL", pad=12, weight="bold")
        ax.set_xticks(range(len(seeds)), [f"seed {seed}" for seed in seeds])
        ax.set_xlim(-0.50, len(seeds) - 0.50)
        ax.set_ylim(-extent, extent)
        ax.grid(axis="y", color="#E2E8F0", linewidth=0.8)
        ax.set_xlabel("Matched initial-state / viscosity seed")
    axes[0, 0].set_ylabel("Received minus requested volume (mL)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.07, 0.855),
               ncols=len(series), frameon=False, fontsize=9 if candidate_audit is not None else 10)
    status = f"All pairs shown, including {failed} unsafe/incomplete result(s)."
    if failed:
        status += " Red crosses identify these results."
    fig.text(0.08, 0.14, "Green band: ±0.2 mL. " + status, fontsize=9, color="#4A5568")
    footer = "Model frozen before test; test cases were not used for fitting or selection."
    if candidate_audit is not None:
        footer = ("Baseline/default reused from original test; 6 new candidate live episodes. "
                  "Original validation alone selects deployment.")
    fig.text(0.08, 0.095, footer,
             fontsize=9, color="#4A5568")
    fig.text(0.08, 0.05, "Volume is privileged particle-mass simulation truth; particle quantum = 0.042875 mL.",
             fontsize=9, color="#4A5568")
    return fig, failed


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, nargs="?", default=DEFAULT_RUN)
    parser.add_argument("--output", type=Path, help="Defaults to the completed run directory.")
    parser.add_argument("--deps-path", type=Path, default=DEFAULT_DEPS)
    parser.add_argument("--candidate-audit", type=Path,
                        help="Completed supplementary audit directory or summary; adds a third comparison figure.")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    summary, curve, provenance = load_completed_run(args.run)
    candidate_audit = (load_completed_candidate_audit(args.candidate_audit, summary, provenance)
                       if args.candidate_audit is not None else None)
    if args.validate_only:
        print(json.dumps({"valid_complete_run": True, "generations": len(curve),
                          "training_episodes": summary["training"]["episodes"],
                          "test_pairs": len(summary["test"]["paired_results"]), **provenance}, indent=2))
        return 0
    matplotlib, plt = _matplotlib(args.deps_path)
    output = (args.output if args.output is not None else args.run).resolve()
    figures = [("training_curve.png", training_figure(plt, summary, curve))]
    heldout, failures = heldout_figure(plt, summary)
    figures.append(("heldout_errors.png", heldout))
    if candidate_audit is not None:
        candidate_figure, _ = heldout_figure(plt, summary, candidate_audit)
        figures.append(("heldout_candidate_errors.png", candidate_figure))
    output.mkdir(parents=True, exist_ok=True)
    pending = []
    try:
        for name, figure in figures:
            temporary = output / (name + ".tmp")
            figure.savefig(temporary, format="png", dpi=180,
                           metadata={"Description": "Actual completed live PBF training results",
                                     "Protocol": str(provenance["protocol_id"])})
            pending.append((temporary, output / name))
        for temporary, destination in pending:
            os.replace(temporary, destination)
    finally:
        for _, figure in figures:
            plt.close(figure)
        for temporary, _ in pending:
            if temporary.exists():
                temporary.unlink()
    print(json.dumps({"plots": [str(output / name) for name, _ in figures],
                      "matplotlib": matplotlib.__version__, "generations": len(curve),
                      "training_episodes": summary["training"]["episodes"],
                      "test_pairs": len(summary["test"]["paired_results"]),
                      "failed_test_results_retained": failures, **provenance}, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (IncompleteRun, ValueError, KeyError, RuntimeError) as error:
        print(f"Final plots not created: {error}", file=sys.stderr)
        raise SystemExit(2)
