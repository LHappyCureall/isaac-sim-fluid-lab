"""Plot only a completed, internally consistent full-budget pouring run.

No Kit, CUDA, live simulation, smoothing or invented epochs are used here.
The three-generation curve and nine paired cases come from committed run
records. Failed/unsafe results remain on the error plot with red crosses.
Recommended results reference the validation-selected candidate/baseline role.
"""

import argparse
import csv
from datetime import datetime
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN = ROOT / "droplet_out/pour_policy_training/run_20261122_full"
DEFAULT_DEPS = ROOT / "_cache/pour_training/plot_deps"


class IncompleteRun(ValueError):
    pass


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode("utf-8")).hexdigest()


def _number(value):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError("Expected a finite numeric result.")
    return float(value)


def _integer(value):
    number = _number(value)
    if int(number) != number:
        raise ValueError("Expected an integer result.")
    return int(number)


def _close(actual, expected):
    if abs(_number(actual) - _number(expected)) > 1e-8:
        raise ValueError("Recorded numeric fields disagree.")


def _file(run, relative):
    path = (run / relative).resolve()
    if not path.is_relative_to(run) or not path.is_file():
        raise IncompleteRun(f"Required run artifact is missing or unsafe: {relative}")
    return path


def _safe(result):
    for name in ("valid_physics", "complete", "upright", "mass_conserved"):
        if type(result[name]) is not bool:
            raise ValueError("Result safety fields must be booleans.")
    return (result["valid_physics"] and result["complete"] and result["upright"]
            and _number(result["max_spilled_ml"]) == 0 and _number(result["inflight_ml"]) <= .1)


def _result(result, target, seed, dt):
    if (_number(result["target_ml"]), _integer(result["seed"])) != (float(target), seed):
        raise ValueError("Episode result target/seed differs from its conditions.")
    if result["randomized"] is not True:
        raise ValueError("Full-budget episodes must retain the declared randomization.")
    _close(result["received_ml"] - target, result["error_ml"])
    _close(abs(result["error_ml"]), result["absolute_error_ml"])
    _close(result["physical_time_s"], _integer(result["physical_steps"]) * dt)
    _safe(result)  # A valid False is retained, never filtered out.


def _metrics(metric, results):
    errors = [abs(_number(result["error_ml"])) for result in results]
    safe = [_safe(result) for result in results]
    if _integer(metric["episodes"]) != len(results):
        raise ValueError("Metric episode count differs from recorded results.")
    _close(metric["mae_ml"], sum(errors) / len(errors))
    _close(metric["worst_absolute_error_ml"], max(errors))
    _close(metric["safe_completion_rate"], sum(safe) / len(results))
    for limit, key in ((.2, "within_0_2_ml_rate"), (.5, "within_0_5_ml_rate")):
        _close(metric[key], sum(ok and error <= limit for ok, error in zip(safe, errors)) / len(results))
    if metric["all_mass_conserved"] != all(result["mass_conserved"] for result in results):
        raise ValueError("Mass-conservation summary differs from recorded outcomes.")
    if _integer(metric["physical_steps"]) != sum(_integer(result["physical_steps"]) for result in results):
        raise ValueError("Physical-step summary differs from recorded outcomes.")
    _close(metric["physical_seconds"], sum(_number(result["physical_time_s"]) for result in results))


def _mean_score(results):
    def score(r):
        return (abs(r["error_ml"]) + 10 * r["max_spilled_ml"]
                + (0 if r["valid_physics"] else 1000)
                + (0 if r["complete"] and r["upright"] else 100) + .0005 * r["physical_time_s"])
    return sum(score(result) for result in results) / len(results)


def load_completed_run(directory):
    """Validate provenance/counts/roles before importing any plotting backend."""
    run = Path(directory).resolve()
    summary_path = _file(run, "summary.json")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("complete") is not True:
        raise IncompleteRun("summary.complete must be true before final plots are created.")
    manifest = json.loads(_file(run, "protocol.json").read_text(encoding="utf-8"))
    protocol = manifest["protocol"]
    protocol_id = _signature(protocol)
    if manifest["protocol_id"] != protocol_id or summary["protocol_id"] != protocol_id or summary["protocol"] != protocol:
        raise ValueError("Summary and frozen full-run protocol disagree.")
    expected = {"kind": "independent_full_budget_live_pouring", "generations": 3,
        "population": 6, "training_seed": 20261122, "training_episodes": 54, "validation_episodes": 18,
        "test_episodes": 18, "total_live_episodes": 90,
        "targets_ml": [20, 30, 40], "validation_seeds": [2101, 2102, 2103],
        "test_seeds": [31001, 31002, 31003],
        "train_common_seed_by_target": {"20": 2011, "30": 2012, "40": 2013},
        "validation_policy_roles": ["candidate", "baseline"], "test_policy_roles": ["candidate", "baseline"]}
    if any(protocol.get(key) != value for key, value in expected.items()):
        raise ValueError("This plotter requires the declared 54/18/18 protocol and its fresh seeds.")
    if len(protocol["source_sha256"]) != 8 or len(protocol["original_seven_source_sha256"]) != 7:
        raise ValueError("Frozen source manifests are incomplete.")
    if (summary.get("gui_deployment_written") is not False
            or summary.get("test_used_for_training_or_selection") is not False
            or summary.get("source_sha256_unchanged_after_run") is not True
            or summary.get("original_seven_source_sha256_unchanged") is not True):
        raise ValueError("Full-run isolation or source-freeze checks did not pass.")
    dt = _number(protocol["physics"]["dt_s"])
    _close(dt, 1 / 60)
    journal_path = _file(run, "episodes.jsonl")
    records = [json.loads(line) for line in journal_path.read_text(encoding="utf-8").splitlines()]
    if len(records) != 90 or len({record["episode_id"] for record in records}) != 90:
        raise IncompleteRun("Require all 90 unique completed episode records.")
    stages = {stage: [] for stage in ("train", "validation", "test")}
    for record in records:
        conditions = record["conditions"]
        if (record["protocol_id"] != protocol_id or record["conditions_sha256"] != _signature(conditions)
                or conditions["stage"] not in stages or conditions["policy_role"] not in ("candidate", "baseline")):
            raise ValueError("Journal protocol/conditions/typed role mismatch.")
        if conditions["randomized"] is not True:
            raise ValueError("A journal record has undeclared nominal conditions.")
        _result(record["result"], _number(conditions["target_ml"]), _integer(conditions["seed"]), dt)
        _close(record["score"], _mean_score([record["result"]]))
        stages[conditions["stage"]].append(record)
    if [len(stages[name]) for name in stages] != [54, 18, 18] or summary["total_recorded_episodes"] != 90:
        raise IncompleteRun("Executed stage counts differ from the 54/18/18 protocol.")
    train_expected = {(generation, candidate, target, seed) for generation in range(3)
                      for candidate in range(6) for target, seed in zip((20, 30, 40), (2011, 2012, 2013))}
    train_actual = {(r["conditions"]["generation"], r["conditions"]["candidate"],
                     r["conditions"]["target_ml"], r["conditions"]["seed"]) for r in stages["train"]}
    if train_actual != train_expected or any(r["conditions"]["policy_role"] != "candidate" for r in stages["train"]):
        raise ValueError("Training candidate/target/seed/generation records are incomplete.")
    for stage, seeds in (("validation", protocol["validation_seeds"]), ("test", protocol["test_seeds"])):
        wanted = {(role, target, seed) for role in ("candidate", "baseline") for target in (20, 30, 40) for seed in seeds}
        found = {(r["conditions"]["policy_role"], r["conditions"]["target_ml"], r["conditions"]["seed"]) for r in stages[stage]}
        if wanted != found:
            raise ValueError("Validation/test typed target/seed pairs are duplicated or missing.")
    _metrics(summary["training"], [r["result"] for r in stages["train"]])
    _metrics(summary["all_rollouts"], [r["result"] for r in records])

    curve_path = _file(run, "training_curve.csv")
    rows = list(csv.DictReader(io.StringIO(curve_path.read_text(encoding="utf-8-sig"))))
    if len(rows) != 3:
        raise IncompleteRun("Require exactly three recorded generations, not inferred epochs.")
    curve, best = [], float("inf")
    for generation, row in enumerate(rows):
        converted = {name: int(value) if name in ("generation", "completed_training_episodes") else float(value)
                     for name, value in row.items()}
        if converted["generation"] != generation or converted["completed_training_episodes"] != (generation + 1) * 18:
            raise ValueError("CSV generation/order/cumulative episode count mismatch.")
        generation_record = json.loads(_file(run, f"generation_{generation:03d}.json").read_text(encoding="utf-8"))
        if generation_record["protocol_id"] != protocol_id or generation_record["generation"] != generation:
            raise ValueError("CSV generation provenance differs from CEM records.")
        scores = generation_record["scores"]
        if len(scores) != 6 or len(generation_record["candidates"]) != 6:
            raise ValueError("CEM population size differs from the curve.")
        values = [r["result"] for r in stages["train"] if r["conditions"]["generation"] == generation]
        for candidate in range(6):
            candidate_records = [r for r in stages["train"] if r["conditions"]["generation"] == generation
                                 and r["conditions"]["candidate"] == candidate]
            if any(r["conditions"]["policy_params"] != generation_record["candidates"][candidate] for r in candidate_records):
                raise ValueError("CEM curve parameters differ from live candidate conditions.")
            _close(scores[candidate], _mean_score([r["result"] for r in candidate_records]))
        best = min(best, min(scores))
        for name, value in (("mean_score", sum(scores) / 6), ("generation_best_score", min(scores)),
                            ("best_training_score", best), ("mean_reward", -sum(scores) / 6),
                            ("mae_ml", sum(abs(r["error_ml"]) for r in values) / 18)):
            _close(converted[name], value)
        curve.append(converted)

    freeze = json.loads(_file(run, "freeze.json").read_text(encoding="utf-8"))
    if freeze["protocol_id"] != protocol_id or freeze["gui_deployment_written"] is not False:
        raise ValueError("Model freeze belongs to another/deployed experiment.")
    frozen_at = datetime.fromisoformat(freeze["frozen_at_utc"])
    for role, key in (("candidate", "candidate_sha256"), ("recommended", "recommended_sha256")):
        entry = summary["models"][role]
        model_path = _file(run, entry["file"])
        if _sha(model_path) != entry["sha256"] or entry["sha256"] != freeze[key]:
            raise ValueError("Frozen model SHA differs from summary/freeze marker.")
        model = json.loads(model_path.read_text(encoding="utf-8"))
        if (model["schema_version"] != 1 or model["policy_kind"] != "cem_linear_tail_compensation_v1"
                or model["parameters"] != entry["parameters"] or set(model["parameters"]) != set(protocol["parameter_bounds"])):
            raise ValueError("Frozen feedback model schema/parameters differ.")
        for name, value in model["parameters"].items():
            low, high = protocol["parameter_bounds"][name]
            if not low <= _number(value) <= high:
                raise ValueError("Frozen model parameter exceeds declared bounds.")
        metadata = model["metadata"]
        if (metadata["protocol_id"] != protocol_id or metadata["policy_role"] != role
                or metadata["source_sha256"] != protocol["source_sha256"]
                or metadata["test_used_for_selection"] is not False
                or metadata["frozen_at_utc"] != freeze["frozen_at_utc"]):
            raise ValueError("Frozen model metadata differs from its pre-test protocol.")
    candidate_parameters = summary["models"]["candidate"]["parameters"]
    search = json.loads(_file(run, "search_state.json").read_text(encoding="utf-8"))
    if (search["protocol_id"] != protocol_id or search["best_params"] != candidate_parameters
            or summary["training"]["best_params"] != candidate_parameters):
        raise ValueError("Candidate is not the frozen training-best result.")
    _close(search["best_score"], best)
    _close(summary["training"]["best_score"], best)
    scores = summary["validation"]["mean_score"]
    for role in ("candidate", "baseline"):
        values = [r["result"] for r in stages["validation"] if r["conditions"]["policy_role"] == role]
        if summary["validation"]["results"][role] != values:
            raise ValueError("Validation summary differs from committed live records.")
        _metrics(summary["validation"]["metrics"][role], values)
        _close(scores[role], _mean_score(values))
    changed = candidate_parameters != protocol["baseline_parameters"]
    recommended_role = "candidate" if changed and scores["candidate"] < scores["baseline"] else "baseline"
    recommendation = summary["recommendation"]
    if (freeze["recommended_role"] != recommended_role or recommendation["test_result_role"] != recommended_role
            or recommendation["additional_test_rollouts"] != 0 or recommendation["test_results_are_references"] is not True):
        raise ValueError("Recommendation must reference its validation-selected role without extra test runs.")
    wanted_parameters = candidate_parameters if recommended_role == "candidate" else protocol["baseline_parameters"]
    if summary["models"]["recommended"]["parameters"] != wanted_parameters:
        raise ValueError("Recommended model differs from validation selection.")
    pairs = summary["test"]["paired_results"]
    if len(pairs) != 9 or summary["test"]["executed_live_episodes"] != 18:
        raise ValueError("Test execution count differs from nine paired conditions.")
    seen = set()
    for pair in pairs:
        key = (_number(pair["target_ml"]), _integer(pair["seed"]))
        if key in seen or key not in {(float(t), s) for t in (20, 30, 40) for s in protocol["test_seeds"]}:
            raise ValueError("Test pair is duplicated or uses an unexpected target/seed.")
        seen.add(key)
        if pair["recommended_result_role"] != recommended_role:
            raise ValueError("A test pair silently changes the recommendation.")
        for role in ("candidate", "baseline"):
            record = next(r for r in stages["test"] if r["conditions"]["policy_role"] == role
                          and (r["conditions"]["target_ml"], r["conditions"]["seed"]) == key)
            if pair[role] != record["result"] or datetime.fromisoformat(record["completed_at_utc"]) < frozen_at:
                raise ValueError("Test pair differs from its committed post-freeze live result.")
    for role in ("candidate", "baseline"):
        _metrics(summary["test"][role + "_metrics"], [pair[role] for pair in pairs])
    if summary["test"]["recommended_metrics"] != summary["test"][recommended_role + "_metrics"]:
        raise ValueError("Recommended metrics do not reference the selected existing group.")
    provenance = {"protocol_id": protocol_id, "summary_sha256": _sha(summary_path),
        "training_curve_sha256": _sha(curve_path), "episodes_sha256": _sha(journal_path),
        "candidate_model_sha256": freeze["candidate_sha256"], "recommended_model_sha256": freeze["recommended_sha256"]}
    return summary, curve, provenance


def _matplotlib(deps):
    path = Path(__file__).with_name("plot_results.py")
    spec = importlib.util.spec_from_file_location("full_plot_backend", path)
    module = importlib.util.module_from_spec(spec)
    exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), module.__dict__)
    return module._matplotlib(deps)


def training_figure(plt, summary, curve):
    fig, ax = plt.subplots(figsize=(11, 6.3))
    fig.subplots_adjust(top=.72, bottom=.23, left=.11, right=.96)
    fig.suptitle("Full-budget CEM: 3 recorded generations", x=.11, y=.97, ha="left", fontsize=17, weight="bold")
    fig.text(.11, .90, "54 live training rollouts; 3 fixed target/seed initial conditions reused across candidates (CRN).", fontsize=10, color="#4A5568")
    x = [row["generation"] + 1 for row in curve]
    ax.plot(x, [r["mean_score"] for r in curve], "o-", color="#2463A6", linewidth=2, markersize=7, label="Generation mean")
    ax.plot(x, [r["generation_best_score"] for r in curve], "s-", color="#D66B1B", linewidth=2, markersize=7, label="Generation best candidate")
    ax.set_xticks(x)
    ax.set_xlim(.85, 3.15)
    ax.set_xlabel("Recorded generation (one-based)")
    ax.set_ylabel("Candidate score averaged over 20/30/40 mL; lower is better")
    ax.grid(axis="y", color="#E2E8F0")
    ax.legend(frameon=False)
    top = ax.secondary_xaxis("top")
    top.set_xticks(x, [str(row["completed_training_episodes"]) for row in curve])
    top.set_xlabel("Completed live training executions", labelpad=8)
    fig.text(.11, .13, "All recorded candidates retained; no smoothing, inferred epochs or discarded failed rollouts.", fontsize=9, color="#4A5568")
    fig.text(.11, .08, f"Physical training time: {summary['training']['physical_seconds']:.2f} s; "
             f"{summary['training']['physical_steps']:,} physical frames. No video clock.", fontsize=9, color="#4A5568")
    same_parameters = summary["models"]["candidate"]["parameters"] == summary["protocol"]["baseline_parameters"]
    note = ("Search best keeps the initial baseline parameters; no learned improvement. Test never selects."
            if same_parameters else "Validation recommendation: " + summary["recommendation"]["selection_origin"] + "; test never changes this choice.")
    fig.text(.11, .035, note, fontsize=9, color="#4A5568")
    return fig


def error_figure(plt, summary):
    pairs = {(float(p["target_ml"]), int(p["seed"])): p for p in summary["test"]["paired_results"]}
    seeds = summary["protocol"]["test_seeds"]
    extent = max(.28, 1.25 * max(abs(p[role]["error_ml"]) for p in pairs.values() for role in ("candidate", "baseline")))
    fig, axes = plt.subplots(1, 3, figsize=(12, 6.2), sharey=True)
    fig.subplots_adjust(left=.08, right=.97, top=.70, bottom=.26, wspace=.13)
    fig.suptitle("Fresh held-out test: frozen CEM candidate vs baseline", x=.08, y=.97, ha="left", fontsize=17, weight="bold")
    fig.text(.08, .90, "9 matched target/seed conditions; 3 fresh seeds; 18 live test executions. All cases shown.", fontsize=10, color="#4A5568")
    failed = 0
    for target, ax in zip((20, 30, 40), axes):
        ax.axhspan(-.2, .2, color="#56B583", alpha=.14, zorder=0)
        ax.axhline(0, color="#64748B", linewidth=.8)
        for boundary in (-.2, .2):
            ax.axhline(boundary, color="#56A783", linestyle="--", linewidth=.9)
        for index, seed in enumerate(seeds):
            pair = pairs[(float(target), seed)]
            ax.plot([index - .1, index + .1], [pair["baseline"]["error_ml"], pair["candidate"]["error_ml"]], color="#CBD5E1", linewidth=1.5)
            for offset, role, marker, color, label in ((-.1, "baseline", "o", "#2463A6", "Original feedback baseline"),
                    (.1, "candidate", "s", "#D66B1B", "Frozen CEM search candidate")):
                value = pair[role]["error_ml"]
                ax.scatter(index + offset, value, marker=marker, s=60, color=color, label=label if index == 0 else None, zorder=3)
                if not _safe(pair[role]):
                    failed += 1
                    ax.scatter(index + offset, value, marker="x", s=110, color="#B91C1C", linewidths=2, zorder=4)
        ax.set_title(f"Target {target} mL", pad=10, weight="bold")
        ax.set_xticks(range(3), [f"seed {seed}" for seed in seeds])
        ax.set_xlim(-.5, 2.5)
        ax.set_ylim(-extent, extent)
        ax.set_xlabel("Matched initialization / viscosity seed")
        ax.grid(axis="y", color="#E2E8F0")
    axes[0].set_ylabel("Received minus requested volume (mL)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(.07, .855), ncols=2, frameon=False, fontsize=10)
    candidate, baseline = summary["test"]["candidate_metrics"], summary["test"]["baseline_metrics"]
    fig.text(.08, .17, f"MAE: candidate {candidate['mae_ml']:.4f} mL; baseline {baseline['mae_ml']:.4f} mL. "
             f"Green band ±0.2 mL; unsafe/incomplete results marked red: {failed}.", fontsize=9, color="#4A5568")
    fig.text(.08, .115, "Recommendation references the " + summary["recommendation"]["test_result_role"] + " group; chosen only on validation, with no extra test executions.", fontsize=9, color="#4A5568")
    fig.text(.08, .065, "3 test seeds are not a broad success-rate study. Volume is privileged particle-mass truth; quantum 0.042875 mL.", fontsize=9, color="#4A5568")
    if summary["models"]["candidate"]["parameters"] == summary["protocol"]["baseline_parameters"]:
        fig.text(.08, .025, "Candidate parameters exactly equal the baseline; coincident results do not demonstrate learned improvement.", fontsize=9, color="#4A5568")
    return fig


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, nargs="?", default=DEFAULT_RUN)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--deps-path", type=Path, default=DEFAULT_DEPS)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    summary, curve, provenance = load_completed_run(args.run)
    if args.validate_only:
        print(json.dumps({"complete_validated": True, "training_executions": 54,
                          "heldout_conditions": 9, "heldout_seeds": 3, **provenance}, indent=2))
        return 0
    matplotlib, plt = _matplotlib(args.deps_path)
    output = (args.output if args.output is not None else args.run).resolve()
    figures = [("full_training_curve.png", training_figure(plt, summary, curve)),
               ("full_heldout_errors.png", error_figure(plt, summary))]
    output.mkdir(parents=True, exist_ok=True)
    pending = []
    try:
        for name, figure in figures:
            temporary = output / (name + ".tmp")
            figure.savefig(temporary, format="png", dpi=180,
                           metadata={"Description": "Completed full-budget live PBF results", **provenance})
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
                      "matplotlib": matplotlib.__version__, **provenance}, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (IncompleteRun, ValueError, KeyError, RuntimeError) as error:
        print(f"Full-run final plots not created: {error}", file=sys.stderr)
        raise SystemExit(2)
