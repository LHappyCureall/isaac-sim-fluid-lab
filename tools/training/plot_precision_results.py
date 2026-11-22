"""Audit and plot a completed 48-episode precision-pouring calibration.

No Kit, GPU, active controller import, fitting or simulation is performed.
Partial runs produce no final figures. Failed/inaccurate episodes are retained.
Use Isaac Sim's Python 3.12; optional Matplotlib deps reuse _cache/pour_training.
"""

import argparse
import csv
from datetime import datetime
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN = ROOT / "droplet_out/pour_policy_precision_v2/run_20261122_calibrated"
DEFAULT_DEPS = ROOT / "_cache/pour_training/plot_deps"
ALGORITHM = "hand_designed_coarse_to_fine_feedback_plus_predeclared_finite_local_calibration"
QUANTUM_ML = 0.042875
FOOTNOTE = ("Privileged particle-mass volume; quantum 0.042875 mL; physical dt 1/60 s. "
            "Kinematic pre-grasp / bounded path; narrow position and viscosity randomization. "
            "No vision estimator or hardware calibration.")


class IncompleteRun(ValueError):
    pass


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode("utf-8")).hexdigest()


def _json(path):
    def reject(value):
        raise ValueError(f"Nonfinite JSON token {value} in {path}")
    return json.loads(Path(path).read_text(encoding="utf-8-sig"), parse_constant=reject)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _number(value):
    _require(not isinstance(value, bool), "Boolean used as a numerical measurement.")
    value = float(value)
    _require(math.isfinite(value), "Nonfinite recorded measurement.")
    return value


def _safe(result):
    # Accuracy is deliberately separate: an inaccurate dose can be physics-safe.
    return bool(result["valid_physics"] and result["mass_conserved"] and result["complete"]
                and result["upright"] and _number(result["max_spilled_ml"]) == 0
                and result["max_below_table_particles"] == 0
                and _number(result["inflight_ml"]) <= 0.1)


def _metrics(results):
    _require(bool(results), "An empty group cannot be plotted.")
    errors = [abs(_number(result["error_ml"])) for result in results]
    n = len(results)
    return {"episodes": n, "mae_ml": sum(errors) / n,
            "worst_absolute_error_ml": max(errors),
            "safe_completion_rate": sum(_safe(result) for result in results) / n,
            **{f"within_{str(threshold).replace('.', '_')}_ml_rate":
               sum(_safe(result) and error <= threshold for result, error in zip(results, errors)) / n
               for threshold in (0.1, 0.2, 0.5)},
            "all_mass_conserved": all(result["mass_conserved"] for result in results),
            "physical_steps": sum(result["physical_steps"] for result in results),
            "physical_seconds": sum(_number(result["physical_time_s"]) for result in results),
            "total_spilled_ml": sum(_number(result["max_spilled_ml"]) for result in results),
            "invalid_physics_episodes": sum(not result["valid_physics"] for result in results),
            "unsafe_or_incomplete_episodes": sum(not _safe(result) for result in results)}


def _check_metrics(recorded, results):
    actual = _metrics(results)
    for key, value in actual.items():
        _require(key in recorded, f"Missing aggregate: {key}")
        if isinstance(value, bool):
            _require(recorded[key] is value, f"Aggregate mismatch: {key}")
        else:
            _require(math.isclose(_number(recorded[key]), value, rel_tol=1e-10, abs_tol=1e-8),
                     f"Aggregate mismatch: {key}")
    return actual


def _score(results):
    m = _metrics(results)
    return (m["mae_ml"] + 0.5 * m["worst_absolute_error_ml"]
            + 10 * m["total_spilled_ml"] / m["episodes"]
            + 1000 * m["invalid_physics_episodes"] / m["episodes"]
            + 100 * m["unsafe_or_incomplete_episodes"] / m["episodes"]
            + 1e-5 * m["physical_seconds"] / m["episodes"])


def _release(metrics):
    return bool(metrics["mae_ml"] <= 0.1 and metrics["worst_absolute_error_ml"] <= 0.2
                and metrics["safe_completion_rate"] == 1 and metrics["all_mass_conserved"])


def _inside(directory, relative):
    path = (directory / relative).resolve()
    _require(path.is_relative_to(directory), f"Artifact escapes its run directory: {relative}")
    _require(path.is_file(), f"Missing recorded artifact: {relative}")
    return path


def _result_key(result):
    return (_number(result["target_ml"]), int(result["seed"]))


def _assert_results(recorded, expected):
    # Compare all fields, including failures; duplicates must not overwrite a case.
    _require(len(recorded) == len(expected), "Summary/journal group size differs.")
    observed = {_result_key(result): result for result in recorded}
    wanted = {_result_key(result): result for result in expected}
    _require(len(observed) == len(recorded) and observed == wanted,
             "A summary result differs from, duplicates, or omits a journal case.")


def _trace(run, record, protocol, *, keep_steps=False):
    path = _inside(run, record["record_path"])
    _require(_sha(path) == record["record_sha256"], "A recorded trace SHA changed.")
    retained, last_step, failure_steps, footer = [], -1, [], None
    condition, result = record["conditions"], record["result"]
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        header = json.loads(next(stream))
        implementation = json.loads(next(stream))
        _require(header.get("type") == "episode" and implementation.get("type") == "controller_implementation",
                 "Missing actual pre-physics controller provenance.")
        _require(implementation == record["controller_implementation"], "Trace/journal implementation differs.")
        _require(implementation["controller_file_sha256"] == condition["controller_source_sha256"],
                 "Actual controller source differs from the declared controller.")
        _require(implementation["wall_clock_start_utc"] == record["episode_wall_clock_start_utc"],
                 "Trace/journal start UTC differs.")
        _require(implementation["physical_solver_changed_by_wrapper"] is False
                 and implementation["episode_result_changed_by_wrapper"] is False,
                 "The precision collector must not rewrite physics or episode results.")
        _require(implementation["precision_wrapper_sha256"] == protocol["source_sha256"][
            "assets/robot_testtube_pour/scripts/precision_training_env.py"], "Wrapper source SHA differs.")
        _require(implementation["base_environment_sha256"] == protocol["source_sha256"][
            "assets/robot_testtube_pour/scripts/pour_training_env.py"], "Base environment SHA differs.")
        _require(header["fluid_mode"] == "live_PBF_action_conditioned"
                 and math.isclose(header["physics_dt_s"], 1 / 60)
                 and math.isclose(header["particle_volume_ml"], QUANTUM_ML),
                 "Unexpected physics time or volume basis.")
        _require(_result_key(header) == _result_key(result), "Trace header conditions differ.")
        for line in stream:
            row = json.loads(line, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
            _require(footer is None, "Records appear after the final physical-result footer.")
            kind = row.get("type")
            if kind == "step":
                _require(footer is None and row["step"] == last_step + 1, "Missing/reordered physical step.")
                last_step = row["step"]
                _require(math.isclose(_number(row["physical_time_s"]), last_step / 60, abs_tol=1e-9),
                         "Trace physical time is not step/60.")
                if keep_steps:
                    retained.append(row)
            elif kind == "failure":
                failure_steps.append(row["step"])
            elif kind == "result":
                _require(footer is None, "Duplicate trace result footer.")
                footer = {key: value for key, value in row.items() if key != "type"}
            else:
                raise ValueError(f"Unexpected trace record: {kind}")
    _require(footer == result, "Raw physical-result footer differs from the journal.")
    n = result["physical_steps"]
    # Invalid physics may fail before the final step can be logged. Keep that
    # episode if its explicit failure event explains the single missing state.
    _require(last_step == n or (last_step == n - 1 and failure_steps == [n]
                               and result["valid_physics"] is False), "Trace step coverage differs.")
    _require(result["valid_physics"] is not True or not failure_steps,
             "A failed trace is incorrectly marked as valid physics.")
    return retained


def load_completed_run(run, *, workspace=ROOT, verify_workspace=True):
    """Validate frozen inputs, all 48 journal/trace cases and summary aggregates.

    This function is CPU-only and writes nothing. Its first check rejects a
    running job before plotting dependencies are loaded or output dirs created.
    """
    run, workspace = Path(run).resolve(), Path(workspace).resolve()
    if not (run / "summary.json").is_file():
        raise IncompleteRun("No completed summary.json; final PNG generation is disabled.")
    summary = _json(run / "summary.json")
    if summary.get("complete") is not True:
        raise IncompleteRun("The 48-episode run has not completed.")
    protocol_record = _json(run / "protocol.json")
    protocol = protocol_record["protocol"]
    _require(summary["protocol"] == protocol and _signature(protocol) == summary["protocol_id"]
             == protocol_record["protocol_id"], "Protocol identity differs.")
    _require(protocol["algorithm"] == summary["algorithm"] == ALGORITHM
             and protocol["neural_policy"] is False and protocol["ppo"] is False
             and summary["neural_policy"] is False, "Unexpected algorithm declaration.")
    _require([protocol[key] for key in ("calibration_episodes", "validation_episodes", "test_episodes", "total_live_episodes")]
             == [18, 12, 18, 48] and summary["total_recorded_episodes"] == 48,
             "The completed predeclared 18/12/18 budget is required.")
    _require(protocol["targets_ml"] == [20, 30, 40] and len(protocol["configs"]) == 3,
             "Expected three targets and three predeclared configurations.")
    seed_groups = [protocol[key] for key in ("train_seeds", "validation_seeds", "test_seeds")]
    _require([len(set(group)) for group in seed_groups] == [2, 2, 3]
             and all(len(group) == len(set(group)) for group in seed_groups), "Seed counts differ.")
    _require(not any(set(seed_groups[i]) & set(seed_groups[j]) for i, j in ((0, 1), (0, 2), (1, 2))),
             "Calibration, validation and test seeds overlap.")
    inputs = _json(run / "input_freeze.json")
    declaration = inputs["declaration"]
    _require(_signature(declaration) == inputs["input_id"] == protocol["input_freeze_id"], "Input freeze differs.")
    config_path = _inside(run, "configs_frozen.json")
    _require(_sha(config_path) == protocol["config_input_sha256"] == declaration["config_input_sha256"],
             "Predeclared raw config bytes changed.")
    raw_configs = _json(config_path)
    _require(raw_configs == declaration["configurations_raw"], "Input config objects differ.")
    for raw, config in zip(raw_configs, protocol["configs"]):
        _require(raw["name"] == config["name"] and config["params"] == {
            **protocol["precision_defaults_at_freeze"], **raw["params"]}, "Expanded configuration differs.")
    _require(protocol["source_sha256"] == declaration["source_sha256"]
             and len(protocol["source_sha256"]) == 11, "The 11-source freeze differs.")
    for relative, digest in protocol["source_sha256"].items():
        _require(_sha(_inside(run, "source_snapshot/" + relative)) == digest, "Frozen source snapshot changed.")
        if verify_workspace:
            _require(_sha(_inside(workspace, relative)) == digest, "Active source differs from frozen source: " + relative)
    _require(protocol["protected_legacy_source_sha256"] == declaration["protected_legacy_source_sha256"]
             and len(protocol["protected_legacy_source_sha256"]) == 8, "Legacy eight-source freeze differs.")
    _require(all(protocol["source_sha256"][key] == value for key, value in
                 protocol["protected_legacy_source_sha256"].items()), "Legacy source is not preserved.")
    journal_path = _inside(run, "episodes.jsonl")
    records = [json.loads(line) for line in journal_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(records) != 48:
        raise IncompleteRun("The completed journal must contain exactly 48 episodes.")
    journal = {record["episode_id"]: record for record in records}
    _require(len(journal) == 48, "Duplicate journal episode IDs.")
    configs = protocol["configs"]
    expected = {f"calibration_{config['name']}_t{target}_s{seed}" for config in configs
                for seed in protocol["train_seeds"] for target in protocol["targets_ml"]}
    expected |= {f"{stage}_{role}_t{target}_s{seed}" for stage, seeds in
                 (("validation", protocol["validation_seeds"]), ("test", protocol["test_seeds"]))
                 for role in ("candidate", "baseline") for seed in seeds for target in protocol["targets_ml"]}
    _require(set(journal) == expected, "A predeclared episode is missing or an extra case appeared.")
    first_key = (protocol["targets_ml"][0], protocol["test_seeds"][0])
    first_trace = {}
    for record in records:
        condition, result = record["conditions"], record["result"]
        _require(record["protocol_id"] == summary["protocol_id"]
                 and _signature(condition) == record["conditions_sha256"], "Journal provenance differs.")
        _require(_result_key(condition) == _result_key(result) and condition["randomized"] is True
                 and result["randomized"] is True, "Journal result conditions differ.")
        stage, policy_role = condition["stage"], condition["policy_role"]
        seeds_by_stage = {"calibration": protocol["train_seeds"],
                          "validation": protocol["validation_seeds"], "test": protocol["test_seeds"]}
        _require(stage in seeds_by_stage and condition["seed"] in seeds_by_stage[stage]
                 and condition["target_ml"] in protocol["targets_ml"], "Unexpected stage/seed/target.")
        if stage == "calibration":
            matching = [config for config in configs if config["name"] == condition["config_name"]]
            _require(len(matching) == 1 and policy_role == "candidate", "Unknown calibration configuration.")
            expected_params = matching[0]["params"]
            expected_id = f"calibration_{condition['config_name']}_t{int(condition['target_ml'])}_s{condition['seed']}"
        else:
            _require(policy_role in ("candidate", "baseline"), "Unknown held-out policy role.")
            selected_config = summary["calibration"]["selected_config"]
            expected_params = selected_config["params"] if policy_role == "candidate" else None
            _require(condition["config_name"] == (selected_config["name"] if policy_role == "candidate" else None),
                     "Held-out configuration differs from the calibration selection.")
            expected_id = f"{stage}_{policy_role}_t{int(condition['target_ml'])}_s{condition['seed']}"
        _require(condition["policy_params"] == expected_params and result["policy_params"] == expected_params
                 and record["episode_id"] == expected_id, "Episode label/actual policy parameters differ.")
        _require(math.isclose(_number(result["received_ml"]) - _number(result["target_ml"]),
                             _number(result["error_ml"]), abs_tol=1e-8), "Water quantity/error differs.")
        _require(math.isclose(_number(result["physical_time_s"]), result["physical_steps"] / 60, abs_tol=1e-8),
                 "Episode physical time differs from fixed 60 Hz steps.")
        relative = "assets/robot_testtube_pour/scripts/" + (
            "target_control.py" if condition["policy_role"] == "baseline" else "precision_control.py")
        _require(condition["controller_source_sha256"] == protocol["source_sha256"][relative],
                 "A policy role used the wrong controller implementation.")
        keep = condition["stage"] == "test" and _result_key(result) == first_key
        steps = _trace(run, record, protocol, keep_steps=keep)
        if keep:
            first_trace[condition["policy_role"]] = steps
    calibration = summary["calibration"]
    _require(calibration["episodes"] == 18 and calibration["configurations"] == 3
             and len(calibration["configs"]) == 3, "Calibration budget differs.")
    for index, (config, entry) in enumerate(zip(configs, calibration["configs"])):
        group = [r["result"] for r in records if r["conditions"]["stage"] == "calibration"
                 and r["conditions"]["config_name"] == config["name"]]
        _require(entry["index"] == index and entry["name"] == config["name"]
                 and entry["params"] == config["params"] and len(group) == 6, "Configuration group differs.")
        _assert_results(entry["results"], group)
        _check_metrics(entry["metrics"], group)
        _require(math.isclose(entry["group_score"], _score(group), abs_tol=1e-8), "Calibration score differs.")
    best = min(calibration["configs"], key=lambda entry: (entry["group_score"], entry["index"]))
    selected = {"name": best["name"], "params": best["params"]}
    _require(selected == calibration["selected_config"], "Calibration did not select its declared minimum.")
    validation = summary["validation"]
    for role in ("candidate", "baseline"):
        group = [r["result"] for r in records if r["conditions"]["stage"] == "validation"
                 and r["conditions"]["policy_role"] == role]
        _require(len(group) == 6, "Validation role count differs.")
        _assert_results(validation["results"][role], group)
        _check_metrics(validation["metrics"][role], group)
        _require(math.isclose(validation["group_score"][role], _score(group), abs_tol=1e-8), "Validation score differs.")
    role = "candidate" if validation["group_score"]["candidate"] < validation["group_score"]["baseline"] else "baseline"
    _require(summary["recommendation"]["role"] == validation["recommended_role"] == role
             and summary["recommendation"]["selected_on_validation_only"] is True
             and summary["recommendation"]["additional_test_rollouts"] == 0
             and validation["test_used_for_selection"] is False
             and summary["test_used_for_training_or_selection"] is False,
             "Recommendation is not exclusively validation-selected.")
    freeze = _json(run / "freeze.json")
    _require(freeze["protocol_id"] == summary["protocol_id"] and freeze["recommended_role"] == role
             and freeze["candidate_config"] == selected and freeze["test_used_for_selection"] is False,
             "Pre-test freeze differs from calibration/validation selection.")
    for model_role in ("candidate", "recommended"):
        entry = summary["models"][model_role]
        path = _inside(run, entry["file"])
        model = _json(path)
        _require(_sha(path) == entry["sha256"] == freeze[model_role + "_sha256"], "Frozen model SHA differs.")
        expected_params = selected["params"] if model_role == "candidate" or role == "candidate" else protocol["baseline_params"]
        _require(model["parameters"] == entry["parameters"] == expected_params
                 and model["metadata"]["protocol_id"] == summary["protocol_id"]
                 and model["metadata"]["source_sha256"] == protocol["source_sha256"]
                 and model["metadata"]["config_input_sha256"] == protocol["config_input_sha256"],
                 "Model parameters or frozen provenance differ.")
    sidecar = dict((parts[1], parts[0]) for line in (run / "models_frozen.sha256").read_text().splitlines()
                   if (parts := line.split()))
    _require(sidecar == {entry["file"]: entry["sha256"] for entry in summary["models"].values()},
             "Model SHA sidecar differs.")
    freeze_time = datetime.fromisoformat(freeze["frozen_at_utc"])
    test_records = [record for record in records if record["conditions"]["stage"] == "test"]
    _require(all(datetime.fromisoformat(record["episode_wall_clock_start_utc"]) > freeze_time
                 for record in test_records), "A test started before model freezing.")
    pairs = summary["test"]["paired_results"]
    expected_pairs = {(target, seed) for target in protocol["targets_ml"] for seed in protocol["test_seeds"]}
    _require(len(pairs) == 9 and {_result_key(pair) for pair in pairs} == expected_pairs,
             "Held-out paired conditions are missing or duplicated.")
    for pair in pairs:
        _require(pair["recommended_result_role"] == role, "Test result changed validation selection.")
        for policy_role in ("candidate", "baseline"):
            key = f"test_{policy_role}_t{int(pair['target_ml'])}_s{int(pair['seed'])}"
            _require(pair[policy_role] == journal[key]["result"]
                     and _result_key(pair[policy_role]) == _result_key(pair), "Plotted paired result differs from journal.")
    for policy_role in ("candidate", "baseline"):
        _check_metrics(summary["test"][policy_role + "_metrics"], [pair[policy_role] for pair in pairs])
    _require(summary["test"]["recommended_metrics"] == summary["test"][role + "_metrics"],
             "Recommendation test metrics are not a reference to the frozen selected role.")
    _require(summary["test"]["distinct_target_seed_conditions"] == 9
             and summary["test"]["independent_initial_state_seeds"] == 3
             and summary["test"]["executed_live_episodes"] == 18, "Test counting semantics differ.")
    acceptance = summary["release_acceptance"]
    _require(acceptance["criteria"] == protocol["release_criteria"]
             and acceptance["does_not_reselect_model"] is True
             and acceptance["automatic_gui_deployment"] is False, "Test must accept/reject release only.")
    for policy_role in ("candidate", "baseline", "recommended"):
        _require(acceptance[policy_role + "_passed"] is _release(summary["test"][policy_role + "_metrics"]),
                 "Release acceptance differs from declared criteria.")
    _check_metrics(summary["all_rollouts"], [record["result"] for record in records])
    _require(summary["gui_checkpoint_sha256_before"] == summary["gui_checkpoint_sha256_after"]
             == protocol["gui_checkpoint_sha256_before"] and summary["gui_checkpoint_unchanged"] is True,
             "The calibration unexpectedly deployed a GUI model.")
    provenance = {"protocol_id": summary["protocol_id"], "completed_at_utc": summary["completed_at_utc"],
        "summary_sha256": _sha(run / "summary.json"), "journal_sha256": _sha(journal_path),
        "config_input_sha256": protocol["config_input_sha256"], "source_sha256": protocol["source_sha256"],
        "model_sha256": {key: value["sha256"] for key, value in summary["models"].items()},
        "counts": {"calibration": 18, "validation": 12, "test": 18, "total_live_episodes": 48,
                   "heldout_target_seed_conditions": 9, "heldout_initial_seeds": 3},
        "recommendation_selected_on_validation_only": True, "test_only_checks_release": True,
        "failures_removed": False, "measurement": FOOTNOTE, "current_workspace_verified": verify_workspace,
        "algorithm": ALGORITHM, "neural_or_ppo_training": False}
    ordered = sorted(pairs, key=lambda pair: (protocol["test_seeds"].index(pair["seed"]), pair["target_ml"]))
    return summary, ordered, first_trace, provenance


def _matplotlib(deps):
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "_cache/pour_training/matplotlib"))
    try:
        import matplotlib
    except ModuleNotFoundError:
        if not Path(deps).is_dir() or sys.version_info[:2] != (3, 12):
            raise RuntimeError("Use SDK Python 3.12 with the existing plot_deps cache.") from None
        sys.path.insert(0, str(Path(deps).resolve()))
        import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
        "figure.facecolor": "white", "savefig.facecolor": "white"})
    return plt


def calibration_figure(plt, summary):
    entries = summary["calibration"]["configs"]
    names = [entry["name"] for entry in entries]
    fig, axes = plt.subplots(1, 3, figsize=(14.4, 5.2))
    choices = (("mae_ml", "Mean absolute dose error", "mL", "#266AA7"),
               ("worst_absolute_error_ml", "Worst absolute dose error", "mL", "#D37A28"),
               ("physical_seconds", "Mean physical time per episode", "seconds", "#27866F"))
    selected = summary["calibration"]["selected_config"]["name"]
    for ax, (key, title, unit, color) in zip(axes, choices):
        values = [entry["metrics"][key] / (6 if key == "physical_seconds" else 1) for entry in entries]
        bars = ax.bar(range(3), values, color=color, width=0.62)
        ax.bar_label(bars, labels=[f"{value:.3f}" for value in values], padding=4, fontsize=10)
        ax.set_xticks(range(3), [name.replace("_", "\n") + ("\nselected" if name == selected else "")
                                for name in names], fontsize=9)
        ax.set(title=title, ylabel=unit, ylim=(0, max(values + [0.01]) * 1.3))
        ax.grid(axis="y", alpha=0.18)
    safe_counts = " | ".join(f"{entry['name']}: {round(entry['metrics']['safe_completion_rate']*6)}/6 safe"
                             for entry in entries)
    fig.suptitle("Predeclared local calibration: 3 configurations x 6 conditions", fontsize=15, y=0.98)
    fig.text(0.5, 0.885, "18 calibration + 12 validation + 18 test = 48 live evaluations; hand-designed feedback, not neural/RL training",
             ha="center", fontsize=10)
    fig.text(0.5, 0.13, safe_counts + "; all failed/inaccurate cases retained", ha="center", fontsize=9)
    fig.text(0.5, 0.045, FOOTNOTE, ha="center", fontsize=8, wrap=True)
    fig.subplots_adjust(top=0.78, bottom=0.26, wspace=0.28)
    return fig


def heldout_figure(plt, summary, pairs):
    fig, axes = plt.subplots(2, 1, figsize=(14.4, 8.6), sharex=True,
                             gridspec_kw={"height_ratios": [1.3, 1]})
    error_ax, time_ax = axes
    styles = (("candidate", "Precision candidate", "#2269A8", -0.16),
              ("baseline", "Original feedback baseline", "#D5822B", 0.16))
    error_ax.axhspan(-0.1, 0.1, color="#DDF2E7", alpha=0.8, label="Dose accuracy band: +/-0.1 mL")
    for boundary in (-0.2, 0.2):
        error_ax.axhline(boundary, color="#82928A", lw=0.8, ls="--")
    error_ax.axhline(0, color="#555555", lw=0.8)
    for role, label, color, offset in styles:
        xs = [index + offset for index in range(len(pairs))]
        errors = [pair[role]["error_ml"] for pair in pairs]
        error_ax.scatter(xs, errors, color=color, s=45, label=label, zorder=3)
        time_ax.bar(xs, [pair[role]["physical_time_s"] for pair in pairs], width=0.28, color=color, label=label)
        for index, pair in enumerate(pairs):
            if not _safe(pair[role]):
                error_ax.scatter(xs[index], errors[index], marker="x", color="#BF263B", s=110, lw=2, zorder=4)
                time_ax.text(xs[index], pair[role]["physical_time_s"], "x", color="#BF263B", ha="center", va="bottom")
    for index, pair in enumerate(pairs):
        error_ax.plot([index - .16, index + .16], [pair["candidate"]["error_ml"], pair["baseline"]["error_ml"]],
                      color="#9BA4AE", lw=0.8, zorder=1)
    worst = max(abs(pair[role]["error_ml"]) for pair in pairs for role, *_rest in styles)
    if worst > 1:
        error_ax.set_yscale("symlog", linthresh=0.25)
        error_ax.set_ylabel("Signed dose error (mL; symmetric log beyond 0.25)")
    else:
        error_ax.set_ylabel("Signed dose error (mL)")
        error_ax.set_ylim(-max(0.25, worst * 1.2), max(0.25, worst * 1.2))
    error_ax.plot([], [], "x", color="#BF263B", ms=8, label="Unsafe/incomplete case (retained)")
    error_ax.legend(ncols=2, loc="best", fontsize=9, frameon=False)
    error_ax.grid(axis="y", alpha=0.18)
    time_ax.set_xticks(range(9), [f"{pair['target_ml']:g} mL\nseed {pair['seed']}" for pair in pairs], fontsize=9)
    time_ax.set_ylabel("Original physical time (s)")
    time_ax.grid(axis="y", alpha=0.18)
    fig.suptitle("Held-out paired dose error and physical-time tradeoff", fontsize=15, y=0.98)
    metrics = summary["test"]
    stats = " | ".join(f"{label}: MAE {metrics[role+'_metrics']['mae_ml']:.4f}, worst "
                       f"{metrics[role+'_metrics']['worst_absolute_error_ml']:.4f} mL, safe "
                       f"{round(metrics[role+'_metrics']['safe_completion_rate']*9)}/9"
                       for role, label, *_rest in styles)
    fig.text(0.5, 0.93, "9 target/seed conditions; 3 fresh seeds; 18 live executions. "
             "Accuracy bands are separate from physics safety.", ha="center", fontsize=10)
    fig.text(0.5, 0.89, stats, ha="center", fontsize=9)
    fig.text(0.5, 0.09, f"Validation-frozen recommendation: {summary['recommendation']['role']}; "
             f"release acceptance: {summary['release_acceptance']['recommended_passed']}. "
             "Test does not reselect or fit a controller.", ha="center", fontsize=10)
    fig.text(0.5, 0.035, FOOTNOTE, ha="center", fontsize=8, wrap=True)
    fig.subplots_adjust(top=0.83, bottom=0.19, hspace=0.1)
    return fig


def first_case_figure(plt, summary, traces):
    from matplotlib.ticker import MaxNLocator
    target = summary["protocol"]["targets_ml"][0]
    seed = summary["protocol"]["test_seeds"][0]
    fig, axes = plt.subplots(2, 1, figsize=(12.8, 7.4), sharex=True,
                             gridspec_kw={"height_ratios": [1.5, 1]})
    for role, label, color in (("candidate", "Precision candidate", "#2269A8"),
                               ("baseline", "Original feedback baseline", "#D5822B")):
        steps = traces[role]
        times = [step["physical_time_s"] for step in steps]
        axes[0].plot(times, [step["truth"]["received_ml"] for step in steps], label=label, color=color)
        if role == "candidate":
            axes[1].step(times, [step.get("pulse_count") or 0 for step in steps], where="post", color=color)
    axes[0].axhline(target, color="#555555", ls="--", lw=1, label=f"Target: {target:g} mL")
    axes[0].axhspan(target - 0.1, target + 0.1, color="#DDF2E7", alpha=0.8)
    axes[0].set_ylabel("Received particle-mass volume (mL)")
    axes[0].legend(frameon=False, fontsize=9)
    axes[1].set(xlabel="Original physical time (s)", ylabel="Candidate pulse count")
    axes[1].yaxis.set_major_locator(MaxNLocator(integer=True))
    for ax in axes:
        ax.grid(alpha=0.18)
    fig.suptitle(f"Predeclared first test case: {target:g} mL / seed {seed} (not selected for best error)",
                 fontsize=14, y=0.97)
    fig.text(0.5, 0.035, FOOTNOTE, ha="center", fontsize=8, wrap=True)
    fig.subplots_adjust(top=0.88, bottom=0.16, hspace=0.12)
    return fig


def per_target_comparison(summary, pairs, provenance):
    """Return all nine held-out cases plus three target-level comparisons.

    Physical time is the primary time measure. Wall-clock measurements are
    retained separately and explicitly depend on the machine's concurrent load.
    This pure helper does not write files; the caller must audit the run first.
    """
    _require(summary.get("complete") is True and summary.get("total_recorded_episodes") == 48,
             "Per-target final tables require the completed 48-episode run.")
    protocol = summary["protocol"]
    targets, seeds = protocol["targets_ml"], protocol["test_seeds"]
    expected = {(target, seed) for target in targets for seed in seeds}
    _require(targets == [20, 30, 40] and len(seeds) == len(set(seeds)) == 3
             and len(pairs) == 9 and {_result_key(pair) for pair in pairs} == expected,
             "Per-target tables require all nine same-seed paired test cases.")
    groups, rows = [], []
    for target in targets:
        cases = sorted((pair for pair in pairs if pair["target_ml"] == target),
                       key=lambda pair: seeds.index(pair["seed"]))
        _require(len(cases) == 3, "Each target requires exactly three test seeds.")
        metrics = {}
        for role in ("candidate", "baseline"):
            results = [case[role] for case in cases]
            _require(all(_result_key(result) == _result_key(case) for case, result in zip(cases, results)),
                     "A per-target candidate/baseline uses a different seed or target.")
            aggregate = _metrics(results)
            metrics[role] = {**aggregate,
                "mean_physical_seconds": aggregate["physical_seconds"] / 3,
                "mean_wall_seconds": sum(_number(result["wall_seconds"]) for result in results) / 3,
                "signed_error_ml_by_seed": [{"seed": case["seed"], "error_ml": result["error_ml"]}
                                            for case, result in zip(cases, results)]}
        groups.append({"target_ml": target, "conditions": 3, "test_seeds": list(seeds),
                       "particle_volume_quantum_ml": QUANTUM_ML,
                       "candidate_metrics": metrics["candidate"], "baseline_metrics": metrics["baseline"],
                       "cases": cases})
        # One CSV row for every paired case. Per-target aggregates are repeated
        # so filtering to a target retains its complete interpretation.
        for case in cases:
            row = {"target_ml": target, "seed": case["seed"], "cases_per_target": 3,
                   "particle_volume_quantum_ml": QUANTUM_ML}
            for role in ("candidate", "baseline"):
                result, aggregate = case[role], metrics[role]
                row.update({role + "_" + key: result[key] for key in (
                    "received_ml", "error_ml", "physical_time_s", "wall_seconds",
                    "valid_physics", "complete", "upright", "mass_conserved")})
                row[role + "_safe_completion"] = _safe(result)
                row[role + "_safe_and_within_0_1_ml"] = _safe(result) and abs(result["error_ml"]) <= 0.1
                row.update({role + "_target_" + key: aggregate[key] for key in (
                    "mae_ml", "worst_absolute_error_ml", "mean_physical_seconds", "mean_wall_seconds",
                    "safe_completion_rate", "within_0_1_ml_rate")})
            row["wall_time_notice"] = "Machine-load-dependent; physical time is the primary comparison"
            rows.append(row)
    report = {"complete": True, "protocol_id": summary["protocol_id"],
        "completed_at_utc": summary["completed_at_utc"], "provenance": provenance,
        "total_live_run_episodes": 48, "paired_test_cases": 9, "test_initial_seeds": 3,
        "cases_per_target": 3, "particle_volume_quantum_ml": QUANTUM_ML,
        "comparison": "Same target and initial-state seed, candidate versus original feedback baseline",
        "primary_time_measure": "original physical seconds, without video retiming",
        "wall_time_notice": "Per-episode wall time is machine-load-dependent; do not treat it as intrinsic controller speed.",
        "all_cases_retained": True, "physics_safety_separate_from_dose_accuracy": True,
        "recommendation_role": summary["recommendation"]["role"],
        "recommendation_selected_on_validation_only": True,
        "test_only_checks_release": True, "targets": groups}
    return report, rows


def write_per_target_comparison(output, summary, pairs, provenance):
    report, rows = per_target_comparison(summary, pairs, provenance)
    output = Path(output)
    json_path = output / "per_target_comparison.json"
    csv_path = output / "per_target_comparison.csv"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    with csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return {json_path.name: _sha(json_path), csv_path.name: _sha(csv_path)}


def self_test():
    import tempfile
    cache = ROOT / "_cache/pour_training"
    cache.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=cache) as temporary:
        try:
            load_completed_run(temporary)
        except IncompleteRun:
            pass
        else:
            raise AssertionError("Missing summary was accepted.")
        (Path(temporary) / "summary.json").write_text('{"complete": false}', encoding="utf-8")
        try:
            load_completed_run(temporary)
        except IncompleteRun:
            pass
        else:
            raise AssertionError("Partial summary was accepted.")
        assert not list(Path(temporary).glob("*.png"))
    base = {"valid_physics": True, "mass_conserved": True, "complete": True, "upright": True,
            "max_spilled_ml": 0.0, "max_below_table_particles": 0, "inflight_ml": 0.0,
            "physical_steps": 60, "physical_time_s": 1.0, "error_ml": 0.3}
    failure = {**base, "valid_physics": False, "complete": False, "error_ml": 4.0}
    metrics = _metrics([base, failure])
    assert _safe(base) and not _safe(failure)
    assert metrics["episodes"] == 2 and metrics["mae_ml"] == 2.15
    assert metrics["worst_absolute_error_ml"] == 4.0 and metrics["safe_completion_rate"] == 0.5
    assert metrics["within_0_1_ml_rate"] == 0.0
    assert not _release(metrics)
    try:
        _check_metrics({**metrics, "episodes": 1}, [base, failure])
    except ValueError:
        pass
    else:
        raise AssertionError("An omitted failed episode was accepted.")
    try:
        per_target_comparison({"complete": False}, [], {})
    except ValueError:
        pass
    else:
        raise AssertionError("Partial per-target final report was accepted.")
    print("PASS: partial-run guards; failures retained; safety and accuracy separated; aggregate mismatch rejected.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", nargs="?", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--deps", type=Path, default=DEFAULT_DEPS)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--with-trace", action="store_true", help="Add the predeclared first test case as a third PNG.")
    parser.add_argument("--tables-only", action="store_true", help="Audit the complete run and export per-target CSV/JSON without plotting.")
    parser.add_argument("--frozen-only", action="store_true", help="Validate frozen snapshots without requiring the current workspace to stay unchanged.")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    if args.self_test:
        self_test()
        return 0
    summary, pairs, traces, provenance = load_completed_run(args.run, verify_workspace=not args.frozen_only)
    if args.check_only:
        print(json.dumps(provenance, indent=2))
        return 0
    figures = []
    if not args.tables_only:
        plt = _matplotlib(args.deps)
        figures = [("precision_calibration.png", calibration_figure(plt, summary)),
                   ("precision_heldout_errors_time.png", heldout_figure(plt, summary, pairs))]
        if args.with_trace:
            figures.append(("precision_first_test_trace.png", first_case_figure(plt, summary, traces)))
    output = (args.output or args.run).resolve()
    output.mkdir(parents=True, exist_ok=True)
    for name, figure in figures:
        path = output / name
        figure.savefig(path, dpi=170, metadata={"Description": FOOTNOTE + " Protocol " + summary["protocol_id"]})
        plt.close(figure)
    provenance["figures"] = {name: _sha(output / name) for name, _figure in figures}
    provenance["per_target_tables"] = write_per_target_comparison(output, summary, pairs, dict(provenance))
    (output / "precision_plot_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "figures": list(provenance["figures"]),
                      "per_target_tables": list(provenance["per_target_tables"])}, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, KeyError, StopIteration) as error:
        print(f"Final plots refused: {error}", file=sys.stderr)
        raise SystemExit(2)
