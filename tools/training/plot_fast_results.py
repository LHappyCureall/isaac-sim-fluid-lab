"""Plot a completed, independently audited 138-episode V3 CEM experiment.

Only standard Python and cached Matplotlib are used. No active experiment
module, Kit, solver, GPU, policy fit, or deployment is imported or performed.
Partial runs are rejected before the episode journal or heldout data are read.
"""

import argparse
import csv
from datetime import datetime
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN = ROOT / "droplet_out/pour_policy_fast_v3/run_20261122_cem5s"
DEPS = ROOT / "_cache/pour_training/plot_deps"
QUANTUM_ML = 0.042875
TARGETS = (20, 30, 40)
ROLES = ("candidate", "baseline")
ALGORITHM = "feasibility_first_CEM_speed_and_stop_feedback_v3"
FOOTNOTE = ("Privileged particle-mass volume; quantum 0.042875 mL; dt 1/60 s. "
            "Fixed kinematic pre-grasp/path, narrow position/viscosity randomization; no vision/hardware calibration.")
COLORS = {"candidate": "#137fbd", "baseline": "#da7b25"}


class IncompleteRun(ValueError):
    """The declared final dataset or its independent audit is unavailable."""


def require(condition, message):
    if not condition:
        raise ValueError(message)


def number(value):
    require(type(value) in (int, float) and math.isfinite(value), "Expected a finite recorded number.")
    return float(value)


def json_read(path):
    def reject(value):
        raise ValueError("Nonfinite JSON token: " + value)
    return json.loads(Path(path).read_text(encoding="utf-8-sig"), parse_constant=reject)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode("utf-8")).hexdigest()


def inside(run, relative):
    path = (run / relative).resolve()
    require(path.is_relative_to(run) and path.is_file(), "Missing or escaped artifact: " + str(relative))
    return path


def close(left, right, message):
    require(math.isclose(number(left), number(right), rel_tol=1e-10, abs_tol=1e-8), message)


def metrics(records):
    require(bool(records), "Cannot aggregate an empty group.")
    n = len(records)
    assessments = [row["assessment"] for row in records]
    errors = [number(item["absolute_error_ml"]) for item in assessments]
    return {"episodes": n, "mae_ml": sum(errors)/n, "worst_absolute_error_ml": max(errors),
            "unsafe_count": sum(not item["physical_safe"] for item in assessments),
            "deadline_or_home_or_late_drop_failures": sum(
                not (item["completed_by_deadline"] and item["late_drop_passed"]) for item in assessments),
            "feasible_count": sum(item["constraint_feasible"] for item in assessments),
            "missing_deadline_measurements": sum(not item["deadline_dose_measurement_available"] for item in assessments),
            "within_0_1_ml_count": sum(item["success_within_0_1_ml"] for item in assessments),
            "within_0_2_ml_count": sum(item["constraint_feasible"] and error <= .2
                                        for item, error in zip(assessments, errors)),
            "all_mass_conserved": all(row["result"]["mass_conserved"] for row in records),
            "mean_action_physical_seconds": sum(number(row["result"]["physical_time_s"]) for row in records)/n,
            "physical_steps": sum(row["result"]["physical_steps"] for row in records),
            "action_physical_seconds": sum(number(row["result"]["physical_time_s"]) for row in records),
            "post_observation_physical_seconds": sum(row["post_observation"].get("duration_s", 0)
                for row in records if isinstance(row["post_observation"], dict))}


def check_metrics(recorded, rows):
    actual = metrics(rows)
    for key, value in actual.items():
        require(key in recorded, "Missing aggregate: " + key)
        if isinstance(value, bool):
            require(recorded[key] is value, "Aggregate differs: " + key)
        else:
            close(recorded[key], value, "Aggregate differs: " + key)
    return actual


def selection_key(rows):
    values = metrics(rows)
    groups = [rows] + [[row for row in rows if row["conditions"]["target_ml"] == target] for target in TARGETS]
    violation = sum(max(0., metrics(group)["mae_ml"]-.1)
                    + max(0., metrics(group)["worst_absolute_error_ml"]-.2) for group in groups if group)
    return [values["unsafe_count"], values["deadline_or_home_or_late_drop_failures"], violation,
            values["mae_ml"] + .5*values["worst_absolute_error_ml"], values["mean_action_physical_seconds"]]


def check_key(recorded, actual):
    require(len(recorded) == len(actual), "Selection key shape differs.")
    for a, b in zip(recorded, actual):
        close(a, b, "Selection key differs.")


def release_passed(rows):
    groups = [rows] + [[row for row in rows if row["conditions"]["target_ml"] == target] for target in TARGETS]
    return all(group and (lambda m: m["unsafe_count"] == 0
               and m["deadline_or_home_or_late_drop_failures"] == 0
               and m["mae_ml"] <= .1 and m["worst_absolute_error_ml"] <= .2
               and m["all_mass_conserved"])(metrics(group)) for group in groups)


def artifact_check(run, row, protocol, *, read_post=False):
    """Independently check artifact bytes/footer; the audit verifies every step."""
    for prefix in ("record", "motion_record", "post_record"):
        relative = row.get(prefix + "_path")
        if relative is not None:
            require(sha(inside(run, relative)) == row[prefix + "_sha256"], "Changed artifact SHA: " + relative)
        else:
            require(prefix == "post_record" and row.get(prefix + "_sha256") is None,
                    "Missing action/motion artifact.")
    motion = json_read(inside(run, row["motion_record_path"]))
    require(motion == row["motion_evidence"] and motion["action_result_modified"] is False,
            "Motion sidecar differs from journal or rewrites action result.")
    with gzip.open(inside(run, row["record_path"]), "rt", encoding="utf-8") as stream:
        header = json.loads(next(stream))
        effective = json.loads(next(stream))
        last = None
        for line in stream:
            last = line
    require(header["type"] == "episode" and effective["type"] == "effective_fast_sources",
            "Missing pre-physics controller identity.")
    condition = row["conditions"]
    require((header["target_ml"], header["seed"]) == (condition["target_ml"], condition["seed"]),
            "Action header conditions differ.")
    close(header["particle_volume_ml"], QUANTUM_ML, "Particle-volume quantum differs.")
    close(header["physics_dt_s"], 1/60, "Physical dt differs.")
    require(effective["source_sha256"] == {name: protocol["source_sha256"][
                "assets/robot_testtube_pour/scripts/"+name] for name in ("fast_control.py", "fast_geometry.py")}
            and effective["cache_sha256"] == protocol["input_files"]["path_cache"]["sha256"],
            "Executed controller/geometry/cache identity differs.")
    require(effective["reset_started_utc"] == row["episode_wall_clock_start_utc"], "Episode start UTC differs.")
    footer = json.loads(last)
    require(footer.pop("type") == "result" and footer == row["result"], "Action footer was changed or omitted.")
    detail = {"deadline_received_ml": row["post_observation"].get("deadline_received_ml")
              if isinstance(row["post_observation"], dict) else None,
              "final_post_received_ml": None, "final_post_physical_time_s": None,
              "max_received_change_after_5s_ml": None, "delta_5s_to_last_post_ml": None}
    if read_post and row.get("post_record_path"):
        received_at_5 = row["result"]["received_ml"] if abs(row["result"]["physical_time_s"]-5) < 1e-10 else None
        observations = []
        with gzip.open(inside(run, row["post_record_path"]), "rt", encoding="utf-8") as stream:
            for line in stream:
                item = json.loads(line)
                require(item["type"] == "post_observation", "Unexpected post event.")
                observations.append((number(item["physical_time_s"]), number(item["measurement"]["received_ml"])))
                if abs(item["physical_time_s"]-5) < 1e-10:
                    received_at_5 = item["measurement"]["received_ml"]
        if observations:
            detail.update(final_post_received_ml=observations[-1][1], final_post_physical_time_s=observations[-1][0])
        if detail["deadline_received_ml"] is not None:
            require(received_at_5 is not None, "Journal claims a missing direct 5-second observation.")
            close(received_at_5, detail["deadline_received_ml"], "Direct 5-second sample differs.")
            later = [abs(ml-received_at_5) for t, ml in observations if t >= 5-1e-10]
            detail["max_received_change_after_5s_ml"] = max(later, default=0.)
            detail["delta_5s_to_last_post_ml"] = (observations[-1][1]-received_at_5) if observations else None
    return detail


def load_completed_run(directory, *, frozen_only=False):
    run = Path(directory).resolve()
    summary_path = run / "summary.json"
    # This gate deliberately precedes all journal, heldout, and audit reads.
    if not summary_path.is_file():
        raise IncompleteRun("Final summary.json is not available; no journal/test data read and no final artifacts generated.")
    summary = json_read(summary_path)
    if summary.get("complete") is not True or summary.get("total_recorded_episodes") != 138:
        raise IncompleteRun("Expected the complete 138-episode summary; no journal/test data read.")
    audit_path = run / "fast_run_audit.json"
    if not audit_path.is_file():
        raise IncompleteRun("The completed run requires its independent fast_run_audit.json before final plotting.")
    audit = json_read(audit_path)
    require(audit.get("artifact_integrity_verified") is True and audit.get("all_failures_retained") is True
            and audit.get("episodes_audited") == 138 and audit.get("raw_action_trace_count") == 138,
            "Independent audit is incomplete or failed artifact integrity.")
    require(audit["summary_sha256"] == sha(summary_path)
            and audit["journal_sha256"] == sha(run / "episodes.jsonl"), "Audit describes different summary/journal bytes.")
    envelope = json_read(run / "protocol.json")
    protocol = envelope["protocol"]
    pid = signature(protocol)
    require(envelope["protocol_id"] == summary["protocol_id"] == audit["protocol_id"] == pid
            and summary["protocol"] == protocol, "Protocol identity differs.")
    require(protocol["algorithm"] == ALGORITHM and protocol["neural_policy"] is False and protocol["ppo"] is False,
            "This plotter handles the declared parameter-CEM experiment only.")
    require(protocol["budget"] == {"generations": 3, "population": 6, "elites": 2,
                "training": 108, "validation": 12, "test": 18, "total": 138}
            and protocol["targets_ml"] == list(TARGETS), "The declared experiment budget differs.")
    require(audit["budgets"] == {"training": 108, "validation": 12, "test": 18, "total": 138},
            "Audit budget differs.")
    seed_groups = [set(protocol[name+"_seeds"]) for name in ("train", "validation", "test")]
    require([len(group) for group in seed_groups] == [2, 2, 3]
            and not any(seed_groups[i] & seed_groups[j] for i in range(3) for j in range(i)), "Seed split is not isolated.")
    frozen = json_read(run / "input_freeze.json")
    require(signature(frozen["declaration"]) == frozen["input_id"] == protocol["input_id"], "Input freeze differs.")
    require(all(protocol.get(key) == value for key, value in frozen["declaration"].items()), "Declaration/protocol changed.")
    require(len(protocol["source_sha256"]) == 20, "Expected all 20 frozen source files.")
    for relative, expected in protocol["source_sha256"].items():
        require(sha(inside(run, "source_snapshot/"+relative)) == expected, "Frozen source changed: " + relative)
        if not frozen_only:
            require(sha(ROOT / relative) == expected, "Active source changed: " + relative)
    for name, entry in protocol["input_files"].items():
        path = "input_snapshot/" + name + Path(entry["path"]).suffix
        require(sha(inside(run, path)) == entry["sha256"], "Frozen input/cache payload changed: " + name)
    freeze = json_read(run / "freeze.json")
    require(freeze["protocol_id"] == pid and freeze["test_used_for_selection"] is False,
            "Model freeze must precede and exclude heldout selection.")
    for role in ("candidate", "recommended"):
        model = summary["models"][role]
        path = inside(run, model["file"])
        require(sha(path) == model["sha256"] == freeze[role+"_sha256"], "Frozen model bytes changed.")
        checkpoint = json_read(path)
        require(checkpoint["parameters"] == model["parameters"] == freeze[role+"_params"]
                and checkpoint["metadata"]["test_used_for_selection"] is False
                and checkpoint["metadata"]["source_sha256"] == protocol["source_sha256"], "Frozen model metadata differs.")
    records = {}
    with (run / "episodes.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            name = row["episode_id"]
            require(name not in records and row["protocol_id"] == pid, "Duplicate/foreign episode.")
            require(signature(row["conditions"]) == row["conditions_sha256"], "Episode conditions changed.")
            records[name] = row
    expected = {f"train_g{g:03d}_c{c:03d}_t{t}_s{s}" for g in range(3) for c in range(6)
                for t in TARGETS for s in protocol["train_seeds"]}
    expected |= {f"{stage}_{role}_t{t}_s{s}" for stage in ("validation", "test") for role in ROLES
                 for t in TARGETS for s in protocol[stage+"_seeds"]}
    require(set(records) == expected and len(records) == 138 and summary["all_episode_ids"] == list(records),
            "The 138-case journal/summary has an omission or extra record.")
    details = {}
    for row in records.values():
        c, a, result = row["conditions"], row["assessment"], row["result"]
        require((result["target_ml"], result["seed"]) == (c["target_ml"], c["seed"]), "Episode condition/result differs.")
        require(result["physical_steps"] <= 300, "Action exceeded the five-second clock.")
        close(result["physical_time_s"], result["physical_steps"]/60, "Physical time must be original step/60.")
        close(a["absolute_error_ml"], abs(a["dose_error_ml"]), "Signed and absolute error differ.")
        require(a["constraint_feasible"] == (a["physical_safe"] and a["completed_by_deadline"] and a["late_drop_passed"])
                and a["success_within_0_1_ml"] == (a["constraint_feasible"] and abs(a["dose_error_ml"]) <= .1),
                "Accuracy and safety/deadline flags differ.")
        detail = artifact_check(run, row, protocol, read_post=c["stage"] == "test")
        available = a["deadline_dose_measurement_available"]
        require(available == (type(detail["deadline_received_ml"]) in (int, float)), "5-second measurement availability differs.")
        basis = "physical_5s" if available else "invalid_terminal_fallback_not_a_5s_measurement"
        require(a["dose_measurement_basis"] == basis, "Fallback dose was relabeled as a 5-second measurement.")
        dose = detail["deadline_received_ml"] if available else result["received_ml"]
        close(a["dose_error_ml"], dose-c["target_ml"], "Recorded dose error differs from its declared time basis.")
        details[row["episode_id"]] = detail
        if c["stage"] == "test":
            require(datetime.fromisoformat(row["episode_wall_clock_start_utc"]) > datetime.fromisoformat(freeze["frozen_at_utc"]),
                    "A holdout reset preceded the model freeze.")
    train = [row for row in records.values() if row["conditions"]["stage"] == "train"]
    check_metrics(summary["training"]["metrics"], train)
    check_metrics(summary["all_rollouts"], list(records.values()))
    generations = summary["training"]["generations"]
    require(len(generations) == 3 and [gen["generation"] for gen in generations] == [0, 1, 2], "Missing/reordered generations.")
    all_candidates = []
    for generation in generations:
        require(len(generation["candidates"]) == 6, "A generation omits failed candidates.")
        require(json_read(run / f"generation_{generation['generation']:03d}.json") == {"protocol_id": pid, **generation},
                "Generation source file/summary differs.")
        for candidate in generation["candidates"]:
            group = [records[name] for name in candidate["episode_ids"]]
            require(len(group) == 6 and all(row["conditions"]["generation"] == generation["generation"]
                and row["conditions"]["candidate"] == candidate["index"]
                and row["conditions"]["policy_params"] == candidate["params"] for row in group),
                "Candidate rows differ from the declared generation.")
            check_metrics(candidate["metrics"], group)
            check_key(candidate["selection_key"], selection_key(group))
            all_candidates.append(candidate)
    best = min(all_candidates, key=lambda item: item["selection_key"])
    require(best["params"] == summary["training"]["best_params"] == freeze["candidate_params"],
            "Frozen candidate was not the cumulative feasibility-first train optimum.")
    check_key(summary["training"]["best_key"], best["selection_key"])
    val = {role: [row for row in records.values() if row["conditions"]["stage"] == "validation"
                  and row["conditions"]["policy_role"] == role] for role in ROLES}
    val_keys = {role: selection_key(group) for role, group in val.items()}
    for role, group in val.items():
        require(len(group) == 6, "Validation budget differs.")
        check_metrics(summary["validation"]["metrics"][role], group)
        check_key(summary["validation"]["selection_keys"][role], val_keys[role])
    baseline = protocol["cem"]["baseline_params"]
    recommendation = "candidate" if best["params"] != baseline and val_keys["candidate"] < val_keys["baseline"] else "baseline"
    require(summary["recommendation"]["role"] == freeze["recommended_role"] == recommendation
            and summary["recommendation"]["selected_on_validation_only"] is True
            and summary["test_used_for_training_or_selection"] is False, "Recommendation was not fixed on validation.")
    pairs = summary["test"]["paired_results"]
    require(len(pairs) == 9 and {(row["target_ml"], row["seed"]) for row in pairs}
            == {(t, s) for t in TARGETS for s in protocol["test_seeds"]}, "Missing heldout target/seed pair.")
    tests = {role: [] for role in ROLES}
    for pair in pairs:
        for role in ROLES:
            row = pair[role]
            require(row == records[row["episode_id"]]
                    and (row["conditions"]["target_ml"], row["conditions"]["seed"]) == (pair["target_ml"], pair["seed"]),
                    "Summary pair differs from the full journal record.")
            require(row["conditions"]["policy_params"] == (best["params"] if role == "candidate" else baseline),
                    "Heldout policy changed from its pre-test parameters.")
            tests[role].append(row)
    for role, group in tests.items():
        check_metrics(summary["test"]["metrics"][role], group)
        require(summary["release_acceptance"][role+"_passed"] is release_passed(group), "Release outcome differs.")
    check_metrics(summary["test"]["recommended_metrics"], tests[recommendation])
    require(summary["release_acceptance"]["recommended_passed"] is release_passed(tests[recommendation])
            and summary["release_acceptance"]["does_not_reselect_model"] is True
            and summary["automatic_gui_deployment"] is False, "Test must not refit, rescue, or deploy the model.")
    return {"run": run, "summary": summary, "protocol": protocol, "records": records,
            "details": details, "audit": audit, "audit_path": audit_path, "frozen_only": frozen_only}


def compact_case(row, detail):
    a, r, c = row["assessment"], row["result"], row["conditions"]
    post = row["post_observation"] if isinstance(row["post_observation"], dict) else {}
    post_check = row["post_trace_check"] or {}
    total_spilled = max(row["trace_check"]["maximum_total_spilled_particles"],
                        post_check.get("maximum_total_spilled_particles", 0)) * QUANTUM_ML
    source_spilled = max(number(r["max_spilled_ml"]),
                         post_check.get("maximum_source_spilled_particles", 0)*QUANTUM_ML)
    return {"episode_id": row["episode_id"], "target_ml": c["target_ml"], "seed": c["seed"],
            "role": c["policy_role"], "dose_received_at_5s_ml": detail["deadline_received_ml"],
            "signed_error_at_5s_ml": a["dose_error_ml"] if a["deadline_dose_measurement_available"] else None,
            "signed_error_for_descriptive_loss_ml": a["dose_error_ml"],
            "descriptive_error_basis": a["dose_measurement_basis"],
            "terminal_fallback_received_ml": None if a["deadline_dose_measurement_available"] else r["received_ml"],
            "action_physical_seconds": r["physical_time_s"], "action_wall_seconds": r["wall_seconds"],
            "total_action_plus_post_wall_seconds": row["total_action_plus_post_wall_seconds"],
            "physical_safe": a["physical_safe"], "completed_by_5s": a["completed_by_deadline"],
            "home_stable_at_action_end": a["home_stable"], "late_drop_passed": a["late_drop_passed"],
            "constraint_feasible": a["constraint_feasible"], "success_within_0_1_ml": a["success_within_0_1_ml"],
            "max_total_water_spilled_ml": total_spilled, "max_source_water_spilled_ml": source_spilled,
            "raw_action_termination_reason": r["termination_reason"], "raw_action_error": r.get("error"),
            "post_observation_error": post.get("error"),
            "post_max_change_since_action_end_ml": post.get("maximum_received_change_ml"),
            **detail, "particle_quantum_ml": QUANTUM_ML}


def comparison_metrics(rows, compact):
    base = metrics(rows)
    n = len(rows)
    available = [row for row in compact if row["signed_error_at_5s_ml"] is not None]
    return {**base, "descriptive_mae_includes_failed_terminal_fallbacks": True,
            "direct_5s_samples": len(available), "terminal_fallback_count": n-len(available),
            "direct_5s_sample_mae_ml": sum(abs(row["signed_error_at_5s_ml"]) for row in available)/len(available) if available else None,
            "completed_by_5s_count": sum(row["completed_by_5s"] for row in compact),
            "late_drop_failed_count": sum(not row["late_drop_passed"] for row in compact),
            "mean_action_wall_seconds": sum(row["action_wall_seconds"] for row in compact)/n,
            "mean_action_plus_post_wall_seconds": sum(row["total_action_plus_post_wall_seconds"] for row in compact)/n,
            "mean_max_total_water_spilled_ml": sum(row["max_total_water_spilled_ml"] for row in compact)/n,
            "worst_total_water_spilled_ml": max(row["max_total_water_spilled_ml"] for row in compact),
            "mean_max_source_water_spilled_ml": sum(row["max_source_water_spilled_ml"] for row in compact)/n,
            "signed_error_for_descriptive_loss_ml": [row["signed_error_for_descriptive_loss_ml"] for row in compact],
            "signed_error_at_5s_ml": [row["signed_error_at_5s_ml"] for row in compact],
            "seeds": [row["seed"] for row in compact]}


def build_report(data):
    records, summary, protocol = data["records"], data["summary"], data["protocol"]
    pairs = sorted(summary["test"]["paired_results"], key=lambda pair: (pair["target_ml"], pair["seed"]))
    compact_pairs = []
    for pair in pairs:
        compact_pairs.append({"target_ml": pair["target_ml"], "seed": pair["seed"],
                              **{role: compact_case(pair[role], data["details"][pair[role]["episode_id"]]) for role in ROLES}})
    per_target = []
    for target in TARGETS:
        entry = {"target_ml": target, "independent_target_seed_conditions": 3,
                 "executed_live_episodes": 6}
        for role in ROLES:
            group = [pair[role] for pair in pairs if pair["target_ml"] == target]
            compact = [pair[role] for pair in compact_pairs if pair["target_ml"] == target]
            entry[role] = comparison_metrics(group, compact)
        per_target.append(entry)
    overall = {role: comparison_metrics([pair[role] for pair in pairs], [pair[role] for pair in compact_pairs]) for role in ROLES}
    curve = []
    cumulative = None
    for generation in summary["training"]["generations"]:
        current = min(generation["candidates"], key=lambda row: row["selection_key"])
        if cumulative is None or current["selection_key"] < cumulative["selection_key"]:
            cumulative = current
        rows = [records[name] for candidate in generation["candidates"] for name in candidate["episode_ids"]]
        curve.append({"generation": generation["generation"]+1, "population_metrics": metrics(rows),
                      "generation_best_metrics": current["metrics"], "generation_best_selection_key": current["selection_key"],
                      "cumulative_lexicographic_best_metrics": cumulative["metrics"],
                      "cumulative_lexicographic_best_selection_key": cumulative["selection_key"],
                      "population_mean_action_plus_post_wall_seconds": sum(row["total_action_plus_post_wall_seconds"] for row in rows)/len(rows),
                      "candidate_maes_ml": [candidate["metrics"]["mae_ml"] for candidate in generation["candidates"]]})
    stages = {stage: {role: metrics(group) for role in ROLES
                     if (group := [row for row in records.values() if row["conditions"]["stage"] == stage
                                   and row["conditions"]["policy_role"] == role])}
              for stage in ("train", "validation", "test")}
    return {"schema_version": 1, "protocol_id": summary["protocol_id"], "complete_declared_episodes": 138,
            "budget": protocol["budget"], "algorithm": protocol["algorithm"], "neural_policy": False, "ppo": False,
            "independent_test_conditions": 9, "executed_test_episodes": 18,
            "recommended_role_selected_on_validation": summary["recommendation"]["role"],
            "candidate_parameters": summary["models"]["candidate"]["parameters"],
            "baseline_parameters": protocol["cem"]["baseline_params"], "release_acceptance": summary["release_acceptance"],
            "source_sha256": protocol["source_sha256"], "models": summary["models"],
            "particle_quantum_ml": QUANTUM_ML, "action_deadline_physical_seconds": 5.,
            "post_observation_cannot_rescue_deadline_or_precision": True,
            "wall_seconds_depend_on_machine_load": True, "unsafe_and_deadline_counts_can_overlap": True,
            "fallback_measurements_are_explicitly_flagged_and_never_pass": True,
            "training_curve": curve, "stage_metrics": stages, "overall_test_comparison": overall,
            "per_target_comparison": per_target, "all_nine_paired_cases_including_failures": compact_pairs,
            "limitations": summary["limitations"] + [FOOTNOTE,
                "Descriptive MAE retains terminal fallback errors when a direct 5-second sample is unavailable; those episodes remain failed.",
                "The recommended role reuses candidate/baseline cases; it is not a third rollout set.",
                "Wall-clock time is descriptive and depends on machine load; physical action time is the primary timing measure."]}


def plotting():
    if DEPS.is_dir():
        sys.path.insert(0, str(DEPS))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.titlesize": 11, "figure.dpi": 135})
    return plt, np


def save_figure(fig, path, subtitle):
    fig.text(.02, .045, subtitle, fontsize=8)
    fig.text(.02, .018, FOOTNOTE, fontsize=7.5, color="#444444")
    fig.tight_layout(rect=(0, .08, 1, .94), h_pad=2.0, w_pad=2.0)
    fig.savefig(path, facecolor="white")


def plot_training(report, path):
    plt, np = plotting()
    curve = report["training_curve"]
    x = np.arange(1, 4)
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.3))
    ax = axes[0, 0]
    for index, generation in enumerate(curve):
        ax.scatter(np.full(6, index+1), generation["candidate_maes_ml"], s=25, color="#aab2bc", alpha=.8)
    for key, label, color in (("population_metrics", "Population (36 episodes/gen)", "#7b8796"),
                              ("generation_best_metrics", "Generation lexicographic best", "#137fbd"),
                              ("cumulative_lexicographic_best_metrics", "Cumulative lexicographic best", "#319668")):
        ax.plot(x, [row[key]["mae_ml"] for row in curve], "o-", label=label, color=color)
    ax.set(title="Training descriptive MAE (all failures retained)", ylabel="Absolute dose error (mL)")
    ax.legend(fontsize=8)
    ax = axes[0, 1]
    for offset, key, label, color in ((-.25, "unsafe_count", "Unsafe", "#bb4c43"),
                                     (0, "deadline_or_home_or_late_drop_failures", "Deadline / home / late failed", "#da9c25"),
                                     (.25, "feasible_count", "Non-dose gates passed", "#319668")):
        ax.bar(x+offset, [row["population_metrics"][key] for row in curve], width=.23, label=label, color=color)
    ax.set(title="Each generation: 36 live episodes", ylabel="Episodes", ylim=(0, 40))
    ax.legend(fontsize=8)
    ax = axes[1, 0]
    for key, label, color in (("population_metrics", "Population mean", "#7b8796"),
                              ("cumulative_lexicographic_best_metrics", "Cumulative best (6 conditions)", "#319668")):
        ax.plot(x, [row[key]["mean_action_physical_seconds"] for row in curve], "o-", label=label, color=color)
    ax.axhline(5, color="#b54848", linestyle="--", linewidth=1, label="Hard action deadline")
    ax.set(title="Original physical action time", ylabel="Physical seconds", ylim=(0, 5.4))
    ax.legend(fontsize=8)
    ax = axes[1, 1]
    stages = [report["stage_metrics"][stage] for stage in ("train", "validation", "test")]
    counts = [sum(group["episodes"] for group in stage.values()) for stage in stages]
    for offset, key, label, color in ((-.24, "unsafe_count", "Unsafe", "#bb4c43"),
                                     (0, "deadline_or_home_or_late_drop_failures", "Deadline / home / late failed", "#da9c25"),
                                     (.24, "feasible_count", "Non-dose gates passed", "#319668")):
        values = [100*sum(group[key] for group in stage.values())/n for stage, n in zip(stages, counts)]
        ax.bar(x+offset, values, width=.22, color=color, label=label)
    ax.set(title="Stage gates are separate and can overlap", ylabel="Percent of stage episodes", ylim=(0, 112))
    ax.legend(fontsize=8)
    for ax in axes.flat:
        ax.set(xticks=x, xlabel="CEM generation")
        ax.grid(axis="y", alpha=.2)
    axes[1, 1].set(xlabel="Experiment stage", xticklabels=[f"{name}\nn={n}" for name, n in zip(
        ("Train", "Validation", "Test"), counts)])
    fig.suptitle("V3 parameter CEM: 3 generations × 6 candidates × 6 conditions = 108 training episodes", fontsize=14)
    save_figure(fig, path, "12 validation + 18 test episodes follow training. Feasibility-first selection can choose a higher MAE to avoid unsafe/late actions.")
    plt.close(fig)


def plot_targets(report, path):
    plt, np = plotting()
    rows = report["per_target_comparison"]
    x = np.arange(3)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    specs = (("mae_ml", "Descriptive MAE (all 3 seeds)", "mL"),
             ("worst_absolute_error_ml", "Worst absolute descriptive error", "mL"),
             ("mean_action_physical_seconds", "Mean original physical action time", "Physical seconds"),
             ("completed_by_5s_count", "Home + 12 stable frames by 5 s", "Count / 3"),
             ("mean_max_total_water_spilled_ml", "Mean maximum spill: all water", "mL / case"),
             ("mean_action_plus_post_wall_seconds", "Mean action + post wall time (load dependent)", "Wall seconds"))
    for ax, (key, title, unit) in zip(axes.flat, specs):
        for index, role in enumerate(ROLES):
            values = [row[role][key] for row in rows]
            bars = ax.bar(x+(index-.5)*.34, values, width=.32, color=COLORS[role], label=role.title())
            for bar, value in zip(bars, values):
                ax.annotate(f"{value:.3g}", (bar.get_x()+bar.get_width()/2, bar.get_height()),
                            xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8)
        ax.set(title=title, ylabel=unit, xticks=x, xticklabels=["20 mL", "30 mL", "40 mL"])
        ax.grid(axis="y", alpha=.2)
        if key == "completed_by_5s_count":
            ax.set_ylim(0, 3.6)
        if key == "mean_action_physical_seconds":
            ax.axhline(5, color="#b54848", linestyle="--", linewidth=1)
            ax.set_ylim(0, 5.8)
    axes[0, 0].legend(fontsize=8)
    role = report["recommended_role_selected_on_validation"]
    fig.suptitle(f"Heldout target comparison: 3 paired seeds per target; recommendation fixed on validation = {role}", fontsize=14)
    save_figure(fig, path, "All 18 rollouts remain included. Missing 5-second samples use flagged terminal fallback only for descriptive loss; never for success.")
    plt.close(fig)


def plot_pairs(report, path):
    plt, np = plotting()
    pairs = report["all_nine_paired_cases_including_failures"]
    x = np.arange(9)
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    for index, role in enumerate(ROLES):
        offset = (index-.5)*.18
        for j, pair in enumerate(pairs):
            row = pair[role]
            marker = "o" if row["signed_error_at_5s_ml"] is not None else "X"
            label = role.title() if j == 0 else None
            axes[0].scatter(j+offset, row["signed_error_for_descriptive_loss_ml"], marker=marker,
                            color=COLORS[role], s=55, label=label, zorder=3)
            axes[1].scatter(j+offset, row["action_physical_seconds"], color=COLORS[role],
                            marker="o" if row["completed_by_5s"] else "x", s=55, zorder=3)
            late = row["delta_5s_to_last_post_ml"]
            if late is not None:
                axes[2].scatter(j+offset, late, color=COLORS[role], s=45, zorder=3)
            else:
                axes[2].scatter(j+offset, 0, color=COLORS[role], marker="X", s=55, zorder=3)
        for j, pair in enumerate(pairs):
            axes[0].plot([j-.09, j+.09], [pair["candidate"]["signed_error_for_descriptive_loss_ml"],
                                        pair["baseline"]["signed_error_for_descriptive_loss_ml"]], color="#c1c7ce", linewidth=1)
    for value in (-.1, .1):
        axes[0].axhline(value, color="#319668", linestyle="--", linewidth=.8)
    for value in (-.2, .2):
        axes[0].axhline(value, color="#bf9e52", linestyle=":", linewidth=.8)
    axes[0].axhline(0, color="#555555", linewidth=.6)
    axes[0].set(title="Signed error at physical 5 s; X = unavailable, terminal fallback shown and remains failed", ylabel="Dose minus target (mL)")
    axes[0].legend(loc="best")
    axes[1].axhline(5, color="#b54848", linestyle="--", linewidth=1)
    axes[1].set(title="Action end time; x = did not finish home + stable by 5 s", ylabel="Original physical seconds", ylim=(0, 5.5))
    axes[2].axhline(0, color="#555555", linewidth=.6)
    axes[2].set(title="Dose change from 5 s to last retained post observation; X at zero = unavailable", ylabel="Later dose minus 5 s dose (mL)")
    axes[2].set(xticks=x, xticklabels=[f"{pair['target_ml']} mL\nseed {pair['seed']}" for pair in pairs], xlabel="All nine paired initial conditions")
    for ax in axes:
        ax.grid(axis="y", alpha=.2)
        for boundary in (2.5, 5.5):
            ax.axvline(boundary, color="#bbbbbb", linewidth=.8)
    fig.suptitle("All heldout pairs, including unsafe, inaccurate, incomplete and late-drop failures", fontsize=14)
    save_figure(fig, path, "Dashed ±0.1 mL = per-case accuracy; dotted ±0.2 mL = release worst-error threshold. Safety and deadline gates also apply; post data never rescues failure.")
    plt.close(fig)


def write_outputs(data, output, *, tables_only=False):
    report = build_report(data)
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / "fast_comparison_table.json"
    json_path.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    rows = []
    for target in report["per_target_comparison"]:
        for role in ROLES:
            row = {"target_ml": target["target_ml"], "role": role, **target[role]}
            row = {key: json.dumps(value) if isinstance(value, (list, dict)) else value for key, value in row.items()}
            rows.append(row)
    csv_path = output / "fast_per_target_comparison.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    files = [json_path, csv_path]
    if not tables_only:
        for filename, fn in (("fast_training_curve.png", plot_training), ("fast_per_target_comparison.png", plot_targets),
                             ("fast_all_heldout_pairs.png", plot_pairs)):
            path = output / filename
            fn(report, path)
            files.append(path)
    run = data["run"]
    provenance = {"protocol_id": data["summary"]["protocol_id"], "complete_episodes": 138,
                  "summary_sha256": sha(run / "summary.json"), "journal_sha256": sha(run / "episodes.jsonl"),
                  "audit_sha256": sha(data["audit_path"]), "active_sources_checked": not data["frozen_only"],
                  "plot_tool_sha256": sha(Path(__file__)),
                  "artifact_sha256": {path.name: sha(path) for path in files},
                  "no_partial_run_or_missing_failure_filtered": True, "no_policy_fit_or_deployment": True}
    (output / "fast_plot_provenance.json").write_text(json.dumps(provenance, indent=2)+"\n", encoding="utf-8")
    return {"output_directory": str(output), "files": [str(path) for path in files],
            "recommended_role": report["recommended_role_selected_on_validation"],
            "release_acceptance": report["release_acceptance"], "overall": report["overall_test_comparison"]}


def self_test():
    # An unreadable journal proves that rejection occurs before heldout access.
    with tempfile.TemporaryDirectory(dir=ROOT / "_cache/pour_training") as temporary:
        directory = Path(temporary)
        (directory / "summary.json").write_text('{"complete":false,"total_recorded_episodes":137}')
        (directory / "episodes.jsonl").write_text("THIS JOURNAL MUST NEVER BE READ")
        try:
            load_completed_run(directory)
        except IncompleteRun:
            pass
        else:
            raise AssertionError("Partial run was accepted.")
        require(not list(directory.glob("*.png")), "Partial run generated final PNGs.")
    rows = []
    for i in range(3):
        a = {"absolute_error_ml": [.05, .2, 3.][i], "dose_error_ml": [.05, -.2, 3.][i],
             "physical_safe": i != 2, "completed_by_deadline": i != 1, "late_drop_passed": True,
             "constraint_feasible": i == 0, "deadline_dose_measurement_available": i != 2,
             "success_within_0_1_ml": i == 0}
        rows.append({"assessment": a, "result": {"mass_conserved": True, "physical_time_s": [4.5, 5., 1.][i],
                     "physical_steps": [270, 300, 60][i]}, "post_observation": {"duration_s": 2.},
                     "conditions": {"target_ml": 20}})
    aggregate = metrics(rows)
    require(aggregate["episodes"] == 3 and aggregate["unsafe_count"] == 1
            and aggregate["deadline_or_home_or_late_drop_failures"] == 1
            and aggregate["missing_deadline_measurements"] == 1 and aggregate["feasible_count"] == 1,
            "Unsafe, deadline, or missing-5-second failures were dropped or conflated.")
    close(aggregate["mae_ml"], 3.25/3, "Failed terminal fallback disappeared from descriptive loss.")
    require(release_passed(rows) is False, "Post observation rescued a failed release.")
    # Synthetic visual fixtures exercise missing samples, late change, spill,
    # and zero-completion bars without accessing any live experiment evidence.
    pairs, targets = [], []
    for target in TARGETS:
        for index, seed in enumerate((71101, 71102, 71103)):
            pair = {"target_ml": target, "seed": seed}
            for role in ROLES:
                unavailable = role == "baseline" and target == 30 and index == 0
                error = 2.0 if unavailable else (.03*(index-1) if role == "candidate" else .15*(index-1))
                pair[role] = {"signed_error_at_5s_ml": None if unavailable else error,
                              "signed_error_for_descriptive_loss_ml": error,
                              "action_physical_seconds": 4.3 if role == "candidate" else 5.,
                              "completed_by_5s": role == "candidate" and index != 2,
                              "delta_5s_to_last_post_ml": None if unavailable else .042875 if index == 2 else 0.}
            pairs.append(pair)
        entry = {"target_ml": target}
        for role in ROLES:
            entry[role] = {"mae_ml": .02 if role == "candidate" else .2,
                           "worst_absolute_error_ml": .03 if role == "candidate" else 2.,
                           "mean_action_physical_seconds": 4.3 if role == "candidate" else 5.,
                           "completed_by_5s_count": 2 if role == "candidate" else 0,
                           "mean_max_total_water_spilled_ml": .0 if role == "candidate" else .042875,
                           "mean_action_plus_post_wall_seconds": 12. if role == "candidate" else 11.}
        targets.append(entry)
    fixture = {"recommended_role_selected_on_validation": "candidate", "per_target_comparison": targets,
               "all_nine_paired_cases_including_failures": pairs,
               "stage_metrics": {stage: {"candidate": aggregate} for stage in ("train", "validation", "test")},
               "training_curve": [{"candidate_maes_ml": [.1, .2, .3, .4, .5, .6],
                   "population_metrics": {**aggregate, "unsafe_count": 12, "feasible_count": 12,
                       "deadline_or_home_or_late_drop_failures": 12},
                   "generation_best_metrics": aggregate, "cumulative_lexicographic_best_metrics": aggregate,
                   "population_mean_action_plus_post_wall_seconds": 12.} for _ in range(3)]}
    with tempfile.TemporaryDirectory(dir=ROOT / "_cache/pour_training") as temporary:
        for index, fn in enumerate((plot_training, plot_targets, plot_pairs)):
            path = Path(temporary) / f"synthetic_fixture_{index}.png"
            fn(fixture, path)
            require(path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n" and path.stat().st_size > 20000,
                    "Matplotlib fixture did not produce a usable PNG.")
    print(json.dumps({"self_test_passed": True, "checks": ["partial rejected before journal/test reads",
        "no partial PNG", "unsafe and deadline kept separate", "missing-5s fallback retained and cannot pass",
        "three synthetic Matplotlib figures rendered; no live results read"]}))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--tables-only", action="store_true")
    parser.add_argument("--frozen-only", action="store_true", help="Verify frozen inputs; allow later active workspace edits.")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    if args.self_test:
        self_test()
        return 0
    try:
        data = load_completed_run(args.run, frozen_only=args.frozen_only)
        report = build_report(data)
        if args.check_only:
            print(json.dumps({"validated_complete_episodes": 138, "paired_test_conditions": 9,
                              "all_failed_cases_retained": True, "recommendation": report["recommended_role_selected_on_validation"],
                              "release_acceptance": report["release_acceptance"]}))
        else:
            result = write_outputs(data, args.output or args.run / "plots", tables_only=args.tables_only)
            print(json.dumps(result, indent=2, allow_nan=False))
    except IncompleteRun as exc:
        print(str(exc), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
