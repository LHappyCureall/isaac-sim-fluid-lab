"""Curate only a COMPLETE42-case, independently audited V9 experiment.

No runtime/solver/SDK/GPU imports, model fitting or model-byte writes occur.
Normal execution writes one explicit JSON destination outside the run. The
original protocol identity is retained; its sanitized excerpt is not rehashed
as if it were that original protocol. Raw logs and NPZ inputs stay local.
"""

import argparse
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import tempfile


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
DEFAULT_RUN = ROOT / "droplet_out/pour_policy_fine_v9/run_20261127"
STUDY = "research/fine_particle_v9_20261127"
KIND = "curated_fine_particle_v9_live_CEM_results"
TOOLS = ("audit_fine_run.py", "plot_fine_results.py", "test_contract.py", "curate_fine_results.py")
INPUT_ROLES = {
    "assets/robot_testtube_pour/scripts/mouth_pivot_fast_geometry.py": "base_pivot_geometry_source",
    "projects/pour_policy_training_v3/mouth_pivot_z022_cache.npz": "base_pivot_path_cache",
    "projects/pour_policy_training_v3/mouth_pivot_z026_cache.npz": "base_import_default_cache",
    "assets/robot_testtube_pour/scripts/geometry.py": "original_geometry_source",
    "projects/pour_policy_training_v3/fast_path_smooth_cache.npz": "ancestral_joint_clock_cache",
    "projects/pour_policy_training_v3/mouth_pivot_lift025_z022_cache.npz": "selected_upright_lift_path_cache",
}
_READER = None
_PUBLISHER = None
_SYNC_TOOL = None


def need(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    json.dumps(value, allow_nan=False)
    return value


def safe_file(root, relative):
    path = (root / relative).resolve()
    need(not Path(relative).is_absolute() and path.is_relative_to(root.resolve()) and path.is_file(),
         "Unsafe or missing relative source/artifact: " + relative)
    return path


def plot_reader():
    global _READER
    if _READER is not None:
        return _READER
    path = HERE / "plot_fine_results.py"
    spec = importlib.util.spec_from_file_location("fine_curator_verified_CPU_reader", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _READER = module
    return module


def load_complete(run):
    # The shared CPU reader checks summary-complete and audit/SHA before any
    # journal/test decoding, then verifies sidecars, sources, models and roles.
    return plot_reader().load_run(run)


def publisher():
    """Use the publisher's exact path conversion, never a second template."""
    global _PUBLISHER
    if _PUBLISHER is not None:
        return _PUBLISHER
    choices = ([_SYNC_TOOL] if _SYNC_TOOL is not None else
               [ROOT / "projects/isaac-sim-fluid-lab/tools/sync_from_workspace.py"]
               + [parent / "tools/sync_from_workspace.py" for parent in HERE.parents])
    path = next((Path(path) for path in choices if Path(path).is_file()), None)
    need(path is not None, "The reviewed sync helper is required; pass --sync-tool explicitly")
    spec = importlib.util.spec_from_file_location("fine_curator_publication_bytes", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    need(callable(getattr(module, "fine_source_bytes", None))
         and module.FINE_PUBLIC.rstrip("/") == STUDY, "Publisher lacks the reviewed V9 source-byte conversion")
    _PUBLISHER = module
    return module


def source_mapping(plan, run):
    entries = []
    for original, expected in plan["source_sha256"].items():
        path = safe_file(run / "source_snapshot", original)
        need(sha(path) == expected, "Frozen runtime source bytes differ")
        raw = path.read_bytes()
        public = publisher().fine_source_bytes(original, raw)
        if original != "assets/robot_testtube_pour/scripts/robot.py":
            need(public == raw, "Only robot.py may have public path conversions")
        entries.append({"original_file": original, "original_sha256": expected,
                        "published_file": STUDY + "/" + original,
                        "published_sha256": hashlib.sha256(public).hexdigest()})
    need(len(entries) == 19, "Runtime19 source coverage differs")
    return entries


def tool_mapping():
    entries = []
    for name in TOOLS:
        path = safe_file(HERE, name)
        digest = sha(path)
        original = "projects/pour_policy_training_v9/" + name
        entries.append({"original_file": original, "original_sha256": digest,
                        "published_file": STUDY + "/" + original, "published_sha256": digest})
    return entries


def descriptive(rows):
    """Every case is included; dose-only success and joint success are distinct."""
    need(bool(rows), "Cannot summarize an empty group")
    n = len(rows)
    errors = [row["result"]["error_ml"] for row in rows]
    percentages = [100 * error / row["target_ml"] for row, error in zip(rows, errors)]
    post_durations = [row["post_observation"].get("duration_s") for row in rows
                      if isinstance(row["post_observation"], dict)
                      and isinstance(row["post_observation"].get("duration_s"), (int, float))]
    return {"n": n, "mae_ml": sum(map(abs, errors)) / n,
            "worst_absolute_error_ml": max(map(abs, errors)), "mean_signed_error_ml": sum(errors) / n,
            "signed_errors_ml": errors, "signed_percent_errors": percentages,
            "mean_absolute_percent_error": sum(map(abs, percentages)) / n,
            "worst_absolute_percent_error": max(map(abs, percentages)),
            "underfill_cases": sum(value < 0 for value in errors), "overfill_cases": sum(value > 0 for value in errors),
            "dose_only_success_count": sum(row["assessment"]["dose_passed"] for row in rows),
            "joint_task_success_count": sum(row["assessment"]["feasible"] for row in rows),
            "valid_evidence_count": sum(row["assessment"]["valid_evidence"] for row in rows),
            "full_home_within_24s_count": sum(row["assessment"]["full_home_within_deadline"] for row in rows),
            "spill_at_most_one_count": sum(row["assessment"]["spill_passed"] for row in rows),
            "no_late_dose_change_count": sum(row["assessment"]["no_late_dose_change"] for row in rows),
            "peak_all_label_spill_particles": max(row["assessment"]["maximum_all_label_spilled_particles"] for row in rows),
            "sum_case_peak_all_label_spill_particles": sum(row["assessment"]["maximum_all_label_spilled_particles"] for row in rows),
            "mean_action_physical_seconds_including_home_and_quiet": sum(row["result"]["physical_time_s"] for row in rows) / n,
            "mean_action_wall_seconds": sum(row["result"]["wall_seconds"] for row in rows) / n,
            "mean_case_wall_seconds_including_post_and_overhead": sum(row["wall_seconds"] for row in rows) / n,
            "mean_post_physical_seconds_when_recorded": sum(post_durations) / len(post_durations) if post_durations else None,
            "post_duration_records": len(post_durations), "separate_post_wall_measurement_available": False,
            "old_footer_success_count": sum(row["result"]["success"] for row in rows),
            "old_footer_success_is_not_the_strict_percentage_task": True}


def compact_case(row, run):
    # The journal row is already a compact footer/status/aggregate record,
    # not raw particle steps. Preserve its JSON value exactly. Artifact keys
    # are safe relative names plus hashes, never the artifact contents.
    value = copy.deepcopy(row)
    for name in value["artifacts"]:
        safe_file(run, name)
    return value


def paired_view(row):
    return {"case_id": row["episode_id"], "received_ml": row["result"]["received_ml"],
            "signed_error_ml": row["result"]["error_ml"],
            "signed_percent_error": 100 * row["result"]["error_ml"] / row["target_ml"],
            "tolerance_ml": .001 * row["target_ml"], "assessment": copy.deepcopy(row["assessment"]),
            "action_physical_seconds_including_home_and_quiet": row["result"]["physical_time_s"],
            "action_wall_seconds": row["result"]["wall_seconds"],
            "case_wall_seconds_including_post_and_overhead": row["wall_seconds"]}


def plot_metadata(folder, data):
    if folder is None:
        return {"included": False, "reason": "No final plot directory was supplied"}, {}
    folder = Path(folder).resolve()
    provenance = read(folder / "fine_plot_provenance.json")
    need(provenance["synthetic_fixture"] is False and provenance["protocol_id"] == data["protocol"]["protocol_id"],
         "Plots are synthetic or describe another protocol")
    for key, path in (("summary_sha256", data["run"] / "summary.json"),
                      ("journal_sha256", data["run"] / "episodes.jsonl"),
                      ("audit_sha256", data["audit_path"]), ("protocol_sha256", data["run"] / "protocol.json")):
        need(provenance[key] == sha(path), "Plot provenance differs: " + key)
    plots = {}
    for name, digest in provenance["files_sha256"].items():
        path = safe_file(folder, name)
        need(sha(path) == digest, "Plot/table artifact changed")
        if name.endswith(".png"):
            need(path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n"), "Invalid PNG")
            plots[name] = {"file": "docs/images/fine_v9_20261127_" + name, "sha256": digest}
    need(len(plots) == 3, "Three final V9 plots required when a plot directory is supplied")
    return {"included": True, "provenance_sha256": sha(folder / "fine_plot_provenance.json"),
            "plot_tool_sha256": provenance["plot_tool_sha256"],
            "synthetic_fixture": False, "no_policy_fitting_or_deployment": True}, plots


def assert_no_machine_paths(value):
    text = json.dumps(value, ensure_ascii=False, allow_nan=False)
    need(re.search(r"[A-Za-z]:[\\/]", text) is None and "/Users/" not in text and "\\\\Users\\\\" not in text,
         "Private machine path found; refuse automatic publication rather than silently rewriting evidence")


def curate(data, plots=None):
    need(data.get("verified") is True and data.get("synthetic") is False, "Only a real fully audited run may be curated")
    run, plan, summary, audit, rows = (data[key] for key in ("run", "protocol", "summary", "audit", "rows"))
    need(len(rows) == 42 and audit["episodes_audited"] == 42, "All42 cases including failures required")
    need(set(plan["input_sha256"]) == set(INPUT_ROLES), "Frozen input role schema differs")
    public_models = {}
    for filename in ("candidate_frozen.json", "recommended_frozen.json"):
        model = data["models"][filename]
        public_models[filename.removesuffix("_frozen.json")] = {
            "file": STUDY + "/models/" + filename, "sha256": model["sha256"],
            "definition": copy.deepcopy(model["definition"]), "raw_bytes_preserved": True}
    protocol_fields = ("kind", "created_at_utc", "user_criterion", "budget", "search", "train_seeds",
        "validation_seeds", "test_seeds", "targets_ml", "spacing_m", "particle_ml", "initial_particle_count",
        "initial_source_particles", "initial_source_ml", "solver", "fixed_engineering_changes", "baseline",
        "observation", "selection", "initial_evidence_gate", "source_sha256", "gui_sha256")
    protocol_summary = {key: copy.deepcopy(plan[key]) for key in protocol_fields}
    protocol_summary["input_sha256"] = {INPUT_ROLES[name]: digest for name, digest in plan["input_sha256"].items()}
    protocol_summary["input_roles"] = {INPUT_ROLES[name]: {"original_file": name, "sha256": digest}
                                        for name, digest in plan["input_sha256"].items()}
    groups = {stage: [row for row in rows if row["stage"] == stage] for stage in ("train", "validation", "test")}
    by_role = {stage: {role: [row for row in groups[stage] if row["role"] == role]
                       for role in ("candidate", "baseline")} for stage in ("validation", "test")}
    pairs = []
    for target in plan["targets_ml"]:
        for seed in plan["test_seeds"]:
            pair = {"target_ml": target, "seed": seed, "recommended_result_role": summary["recommended_role"]}
            for role in ("candidate", "baseline"):
                row = next(row for row in by_role["test"][role] if (row["target_ml"], row["seed"]) == (target, seed))
                pair[role] = paired_view(row)
            pairs.append(pair)
    audit_fields = ("audit_schema_version", "kind", "audited_at_utc", "audit_tool_sha256", "protocol_id",
        "summary_sha256", "journal_sha256", "budgets", "artifact_integrity_verified", "all_failures_retained",
        "episodes_audited", "raw_action_trace_count", "source_count", "input_count", "raw_URDF_reference_sha256",
        "old_3_GUI_models_unchanged", "models_raw_sha256", "validation_only_recommendation_verified",
        "CEM_replay_verified", "freeze_to_first_test_reset_seconds", "initial_source_particles", "initial_source_ml",
        "particle_ml", "joint_240hz_samples_audited", "FK_full_R_60hz_rows_audited", "recommended_role",
        "task_passed", "failed_task_case_ids", "invalid_evidence_case_ids", "no_GPU_or_liquid_execution")
    plot_provenance, image_entries = plot_metadata(plots, data)
    report = {"schema_version": 1, "kind": KIND, "run_id": run.name,
        "complete": True, "experiment_completed": True, "task_passed": summary["task_passed"],
        "completed_at_utc": summary["completed_at_utc"], "protocol_id": plan["protocol_id"],
        "protocol_identity": {"original_protocol_file_sha256": sha(run / "protocol.json"),
            "raw_canonical_protocol_published": False, "sanitized_excerpt_is_not_original_protocol_hash": True},
        "counts": {"train": 18, "validation": 6, "test": 18, "total": 42},
        "method": {"algorithm": "one_dimensional_coarse_speed_CEM_with_fixed_settled_micro_correction",
            "neural_policy": False, "ppo": False, "searched_parameter": "coarse_tilt_speed_multiplier",
            "candidate_engineering_changes_are_not_learned": True,
            "baseline_is_unchanged_legacy_not_candidate_at_multiplier_one": True},
        "protocol_summary": protocol_summary, "versions": copy.deepcopy(summary["versions"]),
        "publication": {"archive_only_not_turnkey_reproduction": True,
            "source_mapping": source_mapping(plan, run), "tool_mapping": tool_mapping(),
            "post_training_tools_are_not_runtime19_dependencies": True,
            "runtime_source_bytes_transformed": True, "checkpoint_bytes_transformed": False,
            "public_source_transform_scope": "robot.py two local paths only; other18runtime raw bytes preserved",
            "relative_module_layout_preserved": True, "literal_source_guards_preserved": True,
            "original_protocol_refers_to_original_runtime19_not_post_training_tools": True,
            "private_NPZ_and_raw_logs_not_published": True,
            "source_rebinding_or_new_preparation_requires_a_new_protocol": True},
        "models": public_models, "freeze": copy.deepcopy(data["freeze"]),
        "generations": copy.deepcopy(summary["generations"]),
        "recommended_role": summary["recommended_role"], "gui_deployment": False,
        "training": {"generations": copy.deepcopy(summary["generations"]),
            "metrics": descriptive(groups["train"]), "case_ids": [row["episode_id"] for row in groups["train"]]},
        "validation": {"original_metrics": copy.deepcopy(summary["validation"]),
            "metrics": {role: descriptive(group) for role, group in by_role["validation"].items()},
            "recommended_role": summary["recommended_role"], "test_used_for_selection": False},
        "test": {"distinct_target_seed_conditions": 9, "executed_live_episodes": 18,
            "original_metrics": copy.deepcopy(summary["test"]),
            "metrics": {role: descriptive(group) for role, group in by_role["test"].items()},
            "paired_cases": pairs, "recommendation_was_not_reselected": True},
        "per_target_comparison": [{"target_ml": target, "tolerance_ml": .001 * target,
            **{role: descriptive([row for row in by_role["test"][role] if row["target_ml"] == target])
               for role in ("candidate", "baseline")}} for target in plan["targets_ml"]],
        "all_cases": [compact_case(row, run) for row in rows], "all_rollouts": descriptive(rows),
        "case_record_identity": {"all_case_JSON_values_equal_original_journal_rows": True,
            "original_footer_success_preserved_but_not_used_for_new_task": True,
            "artifact_contents_not_published": True,
            "completed_sidecar_sha256": {row["episode_id"]: sha(safe_file(run,
                "completed/" + row["episode_id"] + ".json")) for row in rows}},
        "failure_counts": {"task_failed": sum(not row["assessment"]["feasible"] for row in rows),
            "invalid_evidence": sum(not row["assessment"]["valid_evidence"] for row in rows),
            "dose_failed": sum(not row["assessment"]["dose_passed"] for row in rows),
            "home_or_deadline_failed": sum(not row["assessment"]["full_home_within_deadline"] for row in rows),
            "spill_over_one": sum(not row["assessment"]["spill_passed"] for row in rows),
            "late_dose_change": sum(not row["assessment"]["no_late_dose_change"] for row in rows),
            "failure_categories_may_overlap": True,
            "old_footer_success_but_strict_task_failed_ids": [row["episode_id"] for row in rows
                if row["result"]["success"] and not row["assessment"]["feasible"]]},
        "audit": {key: copy.deepcopy(audit[key]) for key in audit_fields},
        "audit_file_sha256": sha(data["audit_path"]), "plots": image_entries, "plot_provenance": plot_provenance,
        "curation": {"tool_sha256": sha(__file__), "CPU_reader_sha256": sha(HERE / "plot_fine_results.py"),
            "no_model_fitting_or_weight_generation": True, "no_solver_or_GPU_execution": True},
        "gui": {"new_deployment": False, "new_validation": False, "old_three_defaults_unchanged": True},
        "limitations": ["Particle-volume truth is privileged simulation state, not a calibrated camera or physical scale.",
            "Spacing2.7mm changes the actual source to2880particles/56.68704mL; it is not the old51mL initial condition.",
            "Dose-only strict0.1percent success is separate from full task success requiring24raw seconds including home/quiet, all-label peakspill<=1, valid physics and stable post observation.",
            "The original footer.success uses an older absolute tolerance and is preserved only as a historical field, never as V9 acceptance.",
            "Action wall and case wall are recorded; isolated post wall is not measured. Their difference also contains reset/overhead, so it is not labeled as measured post wall.",
            "One train seed, one validation seed and three test seeds cover20/30/40mL only, not arbitrary target generalization.",
            "Fixed engineering changes and the coarse-speed search are distinct; this is not PPO or a neural policy.",
            "Original runtime SHA values remain the experiment identity. Only the public robot.py copy changes two local paths; its derived public SHA is separate, other18runtime and model bytes stay raw. Private matching NPZ/SDK are still required, and the public derivative cannot resume the original run or claim byte-identical source reproduction."]}
    report["per_target"] = copy.deepcopy(report["per_target_comparison"])
    tools = {item["original_file"].split("/")[-1]: item["original_sha256"]
             for item in report["publication"]["tool_mapping"]}
    need(tools["audit_fine_run.py"] == audit["audit_tool_sha256"], "Current auditor differs from the accepted audit source")
    if plot_provenance["included"]:
        need(tools["plot_fine_results.py"] == plot_provenance["plot_tool_sha256"], "Current plotter differs from the final plots")
    assert_no_machine_paths(report)
    return report


def self_test():
    checks = []
    with tempfile.TemporaryDirectory(prefix="fine_curator_cpu_", dir=HERE) as name:
        folder = Path(name)
        need(folder.resolve().is_relative_to(HERE.resolve()), "Unsafe temporary fixture directory")
        (folder / "episodes.jsonl").write_text("MUST NOT DECODE JOURNAL/TEST", encoding="utf-8")
        for summary in ({"complete": False, "counts": {"train": 17}},
                        {"complete": True, "counts": {"train": 18, "validation": 6, "test": 18}}):
            (folder / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
            try:
                load_complete(folder)
            except plot_reader().IncompleteRun:
                checks.append("partial_or_unaudited_rejected_before_journal")
            else:
                raise ValueError("Incomplete or unaudited fixture accepted")
    row = {"target_ml": 20, "result": {"error_ml": .08, "physical_time_s": 20., "wall_seconds": 10., "success": True},
           "wall_seconds": 12., "post_observation": {"duration_s": 2.},
           "assessment": {"dose_passed": False, "feasible": False, "valid_evidence": True,
                          "full_home_within_deadline": True, "spill_passed": True,
                          "no_late_dose_change": True, "maximum_all_label_spilled_particles": 1}}
    description = descriptive([row])
    need(description["old_footer_success_count"] == 1 and description["joint_task_success_count"] == 0
         and description["dose_only_success_count"] == 0, "Old absolute success leaked into strict percentage task")
    need(description["separate_post_wall_measurement_available"] is False, "Post wall must not be invented")
    need(description["signed_percent_errors"] == [.4], "Signed percentage units")
    try:
        assert_no_machine_paths({"diagnostic": "C:\\Users\\private\\run.log"})
    except ValueError:
        checks.append("private_machine_path_refused")
    else:
        raise ValueError("Private machine path accepted")
    checks += ["old_footer_success_separate_from_new_task", "signed_percentage_preserved", "post_wall_not_invented"]
    return {"self_test_passed": True, "checks": checks, "real_run_or_test_read": False,
            "GPU_execution": False, "completed_result_or_model_generated": False}


def main():
    global _SYNC_TOOL
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--output", type=Path, help="Explicit curated JSON destination outside the experiment run")
    parser.add_argument("--plots", type=Path, help="Optional final audited plot directory; does not render or copy images")
    parser.add_argument("--sync-tool", type=Path, help="Reviewed sync_from_workspace.py containing the exact public robot path conversion")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    _SYNC_TOOL = args.sync_tool
    if args.self_test:
        print(json.dumps(self_test(), indent=2))
        return 0
    need(args.output is not None, "--output is required; no default writes to active experiment output")
    run, destination = args.run.resolve(), args.output.resolve()
    need(not destination.is_relative_to(run), "Curated output must not alter the original experiment directory")
    need(not destination.exists(), "Refuse to overwrite a previous curated result; use a new explicit destination")
    report = curate(load_complete(run), plots=args.plots)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"file": str(destination), "sha256": sha(destination),
                      "all_cases": len(report["all_cases"]), "task_passed": report["task_passed"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
