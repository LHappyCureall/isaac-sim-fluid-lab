"""Small CPU-only checks for this source distribution; no Kit/GPU required."""
import hashlib
import importlib.util
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _published_file(relative):
    path = (ROOT / relative).resolve()
    assert path.is_relative_to(ROOT) and path.is_file(), f"Missing/unsafe published file: {relative}"
    return path


def _bundle_files(relative):
    """Keep strict source allowlists while ignoring only ordinary local pyc."""
    return {path.relative_to(ROOT).as_posix() for path in (ROOT/relative).rglob("*")
            if path.is_file() and not (path.parent.name == "__pycache__" and path.suffix == ".pyc")}


def _check_ai_results():
    report = json.loads(_published_file("results/ai_training.json").read_text(encoding="utf-8"))
    json.dumps(report, allow_nan=False)
    assert report["schema_version"] == 1 and report["kind"] == "curated_live_pouring_training_results"
    assert report["complete"] is True
    assert type(report["gui_validation"]["accepted"]) is bool
    counts, seeds = report["counts"], report["seeds"]
    targets = {float(target) for target in seeds["training_by_target"]}
    heldout_seeds = set(seeds["heldout_test"])
    assert heldout_seeds.isdisjoint(seeds["validation"])
    assert heldout_seeds.isdisjoint(seeds["training_by_target"].values())
    assert counts["training_episodes"] == counts["generations"] * counts["population"] * len(targets)
    assert counts["validation_episodes"] == 2 * len(targets) * len(seeds["validation"])
    assert counts["distinct_heldout_seeds"] == len(heldout_seeds)
    expected = {(target, seed) for target in targets for seed in heldout_seeds}
    assert counts["distinct_heldout_target_seed_conditions"] == len(expected)
    assert counts["original_heldout_test_episodes"] == 2 * len(expected)
    assert counts["all_executed_test_episodes"] == (counts["original_heldout_test_episodes"]
                                                   + counts["supplementary_candidate_test_episodes"])
    assert counts["all_executed_live_episodes"] == (counts["training_episodes"]
          + counts["validation_episodes"] + counts["all_executed_test_episodes"])
    pairs = report["heldout_pairs"]
    assert len(pairs) == len(expected) and {(float(p["target_ml"]), p["seed"]) for p in pairs} == expected
    for label in ("baseline", "default", "candidate"):
        results = [pair[label] for pair in pairs]
        errors, safe = [], []
        for pair, result in zip(pairs, results):
            assert all(type(result[key]) is bool for key in ("valid_physics", "complete", "upright", "mass_conserved"))
            assert math.isfinite(result["received_ml"]) and math.isfinite(result["error_ml"])
            assert abs(result["received_ml"] - pair["target_ml"] - result["error_ml"]) < 1e-8
            assert abs(result["physical_time_s"] - result["physical_steps"] * report["physics"]["dt_s"]) < 1e-8
            errors.append(abs(result["error_ml"]))
            safe.append(result["valid_physics"] and result["complete"] and result["upright"]
                        and result["max_spilled_ml"] == 0 and result["inflight_ml"] <= .1)
        metric = report["heldout_metrics"][label]
        assert metric["episodes"] == len(results)
        assert abs(metric["mae_ml"] - sum(errors) / len(errors)) < 1e-8
        assert abs(metric["worst_absolute_error_ml"] - max(errors)) < 1e-8
        assert metric["within_0_2_ml_count"] == sum(ok and error <= .2 for ok, error in zip(safe, errors))
        assert metric["within_0_5_ml_count"] == sum(ok and error <= .5 for ok, error in zip(safe, errors))
        assert metric["safe_completed_count"] == sum(safe)
        assert metric["all_mass_conserved"] == all(result["mass_conserved"] for result in results)
    assert len(report["training"]["curve"]) == counts["generations"]
    assert report["training"]["curve"][-1]["completed_training_episodes"] == counts["training_episodes"]
    assert len(report["training"]["candidates"]) == counts["generations"] * counts["population"]
    assert report["selection_protocol"]["test_used_for_fitting_or_selection"] is False
    assert report["selection_protocol"]["candidate_audit_changed_deployment"] is False

    spec = importlib.util.spec_from_file_location("release_sync_validator", ROOT / "tools/sync_from_workspace.py")
    sync = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sync)
    assert set(report["models"]) == {"default", "candidate"}
    for label, relative in zip(("default", "candidate"), sync.MODEL_FILES):
        entry = report["models"][label]
        assert entry["file"] == relative
        path = _published_file(relative)
        raw = path.read_bytes()
        sync._validate_checkpoint(relative, raw.decode("utf-8"))
        assert hashlib.sha256(raw).hexdigest() == entry["sha256"], f"Frozen weight SHA differs: {relative}"
        checkpoint = json.loads(raw)
        assert checkpoint["parameters"] == entry["parameters"]
        assert checkpoint["metadata"]["is_trained"] == entry["is_trained"]
        assert checkpoint["metadata"]["selection_origin"] == entry["selection_origin"]
        assert checkpoint["metadata"]["source_sha256"] == report["provenance"]["original_training_source_sha256"]
    selected = report["models"]["default"]
    origin = selected["selection_origin"]
    assert report["validation"]["selection"] == origin
    scores = report["validation"]["mean_score"]
    if origin == "baseline_fallback":
        assert selected["is_trained"] is False and scores["baseline"] <= scores["training_best"]
        assert all(pair["default"] == pair["baseline"] for pair in pairs)
    else:
        assert origin == "cem_training_best" and selected["is_trained"] is True
        assert scores["training_best"] < scores["baseline"]
    for relative in report["plots"].values():
        assert _published_file(relative).read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    nominal = report["gui_validation"].get("original_dispatch_candidate_nominal")
    if nominal is not None:
        assert nominal["heldout_test"] is False and nominal["included_in_batch_rollouts"] is False
        assert nominal["randomized"] is False
        assert nominal["policy_parameters"] == report["models"]["candidate"]["parameters"]
        assert len(nominal["results"]) == len(targets)
        assert {float(result["target_ml"]) for result in nominal["results"]} == targets
        for result in nominal["results"]:
            assert abs(result["received_ml"] - result["target_ml"] - result["error_ml"]) < 1e-8
    flow = report["gui_validation"].get("flow_display")
    if flow is not None:
        assert flow["display_only"] is True and flow["cuda_graph"] is False
        assert flow["centers_match_live_particles"] is True
        assert flow["positions_velocities_masses_time_unchanged"] is True
        assert flow["clear_hides_glyphs"] is True
        assert flow["shown_particles"] == flow["hud_source_airborne_particles"]
        assert abs(flow["glyph_diameter_m"] - report["physics"]["spacing_m"]) < 1e-12
        _published_file(flow["image"])
    _published_file("docs/AI_TRAINING_RESULTS.md")


def _check_full_results():
    report = json.loads(_published_file("results/ai_training_full_20261122.json").read_text(encoding="utf-8"))
    json.dumps(report, allow_nan=False)
    assert report["schema_version"] == 1 and report["kind"] == "curated_full_budget_live_pouring_results"
    assert report["complete"] is True
    counts, seeds = report["counts"], report["seeds"]
    targets = {float(target) for target in seeds["training_by_target"]}
    test_seeds = set(seeds["heldout_test"])
    quick = json.loads(_published_file("results/ai_training.json").read_text(encoding="utf-8"))
    assert test_seeds.isdisjoint(quick["seeds"]["heldout_test"])
    assert test_seeds.isdisjoint(seeds["validation"])
    assert test_seeds.isdisjoint(seeds["training_by_target"].values())
    assert counts["training_episodes"] == counts["generations"] * counts["population"] * len(targets)
    assert counts["fixed_training_target_seed_conditions"] == len(targets)
    assert counts["validation_episodes"] == 2 * len(targets) * len(seeds["validation"])
    assert counts["distinct_validation_target_seed_conditions"] == len(targets) * len(seeds["validation"])
    assert counts["distinct_test_seeds"] == len(test_seeds)
    expected = {(target, seed) for target in targets for seed in test_seeds}
    assert counts["distinct_test_target_seed_conditions"] == len(expected)
    assert counts["test_episodes"] == 2 * len(expected)
    assert counts["all_executed_live_episodes"] == sum(counts[name] for name in ("training_episodes", "validation_episodes", "test_episodes"))
    pairs = report["heldout_pairs"]
    assert len(pairs) == len(expected) and {(float(p["target_ml"]), p["seed"]) for p in pairs} == expected
    recommended_role = report["recommendation"]["test_result_role"]
    assert recommended_role in ("candidate", "baseline")
    assert counts["recommended_additional_test_rollouts"] == report["recommendation"]["additional_test_rollouts"] == 0
    assert report["recommendation"]["test_results_are_references"] is True
    for role in ("candidate", "baseline"):
        results = [p[role] for p in pairs]
        errors, safe = [], []
        for pair, result in zip(pairs, results):
            assert pair["recommended_result_role"] == recommended_role and "selected" not in pair
            assert abs(result["received_ml"] - pair["target_ml"] - result["error_ml"]) < 1e-8
            assert math.isfinite(result["received_ml"]) and math.isfinite(result["error_ml"])
            assert abs(result["physical_time_s"] - result["physical_steps"] * report["physics"]["dt_s"]) < 1e-8
            errors.append(abs(result["error_ml"]))
            safe.append(result["valid_physics"] and result["complete"] and result["upright"]
                        and result["max_spilled_ml"] == 0 and result["inflight_ml"] <= .1)
        metric = report["heldout_metrics"][role]
        assert metric["episodes"] == len(results)
        assert abs(metric["mae_ml"] - sum(errors) / len(errors)) < 1e-8
        assert abs(metric["worst_absolute_error_ml"] - max(errors)) < 1e-8
        assert metric["within_0_2_ml_count"] == sum(ok and e <= .2 for ok, e in zip(safe, errors))
        assert metric["within_0_5_ml_count"] == sum(ok and e <= .5 for ok, e in zip(safe, errors))
        assert metric["safe_completed_count"] == sum(safe)
        assert metric["all_mass_conserved"] == all(r["mass_conserved"] for r in results)
    assert len(report["training"]["curve"]) == counts["generations"]
    assert report["training"]["curve"][-1]["completed_training_episodes"] == counts["training_episodes"]
    assert len(report["training"]["candidates"]) == counts["generations"] * counts["population"]
    for group in (report["training"], report["all_rollout_checks"]):
        assert abs(group["physical_seconds"] - group["physical_steps"] * report["physics"]["dt_s"]) < 1e-7
    spec = importlib.util.spec_from_file_location("release_full_sync_validator", ROOT / "tools/sync_from_workspace.py")
    sync = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sync)
    for _, (relative, role) in sync.FULL_MODEL_SOURCES.items():
        entry = report["models"][role]
        assert entry["file"] == relative
        raw = _published_file(relative).read_bytes()
        sync._validate_checkpoint(relative, raw.decode("utf-8"))
        assert hashlib.sha256(raw).hexdigest() == entry["sha256"]
        model = json.loads(raw)
        assert model["parameters"] == entry["parameters"]
        assert model["metadata"]["is_trained"] == entry["is_trained"]
        assert model["metadata"]["source_sha256"] == report["provenance"]["original_training_source_sha256"]
        assert model["metadata"]["protocol_id"] == report["provenance"]["protocol_id"]
    candidate = report["models"]["candidate"]
    assert report["training"]["best_parameters"] == candidate["parameters"]
    changed = candidate["parameters"] != report["baseline_parameters"]
    assert candidate["is_trained"] == report["training"]["parameters_changed_from_baseline"] == changed
    scores = report["validation"]["mean_score"]
    expected_role = "candidate" if changed and scores["candidate"] < scores["baseline"] else "baseline"
    assert recommended_role == expected_role == report["validation"]["recommended_role"]
    assert report["models"]["recommended"]["parameters"] == (candidate["parameters"] if expected_role == "candidate" else report["baseline_parameters"])
    assert report["models"]["recommended"]["is_trained"] == (expected_role == "candidate")
    assert report["gui"]["new_deployment"] is False
    for relative in report["plots"].values():
        assert _published_file(relative).read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    _published_file("docs/AI_TRAINING_FULL_20261122.md")


def _check_precision_results():
    report = json.loads(_published_file("results/precision_v2_20261122.json").read_text(encoding="utf-8"))
    json.dumps(report, allow_nan=False)
    assert report["schema_version"] == 1 and report["complete"] is True and report["method"]
    protocol = report["protocol"]
    protocol_id = hashlib.sha256(json.dumps(protocol, sort_keys=True, separators=(",", ":"),
                                            allow_nan=False).encode("utf-8")).hexdigest()
    assert protocol["algorithm"] == "hand_designed_coarse_to_fine_feedback_plus_predeclared_finite_local_calibration"
    assert protocol["neural_policy"] is False and protocol["ppo"] is False
    targets, train, validation, test = (set(protocol[name]) for name in
        ("targets_ml", "train_seeds", "validation_seeds", "test_seeds"))
    assert train.isdisjoint(validation) and train.isdisjoint(test) and validation.isdisjoint(test)
    assert len(protocol["configs"]) == report["calibration"]["configurations"] == 3
    assert protocol["calibration_episodes"] == report["calibration"]["episodes"] == 3 * len(targets) * len(train) == 18
    assert protocol["validation_episodes"] == 2 * len(targets) * len(validation) == 12
    assert protocol["test_episodes"] == 2 * len(targets) * len(test) == 18
    assert protocol["total_live_episodes"] == 48
    for old in ("results/ai_training.json", "results/ai_training_full_20261122.json"):
        old_report = json.loads(_published_file(old).read_text(encoding="utf-8"))
        assert test.isdisjoint(old_report["seeds"]["heldout_test"])
    expected = {(target, seed) for target in targets for seed in test}
    pairs = report["paired_cases"]
    assert len(pairs) == len(expected) == report["test"]["distinct_target_seed_conditions"]
    assert {(p["target_ml"], p["seed"]) for p in pairs} == expected
    assert report["test"]["independent_initial_state_seeds"] == len(test)
    assert report["test"]["executed_live_episodes"] == 2 * len(pairs)
    dt = protocol["physics"]["dt_s"]
    assert abs(dt - 1 / 60) < 1e-12 and protocol["physics"]["pressure_iterations"] == 6

    def check_metrics(metric, results):
        assert results
        for result in results:
            assert all(type(result[key]) is bool for key in
                       ("valid_physics", "mass_conserved", "complete", "upright", "success"))
            assert abs(result["received_ml"] - result["target_ml"] - result["error_ml"]) < 1e-8
            assert abs(result["absolute_error_ml"] - abs(result["error_ml"])) < 1e-8
            assert abs(result["physical_time_s"] - result["physical_steps"] * dt) < 1e-8
            assert result["tolerance_ml"] == protocol["physics"]["tolerance_ml"]
        errors = [abs(r["error_ml"]) for r in results]
        safe = [r["valid_physics"] and r["mass_conserved"] and r["complete"] and r["upright"]
                and r["max_spilled_ml"] == 0 and r["max_below_table_particles"] == 0
                and r["inflight_ml"] <= .1 for r in results]
        assert all(r["success"] == (ok and error <= r["tolerance_ml"])
                   for r, ok, error in zip(results, safe, errors))
        n = len(results)
        computed = {"episodes": n, "mae_ml": sum(errors) / n,
                    "worst_absolute_error_ml": max(errors), "safe_completion_rate": sum(safe) / n,
                    "all_mass_conserved": all(r["mass_conserved"] for r in results),
                    "physical_steps": sum(r["physical_steps"] for r in results),
                    "physical_seconds": sum(r["physical_time_s"] for r in results),
                    "total_spilled_ml": sum(r["max_spilled_ml"] for r in results),
                    "invalid_physics_episodes": sum(not r["valid_physics"] for r in results),
                    "unsafe_or_incomplete_episodes": sum(not ok for ok in safe)}
        for threshold in (.1, .2, .5):
            name = f"within_{str(threshold).replace('.', '_')}_ml_rate"
            computed[name] = sum(ok and e <= threshold for ok, e in zip(safe, errors)) / n
        for name, value in computed.items():
            if type(value) is bool:
                assert metric[name] is value
            elif type(value) is int:
                assert type(metric[name]) is int and metric[name] == value
            else:
                assert abs(metric[name] - value) < 1e-7
        return computed

    def check_group(results, seeds, parameters):
        conditions = {(target, seed) for target in targets for seed in seeds}
        assert len(results) == len(conditions)
        assert {(r["target_ml"], r["seed"]) for r in results} == conditions
        assert all(r["policy_params"] == parameters and r["randomized"] is True for r in results)

    def group_score(metric):
        n = metric["episodes"]
        return (metric["mae_ml"] + .5 * metric["worst_absolute_error_ml"]
                + 10 * metric["total_spilled_ml"] / n
                + 1000 * metric["invalid_physics_episodes"] / n
                + 100 * metric["unsafe_or_incomplete_episodes"] / n
                + 1e-5 * metric["physical_seconds"] / n)

    for role in ("candidate", "baseline"):
        results = [pair[role] for pair in pairs]
        for pair, result in zip(pairs, results):
            assert (result["target_ml"], result["seed"]) == (pair["target_ml"], pair["seed"])
            assert abs(result["received_ml"] - pair["target_ml"] - result["error_ml"]) < 1e-8
            assert abs(result["physical_time_s"] - result["physical_steps"] * dt) < 1e-8
            assert all(type(result[key]) is bool for key in ("valid_physics", "mass_conserved", "complete", "upright"))
        check_metrics(report["test"][role + "_metrics"], results)
    configurations = report["calibration"]["configs"]
    assert len(configurations) == len(protocol["configs"])
    calibration_scores = []
    for config, declared in zip(configurations, protocol["configs"]):
        assert config["name"] == declared["name"] and config["params"] == declared["params"]
        check_group(config["results"], train, declared["params"])
        computed = check_metrics(config["metrics"], config["results"])
        score = group_score(computed)
        assert abs(config["group_score"] - score) < 1e-8
        calibration_scores.append(score)
    best_index = min(range(len(configurations)), key=lambda index: (calibration_scores[index], index))
    best = configurations[best_index]
    selected = report["calibration"]["selected_config"]
    assert selected == {"name": best["name"], "params": best["params"]}
    scores = {}
    for role in ("candidate", "baseline"):
        results = report["validation"]["results"][role]
        check_group(results, validation, selected["params"] if role == "candidate" else None)
        computed = check_metrics(report["validation"]["metrics"][role], results)
        scores[role] = group_score(computed)
        assert abs(report["validation"]["group_score"][role] - scores[role]) < 1e-8
        check_group([pair[role] for pair in pairs], test, selected["params"] if role == "candidate" else None)
    recommended = "candidate" if scores["candidate"] < scores["baseline"] else "baseline"
    assert report["validation"]["recommended_role"] == recommended
    assert report["validation"]["test_used_for_selection"] is False
    assert all(pair["recommended_result_role"] == recommended for pair in pairs)
    assert report["test"]["recommended_metrics"] == report["test"][recommended + "_metrics"]

    spec = importlib.util.spec_from_file_location("release_precision_sync_validator", ROOT / "tools/sync_from_workspace.py")
    sync = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sync)
    for source, (relative, role) in sync.PRECISION_MODEL_SOURCES.items():
        entry = report["models"][role]
        if source.startswith(sync.PRECISION_RUN + "/"):
            assert entry["file"] == relative
        raw = _published_file(relative).read_bytes()
        sync._validate_checkpoint(relative, raw.decode("utf-8"))
        assert hashlib.sha256(raw).hexdigest() == entry["sha256"]
        model = json.loads(raw)
        assert model["parameters"] == entry["parameters"]
        assert model["metadata"]["protocol_id"] == protocol_id
        assert model["metadata"]["source_sha256"] == protocol["source_sha256"]
    assert report["models"]["candidate"]["parameters"] == selected["params"]
    assert report["models"]["recommended"]["parameters"] == (selected["params"] if recommended == "candidate" else protocol["baseline_params"])
    default = report["models"]["default"]
    assert default["file"] == "assets/robot_testtube_pour/policy/precision_v2.json"
    assert default["sha256"] == report["models"]["recommended"]["sha256"]
    assert default["parameters"] == report["models"]["recommended"]["parameters"]
    assert hashlib.sha256(_published_file(default["file"]).read_bytes()).hexdigest() == default["sha256"]
    gui = report["gui_validation"]
    assert gui["complete"] is True and gui["additional_gui_episodes"] == len(targets)
    assert gui["physical_integration_passed"] is True and gui["dose_within_0_1_ml"] is True
    assert gui["protocol_id"] == protocol_id and gui["checkpoint_sha256"] == default["sha256"]
    assert gui["training_or_heldout_test"] is False and gui["cuda_graph"] is False
    assert gui["deployed_default"] == {"file": default["file"], "sha256": default["sha256"]}
    assert len(gui["results"]) == len(targets) and {r["target_ml"] for r in gui["results"]} == targets
    for result in gui["results"]:
        assert all(result[key] is True for key in ("success", "complete", "mass_conserved", "physical_integration_passed"))
        assert abs(result["received_ml"] - result["target_ml"] - result["error_ml"]) < 1e-8
        assert abs(result["error_ml"]) <= .1 and result["precision_checkpoint_sha256"] == default["sha256"]
    configs = json.loads(_published_file(sync.SOURCE_TO_DEST[sync.PRECISION_CONFIG_SOURCE]).read_text(encoding="utf-8"))
    assert configs == protocol["configs"]
    comparison = report["per_target_comparison"]
    assert comparison["complete"] is True and comparison["protocol_id"] == protocol_id
    assert comparison["total_live_run_episodes"] == 48 and comparison["paired_test_cases"] == len(pairs)
    assert comparison["all_cases_retained"] is True and comparison["recommendation_selected_on_validation_only"] is True
    assert {row["target_ml"] for row in comparison["targets"]} == targets
    for row in comparison["targets"]:
        group = [pair for pair in pairs if pair["target_ml"] == row["target_ml"]]
        assert row["cases"] == group
        for role in ("candidate", "baseline"):
            check_metrics(row[role + "_metrics"], [pair[role] for pair in group])
    acceptance = report["release_acceptance"]
    assert acceptance["criteria"] == protocol["release_criteria"]
    assert acceptance["does_not_reselect_model"] is True
    for role in ("candidate", "baseline", "recommended"):
        metric = report["test"][role + "_metrics"]
        passed = (metric["mae_ml"] <= acceptance["criteria"]["mae_ml_at_most"]
                  and metric["worst_absolute_error_ml"] <= acceptance["criteria"]["worst_absolute_error_ml_at_most"]
                  and metric["safe_completion_rate"] == 1 and metric["all_mass_conserved"])
        assert acceptance[role + "_passed"] == passed
    assert report["audit"]["artifact_integrity_verified"] is True
    for relative in ("robot_target_volume_precision.py", "train_target_volume_precision.py",
                     "assets/robot_testtube_pour/scripts/precision_control.py",
                     "assets/robot_testtube_pour/scripts/precision_training_env.py",
                     "tools/training/audit_precision_run.py", "tools/training/plot_precision_results.py"):
        _published_file(relative)


def _check_cem_research_results(version):
    """Check each original failed protocol; V5's new score is not retroactive."""
    assert version in (3, 4)
    pivot = version == 4
    prefix, generation_count, population_count, train_count, source_count = (
        ("pivot_v4", 2, 4, 48, 30) if pivot else ("fast_v3", 3, 6, 108, 20))
    report = json.loads(_published_file("results/"+prefix+"_20261122.json").read_text(encoding="utf-8"))
    json.dumps(report, allow_nan=False)
    assert report["schema_version"] == 1 and report["kind"] == "curated_"+prefix+"_live_CEM_results"
    assert report["complete"] is True
    protocol, counts = report["protocol_summary"], report["counts"]
    assert protocol["algorithm"] == ("feasibility_first_CEM_upright_lift_fixed_mouth_feedback_v4" if pivot
                                      else "feasibility_first_CEM_speed_and_stop_feedback_v3")
    assert report.get("neural_policy", protocol.get("neural_policy", False)) is False
    assert report.get("ppo", protocol.get("ppo", False)) is False
    assert {key: counts[key] for key in ("training", "validation", "test", "total", "generations", "population")} == {
        "training": train_count, "validation": 12, "test": 18, "total": train_count+30,
        "generations": generation_count, "population": population_count}
    targets = set(protocol["targets_ml"])
    train, validation, test = [set(protocol[key+"_seeds"]) for key in ("train", "validation", "test")]
    assert targets == {20, 30, 40} and [len(group) for group in (train, validation, test)] == [2, 2, 3]
    assert train.isdisjoint(validation) and train.isdisjoint(test) and validation.isdisjoint(test)
    assert abs(protocol["physics"]["dt_s"]-1/60) < 1e-12
    assert protocol["physics"]["action_deadline_s"] == 5 and protocol["physics"]["post_observation_s"] == 2
    assert protocol["physics"]["tolerance_ml"] == .1 and protocol["physics"]["maximum_absolute_error_ml"] == .2
    for old in ("results/ai_training.json", "results/ai_training_full_20261122.json"):
        historical = json.loads(_published_file(old).read_text(encoding="utf-8"))
        assert test.isdisjoint(historical["seeds"]["heldout_test"])
    precision = json.loads(_published_file("results/precision_v2_20261122.json").read_text(encoding="utf-8"))
    assert test.isdisjoint(precision["protocol"]["test_seeds"])

    spec = importlib.util.spec_from_file_location("release_fast_sync_validator", ROOT / "tools/sync_from_workspace.py")
    sync = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sync)
    assert len(protocol["source_sha256"]) == source_count
    bounds = sync.PIVOT_SEARCH_BOUNDS if pivot else sync.FAST_PARAMETER_BOUNDS
    validate_parameters = sync._validate_pivot_parameters if pivot else sync._validate_fast_parameters
    assert set(protocol["cem"]["bounds"]) == set(bounds)
    assert all(tuple(protocol["cem"]["bounds"][key]) == limits for key, limits in bounds.items())
    validate_parameters(protocol["cem"]["baseline_params"])
    assert report["protocol_id"] == report["audit"]["protocol_id"]
    assert len(report["protocol_id"]) == 64 and report["audit"]["artifact_integrity_verified"] is True
    assert report["audit"]["budgets"] == {"training": train_count, "validation": 12, "test": 18, "total": train_count+30}
    if pivot:
        assert report["audit"]["all_failures_retained"] is True
        assert report["audit"]["episodes_audited"] == report["audit"]["raw_action_trace_count"] == 78
        assert len(protocol["input_sha256"]) == 15
        assert report["publication"]["archive_only_not_turnkey_reproduction"] is True
        assert report["publication"]["checkpoint_bytes_transformed"] is False
        historical = json.loads(_published_file("results/fast_v3_20261122.json").read_text(encoding="utf-8"))
        assert test.isdisjoint(historical["protocol_summary"]["test_seeds"])
        _published_file("docs/PIVOT_V4_20261122.md")

    def near(one, two):
        assert type(one) in (int, float) and math.isfinite(one) and abs(one-two) < 1e-7

    def computed(rows):
        assert rows
        for row in rows:
            result, a, post = row["result"], row["assessment"], row["post_observation"]
            for flag in ("physical_safe", "home_stable", "completed_by_deadline", "late_drop_passed",
                         "constraint_feasible", "deadline_dose_measurement_available", "success_within_0_1_ml"):
                assert type(a[flag]) is bool
            assert result["physical_steps"] <= 300
            near(result["physical_time_s"], result["physical_steps"]/60)
            near(result["received_ml"]-result["target_ml"], result["error_ml"])
            sample = post.get("deadline_received_ml") if isinstance(post, dict) else None
            available = type(sample) in (int, float) and math.isfinite(sample)
            assert a["deadline_dose_measurement_available"] is available
            near(a["dose_error_ml"], (sample-result["target_ml"]) if available else result["error_ml"])
            near(a["absolute_error_ml"], abs(a["dose_error_ml"]))
            assert a["dose_measurement_basis"] == ("physical_5s" if available else "invalid_terminal_fallback_not_a_5s_measurement")
            assert a["action_result_was_not_modified"] is True
            if a["physical_safe"]:
                assert result["valid_physics"] and result["mass_conserved"] and result["all_states_finite"]
                assert result["max_spilled_ml"] == 0 and result["max_below_table_particles"] == 0
            if a["completed_by_deadline"]:
                assert result["complete"] and result["physical_time_s"] <= 5 and a["home_stable"]
            if a["late_drop_passed"]:
                assert available and post["passed"] and post["maximum_received_change_ml"] == 0
                assert post["end_physical_time_s"] >= 7-1e-9 and post["duration_s"] >= 2-1e-9
                assert post["all_states_finite"] and post["mass_array_unchanged"]
            feasible = a["physical_safe"] and a["completed_by_deadline"] and a["late_drop_passed"]
            assert a["constraint_feasible"] == feasible
            assert a["success_within_0_1_ml"] == (feasible and abs(a["dose_error_ml"]) <= .1)
        n = len(rows)
        errors = [row["assessment"]["absolute_error_ml"] for row in rows]
        return {"episodes": n, "mae_ml": sum(errors)/n, "worst_absolute_error_ml": max(errors),
            "unsafe_count": sum(not row["assessment"]["physical_safe"] for row in rows),
            "deadline_or_home_or_late_drop_failures": sum(not (row["assessment"]["completed_by_deadline"]
                and row["assessment"]["late_drop_passed"]) for row in rows),
            "feasible_count": sum(row["assessment"]["constraint_feasible"] for row in rows),
            "missing_deadline_measurements": sum(not row["assessment"]["deadline_dose_measurement_available"] for row in rows),
            "within_0_1_ml_count": sum(row["assessment"]["success_within_0_1_ml"] for row in rows),
            "within_0_2_ml_count": sum(row["assessment"]["constraint_feasible"] and error <= .2 for row, error in zip(rows, errors)),
            "all_mass_conserved": all(row["result"]["mass_conserved"] for row in rows),
            "mean_action_physical_seconds": sum(row["result"]["physical_time_s"] for row in rows)/n,
            "physical_steps": sum(row["result"]["physical_steps"] for row in rows),
            "action_physical_seconds": sum(row["result"]["physical_time_s"] for row in rows),
            "post_observation_physical_seconds": sum(row["post_observation"].get("duration_s", 0)
                for row in rows if isinstance(row["post_observation"], dict))}

    def check_metrics(recorded, rows):
        actual = computed(rows)
        for key, value in actual.items():
            if type(value) is bool:
                assert recorded[key] is value
            else:
                near(recorded[key], value)
        return actual

    def key(rows):
        value = computed(rows)
        groups = [rows] + [[row for row in rows if row["result"]["target_ml"] == target] for target in targets]
        violation = sum(max(0., computed(group)["mae_ml"]-.1)
                        + max(0., computed(group)["worst_absolute_error_ml"]-.2) for group in groups if group)
        return [value["unsafe_count"], value["deadline_or_home_or_late_drop_failures"], violation,
                value["mae_ml"]+.5*value["worst_absolute_error_ml"], value["mean_action_physical_seconds"]]

    def check_key(recorded, actual):
        assert len(recorded) == len(actual) == 5
        for a, b in zip(recorded, actual):
            near(a, b)

    def check_conditions(rows, seeds, parameters):
        expected = {(target, seed) for target in targets for seed in seeds}
        assert len(rows) == len(expected) and {(row["result"]["target_ml"], row["result"]["seed"]) for row in rows} == expected
        assert all(row["result"]["policy_params"] == parameters and row["result"]["randomized"] is True for row in rows)

    all_train, candidates = [], []
    generations = report["training"]["generations"]
    assert len(generations) == generation_count and [g["generation"] for g in generations] == list(range(generation_count))
    for generation in generations:
        assert len(generation["candidates"]) == population_count and {c["index"] for c in generation["candidates"]} == set(range(population_count))
        for candidate in generation["candidates"]:
            validate_parameters(candidate["params"])
            rows = candidate["cases"]
            check_conditions(rows, train, candidate["params"])
            check_metrics(candidate["metrics"], rows)
            check_key(candidate["selection_key"], key(rows))
            candidates.append(candidate)
            all_train.extend(rows)
    assert len(all_train) == train_count
    assert counts["distinct_parameter_vectors"] == len({json.dumps(c["params"], sort_keys=True) for c in candidates})
    best = min(candidates, key=lambda candidate: candidate["selection_key"])
    validation_cases = report["validation"]["cases"]
    for role in ("candidate", "baseline"):
        rows = validation_cases[role]
        check_conditions(rows, validation, best["params"] if role == "candidate" else protocol["cem"]["baseline_params"])
        check_metrics(report["validation"]["metrics"][role], rows)
        check_key(report["validation"]["selection_keys"][role], key(rows))
    expected_role = ("candidate" if best["params"] != protocol["cem"]["baseline_params"]
                     and key(validation_cases["candidate"]) < key(validation_cases["baseline"]) else "baseline")
    assert report["validation"]["recommended_role"] == report["recommendation"]["role"] == expected_role
    assert report["validation"]["test_used_for_selection"] is False
    assert report["recommendation"]["selected_on_validation_only"] is True
    assert report["recommendation"]["additional_test_rollouts"] == 0

    pairs = report["test"]["paired_cases"]
    assert len(pairs) == 9 and {(row["target_ml"], row["seed"]) for row in pairs} == {(t, s) for t in targets for s in test}
    for role in ("candidate", "baseline"):
        rows = [pair[role] for pair in pairs]
        check_conditions(rows, test, best["params"] if role == "candidate" else protocol["cem"]["baseline_params"])
        for pair, row in zip(pairs, rows):
            assert pair["recommended_result_role"] == expected_role
            assert (pair["target_ml"], pair["seed"]) == (row["result"]["target_ml"], row["result"]["seed"])
        check_metrics(report["test"]["metrics"][role], rows)
    assert report["test"]["recommended_metrics"] == report["test"]["metrics"][expected_role]
    for role in ("candidate", "baseline", "recommended"):
        actual_role = expected_role if role == "recommended" else role
        rows = [pair[actual_role] for pair in pairs]
        groups = [rows] + [[row for row in rows if row["result"]["target_ml"] == target] for target in targets]
        passed = all((lambda m: m["unsafe_count"] == m["deadline_or_home_or_late_drop_failures"] == 0
                     and m["mae_ml"] <= .1 and m["worst_absolute_error_ml"] <= .2
                     and m["all_mass_conserved"])(computed(group)) for group in groups)
        assert report["release_acceptance"][role+"_passed"] is passed
    assert report["release_acceptance"]["does_not_reselect_model"] is True

    model_sources = sync.PIVOT_MODEL_SOURCES if pivot else sync.FAST_MODEL_SOURCES
    for source, (relative, role) in model_sources.items():
        entry = report["models"][role]
        assert entry["file"] == relative
        raw = _published_file(relative).read_bytes()
        sync._validate_checkpoint(relative, raw.decode("utf-8"))
        model = json.loads(raw)
        assert hashlib.sha256(raw).hexdigest() == entry["sha256"]
        assert model["parameters"] == entry["parameters"]
        assert model["metadata"]["protocol_id"] == report["protocol_id"]
        assert model["metadata"]["source_sha256"] == protocol["source_sha256"]
        assert entry["parameters"] == (best["params"] if role == "candidate" or expected_role == "candidate"
                                       else protocol["cem"]["baseline_params"])
    mapping = report["publication"]["source_mapping"]
    assert len(mapping) == source_count and {item["original_file"] for item in mapping} == set(protocol["source_sha256"])
    for item in mapping:
        original, published = item["original_file"], item["published_file"]
        assert item["original_sha256"] == protocol["source_sha256"][original]
        assert published == sync.SOURCE_TO_DEST.get(original, original)
        assert hashlib.sha256(_published_file(published).read_bytes()).hexdigest() == item["published_sha256"]
        if pivot and original in sync.PIVOT_UNCHANGED_SOURCES:
            assert item["original_sha256"] == item["published_sha256"], "Original literal study pins must not be rebound."
    comparison = report["per_target_comparison"]
    assert len(comparison) == 3 and {item["target_ml"] for item in comparison} == targets
    for item in comparison:
        subset = [pair for pair in pairs if pair["target_ml"] == item["target_ml"]]
        if "cases" in item:
            assert item["cases"] == subset
        for role in ("candidate", "baseline"):
            check_metrics(item[role], [pair[role] for pair in subset])
            near(item[role]["mean_action_wall_seconds"], sum(pair[role]["result"]["wall_seconds"] for pair in subset)/3)
            near(item[role]["mean_action_plus_post_wall_seconds"], sum(pair[role]["total_action_plus_post_wall_seconds"] for pair in subset)/3)
    assert report["gui"] == {"new_deployment": False,
                             ("pivot_gui_entry_implemented" if pivot else "fast_gui_entry_implemented"): False}
    assert not (ROOT / ("assets/robot_testtube_pour/policy/"+prefix+".json")).exists()
    assert not (ROOT / ("robot_target_volume_pivot.py" if pivot else "robot_target_volume_fast.py")).exists()
    for entry in report["plots"].values():
        assert set(entry) == {"file", "sha256"}
        raw = _published_file(entry["file"]).read_bytes()
        assert raw.startswith(b"\x89PNG\r\n\x1a\n") and hashlib.sha256(raw).hexdigest() == entry["sha256"]
    runtime = (*sync.PIVOT_RUNTIME_SOURCES, *sync.PIVOT_TOOL_SOURCES) if pivot else (*sync.FAST_RUNTIME_SOURCES, *sync.FAST_TOOL_SOURCES)
    for source in runtime:
        public = _published_file(sync.SOURCE_TO_DEST.get(source, source)).read_text(encoding="utf-8")
        assert "C:\\omninverse" not in public and "127.0.0.1" not in public and "8226" not in public


def _check_fast_results():
    return _check_cem_research_results(3)


def _check_pivot_results():
    return _check_cem_research_results(4)


def _check_v5_score_only():
    """Run the shipped pure-score counterexamples; no controller/SDK imports."""
    import io
    import sys
    import unittest
    folder = _published_file("tools/scoring/test_task_score.py").parent
    previous = sys.path[:]
    previous_module = sys.modules.pop("task_score", None)
    try:
        sys.path.insert(0, str(folder))
        spec = importlib.util.spec_from_file_location("release_v5_score_tests", folder/"test_task_score.py")
        tests = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(tests)
        suite = unittest.defaultTestLoader.loadTestsFromModule(tests)
        assert suite.countTestCases() == 14
        output = io.StringIO()
        result = unittest.TextTestRunner(stream=output, verbosity=0).run(suite)
        assert result.wasSuccessful(), output.getvalue()
    finally:
        sys.path[:] = previous
        sys.modules.pop("task_score", None)
        if previous_module is not None:
            sys.modules["task_score"] = previous_module


def _check_dose_window_results():
    """The 2026-11-23 dose window is a failed study. Do not treat a lower MAE as success."""
    report = json.loads(_published_file("results/dose_window_20261123.json").read_text(encoding="utf-8"))
    json.dumps(report, allow_nan=False)
    assert report["schema_version"] == 1 and report["kind"] == "curated_dose_window_live_CEM_results"
    assert report["complete"] is True and report["neural_policy"] is False and report["ppo"] is False
    assert report["automatic_gui_deployment"] is False and report["test_used_for_selection"] is False
    assert report["recommended_role"] == "candidate"
    assert report["counts"] == {"pilot": 6, "training": 48, "validation": 12, "test": 18, "total": 84,
                                "generations": 2, "population": 4}
    protocol = report["protocol_summary"]
    groups = {name: set(protocol[name]) for name in ("pilot_seeds", "train_seeds", "validation_seeds", "test_seeds")}
    assert groups["test_seeds"] == {104101, 104102, 104103}
    assert groups["pilot_seeds"].isdisjoint(groups["train_seeds"] | groups["validation_seeds"] | groups["test_seeds"])
    assert groups["train_seeds"].isdisjoint(groups["validation_seeds"] | groups["test_seeds"])
    assert groups["validation_seeds"].isdisjoint(groups["test_seeds"])
    for relative in ("results/pivot_v4_20261122.json", "results/fast_v3_20261122.json", "results/precision_v2_20261122.json"):
        historical = json.loads(_published_file(relative).read_text(encoding="utf-8"))
        old = historical.get("protocol_summary", historical.get("protocol", {})).get("test_seeds", [])
        assert groups["test_seeds"].isdisjoint(old)
    assert protocol["physics"]["action_deadline_s"] == 5 and protocol["physics"]["error_tolerance_ml"] == .2
    assert abs(protocol["physics"]["particle_volume_ml"] - .042875) < 1e-12

    def metrics(rows):
        errors = [abs(row["error_ml"]) for row in rows]
        return sum(errors) / len(rows), max(errors), sum(row["error_ml"] for row in rows) / len(rows)

    for role in ("candidate", "baseline"):
        rows = report["test"][role]["cases"]
        assert len(rows) == 9 and {row["seed"] for row in rows} == groups["test_seeds"]
        mae, worst, signed = metrics(rows)
        recorded = report["test"][role]["metrics"]
        assert abs(recorded["mean_absolute_error_ml"] - mae) < 1e-9
        assert abs(recorded["worst_absolute_error_ml"] - worst) < 1e-9
        assert abs(recorded["mean_signed_error_ml"] - signed) < 1e-9
        assert recorded["within_0_2_ml"] == 0 and recorded["full_home_cases"] == 9
        assert all(row["error_ml"] > .2 and row["physical_time_s"] <= 5 and row["full_home"] for row in rows)
        assert all(row["post_received_change_ml"] == 0 for row in rows)
    assert report["test"]["candidate"]["metrics"]["spill_cases"] == 2
    assert report["test"]["baseline"]["metrics"]["spill_cases"] == 1
    for role, path in report["policies"].items():
        payload = json.loads(_published_file(path["file"]).read_text(encoding="utf-8"))
        digest = hashlib.sha256(_published_file(path["file"]).read_bytes()).hexdigest()
        assert digest == path["sha256"] and payload["parameters"] == path["parameters"]
        assert payload["policy_kind"] == "upright_lift_fixed_mouth_planned_return_v2"
        assert payload["metadata"]["automatic_gui_deployment"] is False
        assert payload["metadata"]["test_used_for_selection"] is False
    for item in report["images"].values():
        assert hashlib.sha256(_published_file(item["file"]).read_bytes()).hexdigest() == item["sha256"]
    _published_file("docs/DOSE_WINDOW_20261123.md")


def _check_dose_window_v7_results():
    """Continuation between the V4 under-pour and the first dose-window over-pour."""
    report = json.loads(_published_file("results/dose_window_v7_20261123.json").read_text(encoding="utf-8"))
    json.dumps(report, allow_nan=False)
    assert report["schema_version"] == 1 and report["kind"] == "curated_dose_window_v7_live_CEM_results"
    assert report["complete"] is True and report["neural_policy"] is False and report["ppo"] is False
    assert report["automatic_gui_deployment"] is False and report["test_used_for_selection"] is False
    assert report["recommended_role"] == "candidate"
    assert report["counts"] == {"pilot": 6, "training": 72, "validation": 12, "test": 18, "total": 108,
                                "generations": 3, "population": 4}
    protocol = report["protocol_summary"]
    assert protocol["candidate_free_params"] == protocol["search_center"]
    assert set(protocol["test_seeds"]) == {110101, 110102, 110103}
    previous = json.loads(_published_file("results/dose_window_20261123.json").read_text(encoding="utf-8"))
    assert set(protocol["test_seeds"]).isdisjoint(previous["protocol_summary"]["test_seeds"])
    rows = report["test"]["candidate"]["cases"]
    errors = [abs(row["error_ml"]) for row in rows]
    recorded = report["test"]["candidate"]["metrics"]
    assert abs(recorded["mean_absolute_error_ml"] - sum(errors) / len(errors)) < 1e-9
    assert recorded["within_0_2_ml"] == 2 and recorded["spill_cases"] == 1 and recorded["full_home_cases"] == 9
    assert report["test"]["baseline"]["metrics"]["within_0_2_ml"] == 0
    assert report["test"]["baseline"]["metrics"]["spill_cases"] == 0
    assert all(row["post_received_change_ml"] == 0 and row["physical_time_s"] <= 5 for row in rows)
    for path in report["policies"].values():
        payload = json.loads(_published_file(path["file"]).read_text(encoding="utf-8"))
        digest = hashlib.sha256(_published_file(path["file"]).read_bytes()).hexdigest()
        assert digest == path["sha256"] and payload["parameters"] == path["parameters"]
        assert payload["metadata"]["automatic_gui_deployment"] is False
    for item in report["images"].values():
        assert hashlib.sha256(_published_file(item["file"]).read_bytes()).hexdigest() == item["sha256"]
    _published_file("docs/DOSE_WINDOW_V7_20261123.md")


def _check_correction_12s_results():
    report = json.loads(_published_file("results/correction_12s_20261123.json").read_text(encoding="utf-8"))
    json.dumps(report, allow_nan=False)
    assert report["schema_version"] == 1 and report["kind"] == "curated_timed_correction_12s_results"
    assert report["complete"] is True and report["deadline_s"] == 12.0
    assert report["neural_policy"] is False and report["automatic_gui_deployment"] is False
    assert report["test_used_for_selection"] is False and report["recommended_role"] == "candidate"
    assert report["counts"] == {"pilot_10s": 6, "pilot_12s": 6, "validation": 12, "test": 18, "total": 42}
    assert report["protocol_summary"]["ten_second_pilot_passed"] is False
    assert set(report["protocol_summary"]["test_seeds"]) == {113101, 113102, 113103}
    previous = json.loads(_published_file("results/dose_window_v7_20261123.json").read_text(encoding="utf-8"))
    assert set(report["protocol_summary"]["test_seeds"]).isdisjoint(previous["protocol_summary"]["test_seeds"])
    rows = report["test"]["candidate"]["cases"]
    errors = [abs(row["error_ml"]) for row in rows]
    recorded = report["test"]["candidate"]["metrics"]
    assert len(rows) == 9 and abs(recorded["mean_absolute_error_ml"] - sum(errors) / 9) < 1e-9
    assert recorded["within_0_2_ml"] == 6 and recorded["full_home_cases"] == 9 and recorded["spill_cases"] == 1
    assert report["test"]["baseline"]["metrics"]["within_0_2_ml"] == 3
    for item in report["policies"].values():
        assert hashlib.sha256(_published_file(item["file"]).read_bytes()).hexdigest() == item["sha256"]
    for item in report["images"].values():
        assert hashlib.sha256(_published_file(item["file"]).read_bytes()).hexdigest() == item["sha256"]
    _published_file("docs/CORRECTION_12S_20261123.md")


def _check_sync_source_bindings():
    """Exercise reviewed snapshot selection and rejection of unknown edits."""
    import io
    import unittest
    path = _published_file("tools/test_sync_source_bindings.py")
    spec = importlib.util.spec_from_file_location("release_sync_source_tests", path)
    tests = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tests)
    suite = unittest.defaultTestLoader.loadTestsFromModule(tests)
    assert suite.countTestCases() >= 6
    output = io.StringIO()
    result = unittest.TextTestRunner(stream=output, verbosity=0).run(suite)
    assert result.wasSuccessful(), output.getvalue()


def _check_development_archive():
    """Check reference bytes and selected records, without new physics claims."""
    import ast
    from collections import Counter
    import re
    spec = importlib.util.spec_from_file_location("release_development_sources", ROOT/"tools/sync_from_workspace.py")
    sync = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sync)
    manifest_path = _published_file(sync.DEVELOPMENT_PUBLIC+"source_manifest.json")
    assert hashlib.sha256(manifest_path.read_bytes()).hexdigest() == sync.DEVELOPMENT_MANIFEST_SHA256
    manifest = json.loads(manifest_path.read_bytes())
    assert manifest["source_file_count"] == 19 and manifest["self_written_v8_python_files"] == 16
    assert manifest["notes_files"] == 1 and manifest["development_dependencies"] == 2
    assert manifest["capture_session_date"] == "2026-11-27"
    for key in ("historical_trial_source_binding_established", "independent_physics_audit", "turnkey_reproduction",
                "gui_default", "models_included", "raw_traces_included", "npz_or_sdk_included"):
        assert manifest[key] is False
    entries = manifest["entries"]
    assert len(entries) == 19 and {entry["capture_relative"] for entry in entries} == set(sync.DEVELOPMENT_RELATIVES)
    expected = {sync.DEVELOPMENT_PUBLIC+"source_manifest.json", sync.DEVELOPMENT_PUBLIC+"README.md"}
    py_count = 0
    for entry in entries:
        assert entry["source_alias"] == sync.DEVELOPMENT_CAPTURE+entry["capture_relative"]
        assert entry["published_file"] == sync.DEVELOPMENT_SOURCES[entry["source_alias"]]
        raw = _published_file(entry["published_file"]).read_bytes()
        assert b"\r" not in raw and len(raw) == entry["published_lf_bytes"]
        assert hashlib.sha256(raw).hexdigest() == entry["published_lf_sha256"]
        assert re.fullmatch(r"[0-9a-f]{64}", entry["raw_sha256"])
        assert not Path(entry["original_workspace_relative"]).is_absolute()
        assert ".." not in Path(entry["original_workspace_relative"]).parts
        if entry["published_file"].endswith(".py"):
            ast.parse(raw.decode("utf-8"), filename=entry["published_file"])
            py_count += 1
        expected.add(entry["published_file"])
    assert py_count == 18
    actual = _bundle_files(sync.DEVELOPMENT_PUBLIC)
    assert actual == expected, "Only reviewed self-written reference text belongs in this archive."
    assert hashlib.sha256(_published_file(sync.DEVELOPMENT_PUBLIC+"README.md").read_bytes()).hexdigest() == sync.DEVELOPMENT_README_SHA256
    for source, binding in sync.ARCHIVED_SOURCE_BINDINGS.items():
        assert hashlib.sha256(_published_file(source).read_bytes()).hexdigest() == binding["sha256"]

    path = _published_file("results/development_pilots_20261123_24.json")
    text = path.read_text(encoding="utf-8")
    assert not re.search(r"[A-Za-z]:[\\/]|file://|127\.0\.0\.1|AppData", text)
    report = json.loads(text)
    json.dumps(report, allow_nan=False)
    assert report["schema_version"] == 1 and report["kind"] == "curated_unfrozen_timed_correction_development_archive"
    assert report["archive_complete"] is True
    assert report["experiment_release_accepted"] is report["automatic_gui_deployment"] is False
    counts, runs = report["counts"], report["runs"]
    assert counts["runs"] == len(runs) == 21
    assert counts["committed_case_records"] == 192 and counts["total_attempts"] == 195
    assert counts["orphaned_partial_attempts"] == 3
    assert report["scope"]["excluded_previously_published_cases"] == 42
    assert report["scope"]["evaluation_status"] == "all cases exploratory and already seen; no fresh holdout claim"
    assert report["scope"]["new_simulation_run"] is False
    assert report["scope"]["raw_step_traces_decoded_by_curator"] is False
    all_cases = []

    def near(actual, expected_value):
        assert type(actual) in (int, float) and math.isfinite(actual)
        assert abs(actual-expected_value) < 1e-8

    def check_group(value, cases):
        n = len(cases)
        assert n and value["case_count"] == n
        assert value["case_ids"] == [case["case_id"] for case in cases]
        errors = [case["result"]["error_ml"] for case in cases]
        assert value["signed_errors_ml"] == errors
        near(value["mean_signed_error_ml"], sum(errors)/n)
        near(value["mae_ml"], sum(abs(error) for error in errors)/n)
        near(value["worst_absolute_error_ml"], max(abs(error) for error in errors))
        assert value["within_0_2_ml_descriptive_count"] == sum(abs(error) <= .2 for error in errors)
        near(value["mean_action_physical_seconds"], sum(case["result"]["physical_time_s"] for case in cases)/n)
        near(value["mean_action_wall_seconds"], sum(case["result"]["wall_seconds"] for case in cases)/n)
        assert value["case_footer_complete_count"] == sum(case["result"]["complete"] for case in cases)
        assert value["case_footer_valid_physics_count"] == sum(case["result"]["valid_physics"] for case in cases)
        assert value["full_home_true_recorded_count"] == sum(case["full_home_recorded"] is True for case in cases)
        assert value["case_with_any_maximum_spill_count"] == sum(case["result"]["max_spilled_ml"] > 0 for case in cases)
        assert value["end_spill_positive_recorded_count"] == sum((case["end_spilled_particles_recorded"] or 0) > 0 for case in cases)
        assert value["summary_is_descriptive_not_a_new_scoring_or_acceptance_rule"] is True

    for run in runs:
        assert run["run_id"] != "run_20261123"
        for flag in ("source_snapshot_present", "protocol_present", "input_freeze_present", "freeze_record_present",
                     "independently_audited_experiment", "verified_source_bound_frozen_model"):
            assert run[flag] is False
        cases = run["cases"]
        assert run["case_count"] == len(cases) and len({case["case_id"] for case in cases}) == len(cases)
        assert run["stage_counts"] == dict(Counter(case["stage"] for case in cases))
        for case in cases:
            result = case["result"]
            assert result["target_ml"] == case["target_ml"] and result["seed"] == case["seed"]
            near(result["received_ml"]-case["target_ml"], result["error_ml"])
            near(case["signed_error_percent_from_footer"], 100*result["error_ml"]/case["target_ml"])
            near(case["absolute_error_percent_from_footer"], 100*abs(result["error_ml"])/case["target_ml"])
            assert case["evidence_use"] == "exploratory_seen_not_fresh_holdout"
        check_group(run["statistics_all_cases_descriptive"], cases)
        for group in run["per_target_descriptive"]:
            check_group(group["metrics"], [case for case in cases if case["target_ml"] == group["target_ml"]])
        for group in run["per_stage_role_target"]:
            check_group(group["metrics"], [case for case in cases if all(case[key] == group[key] for key in ("stage", "role", "target_ml"))])
        assert "SOURCE-label" in run["legacy_field_semantics"]["cases.result.max_spilled_ml"]
        assert all(re.fullmatch(r"[0-9a-f]{64}", digest) for digest in run["artifacts_sha256"].values())
        all_cases.extend(cases)
    assert len(all_cases) == 192
    assert counts["unique_target_seed_pairs"] == len({(case["target_ml"], case["seed"]) for case in all_cases}) == 146
    assert counts["stage_counts"] == dict(Counter(case["stage"] for case in all_cases))
    assert counts["case_termination_counts"] == dict(Counter(case["result"]["termination_reason"] for case in all_cases))
    assert counts["case_footer_complete_count"] == sum(case["result"]["complete"] for case in all_cases) == 185
    assert counts["case_footer_valid_physics_count"] == sum(case["result"]["valid_physics"] for case in all_cases) == 190
    assert counts["legacy_case_footer_success_count"] == sum(case["result"]["success"] for case in all_cases) == 43
    assert counts["source_label_maximum_spill_positive_cases"] == 13 and counts["all_label_final_spill_positive_cases"] == 15
    orphans = report["orphaned_attempts"]
    assert len(orphans) == 3 and len({row["attempt_id"] for row in orphans}) == 3
    for orphan in orphans:
        assert orphan["result"] is None
        assert orphan["result_footer_present"] is orphan["committed_journal_row_present"] is False
        assert re.fullmatch(r"[0-9a-f]{64}", orphan["trace_sha256"])
        near(orphan["last_logged_physical_time_s"], orphan["last_logged_physics_step"]/60)
    assert report["metadata_conflicts"][0]["inherited_value"] == 5.0
    assert report["metadata_conflicts"][0]["actual_policy_deadlines_recorded_s"] == [12., 24., 36.]
    curator = _published_file(sync.DEVELOPMENT_PUBLIC+"projects/pour_policy_training_v8/curate_development_archive.py")
    assert hashlib.sha256(curator.read_bytes()).hexdigest() == report["archive_tool"]["sha256"]
    _published_file("docs/DEVELOPMENT_PILOTS_20261124.md")


def _check_fine_v9_results():
    """Consistency of the finished audited V9 study, never new physics proof."""
    import re
    spec = importlib.util.spec_from_file_location("release_fine_v9_sync",ROOT/"tools/sync_from_workspace.py")
    sync = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sync)
    report = json.loads(_published_file(sync.FINE_REPORT).read_text(encoding="utf-8"))
    json.dumps(report,allow_nan=False)
    assert report["schema_version"] == 1 and report["kind"] == "curated_fine_particle_v9_live_CEM_results"
    assert report["complete"] is True and report["counts"] == {"train":18,"validation":6,"test":18,"total":42}
    assert report["gui_deployment"] is False
    protocol,audit,freeze = report["protocol_summary"],report["audit"],report["freeze"]
    assert audit["artifact_integrity_verified"] is audit["all_failures_retained"] is True
    assert audit["episodes_audited"] == audit["raw_action_trace_count"] == 42
    assert audit["budgets"] == report["counts"]
    assert audit["source_count"] == 19 and audit["input_count"] == 6
    assert audit["protocol_id"] == report["protocol_id"] == freeze["protocol_id"]
    assert set(protocol["source_sha256"]) == set(sync.FINE_ORIGINAL_SOURCES)
    assert len(protocol["input_sha256"]) == 6
    assert protocol["user_criterion"] == {"deadline_s":24.,"relative_error":.001,"max_all_label_spill_particles":1,
                                         "full_original_home_and_12_quiet_frames":True}
    assert protocol["initial_particle_count"] == 11625 and protocol["initial_source_particles"] == 2880
    assert abs(protocol["particle_ml"]-.019683) < 1e-12
    assert abs(protocol["initial_source_ml"]-56.68704) < 1e-8
    sources = report["publication"]["source_mapping"]
    assert len(sources) == 19 and {entry["original_file"] for entry in sources} == set(sync.FINE_ORIGINAL_SOURCES)
    for entry in sources:
        assert entry["original_sha256"] == protocol["source_sha256"][entry["original_file"]]
        if entry["original_file"] != sync.FINE_ROBOT_SOURCE:
            assert entry["published_sha256"] == entry["original_sha256"]
        else:
            assert entry["original_sha256"] == sync.FINE_ROBOT_ORIGINAL_SHA256
            assert entry["published_sha256"] == sync.FINE_ROBOT_PUBLISHED_SHA256
        alias = sync.FINE_RUN+"/source_snapshot/"+entry["original_file"]
        assert entry["published_file"] == sync.FINE_RUNTIME_SOURCES[alias]
        raw = _published_file(entry["published_file"]).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == entry["published_sha256"]
        assert not re.search(rb"[A-Za-z]:[\\/]|file://|127\.0\.0\.1|AppData",raw)
        compile(raw.decode("utf-8"),entry["published_file"],"exec")
    tools = report["publication"]["tool_mapping"]
    assert len(tools) == 4 and {entry["original_file"] for entry in tools} == set(sync.FINE_TOOL_SOURCES)
    for entry in tools:
        assert entry["published_file"] == sync.FINE_TOOL_SOURCES[entry["original_file"]]
        assert entry["original_sha256"] == entry["published_sha256"]
        raw = _published_file(entry["published_file"]).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == entry["published_sha256"]
        compile(raw.decode("utf-8"),entry["published_file"],"exec")
    assert _published_file(sync.FINE_PUBLIC+".gitattributes").read_bytes() == sync.FINE_ATTRIBUTES.encode()
    assert _published_file(sync.FINE_PUBLIC+"README.md").read_bytes() == sync.FINE_README.encode()
    helper = sync._fine_helpers(_published_file(sync.FINE_TOOL_SOURCES["projects/pour_policy_training_v9/plot_fine_results.py"]))
    rows = report["all_cases"]
    assert len(rows) == 42 and len({row["episode_id"] for row in rows}) == 42
    assert {stage:sum(row["stage"] == stage for row in rows) for stage in ("train","validation","test")} == {
        "train":18,"validation":6,"test":18}
    for row in rows:
        assert row["assessment"] == helper.task_assessment(row)
        assert (row["result"]["target_ml"],row["result"]["seed"]) == (row["target_ml"],row["seed"])
        helper.near(row["result"]["received_ml"]-row["target_ml"],row["result"]["error_ml"])
        if row["stage"] == "test":
            from datetime import datetime
            assert datetime.fromisoformat(row["started_at_utc"]) > datetime.fromisoformat(freeze["frozen_at_utc"])
    generations = report["generations"]
    assert len(generations) == 2
    candidates = []
    for number,generation in enumerate(generations):
        assert generation["generation"] == number and len(generation["candidates"]) == 3
        for index,candidate in enumerate(generation["candidates"]):
            selected = [row for row in rows if row["stage"] == "train" and row["episode_id"].startswith(f"train_candidate_g{number}c{index}_")]
            helper.conditions(selected,protocol["train_seeds"],candidate["multiplier"],"candidate")
            helper.check_metrics(candidate["metrics"],selected)
            assert candidate["key"] == helper.selection_key(selected)
            candidates.append(candidate)
    best = min((candidate for candidate in candidates if candidate["key"][0] < 2),key=lambda c:c["key"])
    validation = {role:[row for row in rows if row["stage"] == "validation" and row["role"] == role]
                  for role in ("candidate","baseline")}
    for role in ("candidate","baseline"):
        helper.conditions(validation[role],protocol["validation_seeds"],best["multiplier"] if role == "candidate" else 1.,role)
        helper.check_metrics(report["validation"]["original_metrics"][role],validation[role])
    recommended = "candidate" if helper.selection_key(validation["candidate"]) <= helper.selection_key(validation["baseline"]) else "baseline"
    assert report["recommended_role"] == freeze["recommendation"] == recommended
    assert freeze["validation_only_selection"] is True and freeze["test_used_for_selection"] is False
    for role in ("candidate","baseline"):
        selected = [row for row in rows if row["stage"] == "test" and row["role"] == role]
        helper.conditions(selected,protocol["test_seeds"],best["multiplier"] if role == "candidate" else 1.,role)
        helper.check_metrics(report["test"]["original_metrics"][role],selected)
    assert report["task_passed"] is all(row["assessment"]["feasible"] for row in rows if row["stage"] == "test" and row["role"] == recommended)

    def descriptive(recorded,selected):
        actual = helper.descriptive(selected)
        mapped = {"n":"n","mae_ml":"mae_ml","worst_absolute_error_ml":"worst_ml",
            "dose_only_success_count":"dose_passed","joint_task_success_count":"feasible",
            "valid_evidence_count":"valid","peak_all_label_spill_particles":"peak_all_label_spill_including_post",
            "mean_action_physical_seconds_including_home_and_quiet":"mean_physical_s",
            "mean_action_wall_seconds":"mean_action_wall_seconds",
            "mean_case_wall_seconds_including_post_and_overhead":"mean_case_wall_seconds_including_post"}
        for name,key in mapped.items():
            helper.near(recorded[name],actual[key])
        assert recorded["signed_errors_ml"] == actual["signed_errors_ml"]
        assert recorded["old_footer_success_is_not_the_strict_percentage_task"] is True

    for stage in ("validation","test"):
        for role in ("candidate","baseline"):
            descriptive(report[stage]["metrics"][role],[row for row in rows if row["stage"] == stage and row["role"] == role])
    per_target = report["per_target"]
    assert len(per_target) == 3 and {entry["target_ml"] for entry in per_target} == {20,30,40}
    for entry in per_target:
        assert entry["tolerance_ml"] == entry["target_ml"]*.001
        for role in ("candidate","baseline"):
            descriptive(entry[role],[row for row in rows if row["stage"] == "test" and row["role"] == role and row["target_ml"] == entry["target_ml"]])
    pairs = report["test"]["paired_cases"]
    assert len(pairs) == 9 and {(pair["target_ml"],pair["seed"]) for pair in pairs} == {
        (target,seed) for target in (20,30,40) for seed in protocol["test_seeds"]}
    by_id = {row["episode_id"]:row for row in rows}
    for pair in pairs:
        assert pair["recommended_result_role"] == recommended
        for role in ("candidate","baseline"):
            view = pair[role]
            row = by_id[view["case_id"]]
            assert (row["stage"],row["role"],row["target_ml"],row["seed"]) == ("test",role,pair["target_ml"],pair["seed"])
            helper.near(view["received_ml"],row["result"]["received_ml"])
            helper.near(view["signed_error_ml"],row["result"]["error_ml"])
            assert view["assessment"] == row["assessment"]
    for source,(destination,role) in sync.FINE_MODEL_SOURCES.items():
        entry = report["models"][role]
        assert entry["file"] == destination
        raw = _published_file(destination).read_bytes()
        sync._validate_fine_checkpoint(raw.decode("utf-8"),role)
        assert hashlib.sha256(raw).hexdigest() == entry["sha256"] == freeze["models_sha256"][role+"_frozen.json"]
        model = json.loads(raw)
        expected_role = "candidate" if role == "candidate" else recommended
        assert model["implementation"] == expected_role
        assert model["source_sha256"] == protocol["source_sha256"] and model["protocol_id"] == report["protocol_id"]
        assert model["coarse_speed_multiplier"] == (best["multiplier"] if expected_role == "candidate" else 1.)
    # Exact bundle allowlist: reference sources, tools, two research models and
    # local byte-preservation/provenance metadata; no raw/NPZ/SDK/default entry.
    expected = {entry["published_file"] for entry in (*sources,*tools)}
    expected.update(entry["file"] for entry in report["models"].values())
    expected.update((sync.FINE_PUBLIC+".gitattributes",sync.FINE_PUBLIC+"README.md"))
    assert _bundle_files(sync.FINE_PUBLIC) == expected
    text = json.dumps(report,ensure_ascii=False)
    assert not re.search(r"[A-Za-z]:[\\/]|file://|127\.0\.0\.1|AppData",text)
    assert not (ROOT/"assets/robot_testtube_pour/policy/fine_v9.json").exists()
    for entry in report.get("plots",{}).values():
        raw = _published_file(entry["file"]).read_bytes()
        assert raw.startswith(b"\x89PNG\r\n\x1a\n") and hashlib.sha256(raw).hexdigest() == entry["sha256"]


def main():
    sources = list(ROOT.glob("*.py")) + list((ROOT / "tools").rglob("*.py"))
    sources += list((ROOT / "assets/robot_testtube_pour/scripts").glob("*.py"))
    for source in sources:
        compile(source.read_text(encoding="utf-8"), str(source), "exec")
    required = ("droplet_milkcrown_newton.py", "droplet_milkcrown_newton_surface.py",
                "droplet_milkcrown_newton_view.py",
                "robot_testtube_pour.py", "robot_target_volume_pour.py", "robot_target_volume_ai.py",
                "train_target_volume_policy.py", "train_target_volume_policy_full.py", "launch.py", "lab_paths.py")
    assert all((ROOT / name).is_file() for name in required), "Missing experiment source"
    for path in ROOT.rglob("*"):
        if not path.is_file() or ".git" in path.relative_to(ROOT).parts:
            continue
        rel = path.relative_to(ROOT)
        # Local runtime outputs are ignored by Git and do not affect a checkout.
        if rel.parts[0] in ("droplet_out", "test", "__pycache__"):
            continue
        if path.suffix in (".py", ".md", ".json"):
            assert path.stat().st_size < 2_000_000, f"Unexpectedly large text file: {rel}"
    data = json.loads((ROOT / "results/target_volume.json").read_text(encoding="utf-8"))
    assert data["success"] and len(data["results"]) == 3
    assert [r["target_ml"] for r in data["results"]] == [20, 30, 40]
    for result in data["results"]:
        assert abs(result["received_ml"] - result["target_ml"] - result["error_ml"]) < 1e-8
        assert abs(result["error_ml"]) < 2 and result["spilled_ml"] == 0
        assert result["mass_conserved"] and result["particle_count"] == 5330
    _check_ai_results()
    _check_full_results()
    _check_precision_results()
    _check_fast_results()
    _check_pivot_results()
    _check_v5_score_only()
    _check_dose_window_results()
    _check_dose_window_v7_results()
    _check_correction_12s_results()
    _check_sync_source_bindings()
    _check_development_archive()
    _check_fine_v9_results()
    spec = importlib.util.spec_from_file_location("check_adaptive_response_bundle", ROOT / "tools/adaptive_response_bundle.py")
    adaptive_bundle = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adaptive_bundle)
    adaptive_bundle.verify_public(ROOT)
    spec = importlib.util.spec_from_file_location("check_prefix_calibration_bundle", ROOT / "tools/prefix_calibration_bundle.py")
    prefix_bundle = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prefix_bundle)
    prefix_bundle.verify_public(ROOT)
    spec = importlib.util.spec_from_file_location("check_easy20_bundle", ROOT / "tools/easy20_bundle.py")
    easy_bundle = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(easy_bundle)
    easy_bundle.verify_public(ROOT)
    spec = importlib.util.spec_from_file_location("check_refine20_bundle", ROOT / "tools/refine20_bundle.py")
    refine_bundle = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(refine_bundle)
    refine_bundle.verify_public(ROOT)
    print(f"PASS: {len(sources)} Python sources compile; baseline, quick, full, V2, V3/V4, dose-window, 12 s correction results and 14 pure V5 score checks are consistent.")
    print("GPU physics is validated separately in Isaac Sim; this check does not run it.")
    print("Archived source selection rejects unreviewed edits and missing or changed snapshots.")
    print("Development archive: 19 reference files, 192 committed cases + 3 partial attempts; no new physics acceptance.")
    print("V9: complete audited 18/6/18 study, frozen raw research sources/models; no default GUI deployment.")
    print("V12: audited60 response collection plus10 development goals; empirical model schema/source/plots bound, no full-range acceptance or GUI default.")
    print("V13: twelve paired prefix-only cases archived separately; original goals unchanged, no new model/default/range claim.")
    print("Easy20: separate6/4/10 beginner task, original strict criterion retained; baseline/candidate/recommendation and paired tests bound, no GUI/default or arbitrary-range claim.")
    print("Refine20: separate3/6/4/10 single-correction curriculum; primary +/-0.5mL and both secondary criteria bound, train/validation-only choice, no default or arbitrary-range claim.")


if __name__ == "__main__":
    main()
