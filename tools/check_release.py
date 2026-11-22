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


def main():
    sources = list(ROOT.glob("*.py")) + list((ROOT / "tools").rglob("*.py"))
    sources += list((ROOT / "assets/robot_testtube_pour/scripts").glob("*.py"))
    for source in sources:
        compile(source.read_text(encoding="utf-8"), str(source), "exec")
    required = ("droplet_milkcrown_newton.py", "droplet_milkcrown_newton_surface.py",
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
    print(f"PASS: {len(sources)} Python sources compile; baseline, quick, full and precision results/weights are consistent.")
    print("GPU physics is validated separately in Isaac Sim; this check does not run it.")


if __name__ == "__main__":
    main()
