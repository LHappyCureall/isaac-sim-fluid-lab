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
    print(f"PASS: {len(sources)} Python sources compile; baseline, quick and full results/weights/plots are consistent.")
    print("GPU physics is validated separately in Isaac Sim; this check does not run it.")


if __name__ == "__main__":
    main()
