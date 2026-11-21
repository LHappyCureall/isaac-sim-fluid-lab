"""Supplementary live audit of the already-trained CEM candidate.

This never fits a model or changes the deployment chosen by the original
validation set. Six new candidate episodes reuse the completed quick run's
six paired baseline results. Candidate parameters and a separate audit
protocol are frozen before the first supplementary candidate episode.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN = ROOT / "droplet_out" / "pour_policy_training" / "run_20261121_quick"
GUI_CANDIDATE = ROOT / "assets" / "robot_testtube_pour" / "policy" / "target_volume_candidate.json"


def _now():
    return datetime.now(timezone.utc).isoformat()


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode("utf-8")).hexdigest()


def _write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _load_source(run):
    run = Path(run).resolve()
    summary_path = run / "summary.json"
    if not summary_path.is_file():
        raise ValueError("The original quick summary is not complete; no candidate audit can start yet.")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("complete") is not True or summary.get("test_used_for_training_or_selection") is not False:
        raise ValueError("Require a completed original run with test excluded from model selection.")
    search = json.loads((run / "search_state.json").read_text(encoding="utf-8"))
    if search["protocol_id"] != summary["protocol_id"]:
        raise ValueError("Search state and original summary have different protocols.")
    params = search["best_params"]
    if params != summary["training"]["best_params"]:
        raise ValueError("The candidate is not the original training-best parameter set.")
    bounds = summary["protocol"]["parameter_bounds"]
    if set(params) != set(bounds):
        raise ValueError("Candidate parameter names differ from the original protocol.")
    for name, value in params.items():
        if not math.isfinite(float(value)) or not bounds[name][0] <= value <= bounds[name][1]:
            raise ValueError("Candidate parameters are outside the original reviewed bounds.")
    expected = {(float(target), int(seed)) for target in summary["protocol"]["targets_ml"]
                for seed in summary["protocol"]["test_seeds"]}
    if expected != {(float(target), seed) for target in (20, 30, 40) for seed in (1001, 1002)}:
        raise ValueError("This supplementary audit expects exactly targets 20/30/40 and test seeds 1001/1002.")
    pairs = {}
    for pair in summary["test"]["paired_results"]:
        key = (float(pair["target_ml"]), int(pair["seed"]))
        if key in pairs or key not in expected:
            raise ValueError("Original held-out pairs are duplicated or unexpected.")
        result = pair["baseline"]
        if (float(result["target_ml"]), int(result["seed"])) != key:
            raise ValueError("Original baseline conditions differ from their paired labels.")
        pairs[key] = pair
    if set(pairs) != expected:
        raise ValueError("Original baseline results are incomplete.")
    selected = run / "model_frozen.json"
    if _sha(selected) != summary["selected"]["model_sha256"]:
        raise ValueError("The original frozen deployment checkpoint changed.")
    template = json.loads(selected.read_text(encoding="utf-8"))
    for relative, expected_sha in summary["protocol"]["source_sha256"].items():
        path = (ROOT / relative).resolve()
        if not path.is_relative_to(ROOT) or _sha(path) != expected_sha:
            raise ValueError(f"Frozen original source changed: {relative}")
    return summary, search, pairs, template


def _prepare(run, output, resume):
    summary, search, pairs, template = _load_source(run)
    output.mkdir(parents=True, exist_ok=True)
    model_path = output / "candidate_frozen.json"
    protocol_path = output / "candidate_audit_protocol.json"
    existing = None
    if protocol_path.exists():
        if not resume:
            raise FileExistsError("Audit protocol already exists; use --resume or a new output directory.")
        existing = json.loads(protocol_path.read_text(encoding="utf-8"))
        if not model_path.is_file():
            raise ValueError("The audit's frozen candidate is missing.")
        model = json.loads(model_path.read_text(encoding="utf-8"))
    else:
        if any((output / name).exists() for name in ("candidate_frozen.json", "audit_episodes.jsonl", "candidate_audit_summary.json")):
            raise FileExistsError("Partial or unrelated audit output exists; choose a clean output directory.")
        scores = summary["validation"]["scores"]
        model = {**template, "parameters": dict(search["best_params"]), "metadata": {
            "frozen_at_utc": _now(), "source_protocol_id": summary["protocol_id"],
            "candidate_origin": "original_CEM_training_best; no refitting",
            "selection_origin": "research_candidate_not_default_deployment",
            "is_trained": True, "default_deployment": False,
            "training_episodes": summary["training"]["episodes"],
            "original_validation_scores": scores,
            "candidate_weaker_on_original_validation": scores["training_best"] >= scores["baseline"],
            "original_validation_deployment": summary["selected"]["selection"],
            "supplementary_test_used_for_selection": False,
            "notice": "Trained candidate for comparison; original validation selected the default policy.",
            "source_sha256": summary["protocol"]["source_sha256"]}}
        _write(model_path, model)
    if model["parameters"] != search["best_params"]:
        raise ValueError("Frozen audit candidate differs from the original training winner.")
    protocol = {"schema_version": 1, "kind": "supplementary_candidate_evaluation_not_training",
        "source_protocol_id": summary["protocol_id"], "source_summary_sha256": _sha(run / "summary.json"),
        "search_state_sha256": _sha(run / "search_state.json"), "audit_script_sha256": _sha(__file__),
        "candidate_model_sha256": _sha(model_path), "candidate_parameters": dict(search["best_params"]),
        "candidate_frozen_at_utc": model["metadata"]["frozen_at_utc"],
        "targets_ml": [20, 30, 40], "test_seeds": [1001, 1002], "new_live_episodes": 6,
        "new_training_episodes": 0, "baseline_reused_from_completed_quick_summary": True,
        "baseline_pairs_sha256": _signature([pairs[key]["baseline"] for key in sorted(pairs)]),
        "physics": summary["protocol"]["physics"], "randomization": summary["protocol"]["randomization"],
        "original_validation_scores": summary["validation"]["scores"],
        "deployment_selection": summary["selected"]["selection"],
        "deployment_model_sha256": summary["selected"]["model_sha256"],
        "test_used_for_training_or_deployment_selection": False,
        "timing_notice": "Supplementary audit specified after original validation/test; candidate fixed by training before this audit.",
        "source_sha256": summary["protocol"]["source_sha256"]}
    protocol_id = _signature(protocol)
    if existing is not None:
        if existing.get("audit_protocol_id") != protocol_id or existing.get("protocol") != protocol:
            raise ValueError("Resume rejected: candidate, source, baseline or audit protocol changed.")
    else:
        _write(protocol_path, {"created_at_utc": _now(), "audit_protocol_id": protocol_id, "protocol": protocol})
        (output / "candidate_frozen.sha256").write_text(protocol["candidate_model_sha256"] + "  candidate_frozen.json\n", encoding="utf-8")
    return summary, pairs, model, protocol, protocol_id


def _load_journal(output, protocol_id):
    path = output / "audit_episodes.jsonl"
    records = {}
    if not path.exists():
        return records
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    good = []
    for index, line in enumerate(lines):
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            if index == len(lines) - 1 and not line.endswith("\n"):
                path.write_text("".join(good), encoding="utf-8")
                break
            raise ValueError("Malformed committed candidate-audit journal.")
        if record["audit_protocol_id"] != protocol_id or record["episode_id"] in records:
            raise ValueError("Journal protocol differs or contains a duplicate episode.")
        raw = (output / record["record_path"]).resolve()
        if not raw.is_relative_to(output) or not raw.is_file() or _sha(raw) != record["record_sha256"]:
            raise ValueError("Candidate-audit trace is missing, changed or outside its output directory.")
        if _signature(record["conditions"]) != record["conditions_sha256"]:
            raise ValueError("Candidate-audit episode conditions changed.")
        records[record["episode_id"]] = record
        good.append(line if line.endswith("\n") else line + "\n")
    if lines and not lines[-1].endswith("\n") and len(good) == len(lines):
        path.write_text("".join(good), encoding="utf-8")
    return records


def _safe(result):
    return (result["valid_physics"] and result["complete"] and result["upright"]
            and result["max_spilled_ml"] == 0 and result["inflight_ml"] <= 0.1)


def _metrics(results):
    errors = [abs(float(result["error_ml"])) for result in results]
    return {"episodes": len(results), "mae_ml": sum(errors) / len(errors),
        "worst_absolute_error_ml": max(errors),
        "within_0_2_ml_rate": sum(_safe(r) and e <= .2 for r, e in zip(results, errors)) / len(results),
        "within_0_5_ml_rate": sum(_safe(r) and e <= .5 for r, e in zip(results, errors)) / len(results),
        "safe_completion_rate": sum(bool(_safe(result)) for result in results) / len(results),
        "all_mass_conserved": all(result["mass_conserved"] for result in results),
        "max_spilled_ml": max(result["max_spilled_ml"] for result in results),
        "physical_steps": sum(result["physical_steps"] for result in results),
        "physical_seconds": sum(result["physical_time_s"] for result in results)}


def run_audit(run, output=None, *, resume=False, prepare_only=False,
              environment_factory=None, policy_factory=None, publish_candidate=True):
    """CPU-testable orchestration; the normal path uses the original live env."""
    run = Path(run).resolve()
    output = (Path(output) if output is not None else run / "candidate_audit").resolve()
    if output == run:
        raise ValueError("Use a separate candidate_audit directory; do not write over the original run.")
    summary, baseline_pairs, model, protocol, protocol_id = _prepare(run, output, resume)
    records = _load_journal(output, protocol_id)
    if prepare_only:
        return {"prepared": True, "audit_protocol_id": protocol_id,
                "candidate_model_sha256": protocol["candidate_model_sha256"], "new_live_episodes": 6}
    original_model = run / "model_frozen.json"
    environment = None
    started = time.perf_counter()
    newly_run = 0
    for seed in protocol["test_seeds"]:
        for target in protocol["targets_ml"]:
            episode_id = f"candidate_test_t{target}_s{seed}"
            conditions = {"target_ml": float(target), "seed": int(seed), "randomized": True,
                          "policy_parameters": model["parameters"],
                          "candidate_model_sha256": protocol["candidate_model_sha256"]}
            if episode_id in records:
                if records[episode_id]["conditions"] != conditions:
                    raise ValueError("Resumed candidate-audit conditions differ.")
                continue
            if environment is None:
                if environment_factory is None:
                    sys.path.insert(0, str(ROOT))
                    import train_target_volume_policy as training
                    training.configure_sdk()
                    from pour_training_env import PourTrainingEnv
                    from learned_policy import LearnedPourPolicy
                    environment = PourTrainingEnv()
                    if protocol["physics"]["cuda_graph"]:
                        training._enable_graph(environment)
                    policy_factory = LearnedPourPolicy
                else:
                    environment = environment_factory()
                    if policy_factory is None:
                        raise ValueError("A CPU fake environment requires an explicit fake policy factory.")
            def progress(event):
                _write(output / "audit_progress.json", {"state": "running", "updated_at_utc": _now(),
                    "audit_protocol_id": protocol_id, "episode_id": episode_id,
                    "completed_new_episodes": len(records), "total_new_episodes": 6,
                    "stage": "supplementary_independent_candidate_evaluation", **event})
            progress({"event": "start"})
            print(f"CANDIDATE AUDIT {len(records) + 1}/6: {episode_id}", flush=True)
            raw = output / "traces" / f"{episode_id}.steps.jsonl.gz"
            result = environment.run_episode(target, seed, policy=policy_factory(model["parameters"]),
                                             randomize=True, record_path=raw, progress=progress)
            if (float(result["target_ml"]), int(result["seed"])) != (float(target), int(seed)):
                raise ValueError("Live candidate result returned different conditions.")
            if not raw.is_file():
                raise ValueError("Live candidate episode did not preserve its physical-step trace.")
            record = {"episode_id": episode_id, "audit_protocol_id": protocol_id,
                "conditions": conditions, "conditions_sha256": _signature(conditions),
                "record_path": raw.relative_to(output).as_posix(), "record_sha256": _sha(raw),
                "completed_at_utc": _now(), "result": result}
            with (output / "audit_episodes.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, allow_nan=False, separators=(",", ":")) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            records[episode_id] = record
            newly_run += 1
            progress({"event": "episode_recorded", "received_ml": result["received_ml"], "error_ml": result["error_ml"]})
    _load_source(run)  # Recheck the frozen physical sources after all live episodes.
    if (_sha(run / "summary.json") != protocol["source_summary_sha256"]
            or _sha(run / "search_state.json") != protocol["search_state_sha256"]
            or _sha(__file__) != protocol["audit_script_sha256"]):
        raise ValueError("Original results or audit implementation changed during supplementary testing.")
    if _sha(original_model) != protocol["deployment_model_sha256"] or _sha(output / "candidate_frozen.json") != protocol["candidate_model_sha256"]:
        raise ValueError("The original deployment or frozen audit candidate changed during testing.")
    paired = []
    for seed in protocol["test_seeds"]:
        for target in protocol["targets_ml"]:
            old = baseline_pairs[(float(target), int(seed))]
            paired.append({"target_ml": target, "seed": seed,
                "candidate": records[f"candidate_test_t{target}_s{seed}"]["result"],
                "baseline": old["baseline"], "original_selected": old["selected"]})
    candidate_metrics = _metrics([pair["candidate"] for pair in paired])
    baseline_metrics = _metrics([pair["baseline"] for pair in paired])
    report = {"complete": True, "completed_at_utc": _now(), "audit_protocol_id": protocol_id,
        "protocol": protocol, "source_protocol_id": summary["protocol_id"],
        "candidate_parameters": model["parameters"], "paired_results": paired,
        "candidate_metrics": candidate_metrics, "baseline_metrics": baseline_metrics,
        "candidate_mae_improved": candidate_metrics["mae_ml"] < baseline_metrics["mae_ml"],
        "candidate_safe_completion_not_worse": candidate_metrics["safe_completion_rate"] >= baseline_metrics["safe_completion_rate"],
        "new_training_episodes": 0, "additional_evaluation_episodes": len(records),
        "new_episodes_this_invocation": newly_run,
        "total_batch_rollouts_with_this_audit": summary["total_recorded_episodes"] + len(records),
        "total_independent_evaluation_rollouts": 2 * len(paired) + len(records),
        "original_training_episodes": summary["training"]["episodes"],
        "original_validation_deployment": summary["selected"]["selection"],
        "deployment_changed": False, "test_used_for_training_or_deployment_selection": False,
        "wall_seconds_this_invocation": time.perf_counter() - started,
        "notice": "Supplementary test describes the trained candidate; original validation alone determines default deployment."}
    _write(output / "candidate_audit_summary.json", report)
    if publish_candidate:
        GUI_CANDIDATE.parent.mkdir(parents=True, exist_ok=True)
        temporary = GUI_CANDIDATE.with_name(GUI_CANDIDATE.name + ".tmp")
        temporary.write_bytes((output / "candidate_frozen.json").read_bytes())
        os.replace(temporary, GUI_CANDIDATE)
    _write(output / "audit_progress.json", {"state": "complete", "updated_at_utc": _now(),
        "audit_protocol_id": protocol_id, "completed_new_episodes": len(records),
        "total_new_episodes": 6, "candidate_model_sha256": protocol["candidate_model_sha256"],
        "deployment_changed": False})
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", nargs="?", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--prepare-only", action="store_true", help="Freeze protocol/checkpoint without starting GPU episodes.")
    args = parser.parse_args(argv)
    try:
        report = run_audit(args.run, args.output, resume=args.resume, prepare_only=args.prepare_only)
        print(json.dumps({key: report[key] for key in report if key not in ("protocol", "paired_results")}, indent=2), flush=True)
        return 0
    except BaseException as error:
        if not args.prepare_only:
            output = (args.output if args.output is not None else args.run / "candidate_audit").resolve()
            if (output / "candidate_audit_protocol.json").is_file():
                _write(output / "audit_progress.json", {"state": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                    "updated_at_utc": _now(), "error_type": type(error).__name__, "error": str(error)})
        if isinstance(error, KeyboardInterrupt):
            raise
        print(f"Candidate audit stopped: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
