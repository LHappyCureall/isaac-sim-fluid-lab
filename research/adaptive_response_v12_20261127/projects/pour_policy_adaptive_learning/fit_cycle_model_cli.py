"""Fit an anchor-to-cutoff whole-cycle response model from one independently audited live collection.

This post-hoc CPU tool never runs physics, changes dataset splits, edits source
logs, or claims full pouring task acceptance.  A fitting output directory is
created once, outside the immutable collection directory.  Require the audit's
raw SHA explicitly so selecting a newer/different audit is a reviewable action.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import cycle_response_model as response_model
import cycle_samples


REQUIRED_ARTIFACTS = ("protocol.json", "summary.json", "job_state.json", "episodes.jsonl",
                      "responses.jsonl", "matrix.json", "versions.json")
AUDIT_SCOPE = "record_integrity_and_independent_physical_motion_mass_response_audit_not_task_success"


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _json(path):
    def invalid_constant(value):
        raise ValueError("JSON nonfinite constant is forbidden: " + value)
    return json.loads(Path(path).read_text(encoding="utf8"), parse_constant=invalid_constant)


def _jsonl(path, allow_empty=False):
    lines = Path(path).read_text(encoding="utf8").splitlines()
    _require((lines or allow_empty) and all(line.strip() for line in lines), "Dataset contains blank or missing records")
    rows = []
    for line in lines:
        value = json.loads(line)
        _require(isinstance(value, dict), "Dataset rows must be JSON objects")
        json.dumps(value, allow_nan=False)
        rows.append(value)
    return rows


def validate_audited_collection(run, audit_path, expected_audit_sha256):
    """Check finalized artifacts against the independently pinned audit."""
    run, audit_path = Path(run).resolve(), Path(audit_path).resolve()
    _require(run.is_dir(), "Collection directory does not exist")
    _require(isinstance(expected_audit_sha256, str) and len(expected_audit_sha256) == 64
             and all(char in "0123456789abcdef" for char in expected_audit_sha256),
             "--audit-sha256 must be an explicit lowercase SHA-256")
    _require(_sha(audit_path) == expected_audit_sha256, "Independent audit raw SHA mismatch")
    audit = _json(audit_path)
    _require(audit.get("complete") is True and audit.get("pass") is True
             and audit.get("scope") == AUDIT_SCOPE, "A complete independent physical/evidence audit is required")
    bindings = audit.get("input_artifact_sha256")
    _require(isinstance(bindings, dict) and set(REQUIRED_ARTIFACTS) <= set(bindings),
             "Audit lacks required collection artifact bindings")
    for name in REQUIRED_ARTIFACTS:
        _require(_sha(run / name) == bindings[name], "Audited collection artifact changed: " + name)
    protocol, summary, job, matrix = (_json(run / name) for name in
                                     ("protocol.json", "summary.json", "job_state.json", "matrix.json"))
    _require(summary.get("complete") is True and job.get("state") == "complete"
             and job.get("pid") is None, "Collection must be finalized before fitting")
    _require(not (run / "worker.lock").exists(), "An active/stale collection lock needs review before fitting")
    identifier = protocol.get("protocol_id")
    _require(isinstance(identifier, str) and len(identifier) == 64
             and summary.get("protocol_id") == audit.get("protocol_id") == identifier,
             "Protocol ID differs across audit and completed collection")
    _require(protocol.get("kind") == "adaptive_real_cycle_response_collection_v12"
             and summary.get("kind") == protocol["kind"], "Unexpected collection protocol kind")
    _require(protocol.get("deadline_s") == 24. and protocol.get("relative_tolerance") == .001
             and protocol.get("peak_all_label_spill_limit") == 1
             and protocol.get("particle_ml") == .0027 ** 3 * 1e6,
             "Collection differs from fixed 24s/0.1%/one-particle contract")
    rows = _jsonl(run / "responses.jsonl", allow_empty=True)
    episodes = _jsonl(run / "episodes.jsonl")
    _require(summary.get("responses_sha256") == audit.get("responses_sha256") == bindings["responses.jsonl"],
             "Response dataset differs from audit/summary raw SHA")
    _require(summary.get("response_samples") == audit.get("response_samples") == len(rows),
             "Response counts differ across artifacts")
    _require(summary.get("episodes") == audit.get("episodes_audited") == len(episodes),
             "Completed episode counts differ across artifacts")
    _require(matrix.get("draft") is False and protocol.get("matrix") == matrix
             and len(matrix.get("cases", [])) == len(episodes), "Final matrix does not match complete collection")
    _require(isinstance(audit.get("audit_tool_sha256"), str)
             and len(audit["audit_tool_sha256"]) == 64, "Audit source SHA is required")
    # Rows must also match their raw journal projection byte-for-value. The
    # independent audit proves the endpoint/event/physics meaning of this data.
    projected = [sample for episode in episodes for sample in episode.get("response_samples", [])]
    _require(rows == projected, "Response dataset is not exactly the audited episode projection")
    cycle_projection = {"samples": [], "partial_cycles": []}
    for episode in episodes:
        derived = cycle_samples.cycles_from_episode(episode)
        cycle_projection["samples"].extend(derived["samples"])
        cycle_projection["partial_cycles"].extend(derived["partial_cycles"])
    projection_sha = hashlib.sha256(json.dumps(cycle_projection, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode("utf8")).hexdigest()
    _require(audit.get("cycle_projection_sha256") == projection_sha,
             "Whole-cycle projection differs from independent audit")
    _require(audit.get("cycle_samples_source_sha256") == _sha(cycle_samples.__file__),
             "Whole-cycle endpoint projector differs from independently audited source")
    rows = cycle_projection["samples"]
    partials = cycle_projection["partial_cycles"]
    _require(all(row.get("split") in ("train", "validation") for row in partials),
             "Development/test partial cycles cannot enter formal fitting")
    _require(all(row.get("split") in ("train", "validation") for row in rows),
             "Development/test rows cannot be relabeled or pooled into fitting")
    training = [row for row in rows if row["split"] == "train"]
    validation = [row for row in rows if row["split"] == "validation"]
    _require(training and validation, "Fitting requires explicit independent train and validation responses")
    episode_by_id = {episode["episode_id"]: episode for episode in episodes}
    _require(len(episode_by_id) == len(episodes), "Duplicate completed episode identity")
    matrix_by_id = {case["case_id"]: case for case in matrix["cases"]}
    _require(set(matrix_by_id) == set(episode_by_id), "Matrix/episode identity coverage differs")
    for episode in episodes:
        case = matrix_by_id[episode["episode_id"]]
        _require(episode.get("split") == case.get("split")
                 and episode.get("seed") == case.get("seed")
                 and episode.get("target_ml") == case.get("target_ml"),
                 "An episode or partial response changed its frozen split/seed/target")
    for row in rows:
        episode = episode_by_id.get(row.get("episode_id"))
        _require(episode is not None, "Response does not identify a completed source episode")
        case = matrix_by_id[episode["episode_id"]]
        _require(row["split"] == episode.get("split") == case.get("split")
                 and row.get("seed") == episode.get("seed") == case.get("seed")
                 and row.get("target_ml") == episode.get("target_ml") == case.get("target_ml"),
                 "Response split/seed/target changed from its frozen matrix")
    return {"run": run, "audit": audit, "protocol": protocol, "summary": summary,
            "artifact_sha256": {name: bindings[name] for name in REQUIRED_ARTIFACTS},
            "training": training, "validation": validation,
            "cycle_projection": cycle_projection, "cycle_projection_sha256": projection_sha,
            "partial_training": [row for row in partials if row["split"] == "train"],
            "partial_validation": [row for row in partials if row["split"] == "validation"]}


def fit_audited_collection(run, audit_path, expected_audit_sha256, output, *,
                           config_json=None, dry_run=False):
    """CPU-fit and create candidate/metrics once, with unchanged raw evidence."""
    validated = validate_audited_collection(run, audit_path, expected_audit_sha256)
    run, output = validated["run"], Path(output).resolve()
    _require(not output.is_relative_to(run) and not run.is_relative_to(output),
             "Model output must be outside and not an ancestor of the immutable collection")
    _require(not output.exists(), "Model output already exists; use a new named fit directory")
    configs, config_sha = None, None
    if config_json is not None:
        config_json = Path(config_json).resolve()
        configs = _json(config_json)
        _require(isinstance(configs, list) and configs, "Configuration file must be an explicit nonempty list")
        config_sha = _sha(config_json)
    provenance = {"collection_protocol_id": validated["protocol"]["protocol_id"],
                  "collection_artifact_sha256": validated["artifact_sha256"],
                  "independent_audit_raw_sha256": expected_audit_sha256,
                  "independent_audit_tool_sha256": validated["audit"]["audit_tool_sha256"],
                  "configuration_file_raw_sha256": config_sha,
                  "fit_tool_sha256": _sha(__file__),
                  "cycle_projection_sha256": validated["cycle_projection_sha256"],
                  "cycle_samples_source_sha256": _sha(cycle_samples.__file__)}
    model = response_model.fit_response_model(validated["training"], validated["validation"],
                                              configs=configs, provenance=provenance,
                                              partial_training_rows=validated["partial_training"],
                                              partial_validation_rows=validated["partial_validation"])
    selected = model["configuration_selection"][model["selected_config_index"]]
    manifest = {"complete": True, "kind": "audited_whole_cycle_response_model_fit_v1",
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                **provenance, "selection_kind": model["selection_kind"],
                "selected_config_index": model["selected_config_index"],
                "training_rows": model["training_row_count"],
                "training_eligible_rows": model["training_eligible_rows"],
                "training_hazard_rows": model["training_hazard_rows"],
                "training_zero_dose_rows": model["training_zero_dose_rows"],
                "partial_training_cycles": model["partial_training_cycles"],
                "partial_validation_cycles": model["partial_validation_cycles"],
                "validation_rows": model["validation_row_count"],
                "training_seeds": model["training_seeds"],
                "validation_seeds": model["validation_seeds"],
                "validation_metrics": selected["validation"],
                "validation_complete_local_support": model["validation_complete_local_support"],
                "validation_all_responses_valid": model["validation_all_responses_valid"],
                "candidate_usable_for_local_engineering": bool(model["validation_complete_local_support"]
                    and model["validation_all_responses_valid"]),
                "terminal_task_acceptance_proven": False,
                "new_physics_execution": False, "default_gui_deployment": False,
                "test_data_used": False, "dataset_splits_reassigned": False,
                "model_content_sha256": model["content_sha256"]}
    if dry_run:
        return {"dry_run": True, **manifest}
    # Exclusive directory creation and exclusive model files prevent accidental
    # overwrites of a previously frozen candidate or an interrupted fit.
    output.mkdir(parents=True, exist_ok=False)
    with (output / "cycle_responses.jsonl").open("x", encoding="utf8") as stream:
        for row in validated["cycle_projection"]["samples"]:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
    with (output / "partial_cycles.json").open("x", encoding="utf8") as stream:
        json.dump(validated["cycle_projection"]["partial_cycles"], stream, indent=2, allow_nan=False)
        stream.write("\n")
    manifest["cycle_responses_raw_sha256"] = _sha(output / "cycle_responses.jsonl")
    manifest["partial_cycles_raw_sha256"] = _sha(output / "partial_cycles.json")
    receipt = response_model.save_response_model(output / "candidate.json", model)
    manifest["candidate_raw_sha256"] = receipt["raw_sha256"]
    with (output / "fit_manifest.json").open("x", encoding="utf8") as stream:
        json.dump(manifest, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return {"dry_run": False, **manifest}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--audit-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config-json", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    result = fit_audited_collection(args.run, args.audit, args.audit_sha256, args.output,
                                    config_json=args.config_json, dry_run=args.dry_run)
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, TypeError) as error:
        print("Whole-cycle fitting refused: " + str(error), file=sys.stderr)
        raise SystemExit(1)
