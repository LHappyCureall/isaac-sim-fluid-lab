"""Explicit additive prefix-calibration archive; no model or default upgrade."""

import ast
import hashlib
import importlib.util
import json
from pathlib import Path


RUN = "droplet_out/pour_policy_adaptive_response/prefix_calibration_001"
PUBLIC = "research/adaptive_response_v12_20261127/"
HERE = "projects/pour_policy_adaptive_learning/"
REPORT = "results/prefix_calibration_v13_20261127.json"
PROTOCOL_ID = "c5eaf9b6a2f8e6ac9823182e20936ccf31e6ecd287705075b3bd5c5483cd47d4"
RUNTIME = {
    HERE + "prefix_control.py": "7fd23b7d35bc1de8b181a8a2784231b0c6812a6a426211a8a000cb0f83dea699",
    HERE + "run_prefix_calibration.py": "d9b892bbea6dd392f06c6b16c2e88de1ccaa1f4d5c7cf2ce2efdd1fbaa67bccf",
}
TOOLS = {
    HERE + "audit_prefix_calibration.py": "d0e6acc33e93baa57edb6e8b689f79d7b5d251821bc20a8167b192ee49d9aec6",
    HERE + "test_prefix_control.py": "2be8dc69b9338722d89e32cdc4273e1e2b1ebee7afd5cb9dfe648e73fb712b63",
    HERE + "test_audit_prefix_calibration.py": "229541d2211fed25b09cebe29fb7bf42633113e964b162d66855ca79015e632d",
    HERE + "curate_prefix_results.py": "7868cffd4a3be291bc3e03afc83babbfe86b1d8d838c9bd57091b92d402f4e25",
}
EXTRA_NAMES = (*RUNTIME, *TOOLS)
MANIFEST = PUBLIC + "prefix_source_manifest.json"


def need(value, message):
    if not value:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read(path):
    value = json.loads(Path(path).read_text(encoding="utf8"))
    json.dumps(value, allow_nan=False)
    return value


def load_base(project):
    spec = importlib.util.spec_from_file_location("prefix_base_publication_rules", project / "tools/adaptive_response_bundle.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare(workspace, project, prior_prepared):
    workspace, project = Path(workspace).resolve(), Path(project).resolve()
    base = load_base(project)
    summary, job = read(workspace / RUN / "summary.json"), read(workspace / RUN / "job_state.json")
    need(summary["complete"] is True and summary["episodes"] == 12 and job["state"] == "complete"
         and job["pid"] is None, "Prefix calibration is incomplete; publication is premature")
    for name, expected in TOOLS.items():
        need(digest((workspace / name).read_bytes()) == expected, "Reviewed prefix tool changed: " + name)
    spec = importlib.util.spec_from_file_location("publication_checked_prefix_curator", workspace / HERE / "curate_prefix_results.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    audit = workspace / RUN / "prefix_audit.json"
    expected = module.curate(workspace / RUN, audit, digest(audit.read_bytes()), workspace)
    selected = read(project / REPORT)
    base.private_paths((project / REPORT).read_bytes())
    need({k: v for k, v in expected.items() if k != "record_created_at_utc"}
         == {k: v for k, v in selected.items() if k != "record_created_at_utc"},
         "Selected prefix data differ from all twelve audited cases")
    plan = read(workspace / RUN / "protocol.json")
    need(plan["protocol_id"] == PROTOCOL_ID and len(plan["source_sha256"]) == 27
         and len(plan["input_sha256"]) == 6, "Prefix runtime/source/input identity changed")
    prepared = dict(prior_prepared)
    parent = json.loads(prepared[PUBLIC + "source_manifest.json"])
    parent_runtime = {entry["original_file"]: entry for entry in parent["files"]
                      if entry["role"] == "collection_runtime"}
    shared = set(parent_runtime) - {HERE + "collect_cycles.py"}
    need(set(plan["source_sha256"]) == shared | set(RUNTIME), "Prefix must preserve exact25 shared dependencies")
    shared_bindings = []
    for name in sorted(shared):
        entry = parent_runtime[name]
        need(plan["source_sha256"][name] == entry["original_sha256"]
             and digest(prepared[entry["published_file"]]) == entry["published_sha256"],
             "Shared prefix dependency differs from unchanged V12 archive: " + name)
        shared_bindings.append(entry)
    files, mapping = [], []
    for name, expected_sha in {**RUNTIME, **TOOLS}.items():
        source = workspace / RUN / "source_snapshot" / name if name in RUNTIME else workspace / name
        raw = source.read_bytes()
        need(digest(raw) == expected_sha, "Explicit prefix source changed: " + name)
        base.private_paths(raw)
        ast.parse(raw.decode("utf8"), filename=name)
        destination = PUBLIC + name
        files.append((destination, raw))
        mapping.append({"role": "prefix_runtime" if name in RUNTIME else "prefix_posthoc_or_test",
            "original_file": name, "published_file": destination,
            "original_sha256": expected_sha, "published_sha256": expected_sha})
    manifest = {"schema_version": 1, "kind": "prefix_calibration_v13_additive_publication",
        "protocol_id": PROTOCOL_ID, "source_count": 27, "shared_source_count": 25,
        "report_file": REPORT, "report_sha256": digest((project / REPORT).read_bytes()),
        "independent_audit_raw_sha256": selected["independent_audit_raw_sha256"],
        "audit_tool_sha256": TOOLS[HERE + "audit_prefix_calibration.py"],
        "parent_bundle_manifest_sha256": digest(prepared[PUBLIC + "source_manifest.json"]),
        "parent_bundle_manifest_unchanged": True, "files": mapping, "shared_source_bindings": shared_bindings,
        "cases": 12, "new_model": False, "default_gui_deployment": False,
        "only_internal_coarse_control_target_offset_changed": True,
        "actual_requested_target_and_measurement_unchanged": True,
        "raw_logs_or_caches_distributed": False}
    data = json.dumps(manifest, indent=2, allow_nan=False).encode() + b"\n"
    base.private_paths(data)
    files.append((MANIFEST, data))
    return files


def verify_public(project):
    project = Path(project).resolve()
    base = load_base(project)
    manifest, selected = read(project / MANIFEST), read(project / REPORT)
    need(manifest["schema_version"] == 1 and manifest["kind"] == "prefix_calibration_v13_additive_publication"
         and manifest["protocol_id"] == selected["protocol_id"] == PROTOCOL_ID
         and selected["kind"] == "adaptive_coarse_prefix_calibration_v13" and selected["complete"] is True,
         "Prefix selected schema/protocol differs")
    need(manifest["report_sha256"] == digest((project / REPORT).read_bytes())
         and manifest["report_file"] == REPORT and manifest["source_count"] == 27
         and manifest["shared_source_count"] == 25 and manifest["cases"] == 12,
         "Prefix report/source count binding differs")
    need(manifest["parent_bundle_manifest_unchanged"] is True and
         manifest["parent_bundle_manifest_sha256"] == digest((project / PUBLIC / "source_manifest.json").read_bytes()),
         "Prefix archive changed its V12 parent proof")
    need({entry["original_file"] for entry in manifest["files"]} == set(EXTRA_NAMES)
         and len(manifest["files"]) == 6, "Explicit prefix source allowlist differs")
    for entry in manifest["files"]:
        name = entry["original_file"]
        raw = (project / entry["published_file"]).read_bytes()
        need(entry["published_file"] == PUBLIC + name and entry["original_sha256"] == entry["published_sha256"]
             == digest(raw) == {**RUNTIME, **TOOLS}[name], "Published prefix source bytes changed")
        base.private_paths(raw)
        compile(raw.decode("utf8"), name, "exec")
    parent = read(project / PUBLIC / "source_manifest.json")
    lookup = {entry["original_file"]: entry for entry in parent["files"] if entry["role"] == "collection_runtime"}
    need(len(manifest["shared_source_bindings"]) == 25 and set(selected["source_sha256"])
         == {entry["original_file"] for entry in manifest["shared_source_bindings"]} | set(RUNTIME),
         "Prefix shared-source mapping changed")
    for entry in manifest["shared_source_bindings"]:
        need(lookup[entry["original_file"]] == entry and selected["source_sha256"][entry["original_file"]]
             == entry["original_sha256"] and digest((project / entry["published_file"]).read_bytes())
             == entry["published_sha256"], "Prefix shared dependency changed")
    rows = selected["cases"]
    conditions = {(25., -.5), (25., -1.), (35., 0.), (35., .5), (40., 1.), (40., 1.5)}
    need(len(rows) == 12 and len({row["episode_id"] for row in rows}) == 12
         and {(row["target_ml"], row["control_target_offset_ml"], row["seed"]) for row in rows}
         == {(target, offset, seed) for target, offset in conditions for seed in (193101, 193102)},
         "Twelve paired developmental prefix conditions changed")
    for row in rows:
        target, dose, gate = row["target_ml"], row["received_ml"], row["strict_assessment"]
        need(row["internal_control_target_ml"] == target + row["control_target_offset_ml"]
             and gate["dose_passed"] is (target - .001 * target <= dose <= target + .001 * target)
             and gate["tolerance_ml"] == .001 * target and abs(row["signed_error_ml"] - (dose - target)) < 1e-9,
             "Internal actuator offset changed the actual goal/tolerance")
        need(row["split"] == "development" and row["physical_action_seconds"] <= 24.
             and row["wall_seconds_including_post"] > 0 and gate["valid_evidence"] is True
             and gate["full_home_within_deadline"] is True and gate["no_late_dose_change"] is True
             and gate["maximum_all_label_spilled_particles"] <= 1,
             "Prefix measurement/home/spill/time evidence changed")
    need(selected["strict_task_passes"] == sum(row["strict_assessment"]["feasible"] for row in rows) == 0
         and selected["prefix_only_no_fine_dosing"] is True and selected["new_goal_policy_trained"] is False
         and selected["default_gui_deployment"] is False and selected["all_continuous_goals_guaranteed"] is False
         and manifest["new_model"] is False and manifest["default_gui_deployment"] is False,
         "Prefix calibration was promoted to a precision model/default")
    need(selected["fixed_coarse_parameters"] == {"prepare_fraction": 1., "dose_tilt_deg_s": 25.6,
        "return_fraction": 1., "tail_lead_s": .26, "stop_bias_ml": 10., "max_dose_u_accel": 6.},
        "Prefix calibration changed speeds or another original parameter")
    base.private_paths((project / REPORT).read_bytes())
    base.private_paths((project / MANIFEST).read_bytes())
