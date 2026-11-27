"""Curate selected journal records from unversioned development trials.

CPU-only: read small summary/evaluation/parameter JSON and committed journal
rows. Never decode particle/step traces or run a live controller; only three
orphaned gzip byte streams are hashed. Historical claims stay historical;
this tool does not select a policy or run a new acceptance test.
"""

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import re
import tempfile


ROOT = Path(__file__).resolve().parents[2]
RUN_HOME = ROOT / "droplet_out/pour_policy_timed_correction"
REPO = ROOT / "projects/isaac-sim-fluid-lab"
REPORT = REPO / "results/development_pilots_20261123_24.json"
DOCUMENT = REPO / "docs/DEVELOPMENT_PILOTS_20261124.md"
KIND = "curated_unfrozen_timed_correction_development_archive"
LOCAL = timezone(timedelta(hours=8), "Asia/Macau")
EXCLUDED_RUN = "run_20261123"
RUN_IDS = (
    "run_20261123_bump", "run_20261123_bump2", "run_20261123_early",
    "run_20261123_fill26", "run_20261123_fill26b", "run_20261123_fill26c",
    "run_20261123_fill26d", "run_20261123_lead_probe", "run_20261123_percent",
    "run_20261123_reserve", "run_20261123_scale", "run_20261123_scale_b",
    "run_20261123_scale_pulse", "run_20261123_scale_pulse_b", "run_20261123_scale_pulse_c",
    "run_20261123_span", "run_20261123_span_b", "run_20261123_tilt22",
    "run_20261123_tilt22b", "run_20261124_fine_particle", "run_20261124_fine_trim",
)
RESULT_FIELDS = (
    "target_ml", "received_ml", "error_ml", "absolute_error_ml", "tolerance_ml",
    "success", "valid_physics", "complete", "termination_reason", "seed", "randomized",
    "policy_name", "policy_params", "spilled_ml", "max_spilled_ml", "inflight_ml",
    "remaining_ml", "final_tilt_degrees", "upright", "physical_time_s", "physical_steps",
    "physics_dt_s", "measurement_physical_time_s", "measurement_basis", "wall_seconds",
    "particle_count", "source_particle_count", "particle_volume_ml", "initial_source_ml",
    "actual_initial_source_mass_kg", "actual_initial_total_mass_kg", "actual_final_total_mass_kg",
    "maximum_mass_drift_kg", "mass_conserved", "particle_mass_array_unchanged",
    "mass_volume_units_consistent", "all_states_finite", "source_partition_conserved",
    "max_below_table_particles", "minimum_substeps", "maximum_substeps",
    "fluid_mode", "rendered_training", "volume_measurement", "contact_force_grasp_simulated",
)
STATUS_FIELDS = (
    "phase", "direct_phase", "done", "home_static", "stable_frames", "timed_out",
    "completed_by_deadline", "completed_physical_time_s", "source_airborne_particles",
    "correction_count", "correction_enabled", "deadline_s", "mode_event", "sensor",
    "inflight_used_for_control",
)
ORPHANED_ATTEMPTS = (
    ("run_20261123_early", "pilot1_candidate_lead0.62_t30_s117101.steps.jsonl.gz", 456, 7.6,
     "cf2c4fe77381181b4b49b62e4e885eb6a3451afce8130c6e9adad6ace79be25f"),
    ("run_20261123_fill26c", "pilot_candidate_lead0.26_t30_s127101.steps.jsonl.gz", 437, 437/60,
     "602f86b9e3c119736787936dad528fa630b91752394ff862623764c6e597fded"),
    ("run_20261123_scale", "pilot_candidate_d0.72_t40_s145101.steps.jsonl.gz", 128, 128/60,
     "5e7bc79672df99818b82d98e3ff064ba3de390d5f356d791ca302f02dccc4ac9"),
)
SETTING_FIELDS = (
    "spec", "free_params", "lead_s", "enable_topup", "dose_duration_s", "spacing_m",
    "particle_ml", "limit_ml", "within_band", "corrections", "topups", "sensor",
)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def file_sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finite(value):
    json.dumps(value, allow_nan=False)


def timestamp(value):
    result = datetime.fromisoformat(value)
    require(result.tzinfo is not None, "A recorded wall timestamp has no timezone.")
    return result


def times(values):
    values = [timestamp(value) for value in values if value]
    require(values, "No original case wall timestamps are available.")
    return {"first_recorded_at_utc": min(values).astimezone(timezone.utc).isoformat(),
            "last_recorded_at_utc": max(values).astimezone(timezone.utc).isoformat(),
            "first_recorded_at_local": min(values).astimezone(LOCAL).isoformat(),
            "last_recorded_at_local": max(values).astimezone(LOCAL).isoformat(),
            "timezone": "Asia/Macau (UTC+08:00)"}


def pick(value, fields):
    return {key: value[key] for key in fields if key in value}


def selected_case(row, ordinal):
    result = row["result"]
    finite(row)
    require(result["target_ml"] == row["target_ml"] and result["seed"] == row["seed"],
            "Journal condition/footer mismatch.")
    require(abs(result["received_ml"] - result["target_ml"] - result["error_ml"]) < 1e-8,
            "Journal signed dose error differs from its original footer.")
    status = pick(row.get("controller", {}), STATUS_FIELDS)
    error = result.get("error")
    # Original exception text is retained only if it passes the final public
    # privacy scan; raw journals and controller trajectories stay local.
    return {
        "journal_ordinal": ordinal, "case_id": row["episode_id"],
        "stage_recorded": row.get("stage"), "stage": row.get("stage", "unlabelled_probe"),
        "role_recorded": row.get("role"), "role": row.get("role", "unspecified"),
        "target_ml": row["target_ml"], "seed": row["seed"],
        "signed_error_percent_from_footer": result["error_ml"] / row["target_ml"] * 100,
        "absolute_error_percent_from_footer": abs(result["error_ml"]) / row["target_ml"] * 100,
        "recorded_at_utc": row["recorded_at_utc"],
        "recorded_at_local": timestamp(row["recorded_at_utc"]).astimezone(LOCAL).isoformat(),
        "settings_recorded": pick(row, SETTING_FIELDS),
        "result": pick(result, RESULT_FIELDS),
        "error_present": error is not None,
        "error_message_recorded": error,
        "error_text_sha256": sha(str(error).encode("utf-8")) if error is not None else None,
        "terminal_status_recorded": status,
        "full_home_recorded": row.get("full_home"),
        "end_spilled_particles_recorded": row.get("spill_particles"),
        "post_observation_recorded": row.get("post_observation"),
        "case_wall_seconds_recorded": row.get("wall_seconds"),
        "evidence_use": "exploratory_seen_not_fresh_holdout",
        "original_footer_success_is_not_a_new_release_decision": True,
    }


def mean(values):
    return sum(values) / len(values) if values else None


def statistics(cases):
    require(cases, "Cannot summarize an empty case group.")
    results = [case["result"] for case in cases]
    errors = [result["error_ml"] for result in results]
    maximum_spill = [result.get("max_spilled_ml") for result in results]
    return {
        "case_count": len(cases), "case_ids": [case["case_id"] for case in cases],
        "signed_errors_ml": errors, "mean_signed_error_ml": mean(errors),
        "mae_ml": mean([abs(error) for error in errors]),
        "worst_absolute_error_ml": max(abs(error) for error in errors),
        "within_0_2_ml_descriptive_count": sum(abs(error) <= .2 for error in errors),
        "mean_action_physical_seconds": mean([result["physical_time_s"] for result in results]),
        "mean_action_wall_seconds": mean([result["wall_seconds"] for result in results]),
        "mean_case_wall_seconds_recorded": mean([case["case_wall_seconds_recorded"]
            for case in cases if case["case_wall_seconds_recorded"] is not None]),
        "case_footer_complete_count": sum(result["complete"] is True for result in results),
        "case_footer_valid_physics_count": sum(result["valid_physics"] is True for result in results),
        "case_footer_mass_conserved_count": sum(result["mass_conserved"] is True for result in results),
        "full_home_true_recorded_count": sum(case["full_home_recorded"] is True for case in cases),
        "full_home_missing_recorded_count": sum(case["full_home_recorded"] is None for case in cases),
        "case_with_any_maximum_spill_count": sum(value > 0 for value in maximum_spill if value is not None),
        "maximum_spill_missing_count": sum(value is None for value in maximum_spill),
        "maximum_spilled_ml": max((value for value in maximum_spill if value is not None), default=None),
        "end_spill_positive_recorded_count": sum((case["end_spilled_particles_recorded"] or 0) > 0 for case in cases),
        "summary_is_descriptive_not_a_new_scoring_or_acceptance_rule": True,
    }


def groups(cases, fields):
    grouped = defaultdict(list)
    for case in cases:
        grouped[tuple(case[field] for field in fields)].append(case)
    return [{**dict(zip(fields, key)), "metrics": statistics(rows)}
            for key, rows in sorted(grouped.items())]


def particle_bases(cases):
    fields = ("particle_count", "source_particle_count", "particle_volume_ml", "initial_source_ml")
    values = {tuple(case["result"][key] for key in fields) for case in cases}
    return [{**dict(zip(fields, value)),
             "initial_beaker_ml_from_counts": (value[0]-value[1])*value[2],
             "initial_total_ml_from_counts": value[0]*value[2]} for value in sorted(values)]


def build():
    captured = {}

    def read(path):
        before = path.read_bytes()
        captured[path] = sha(before)
        return before

    def document(path):
        return json.loads(read(path))

    existing = {path.name for path in RUN_HOME.iterdir() if path.is_dir()}
    require(existing == {*RUN_IDS, EXCLUDED_RUN}, "Run inventory changed; review scope before extending this archive.")
    previous = document(REPO / "results/correction_12s_20261123.json")
    require(previous["counts"]["total"] == 42, "The already published 42-case study changed.")
    records, all_cases, model_count = [], [], 0
    for run_id in RUN_IDS:
        run = RUN_HOME / run_id
        raw = read(run / "episodes.jsonl")
        rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
        require(rows and len({row["episode_id"] for row in rows}) == len(rows), "Duplicate/missing committed case IDs.")
        cases = [selected_case(row, ordinal) for ordinal, row in enumerate(rows, 1)]
        summaries, artifacts = {}, {"episodes.jsonl": sha(raw)}
        for filename in ("summary.json", "eval_summary.json", "pilot_decision.json"):
            path = run / filename
            if path.is_file():
                summaries[filename] = document(path)
                artifacts[filename] = captured[path]
        models = []
        for filename in ("candidate_frozen.json", "recommended_frozen.json"):
            path = run / filename
            if path.is_file():
                model = document(path)
                artifacts[filename] = captured[path]
                models.append({"original_basename": filename, "sha256": captured[path],
                    "policy_kind_recorded": model.get("policy_kind"),
                    "role_recorded": model.get("role", model.get("metadata", {}).get("role")),
                    "parameters_recorded": model.get("parameters"),
                    "verified_source_bound_frozen_model": False,
                    "limitation": "Filename says frozen, but no run source/input binding or model freeze-time record verifies it."})
        model_count += len(models)
        completion = summaries.get("summary.json", {}).get("complete")
        evaluation_seen = any(case["stage"] in ("validation", "test") for case in cases)
        status = ("unfrozen_exploratory_evaluation_already_seen" if evaluation_seen else
                  "no_completion_summary_partial_committed_records" if "summary.json" not in summaries else
                  "historical_pilot_gate_not_passed" if completion is False else "development_probe_record_only")
        claims = [summary.get("recorded_at_utc") for summary in summaries.values()]
        records.append({
            "run_id": run_id, "dates": times([case["recorded_at_utc"] for case in cases] + claims),
            "archive_status": status, "historical_summary_complete": completion,
            "complete_false_is_not_evidence_of_a_currently_running_process": True,
            "case_count": len(cases), "stage_counts": dict(sorted(Counter(case["stage"] for case in cases).items())),
            "artifacts_sha256": artifacts,
            "source_snapshot_present": (run/"source_snapshot").is_dir(),
            "protocol_present": (run/"protocol.json").is_file(),
            "input_freeze_present": (run/"input_freeze.json").is_file(),
            "freeze_record_present": (run/"freeze.json").is_file(),
            "independently_audited_experiment": False,
            "verified_source_bound_frozen_model": False, "parameter_files": models,
            "recorded_summaries": summaries, "particle_bases_recorded": particle_bases(cases),
            "legacy_field_semantics": {"recorded_summaries.*.pilot.signed_percent":
                "The historical field is misnamed: recorded values are absolute percent, not signed percent. "
                "They are retained unchanged. Case signed_error_percent_from_footer is separately recomputed from signed error.",
                "cases.result.max_spilled_ml": "Legacy maximum SOURCE-label spill, not a maximum over all particle labels.",
                "cases.end_spilled_particles_recorded": "Final all-label spilled particle count, not a whole-episode maximum.",
                "statistics.*maximum*spill*": "Derived only from legacy maximum SOURCE-label spill; no all-label maximum is established."},
            "historical_rule_note": ("max(one 0.042875 mL particle, 0.1 percent of target); not a strict 0.1 percent rule."
                if run_id == "run_20261123_percent" else
                "Use the individually recorded rule/limit; no global rule is retroactively applied."),
            "statistics_all_cases_descriptive": statistics(cases),
            "per_target_descriptive": groups(cases, ("target_ml",)),
            "per_stage_role_target": groups(cases, ("stage", "role", "target_ml")),
            "cases": cases,
            "raw_step_traces_decoded_by_curator": False,
        })
        all_cases.extend(cases)
    require(len(records) == 21 and len(all_cases) == 192, "Expected all 21 runs and all 192 committed journal cases.")
    require(all(not run[key] for run in records for key in
        ("source_snapshot_present", "protocol_present", "input_freeze_present", "freeze_record_present")),
        "New provenance artifacts appeared; this unversioned archive needs review.")
    orphans = []
    for run_id, basename, step, physical_time, expected_sha in ORPHANED_ATTEMPTS:
        path = RUN_HOME / run_id / "traces" / basename
        captured[path] = file_sha(path)
        require(captured[path] == expected_sha, "An inventoried orphaned trace changed.")
        orphans.append({"run_id": run_id, "attempt_id": basename.removesuffix(".steps.jsonl.gz"),
            "trace_basename": basename, "trace_sha256": captured[path],
            "last_logged_physics_step": step, "last_logged_physical_time_s": physical_time,
            "result_footer_present": False, "committed_journal_row_present": False,
            "last_physics_step_has_matching_direct_evidence": False,
            "trace_decode_status_from_readonly_inventory": "EOFError_truncated_gzip",
            "result": None, "acceptance": "unknown_no_complete_terminal_evidence",
            "provenance": "separate read-only trace inventory; not an independent physical audit",
            "curator_reads_payload": False, "curator_only_hashes_raw_trace_bytes": True})
    for path, expected in captured.items():
        require(file_sha(path) == expected, "An input changed while the archive was being assembled.")
    report = {
        "schema_version": 1, "kind": KIND, "archive_complete": True,
        "experiment_release_accepted": False, "automatic_gui_deployment": False,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "archive_tool": {"basename": Path(__file__).name, "sha256": sha(Path(__file__).read_bytes())},
        "scope": {"recorded_dates": times([case["recorded_at_utc"] for case in all_cases]),
            "excluded_already_published_run": EXCLUDED_RUN,
            "excluded_public_report": "results/correction_12s_20261123.json",
            "excluded_public_report_sha256": captured[REPO/"results/correction_12s_20261123.json"],
            "excluded_previously_published_cases": 42,
            "unit_of_record": "one committed journal row; not necessarily one independent condition",
            "evaluation_status": "all cases exploratory and already seen; no fresh holdout claim",
            "raw_step_traces_decoded_by_curator": False, "orphan_trace_bytes_hashed_only": 3,
            "readonly_inventory_action_trace_attempts": 195,
            "readonly_inventory_auxiliary_post_traces": 55,
            "new_simulation_run": False,
            "historical_controller_source_binding_available": False},
        "counts": {"runs": len(records), "total_attempts": len(all_cases)+len(orphans),
            "committed_case_records": len(all_cases), "orphaned_partial_attempts": len(orphans),
            "unique_target_seed_pairs": len({(case["target_ml"], case["seed"]) for case in all_cases}),
            "unique_target_seed_pair_count_scope": "committed journal cases only; not independent heldout episodes",
            "stage_counts": dict(sorted(Counter(case["stage"] for case in all_cases).items())),
            "case_footer_complete_count": sum(case["result"]["complete"] for case in all_cases),
            "case_footer_valid_physics_count": sum(case["result"]["valid_physics"] for case in all_cases),
            "legacy_case_footer_success_count": sum(case["result"]["success"] for case in all_cases),
            "source_label_maximum_spill_positive_cases": sum(case["result"]["max_spilled_ml"] > 0 for case in all_cases),
            "all_label_final_spill_positive_cases": sum((case["end_spilled_particles_recorded"] or 0) > 0 for case in all_cases),
            "case_termination_counts": dict(sorted(Counter(case["result"]["termination_reason"] for case in all_cases).items())),
            "parameter_files_named_frozen": model_count, "verified_source_bound_frozen_models": 0},
        "metadata_conflicts": [{"record_type": "effective_direct_sources",
            "field": "deadline_physical_seconds", "inherited_value": 5.0,
            "affected_action_trace_headers_from_readonly_inventory": 195,
            "actual_policy_deadlines_recorded_s": [12.0, 24.0, 36.0],
            "meaning": "Inherited collector metadata conflicts with actual policy parameters; it is not evidence of five-second completion.",
            "provenance": "separate read-only trace inventory, not a new physics audit"}],
        "limitations": [
            "This is a development archive, not a new release or an independently audited experiment.",
            "All 192 committed rows and 3 orphaned truncated attempts are retained; no terminal result is invented for orphaned attempts.",
            "No historical source/input snapshot binds these cases to the present controller; publishing present source cannot validate past runs.",
            "Summary complete/role/pass labels are retained as historical statements, not adopted as new acceptance or policy selection.",
            "The reserve run has two small parameter files named frozen, without verifiable source binding/freeze time; these are not a deployed model.",
            "fill26d validation/test ran after a failed pilot and without a frozen model; they are exploratory already-seen evidence, never a fresh test set.",
            "The 0.2 mL count is descriptive only; historical rules, spill allowances and 12/24/36-second deadlines differ and remain separately recorded.",
            "The fine-particle trials use 0.019683 mL particles and 56.68704 mL initial source, versus 0.042875 mL and 51.36425 mL previously; accuracy comparisons are confounded.",
            "Particle-count mass is privileged simulation truth, not an implemented physical scale or camera sensor; the arm path remains kinematic.",
            "Wall time depends on machine load; full-home flags and physical validity here are recorded claims, not a new raw-trace audit.",
            "Legacy summary signed_percent arrays actually contain absolute percentages; original values are preserved with an explicit semantics note.",
            "The 43 legacy footer success flags use an old 0.1 mL condition, not the later 0.1 percent experiment rule or a new release decision.",
            "The percent trial uses max(one 0.042875 mL particle, 0.1 percent of target), unlike the strict 0.1 percent fine-particle trials; its rule is not pooled with them.",
            "Legacy result.max_spilled_ml covers source-label liquid only; final spill_particles covers all labels. No whole-episode all-label maximum spill has been independently established.",
        ],
        "orphaned_attempts": orphans, "runs": records,
    }
    finite(report)
    public = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    require(not re.search(r"[A-Za-z]:[\\/]|(?:file|plugin)://|127\.0\.0\.1|AppData|mathlab", public),
            "Private absolute path or local connection data appeared in the selected archive.")
    require(len(public.encode("utf-8")) < 1_800_000, "Archive unexpectedly large; review selection.")
    return report, public


def markdown(report):
    lines = ["# 2026-11-23–24 倒水开发试点归档", "",
        "补录21组、192条已提交案例及3条孤立截断尝试（合计195次动作尝试），排除此前公开的V8主轮42例；146组不同目标/seed组合不等于192个独立测试。这里没有新验收通过的模型，也没有更新GUI默认。", "",
        "所有失败与缺少完成摘要的记录均保留。`complete=false`表示旧试点未过门槛或记录不完整，不能据此判断程序仍在运行。没有历史源码/输入快照、可核验的模型冻结或整轮独立审计，当前控制器不能替旧试点补来源证明。", "",
        "| 原试点 | 已提交案例 | MAE / 最坏误差（mL） | 物理均时（s） | 记录状态 |", "| --- | ---: | ---: | ---: | --- |"]
    for run in report["runs"]:
        metric = run["statistics_all_cases_descriptive"]
        state = ("探索验证/测试，已见" if "evaluation" in run["archive_status"] else
                 "无完成摘要，部分记录" if "partial" in run["archive_status"] else "试点未过门槛/探测记录")
        lines.append(f"| `{run['run_id']}` | {run['case_count']} | {metric['mae_ml']:.4f} / {metric['worst_absolute_error_ml']:.4f} | {metric['mean_action_physical_seconds']:.3f} | {state} |")
    lines += ["", "表中跨阶段均值只是日志描述，不能混合候选/基准或不同初态进行选优；逐目标、阶段、角色、有符号误差与墙钟耗时见[精选JSON](../results/development_pilots_20261123_24.json)。", "",
        "- `fill26d`在试点未过后另做的验证/测试没有冻结模型：测试9/9剂量落在±0.2mL，但有洒水，验证也有误差/回家失败；只作为已见探索数据，不能再次当新留出集。",
        "- `reserve`虽有两份名为`*_frozen.json`的参数文件，但缺来源绑定和可验证冻结时间；JSON如实保存其参数及SHA，不发布为默认权重。",
        "- `early`、`fill26c`、`scale`各有一份未写入journal的截断尝试，最后记录分别到7.600、7.283、2.133物理秒；没有result，不能当未发生或补造结果。归档保留其原字节SHA和末帧元数据，不发布轨迹内容。",
        "- 后期改用12/24/36秒、0.1%指标及最多1粒洒水等不同门槛，不能冒称原5秒、零洒水要求通过。",
        "- `fine_particle`与`fine_trim`粒距改为2.7mm，源水变为56.68704mL，较原51.36425mL多约10.36%，杯内初态也由177.1595变为172.127835mL；不能只比较误差就声称控制改进。前者pilot MAE0.04627mL，剂量0.1%指标3/6通过，但联合≤1粒洒水/回家条件仅2/6；后者MAE0.28696mL，仅2/6通过。",
        "- 旧summary的`signed_percent`实际是绝对百分比，原值保留并加语义说明；真正有符号百分比另由原始有符号误差计算。43个旧footer的`success=true`也不等于后期0.1%试验通过。",
        "- `percent`试点按`max(一粒0.042875mL, 目标的0.1%)`计算容差，不是严格0.1%；3.5mm粒距的20mL最近误差0.02025mL已大于严格0.020mL，不能和细粒试点混算通过率。",
        "- 独立只读清点发现195份动作轨迹的`effective_direct_sources.deadline_physical_seconds`继承写为5秒，与实际policy的12/24/36秒冲突，不能据该字段宣称5秒完成。清点仅核对记录，不是GPU数组/关节运动的独立物理审计。",
        "- 旧`result.max_spilled_ml`只是源标签水的最大洒水（13例大于零）；`spill_particles`是全部标签的末态洒水（15例大于零）。两者均原样保留，不能据此声称已核对全程全部粒子的最大洒水。",
        "- 最后记录为11月24日00:48（UTC+8）的`fine_trim`；本次仅整理历史，不运行新仿真、不读取逐步液体轨迹，不把后续观察当作动作按时完成。", ""]
    return "\n".join(lines)


def atomic(path, text):
    require(path in (REPORT, DOCUMENT), "Unexpected output target.")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="\n", dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(text)
        temporary.replace(path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="Write only the two new selected archive files.")
    args = parser.parse_args()
    report, public = build()
    if args.write:
        atomic(REPORT, public)
        atomic(DOCUMENT, markdown(report))
    print(json.dumps({"checked": True, "written": args.write, "counts": report["counts"],
        "json_bytes": len(public.encode("utf-8")), "raw_step_traces_decoded": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
