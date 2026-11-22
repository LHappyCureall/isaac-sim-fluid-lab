"""Independent 54-train / 18-validation / 18-test live pouring experiment.

Fresh seeds and a new protocol protect the quick-run history. Test episodes
always compare the frozen training-best candidate with the baseline. The
validation recommendation references one of these groups without rerunning
it. This process never deploys or writes a GUI checkpoint.
"""

import argparse
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path

from lab_paths import PROJECT_ROOT, require_isaac_sim_root
import platform
import sys
import time


ROOT = PROJECT_ROOT
SCRIPTS = ROOT / "assets" / "robot_testtube_pour" / "scripts"
SHARED_PATH = ROOT / "train_target_volume_policy.py"
_spec = importlib.util.spec_from_file_location("full_pour_shared", SHARED_PATH)
shared = importlib.util.module_from_spec(_spec)
exec(compile(SHARED_PATH.read_text(encoding="utf-8"), str(SHARED_PATH), "exec"), shared.__dict__)

TARGETS = (20, 30, 40)
TRAIN_SEEDS = (2011, 2012, 2013)
VALIDATION_SEEDS = (2101, 2102, 2103)
TEST_SEEDS = (31001, 31002, 31003)
CEM_SEED = 20261122
GENERATIONS, POPULATION, ELITES = 3, 6, 2
TOTAL_EPISODES = 90


def now():
    return datetime.now(timezone.utc).isoformat()


def source_snapshot():
    paths = {"train_target_volume_policy_full.py": ROOT / "train_target_volume_policy_full.py",
             "train_target_volume_policy.py": SHARED_PATH}
    for name in ("geometry.py", "liquid.py", "target_control.py", "learned_policy.py",
                 "pour_training_env.py", "pbf_graph.py"):
        paths[f"assets/robot_testtube_pour/scripts/{name}"] = SCRIPTS / name
    return {name: shared._sha(path) for name, path in paths.items()}


def require_sources(expected):
    actual = source_snapshot()
    if actual != expected:
        changed = [name for name in expected if expected[name] != actual.get(name)]
        raise RuntimeError("Frozen experiment source changed: " + ", ".join(changed))


def write_report(path, report):
    candidate = report["test"]["candidate_metrics"]
    baseline = report["test"]["baseline_metrics"]
    recommended = report["recommendation"]
    lines = ["# 完整预算实时倒水实验", "",
        f"完成时间（UTC）：{report['completed_at_utc']}", "",
        "54 个训练、18 个验证、18 个测试回合，共 90 次实时液体运行。"
        "CEM 为 3 代 × 6 候选 × 3 目标，学习三个停止/扶正反馈参数，不是 PPO 或神经网络关节策略。", "",
        "训练初始种子：2011/2012/2013；验证：2101/2102/2103；测试：31001/31002/31003。"
        "测试覆盖 9 个目标×种子条件，候选/基准各 9 次执行。", "",
        f"验证选择：`{recommended['selection_origin']}`，对应测试组 `{recommended['test_result_role']}`。"
        "选择在测试前冻结，测试不用于拟合或更换推荐。GUI 默认与历史权重未修改。", "",
        "| 独立测试 | MAE (mL) | 最大绝对误差 (mL) | 安全且 ±0.2 mL | 安全且 ±0.5 mL |",
        "| --- | ---: | ---: | ---: | ---: |"]
    for label, metrics in (("冻结训练候选", candidate), ("原反馈基准", baseline)):
        lines.append(f"| {label} | {metrics['mae_ml']:.6f} | {metrics['worst_absolute_error_ml']:.6f} | "
                     f"{metrics['within_0_2_ml_rate']:.1%} | {metrics['within_0_5_ml_rate']:.1%} |")
    lines += ["", "| 目标 (mL) | 种子 | 候选入杯 (mL) | 候选误差 (mL) | 基准入杯 (mL) | 基准误差 (mL) |",
              "| ---: | ---: | ---: | ---: | ---: | ---: |"]
    for pair in report["test"]["paired_results"]:
        a, b = pair["candidate"], pair["baseline"]
        lines.append(f"| {pair['target_ml']:g} | {pair['seed']} | {a['received_ml']:.6f} | {a['error_ml']:+.6f} | "
                     f"{b['received_ml']:.6f} | {b['error_ml']:+.6f} |")
    lines += ["", f"训练物理帧：{report['training']['physical_steps']:,}；"
        f"训练物理时间：{report['training']['physical_seconds']:.2f} s。"
        f"候选/基准质量守恒：{candidate['all_mass_conserved']} / {baseline['all_mass_conserved']}。", "",
        "mL 是固定粒子质量/密度给出的仿真真值，粒子量化为 0.042875 mL；尚无视觉估水或真机称重校准。"
        "所有动作通过原实时 PBF、1/60 s 物理帧、自适应子步与六次压力迭代执行。"
        "CUDA graph 只在已验证的无渲染批量环境可选，展示视频时间不参与训练。", "",
        "本轮只含三个目标、窄幅位置/黏性随机化和三个新测试种子；不代表杯型/液体/真实机器人泛化。"
        "机器人采用运动学预握和受限路径，不模拟摩擦抓持或流体力矩反馈。", "",
        "`protocol.json` 保存八个源码 SHA、预算和种子；`episodes.jsonl` 与 `traces/*.steps.jsonl.gz` "
        "保留每次真实执行及原始物理步。`candidate_frozen.json` 与 `recommended_frozen.json` "
        "均在测试前冻结，`freeze.json` 保存各自 SHA。推荐结果引用已有候选/基准测试组，没有额外测试执行。", ""]
    shared._atomic_text(path, "\n".join(lines))


def run_full(args, versions):
    import numpy as np
    from learned_policy import CEMSearch, DEFAULT_PARAMS, LearnedPourPolicy, PARAMETER_BOUNDS
    from pour_training_env import PourTrainingEnv

    output = Path(args.output).resolve()
    if output == ROOT or output.is_relative_to(ROOT / "assets"):
        raise ValueError("Use a dedicated run output directory outside GUI assets.")
    output.mkdir(parents=True, exist_ok=True)
    frozen_sources = source_snapshot()
    original_seven = {name: value for name, value in frozen_sources.items()
                      if name != "train_target_volume_policy_full.py"}
    sdk = require_isaac_sim_root()
    version_file = sdk / "VERSION"
    software = {**versions, "python": sys.version, "platform": platform.platform(),
                "isaac_sim": version_file.read_text(encoding="utf-8").strip() if version_file.is_file() else "unavailable"}
    protocol = {"schema_version": 1, "kind": "independent_full_budget_live_pouring",
        "algorithm": "CEM_3_parameter_stopping_policy", "targets_ml": list(TARGETS),
        "generations": GENERATIONS, "population": POPULATION, "elite_count": ELITES,
        "training_seed": CEM_SEED, "training_episodes": 54, "validation_episodes": 18,
        "test_episodes": 18, "total_live_episodes": TOTAL_EPISODES,
        "train_common_seed_by_target": dict(zip(map(str, TARGETS), TRAIN_SEEDS)),
        "validation_seeds": list(VALIDATION_SEEDS), "test_seeds": list(TEST_SEEDS),
        "distinct_validation_target_seed_conditions": 9, "distinct_test_target_seed_conditions": 9,
        "validation_policy_roles": ["candidate", "baseline"], "test_policy_roles": ["candidate", "baseline"],
        "recommendation_test_results": "references validation-selected role; no additional rollouts",
        "parameter_bounds": {name: list(bounds) for name, bounds in PARAMETER_BOUNDS.items()},
        "baseline_parameters": dict(DEFAULT_PARAMS),
        "randomization": {"initial_jitter_max_spacing": .03, "viscosity_factor": [.9, 1.1]},
        "physics": {"mode": "live_PBF_action_conditioned", "dt_s": 1 / 60, "spacing_m": .0035,
            "motion_time_scale": 2.0, "minimum_substeps": 4, "pressure_iterations": 6,
            "max_seconds": 35.0, "cuda_graph": bool(args.graph)},
        "score": "abs(error_ml)+10*max_spilled_ml+1000*invalid_physics+100*incomplete_or_not_upright+0.0005*physical_seconds",
        "selection": "lower validation mean score; ties or unchanged parameters recommend baseline; test never selects",
        "source_sha256": frozen_sources, "original_seven_source_sha256": original_seven,
        "software": software, "gui_deployment_written": False}
    assert set(TRAIN_SEEDS).isdisjoint(VALIDATION_SEEDS + TEST_SEEDS)
    assert set(VALIDATION_SEEDS).isdisjoint(TEST_SEEDS)
    protocol_id = shared._signature(protocol)
    protocol_path = output / "protocol.json"
    if protocol_path.exists():
        if not args.resume:
            raise FileExistsError("Configured run already exists; use --resume or a new output directory.")
        previous = json.loads(protocol_path.read_text(encoding="utf-8"))
        if previous.get("protocol_id") != protocol_id or previous.get("protocol") != protocol:
            raise ValueError("Resume rejected: source, seeds, versions, graph or protocol changed.")
    else:
        if any((output / name).exists() for name in ("episodes.jsonl", "summary.json", "freeze.json")):
            raise FileExistsError("Unconfigured journal/results exist; do not mix experiments.")
        # PLAN.md, job_state.json, source archives and supervisor logs may exist.
        shared.write_json(protocol_path, {"created_at_utc": now(), "protocol_id": protocol_id, "protocol": protocol})
    require_sources(frozen_sources)
    journal = output / "episodes.jsonl"
    records = shared._load_journal(journal, protocol_id) if args.resume else {}
    environment = PourTrainingEnv()
    if args.graph:
        shared._enable_graph(environment)
    search = CEMSearch(seed=CEM_SEED, population=POPULATION, elite_count=ELITES)
    started, newly_run = time.perf_counter(), 0
    progress_base = {"protocol_id": protocol_id, "total_planned_episodes": TOTAL_EPISODES,
                     "started_or_resumed_at_utc": now()}

    def reward_csv():
        rows = []
        for record in records.values():
            result = record["result"]
            rows.append({"episode_id": record["episode_id"], **record["conditions"],
                "score": record["score"], "reward": -record["score"],
                **{name: result[name] for name in ("received_ml", "error_ml", "physical_steps", "physical_time_s", "wall_seconds")},
                "record_path": record["record_path"]})
        fields = ("episode_id", "stage", "policy_role", "generation", "candidate", "target_ml", "seed",
                  "score", "reward", "received_ml", "error_ml", "physical_steps", "physical_time_s", "wall_seconds", "record_path")
        shared._write_csv(output / "rewards.csv", rows, fields)

    def episode(episode_id, stage, role, target, seed, params, generation=None, candidate=None):
        nonlocal newly_run
        conditions = {"stage": stage, "policy_role": role, "target_ml": float(target), "seed": int(seed),
            "randomized": True, "policy_params": None if params is None else dict(params),
            "generation": generation, "candidate": candidate}
        if episode_id in records:
            record = records[episode_id]
            if record["conditions"] != conditions or record["conditions_sha256"] != shared._signature(conditions):
                raise ValueError("Resumed policy/seed/episode conditions differ: " + episode_id)
            return record["result"]
        require_sources(frozen_sources)
        raw = output / "traces" / f"{episode_id}.steps.jsonl.gz"

        def progress(event):
            shared.write_json(output / "progress.json", {**progress_base, "state": "running",
                "updated_at_utc": now(), "episode_id": episode_id, "stage": stage,
                "policy_role": role, "conditions": conditions, "completed_episodes": len(records),
                "wall_seconds_this_invocation": time.perf_counter() - started, **event})

        print(f"FULL EPISODE {len(records) + 1}/{TOTAL_EPISODES}: {episode_id}", flush=True)
        policy = None if params is None else LearnedPourPolicy(params)
        result = environment.run_episode(target, seed, policy=policy, randomize=True,
                                         record_path=raw, progress=progress)
        if not raw.is_file() or float(result["target_ml"]) != float(target) or int(result["seed"]) != int(seed):
            raise RuntimeError("Missing trace or different live episode conditions: " + episode_id)
        record = {"episode_id": episode_id, "protocol_id": protocol_id, "conditions": conditions,
            "conditions_sha256": shared._signature(conditions), "record_path": raw.relative_to(output).as_posix(),
            "record_sha256": shared._sha(raw), "completed_at_utc": now(),
            "score": float(shared.score_episode(result)), "result": result}
        with journal.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, allow_nan=False, separators=(",", ":")) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        records[episode_id] = record
        newly_run += 1
        reward_csv()
        progress({"event": "episode_recorded", "received_ml": result["received_ml"],
                  "error_ml": result["error_ml"], "score": record["score"]})
        print(f"FULL RESULT {episode_id}: error={result['error_ml']:+.6f} mL, score={record['score']:.6f}", flush=True)
        return result

    try:
        training_results, curve = [], []
        for generation in range(GENERATIONS):
            candidates = search.ask()
            scores, generation_results = [], []
            for index, params in enumerate(candidates):
                values = [episode(f"train_g{generation:03d}_c{index:03d}_t{target}_s{seed}",
                    "train", "candidate", target, seed, params, generation, index)
                    for target, seed in zip(TARGETS, TRAIN_SEEDS)]
                scores.append(float(np.mean([shared.score_episode(result) for result in values])))
                generation_results.extend(values)
            generation_summary = search.tell(scores)
            training_results.extend(generation_results)
            shared.write_json(output / f"generation_{generation:03d}.json", {"protocol_id": protocol_id, **generation_summary})
            shared.write_json(output / "search_state.json", {"protocol_id": protocol_id, **search.state_dict()})
            curve.append({"generation": generation, "mean_score": float(np.mean(scores)),
                "generation_best_score": float(min(scores)), "best_training_score": search.best_score,
                "mean_reward": -float(np.mean(scores)), "mae_ml": shared.summarize(generation_results)["mae_ml"],
                "completed_training_episodes": len(training_results)})
            shared._write_csv(output / "training_curve.csv", curve, tuple(curve[0]))
            print("FULL GENERATION " + json.dumps(curve[-1]), flush=True)

        validation = {"candidate": [], "baseline": []}
        for seed in VALIDATION_SEEDS:
            for target in TARGETS:
                for role, params in (("candidate", search.best_params), ("baseline", None)):
                    validation[role].append(episode(f"validation_{role}_t{target}_s{seed}",
                                                   "validation", role, target, seed, params))
        scores = {role: float(np.mean([shared.score_episode(result) for result in values]))
                  for role, values in validation.items()}
        candidate_changed = search.best_params != DEFAULT_PARAMS
        recommended_role = "candidate" if candidate_changed and scores["candidate"] < scores["baseline"] else "baseline"
        selection = "cem_training_best" if recommended_role == "candidate" else "baseline_fallback"
        recommended_params = dict(search.best_params if recommended_role == "candidate" else DEFAULT_PARAMS)
        validation_report = {"protocol_id": protocol_id, "mean_score": scores,
            "metrics": {role: shared.summarize(values) for role, values in validation.items()},
            "results": validation, "recommended_role": recommended_role, "selection_origin": selection}
        shared.write_json(output / "validation.json", validation_report)
        require_sources(frozen_sources)

        candidate_path = output / "candidate_frozen.json"
        recommended_path = output / "recommended_frozen.json"
        freeze_path = output / "freeze.json"
        if freeze_path.exists():
            freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
            if (not args.resume or freeze.get("protocol_id") != protocol_id
                    or freeze["candidate_sha256"] != shared._sha(candidate_path)
                    or freeze["recommended_sha256"] != shared._sha(recommended_path)):
                raise ValueError("Frozen full-run models differ from their intact source/protocol.")
            if (LearnedPourPolicy.load(candidate_path).params != search.best_params
                    or LearnedPourPolicy.load(recommended_path).params != recommended_params
                    or freeze["recommended_role"] != recommended_role):
                raise ValueError("Frozen model parameters/recommendation differ from validation.")
        else:
            if any(record["conditions"]["stage"] == "test" for record in records.values()):
                raise ValueError("Cannot reuse test episodes without the pre-test model freeze.")
            if candidate_path.exists() or recommended_path.exists():
                raise ValueError("Partial model freeze; preserve files and review before resuming.")
            frozen_at = now()
            metadata = {"frozen_at_utc": frozen_at, "protocol_id": protocol_id,
                "training_episodes": 54, "training_seed": CEM_SEED, "validation_mean_score": scores,
                "source_sha256": frozen_sources, "test_used_for_selection": False, "gui_deployment_written": False}
            LearnedPourPolicy(search.best_params).save(candidate_path, metadata={**metadata,
                "policy_role": "candidate", "selection_origin": "full_CEM_training_best",
                "is_trained": bool(candidate_changed), "parameters_changed_from_baseline": bool(candidate_changed)})
            LearnedPourPolicy(recommended_params).save(recommended_path, metadata={**metadata,
                "policy_role": "recommended", "selection_origin": selection,
                "is_trained": recommended_role == "candidate", "test_result_role": recommended_role})
            freeze = {"protocol_id": protocol_id, "frozen_at_utc": frozen_at,
                "candidate_sha256": shared._sha(candidate_path), "recommended_sha256": shared._sha(recommended_path),
                "recommended_role": recommended_role, "selection_origin": selection,
                "gui_deployment_written": False}
            shared.write_json(freeze_path, freeze)
            shared._atomic_text(output / "models_frozen.sha256", freeze["candidate_sha256"] + "  candidate_frozen.json\n"
                               + freeze["recommended_sha256"] + "  recommended_frozen.json\n")

        # Both model files, their SHA and timestamp exist before any test step.
        pairs = []
        for seed in TEST_SEEDS:
            for target in TARGETS:
                candidate = episode(f"test_candidate_t{target}_s{seed}", "test", "candidate", target, seed, search.best_params)
                baseline = episode(f"test_baseline_t{target}_s{seed}", "test", "baseline", target, seed, None)
                pairs.append({"target_ml": target, "seed": seed, "candidate": candidate,
                              "baseline": baseline, "recommended_result_role": recommended_role})
        require_sources(frozen_sources)
        if (shared._sha(candidate_path) != freeze["candidate_sha256"]
                or shared._sha(recommended_path) != freeze["recommended_sha256"]):
            raise RuntimeError("A pre-test frozen model changed during independent evaluation.")
        if len(records) != TOTAL_EPISODES:
            raise RuntimeError("Unexpected number of recorded live rollouts.")
        all_results = [record["result"] for record in records.values()]
        candidate_metrics = shared.summarize([pair["candidate"] for pair in pairs])
        baseline_metrics = shared.summarize([pair["baseline"] for pair in pairs])
        summary = {"complete": True, "completed_at_utc": now(), "protocol_id": protocol_id, "protocol": protocol,
            "training": {**shared.summarize(training_results), "generations": GENERATIONS, "population": POPULATION,
                "best_score": search.best_score, "best_params": dict(search.best_params)},
            "validation": validation_report, "models": {
                "candidate": {"file": candidate_path.name, "sha256": freeze["candidate_sha256"], "parameters": dict(search.best_params)},
                "recommended": {"file": recommended_path.name, "sha256": freeze["recommended_sha256"], "parameters": recommended_params}},
            "recommendation": {"selection_origin": selection, "is_trained": recommended_role == "candidate",
                "test_result_role": recommended_role, "test_results_are_references": True, "additional_test_rollouts": 0},
            "test": {"paired_results": pairs, "candidate_metrics": candidate_metrics, "baseline_metrics": baseline_metrics,
                "recommended_metrics": candidate_metrics if recommended_role == "candidate" else baseline_metrics,
                "distinct_target_seed_conditions": 9, "executed_live_episodes": 18,
                "candidate_and_baseline_frozen_before_test": True},
            "all_rollouts": shared.summarize(all_results), "total_recorded_episodes": len(records),
            "new_episodes_this_invocation": newly_run, "wall_seconds_this_invocation": time.perf_counter() - started,
            "total_rollout_wall_seconds": sum(result["wall_seconds"] for result in all_results),
            "test_used_for_training_or_selection": False, "gui_deployment_written": False,
            "source_sha256_unchanged_after_run": source_snapshot() == frozen_sources,
            "original_seven_source_sha256_unchanged": True}
        # Verify again immediately before marking the experiment complete.
        require_sources(frozen_sources)
        shared.write_json(output / "summary.json", summary)
        write_report(output / "REPORT.md", summary)
        shared.write_json(output / "progress.json", {**progress_base, "state": "complete", "updated_at_utc": now(),
            "completed_episodes": len(records), "new_episodes_this_invocation": newly_run,
            "recommended_role": recommended_role, "gui_deployment_written": False})
        print("FULL COMPLETE " + json.dumps({"episodes": len(records), "recommended_role": recommended_role,
            "candidate_test_mae_ml": candidate_metrics["mae_ml"], "baseline_test_mae_ml": baseline_metrics["mae_ml"]}), flush=True)
        return summary
    except BaseException as error:
        shared.write_json(output / "progress.json", {**progress_base,
            "state": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed", "updated_at_utc": now(),
            "completed_episodes": len(records), "error_type": type(error).__name__, "error": str(error)})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "droplet_out/pour_policy_training/run_20261122_full")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--graph", action="store_true")
    args = parser.parse_args()
    before = source_snapshot()
    versions = shared.configure_sdk()
    require_sources(before)
    print("FULL ENVIRONMENT " + json.dumps(versions), flush=True)
    run_full(args, versions)


if __name__ == "__main__":
    main()
