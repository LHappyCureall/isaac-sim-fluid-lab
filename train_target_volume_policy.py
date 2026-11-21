"""Train an interpretable 20/30/40 mL policy on live, physical-time PBF.

Run with Isaac Sim's bundled Python. This batch process uses Newton/Warp
directly and creates no Kit app, viewport, robot visual or video playback.
The visible handoff is robot_target_volume_ai.py.
"""

import argparse
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path

from lab_paths import PROJECT_ROOT, require_isaac_sim_root
import platform
import subprocess
import sys
import time


ROOT = PROJECT_ROOT
SCRIPTS = ROOT / "assets" / "robot_testtube_pour" / "scripts"


def configure_sdk():
    sdk = require_isaac_sim_root()
    # NVRTC cannot use the desktop sandbox's redirected #!sandbox temp path.
    # These settings affect this training process only.
    cache = Path(os.environ.get("POUR_TRAIN_CACHE", str(ROOT / "_cache" / "pour_training")))
    (cache / "tmp").mkdir(parents=True, exist_ok=True)
    os.environ["TEMP"] = str(cache / "tmp")
    os.environ["TMP"] = str(cache / "tmp")
    candidates = list((sdk / "extscache").glob("omni.kit.pip_archive-*/pip_prebundle"))
    candidates += [sdk / "exts" / "omni.pip.compute" / "pip_prebundle"]
    candidates += list((sdk / "extscache").glob("omni.warp.core-*"))
    candidates += [sdk / "exts" / "isaacsim.pip.newton" / "pip_prebundle", SCRIPTS]
    for path in candidates:
        if path.is_dir() and str(path) not in sys.path:
            sys.path.insert(0, str(path))
    import numpy as np
    import warp as wp
    import newton
    wp.config.kernel_cache_dir = str(cache / "warp")
    wp.init()
    if not wp.get_device("cuda:0").is_cuda:
        raise RuntimeError("This training run requires the validated CUDA liquid solver.")
    return {"numpy": np.__version__, "warp": wp.__version__,
            "newton": newton.__version__, "device": str(wp.get_device("cuda:0"))}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                         encoding="utf-8")
    os.replace(temporary, path)


def score_episode(result):
    return (result["absolute_error_ml"] + 10.0 * result["max_spilled_ml"]
            + (0 if result["valid_physics"] else 1000.0)
            + (0 if result["complete"] and result["upright"] else 100.0)
            + 0.0005 * result["physical_time_s"])


def summarize(results):
    import numpy as np
    errors = np.asarray([r["absolute_error_ml"] for r in results])
    safe = [r["valid_physics"] and r["complete"] and r["upright"]
            and r["max_spilled_ml"] == 0 and r["inflight_ml"] <= .1 for r in results]
    return {"episodes": len(results), "mae_ml": float(errors.mean()),
            "worst_absolute_error_ml": float(errors.max()),
            "safe_completion_rate": float(np.mean(safe)),
            "within_0_2_ml_rate": float(np.mean([s and e <= .2 for s, e in zip(safe, errors)])),
            "within_0_5_ml_rate": float(np.mean([s and e <= .5 for s, e in zip(safe, errors)])),
            "spilled_ml": float(sum(r["max_spilled_ml"] for r in results)),
            "physical_steps": sum(r["physical_steps"] for r in results),
            "physical_seconds": sum(r["physical_time_s"] for r in results),
            "all_mass_conserved": all(r["mass_conserved"] for r in results)}


def _enable_graph(environment):
    """Install the optional exact-step acceleration after lazy fluid creation."""
    original_reset = environment._reset

    def reset_with_graph(*args, **kwargs):
        result = original_reset(*args, **kwargs)
        if not getattr(environment.fluid, "_training_graph_installed", False):
            from pbf_graph import install_graph_step
            install_graph_step(environment.fluid)
            environment.fluid._training_graph_installed = True
        return result

    environment._reset = reset_with_graph


def benchmark(output, graph=False):
    from pour_training_env import PourTrainingEnv
    environment = PourTrainingEnv()
    if graph:
        _enable_graph(environment)
    started = time.perf_counter()
    result = environment.run_episode(30, 0, randomize=False,
        record_path=output / "benchmark_steps.jsonl.gz",
        progress=lambda event: print(json.dumps(event), flush=True)
        if event["event"] in ("step", "end") else None)
    result["benchmark_wall_seconds"] = time.perf_counter() - started
    write_json(output / "benchmark.json", result)
    return result


def _now():
    return datetime.now(timezone.utc).isoformat()


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode("utf-8")).hexdigest()


def _atomic_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _write_csv(path, rows, fields):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _load_journal(path, protocol_id):
    """A committed line is reusable only with its complete, unchanged raw trace."""
    if not path.exists():
        return {}
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    records = {}
    valid_lines = []
    for index, line in enumerate(lines):
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            if index == len(lines) - 1 and not line.endswith("\n"):
                _atomic_text(path, "".join(valid_lines))
                break
            raise ValueError(f"Corrupt completed-episode journal at line {index + 1}.")
        if record.get("protocol_id") != protocol_id:
            raise ValueError("Journal contains an episode from another source/protocol.")
        episode_id = record["episode_id"]
        if episode_id in records:
            raise ValueError(f"Duplicate completed episode: {episode_id}")
        raw = path.parent / record["record_path"]
        if not raw.resolve().is_relative_to(path.parent.resolve()):
            raise ValueError("A recorded trace path escapes the output directory.")
        if not raw.is_file() or _sha(raw) != record["record_sha256"]:
            raise ValueError(f"Missing or altered raw trace: {episode_id}")
        records[episode_id] = record
        valid_lines.append(line if line.endswith("\n") else line + "\n")
    if lines and not lines[-1].endswith("\n") and len(valid_lines) == len(lines):
        _atomic_text(path, "".join(valid_lines))
    return records


def _report_markdown(summary):
    training = summary["training"]
    selected = summary["selected"]
    learned = summary["test"]["selected_metrics"]
    baseline = summary["test"]["baseline_metrics"]
    validation_seeds = summary["protocol"]["validation_seeds"]
    test_seeds = summary["protocol"]["test_seeds"]
    test_seed_text = "/".join(str(seed) for seed in test_seeds)
    lines = ["# 目标水量策略训练记录", "",
        f"完成时间（UTC）：{summary['completed_at_utc']}", "",
        "算法：三参数 CEM episodic policy search，只学习倒水停止/扶正时机。"
        "不是 PPO、神经网络训练或六关节端到端控制。", "",
        f"真实训练：{training['episodes']} 回合，{training['generations']} 代 × "
        f"{training['population']} 候选 × 3 个目标；"
        f"{training['physical_steps']} 个物理帧，{training['physical_seconds']:.2f} s 原始物理时间。",
        f"验证种子：{'/'.join(str(seed) for seed in validation_seeds)}；"
        f"共 {len(validation_seeds) * 3 * 2} 个模型选择对照回合。",
        f"模型选择：{selected['selection']}；部署参数来自实际训练：{selected['is_trained']}。",
        f"冻结 SHA-256：`{selected['model_sha256']}`。", "",
        "## 独立测试", "",
        f"冻结后使用 {test_seed_text} 共 {len(test_seeds)} 个新种子，三个目标各自与基准配对；"
        f"共 {len(test_seeds) * 3 * 2} 回合，测试成绩不参与 CEM 或模型选择。", "",
        "| 指标 | 冻结策略 | 原反馈基准 |", "| --- | ---: | ---: |",
        f"| MAE (mL) | {learned['mae_ml']:.6f} | {baseline['mae_ml']:.6f} |",
        f"| 最大绝对误差 (mL) | {learned['worst_absolute_error_ml']:.6f} | {baseline['worst_absolute_error_ml']:.6f} |",
        f"| 安全且 ±0.2 mL | {learned['within_0_2_ml_rate']:.1%} | {baseline['within_0_2_ml_rate']:.1%} |",
        f"| 安全且 ±0.5 mL | {learned['within_0_5_ml_rate']:.1%} | {baseline['within_0_5_ml_rate']:.1%} |",
        f"| 质量守恒 | {learned['all_mass_conserved']} | {baseline['all_mass_conserved']} |", "",
        "| 目标 (mL) | 种子 | 冻结策略入杯 (mL) | 误差 (mL) | 基准入杯 (mL) | 误差 (mL) |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |"]
    for pair in summary["test"]["paired_results"]:
        a, b = pair["selected"], pair["baseline"]
        lines.append(f"| {pair['target_ml']:g} | {pair['seed']} | {a['received_ml']:.6f} | "
                     f"{a['error_ml']:+.6f} | {b['received_ml']:.6f} | {b['error_ml']:+.6f} |")
    lines += ["", "## 边界与复现", "",
        "所有动作由实时 PBF 求解，物理 dt=1/60 s，控制步与物理帧一致。"
        "CUDA graph 若启用仅减少 kernel 调用开销，不跳过物理子步或压缩时间。",
        "mL 来自固定粒子质量除以密度；不是从透明画面估计，也没有真实量杯/秤校准。"
        "单粒子为 0.042875 mL，20/30/40 mL 最近可实现体积为 "
        "19.979750 / 30.012500 / 40.002375 mL。",
        "训练只涉及目标条件、初始位置小扰动与黏性 ±10%；不代表其他杯型、液体、视觉观测或真机泛化。"
        "试管与机器人采用预握和受限运动学路径，无摩擦抓持或液体力矩反馈。",
        "`protocol.json` 保存种子、版本、源文件 SHA 与预算；`episodes.jsonl` 保存逐回合结果，"
        "`traces/*.steps.jsonl.gz` 保存每个原始物理步的观测、动作和守恒量。"
        "`training_curve.csv` 与 `rewards.csv` 是 CEM 分数/回报记录，没有 value loss。", ""]
    return "\n".join(lines)


def run_training(args, versions):
    """Run true live rollouts, select on validation, freeze, then test once."""
    import numpy as np
    from learned_policy import CEMSearch, DEFAULT_PARAMS, LearnedPourPolicy, PARAMETER_BOUNDS
    from pour_training_env import PourTrainingEnv

    if args.generations < 1 or args.population < 3:
        raise ValueError("Use at least one generation and three candidates.")
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    resume = bool(getattr(args, "resume", False))
    graph = bool(getattr(args, "graph", False))
    nominal = bool(getattr(args, "nominal", False))
    quick = bool(getattr(args, "quick", False))
    targets = (20, 30, 40)
    validation_seeds = (101,) if quick else (101, 102, 103)
    test_seeds = (1001, 1002) if quick else (1001, 1002, 1003)
    sources = {"train_target_volume_policy.py": ROOT / "train_target_volume_policy.py"}
    for name in ("liquid.py", "geometry.py", "target_control.py", "learned_policy.py", "pour_training_env.py"):
        sources[f"assets/robot_testtube_pour/scripts/{name}"] = SCRIPTS / name
    if graph:
        sources["assets/robot_testtube_pour/scripts/pbf_graph.py"] = SCRIPTS / "pbf_graph.py"
    source_hashes = {name: _sha(path) for name, path in sources.items()}
    sdk = require_isaac_sim_root()
    sdk_version = sdk / "VERSION"
    software = {**versions, "python": sys.version, "platform": platform.platform(),
                "isaac_sim": sdk_version.read_text(encoding="utf-8").strip() if sdk_version.is_file() else "unavailable"}
    elite_count = min(2, args.population - 1)
    protocol = {"schema_version": 1, "algorithm": "CEM_3_parameter_stopping_policy",
        "targets_ml": list(targets), "training_seed": int(args.seed),
        "generations": int(args.generations), "population": int(args.population), "elite_count": elite_count,
        "quick_mode": quick,
        "parameter_bounds": {name: list(bounds) for name, bounds in PARAMETER_BOUNDS.items()},
        "baseline_parameters": dict(DEFAULT_PARAMS),
        "train_common_seed_by_target": {str(target): seed for target, seed in zip(targets, (11, 12, 13))},
        "validation_seeds": list(validation_seeds), "test_seeds": list(test_seeds),
        "validation_policies": ["training_best", "baseline"],
        "test_policies": ["frozen_selected", "baseline"], "nominal_pairs": nominal,
        "randomization": {"initial_jitter_max_spacing": 0.03, "viscosity_factor": [0.9, 1.1]},
        "physics": {"mode": "live_PBF_action_conditioned", "dt_s": 1 / 60,
                    "spacing_m": 0.0035, "motion_time_scale": 2.0,
                    "minimum_substeps": 4, "max_seconds": 35.0, "cuda_graph": graph},
        "score": "abs(error_ml)+10*max_spilled_ml+1000*invalid_physics+100*incomplete_or_not_upright+0.0005*physical_seconds",
        "selection": "lower validation mean score; ties choose baseline; test never used for fitting or selection",
        "source_sha256": source_hashes, "software": software}
    protocol_id = _signature(protocol)
    protocol_path = output / "protocol.json"
    if protocol_path.exists():
        if not resume:
            raise FileExistsError("Output already contains a protocol; use --resume or a new --output directory.")
        existing = json.loads(protocol_path.read_text(encoding="utf-8"))
        if existing.get("protocol_id") != protocol_id or existing.get("protocol") != protocol:
            raise ValueError("Resume rejected: source, model bounds, seeds, versions or protocol changed.")
    else:
        if resume and (output / "episodes.jsonl").exists():
            raise ValueError("Cannot resume an episode journal without its source/protocol manifest.")
        if any((output / name).exists() for name in ("config.json", "episodes.jsonl", "model_frozen.json", "summary.json")):
            raise FileExistsError("This directory contains another configured run; use a new output directory.")
        write_json(protocol_path, {"created_at_utc": _now(), "protocol_id": protocol_id, "protocol": protocol})

    journal_path = output / "episodes.jsonl"
    records = _load_journal(journal_path, protocol_id) if resume else {}
    search = CEMSearch(seed=args.seed, population=args.population, elite_count=elite_count)
    environment = PourTrainingEnv()
    if graph:
        _enable_graph(environment)
    started, newly_run = time.perf_counter(), 0
    total_planned = (args.generations * args.population * 3 + len(validation_seeds) * 3 * 2
                     + len(test_seeds) * 3 * 2 + (6 if nominal else 0))
    progress_base = {"protocol_id": protocol_id, "started_or_resumed_at_utc": _now(),
                     "total_planned_episodes": total_planned, "completed_episodes": len(records)}
    reward_fields = ("episode_id", "stage", "generation", "candidate", "target_ml", "seed",
                     "randomized", "score", "reward", "received_ml", "error_ml", "physical_steps",
                     "physical_time_s", "wall_seconds", "record_path")

    def update_csv():
        rows = []
        for record in records.values():
            result = record["result"]
            rows.append({**record["conditions"], "episode_id": record["episode_id"],
                         "score": record["score"], "reward": -record["score"],
                         **{key: result[key] for key in ("received_ml", "error_ml", "physical_steps", "physical_time_s", "wall_seconds")},
                         "record_path": record["record_path"]})
        _write_csv(output / "rewards.csv", rows, reward_fields)

    def episode(episode_id, stage, target, seed, params, *, randomized=True, generation=None, candidate=None):
        nonlocal newly_run
        conditions = {"stage": stage, "target_ml": float(target), "seed": int(seed),
                      "policy_params": None if params is None else dict(params),
                      "randomized": bool(randomized), "generation": generation, "candidate": candidate}
        if episode_id in records:
            previous = records[episode_id]
            if previous["conditions"] != conditions or previous["conditions_sha256"] != _signature(conditions):
                raise ValueError(f"Resume episode/model conditions differ: {episode_id}")
            return previous["result"]
        record_path = output / "traces" / f"{episode_id}.steps.jsonl.gz"
        policy = None if params is None else LearnedPourPolicy(params)

        def progress(event):
            write_json(output / "progress.json", {**progress_base,
                "state": "running", "updated_at_utc": _now(), "episode_id": episode_id,
                "stage": stage, "conditions": conditions,
                "completed_episodes": len(records), "wall_seconds_this_invocation": time.perf_counter() - started,
                **event})

        print(f"EPISODE {len(records) + 1}/{total_planned}: {episode_id}", flush=True)
        result = environment.run_episode(target, seed, policy=policy, randomize=randomized,
                                        record_path=record_path, progress=progress)
        if not record_path.is_file():
            raise RuntimeError(f"Solver did not preserve its physical-step trace: {episode_id}")
        record = {"episode_id": episode_id, "protocol_id": protocol_id,
            "conditions": conditions, "conditions_sha256": _signature(conditions),
            "record_path": record_path.relative_to(output).as_posix(), "record_sha256": _sha(record_path),
            "completed_at_utc": _now(), "score": float(score_episode(result)), "result": result}
        with journal_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        records[episode_id] = record
        newly_run += 1
        update_csv()
        progress({"event": "episode_recorded", "received_ml": result["received_ml"],
                  "error_ml": result["error_ml"], "score": record["score"]})
        print(f"RESULT {episode_id}: {result['received_ml']:.6f} mL, "
              f"error={result['error_ml']:+.6f}, score={record['score']:.6f}", flush=True)
        return result

    try:
        curve = []
        training_results = []
        for generation in range(args.generations):
            candidates = search.ask()
            scores, generation_results = [], []
            for candidate_index, params in enumerate(candidates):
                candidate_results = []
                for target, seed in zip(targets, (11, 12, 13)):
                    episode_id = f"train_g{generation:03d}_c{candidate_index:03d}_t{target}_s{seed}"
                    result = episode(episode_id, "train", target, seed, params,
                                     generation=generation, candidate=candidate_index)
                    candidate_results.append(result)
                scores.append(float(np.mean([score_episode(result) for result in candidate_results])))
                generation_results.extend(candidate_results)
            generation_summary = search.tell(scores)
            training_results.extend(generation_results)
            write_json(output / f"generation_{generation:03d}.json",
                       {"protocol_id": protocol_id, **generation_summary})
            write_json(output / "search_state.json", {"protocol_id": protocol_id, **search.state_dict()})
            curve.append({"generation": generation, "mean_score": float(np.mean(scores)),
                          "generation_best_score": float(min(scores)), "best_training_score": search.best_score,
                          "mean_reward": -float(np.mean(scores)),
                          "mae_ml": summarize(generation_results)["mae_ml"],
                          "completed_training_episodes": len(training_results)})
            _write_csv(output / "training_curve.csv", curve, tuple(curve[0]))
            print("GENERATION " + json.dumps(curve[-1]), flush=True)

        validation = {"training_best": [], "baseline": []}
        for seed in validation_seeds:
            for target in targets:
                for label, params in (("training_best", search.best_params), ("baseline", None)):
                    validation[label].append(episode(f"validation_{label}_t{target}_s{seed}",
                                                      "validation", target, seed, params))
        validation_scores = {label: float(np.mean([score_episode(result) for result in values]))
                             for label, values in validation.items()}
        is_trained = (search.best_params != DEFAULT_PARAMS and
                      validation_scores["training_best"] < validation_scores["baseline"])
        selected_params = dict(search.best_params if is_trained else DEFAULT_PARAMS)
        selection = "cem_training_best" if is_trained else "baseline_fallback"
        validation_report = {"protocol_id": protocol_id, "scores": validation_scores,
            "metrics": {label: summarize(values) for label, values in validation.items()},
            "results": validation, "selected": selection, "selected_params": selected_params}
        write_json(output / "validation.json", validation_report)

        frozen_path, freeze_path = output / "model_frozen.json", output / "freeze.json"
        if frozen_path.exists() or freeze_path.exists():
            if not resume or not frozen_path.is_file() or not freeze_path.is_file():
                raise ValueError("Incomplete/existing model freeze requires an intact --resume run.")
            frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
            freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
            if (freeze.get("protocol_id") != protocol_id or freeze.get("model_sha256") != _sha(frozen_path)
                    or frozen.get("parameters") != selected_params
                    or frozen.get("metadata", {}).get("selection") != selection):
                raise ValueError("Frozen model no longer matches validated model/protocol.")
        else:
            frozen_at = _now()
            frozen = LearnedPourPolicy(selected_params).save(frozen_path, metadata={
                "frozen_at_utc": frozen_at, "protocol_id": protocol_id, "selection": selection,
                "selection_origin": selection,
                "is_trained": is_trained, "training_seed": int(args.seed),
                "training_episodes": len(training_results), "validation_scores": validation_scores,
                "source_sha256": source_hashes, "test_used_for_selection": False,
                "observations": "privileged simulator particle state; no camera estimator"})
            freeze = {"protocol_id": protocol_id, "frozen_at_utc": frozen_at,
                      "model_sha256": _sha(frozen_path), "selection": selection,
                      "selection_origin": selection, "is_trained": is_trained}
            write_json(freeze_path, freeze)
            _atomic_text(output / "model_frozen.sha256", freeze["model_sha256"] + "  model_frozen.json\n")

        # The checkpoint and its SHA/timestamp exist before the first test step.
        test_pairs = []
        for seed in test_seeds:
            for target in targets:
                selected = episode(f"test_selected_t{target}_s{seed}", "test", target, seed, selected_params)
                baseline = episode(f"test_baseline_t{target}_s{seed}", "test", target, seed, None)
                test_pairs.append({"target_ml": target, "seed": seed, "selected": selected, "baseline": baseline})
        if _sha(frozen_path) != freeze["model_sha256"]:
            raise RuntimeError("The frozen model changed during independent testing.")
        nominal_pairs = []
        if nominal:
            for target in targets:
                selected = episode(f"nominal_selected_t{target}", "nominal", target, 0, selected_params, randomized=False)
                baseline = episode(f"nominal_baseline_t{target}", "nominal", target, 0, None, randomized=False)
                nominal_pairs.append({"target_ml": target, "seed": 0, "selected": selected, "baseline": baseline})

        published_path = ROOT / "assets" / "robot_testtube_pour" / "policy" / "target_volume_policy.json"
        published_path.parent.mkdir(parents=True, exist_ok=True)
        published_temporary = published_path.with_name(published_path.name + ".tmp")
        published_temporary.write_bytes(frozen_path.read_bytes())
        os.replace(published_temporary, published_path)
        all_results = [record["result"] for record in records.values()]
        summary = {"complete": True, "completed_at_utc": _now(), "protocol_id": protocol_id,
            "protocol": protocol, "training": {**summarize(training_results),
                "generations": args.generations, "population": args.population,
                "best_score": search.best_score, "best_params": dict(search.best_params)},
            "validation": validation_report,
            "selected": {**freeze, "parameters": selected_params,
                         "gui_checkpoint": published_path.relative_to(ROOT).as_posix()},
            "test": {"paired_results": test_pairs,
                "selected_metrics": summarize([pair["selected"] for pair in test_pairs]),
                "baseline_metrics": summarize([pair["baseline"] for pair in test_pairs])},
            "nominal_paired_results": nominal_pairs, "all_rollouts": summarize(all_results),
            "total_recorded_episodes": len(records), "new_episodes_this_invocation": newly_run,
            "wall_seconds_this_invocation": time.perf_counter() - started,
            "total_rollout_wall_seconds": sum(result["wall_seconds"] for result in all_results),
            "test_used_for_training_or_selection": False,
            "limitations": ["privileged particle-state observations; not vision or a real scale",
                "kinematic pre-grasp; no contact-force grasp or fluid torque feedback",
                "only three targets and narrow initial-state/viscosity randomization",
                "particle-volume resolution is not real-world dosing accuracy"]}
        write_json(output / "summary.json", summary)
        _atomic_text(output / "REPORT.md", _report_markdown(summary))
        write_json(output / "progress.json", {**progress_base, "state": "complete", "updated_at_utc": _now(),
            "completed_episodes": len(records), "new_episodes_this_invocation": newly_run,
            "model_sha256": freeze["model_sha256"], "selection": selection})
        print("TRAINING COMPLETE " + json.dumps({"episodes": len(records), "selection": selection,
              "test_mae_ml": summary["test"]["selected_metrics"]["mae_ml"], "output": str(output)}), flush=True)
        return summary
    except BaseException as error:
        write_json(output / "progress.json", {**progress_base,
            "state": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
            "updated_at_utc": _now(), "completed_episodes": len(records),
            "error_type": type(error).__name__, "error": str(error)})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--generations", type=int, default=None)
    parser.add_argument("--population", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20261121)
    parser.add_argument("--resume", action="store_true", help="Reuse only matching completed live episodes.")
    parser.add_argument("--graph", action="store_true", help="Use optional exact PBF CUDA-graph stepping.")
    parser.add_argument("--nominal", action="store_true", help="Also run six unperturbed paired episodes.")
    parser.add_argument("--quick", action="store_true",
                        help="Default to 2x3 candidates, one validation seed and two independent test seeds.")
    parser.add_argument("--output", type=Path, default=ROOT / "droplet_out" / "pour_policy_training")
    args = parser.parse_args()
    if args.generations is None:
        args.generations = 2 if args.quick else 3
    if args.population is None:
        args.population = 3 if args.quick else 6
    args.output.mkdir(parents=True, exist_ok=True)
    versions = configure_sdk()
    print("TRAINING ENVIRONMENT " + json.dumps(versions), flush=True)
    if args.benchmark:
        benchmark(args.output, graph=args.graph)
        return
    run_training(args, versions)


if __name__ == "__main__":
    main()
