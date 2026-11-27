"""CPU-only plots for a complete, independently audited 42-case V9 study.

No solver/controller/SDK imports. Partial summaries and missing audits are
rejected before journal/test decoding. Synthetic fixtures use temporary files
with explicit watermarks and never enter the experiment output directory.
"""

import argparse
import csv
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN = ROOT / "droplet_out/pour_policy_fine_v9/run_20261127"
BUDGET = {"train": 18, "validation": 6, "test": 18, "total": 42}
TARGETS = (20, 30, 40)
QUANTUM = .019683


class IncompleteRun(ValueError):
    pass


def need(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    json.dumps(value, allow_nan=False)
    return value


def inside(root, relative):
    path = (root / relative).resolve()
    need(not Path(relative).is_absolute() and path.is_relative_to(root.resolve()) and path.is_file(),
         "Missing or unsafe artifact: "+relative)
    return path


def signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def near(actual, expected):
    need(type(actual) in (int, float) and math.isfinite(actual) and abs(actual-expected) < 1e-8,
         "Recorded numeric value differs from recomputed evidence.")


def task_assessment(row):
    r, s, p = row["result"], row["status"], row["post_observation"]
    valid = (r["valid_physics"] and r["mass_conserved"] and r["all_states_finite"]
        and r["source_partition_conserved"] and r["max_below_table_particles"] == 0
        and abs(r["measurement_physical_time_s"]-r["physical_time_s"]) < 1e-8
        and isinstance(p, dict) and p["all_states_finite"] and p["mass_array_unchanged"]
        and p.get("maximum_below_table_particles", 0) == 0
        and abs(p.get("start_physical_time_s", -1)-r["physical_time_s"]) < 1e-8
        and p.get("physical_steps", 0) >= 120
        and p.get("end_physical_time_s", -1)-p.get("start_physical_time_s", 0) >= 2.-1e-8)
    home = bool(r["complete"] and s.get("done") and s.get("home_static")
        and s.get("stable_frames", 0) >= 12 and s.get("source_airborne_particles") == 0
        and not s.get("timed_out") and r["physical_time_s"] <= 24.)
    dose = abs(r["error_ml"]) <= .001*r["target_ml"]
    late = isinstance(p, dict) and p.get("maximum_received_change_ml") == 0.
    peak = row["assessment"]["maximum_all_label_spilled_particles"]
    need(type(peak) is int and peak >= 0, "All-label peak spill must be a recorded nonnegative integer.")
    return {"valid_evidence": bool(valid), "dose_passed": dose, "full_home_within_deadline": home,
        "maximum_all_label_spilled_particles": peak, "spill_passed": peak <= 1,
        "no_late_dose_change": late, "tolerance_ml": .001*r["target_ml"],
        "feasible": bool(valid and dose and home and peak <= 1 and late)}


def original_metrics(rows):
    errors = [abs(row["result"]["error_ml"]) for row in rows]
    n = len(rows)
    return {"n": n, "mae_ml": sum(errors)/n, "worst_ml": max(errors),
        "feasible": sum(row["assessment"]["feasible"] for row in rows),
        "valid": sum(row["assessment"]["valid_evidence"] for row in rows),
        "dose_passed": sum(row["assessment"]["dose_passed"] for row in rows),
        "spill_particles": sum(row["assessment"]["maximum_all_label_spilled_particles"] for row in rows),
        "mean_physical_s": sum(row["result"]["physical_time_s"] for row in rows)/n,
        "mean_wall_s": sum(row["wall_seconds"] for row in rows)/n}


def selection_key(rows):
    m = original_metrics(rows)
    if m["valid"] != m["n"]:
        return [2, m["n"]-m["valid"], m["mae_ml"]]
    if m["feasible"] == m["n"]:
        return [0, m["mean_physical_s"]]
    return [1, m["mae_ml"], m["spill_particles"]]


def check_metrics(recorded, rows):
    actual = original_metrics(rows)
    need(set(recorded) == set(actual), "Summary metric schema changed.")
    for key, value in actual.items():
        near(recorded[key], value)


def load_run(directory, audit_path=None):
    run = Path(directory).resolve()
    spath = run / "summary.json"
    if not spath.is_file():
        raise IncompleteRun("No complete summary; journal and test were not decoded.")
    summary = read(spath)
    if summary.get("complete") is not True or summary.get("counts") != {"train":18,"validation":6,"test":18}:
        raise IncompleteRun("Full 18/6/18 budget required; journal and test were not decoded.")
    apath = Path(audit_path).resolve() if audit_path else run / "fine_run_audit.json"
    if not apath.is_file():
        raise IncompleteRun("Final independent fine_run_audit.json is required before journal/test decoding.")
    audit = read(apath)
    need(audit.get("artifact_integrity_verified") is True and audit.get("all_failures_retained") is True,
         "Independent artifact audit has not passed.")
    need(audit.get("episodes_audited") == 42 and audit.get("raw_action_trace_count") == 42
         and audit.get("budgets") == BUDGET and audit.get("source_count") == 19 and audit.get("input_count") == 6,
         "Audit describes a different experiment budget or provenance set.")
    need(audit.get("summary_sha256") == sha(spath) and audit.get("journal_sha256") == sha(run/"episodes.jsonl"),
         "Audit does not bind the current final summary/journal bytes.")
    p = read(run/"protocol.json")
    pid = p["protocol_id"]
    need(signature({key:value for key,value in p.items() if key != "protocol_id"}) == pid == audit["protocol_id"],
         "Protocol identity mismatch.")
    need(p["kind"] == "fine_particle_24s_coarse_speed_CEM_v9" and p["budget"] == BUDGET,
         "Expected the V9 one-dimensional coarse-speed CEM protocol.")
    need(p["targets_ml"] == list(TARGETS) and p["user_criterion"] == {
        "deadline_s":24., "relative_error":.001, "max_all_label_spill_particles":1,
        "full_original_home_and_12_quiet_frames":True}, "Strict percent/home/spill criterion changed.")
    need(p["initial_particle_count"] == 11625 and p["initial_source_particles"] == 2880,
         "Fine-particle initial total/source identities changed.")
    near(p["particle_ml"], QUANTUM)
    near(p["initial_source_ml"], 2880*QUANTUM)
    seeds = [set(p[key]) for key in ("train_seeds","validation_seeds","test_seeds")]
    need([len(group) for group in seeds] == [1,1,3] and all(seeds[i].isdisjoint(seeds[j]) for i in range(3) for j in range(i)),
         "Train/validation/test conditions are not isolated.")
    need(len(p["source_sha256"]) == 19 and len(p["input_sha256"]) == 6, "Provenance count changed.")
    for field, folder in (("source_sha256","source_snapshot"),("input_sha256","input_snapshot")):
        for name, expected in p[field].items():
            need(sha(inside(run, folder+"/"+name)) == expected, "A frozen source/input changed.")
    frozen = read(run/"freeze.json")
    need(summary["freeze"] == frozen and frozen["protocol_id"] == pid and frozen["source_sha256"] == p["source_sha256"],
         "Summary/model/source freeze differs.")
    need(frozen["validation_only_selection"] is True and frozen["test_used_for_selection"] is False
         and frozen["gui_deployment"] is False and summary["gui_deployment"] is False, "Selection/deployment boundary changed.")
    models = {}
    for filename, expected in frozen["models_sha256"].items():
        need(filename in ("candidate_frozen.json","recommended_frozen.json"), "Unknown frozen model.")
        path = inside(run, filename)
        need(sha(path) == expected, "Frozen model bytes changed.")
        model = read(path)
        need(model["protocol_id"] == pid and model["source_sha256"] == p["source_sha256"]
             and model["test_used_for_selection"] is False and model["gui_deployment"] is False, "Model provenance changed.")
        models[filename] = {"sha256":expected,"definition":model}
    need(set(models) == {"candidate_frozen.json","recommended_frozen.json"}, "Both pre-test models are required.")
    rows = [json.loads(line) for line in (run/"episodes.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    need(len(rows) == 42 and len({row["episode_id"] for row in rows}) == 42, "Cases missing or duplicated.")
    need({stage:sum(row["stage"] == stage for row in rows) for stage in ("train","validation","test")} == summary["counts"],
         "Stages do not match the committed budget.")
    for row in rows:
        json.dumps(row, allow_nan=False)
        r = row["result"]
        need((r["target_ml"],r["seed"]) == (row["target_ml"],row["seed"]), "Footer condition mismatch.")
        near(r["received_ml"]-row["target_ml"], r["error_ml"])
        near(r["physical_time_s"], r["physical_steps"]/60)
        need(row["assessment"] == task_assessment(row), "Task gate does not match recorded action/post evidence.")
        need(read(inside(run,"completed/"+row["episode_id"]+".json")) == row, "Completed sidecar mismatch.")
        for name, expected in row["artifacts"].items():
            need(sha(inside(run,name)) == expected, "Action/post artifact changed after final audit.")
        if row["stage"] == "test":
            need(datetime.fromisoformat(row["started_at_utc"]) > datetime.fromisoformat(frozen["frozen_at_utc"]),
                 "Test started before model freeze.")
    generations = summary["generations"]
    need(len(generations) == 2, "Two actual training generations are required.")
    candidates = []
    for generation, value in enumerate(generations):
        need(value["generation"] == generation and len(value["candidates"]) == 3, "Training population differs.")
        need(read(run/f"generation_{generation}.json") == value, "Generation snapshot differs.")
        for index, candidate in enumerate(value["candidates"]):
            need((candidate["generation"],candidate["index"]) == (generation,index), "Candidate index differs.")
            selected = [row for row in rows if row["stage"] == "train" and row["episode_id"].startswith(f"train_candidate_g{generation}c{index}_")]
            conditions(selected,p["train_seeds"],candidate["multiplier"],"candidate")
            check_metrics(candidate["metrics"],selected)
            need(list(candidate["key"]) == selection_key(selected), "Training selection key differs.")
            candidates.append(candidate)
    best = min((candidate for candidate in candidates if candidate["key"][0] < 2), key=lambda value:value["key"])
    for stage, group_seeds in (("validation",p["validation_seeds"]),("test",p["test_seeds"])):
        for role in ("candidate","baseline"):
            selected = [row for row in rows if row["stage"] == stage and row["role"] == role]
            conditions(selected,group_seeds,best["multiplier"] if role == "candidate" else 1.,role)
            check_metrics(summary[stage][role],selected)
    validation = {role:[row for row in rows if row["stage"] == "validation" and row["role"] == role] for role in ("candidate","baseline")}
    recommendation = "candidate" if selection_key(validation["candidate"]) <= selection_key(validation["baseline"]) else "baseline"
    need(summary["recommended_role"] == frozen["recommendation"] == recommendation, "Recommendation is not validation-only.")
    near(frozen["candidate_multiplier"],best["multiplier"])
    for filename, role in (("candidate_frozen.json","candidate"),("recommended_frozen.json",recommendation)):
        model = models[filename]["definition"]
        need(model["implementation"] == role, "Frozen model implementation differs from selected role.")
        near(model["coarse_speed_multiplier"],best["multiplier"] if role == "candidate" else 1.)
    need(summary["task_passed"] is all(row["assessment"]["feasible"] for row in rows if row["stage"] == "test" and row["role"] == recommendation),
         "Final task acceptance differs; test must not reselect the model.")
    return {"verified":True,"synthetic":False,"run":run,"audit_path":apath,"audit":audit,
        "protocol":p,"summary":summary,"freeze":frozen,"models":models,"rows":rows}


def conditions(rows, seeds, multiplier, role):
    expected = {(target,seed) for target in TARGETS for seed in seeds}
    need(len(rows) == len(expected) and {(row["target_ml"],row["seed"]) for row in rows} == expected,
         "Condition coverage differs or a failed case was omitted.")
    need(all(row["role"] == role and row["multiplier"] == multiplier for row in rows), "Role or multiplier changed.")


def descriptive(rows):
    value = original_metrics(rows)
    value.update(signed_errors_ml=[row["result"]["error_ml"] for row in rows],
        mean_signed_error_ml=sum(row["result"]["error_ml"] for row in rows)/len(rows),
        peak_all_label_spill_including_post=max(row["assessment"]["maximum_all_label_spilled_particles"] for row in rows),
        home_within_24s_count=sum(row["assessment"]["full_home_within_deadline"] for row in rows),
        no_late_dose_change_count=sum(row["assessment"]["no_late_dose_change"] for row in rows),
        mean_action_wall_seconds=sum(row["result"]["wall_seconds"] for row in rows)/len(rows),
        mean_case_wall_seconds_including_post=value["mean_wall_s"],
        joint_task_feasible_count=value["feasible"],
        post_observation_available_count=sum(isinstance(row["post_observation"],dict) for row in rows),
        post_wall_seconds_separately_recorded=False)
    durations = [row["post_observation"].get("duration_s",row["post_observation"].get("end_physical_time_s",0)-row["post_observation"].get("start_physical_time_s",0))
                 for row in rows if isinstance(row["post_observation"],dict)]
    value["mean_post_physical_seconds_when_available"] = sum(durations)/len(durations) if durations else None
    return value


def compact(row):
    p = row["post_observation"]
    return {"case_id":row["episode_id"],"target_ml":row["target_ml"],"seed":row["seed"],"multiplier":row["multiplier"],
        "received_ml":row["result"]["received_ml"],"signed_error_ml":row["result"]["error_ml"],
        "action_physical_seconds":row["result"]["physical_time_s"],"action_wall_seconds":row["result"]["wall_seconds"],
        "case_wall_seconds_including_post":row["wall_seconds"],"assessment":row["assessment"],
        "post":{key:p[key] for key in ("start_physical_time_s","end_physical_time_s","duration_s","physical_steps",
            "maximum_received_change_ml","all_states_finite","mass_array_unchanged") if isinstance(p,dict) and key in p}
            if isinstance(p,dict) else None}


def table(data):
    p, rows, s = data["protocol"],data["rows"],data["summary"]
    pairs = []
    tests = [row for row in rows if row["stage"] == "test"]
    for target in TARGETS:
        for seed in p["test_seeds"]:
            pair = {"target_ml":target,"seed":seed}
            for role in ("candidate","baseline"):
                pair[role] = compact(next(row for row in tests if (row["target_ml"],row["seed"],row["role"]) == (target,seed,role)))
            pairs.append(pair)
    return {"schema_version":1,"kind":"audited_fine_v9_descriptive_plot_table","synthetic_fixture":data["synthetic"],
        "protocol_id":p["protocol_id"],"counts":BUDGET,"particle_ml":QUANTUM,
        "initial_source_particles":2880,"initial_source_ml":2880*QUANTUM,"initial_total_particles":11625,
        "recommended_role":s["recommended_role"],"task_passed":s["task_passed"],
        "generations":s["generations"],"validation":s["validation"],"test":s["test"],
        "models":data["models"],"paired_cases":pairs,
        "per_target":[{"target_ml":target,"tolerance_ml":.001*target,
            **{role:descriptive([row for row in tests if row["target_ml"] == target and row["role"] == role])
               for role in ("candidate","baseline")}} for target in TARGETS],
        "limitations":["One-dimensional coarse-speed CEM plus fixed hand-written engineering changes, not PPO/neural training.",
            "Privileged particle-mass volume; kinematic pregrasp/path; one train seed and three narrow-randomization test seeds.",
            "Action physical time includes full original home and quiet interval; post observation is separate and cannot rescue action failures.",
            "Dose-only counts differ from joint task feasibility; all failed cases are retained. Test never chooses the policy.",
            "Peak all-label spill is the recorded action-plus-post peak. Case wall includes reset/post/overhead; isolated post wall is not recorded.",
            "Candidate includes fixed micro-backoff/settled-hold/topup-margin changes; baseline is the unchanged legacy controller, even at multiplier 1."]}


def plotting():
    deps = ROOT/"_cache/pour_training/plot_deps"
    if deps.is_dir():
        sys.path.insert(0,str(deps))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def figures(report, output):
    plt = plotting()
    plt.rcParams.update({"font.size":9,"figure.dpi":150,"axes.spines.top":False,"axes.spines.right":False})
    prefix = "SYNTHETIC_" if report["synthetic_fixture"] else ""
    footer = ("SYNTHETIC FIXTURE - NOT A SIMULATION RESULT" if report["synthetic_fixture"] else
              "Privileged mass | kinematic path | narrow reset randomization | 42 live cases; no failures removed")
    paths = []

    def save(fig, name):
        fig.text(.5,.012,footer,ha="center",fontsize=8)
        fig.tight_layout(rect=(0,.045,1,.94))
        path = output/(prefix+name)
        fig.savefig(path,facecolor="white")
        plt.close(fig)
        paths.append(path)

    fig, axes = plt.subplots(1,3,figsize=(12.5,3.7))
    selected = []
    for generation in report["generations"]:
        x = generation["generation"]+1
        for candidate in generation["candidates"]:
            m = candidate["metrics"]
            axes[0].scatter(x+(candidate["index"]-1)*.06,m["mae_ml"],c="#aaa",marker="x" if m["valid"]<3 else "o",s=28)
        best = min(generation["candidates"],key=lambda candidate:candidate["key"])
        selected.append(best)
    xs = [1,2]
    axes[0].plot(xs,[c["metrics"]["mae_ml"] for c in selected],"o-",label="Generation-selected MAE")
    axes[0].plot(xs,[c["metrics"]["worst_ml"] for c in selected],"s--",label="Its worst error")
    axes[0].set(ylabel="Absolute dose error (mL)",title="Training only: 2 generations")
    axes[0].legend(fontsize=8)
    for key, label, marker in (("dose_passed","Dose only","o"),("feasible","All task gates","s"),("valid","Valid evidence","^")):
        axes[1].plot(xs,[c["metrics"][key] for c in selected],marker+"-",label=label)
    axes[1].set(ylim=(-.2,3.3),yticks=(0,1,2,3),ylabel="Cases / 3 training conditions",title="Same selected candidate")
    axes[1].legend(fontsize=8)
    axes[2].plot(xs,[c["metrics"]["mean_physical_s"] for c in selected],"o-",color="#bd5d19")
    axes[2].axhline(24,color="#b33",ls="--",label="24 s action limit")
    axes[2].set(ylabel="Mean raw physical action (s)",title="Home + quiet included")
    axes[2].legend(fontsize=8)
    for ax in axes:
        ax.set(xticks=xs,xlabel="Generation")
        ax.grid(alpha=.2)
    fig.suptitle("V9 coarse-speed CEM; fixed engineering changes, not PPO")
    save(fig,"fine_training_curve.png")

    fig, axes = plt.subplots(1,3,figsize=(12.5,4.1))
    colors = {"candidate":"#d46b23","baseline":"#2479ad"}
    for ax, item in zip(axes,report["per_target"]):
        target = item["target_ml"]
        pairs = [pair for pair in report["paired_cases"] if pair["target_ml"] == target]
        ax.axhspan(-.001*target,.001*target,color="#d1ead5",label="Strict +/-0.1% dose band")
        ax.axhline(0,color="#555",lw=.7)
        for index,pair in enumerate(pairs,1):
            ax.plot([index-.10,index+.10],[pair["candidate"]["signed_error_ml"],pair["baseline"]["signed_error_ml"]],color="#aaa",lw=.7)
            for role,offset in (("candidate",-.10),("baseline",.10)):
                case = pair[role]
                marker = "o" if case["assessment"]["feasible"] else "x"
                ax.scatter(index+offset,case["signed_error_ml"],c=colors[role],marker=marker,s=46,
                           label=role if index==1 else None,zorder=3)
        c,b = item["candidate"],item["baseline"]
        ax.set(title=f"{target} mL | tolerance {target*.001:.3f} mL",xticks=[1,2,3],
               xticklabels=[str(pair["seed"]) for pair in pairs],xlabel="Paired test seed",ylabel="Signed error (mL)")
        ax.text(.02,.98,f"C/B MAE {c['mae_ml']:.4g}/{b['mae_ml']:.4g}\nWorst {c['worst_ml']:.4g}/{b['worst_ml']:.4g}\nDose {c['dose_passed']}/3, {b['dose_passed']}/3; all gates {c['feasible']}/3, {b['feasible']}/3",
                transform=ax.transAxes,va="top",fontsize=8,bbox={"facecolor":"white","alpha":.8,"edgecolor":"none"})
        ax.margins(y=.28)
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=7,loc="lower left")
    fig.suptitle("All 9 paired conditions retained | circle = all task gates passed; x = any task-gate failure")
    save(fig,"fine_paired_errors.png")

    fig, axes = plt.subplots(2,2,figsize=(12,6.2))
    for role,offset in (("candidate",-.16),("baseline",.16)):
        values = [item[role] for item in report["per_target"]]
        x = [index+offset for index in range(3)]
        axes[0,0].bar(x,[v["mean_physical_s"] for v in values],.30,color=colors[role],label=role)
        axes[1,0].bar(x,[v["mean_case_wall_seconds_including_post"] for v in values],.30,
                      color=colors[role],alpha=.35,label=role+" case incl. post")
        axes[1,0].bar(x,[v["mean_action_wall_seconds"] for v in values],.20,color=colors[role],label=role+" action wall")
        axes[1,1].bar(x,[v["mean_post_physical_seconds_when_available"] if v["mean_post_physical_seconds_when_available"] is not None else float("nan")
                        for v in values],.30,color=colors[role],label=role)
        for location,value in zip(x,values):
            if value["mean_post_physical_seconds_when_available"] is None:
                axes[1,1].text(location,0,"missing",rotation=90,ha="center",va="bottom",fontsize=7)
        for index,pair in enumerate(report["paired_cases"]):
            axes[0,1].scatter(index+offset,pair[role]["assessment"]["maximum_all_label_spilled_particles"],c=colors[role],s=30,label=role if index==0 else None)
    axes[0,0].axhline(24,color="#b33",ls="--")
    axes[0,0].set(ylabel="Mean raw physical action (s)",title="Full home and quiet; deadline 24 s")
    axes[0,1].axhspan(0,1,color="#d1ead5")
    axes[0,1].set(ylabel="Peak all-label spilled particles",xlabel="9 target-seed pairs",title="Recorded action + post peak; limit 1",xticks=range(9),xticklabels=[str(i) for i in range(1,10)])
    axes[1,0].set(ylabel="Mean wall time (s)",title="Machine/load dependent; post wall not isolated")
    axes[1,1].set(ylabel="Mean available post physical time (s)",title="Separate observation; never deadline rescue")
    for ax in (axes[0,0],axes[1,0],axes[1,1]):
        ax.set(xticks=range(3),xticklabels=["20 mL","30 mL","40 mL"])
    for ax in axes.flat:
        ax.legend(fontsize=7)
        ax.grid(axis="y",alpha=.2)
    fig.suptitle("V9 time/spill tradeoff; dose feasibility is reported separately")
    save(fig,"fine_time_spill.png")
    return paths


def write_outputs(data, directory):
    need(data.get("verified") is True or data.get("synthetic") is True,"Unverified data cannot produce final artifacts.")
    output = Path(directory).resolve()
    if data["synthetic"]:
        need(output.is_relative_to((ROOT/"_cache/pour_training").resolve()),"Synthetic output must stay in the temporary fixture area.")
    output.mkdir(parents=True,exist_ok=True)
    report = table(data)
    pngs = figures(report,output)
    prefix = "SYNTHETIC_" if data["synthetic"] else ""
    jpath = output/(prefix+"fine_results_table.json")
    jpath.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    csvpath = output/(prefix+"fine_per_target.csv")
    fields = ["target_ml","role","tolerance_ml","n","mae_ml","worst_ml","signed_errors_ml","dose_passed","feasible","joint_task_feasible_count","valid",
              "peak_all_label_spill_including_post","mean_physical_s","mean_action_wall_seconds",
              "mean_case_wall_seconds_including_post","mean_post_physical_seconds_when_available"]
    with csvpath.open("w",newline="",encoding="utf-8") as stream:
        writer = csv.DictWriter(stream,fieldnames=fields)
        writer.writeheader()
        for item in report["per_target"]:
            for role in ("candidate","baseline"):
                row = {key:item[role].get(key) for key in fields}
                row.update(target_ml=item["target_ml"],role=role,tolerance_ml=item["tolerance_ml"],
                           signed_errors_ml=json.dumps(item[role]["signed_errors_ml"]))
                writer.writerow(row)
    provenance = {"schema_version":1,"protocol_id":report["protocol_id"],"synthetic_fixture":data["synthetic"],
        "plot_tool_sha256":sha(__file__),"files_sha256":{p.name:sha(p) for p in (*pngs,jpath,csvpath)}}
    if not data["synthetic"]:
        provenance.update(summary_sha256=sha(data["run"]/"summary.json"),journal_sha256=sha(data["run"]/"episodes.jsonl"),
            audit_sha256=sha(data["audit_path"]),protocol_sha256=sha(data["run"]/"protocol.json"),
            source_sha256=data["protocol"]["source_sha256"],input_sha256=data["protocol"]["input_sha256"])
    (output/(prefix+"fine_plot_provenance.json")).write_text(json.dumps(provenance,indent=2)+"\n",encoding="utf-8")
    return report


def self_test():
    home = ROOT/"_cache/pour_training"
    home.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="fine_plot_fixture_",dir=home) as name:
        folder = Path(name)
        (folder/"summary.json").write_text('{"complete":false,"counts":{"train":17}}',encoding="utf-8")
        (folder/"episodes.jsonl").write_text("MUST NOT DECODE THIS JOURNAL",encoding="utf-8")
        try:
            load_run(folder)
        except IncompleteRun:
            pass
        else:
            raise AssertionError("Partial data accepted")
        (folder/"summary.json").write_text(json.dumps({"complete":True,"counts":{"train":18,"validation":6,"test":18}}))
        try:
            load_run(folder)
        except IncompleteRun:
            pass
        else:
            raise AssertionError("Missing independent audit accepted")
        rows = []
        for role in ("candidate","baseline"):
            for seed in (11,12,13):
                for target in TARGETS:
                    error = (.01 if role=="candidate" else -.15)*(seed-10)
                    a = {"maximum_all_label_spilled_particles":2 if seed==13 else 0}
                    row = {"episode_id":f"SYNTHETIC_{role}_{target}_{seed}","stage":"test","role":role,"target_ml":target,"seed":seed,"multiplier":1.,
                        "result":{"target_ml":target,"error_ml":error,"received_ml":target+error,"valid_physics":True,"mass_conserved":True,"all_states_finite":True,
                            "source_partition_conserved":True,"max_below_table_particles":0,"measurement_physical_time_s":10.,"physical_time_s":10.,"complete":True,"wall_seconds":20.},
                        "status":{"done":True,"home_static":True,"stable_frames":12,"source_airborne_particles":0,"timed_out":False},
                        "post_observation":{"all_states_finite":True,"mass_array_unchanged":True,"physical_steps":120,"start_physical_time_s":10.,"end_physical_time_s":12.,"duration_s":2.,"maximum_received_change_ml":0.},
                        "assessment":a,"wall_seconds":24.}
                    row["assessment"] = task_assessment(row)
                    rows.append(row)
        data = {"verified":False,"synthetic":True,"protocol":{"protocol_id":"SYNTHETIC_NOT_AN_EXPERIMENT","test_seeds":[11,12,13]},"rows":rows,"models":{},
            "summary":{"recommended_role":"candidate","task_passed":False,"validation":{},
                "test":{role:original_metrics([row for row in rows if row["role"]==role]) for role in ("candidate","baseline")},
                "generations":[{"generation":g,"candidates":[{"index":i,"multiplier":1.,"key":[1,.02+i*.01,0],
                    "metrics":{"mae_ml":.02+i*.01,"worst_ml":.06,"valid":3,"feasible":i,"dose_passed":2,"mean_physical_s":10.+g}} for i in range(3)]} for g in range(2)]}}
        report = write_outputs(data,folder/"render")
        need(len(report["paired_cases"]) == 9 and sum(item["candidate"]["feasible"] for item in report["per_target"]) < 9,
             "Synthetic failed cases disappeared.")
        need(len(list((folder/"render").glob("SYNTHETIC_*.png"))) == 3,"Expected three watermarked fixture figures.")
    print("PASS: partial/missing-audit refusal, signed pairs, failed-case retention and 3 synthetic PNGs; no formal OUT writes.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",type=Path,default=DEFAULT_RUN)
    parser.add_argument("--audit",type=Path)
    parser.add_argument("--output",type=Path)
    parser.add_argument("--check-only",action="store_true")
    parser.add_argument("--self-test",action="store_true")
    args = parser.parse_args()
    if args.self_test:
        return self_test()
    data = load_run(args.run,args.audit)
    if args.check_only:
        print(json.dumps({"verified":True,"cases":42,"protocol_id":data["protocol"]["protocol_id"],"task_passed":data["summary"]["task_passed"]}))
        return
    report = write_outputs(data,args.output or args.run/"plots")
    print(json.dumps({"written":True,"protocol_id":report["protocol_id"],"per_target":report["per_target"]}))


if __name__ == "__main__":
    main()
