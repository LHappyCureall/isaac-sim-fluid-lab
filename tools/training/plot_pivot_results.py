"""Plot only a completed and independently audited 78-case V4 pivot run.

No controller, collector, SDK, Warp, Kit, GPU, training or deployment is used.
The summary-complete gate precedes all journal/heldout reads. Failed cases
remain in every comparison; post observation never rescues an action miss.
"""
import argparse
from datetime import datetime
import csv
import gzip
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN = ROOT / "droplet_out/pour_policy_pivot_v4/run_20261122_joint_cem"
ALGORITHM = "feasibility_first_CEM_upright_lift_fixed_mouth_feedback_v4"
POLICY_KIND = "upright_lift_fixed_mouth_planned_return_v2"
BUDGET = {"generations":2,"population":4,"elites":2,"training":48,"validation":12,"test":18,"total":78}
TARGETS, ROLES = (20,30,40), ("candidate","baseline")
QUANTUM = .042875
RENDERER_PATH = ROOT / "tools/training/plot_fast_results.py"
RENDERER_SHA256 = hashlib.sha256(RENDERER_PATH.read_bytes()).hexdigest()
_spec = importlib.util.spec_from_file_location("pivot_read_only_plot_helpers",RENDERER_PATH)
draw = importlib.util.module_from_spec(_spec)
exec(compile(RENDERER_PATH.read_text(encoding="utf-8"),str(RENDERER_PATH),"exec"),draw.__dict__)
# Only generic CPU I/O, descriptive statistics and the nine-pair renderer are
# reused. The V3 loader, 138-case schema and three-generation renderer are not.
number, require, sha, signature = draw.number, draw.require, draw.sha, draw.signature
read, inside, metrics, close = draw.json_read, draw.inside, draw.metrics, draw.close
selection_key, release_passed = draw.selection_key, draw.release_passed


class IncompleteRun(ValueError):
    pass


def artifacts(run,row,d,*,read_post=False):
    for prefix in ("record","motion_record","post_record"):
        relative=row.get(prefix+"_path")
        if relative is None:
            require(prefix=="post_record" and row.get(prefix+"_sha256") is None,"Missing action or motion evidence.")
        else:require(sha(inside(run,relative))==row[prefix+"_sha256"],"Changed raw/sidecar SHA: "+relative)
    evidence=read(inside(run,row["motion_record_path"]))
    require(evidence==row["motion_evidence"] and evidence["action_result_modified"] is False
        and evidence["original_action_result"]==row["result"] and evidence["conditions"]==row["conditions"]
        and evidence["action_motion_metrics"]==row["motion_metrics"] and evidence["assessment"]==row["assessment"]
        and evidence["total_action_plus_post_wall_seconds"]==row["total_action_plus_post_wall_seconds"]
        and evidence["completed_at_utc"]==row["completed_at_utc"],"Original action/conditions/fixed wall sidecar link differs.")
    with gzip.open(inside(run,row["record_path"]),"rt",encoding="utf-8") as stream:
        header=json.loads(next(stream));effective=json.loads(next(stream));last=None
        for line in stream:last=line
    c,r=row["conditions"],row["result"]
    require(header["type"]=="episode" and effective["type"]=="effective_direct_sources","Missing actual direct implementation header.")
    require((header["target_ml"],header["seed"],header["policy_params"],header["randomized"])==
            (c["target_ml"],c["seed"],c["policy_params"],c["randomized"]),"Raw header and journal conditions differ.")
    require(header["policy_name"]==r["policy_name"]==POLICY_KIND,"Incorrect independent V4 policy kind.")
    close(header["physics_dt_s"],1/60,"Physical dt differs.");close(header["particle_volume_ml"],QUANTUM,"Particle quantum differs.")
    expected={Path(d["module_binding"][role]).name:d["source_sha256"][d["module_binding"][role]] for role in ("geometry_file","controller_file")}
    require(effective["source_sha256"]==expected and effective["cache_sha256"]==d["input_files"]["path_cache"]["sha256"]
        and effective["tube_containment"]=="full_rotation_matrix_not_a_pitch_projection"
        and effective["physics_solver_changed"] is False,"Actual geometry/controller/cache/full-R identity differs.")
    require(effective["reset_started_utc"]==row["episode_wall_clock_start_utc"],"Pre-physics UTC differs.")
    footer=json.loads(last);require(footer.pop("type")=="result" and footer==r,"An unchanged original action footer is missing.")
    post=row["post_observation"] if isinstance(row["post_observation"],dict) else {}
    detail={"deadline_received_ml":post.get("deadline_received_ml"),"final_post_received_ml":None,
            "final_post_physical_time_s":None,"max_received_change_after_5s_ml":None,"delta_5s_to_last_post_ml":None}
    if read_post and row.get("post_record_path"):
        at5=r["received_ml"] if abs(r["physical_time_s"]-5)<1e-10 else None
        observations=[]
        with gzip.open(inside(run,row["post_record_path"]),"rt",encoding="utf-8") as stream:
            for line in stream:
                item=json.loads(line);require(item["type"]=="post_observation","Unexpected post event.")
                observations.append((number(item["physical_time_s"]),number(item["measurement"]["received_ml"])))
                if abs(item["physical_time_s"]-5)<1e-10:at5=item["measurement"]["received_ml"]
        if observations:detail.update(final_post_received_ml=observations[-1][1],final_post_physical_time_s=observations[-1][0])
        if detail["deadline_received_ml"] is not None:
            require(at5 is not None,"A claimed 5-second sample is absent from raw observations.")
            close(at5,detail["deadline_received_ml"],"The actual 5-second sample differs.")
            detail["max_received_change_after_5s_ml"]=max((abs(ml-at5) for t,ml in observations if t>=5-1e-10),default=0.)
            detail["delta_5s_to_last_post_ml"]=observations[-1][1]-at5 if observations else None
    return detail


def load_complete(directory,*,frozen_only=False):
    run=Path(directory).resolve();summary_path=run/"summary.json"
    if not summary_path.is_file():raise IncompleteRun("Final 78-case summary is unavailable; journal/test were not read and no final artifact was generated.")
    s=read(summary_path)
    if s.get("complete") is not True or s.get("total_recorded_episodes")!=78:
        raise IncompleteRun("The declared 78 cases are incomplete; journal/test were not read.")
    apath=run/"pivot_run_audit.json"
    if not apath.is_file():raise IncompleteRun("Completed V4 requires the independent pivot_run_audit.json before plotting.")
    audit=read(apath)
    require(audit.get("artifact_integrity_verified") is True and audit.get("all_failures_retained") is True
        and audit.get("episodes_audited")==audit.get("raw_action_trace_count")==78
        and audit.get("budgets")=={"training":48,"validation":12,"test":18,"total":78},"Independent V4 artifact audit is incomplete.")
    require(audit["summary_sha256"]==sha(summary_path) and audit["journal_sha256"]==sha(run/"episodes.jsonl"),"Audit describes different final bytes.")
    document=read(run/"protocol.json");p=document["protocol"];d=p["declaration"];pid=signature(p)
    require(document["protocol_id"]==s["protocol_id"]==audit["protocol_id"]==pid and s["protocol"]==p,"Protocol identity differs.")
    require(p["kind"]=="formal_upright_lift_fixed_mouth_CEM_78_live_episodes" and d["algorithm"]==ALGORITHM
        and d["policy_kind"]==POLICY_KIND and d["neural_policy"] is False and d["ppo"] is False,"Different experiment family.")
    require(d["budget"]==s["budget"]==BUDGET and d["targets_ml"]==list(TARGETS),"Declared V4 budget/targets differ.")
    require(d["module_binding"]["collector_class"]=="DirectTrainingEnv"
        and d["module_binding"]["effective_record_type"]=="effective_direct_sources"
        and d["module_binding"]["step_evidence_type"]=="direct_step_evidence"
        and d["module_binding"]["containment"]=="actual_solver_body_q_full_R","Actual V4 binding was replaced by legacy assumptions.")
    require(d["engineering_gate"]["eligible"] is True and d["search_config_is_draft"] is False,"No finalized three-target engineering gate.")
    groups=[set(d[name+"_seeds"]) for name in ("train","validation","test")]
    require([len(g) for g in groups]==[2,2,3] and not any(groups[i]&groups[j] for i in range(3) for j in range(i)),"Seed split is not isolated.")
    frozen=read(run/"input_freeze.json")
    require(frozen["declaration"]==d and frozen["input_id"]==p["input_id"]==signature(d),"Input freeze differs.")
    require(len(d["source_sha256"])==30 and len(d["input_files"])==15,"Expected the frozen thirty sources and fifteen inputs.")
    for relative,digest in d["source_sha256"].items():
        require(sha(inside(run,"source_snapshot/"+relative))==digest,"Frozen source changed: "+relative)
        if not frozen_only:require(sha(ROOT/relative)==digest,"Active source changed: "+relative)
    for name,item in d["input_files"].items():
        require(sha(inside(run,"input_snapshot/"+name+Path(item["path"]).suffix))==item["sha256"],"Frozen input changed: "+name)
    require(s["gui_checkpoint_sha256_after"]==d["gui_checkpoint_sha256_before"],"Original GUI checkpoint bytes changed during this experiment.")
    if not frozen_only:
        for relative,digest in d["gui_checkpoint_sha256_before"].items():
            require(sha(ROOT/relative)==digest,"Current original GUI checkpoint changed: "+relative)
    freeze=read(run/"freeze.json")
    require(freeze["protocol_id"]==pid and freeze["test_used_for_selection"] is False,"Models were not frozen before heldout use.")
    for role in ("candidate","recommended"):
        model=s["models"][role];path=inside(run,model["file"]);checkpoint=read(path);meta=checkpoint["metadata"]
        require(sha(path)==model["sha256"]==freeze[role+"_sha256"] and checkpoint["parameters"]==model["parameters"]==freeze[role+"_params"],"Frozen model bytes/parameters changed.")
        require(checkpoint["policy_kind"]==POLICY_KIND and set(checkpoint["parameters"])==set(d["cem"]["parameter_names"])
            and meta["source_sha256"]==d["source_sha256"] and meta["module_binding"]==d["module_binding"]
            and meta["algorithm"]==ALGORITHM and meta["training_episodes"]==48 and meta["role"]==role
            and meta["frozen_at_utc"]==freeze["frozen_at_utc"] and meta["protocol_id"]==pid
            and meta["test_used_for_selection"] is False and meta["automatic_gui_deployment"] is False,"V4 model provenance/kind differs.")
    rows={}
    with (run/"episodes.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            row=json.loads(line);name=row["episode_id"]
            require(name not in rows and row["protocol_id"]==pid and row["conditions_sha256"]==signature(row["conditions"]),"Duplicate/foreign/modified case.")
            rows[name]=row
    expected={f"train_g{g:03d}_c{c:03d}_t{t}_s{seed}" for g in range(2) for c in range(4) for t in TARGETS for seed in d["train_seeds"]}
    expected|={f"{stage}_{role}_t{t}_s{seed}" for stage in ("validation","test") for role in ROLES for t in TARGETS for seed in d[stage+"_seeds"]}
    require(set(rows)==expected and len(rows)==78 and s["all_episode_ids"]==list(rows),"Final journal omitted/added a declared case.")
    details={}
    for row in rows.values():
        c,r,a=row["conditions"],row["result"],row["assessment"]
        require((r["target_ml"],r["seed"],r["policy_params"],r["randomized"])==
                (c["target_ml"],c["seed"],c["policy_params"],c["randomized"]),"Actual result/condition linkage differs.")
        require(r["physical_steps"]<=300,"An action exceeded five physical seconds.")
        close(r["physical_time_s"],r["physical_steps"]/60,"Action clock is not original physical step/60.")
        close(a["absolute_error_ml"],abs(a["dose_error_ml"]),"Signed/absolute error differs.")
        require(a["constraint_feasible"]==(a["physical_safe"] and a["completed_by_deadline"] and a["late_drop_passed"])
            and a["success_within_0_1_ml"]==(a["constraint_feasible"] and a["absolute_error_ml"]<=.1),"Safety/deadline/accuracy were conflated.")
        detail=artifacts(run,row,d,read_post=c["stage"]=="test");available=type(detail["deadline_received_ml"]) in (int,float)
        require(a["deadline_dose_measurement_available"]==available,"Direct 5-second availability differs.")
        require(a["dose_measurement_basis"]==("physical_5s" if available else "invalid_terminal_fallback_not_a_5s_measurement"),"A fallback was relabeled as a 5-second sample.")
        close(a["dose_error_ml"],(detail["deadline_received_ml"] if available else r["received_ml"])-r["target_ml"],"Dose error/time basis differs.")
        details[row["episode_id"]]=detail
        if c["stage"]=="test":require(datetime.fromisoformat(row["episode_wall_clock_start_utc"])>datetime.fromisoformat(freeze["frozen_at_utc"]),"A test reset preceded model freeze.")
    train=[r for r in rows.values() if r["conditions"]["stage"]=="train"]
    draw.check_metrics(s["training"]["metrics"],train);draw.check_metrics(s["all_rollouts"],list(rows.values()))
    generations=s["training"]["generations"];require(len(generations)==2 and [g["generation"] for g in generations]==[0,1],"Missing/reordered CEM generations.")
    candidates=[]
    for generation in generations:
        require(len(generation["candidates"])==4,"An unsuccessful candidate was dropped.")
        require(read(run/f"generation_{generation['generation']:03d}.json")=={"protocol_id":pid,**generation},"Generation snapshot/summary differs.")
        for candidate in generation["candidates"]:
            case_rows=[rows[name] for name in candidate["episode_ids"]]
            require(len(case_rows)==6 and all(row["conditions"]["generation"]==generation["generation"]
                and row["conditions"]["candidate"]==candidate["index"] and row["conditions"]["policy_params"]==candidate["params"] for row in case_rows),"Candidate condition group differs.")
            draw.check_metrics(candidate["metrics"],case_rows);draw.check_key(candidate["selection_key"],selection_key(case_rows));candidates.append(candidate)
    best=min(candidates,key=lambda candidate:candidate["selection_key"])
    require(best["params"]==s["training"]["best_params"]==freeze["candidate_params"],"Cumulative feasibility-first training optimum differs.")
    draw.check_key(s["training"]["best_key"],best["selection_key"])
    val={role:[r for r in rows.values() if r["conditions"]["stage"]=="validation" and r["conditions"]["policy_role"]==role] for role in ROLES}
    scores={role:selection_key(group) for role,group in val.items()}
    for role,group in val.items():
        require(len(group)==6,"Validation group omitted a case.")
        draw.check_metrics(s["validation"]["metrics"][role],group);draw.check_key(s["validation"]["selection_keys"][role],scores[role])
    baseline=d["cem"]["baseline_params"]
    recommended="candidate" if best["params"]!=baseline and scores["candidate"]<scores["baseline"] else "baseline"
    require(s["recommendation"]=={"role":recommended,"selected_on_validation_only":True,"additional_test_rollouts":0}
        and freeze["recommended_role"]==recommended and s["validation"]["recommended_role"]==recommended
        and s["validation"]["test_used_for_selection"] is False,"Recommendation was not fixed on validation.")
    require(s["models"]["recommended"]["parameters"]==(best["params"] if recommended=="candidate" else baseline),"Recommended model role was confused with candidate.")
    pairs=s["test"]["paired_results"]
    require(len(pairs)==9 and {(pair["target_ml"],pair["seed"]) for pair in pairs}=={(t,seed) for t in TARGETS for seed in d["test_seeds"]},"Missing one of nine paired holdout conditions.")
    tests={role:[] for role in ROLES}
    for pair in pairs:
        require(pair["recommended_result_role"]==recommended,"A paired recommendation differs.")
        for role in ROLES:
            row=pair[role]
            require(row==rows[row["episode_id"]] and (row["conditions"]["target_ml"],row["conditions"]["seed"])==(pair["target_ml"],pair["seed"])
                and row["conditions"]["policy_params"]==(best["params"] if role=="candidate" else baseline),"Holdout result was modified, omitted, or used a different policy.")
            tests[role].append(row)
    for role,group in tests.items():
        draw.check_metrics(s["test"]["metrics"][role],group)
        require(s["release_acceptance"][role+"_passed"] is release_passed(group),"Heldout acceptance differs.")
    require(s["test"]["recommended_metrics"]==s["test"]["metrics"][recommended]
        and s["release_acceptance"]["recommended_passed"] is release_passed(tests[recommended])
        and s["release_acceptance"]["does_not_reselect_model"] is True and s["automatic_gui_deployment"] is False,
        "Heldout test must not reselect, refit or deploy a model.")
    return {"run":run,"summary":s,"declaration":d,"rows":rows,"details":details,"audit_path":apath,"frozen_only":frozen_only}


def comparison(rows,compact):
    out=draw.comparison_metrics(rows,compact)
    out.update(physical_safe_count=sum(row["physical_safe"] for row in compact),
        completed_by_5s_zero_total_spill_count=sum(row["completed_by_5s"] and row["max_total_water_spilled_ml"]==0 for row in compact),
        post_gate_passed_count=sum(row["late_drop_passed"] for row in compact))
    return out


def report(data):
    s,d,rows=data["summary"],data["declaration"],data["rows"]
    pairs=sorted(s["test"]["paired_results"],key=lambda pair:(pair["target_ml"],pair["seed"]))
    compact=[{"target_ml":pair["target_ml"],"seed":pair["seed"],**{
        role:draw.compact_case(pair[role],data["details"][pair[role]["episode_id"]]) for role in ROLES}} for pair in pairs]
    per_target=[]
    for target in TARGETS:
        per_target.append({"target_ml":target,"independent_target_seed_conditions":3,"executed_live_episodes":6,
            **{role:comparison([pair[role] for pair in pairs if pair["target_ml"]==target],
                               [pair[role] for pair in compact if pair["target_ml"]==target]) for role in ROLES}})
    curve=[];cumulative=None
    for generation in s["training"]["generations"]:
        current=min(generation["candidates"],key=lambda candidate:candidate["selection_key"])
        if cumulative is None or current["selection_key"]<cumulative["selection_key"]:cumulative=current
        group=[rows[name] for candidate in generation["candidates"] for name in candidate["episode_ids"]]
        curve.append({"generation":generation["generation"]+1,"population_metrics":metrics(group),
            "generation_best_metrics":current["metrics"],"cumulative_lexicographic_best_metrics":cumulative["metrics"],
            "generation_best_selection_key":current["selection_key"],"cumulative_lexicographic_best_selection_key":cumulative["selection_key"],
            "candidate_maes_ml":[candidate["metrics"]["mae_ml"] for candidate in generation["candidates"]]})
    stages={stage:{role:metrics(group) for role in ROLES if (group:=[row for row in rows.values()
        if row["conditions"]["stage"]==stage and row["conditions"]["policy_role"]==role])} for stage in ("train","validation","test")}
    return {"schema_version":1,"protocol_id":s["protocol_id"],"complete_committed_cases":78,"budget":BUDGET,
        "algorithm":ALGORITHM,"policy_kind":POLICY_KIND,"neural_policy":False,"ppo":False,
        "baseline_is_engineering_008_policy_not_historical_v3":True,"baseline_parameters":d["cem"]["baseline_params"],
        "cem_initial_center_parameters":d["cem"]["initial_params"],"candidate_parameters":s["models"]["candidate"]["parameters"],
        "recommended_role_selected_on_validation":s["recommendation"]["role"],"models":s["models"],
        "recommendation_reuses_one_of_two_test_roles_no_third_rollout_set":True,"independent_test_conditions":9,"executed_test_episodes":18,
        "release_acceptance":s["release_acceptance"],"particle_quantum_ml":QUANTUM,"physical_action_deadline_s":5.,
        "post_observation_absolute_end_at_least_s":7.,"post_observation_never_rescues_a_failed_action":True,
        "wall_seconds_depend_on_machine_load":True,"unsafe_and_deadline_counts_can_overlap":True,
        "training_curve":curve,"stage_metrics":stages,"per_target_comparison":per_target,
        "overall_test_comparison":{role:comparison([pair[role] for pair in pairs],[pair[role] for pair in compact]) for role in ROLES},
        "all_nine_paired_cases_including_failures":compact,"interrupted_artifact_records":s.get("interrupted_artifact_records",0),
        "source_sha256":d["source_sha256"],"module_binding":d["module_binding"],
        "limitations":s["limitations"]+[draw.FOOTNOTE,
            "48 training, 12 validation and 18 heldout committed cases. Development nominal pilots are not heldout cases.",
            "Missing direct 5s observations remain failures and use explicitly flagged terminal fallback only in descriptive loss.",
            "Completion, zero spill, all non-dose gates and dose precision are separate counts.",
            "Physical seconds are the primary timing measure; measured wall includes live return planning and machine load."]}


def training_figure(r,path):
    plt,np=draw.plotting();curve=r["training_curve"];x=np.arange(1,3)
    fig,axes=plt.subplots(2,2,figsize=(13,8.3))
    for i,row in enumerate(curve):axes[0,0].scatter(np.full(4,i+1),row["candidate_maes_ml"],color="#aab2bc",s=25)
    for key,label,color in (("population_metrics","Population (24 cases/generation)","#7b8796"),
        ("generation_best_metrics","Generation feasibility-first best","#137fbd"),
        ("cumulative_lexicographic_best_metrics","Cumulative feasibility-first best","#319668")):
        axes[0,0].plot(x,[row[key]["mae_ml"] for row in curve],"o-",label=label,color=color)
    axes[0,0].set(title="Descriptive training MAE, including failed cases",ylabel="Absolute error (mL)");axes[0,0].legend(fontsize=8)
    for offset,key,label,color in ((-.24,"unsafe_count","Unsafe","#bb4c43"),(0,"deadline_or_home_or_late_drop_failures","Deadline / home / post failed","#da9c25"),(.24,"feasible_count","Non-dose gates passed","#319668")):
        axes[0,1].bar(x+offset,[row["population_metrics"][key] for row in curve],width=.22,label=label,color=color)
    axes[0,1].set(title="24 true live cases per generation",ylabel="Cases",ylim=(0,27));axes[0,1].legend(fontsize=8)
    for key,label,color in (("population_metrics","Population mean","#7b8796"),("cumulative_lexicographic_best_metrics","Cumulative best (6 conditions)","#319668")):
        axes[1,0].plot(x,[row[key]["mean_action_physical_seconds"] for row in curve],"o-",label=label,color=color)
    axes[1,0].axhline(5,color="#b54848",linestyle="--",linewidth=1,label="Hard action deadline")
    axes[1,0].set(title="Original physical action time",ylabel="Physical seconds",ylim=(0,5.4));axes[1,0].legend(fontsize=8)
    stages=[r["stage_metrics"][stage] for stage in ("train","validation","test")];n=[sum(group["episodes"] for group in stage.values()) for stage in stages];sx=np.arange(3)
    for offset,key,label,color in ((-.24,"unsafe_count","Unsafe","#bb4c43"),(0,"deadline_or_home_or_late_drop_failures","Deadline / home / post failed","#da9c25"),(.24,"feasible_count","Non-dose gates passed","#319668")):
        axes[1,1].bar(sx+offset,[100*sum(group[key] for group in stage.values())/count for stage,count in zip(stages,n)],width=.22,label=label,color=color)
    axes[1,1].set(title="Separate stage gates can overlap",ylabel="Percent of stage cases",ylim=(0,112),xticks=sx,xticklabels=[f"{name}\nn={count}" for name,count in zip(("Train","Validation","Test"),n)])
    axes[1,1].legend(fontsize=8)
    for ax in (axes[0,0],axes[0,1],axes[1,0]):ax.set(xticks=x,xlabel="CEM generation")
    for ax in axes.flat:ax.grid(axis="y",alpha=.2)
    fig.suptitle("V4 fixed-mouth parameter CEM: 2 generations × 4 candidates × 6 conditions = 48 training cases",fontsize=14)
    draw.save_figure(fig,path,"12 validation + 18 paired test cases follow training. Non-dose gates exclude the dose-precision threshold; all failures remain included.");plt.close(fig)


def targets_figure(r,path):
    plt,np=draw.plotting();x=np.arange(3);rows=r["per_target_comparison"]
    fig,axes=plt.subplots(2,3,figsize=(15,8))
    specs=(("mae_ml","Descriptive MAE: all three paired seeds","mL"),
        ("worst_absolute_error_ml","Worst absolute descriptive error","mL"),
        ("mean_action_physical_seconds","Mean original physical action time","Physical seconds"),
        ("completed_by_5s_zero_total_spill_count","Home by 5s + zero total spill; bracket = home only","Count / 3"),
        ("mean_max_total_water_spilled_ml","Mean maximum spill: all water","mL / case"),
        ("mean_action_plus_post_wall_seconds","Action + post wall time; load dependent","Wall seconds"))
    for ax,(key,title,unit) in zip(axes.flat,specs):
        for index,role in enumerate(ROLES):
            values=[row[role][key] for row in rows];bars=ax.bar(x+(index-.5)*.34,values,width=.32,color=draw.COLORS[role],label=role.title())
            for bar,value,row in zip(bars,values,rows):
                text=f"{value:.3g}"+(f" [{row[role]['completed_by_5s_count']}]" if key=="completed_by_5s_zero_total_spill_count" else "")
                ax.annotate(text,(bar.get_x()+bar.get_width()/2,bar.get_height()),xytext=(0,3),textcoords="offset points",ha="center",fontsize=8)
        ax.set(title=title,ylabel=unit,xticks=x,xticklabels=["20 mL","30 mL","40 mL"]);ax.grid(axis="y",alpha=.2)
        if key=="completed_by_5s_zero_total_spill_count":ax.set_ylim(0,3.7)
        if key=="mean_action_physical_seconds":ax.axhline(5,color="#b54848",linestyle="--",linewidth=1);ax.set_ylim(0,5.8)
    axes[0,0].legend(fontsize=8)
    fig.suptitle("V4 heldout comparison: 3 paired seeds per target; validation recommendation = "+r["recommended_role_selected_on_validation"],fontsize=14)
    draw.save_figure(fig,path,"Two test roles only. Baseline is the finalized V4 engineering policy; completion + zero spill does not itself pass mass, post or precision gates.");plt.close(fig)


def outputs(data,output,*,tables_only=False):
    r=report(data);output.mkdir(parents=True,exist_ok=True);files=[]
    path=output/"pivot_comparison_table.json";path.write_text(json.dumps(r,indent=2,allow_nan=False)+"\n",encoding="utf-8");files.append(path)
    csv_rows=[]
    for target in r["per_target_comparison"]:
        for role in ROLES:
            row={"target_ml":target["target_ml"],"role":role,**target[role]}
            csv_rows.append({k:json.dumps(v) if isinstance(v,(list,dict)) else v for k,v in row.items()})
    path=output/"pivot_per_target_comparison.csv"
    with path.open("w",encoding="utf-8-sig",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(csv_rows[0]));writer.writeheader();writer.writerows(csv_rows)
    files.append(path)
    if not tables_only:
        for name,fn in (("pivot_training_curve.png",training_figure),("pivot_per_target_comparison.png",targets_figure),("pivot_all_heldout_pairs.png",draw.plot_pairs)):
            path=output/name;fn(r,path);files.append(path)
    require(sha(RENDERER_PATH)==RENDERER_SHA256,"Generic read-only renderer changed during plotting.")
    provenance={"protocol_id":r["protocol_id"],"complete_committed_cases":78,"summary_sha256":sha(data["run"]/"summary.json"),
        "journal_sha256":sha(data["run"]/"episodes.jsonl"),"audit_sha256":sha(data["audit_path"]),"plot_tool_sha256":sha(__file__),
        "generic_read_only_renderer_sha256":RENDERER_SHA256,"active_sources_checked":not data["frozen_only"],
        "artifact_sha256":{p.name:sha(p) for p in files},"no_policy_fitting_or_deployment":True,"all_failures_retained":True}
    (output/"pivot_plot_provenance.json").write_text(json.dumps(provenance,indent=2)+"\n",encoding="utf-8")
    return {"output_directory":str(output),"files":[str(p) for p in files],"recommended_role":r["recommended_role_selected_on_validation"],"release_acceptance":r["release_acceptance"],"overall":r["overall_test_comparison"]}


def self_test():
    with tempfile.TemporaryDirectory(dir=ROOT/"_cache/pour_training") as temporary:
        run=Path(temporary);require(run.resolve().is_relative_to(ROOT/"_cache/pour_training"),"Unsafe temporary fixture directory.")
        (run/"summary.json").write_text('{"complete":false,"total_recorded_episodes":77}')
        (run/"episodes.jsonl").write_text("THIS JOURNAL MUST NOT BE READ")
        try:load_complete(run)
        except IncompleteRun:pass
        else:raise AssertionError("Partial run was accepted.")
        require(not list(run.glob("*.png")),"Partial run generated final figures.")
    rows=[]
    for i in range(3):
        rows.append({"assessment":{"absolute_error_ml":[.03,.4,2.][i],"dose_error_ml":[.03,-.4,2.][i],
            "physical_safe":i!=2,"completed_by_deadline":i!=1,"late_drop_passed":True,"constraint_feasible":i==0,
            "deadline_dose_measurement_available":i!=2,"success_within_0_1_ml":i==0},
            "result":{"mass_conserved":True,"physical_time_s":[4.4,5.,1.][i],"physical_steps":[264,300,60][i]},"post_observation":{"duration_s":2.}})
    m=metrics(rows);require(m["episodes"]==3 and m["unsafe_count"]==1 and m["missing_deadline_measurements"]==1
        and m["deadline_or_home_or_late_drop_failures"]==1 and m["feasible_count"]==1,"Failed cases or independent gates were lost.")
    close(m["mae_ml"],2.43/3,"Terminal fallback disappeared from descriptive loss.")
    # Matplotlib fixtures are explicitly synthetic and temporary, never final
    # experiment artifacts. They cover zero bars and a missing-5s marker.
    pairs=[];targets=[]
    for target in TARGETS:
        for index,seed in enumerate((91101,91102,91103)):
            pair={"target_ml":target,"seed":seed}
            for role in ROLES:
                unavailable=role=="baseline" and target==30 and index==0
                error=2. if unavailable else .1*(index-1)
                pair[role]={"signed_error_at_5s_ml":None if unavailable else error,"signed_error_for_descriptive_loss_ml":error,
                    "action_physical_seconds":4.5 if role=="candidate" else 5.,"completed_by_5s":role=="candidate",
                    "delta_5s_to_last_post_ml":None if unavailable else 0.}
            pairs.append(pair)
        targets.append({"target_ml":target,**{role:{"mae_ml":.1,"worst_absolute_error_ml":.2,
            "mean_action_physical_seconds":4.5,"completed_by_5s_zero_total_spill_count":2 if role=="candidate" else 0,
            "completed_by_5s_count":3 if role=="candidate" else 1,"mean_max_total_water_spilled_ml":0. if role=="candidate" else .042875,
            "mean_action_plus_post_wall_seconds":12.} for role in ROLES}})
    fixture={"recommended_role_selected_on_validation":"baseline","per_target_comparison":targets,
        "all_nine_paired_cases_including_failures":pairs,"stage_metrics":{stage:{"candidate":m} for stage in ("train","validation","test")},
        "training_curve":[{"candidate_maes_ml":[.1,.2,.3,.4],"population_metrics":m,"generation_best_metrics":m,"cumulative_lexicographic_best_metrics":m} for _ in range(2)]}
    with tempfile.TemporaryDirectory(dir=ROOT/"_cache/pour_training") as temporary:
        require(Path(temporary).resolve().is_relative_to(ROOT/"_cache/pour_training"),"Unsafe temporary renderer fixture directory.")
        for i,fn in enumerate((training_figure,targets_figure,draw.plot_pairs)):
            path=Path(temporary)/f"synthetic_fixture_{i}.png";fn(fixture,path)
            require(path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n") and path.stat().st_size>20000,"Synthetic renderer failed.")
    print(json.dumps({"self_test_passed":True,"checks":["78-complete gate precedes journal/test reads","no partial final artifacts",
        "unsafe/deadline/missing-5s cases retained separately","two-generation and three-target/nine-pair synthetic PNGs rendered without live results"]}))


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--run",type=Path,default=DEFAULT_RUN);p.add_argument("--output",type=Path)
    p.add_argument("--check-only",action="store_true");p.add_argument("--tables-only",action="store_true");p.add_argument("--frozen-only",action="store_true");p.add_argument("--self-test",action="store_true")
    args=p.parse_args(argv)
    if args.self_test:self_test();return 0
    try:
        data=load_complete(args.run,frozen_only=args.frozen_only)
        if args.check_only:
            r=report(data);print(json.dumps({"validated_complete_cases":78,"paired_test_conditions":9,"all_failures_retained":True,
                "recommended_role":r["recommended_role_selected_on_validation"],"release_acceptance":r["release_acceptance"]}))
        else:print(json.dumps(outputs(data,args.output or args.run/"plots",tables_only=args.tables_only),indent=2,allow_nan=False))
    except IncompleteRun as exc:print(str(exc),file=sys.stderr);return 2
    return 0


if __name__=="__main__":raise SystemExit(main())
