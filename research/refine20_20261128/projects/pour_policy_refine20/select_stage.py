"""Select only from an audited train or validation stage; never reads test rows."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import task_score


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_text(encoding="utf8"))
def now(): return datetime.now(timezone.utc).isoformat()
def write(path, value):
    with path.open("x", encoding="utf8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment",type=Path,required=True)
    parser.add_argument("--stage",choices=("train","validation"),required=True)
    args=parser.parse_args()
    root=args.experiment.resolve()
    plan=read(root/"experiment_plan.json")
    stage=root/args.stage
    audit=read(stage/"audit.json")
    protocol=read(stage/"protocol.json")
    if not audit.get("pass") or not audit.get("complete"):
        raise ValueError("Independent complete physical audit required before selection")
    if audit.get("stage") != args.stage:
        raise ValueError("Independent audit belongs to a different stage")
    for name,digest in audit["input_artifact_sha256"].items():
        if sha(stage/name) != digest:
            raise ValueError("Audited evidence changed before selection: "+name)
    if sha(Path(__file__).with_name("audit_refine_run.py")) != audit["auditor_source_sha256"]:
        raise ValueError("Selection must use the exact reviewed independent auditor")
    if protocol["matrix"]["parent_plan_sha256"] != sha(root/"experiment_plan.json"):
        raise ValueError("Preregistered plan changed")
    task_path=Path(__file__).with_name("task_score.py")
    if sha(task_path) != protocol["source_sha256"]["projects/pour_policy_refine20/task_score.py"]:
        raise ValueError("Frozen ranking implementation changed")
    rows=[json.loads(line) for line in (stage/"episodes.jsonl").read_text(encoding="utf8").splitlines()]
    groups={}
    for row in rows:
        if row["split"] != args.stage or row["stage"] != args.stage:
            raise ValueError("Selection may only inspect its declared split")
        groups.setdefault(row["role"],[]).append(row)
    selected=task_score.select(groups)
    scores={name:list(task_score.score(items)) for name,items in groups.items()}
    if args.stage == "train":
        engineering=root/"engineering"
        engineering_audit=read(engineering/"audit.json")
        if not (engineering_audit.get("complete") and engineering_audit.get("pass")
                and engineering_audit.get("stage")=="engineering"):
            raise ValueError("Three completed engineering cases need an independent audit")
        for name,digest in engineering_audit["input_artifact_sha256"].items():
            if sha(engineering/name)!=digest:
                raise ValueError("Engineering evidence changed before training selection")
        engineering_rows=[json.loads(line) for line in (engineering/"episodes.jsonl").read_text(encoding="utf8").splitlines()]
        if len(engineering_rows)!=3:
            raise ValueError("Exactly three engineering cases are required")
        for row in engineering_rows:
            gate=row["refine_assessment"]
            if not (gate["valid_evidence"] and gate["full_home_within_deadline"]
                    and gate["spill_passed"] and gate["no_late_dose_change"]):
                raise ValueError("An engineering safety/time gate failed; no promotion")
        if len(rows)!=6 or list(groups)!=["angle_84","angle_84.1","angle_84.2"]:
            raise ValueError("Complete six-case training grid required")
        for role,policy in (("baseline",{"enable_correction":False,"approach_tilt_deg":84.}),
                            ("candidate",groups[selected][0]["declared_policy"])):
            write(root/(role+".json"),{"kind":"refine20_settled_angle_v1",
                "created_at_utc":now(),"task_kind":task_score.TASK_KIND,"role":role,
                "policy":policy,"selected_group":selected if role=="candidate" else "coarse_only_reference",
                "selection_scores":scores,"training_protocol_id":protocol["protocol_id"],
                "training_journal_sha256":sha(stage/"episodes.jsonl"),
                "training_audit_sha256":sha(stage/"audit.json"),
                "engineering_audit_sha256":sha(engineering/"audit.json"),
                "source_sha256":protocol["source_sha256"],"selection_tool_sha256":sha(__file__),
                "is_neural_policy":False,"default_gui_deployment":False})
        next_stage,seeds="validation",plan["validation_seeds"]
    else:
        if len(rows)!=4 or list(groups)!=["baseline","candidate"]:
            raise ValueError("Complete four-case validation comparison required")
        checkpoint=root/(selected+".json")
        write(root/"recommended.json",{"kind":"refine20_validation_recommendation_v1",
            "created_at_utc":now(),"task_kind":task_score.TASK_KIND,"recommended_role":selected,
            "policy":read(checkpoint)["policy"],"checkpoint_sha256":sha(checkpoint),
            "validation_protocol_id":protocol["protocol_id"],
            "validation_journal_sha256":sha(stage/"episodes.jsonl"),
            "validation_audit_sha256":sha(stage/"audit.json"),"selection_scores":scores,
            "selection_tool_sha256":sha(__file__),"test_used_for_selection":False})
        next_stage,seeds="test",plan["test_seeds"]
    policies={role:read(root/(role+".json"))["policy"] for role in ("baseline","candidate")}
    cases=[{"case_id":f"{next_stage}_{role}_s{seed}","target_ml":20.,"seed":seed,
            "split":next_stage,"role":role,"policy":policies[role]}
           for role in ("baseline","candidate") for seed in seeds]
    matrix={"draft":False,"stage":next_stage,"parent_plan_sha256":sha(root/"experiment_plan.json"),
        "profile_sha256":{role:sha(root/(role+".json")) for role in ("baseline","candidate")},
        "selection_tool_sha256":sha(__file__),"cases":cases}
    if next_stage=="test": matrix["recommendation_sha256"]=sha(root/"recommended.json")
    write(root/(next_stage+"_matrix.json"),matrix)
    print(json.dumps({"selected":selected,"scores":scores,"next_stage":next_stage},indent=2))


if __name__=="__main__": main()
