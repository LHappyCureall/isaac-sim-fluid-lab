"""Freeze the predeclared fixed retry policy after the known-case liquid gate.

This does not fit or select a parameter and never reads fresh test outcomes.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_text(encoding="utf8"))
def write(path,value):
    with path.open("x",encoding="utf8") as stream:
        json.dump(value,stream,indent=2,allow_nan=False)
        stream.write("\n")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment",type=Path,required=True)
    args=parser.parse_args()
    root=args.experiment.resolve()
    if (root/"test").exists():
        raise ValueError("Strategy must freeze before any fresh test initialization")
    plan=read(root/"experiment_plan.json")
    pilot=root/"pilot"
    audit=read(pilot/"audit.json")
    protocol=read(pilot/"protocol.json")
    if not (audit.get("complete") and audit.get("pass") and audit.get("stage")=="pilot"
            and audit.get("pilot_gate_passed") is True):
        raise ValueError("Independent known-case liquid/second-cycle gate did not pass")
    if audit["auditor_source_sha256"]!=sha(HERE/"audit_retry_run.py"):
        raise ValueError("Audited pilot implementation changed")
    for name,digest in audit["input_artifact_sha256"].items():
        if sha(pilot/name)!=digest:
            raise ValueError("Pilot evidence changed: "+name)
    if protocol["matrix"]["parent_plan_sha256"]!=sha(root/"experiment_plan.json"):
        raise ValueError("Predeclared policy/coverage changed")
    for category in ("source_sha256","input_sha256","gui_sha256"):
        for name,digest in protocol[category].items():
            if sha(ROOT/name)!=digest:
                raise ValueError("Source/input/old GUI changed before freeze: "+name)
    for name,digest in plan["pilot_reference_sha256"].items():
        if sha(ROOT/name)!=digest:
            raise ValueError("The previous known failure was rewritten")
    rows=[json.loads(line) for line in (pilot/"episodes.jsonl").read_text(encoding="utf8").splitlines()]
    if len(rows)!=1 or not rows[0]["refine_assessment"]["feasible"]:
        raise ValueError("Pilot must really satisfy dose/time/spill/home/late requirements")
    starts=[e for e in rows[0]["response_events"] if e["type"]=="response_cycle_start"]
    if len(starts)!=2 or [e["approach_tilt_deg"] for e in starts]!=[84.,84.1]:
        raise ValueError("Known case did not actually execute the two declared cycles")
    params=rows[0]["result"]["policy_params"]
    names=("prepare_fraction","dose_tilt_deg_s","return_fraction","tail_lead_s","stop_bias_ml",
           "max_dose_u_accel","approach_tilt_deg_s","action_delta_deg","action_tilt_deg_s",
           "hold_seconds","backoff_delta_deg","backoff_tilt_deg_s","anchor_return_tilt_deg_s",
           "quiet_seconds","maximum_measurement_hold_s")
    fixed={key:params[key] for key in names}
    fixed.update(first_approach_tilt_deg=84.,retry_approach_tilt_deg=84.1,
                 maximum_cycles_baseline=1,maximum_cycles_candidate=2,
                 target_ml=20.,absolute_tolerance_ml=.5,deadline_s=24.,peak_all_label_spill_limit=1)
    strategy={"kind":"fixed20_conditional_second_cycle_strategy_v1",
        "task_kind":"fixed20ml_absolute05ml_24s_onegrain_v1",
        "created_at_utc":datetime.now(timezone.utc).isoformat(),
        "experiment_plan_sha256":sha(root/"experiment_plan.json"),
        "profiles":plan["profiles"],"fixed_parameters":fixed,
        "pilot_protocol_id":protocol["protocol_id"],
        "pilot_journal_sha256":sha(pilot/"episodes.jsonl"),"pilot_audit_sha256":sha(pilot/"audit.json"),
        "pilot_uses_seen_previous_test_seed":True,"pilot_joint_task_passed":True,
        "no_data_fitting_or_model_selection":True,"test_used_for_selection":False,
        "default_gui_deployment":False,"freeze_helper_sha256":sha(__file__),
        **{key:protocol[key] for key in ("source_sha256","input_sha256","gui_sha256")}}
    write(root/"fixed_strategy.json",strategy)
    cases=[{"case_id":f"test_{role}_s{seed}","target_ml":20.,"seed":seed,"split":"test",
            "role":role,"policy":policy} for role,policy in plan["profiles"].items()
           for seed in plan["test_seeds"]]
    write(root/"test_matrix.json",{"draft":False,"stage":"test",
        "parent_plan_sha256":sha(root/"experiment_plan.json"),
        "fixed_strategy_sha256":sha(root/"fixed_strategy.json"),"cases":cases})
    print(json.dumps({"fixed_strategy_sha256":sha(root/"fixed_strategy.json"),
                      "fresh_test_cases":len(cases),"data_fitting":False},indent=2))


if __name__=="__main__": main()
