"""Independent CPU audit of the bounded twelve-case V13 coarse-prefix study.

Physics mathematics comes from the explicitly pinned response/V9 auditors.
Only experiment ownership and prefix-observation metadata are new here.  This
does not import a controller, Warp, Kit or a liquid solver.  Audit success is
not a claim of precise or arbitrary-goal pouring.
"""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from collections import Counter
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
HERE = "projects/pour_policy_adaptive_learning/"
PARENT_AUDIT = HERE + "audit_response_collection.py"
PARENT_AUDIT_SHA256 = "3fc83116776a204349dcf76242052e8f40749b485efa3513aebdf6cdaaadc7d9"
PREFIX_KIND = "adaptive_coarse_prefix_calibration_v13"
PREFIX_SCOPE = "record_integrity_and_independent_physical_motion_mass_prefix_audit_not_task_success"
CONTROLLER_SOURCE = HERE + "prefix_control.py"
COLLECTOR_SOURCE = HERE + "run_prefix_calibration.py"
DT, QUANTUM = 1 / 60, .019683
CYCLE_KINDS = (PREFIX_KIND,)
COARSE = {"prepare_fraction": 1., "dose_tilt_deg_s": 25.6, "return_fraction": 1.,
          "tail_lead_s": .26, "stop_bias_ml": 10., "max_dose_u_accel": 6.}
CONDITIONS = ((25., -.5), (25., -1.), (35., 0.), (35., .5), (40., 1.), (40., 1.5))
SEEDS = (193101, 193102)
ARTIFACTS = ("protocol.json", "summary.json", "job_state.json", "episodes.jsonl",
             "responses.jsonl", "matrix.json", "versions.json")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


path = ROOT / PARENT_AUDIT
if sha(path) != PARENT_AUDIT_SHA256:
    raise ValueError("Pinned independent response audit implementation changed")
spec = importlib.util.spec_from_file_location("prefix_pinned_response_auditor", path)
parent = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = parent
spec.loader.exec_module(parent)
AuditError, need, close, read = parent.AuditError, parent.need, parent.close, parent.read
load_cpu_reference, response_bindings = parent.load_cpu_reference, parent.response_bindings
audit_post = parent.audit_post


def required_runtime_sources():
    return tuple(HERE + name for name in (
        "response_control.py", "cycle_control.py", "prefix_control.py",
        "response_samples.py", "response_env.py", "cycle_env.py", "run_prefix_calibration.py"))


def validate_schedule(cases):
    need(isinstance(cases, list) and len(cases) == 12, "Exactly twelve declared prefix cases required")
    seen, conditions = set(), []
    for case in cases:
        need(isinstance(case, dict), "Prefix cases must be declared objects")
        identifier = case.get("case_id")
        need(isinstance(identifier, str) and identifier and identifier not in seen
             and all(c in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in identifier),
             "Prefix identities must be unique safe strings")
        seen.add(identifier)
        need(case.get("split") == "development" and type(case.get("seed")) is int,
             "Prefix calibration is development, never a held-out task test")
        policy = case.get("policy")
        need(isinstance(policy, dict) and set(policy) == {"control_target_offset_ml"},
             "Only the internal coarse target offset may vary in this bounded study")
        target, offset = case.get("target_ml"), policy["control_target_offset_ml"]
        need(type(target) in (int, float) and type(offset) in (int, float),
             "Prefix target and offset must be physical numbers, not Boolean")
        conditions.append((target, offset, case["seed"]))
    expected = Counter((target, offset, seed) for target, offset in CONDITIONS for seed in SEEDS)
    need(Counter(conditions) == expected,
         "Prefix schedule differs from declared25/35/40mL offsets and two fresh development seeds")


def validate_prefix_status(status, actual_target, offset, controller_sha):
    """Recompute the old coarse formula with only its local target changed."""
    for mapping in (status, status["policy_action"], status["policy_action"]["coarse_action"]):
        close(mapping["actual_target_ml"], actual_target, "Actual requested goal was changed")
        close(mapping["control_target_ml"], actual_target + offset, "Internal coarse target differs")
        close(mapping["prefix_target_offset_ml"], offset, "Prefix declared offset differs")
        close(mapping["control_target_offset_ml"], offset, "Prefix offset aliases differ")
    close(status["target_ml"], actual_target, "Final measurement goal was replaced")
    need(status["prefix_controller_source_sha256"] == controller_sha,
         "Actual prefix controller metadata source differs")
    for key, expected in COARSE.items():
        close(status["policy_params"][key], expected, "Original coarse parameter changed: " + key)
    need(status["response_events"] == [] and status["response_cycle_count"] == 0
         and status["response_model_used"] is False, "Prefix calibration executed a response/model cycle")
    action = status["policy_action"]["coarse_action"]
    predicted = status["received_ml"] + status["inflight_ml"] + max(0., status["flow_ml_s"]) * .26
    close(action["predicted_final_ml"], predicted, "Original coarse prediction arithmetic differs")
    close(action["remaining_committed_ml"], actual_target + offset - status["received_ml"] - status["inflight_ml"],
          "Internal target was not used only in the coarse decision observation")
    close(action["requested_tilt_deg_s"], 25.6, "Original coarse angular speed changed")
    need(action["stop"] is (predicted + 10. >= actual_target + offset),
         "Internal coarse stop decision does not match the original formula")


def cycle_bindings(row, steps, direct):
    need(row["response_events"] == [] and row["response_samples"] == [],
         "Prefix-only experiment invented response cycles or fitted samples")
    return None


def prefix_bindings(row, report, plan):
    target = row["target_ml"]
    offset = row["declared_policy"]["control_target_offset_ml"]
    controller_sha = plan["source_sha256"][CONTROLLER_SOURCE]
    recorded_anchor = None
    for detail in report["raw_direct"]:
        status = detail["controller_status"]
        validate_prefix_status(status, target, offset, controller_sha)
        anchor = status.get("prefix_anchor_observation")
        if anchor is not None:
            if recorded_anchor is None:
                close(anchor["physical_time_s"], detail["physical_time_s"],
                      "Prefix anchor was not recorded at its actual observation frame", 1e-8)
            else:
                need(anchor == recorded_anchor, "Retained actual prefix anchor was rewritten")
            recorded_anchor = anchor
        else:
            need(recorded_anchor is None, "Retained prefix anchor disappeared")
    observation = row["status"].get("prefix_anchor_observation")
    if row["result"]["complete"]:
        need(isinstance(observation, dict), "Completed prefix lacks its actual quiet78-degree anchor")
    if observation is None:
        return None
    need(observation == recorded_anchor, "Final prefix anchor differs from its retained raw observation")
    moment = observation["physical_time_s"]
    frames = {round(item["physical_time_s"] / DT): (step, item)
              for step, item in zip(report["raw_steps"], report["raw_direct"])}
    frame = int(round(moment / DT))
    need(frame in frames, "Prefix anchor is not a recorded actual frame")
    step, detail = frames[frame]
    close(step["physical_time_s"], moment, "Actual prefix anchor time", 1e-8)
    parent.observe_endpoint(observation, step, detail, "prefix_anchor")
    close(observation["tilt_deg"], 78., "Actual prefix anchor tilt", 1e-6)
    need(observation["settled_measurement"] is True and observation["source_airborne_particles"] == 0
         and observation["quiet_measured_seconds"] >= .25 - 1e-9,
         "Prefix anchor was not an actual stationary dose-quiet measurement")
    return {"physical_time_s": moment, "received_ml": observation["received_ml"],
            "remaining_ml": observation["tube_remaining_ml"],
            "actual_target_ml": target, "control_target_ml": target + offset,
            "control_target_offset_ml": offset}


# Copied physical audit from the pinned3fc implementation. Only ownership
# literals and returned raw endpoint references differ; numerical gates remain exact.
def audit_episode(run, row, plan, geometry, ref):
    name, result = row["episode_id"], row["result"]
    ref.finite(row)
    need(ref.read(ref.inside(run, "completed/" + name + ".json")) == row, "Sidecar/journal differs")
    cases = {case["case_id"]: case for case in plan["matrix"]["cases"]}
    case = cases[name]
    need(row["target_ml"] == case["target_ml"] == result["target_ml"]
         and row["seed"] == case["seed"] == result["seed"] and row["split"] == case["split"]
         and row["declared_policy"] == case["policy"], "Scheduled conditions differ")
    action_name = "traces/" + name + ".steps.jsonl.gz"
    post_name = "traces/" + name + ".post.jsonl.gz"
    need(action_name in row["artifacts"] and set(row["artifacts"]) <= {action_name, post_name},
         "Unexpected raw artifact set")
    for relative, digest in row["artifacts"].items():
        need(sha(ref.inside(run, relative)) == digest, "Artifact bytes changed: " + relative)
    action, _ = ref.gz(ref.inside(run, action_name))
    header = action[0]
    need(header["type"] == "episode" and header["fluid_mode"] == "live_PBF_action_conditioned",
         "Original live physics header missing")
    need(header["randomized"] is True and result["randomized"] is True, "Randomized physical prefix missing")
    for key in ("target_ml", "seed", "policy_name", "policy_params", "randomized"):
        ref.same(header[key], result[key], "Original header/footer " + key)
    close(header["physics_dt_s"], DT, "Original physical dt")
    close(header["particle_spacing_m"], .0027, "Unchanged spacing")
    close(header["particle_volume_ml"], QUANTUM, "Unchanged particle quantum")
    close(header["density_kg_m3"], 1000., "Unchanged density")
    need(header["particle_count"] == result["particle_count"] == 11625
         and result["source_particle_count"] == 2880, "Fixed identities changed")
    close(result["initial_source_ml"], 56.68704, "Source mass volume")
    close(result["physical_time_s"], result["physical_steps"] * DT, "Action clock step/60", 1e-8)
    need(result["physical_steps"] <= 1440 and result["minimum_substeps"] >= 4,
         "Deadline or adaptive minimum changed")
    close(result["error_ml"], result["received_ml"] - result["target_ml"], "Signed actual dose error")
    effective = [item for item in action if item["type"] == "effective_direct_sources"]
    need(len(effective) == 1, "One effective physical-source header required")
    effective = effective[0]
    expected_controller = CONTROLLER_SOURCE
    expected_collector = COLLECTOR_SOURCE
    if plan["kind"] in CYCLE_KINDS:
        need(plan["controller_source"] == expected_controller
             and plan["collector_source"] == expected_collector, "Declared V12 source ownership differs")
    need(effective["actual_controller_file"] == expected_controller
         and effective["actual_controller_sha256"] == plan["source_sha256"][expected_controller],
         "Actual controller differs from protocol")
    expected_bindings = {"mouth_pivot_lift_geometry.py": plan["source_sha256"][
        "assets/robot_testtube_pour/scripts/mouth_pivot_lift_geometry.py"],
        Path(expected_collector).name: plan["source_sha256"][expected_collector]}
    need(effective["source_sha256"] == expected_bindings, "Actual geometry/factory source binding differs")
    selected_cache = "projects/pour_policy_training_v3/mouth_pivot_lift025_z022_cache.npz"
    need(effective["cache_sha256"] == plan["input_sha256"][selected_cache], "Actual path cache binding differs")
    need(effective["cuda_graph"] is False and effective["physics_solver_changed"] is False
         and effective["tube_containment"] == "actual_solver_body_q_full_R"
         and effective["deadline_physical_seconds"] == 24.
         and effective["relative_volume_tolerance"] == .001
         and effective["maximum_all_label_spill_particles"] == 1, "Actual task/physics contract changed")
    steps = [item for item in action if item["type"] == "step"]
    direct = [item for item in action if item["type"] == "direct_step_evidence"]
    spill_observations = [item for item in action if item["type"] == "cycle_spill_observation"]
    footers = [item for item in action if item["type"] == "result"]
    failures = [item for item in action if item["type"] == "failure"]
    need(len(footers) == 1 and action[-1] is footers[0]
         and {k: v for k, v in footers[0].items() if k != "type"} == result,
         "Original result footer changed")
    need(len(steps) == len(direct) and steps, "Every real step requires raw state and direct evidence")
    cycle_mode = plan["kind"] in CYCLE_KINDS
    if cycle_mode:
        need(len(spill_observations) == len(steps), "Every cycle frame needs same-frame all-label spill feedback")
    motion = ref.Motion(geometry)
    peak = below = 0
    drift = 0.
    initial_mass = result["actual_initial_total_mass_kg"]
    for index, (step, detail) in enumerate(zip(steps, direct)):
        need(step["step"] == detail["step"] == index and step["solver_substeps"] >= 4,
             "Real step/substep sequence differs")
        close(step["physical_time_s"], index * DT, "60Hz actual frame time", 1e-8)
        close(detail["physical_time_s"], step["physical_time_s"], "Paired physical evidence time", 1e-8)
        ref.partitions(detail["partitions"], step["truth"])
        peak = max(peak, detail["partitions"]["spilled"])
        below = max(below, detail["partitions"]["below_table"])
        need(detail["maximum_all_label_spilled_particles"] == peak, "All-label peak tally differs")
        if cycle_mode:
            observed = spill_observations[index]
            need(observed["step"] == index and observed["same_frame_read_only_observation"] is True,
                 "Same-frame cycle observer sequence differs")
            close(observed["physical_time_s"], step["physical_time_s"], "Cycle spill observation clock", 1e-8)
            need(observed["current_all_label_spilled_particles"] == detail["partitions"]["spilled"]
                 and observed["current_below_table_particles"] == detail["partitions"]["below_table"]
                 and observed["maximum_all_label_spilled_particles"] == peak,
                 "Cycle spill feedback differs from current actual full-R membership")
            need(observed["observer_source_sha256"] == plan["source_sha256"][HERE + "cycle_env.py"],
                 "Actual cycle observer source differs")
            need(detail["controller_status"]["response_cycle_observed_peak_spill_particles"] == peak,
                 "Cycle choice used missing or delayed all-label peak spill feedback")
        drift = max(drift, abs(step["truth"]["mass_kg"] - initial_mass))
        motion.row(detail["physical_time_s"], step["source_time"], detail["kinematics"], detail["controller_status"])
    action_rows = motion.rows
    for failure in failures:
        need(not result["valid_physics"], "Partial physics action relabelled valid")
        motion.dense(failure.get("last_motion_evidence", {}).get("dense_240_hz_samples", []), terminal=True)
    gap = result["physical_steps"] - steps[-1]["step"]
    need(gap in (0, 1) and (gap == 0 or failures), "Unexplained terminal frame gap")
    if gap == 0:
        close(result["received_ml"], steps[-1]["truth"]["received_ml"], "Footer actual received amount")
    if result["complete"]:
        need(row["status"] == direct[-1]["controller_status"], "Action endpoint status differs")
        need(len(direct) >= 13 and all(item["controller_status"]["home_static"] for item in direct[-13:]),
             "Full physical home and12quiet interval missing")
    canonical = response_bindings(row, steps, direct)
    cycles = (cycle_bindings(row, steps, direct)
              if plan["kind"] in CYCLE_KINDS else None)
    return {"episode_id": name, "action_rows": action_rows, "unlogged_terminal_action_frames": gap,
        "failures": failures, "canonical_samples": canonical, "motion": motion,
        "peak": peak, "below": below, "drift": drift, "initial_mass": initial_mass,
        "effective": effective, "post_name": post_name, "cycles": cycles, "cycle_mode": cycle_mode,
        "raw_steps": steps, "raw_direct": direct}


def audit(run, root=ROOT):
    run, root = Path(run).resolve(), Path(root).resolve()
    summary, job = read(run / "summary.json"), read(run / "job_state.json")
    need(summary.get("complete") is True and job.get("state") == "complete"
         and job.get("pid") is None and not (run / "worker.lock").exists(),
         "Complete immutable prefix experiment required before journal inspection")
    plan = read(run / "protocol.json")
    need(plan.get("kind") == PREFIX_KIND, "Unknown prefix experiment kind")
    need(plan.get("controller_source") == CONTROLLER_SOURCE
         and plan.get("collector_source") == COLLECTOR_SOURCE,
         "Declared prefix controller/factory ownership differs")
    ref = load_cpu_reference(root)
    protocol_id = plan["protocol_id"]
    need(ref.signature({k:v for k,v in plan.items() if k != "protocol_id"}) == protocol_id
         and summary["protocol_id"] == protocol_id, "Canonical prefix protocol identity differs")
    matrix = plan["matrix"]
    need(matrix["draft"] is False and read(run / "matrix.json") == matrix,
         "Finalized prefix matrix snapshot differs")
    validate_schedule(matrix["cases"])
    criteria = {"deadline_s":24., "relative_tolerance":.001, "peak_all_label_spill_limit":1,
        "spacing_m":.0027, "particle_ml":.0027 ** 3 * 1e6, "total_particles":11625,
        "source_particles":2880, "initial_source_ml":56.68704, "post_seconds":2.}
    for key, value in criteria.items():
        close(plan[key], value, "Fixed prefix protocol criterion " + key)
    need(plan["solver"] == "unchanged_live_PBF_60Hz_pressure6_min4_graph_off"
         and plan["collection_task_success_not_required"] is True
         and plan["phase_training_time_not_hidden"] is True and plan["no_saved_liquid_replay"] is True,
         "Prefix physics/time/claim contract differs")
    for category, folder in (("source_sha256","source_snapshot"),("input_sha256","input_snapshot")):
        need(isinstance(plan[category],dict) and plan[category], "Missing prefix source/input freeze")
        for name,digest in plan[category].items():
            need(sha(ref.inside(run / folder,name)) == digest, "Prefix snapshot changed: " + name)
            need(sha(ref.inside(root,name)) == digest, "Prefix current source/input changed: " + name)
    for name,digest in plan["gui_sha256"].items():
        need(sha(ref.inside(root,name)) == digest, "Old GUI checkpoint changed: " + name)
    need(set(required_runtime_sources()) <= set(plan["source_sha256"]),
         "Prefix runtime owner is absent from its source freeze")
    rows = [json.loads(line,parse_constant=ref.reject_constant)
            for line in (run / "episodes.jsonl").read_text(encoding="utf8").splitlines() if line.strip()]
    ids = [case["case_id"] for case in matrix["cases"]]
    need([row["episode_id"] for row in rows] == ids and len(rows) == summary["episodes"] == 12,
         "Prefix journal does not contain all twelve unique scheduled cases")
    need({path.stem for path in (run / "completed").glob("*.json")} == set(ids),
         "Unjournaled or missing prefix completed sidecar")
    for journal in (root / "droplet_out").glob("pour_policy*/*/episodes.jsonl"):
        if journal.resolve() == (run / "episodes.jsonl").resolve():
            continue
        for line in journal.read_text(encoding="utf8").splitlines():
            if not line.strip():
                continue
            previous = json.loads(line,parse_constant=ref.reject_constant)
            seed = previous.get("seed",previous.get("result",{}).get("seed"))
            need(seed not in SEEDS, "Prefix development seed occurs in another physical experiment")
    expected_physics = hashlib.sha256(json.dumps({"sources":plan["source_sha256"],
        "inputs":plan["input_sha256"], "spacing":.0027},sort_keys=True).encode()).hexdigest()
    geometry = ref.Geometry(run,root,plan)
    reports = []
    for row in rows:
        need(row["protocol_id"] == protocol_id and row["physics_fingerprint"] == expected_physics,
             "Prefix episode source/input identity differs")
        need(row["response_events"] == [] and row["response_samples"] == [],
             "Prefix calibration invented response actions or model training data")
        report = audit_episode(run,row,plan,geometry,ref)
        anchor = prefix_bindings(row,report,plan)
        # The pinned post auditor preserves its exact q240/FK/mass/full-R and
        # all-label spill checks. Prefix metadata is additionally checked on
        # each of those post frames; no post time can rescue the24s deadline.
        audited = audit_post(run,row,report,ref)
        for item in ref.gz(ref.inside(run,report["post_name"]))[0] if report["post_name"] in row["artifacts"] else []:
            validate_prefix_status(item["controller_status"],row["target_ml"],
                row["declared_policy"]["control_target_offset_ml"],plan["source_sha256"][CONTROLLER_SOURCE])
            need(item["controller_status"].get("prefix_anchor_observation") == row["status"].get("prefix_anchor_observation"),
                 "Actual prefix anchor changed during post observation")
        audited.update(control_target_offset_ml=row["declared_policy"]["control_target_offset_ml"],
                       control_target_ml=row["target_ml"]+row["declared_policy"]["control_target_offset_ml"],
                       actual_prefix_anchor=anchor)
        reports.append(audited)
    need((run / "responses.jsonl").read_text(encoding="utf8").strip() == ""
         and summary["response_samples"] == 0
         and summary["responses_sha256"] == sha(run / "responses.jsonl"),
         "Prefix-only run should contain no response-cycle fitting labels")
    passes = sum(report["strict_assessment"]["feasible"] for report in reports)
    need(summary["strict_task_passes"] == passes and summary["new_goal_policy_trained"] is False
         and summary["gui_deployment"] is False, "Prefix summary fabricated goal-policy acceptance")
    return {"complete":True,"pass":True,"kind":PREFIX_KIND,"scope":PREFIX_SCOPE,
        "created_at_utc":datetime.now(timezone.utc).isoformat(),"protocol_id":protocol_id,
        "episodes_audited":12,"response_samples":0,"actual_measurement_target_unchanged":True,
        "only_internal_coarse_observation_target_offset_changed":True,"all_six_coarse_parameters_fixed":True,
        "no_response_or_model_cycles":True,"new_goal_policy_trained":False,"gui_deployment":False,
        "strict_task_passes":passes,"all_strict_task_cases_passed":passes == 12,
        "input_artifact_sha256":{name:sha(run / name) for name in ARTIFACTS},
        "audit_tool_sha256":sha(__file__),"pinned_parent_audit_sha256":PARENT_AUDIT_SHA256,
        "independent_geometry_audit_sha256":parent.OLD_AUDIT_SHA,"episode_reports":reports}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",type=Path,required=True)
    parser.add_argument("--workspace",type=Path,default=ROOT)
    parser.add_argument("--output",type=Path)
    args=parser.parse_args()
    result=audit(args.run,args.workspace)
    if args.output:
        with args.output.open("x",encoding="utf8") as stream:
            json.dump(result,stream,indent=2,allow_nan=False)
            stream.write("\n")
    print(json.dumps({key:value for key,value in result.items() if key != "episode_reports"},indent=2))


if __name__ == "__main__":
    main()
