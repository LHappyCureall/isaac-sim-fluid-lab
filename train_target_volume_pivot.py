"""Independent V4 fixed-mouth CEM: 48 train + 12 validation + 18 test.

--dry-run is CPU-only and writes no frozen experiment. Actual start requires
the matching three-target, <=5-second, zero-spill engineering pilot first.
Raw DirectTrainingEnv footers remain unchanged; action, 5s dose and post-to-7s
constraints are recorded separately. Nothing is deployed to the GUI.
"""
import argparse
import copy
import ctypes
from datetime import datetime
import gzip
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path

from lab_paths import require_isaac_sim_root
import platform
import random
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parent
LOADED_ENTRY_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
PREFIX = "assets/robot_testtube_pour/scripts/"
SCRIPTS = ROOT / PREFIX
TRAIN_SEEDS, VALIDATION_SEEDS, TEST_SEEDS = (81101, 81102), (82101, 82102), (91101, 91102, 91103)
TARGETS = (20, 30, 40)
GENERATIONS, POPULATION, ELITES, CEM_SEED = 2, 4, 2, 2026112204
BUDGET = {"generations": 2, "population": 4, "elites": 2, "training": 48, "validation": 12, "test": 18, "total": 78}
ALGORITHM = "feasibility_first_CEM_upright_lift_fixed_mouth_feedback_v4"
POLICY_KIND = "upright_lift_fixed_mouth_planned_return_v2"
GEOMETRY_FILE, CONTROL_FILE, COLLECTOR_FILE = (PREFIX+name for name in (
    "mouth_pivot_lift_geometry.py", "mouth_pivot_lift_control_v2.py", "direct_training_env.py"))
MODULE_PINS = {
    GEOMETRY_FILE: "6f5ee0f694958ede3f111b1d27134c6571c4d7ef0ce5dae96425c4829638bfa9",
    CONTROL_FILE: "6b17db742e812777d562e6cd15f2c53ee5009fcf75c60052b702d4e5707c95ca",
    PREFIX+"mouth_pivot_fast_geometry.py": "44b11468f1ecfbad56127a1747d7c54848b0b56475c0c3ea0c27b6d20eaaaffb",
}
CACHE_PINS = {
    "path_cache": ("projects/pour_policy_training_v3/mouth_pivot_lift025_z022_cache.npz", "106706110581ef85ac87137aafaf5f513491b37f9f6ea16c2f427f7685c06a1f"),
    "base_path_cache": ("projects/pour_policy_training_v3/mouth_pivot_z022_cache.npz", "f0dec80ef890b2ef8946627af98be8d736e2a3d684e7f8f7bf2adb120aa46d40"),
    "base_import_cache": ("projects/pour_policy_training_v3/mouth_pivot_z026_cache.npz", "86ebd4f519f9341fc39c21b02fea47fb9cebe6117d367da68ef6729391dc1bfd"),
    "ancestral_reference_joint_cache": ("projects/pour_policy_training_v3/fast_path_smooth_cache.npz", "9d4e24e28e93d21320856ed7ae67282b32d8ce62587d3e46131b2d9a8bcbe7ae"),
}
PARAMETER_UNITS = {"prepare_fraction": "nominal embedded C2 clock seconds / physical second",
    "dose_tilt_deg_s": "degrees / physical second", "return_fraction": "nominal embedded C2 clock seconds / physical second",
    "tail_lead_s": "physical seconds", "stop_bias_ml": "mL",
    "max_dose_u_accel": "embedded C2 clock seconds / physical second squared; not a joint acceleration rating"}
REFERENCE = ROOT / "train_target_volume_fast.py"
if hashlib.sha256(REFERENCE.read_bytes()).hexdigest() != "2e742a7b46a051bc24a6f83ff42779ddd3fdfd6262f5adfd01ee0ede2d69a398":
    raise RuntimeError("Archived V4 study requires the exact private V3 helper/protocol/cache lineage. Public portable copies and rebuilt caches are different versions; do not rebind pins or resume original logs. See docs/PIVOT_V4_20261122.md.")
_spec = importlib.util.spec_from_file_location("pivot_frozen_pure_helpers", REFERENCE)
ref = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = ref
exec(compile(REFERENCE.read_text(encoding="utf-8"),str(REFERENCE),"exec"),ref.__dict__)
now, sha, signature, normalized, read_json = ref.now, ref.sha, ref.signature, ref.normalized, ref.read_json
write_json, append_json, write_csv, finite, numeric = ref.write_json, ref.append_json, ref.write_csv, ref.finite, ref.numeric
group_metrics, selection_key, release_passed = ref.group_metrics, ref.selection_key, ref.release_passed
DT, DEADLINE, POST_SECONDS, QUANTUM = 1/60, 5., 2., .0035**3*1e6


def need(condition, message):
    if not condition:
        raise ValueError(message)


def process_alive(pid):
    """Read-only exit check; inability to verify is never permission to start."""
    if pid is None:return False
    need(type(pid) is int and pid>0,"Invalid owner PID.")
    if os.name!="nt":
        try:os.kill(pid,0);return True
        except ProcessLookupError:return False
    kernel=ctypes.WinDLL("kernel32",use_last_error=True)
    kernel.OpenProcess.argtypes=[ctypes.c_uint32,ctypes.c_int,ctypes.c_uint32];kernel.OpenProcess.restype=ctypes.c_void_p
    kernel.GetExitCodeProcess.argtypes=[ctypes.c_void_p,ctypes.POINTER(ctypes.c_uint32)]
    kernel.CloseHandle.argtypes=[ctypes.c_void_p]
    handle=kernel.OpenProcess(0x1000,0,pid)
    if not handle:
        need(ctypes.get_last_error()==87,"Cannot verify the previous owner exited; refusing another worker.")
        return False
    try:
        code=ctypes.c_uint32()
        need(kernel.GetExitCodeProcess(handle,ctypes.byref(code)),"Cannot read previous owner exit code.")
        return code.value==259
    finally:kernel.CloseHandle(handle)


def acquire_worker(output,resume):
    output.mkdir(parents=True,exist_ok=True);lock=output/"worker.lock"
    if lock.exists():
        old=read_json(lock)
        need(not process_alive(old.get("worker_pid")),"This output already has a live worker; no process was stopped.")
        need(resume,"A stale worker lock requires explicit same-protocol --resume.")
        destination=output/"interrupted_artifacts"/("worker.lock."+str(time.time_ns())+"."+sha(lock)[:12]+".json")
        destination.parent.mkdir(parents=True,exist_ok=True);lock.replace(destination)
    token=str(time.time_ns())
    with lock.open("x",encoding="utf-8") as stream:
        json.dump({"worker_pid":os.getpid(),"started_at_utc":now(),"token":token,"entry_sha256":LOADED_ENTRY_SHA256},stream)
    return lock,token


def load(name, path):
    before = sha(path)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    exec(compile(Path(path).read_text(encoding="utf-8"), str(path), "exec"), module.__dict__)
    need(sha(path) == before, "Source changed during import: " + str(path))
    module._runner_loaded_sha256 = before
    return module


def gui_hashes():
    return {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted((ROOT / "assets/robot_testtube_pour/policy").glob("*.json"))}


def protected_legacy():
    path = ROOT / "droplet_out/pour_policy_fast_v3/run_20261122_cem5s/protocol.json"
    document = read_json(path)
    sources = document["protocol"]["source_sha256"]
    need(len(sources) == 20, "Expected the complete frozen V3/legacy source guard.")
    for relative, expected in sources.items():
        need(sha(ROOT/relative) == expected, "Protected old V3/V2 source changed: " + relative)
    return sources, {"path": str(path), "sha256": sha(path)}


def source_manifest():
    old, _ = protected_legacy()
    additions = ("train_target_volume_pivot.py", "tools/training/run_pivot_job.py",
        GEOMETRY_FILE, CONTROL_FILE, COLLECTOR_FILE, PREFIX+"mouth_pivot_fast_geometry.py",
        PREFIX+"direct_fast_geometry.py", PREFIX+"direct_fast_control.py",
        "tools/training/prepare_mouth_pivot_path.py",
        "tools/training/prepare_mouth_pivot_lift_path.py")
    result = {**old, **{name: sha(ROOT/name) for name in additions}}
    for name, expected in MODULE_PINS.items():
        need(result[name] == expected, "Selected V4 module is not its actual frozen version: " + name)
    return result


def sdk_root():
    sdk = require_isaac_sim_root().resolve()
    need((sdk/"kit").is_dir() and (sdk/"exts").is_dir(), "ISAAC_SIM_ROOT must be an installed SDK.")
    return sdk


def cpu_runtime():
    # Import only NumPy, pure geometry/policy and the lazy environment class.
    # shared.configure_sdk / Warp initialization is deferred until all gates pass.
    sdk = sdk_root()
    for p in [*sorted((sdk/"extscache").glob("omni.kit.pip_archive-*/pip_prebundle")),
              sdk/"exts/omni.pip.compute/pip_prebundle", SCRIPTS]:
        if p.is_dir() and str(p) not in sys.path:
            sys.path.insert(0, str(p))
    import numpy as np
    geometry_module = load("pivot_explicit_lift_geometry", ROOT/GEOMETRY_FILE)
    path, expected = CACHE_PINS["path_cache"]
    geometry = geometry_module.load_geometry(ROOT/path, expected)
    control = load("pivot_explicit_planned_return_control", ROOT/CONTROL_FILE)
    collector = load("pivot_explicit_full_R_collector", ROOT/COLLECTOR_FILE)
    need(control.POLICY_KIND == POLICY_KIND and set(control.PARAMETER_NAMES) == set(PARAMETER_UNITS), "V4 six-parameter interface changed.")
    need(control.EXPECTED_GEOMETRY_SHA256 == MODULE_PINS[GEOMETRY_FILE]
         and control.EXPECTED_CACHE_SHA256 == expected and geometry.CACHE_SHA256 == expected,
         "Controller and actual geometry/cache are not explicitly bound.")
    points, labels = geometry.initial_water_points(.0035)
    need(np.asarray(points).shape == (5330,3) and int(np.count_nonzero(np.asarray(labels)==1)) == 1198,
         "The original 5330/1198 liquid initial condition changed.")
    return {"geometry": geometry, "control": control, "collector": collector, "numpy": np.__version__,
        "loaded_sha256":{GEOMETRY_FILE:geometry_module._runner_loaded_sha256,
            CONTROL_FILE:control._runner_loaded_sha256,COLLECTOR_FILE:collector._runner_loaded_sha256}}


def history(output):
    planned = set((*TRAIN_SEEDS, *VALIDATION_SEEDS, *TEST_SEEDS, CEM_SEED))
    found = {}
    for path in sorted((ROOT/"droplet_out").rglob("*protocol*.json")):
        if path.resolve().is_relative_to(output):
            continue
        value = read_json(path)
        overlap = planned & set(ref.declared_seeds(value))
        need(not overlap, "A proposed seed was already used in historical protocol " + str(path) + ": " + str(sorted(overlap)))
        found[path.relative_to(ROOT).as_posix()] = sha(path)
    return found


def trace_check(path, result, declaration):
    """Verify new direct trace identities, 60Hz clock, mass and dense evidence.

    Unsafe episodes are retained; integrity checks must not erase failed data.
    """
    steps, evidence, failures, effective, footer = [], [], [], None, None
    previous_q,previous_dq,last_step = None,[0.]*7,None
    max_position = max_spilled = max_below = 0
    last_partitions, max_logged_qacc, max_logged_speed_ratio, dense_rows = None, 0., 0., 0
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        header = json.loads(next(stream))
        need(header["type"] == "episode" and (header["target_ml"], header["seed"]) == (result["target_ml"], result["seed"]), "Action header conditions differ.")
        need(header["policy_name"] == declaration["policy_kind"] and header["fluid_mode"] == "live_PBF_action_conditioned"
             and abs(header["physics_dt_s"]-DT)<1e-12 and header["minimum_substeps"]==4
             and header["particle_count"]==5330 and abs(header["particle_volume_ml"]-QUANTUM)<1e-12,
             "The action uses a different policy, physics clock or particle basis.")
        need(header["policy_params"]==result["policy_params"] and header["policy_name"]==result["policy_name"]
             and header["randomized"]==result["randomized"],"Action header and unchanged footer policy/conditions differ.")
        for name, digest in header["source_sha256"].items():
            need(declaration["source_sha256"].get(PREFIX+name)==digest, "A legacy collector-header dependency changed.")
        for line in stream:
            row = json.loads(line); finite(row)
            need(footer is None, "Data appears after the unchanged action footer.")
            kind = row.get("type")
            if kind == "effective_direct_sources":
                need(effective is None, "Duplicate actual direct implementation record.")
                effective = row
            elif kind == "step":
                need(row["step"]==len(steps) and abs(row["physical_time_s"]-row["step"]*DT)<1e-12, "Action clock is not original step/60.")
                steps.append(row["step"])
                last_step=row
                need(abs(row["truth"]["mass_kg"]-result["actual_initial_total_mass_kg"])<=1e-12
                     and row["truth"]["source_partition_count"]==1198 and row["truth"]["target_ml"]==result["target_ml"],
                     "Logged mass or requested target/source identities changed.")
            elif kind == "direct_step_evidence":
                need(row["step"]==len(evidence) and abs(row["physical_time_s"]-row["step"]*DT)<1e-12, "Direct evidence omitted a physical frame.")
                part = row["partitions"]
                need(part["total"]==5330 and part["initial_tube"]==1198
                     and sum(part[k] for k in ("beaker","tube","airborne","spilled"))==5330
                     and sum(part[k] for k in ("tube_to_beaker","tube_remaining","tube_airborne","tube_spilled"))==1198,
                     "Actual full-R particle partitions lose identities.")
                max_spilled, max_below = max(max_spilled,part["spilled"]), max(max_below,part["below_table"])
                last_partitions = part
                need(last_step is not None and last_step["step"]==row["step"],"Direct evidence lacks its matching truth row.")
                for key,count_key in (("received_ml","tube_to_beaker"),("remaining_ml","tube_remaining"),
                                      ("inflight_ml","tube_airborne"),("spilled_ml","tube_spilled")):
                    need(abs(last_step["truth"][key]-part[count_key]*QUANTUM)<1e-10,"Full-R mass/count truth differs: "+key)
                need(last_step["truth"]["below_table_particles"]==part["below_table"],"Below-table evidence differs.")
                kin = row["kinematics"]
                q = kin["joint_position_rad"]
                need(len(q)==7 and len(kin["actual_tube_pose_xyzw"])==7, "Missing actual seven-joint/full-R collider evidence.")
                lower, upper = declaration["robot_envelope"]["joint_lower_limits_rad"], declaration["robot_envelope"]["joint_upper_limits_rad"]
                max_position = max(max_position, max(max(lo-v,v-hi,0.) for v,lo,hi in zip(q,lower,upper)))
                dense=kin.get("dense_240_hz_samples",[])
                need(len(dense)==(0 if row["step"]==0 else 4),"Every completed nonzero frame requires exactly four 240Hz joint samples.")
                if row["step"]==0:previous_q=list(q)
                for j,sample in enumerate(dense,1):
                    need(abs(sample["time_s"]-((row["step"]-1)*DT+j*DT/4))<1e-10,"Dense joint sample is not the actual 240Hz sub-frame time.")
                    need(all(len(sample[key])==7 for key in ("q_rad","dq_rad_s","ddq_rad_s2")), "Missing 240Hz joint evidence.")
                    dq=[(v-old)*240 for v,old in zip(sample["q_rad"],previous_q)]
                    ddq=[(v-old)*240 for v,old in zip(dq,previous_dq)]
                    need(max(abs(a-b) for a,b in zip(dq,sample["dq_rad_s"]))<1e-7
                         and max(abs(a-b) for a,b in zip(ddq,sample["ddq_rad_s2"]))<2e-5,
                         "Dense velocities/accelerations differ from actual joint finite differences.")
                    previous_q,previous_dq=list(sample["q_rad"]),list(sample["dq_rad_s"])
                    max_logged_qacc = max(max_logged_qacc,max(map(abs,sample["ddq_rad_s2"])))
                    max_logged_speed_ratio = max(max_logged_speed_ratio,sample["speed_limit_ratio"])
                    dense_rows += 1
                evidence.append(row["step"])
            elif kind == "failure": failures.append(row)
            elif kind == "result": footer = {k:v for k,v in row.items() if k!="type"}
            else: raise ValueError("Unrecognized V4 raw trace event: " + str(kind))
    need(footer==result and effective is not None and steps==evidence and bool(steps), "Action footer or evidence coverage differs.")
    expected = {Path(name).name:declaration["source_sha256"][name] for name in (GEOMETRY_FILE, CONTROL_FILE)}
    need(effective["source_sha256"]==expected and effective["cache_sha256"]==declaration["input_files"]["path_cache"]["sha256"]
         and effective["tube_containment"]=="full_rotation_matrix_not_a_pitch_projection"
         and effective["physics_solver_changed"] is False, "Actual selected geometry/controller/cache/full-R binding differs.")
    gap = result["physical_steps"]-steps[-1]
    need(gap in (0,1) and (gap==0 or bool(failures)) and (not failures or not result["valid_physics"]), "Unexplained terminal failure coverage gap.")
    if gap==0:
        need(abs(last_step["truth"]["received_ml"]-result["received_ml"])<1e-10
             and abs(last_step["truth"]["spilled_ml"]-result["spilled_ml"])<1e-10,"Final action footer is not the last actual full-R truth.")
    need(result["physical_steps"]<=300 and abs(result["physical_time_s"]-result["physical_steps"]*DT)<1e-12, "Action exceeded original five-second limit.")
    need(datetime.fromisoformat(effective["reset_started_utc"]).tzinfo is not None, "Missing pre-physics UTC identity.")
    return {"reset_started_utc":effective["reset_started_utc"], "raw_step_rows":len(steps), "dense_240hz_rows":dense_rows,
        "unlogged_terminal_failure_steps":gap, "failure_events":failures, "maximum_joint_position_violation_rad":max_position,
        "maximum_total_spilled_particles":max_spilled, "maximum_below_table_particles":max_below,
        "last_recorded_partitions":last_partitions, "maximum_logged_joint_acceleration_rad_s2":max_logged_qacc,
        "maximum_logged_joint_speed_limit_ratio":max_logged_speed_ratio, "footer_equals_original_returned_result":True}


def assess(result, status, motion, post, action_check, post_check, post_motion):
    def safe_motion(values):
        return all(numeric(values.get(key))<=limit for key,limit in {
            "maximum_joint_speed_limit_ratio":1.00001, "maximum_joint_acceleration_rad_s2":25.0001,
            "maximum_hand_fk_position_error_m":.003, "maximum_hand_fk_orientation_error_rad":.02,
            "maximum_actual_body_position_error_m":1e-5, "maximum_actual_body_orientation_error_rad":1e-4,
            "maximum_actual_grasp_position_error_m":.003}.items())
    physical_safe = bool(result["valid_physics"] and result["mass_conserved"] and result["all_states_finite"]
        and result["source_partition_conserved"] and result["max_spilled_ml"]==0 and result["max_below_table_particles"]==0
        and safe_motion(motion) and safe_motion(post_motion) and action_check["maximum_joint_position_violation_rad"]<=1e-6
        and action_check["maximum_total_spilled_particles"]==action_check["maximum_below_table_particles"]==0
        and (post_check is None or (post_check["maximum_total_spilled_particles"]==0
             and post_check["all_logged_states_finite"] and post_check["all_logged_mass_arrays_unchanged"]))
        and not (isinstance(post,dict) and (post.get("error") or post.get("exception"))))
    home = bool(status.get("home_static") is True and numeric(status.get("home_position_error_m"))<=.001
        and numeric(status.get("home_orientation_error_rad"))<=math.radians(.1) and numeric(status.get("home_joint_error_rad"))<=.01
        and abs(numeric(status.get("u")))<=1e-7 and status.get("source_airborne_particles")==0
        and numeric(status.get("stable_frames"),-1)>=12)
    on_time = bool(result["complete"] and result["physical_time_s"]<=5+1e-12 and result["physical_steps"]<=300
        and status.get("completed_by_deadline") is True and status.get("timed_out") is False and status.get("done") is True and home)
    dose = post.get("deadline_received_ml") if isinstance(post,dict) else None
    available = type(dose) in (int,float) and math.isfinite(dose)
    late = bool(isinstance(post,dict) and post.get("passed") is True and available
        and numeric(post.get("duration_s"),-1)>=2-1e-10 and numeric(post.get("end_physical_time_s"),-1)>=7-1e-10
        and abs(numeric(post.get("start_physical_time_s"))-result["physical_time_s"])<1e-10
        and post.get("maximum_received_change_ml")==0 and post.get("all_states_finite") is True and post.get("mass_array_unchanged") is True)
    error = dose-result["target_ml"] if available else result["error_ml"]
    return {"physical_safe":physical_safe, "home_stable":home, "completed_by_deadline":on_time, "late_drop_passed":late,
        "constraint_feasible":physical_safe and on_time and late, "dose_error_ml":error, "absolute_error_ml":abs(error),
        "deadline_dose_measurement_available":available,
        "dose_measurement_basis":"physical_5s" if available else "invalid_terminal_fallback_not_a_5s_measurement",
        "fallback_measurement_physical_time_s":result.get("measurement_physical_time_s") if not available else None,
        "action_physical_seconds":result["physical_time_s"],
        "success_within_0_1_ml":physical_safe and on_time and late and abs(error)<=.1,
        "action_result_was_not_modified":True, "action_footer_success_alone_is_not_V4_release_success":True}


def engineering_gate(report_path, runtime, sources):
    need(report_path is not None and Path(report_path).is_file(), "A matching complete three-target engineering report is required before training.")
    path = Path(report_path).resolve(); report, p = read_json(path), read_json(path.parent/"protocol.json")
    need(report.get("complete") is True and p.get("purpose")=="direct_path_development_pilot_not_heldout", "Engineering pilot is incomplete or has a different purpose.")
    need(p.get("effective_geometry_file")==GEOMETRY_FILE and p.get("effective_controller_file")==CONTROL_FILE
        and p.get("cache_sha256")==CACHE_PINS["path_cache"][1] and p.get("geometry_kind")=="C2_upright_lift_fixed_mouth_offline_IK"
        and p.get("physical_dt_s")==DT and p.get("minimum_substeps")==4 and p.get("pressure_iterations")==6
        and p.get("cuda_graph") is False and p.get("liquid_solver_modified") is False, "Engineering module/cache/physics identity differs.")
    baseline = normalized(runtime["control"].DirectPolicy(p["policy_parameters"]).params)
    for name,digest in p["sources"].items():
        need(sha(ROOT/name)==digest and sha(path.parent/"source_snapshot"/name)==digest, "Engineering source/snapshot changed: "+name)
    for name in (GEOMETRY_FILE,CONTROL_FILE,COLLECTOR_FILE,PREFIX+"liquid.py",PREFIX+"mouth_pivot_fast_geometry.py"):
        need(p["sources"].get(name)==sources[name], "Engineering does not bind the actual V4 transitive modules.")
    declaration = {"source_sha256":sources, "policy_kind":POLICY_KIND,
        "robot_envelope":{"joint_lower_limits_rad":normalized(runtime["geometry"].JOINT_LOWER_LIMITS_RAD),
                          "joint_upper_limits_rad":normalized(runtime["geometry"].JOINT_UPPER_LIMITS_RAD)},
        "input_files":{"path_cache":{"sha256":CACHE_PINS["path_cache"][1]}}}
    rows = report.get("results",[])
    need(len(rows)==3 and {r["result"]["target_ml"] for r in rows}==set(TARGETS), "Engineering must cover all three targets, not only 20mL.")
    files = {"engineering_results":path, "engineering_protocol":path.parent/"protocol.json"}
    cases = []
    for row in rows:
        result = row["result"]; target = int(result["target_ml"])
        need(result["policy_params"]==baseline and result["policy_name"]==POLICY_KIND, "Engineering baseline is not the selected actual policy.")
        action, post_path = path.parent/f"target_{target}.steps.jsonl.gz", path.parent/f"target_{target}.post.jsonl.gz"
        ac = trace_check(action,result,declaration)
        pc = ref.verify_post_trace(post_path,row["post_observation"],result)
        assessment = assess(result,row["controller"],row["motion_metrics"],row["post_observation"],ac,pc,row["post_motion_metrics"])
        need(assessment["constraint_feasible"], "Engineering target has not passed <=5s full home, zero total spill, mass/motion and post-to-7s gates: "+str(target))
        files["engineering_action_t"+str(target)],files["engineering_post_t"+str(target)] = action,post_path
        cases.append({"target_ml":target,"assessment":assessment,"action_sha256":sha(action),"post_sha256":sha(post_path)})
    return {"eligible":True,"baseline_params":baseline,"all_three_targets_passed_non_dose_gates":True,
            "dose_precision_required_for_engineering":False,"cases":cases}, files


class LexCEM:
    def __init__(self,names,bounds,initial,std_fraction):
        self.names,self.bounds,self.mean = tuple(names),bounds,dict(initial)
        self.std = {k:(bounds[k][1]-bounds[k][0])*std_fraction for k in self.names}
        self.rng = random.Random(CEM_SEED); self.best_params=self.best_key=None
    def ask(self):
        result=[dict(self.mean)]
        if self.best_params is not None and self.best_params!=self.mean: result.append(dict(self.best_params))
        while len(result)<POPULATION:
            result.append({k:min(self.bounds[k][1],max(self.bounds[k][0],self.rng.gauss(self.mean[k],self.std[k]))) for k in self.names})
        return result
    def tell(self,params,keys):
        need(len(params)==len(keys)==POPULATION,"CEM population differs.")
        order=sorted(range(POPULATION),key=lambda i:(tuple(keys[i]),i))
        for i in order:
            if self.best_key is None or tuple(keys[i])<self.best_key:self.best_key,self.best_params=tuple(keys[i]),dict(params[i])
        elites=[params[i] for i in order[:ELITES]]
        for k in self.names:
            mean=sum(p[k] for p in elites)/ELITES;std=math.sqrt(sum((p[k]-mean)**2 for p in elites)/ELITES)
            self.mean[k]=.65*self.mean[k]+.35*mean
            self.std[k]=max(.02*(self.bounds[k][1]-self.bounds[k][0]),.65*self.std[k]+.35*std)
        return {"elite_indices":order[:ELITES],"best_key":list(self.best_key),"best_params":dict(self.best_params),"next_mean":dict(self.mean),"next_std":dict(self.std)}


def declaration(args,runtime):
    sources=source_manifest(); historical=history(args.output.resolve())
    need(sources["train_target_volume_pivot.py"]==LOADED_ENTRY_SHA256,"Runner source changed after code load.")
    need(all(sources[name]==digest for name,digest in runtime["loaded_sha256"].items()),"A selected runtime module changed between actual import and freezing.")
    config=read_json(args.search_config) if args.search_config else {}
    need(not set(config)-{"draft","bounds","initial_params","initial_std_fraction","notes"},"Unknown V4 search configuration fields.")
    control=runtime["control"];bounds=normalized(config.get("bounds",control.PARAMETER_BOUNDS))
    need(set(bounds)==set(control.PARAMETER_NAMES),"V4 search bounds must specify all six parameters.")
    for name,(lo,hi) in bounds.items():
        outer=control.PARAMETER_BOUNDS[name]
        need(type(lo) in (int,float) and type(hi) in (int,float) and math.isfinite(lo) and math.isfinite(hi)
             and outer[0]<=lo<hi<=outer[1],"Draft/search bound outside the selected controller API: "+name)
    gate,engineering_files={"eligible":False,"reason":"matching completed three-target engineering report not supplied"},{}
    if args.engineering_report:
        try:gate,engineering_files=engineering_gate(args.engineering_report,runtime,sources)
        except (ValueError,OSError,KeyError) as exc:
            if not args.dry_run:raise
            gate={"eligible":False,"reason":str(exc)}
    baseline=gate.get("baseline_params")
    center=normalized(control.DirectPolicy(config.get("initial_params",baseline or control.DEFAULT_PARAMS)).params)
    need(all(bounds[k][0]<=center[k]<=bounds[k][1] for k in bounds),"CEM center lies outside its declared bounds.")
    std=float(config.get("initial_std_fraction",.2));need(math.isfinite(std) and 0<std<=.5,"Invalid CEM std fraction.")
    files={key:ROOT/path for key,(path,pin) in CACHE_PINS.items()}
    for key,(path,pin) in CACHE_PINS.items():need(sha(ROOT/path)==pin,"Transitive robot cache changed: "+key)
    _,old_protocol=protected_legacy();files["protected_legacy_protocol"]=Path(old_protocol["path"])
    files["urdf"]=sdk_root()/"exts/isaacsim.asset.importer.urdf/data/urdf/robots/franka_description/robots/panda_arm_hand.urdf"
    import numpy as np
    with np.load(files["ancestral_reference_joint_cache"],allow_pickle=False) as archive:
        need(str(archive["urdf_sha256"])==sha(files["urdf"]),"The actual SDK URDF differs from the pinned joint/FK ancestry.")
    need(str(runtime["geometry"]._CACHE["reference_cache_sha256"])==CACHE_PINS["ancestral_reference_joint_cache"][1]
         and str(runtime["geometry"]._CACHE["original_geometry_sha256"])==sources[PREFIX+"geometry.py"],
         "The selected lift cache does not bind the declared geometry/FK ancestry.")
    files.update(engineering_files)
    if args.search_config:files["search_config"]=args.search_config.resolve()
    historical_frozen=historical
    freeze_path=args.output/"input_freeze.json"
    if args.resume and freeze_path.is_file():
        historical_frozen=read_json(freeze_path)["declaration"]["historical_protocol_sha256"]
        need(all(sha(ROOT/name)==digest for name,digest in historical_frozen.items()),"A pinned historical protocol changed.")
    return {"schema_version":1,"algorithm":ALGORITHM,"policy_kind":POLICY_KIND,"neural_policy":False,"ppo":False,
        "targets_ml":list(TARGETS),"train_seeds":list(TRAIN_SEEDS),"validation_seeds":list(VALIDATION_SEEDS),"test_seeds":list(TEST_SEEDS),
        "budget":BUDGET,"cem":{"seed":CEM_SEED,"parameter_names":list(control.PARAMETER_NAMES),"bounds":bounds,
            "initial_params":center,"baseline_params":baseline,"initial_std_fraction":std,"update_fraction":.35,"std_floor_range_fraction":.02},
        "parameter_units":PARAMETER_UNITS,"engineering_gate":gate,"search_config_is_draft":config.get("draft",False),
        "module_binding":{"geometry_file":GEOMETRY_FILE,"controller_file":CONTROL_FILE,"collector_file":COLLECTOR_FILE,
            "collector_class":"DirectTrainingEnv","effective_record_type":"effective_direct_sources","step_evidence_type":"direct_step_evidence",
            "containment":"actual_solver_body_q_full_R","geometry_imports":normalized(runtime["geometry"].INPUT_SHA256)},
        "source_sha256":sources,"input_files":{k:{"path":str(p.resolve()),"sha256":sha(p)} for k,p in files.items()},
        "historical_protocol_sha256":historical_frozen,"gui_checkpoint_sha256_before":gui_hashes(),
        "physics":{"mode":"live_PBF_action_conditioned","dt_s":DT,"spacing_m":.0035,"pressure_iterations":6,"minimum_substeps":4,
            "cuda_graph":False,"action_deadline_s":5.,"post_observation_s":2.,"post_observation_until_physical_time_s":7.,"tolerance_ml":.1,"maximum_absolute_error_ml":.2},
        "randomization":{"initial_jitter_max_spacing":.03,"viscosity_factor":[.9,1.1]},
        "completion":{"profile":"pregrasp_upright_lift_fixed_mouth_dose_exact_reverse_prepare_original_pose_stable",
            "stable_frames":12,"home_position_validation_tolerance_m":.001,"home_orientation_validation_tolerance_rad":math.radians(.1),
            "home_joint_validation_tolerance_rad":.01,"home_coordinate_absolute_max":1e-7,"source_airborne_particles_max":0,
            "maximum_post_received_change_ml":0.},
        "robot_envelope":{"joint_lower_limits_rad":normalized(runtime["geometry"].JOINT_LOWER_LIMITS_RAD),
            "joint_upper_limits_rad":normalized(runtime["geometry"].JOINT_UPPER_LIMITS_RAD),
            "joint_velocity_limits_rad_s":normalized(runtime["geometry"].JOINT_VELOCITY_LIMITS_RAD_S),
            "joint_acceleration_simulation_cap_rad_s2":25.,"coordinate_acceleration_is_a_physical_release_gate":False},
        "selection_tuple":["unsafe_count","deadline_or_home_or_late_drop_failures","precision_violation_ml","MAE_plus_half_worst_ml","mean_action_physical_seconds"],
        "release":{"overall_and_each_target_mae_ml_at_most":.1,"worst_absolute_error_ml_at_most":.2,
            "all_cases_require_5s_full_home_zero_spill_mass_finite_no_late":True},"automatic_gui_deployment":False}


def unchanged(d):
    need(source_manifest()==d["source_sha256"] and gui_hashes()==d["gui_checkpoint_sha256_before"],"Frozen V4/legacy source or original GUI checkpoint changed.")
    for entry in d["input_files"].values():need(sha(entry["path"])==entry["sha256"],"A frozen V4 input changed.")


def freeze_inputs(output,d,resume):
    output.mkdir(parents=True,exist_ok=True);path=output/"input_freeze.json";iid=signature(d)
    if path.is_file():
        previous=read_json(path)
        need(resume and previous["input_id"]==iid and previous["declaration"]==d,"Resume input/config/source identity changed.")
    else:
        need(not any((output/name).exists() for name in ("protocol.json","episodes.jsonl","summary.json","freeze.json")),"Existing unrelated evidence cannot be adopted.")
        for relative,digest in d["source_sha256"].items():
            destination=output/"source_snapshot"/relative;destination.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/relative,destination)
            need(sha(destination)==digest,"Source changed while snapshotting.")
        for key,entry in d["input_files"].items():
            destination=output/"input_snapshot"/(key+Path(entry["path"]).suffix);destination.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(entry["path"],destination)
            need(sha(destination)==entry["sha256"],"Input changed while snapshotting.")
        write_json(path,{"input_id":iid,"frozen_at_utc":now(),"declaration":d})
    for name,digest in d["source_sha256"].items():need(sha(output/"source_snapshot"/name)==digest,"Frozen source snapshot changed.")
    for key,entry in d["input_files"].items():need(sha(output/"input_snapshot"/(key+Path(entry["path"]).suffix))==entry["sha256"],"Frozen input snapshot changed.")
    unchanged(d);return iid


def expected_ids():
    return ({f"train_g{g:03d}_c{c:03d}_t{t}_s{s}" for g in range(2) for c in range(4) for t in TARGETS for s in TRAIN_SEEDS}
        | {f"{stage}_{role}_t{t}_s{s}" for stage,seeds in (("validation",VALIDATION_SEEDS),("test",TEST_SEEDS)) for role in ("candidate","baseline") for t in TARGETS for s in seeds})


def completed_action_guard(output,records):
    """A returned action is never physically repeated because commit crashed."""
    for path in sorted((output/"traces").glob("*.steps.jsonl.gz")):
        name=path.name.removesuffix(".steps.jsonl.gz")
        if name in records:continue
        has_footer=False
        try:
            with gzip.open(path,"rt",encoding="utf-8") as stream:
                for line in stream:
                    try:row=json.loads(line)
                    except json.JSONDecodeError:continue
                    if row.get("type")=="result":has_footer=True;break
        except (EOFError,OSError):
            # A genuinely incomplete attempt may be preserved before retry.
            # A footer already read remains authoritative even in a broken gzip.
            pass
        need(not has_footer,"A completed action footer has no committed journal row: "+name+
             ". Preserve and recover it offline; automatic GPU/physical replay is forbidden.")


def record_links(row):
    c,result,evidence=row["conditions"],row["result"],row["motion_evidence"]
    need((result["target_ml"],result["seed"],result["randomized"],result["policy_params"])==
         (c["target_ml"],c["seed"],c["randomized"],c["policy_params"]),"Journal condition/footer linkage differs.")
    need(evidence["original_action_result"]==result and evidence["conditions"]==c
         and evidence["action_motion_metrics"]==row["motion_metrics"] and evidence["assessment"]==row["assessment"]
         and evidence["completed_at_utc"]==row["completed_at_utc"]
         and evidence["total_action_plus_post_wall_seconds"]==row["total_action_plus_post_wall_seconds"]
         and math.isfinite(row["total_action_plus_post_wall_seconds"]) and row["total_action_plus_post_wall_seconds"]>=0,
         "Original result/conditions or fixed precommit wall in the motion sidecar differs from the journal.")


def load_records(output,pid,d,resume):
    records={};path=output/"episodes.jsonl"
    if not path.exists():return records
    need(resume,"Existing journal requires --resume.")
    lines=path.read_text(encoding="utf-8").splitlines(keepends=True);good=[]
    for i,line in enumerate(lines):
        try:row=json.loads(line)
        except json.JSONDecodeError:
            need(i==len(lines)-1 and not line.endswith("\n"),"Corrupt committed V4 journal row.")
            ref.atomic_text(output/"interrupted_artifacts"/("journal_tail."+str(time.time_ns())+".txt"),line)
            ref.atomic_text(path,"".join(good));break
        name=row["episode_id"];need(name not in records and row["protocol_id"]==pid and row["conditions_sha256"]==signature(row["conditions"]),"Duplicate/foreign/damaged journal record.")
        for key in ("record_path","motion_record_path","post_record_path"):
            if row[key] is not None:
                artifact=(output/row[key]).resolve();need(artifact.is_relative_to(output) and sha(artifact)==row[key.replace("_path","_sha256")],"Resumed raw artifact SHA changed.")
        ac=trace_check(output/row["record_path"],row["result"],d)
        pc=ref.verify_post_trace(output/row["post_record_path"] if row["post_record_path"] else None,row["post_observation"],row["result"])
        need(ac==row["trace_check"] and pc==row["post_trace_check"] and read_json(output/row["motion_record_path"])==row["motion_evidence"],"Resumed original footer/evidence changed.")
        record_links(row)
        need(assess(row["result"],row["controller_summary"],row["motion_metrics"],row["post_observation"],ac,pc,row["motion_evidence"]["post_motion_metrics"])==row["assessment"],"Resumed V4 assessment changed.")
        records[name]=row;good.append(line if line.endswith("\n") else line+"\n")
    need(set(records)<=expected_ids(),"Undeclared V4 journal episode.")
    return records


def verify_complete(output,summary,pid,d,records):
    need(summary["complete"] is True and summary["protocol_id"]==pid and set(records)==expected_ids()
        and summary["total_recorded_episodes"]==78 and set(summary["all_episode_ids"])==set(records),"Completed run is not its original full 78-case protocol.")
    need(summary["protocol"]==read_json(output/"protocol.json")["protocol"] and summary["raw_action_result_footers_modified"] is False
         and summary["test_used_for_training_or_selection"] is False and summary["automatic_gui_deployment"] is False,
         "Completed protocol or isolation semantics changed.")
    search=LexCEM(d["cem"]["parameter_names"],d["cem"]["bounds"],d["cem"]["initial_params"],d["cem"]["initial_std_fraction"])
    training=[]
    def condition(row,stage,role,target,seed,params,generation=None,candidate=None):
        expected={"stage":stage,"policy_role":role,"target_ml":float(target),"seed":seed,"policy_params":params,
                  "randomized":True,"generation":generation,"candidate":candidate}
        need(row["conditions"]==expected,"Completed scheduled parameters/conditions changed.")
    for g in range(2):
        params=search.ask();groups=[]
        for index,parameters in enumerate(params):
            rows=[records[f"train_g{g:03d}_c{index:03d}_t{t}_s{s}"] for t in TARGETS for s in TRAIN_SEEDS]
            for row in rows:condition(row,"train","candidate",row["result"]["target_ml"],row["result"]["seed"],parameters,g,index)
            groups.append(rows);training.extend(rows)
        keys=[selection_key(rows) for rows in groups];update=search.tell(params,keys)
        entry={"generation":g,"candidates":[{"index":i,"params":v,"selection_key":list(k),"metrics":group_metrics(rows),"episode_ids":[r["episode_id"] for r in rows]} for i,(v,k,rows) in enumerate(zip(params,keys,groups))],"completed_training_episodes":len(training),**update}
        need(summary["training"]["generations"][g]==entry and read_json(output/f"generation_{g:03d}.json")=={"protocol_id":pid,**entry},"Completed CEM replay/generation changed.")
    candidate,baseline=dict(search.best_params),dict(d["cem"]["baseline_params"])
    val={role:[records[f"validation_{role}_t{t}_s{s}"] for s in VALIDATION_SEEDS for t in TARGETS] for role in ("candidate","baseline")}
    for role,rows in val.items():
        for row in rows:condition(row,"validation",role,row["result"]["target_ml"],row["result"]["seed"],candidate if role=="candidate" else baseline)
        need(summary["validation"]["metrics"][role]==group_metrics(rows)
             and summary["validation"]["selection_keys"][role]==list(selection_key(rows)),"Completed validation metrics/selection changed.")
    recommended="candidate" if candidate!=baseline and selection_key(val["candidate"])<selection_key(val["baseline"]) else "baseline"
    need(summary["recommendation"]=={"role":recommended,"selected_on_validation_only":True,"additional_test_rollouts":0}
         and summary["validation"]["recommended_role"]==recommended and summary["validation"]["test_used_for_selection"] is False,
         "Completed recommendation was reselected.")
    freeze=read_json(output/"freeze.json")
    need(freeze["protocol_id"]==pid and freeze["recommended_role"]==recommended and freeze["test_used_for_selection"] is False,
         "Completed pre-test recommendation freeze changed.")
    for role,params in (("candidate",candidate),("recommended",candidate if recommended=="candidate" else baseline)):
        model=read_json(output/(role+"_frozen.json"))
        need(sha(output/(role+"_frozen.json"))==freeze[role+"_sha256"]==summary["models"][role]["sha256"]
             and model["parameters"]==params==freeze[role+"_params"]==summary["models"][role]["parameters"]
             and model["policy_kind"]==POLICY_KIND and model["metadata"]["protocol_id"]==pid
             and model["metadata"]["source_sha256"]==d["source_sha256"] and model["metadata"]["test_used_for_selection"] is False
             and model["metadata"]["frozen_at_utc"]==freeze["frozen_at_utc"] and model["metadata"]["role"]==role
             and model["metadata"]["algorithm"]==ALGORITHM and model["metadata"]["automatic_gui_deployment"] is False,
             "Completed checkpoint bytes/parameters/provenance changed.")
    need(summary["training"]["metrics"]==group_metrics(training) and summary["training"]["best_params"]==candidate
         and summary["training"]["best_key"]==list(search.best_key) and summary["all_rollouts"]==group_metrics(list(records.values())),
         "Completed aggregate or training optimum changed.")
    for role in ("candidate","baseline"):
        rows=[r for r in records.values() if r["conditions"]["stage"]=="test" and r["conditions"]["policy_role"]==role]
        for row in rows:condition(row,"test",role,row["result"]["target_ml"],row["result"]["seed"],candidate if role=="candidate" else baseline)
        need(group_metrics(rows)==summary["test"]["metrics"][role] and release_passed(rows)==summary["release_acceptance"][role+"_passed"],"Completed holdout metrics changed.")
        need(all(datetime.fromisoformat(r["episode_wall_clock_start_utc"])>datetime.fromisoformat(freeze["frozen_at_utc"]) for r in rows),"A heldout reset preceded model freeze.")
    pairs=summary["test"]["paired_results"]
    need(len(pairs)==9 and {(pair["target_ml"],pair["seed"]) for pair in pairs}=={(t,s) for t in TARGETS for s in TEST_SEEDS},"Completed summary omitted a test pair.")
    for pair in pairs:
        for role in ("candidate","baseline"):need(pair[role]==records[f"test_{role}_t{pair['target_ml']}_s{pair['seed']}"] and pair["recommended_result_role"]==recommended,"Completed paired case changed.")
    need(summary["test"]["recommended_metrics"]==summary["test"]["metrics"][recommended]
         and summary["release_acceptance"]["recommended_passed"]==summary["release_acceptance"][recommended+"_passed"],"Completed recommended metrics changed.")
    unchanged(d)
    return {"already_complete_verified_no_gpu_or_retraining":True,"protocol_id":pid,"total_recorded_episodes":78,"release_acceptance":summary["release_acceptance"]}


def run(args):
    need(not args.graph,"V4 uses original graph-off PBF dispatch only.")
    runtime=cpu_runtime();d=declaration(args,runtime)
    if args.dry_run:return {"dry_run":True,"no_gpu_or_frozen_run_written":True,"eligible_for_actual_start":d["engineering_gate"]["eligible"] and not d["search_config_is_draft"],"proposed_declaration":d}
    need(d["engineering_gate"]["eligible"] and not d["search_config_is_draft"],"Only --dry-run is allowed until all-three engineering gates pass and the draft configuration is finalized.")
    need(sys.version_info[:2]==(3,12) and Path(sys.executable).samefile(sdk_root()/"kit/python/python.exe"),
         "Actual runs/resumes require the installed SDK Python3.12 interpreter.")
    output=args.output.resolve()
    need(not args.verify_only or (output/"protocol.json").is_file(),"Read-only verification requires an existing protocol; no freeze was created.")
    iid=freeze_inputs(output,d,args.resume)
    ppath=output/"protocol.json"
    initialized=False
    if ppath.exists():
        need(args.resume,"Protocol already exists; use --resume.");document=read_json(ppath);p=document["protocol"];pid=signature(p)
        need(document["protocol_id"]==pid and p["declaration"]==d and p["input_id"]==iid,"Resume protocol differs.")
        records=load_records(output,pid,d,True)
        completed_action_guard(output,records)
        if (output/"summary.json").exists():return verify_complete(output,read_json(output/"summary.json"),pid,d,records)
        if args.verify_only:return {"partial_protocol_verified_no_gpu":True,"protocol_id":pid,"committed_episodes":len(records),"declared_total":78}
    else:
        need(not args.verify_only,"No existing protocol to verify.")
        completed_action_guard(output,{})
        shared=load("pivot_sdk_configuration",ROOT/"train_target_volume_policy.py");versions=shared.configure_sdk();initialized=True;unchanged(d)
        p={"schema_version":1,"kind":"formal_upright_lift_fixed_mouth_CEM_78_live_episodes","input_id":iid,"declaration":d,
           "software":{**versions,"python":sys.version,"platform":platform.platform()},"raw_action_footer_modified":False}
        pid=signature(p);write_json(ppath,{"created_at_utc":now(),"protocol_id":pid,"protocol":p});records={}
    if args.resume and not initialized:
        shared=load("pivot_resume_sdk_configuration",ROOT/"train_target_volume_policy.py");versions=shared.configure_sdk()
        need(all(p["software"][k]==v for k,v in versions.items())
             and p["software"]["python"]==sys.version and p["software"]["platform"]==platform.platform(),"Resume SDK/Python/platform versions changed.")
    env=None;new_count=0;started=time.perf_counter()
    def episode(name,stage,role,target,seed,params,generation=None,candidate=None):
        nonlocal env,new_count
        params=normalized(runtime["control"].DirectPolicy(params).params)
        c={"stage":stage,"policy_role":role,"target_ml":float(target),"seed":seed,"policy_params":params,"randomized":True,"generation":generation,"candidate":candidate}
        unchanged(d)
        if name in records:need(records[name]["conditions"]==c,"Resumed candidate/condition differs.");return records[name]
        paths={"action":output/"traces"/(name+".steps.jsonl.gz"),"post":output/"traces"/(name+".post.jsonl.gz"),"motion":output/"motion"/(name+".json")}
        completed_action_guard(output,records)
        ref.preserve_uncommitted(output,paths.values())
        if env is None:env=runtime["collector"].DirectTrainingEnv(geometry=runtime["geometry"],control=runtime["control"])
        def progress(event):write_json(output/"progress.json",{"state":"running","updated_at_utc":now(),"protocol_id":pid,"episode_id":name,"conditions":c,"completed_episodes":len(records),"total_planned_episodes":78,**normalized(event)})
        wall=time.perf_counter();print(f"PIVOT EPISODE {len(records)+1}/78: {name}",flush=True)
        result=normalized(env.run_episode(target,seed,policy=runtime["control"].DirectPolicy(params),randomize=True,record_path=paths["action"],progress=progress));finite(result)
        need((result["target_ml"],result["seed"],result["randomized"],result["policy_params"])==(float(target),seed,True,params),"Collector returned different actual conditions/policy.")
        status,motion=normalized(copy.deepcopy(env.last_controller.status)),normalized(copy.deepcopy(env.fast_metrics))
        action_terminal={"fluid_time_s":float(env.fluid.time),"last_measurement_time_s":env._last_measurement_time,
            "kinematics":normalized(copy.deepcopy(env._current_kinematics))}
        ac=trace_check(paths["action"],result,d);post=None
        if result["valid_physics"]:
            try:post=normalized(env.post_observe(paths["post"],seconds=2.))
            except (RuntimeError,ValueError,FloatingPointError) as exc:post={"passed":False,"error":str(exc),"error_type":type(exc).__name__,"purpose":"failed_post_observation_retained_no_action_rescue"}
        pc=ref.verify_post_trace(paths["post"] if paths["post"].exists() else None,post,result)
        post_motion=normalized(copy.deepcopy(env.fast_metrics))
        assessment=assess(result,status,motion,post,ac,pc,post_motion)
        completed_at=now();total_wall=time.perf_counter()-wall
        evidence={"episode_id":name,"protocol_id":pid,"conditions":c,"original_action_result":result,
            "completed_at_utc":completed_at,"assessment":assessment,"total_action_plus_post_wall_seconds":total_wall,
            "wall_measurement_end":"before_atomic_motion_and_journal_commit",
            "action_motion_metrics":motion,"post_motion_metrics":post_motion,
            "controller_at_action_end":status,"action_terminal_observation":action_terminal,
            "post_terminal_observation":{"fluid_time_s":float(env.fluid.time),"last_measurement_time_s":env._last_measurement_time,
                "kinematics":normalized(copy.deepcopy(env._current_kinematics)),
                "unlogged_terminal_post_frame_possible":bool(isinstance(post,dict) and post.get("error")),
                "partial_fluid_frame_possible":bool(isinstance(post,dict) and post.get("error"))},
            "post_observation":post,"trace_check":ac,"post_trace_check":pc,"action_result_modified":False}
        write_json(paths["motion"],evidence)
        row={"episode_id":name,"protocol_id":pid,"conditions":c,"conditions_sha256":signature(c),"completed_at_utc":completed_at,
            "episode_wall_clock_start_utc":ac["reset_started_utc"],"record_path":paths["action"].relative_to(output).as_posix(),"record_sha256":sha(paths["action"]),
            "post_record_path":paths["post"].relative_to(output).as_posix() if paths["post"].exists() else None,"post_record_sha256":sha(paths["post"]) if paths["post"].exists() else None,
            "motion_record_path":paths["motion"].relative_to(output).as_posix(),"motion_record_sha256":sha(paths["motion"]),"result":result,"controller_summary":status,"motion_metrics":motion,
            "motion_evidence":evidence,"post_observation":post,"trace_check":ac,"post_trace_check":pc,
            "assessment":assessment,"total_action_plus_post_wall_seconds":total_wall}
        record_links(row)
        append_json(output/"episodes.jsonl",row);records[name]=row;new_count+=1;progress({"event":"episode_recorded","assessment":row["assessment"]});return row
    try:
        search=LexCEM(d["cem"]["parameter_names"],d["cem"]["bounds"],d["cem"]["initial_params"],d["cem"]["initial_std_fraction"])
        generations,train_records,curve=[],[],[]
        for g in range(2):
            params=[normalized(runtime["control"].DirectPolicy(v).params) for v in search.ask()];groups=[]
            for index,candidate in enumerate(params):groups.append([episode(f"train_g{g:03d}_c{index:03d}_t{t}_s{s}","train","candidate",t,s,candidate,g,index) for t in TARGETS for s in TRAIN_SEEDS])
            keys=[selection_key(group) for group in groups];update=search.tell(params,keys);train_records.extend(row for group in groups for row in group)
            entry={"generation":g,"candidates":[{"index":i,"params":v,"selection_key":list(k),"metrics":group_metrics(rows),"episode_ids":[r["episode_id"] for r in rows]} for i,(v,k,rows) in enumerate(zip(params,keys,groups))],"completed_training_episodes":len(train_records),**update}
            generations.append(entry);write_json(output/f"generation_{g:03d}.json",{"protocol_id":pid,**entry})
            curve.append({"generation":g,"completed_training_episodes":len(train_records),"population_mean_mae_ml":sum(group_metrics(rows)["mae_ml"] for rows in groups)/4,"best_key":json.dumps(list(search.best_key))})
            write_csv(output/"training_curve.csv",curve,tuple(curve[0]))
        candidate,baseline=dict(search.best_params),dict(d["cem"]["baseline_params"])
        val={role:[episode(f"validation_{role}_t{t}_s{s}","validation",role,t,s,params) for s in VALIDATION_SEEDS for t in TARGETS] for role,params in (("candidate",candidate),("baseline",baseline))}
        scores={role:selection_key(rows) for role,rows in val.items()};recommended="candidate" if candidate!=baseline and scores["candidate"]<scores["baseline"] else "baseline"
        recommendation=candidate if recommended=="candidate" else baseline
        validation={"protocol_id":pid,"selection_keys":{k:list(v) for k,v in scores.items()},"metrics":{k:group_metrics(v) for k,v in val.items()},"episode_ids":{k:[r["episode_id"] for r in v] for k,v in val.items()},"recommended_role":recommended,"test_used_for_selection":False}
        write_json(output/"validation.json",validation)
        fpath=output/"freeze.json"
        if fpath.exists():
            frozen=read_json(fpath);need(frozen["protocol_id"]==pid and frozen["candidate_params"]==candidate and frozen["recommended_params"]==recommendation and frozen["recommended_role"]==recommended,"Pre-test model freeze differs.")
        else:
            need(not any(r["conditions"]["stage"]=="test" for r in records.values()),"Holdout exists without a prior model freeze.")
            need(not any((output/(r+"_frozen.json")).exists() for r in ("candidate","recommended")),"Partial model freeze must be preserved and reviewed.")
            timestamp=now();metadata={"protocol_id":pid,"input_id":iid,"frozen_at_utc":timestamp,"algorithm":ALGORITHM,"neural_policy":False,"ppo":False,"training_episodes":48,
                "source_sha256":d["source_sha256"],"module_binding":d["module_binding"],"test_used_for_selection":False,"automatic_gui_deployment":False,"recommended_role":recommended}
            for role,params in (("candidate",candidate),("recommended",recommendation)):runtime["control"].DirectPolicy(params).save(output/(role+"_frozen.json"),metadata={**metadata,"role":role})
            frozen={"protocol_id":pid,"frozen_at_utc":timestamp,"recommended_role":recommended,"candidate_params":candidate,"recommended_params":recommendation,
                "candidate_sha256":sha(output/"candidate_frozen.json"),"recommended_sha256":sha(output/"recommended_frozen.json"),"test_used_for_selection":False,"automatic_gui_deployment":False}
            write_json(fpath,frozen)
        for role,params in (("candidate",candidate),("recommended",recommendation)):need(sha(output/(role+"_frozen.json"))==frozen[role+"_sha256"] and read_json(output/(role+"_frozen.json"))["parameters"]==params,"Frozen checkpoint changed before test.")
        pairs=[];tests={role:[] for role in ("candidate","baseline")}
        for seed in TEST_SEEDS:
            for target in TARGETS:
                pair={"target_ml":target,"seed":seed,"recommended_result_role":recommended}
                for role,params in (("candidate",candidate),("baseline",baseline)):
                    row=episode(f"test_{role}_t{target}_s{seed}","test",role,target,seed,params)
                    need(datetime.fromisoformat(row["episode_wall_clock_start_utc"])>datetime.fromisoformat(frozen["frozen_at_utc"]),"Heldout reset preceded freeze.")
                    pair[role]=row;tests[role].append(row)
                pairs.append(pair)
        need(set(records)==expected_ids() and len(records)==78,"Final V4 journal budget differs.");unchanged(d)
        for role in ("candidate","recommended"):need(sha(output/(role+"_frozen.json"))==frozen[role+"_sha256"],"Test changed a frozen model.")
        summary={"schema_version":1,"complete":True,"completed_at_utc":now(),"protocol_id":pid,"protocol":p,"budget":BUDGET,
            "training":{"generations":generations,"metrics":group_metrics(train_records),"best_key":list(search.best_key),"best_params":candidate},"validation":validation,
            "models":{role:{"file":role+"_frozen.json","sha256":frozen[role+"_sha256"],"parameters":params} for role,params in (("candidate",candidate),("recommended",recommendation))},
            "recommendation":{"role":recommended,"selected_on_validation_only":True,"additional_test_rollouts":0},
            "test":{"paired_results":pairs,"metrics":{k:group_metrics(v) for k,v in tests.items()},"distinct_target_seed_conditions":9,"executed_live_episodes":18,"recommended_metrics":group_metrics(tests[recommended])},
            "release_acceptance":{**{k+"_passed":release_passed(v) for k,v in (*tests.items(),("recommended",tests[recommended]))},"does_not_reselect_model":True,"automatic_gui_deployment":False},
            "total_recorded_episodes":78,"all_episode_ids":list(records),"all_rollouts":group_metrics(list(records.values())),"new_episodes_this_invocation":new_count,
            "wall_seconds_this_invocation":time.perf_counter()-started,"total_action_plus_post_wall_seconds":sum(r["total_action_plus_post_wall_seconds"] for r in records.values()),
            "interrupted_artifact_records":len((output/"interrupted_artifacts.jsonl").read_text().splitlines()) if (output/"interrupted_artifacts.jsonl").exists() else 0,
            "test_used_for_training_or_selection":False,"raw_action_result_footers_modified":False,"source_sha256_unchanged_after_run":True,"gui_checkpoint_sha256_after":gui_hashes(),"automatic_gui_deployment":False,
            "limitations":["Parameter CEM on privileged simulation particle mass, not neural/PPO or visual/hardware learning.","Baseline is this V4 engineering protocol policy, not historical V3.","All failed committed cases and interrupted artifacts remain retained; post observations cannot rescue a 5-second miss.","Joint acceleration25rad/s² is a simulation envelope, not a manufacturer specification; torque/contact grasp safety is unproven."]}
        write_json(output/"summary.json",summary);write_json(output/"progress.json",{"state":"complete","protocol_id":pid,"completed_episodes":78,"release_acceptance":summary["release_acceptance"]});return summary
    except BaseException as exc:
        write_json(output/"progress.json",{"state":"interrupted","updated_at_utc":now(),"protocol_id":pid,"completed_episodes":len(records),"error_type":type(exc).__name__,"error":str(exc)});raise


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,default=ROOT/"droplet_out/pour_policy_pivot_v4/run_draft")
    p.add_argument("--engineering-report",type=Path)
    p.add_argument("--search-config",type=Path)
    p.add_argument("--dry-run",action="store_true");p.add_argument("--resume",action="store_true")
    p.add_argument("--verify-only",action="store_true");p.add_argument("--graph",action="store_true")
    return p


def main(argv=None):
    args=parser().parse_args(argv);args.output=args.output.resolve()
    need(args.output.is_relative_to(ROOT/"droplet_out/pour_policy_pivot_v4"),"Use the independent V4 output tree.")
    need(not args.verify_only or args.resume,"Read-only verification requires --resume.")
    if (args.output/"summary.json").exists() and not args.dry_run:
        need(args.resume,"A completed run requires --resume for CPU verification, never retraining.")
        args.verify_only=True
    lock=token=None
    try:
        if not args.dry_run and not args.verify_only:lock,token=acquire_worker(args.output,args.resume)
        result=run(args);print(json.dumps(result,indent=2,allow_nan=False),flush=True);return 0
    finally:
        if lock is not None and lock.exists() and read_json(lock).get("token")==token:lock.unlink()


if __name__=="__main__":raise SystemExit(main())
