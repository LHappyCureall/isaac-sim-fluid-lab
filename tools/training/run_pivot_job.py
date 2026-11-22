"""Explicit V4 supervisor; --dry-run never starts Warp or a training worker.

Actual --start needs a complete all-target engineering report and finalized
configuration. A stale supervisor may resume only the same frozen protocol,
after verifying its previous worker exited. No process is ever terminated.
"""
import argparse
import ctypes
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import shutil

ROOT=Path(__file__).resolve().parents[2]


def alive(pid):
    if not pid:return False
    if os.name!="nt":
        try:os.kill(int(pid),0);return True
        except ProcessLookupError:return False
    kernel=ctypes.WinDLL("kernel32",use_last_error=True)
    kernel.OpenProcess.argtypes=[ctypes.c_uint32,ctypes.c_int,ctypes.c_uint32];kernel.OpenProcess.restype=ctypes.c_void_p
    kernel.GetExitCodeProcess.argtypes=[ctypes.c_void_p,ctypes.POINTER(ctypes.c_uint32)]
    kernel.CloseHandle.argtypes=[ctypes.c_void_p]
    handle=kernel.OpenProcess(0x1000,0,int(pid))
    if not handle:
        if ctypes.get_last_error()==87:return False
        raise RuntimeError("Cannot prove the previous process exited; refusing another worker.")
    try:
        code=ctypes.c_uint32()
        if not kernel.GetExitCodeProcess(handle,ctypes.byref(code)):raise RuntimeError("Cannot inspect previous process exit status.")
        return code.value==259
    finally:kernel.CloseHandle(handle)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,required=True);p.add_argument("--engineering-report",type=Path)
    p.add_argument("--search-config",type=Path);p.add_argument("--resume",action="store_true")
    mode=p.add_mutually_exclusive_group(required=True);mode.add_argument("--dry-run",action="store_true");mode.add_argument("--start",action="store_true")
    p.add_argument("--max-restarts",type=int,default=0,help="Explicit same-protocol retries after worker exit; interrupted evidence is preserved.")
    args=p.parse_args();output=args.output.resolve()
    if not output.is_relative_to(ROOT/"droplet_out/pour_policy_pivot_v4") or not 0<=args.max_restarts<=3:raise ValueError("Invalid isolated output or retry budget.")
    spec=importlib.util.spec_from_file_location("pivot_supervised_runner",ROOT/"train_target_volume_pivot.py")
    runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
    command=[sys.executable,"-X","utf8","-u",str(ROOT/"train_target_volume_pivot.py"),"--output",str(output)]
    if args.engineering_report:command += ["--engineering-report",str(args.engineering_report.resolve())]
    if args.search_config:command += ["--search-config",str(args.search_config.resolve())]
    if args.resume:command += ["--resume"]
    if args.dry_run:return runner.main(command[5:]+["--dry-run"])
    if sys.version_info[:2]!=(3,12) or not Path(sys.executable).samefile(runner.sdk_root()/"kit/python/python.exe"):raise RuntimeError("Actual supervisor must use the SDK Python3.12 interpreter.")
    # Preflight is in-process CPU only. It must pass before creating a worker.
    pre_args=runner.parser().parse_args(command[5:]+["--dry-run"]);pre_args.output=output
    proposed=runner.run(pre_args)
    if not proposed["eligible_for_actual_start"]:raise RuntimeError("Only dry-run is allowed until the strict three-target engineering gate and non-draft configuration pass.")
    output.mkdir(parents=True,exist_ok=True);lock=output/"job.lock";state_path=output/"job_state.json"
    previous_state={}
    if state_path.exists():
        previous_state=runner.read_json(state_path)
        if alive(previous_state.get("worker_pid")) or alive(previous_state.get("supervisor_pid")):raise RuntimeError("An earlier supervisor/worker is still alive; no process was stopped.")
        if not args.resume:raise RuntimeError("Existing evidence requires explicit --resume.")
        destination=output/"interrupted_artifacts"/("job_state."+str(time.time_ns())+"."+runner.sha(state_path)[:12]+".json")
        destination.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(state_path,destination)
    if lock.exists():
        old=runner.read_json(lock)
        if alive(old.get("supervisor_pid")):raise RuntimeError("A supervisor owns this protocol.")
        destination=output/"interrupted_artifacts"/("job.lock."+str(time.time_ns())+".json");destination.parent.mkdir(parents=True,exist_ok=True);lock.replace(destination)
    token=str(time.time_ns())
    with lock.open("x",encoding="utf-8") as stream:json.dump({"supervisor_pid":os.getpid(),"created_at_utc":runner.now(),"token":token},stream)
    sources,gui=runner.source_manifest(),runner.gui_hashes()
    state={"state":"starting","supervisor_pid":os.getpid(),"created_at_utc":previous_state.get("created_at_utc",runner.now()),"supervisor_sha256":runner.sha(__file__),"budget":runner.BUDGET,
        "command":command,"worker_pid":None,"attempts":list(previous_state.get("attempts",[])),"gui_checkpoint_sha256_before":gui,"source_sha256":sources}
    runner.write_json(state_path,state)
    try:
        for attempt in range(args.max_restarts+1):
            if sources!=runner.source_manifest() or gui!=runner.gui_hashes():raise RuntimeError("A source or original GUI checkpoint changed; no restart allowed.")
            current=list(command)
            if (output/"input_freeze.json").exists() and "--resume" not in current:current.append("--resume")
            if (output/"summary.json").exists():current.append("--verify-only")
            suffix=str(time.time_ns());log=output/("worker."+suffix+".stdout.log");err=output/("worker."+suffix+".stderr.log")
            with log.open("xb") as stdout,err.open("xb") as stderr:
                process=subprocess.Popen(current,cwd=ROOT,env=dict(os.environ,ISAAC_SIM_ROOT=str(runner.sdk_root())),stdout=stdout,stderr=stderr,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name=="nt" else 0)
                state.update(state="running",worker_pid=process.pid,updated_at_utc=runner.now())
                state["attempts"].append({"attempt":attempt,"worker_pid":process.pid,"command":current,"started_at_utc":runner.now(),"stdout":log.name,"stderr":err.name})
                runner.write_json(state_path,state);code=process.wait()
            state["attempts"][-1].update(exit_code=code,exited_at_utc=runner.now());state["worker_pid"]=None
            unchanged=sources==runner.source_manifest() and gui==runner.gui_hashes()
            summary=runner.read_json(output/"summary.json") if (output/"summary.json").exists() else {}
            complete=code==0 and unchanged and summary.get("complete") is True and summary.get("total_recorded_episodes")==78
            state.update(state="complete" if complete else "worker_failed",updated_at_utc=runner.now(),sources_unchanged=unchanged,
                summary_complete=summary.get("complete",False),release_acceptance=summary.get("release_acceptance"))
            runner.write_json(state_path,state)
            if complete:return 0
            if not unchanged or code==0:break
        return 1
    except BaseException as exc:
        state.update(state="supervisor_interrupted",updated_at_utc=runner.now(),error_type=type(exc).__name__,error=str(exc))
        runner.write_json(state_path,state)
        raise
    finally:
        if lock.exists() and runner.read_json(lock).get("token")==token:lock.unlink()


if __name__=="__main__":raise SystemExit(main())
