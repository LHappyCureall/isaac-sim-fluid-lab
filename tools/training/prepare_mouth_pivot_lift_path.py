"""Save optional upright-lift IK coefficients for a future separate runtime.

This cache is intentionally not compatible with the active PivotGeometry:
that runtime implements a direct home-to-78 prepare and is not modified here.
The added lift requires a new explicitly versioned runtime/controller pilot.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from estimate_upright_lift_pivot import compare, duration_for_joint_line, lift_curve
from verify_mouth_pivot_path import load


HERE = Path(__file__).resolve().parents[2] / "projects/pour_policy_training_v3"
Z022_SHA256 = "f0dec80ef890b2ef8946627af98be8d736e2a3d684e7f8f7bf2adb120aa46d40"


def prepare(height, output):
    if not .02 <= height <= .03:
        raise ValueError("The engineering comparison only supports 2--3 cm upright lifts.")
    output = Path(output)
    if output.exists() or output.with_suffix(".json").exists():
        raise FileExistsError("Keep prior engineering caches immutable; use a new output path.")
    module = load()
    g = module.load_geometry(HERE / "mouth_pivot_z022_cache.npz", Z022_SHA256)
    grid, coefficients, qlift, duration, error = lift_curve(g, height)
    report = compare(g, height, module._progress)
    if not report["cpu_constraints_passed"]:
        raise ValueError("Optional lift failed its independent CPU constraints.")
    data = {key: value.copy() for key, value in g._CACHE.items()}
    data.update(kind=np.asarray("optional_upright_lift_fixed_mouth_not_active_runtime"),
                lift_progress_grid=grid, lift_joint_coefficients=coefficients,
                q_lift_end=qlift, lift_height_m=np.asarray(height),
                lift_duration_s=np.asarray(duration),
                lifted_to_78_duration_s=np.asarray(duration_for_joint_line(
                    g.Q_78 - qlift, g.JOINT_VELOCITY_LIMITS_RAD_S)),
                base_cache_sha256=np.asarray(Z022_SHA256),
                preparation_source_sha256=np.asarray(hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, **data)
    result = {"schema_version": 1, "kind": "optional_upright_lift_fixed_mouth_not_active_runtime",
              "cache_file": output.name, "cache_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
              "base_cache_file": g.CACHE_FILE.name, "base_cache_sha256": Z022_SHA256,
              "geometry_source_sha256": hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(),
              "preparation_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "lift_IK_max_error_m_or_rad": error, "cpu_report": report,
              "notice": "CPU preparation only. Not loadable by the frozen direct-prepare PivotGeometry; "
                        "requires a new explicitly versioned runtime before any live test. No liquid benefit established."}
    output.with_suffix(".json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lift-height", type=float, default=.025)
    parser.add_argument("--output", type=Path, default=HERE / "mouth_pivot_lift025_z022_cache.npz")
    args = parser.parse_args()
    print(json.dumps(prepare(args.lift_height, args.output), indent=2, allow_nan=False))
