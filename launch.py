"""Kit --exec launcher that makes this checkout importable.

Set ISAAC_FLUID_DEMO to target-volume (default), ai, pour, milkcrown, or bake-cache.
For Script Editor buffers, use the explicit project-path snippet in README.
"""
import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
DEMOS = {
    "target-volume": "robot_target_volume_pour.py",
    "ai": "robot_target_volume_ai.py",
    "pour": "robot_testtube_pour.py",
    "milkcrown": "droplet_milkcrown_newton_surface.py",
    "bake-cache": "tools/bake_pour_cache.py",
}


def main():
    name = os.environ.get("ISAAC_FLUID_DEMO", "target-volume")
    if name not in DEMOS:
        raise ValueError(f"ISAAC_FLUID_DEMO must be one of {tuple(DEMOS)}")
    path = PROJECT_ROOT / DEMOS[name]
    exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"),
         {"__name__": "__main__", "__file__": str(path)})


if __name__ == "__main__":
    main()
