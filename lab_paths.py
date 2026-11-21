"""Project paths and discovery of an independently installed Isaac Sim SDK.

Importing PROJECT_ROOT needs only standard Python. ISAAC_SIM_ROOT is resolved
when explicitly imported, so texture preparation also works outside Isaac Sim.
"""

import os
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parent
__all__ = ["PROJECT_ROOT", "ISAAC_SIM_ROOT", "find_isaac_sim_root", "require_isaac_sim_root"]


def _is_sdk_root(path):
    return (path / "kit").is_dir() and (path / "exts").is_dir()


def _sdk_ancestor(path):
    candidate = Path(path).expanduser().resolve()
    if not candidate.is_dir():
        candidate = candidate.parent
    for parent in (candidate, *candidate.parents):
        if _is_sdk_root(parent):
            return parent
    return None


def find_isaac_sim_root():
    """Return the SDK installation directory, or None outside its runtime.

    An explicit environment override is authoritative and is validated. This
    never treats the experiments repository as the SDK installation.
    """
    override = os.environ.get("ISAAC_SIM_ROOT")
    if override:
        root = Path(override).expanduser().resolve()
        if not _is_sdk_root(root):
            raise RuntimeError("ISAAC_SIM_ROOT must name an Isaac Sim installation containing kit/ and exts/.")
        return root
    root = _sdk_ancestor(sys.executable)
    if root is not None:
        return root
    try:
        import omni.kit.app

        app = omni.kit.app.get_app()
        manager = app.get_extension_manager() if app is not None else None
        if manager is not None:
            for module in ("isaacsim.asset.importer.urdf", "isaacsim.pip.newton", "omni.kit.app"):
                extension_path = manager.get_extension_path_by_module(module)
                if extension_path:
                    root = _sdk_ancestor(extension_path)
                    if root is not None:
                        return root
    except (ImportError, AttributeError, RuntimeError):
        pass
    return None


def require_isaac_sim_root():
    root = find_isaac_sim_root()
    if root is None:
        raise RuntimeError("Cannot locate Isaac Sim. Run with its python launcher, or set ISAAC_SIM_ROOT.")
    return root


def __getattr__(name):
    if name == "ISAAC_SIM_ROOT":
        return require_isaac_sim_root()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
