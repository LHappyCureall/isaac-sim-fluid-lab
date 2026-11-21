"""Publish an explicit source allowlist without changing the local experiments.

The Windows workspace default is a convenience for the author's daily sync.
The published examples use lab_paths instead. Every transformation has a fixed
anchor count; any mismatch aborts before destination files are changed.
"""

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WORKSPACE = Path(r"C:\omninverse")
ALLOWLIST = (
    "droplet_milkcrown_newton.py",
    "droplet_milkcrown_newton_surface.py",
    "robot_testtube_pour.py",
    "robot_target_volume_pour.py",
    "assets/robot_testtube_pour/scripts/geometry.py",
    "assets/robot_testtube_pour/scripts/robot.py",
    "assets/robot_testtube_pour/scripts/liquid.py",
    "assets/robot_testtube_pour/scripts/target_control.py",
    "assets/robot_testtube_pour/scripts/video.py",
    "assets/robot_testtube_pour/README.md",
    "assets/robot_testtube_pour/TARGET_VOLUME.md",
    "textures/table_options/sources.json",
)


class SyncError(RuntimeError):
    pass


def _replace(text, old, new, *, count=1):
    found = text.count(old)
    if found != count:
        raise SyncError(f"Anchor {old!r}: expected {count} occurrence(s), found {found}.")
    return text.replace(old, new)


def transform(relative, text):
    """Return the fixed publication variant; never evaluate imported scripts."""
    if relative == "droplet_milkcrown_newton.py":
        text = _replace(text, "import os\n", "import os\n\nfrom lab_paths import PROJECT_ROOT\n")
        text = _replace(text, 'WOOD_TEXTURE = r"C:\\omninverse\\textures\\wooden_table.jpg"',
                        'WOOD_TEXTURE = str(PROJECT_ROOT / "textures" / "wooden_table.jpg")')
        text = _replace(text, "/mnt/c/omninverse/", "./", count=4)
        text = _replace(text, "# Milk-crown cup in Newton 1.7 implicit MPM.",
                        "# Milk-crown cup using the Isaac Sim bundled Newton implicit MPM solver.")
    elif relative == "droplet_milkcrown_newton_surface.py":
        text = _replace(text, "from pathlib import Path\n", "from pathlib import Path\n\nfrom lab_paths import PROJECT_ROOT\n")
        text = _replace(text, 'TABLE_STYLE = "original_wood"', 'TABLE_STYLE = "white_matte"')
        text = _replace(text, 'TEXTURE_DIRECTORY = Path(r"C:\\omninverse\\textures\\table_options")',
                        'TEXTURE_DIRECTORY = PROJECT_ROOT / "textures" / "table_options"')
        text = _replace(text, '    r"C:\\omninverse\\droplet_milkcrown_newton.py",',
                        '    str(PROJECT_ROOT / "droplet_milkcrown_newton.py"),')
    elif relative in ("robot_testtube_pour.py", "robot_target_volume_pour.py"):
        text = _replace(text, "from pathlib import Path\n", "from pathlib import Path\n\nfrom lab_paths import PROJECT_ROOT\n")
        text = _replace(text, 'TABLE_STYLE = "pale_oak"', 'TABLE_STYLE = "white_matte"')
        text = _replace(text, 'INSTALL = Path(r"C:\\omninverse")', 'INSTALL = PROJECT_ROOT')
        if relative == "robot_testtube_pour.py":
            text = _replace(text, 'OUTPUT = Path(r"C:\\omninverse\\droplet_out\\robot_testtube_pour")',
                            'OUTPUT = PROJECT_ROOT / "droplet_out" / "robot_testtube_pour"')
            text = _replace(text, 'LIQUID_MODE = "replay"', 'LIQUID_MODE = "simulate"')
    elif relative.endswith("/scripts/robot.py"):
        text = _replace(text, "from pathlib import Path\n", "from pathlib import Path\n\nfrom lab_paths import PROJECT_ROOT, ISAAC_SIM_ROOT\n")
        text = _replace(text,
                        "FRANKA_PACKAGE = Path(r'C:\\omninverse\\exts\\isaacsim.asset.importer.urdf\\data\\urdf\\robots\\franka_description')",
                        "FRANKA_PACKAGE = ISAAC_SIM_ROOT / 'exts' / 'isaacsim.asset.importer.urdf' / 'data' / 'urdf' / 'robots' / 'franka_description'")
        text = _replace(text, "FRANKA_CACHE = Path(r'C:\\omninverse\\assets\\robot_testtube_pour\\franka')",
                        "FRANKA_CACHE = PROJECT_ROOT / 'assets' / 'robot_testtube_pour' / 'franka'")
    elif relative.endswith("/scripts/video.py"):
        pattern = r"os\.environ\.get\('LOCALAPPDATA', r'C:\\Users\\[^\\/'\r\n]+\\AppData\\Local'\)"
        text, found = re.subn(pattern,
                             "os.environ.get('LOCALAPPDATA', str(Path.home() / 'AppData' / 'Local'))", text)
        if found != 1:
            raise SyncError(f"Expected exactly one LOCALAPPDATA Windows-user fallback; found {found}.")
    elif relative.endswith("/README.md"):
        text = _replace(text, r"C:\omninverse", "<PROJECT_ROOT>", count=5)
        text = _replace(text, "# 机器人用试管向烧杯倒水\n", "# 机器人用试管向烧杯倒水\n\n"
                        "> 公开仓库默认 `white_matte` 桌面与 `simulate` 实时求解；原实验的回放、视频与节奏说明"
                        "适用于可选物理缓存。Franka 从 SDK 本地转换，缓存不随源码提供。"
                        "`<PROJECT_ROOT>` 表示本仓库目录，启动配置见根目录 README。\n")
        text = _replace(text, "入口：`<PROJECT_ROOT>\\robot_testtube_pour.py`。",
                        "入口：项目根目录的 `robot_testtube_pour.py`。")
        text = _replace(text, "在 Isaac Sim 的 Window > Script Editor 中打开入口文件，点击 Run。",
                        "在 Isaac Sim 的 Window > Script Editor 中使用[根目录 README 的加载片段](../../README.md#环境与启动)，"
                        "保留其中的 `sys.path` 和真实 `__file__` 设置，并将 `script` 改为 `project / \"robot_testtube_pour.py\"`，再点击 Run。")
        text = _replace(text, "点击 Export 10s MP4 直接生成短片。",
                        "先生成可选物理缓存并切换到 `replay`，才可点击 Export 10s MP4 生成短片。")
        text = _replace(text,
                        "也可在 PowerShell 中从安装目录启动：\n\n```powershell\n"
                        "cd <PROJECT_ROOT>\n.\\isaac-sim.bat --exec <PROJECT_ROOT>\\robot_testtube_pour.py\n```",
                        "也可在 PowerShell 中设置独立的 SDK 安装目录，并从仓库根目录启动。把下方 SDK 路径改为实际安装位置：\n\n"
                        '```powershell\n$env:ISAAC_SIM_ROOT = "D:\\apps\\isaac-sim"\n'
                        '$env:ISAAC_FLUID_DEMO = "pour"\n'
                        '& "$env:ISAAC_SIM_ROOT\\isaac-sim.bat" --exec "$PWD\\launch.py"\n```')
        text = _replace(text,
                        "默认回放约 9.85 秒，扶正后结束。倒水后的等待为 0.25 秒，扶正用时 1.2 秒。",
                        "公开版默认实时求解，仿真时间约 15.6 秒，扶正后结束。可选缓存回放约 9.85 秒；"
                        "回放中的倒水后等待为 0.25 秒，扶正用时 1.2 秒。")
        text = _replace(text,
                        '- `LIQUID_MODE = "replay"`：默认，回放经过物理计算的 60 Hz 粒子轨迹，实时重建水面，减少卡顿。\n'
                        '- `LIQUID_MODE = "simulate"`：重新实时求解液体，计算量更大。修改水粒子间距、运动时长或容器几何后使用此模式。',
                        '- `LIQUID_MODE = "simulate"`：公开版默认，实时求解液体，不需要物理缓存。修改水粒子间距、运动时长或容器几何后也使用此模式。\n'
                        '- `LIQUID_MODE = "replay"`：可选，先按[根目录 README](../../README.md#缓存桌面与视频)生成 `pour_cache.npz`，'
                        '再回放经过物理计算的 60 Hz 粒子轨迹，实时重建水面，减少卡顿。缓存不随源码发布。')
        text = _replace(text, "默认推荐直接使用本示例的 Export 按钮，",
                        "在已准备好缓存的 `replay` 模式下推荐使用本示例的 Export 按钮，")
        text = _replace(text,
                        "`scripts/` 中四个模块分别提供几何、机器人、液体和视频导出实现。`franka/` 是官方 URDF 转换后的必要资源；"
                        "`pour_cache.npz` 是完整物理轨迹，不是视频。材质复用现有 milkcrown surface 脚本及桌面纹理。",
                        "`scripts/` 提供几何、机器人、液体、反馈控制与视频导出实现。`franka/` 是首次运行时从 SDK 自带 URDF 转换生成的本地缓存；"
                        "可选的 `pour_cache.npz` 是物理轨迹，不是视频，可通过 `tools/bake_pour_cache.py` 生成。两类缓存均不进入 Git。"
                        "材质复用 milkcrown surface 脚本，默认纯色桌面无需贴图。")
        text = _replace(text,
                        "两个文件可以同时在 Script Editor 中打开；选择目标文件并点击 Run 才会切换场景。"
                        "磁盘文件更新后，旧标签不会自动读取新代码，应关闭后重新打开该文件。",
                        "可以在 Script Editor 中保留多个加载片段；用根目录 README 的方式选择入口并点击 Run 切换场景。"
                        "该加载片段每次从磁盘读取源码，更新后重新运行片段即可。")
    elif relative.endswith("/TARGET_VOLUME.md"):
        text = _replace(text, r"C:\omninverse", "<PROJECT_ROOT>", count=2)
        text = _replace(text, "# 目标水量倒水\n", "# 目标水量倒水\n\n"
                        "> `<PROJECT_ROOT>` 表示本仓库目录。公开版默认使用无需下载贴图的 `white_matte` 桌面，"
                        "SDK 与项目路径配置见根目录 README。\n")
        text = _replace(text, "新入口：`<PROJECT_ROOT>\\robot_target_volume_pour.py`。",
                        "入口：项目根目录的 `robot_target_volume_pour.py`。")
        text = _replace(text,
                        "在 Isaac Sim 中选择 Window > Script Editor，在 File > Open 中打开新入口并 Run。",
                        "在 Isaac Sim 中选择 Window > Script Editor，使用[根目录 README 的加载片段](../../README.md#环境与启动)，"
                        "保留 `sys.path` 和真实 `__file__` 设置，将 `script` 指向 `project / \"robot_target_volume_pour.py\"` 后 Run。")
        text = _replace(text,
                        "文件顶部 `TARGET_VOLUME_ML = 30.0` 设置初始目标。",
                        "也可从仓库根目录在 PowerShell 中启动，把 SDK 路径改为实际安装位置：\n\n"
                        '```powershell\n$env:ISAAC_SIM_ROOT = "D:\\apps\\isaac-sim"\n'
                        '$env:ISAAC_FLUID_DEMO = "target-volume"\n'
                        '& "$env:ISAAC_SIM_ROOT\\isaac-sim.bat" --exec "$PWD\\launch.py"\n```\n\n'
                        "文件顶部 `TARGET_VOLUME_ML = 30.0` 设置初始目标。")
        text = _replace(text,
                        "原 `robot_testtube_pour.py` 的回放加速和 10 秒视频设置不应用于本例。",
                        "本例始终实时求解；旧 `robot_testtube_pour.py` 入口的公开默认也是 `LIQUID_MODE=\"simulate\"`，"
                        "其 `replay` 是需要另生成物理缓存的可选模式。旧入口的回放加速和 10 秒视频设置不应用于本例。")
    elif relative.endswith("/geometry.py"):
        _replace(text, '"""Pure NumPy geometry and prescribed motion for the robot pouring example.',
                 '"""Pure NumPy geometry and prescribed motion for the robot pouring example.')
    elif relative.endswith("/liquid.py"):
        _replace(text, '"""Fixed-particle GPU position-based liquid for the robot pouring example.',
                 '"""Fixed-particle GPU position-based liquid for the robot pouring example.')
    elif relative.endswith("/target_control.py"):
        _replace(text, '"""Feedback dosing controller; pure NumPy, no particle/state modifications.',
                 '"""Feedback dosing controller; pure NumPy, no particle/state modifications.')
    elif relative == "textures/table_options/sources.json":
        manifest = json.loads(text)
        if set(manifest) != {"gray_concrete", "pale_oak", "white_marble"}:
            raise SyncError("The CC0 manifest's asset allowlist changed; review the sync mapping.")
        if any(item.get("license") != "CC0 1.0" for item in manifest.values()):
            raise SyncError("Only the reviewed CC0 texture manifest can be copied.")
    else:
        raise SyncError(f"No publication mapping for {relative}.")
    if "C:\\omninverse" in text or re.search(r"(?i)[a-z]:\\Users\\", text):
        raise SyncError(f"Unmapped machine-specific path or username in {relative}.")
    if relative.endswith(".py"):
        ast.parse(text, filename=relative)
    return text.rstrip() + "\n"


def prepare_sync(workspace):
    """Validate the entire allowlist before returning any content to write."""
    root = Path(workspace).expanduser().resolve()
    prepared = []
    for relative in ALLOWLIST:
        source = (root / relative).resolve()
        if not source.is_relative_to(root) or not source.is_file():
            raise SyncError(f"Missing or unsafe allowlisted source: {relative}.")
        original = source.read_text(encoding="utf-8")
        try:
            published = transform(relative, original)
        except (ValueError, SyntaxError, SyncError) as error:
            raise SyncError(f"{relative}: {error}") from error
        prepared.append((relative, published.encode("utf-8")))
    return prepared


def _write_atomic(destination, content):
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(content)
        os.replace(temporary, destination)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument("--dry-run", action="store_true", help="Validate all anchors and list changes; write nothing.")
    args = parser.parse_args(argv)
    prepared = prepare_sync(args.workspace)
    changed = []
    for relative, content in prepared:
        destination = PROJECT_ROOT / relative
        if not destination.resolve().is_relative_to(PROJECT_ROOT):
            raise SyncError(f"Unsafe publication destination: {relative}.")
        if not destination.is_file() or destination.read_bytes() != content:
            changed.append((relative, content))
    if not args.dry_run:
        for relative, content in changed:
            _write_atomic(PROJECT_ROOT / relative, content)
    print(json.dumps({"dry_run": args.dry_run, "validated_files": len(prepared),
                      "changed_files": [name for name, _ in changed],
                      "sha256": {name: hashlib.sha256(content).hexdigest() for name, content in prepared}},
                     indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, SyncError, ValueError) as error:
        print(f"Sync aborted: {error}", file=sys.stderr)
        raise SystemExit(1)
