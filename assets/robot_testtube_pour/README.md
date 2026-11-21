# 机器人用试管向烧杯倒水

> 公开仓库默认 `white_matte` 桌面与 `simulate` 实时求解；原实验的回放、视频与节奏说明适用于可选物理缓存。Franka 从 SDK 本地转换，缓存不随源码提供。`<PROJECT_ROOT>` 表示本仓库目录，启动配置见根目录 README。

入口：项目根目录的 `robot_testtube_pour.py`。

在 Isaac Sim 的 Window > Script Editor 中使用[根目录 README 的加载片段](../../README.md#环境与启动)，保留其中的 `sys.path` 和真实 `__file__` 设置，并将 `script` 改为 `project / "robot_testtube_pour.py"`，再点击 Run。初始化完成后，在 Robot pouring 面板中点击 Play。Pause 暂停，Reset 恢复初始水和机器人；Overview / Pour close-up 切换视角。先生成可选物理缓存并切换到 `replay`，才可点击 Export 10s MP4 生成短片。也可使用主工具栏的播放、暂停、停止按钮。

也可在 PowerShell 中设置独立的 SDK 安装目录，并从仓库根目录启动。把下方 SDK 路径改为实际安装位置：

```powershell
$env:ISAAC_SIM_ROOT = "D:\apps\isaac-sim"
$env:ISAAC_FLUID_DEMO = "pour"
& "$env:ISAAC_SIM_ROOT\isaac-sim.bat" --exec "$PWD\launch.py"
```

机器人预先侧握试管，抬起、移至烧杯上方、倾倒并扶正，公开版默认实时求解，仿真时间约 15.6 秒，扶正后结束。可选缓存回放约 9.85 秒；回放中的倒水后等待为 0.25 秒，扶正用时 1.2 秒。试管初始标称水量约 51 mL，烧杯中原有一层水。使用官方离线 Franka 资源与 RobotPoser 逆运动学；机械臂和夹持关系为运动学控制，不包含摩擦抓持、液体反力或力矩反馈。

水使用固定数量的 GPU PBF 粒子：重力、邻域压力约束、黏性及开放玻璃容器边界决定流动，不使用发射器或将水强制拉回烧杯的规则。它是可视化原型，并未校准为定量流量实验。

入口顶部可调整：

- `TABLE_STYLE`：`pale_oak`、`white_marble`、`gray_concrete`、`white_matte`。
- `LIQUID_MODE = "simulate"`：公开版默认，实时求解液体，不需要物理缓存。修改水粒子间距、运动时长或容器几何后也使用此模式。
- `LIQUID_MODE = "replay"`：可选，先按[根目录 README](../../README.md#缓存桌面与视频)生成 `pour_cache.npz`，再回放经过物理计算的 60 Hz 粒子轨迹，实时重建水面，减少卡顿。缓存不随源码发布。
- `RENDER_MODE = "RealTimePathTracing"`：RTX Real-Time 2.0；暂停后可在软件中切换 RTX Interactive 检查静态效果。
- `STOP_AFTER_UPRIGHT = True`：扶正试管后结束；改为 False 可保留最后返回起点的动作。
- `POST_POUR_HOLD_SECONDS = 0.25`：倒水后的回放等待秒数，调小可减少静止时间。
- `UPRIGHT_PLAYBACK_SPEED = 2.0`：扶正的回放速度倍率，越大越快；`RETURN_PLAYBACK_SPEED = 1.6` 控制可选的返回起点动作。
- 上述节奏同时用于普通回放和 MP4，机器人、试管、水同步采样。物理缓存及其运动学验证时间不变；`simulate` 模式仍按原始物理时间逐步求解，默认需 15.6 秒。
- `VIDEO_SECONDS = 10.0`、`VIDEO_FPS = 30`：控制 MP4 播放时长和帧率。默认 300 帧，播放恰好 10 秒。短片压缩倾倒后的等待，在扶正后结束。

MP4 输出：`<PROJECT_ROOT>\droplet_out\robot_testtube_pour\robot_pour_10s.mp4`，编码检查记录为同目录的 `video_export.json`。导出当前相机视角；先选择 Overview 或 Pour close-up。导出使用暂停的缓存逐帧采样，确保截图时额外的渲染更新不会推进动作。PNG 中间帧在编码成功或失败后自动清理。需要本机已安装的 FFmpeg/FFprobe。

播放时长 = 总帧数 ÷ 输出 FPS。Movie Capture 的帧范围包含末帧，因此手动录制 10 秒、30 FPS 时应使用 0–299；动画采样 FPS 也应为 30。在已准备好缓存的 `replay` 模式下推荐使用本示例的 Export 按钮，它还会将完整倒水与扶正动作剪辑到这 10 秒中。实际导出耗时取决于逐帧渲染速度，与成片播放时长不同。

`scripts/` 提供几何、机器人、液体、反馈控制与视频导出实现。`franka/` 是首次运行时从 SDK 自带 URDF 转换生成的本地缓存；可选的 `pour_cache.npz` 是物理轨迹，不是视频，可通过 `tools/bake_pour_cache.py` 生成。两类缓存均不进入 Git。材质复用 milkcrown surface 脚本，默认纯色桌面无需贴图。

验证结果保存在 `<PROJECT_ROOT>\droplet_out\robot_testtube_pour`。物理测试使用 5330 个粒子；试管中的 1198 个粒子最终有 1197 个进入烧杯（99.92%），1 个落在桌面，无穿桌。固定粒子数量和总质量，检查直立水柱保持，并通过复位重算检查。机器人完整轨迹逐帧检查末端误差及 URDF 速度限制。

本次实现中的两个修正：原 MPM 在细试管中出现严重压缩，因此使用 PBF 压力约束；机器人跟踪关闭持续零空间回中，避免初始关节快速调整。

水面采用原生 particle_sdf 重建，减少进入玻璃内部的外扩。玻璃杯底与桌面保留 0.3 mm 的光学间距，避免两张面完全重合导致白色折射伪影；杯子内腔与液体碰撞几何维持原尺寸。

与 milkcrown 切换时，两个入口会先停止旧例子的播放回调、取消旧的构建和视频任务，并清除旧例子的几何与控制面板。可以在 Script Editor 中保留多个加载片段；用根目录 README 的方式选择入口并点击 Run 切换场景。该加载片段每次从磁盘读取源码，更新后重新运行片段即可。
