# 机器人用试管向烧杯倒水

> 公开仓库默认 `white_matte` 桌面与 `simulate` 实时求解；原实验的回放、视频与节奏说明适用于可选物理缓存。Franka 从 SDK 本地转换，缓存不随源码提供。`<PROJECT_ROOT>` 表示本仓库目录，启动配置见根目录 README。

入口：`<PROJECT_ROOT>\robot_testtube_pour.py`。

在 Isaac Sim 的 Window > Script Editor 中打开入口文件，点击 Run。初始化完成后，在 Robot pouring 面板中点击 Play。Pause 暂停，Reset 恢复初始水和机器人；Overview / Pour close-up 切换视角。点击 Export 10s MP4 直接生成短片。也可使用主工具栏的播放、暂停、停止按钮。

也可在 PowerShell 中从安装目录启动：

```powershell
cd <PROJECT_ROOT>
.\isaac-sim.bat --exec <PROJECT_ROOT>\robot_testtube_pour.py
```

机器人预先侧握试管，抬起、移至烧杯上方、倾倒并扶正，默认回放约 9.85 秒，扶正后结束。倒水后的等待为 0.25 秒，扶正用时 1.2 秒。试管初始标称水量约 51 mL，烧杯中原有一层水。使用官方离线 Franka 资源与 RobotPoser 逆运动学；机械臂和夹持关系为运动学控制，不包含摩擦抓持、液体反力或力矩反馈。

水使用固定数量的 GPU PBF 粒子：重力、邻域压力约束、黏性及开放玻璃容器边界决定流动，不使用发射器或将水强制拉回烧杯的规则。它是可视化原型，并未校准为定量流量实验。

入口顶部可调整：

- `TABLE_STYLE`：`pale_oak`、`white_marble`、`gray_concrete`、`white_matte`。
- `LIQUID_MODE = "replay"`：默认，回放经过物理计算的 60 Hz 粒子轨迹，实时重建水面，减少卡顿。
- `LIQUID_MODE = "simulate"`：重新实时求解液体，计算量更大。修改水粒子间距、运动时长或容器几何后使用此模式。
- `RENDER_MODE = "RealTimePathTracing"`：RTX Real-Time 2.0；暂停后可在软件中切换 RTX Interactive 检查静态效果。
- `STOP_AFTER_UPRIGHT = True`：扶正试管后结束；改为 False 可保留最后返回起点的动作。
- `POST_POUR_HOLD_SECONDS = 0.25`：倒水后的回放等待秒数，调小可减少静止时间。
- `UPRIGHT_PLAYBACK_SPEED = 2.0`：扶正的回放速度倍率，越大越快；`RETURN_PLAYBACK_SPEED = 1.6` 控制可选的返回起点动作。
- 上述节奏同时用于普通回放和 MP4，机器人、试管、水同步采样。物理缓存及其运动学验证时间不变；`simulate` 模式仍按原始物理时间逐步求解，默认需 15.6 秒。
- `VIDEO_SECONDS = 10.0`、`VIDEO_FPS = 30`：控制 MP4 播放时长和帧率。默认 300 帧，播放恰好 10 秒。短片压缩倾倒后的等待，在扶正后结束。

MP4 输出：`<PROJECT_ROOT>\droplet_out\robot_testtube_pour\robot_pour_10s.mp4`，编码检查记录为同目录的 `video_export.json`。导出当前相机视角；先选择 Overview 或 Pour close-up。导出使用暂停的缓存逐帧采样，确保截图时额外的渲染更新不会推进动作。PNG 中间帧在编码成功或失败后自动清理。需要本机已安装的 FFmpeg/FFprobe。

播放时长 = 总帧数 ÷ 输出 FPS。Movie Capture 的帧范围包含末帧，因此手动录制 10 秒、30 FPS 时应使用 0–299；动画采样 FPS 也应为 30。默认推荐直接使用本示例的 Export 按钮，它还会将完整倒水与扶正动作剪辑到这 10 秒中。实际导出耗时取决于逐帧渲染速度，与成片播放时长不同。

`scripts/` 中四个模块分别提供几何、机器人、液体和视频导出实现。`franka/` 是官方 URDF 转换后的必要资源；`pour_cache.npz` 是完整物理轨迹，不是视频。材质复用现有 milkcrown surface 脚本及桌面纹理。

验证结果保存在 `<PROJECT_ROOT>\droplet_out\robot_testtube_pour`。物理测试使用 5330 个粒子；试管中的 1198 个粒子最终有 1197 个进入烧杯（99.92%），1 个落在桌面，无穿桌。固定粒子数量和总质量，检查直立水柱保持，并通过复位重算检查。机器人完整轨迹逐帧检查末端误差及 URDF 速度限制。

本次实现中的两个修正：原 MPM 在细试管中出现严重压缩，因此使用 PBF 压力约束；机器人跟踪关闭持续零空间回中，避免初始关节快速调整。

水面采用原生 particle_sdf 重建，减少进入玻璃内部的外扩。玻璃杯底与桌面保留 0.3 mm 的光学间距，避免两张面完全重合导致白色折射伪影；杯子内腔与液体碰撞几何维持原尺寸。

与 milkcrown 切换时，两个入口会先停止旧例子的播放回调、取消旧的构建和视频任务，并清除旧例子的几何与控制面板。两个文件可以同时在 Script Editor 中打开；选择目标文件并点击 Run 才会切换场景。磁盘文件更新后，旧标签不会自动读取新代码，应关闭后重新打开该文件。
