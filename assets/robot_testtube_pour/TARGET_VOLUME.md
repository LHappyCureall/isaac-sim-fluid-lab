# 目标水量倒水

> `<PROJECT_ROOT>` 表示本仓库目录。公开版默认使用无需下载贴图的 `white_matte` 桌面，SDK 与项目路径配置见根目录 README。

入口：项目根目录的 `robot_target_volume_pour.py`。

在 Isaac Sim 中选择 Window > Script Editor，使用[根目录 README 的加载片段](../../README.md#环境与启动)，保留 `sys.path` 和真实 `__file__` 设置，将 `script` 指向 `project / "robot_target_volume_pour.py"` 后 Run。初始化后会出现 **Target volume pouring** 面板。选择 **20 mL / 30 mL / 40 mL**，再点击 Play。Pause 暂停；Reset 或主工具栏 Stop 恢复初始水量和姿态。切换目标也会复位。

也可从仓库根目录在 PowerShell 中启动，把 SDK 路径改为实际安装位置：

```powershell
$env:ISAAC_SIM_ROOT = "D:\apps\isaac-sim"
$env:ISAAC_FLUID_DEMO = "target-volume"
& "$env:ISAAC_SIM_ROOT\isaac-sim.bat" --exec "$PWD\launch.py"
```

文件顶部 `TARGET_VOLUME_ML = 30.0` 设置初始目标。`TABLE_STYLE` 设置桌面；`VOLUME_TOLERANCE_ML` 设置测试的允许误差，不改变控制目标。`Test 20 / 30 / 40 mL` 会依次运行三个独立测试，并自动复位。Cancel test 或 Stop 可中止批量测试。

本例使用实时 GPU PBF 液体求解，每一帧根据试管来源的粒子在烧杯内、试管内和空中的位置反馈控制。控制器估计空中水落入烧杯的量和扶正前的余流，提前减小倾角并自动扶正，最后等待已流出的水落稳后计算误差。这里的目标量是**新倒入烧杯的水量**，不包含烧杯原有的水。水量按粒子体积估算；真实实验精度需要实测校准。

机器人继续采用预握试管与运动学控制。控制器沿已有、已检查的空间路径前进或回退，机器人和试管同步；液体响应由当前运动重新求解。它是反馈控制基准，尚未训练神经网络策略，也没有模拟接触摩擦抓持或液体力矩反馈。

实时求解耗时取决于机器性能，面板中的秒数为仿真时间。本例始终实时求解；旧 `robot_testtube_pour.py` 入口的公开默认也是 `LIQUID_MODE="simulate"`，其 `replay` 是需要另生成物理缓存的可选模式。旧入口的回放加速和 10 秒视频设置不应用于本例。

输出目录：`<PROJECT_ROOT>\droplet_out\robot_target_volume_pour`。每个目标保存 `target_20ml.json` 等报告；批量测试还保存 `validation.json` 和各目标完成时的截图。检查项目包括最终水量误差、扶正、洒水、穿桌、粒子数与质量守恒、实际手爪与试管的夹持位置/姿态误差，以及复位。

新例子复用 `robot_testtube_pour.py` 与同目录脚本、Franka 和材质资源。三个水滴/倒水入口共享场景互斥机制：运行一个入口会停止旧例子的回调和测试任务。

Isaac Sim 界面内批量测试结果：20 mL → 19.936875 mL；30 mL → 29.755250 mL；40 mL → 39.745125 mL。三次均完成扶正，未检测到洒水、未落稳的空中水或穿桌；5330 个粒子及总质量保持不变，复位误差为 0。逐帧实际手爪与试管 TCP 最大位置误差约 0.057 mm，最大姿态误差约 0.00057 rad。详情以输出目录的 `validation.json` 为准。
