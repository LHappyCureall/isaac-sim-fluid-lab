# 20 / 30 / 40 mL：第一轮学习策略

入口：项目根目录的 `robot_target_volume_ai.py`。结果与模型选择见 `results/ai_training.json`；冻结检查点准备好后，在 Isaac Sim Script Editor 中使用[根目录 README 的加载片段](../../README.md#环境与启动)，保留 `sys.path` 与真实 `__file__`，将 `script` 指向该入口，选择目标水量后按 Play。

- **Learned candidate / Baseline (validated)**：切换已训练候选与原控制器，切换会复位场景。默认使用验证集选定的策略；若学习策略未胜出，面板会如实标明选择了基准，仍可切换候选做实验。
- **Blue water / Clear water**：蓝色提高水流对比度；清水保留原折射外观。两种模式使用同一液体物理。
- 蓝色模式额外显示水面重建容易漏掉的真实空中粒子，球心来自当前求解器位置、直径为一个粒子间距；这些仅是显示标记，不是新增水量。
- 面板显示新增入杯量、目标误差、空中余流和流速。橙线为目标、青线为实际水量对应的等效静态液位，不通过画面估水量。

## 训练什么

采用 CEM（交叉熵方法）优化三个反馈参数：固定余流补偿、流速提前时间，以及目标水量修正。观测来自当前液体状态；策略每个物理帧判断继续倾倒或扶正。机械臂仍使用已有的运动学预握和安全路径。

每个候选策略都重新运行实时 GPU PBF，不读取液体回放缓存。固定物理步长为 1/60 s，并保留原求解器的自适应子步和压力迭代。减少渲染或合并 CUDA 调用不会改变训练数据的物理时间；10 秒展示视频的时间映射不参与训练。

从仓库根目录的 PowerShell 运行一轮训练，并将 `ISAAC_SIM_ROOT` 改为实际 SDK 安装位置：

```powershell
$env:ISAAC_SIM_ROOT = "D:\apps\isaac-sim"
& "$env:ISAAC_SIM_ROOT\kit\python\python.exe" "$PWD\train_target_volume_policy.py" --quick --graph --output "$PWD\droplet_out\pour_policy_training\run_quick"
```

快速模式为 18 个训练回合、6 个验证回合和 12 次配对测试执行；测试覆盖 6 个目标×种子条件、2 个随机种子。去掉 `--quick` 可扩大预算；用 `--output` 指定新实验目录，`--generations` / `--population` 设置预算；已有实验通过 `--resume` 续跑。先运行 `--benchmark` 可检查环境与单回合耗时。具体回合数以每次的 `protocol.json` 与 `summary.json` 为准。

GUI 默认使用原始逐步调度。`--graph` 仅用于已验证等价的无渲染批量运行；与表面重建交错运行尚未验证通过。

默认记录目录：`droplet_out/pour_policy_training/`。保存逐物理帧的观测/动作/时间压缩日志、每回合指标、搜索过程、冻结模型、训练曲线数据与独立测试报告。默认部署模型位于 `assets/robot_testtube_pour/policy/target_volume_policy.json`；单独保存的学习候选为同目录 `target_volume_candidate.json`。

## 水量精度

每粒体积为 **0.042875 mL**。20 / 30 / 40 mL 最近可实现的粒子水量分别为 **19.979750 / 30.012500 / 40.002375 mL**，所以不能承诺数学上的零误差。数值是固定粒子质量对应的仿真真值，尚未实现摄像机估水、真实称重校准或接触力抓持。

训练、验证和测试种子预先分开；模型冻结后才做独立测试。报告保留未达标、洒水和超时案例，不把训练成绩当成独立测试成绩。

首轮共完成 42 次批量执行（18 训练、6 验证、18 测试，其中 6 次为冻结候选的补充对比）。默认反馈控制在 6 个留出条件上的平均绝对误差为 **0.0824 mL**、最大 **0.2346 mL**；训练候选为 **0.1693 / 0.3883 mL**，未胜出，因此默认保留基准。完整物理时间、动作和质量检查记录在 `droplet_out/pour_policy_training/run_20261121_quick/`。
