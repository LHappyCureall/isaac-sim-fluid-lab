# 自适应定量倒水：官方方法与本项目策略

核对日期：2026-11-27。官方源码固定到 `release/3.0.0` 的 `39394453ceaac256349ad822e675a93485373667`，不把不断变化的 main、标签及网页示例拼成同一个环境。本页是方法决定和准备工作，未训练新模型、未切换现有物理或 GUI。

## 官网实际训练什么

| 部分 | 官方 Franka Pour | 本项目需要的改动 |
| --- | --- | --- |
| 学习 | RSL-RL PPO，每步动作反馈 | 当前先保留此前不换PPO的约束，学习小动作响应并做保守滚动决策 |
| 动作 | 7关节位置增量及夹爪，30Hz | 保留已验证预握/运输，只学习倾角小动作、保持及收流决策 |
| 观测 | 目标比例、剩余时间、机器人/容器状态及粒子摘要 | 增加目标mL、细尺度误差、实测落稳增量与回家时间预算 |
| 成功 | 达到初始载荷的目标比例后结束，默认70% | 必须双向限制超量/欠量，并完整回家、稳定、检查晚滴 |

上述来自[官方PPO配置](https://github.com/isaac-sim/IsaacLab/blob/39394453ceaac256349ad822e675a93485373667/source/isaaclab_tasks/isaaclab_tasks/contrib/franka_pour/config/franka/agents/rsl_rl_ppo_cfg.py)、[动作与观测配置](https://github.com/isaac-sim/IsaacLab/blob/39394453ceaac256349ad822e675a93485373667/source/isaaclab_tasks/isaaclab_tasks/contrib/franka_pour/pour_env_cfg.py)、[终止条件](https://github.com/isaac-sim/IsaacLab/blob/39394453ceaac256349ad822e675a93485373667/source/isaaclab_tasks/isaaclab_tasks/contrib/franka_pour/mdp/terminations.py)。目标比例虽是观测，默认reset仍使用固定配置值；不能把“有目标输入”说成已经学会随机任意mL。[reset代码](https://github.com/isaac-sim/IsaacLab/blob/39394453ceaac256349ad822e675a93485373667/source/isaaclab_tasks/isaaclab_tasks/contrib/franka_pour/mdp/reset_dataset.py)。

官方14阶段、约两万个初态构成reset课程，液体运行时仍实时求解。采样器记每行最近50次安全局部进展，优先级为 `sqrt(p*(1-p))+epsilon`，并保留50%循环覆盖；局部进展不等于最终成功，也不直接充当actor奖励。[官方学习契约](https://github.com/isaac-sim/IsaacLab/blob/39394453ceaac256349ad822e675a93485373667/source/isaaclab_tasks/isaaclab_tasks/contrib/franka_pour/README.md)、[reset采样器](https://github.com/isaac-sim/IsaacLab/blob/39394453ceaac256349ad822e675a93485373667/source/isaaclab_tasks/isaaclab_tasks/contrib/franka_pour/reset_sampler.py)。

## 先把物理基线分清

官网MPM是粒子—网格材料求解，刚体双向作用另由耦合器实现；文档明确这部分仍属实验功能。[MPM说明](https://docs.nvidia.com/isaaclab/concepts/implicit-mpm)、[耦合说明](https://docs.nvidia.com/isaaclab/concepts/coupled-solvers)。官方任务的默认材料是有摩擦的颗粒介质，不应直接称作校准过的水；粒子质量要读实际生成结果，不能仅把标称spacing立方当作精度。[材料配置](https://github.com/isaac-sim/IsaacLab/blob/39394453ceaac256349ad822e675a93485373667/source/isaaclab_tasks/isaaclab_tasks/contrib/franka_pour/pour_env_cfg.py)、[生成器质量计算](https://github.com/isaac-sim/IsaacLab/blob/39394453ceaac256349ad822e675a93485373667/source/isaaclab_newton/isaaclab_newton/sim/spawners/mpm/mpm.py)。

本项目 `liquid.py:450` 当前调用自写PBF kernels，没有调用 `self.solver.step`。创建了 `SolverImplicitMPM` 对象不代表实际用MPM推进。当前试管运动也是规定的运动学状态，不能由此声称机械臂感受到了水的质量变化。V9/V10仅是这个PBF基线的结果，不能改名为官网MPM训练。

本机Isaac Sim 6.2 RC14、Python3.12、Newton1.6.1、Warp1.17的核心版本接近官方基线，但未找到完整Isaac Lab任务/RSL-RL训练栈。后续若做官网对照，单独建立固定版本环境；先单环境验证真MPM推进、完整reset、容器包含和耦合，再增加并行数。保持现有SDK和GUI，不直接把示例2048环境数用于本机。数值应先检查分辨率、步长和收敛，再解释材料或策略效果。[官方数值调试顺序](https://docs.nvidia.com/isaaclab/3.0/isaac-lab/mpm-solver)。

## 本轮确定的学习策略

**目标条件化 + 自适应课程 + 实际落稳响应模型 + 受约束滚动控制。** PPO保留为后续独立对照，不默默替换当前算法。官方PPO与本方案是两种算法实现，不能把新采样工具称作已训练的PPO策略。

1. **给模型明确目标。** 输入 `g=目标mL/初始源水量`，以及粗误差 `(目标-已收)/初始源水量`、精误差 `(目标-已收)/(0.001*目标)`。精误差可为特征限幅，但验收始终使用未限幅的原始误差。固定56.68704mL源水时，20–40mL对应目标比例约0.352814–0.705629。
2. **学习真实动作响应。** 训练样本为“已落稳状态、倾角/角速度、剩余源水量、前次动作、时间预算、小动作”到“实际落稳增量、持续时间、回撤余量、洒水风险”。先用可解释的小模型和不确定度，数据不足时不直接训练高容量关节策略。当前静水几何只作候选角建议，不能代替这个实测模型。
3. **约束下选下一动作。** 从实测支持的小角移动、保持、收流/回家组合中选择动作；留足回家时间，限制关节速度/加速度，拒绝缺乏数据支持的激进动作。模型负责建议动作，补水和最终停表仍由实际截止流动后的落稳复测决定，不采用固定 `Q*0.5s` 判成功。
4. **从关键阶段扩展。** 先学习接近出水、少量补水、截止及回撤的可控响应，再覆盖完整预握到回家。训练初态只能来自可复现的真实物理轨迹/完整状态reset；不能只恢复关节位置就假定液体已恢复。阶段训练须记录原已消耗时间，最终评估一律从预握重跑全周期。
5. **借鉴课程采样，不回放水。** 对不同目标/剩余量/阶段的合法reset行保留覆盖，并多采样安全局部进展概率接近中间的行；课程统计独立于严格成功。冻结验证选出的模型后，测试期间不跨案例继续训练。

当前新增的[CPU课程与特征工具](../tools/training/adaptive_curriculum.py)只实现第1项和第5项的可复用部分；它不是液体reset writer、响应模型、完整环境或已经训练好的策略。

## 验收与下一阶段门槛

保持24原始物理秒、目标±0.1%、全标签洒水峰值≤1粒、原home及12帧稳定；另观察2秒，晚滴可判失败但不能救回deadline。量取固定粒子质量/密度的仿真真值，后续真实秤/视觉方案须另验证噪声、延迟和分辨率。

所有21个整数目标逐项测试，并保留未用于开发/验证的小数目标和新初态。整数已参加验证，只能称新初态覆盖；小数才用于未见目标泛化。任一案例失败就不能报告全覆盖通过，也不能由有限测试保证所有连续目标与所有环境。

V10四组工程共12次均失败，[精选记录](V10_ENGINEERING_20261127.md)已保留；原180回合只搜索粗倒倍率，工程门槛没有通过，**未启动**。下一阶段先证明稳定的小动作响应和环境reset，而不是直接增加旧CEM回合数。实际更换MPM、动作空间或课程reset均须新协议、新来源快照、新留出集，旧结果只作历史基线。
