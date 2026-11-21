# 外部依赖与资源

- **NVIDIA Isaac Sim / Omniverse Kit**：外部安装依赖，使用其配套 Python、扩展与运行库；这些安装文件不随本仓库分发。参考 <https://docs.isaacsim.omniverse.nvidia.com/>。
- **Newton / NVIDIA Warp**：使用 Isaac Sim 配套版本。参考 <https://github.com/newton-physics/newton>、<https://github.com/NVIDIA/warp>。
- **Franka description**：运行时使用 Isaac Sim URDF importer 附带的 `franka_description`；本机包声明 Apache 2.0。转换结果和第三方模型未随源码发布，使用时应保留其原有许可。
- **可选桌面贴图**：来源、CC0 声明、文件名与 SHA-256 在 `textures/table_options/sources.json`。下载后仍可追溯至 Poly Haven / ambientCG 的原始页面。
- **FFmpeg / FFprobe**：可选视频导出依赖，需自行安装，二进制不随本仓库发布。

本仓库的实验结果不代表上述项目对实验精度或机器人真机部署效果的认证。
