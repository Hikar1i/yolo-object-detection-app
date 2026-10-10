# 已知问题

## 文件监听达到 inotify 上限

2026-10-09 根目录 `app.log` 多次记录 `OSError: [Errno 28] inotify watch limit reached`，发生于 Streamlit 的 watchdog 文件监听注册阶段。该错误影响代码变更监听，日志未显示它导致推理失败；这里的 Errno 28 指监听资源耗尽。

建议在 `.streamlit/config.toml` 的 `[server]` 下配置 `fileWatcherType = "poll"`，避免占用 inotify 监听配额，修改后重启应用。也可由管理员排查监听占用并调整系统配额。当前已配置 `fileWatcherType = "poll"` 并重启服务，健康端点返回 `ok`。

## 运行目录写权限不足

2026-10-09 最初检查时，运行用户为 `ieds`，`models/`、`outputs/`、`logs/` 和 `logs/app.log` 属于 `root`，当时用户无写权限。

日志已记录文件日志创建失败，应用降级为控制台日志。代码中的视频保存与历史 CSV 写入也依赖上述目录；模型上传同样需要 `models/` 可写。应将这些运行目录及已有运行文件的所有权调整为实际启动用户，或给予该用户所需的写权限。用户已修复目录权限，本次复查确认上述目录和 `logs/app.log` 对 `ieds` 可写。

## 摄像头组件未安装

本机 `cu12-dev` 环境未提供 `streamlit-webrtc`，实时检测页面会显示依赖提示。图片、视频检测不依赖该组件；本次按用户要求未安装依赖。

## 页面语言与 API 标签

页面中文范围见 [使用指南](11-usage.md)。模型显示名称也用于 REST API 的 `/models` 返回键；权重标识、模型参数及历史 CSV 存储字段不变，详见 [API 接口](04-api.md)。

## 自动 GPU 设备名称不兼容（已修复）

真实模型上传推理验证发现 `get_best_device()` 自动检测 CUDA 后返回字符串 `"0"`，传给 PyTorch `model.to()` 时出现 `Invalid device string: '0'`。当前已统一自动 CUDA 返回值为 `"cuda:0"`，该值也可用于 Ultralytics 推理参数；已通过回归测试和真实上传权重 GPU 图片推理。

## Ultralytics half 参数弃用提示（已修复）

`cu12-dev` 当前使用 Ultralytics 8.4.83；显式传入 `half=False` 也会触发弃用警告。图片、批量和视频/跟踪统一通过 `InferenceConfig.precision_kwargs` 选择参数：支持 `quantize` 的版本用 `16` 表示 FP16、`None` 表示 FP32；旧版本仍传入 `half` 布尔值。公开的 `half_precision` 配置保持原语义。

项目 `requirements.txt` 锁定 Ultralytics 8.3.40，与本机环境不同；通过检测已安装库的配置字段兼容这两种接口，本次不更改依赖。


## 其他安装与运行问题

| 现象 | 排查与处理 |
|---|---|
| `ModuleNotFoundError: No module named 'src'` | 从仓库根目录执行命令；Web 入口已加入项目路径，其他脚本仍需正确工作目录。 |
| Linux 缺少 `libGL.so.1` | 原安装指南提供的系统依赖命令为 `sudo apt-get install libgl1 libglib2.0-0`；Dockerfile 已安装这些包。 |
| 模型首次下载失败 | 检查网络，或从 Ultralytics 官方 releases 获取可信 `.pt` 文件，通过侧栏上传或手动放入 `models/` 后选择本地模型。 |
| GPU 存在但 CUDA 不可用 | 检查驱动及 PyTorch CUDA 构建；步骤见 [环境与安装](08-environments.md)。 |

完整安装步骤见 [环境与安装](08-environments.md)，部署持久化与日志轮转见 [部署指南](07-deployment.md)。
