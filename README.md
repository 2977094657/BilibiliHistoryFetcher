<div align="center">
  <img src="./logo.png" alt="BilibiliHistoryFetcher Logo" />
</div>

该项目用于获取、处理、分析和可视化哔哩哔哩用户的观看历史数据。它提供了完整的数据处理流程，从数据获取到可视化展示，并支持自动化运行和邮件通知。

## 配套前端请前往此项目[BiliHistoryFrontend](https://github.com/2977094657/BiliHistoryFrontend) 

## 主要功能

- [x] 获取历史记录
- [x] 年度总结
- [x] 视频和图片下载
- [x] 自动化任务
- [x] AI 摘要
- [x] 获取用户评论
- [x] 获取收藏夹
  - [x] 批量收藏
  - [x] 修复失效视频
  - [x] 一键下载收藏夹所有视频

## 系统要求

- Python 3.10+
- SQLite 3
- FFmpeg（用于视频下载）
- 必要的 Python 包（见 requirements.txt）
- NVIDIA GPU（可选，用于加速音频转文字）
  - 支持 CUDA 9.0 及以上版本
  - 建议使用 CUDA 11.8 或更高版本以获得最佳性能
  - 如果没有 GPU，将自动使用 CPU 模式运行

## 快速开始

### 使用 Docker 安装

1. 安装[Docker](https://docs.docker.com/get-started/get-docker/)。
2. 根据你使用的系统构建 Docker 镜像：
   1. 如果你使用 NVIDIA 显卡，并配置好了 Docker 驱动，请使用`docker build -t bilibili-api:dev -f docker/Dockerfile.cuda .`构建镜像。
   2. 否则，请使用`docker build -t bilibili-api:dev -f docker/Dockerfile.cpu .`构建镜像。
3. 根据你使用的镜像创建 Docker 容器：
   1. 如果你是用`cuda`镜像，则使用`docker run -d -v ./config:/app/config -v ./output:/app/output -p 8899:8899 --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --name bilibili-api bilibili-api:dev`创建容器。
   1. 否则，请使用`docker run -d -v ./config:/app/config -v ./output:/app/output -p 8899:8899 --name bilibili-api bilibili-api:dev`创建容器。

### 使用`uv`安装

首先，根据[官方文档](https://docs.astral.sh/uv/getting-started/installation/)安装``uv`。

在项目根目录下运行`uv sync`。

如果你是用 NVIDIA 显卡并希望使用 CUDA 加速推理，请使用如下的指令安装`pytorch`：

```bash
UV_TORCH_BACKEND=auto uv pip install torch torchaudio torchvision
```

否则，使用如下的指令安装`pytorch`：

```bash
UV_TORCH_BACKEND=cpu uv pip install torch torchaudio torchvision
```

最后，使用`uv run main.py`运行程序。

### 使用源码安装

1. **安装依赖**

项目提供了自动化的依赖安装脚本，可以自动检测您的系统环境并安装合适的依赖：

```bash
python install_dependencies.py
```

该脚本会：
- 安装基本依赖（requirements.txt 中的包）
- 检测系统 CUDA 环境
- 安装适配您系统的 PyTorch 版本
- 安装音频转文字所需的依赖（faster-whisper 等）

您也可以使用以下选项：
```bash
# 查看系统环境信息和CUDA兼容性
python install_dependencies.py --info

# 强制使用CPU版本（不使用GPU加速）
python install_dependencies.py --force-cpu

# 指定CUDA版本安装
python install_dependencies.py --force-cuda 12.7

# 跳过PyTorch相关依赖安装
python install_dependencies.py --skip-torch
```

音频转文字功能依赖说明：
- 如果您的系统有 NVIDIA GPU，脚本会自动安装 GPU 加速版本的依赖
- 如果没有检测到 GPU，会安装 CPU 版本（性能较低但仍可使用）
- 所需的主要依赖包括：
  - PyTorch：深度学习框架
  - faster-whisper：优化版语音识别模型
  - huggingface-hub：模型下载和管理

详细的 CUDA 和 PyTorch 版本对应关系请参考 [CUDA_SETUP.md](CUDA_SETUP.md)

2. **配置文件**

在 `config/config.yaml` 中配置以下信息：
```yaml
# B 站用户认证
SESSDATA: "你的 SESSDATA"

# 邮件通知配置
email:
  smtp_server: smtp.qq.com
  smtp_port: 587
  sender: "发件人邮箱"
  password: "邮箱授权码"
  receiver: "收件人邮箱"

# DeepSeek AI API 配置
deepseek:
  api_key: "你的 DeepSeek API 密钥"
  api_base: "https://api.deepseek.com/v1"
  default_model: "deepseek-reasoner"

# 服务器配置
server:
  host: "localhost"
  port: 8899
```


3. **运行程序**
```bash
python main.py
```

## API 接口

基础 URL: `http://localhost:8899`

完整 API 文档访问：`http://localhost:8899/docs`

## 自动化任务配置

在 `config/scheduler_config.yaml` 中配置定时任务：
```yaml
tasks:
  fetch_history:
    schedule:
      type: "daily"
      time: "00:00"
  
  analyze_data:
    schedule:
      type: "chain"
    requires:
      - fetch_history
```

> **注意**：当程序打包为可执行文件后，计划任务配置文件位于 `_internal/config/scheduler_config.yaml`。

## 应用打包

本项目提供了自动化打包脚本，可以将应用打包成独立的可执行文件，便于分发和部署。打包过程会自动处理依赖并清理敏感信息。

**打包前准备**

确保已经安装了 PyInstaller：

```bash
pip install pyinstaller
```

**打包命令**

使用以下命令进行打包：

```bash
# 打包精简版（不含torch，不支持音频转文字功能）
python build.py lite

# 打包完整版（含torch，支持所有功能）
python build.py full

# 同时打包两个版本
python build.py all
```

**打包版本说明**

1. **精简版 (Lite)**
   - 不包含 PyTorch 和音频转文字相关模块
   - 体积较小，适合只需基本功能的用户
   - 不支持音频转文字和本地视频摘要功能
   - 输出目录：`dist/BilibiliHistoryAnalyzer_Lite/`

2. **完整版 (Full)**
   - 包含所有功能，包括 PyTorch 和音频转文字
   - 体积较大，适合需要完整功能的用户
   - 支持所有功能，包括音频转文字和本地视频摘要
   - 输出目录：`dist/BilibiliHistoryAnalyzer_Full/`

**敏感信息处理**

打包过程会自动处理配置文件中的敏感信息：

- 创建临时清理过的配置文件，替换以下敏感字段为示例值：
  - `SESSDATA`：替换为"你的 SESSDATA"
  - `email`：邮箱相关信息替换为示例邮箱地址和说明
  - `ssl_certfile`/`ssl_keyfile`：替换为示例路径
  - `api_key`：替换为"你的 API 密钥"
  
- 打包完成后，临时文件会被自动删除
- 打包版本中的配置文件包含原始结构但敏感字段被替换为示例值，用户需要首次运行时填写实际信息

**运行打包后的应用**

在目标系统上直接运行可执行文件：

```
# Windows系统
BilibiliHistoryAnalyzer_Full.exe
```

首次运行注意事项：

1. 打开 `_internal/config/config.yaml` 文件
2. 将示例值替换为您的实际配置信息：
   - 填写您的 B 站 `SESSDATA`
   - 配置邮箱信息（如需使用邮件通知功能）
   - 配置 DeepSeek API 密钥（如需使用 AI 摘要功能）
3. 保存配置文件后重新启动应用

## 贡献指南

1. Fork 项目
2. 创建特性分支
3. 提交更改
4. 发起 Pull Request

## 许可证

MIT License

## 致谢

- [bilibili-API-collect](https://github.com/SocialSisterYi/bilibili-API-collect) - 没有它就没有这个项目
- [Yutto](https://yutto.nyakku.moe/) - 可爱的 B 站视频下载工具
- [FasterWhisper](https://github.com/SYSTRAN/faster-whisper) - 音频转文字
- [DeepSeek](https://github.com/deepseek-ai/DeepSeek-R1) - DeepSeek AI API
- 所有贡献者

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=2977094657/BilibiliHistoryFetcher&type=Date)](https://star-history.com/#2977094657/BilibiliHistoryFetcher&Date)
