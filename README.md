# AI Caption (NeuralSub 智能字幕系统)

## 项目简介
本项目是一个基于 AI 的视频字幕生成与翻译系统，旨在为视频内容提供高质量的自动字幕生成及多语言翻译服务。系统集成了 OpenAI Whisper 进行语音转写，并结合神经机器翻译（NLLB）与大语言模型（Qwen）实现高精度的字幕翻译，同时引入了“反思模式”以提升翻译的自然度和准确性。

## 核心功能
- **自动语音转写 (ASR)**: 使用 Whisper 模型将视频/音频转换为文本。
- **智能翻译**: 
  - 支持 NLLB (No Language Left Behind) 进行基础翻译。
  - 集成 Qwen 大模型进行“反思式”翻译优化。
- **质量评估 (QE)**: 使用 SentenceTransformers (Bi-Encoder) 对翻译结果进行语义相似度评分。
- **多端支持**: 提供 Web 可视化界面 (Flask) 和命令行工具 (CLI)。
- **多格式输出**: 支持导出 SRT, VTT, JSON 等多种字幕格式。

## 项目结构

```
AI Caption/
├── app.py                  # Web 应用入口 (Flask)
├── cli.py                  # 命令行工具入口
├── config.py               # 项目配置文件
├── requirements.txt        # 项目依赖清单
├── install.sh              # 安装脚本
├── models/                 # AI 模型核心组件
│   ├── whisper_model_fixed.py  # Whisper ASR 模型封装
│   ├── translator.py           # 神经翻译器 (NLLB + Qwen)
│   ├── quality_estimator.py    # 翻译质量评估模块
│   └── finetune.py             # 模型微调脚本 (实验性)
├── utils/                  # 通用工具库
│   ├── audio_processor.py      # 音频处理与提取 (FFmpeg)
│   ├── subtitle_generator.py   # 字幕文件生成逻辑
│   └── file_handler.py         # 文件 I/O 管理
├── static/                 # Web 静态资源 (CSS, JS)
├── templates/              # Web 页面模板 (HTML)
├── output/                 # 默认输出目录
└── temp/                   # 临时文件目录
```

## 快速开始

### 环境要求
- Python 3.11
- FFmpeg (必须安装并配置环境变量)
- CUDA (推荐，用于 GPU 加速)
- torch版本：2.6.0+cu124

### 安装依赖
```bash
pip install -r requirements.txt
```

### 运行 Web 应用
```bash
python app.py
# 访问 http://localhost:5000
```


## 待办事项 (TODO)
- [ ] 升级至更大参数的模型以提升精度。
- [ ] 丰富视觉模块，利用画面信息辅助字幕生成。
- [ ] 建立合理的翻译性能评估体系。
- [ ] 探索迁移学习或微调方案以适应特定领域视频。
