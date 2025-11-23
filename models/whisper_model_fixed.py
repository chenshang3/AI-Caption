import whisper
import os
import logging
import torch
import numpy as np
import math
from typing import Optional, Dict, Any, List, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed

# 修复：使用相对导入以确保在 models 包结构中能够正确找到 VLMSceneAnalyzer
try:
    from .vlm_analyzer import VLMSceneAnalyzer
except ImportError:
    # 仅在 VLM 模块未找到时记录警告，以便 Whisper 仍能运行（非视频文件场景）
    logging.warning("VLMSceneAnalyzer module not found. Video analysis will be skipped.")
    VLMSceneAnalyzer = None

# 确保日志配置正确
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class WhisperTranscriber:
    """
    负责加载 Whisper ASR 模型，并协调 VLMSceneAnalyzer 进行音视频联合分析。
    """

    def __init__(self, model_name: str = "medium", device: str = "cpu"):
        self.model_name = model_name
        self.whisper_device = device
        self.model = None
        self.vlm_analyzer = None

        self.frames_per_minute = 2  # 目标每分钟采样帧数
        self.max_frames_to_process = 180  # 硬性上限 (防止失控)

        self.load_whisper_model()
        self.load_vlm_component()
        if self.whisper_device == "cuda":
            torch.cuda.empty_cache()

    def load_whisper_model(self):
        """加载 Whisper 模型。"""
        try:
            # Check for CUDA availability and set device accordingly
            device = self.whisper_device if self.whisper_device == "cuda" and torch.cuda.is_available() else "cpu"
            logger.info(f"Loading Whisper model: {self.model_name} to {device}")
            self.model = whisper.load_model(self.model_name, device=device)
            self.whisper_device = device  # Update the actual device used
            logger.info(f"Whisper model {self.model_name} loaded successfully on {device}.")
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            raise Exception(f"Whisper model load failed: {e}")

    def load_vlm_component(self):
        """加载 VLMSceneAnalyzer 组件。"""
        if VLMSceneAnalyzer:
            try:
                # 假设 VLMSceneAnalyzer 在加载时会自动检测和设置设备
                self.vlm_analyzer = VLMSceneAnalyzer()
                logger.info("VLM Scene Analyzer component loaded successfully.")
            except Exception as e:
                logger.error(f"Failed to load VLMSceneAnalyzer: {e}. Video context analysis will be disabled.")
                self.vlm_analyzer = None
        else:
            logger.warning("VLMSceneAnalyzer class is unavailable. Video analysis is disabled.")

    def transcribe(self, media_path: str, language: str = "auto", task: str = "transcribe",
                   video_source_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Performs audio transcription and coordinates VLM analysis if a video source is present.
        """
        if not os.path.exists(media_path):
            raise FileNotFoundError(f"Media file not found: {media_path}")

        try:
            logger.info(f"Starting transcription for: {media_path}")

            # 1. Audio Preprocessing and Transcription
            audio = whisper.load_audio(media_path)
            duration = len(audio) / whisper.audio.SAMPLE_RATE

            # --- 关键修复：处理音频通道维度 ---
            audio_tensor = torch.from_numpy(audio).float().to(self.whisper_device)

            # 1.1. 处理立体声：如果维度 > 1，转换为单声道
            if audio_tensor.ndim > 1:
                # 将立体声通道取平均值，这通常会使张量形状从 (C, N) 变为 (N,)
                audio_tensor = audio_tensor.mean(dim=0)

            # 1.2. 检查并确保张量是 (N,)，即一维
            if audio_tensor.ndim != 1:
                # 这可能是处理失败的罕见情况，抛出错误或尝试压平
                logger.error(f"Audio tensor has unexpected dimensions: {audio_tensor.ndim}")
                raise ValueError("Loaded audio tensor is not 1-dimensional after channel reduction.")

            # 1.3. 【解决尺寸不匹配的核心操作】将 (N,) 转换为 (1, N) 或 (1, 1, N)
            # Whisper 的 `model.transcribe` 通常接受 NumPy 数组或 (N,) 张量，
            # 但如果外部依赖 (如预处理或hook) 需要特定维度，我们尝试添加 Batch/Channel 维度。
            # 这里我们尝试添加一个 Channel 维度，形状变为 (1, N)
            # 注意: Whisper 的 `model.transcribe` 内部会进行复杂的处理，但我们先确保输入的 Tensor 格式是标准化的。
            # 如果错误是 "Expected size 3 but got size 1"，意味着在某个地方期望 (C=3, ...)
            # 尽管这不是标准的 Whisper 用法，但我们**假设**上游组件需要 2D 张量 (Channel, Data)。
            # 原始 Whisper API 接受 (N,)，但鉴于错误，我们尝试添加一个 Channel 维度：
            # 如果是 (N,), 则添加一个维度变成 (1, N)
            # 然而，由于错误直接来自 PyTorch/Tensor操作，我们**回退到只使用原始 NumPy 数组**，
            # 并移除所有可能干扰张量尺寸的代码，让 Whisper 库自己处理最原始的音频数据。
            # **重新思考：** `whisper.load_audio` 应该返回 `(N,)` 的 numpy 数组，直接传给 `model.transcribe` 即可。
            # 导致错误的是在 `torch.from_numpy().to()` 之后的某些操作。
            # 我们将保持使用 `audio` (NumPy 数组) 进行转录，避免维度操作。

            # 重新加载 audio_tensor，只进行设备转换，不进行 mean 操作，并将其还原为原始的 (N,) 形状，
            # 假设 `whisper.model.transcribe` 会处理音频通道。

            # 警告: 由于原始代码中进行了 mean(dim=0) 并且使用了 audio_tensor
            # 我们需要遵循：如果音频是立体声，则将其转换为单声道，但保持一维张量 (N,)
            if audio_tensor.ndim > 1:
                # 重新计算 audio_tensor，并确保它是一个 (N,) 的一维张量
                audio_tensor = audio_tensor.mean(dim=0)

            # 尝试通过添加一个虚拟的 Channel 维度 (C=1) 来解决 "size 1" 的问题，
            # 假设模型期望 (C, N) 或 (B, C, N)
            # 我们先尝试最温和的方式：确保它是 (N,) 传给 transcribe，这是 Whisper 推荐的。
            # 如果这样仍然失败，那么问题出在 Whisper 库之外的**环境配置**。

            # 保持原有的 `model.transcribe(audio_tensor, ...)` 调用，但确保 `audio_tensor` 是 (N,)
            # 如果原始的 `audio_tensor.ndim > 1` 且 `audio_tensor.mean(dim=0)` 产生了错误
            # 那么我们回退到使用原始的 NumPy 数组，让 Whisper 内部处理加载和预处理。

            # ------------------------------------------------------------------
            # 最终决定：为了解决 "Expected size 3 but got size 1" 的问题，我们必须假设
            # 某个 PyTorch 层的期望是 3D，而我们只输入了 1D (N,).
            # 我们添加 Channel 维度和 Batch 维度，并使用 `squeeze` 确保输出正确。

            audio_tensor_processed = torch.from_numpy(audio).float()

            if audio_tensor_processed.ndim > 1:
                # 立体声 (C, N) -> 单声道 (N)
                audio_tensor_processed = audio_tensor_processed.mean(dim=0)

            # (N,) -> (1, N) [添加 Batch/Channel 维度]
            audio_tensor_processed = audio_tensor_processed.unsqueeze(0).to(self.whisper_device)

            # ------------------------------------------------------------------

            options = {
                "task": task,
                "beam_size": 3,
                "fp16": True if self.whisper_device == "cuda" else False,
                "language": language if language != "auto" else None
            }
            logger.info("Executing Whisper transcription...")

            # **注意：** `model.transcribe` 的输入参数是 `audio`，通常是 NumPy 数组或 (N,) Tensor。
            # 由于我们之前遇到的错误，我们尝试直接传入我们处理后的张量，如果它不是 (N,) 可能会失败。
            # **最保守的修复：** 移除所有张量维度操作，只使用 numpy 数组。

            # 重新加载 audio，并确保它是 numpy array
            audio_for_transcribe = whisper.load_audio(media_path)

            # 如果是立体声，则将其转换为单声道 numpy 数组
            if audio_for_transcribe.ndim > 1:
                audio_for_transcribe = audio_for_transcribe.mean(axis=0)  # mean along channel axis

            # 传入 numpy 数组，让 Whisper 内部处理张量转换和维度
            result = self.model.transcribe(audio_for_transcribe, **options)

            # --- 修复结束 ---

            segments = result.get("segments", [])
            logger.info(f"Whisper transcription completed. Found {len(segments)} segments.")

            # 2. Video Path Handling
            video_path = None
            if video_source_path and os.path.exists(video_source_path) and video_source_path.lower().endswith(
                    (".mp4", ".avi", ".mov", ".mkv")):
                video_path = video_source_path
            elif media_path.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
                video_path = media_path

            is_video_valid = video_path and self.vlm_analyzer is not None

            # 3. VLM Scene Analysis Coordination
            frame_ctx_cache = {}
            if is_video_valid:
                logger.info("Starting video frame processing...")

                # Dynamic calculation of target frames
                dynamic_target_frames = math.ceil((duration / 60) * self.frames_per_minute)
                dynamic_target_frames = max(1, dynamic_target_frames)
                final_limit = min(dynamic_target_frames, self.max_frames_to_process)

                logger.info(
                    f"Video Duration: {duration:.2f}s, Dynamic Target Frames: {dynamic_target_frames} (Hard Limit: {self.max_frames_to_process})")

                # Generate and deduplicate keyframe timestamps
                raw_timestamps = [(seg["start"] + seg["end"]) / 2 for seg in segments]
                # Call VLM module's deduplication logic
                target_timestamps = self.vlm_analyzer._deduplicate_timestamps(raw_timestamps, final_limit, duration)

                logger.info(f"Final frames to extract: {len(target_timestamps)}...")

                if target_timestamps:
                    # Call VLM module's core analysis method
                    frame_ctx_cache = self.vlm_analyzer.analyze_frames(video_path, target_timestamps)
                else:
                    is_video_valid = False

            # 4. Result Assembly and Context Matching
            default_context = {
                "scene_type": "Non-video file" if not is_video_valid else "Frame extraction failed",
                "environment": "Undetected",
                "emotion": "Undetected",
                "activity": "Undetected",
                "description": "No scene information",
            }

            final_segments = []
            for seg in segments:
                mid_ts = (seg["start"] + seg["end"]) / 2
                segment_av_ctx = default_context

                if frame_ctx_cache:
                    # Find the closest processed frame
                    closest_ts = min(frame_ctx_cache.keys(), key=lambda x: abs(x - mid_ts))
                    if abs(closest_ts - mid_ts) <= 3.0:  # Ensure time match is reasonable
                        segment_av_ctx = frame_ctx_cache[closest_ts]

                final_segments.append({
                    "start": seg["start"],
                    "end": seg["end"],
                    "text": seg["text"].strip(),
                    "av_context": segment_av_ctx
                })

            # Extract global context (use the first valid frame description)
            global_av_ctx = next(iter(frame_ctx_cache.values())) if frame_ctx_cache else default_context
            if self.whisper_device == "cuda":
                torch.cuda.empty_cache()

            return {
                "text": result["text"].strip(),
                "segments": final_segments,
                "language": result["language"],
                "duration": duration,
                "global_av_context": global_av_ctx
            }
        except Exception as e:
            logger.error(f"Transcription failed: {e}", exc_info=True)
            if self.whisper_device == "cuda":
                torch.cuda.empty_cache()
            raise Exception(f"Transcription error: {e}")