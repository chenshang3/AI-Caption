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

# 确保日志配置正确（修正了用户代码中的 log format typo）
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

            audio_tensor = torch.from_numpy(audio).float().to(self.whisper_device)
            if audio_tensor.ndim > 1:
                audio_tensor = audio_tensor.mean(dim=0)

            options = {
                "task": task,
                "beam_size": 3,
                "fp16": True if self.whisper_device == "cuda" else False,
                "language": language if language != "auto" else None
            }
            logger.info("Executing Whisper transcription...")
            result = self.model.transcribe(audio_tensor, **options)
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