import cv2
import logging
import numpy as np
import os  # 导入 os 模块以使用 os.path.exists
from typing import Optional

logger = logging.getLogger(__name__)


class VideoProcessor:
    def __init__(self):
        # 检查 OpenCV 是否可用（如果 cv2 导入失败，此处会捕获，但在当前结构下，通常在导入时就失败了）
        logger.info("VideoProcessor initialized.")
        pass

    def extract_frame_at_time(self, video_path: str, timestamp: float) -> Optional[np.ndarray]:
        """
        从视频的指定时间戳（秒）提取一帧（RGB格式）
        :param video_path: 视频文件路径
        :param timestamp: 目标时间戳（秒）
        :return: RGB格式的帧（np.ndarray），失败返回None
        """
        # 使用 os.path.exists
        if not video_path or not os.path.exists(video_path):
            logger.error(f"视频文件不存在：{video_path}")
            return None

        cap = None
        try:
            # 尝试打开视频文件
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                # 📢 这是最容易失败的地方（FFmpeg/编解码器问题）
                logger.error(f"无法打开视频：{video_path}。请检查 FFmpeg/编解码器配置。")
                return None

            # 获取视频帧率和总帧数
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            total_duration = total_frames / fps if fps > 0 else 0

            # 校验时间戳有效性
            if timestamp < 0 or timestamp > total_duration:
                logger.warning(f"时间戳 {timestamp}s 超出视频范围（总时长 {total_duration:.2f}s），调整为中间帧")
                # 使用中间帧作为安全回退
                timestamp = max(0, min(timestamp, total_duration / 2))  # 确保至少是 0

            # 定位到目标帧
            target_frame_idx = int(timestamp * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame_idx)

            # 读取帧并转换为RGB（cv2默认BGR）
            ret, frame = cap.read()
            if not ret or frame is None:
                logger.error(f"在时间戳 {timestamp:.2f}s（帧索引 {target_frame_idx}）读取帧失败")
                return None

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            logger.info(f"成功提取视频 {video_path} 在 {timestamp:.2f}s 的帧（索引 {target_frame_idx}）")
            return rgb_frame

        except Exception as e:
            logger.error(f"提取视频帧失败：{e}", exc_info=True)
            return None
        finally:
            if cap is not None:
                cap.release()


video_processor = VideoProcessor()