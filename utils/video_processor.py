import cv2
import logging
import numpy as np
from typing import Optional

logger = logging.getLogger(__name__)

class VideoProcessor:
    def __init__(self):
        pass

    def extract_frame_at_time(self, video_path: str, timestamp: float) -> Optional[np.ndarray]:
        """
        从视频的指定时间戳（秒）提取一帧（RGB格式）
        :param video_path: 视频文件路径
        :param timestamp: 目标时间戳（秒）
        :return: RGB格式的帧（np.ndarray），失败返回None
        """
        if not video_path or not cv2.os.path.exists(video_path):
            logger.error(f"视频文件不存在：{video_path}")
            return None

        cap = None
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logger.error(f"无法打开视频：{video_path}")
                return None

            # 获取视频帧率和总帧数
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            total_duration = total_frames / fps if fps > 0 else 0

            # 校验时间戳有效性
            if timestamp < 0 or timestamp > total_duration:
                logger.warning(f"时间戳 {timestamp}s 超出视频范围（总时长 {total_duration}s），使用中间帧")
                timestamp = total_duration / 2

            # 计算目标帧索引并定位
            target_frame_idx = int(timestamp * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame_idx)

            # 读取帧并转换为RGB（cv2默认BGR）
            ret, frame = cap.read()
            if not ret:
                logger.error(f"在时间戳 {timestamp}s（帧索引 {target_frame_idx}）读取帧失败")
                return None

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            logger.info(f"成功提取视频 {video_path} 在 {timestamp:.2f}s 的帧（索引 {target_frame_idx}）")
            return rgb_frame

        except Exception as e:
            logger.error(f"提取视频帧失败：{e}")
            return None
        finally:
            if cap is not None:
                cap.release()

# 全局实例（供其他模块调用）
video_processor = VideoProcessor()
video_processor = VideoProcessor()