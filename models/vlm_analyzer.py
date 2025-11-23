import logging
import torch
import numpy as np
import cv2
import os
import math
from typing import Dict, Any, List, Tuple
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, GPT2Tokenizer
from concurrent.futures import ProcessPoolExecutor
from PIL import Image

# 确保日志配置正确
logger = logging.getLogger(__name__)


class VLMSceneAnalyzer:
    """
    负责加载 ViT-GPT2 模型、提取关键帧和批量生成场景描述。
    这是独立的 VLM 组件，负责所有视觉分析工作。
    """

    def __init__(self):
        self.vit_gpt2_model = None
        self.vit_processor = None
        self.gpt2_tokenizer = None
        # VLM 设备设置
        self.vlm_device = "cuda" if torch.cuda.is_available() else "cpu"
        # 优化参数
        self.frame_size = (224, 224)  # ViT-GPT2的最佳输入尺寸
        self.min_frame_interval = 1.0  # 关键帧之间的最小时间间隔（秒）

        self.load_model()

    def load_model(self):
        """加载 ViT-GPT2 模型及其组件。"""
        vlm_model_id = "nlpconnect/vit-gpt2-image-captioning"
        logger.info(f"Initializing ViT-GPT2 model '{vlm_model_id}' on device: {self.vlm_device}")
        try:
            self.vit_processor = ViTImageProcessor.from_pretrained(vlm_model_id)
            self.gpt2_tokenizer = GPT2Tokenizer.from_pretrained(vlm_model_id)

            # 使用 fp16 优化 VRAM (仅限 CUDA)
            vlm_dtype = torch.float16 if self.vlm_device == "cuda" else torch.float32

            self.vit_gpt2_model = VisionEncoderDecoderModel.from_pretrained(
                vlm_model_id,
                torch_dtype=vlm_dtype
            ).to(self.vlm_device)

            if self.gpt2_tokenizer.pad_token is None:
                self.gpt2_tokenizer.pad_token = self.gpt2_tokenizer.eos_token
                logger.warning("Set pad_token to eos_token for GPT2Tokenizer")
            logger.info("✅ ViT-GPT2 model and components loaded successfully.")
        except Exception as e:
            logger.error(f"ViT-GPT2 load failed: {e}")
            self.vit_gpt2_model = None
        finally:
            if self.vlm_device == "cuda":
                torch.cuda.empty_cache()

    # --- 场景解析辅助函数 (中文解析逻辑) ---

    def _parse_environment(self, desc: str) -> str:
        """根据描述解析环境/场所类型 (中文)。"""
        special_places = {
            "监狱": ["prison", "jail", "cell", "inmate", "correctional facility", "guard", "bars"],
            "警察局": ["police station", "police office", "cop shop", "detention center", "police car", "officer"],
            "医院": ["hospital", "clinic", "medical center", "ward", "emergency room", "doctor", "nurse", "patient",
                     "bed"],
            "商店": ["store", "shop", "market", "mall", "retail", "counter", "customer", "product"],
            "学校": ["school", "classroom", "university", "college", "student", "teacher", "desk", "blackboard"],
            "办公室": ["office", "workplace", "desk", "computer", "employee", "meeting room", "cubicle"],
            "餐厅": ["restaurant", "cafe", "diner", "table", "chair", "menu", "waiter", "food"],
            "酒店": ["hotel", "motel", "lobby", "room", "reception", "guest"],
            "银行": ["bank", "teller", "ATM", "vault", "customer service"],
            "机场": ["airport", "terminal", "plane", "gate", "passenger", "luggage"],
            "车站": ["train station", "bus station", "platform", "ticket", "passenger"],
            "图书馆": ["library", "book", "shelf", "reader", "desk"],
            "博物馆": ["museum", "exhibit", "artifact", "display", "visitor"],
            "体育馆": ["stadium", "gym", "court", "field", "player", "audience"],
            "电影院": ["cinema", "theater", "movie", "screen", "seat", "audience"],
            "教堂": ["church", "temple", "mosque", "prayer", "worship", "altar"],
        }
        outdoor_scenes = {
            "城市街道": ["street", "road", "car", "traffic", "building", "sidewalk", "crosswalk", "traffic light"],
            "公园": ["park", "garden", "tree", "flower", "bench", "path", "playground"],
            "森林": ["forest", "woods", "tree", "leaf", "animal", "trail"],
            "海滩": ["beach", "sand", "ocean", "sea", "wave", "umbrella", "swimmer"],
            "山脉": ["mountain", "hill", "peak", "valley", "hiker", "trail"],
            "田野": ["field", "farm", "crop", "tractor", "farmer", "grass"],
            "工地": ["construction site", "worker", "crane", "building", "material"],
            "停车场": ["parking lot", "car", "parking space", "vehicle"],
            "加油站": ["gas station", "fuel", "pump", "car", "attendant"],
        }
        indoor_scenes = {
            "家庭住宅": ["house", "home", "living room", "kitchen", "bedroom", "bathroom", "sofa", "TV"],
            "公寓": ["apartment", "flat", "living room", "kitchen", "bedroom", "tenant"],
            "宿舍": ["dormitory", "dorm", "room", "student", "bed", "desk"],
        }
        urban_keywords = ["city", "urban", "building", "street", "car", "traffic", "skyscraper", "apartment"]
        rural_keywords = ["countryside", "rural", "farm", "field", "village", "cottage", "tractor", "animal"]

        desc_lower = desc.lower()
        for place, keywords in special_places.items():
            if any(kw in desc_lower for kw in keywords): return place
        for scene, keywords in outdoor_scenes.items():
            if any(kw in desc_lower for kw in keywords): return scene
        for scene, keywords in indoor_scenes.items():
            if any(kw in desc_lower for kw in keywords): return scene
        if any(kw in desc_lower for kw in urban_keywords): return "城市区域"
        if any(kw in desc_lower for kw in rural_keywords): return "农村区域"
        if any(kw in desc_lower for kw in ["indoor", "inside", "room", "building"]): return "室内场所"
        if any(kw in desc_lower for kw in ["outdoor", "outside", "open area"]): return "室外场所"
        return "未知场所"

    def _parse_emotion(self, desc: str) -> str:
        """根据描述解析人物情绪 (中文)。"""
        positive_keywords = ["smiling", "happy", "laughing", "excited", "joyful", "cheerful", "grinning", "delighted"]
        calm_keywords = ["calm", "relaxed", "quiet", "still", "peaceful", "serene", "composed"]
        negative_keywords = ["sad", "angry", "upset", "frowning", "frustrated", "crying", "mad", "serious"]
        desc_lower = desc.lower()
        if any(kw in desc_lower for kw in positive_keywords): return "开心/兴奋"
        if any(kw in desc_lower for kw in calm_keywords): return "平静/放松"
        if any(kw in desc_lower for kw in negative_keywords): return "悲伤/愤怒/严肃"
        return "中性"

    def _parse_activity(self, desc: str) -> str:
        """根据描述解析人物活动 (中文)。"""
        talking_keywords = ["talking", "speaking", "discussing", "interview", "chatting", "conversing", "explaining"]
        action_keywords = ["holding", "using", "playing", "running", "walking", "skateboarding", "dancing", "eating",
                           "drinking", "writing", "reading", "gaming", "playing a game"]
        static_keywords = ["standing", "sitting", "posing", "looking", "watching", "listening", "sleeping", "resting"]
        desc_lower = desc.lower()
        if any(kw in desc_lower for kw in talking_keywords): return "交谈/说话/解说"
        if any(kw in desc_lower for kw in action_keywords): return "进行动作（持物/运动/游戏等）"
        if any(kw in desc_lower for kw in static_keywords): return "静止状态（站立/坐姿等）"
        return "未知活动"

    def _parse_scene_type(self, desc: str) -> str:
        """根据描述解析场景类型 (中文)。"""
        live_stream_keywords = ["live", "stream", "streamer", "主播", "直播", "解说", "commentary", "ui", "interface",
                                "弹幕", "danmu", "chat", "聊天", "礼物", "关注", "点赞"]
        game_keywords = ["game", "gaming", "video game", "character", "角色", "player", "玩家", "level", "地图", "map",
                         "quest", "任务", "hp", "mp", "health", "mana", "score", "得分", "loading", "menu", "inventory",
                         "装备", "weapon", "武器", "敌人", "boss", "战斗", "战斗场景", "像素", "pixel",
                         "3d render", "animated"]
        desc_lower = desc.lower()
        if any(kw in desc_lower for kw in live_stream_keywords):
            return "游戏直播解说画面" if any(kw in desc_lower for kw in game_keywords) else "直播解说画面"
        if any(kw in desc_lower for kw in game_keywords): return "游戏画面"
        return "真实世界场景"

    # --- 帧提取和 VLM 推理核心函数 ---

    @staticmethod
    def _extract_frames_worker(video_path: str, timestamps: List[float], frame_size: Tuple[int, int]) -> Dict[
        float, np.ndarray]:
        """
        【多进程工作单元】从视频中提取指定时间戳的帧，并进行尺寸缩放。
        """
        frame_cache = {}
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Failed to open video file: {video_path}")
            return frame_cache

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        for ts in timestamps:
            frame_idx = int(ts * fps)
            frame_idx = min(max(0, frame_idx), total_frames - 1)

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_resized = cv2.resize(frame_rgb, frame_size)
                frame_cache[ts] = frame_resized

        cap.release()
        return frame_cache

    def _deduplicate_timestamps(self, timestamps: List[float], final_limit: int, duration: float) -> List[float]:
        """
        对关键帧时间戳进行去重，确保在指定的最小间隔内只保留一个，并使用最终数量限制进行均匀采样。
        """
        if not timestamps: return []

        # 1. 最小间隔去重
        sorted_ts = sorted(timestamps)
        deduplicated = [sorted_ts[0]]
        for ts in sorted_ts[1:]:
            if ts - deduplicated[-1] >= self.min_frame_interval:
                deduplicated.append(ts)

        original_count = len(sorted_ts)
        dedup_count = len(deduplicated)

        logger.info(
            f"🎬 VLM Frame Sampling: Original {original_count} frames -> Deduplicated {dedup_count} frames"
        )

        # 2. 均匀采样限制
        if dedup_count > final_limit:
            indices = np.linspace(0, dedup_count - 1, final_limit, dtype=int)
            deduplicated = [deduplicated[i] for i in indices]

            final_count = len(deduplicated)
            logger.info(
                f"➡️ VLM Final Sampling: Exceeded limit ({final_limit} frames), uniformly sampled {final_count} frames"
            )

        return deduplicated

    def _process_frames_batch(self, frames_data: List[Tuple[float, np.ndarray]]) -> List[Dict[str, Any]]:
        """
        【主进程执行】批量处理帧数据，生成场景描述，并进行解析。
        """
        if not frames_data or self.vit_gpt2_model is None:
            logger.warning("VLM model is not loaded, skipping batch processing.")
            return []

        timestamps, frames = zip(*frames_data)
        vlm_dtype = torch.float16 if self.vlm_device == "cuda" else torch.float32

        # 批量预处理 (frames 是 np.ndarray 列表)
        pixel_values = self.vit_processor(
            images=[Image.fromarray(f) for f in frames],
            return_tensors="pt",
        ).pixel_values.to(self.vlm_device, dtype=vlm_dtype)

        # 批量生成描述
        with torch.no_grad():
            gen_ids = self.vit_gpt2_model.generate(
                pixel_values,
                max_length=100,
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=2,
                pad_token_id=self.gpt2_tokenizer.pad_token_id,
                eos_token_id=self.gpt2_tokenizer.eos_token_id
            )

        raw_descriptions = self.gpt2_tokenizer.batch_decode(gen_ids, skip_special_tokens=True)

        # 批量解析结果
        results = []
        for ts, desc in zip(timestamps, raw_descriptions):
            desc_stripped = desc.strip()
            result = {
                "timestamp": round(ts, 2),
                "description": desc_stripped,
                "scene_type": self._parse_scene_type(desc_stripped),
                "environment": self._parse_environment(desc_stripped),
                "emotion": self._parse_emotion(desc_stripped),
                "activity": self._parse_activity(desc_stripped),
            }
            results.append(result)

            logger.info(
                f"🖼️ Frame Context Analysis (TS: {result['timestamp']}s): "
                f"[{result['scene_type']}/{result['environment']}] "
                f"Emotion: {result['emotion']}, "
                f"Activity: {result['activity']}. "
                f"Description: '{result['description']}'"
            )

        return results

    def analyze_frames(self, video_path: str, target_timestamps: List[float]) -> Dict[float, Dict[str, Any]]:
        """
        Executes frame extraction (multi-process) and VLM inference (main process).
        Returns a map from timestamp to scene context.
        """
        frame_ctx_cache = {}
        if self.vit_gpt2_model is None:
            logger.error("VLM model is not available. Cannot analyze frames.")
            return {}

        # 1. Multi-process frame extraction (I/O)
        frame_cache = {}
        try:
            logger.info(f"Entering ProcessPoolExecutor, preparing to extract {len(target_timestamps)} frames...")

            with ProcessPoolExecutor(max_workers=min(4, os.cpu_count() or 1)) as executor:
                extract_future = executor.submit(
                    VLMSceneAnalyzer._extract_frames_worker,
                    video_path,
                    target_timestamps,
                    self.frame_size
                )
                frame_cache = extract_future.result()

            logger.info(f"ProcessPoolExecutor exited, successfully extracted {len(frame_cache)} frames.")

            if not frame_cache:
                logger.error("Failed to extract any frames in the multi-process pool.")
                return {}

            # 2. Main process batch VLM inference (GPU)
            frames_to_process = sorted(frame_cache.items(), key=lambda item: item[0])
            batch_size = 16
            frame_batches = [frames_to_process[i:i + batch_size] for i in
                             range(0, len(frames_to_process), batch_size)]

            logger.info(f"Starting VLM batch inference ({len(frames_to_process)} frames, Batch={batch_size})...")

            for i, batch in enumerate(frame_batches):
                logger.info(f"Processing VLM batch {i + 1}/{len(frame_batches)}")
                batch_results = self._process_frames_batch(batch)
                for res in batch_results:
                    frame_ctx_cache[res.pop("timestamp")] = res

            logger.info(f"VLM inference completed, generated {len(frame_ctx_cache)} valid descriptions.")
            return frame_ctx_cache

        except Exception as e:
            logger.error(f"Video scene analysis failed: {e}", exc_info=True)
            return {}
        finally:
            if self.vlm_device == "cuda":
                torch.cuda.empty_cache()