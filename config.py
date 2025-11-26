import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    # Flask配置
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'course-project-secret-key'
    UPLOAD_FOLDER = 'static/uploads'
    OUTPUT_FOLDER = 'output'
    PORT = int(os.environ.get('PORT', 5000))  # 之前添加的端口配置

    # ML Model Configuration
    # 1. Whisper ASR Model
    WHISPER_MODEL = 'small'
    WHISPER_DEVICE = 'cuda'  # 若有N卡改为 'cuda'

    # 2. NMT Model（新增 TRANSLATOR_DEVICE 配置）
    NMT_MODEL_ID = "facebook/nllb-200-1.3B"  # 原有的NMT模型ID
    TRANSLATOR_DEVICE = 'cuda'  # 翻译模型的设备（与Whisper保持一致，无GPU则改为 'cpu'）

    # 3. Reflection LLM Model
    REFLECTION_MODEL_ID = "Qwen/Qwen3-8B"

    # 4. Transfer Learning Config
    FINETUNE_EPOCHS = 3
    FINETUNE_LEARNING_RATE = 2e-5
    FINETUNE_BATCH_SIZE = 16

    # 5. Quality Estimation (QE) Model
    QE_MODEL_ID = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    QE_THRESHOLD = 0.7
    ENABLE_QE = True

    # 功能开关
    ENABLE_REFLECTION = True

    # 其他配置
    SUPPORTED_FORMATS = ['.mp3', '.wav', '.m4a', '.flac', '.mp4', '.mkv']
    SUBTITLE_FORMATS = ['srt', 'vtt']
    TEMP_FOLDER = 'temp'

    @staticmethod
    def init_app(app):
        os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
        os.makedirs(Config.OUTPUT_FOLDER, exist_ok=True)
        os.makedirs(Config.TEMP_FOLDER, exist_ok=True)