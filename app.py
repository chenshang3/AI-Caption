from flask import Flask, render_template, request, jsonify, send_file, url_for
import os
import json
import logging
from datetime import datetime
from werkzeug.utils import secure_filename

# 引入配置
from config import Config

# 引入模型组件 (根据新的协调类结构修改)
# 现在使用 WhisperTranscriber 作为 ASR 和 VLM 协调器
from models.whisper_model_fixed import WhisperTranscriber
from models.translator import NeuralTranslator
from utils.audio_processor import AudioProcessor
from utils.subtitle_generator import SubtitleGenerator
from utils.file_handler import FileHandler

# 配置日志格式
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 创建Flask应用
app = Flask(__name__)
app.config.from_object(Config)
Config.init_app(app)

# ===========================================================
# 全局组件初始化 (Loading Models)
# ===========================================================
try:
    logger.info("正在初始化 AI 核心组件...")

    # 1. 初始化 Whisper Transcriber (ASR + VLM 协调器)
    # 该类内部会加载 Whisper 模型和 VLMSceneAnalyzer
    transcriber = WhisperTranscriber(
        model_name=Config.WHISPER_MODEL,
        device=Config.WHISPER_DEVICE
    )
    logger.info(f"Whisper Transcriber (ASR/VLM Coordinator) 初始化完成。")


    # 2. 初始化 NeuralTranslator (NMT + LLM 模型)
    translator = NeuralTranslator(
        nmt_model_id=getattr(Config, 'NMT_MODEL_ID', "facebook/nllb-200-distilled-600M"),
        reflection_model_id=getattr(Config, 'REFLECTION_MODEL_ID', "Qwen/Qwen2.5-0.5B-Instruct"),
        device=Config.WHISPER_DEVICE
    )
    logger.info("神经翻译引擎加载完成 (NLLB + Reflection Agent)")

    # 3. 初始化工具类
    audio_processor = AudioProcessor()
    subtitle_generator = SubtitleGenerator()
    file_handler = FileHandler()

    logger.info("✅ 所有系统组件初始化成功")

except Exception as e:
    logger.critical(f"组件初始化失败: {e}")
    raise e


# ===========================================================
# Web 路由定义
# ===========================================================

@app.route('/')
def index():
    """渲染 Dashboard 界面"""
    return render_template('index.html')


@app.route('/api/upload', methods=['POST'])
def upload_file():
    """上传文件API"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': '没有上传文件'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': '没有选择文件'}), 400

        # 保存文件
        file_path, filename = file_handler.save_uploaded_file(file)
        logger.info(f"收到文件上传: {filename}，路径: {file_path}")

        return jsonify({
            'success': True,
            'file_path': file_path,
            'filename': filename,
            'original_name': file.filename
        })

    except Exception as e:
        logger.error(f"文件上传失败: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/transcribe', methods=['POST'])
def transcribe_audio():
    """音频转录 + VLM 分析 API (调用协调器)"""
    try:
        data = request.get_json()
        file_path = data.get('file_path')  # 原始上传的视频文件路径
        language = data.get('language', 'auto')

        if not file_path or not os.path.exists(file_path):
            return jsonify({'error': '文件不存在'}), 400

        logger.info(f"开始处理: {file_path} (Lang: {language})")

        temp_dir = file_handler.create_temp_directory()

        try:
            # 1. 预处理音频：从视频中提取音频
            processed_audio_path = audio_processor.process_audio_for_transcription(
                file_path, temp_dir
            )

            # 2. 调用 Whisper Transcriber 协调器
            # 它将负责：ASR (使用 processed_audio_path) -> VLM (使用 file_path 和时间戳) -> 结果合并
            logger.info("调用 Whisper Transcriber 进行 ASR 和 VLM 协调分析...")
            result = transcriber.transcribe(
                media_path=processed_audio_path,
                language=language,
                video_source_path=file_path  # 传入原始视频路径供 VLM 使用
            )

            segments = result.get('segments', [])
            logger.info(f"协调分析完成，返回 {len(segments)} 个片段。")

            # 组装最终结果
            return jsonify({
                'success': True,
                'text': result['text'],
                'segments': segments,
                'language': result['language'],
                'duration': result['duration']
            })

        finally:
            file_handler.cleanup_temp_files(temp_dir)

    except Exception as e:
        logger.error(f"媒体处理失败: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/api/translate', methods=['POST'])
def translate_subtitle():
    """
    字幕翻译API - 对接 NeuralTranslator
    """
    try:
        data = request.get_json()
        segments = data.get('segments', [])
        target_lang = data.get('target_language', 'zh-cn')
        source_lang = data.get('source_language', 'auto')

        use_reflection = data.get('use_reflection', False)

        if not segments:
            return jsonify({'error': '没有字幕内容'}), 400

        logger.info(f"开始翻译请求: {len(segments)} segments -> {target_lang}")
        if use_reflection:
            logger.info("🚀 启用 Agent 反思模式 (Reflection Mode)")

        # 调用神经翻译器
        translated_segments = translator.translate_segments(
            segments=segments,
            target_lang=target_lang,
            source_lang=source_lang,
            use_reflection=use_reflection
        )

        return jsonify({
            'success': True,
            'segments': translated_segments
        })

    except Exception as e:
        logger.error(f"字幕翻译失败: {e}")
        return jsonify({'error': f"翻译引擎错误: {str(e)}"}), 500


@app.route('/api/generate-subtitle', methods=['POST'])
def generate_subtitle():
    """生成字幕文件API"""
    try:
        data = request.get_json()
        segments = data.get('segments', [])
        format_type = data.get('format', 'srt')
        filename = data.get('filename', 'subtitle')
        suffix = data.get('suffix', 'translated')

        if not segments:
            return jsonify({'error': '没有字幕内容'}), 400

        output_filename = file_handler.generate_output_filename(
            filename, suffix, f".{format_type}"
        )
        output_path = os.path.join(Config.OUTPUT_FOLDER, output_filename)

        subtitle_path = subtitle_generator.create_subtitle(
            segments, output_path, format_type
        )

        return jsonify({
            'success': True,
            'download_url': url_for('download_file', filename=output_filename),
            'filename': output_filename
        })

    except Exception as e:
        logger.error(f"字幕文件生成失败: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/download/<filename>')
def download_file(filename):
    """文件下载路由"""
    try:
        file_path = os.path.join(Config.OUTPUT_FOLDER, filename)
        if os.path.exists(file_path):
            return send_file(file_path, as_attachment=True)
        else:
            return jsonify({'error': '文件不存在'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/languages')
def get_languages():
    """获取支持语言 - 对接翻译模型"""
    try:
        languages = translator.get_supported_languages()
        return jsonify(languages)
    except Exception as e:
        logger.error(f"获取语言列表失败: {e}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    # 检查 FFmpeg
    if not audio_processor.check_ffmpeg():
        print("\n⚠️  警告: 未检测到 FFmpeg!")
        print("    这会导致音频提取失败。请先安装 FFmpeg 并添加到环境变量。\n")

    print(f"\n{'=' * 50}")
    print(f"🤖 NeuralSub 智能字幕系统启动中...")
    print(f"💻 访问地址: http://localhost:5000")
    print(f"{'=' * 50}\n")

    # 启动应用
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)