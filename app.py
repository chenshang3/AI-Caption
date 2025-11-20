from flask import Flask, render_template, request, jsonify, send_file, url_for
import os
import json
import logging
import shutil
from datetime import datetime
from werkzeug.utils import secure_filename
from models.evaluator_bleu import SacreBLEUEvaluator
from utils.srt_utils import load_srt_as_sentences


# 引入配置
from config import Config

# 引入模型组件
# 注意：这里我们替换了原来的 Translator，改用新的 NeuralTranslator
from models.whisper_model_fixed import WhisperModel
from models.translator import NeuralTranslator 
from utils.audio_processor import AudioProcessor
from utils.subtitle_generator import SubtitleGenerator
from utils.file_handler import FileHandler

# 配置日志格式，增加时间戳以便在前端终端中查看
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
# 这里是机器学习工作量的体现：我们在启动时加载所有模型到内存
try:
    logger.info("正在初始化 AI 核心组件...")
    
    # 1. 初始化 Whisper (ASR 模型)
    # 这里的参数从 Config 中读取
    whisper_model = WhisperModel(
        model_name=Config.WHISPER_MODEL,
        device=Config.WHISPER_DEVICE
    )
    logger.info(f"Whisper 模型加载完成 (Device: {Config.WHISPER_DEVICE})")

    # 2. 初始化 NeuralTranslator (NMT + LLM 模型)
    # 这是本次改造的核心：本地加载 NLLB 和 Qwen
    translator = NeuralTranslator(
        nmt_model_id=getattr(Config, 'NMT_MODEL_ID', "facebook/nllb-200-distilled-600M"),
        reflection_model_id=getattr(Config, 'REFLECTION_MODEL_ID', "Qwen/Qwen2.5-0.5B-Instruct"),
        device=Config.WHISPER_DEVICE # 复用设备配置
    )
    logger.info("神经翻译引擎加载完成 (NLLB + Reflection Agent)")

    # 3. 初始化工具类
    audio_processor = AudioProcessor()
    subtitle_generator = SubtitleGenerator()
    file_handler = FileHandler()
    bleu_evaluator = SacreBLEUEvaluator()
    logger.info("BLEU 评估器加载完成")

    logger.info("✅ 所有系统组件初始化成功")

except Exception as e:
    logger.critical(f"组件初始化失败: {e}")
    # 在课设演示中，如果模型加载失败，抛出异常让程序崩溃是合理的，
    # 这样你能直接看到报错信息（通常是显存不足或网络下载超时）
    raise e

# ===========================================================
# Web 路由定义
# ===========================================================

@app.route('/')
def index():
    """渲染新的 Dashboard 界面"""
    return render_template('index.html')

@app.route('/api/upload', methods=['POST'])
def upload_file():
    """上传文件API - 保持不变"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': '没有上传文件'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': '没有选择文件'}), 400
        
        # 保存文件
        file_path, filename = file_handler.save_uploaded_file(file)
        logger.info(f"收到文件上传: {filename}")
        
        return jsonify({
            'success': True,
            'file_path': file_path,
            'filename': filename,
            'original_name': file.filename,
        })
        
    except Exception as e:
        logger.error(f"文件上传失败: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/transcribe', methods=['POST'])
def transcribe_audio():
    """音频转录API - 逻辑微调"""
    try:
        data = request.get_json()
        file_path = data.get('file_path')
        language = data.get('language', 'auto')
        
        if not file_path or not os.path.exists(file_path):
            return jsonify({'error': '文件不存在'}), 400
        
        logger.info(f"开始转录: {file_path} (Lang: {language})")
        
        # 创建临时目录
        temp_dir = file_handler.create_temp_directory()
        
        try:
            # 预处理音频
            processed_audio_path = audio_processor.process_audio_for_transcription(
                file_path, temp_dir
            )
            
            # 执行 Whisper 转录
            result = whisper_model.transcribe(processed_audio_path, language=language)
            
            logger.info(f"转录完成，生成 {len(result['segments'])} 个片段")
            
            return jsonify({
                'success': True,
                'text': result['text'],
                'segments': result['segments'],
                'language': result['language'],
                'duration': result['duration']
            })
            
        finally:
            # 清理临时音频切片，但不删除原上传文件
            file_handler.cleanup_temp_files(temp_dir)
            
    except Exception as e:
        logger.error(f"音频转录失败: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/translate', methods=['POST'])
def translate_subtitle():
    """
    字幕翻译API - 核心修改点
    对接 NeuralTranslator 并支持反思模式
    """
    try:
        data = request.get_json()
        segments = data.get('segments', [])
        target_lang = data.get('target_language', 'zh-cn')
        source_lang = data.get('source_language', 'auto')
        
        # 获取前端传来的开关状态
        use_reflection = data.get('use_reflection', False)
        
        if not segments:
            return jsonify({'error': '没有字幕内容'}), 400
        
        logger.info(f"开始翻译请求: {len(segments)} segments -> {target_lang}")
        if use_reflection:
            logger.info("🚀 启用 Agent 反思模式 (Reflection Mode)")
        
        # 调用本地神经翻译器
        translated_segments = translator.translate_segments(
            segments=segments, 
            target_lang=target_lang, 
            source_lang=source_lang,
            use_reflection=use_reflection # 传递给后端模型
        )
        
        return jsonify({
            'success': True,
            'segments': translated_segments
        })
        
    except Exception as e:
        logger.error(f"字幕翻译失败: {e}")
        # 在前端显示具体错误，有助于调试 CUDA OOM 等问题
        return jsonify({'error': f"翻译引擎错误: {str(e)}"}), 500

@app.route('/api/generate-subtitle', methods=['POST'])
def generate_subtitle():
    """生成字幕文件API"""
    try:
        data = request.get_json()
        segments = data.get('segments', [])
        format_type = data.get('format', 'srt')
        filename = data.get('filename', 'subtitle')
        # 新增：从前端获取后缀，默认为 translated
        suffix = data.get('suffix', 'translated') 
        
        if not segments:
            return jsonify({'error': '没有字幕内容'}), 400
        
        # 使用动态后缀
        output_filename = file_handler.generate_output_filename(
            filename, suffix, f".{format_type}"
        )
        output_path = os.path.join(Config.OUTPUT_FOLDER, output_filename)
        
        # 创建字幕文件
        subtitle_path = subtitle_generator.create_subtitle(
            segments, output_path, format_type
        )
        # 如果生成的是 original 字幕，则自动创建参考字幕
        if suffix == "original":
            reference_path = "data/reference.srt"
            # 仅在用户没有上传自定义 reference 时自动生成
            if not os.path.exists(reference_path):
                shutil.copy(subtitle_path, reference_path)
            logger.info(f"已自动生成参考字幕: {reference_path}")

        return jsonify({
            'success': True,
            'download_url': url_for('download_file', filename=output_filename),
            'filename': output_filename,
            'file_path': subtitle_path   # 给 BLEU           
        })
        
    except Exception as e:
        logger.error(f"字幕文件生成失败: {e}")
        return jsonify({'error': str(e)}), 500
    
@app.post("/api/upload-reference")
def upload_reference():
    try:
        file = request.files["file"]
        save_path = os.path.join("data", "reference.srt")
        file.save(save_path)
        return jsonify({"success": True, "message": "参考字幕上传成功！", "path": save_path})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@app.route('/api/evaluate', methods=['POST'])
def evaluate_translation():
    try:
        data = request.get_json()
        reference_path = data.get('reference_path')
        candidate_path = data.get('candidate_path')

        if not reference_path or not candidate_path:
            return jsonify({'error': '缺少 reference_path 或 candidate_path'}), 400

        if not os.path.exists(reference_path):
            return jsonify({'error': f'参考文件不存在: {reference_path}'}), 400

        if not os.path.exists(candidate_path):
            return jsonify({'error': f'候选翻译文件不存在: {candidate_path}'}), 400

        # 直接把路径传进去，不做任何读取！
        bleu = bleu_evaluator.evaluate(reference_path, candidate_path)

        return jsonify({
            'success': True,
            'bleu': bleu
        })

    except Exception as e:
        logger.error(f"BLEU 评估失败: {e}")
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
    """获取支持语言 - 对接新模型"""
    try:
        # 直接从新的翻译器获取支持列表
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
    
    print(f"\n{'='*50}")
    print(f"🤖 NeuralSub 智能字幕系统启动中...")
    print(f"💻 访问地址: http://localhost:5000")
    print(f"{'='*50}\n")
    
    # 启动应用
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False) 
    # use_reloader=False 防止加载两次模型导致显存爆炸