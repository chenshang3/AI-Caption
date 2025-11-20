# models/evaluator_bleu.py

import os
import re
import codecs
from sacrebleu.metrics import BLEU

class SacreBLEUEvaluator:
    def __init__(self):
        self.metric = BLEU()

    def load_srt(self, path):
        """
        读取 SRT 文件，并提取字幕文本（不带序号、时间戳）
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"文件不存在: {path}")

        with codecs.open(path, 'r', 'utf-8', errors='ignore') as f:
            content = f.read()

        # 删除序号
        content = re.sub(r"\n\d+\n", "\n", content)

        # 删除时间轴
        content = re.sub(
            r"\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}",
            "",
            content,
        )

        # 清理空行
        lines = [line.strip() for line in content.split("\n") if line.strip()]
        return " ".join(lines)

    def evaluate(self, reference_path, candidate_path):
        """
        计算 BLEU 分数
        """
        ref = self.load_srt(reference_path)
        cand = self.load_srt(candidate_path)

        score_obj = self.metric.corpus_score([cand], [[ref]])

        return {
            "score": score_obj.score,
            "precisions": score_obj.precisions,
            "bp": score_obj.bp,
            "sys_len": score_obj.sys_len,
            "ref_len": score_obj.ref_len
        }
