# utils/srt_utils.py
import srt
from typing import List, Tuple

def load_srt_as_sentences(path: str) -> List[str]:
    """
    Read an SRT and return a list of strings, one per subtitle block.
    """
    with open(path, "r", encoding="utf8") as f:
        text = f.read()
    subs = list(srt.parse(text))
    # join internal newlines; keep block as one sentence line
    return [s.content.replace("\n", " ").strip() for s in subs]

def load_srt_with_times(path: str) -> List[Tuple[float, float, str]]:
    """
    Return list of (start_seconds, end_seconds, text) for each srt block.
    """
    with open(path, "r", encoding="utf8") as f:
        text = f.read()
    subs = list(srt.parse(text))
    return [(s.start.total_seconds(), s.end.total_seconds(), s.content.replace("\n", " ").strip()) for s in subs]

def align_by_timestamps(ref_with_times, cand_with_times, max_gap=0.5):
    """
    A simple time-overlap alignment: for each candidate subtitle pick the reference subtitle with largest overlap.
    Returns two lists (ref_lines_aligned, cand_lines_aligned) that are aligned 1:1.
    Note: this is a heuristic — for difficult mismatches you'd want a more advanced aligner.
    """
    ref_lines = []
    cand_lines = []
    r_idx = 0
    for c_start, c_end, c_text in cand_with_times:
        # find ref with max overlap
        best_ref_text = None
        best_overlap = 0.0
        for rs, re, rtext in ref_with_times:
            # compute overlap
            overlap = max(0.0, min(re, c_end) - max(rs, c_start))
            if overlap > best_overlap:
                best_overlap = overlap
                best_ref_text = rtext
        # only accept if overlap positive (or within max_gap)
        if best_overlap > 0 or best_ref_text is not None:
            ref_lines.append(best_ref_text or "")
            cand_lines.append(c_text)
    return ref_lines, cand_lines
