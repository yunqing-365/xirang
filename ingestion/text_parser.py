# ingestion/text_parser.py
"""
古籍文本解析器
处理史料文献的特殊格式：繁体字、竖排OCR噪音、异体字、标点缺失等。

支持输入格式：
  .txt / .md  — 直接读取
  .pdf        — pdfplumber 提取（含图片OCR可选）

输出：标准化的 List[str] 文本块，每块约 300-600 字，
      附带 metadata: {source, year, era, chunk_idx, char_count}
"""
from __future__ import annotations
import re
import unicodedata
from pathlib import Path
from typing import Iterator, List, Optional, Tuple
import opencc          # 繁简转换

from ingestion.year_normalizer import extract_year_from_text, normalize_era_name

# ── 繁简转换器（一次性初始化）──────────────────────────────────
_T2S = opencc.OpenCC("t2s")    # 繁体→简体（保留异体字辨义）
_S2T = opencc.OpenCC("s2t")    # 简体→繁体（用于验证）

def trad_to_simp(text: str) -> str:
    return _T2S.convert(text)


# ── 异体字/通假字规范化 ─────────────────────────────────────────
# 格式：异体字 → 标准字（仅影响索引，不修改原文显示）
_VARIANT_MAP = {
    # 常见异体字
    "喆": "哲", "祇": "祗", "昇": "升", "廸": "迪",
    "迺": "乃", "敺": "驱", "糴": "籴", "糶": "粜",
    "鍼": "针", "��廳": "厅", "覈": "核", "覩": "睹",
    "矚": "嘱", "攬": "揽", "籲": "吁",
    # 通假常见
    "说": "悦",  # 仅在特定语境，保守处理
    # 竖排OCR常见错误字
    "己": None,  # 已/己/巳混淆，不强制
}

def normalize_variants(text: str) -> str:
    """将异体字替换为标准字（用于索引建立）"""
    result = []
    for ch in text:
        std = _VARIANT_MAP.get(ch)
        if std is not None:
            result.append(std)
        else:
            result.append(ch)
    return "".join(result)


# ── 竖排 OCR 噪音清理 ────────────────────────────────────────
# 竖排 PDF OCR 常见问题：字符串中插入换行、多余空格
_VERTICAL_NOISE = re.compile(
    r"(?<=[^\x00-\x7F])\s+(?=[^\x00-\x7F])"  # 中文字之间的空白
)

def clean_vertical_ocr(text: str) -> str:
    """清除竖排OCR产生的字间空白，保留段落换行"""
    # 先保留真正的段落分隔（连续2+换行）
    text = re.sub(r"\n{3,}", "\n\n", text)
    # 再去除中文字符之间的单个空白/换行
    text = _VERTICAL_NOISE.sub("", text)
    # 去除行首行尾多余空格
    lines = [l.strip() for l in text.splitlines()]
    return "\n".join(lines)


# ── 通用文本清洗 ─────────────────────────────────────────────
_JUNK_PATTERNS = [
    re.compile(r"第\s*[一二三四五六七八九十百\d]+\s*[页頁]"),    # 页码
    re.compile(r"[-—─]{3,}"),                                    # 分隔线
    re.compile(r"[■□●○◆◇▲△▼▽]+"),                              # 装饰符号
    re.compile(r"\s{4,}"),                                        # 4+连续空白 → 段落
    re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]"),           # 控制字符
]

def clean_text(raw: str) -> str:
    """通用清洗：去除页码、分隔线、控制字符"""
    for pat in _JUNK_PATTERNS:
        raw = pat.sub("", raw)
    # 合并多余空行
    raw = re.sub(r"\n{3,}", "\n\n", raw)
    return raw.strip()


# ── 语义分块器 ───────────────────────────────────────────────
# 中文句子结束标点
_SENT_END = re.compile(r"[。！？；…]{1,3}")
# 段落分隔
_PARA_SEP = re.compile(r"\n\s*\n")

def semantic_chunk(
    text: str,
    target_size: int = 400,
    max_size: int = 600,
    overlap_chars: int = 60,
) -> List[str]:
    """
    语义感知分块：
    1. 优先按段落分割
    2. 段落过长时按句子分割
    3. 块间保留 overlap_chars 字符上下文重叠
    4. 过短段落自动合并
    """
    paragraphs = _PARA_SEP.split(text)
    chunks: List[str] = []
    buffer = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        # 段落本身就超过 max_size → 按句子拆
        if len(para) > max_size:
            sentences = _split_sentences(para)
            for sent in sentences:
                if len(buffer) + len(sent) <= max_size:
                    buffer += sent
                else:
                    if buffer:
                        chunks.append(buffer)
                    # overlap：保留上一块末尾
                    tail = buffer[-overlap_chars:] if overlap_chars > 0 else ""
                    buffer = tail + sent
        else:
            if len(buffer) + len(para) <= max_size:
                buffer += ("\n" if buffer else "") + para
            else:
                if len(buffer) >= target_size // 2:
                    chunks.append(buffer)
                    tail = buffer[-overlap_chars:] if overlap_chars > 0 else ""
                    buffer = tail + para
                else:
                    buffer += ("\n" if buffer else "") + para

    if buffer.strip():
        chunks.append(buffer)

    # 合并过短的尾块
    if len(chunks) >= 2 and len(chunks[-1]) < target_size // 3:
        chunks[-2] += "\n" + chunks.pop()

    return [c.strip() for c in chunks if c.strip()]


def _split_sentences(text: str) -> List[str]:
    """按中文标点分割句子，保留标点"""
    parts = _SENT_END.split(text)
    puncts = _SENT_END.findall(text)
    sentences = []
    for i, part in enumerate(parts):
        sent = part + (puncts[i] if i < len(puncts) else "")
        if sent.strip():
            sentences.append(sent)
    return sentences


# ── PDF 解析 ─────────────────────────────────────────────────
def extract_pdf(path: Path) -> str:
    """
    用 pdfplumber 提取 PDF 文本。
    自动检测竖排并做 OCR 噪音清理。
    """
    try:
        import pdfplumber
    except ImportError:
        raise RuntimeError("请安装 pdfplumber: pip install pdfplumber")

    pages_text: List[str] = []
    with pdfplumber.open(str(path)) as pdf:
        for page in pdf.pages:
            # 先尝试普通文本提取
            text = page.extract_text(layout=True) or ""
            if not text.strip() and page.images:
                # 纯图片页：标记为待OCR
                text = f"[图片页 {page.page_number}，需OCR处理]"
            pages_text.append(text)

    full_text = "\n\n".join(pages_text)
    # 判断是否为竖排（竖排文本字间空白较多）
    sample = full_text[:2000]
    space_ratio = sample.count(" ") / max(len(sample), 1)
    if space_ratio > 0.15:
        full_text = clean_vertical_ocr(full_text)

    return full_text


# ── 主解析器类 ──────────────────────────────────────────────
class AncientTextParser:
    """
    将原始文献文件解析为标准化知识块列表。

    用法：
        parser = AncientTextParser(era="song", simplify=True)
        chunks = parser.parse(Path("data/raw_documents/song/黄州寒食帖.txt"))
        for chunk, meta in chunks:
            print(meta, chunk[:50])
    """

    def __init__(
        self,
        era: str = "unknown",
        simplify: bool = True,         # 是否繁→简（索引用）
        normalize_variants_flag: bool = True,
        target_chunk_size: int = 400,
    ):
        self.era = normalize_era_name(era)
        self.simplify = simplify
        self.norm_variants = normalize_variants_flag
        self.target_chunk_size = target_chunk_size
        self._dynasty_ce = {
            "song": 1100, "tang": 750, "ming": 1520, "qing": 1780,
            "han": 100, "yuan": 1320, "pre_qin": -400,
        }.get(self.era, 1000)

    def parse(self, path: Path) -> List[Tuple[str, dict]]:
        """
        解析单个文件，返回 [(chunk_text, metadata), ...] 列表。
        metadata 包含：source, era, year, chunk_idx, char_count, original_path
        """
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            raw = extract_pdf(path)
        elif suffix in (".txt", ".md", ".text"):
            raw = path.read_text(encoding="utf-8", errors="replace")
        else:
            raise ValueError(f"不支持的格式：{suffix}")

        # 清洗
        cleaned = clean_text(raw)
        if not cleaned:
            return []

        # 繁简转换（保留原文同时建索引用简体）
        index_text = trad_to_simp(cleaned) if self.simplify else cleaned
        if self.norm_variants:
            index_text = normalize_variants(index_text)

        # 提取年份（用于时间围栏）
        year = extract_year_from_text(index_text[:1000], dynasty_hint=self.era)

        # 分块
        chunks = semantic_chunk(
            index_text,
            target_size=self.target_chunk_size,
        )

        results = []
        for i, chunk in enumerate(chunks):
            # 每块单独提取年份（可能比文件级更精确）
            chunk_year = extract_year_from_text(chunk, dynasty_hint=self.era) or year
            meta = {
                "source": path.name,
                "original_path": str(path),
                "era": self.era,
                "year": chunk_year or self._dynasty_ce,
                "chunk_idx": i,
                "char_count": len(chunk),
                "type": "text",
            }
            results.append((chunk, meta))

        return results

    def parse_directory(self, dir_path: Path) -> Iterator[Tuple[str, dict]]:
        """批量解析目录下所有支持格式的文件"""
        supported = {".txt", ".md", ".pdf", ".text"}
        for p in sorted(dir_path.rglob("*")):
            if p.suffix.lower() in supported and p.is_file():
                try:
                    for chunk, meta in self.parse(p):
                        yield chunk, meta
                except Exception as e:
                    print(f"  ⚠️  解析失败 {p.name}: {e}")


# ── 命令行自测 ───────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    test_text = """
    元丰三年（公元1080年），苏轼以"乌台诗案"被贬黄州团练副使。
    
    黄州僻陋，苏子瞻初至，衣食拮据。然其性豁达，与渔樵杂处，
    躬耕东坡，自号"东坡居士"。
    
    时有佛印禅师，居金山寺，与苏轼诗文往来，互为知己。
    一日，苏轼书偈云："八风吹不动，端坐紫金莲。"
    佛印批曰："放屁。"苏轼怒而过江，佛印笑曰："八风吹不动，
    一屁打过江。"
    
    王朝云者，苏轼侍姬也，随侍黄州，伺候起居，与苏轼情谊深厚。
    朝云常言："学士满腹皆不合时宜。"苏轼深以为然。
    """

    parser = AncientTextParser(era="song")
    # 手动测试（不从文件读）
    cleaned = clean_text(test_text)
    index_text = trad_to_simp(cleaned)
    chunks = semantic_chunk(index_text, target_size=200)

    print("=== 古籍文本解析器自测 ===")
    print(f"原文 {len(test_text)} 字 → {len(chunks)} 块")
    for i, c in enumerate(chunks):
        year = extract_year_from_text(c, dynasty_hint="song")
        print(f"\n  [块{i+1}] {len(c)}字 | 年份={year}")
        print(f"  {c[:80]}...")
