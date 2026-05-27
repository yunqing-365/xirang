# source_workshop.py
"""
息壤 · Phase 14A · 史料直面工作坊

核心理念：
  让学生直接面对真实历史文献，而非二手叙述。
  奏折、日记、方志——不是装饰，是证据。

功能：
  1. SourceLibrary     — 史料片段库（按时代/类型/人物索引）
  2. AnnotationEngine  — 逐句注释（生僻字/典故/制度背景即点即解）
  3. CitationTracker   — 学生在对话中引用史料获得「实证加分」
  4. SourceComparator  — 同一事件的不同史料对比（矛盾与共识）
  5. WorkshopSession   — 会话级工作坊状态管理

史料嵌入流程：
  场景检索 → RAG 命中史料片段 → 工作坊解析注释 → 渲染到前端
  学生点击引用 → CitationTracker 记录 → agent 感知「玩家引用了史料」
"""
import asyncio
import json
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

from openai import AsyncOpenAI

from config import get_settings

_settings = get_settings()
_client = AsyncOpenAI(api_key=_settings.API_KEY, base_url=_settings.BASE_URL)


# ═══════════════════════════════════════════════════════════════
# 史料类型
# ═══════════════════════════════════════════════════════════════

class SourceDocType(str, Enum):
    MEMORIAL   = "奏折"      # 官员上书皇帝
    DIARY      = "日记"      # 私人日记
    GAZETTEER  = "方志"      # 地方志
    OFFICIAL   = "实录"      # 官方实录
    POETRY     = "诗词"      # 诗词（含史料价值）
    LETTER     = "书信"      # 私人书信
    ANNOTATION = "注疏"      # 经典注疏
    MISC       = "笔记"      # 私人笔记（如《梦溪笔谈》）
    EDICT      = "诏书"      # 皇帝诏令


# ═══════════════════════════════════════════════════════════════
# 史料片段
# ═══════════════════════════════════════════════════════════════

@dataclass
class SourceFragment:
    """一条史料片段——可独立展示给学生的最小单元"""
    fragment_id: str
    doc_type: SourceDocType
    title: str                    # 来源文献名称（如《宋史·苏轼传》）
    author: str                   # 作者/来源
    era: str                      # 时代
    year_hint: str                # 模糊年代（如「元丰二年」「约1079年」）
    original_text: str            # 原文（古文）
    modern_paraphrase: str        # 现代文意译（供参考）
    context_note: str             # 史料背景说明（50字以内）
    credibility_score: int        # 可信度 0–100
    related_figures: List[str]    # 涉及人物
    related_events: List[str]     # 涉及事件
    tags: List[str]               # 检索标签

    def to_dict(self) -> dict:
        return {
            "id": self.fragment_id,
            "doc_type": self.doc_type.value,
            "title": self.title,
            "author": self.author,
            "era": self.era,
            "year_hint": self.year_hint,
            "original_text": self.original_text,
            "modern_paraphrase": self.modern_paraphrase,
            "context_note": self.context_note,
            "credibility_score": self.credibility_score,
            "related_figures": self.related_figures,
            "tags": self.tags,
        }


# ═══════════════════════════════════════════════════════════════
# 注释引擎
# ═══════════════════════════════════════════════════════════════

@dataclass
class Annotation:
    """单条词汇注释"""
    term: str               # 被注释的词（可以是单字、词组、典故）
    annotation_type: str    # "字义" | "典故" | "制度" | "人名" | "地名"
    explanation: str        # 简要解释（30字以内）
    extended_note: str      # 详细背景（100字以内，点击展开）

    def to_dict(self) -> dict:
        return {
            "term": self.term,
            "type": self.annotation_type,
            "explanation": self.explanation,
            "extended_note": self.extended_note,
        }


class AnnotationEngine:
    """
    逐句注释引擎：对史料原文进行词汇级别的注解。
    生僻字 / 典故 / 官制 / 地名 → 即点即解。
    """

    # 常用注释缓存（避免重复 LLM 调用）
    _cache: Dict[str, List[Annotation]] = {}

    async def annotate(
        self,
        original_text: str,
        era: str,
        doc_type: str = "文言文",
    ) -> List[Annotation]:
        """
        对原文进行注释生成。
        返回该文本中需要注解的词汇列表。
        """
        cache_key = original_text[:40]
        if cache_key in self._cache:
            return self._cache[cache_key]

        prompt = (
            f"以下是一段{era}时期的{doc_type}原文：\n\n「{original_text[:300]}」\n\n"
            f"请识别其中需要注释的词汇（生僻字、典故、官制术语、地名、人名），"
            f"每个词给出简明注释。\n"
            f"以纯JSON数组输出，每项：\n"
            '{{"term": "词汇", "type": "字义/典故/制度/人名/地名", '
            '"explanation": "简要解释（20字以内）", '
            '"extended_note": "详细背景（60字以内）"}}\n'
            f"最多返回8个最重要的词汇，不要注释常见字。"
        )
        try:
            resp = await _client.chat.completions.create(
                model=_settings.MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                timeout=20,
                max_tokens=600,
            )
            raw = _strip_json(resp.choices[0].message.content)
            items = json.loads(raw)
            annotations = [
                Annotation(
                    term=item["term"],
                    annotation_type=item.get("type", "字义"),
                    explanation=item.get("explanation", ""),
                    extended_note=item.get("extended_note", ""),
                )
                for item in items
            ]
            self._cache[cache_key] = annotations
            return annotations
        except Exception as e:
            print(f"⚠️ [注释引擎] 失败: {e}")
            return []

    def build_annotated_html(self, original_text: str, annotations: List[Annotation]) -> str:
        """
        将注释嵌入原文，返回带 <ruby>/<span> 标注的 HTML 片段。
        前端渲染时词汇可高亮 + 悬停显示注释。
        """
        result = original_text
        # 按词长从长到短排序，避免短词覆盖长词
        sorted_annots = sorted(annotations, key=lambda a: len(a.term), reverse=True)
        for ann in sorted_annots:
            if ann.term not in result:
                continue
            tooltip = ann.explanation
            span = (
                f'<span class="src-term" '
                f'data-type="{ann.annotation_type}" '
                f'data-tip="{tooltip}" '
                f'data-ext="{ann.extended_note}" '
                f'onclick="showAnnotation(this)">'
                f'{ann.term}</span>'
            )
            result = result.replace(ann.term, span, 1)
        return result


# ═══════════════════════════════════════════════════════════════
# 引用追踪器
# ═══════════════════════════════════════════════════════════════

@dataclass
class CitationRecord:
    """学生引用史料的一条记录"""
    session_id: str
    fragment_id: str
    fragment_title: str
    cited_text: str          # 学生实际引用的文字
    round_num: int
    evidence_score: int      # 实证加分（5–20，根据引用质量）
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "fragment_id": self.fragment_id,
            "title": self.fragment_title,
            "cited": self.cited_text,
            "round": self.round_num,
            "score": self.evidence_score,
        }


class CitationTracker:
    """
    学生引用史料积分追踪器。
    当学生在对话中引用史料时，自动识别并给予「实证加分」。
    """

    def __init__(self):
        self.citations: List[CitationRecord] = []
        self.total_evidence_score: int = 0
        self._active_fragments: Dict[str, SourceFragment] = {}  # 当前会话的活跃史料

    def register_fragment(self, fragment: SourceFragment):
        """注册一条活跃史料（供引用识别）"""
        self._active_fragments[fragment.fragment_id] = fragment

    def check_citation(
        self,
        player_input: str,
        session_id: str,
        round_num: int,
    ) -> Optional[CitationRecord]:
        """
        检测玩家输入中是否引用了已展示的史料片段。
        返回 CitationRecord（若有）。
        """
        for fid, fragment in self._active_fragments.items():
            # 检测原文的4字以上子串
            for length in range(min(20, len(fragment.original_text)), 3, -1):
                for start in range(len(fragment.original_text) - length + 1):
                    snippet = fragment.original_text[start:start + length]
                    if snippet in player_input:
                        score = min(20, max(5, length * 2))
                        record = CitationRecord(
                            session_id=session_id,
                            fragment_id=fid,
                            fragment_title=fragment.title,
                            cited_text=snippet,
                            round_num=round_num,
                            evidence_score=score,
                        )
                        self.citations.append(record)
                        self.total_evidence_score += score
                        return record
        return None

    def get_agent_awareness(self) -> str:
        """
        生成给 Agent 的感知提示：
        告知 NPC 玩家刚刚引用了某段史料，让 NPC 能自然回应。
        """
        if not self.citations:
            return ""
        last = self.citations[-1]
        return (
            f"【重要提示】玩家刚刚引用了史料「{last.fragment_title}」中的文字，"
            f"说明他们认真研读了这份文献。请在回应中适当认可这种实证态度，"
            f"可以顺势深入讨论该史料的背景或细节。"
        )

    def to_dict(self) -> dict:
        return {
            "total_score": self.total_evidence_score,
            "citations": [c.to_dict() for c in self.citations],
            "count": len(self.citations),
        }


# ═══════════════════════════════════════════════════════════════
# 史料对比器
# ═══════════════════════════════════════════════════════════════

@dataclass
class ComparisonResult:
    """两条史料的对比分析结果"""
    event_desc: str
    fragment_a: SourceFragment
    fragment_b: SourceFragment
    agreements: List[str]       # 两者一致的地方
    contradictions: List[str]   # 矛盾与出入
    possible_reasons: List[str] # 为什么有出入（立场/时间/目的）
    study_question: str         # 对学生的追问

    def to_dict(self) -> dict:
        return {
            "event": self.event_desc,
            "source_a": {"title": self.fragment_a.title, "text": self.fragment_a.original_text[:100]},
            "source_b": {"title": self.fragment_b.title, "text": self.fragment_b.original_text[:100]},
            "agreements": self.agreements,
            "contradictions": self.contradictions,
            "possible_reasons": self.possible_reasons,
            "study_question": self.study_question,
        }


class SourceComparator:
    """同一事件的不同史料对比"""

    async def compare(
        self,
        fragment_a: SourceFragment,
        fragment_b: SourceFragment,
        event_desc: str,
    ) -> ComparisonResult:
        prompt = (
            f"请对比以下两段关于「{event_desc}」的史料：\n\n"
            f"【史料甲】{fragment_a.title}（{fragment_a.doc_type.value}·{fragment_a.year_hint}）\n"
            f"{fragment_a.original_text[:200]}\n\n"
            f"【史料乙】{fragment_b.title}（{fragment_b.doc_type.value}·{fragment_b.year_hint}）\n"
            f"{fragment_b.original_text[:200]}\n\n"
            f"以纯JSON输出：\n"
            '{{"agreements": ["一致点1","一致点2"], '
            '"contradictions": ["矛盾点1","矛盾点2"], '
            '"possible_reasons": ["原因1","原因2"], '
            '"study_question": "给学生的探究追问（1句）"}}'
        )
        try:
            resp = await _client.chat.completions.create(
                model=_settings.MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.4,
                timeout=20,
                max_tokens=500,
            )
            data = json.loads(_strip_json(resp.choices[0].message.content))
            return ComparisonResult(
                event_desc=event_desc,
                fragment_a=fragment_a,
                fragment_b=fragment_b,
                agreements=data.get("agreements", []),
                contradictions=data.get("contradictions", []),
                possible_reasons=data.get("possible_reasons", []),
                study_question=data.get("study_question", ""),
            )
        except Exception as e:
            print(f"⚠️ [史料对比] 失败: {e}")
            return ComparisonResult(
                event_desc=event_desc,
                fragment_a=fragment_a, fragment_b=fragment_b,
                agreements=[], contradictions=[], possible_reasons=[],
                study_question="两份史料对同一事件的记述有何不同？",
            )


# ═══════════════════════════════════════════════════════════════
# 内置史料片段库（北宋·苏轼时代示例，可扩展）
# ═══════════════════════════════════════════════════════════════

BUILTIN_SOURCES: List[SourceFragment] = [
    SourceFragment(
        fragment_id="src_susong_zizhi",
        doc_type=SourceDocType.OFFICIAL,
        title="《续资治通鉴长编》卷二百九十九",
        author="李焘（南宋）",
        era="北宋",
        year_hint="元丰二年（1079年）",
        original_text=(
            "御史李定、何正臣、舒亶等劾轼以诗文讪谤，"
            "轼坐系御史台狱。"
            "神宗览其诗，叹曰：『轼固大才，岂可以此杀之。』"
            "乃贬黄州团练副使，本州安置，不得签书公事。"
        ),
        modern_paraphrase=(
            "御史台的官员弹劾苏轼，认为他的诗文讽刺朝政。"
            "苏轼因此被关押在御史台。神宗皇帝看了他的诗，叹息说苏轼是大才，"
            "不能因此杀了他。于是将他贬为黄州团练副使，留在本州，不许处理公务。"
        ),
        context_note="乌台诗案的官方记载，出自南宋史家李焘的编年史著作，距事件约百年。",
        credibility_score=68,
        related_figures=["苏轼", "李定", "宋神宗"],
        related_events=["乌台诗案", "苏轼贬谪黄州"],
        tags=["乌台诗案", "贬谪", "官方记载", "神宗", "苏轼"],
    ),
    SourceFragment(
        fragment_id="src_su_letter_ziyou",
        doc_type=SourceDocType.LETTER,
        title="《与子由书》",
        author="苏轼",
        era="北宋",
        year_hint="元丰三年（1080年）",
        original_text=(
            "吾侪小人，才德俱不逮古人远甚，"
            "每自揣量，辄便汗出。"
            "到黄州，颇有所得，真所谓『因祸得福』者。"
            "只是少饮食，少睡，然此非病，乃所以养病也。"
        ),
        modern_paraphrase=(
            "我们这样的小人物，才能和品德都远不及古人，"
            "每次自我反省，都不禁汗流浃背。"
            "到了黄州之后，倒颇有些收获，真可谓因祸得福。"
            "只是吃得少，睡得少，但这并非生病，而是在休养调理。"
        ),
        context_note="苏轼贬谪黄州初期写给弟弟苏辙的信，一手史料，情感真实。",
        credibility_score=88,
        related_figures=["苏轼", "苏辙"],
        related_events=["黄州贬谪生活"],
        tags=["书信", "黄州", "一手史料", "苏辙", "心态"],
    ),
    SourceFragment(
        fragment_id="src_dongpo_zhi_lin",
        doc_type=SourceDocType.MISC,
        title="《东坡志林》",
        author="苏轼",
        era="北宋",
        year_hint="元丰至元祐年间（约1080–1094年）",
        original_text=(
            "余谪居黄州，春夜行蕲水中，"
            "过酒家，饮酒醉，乘月至一溪桥上，"
            "解鞍曲肱，醉卧少休，及觉已晓，"
            "乱山攒拥，流水铿然，疑非尘世也。"
        ),
        modern_paraphrase=(
            "我被贬到黄州时，春天夜里在蕲水中行走，"
            "路过一家酒馆，喝了酒醉了，借着月色到一座溪桥上，"
            "解下马鞍枕着手臂，醉倒小憩。等到醒来，天已经亮了。"
            "四周群山环抱，流水声叮咚悦耳，仿佛不是人间。"
        ),
        context_note="苏轼亲笔随笔，记录黄州贬谪期间的生活片段，极具个人风格。",
        credibility_score=90,
        related_figures=["苏轼"],
        related_events=["黄州贬谪生活", "苏轼山水情怀"],
        tags=["随笔", "黄州", "自然", "一手史料", "生活"],
    ),
]

# 快速检索索引
_SOURCE_INDEX: Dict[str, SourceFragment] = {s.fragment_id: s for s in BUILTIN_SOURCES}


# ═══════════════════════════════════════════════════════════════
# 会话级工作坊
# ═══════════════════════════════════════════════════════════════

class WorkshopSession:
    """
    会话级史料工作坊：管理当前会话中的史料展示、注释和引用。
    """

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.annotation_engine = AnnotationEngine()
        self.citation_tracker = CitationTracker()
        self.comparator = SourceComparator()
        self._shown_fragments: List[SourceFragment] = []
        self._annotations_cache: Dict[str, List[Annotation]] = {}

    async def present_fragment(
        self,
        fragment: SourceFragment,
        era: str = "",
    ) -> dict:
        """
        向学生呈现一条史料，同时生成注释。
        返回前端所需的完整展示数据。
        """
        self._shown_fragments.append(fragment)
        self.citation_tracker.register_fragment(fragment)

        # 并发生成注释
        annotations = await self.annotation_engine.annotate(
            fragment.original_text,
            era or fragment.era,
            fragment.doc_type.value,
        )
        self._annotations_cache[fragment.fragment_id] = annotations
        annotated_html = self.annotation_engine.build_annotated_html(
            fragment.original_text, annotations
        )

        return {
            "fragment": fragment.to_dict(),
            "annotated_html": annotated_html,
            "annotations": [a.to_dict() for a in annotations],
        }

    def on_player_input(self, player_input: str, round_num: int) -> Optional[dict]:
        """
        检测玩家输入中是否引用了史料，返回引用记录（如有）。
        """
        record = self.citation_tracker.check_citation(
            player_input, self.session_id, round_num
        )
        if record:
            return {
                "cited": True,
                "record": record.to_dict(),
                "total_score": self.citation_tracker.total_evidence_score,
                "agent_awareness": self.citation_tracker.get_agent_awareness(),
            }
        return None

    async def compare_fragments(self, id_a: str, id_b: str, event_desc: str) -> dict:
        fa = _SOURCE_INDEX.get(id_a)
        fb = _SOURCE_INDEX.get(id_b)
        if not fa or not fb:
            # 从已展示的片段中查找
            shown_index = {f.fragment_id: f for f in self._shown_fragments}
            fa = fa or shown_index.get(id_a)
            fb = fb or shown_index.get(id_b)
        if not fa or not fb:
            return {"error": "史料片段不存在"}
        result = await self.comparator.compare(fa, fb, event_desc)
        return result.to_dict()

    def search_sources(self, query: str, era: str = "") -> List[dict]:
        """本地关键词搜索内置史料库"""
        results = []
        for src in BUILTIN_SOURCES:
            if era and src.era != era:
                continue
            score = 0
            for tag in src.tags:
                if tag in query:
                    score += 2
            if query in src.original_text:
                score += 3
            for figure in src.related_figures:
                if figure in query:
                    score += 2
            if score > 0:
                results.append((score, src))
        results.sort(key=lambda x: x[0], reverse=True)
        return [s.to_dict() for _, s in results[:5]]

    def get_citation_summary(self) -> dict:
        return self.citation_tracker.to_dict()

    def get_agent_citation_context(self) -> str:
        return self.citation_tracker.get_agent_awareness()


# ── 全局会话注册表 ────────────────────────────────────────────

_workshop_sessions: Dict[str, WorkshopSession] = {}


def get_workshop(session_id: str) -> WorkshopSession:
    if session_id not in _workshop_sessions:
        _workshop_sessions[session_id] = WorkshopSession(session_id)
    return _workshop_sessions[session_id]


def get_source_by_id(fragment_id: str) -> Optional[SourceFragment]:
    return _SOURCE_INDEX.get(fragment_id)


def get_all_sources(era: str = "") -> List[dict]:
    sources = BUILTIN_SOURCES if not era else [s for s in BUILTIN_SOURCES if s.era == era]
    return [s.to_dict() for s in sources]


# ── 工具函数 ──────────────────────────────────────────────────

def _strip_json(text: str) -> str:
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()
