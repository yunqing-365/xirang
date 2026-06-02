# concept_engine.py
"""
息壤 · Phase 12B · 大概念叙事锚点系统

核心理念：
  每一次历史体验都在回答一个历史学大问题。
  玩家不是在"背知识"，而是在不知不觉中触碰历史学的深层概念。

12 个历史学核心大概念（参考 IB History + 国内历史课程标准）：
  变革(CHANGE) · 延续(CONTINUITY) · 因果(CAUSATION)
  后果(CONSEQUENCE) · 视角(PERSPECTIVE) · 证据(EVIDENCE)
  权力(POWER) · 文明交流(EXCHANGE) · 集体记忆(MEMORY)
  身份认同(IDENTITY) · 道德判断(ETHICS) · 时序(CHRONOLOGY)

功能：
  1. 每次会话自动识别当前涉及的大概念（基于对话关键词）
  2. 剧情选项携带隐性概念标签（引导学生思维方向）
  3. 会话结束生成「本次你触碰了哪些历史学问题」总结卡
  4. 跨会话积累学生的概念掌握图谱（写入 UserProfile）
"""
import asyncio
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set

from openai import AsyncOpenAI
from infra.resilience import llm_guard, result_cache, annotation_cache, cross_link_cache, perspective_cache, graceful_degradation

from config import get_settings
from prompt_templates import _strip_code_fence_compat  # noqa – 若不存在则下方自定义

_settings = get_settings()
_client = AsyncOpenAI(api_key=_settings.API_KEY, base_url=_settings.BASE_URL)


# ═══════════════════════════════════════════════════════════════
# 12 大核心概念
# ═══════════════════════════════════════════════════════════════

class HistoryConcept(str, Enum):
    CHANGE       = "变革"       # 事物如何改变？为什么改变？
    CONTINUITY   = "延续"       # 什么保持不变？为什么？
    CAUSATION    = "因果"       # 是什么导致了这件事？
    CONSEQUENCE  = "后果"       # 这件事带来了什么影响？
    PERSPECTIVE  = "视角"       # 不同的人如何看待同一件事？
    EVIDENCE     = "证据"       # 我们怎么知道历史上发生了什么？
    POWER        = "权力"       # 谁有权力？权力如何运作？
    EXCHANGE     = "文明交流"   # 人、思想、物品如何流通？
    MEMORY       = "集体记忆"   # 历史如何被记住？被谁记住？
    IDENTITY     = "身份认同"   # 人们如何定义自己？
    ETHICS       = "道德判断"   # 如何评价历史人物的选择？
    CHRONOLOGY   = "时序"       # 事件的先后顺序如何影响理解？


# 每个概念的简短定义（用于生成总结卡）
CONCEPT_DEFINITIONS: Dict[HistoryConcept, str] = {
    HistoryConcept.CHANGE:      "历史中的变革——是渐变还是突变？谁推动了变化？",
    HistoryConcept.CONTINUITY:  "历史的延续——哪些东西跨越时代保持不变？",
    HistoryConcept.CAUSATION:   "历史的因果——是什么引发了这件事？近因与远因有何不同？",
    HistoryConcept.CONSEQUENCE: "历史的后果——这件事改变了什么？影响延续了多久？",
    HistoryConcept.PERSPECTIVE: "历史的视角——同一事件，不同身份的人看法截然不同。",
    HistoryConcept.EVIDENCE:    "历史的证据——史料如何塑造我们对过去的认知？",
    HistoryConcept.POWER:       "历史中的权力——权力如何被获取、运用和失去？",
    HistoryConcept.EXCHANGE:    "文明的交流——人、思想与物品如何跨越边界流动？",
    HistoryConcept.MEMORY:      "集体的记忆——历史如何被选择性地记住与遗忘？",
    HistoryConcept.IDENTITY:    "身份与认同——历史如何塑造人们对自己和群体的认知？",
    HistoryConcept.ETHICS:      "道德的评判——我们能用今天的标准评价历史人物吗？",
    HistoryConcept.CHRONOLOGY:  "时序的意义——事件的先后如何影响我们对历史的理解？",
}

# 概念的触发关键词（轻量级本地识别）
CONCEPT_KEYWORDS: Dict[HistoryConcept, List[str]] = {
    HistoryConcept.CHANGE:      ["变法", "改革", "变化", "革新", "废除", "建立", "取代"],
    HistoryConcept.CONTINUITY:  ["传统", "沿袭", "延续", "保留", "不变", "祖制", "惯例"],
    HistoryConcept.CAUSATION:   ["为什么", "原因", "导致", "因为", "引发", "根源", "缘由"],
    HistoryConcept.CONSEQUENCE: ["结果", "影响", "后果", "导致了", "从此", "因此", "使得"],
    HistoryConcept.PERSPECTIVE: ["认为", "在他看来", "不同角度", "对方", "立场", "视角", "理解"],
    HistoryConcept.EVIDENCE:    ["史料", "记载", "奏折", "日记", "文献", "考证", "史书"],
    HistoryConcept.POWER:       ["权力", "皇帝", "朝廷", "官职", "变法", "派系", "弹劾", "贬谪"],
    HistoryConcept.EXCHANGE:    ["丝绸之路", "传播", "影响", "交流", "贸易", "传入", "流通"],
    HistoryConcept.MEMORY:      ["历史评价", "后人", "纪念", "遗忘", "铭记", "形象", "评价"],
    HistoryConcept.IDENTITY:    ["身份", "归属", "认同", "士大夫", "汉人", "文人", "民族"],
    HistoryConcept.ETHICS:      ["对错", "是否正确", "应该", "评价", "功过", "道德", "是非"],
    HistoryConcept.CHRONOLOGY:  ["之前", "之后", "先后", "年代", "时期", "顺序", "背景"],
}

# 概念对应的探究提问（Bloom 高阶）
CONCEPT_INQUIRY_QUESTIONS: Dict[HistoryConcept, List[str]] = {
    HistoryConcept.CHANGE: [
        "这场变革是由上而下推动的，还是由下而上涌现的？",
        "谁是变革的受益者？谁是损失者？",
        "如果没有这场变革，历史会怎样发展？",
    ],
    HistoryConcept.CAUSATION: [
        "你认为最根本的原因是什么？是偶然还是必然？",
        "如果去掉其中一个原因，事情还会发生吗？",
        "历史上有没有类似的『导火索』？",
    ],
    HistoryConcept.PERSPECTIVE: [
        "如果你是另一个立场的人，你会如何看待这件事？",
        "是什么决定了人们各自的立场？",
        "历史上谁的声音更容易被记录下来？谁的声音被遗失了？",
    ],
    HistoryConcept.POWER: [
        "权力是如何从一个人转移到另一个人手中的？",
        "普通人在这段历史中有没有任何力量？",
        "这种权力结构今天还存在吗？",
    ],
    HistoryConcept.ETHICS: [
        "用今天的标准评判古人公平吗？为什么？",
        "如果你身处同样的处境，你会做出相同的选择吗？",
        "历史上有没有『虽然错了，但可以理解』的选择？",
    ],
    HistoryConcept.EVIDENCE: [
        "这段历史是谁记录下来的？他们有没有立场？",
        "如果这份史料是伪造的，我们的判断会改变吗？",
        "哪些历史是没有文字记载的？我们如何重建它？",
    ],
    HistoryConcept.MEMORY: [
        "为什么这个人/这件事被记住了，而其他的被遗忘了？",
        "不同时代的人如何评价同一个历史人物？为什么评价会变？",
        "今天我们讲述这段历史，我们选择了什么？省略了什么？",
    ],
    HistoryConcept.CONSEQUENCE: [
        "这件事最重要的短期后果是什么？长期后果是什么？",
        "有没有意想不到的后果？",
        "如果历史可以重来，哪个决定会最大程度改变结果？",
    ],
    HistoryConcept.IDENTITY: [
        "这个人是如何定义自己的？什么影响了他的自我认知？",
        "群体认同如何影响个人的选择？",
        "今天的你和历史中的人物在身份认同上有什么相似之处？",
    ],
    HistoryConcept.CONTINUITY: [
        "这段历史中，什么东西一直延续到了今天？",
        "传统与革新之间的张力是如何解决的？",
        "为什么有些东西看似改变，实质上却从未变过？",
    ],
    HistoryConcept.EXCHANGE: [
        "这次交流是对等的还是单向的？谁更受益？",
        "思想和物品的流通如何改变了接受者？",
        "今天的全球化和历史上的文明交流有何相似？",
    ],
    HistoryConcept.CHRONOLOGY: [
        "如果这两件事的顺序互换，结果会不同吗？",
        "为什么了解背景（前因）对理解历史至关重要？",
        "这段历史的『起点』应该从哪里算起？为什么？",
    ],
}


# ═══════════════════════════════════════════════════════════════
# 会话级概念追踪
# ═══════════════════════════════════════════════════════════════

@dataclass
class ConceptTouch:
    """一次「触碰」记录"""
    concept: HistoryConcept
    round_num: int
    context_snippet: str      # 触发该概念的对话片段（<= 80字）
    depth_score: int = 1      # 1=浅触（提及）2=中度（讨论）3=深度（推理）

    def to_dict(self) -> dict:
        return {
            "concept": self.concept.value,
            "round": self.round_num,
            "context": self.context_snippet,
            "depth": self.depth_score,
        }


class ConceptTracker:
    """
    单个会话的概念追踪器。
    记录本次体验中触碰了哪些概念、深度如何。
    """

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.touches: List[ConceptTouch] = []
        self.active_concepts: Set[HistoryConcept] = set()
        self._round = 0

    def advance_round(self):
        self._round += 1

    def record_touch(self, concept: HistoryConcept, context: str, depth: int = 1):
        self.touches.append(ConceptTouch(
            concept=concept,
            round_num=self._round,
            context_snippet=context[:80],
            depth_score=depth,
        ))
        self.active_concepts.add(concept)

    def get_active_concepts(self) -> List[HistoryConcept]:
        """返回本会话中触碰过的所有概念（去重）"""
        return list(self.active_concepts)

    def get_top_concepts(self, n: int = 3) -> List[HistoryConcept]:
        """返回被触碰最多次的 N 个概念"""
        from collections import Counter
        counts = Counter(t.concept for t in self.touches)
        return [c for c, _ in counts.most_common(n)]

    def generate_summary_card(self) -> dict:
        """
        生成「本次你触碰了哪些历史学问题」总结卡。
        返回结构化数据，供前端渲染。
        """
        top = self.get_top_concepts(3)
        return {
            "session_id": self.session_id,
            "total_touches": len(self.touches),
            "concepts_explored": len(self.active_concepts),
            "highlighted_concepts": [
                {
                    "concept": c.value,
                    "definition": CONCEPT_DEFINITIONS[c],
                    "times_touched": sum(1 for t in self.touches if t.concept == c),
                    "inquiry_question": CONCEPT_INQUIRY_QUESTIONS[c][0],
                }
                for c in top
            ],
            "all_concepts": [c.value for c in self.active_concepts],
        }

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "touches": [t.to_dict() for t in self.touches],
            "active_concepts": [c.value for c in self.active_concepts],
        }


# ═══════════════════════════════════════════════════════════════
# 大概念引擎
# ═══════════════════════════════════════════════════════════════

class ConceptEngine:
    """
    系统级大概念引擎：跨会话管理所有 ConceptTracker。
    提供：本地关键词识别（快） + LLM 深度分析（慢，可选）。
    """

    def __init__(self):
        self._trackers: Dict[str, ConceptTracker] = {}

    def get_or_create_tracker(self, session_id: str) -> ConceptTracker:
        if session_id not in self._trackers:
            self._trackers[session_id] = ConceptTracker(session_id)
        return self._trackers[session_id]

    # ── 本地轻量识别（每回合调用） ────────────────────────────

    def scan_text(
        self,
        session_id: str,
        text: str,
        round_num: int = 0,
    ) -> List[HistoryConcept]:
        """
        从对话文本中快速识别涉及的大概念（本地关键词匹配）。
        每命中一个概念，自动记录到 tracker。
        """
        tracker = self.get_or_create_tracker(session_id)
        found: List[HistoryConcept] = []
        text_lower = text.lower()

        for concept, keywords in CONCEPT_KEYWORDS.items():
            hits = sum(1 for kw in keywords if kw in text_lower)
            if hits >= 1:
                depth = min(3, hits)
                tracker.record_touch(concept, text[:80], depth)
                found.append(concept)

        return found

    # ── LLM 深度分析（会话结束或按需调用） ───────────────────

    async def analyze_with_llm(
        self,
        session_id: str,
        dialogue_summary: str,
    ) -> List[HistoryConcept]:
        """
        使用 LLM 对完整对话进行深度概念分析。
        比本地扫描更准确，但耗时更长，建议会话结束时调用。
        """
        tracker = self.get_or_create_tracker(session_id)
        concept_list = "\n".join(
            f"- {c.value}：{CONCEPT_DEFINITIONS[c]}" for c in HistoryConcept
        )
        prompt = (
            f"以下是一段历史沉浸体验的对话摘要：\n{dialogue_summary[:800]}\n\n"
            f"请从以下12个历史学大概念中，识别出这段对话最主要涉及的概念（最多5个）：\n"
            f"{concept_list}\n\n"
            f"请只返回JSON数组，例如：[\"变革\", \"权力\", \"道德判断\"]"
        )
        try:
            resp = await _client.chat.completions.create(
                model=_settings.MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                timeout=15,
                max_tokens=100,
            )
            raw = resp.choices[0].message.content.strip()
            raw = _strip_json(raw)
            names = json.loads(raw)
            result = []
            for name in names:
                for c in HistoryConcept:
                    if c.value == name:
                        tracker.record_touch(c, "LLM深度分析", depth=2)
                        result.append(c)
            return result
        except Exception as e:
            print(f"⚠️ [概念引擎] LLM分析失败: {e}")
            return tracker.get_active_concepts()

    # ── 获取本回合探究问题 ────────────────────────────────────

    def get_round_questions(
        self,
        session_id: str,
        n: int = 3,
    ) -> List[dict]:
        """
        为本回合涉及的概念生成探究问题。
        这是 Phase 13B（探究式问题生成器）的协作接口。
        """
        tracker = self.get_or_create_tracker(session_id)
        top = tracker.get_top_concepts(n)
        questions = []
        for concept in top:
            qs = CONCEPT_INQUIRY_QUESTIONS.get(concept, [])
            if qs:
                questions.append({
                    "concept": concept.value,
                    "question": qs[min(tracker._round % len(qs), len(qs) - 1)],
                    "depth_hint": f"💡 这是一个关于「{concept.value}」的问题",
                })
        return questions

    # ── 会话总结卡 ────────────────────────────────────────────

    def get_session_summary(self, session_id: str) -> Optional[dict]:
        tracker = self._trackers.get(session_id)
        if not tracker:
            return None
        return tracker.generate_summary_card()

    # ── 标注选项的隐性概念标签 ───────────────────────────────

    def annotate_choice(self, choice_text: str) -> Optional[HistoryConcept]:
        """
        为一个叙事选项文本推断其主要涉及的大概念（隐性标签）。
        用于前端在选项旁边显示小徽章，引导学生注意概念。
        """
        scores: Dict[HistoryConcept, int] = {}
        for concept, keywords in CONCEPT_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in choice_text)
            if score > 0:
                scores[concept] = score
        if not scores:
            return None
        return max(scores, key=lambda c: scores[c])


# ═══════════════════════════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════════════════════════

def _strip_json(text: str) -> str:
    text = text.strip()
    for prefix in ("```json", "```"):
        if text.startswith(prefix):
            text = text[len(prefix):]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()

# Phase 15B: 全局实例引用槽（由 server.py 启动时注入）
_concept_engine_global = None
