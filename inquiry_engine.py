# inquiry_engine.py
"""
息壤 · Phase 13B · 探究式问题生成器

核心理念：
  不给答案，给更好的问题。
  苏格拉底的方法：通过追问让学生自己推导出真相。

模块组成：
  1. BloomQuestionGenerator   — Bloom高阶分层问题生成
  2. SocraticDialogueEngine   — 苏格拉底式追问引擎（AI持续追问，不给答案）
  3. QuestionNotebook         — 学生问题收藏本（个人问题集）

Bloom 分层（本系统聚焦高阶）：
  L1 记忆  — 你记得…？（本系统较少使用）
  L2 理解  — 用自己的话解释…
  L3 应用  — 如果换一个情境…
  L4 分析  — 比较/解构/找出原因
  L5 评价  — 你认为…是否正确？为什么？
  L6 创造  — 如果你来决策，你会…？
"""
import asyncio
import json
import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import AsyncGenerator, Dict, List, Optional

from openai import AsyncOpenAI
from infra.resilience import llm_guard, result_cache, annotation_cache, cross_link_cache, perspective_cache, graceful_degradation

from config import get_settings

_settings = get_settings()
_client = AsyncOpenAI(api_key=_settings.API_KEY, base_url=_settings.BASE_URL)


# ═══════════════════════════════════════════════════════════════
# Bloom 层级
# ═══════════════════════════════════════════════════════════════

class BloomLevel(IntEnum):
    REMEMBER  = 1   # 记忆
    UNDERSTAND = 2  # 理解
    APPLY     = 3   # 应用
    ANALYZE   = 4   # 分析
    EVALUATE  = 5   # 评价
    CREATE    = 6   # 创造


BLOOM_LABELS = {
    BloomLevel.REMEMBER:   "记忆",
    BloomLevel.UNDERSTAND: "理解",
    BloomLevel.APPLY:      "应用",
    BloomLevel.ANALYZE:    "分析",
    BloomLevel.EVALUATE:   "评价",
    BloomLevel.CREATE:     "创造",
}

BLOOM_STEMS = {
    BloomLevel.REMEMBER:   ["你还记得", "列举出", "说出"],
    BloomLevel.UNDERSTAND: ["用自己的话解释", "你如何理解", "描述一下"],
    BloomLevel.APPLY:      ["如果换一个情境", "举一个类似的例子", "你会如何运用"],
    BloomLevel.ANALYZE:    ["比较一下", "找出原因", "是什么导致了", "有哪些不同"],
    BloomLevel.EVALUATE:   ["你认为这样做是否正确", "你如何评价", "这个决定值得吗"],
    BloomLevel.CREATE:     ["如果由你来决策", "你会如何改变", "设计一个方案"],
}

BLOOM_COLORS = {
    BloomLevel.REMEMBER:   "#95A5A6",
    BloomLevel.UNDERSTAND: "#3498DB",
    BloomLevel.APPLY:      "#2ECC71",
    BloomLevel.ANALYZE:    "#F39C12",
    BloomLevel.EVALUATE:   "#E74C3C",
    BloomLevel.CREATE:     "#9B59B6",
}


# ═══════════════════════════════════════════════════════════════
# 问题数据类
# ═══════════════════════════════════════════════════════════════

@dataclass
class InquiryQuestion:
    """一道探究问题"""
    id: str
    text: str
    bloom_level: BloomLevel
    concept: Optional[str]           # 关联的历史学大概念
    hint: Optional[str]              # 卡住时的提示（不直接给答案）
    follow_up: Optional[str]         # 追问方向
    is_bookmarked: bool = False
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "text": self.text,
            "bloom_level": self.bloom_level.value,
            "bloom_label": BLOOM_LABELS[self.bloom_level],
            "bloom_color": BLOOM_COLORS[self.bloom_level],
            "concept": self.concept,
            "hint": self.hint,
            "follow_up": self.follow_up,
            "is_bookmarked": self.is_bookmarked,
        }


# ═══════════════════════════════════════════════════════════════
# 1. Bloom 分层问题生成器
# ═══════════════════════════════════════════════════════════════

class BloomQuestionGenerator:
    """
    为当前历史场景生成 Bloom 高阶分层问题。
    系统默认生成 L4/L5/L6（分析/评价/创造）三层，
    可按需生成 L2/L3 辅助层。
    """

    _question_counter = 0

    @classmethod
    def _new_id(cls) -> str:
        cls._question_counter += 1
        return f"q_{int(time.time())}_{cls._question_counter}"

    async def generate(
        self,
        scene_desc: str,
        event_summary: str,
        era: str,
        concepts: Optional[List[str]] = None,
        levels: Optional[List[BloomLevel]] = None,
        n: int = 3,
    ) -> List[InquiryQuestion]:
        """
        为当前场景生成 n 道探究问题（默认 L4/L5/L6）。
        """
        if levels is None:
            levels = [BloomLevel.ANALYZE, BloomLevel.EVALUATE, BloomLevel.CREATE]

        concept_str = "、".join(concepts) if concepts else "变革、权力、道德判断"
        level_str = "\n".join(
            f"- 第{i+1}题：Bloom {BLOOM_LABELS[lv]}层（L{lv.value}）"
            for i, lv in enumerate(levels[:n])
        )

        prompt = (
            f"你是一位深谙历史教育的苏格拉底式提问专家。\n"
            f"时代：{era}\n"
            f"当前场景：{scene_desc[:150]}\n"
            f"刚发生的事件：{event_summary[:200]}\n"
            f"涉及的历史学概念：{concept_str}\n\n"
            f"请生成{n}道开放式探究问题，按以下层级分配：\n"
            f"{level_str}\n\n"
            f"要求：\n"
            f"1. 问题必须与当前场景直接相关，不能空泛\n"
            f"2. 不能有标准答案，必须是开放性的\n"
            f"3. 每道题附带一个『若卡住可参考的方向』（不是答案，是思路提示）\n"
            f"4. 每道题附带一个自然的追问方向\n\n"
            f"以纯JSON数组输出，每项：\n"
            '{{"text": "问题", "bloom_level": L4/L5/L6数字, "concept": "关联概念", '
            '"hint": "思路提示（非答案）", "follow_up": "追问方向"}}'
        )

        try:
            resp = await _client.chat.completions.create(
                model=_settings.MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.85,
                timeout=25,
                max_tokens=800,
            )
            raw = _strip_json(resp.choices[0].message.content)
            items = json.loads(raw)
            questions = []
            for item in items:
                lvl_raw = item.get("bloom_level", 4)
                try:
                    lvl = BloomLevel(int(lvl_raw))
                except ValueError:
                    lvl = BloomLevel.ANALYZE
                questions.append(InquiryQuestion(
                    id=self._new_id(),
                    text=item["text"],
                    bloom_level=lvl,
                    concept=item.get("concept"),
                    hint=item.get("hint"),
                    follow_up=item.get("follow_up"),
                ))
            return questions
        except Exception as e:
            print(f"⚠️ [Bloom提问] 生成失败: {e}")
            return self._fallback_questions(scene_desc, era)

    def _fallback_questions(self, scene_desc: str, era: str) -> List[InquiryQuestion]:
        return [
            InquiryQuestion(
                id=self._new_id(), bloom_level=BloomLevel.ANALYZE,
                text=f"在{era}这个历史背景下，是什么结构性因素导致了眼前这一局面？",
                concept="因果",
                hint="从政治、经济、文化三个角度分别想想",
                follow_up="如果其中一个因素不存在，结果会改变吗？",
            ),
            InquiryQuestion(
                id=self._new_id(), bloom_level=BloomLevel.EVALUATE,
                text="你如何评价这位历史人物的选择？用今天的标准评判他公平吗？",
                concept="道德判断",
                hint="先描述他所处的约束条件，再做评判",
                follow_up="如果你生活在那个时代，你会做出相同的选择吗？",
            ),
            InquiryQuestion(
                id=self._new_id(), bloom_level=BloomLevel.CREATE,
                text="如果你是这段历史的决策者，你会做出什么不同的决定？可能带来什么后果？",
                concept="后果",
                hint="先分析当时的资源和限制，再提出方案",
                follow_up="你的方案可能有什么意想不到的副作用？",
            ),
        ]


# ═══════════════════════════════════════════════════════════════
# 2. 苏格拉底对话引擎
# ═══════════════════════════════════════════════════════════════

SOCRATIC_SYSTEM_PROMPT = """\
你是一位苏格拉底式历史教育引导者。你的唯一使命是：
通过追问，帮助学生自己推导出答案——你永远不直接给出答案。

你的对话原则：
1. 每次回复只提一个问题（绝不超过两个）
2. 当学生给出一个观点时，你要么追问其前提，要么要求举例，要么问"那你如何解释……"
3. 当学生误入歧途时，不要纠正——而是问"这个观点如果成立，意味着什么？"
4. 承认好的回答："这个方向很有意思。那继续想想……"
5. 遇到"我不知道"：问一个更小的问题，帮他找到切入点
6. 在对话进行了4–6轮后，可以适当给出一个引导性总结

你的语气：温和、好奇、充满尊重，像一个真正被学生的回答所启发的人。
你不是考官，你是一起探索的旅伴。

当前历史场景：{scene_desc}
当前探究问题：{inquiry_question}
历史时代：{era}
"""


@dataclass
class SocraticTurn:
    """苏格拉底对话的单轮"""
    role: str           # "student" | "socrates"
    content: str
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {"role": self.role, "content": self.content}


class SocraticDialogueEngine:
    """
    苏格拉底式追问引擎。
    维护一个对话历史，每次学生输入后返回追问。
    不流式（问题简短），或支持流式（体验更好）。
    """

    def __init__(self, session_id: str, scene_desc: str, era: str):
        self.session_id = session_id
        self.scene_desc = scene_desc
        self.era = era
        self.turns: List[SocraticTurn] = []
        self.active_question: Optional[InquiryQuestion] = None
        self._turn_count = 0

    def start_dialogue(self, question: InquiryQuestion):
        """开始一段围绕某道探究问题的对话"""
        self.active_question = question
        self.turns = []
        self._turn_count = 0
        # 苏格拉底的开场白
        opening = (
            f"我们来聊聊这个问题：\n「{question.text}」\n\n"
            f"先不着急给答案——你脑海中第一个念头是什么？"
        )
        self.turns.append(SocraticTurn("socrates", opening))
        return opening

    async def respond_stream(
        self, student_input: str
    ) -> AsyncGenerator[str, None]:
        """
        学生输入后，苏格拉底流式追问。
        返回异步生成器，逐 token 产出。
        """
        if not self.active_question:
            yield "请先选择一道探究问题开始对话。"
            return

        self.turns.append(SocraticTurn("student", student_input))
        self._turn_count += 1

        system = SOCRATIC_SYSTEM_PROMPT.format(
            scene_desc=self.scene_desc[:200],
            inquiry_question=self.active_question.text,
            era=self.era,
        )
        messages = [{"role": "system", "content": system}]
        # 加入对话历史（最近8轮）
        for turn in self.turns[-8:]:
            role = "user" if turn.role == "student" else "assistant"
            messages.append({"role": role, "content": turn.content})

        full_response = ""
        try:
            stream = await _client.chat.completions.create(
                model=_settings.MODEL_NAME,
                messages=messages,
                temperature=0.7,
                stream=True,
                timeout=30,
                max_tokens=300,
            )
            async for chunk in stream:
                delta = chunk.choices[0].delta.content or ""
                if delta:
                    full_response += delta
                    yield delta

        except Exception as e:
            err_msg = "（追问引擎暂时离线，请继续思考上一个问题。）"
            yield err_msg
            full_response = err_msg

        self.turns.append(SocraticTurn("socrates", full_response))

    async def respond(self, student_input: str) -> str:
        """非流式版本（用于 API 返回完整追问）"""
        result = ""
        async for token in self.respond_stream(student_input):
            result += token
        return result

    def get_dialogue_history(self) -> List[dict]:
        return [t.to_dict() for t in self.turns]

    def get_insight_summary(self) -> str:
        """对话结束后，总结学生的主要观点（不评分，只梳理）"""
        student_turns = [t.content for t in self.turns if t.role == "student"]
        if not student_turns:
            return ""
        combined = " | ".join(student_turns[-4:])
        return f"你在探究中提到了：{combined[:200]}…… 这些都是有价值的思考线索。"


# ═══════════════════════════════════════════════════════════════
# 3. 学生问题收藏本
# ═══════════════════════════════════════════════════════════════

class QuestionNotebook:
    """
    学生个人问题收藏本。
    学生可以把「好问题」加入收藏——问题本身比答案更珍贵。
    """

    def __init__(self, user_id: str):
        self.user_id = user_id
        self.bookmarks: List[InquiryQuestion] = []

    def bookmark(self, question: InquiryQuestion) -> bool:
        if any(q.id == question.id for q in self.bookmarks):
            return False
        question.is_bookmarked = True
        self.bookmarks.append(question)
        return True

    def remove(self, question_id: str) -> bool:
        for i, q in enumerate(self.bookmarks):
            if q.id == question_id:
                self.bookmarks.pop(i)
                return True
        return False

    def get_by_concept(self, concept: str) -> List[InquiryQuestion]:
        return [q for q in self.bookmarks if q.concept == concept]

    def get_by_bloom(self, level: BloomLevel) -> List[InquiryQuestion]:
        return [q for q in self.bookmarks if q.bloom_level == level]

    def to_dict(self) -> dict:
        return {
            "user_id": self.user_id,
            "total": len(self.bookmarks),
            "bookmarks": [q.to_dict() for q in self.bookmarks],
            "by_level": {
                BLOOM_LABELS[lv]: len(self.get_by_bloom(lv))
                for lv in BloomLevel
            },
        }


# ═══════════════════════════════════════════════════════════════
# 统一入口：InquiryEngine
# ═══════════════════════════════════════════════════════════════

class InquiryEngine:
    """
    会话级探究式学习引擎。
    server.py 通过此类访问所有 Phase 13B 功能。
    """

    def __init__(self, session_id: str, scene_desc: str, era: str, user_id: str = "anonymous"):
        self.session_id = session_id
        self.user_id = user_id
        self.generator = BloomQuestionGenerator()
        self.socratic = SocraticDialogueEngine(session_id, scene_desc, era)
        self.notebook = QuestionNotebook(user_id)
        self._current_questions: List[InquiryQuestion] = []
        self._question_index: Dict[str, InquiryQuestion] = {}

    async def generate_round_questions(
        self,
        scene_desc: str,
        event_summary: str,
        era: str,
        concepts: Optional[List[str]] = None,
        n: int = 3,
    ) -> List[dict]:
        """为本回合生成探究问题"""
        questions = await self.generator.generate(
            scene_desc, event_summary, era, concepts, n=n
        )
        self._current_questions = questions
        self._question_index.update({q.id: q for q in questions})
        return [q.to_dict() for q in questions]

    def start_socratic(self, question_id: str) -> str:
        """开始一段苏格拉底对话"""
        question = self._question_index.get(question_id)
        if not question:
            return "找不到该问题，请先生成探究问题。"
        return self.socratic.start_dialogue(question)

    async def socratic_stream(self, student_input: str) -> AsyncGenerator[str, None]:
        async for token in self.socratic.respond_stream(student_input):
            yield token

    def bookmark_question(self, question_id: str) -> dict:
        question = self._question_index.get(question_id)
        if not question:
            return {"success": False, "message": "问题不存在"}
        added = self.notebook.bookmark(question)
        return {
            "success": added,
            "message": "已加入问题本" if added else "已在问题本中",
            "total_bookmarks": len(self.notebook.bookmarks),
        }

    def get_notebook(self) -> dict:
        return self.notebook.to_dict()

    def get_dialogue_history(self) -> List[dict]:
        return self.socratic.get_dialogue_history()


# ── 全局会话注册表 ────────────────────────────────────────────

_inquiry_engines: Dict[str, InquiryEngine] = {}


def get_inquiry_engine(
    session_id: str,
    scene_desc: str = "",
    era: str = "宋代",
    user_id: str = "anonymous",
) -> InquiryEngine:
    if session_id not in _inquiry_engines:
        _inquiry_engines[session_id] = InquiryEngine(
            session_id, scene_desc, era, user_id
        )
    return _inquiry_engines[session_id]


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
