# emotion_engine.py
"""
息壤 · Phase 12A · NPC 情感弧线引擎

核心理念：
  历史人物不是"信息容器"，而是有七情六欲的活人。
  苏轼失意时说话的方式，与得意时截然不同。
  玩家的行为应该真实地影响 NPC 的情绪——情绪再影响历史走向。

七情状态机（基于儒家「七情」概念）:
  喜 (JOY)       · 悦 (PLEASURE)   · 怒 (ANGER)
  哀 (SORROW)    · 惧 (FEAR)       · 爱 (AFFECTION)
  恶 (AVERSION)  + 矛盾 (CONFLICT) 扩展状态

每个 NPC 实例持有一个 EmotionState：
  - 主情绪 + 强度（0–100）
  - 情绪历史（用于绘制弧线图）
  - 内心独白解锁系统（玩家累积亲密度可解锁）
  - 情绪→语言风格映射（对话生成时注入 prompt）
"""
import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

# ═══════════════════════════════════════════════════════════════
# 七情枚举
# ═══════════════════════════════════════════════════════════════

class Emotion(str, Enum):
    JOY       = "喜"      # 欢欣、兴奋、得意
    PLEASURE  = "悦"      # 内心平静的满足
    ANGER     = "怒"      # 愤慨、不平
    SORROW    = "哀"      # 悲伤、忧郁、失落
    FEAR      = "惧"      # 担忧、惶恐
    AFFECTION = "爱"      # 欣赏、情谊、眷恋
    AVERSION  = "恶"      # 厌倦、抗拒、鄙视
    CONFLICT  = "矛盾"    # 两种情绪交织（扩展）


# 每种情绪对应的语言风格提示（注入 AGENT_SYSTEM）
EMOTION_STYLE_HINTS: Dict[Emotion, str] = {
    Emotion.JOY:       "你此刻心情愉悦，话语中带着笑意与活力，可能会多用比喻和典故，语调明快。",
    Emotion.PLEASURE:  "你内心平静，话语沉稳，带有一种看透世事的从容，偶尔引经据典。",
    Emotion.ANGER:     "你内心愤懑，虽竭力克制，但语气中难免透出锋芒，措辞更加直接有力。",
    Emotion.SORROW:    "你心中有隐痛，话语比平时更沉，偶有停顿，可能会触景生情引发感慨。",
    Emotion.FEAR:      "你心存顾虑，说话更加谨慎迂回，会主动观察他人反应，有时话到嘴边又咽回去。",
    Emotion.AFFECTION: "你对眼前的人或事怀有好感，话语温和，更愿意倾听，眼神里有真诚的关注。",
    Emotion.AVERSION:  "你对当前处境感到厌倦或抗拒，话语中带着疏离感，不愿多说，言简意赅。",
    Emotion.CONFLICT:  "你内心有两种情绪相互拉扯，话语时而激动时而低沉，前后可能略有矛盾，这是真实的人性。",
}

# 情绪触发关键词（用于从玩家行为自动推断情绪影响）
EMOTION_TRIGGER_KEYWORDS: Dict[Emotion, List[str]] = {
    Emotion.JOY:       ["赞美", "称赞", "欣喜", "好消息", "升官", "得意", "庆贺", "喜事"],
    Emotion.PLEASURE:  ["理解", "共鸣", "平静", "欣赏", "满足", "自然", "山水"],
    Emotion.ANGER:     ["指责", "冤枉", "陷害", "不公", "小人", "变法", "排挤", "弹劾"],
    Emotion.SORROW:    ["贬谪", "离别", "思念", "失去", "亡故", "遗憾", "落寞", "流放"],
    Emotion.FEAR:      ["威胁", "皇帝", "圣旨", "审判", "牢狱", "死亡", "连累"],
    Emotion.AFFECTION: ["朋友", "子弟", "妻子", "诗友", "故乡", "旧日", "师长"],
    Emotion.AVERSION:  ["虚伪", "小人", "官场", "逢迎", "无聊", "无趣", "敷衍"],
    Emotion.CONFLICT:  ["两难", "矛盾", "既…又…", "一方面", "另一方面", "进退两难"],
}


# ═══════════════════════════════════════════════════════════════
# 内心独白条目
# ═══════════════════════════════════════════════════════════════

@dataclass
class InnerMonologue:
    """NPC 内心独白（玩家积累亲密度后可解锁）"""
    unlock_threshold: int      # 需要多少「情感共鸣点」才能解锁
    emotion_context: Emotion   # 在什么情绪状态下触发
    content: str               # 独白文本
    is_unlocked: bool = False
    unlocked_at: Optional[float] = None

    def unlock(self):
        self.is_unlocked = True
        self.unlocked_at = time.time()

    def to_dict(self) -> dict:
        return {
            "unlock_threshold": self.unlock_threshold,
            "emotion_context": self.emotion_context.value,
            "content": self.content,
            "is_unlocked": self.is_unlocked,
        }


# ═══════════════════════════════════════════════════════════════
# 情绪历史记录
# ═══════════════════════════════════════════════════════════════

@dataclass
class EmotionSnapshot:
    """单个时刻的情绪快照（用于绘制弧线）"""
    round_num: int
    emotion: Emotion
    intensity: int          # 0–100
    trigger: str            # 是什么引发了这次情绪变化
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "round": self.round_num,
            "emotion": self.emotion.value,
            "intensity": self.intensity,
            "trigger": self.trigger,
        }


# ═══════════════════════════════════════════════════════════════
# NPC 情绪状态
# ═══════════════════════════════════════════════════════════════

class EmotionState:
    """
    单个 NPC 的情绪状态容器。
    由 EmotionEngine 统一管理，挂载到 SocialAgent。
    """

    def __init__(self, agent_name: str, initial_emotion: Emotion = Emotion.PLEASURE):
        self.agent_name = agent_name
        self.current_emotion: Emotion = initial_emotion
        self.intensity: int = 50                    # 情绪强度 0–100
        self.history: List[EmotionSnapshot] = []
        self.resonance_points: int = 0              # 玩家与 NPC 的情感共鸣积累
        self.monologues: List[InnerMonologue] = []  # 内心独白库（由外部注册）
        self._round = 0

    # ── 情绪更新 ──────────────────────────────────────────────

    def update(
        self,
        new_emotion: Emotion,
        intensity: int,
        trigger: str = "",
        resonance_delta: int = 0,
    ) -> bool:
        """
        更新情绪状态。
        返回 True 表示情绪发生了显著变化（强度差 > 20 或情绪种类变化）。
        """
        changed = (
            new_emotion != self.current_emotion
            or abs(intensity - self.intensity) > 20
        )
        self._round += 1
        self.history.append(EmotionSnapshot(
            round_num=self._round,
            emotion=new_emotion,
            intensity=max(0, min(100, intensity)),
            trigger=trigger,
        ))
        # 只保留最近 20 条历史
        if len(self.history) > 20:
            self.history = self.history[-20:]

        self.current_emotion = new_emotion
        self.intensity = max(0, min(100, intensity))
        self.resonance_points = max(0, self.resonance_points + resonance_delta)
        return changed

    def infer_from_text(self, text: str, delta_intensity: int = 15) -> Tuple[Emotion, int]:
        """
        从文本（玩家行为/对话）中简单推断情绪影响。
        返回 (推断情绪, 强度变化建议)。
        """
        text_lower = text.lower()
        scores: Dict[Emotion, int] = {e: 0 for e in Emotion}
        for emotion, keywords in EMOTION_TRIGGER_KEYWORDS.items():
            for kw in keywords:
                if kw in text_lower:
                    scores[emotion] += 1
        best = max(scores, key=lambda e: scores[e])
        if scores[best] == 0:
            return self.current_emotion, 0
        new_intensity = min(100, self.intensity + delta_intensity * scores[best])
        return best, new_intensity

    # ── 独白解锁 ──────────────────────────────────────────────

    def register_monologue(self, monologue: InnerMonologue):
        """注册一条内心独白（由 EmotionEngine 在初始化时批量注册）"""
        self.monologues.append(monologue)

    def check_unlock(self) -> Optional[InnerMonologue]:
        """
        检查是否有独白刚好达到解锁条件。
        返回刚解锁的那条独白（如有）。
        """
        for m in self.monologues:
            if (
                not m.is_unlocked
                and self.resonance_points >= m.unlock_threshold
                and m.emotion_context == self.current_emotion
            ):
                m.unlock()
                return m
        return None

    def get_unlocked_monologues(self) -> List[InnerMonologue]:
        return [m for m in self.monologues if m.is_unlocked]

    # ── 对话风格提示 ──────────────────────────────────────────

    def get_style_hint(self) -> str:
        """返回注入 Agent System Prompt 的情绪驱动语言风格提示"""
        base = EMOTION_STYLE_HINTS.get(self.current_emotion, "")
        intensity_modifier = ""
        if self.intensity >= 80:
            intensity_modifier = "（情绪非常强烈，难以完全掩饰）"
        elif self.intensity >= 60:
            intensity_modifier = "（情绪较为明显，但尚在控制之内）"
        elif self.intensity <= 25:
            intensity_modifier = "（情绪较为平淡，内心更多是疲惫或麻木）"
        return f"{base}{intensity_modifier}"

    # ── 序列化 ────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "agent": self.agent_name,
            "current_emotion": self.current_emotion.value,
            "intensity": self.intensity,
            "resonance_points": self.resonance_points,
            "history": [s.to_dict() for s in self.history[-10:]],
            "unlocked_monologues": [m.to_dict() for m in self.get_unlocked_monologues()],
        }

    @property
    def summary(self) -> str:
        """单行摘要，用于 SSE 事件 payload"""
        return f"{self.agent_name}·{self.current_emotion.value}·强度{self.intensity}"


# ═══════════════════════════════════════════════════════════════
# 情绪引擎（管理所有 NPC 的情绪状态）
# ═══════════════════════════════════════════════════════════════

class EmotionEngine:
    """
    会话级情绪引擎：统一管理一个场景中所有 NPC 的情绪状态。

    使用方式：
        engine = EmotionEngine()
        engine.init_npc("苏轼", Emotion.SORROW, monologues=[...])
        engine.on_player_action("苏轼", "你问苏轼：此刻心中是否后悔当初上书？")
        hint = engine.get_dialogue_hint("苏轼")  # → 注入 prompt
    """

    def __init__(self):
        self._states: Dict[str, EmotionState] = {}

    def init_npc(
        self,
        name: str,
        initial_emotion: Emotion = Emotion.PLEASURE,
        monologues: Optional[List[InnerMonologue]] = None,
    ) -> EmotionState:
        state = EmotionState(name, initial_emotion)
        if monologues:
            for m in monologues:
                state.register_monologue(m)
        self._states[name] = state
        return state

    def get_state(self, name: str) -> Optional[EmotionState]:
        return self._states.get(name)

    def ensure_state(self, name: str) -> EmotionState:
        if name not in self._states:
            self._states[name] = EmotionState(name)
        return self._states[name]

    def on_player_action(
        self,
        npc_name: str,
        action_text: str,
        resonance_delta: int = 5,
    ) -> Tuple[Emotion, int, Optional[InnerMonologue]]:
        """
        玩家行为触发情绪更新。
        返回 (新情绪, 新强度, 解锁的独白 or None)
        """
        state = self.ensure_state(npc_name)
        inferred_emotion, new_intensity = state.infer_from_text(action_text)

        if new_intensity == 0:
            # 无明显情绪触发，小幅度共鸣加分
            state.resonance_points += resonance_delta
            unlocked = state.check_unlock()
            return state.current_emotion, state.intensity, unlocked

        state.update(
            inferred_emotion,
            new_intensity,
            trigger=action_text[:40],
            resonance_delta=resonance_delta,
        )
        unlocked = state.check_unlock()
        return inferred_emotion, new_intensity, unlocked

    def on_agent_response(
        self,
        npc_name: str,
        emotion_keyword: str,
        intensity_hint: int = 50,
    ) -> EmotionState:
        """
        Agent 生成回应后，用其 emotion_keyword 更新情绪状态。
        emotion_keyword 来自 AGENT_SYSTEM JSON 输出字段。
        """
        state = self.ensure_state(npc_name)
        # 尝试把 LLM 返回的情绪词映射到枚举
        mapped = _map_keyword_to_emotion(emotion_keyword)
        state.update(mapped, intensity_hint, trigger=f"自身回应：{emotion_keyword}")
        return state

    def get_dialogue_hint(self, npc_name: str) -> str:
        """获取该 NPC 当前的情绪驱动语言风格提示（注入 Agent System Prompt）"""
        state = self.get_state(npc_name)
        if not state:
            return ""
        return state.get_style_hint()

    def get_all_states_summary(self) -> Dict[str, dict]:
        return {name: s.to_dict() for name, s in self._states.items()}

    def get_arc_data(self, npc_name: str) -> List[dict]:
        """返回 NPC 的情感弧线数据（用于前端可视化）"""
        state = self.get_state(npc_name)
        if not state:
            return []
        return [s.to_dict() for s in state.history]


# ═══════════════════════════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════════════════════════

_KEYWORD_MAPPING: Dict[str, Emotion] = {
    # 喜
    "喜悦": Emotion.JOY, "欢欣": Emotion.JOY, "得意": Emotion.JOY,
    "兴奋": Emotion.JOY, "高兴": Emotion.JOY, "欣喜": Emotion.JOY,
    # 悦
    "平静": Emotion.PLEASURE, "满足": Emotion.PLEASURE, "豁达": Emotion.PLEASURE,
    "从容": Emotion.PLEASURE, "淡然": Emotion.PLEASURE, "坦然": Emotion.PLEASURE,
    # 怒
    "愤怒": Emotion.ANGER, "愤慨": Emotion.ANGER, "激愤": Emotion.ANGER,
    "气愤": Emotion.ANGER, "不平": Emotion.ANGER, "义愤": Emotion.ANGER,
    # 哀
    "悲凉": Emotion.SORROW, "忧郁": Emotion.SORROW, "失落": Emotion.SORROW,
    "哀伤": Emotion.SORROW, "沉郁": Emotion.SORROW, "惆怅": Emotion.SORROW,
    "悲伤": Emotion.SORROW, "落寞": Emotion.SORROW,
    # 惧
    "紧张": Emotion.FEAR, "惶恐": Emotion.FEAR, "担忧": Emotion.FEAR,
    "不安": Emotion.FEAR, "焦虑": Emotion.FEAR,
    # 爱
    "欣赏": Emotion.AFFECTION, "感激": Emotion.AFFECTION, "温情": Emotion.AFFECTION,
    "怀念": Emotion.AFFECTION, "眷恋": Emotion.AFFECTION,
    # 恶
    "厌倦": Emotion.AVERSION, "抗拒": Emotion.AVERSION, "鄙视": Emotion.AVERSION,
    "厌恶": Emotion.AVERSION, "疏离": Emotion.AVERSION,
    # 矛盾
    "矛盾": Emotion.CONFLICT, "复杂": Emotion.CONFLICT, "纠结": Emotion.CONFLICT,
}

def _map_keyword_to_emotion(keyword: str) -> Emotion:
    """将 LLM 返回的情绪词尽量映射到七情枚举"""
    if keyword in _KEYWORD_MAPPING:
        return _KEYWORD_MAPPING[keyword]
    # 模糊匹配
    for k, v in _KEYWORD_MAPPING.items():
        if k in keyword or keyword in k:
            return v
    return Emotion.CONFLICT  # 无法识别时标记为"矛盾"


# ═══════════════════════════════════════════════════════════════
# 预置内心独白库（以苏轼为示例，可扩展到任意 NPC）
# ═══════════════════════════════════════════════════════════════

SUSHI_MONOLOGUES = [
    InnerMonologue(
        unlock_threshold=10,
        emotion_context=Emotion.SORROW,
        content=(
            "乌台诗案之后，我以为自己已学会沉默。"
            "可这颗心啊，它偏偏不肯死——每逢秋风，"
            "每逢月圆，它还是会想起汴京，想起子由，"
            "想起那些年指点江山的意气……"
            "我只是不说了，不代表我不想了。"
        ),
    ),
    InnerMonologue(
        unlock_threshold=20,
        emotion_context=Emotion.CONFLICT,
        content=(
            "他们说我『旷达』，说我『随缘自适』。"
            "可那些诗，那些词，哪一首不是在对抗？"
            "『竹杖芒鞋轻胜马』——那是我告诉自己的，"
            "不是我真的觉得轻松。"
            "我只是……不想让那些把我贬到这里的人，"
            "看见我倒下而已。"
        ),
    ),
    InnerMonologue(
        unlock_threshold=35,
        emotion_context=Emotion.JOY,
        content=(
            "今日酿酒成功，喝了两碗，微醺。"
            "突然想起，人生到处知何似，"
            "应似飞鸿踏雪泥。"
            "有时候我真的觉得——"
            "这一生，东奔西走，颠沛流离，"
            "也许正因如此，才见了那么多风景，"
            "才写出了那些东西。"
            "若我一直在汴京高居庙堂，"
            "我能写出赤壁赋吗？"
        ),
    ),
]

# Phase 15B: 全局实例引用槽（由 server.py 启动时注入）
_emotion_engine_global = None
