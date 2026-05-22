# world_engine.py
"""
息壤 · 有限状态机世界引擎

替换原有 environment.py 的简单字典模式。

核心升级：
  1. 有限状态机（FSM）：世界有明确的情绪状态，转换有规则
  2. LLM 驱动的电影级环境描述（不再是硬编码字符串列表）
  3. 事件驱动的状态转移（接入 EventBus）
  4. 环境变量的"因果链"追踪（谁在什么时候改变了什么）
  5. 诗意化的时间描述（子时/丑时/寅时…而不是简单数字）
"""
import asyncio
import json
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

from openai import AsyncOpenAI

from config import get_settings
from prompt_templates import WORLD_CINEMATIC_DESC, WORLD_STATE_TRANSITION

_settings = get_settings()
_client = AsyncOpenAI(api_key=_settings.API_KEY, base_url=_settings.BASE_URL)


# ═══════════════════════════════════════════════════════════════
# 世界情绪状态枚举
# ═══════════════════════════════════════════════════════════════

class WorldMood(str, Enum):
    SERENE    = "SERENE"     # 平静：云淡风轻
    MELANCHOLY= "MELANCHOLY" # 忧郁：秋雨潇潇
    TENSE     = "TENSE"      # 紧张：山雨欲来
    JOYFUL    = "JOYFUL"     # 欢愉：春和景明
    SOLEMN    = "SOLEMN"     # 庄严：松柏凌霜
    CHAOTIC   = "CHAOTIC"    # 动荡：风云变色


# 状态的中文映射和默认意象
_MOOD_META = {
    WorldMood.SERENE:    {"cn": "平静", "motifs": ["清风徐来，水波不兴", "白云悠然，天高地阔"]},
    WorldMood.MELANCHOLY:{"cn": "忧郁", "motifs": ["残烛明灭，暗影浮动", "秋雨敲窗，寒意彻骨"]},
    WorldMood.TENSE:     {"cn": "紧张", "motifs": ["乌云压顶，风雨欲来", "四周死寂，落针可闻"]},
    WorldMood.JOYFUL:    {"cn": "欢愉", "motifs": ["春和景明，鸟语花香", "晨光微露，暖意融融"]},
    WorldMood.SOLEMN:    {"cn": "庄严", "motifs": ["古松挺立，暮鼓晨钟", "山河无语，星辰肃穆"]},
    WorldMood.CHAOTIC:   {"cn": "动荡", "motifs": ["惊鸟四散，尘土飞扬", "烛火摇曳，人心惶惶"]},
}

# 情绪词到 WorldMood 的映射（从 Agent emotion_keyword 自动转换）
_EMOTION_TO_MOOD: Dict[str, WorldMood] = {
    "悲凉": WorldMood.MELANCHOLY, "忧伤": WorldMood.MELANCHOLY, "孤独": WorldMood.MELANCHOLY,
    "紧张": WorldMood.TENSE,      "愤怒": WorldMood.TENSE,      "恐惧": WorldMood.TENSE,
    "喜悦": WorldMood.JOYFUL,     "感恩": WorldMood.JOYFUL,     "欣慰": WorldMood.JOYFUL,
    "豁达": WorldMood.SERENE,     "平静": WorldMood.SERENE,
    "庄严": WorldMood.SOLEMN,     "肃穆": WorldMood.SOLEMN,
    "动荡": WorldMood.CHAOTIC,    "混乱": WorldMood.CHAOTIC,
}

# 古代时辰描述
_SHICHEN = [
    "子时（深夜）", "丑时（凌晨）", "寅时（黎明前）", "卯时（日出）",
    "辰时（清晨）", "巳时（上午）", "午时（正午）", "未时（午后）",
    "申时（傍晚前）", "酉时（日落）", "戌时（黄昏）", "亥时（入夜）",
]


# ═══════════════════════════════════════════════════════════════
# 环境变化记录
# ═══════════════════════════════════════════════════════════════

@dataclass
class EnvChange:
    agent: str
    key: str
    old_value: str
    new_value: str
    timestamp: float = field(default_factory=time.time)
    round_number: int = 0


# ═══════════════════════════════════════════════════════════════
# 世界引擎主体
# ═══════════════════════════════════════════════════════════════

class WorldEngine:
    """
    有限状态机驱动的世界引擎。
    取代原来 WorldEnvironment 的简单字典实现。
    """

    def __init__(self, initial_vars: Dict[str, str]):
        self.state: Dict[str, str] = dict(initial_vars) if initial_vars else {}
        self.mood: WorldMood = WorldMood.SERENE
        self.time_passed: int = 0
        self._change_log: List[EnvChange] = []
        self._cinematic_cache: Optional[str] = None  # 缓存，避免每帧都调 LLM
        self._cache_round: int = -1

        # 初始时辰（从环境变量中提取，否则随机）
        init_time = initial_vars.get("时间", "")
        self._shichen_index = next(
            (i for i, s in enumerate(_SHICHEN) if s in init_time),
            random.randint(6, 11),
        )

    # ── FSM 状态转移 ──────────────────────────────────────────

    async def try_transition(self, trigger_event: str, emotion_keyword: str) -> bool:
        """
        调用 LLM 判断是否触发世界状态转移。
        返回 True 表示发生了转变。
        """
        # 优先用确定性规则（快速路径，不调 LLM）
        deterministic = self._deterministic_transition(emotion_keyword)
        if deterministic and deterministic != self.mood:
            old = self.mood
            self.mood = deterministic
            self._cinematic_cache = None
            print(f"🌀 [世界引擎] 确定性转移: {old.value} → {self.mood.value}")
            return True

        # 不确定时调 LLM 判断
        try:
            prompt = WORLD_STATE_TRANSITION.substitute(
                current_state=f"{self.mood.value}（{_MOOD_META[self.mood]['cn']}）",
                trigger_event=trigger_event[:200],
                emotion_keyword=emotion_keyword,
            )
            resp = await _client.chat.completions.create(
                model=_settings.MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                timeout=8,
            )
            raw = _strip_json(resp.choices[0].message.content)
            data = json.loads(raw)

            if data.get("should_transition"):
                new_state_str = data.get("new_state", "").upper()
                try:
                    new_mood = WorldMood(new_state_str)
                    if new_mood != self.mood:
                        old = self.mood
                        self.mood = new_mood
                        self._cinematic_cache = None
                        print(f"🌀 [世界引擎] LLM 判断转移: {old.value} → {self.mood.value} | {data.get('reason', '')}")
                        return True
                except ValueError:
                    pass
        except Exception as e:
            print(f"⚠️ [世界引擎] 状态转移判断失败: {e}")

        return False

    def _deterministic_transition(self, emotion_keyword: str) -> Optional[WorldMood]:
        """快速确定性规则（不需要 LLM）"""
        for keyword, mood in _EMOTION_TO_MOOD.items():
            if keyword in emotion_keyword:
                return mood
        return None

    # ── 环境变量操作 ──────────────────────────────────────────

    def apply_impact(self, agent_name: str, impacts: Dict[str, str]) -> None:
        if not impacts or impacts == "无":
            return
        for key, new_value in impacts.items():
            old_value = self.state.get(key, "未知")
            self.state[key] = new_value
            self._change_log.append(EnvChange(
                agent=agent_name, key=key,
                old_value=old_value, new_value=new_value,
                round_number=self.time_passed,
            ))
            print(f"  🌍 环境变化 [{agent_name}]: {key} → {new_value}")
        self._cinematic_cache = None  # 环境变了，清缓存

    def advance_time(self) -> None:
        self.time_passed += 1
        # 时辰缓慢推进（每 3 回合推进一个时辰）
        if self.time_passed % 3 == 0:
            self._shichen_index = (self._shichen_index + 1) % len(_SHICHEN)
        self._cinematic_cache = None

    # ── 环境描述生成 ──────────────────────────────────────────

    async def get_cinematic_description(self) -> str:
        """
        LLM 驱动的电影级环境描述（每回合最多调用一次，有缓存）。
        如果 LLM 调用失败，回退到确定性意象。
        """
        if self._cinematic_cache and self._cache_round == self.time_passed:
            return self._cinematic_cache

        try:
            state_vars_str = "、".join(f"{k}:{v}" for k, v in list(self.state.items())[:5])
            prompt = WORLD_CINEMATIC_DESC.substitute(
                state_vars=state_vars_str or "无特殊变化",
                emotional_tone=f"{self.mood.value}（{_MOOD_META[self.mood]['cn']}）",
                time_passed=self.time_passed,
            )
            resp = await _client.chat.completions.create(
                model=_settings.MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.8,
                max_tokens=60,
                timeout=6,
            )
            desc = resp.choices[0].message.content.strip()
            self._cinematic_cache = desc
            self._cache_round = self.time_passed
            return desc
        except Exception:
            # 降级到确定性意象
            motifs = _MOOD_META[self.mood]["motifs"]
            return random.choice(motifs)

    def get_current_state_text(self) -> str:
        """
        同步版环境描述（给 Agent System Prompt 用）。
        cinematic 描述需要异步获取，这里用缓存或静态意象。
        """
        state_strs = [f"{k}: {v}" for k, v in self.state.items()] or ["无特殊物理变化"]
        current_motif = self._cinematic_cache or random.choice(_MOOD_META[self.mood]["motifs"])

        return (
            f"当前时辰：{_SHICHEN[self._shichen_index]}（第 {self.time_passed} 回合）\n"
            f"世界情绪基调：{self.mood.value}（{_MOOD_META[self.mood]['cn']}）\n"
            f"【世界物理参数】: {', '.join(state_strs)}\n"
            f"【电影意象】: {current_motif}（请在对话中巧妙化用此景借景抒情）"
        )

    # ── 历史变化追溯 ──────────────────────────────────────────

    def get_change_log_text(self, last_n: int = 5) -> str:
        recent = self._change_log[-last_n:]
        if not recent:
            return "无记录"
        return "\n".join(
            f"  第{c.round_number}回合 [{c.agent}] {c.key}: {c.old_value}→{c.new_value}"
            for c in recent
        )

    def to_dict(self) -> dict:
        return {
            "state": self.state,
            "mood": self.mood.value,
            "time_passed": self.time_passed,
            "shichen_index": self._shichen_index,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "WorldEngine":
        engine = cls(data.get("state", {}))
        engine.time_passed = data.get("time_passed", 0)
        engine.mood = WorldMood(data.get("mood", "SERENE"))
        engine._shichen_index = data.get("shichen_index", 6)
        return engine


def _strip_json(text: str) -> str:
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()
