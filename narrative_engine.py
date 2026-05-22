# narrative_engine.py
"""
息壤 · 叙事引擎

核心职责：
  1. 每隔 N 回合，向玩家生成 3 个有意义的"干预选项"
  2. 跟踪叙事里程碑（防止重复推送相似选项）
  3. 将玩家选择转化为注入给下一回合 Agent 的"神谕指令"
  4. 维护故事弧线状态（开端 → 发展 → 高潮 → 余韵）

这套机制让玩家从"旁观者"变成"故事共创者"，
是实现"爱上人文学科"体验跃升的核心杠杆。
"""
import asyncio
import json
import re
from enum import Enum
from typing import Dict, List, Optional

from openai import AsyncOpenAI

from config import get_settings
from prompt_templates import NARRATIVE_CHOICES

_settings = get_settings()
_client = AsyncOpenAI(api_key=_settings.API_KEY, base_url=_settings.BASE_URL)


class ArcPhase(str, Enum):
    OPENING = "开端"       # 场景建立，人物登场
    RISING = "发展"        # 矛盾升温，情感铺垫
    CLIMAX = "高潮"        # 核心冲突或顿悟时刻
    DENOUEMENT = "余韵"    # 余音绕梁，人文启示


class NarrativeChoice:
    def __init__(self, id: int, text: str, choice_type: str, intent: str):
        self.id = id
        self.text = text
        self.choice_type = choice_type  # 情感共鸣 / 历史知识 / 命运转折
        self.intent = intent            # 给系统用，不展示给玩家

    def to_dict(self) -> dict:
        return {"id": self.id, "text": self.text, "type": self.choice_type}


class NarrativeState:
    """单个会话的叙事状态"""

    def __init__(self):
        self.phase: ArcPhase = ArcPhase.OPENING
        self.rounds: int = 0
        self.milestones: List[str] = []       # 已发生的重要叙事事件
        self.player_choices: List[str] = []   # 玩家做出的历史选择记录
        self.pending_choices: Optional[List[NarrativeChoice]] = None  # 当前等待玩家选择的选项
        self.choice_trigger_interval: int = _settings.NARRATIVE_CHOICES_PER_TURN

    def advance_round(self):
        self.rounds += 1
        # 弧线自动推进
        if self.rounds <= 2:
            self.phase = ArcPhase.OPENING
        elif self.rounds <= 5:
            self.phase = ArcPhase.RISING
        elif self.rounds <= 9:
            self.phase = ArcPhase.CLIMAX
        else:
            self.phase = ArcPhase.DENOUEMENT

    def should_offer_choices(self) -> bool:
        """判断本回合是否应弹出选项面板"""
        return self.rounds > 0 and self.rounds % 3 == 0

    def record_milestone(self, event: str):
        if event and event != "无" and event not in self.milestones:
            self.milestones.append(event)

    def record_player_choice(self, choice_text: str):
        self.player_choices.append(choice_text)

    def to_dict(self) -> dict:
        return {
            "phase": self.phase.value,
            "rounds": self.rounds,
            "milestones": self.milestones,
            "player_choices": self.player_choices,
        }


class NarrativeEngine:
    """叙事引擎：生成选项 + 将选择翻译为角色干预指令"""

    async def generate_choices(
        self,
        scene_desc: str,
        dialogue_summary: str,
        agent_names: List[str],
        narrative_state: NarrativeState,
        n: int = 3,
    ) -> List[NarrativeChoice]:
        """调用 LLM 生成本回合的 N 个玩家选项"""
        prompt = NARRATIVE_CHOICES.substitute(
            scene_desc=scene_desc,
            dialogue_summary=dialogue_summary[-600:],
            agent_names=", ".join(agent_names),
            narrative_milestones="; ".join(narrative_state.milestones[-5:]) or "无",
            choice_count=n,
        )

        try:
            resp = await _client.chat.completions.create(
                model=_settings.MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.8,
                timeout=15,
            )
            raw = resp.choices[0].message.content.strip()
            raw = _strip_code_fence(raw)

            items = json.loads(raw)
            choices = [
                NarrativeChoice(
                    id=item["id"],
                    text=item["text"],
                    choice_type=item.get("type", "情感共鸣"),
                    intent=item.get("intent", ""),
                )
                for item in items
            ]
            return choices

        except Exception as e:
            print(f"⚠️ [叙事引擎] 选项生成失败，使用保底选项: {e}")
            return _fallback_choices(agent_names)

    def choice_to_intervention(
        self,
        choice: NarrativeChoice,
        narrative_state: NarrativeState,
    ) -> str:
        """将玩家的点击选择翻译成注入给 Agent 系统的高维指令"""
        phase_instructions = {
            ArcPhase.OPENING: "场景刚刚建立，请顺势引出这个要求，让氛围自然流动。",
            ArcPhase.RISING: "矛盾正在升温，请将这个要求融入紧张的情感线索中。",
            ArcPhase.CLIMAX: "故事正处高潮，请让这个要求成为关键的转折点或顿悟时刻。",
            ArcPhase.DENOUEMENT: "故事接近尾声，请让这个要求升华为余韵悠长的人文感悟。",
        }

        directive = (
            f"【高维观察者的意志】: {choice.text}\n"
            f"【系统内部意图】: {choice.intent}\n"
            f"【当前叙事阶段】: {narrative_state.phase.value} — "
            f"{phase_instructions.get(narrative_state.phase, '')}"
        )
        narrative_state.record_player_choice(choice.text)
        return directive


# ── 工具函数 ──────────────────────────────────────────────────

def _strip_code_fence(text: str) -> str:
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()


def _fallback_choices(agent_names: List[str]) -> List[NarrativeChoice]:
    name = agent_names[0] if agent_names else "他"
    return [
        NarrativeChoice(1, f"问{name}：此刻你心中最放不下的是什么？", "情感共鸣",
                        "引导角色吐露内心深处的情感"),
        NarrativeChoice(2, "让旁白揭示此刻的历史背景", "历史知识",
                        "触发一段相关历史知识的旁白"),
        NarrativeChoice(3, "一阵意外打断了众人", "命运转折",
                        "制造突发事件，扭转当前对话走向"),
    ]
