# reflection_engine.py
"""
息壤 · 人文反思引擎

在恰当的时机（每 N 回合 / 会话结束 / 高潮事件后），
生成一段将历史体验与玩家当下生命联结的「人文回响」。

这是整个产品最具启发性的核心功能：
  "不是让人记住历史，而是让历史照亮当下。"
"""
import asyncio
import json
from typing import Optional

from openai import AsyncOpenAI

from config import get_settings
from prompt_templates import REFLECTION_ENGINE
from user_profile import ReflectionRecord

_settings = get_settings()
_client = AsyncOpenAI(api_key=_settings.API_KEY, base_url=_settings.BASE_URL)


class ReflectionResult:
    def __init__(self, insight: str, reflection_question: str, era_fact: str):
        self.insight = insight
        self.reflection_question = reflection_question
        self.era_fact = era_fact          # 历史彩蛋（可选）

    def to_dict(self) -> dict:
        return {
            "insight": self.insight,
            "reflection_question": self.reflection_question,
            "era_fact": self.era_fact,
        }

    def to_profile_record(self, session_id: str) -> ReflectionRecord:
        return ReflectionRecord(
            session_id=session_id,
            insight=self.insight,
            reflection_question=self.reflection_question,
            era_fact=self.era_fact,
        )


class ReflectionEngine:
    """
    负责在正确时机触发人文反思，并将结果持久化到用户档案。
    """

    async def generate(
        self,
        scene_desc: str,
        dialogue_summary: str,
        player_choices: list,
        player_context: str = "",
    ) -> Optional[ReflectionResult]:
        """
        生成本次体验的「人文回响」。
        返回 None 表示当前对话还不足以触发有质量的反思。
        """
        # 对话太短则不触发
        if len(dialogue_summary.strip()) < 100:
            return None

        prompt = REFLECTION_ENGINE.substitute(
            scene_desc=scene_desc,
            dialogue_summary=dialogue_summary[-800:],
            player_choices="; ".join(player_choices[-5:]) or "无",
            player_context=player_context or "无",
        )

        try:
            resp = await _client.chat.completions.create(
                model=_settings.MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.6,
                timeout=20,
            )
            raw = resp.choices[0].message.content.strip()
            raw = _strip_code_fence(raw)
            data = json.loads(raw)

            return ReflectionResult(
                insight=data.get("insight", ""),
                reflection_question=data.get("reflection_question", ""),
                era_fact=data.get("related_era_fact", ""),
            )

        except Exception as e:
            print(f"⚠️ [反思引擎] 生成失败: {e}")
            return None

    def should_trigger(self, rounds: int, force: bool = False) -> bool:
        """判断当前是否应触发反思"""
        if force:
            return True
        return rounds > 0 and rounds % _settings.REFLECTION_TRIGGER_ROUNDS == 0


def _strip_code_fence(text: str) -> str:
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()
