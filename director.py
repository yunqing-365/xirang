# director.py  ── 架构升级版
"""
核心变更：
  1. direct_next_scene 改为 async，使用 AsyncOpenAI。
  2. Prompt 从此剥离，走 prompt_templates.DIRECTOR_SYSTEM。
"""
import json
import random

from openai import AsyncOpenAI

from config import get_settings
from prompt_templates import DIRECTOR_SYSTEM

_settings = get_settings()
_client = AsyncOpenAI(api_key=_settings.API_KEY, base_url=_settings.BASE_URL)


class SpatiotemporalDirector:
    def __init__(self, agents):
        self.agent_names = [a.name for a in agents]

    async def direct_next_scene(
        self,
        scene_desc: str,
        current_dialogue: str,
        env_text: str,
    ) -> dict:
        """完全异步，不再阻塞事件循环。"""
        prompt = DIRECTOR_SYSTEM.substitute(
            scene_desc=scene_desc,
            env_text=env_text,
            current_dialogue=current_dialogue[-800:],
            agent_names=str(self.agent_names),
        )

        try:
            resp = await _client.chat.completions.create(
                model=_settings.MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.6,
                timeout=15,
            )
            raw = resp.choices[0].message.content.strip()
            raw = _strip_code_fence(raw)
            match = __import__("re").search(r'\{.*\}', raw, __import__("re").DOTALL)
            if match:
                return json.loads(match.group(0))
        except Exception as e:
            print(f"🎬 [导演引擎出错]: {e}")

        return {
            "next_speaker": random.choice(self.agent_names),
            "narrator_event": "无",
            "historical_echo": "无",
        }


def _strip_code_fence(text: str) -> str:
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()
