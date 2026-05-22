# persona_engine.py
"""
息壤 · 历史人格一致性守护引擎

核心问题：大模型很容易让苏轼说出"你知道吗，资本主义其实……"
这个引擎确保每个历史人物的发言都经过三道防线：

  防线1 - 人格指纹（Fingerprint）：
    为每个角色生成一次性的深度人格画像（核心价值观/语言风格/禁忌/认知边界）

  防线2 - 一致性审查（Consistency Check）：
    每次 Agent 生成回应后，评分并检测违规
    高违规 → 触发 PERSONA_VIOLATION 事件 → 前端可以提示或要求重新生成

  防线3 - 时代语言美化（Era Stylizer）：
    对生成的台词进行时代语言风格微调（不改变意思，只调整措辞）
    使语言更贴近历史语境
"""
import asyncio
import json
import os
from typing import Dict, Optional

from openai import AsyncOpenAI

from config import get_settings
from prompt_templates import PERSONA_FINGERPRINT, PERSONA_CONSISTENCY_CHECK

_settings = get_settings()
_client = AsyncOpenAI(api_key=_settings.API_KEY, base_url=_settings.BASE_URL)

# 指纹缓存目录
_FINGERPRINT_DIR = os.path.join(_settings.DATA_DIR, "persona_fingerprints")
os.makedirs(_FINGERPRINT_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════════
# 人格指纹数据类
# ═══════════════════════════════════════════════════════════════

class PersonaFingerprint:
    def __init__(self, name: str, raw: Dict):
        self.name = name
        self.core_values: list = raw.get("core_values", [])
        self.speech_patterns: list = raw.get("speech_patterns", [])
        self.taboo_topics: list = raw.get("taboo_topics", [])
        self.knowledge_boundaries: str = raw.get("knowledge_boundaries", "")

    def to_prompt_str(self) -> str:
        return (
            f"核心价值观：{'; '.join(self.core_values)}\n"
            f"语言风格：{'; '.join(self.speech_patterns)}\n"
            f"禁忌话题：{'; '.join(self.taboo_topics)}\n"
            f"认知边界：{self.knowledge_boundaries}"
        )

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "core_values": self.core_values,
            "speech_patterns": self.speech_patterns,
            "taboo_topics": self.taboo_topics,
            "knowledge_boundaries": self.knowledge_boundaries,
        }


# ═══════════════════════════════════════════════════════════════
# 一致性检查结果
# ═══════════════════════════════════════════════════════════════

class ConsistencyResult:
    def __init__(self, raw: Dict):
        self.is_consistent: bool = raw.get("is_consistent", True)
        self.violations: list = raw.get("violations", [])
        self.consistency_score: int = raw.get("consistency_score", 100)
        self.suggested_fix: str = raw.get("suggested_fix", "无")

    @property
    def has_violation(self) -> bool:
        return not self.is_consistent or self.consistency_score < 60

    def __repr__(self) -> str:
        return f"ConsistencyResult(score={self.consistency_score}, violations={self.violations})"


# ═══════════════════════════════════════════════════════════════
# 人格引擎主体
# ═══════════════════════════════════════════════════════════════

class PersonaEngine:
    """
    每个 SocialAgent 挂载一个 PersonaEngine 实例。
    指纹一次生成，持久化到磁盘，后续直接加载。
    """

    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        self._fingerprint: Optional[PersonaFingerprint] = self._load_fingerprint()

    # ── 人格指纹 ──────────────────────────────────────────────

    async def ensure_fingerprint(
        self,
        identity: str,
        personality: str,
        era: str,
    ) -> PersonaFingerprint:
        """确保指纹存在（不存在则生成并缓存）"""
        if self._fingerprint:
            return self._fingerprint

        print(f"🔬 [人格引擎] 为 [{self.agent_name}] 生成人格指纹…")
        prompt = PERSONA_FINGERPRINT.substitute(
            name=self.agent_name,
            identity=identity,
            personality=personality,
            era=era,
        )
        try:
            resp = await _client.chat.completions.create(
                model=_settings.MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                timeout=15,
            )
            raw = _strip_json(resp.choices[0].message.content)
            data = json.loads(raw)
            fp = PersonaFingerprint(self.agent_name, data)
            self._fingerprint = fp
            self._save_fingerprint(fp)
            print(f"✅ [{self.agent_name}] 人格指纹生成完毕: {fp.core_values}")
            return fp
        except Exception as e:
            print(f"⚠️ [人格引擎] 指纹生成失败，使用空指纹: {e}")
            empty = PersonaFingerprint(self.agent_name, {})
            self._fingerprint = empty
            return empty

    def get_fingerprint_for_prompt(self) -> str:
        """返回给 Agent System Prompt 注入的人格约束文本"""
        if not self._fingerprint:
            return "（人格指纹加载中…）"
        return self._fingerprint.to_prompt_str()

    # ── 一致性检查 ────────────────────────────────────────────

    async def check_consistency(
        self,
        action: str,
        dialogue: str,
    ) -> ConsistencyResult:
        """
        对 Agent 刚生成的回应进行一致性审查。
        轻量级：temperature=0.1，timeout=8s
        """
        if not self._fingerprint:
            return ConsistencyResult({"is_consistent": True, "consistency_score": 100})

        prompt = PERSONA_CONSISTENCY_CHECK.substitute(
            name=self.agent_name,
            fingerprint=self._fingerprint.to_prompt_str(),
            action=action,
            dialogue=dialogue,
        )
        try:
            resp = await _client.chat.completions.create(
                model=_settings.MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                timeout=8,
            )
            raw = _strip_json(resp.choices[0].message.content)
            return ConsistencyResult(json.loads(raw))
        except Exception as e:
            print(f"⚠️ [人格引擎] 一致性检查失败: {e}")
            return ConsistencyResult({"is_consistent": True, "consistency_score": 100})

    # ── 时代语言美化（轻量版，不改意思只调语气）─────────────

    async def stylize_dialogue(self, dialogue: str, era: str) -> str:
        """
        将对话微调为更贴近时代语境的表达。
        失败时原样返回，不影响主流程。
        """
        if not dialogue or len(dialogue) < 5:
            return dialogue

        prompt = (
            f"你是{era}的语言顾问。请将以下台词微调为更符合{era}语境的表达，"
            f"不要改变意思，只调整措辞和语气，不超过原文字数 120%。\n"
            f"直接输出调整后的台词，不要解释。\n台词：{dialogue}"
        )
        try:
            resp = await _client.chat.completions.create(
                model=_settings.MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=200,
                timeout=6,
            )
            return resp.choices[0].message.content.strip()
        except Exception:
            return dialogue

    # ── 磁盘持久化 ────────────────────────────────────────────

    def _load_fingerprint(self) -> Optional[PersonaFingerprint]:
        path = self._fp_path()
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return PersonaFingerprint(self.agent_name, data)
            except Exception:
                pass
        return None

    def _save_fingerprint(self, fp: PersonaFingerprint) -> None:
        with open(self._fp_path(), "w", encoding="utf-8") as f:
            json.dump(fp.to_dict(), f, ensure_ascii=False, indent=2)

    def _fp_path(self) -> str:
        safe = "".join(c for c in self.agent_name if c.isalnum() or c in "_-")
        return os.path.join(_FINGERPRINT_DIR, f"{safe}_fingerprint.json")


def _strip_json(text: str) -> str:
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()
