# agent.py  ── Phase 12A 升级版
"""
本轮升级（Phase 12A）：
  1. 挂载 PersonaEngine：人格指纹初始化 + 每轮一致性检查
  2. 接入 EventBus：Agent 行为通过事件广播，解耦下游消费
  3. 记忆写入改用 add_episodic_memory_async（避免 to_thread 嵌套）
  4. 情绪传递给记忆检索（情绪相似性加权）
  5. 支持人格违规时触发 PERSONA_VIOLATION 事件
  6. [NEW] 挂载 EmotionEngine：七情状态机，情绪驱动对话风格
     - Agent 响应后自动更新情绪状态
     - 情绪风格提示注入 System Prompt
     - 内心独白解锁时触发 MONOLOGUE_UNLOCKED 事件
"""
import asyncio
import json
import re

from infra.llm_client import aclient as _client, llm_chat, llm_chat_stream

from config import get_settings
from memory import SocialMemory
from persona_engine import PersonaEngine
from emotion_engine import EmotionEngine, Emotion
from event_bus import bus, Event, EventType
from prompt_templates import AGENT_SYSTEM, EMOTION_STYLE_INJECTION

_settings = get_settings()


class SocialAgent:
    def __init__(self, name, identity, personality, initial_metrics, task_role):
        self.name = name
        self.identity = identity
        self.personality = personality
        self.metrics = initial_metrics
        self.task_role = task_role
        self.era = ""  # 由 ScenarioManager 注入

        self.memory = SocialMemory(name)
        self.persona = PersonaEngine(name)
        self.rag_engine = None
        # Phase 12A: 情绪状态（由外部 EmotionEngine 统一管理，此处存引用）
        self._emotion_engine = None  # type: Optional[EmotionEngine]

    def mount_knowledge(self, rag_engine):
        self.rag_engine = rag_engine

    def set_era(self, era: str):
        self.era = era

    def mount_emotion_engine(self, emotion_engine, initial_emotion: Emotion = Emotion.PLEASURE):
        """挂载情绪引擎，初始化该 NPC 的情绪状态。Phase 12A"""
        self._emotion_engine = emotion_engine
        emotion_engine.ensure_state(self.name)
        # 用初始情绪初始化
        state = emotion_engine.get_state(self.name)
        if state:
            state.current_emotion = initial_emotion

    async def initialize(self):
        """
        首次使用前调用：生成人格指纹（已有缓存则直接加载）。
        由 ScenarioManager.load_era 在 agents 创建后 gather 调用。
        """
        await self.persona.ensure_fingerprint(
            self.identity, self.personality, self.era
        )

    # ── 核心生成流 ────────────────────────────────────────────

    async def generate_response_stream(
        self,
        scene_desc: str,
        current_task: str,
        shared_workspace: str,
        current_dialogue: str,
        env_state_text: str,
        session_id: str = "",
        user_context: str = "",
        current_emotion: str = "",    # 上一回合的世界情绪（用于记忆检索加权）
    ):
        """
        Async generator，逐 token yield 给 server.py 的 SSE 流。
        """
        relationships_str = json.dumps(
            self.memory.data["relationships"], ensure_ascii=False
        )

        # ── 并发：RAG检索 + 记忆检索 ─────────────────────────
        async def _rag():
            if not self.rag_engine:
                return "无"
            return await self.rag_engine.aretrieve(
                f"{current_task} {current_dialogue[-300:]}"
            )

        def _mem():
            return self.memory.retrieve_episodic_memory(
                current_dialogue[-300:],
                current_emotion=current_emotion,
            )

        reference_knowledge, past_memories = await asyncio.gather(
            _rag(),
            asyncio.to_thread(_mem),
        )

        # ── 注入人格约束到 System Prompt ─────────────────────
        persona_constraint = self.persona.get_fingerprint_for_prompt()
        system_prompt = AGENT_SYSTEM.substitute(
            name=self.name,
            identity=self.identity,
            personality=self.personality,
            metrics_json=json.dumps(self.metrics, ensure_ascii=False),
            relationships_json=relationships_str,
            scene_desc=scene_desc,
            env_state_text=env_state_text,
            reference_knowledge=reference_knowledge,
            past_memories=past_memories,
            current_task=current_task,
            task_role=self.task_role,
            shared_workspace=shared_workspace,
        )
        if user_context:
            system_prompt += f"\n\n=== 【与你对话的高维观察者背景】 ===\n{user_context}"
        if persona_constraint:
            system_prompt += f"\n\n=== 【你的人格约束（必须严格遵守）】 ===\n{persona_constraint}"

        # ── Phase 12A：注入情绪风格提示 ──────────────────────
        if self._emotion_engine:
            emotion_state = self._emotion_engine.get_state(self.name)
            if emotion_state:
                emotion_hint = EMOTION_STYLE_INJECTION.substitute(
                    emotion=emotion_state.current_emotion.value,
                    intensity=emotion_state.intensity,
                    style_hint=emotion_state.get_style_hint(),
                )
                system_prompt += f"\n\n{emotion_hint}"

        # ── EventBus：通知"开始思考" ─────────────────────────
        await bus.emit(Event(
            type=EventType.AGENT_THINKING,
            session_id=session_id,
            payload={"name": self.name},
        ))

        try:
            raw_full_text = ""
            async for token in llm_chat_stream(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": current_dialogue[-500:]},
                ],
                temperature=0.7,
                max_tokens=2000,
                timeout=90.0,
            ):
                raw_full_text += token
                yield {"type": "token", "content": token}

            # ── 流结束：解析 + 人格一致性检查 ─────────────────
            raw = _strip_json(raw_full_text.strip())
            match = re.search(r'\{.*\}', raw, re.DOTALL)
            if not match:
                yield {"type": "error", "content": "JSON 解析失败"}
                return

            res = json.loads(match.group(0))
            action   = res.get("action",   "静坐")
            dialogue = res.get("dialogue", "…")
            emotion  = res.get("emotion_keyword", "平静")

            # ── 人格一致性检查（并发，不阻塞主流）────────────
            consistency = await self.persona.check_consistency(action, dialogue)
            if consistency.has_violation:
                await bus.emit(Event(
                    type=EventType.PERSONA_VIOLATION,
                    session_id=session_id,
                    priority=2,
                    payload={
                        "name": self.name,
                        "score": consistency.consistency_score,
                        "violations": consistency.violations,
                        "fix": consistency.suggested_fix,
                    },
                ))
                print(f"🚨 [{self.name}] 人格违规 score={consistency.consistency_score} "
                      f"| {consistency.violations}")

            # ── 时代语言美化（高一致性时可选）────────────────
            if consistency.consistency_score < 70 and consistency.suggested_fix != "无":
                print(f"  ✏️  [{self.name}] 时代语言微调中…")
                dialogue = await self.persona.stylize_dialogue(dialogue, self.era)
                res["dialogue"] = dialogue

            # ── 社会关系 + 记忆落盘（并发）──────────────────
            for target, changes in res.get("social_impact", {}).items():
                self.memory.update_relationship(
                    target,
                    changes.get("affinity", 0),
                    changes.get("trust", 0),
                )
            self.memory.save()
            self.memory.current_round += 1

            # 记忆写入（用新的异步版，不需要 to_thread）
            asyncio.create_task(
                self.memory.add_episodic_memory_async(
                    env_state_text, action, dialogue
                )
            )

            # ── Phase 12A：更新情绪状态 ──────────────────────
            if self._emotion_engine:
                updated_state = self._emotion_engine.on_agent_response(
                    self.name,
                    emotion_keyword=emotion,
                    intensity_hint=60,
                )
                # 广播情绪更新事件
                await bus.emit(Event(
                    type=EventType.EMOTION_UPDATED,
                    session_id=session_id,
                    payload={
                        "name": self.name,
                        "emotion": updated_state.current_emotion.value,
                        "intensity": updated_state.intensity,
                        "arc": updated_state.history[-1].to_dict() if updated_state.history else {},
                    },
                ))
                # 检查内心独白解锁
                unlocked = updated_state.check_unlock()
                if unlocked:
                    await bus.emit(Event(
                        type=EventType.MONOLOGUE_UNLOCKED,
                        session_id=session_id,
                        payload={
                            "name": self.name,
                            "monologue": unlocked.content,
                            "emotion_context": unlocked.emotion_context.value,
                        },
                    ))

            # ── EventBus：广播完整行动 ────────────────────────
            await bus.emit(Event(
                type=EventType.AGENT_SPOKE,
                session_id=session_id,
                payload={
                    "name": self.name,
                    "action": action,
                    "dialogue": dialogue,
                    "emotion": emotion,
                    "contribution": res.get("contribution", "无"),
                    "show_image": res.get("show_image", "无"),
                    "env_impact": res.get("env_impact"),
                    "social_impact": res.get("social_impact", {}),
                    "consistency_score": consistency.consistency_score,
                },
            ))

            yield {"type": "done", "parsed_data": res}

        except Exception as e:
            print(f"[{self.name}] 引擎出错: {e}")
            yield {"type": "error", "content": str(e)}


def _strip_json(text: str) -> str:
    if text.startswith("```json"): text = text[7:]
    elif text.startswith("```"):   text = text[3:]
    if text.endswith("```"):       text = text[:-3]
    return text.strip()
