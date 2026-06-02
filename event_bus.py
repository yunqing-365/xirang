# event_bus.py
"""
息壤 · 异步事件总线

架构目标：彻底解耦各组件间的直接调用依赖。
当前紧耦合链路（升级前）：
  server.py → director → agent → memory → rag_engine（线性同步）

升级后的事件驱动链路：
  任意组件 emit(event) → EventBus 分发 → 所有订阅者并发响应

设计原则：
  - Fire-and-forget：发布者不等待订阅者完成
  - 异步优先：所有 handler 必须是 async def
  - 事件日志：自动记录事件流，用于调试和回放
  - 优先级队列：关键事件（如错误）优先处理
"""
import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Coroutine, Dict, List, Optional
from collections import defaultdict


# ═══════════════════════════════════════════════════════════════
# 事件类型定义
# ═══════════════════════════════════════════════════════════════

class EventType(Enum):
    # ── 世界/场景事件 ────────────────────────────────────────
    WORLD_CREATED        = auto()   # 新世界创建完成
    WORLD_STATE_CHANGED  = auto()   # 世界物理/情绪状态变化
    TIME_ADVANCED        = auto()   # 时间流逝一个回合

    # ── Agent 行为事件 ────────────────────────────────────────
    AGENT_THINKING       = auto()   # Agent 开始思考
    AGENT_SPOKE          = auto()   # Agent 完成一轮发言（携带完整 parsed_data）
    AGENT_STREAM_TOKEN   = auto()   # Agent 流式 token（逐字）
    AGENT_MEMORY_UPDATED = auto()   # Agent 记忆落盘完成

    # ── 叙事事件 ──────────────────────────────────────────────
    NARRATOR_EVENT       = auto()   # 旁白/突发事件
    HISTORICAL_ECHO      = auto()   # 时空回响触发
    CHOICES_READY        = auto()   # 本回合叙事选项就绪
    PLAYER_CHOSE         = auto()   # 玩家提交了选择

    # ── 反思/学习事件 ────────────────────────────────────────
    REFLECTION_TRIGGERED = auto()   # 人文反思触发
    REFLECTION_DONE      = auto()   # 反思内容生成完毕
    USER_PROFILE_UPDATED = auto()   # 用户成长档案更新

    # ── Phase 12A · 情绪事件 ─────────────────────────────────
    EMOTION_UPDATED      = auto()   # NPC 情绪状态更新
    MONOLOGUE_UNLOCKED   = auto()   # 玩家解锁 NPC 内心独白
    EMOTION_ARC_SNAPSHOT = auto()   # 情感弧线快照（用于前端弧线图）

    # ── Phase 12B · 大概念事件 ───────────────────────────────
    CONCEPT_TOUCHED      = auto()   # 检测到大概念被触碰
    CONCEPT_SUMMARY_READY = auto()  # 会话概念总结卡就绪

    # ── Phase 13B · 探究问题事件 ─────────────────────────────
    INQUIRY_QUESTIONS_READY = auto()  # 本回合探究问题生成完毕

    # ── 系统事件 ──────────────────────────────────────────────
    SESSION_CREATED      = auto()
    SESSION_EXPIRED      = auto()
    ERROR_OCCURRED       = auto()
    RAG_RETRIEVED        = auto()   # RAG 检索完成（带命中质量分）
    PERSONA_VIOLATION    = auto()   # 人格一致性检测到违规


# ═══════════════════════════════════════════════════════════════
# 事件数据类
# ═══════════════════════════════════════════════════════════════

@dataclass
class Event:
    type: EventType
    session_id: str
    payload: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    priority: int = 5   # 1=最高优先级，10=最低

    def __lt__(self, other: "Event") -> bool:
        # asyncio.PriorityQueue 需要可比较
        return self.priority < other.priority


# ═══════════════════════════════════════════════════════════════
# 事件总线核心
# ═══════════════════════════════════════════════════════════════

AsyncHandler = Callable[[Event], Coroutine[Any, Any, None]]


class EventBus:
    """
    单例事件总线。
    用法：
        # 订阅
        bus.subscribe(EventType.AGENT_SPOKE, my_async_handler)

        # 发布（fire-and-forget）
        await bus.emit(Event(EventType.AGENT_SPOKE, session_id="xxx", payload={...}))

        # 发布并等待所有订阅者完成（用于需要顺序保证的场景）
        await bus.emit_and_wait(event)
    """

    def __init__(self, log_capacity: int = 500):
        self._handlers: Dict[EventType, List[AsyncHandler]] = defaultdict(list)
        self._global_handlers: List[AsyncHandler] = []   # 订阅所有事件
        self._event_log: List[Event] = []
        self._log_capacity = log_capacity
        self._lock = asyncio.Lock()

    # ── 订阅 ──────────────────────────────────────────────────

    def subscribe(self, event_type: EventType, handler: AsyncHandler) -> None:
        """订阅特定类型事件"""
        self._handlers[event_type].append(handler)

    def subscribe_all(self, handler: AsyncHandler) -> None:
        """订阅所有事件（用于日志/监控）"""
        self._global_handlers.append(handler)

    def unsubscribe(self, event_type: EventType, handler: AsyncHandler) -> None:
        handlers = self._handlers.get(event_type, [])
        if handler in handlers:
            handlers.remove(handler)

    # ── 发布 ──────────────────────────────────────────────────

    async def emit(self, event: Event) -> None:
        """Fire-and-forget：发布事件，不等待 handler 完成"""
        self._log(event)
        handlers = self._handlers.get(event.type, []) + self._global_handlers
        for handler in handlers:
            asyncio.create_task(self._safe_call(handler, event))

    async def emit_and_wait(self, event: Event) -> None:
        """发布并等待所有 handler 完成（顺序保证）"""
        self._log(event)
        handlers = self._handlers.get(event.type, []) + self._global_handlers
        if handlers:
            await asyncio.gather(*(self._safe_call(h, event) for h in handlers))

    # ── 查询事件日志 ──────────────────────────────────────────

    def get_session_events(
        self,
        session_id: str,
        event_type: Optional[EventType] = None,
        limit: int = 50,
    ) -> List[Event]:
        events = [e for e in self._event_log if e.session_id == session_id]
        if event_type:
            events = [e for e in events if e.type == event_type]
        return events[-limit:]

    def get_agent_dialogue_history(self, session_id: str) -> List[Dict]:
        """从事件日志中重建对话历史（用于断线重连后的 UI 恢复）"""
        events = self.get_session_events(session_id, EventType.AGENT_SPOKE, limit=100)
        return [
            {
                "name": e.payload.get("name"),
                "action": e.payload.get("action"),
                "dialogue": e.payload.get("dialogue"),
                "timestamp": e.timestamp,
            }
            for e in events
        ]

    # ── 内部工具 ──────────────────────────────────────────────

    def _log(self, event: Event) -> None:
        self._event_log.append(event)
        # 环形缓冲：超容后丢弃最老的事件
        if len(self._event_log) > self._log_capacity:
            self._event_log = self._event_log[-self._log_capacity:]

    @staticmethod
    async def _safe_call(handler: AsyncHandler, event: Event) -> None:
        try:
            await handler(event)
        except Exception as e:
            print(f"⚠️ [EventBus] handler {handler.__name__} 处理 {event.type.name} 出错: {e}")


# ── 单例 ──────────────────────────────────────────────────────
bus = EventBus()


# ═══════════════════════════════════════════════════════════════
# 内置监控 handler（订阅所有事件，用于控制台输出）
# ═══════════════════════════════════════════════════════════════

_EMOJI_MAP = {
    EventType.WORLD_CREATED:          "🌌",
    EventType.WORLD_STATE_CHANGED:    "🌍",
    EventType.AGENT_SPOKE:            "🗣️",
    EventType.AGENT_THINKING:         "🧠",
    EventType.NARRATOR_EVENT:         "📜",
    EventType.HISTORICAL_ECHO:        "⏳",
    EventType.CHOICES_READY:          "🎮",
    EventType.PLAYER_CHOSE:           "⚡",
    EventType.REFLECTION_DONE:        "✨",
    EventType.ERROR_OCCURRED:         "❌",
    EventType.PERSONA_VIOLATION:      "🚨",
    # Phase 12A
    EventType.EMOTION_UPDATED:        "💫",
    EventType.MONOLOGUE_UNLOCKED:     "📖",
    EventType.EMOTION_ARC_SNAPSHOT:   "📈",
    # Phase 12B
    EventType.CONCEPT_TOUCHED:        "🔍",
    EventType.CONCEPT_SUMMARY_READY:  "🎓",
    # Phase 13B
    EventType.INQUIRY_QUESTIONS_READY: "❓",
    EventType.RAG_RETRIEVED:        "🔍",
    EventType.SESSION_EXPIRED:      "💤",
}

async def _console_monitor(event: Event) -> None:
    emoji = _EMOJI_MAP.get(event.type, "•")
    session_short = event.session_id[:12] if event.session_id else "—"
    name = event.payload.get("name", "")
    label = f" [{name}]" if name else ""
    print(f"{emoji} [{session_short}]{label} {event.type.name}")

bus.subscribe_all(_console_monitor)
