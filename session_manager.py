# session_manager.py
"""
息壤 · 异步会话管理器

解决原有架构中 active_sessions / current_intervention 作为裸全局字典
在并发请求下产生的竞态条件问题。

设计原则：
  - 所有读写通过 asyncio.Lock 保护
  - 超时自动过期（TTL），防止内存无限增长
  - 提供统一的 CRUD 接口，server.py 不再直接操作字典
"""
import asyncio
import time
from typing import Any, Dict, Optional
from dataclasses import dataclass, field

from config import get_settings

_settings = get_settings()


@dataclass
class SessionEntry:
    data: Dict[str, Any]
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)

    def touch(self):
        self.last_accessed = time.time()

    def is_expired(self, ttl: int) -> bool:
        return (time.time() - self.last_accessed) > ttl


class SessionManager:
    """线程安全的异步会话存储，带 TTL 自动过期。"""

    def __init__(self):
        self._sessions: Dict[str, SessionEntry] = {}
        self._interventions: Dict[str, str] = {}
        self._lock = asyncio.Lock()
        self._ttl = _settings.SESSION_TTL_SECONDS
        self._max_sessions = _settings.MAX_ACTIVE_SESSIONS

    # ── 会话 CRUD ─────────────────────────────────────────────

    async def create(self, session_id: str, data: Dict[str, Any]) -> None:
        async with self._lock:
            await self._evict_if_needed()
            self._sessions[session_id] = SessionEntry(data=data)

    async def get(self, session_id: str) -> Optional[Dict[str, Any]]:
        async with self._lock:
            entry = self._sessions.get(session_id)
            if entry is None:
                return None
            if entry.is_expired(self._ttl):
                del self._sessions[session_id]
                return None
            entry.touch()
            return entry.data

    async def update(self, session_id: str, patch: Dict[str, Any]) -> bool:
        async with self._lock:
            entry = self._sessions.get(session_id)
            if entry is None:
                return False
            entry.data.update(patch)
            entry.touch()
            return True

    async def delete(self, session_id: str) -> None:
        async with self._lock:
            self._sessions.pop(session_id, None)
            self._interventions.pop(session_id, None)

    async def exists(self, session_id: str) -> bool:
        return await self.get(session_id) is not None

    # ── 干预指令 ──────────────────────────────────────────────

    async def set_intervention(self, session_id: str, message: str) -> None:
        async with self._lock:
            self._interventions[session_id] = message

    async def pop_intervention(self, session_id: str) -> Optional[str]:
        """取出并消费干预指令（一次性读取）"""
        async with self._lock:
            return self._interventions.pop(session_id, None)

    # ── 内部工具 ──────────────────────────────────────────────

    async def _evict_if_needed(self):
        """过期清理 + 容量控制（已持有锁时调用）"""
        # 1. 清除超时会话
        expired = [sid for sid, e in self._sessions.items() if e.is_expired(self._ttl)]
        for sid in expired:
            del self._sessions[sid]
            self._interventions.pop(sid, None)

        # 2. 若仍超容，按最久未访问淘汰
        if len(self._sessions) >= self._max_sessions:
            oldest = sorted(self._sessions.items(), key=lambda x: x[1].last_accessed)
            to_remove = oldest[: len(self._sessions) - self._max_sessions + 1]
            for sid, _ in to_remove:
                del self._sessions[sid]
                self._interventions.pop(sid, None)

    async def active_count(self) -> int:
        async with self._lock:
            return len(self._sessions)


# ── 单例 ──────────────────────────────────────────────────────
session_mgr = SessionManager()
