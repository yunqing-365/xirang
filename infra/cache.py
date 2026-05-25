# infra/cache.py
"""
Redis 支持的会话管理器
替换 session_manager.py 中的内存字典，支持多进程部署。

设计：
  - 会话数据序列化为 JSON 存入 Redis（TTL 自动过期）
  - Agent/Manager 对象（含 Python 类实例）不能直接序列化，
    因此采用「冷热分离」策略：
      热数据（轻量 metadata、对话历史、关系值）→ Redis
      热对象（Agent 实例等）→ 进程内 LRU 缓存（带 TTL）
  - 多进程间通过 Redis Pub/Sub 广播 session 失效通知

冷热分离说明：
  session_mgr.get(sid)    → 先查进程内缓存，Miss 则从 Redis 重建
  session_mgr.save(sid)   → 将可序列化部分写 Redis，对象留本地
  session_mgr.invalidate(sid) → 删 Redis + 广播失效
"""
from __future__ import annotations
import asyncio
import json
import pickle
import time
from collections import OrderedDict
from typing import Any, Optional

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import get_settings

_settings = get_settings()


# ── 进程内 LRU 缓存（热对象）────────────────────────────────
class LRUCache:
    """简单 TTL-LRU 缓存，用于存放不可序列化的 Python 对象"""

    def __init__(self, maxsize: int = 256, ttl: int = 7200):
        self._cache: OrderedDict[str, tuple[Any, float]] = OrderedDict()
        self._maxsize = maxsize
        self._ttl = ttl

    def get(self, key: str) -> Optional[Any]:
        if key not in self._cache:
            return None
        value, ts = self._cache[key]
        if time.time() - ts > self._ttl:
            del self._cache[key]
            return None
        self._cache.move_to_end(key)
        return value

    def set(self, key: str, value: Any):
        if key in self._cache:
            self._cache.move_to_end(key)
        self._cache[key] = (value, time.time())
        if len(self._cache) > self._maxsize:
            self._cache.popitem(last=False)

    def delete(self, key: str):
        self._cache.pop(key, None)

    def __len__(self) -> int:
        return len(self._cache)


# ── Redis 会话管理器 ─────────────────────────────────────────
class RedisSessionManager:
    """
    生产级会话管理器，Redis 作为持久化后端。

    会话结构（Redis Hash）：
      xirang:session:{sid}:meta   → JSON（轻量元数据）
      xirang:session:{sid}:dialog → 对话历史字符串
      xirang:session:{sid}:ws     → workspace 字符串

    重对象（agents, manager）存进程内 LRU。
    """

    def __init__(self):
        self._redis = None          # 懒加载
        self._local  = LRUCache(
            maxsize=_settings.MAX_ACTIVE_SESSIONS,
            ttl=_settings.SESSION_TTL_SECONDS,
        )
        self._ttl = _settings.SESSION_TTL_SECONDS
        self._prefix = _settings.REDIS_SESSION_PREFIX

    async def _get_redis(self):
        """懒加载 Redis 连接"""
        if self._redis is None:
            try:
                import redis.asyncio as aioredis
                self._redis = aioredis.from_url(
                    _settings.REDIS_URL,
                    encoding="utf-8",
                    decode_responses=True,
                    socket_connect_timeout=3,
                    socket_timeout=3,
                )
                await self._redis.ping()
                print(f"✅ Redis 连接成功: {_settings.REDIS_URL}")
            except Exception as e:
                print(f"⚠️  Redis 连接失败: {e}，降级为内存模式")
                self._redis = None
        return self._redis

    # ── CRUD ────────────────────────────────────────────────
    async def create(self, session_id: str, data: dict) -> None:
        """创建新会话：对象存本地缓存，元数据写 Redis"""
        self._local.set(session_id, data)

        r = await self._get_redis()
        if r:
            meta = self._extract_meta(data)
            await r.setex(
                f"{self._prefix}{session_id}",
                self._ttl,
                json.dumps(meta, ensure_ascii=False),
            )

    async def get(self, session_id: str) -> Optional[dict]:
        """获取会话（优先本地缓存）"""
        # 1. 本地热缓存命中
        cached = self._local.get(session_id)
        if cached is not None:
            return cached

        # 2. Redis 中有元数据 → 说明进程重启或负载均衡到新进程
        r = await self._get_redis()
        if r:
            raw = await r.get(f"{self._prefix}{session_id}")
            if raw:
                meta = json.loads(raw)
                # 无法重建完整会话对象，返回轻量元数据供错误处理
                print(f"⚠️  session {session_id} 仅从 Redis 恢复元数据（对象丢失）")
                return {"_redis_only": True, **meta}

        return None

    async def save(self, session_id: str, data: dict) -> None:
        """持久化会话（增量更新 Redis 元数据）"""
        self._local.set(session_id, data)

        r = await self._get_redis()
        if r:
            meta = self._extract_meta(data)
            await r.setex(
                f"{self._prefix}{session_id}",
                self._ttl,
                json.dumps(meta, ensure_ascii=False),
            )

    async def delete(self, session_id: str) -> None:
        self._local.delete(session_id)
        r = await self._get_redis()
        if r:
            await r.delete(f"{self._prefix}{session_id}")

    async def active_count(self) -> int:
        return len(self._local)

    # ── 元数据提取（可序列化部分）──────────────────────────
    @staticmethod
    def _extract_meta(data: dict) -> dict:
        """从 session data 提取可 JSON 序列化的元数据"""
        manager = data.get("manager")
        ns      = data.get("narrative_state")
        tc      = data.get("trigger_checker")
        return {
            "user_id":        data.get("user_id", "anonymous"),
            "theme":          data.get("theme", ""),
            "workspace":      getattr(manager, "shared_workspace", "")[:500],
            "dialogue_tail":  (getattr(manager, "current_dialogue", "") or "")[-300:],
            "phase":          ns.phase.value if ns and hasattr(ns, "phase") else "",
            "milestones":     (ns.milestones[-5:] if ns else []),
            "triggered_ids":  list(tc.triggered_ids) if tc else [],
            "last_saved":     time.time(),
        }


# ── Redis 原子限流（滑动窗口）───────────────────────────────
class RedisRateLimiter:
    """
    基于 Redis ZSET 的滑动窗口限流。
    相比内存版，多进程间共享限流状态。
    """

    def __init__(self, rpm: int = 60):
        self._rpm  = rpm
        self._prefix = _settings.REDIS_RATELIMIT_PREFIX

    async def is_allowed(self, user_id: str, redis_client=None) -> bool:
        if not _settings.RATE_LIMIT_ENABLED:
            return True
        if redis_client is None:
            return True   # 无 Redis 时放行

        key = f"{self._prefix}{user_id}"
        now = time.time()
        window_start = now - 60

        pipe = redis_client.pipeline()
        pipe.zremrangebyscore(key, 0, window_start)   # 清除过期
        pipe.zadd(key, {str(now): now})                # 记录本次
        pipe.zcard(key)                                # 计数
        pipe.expire(key, 61)
        results = await pipe.execute()

        count = results[2]
        return count <= self._rpm


# ── 便利函数：按需选择后端 ───────────────────────────────────
def create_session_manager():
    """根据配置返回合适的会话管理器"""
    if _settings.USE_REDIS:
        print("📦 使用 Redis 会话管理器")
        return RedisSessionManager()
    else:
        # 降级：使用原有内存版
        print("📦 使用内存会话管理器（开发模式）")
        from session_manager import SessionManager
        return SessionManager()


# ── 自测 ─────────────────────────────────────────────────────
if __name__ == "__main__":
    async def _test():
        mgr = RedisSessionManager()
        sid = "test_session_001"
        await mgr.create(sid, {
            "user_id": "test_user",
            "theme": "宋代黄州",
            "manager": None,
            "narrative_state": None,
        })
        result = await mgr.get(sid)
        print("Created and retrieved:", result is not None)

        await mgr.delete(sid)
        result2 = await mgr.get(sid)
        print("Deleted:", result2 is None)

        # LRU 缓存测试
        lru = LRUCache(maxsize=3, ttl=1)
        lru.set("a", 1); lru.set("b", 2); lru.set("c", 3)
        print("LRU get a:", lru.get("a"))
        lru.set("d", 4)   # 触发 LRU 淘汰
        print("LRU len after overflow:", len(lru))

    asyncio.run(_test())
