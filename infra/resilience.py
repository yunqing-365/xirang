# infra/resilience.py
"""
息壤 · 生产加固层

解决审计发现的三类问题：
  1. 12 个 LLM 调用点无重试保护 → LLMCallGuard（指数退避 + 熔断）
  2. 73 个 API 无限速 → RateLimiter（滑动窗口，内存 or Redis）
  3. 高频调用无缓存 → ResultCache（TTL LRU，按内容哈希去重）

使用方式：
  # LLM 调用保护
  from infra.resilience import llm_guard
  result = await llm_guard.call(client.chat.completions.create, **kwargs)

  # API 限速（在 server.py 的路由上装饰）
  from infra.resilience import rate_limit
  @rate_limit("emotion_arc", max_calls=30, window_seconds=60)

  # 结果缓存
  from infra.resilience import result_cache
  cached = await result_cache.get_or_set("key", async_fn, ttl=300)
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Callable, Coroutine, Dict, Optional

logger = logging.getLogger("xirang.resilience")


# ═══════════════════════════════════════════════════════════════
# 1. LLM 调用守卫（重试 + 指数退避 + 熔断）
# ═══════════════════════════════════════════════════════════════

@dataclass
class CircuitState:
    failures: int = 0
    last_failure_ts: float = 0.0
    is_open: bool = False          # True = 熔断中，拒绝请求


class LLMCallGuard:
    """
    LLM API 调用守卫。
    - 失败自动重试（指数退避）
    - 连续失败超阈值 → 熔断（Circuit Breaker）
    - 熔断恢复期后自动半开探测
    """

    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 30.0,
        failure_threshold: int = 5,    # 熔断触发连续失败次数
        recovery_seconds: float = 60.0, # 熔断恢复等待时间
    ):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.failure_threshold = failure_threshold
        self.recovery_seconds = recovery_seconds
        self._circuit = CircuitState()
        self._lock = asyncio.Lock()

    def _is_circuit_open(self) -> bool:
        if not self._circuit.is_open:
            return False
        # 检查是否超过恢复时间（进入半开状态）
        if time.time() - self._circuit.last_failure_ts > self.recovery_seconds:
            self._circuit.is_open = False
            logger.info("🔌 LLM 熔断器半开，尝试恢复")
            return False
        return True

    async def call(
        self,
        fn: Callable[..., Coroutine],
        *args,
        fallback: Any = None,
        **kwargs,
    ) -> Any:
        """
        安全调用 LLM API，带重试和熔断。
        fallback: 所有重试失败后的兜底返回值。
        """
        if self._is_circuit_open():
            logger.warning("⚡ LLM 熔断中，返回兜底值")
            return fallback

        last_exc = None
        for attempt in range(self.max_retries + 1):
            try:
                result = await fn(*args, **kwargs)
                # 成功：重置熔断计数
                async with self._lock:
                    self._circuit.failures = 0
                return result

            except Exception as exc:
                last_exc = exc
                async with self._lock:
                    self._circuit.failures += 1
                    self._circuit.last_failure_ts = time.time()
                    if self._circuit.failures >= self.failure_threshold:
                        self._circuit.is_open = True
                        logger.error(
                            f"🔴 LLM 熔断触发（连续失败 {self._circuit.failures} 次）: {exc}"
                        )
                        return fallback

                if attempt < self.max_retries:
                    delay = min(
                        self.base_delay * (2 ** attempt),
                        self.max_delay,
                    )
                    logger.warning(
                        f"⚠️ LLM 调用失败（第{attempt+1}次），{delay:.1f}s 后重试: {exc}"
                    )
                    await asyncio.sleep(delay)

        logger.error(f"❌ LLM 调用最终失败（{self.max_retries+1}次）: {last_exc}")
        return fallback

    def get_status(self) -> dict:
        return {
            "is_open": self._circuit.is_open,
            "failures": self._circuit.failures,
            "last_failure": self._circuit.last_failure_ts,
        }


# 全局守卫实例（所有引擎共用）
llm_guard = LLMCallGuard()


# ═══════════════════════════════════════════════════════════════
# 2. 滑动窗口限速器
# ═══════════════════════════════════════════════════════════════

class RateLimiter:
    """
    内存版滑动窗口限速器。
    适合单进程；多进程场景接 Redis 版本。
    """

    def __init__(self):
        # {key: deque[timestamp]}
        self._windows: Dict[str, deque] = defaultdict(deque)
        self._lock = asyncio.Lock()

    async def is_allowed(
        self,
        key: str,
        max_calls: int,
        window_seconds: float,
    ) -> bool:
        now = time.time()
        cutoff = now - window_seconds
        async with self._lock:
            window = self._windows[key]
            # 清除过期记录
            while window and window[0] < cutoff:
                window.popleft()
            if len(window) >= max_calls:
                return False
            window.append(now)
            return True

    async def wait_and_acquire(
        self,
        key: str,
        max_calls: int,
        window_seconds: float,
        max_wait: float = 5.0,
    ) -> bool:
        """等待直到获得配额（或超时）"""
        deadline = time.time() + max_wait
        while time.time() < deadline:
            if await self.is_allowed(key, max_calls, window_seconds):
                return True
            await asyncio.sleep(0.1)
        return False


# 全局限速器
rate_limiter = RateLimiter()


def rate_limit(
    key_prefix: str,
    max_calls: int = 60,
    window_seconds: float = 60.0,
    per_user: bool = True,
):
    """
    FastAPI 路由限速装饰器。

    用法：
        @app.get("/api/emotion/arc/{npc_name}")
        @rate_limit("emotion_arc", max_calls=30, window_seconds=60)
        async def get_emotion_arc(npc_name: str, ...):
    """
    def decorator(fn: Callable) -> Callable:
        @wraps(fn)
        async def wrapper(*args, **kwargs):
            # 尝试从 kwargs 提取 user_id 或 session_id 作为限速 key
            uid = kwargs.get("user_id") or kwargs.get("session_id") or "global"
            key = f"{key_prefix}:{uid}" if per_user else key_prefix

            allowed = await rate_limiter.is_allowed(key, max_calls, window_seconds)
            if not allowed:
                from fastapi import HTTPException
                raise HTTPException(
                    status_code=429,
                    detail=f"请求过于频繁，请 {window_seconds:.0f} 秒后重试",
                )
            return await fn(*args, **kwargs)
        return wrapper
    return decorator


# ═══════════════════════════════════════════════════════════════
# 3. 结果缓存（TTL LRU）
# ═══════════════════════════════════════════════════════════════

@dataclass
class CacheEntry:
    value: Any
    created_at: float = field(default_factory=time.time)
    hits: int = 0


class ResultCache:
    """
    异步 TTL-LRU 结果缓存。
    用于缓存 LLM 生成的注释、跨学科连接、多视角等高成本结果。
    """

    def __init__(self, maxsize: int = 512, default_ttl: float = 600.0):
        self._store: Dict[str, CacheEntry] = {}
        self._access_order: deque = deque()  # LRU 顺序
        self._maxsize = maxsize
        self._default_ttl = default_ttl
        self._lock = asyncio.Lock()
        self._hits = 0
        self._misses = 0

    @staticmethod
    def make_key(*args, **kwargs) -> str:
        """从参数生成缓存 key（内容哈希）"""
        raw = json.dumps({"args": args, "kwargs": kwargs}, ensure_ascii=False, sort_keys=True)
        return hashlib.md5(raw.encode()).hexdigest()

    async def get(self, key: str, ttl: Optional[float] = None) -> Optional[Any]:
        async with self._lock:
            entry = self._store.get(key)
            if entry is None:
                self._misses += 1
                return None
            age = time.time() - entry.created_at
            effective_ttl = ttl or self._default_ttl
            if age > effective_ttl:
                del self._store[key]
                self._misses += 1
                return None
            entry.hits += 1
            self._hits += 1
            return entry.value

    async def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        async with self._lock:
            # LRU 淘汰
            if len(self._store) >= self._maxsize and key not in self._store:
                if self._access_order:
                    oldest = self._access_order.popleft()
                    self._store.pop(oldest, None)
            self._store[key] = CacheEntry(value=value)
            self._access_order.append(key)

    async def get_or_set(
        self,
        key: str,
        fn: Callable[[], Coroutine],
        ttl: Optional[float] = None,
    ) -> Any:
        """缓存穿透保护：有缓存则返回，否则调用 fn 并缓存结果"""
        cached = await self.get(key, ttl)
        if cached is not None:
            return cached
        result = await fn()
        if result is not None:
            await self.set(key, result, ttl)
        return result

    def stats(self) -> dict:
        total = self._hits + self._misses
        return {
            "size": len(self._store),
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(self._hits / total * 100, 1) if total else 0,
        }

    async def invalidate(self, key: str) -> None:
        async with self._lock:
            self._store.pop(key, None)

    async def clear(self) -> None:
        async with self._lock:
            self._store.clear()
            self._access_order.clear()


# 全局缓存实例
result_cache = ResultCache(maxsize=512, default_ttl=600.0)

# 特化缓存（不同 TTL）
annotation_cache = ResultCache(maxsize=256, default_ttl=3600.0)   # 注释：1小时
cross_link_cache  = ResultCache(maxsize=128, default_ttl=1800.0)  # 跨学科：30分钟
perspective_cache = ResultCache(maxsize=128, default_ttl=900.0)   # 多视角：15分钟


# ═══════════════════════════════════════════════════════════════
# 4. 优雅降级装饰器
# ═══════════════════════════════════════════════════════════════

def graceful_degradation(
    fallback: Any = None,
    log_errors: bool = True,
    error_msg: str = "",
):
    """
    优雅降级：LLM 调用失败时返回兜底值，不中断用户体验。

    用法：
        @graceful_degradation(fallback=[], log_errors=True)
        async def generate_perspectives(self, ...):
            ...
    """
    def decorator(fn: Callable) -> Callable:
        @wraps(fn)
        async def wrapper(*args, **kwargs):
            try:
                return await fn(*args, **kwargs)
            except Exception as exc:
                if log_errors:
                    msg = error_msg or f"[{fn.__qualname__}] 降级触发"
                    logger.warning(f"⚡ {msg}: {exc}")
                return fallback
        return wrapper
    return decorator


# ═══════════════════════════════════════════════════════════════
# 5. 健康检查聚合
# ═══════════════════════════════════════════════════════════════

async def get_resilience_status() -> dict:
    """返回所有弹性组件的当前状态，供 /health 端点使用"""
    return {
        "llm_circuit": llm_guard.get_status(),
        "cache_stats": {
            "result": result_cache.stats(),
            "annotation": annotation_cache.stats(),
            "cross_link": cross_link_cache.stats(),
            "perspective": perspective_cache.stats(),
        },
    }
