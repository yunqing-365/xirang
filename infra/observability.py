# infra/observability.py
"""
可观测性模块：Prometheus 指标 + 结构化日志

指标体系：
  xirang_requests_total          请求计数（按路由/状态）
  xirang_request_duration_seconds 请求耗时直方图
  xirang_active_sessions         活跃会话数（Gauge）
  xirang_llm_calls_total         LLM API 调用计数（按模型/成功/失败）
  xirang_llm_duration_seconds    LLM 响应耗时
  xirang_rag_queries_total       RAG 检索计数
  xirang_narrative_events_total  叙事事件计数（按类型）
  xirang_trigger_fires_total     历史触发器触发计数（按 trigger_id）
  xirang_offline_evolutions_total 离线推演计数

挂载方式（server.py）：
  from infra.observability import setup_observability, metrics_router
  setup_observability(app)
  app.include_router(metrics_router)
"""
from __future__ import annotations
import time
import logging
import sys
from contextlib import asynccontextmanager
from functools import wraps
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import get_settings

_settings = get_settings()

# ── 结构化日志 ────────────────────────────────────────────────
def _setup_logging():
    """配置结构化日志（JSON 或普通格式）"""
    level = getattr(logging, _settings.LOG_LEVEL.upper(), logging.INFO)

    if _settings.LOG_JSON:
        try:
            import structlog
            structlog.configure(
                processors=[
                    structlog.stdlib.filter_by_level,
                    structlog.stdlib.add_logger_name,
                    structlog.stdlib.add_log_level,
                    structlog.processors.TimeStamper(fmt="iso"),
                    structlog.processors.StackInfoRenderer(),
                    structlog.processors.format_exc_info,
                    structlog.processors.JSONRenderer(),
                ],
                wrapper_class=structlog.stdlib.BoundLogger,
                context_class=dict,
                logger_factory=structlog.stdlib.LoggerFactory(),
                cache_logger_on_first_use=True,
            )
            return structlog.get_logger("xirang")
        except ImportError:
            pass

    # 标准格式
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger("xirang")


logger = _setup_logging()


# ── Prometheus 指标 ───────────────────────────────────────────
_metrics_initialized = False
_METRICS: dict = {}

def _init_metrics():
    global _metrics_initialized, _METRICS
    if _metrics_initialized:
        return
    try:
        from prometheus_client import (
            Counter, Histogram, Gauge, CollectorRegistry, REGISTRY
        )
        _METRICS = {
            "requests_total": Counter(
                "xirang_requests_total",
                "HTTP 请求计数",
                ["method", "path", "status"],
            ),
            "request_duration": Histogram(
                "xirang_request_duration_seconds",
                "HTTP 请求耗时",
                ["method", "path"],
                buckets=[0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10],
            ),
            "active_sessions": Gauge(
                "xirang_active_sessions",
                "当前活跃会话数",
            ),
            "llm_calls_total": Counter(
                "xirang_llm_calls_total",
                "LLM API 调用计数",
                ["model", "endpoint", "status"],
            ),
            "llm_duration": Histogram(
                "xirang_llm_duration_seconds",
                "LLM 响应耗时",
                ["model", "endpoint"],
                buckets=[0.5, 1, 2, 5, 10, 20, 30, 60],
            ),
            "rag_queries_total": Counter(
                "xirang_rag_queries_total",
                "RAG 检索计数",
                ["era", "status"],
            ),
            "narrative_events_total": Counter(
                "xirang_narrative_events_total",
                "叙事事件计数",
                ["event_type"],
            ),
            "trigger_fires_total": Counter(
                "xirang_trigger_fires_total",
                "历史触发器触发计数",
                ["trigger_id", "era"],
            ),
            "offline_evolutions_total": Counter(
                "xirang_offline_evolutions_total",
                "离线推演触发计数",
                ["era"],
            ),
        }
        _metrics_initialized = True
        logger.info("Prometheus 指标初始化成功")
    except ImportError:
        logger.warning("prometheus_client 未安装，指标采集禁用")
    except Exception as e:
        logger.warning(f"指标初始化失败: {e}")


def get_metric(name: str):
    """安全获取指标对象（未初始化时返回 None）"""
    if not _metrics_initialized:
        _init_metrics()
    return _METRICS.get(name)


# ── 便捷记录函数 ─────────────────────────────────────────────
def record_request(method: str, path: str, status: int, duration: float):
    c = get_metric("requests_total")
    if c: c.labels(method=method, path=path, status=str(status)).inc()
    h = get_metric("request_duration")
    if h: h.labels(method=method, path=path).observe(duration)

def record_llm_call(model: str, endpoint: str, success: bool, duration: float):
    c = get_metric("llm_calls_total")
    if c: c.labels(model=model, endpoint=endpoint,
                   status="ok" if success else "error").inc()
    h = get_metric("llm_duration")
    if h: h.labels(model=model, endpoint=endpoint).observe(duration)

def record_rag_query(era: str, success: bool):
    c = get_metric("rag_queries_total")
    if c: c.labels(era=era, status="ok" if success else "miss").inc()

def record_narrative_event(event_type: str):
    c = get_metric("narrative_events_total")
    if c: c.labels(event_type=event_type).inc()

def record_trigger_fire(trigger_id: str, era: str):
    c = get_metric("trigger_fires_total")
    if c: c.labels(trigger_id=trigger_id, era=era).inc()

def record_offline_evolution(era: str):
    c = get_metric("offline_evolutions_total")
    if c: c.labels(era=era).inc()

def update_active_sessions(count: int):
    g = get_metric("active_sessions")
    if g: g.set(count)


# ── FastAPI 中间件 ────────────────────────────────────────────
from fastapi import Request, Response
from fastapi.routing import APIRouter
import asyncio

metrics_router = APIRouter(tags=["observability"])

@metrics_router.get("/metrics")
async def prometheus_metrics():
    """Prometheus scrape 端点（/metrics）"""
    if not _settings.METRICS_ENABLED:
        return Response(content="# metrics disabled", media_type="text/plain")
    try:
        from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
        return Response(
            content=generate_latest(),
            media_type=CONTENT_TYPE_LATEST,
        )
    except ImportError:
        return Response(content="# prometheus_client not installed",
                        media_type="text/plain")

@metrics_router.get("/health")
async def health_check():
    """健康检查端点"""
    return {
        "status": "ok",
        "env": _settings.ENV,
        "metrics_enabled": _settings.METRICS_ENABLED,
        "auth_enabled": _settings.AUTH_ENABLED,
        "redis_enabled": _settings.USE_REDIS,
    }

@metrics_router.get("/ready")
async def readiness_check():
    """就绪检查（含 Redis 连通性）"""
    checks: dict[str, str] = {"api": "ok"}

    if _settings.USE_REDIS:
        try:
            import redis.asyncio as aioredis
            r = aioredis.from_url(_settings.REDIS_URL, socket_connect_timeout=2)
            await asyncio.wait_for(r.ping(), timeout=2)
            checks["redis"] = "ok"
            await r.aclose()
        except Exception as e:
            checks["redis"] = f"error: {e}"
            return Response(
                content=str({"status": "not_ready", "checks": checks}),
                status_code=503,
            )

    return {"status": "ready", "checks": checks}


# ── 请求计时中间件 ────────────────────────────────────────────
def setup_observability(app):
    """在 FastAPI app 上注册可观测性中间件和路由"""
    if not _settings.METRICS_ENABLED:
        app.include_router(metrics_router)
        return

    _init_metrics()

    from starlette.middleware.base import BaseHTTPMiddleware

    class MetricsMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request: Request, call_next: Callable) -> Response:
            start = time.perf_counter()
            try:
                response = await call_next(request)
                duration = time.perf_counter() - start
                # 只记录 API 路由（跳过静态文件）
                if request.url.path.startswith("/api"):
                    record_request(
                        method=request.method,
                        path=request.url.path,
                        status=response.status_code,
                        duration=duration,
                    )
                return response
            except Exception as exc:
                duration = time.perf_counter() - start
                record_request(request.method, request.url.path, 500, duration)
                raise

    app.add_middleware(MetricsMiddleware)
    app.include_router(metrics_router)
    logger.info("可观测性中间件已注册")


if __name__ == "__main__":
    _init_metrics()
    # 模拟记录几条指标
    record_request("GET", "/api/stream_next/test", 200, 1.23)
    record_llm_call("gpt-4o-mini", "chat", True, 2.5)
    record_narrative_event("agent_action")
    record_trigger_fire("song_wutai_case", "song")
    print("✅ 指标记录测试通过")
    print(f"   已初始化 {len(_METRICS)} 个指标")
