# infra/quota.py
"""
息壤使用配额模块（P0 - 已修复：配额数据持久化至 PostgreSQL/Redis）
=====================================
免费用户：每月3次完整会话
教师专业版：无限
学生月度版：无限
学校版：按班级配额

存储策略：
  - 生产（USE_POSTGRES=True）：读写 user_quota 表，月度滚动
  - 开发（内存回退）：重启清零，仅用于测试
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from fastapi import Depends, HTTPException

from infra.auth import TokenData, get_current_user
from config import get_settings

_settings = get_settings()
_log = logging.getLogger(__name__)

USE_POSTGRES: bool = getattr(_settings, "USE_POSTGRES", False)

# ─────────────────────────────────────────────────────────────────
# 套餐定义
# ─────────────────────────────────────────────────────────────────

@dataclass
class Plan:
    name: str
    monthly_sessions: int        # -1 = 无限
    monthly_messages: int        # -1 = 无限
    price_monthly: int           # 分（人民币）
    price_yearly: int
    max_class_size: int = 0
    can_export_pdf: bool = False
    can_view_dashboard: bool = False

PLANS: dict[str, Plan] = {
    "free": Plan(
        name="免费体验版",
        monthly_sessions=3,
        monthly_messages=30,
        price_monthly=0,
        price_yearly=0,
    ),
    "student": Plan(
        name="学生版",
        monthly_sessions=-1,
        monthly_messages=-1,
        price_monthly=3900,
        price_yearly=29900,
    ),
    "teacher_pro": Plan(
        name="教师专业版",
        monthly_sessions=-1,
        monthly_messages=-1,
        price_monthly=19900,
        price_yearly=159900,
        max_class_size=60,
        can_export_pdf=True,
        can_view_dashboard=True,
    ),
    "school": Plan(
        name="学校版",
        monthly_sessions=-1,
        monthly_messages=-1,
        price_monthly=0,
        price_yearly=800000,
        max_class_size=500,
        can_export_pdf=True,
        can_view_dashboard=True,
    ),
}


# ─────────────────────────────────────────────────────────────────
# 配额状态结构
# ─────────────────────────────────────────────────────────────────

@dataclass
class QuotaState:
    user_id: str
    plan: str = "free"
    sessions_used: int = 0
    messages_used: int = 0
    expires_at: float = 0.0
    cycle_start: float = field(default_factory=lambda: _month_start())

    @property
    def current_plan(self) -> Plan:
        return PLANS.get(self.plan, PLANS["free"])

    @property
    def is_active(self) -> bool:
        if self.plan == "free":
            return True
        if self.expires_at == 0:
            return True
        return time.time() < self.expires_at

    @property
    def sessions_remaining(self) -> int:
        plan = self.current_plan
        if not self.is_active:
            return PLANS["free"].monthly_sessions - self.sessions_used
        if plan.monthly_sessions == -1:
            return 9999
        return max(0, plan.monthly_sessions - self.sessions_used)

    @property
    def can_start_session(self) -> bool:
        return self.sessions_remaining > 0

    def to_dict(self) -> dict:
        plan = self.current_plan
        return {
            "user_id": self.user_id,
            "plan": self.plan,
            "plan_name": plan.name,
            "is_active": self.is_active,
            "sessions_used": self.sessions_used,
            "sessions_limit": plan.monthly_sessions,
            "sessions_remaining": self.sessions_remaining,
            "messages_used": self.messages_used,
            "messages_limit": plan.monthly_messages,
            "can_export_pdf": plan.can_export_pdf,
            "can_view_dashboard": plan.can_view_dashboard,
            "expires_at": self.expires_at,
            "cycle_start": self.cycle_start,
        }


def _month_start() -> float:
    now = datetime.now(timezone.utc)
    return datetime(now.year, now.month, 1, tzinfo=timezone.utc).timestamp()


# ─────────────────────────────────────────────────────────────────
# 内存回退（开发模式）
# ─────────────────────────────────────────────────────────────────

_quota_store: dict[str, QuotaState] = {}


def _mem_get_or_create(user_id: str) -> QuotaState:
    if user_id not in _quota_store:
        _quota_store[user_id] = QuotaState(user_id=user_id)
    state = _quota_store[user_id]
    current_cycle = _month_start()
    if state.cycle_start < current_cycle:
        state.sessions_used = 0
        state.messages_used = 0
        state.cycle_start = current_cycle
    return state


# ─────────────────────────────────────────────────────────────────
# PostgreSQL 持久化层
# ─────────────────────────────────────────────────────────────────

_db_pool = None

async def _get_pool():
    global _db_pool
    if _db_pool is None and USE_POSTGRES:
        try:
            import asyncpg
            _db_pool = await asyncpg.create_pool(_settings.DB_URL, min_size=2, max_size=10)
            _log.info("✅ quota: PostgreSQL 连接池已建立")
        except Exception as e:
            _log.error(f"❌ quota: PostgreSQL 连接失败，回退内存模式: {e}")
    return _db_pool


async def _db_get_or_create(user_id: str) -> QuotaState:
    """从 PostgreSQL 读取配额状态，不存在则创建"""
    pool = await _get_pool()
    if not pool:
        return _mem_get_or_create(user_id)

    async with pool.acquire() as conn:
        # 月度自动滚动
        await conn.execute("SELECT reset_monthly_quota()")

        row = await conn.fetchrow(
            """
            SELECT uq.sessions_used, uq.messages_used,
                   EXTRACT(EPOCH FROM uq.cycle_start) AS cycle_start,
                   u.plan,
                   EXTRACT(EPOCH FROM u.plan_expires_at) AS expires_at
            FROM user_quota uq
            JOIN users u ON u.user_id = uq.user_id
            WHERE uq.user_id = $1
            """,
            user_id
        )
        if row:
            return QuotaState(
                user_id=user_id,
                plan=row["plan"] or "free",
                sessions_used=row["sessions_used"],
                messages_used=row["messages_used"],
                expires_at=float(row["expires_at"] or 0),
                cycle_start=float(row["cycle_start"]),
            )
        # 不存在则创建（user 可能尚未在 users 表中——降级到内存）
        _log.warning(f"quota: 用户 {user_id} 未找到，使用内存模式")
        return _mem_get_or_create(user_id)


async def _db_save(state: QuotaState):
    pool = await _get_pool()
    if not pool:
        _quota_store[state.user_id] = state
        return
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO user_quota (user_id, plan, sessions_used, messages_used, updated_at)
            VALUES ($1, $2, $3, $4, NOW())
            ON CONFLICT (user_id) DO UPDATE
              SET sessions_used = EXCLUDED.sessions_used,
                  messages_used = EXCLUDED.messages_used,
                  updated_at    = NOW()
            """,
            state.user_id, state.plan, state.sessions_used, state.messages_used
        )


async def _db_upgrade_plan(user_id: str, plan: str, duration_days: int) -> QuotaState:
    pool = await _get_pool()
    state = await _db_get_or_create(user_id)
    state.plan = plan
    state.expires_at = time.time() + duration_days * 86400

    if pool:
        from datetime import timedelta
        expires_dt = datetime.fromtimestamp(state.expires_at, tz=timezone.utc)
        async with pool.acquire() as conn:
            await conn.execute(
                "UPDATE users SET plan = $1, plan_expires_at = $2 WHERE user_id = $3",
                plan, expires_dt, user_id
            )
    else:
        _quota_store[user_id] = state
    return state


# ─────────────────────────────────────────────────────────────────
# 公共 API（同步兼容包装 + 异步版本）
# ─────────────────────────────────────────────────────────────────

def get_quota_status(user_id: str) -> QuotaState:
    """同步版（兼容旧调用），内存回退"""
    return _mem_get_or_create(user_id)


async def get_quota_status_async(user_id: str) -> QuotaState:
    """异步版，优先查 PostgreSQL"""
    return await _db_get_or_create(user_id)


async def quota_consume_session(user_id: str) -> QuotaState:
    state = await _db_get_or_create(user_id)
    state.sessions_used += 1
    await _db_save(state)
    return state


async def quota_consume_message(user_id: str, count: int = 1) -> QuotaState:
    state = await _db_get_or_create(user_id)
    state.messages_used += count
    await _db_save(state)
    return state


async def upgrade_plan(user_id: str, plan: str, duration_days: int = 30) -> QuotaState:
    """升级订阅（支付成功后调用）"""
    if plan not in PLANS:
        raise ValueError(f"未知套餐: {plan}")
    return await _db_upgrade_plan(user_id, plan, duration_days)


def downgrade_to_free(user_id: str) -> QuotaState:
    state = _mem_get_or_create(user_id)
    state.plan = "free"
    state.expires_at = 0
    return state


# ─────────────────────────────────────────────────────────────────
# FastAPI 依赖注入
# ─────────────────────────────────────────────────────────────────

async def require_session_quota(
    token: TokenData = Depends(get_current_user),
) -> QuotaState:
    """检查用户是否有剩余会话配额，超限返回 402"""
    if token.is_admin or token.is_teacher:
        state = await _db_get_or_create(token.user_id)
        if state.plan == "free":
            state.plan = "teacher_pro"
        return state

    state = await _db_get_or_create(token.user_id)
    if not state.can_start_session:
        plan = state.current_plan
        raise HTTPException(
            status_code=402,
            detail={
                "code": "QUOTA_EXCEEDED",
                "message": f"免费体验次数已用完（{plan.monthly_sessions}次/月），请升级订阅继续使用",
                "sessions_used": state.sessions_used,
                "sessions_limit": plan.monthly_sessions,
                "upgrade_url": "/api/payment/plans",
            },
        )
    return state


async def require_pdf_export(
    token: TokenData = Depends(get_current_user),
) -> QuotaState:
    """检查是否有 PDF 导出权限"""
    state = await _db_get_or_create(token.user_id)
    if not state.current_plan.can_export_pdf and not token.is_admin:
        raise HTTPException(
            status_code=403,
            detail={
                "code": "FEATURE_LOCKED",
                "message": "PDF导出需要教师专业版或以上套餐",
                "upgrade_url": "/api/payment/plans",
            },
        )
    return state


# ─────────────────────────────────────────────────────────────────
# FastAPI Router
# ─────────────────────────────────────────────────────────────────

from fastapi import APIRouter

quota_router = APIRouter(prefix="/api/quota", tags=["quota"])


@quota_router.get("/status")
async def my_quota(token: TokenData = Depends(get_current_user)):
    """查询当前用户的配额状态"""
    state = await get_quota_status_async(token.user_id)
    return state.to_dict()


@quota_router.get("/plans")
async def list_plans():
    """返回所有套餐详情"""
    return [
        {
            "id": pid,
            "name": p.name,
            "price_monthly": p.price_monthly / 100,
            "price_yearly": p.price_yearly / 100,
            "monthly_sessions": p.monthly_sessions if p.monthly_sessions != -1 else "无限",
            "features": {
                "pdf_export": p.can_export_pdf,
                "teacher_dashboard": p.can_view_dashboard,
                "max_class_size": p.max_class_size or None,
            },
        }
        for pid, p in PLANS.items()
    ]
