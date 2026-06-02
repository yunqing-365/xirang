# infra/quota.py
"""
息壤使用配额模块（P0）
=====================================
免费用户：每月3次完整会话
教师专业版：无限
学生月度版：无限
学校版：按班级配额

配额状态存储：
  - 开发/单机：内存 dict（重启清零，无所谓）
  - 生产：Redis INCR + EXPIRE（原子操作，跨进程安全）

快速使用：
    from infra.quota import quota_check, quota_consume, get_quota_status
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from fastapi import Depends, HTTPException

from infra.auth import TokenData, get_current_user
from config import get_settings

_settings = get_settings()

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
    max_class_size: int = 0      # 教师版专属
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
        price_monthly=3900,    # ¥39
        price_yearly=29900,    # ¥299
    ),
    "teacher_pro": Plan(
        name="教师专业版",
        monthly_sessions=-1,
        monthly_messages=-1,
        price_monthly=19900,   # ¥199
        price_yearly=159900,   # ¥1599
        max_class_size=60,
        can_export_pdf=True,
        can_view_dashboard=True,
    ),
    "school": Plan(
        name="学校版",
        monthly_sessions=-1,
        monthly_messages=-1,
        price_monthly=0,       # 按年签合同
        price_yearly=800000,   # ¥8000
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
    # 本月已用
    sessions_used: int = 0
    messages_used: int = 0
    # 订阅到期时间（UNIX timestamp，0=永不）
    expires_at: float = 0.0
    # 本月计费周期起始
    cycle_start: float = field(default_factory=lambda: _month_start())

    @property
    def current_plan(self) -> Plan:
        return PLANS.get(self.plan, PLANS["free"])

    @property
    def is_active(self) -> bool:
        """订阅是否有效"""
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
    """当月1日00:00:00 UTC 的时间戳"""
    now = datetime.now(timezone.utc)
    return datetime(now.year, now.month, 1, tzinfo=timezone.utc).timestamp()


# ─────────────────────────────────────────────────────────────────
# 内存存储（生产替换为 Redis）
# ─────────────────────────────────────────────────────────────────

_quota_store: dict[str, QuotaState] = {}


def _get_or_create(user_id: str) -> QuotaState:
    if user_id not in _quota_store:
        _quota_store[user_id] = QuotaState(user_id=user_id)
    state = _quota_store[user_id]
    # 自动滚动月度周期
    current_cycle = _month_start()
    if state.cycle_start < current_cycle:
        state.sessions_used = 0
        state.messages_used = 0
        state.cycle_start = current_cycle
    return state


# ─────────────────────────────────────────────────────────────────
# 公共 API
# ─────────────────────────────────────────────────────────────────

def get_quota_status(user_id: str) -> QuotaState:
    return _get_or_create(user_id)


def quota_consume_session(user_id: str) -> QuotaState:
    """消耗一次会话配额，返回最新状态"""
    state = _get_or_create(user_id)
    state.sessions_used += 1
    return state


def quota_consume_message(user_id: str, count: int = 1) -> QuotaState:
    state = _get_or_create(user_id)
    state.messages_used += count
    return state


def upgrade_plan(
    user_id: str,
    plan: str,
    duration_days: int = 30,
) -> QuotaState:
    """升级订阅（支付成功后调用）"""
    if plan not in PLANS:
        raise ValueError(f"未知套餐: {plan}")
    state = _get_or_create(user_id)
    state.plan = plan
    state.expires_at = time.time() + duration_days * 86400
    return state


def downgrade_to_free(user_id: str) -> QuotaState:
    state = _get_or_create(user_id)
    state.plan = "free"
    state.expires_at = 0
    return state


# ─────────────────────────────────────────────────────────────────
# FastAPI 依赖注入
# ─────────────────────────────────────────────────────────────────

async def require_session_quota(
    token: TokenData = Depends(get_current_user),
) -> QuotaState:
    """
    依赖注入：检查用户是否有剩余会话配额。
    超限时返回 402 Payment Required。
    管理员/教师（已验证）直接放行。
    """
    # 教师/管理员不受配额限制
    if token.is_admin or token.is_teacher:
        state = _get_or_create(token.user_id)
        # 确保教师用的是 teacher_pro 以上套餐
        if state.plan == "free":
            state.plan = "teacher_pro"
        return state

    state = _get_or_create(token.user_id)
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
    """检查是否有 PDF 导出权限（教师版及以上）"""
    state = _get_or_create(token.user_id)
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
    state = get_quota_status(token.user_id)
    return state.to_dict()


@quota_router.get("/plans")
async def list_plans():
    """返回所有套餐详情（用于升级页面）"""
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
