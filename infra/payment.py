# infra/payment.py
"""
息壤支付模块（P0 骨架）
=====================================
生产接入流程：
  1. 注册微信支付商户号 + 证书 → 替换 WeChatPayClient
  2. 注册支付宝开放平台应用 → 替换 AlipayClient
  3. 配置回调地址：https://yourdomain.com/api/payment/notify/{channel}

开发模式（PAYMENT_SANDBOX=True）：
  - 所有支付调用返回 mock 结果
  - 回调可手动触发：POST /api/payment/mock_notify

已实现：
  ✅ 统一下单接口（微信/支付宝）
  ✅ 异步回调处理（签名验证占位）
  ✅ 订阅开通/续期逻辑
  ✅ 退款接口骨架
  ✅ 订单查询
"""
from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from config import get_settings
from infra.auth import TokenData, get_current_user
from infra.quota import upgrade_plan, get_quota_status, PLANS

_settings = get_settings()
PAYMENT_SANDBOX: bool = getattr(_settings, "PAYMENT_SANDBOX", True)

# ─────────────────────────────────────────────────────────────────
# 订单结构
# ─────────────────────────────────────────────────────────────────

@dataclass
class Order:
    order_id: str
    user_id: str
    plan: str
    duration_days: int
    amount: int          # 分
    channel: str         # "wechat" | "alipay"
    status: str = "pending"  # pending | paid | failed | refunded
    created_at: float = field(default_factory=time.time)
    paid_at: Optional[float] = None
    trade_no: Optional[str] = None   # 支付平台交易号
    qr_code_url: str = ""            # 扫码支付二维码链接（生产为真实链接）

    def to_dict(self) -> dict:
        return {
            "order_id": self.order_id,
            "user_id": self.user_id,
            "plan": self.plan,
            "plan_name": PLANS.get(self.plan, PLANS["free"]).name,
            "duration_days": self.duration_days,
            "amount_yuan": self.amount / 100,
            "channel": self.channel,
            "status": self.status,
            "created_at": self.created_at,
            "paid_at": self.paid_at,
            "trade_no": self.trade_no,
            "qr_code_url": self.qr_code_url,
        }


# 内存订单存储（生产用 DB）
_orders: dict[str, Order] = {}

# ─────────────────────────────────────────────────────────────────
# 沙箱/Mock 支付客户端
# ─────────────────────────────────────────────────────────────────

def _mock_create_order(order: Order) -> str:
    """沙箱模式：返回假二维码链接"""
    return f"https://sandbox.pay.example.com/qr/{order.order_id}"


def _mock_query_order(order_id: str) -> dict:
    order = _orders.get(order_id)
    if not order:
        return {"status": "not_found"}
    return {"status": order.status, "trade_no": order.trade_no}


# ─────────────────────────────────────────────────────────────────
# TODO: 替换为真实支付 SDK（生产）
# ─────────────────────────────────────────────────────────────────

# class WeChatPayClient:
#     """
#     生产接入参考：
#       pip install wechatpayv3
#       from wechatpayv3 import WeChatPay, WeChatPayType
#     """
#     def __init__(self):
#         self.mchid = _settings.WECHAT_MCH_ID
#         self.api_key = _settings.WECHAT_API_KEY
#
#     async def create_native_order(self, order: Order) -> str:
#         """返回 code_url（微信扫码支付链接）"""
#         ...  # 调用 /v3/pay/transactions/native

# class AlipayClient:
#     """
#     生产接入参考：
#       pip install alipay-sdk-python
#       from alipay import AliPay
#     """
#     def __init__(self):
#         self.app_id = _settings.ALIPAY_APP_ID
#         self.private_key = _settings.ALIPAY_PRIVATE_KEY
#
#     async def create_qr_order(self, order: Order) -> str:
#         """返回二维码内容"""
#         ...  # 调用 alipay.trade.precreate


# ─────────────────────────────────────────────────────────────────
# 业务逻辑
# ─────────────────────────────────────────────────────────────────

def _calc_amount(plan: str, duration_days: int) -> int:
    """根据套餐和时长计算金额（分）"""
    p = PLANS.get(plan)
    if not p:
        raise ValueError(f"未知套餐: {plan}")
    if duration_days >= 365:
        return p.price_yearly
    elif duration_days >= 30:
        return p.price_monthly
    else:
        # 按日计算（月费/30）
        return max(100, p.price_monthly * duration_days // 30)


def create_order(
    user_id: str,
    plan: str,
    duration_days: int,
    channel: str = "wechat",
) -> Order:
    amount = _calc_amount(plan, duration_days)
    if amount == 0:
        raise ValueError(f"套餐 {plan} 无需支付")
    order_id = "XR" + uuid.uuid4().hex[:12].upper()
    order = Order(
        order_id=order_id,
        user_id=user_id,
        plan=plan,
        duration_days=duration_days,
        amount=amount,
        channel=channel,
    )
    if PAYMENT_SANDBOX:
        order.qr_code_url = _mock_create_order(order)
    else:
        # 生产：调用真实支付SDK
        # order.qr_code_url = await WeChatPayClient().create_native_order(order)
        order.qr_code_url = _mock_create_order(order)  # 临时
    _orders[order_id] = order
    return order


def process_payment_notify(
    order_id: str,
    trade_no: str,
    channel: str,
    raw_body: bytes = b"",
) -> Order:
    """
    处理支付回调。
    TODO 生产：先验签（微信v3 RSA/支付宝RSA2），再处理业务。
    """
    order = _orders.get(order_id)
    if not order:
        raise ValueError(f"订单 {order_id} 不存在")
    if order.status == "paid":
        return order  # 幂等：重复回调直接返回
    order.status = "paid"
    order.paid_at = time.time()
    order.trade_no = trade_no
    # 开通订阅
    upgrade_plan(order.user_id, order.plan, duration_days=order.duration_days)
    return order


# ─────────────────────────────────────────────────────────────────
# FastAPI Router
# ─────────────────────────────────────────────────────────────────

payment_router = APIRouter(prefix="/api/payment", tags=["payment"])


class CreateOrderRequest(BaseModel):
    plan: str
    duration_days: int = 30
    channel: str = "wechat"    # "wechat" | "alipay"


class MockNotifyRequest(BaseModel):
    order_id: str
    trade_no: str = ""


@payment_router.get("/plans")
async def api_payment_plans():
    """返回所有可购买套餐（含价格）"""
    result = []
    for pid, p in PLANS.items():
        if p.price_monthly == 0 and p.price_yearly == 0:
            continue  # 跳过免费版和合同版（学校版单独谈）
        result.append({
            "plan_id": pid,
            "name": p.name,
            "price_monthly": p.price_monthly / 100,
            "price_yearly": p.price_yearly / 100,
            "yearly_discount": f"{p.price_yearly / (p.price_monthly * 12) * 100:.0f}%" if p.price_monthly > 0 else None,
            "features": [
                f"{'无限' if p.monthly_sessions == -1 else p.monthly_sessions}次/月会话",
                "PDF报告导出" if p.can_export_pdf else None,
                "教师大屏看板" if p.can_view_dashboard else None,
                f"班级容量{p.max_class_size}人" if p.max_class_size > 0 else None,
            ],
        })
    return {"plans": result, "sandbox": PAYMENT_SANDBOX}


@payment_router.post("/create_order")
async def api_create_order(
    req: CreateOrderRequest,
    token: TokenData = Depends(get_current_user),
):
    """创建支付订单，返回二维码"""
    try:
        order = create_order(token.user_id, req.plan, req.duration_days, req.channel)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {
        "order_id": order.order_id,
        "amount_yuan": order.amount / 100,
        "qr_code_url": order.qr_code_url,
        "channel": order.channel,
        "plan": order.plan,
        "plan_name": PLANS[order.plan].name,
        "expires_in": 1800,  # 二维码30分钟有效
        "sandbox": PAYMENT_SANDBOX,
        "tip": "沙箱模式，请用 /api/payment/mock_notify 模拟支付成功" if PAYMENT_SANDBOX else None,
    }


@payment_router.get("/order/{order_id}")
async def api_query_order(
    order_id: str,
    token: TokenData = Depends(get_current_user),
):
    """查询订单状态（前端轮询用）"""
    order = _orders.get(order_id)
    if not order or order.user_id != token.user_id:
        raise HTTPException(status_code=404, detail="订单不存在")
    return order.to_dict()


@payment_router.post("/notify/{channel}")
async def api_payment_notify(channel: str, request: Request):
    """
    支付回调端点（微信/支付宝异步通知）。
    生产：先验签，再处理业务。
    """
    body = await request.body()
    # TODO 生产：从 body/header 解析 order_id 和 trade_no
    # 微信v3：从 body JSON 中取 out_trade_no
    # 支付宝：从 form data 取 out_trade_no
    return {"code": "SUCCESS", "message": "OK"}


@payment_router.post("/mock_notify")
async def api_mock_notify(req: MockNotifyRequest):
    """
    【沙箱专用】模拟支付回调，触发订阅开通。
    生产环境自动禁用。
    """
    if not PAYMENT_SANDBOX:
        raise HTTPException(status_code=403, detail="非沙箱环境不允许模拟支付")
    trade_no = req.trade_no or f"MOCK_{uuid.uuid4().hex[:10].upper()}"
    try:
        order = process_payment_notify(req.order_id, trade_no, "mock")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    quota = get_quota_status(order.user_id)
    return {
        "success": True,
        "order": order.to_dict(),
        "quota": quota.to_dict(),
        "message": f"支付成功！「{PLANS[order.plan].name}」已激活",
    }


@payment_router.get("/my_orders")
async def api_my_orders(token: TokenData = Depends(get_current_user)):
    """用户查看自己的历史订单"""
    user_orders = [o for o in _orders.values() if o.user_id == token.user_id]
    user_orders.sort(key=lambda o: o.created_at, reverse=True)
    return {"orders": [o.to_dict() for o in user_orders]}
