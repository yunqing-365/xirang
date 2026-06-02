# infra/sms_auth.py
"""
息壤手机号登录模块（P0）
=====================================
流程：
  1. POST /api/auth/send_sms   { phone }
     → 生成6位验证码，发SMS（沙箱直接返回code）
  2. POST /api/auth/verify_sms { phone, code, role? }
     → 验证通过 → 返回 JWT + 用户信息（自动注册新用户）

生产接入（任选）：
  - 阿里云短信：aliyun-python-sdk-core + aliyun-python-sdk-dysmsapi
  - 腾讯云短信：tencentcloud-sdk-python
  - 替换 _send_sms_real() 即可

微信登录（预留接口，需微信开放平台 AppID/Secret）：
  POST /api/auth/wechat_login  { code }
  → 换取 openid → 查/建用户 → 返回 JWT
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from config import get_settings
from infra.auth import create_access_token, TokenData

_settings = get_settings()
SMS_SANDBOX: bool = getattr(_settings, "SMS_SANDBOX", True)

# ─────────────────────────────────────────────────────────────────
# 验证码存储（生产用 Redis TTL）
# ─────────────────────────────────────────────────────────────────

@dataclass
class SmsRecord:
    phone: str
    code: str
    created_at: float = field(default_factory=time.time)
    attempts: int = 0
    verified: bool = False

    @property
    def is_expired(self) -> bool:
        return time.time() - self.created_at > 300   # 5分钟过期

    @property
    def is_locked(self) -> bool:
        return self.attempts >= 5   # 5次错误锁定


_sms_store: dict[str, SmsRecord] = {}   # phone → SmsRecord

# 用户数据库（手机号 → 用户信息）生产替换为 PostgreSQL
_users: dict[str, dict] = {}   # phone → {user_id, display_name, role, ...}

# 发送频率限制：同一手机号60秒内只能发一次
_send_throttle: dict[str, float] = {}


# ─────────────────────────────────────────────────────────────────
# 核心逻辑
# ─────────────────────────────────────────────────────────────────

def _gen_code() -> str:
    return str(random.randint(100000, 999999))


def _phone_to_user_id(phone: str) -> str:
    return f"u_{phone[-4:]}_" + hex(abs(hash(phone)) % 0xFFFF)[2:]


async def _send_sms_real(phone: str, code: str) -> bool:
    """
    TODO：接入真实短信服务商
    阿里云示例：
        client = AcsClient(access_key_id, access_key_secret, 'cn-hangzhou')
        request = SendSmsRequest.SendSmsRequest()
        request.set_PhoneNumbers(phone)
        request.set_SignName('息壤历史')
        request.set_TemplateCode('SMS_xxxxxxxx')
        request.set_TemplateParam(json.dumps({'code': code}))
        response = client.do_action_with_exception(request)
    """
    return True   # 占位


async def send_sms_code(phone: str) -> dict:
    """发送验证码，返回结果（沙箱直接返回明文code）"""
    # 手机号格式简单校验
    if not phone.isdigit() or len(phone) != 11 or not phone.startswith("1"):
        raise ValueError("请输入正确的11位手机号")

    # 60秒频率限制
    last_sent = _send_throttle.get(phone, 0)
    wait = 60 - (time.time() - last_sent)
    if wait > 0:
        raise ValueError(f"发送太频繁，请{int(wait)+1}秒后重试")

    code = _gen_code()
    _sms_store[phone] = SmsRecord(phone=phone, code=code)
    _send_throttle[phone] = time.time()

    if SMS_SANDBOX:
        # 沙箱：直接返回验证码（前端展示用于测试）
        return {"sent": True, "sandbox_code": code, "tip": "沙箱模式，验证码已在返回值中"}
    else:
        success = await _send_sms_real(phone, code)
        if not success:
            raise RuntimeError("短信发送失败，请稍后重试")
        return {"sent": True}


def verify_sms_code(phone: str, code: str, role: str = "user") -> dict:
    """验证验证码，返回 JWT token"""
    record = _sms_store.get(phone)
    if not record:
        raise ValueError("请先获取验证码")
    if record.is_expired:
        del _sms_store[phone]
        raise ValueError("验证码已过期，请重新获取")
    if record.is_locked:
        raise ValueError("验证次数过多，请重新获取验证码")

    record.attempts += 1
    if record.code != code.strip():
        raise ValueError(f"验证码错误（还可尝试{5 - record.attempts}次）")

    # 验证通过
    record.verified = True

    # 查找或创建用户
    user = _users.get(phone)
    if not user:
        user_id = _phone_to_user_id(phone)
        user = {
            "user_id": user_id,
            "phone": phone,
            "display_name": f"用户{phone[-4:]}",
            "role": role,
            "roles": [role],
            "created_at": time.time(),
            "is_new": True,
        }
        _users[phone] = user
    else:
        user["is_new"] = False

    # 签发 JWT
    token = create_access_token(
        user_id=user["user_id"],
        roles=user.get("roles", ["user"]),
    )
    return {
        "access_token": token,
        "token_type": "bearer",
        "user_id": user["user_id"],
        "display_name": user["display_name"],
        "roles": user.get("roles", ["user"]),
        "is_new_user": user.get("is_new", False),
    }


# ─────────────────────────────────────────────────────────────────
# FastAPI Router
# ─────────────────────────────────────────────────────────────────

sms_router = APIRouter(prefix="/api/auth", tags=["auth-sms"])


class SendSmsRequest(BaseModel):
    phone: str


class VerifySmsRequest(BaseModel):
    phone: str
    code: str
    role: str = "user"       # "user" | "teacher"（教师注册时传 teacher）
    display_name: str = ""   # 可选，首次注册设置昵称


class WechatLoginRequest(BaseModel):
    code: str                # 微信 wx.login() 返回的临时code
    role: str = "user"


@sms_router.post("/send_sms")
async def api_send_sms(req: SendSmsRequest):
    """获取手机验证码"""
    try:
        result = await send_sms_code(req.phone)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return result


@sms_router.post("/verify_sms")
async def api_verify_sms(req: VerifySmsRequest):
    """验证码登录/注册"""
    try:
        result = verify_sms_code(req.phone, req.code, req.role)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # 首次注册更新昵称
    if req.display_name and result.get("is_new_user"):
        user = _users.get(req.phone)
        if user:
            user["display_name"] = req.display_name

    return result


@sms_router.post("/wechat_login")
async def api_wechat_login(req: WechatLoginRequest):
    """
    微信登录（预留接口）
    TODO 生产：
      1. 用 code 换 openid：
         GET https://api.weixin.qq.com/sns/jscode2session
             ?appid=APPID&secret=SECRET&js_code=CODE&grant_type=authorization_code
      2. 查 DB：openid → user_id（不存在则新建）
      3. 签发 JWT
    """
    raise HTTPException(
        status_code=501,
        detail={
            "message": "微信登录接口待接入，请先完成微信开放平台认证",
            "docs": "https://developers.weixin.qq.com/miniprogram/dev/api/open-api/login/wx.login.html",
            "todo": "替换本函数中的 TODO 部分后即可启用",
        },
    )
