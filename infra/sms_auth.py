# infra/sms_auth.py
"""
息壤手机号登录模块（P0 - 已修复：用户数据持久化至 PostgreSQL）
=====================================
流程：
  1. POST /api/auth/send_sms   { phone }
     → 生成6位验证码，发SMS（沙箱直接返回code）
  2. POST /api/auth/verify_sms { phone, code, role? }
     → 验证通过 → 返回 JWT + 用户信息（自动注册新用户）

存储策略：
  - 验证码：Redis（USE_REDIS=True）或内存（开发模式）
  - 用户数据：PostgreSQL（USE_POSTGRES=True）或内存（开发模式）
"""
from __future__ import annotations

import json
import logging
import random
import time
from dataclasses import dataclass, field
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from config import get_settings
from infra.auth import create_access_token, TokenData

_settings = get_settings()
_log = logging.getLogger(__name__)

SMS_SANDBOX: bool = getattr(_settings, "SMS_SANDBOX", True)
USE_POSTGRES: bool = getattr(_settings, "USE_POSTGRES", False)
USE_REDIS: bool = getattr(_settings, "USE_REDIS", False)

# ─────────────────────────────────────────────────────────────────
# 验证码存储
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
        return self.attempts >= 5


# 内存回退（开发模式）
_sms_store: dict[str, SmsRecord] = {}
_send_throttle: dict[str, float] = {}
# 内存用户表（开发模式回退）
_users_mem: dict[str, dict] = {}


# ─────────────────────────────────────────────────────────────────
# PostgreSQL 持久化层
# ─────────────────────────────────────────────────────────────────

_db_pool = None

async def _get_pool():
    """惰性初始化数据库连接池"""
    global _db_pool
    if _db_pool is None and USE_POSTGRES:
        try:
            import asyncpg
            _db_pool = await asyncpg.create_pool(_settings.DB_URL, min_size=2, max_size=10)
            _log.info("✅ sms_auth: PostgreSQL 连接池已建立")
        except Exception as e:
            _log.error(f"❌ sms_auth: PostgreSQL 连接失败，回退内存模式: {e}")
    return _db_pool


async def _db_get_user(phone: str) -> Optional[dict]:
    """从 PostgreSQL 查询用户"""
    pool = await _get_pool()
    if not pool:
        return _users_mem.get(phone)
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT user_id, phone, display_name, roles, plan FROM users WHERE phone = $1", phone
        )
        if row:
            return {
                "user_id": row["user_id"],
                "phone": row["phone"],
                "display_name": row["display_name"],
                "role": json.loads(row["roles"])[0] if row["roles"] else "user",
                "roles": json.loads(row["roles"]) if row["roles"] else ["user"],
                "plan": row["plan"],
            }
    return None


async def _db_create_user(user: dict) -> dict:
    """在 PostgreSQL 中创建用户"""
    pool = await _get_pool()
    if not pool:
        _users_mem[user["phone"]] = user
        return user
    async with pool.acquire() as conn:
        roles_json = json.dumps(user.get("roles", ["user"]))
        await conn.execute(
            """
            INSERT INTO users (user_id, phone, display_name, roles, plan)
            VALUES ($1, $2, $3, $4::jsonb, $5)
            ON CONFLICT (phone) DO UPDATE
              SET display_name = EXCLUDED.display_name,
                  last_login_at = NOW()
            """,
            user["user_id"], user["phone"], user["display_name"], roles_json, "free"
        )
        # 初始化配额记录
        await conn.execute(
            """
            INSERT INTO user_quota (user_id)
            VALUES ($1)
            ON CONFLICT (user_id) DO NOTHING
            """,
            user["user_id"]
        )
    return user


async def _db_update_display_name(phone: str, display_name: str):
    """更新用户昵称"""
    pool = await _get_pool()
    if not pool:
        if phone in _users_mem:
            _users_mem[phone]["display_name"] = display_name
        return
    async with pool.acquire() as conn:
        await conn.execute(
            "UPDATE users SET display_name = $1 WHERE phone = $2",
            display_name, phone
        )


# ─────────────────────────────────────────────────────────────────
# Redis 验证码存储
# ─────────────────────────────────────────────────────────────────

_redis_client = None

async def _get_redis():
    global _redis_client
    if _redis_client is None and USE_REDIS:
        try:
            import redis.asyncio as aioredis
            _redis_client = aioredis.from_url(_settings.REDIS_URL, decode_responses=True)
            await _redis_client.ping()
            _log.info("✅ sms_auth: Redis 连接成功")
        except Exception as e:
            _log.error(f"❌ sms_auth: Redis 连接失败，回退内存模式: {e}")
    return _redis_client


async def _store_sms_code(phone: str, record: SmsRecord):
    r = await _get_redis()
    if r:
        data = json.dumps({
            "code": record.code,
            "created_at": record.created_at,
            "attempts": record.attempts,
            "verified": record.verified,
        })
        await r.setex(f"sms:{phone}", 300, data)
    else:
        _sms_store[phone] = record


async def _get_sms_record(phone: str) -> Optional[SmsRecord]:
    r = await _get_redis()
    if r:
        raw = await r.get(f"sms:{phone}")
        if not raw:
            return None
        data = json.loads(raw)
        return SmsRecord(
            phone=phone,
            code=data["code"],
            created_at=data["created_at"],
            attempts=data["attempts"],
            verified=data["verified"],
        )
    return _sms_store.get(phone)


async def _delete_sms_record(phone: str):
    r = await _get_redis()
    if r:
        await r.delete(f"sms:{phone}")
    else:
        _sms_store.pop(phone, None)


async def _update_sms_attempts(phone: str, record: SmsRecord):
    r = await _get_redis()
    if r:
        ttl = await r.ttl(f"sms:{phone}")
        if ttl > 0:
            data = json.dumps({
                "code": record.code,
                "created_at": record.created_at,
                "attempts": record.attempts,
                "verified": record.verified,
            })
            await r.setex(f"sms:{phone}", ttl, data)
    else:
        _sms_store[phone] = record


async def _check_throttle(phone: str) -> float:
    """返回还需等待的秒数（0=可以发送）"""
    r = await _get_redis()
    if r:
        ttl = await r.ttl(f"throttle:{phone}")
        return max(0, ttl)
    last_sent = _send_throttle.get(phone, 0)
    wait = 60 - (time.time() - last_sent)
    return max(0, wait)


async def _set_throttle(phone: str):
    r = await _get_redis()
    if r:
        await r.setex(f"throttle:{phone}", 60, "1")
    else:
        _send_throttle[phone] = time.time()


# ─────────────────────────────────────────────────────────────────
# 核心逻辑
# ─────────────────────────────────────────────────────────────────

def _gen_code() -> str:
    return str(random.randint(100000, 999999))


def _phone_to_user_id(phone: str) -> str:
    return f"u_{phone[-4:]}_" + hex(abs(hash(phone)) % 0xFFFF)[2:]


async def _send_sms_real(phone: str, code: str) -> bool:
    """
    接入阿里云短信服务。
    配置：ALIYUN_ACCESS_KEY_ID / ALIYUN_ACCESS_KEY_SECRET / XIRANG_SMS_SIGN_NAME / XIRANG_SMS_TEMPLATE_CODE
    """
    try:
        from aliyunsdkcore.client import AcsClient
        from aliyunsdkcore.request import CommonRequest
        client = AcsClient(
            _settings.ALIYUN_ACCESS_KEY_ID,
            _settings.ALIYUN_ACCESS_KEY_SECRET,
            "cn-hangzhou",
        )
        request = CommonRequest()
        request.set_accept_format("json")
        request.set_domain("dysmsapi.aliyuncs.com")
        request.set_method("POST")
        request.set_protocol_type("https")
        request.set_version("2017-05-25")
        request.set_action_name("SendSms")
        request.add_query_param("RegionId", "cn-hangzhou")
        request.add_query_param("PhoneNumbers", phone)
        request.add_query_param("SignName", _settings.SMS_SIGN_NAME)
        request.add_query_param("TemplateCode", _settings.SMS_TEMPLATE_CODE)
        request.add_query_param("TemplateParam", json.dumps({"code": code}))
        response = json.loads(client.do_action_with_exception(request))
        return response.get("Code") == "OK"
    except ImportError:
        _log.error("阿里云短信 SDK 未安装: pip install aliyun-python-sdk-core aliyun-python-sdk-dysmsapi")
        return False
    except Exception as e:
        _log.error(f"短信发送失败: {e}")
        return False


async def send_sms_code(phone: str) -> dict:
    """发送验证码"""
    if not phone.isdigit() or len(phone) != 11 or not phone.startswith("1"):
        raise ValueError("请输入正确的11位手机号")

    wait = await _check_throttle(phone)
    if wait > 0:
        raise ValueError(f"发送太频繁，请{int(wait)+1}秒后重试")

    code = _gen_code()
    record = SmsRecord(phone=phone, code=code)
    await _store_sms_code(phone, record)
    await _set_throttle(phone)

    if SMS_SANDBOX:
        return {"sent": True, "sandbox_code": code, "tip": "沙箱模式，验证码已在返回值中"}
    else:
        success = await _send_sms_real(phone, code)
        if not success:
            raise RuntimeError("短信发送失败，请稍后重试")
        return {"sent": True}


async def verify_sms_code(phone: str, code: str, role: str = "user") -> dict:
    """验证验证码，返回 JWT token（异步，支持DB查询）"""
    record = await _get_sms_record(phone)
    if not record:
        raise ValueError("请先获取验证码")
    if record.is_expired:
        await _delete_sms_record(phone)
        raise ValueError("验证码已过期，请重新获取")
    if record.is_locked:
        raise ValueError("验证次数过多，请重新获取验证码")

    record.attempts += 1
    if record.code != code.strip():
        await _update_sms_attempts(phone, record)
        raise ValueError(f"验证码错误（还可尝试{5 - record.attempts}次）")

    record.verified = True
    await _delete_sms_record(phone)

    # 查找或创建用户（持久化到 PostgreSQL）
    user = await _db_get_user(phone)
    is_new = user is None
    if is_new:
        user_id = _phone_to_user_id(phone)
        user = {
            "user_id": user_id,
            "phone": phone,
            "display_name": f"用户{phone[-4:]}",
            "role": role,
            "roles": [role],
            "plan": "free",
        }
        await _db_create_user(user)
    else:
        is_new = False

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
        "is_new_user": is_new,
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
    role: str = "user"
    display_name: str = ""


class WechatLoginRequest(BaseModel):
    code: str
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
        result = await verify_sms_code(req.phone, req.code, req.role)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # 首次注册更新昵称
    if req.display_name and result.get("is_new_user"):
        await _db_update_display_name(req.phone, req.display_name)
        result["display_name"] = req.display_name

    return result


@sms_router.post("/wechat_login")
async def api_wechat_login(req: WechatLoginRequest):
    """
    微信登录（预留接口）
    TODO 生产：
      1. 用 code 换 openid：GET https://api.weixin.qq.com/sns/jscode2session
      2. 查 DB：openid → user_id（不存在则新建）
      3. 签发 JWT
    """
    raise HTTPException(
        status_code=501,
        detail={
            "message": "微信登录接口待接入，请先完成微信开放平台认证",
            "docs": "https://developers.weixin.qq.com/miniprogram/dev/api/open-api/login/wx.login.html",
        },
    )
