# infra/wechat_auth.py
"""
息壤微信登录模块
=====================================
支持两种微信登录方式：
  1. 微信小程序登录（wx.login code → openid）
  2. 微信网页授权登录（OAuth2 code → openid）

接入步骤：
  1. 登录微信开放平台 https://open.weixin.qq.com
  2. 创建小程序/网站应用，获取 AppID 和 AppSecret
  3. 在 .env 中设置：
       WECHAT_APP_ID=wx_your_appid
       WECHAT_APP_SECRET=your_appsecret
  4. 将 XIRANG_WECHAT_ENABLED=true

当前状态：代码完整，等待填入真实 AppID/Secret 即可启用
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from typing import Optional

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from config import get_settings
from infra.auth import create_access_token

_settings = get_settings()
_log = logging.getLogger(__name__)

# 从配置读取（在 .env 中设置）
WECHAT_APP_ID: str = getattr(_settings, "WECHAT_APP_ID", "")
WECHAT_APP_SECRET: str = getattr(_settings, "WECHAT_APP_SECRET", "")
WECHAT_ENABLED: bool = bool(WECHAT_APP_ID and WECHAT_APP_SECRET)

if not WECHAT_ENABLED:
    _log.warning(
        "⚠️ 微信登录未启用：请在 .env 中设置 WECHAT_APP_ID 和 WECHAT_APP_SECRET"
    )

# ─────────────────────────────────────────────────────────────────
# 微信 API 调用
# ─────────────────────────────────────────────────────────────────

MINIPROGRAM_CODE2SESSION = "https://api.weixin.qq.com/sns/jscode2session"
WEB_ACCESS_TOKEN_URL = "https://api.weixin.qq.com/sns/oauth2/access_token"
WEB_USERINFO_URL = "https://api.weixin.qq.com/sns/userinfo"


async def _miniprogram_code_to_openid(code: str) -> dict:
    """
    小程序 wx.login() 换取 openid + session_key
    参考文档：https://developers.weixin.qq.com/miniprogram/dev/api-backend/open-api/login/auth.code2Session.html
    """
    params = {
        "appid": WECHAT_APP_ID,
        "secret": WECHAT_APP_SECRET,
        "js_code": code,
        "grant_type": "authorization_code",
    }
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(MINIPROGRAM_CODE2SESSION, params=params)
        resp.raise_for_status()
        data = resp.json()

    if "errcode" in data and data["errcode"] != 0:
        errmsg = data.get("errmsg", "未知错误")
        errcode = data.get("errcode")
        _log.error(f"微信 code2session 失败: {errcode} {errmsg}")
        if errcode == 40029:
            raise ValueError("微信登录码无效或已过期，请重新登录")
        if errcode == 45011:
            raise ValueError("请求频率超限，请稍后重试")
        raise ValueError(f"微信登录失败: {errmsg}")

    return {
        "openid": data["openid"],
        "session_key": data.get("session_key", ""),
        "unionid": data.get("unionid", ""),  # 需开放平台绑定才有
    }


async def _web_code_to_openid(code: str) -> dict:
    """
    网页授权 code 换取 access_token + openid
    参考文档：https://developers.weixin.qq.com/doc/offiaccount/OA_Web_Apps/Wechat_webpage_authorization.html
    """
    params = {
        "appid": WECHAT_APP_ID,
        "secret": WECHAT_APP_SECRET,
        "code": code,
        "grant_type": "authorization_code",
    }
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(WEB_ACCESS_TOKEN_URL, params=params)
        resp.raise_for_status()
        data = resp.json()

    if "errcode" in data:
        raise ValueError(f"微信网页授权失败: {data.get('errmsg', '未知错误')}")

    return {
        "openid": data["openid"],
        "access_token": data.get("access_token", ""),
        "unionid": data.get("unionid", ""),
    }


# ─────────────────────────────────────────────────────────────────
# PostgreSQL 用户操作
# ─────────────────────────────────────────────────────────────────

_db_pool = None

async def _get_pool():
    global _db_pool
    if _db_pool is None:
        try:
            import asyncpg
            _db_pool = await asyncpg.create_pool(_settings.DB_URL, min_size=1, max_size=5)
        except Exception as e:
            _log.error(f"wechat_auth: DB 连接失败: {e}")
    return _db_pool


async def _get_or_create_wechat_user(
    openid: str,
    unionid: str = "",
    display_name: str = "",
    role: str = "user",
) -> dict:
    """通过 openid 查找或创建用户，返回用户信息"""
    pool = await _get_pool()

    if pool:
        async with pool.acquire() as conn:
            # 先用 openid 查
            row = await conn.fetchrow(
                "SELECT user_id, display_name, roles, plan FROM users WHERE wechat_openid = $1",
                openid
            )
            if row:
                return {
                    "user_id": row["user_id"],
                    "display_name": row["display_name"],
                    "roles": json.loads(row["roles"]) if row["roles"] else ["user"],
                    "plan": row["plan"],
                    "is_new": False,
                }

            # 新用户：创建
            user_id = f"wx_{openid[-8:]}_{uuid.uuid4().hex[:6]}"
            name = display_name or f"微信用户{openid[-4:]}"
            roles_json = json.dumps([role])
            await conn.execute(
                """
                INSERT INTO users (user_id, wechat_openid, display_name, roles, plan)
                VALUES ($1, $2, $3, $4::jsonb, 'free')
                ON CONFLICT (wechat_openid) DO UPDATE
                  SET last_login_at = NOW()
                """,
                user_id, openid, name, roles_json
            )
            await conn.execute(
                "INSERT INTO user_quota (user_id) VALUES ($1) ON CONFLICT DO NOTHING",
                user_id
            )
            return {
                "user_id": user_id,
                "display_name": name,
                "roles": [role],
                "plan": "free",
                "is_new": True,
            }
    else:
        # 内存回退（开发模式）
        _log.warning("wechat_auth: 使用内存模式，用户数据重启丢失")
        user_id = f"wx_{openid[-8:]}_{uuid.uuid4().hex[:6]}"
        return {
            "user_id": user_id,
            "display_name": display_name or f"微信用户{openid[-4:]}",
            "roles": [role],
            "plan": "free",
            "is_new": True,
        }


# ─────────────────────────────────────────────────────────────────
# 业务逻辑
# ─────────────────────────────────────────────────────────────────

async def wechat_miniprogram_login(code: str, role: str = "user") -> dict:
    """小程序登录主流程"""
    if not WECHAT_ENABLED:
        raise ValueError(
            "微信登录未配置，请在 .env 中设置 WECHAT_APP_ID 和 WECHAT_APP_SECRET"
        )
    wx_data = await _miniprogram_code_to_openid(code)
    openid = wx_data["openid"]
    unionid = wx_data.get("unionid", "")

    user = await _get_or_create_wechat_user(openid, unionid, role=role)

    token = create_access_token(
        user_id=user["user_id"],
        roles=user["roles"],
    )
    return {
        "access_token": token,
        "token_type": "bearer",
        "user_id": user["user_id"],
        "display_name": user["display_name"],
        "roles": user["roles"],
        "is_new_user": user["is_new"],
    }


async def wechat_web_login(code: str, role: str = "user") -> dict:
    """网页授权登录主流程"""
    if not WECHAT_ENABLED:
        raise ValueError("微信登录未配置")
    wx_data = await _web_code_to_openid(code)
    openid = wx_data["openid"]
    unionid = wx_data.get("unionid", "")

    user = await _get_or_create_wechat_user(openid, unionid, role=role)

    token = create_access_token(user_id=user["user_id"], roles=user["roles"])
    return {
        "access_token": token,
        "token_type": "bearer",
        "user_id": user["user_id"],
        "display_name": user["display_name"],
        "roles": user["roles"],
        "is_new_user": user["is_new"],
    }


# ─────────────────────────────────────────────────────────────────
# FastAPI Router（挂载到 sms_auth 的 sms_router，共享 /api/auth 前缀）
# ─────────────────────────────────────────────────────────────────

wechat_router = APIRouter(prefix="/api/auth", tags=["auth-wechat"])


class WechatMiniLoginRequest(BaseModel):
    code: str                  # wx.login() 返回的临时 code
    role: str = "user"         # user | teacher


class WechatWebLoginRequest(BaseModel):
    code: str                  # 网页授权 code
    role: str = "user"


@wechat_router.post("/wechat_miniprogram")
async def api_wechat_miniprogram_login(req: WechatMiniLoginRequest):
    """
    微信小程序登录
    前端调用 wx.login() 获取 code，传给此接口换取 JWT
    """
    try:
        result = await wechat_miniprogram_login(req.code, req.role)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        _log.error(f"微信小程序登录异常: {e}")
        raise HTTPException(status_code=500, detail="微信登录服务暂时不可用")
    return result


@wechat_router.post("/wechat_web")
async def api_wechat_web_login(req: WechatWebLoginRequest):
    """
    微信网页授权登录
    用户点击微信授权后，将 code 传给此接口换取 JWT
    """
    try:
        result = await wechat_web_login(req.code, req.role)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        _log.error(f"微信网页登录异常: {e}")
        raise HTTPException(status_code=500, detail="微信登录服务暂时不可用")
    return result


@wechat_router.get("/wechat_status")
async def api_wechat_status():
    """查询微信登录配置状态"""
    return {
        "enabled": WECHAT_ENABLED,
        "app_id": WECHAT_APP_ID[:8] + "***" if WECHAT_APP_ID else "",
        "message": "微信登录已启用" if WECHAT_ENABLED else "请配置 WECHAT_APP_ID 和 WECHAT_APP_SECRET",
    }
