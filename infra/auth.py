# infra/auth.py
"""
息壤多租户鉴权模块
支持两种认证方式：
  1. API Key（Header: X-API-Key）——适合服务端集成
  2. Bearer JWT（Header: Authorization: Bearer <token>）——适合前端用户

当 AUTH_ENABLED=False（开发模式）时所有请求直通。

多租户设计：
  每个 tenant 有独立的 user_id 命名空间、速率限制配额、
  可访问的 ERA 白名单（可选）。
"""
from __future__ import annotations
import time
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import Depends, Header, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from jose import JWTError, jwt
from passlib.context import CryptContext

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import get_settings

_settings = get_settings()
_pwd_ctx  = CryptContext(schemes=["bcrypt"], deprecated="auto")
_bearer   = HTTPBearer(auto_error=False)


# ── Token 结构 ────────────────────────────────────────────────
class TokenData:
    def __init__(self, user_id: str, tenant_id: str = "default",
                 roles: list[str] | None = None, exp: int = 0):
        self.user_id   = user_id
        self.tenant_id = tenant_id
        self.roles     = roles or ["user"]
        self.exp       = exp

    @property
    def is_teacher(self) -> bool:
        return "teacher" in self.roles

    @property
    def is_admin(self) -> bool:
        return "admin" in self.roles


def create_access_token(
    user_id: str,
    tenant_id: str = "default",
    roles: list[str] | None = None,
    expires_delta: timedelta | None = None,
) -> str:
    """签发 JWT"""
    expire = datetime.now(timezone.utc) + (
        expires_delta or timedelta(minutes=_settings.JWT_EXPIRE_MINUTES)
    )
    payload = {
        "sub":    user_id,
        "tenant": tenant_id,
        "roles":  roles or ["user"],
        "exp":    expire,
        "iat":    datetime.now(timezone.utc),
    }
    return jwt.encode(payload, _settings.JWT_SECRET,
                      algorithm=_settings.JWT_ALGORITHM)


def decode_token(token: str) -> TokenData:
    """验证并解码 JWT，失败抛 HTTPException 401"""
    try:
        payload = jwt.decode(
            token,
            _settings.JWT_SECRET,
            algorithms=[_settings.JWT_ALGORITHM],
        )
        return TokenData(
            user_id=payload["sub"],
            tenant_id=payload.get("tenant", "default"),
            roles=payload.get("roles", ["user"]),
            exp=int(payload.get("exp", 0)),
        )
    except JWTError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"无效的认证凭据: {e}",
            headers={"WWW-Authenticate": "Bearer"},
        )


def verify_api_key(key: str) -> Optional[TokenData]:
    """验证静态 API Key，返回对应的 TokenData 或 None"""
    if key in _settings.STATIC_API_KEYS:
        # API Key 默认给 admin 权限
        return TokenData(user_id=f"apikey_{key[:8]}", tenant_id="default",
                         roles=["admin"])
    return None


# ── FastAPI 依赖注入 ──────────────────────────────────────────
async def get_current_user(
    request: Request,
    credentials: HTTPAuthorizationCredentials = Depends(_bearer),
    x_api_key: str = Header(default=""),
) -> TokenData:
    """
    FastAPI 依赖：提取并验证当前请求的认证信息。
    AUTH_ENABLED=False 时返回匿名用户，允许所有请求通过。
    """
    if not _settings.AUTH_ENABLED:
        # 开发模式：从 query param 或 header 读取 user_id，不验证
        uid = request.query_params.get("user_id", "anonymous")
        return TokenData(user_id=uid, roles=["user", "admin"])

    # 1. API Key 优先
    if x_api_key:
        token_data = verify_api_key(x_api_key)
        if token_data:
            return token_data
        raise HTTPException(status_code=401, detail="无效的 API Key")

    # 2. Bearer JWT
    if credentials:
        return decode_token(credentials.credentials)

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="请提供认证凭据（X-API-Key 或 Bearer Token）",
        headers={"WWW-Authenticate": "Bearer"},
    )


def require_teacher(token: TokenData = Depends(get_current_user)) -> TokenData:
    """依赖：要求 teacher 或 admin 角色"""
    if not (token.is_teacher or token.is_admin):
        raise HTTPException(status_code=403, detail="仅教师或管理员可访问")
    return token


def require_admin(token: TokenData = Depends(get_current_user)) -> TokenData:
    """依赖：要求 admin 角色"""
    if not token.is_admin:
        raise HTTPException(status_code=403, detail="仅管理员可访问")
    return token


# ── 速率限制（内存版，生产用 Redis 版）────────────────────────
_rate_buckets: dict[str, list[float]] = {}

def check_rate_limit(user_id: str, rpm: int | None = None) -> bool:
    """
    令牌桶限流（内存版）。
    返回 True=允许, False=超限。
    生产环境请替换为 infra/cache.py 中的 Redis 原子限流。
    """
    if not _settings.RATE_LIMIT_ENABLED:
        return True
    limit = rpm or _settings.RATE_LIMIT_RPM
    now = time.time()
    window = 60.0
    bucket = _rate_buckets.setdefault(user_id, [])
    # 清除 60s 外的记录
    _rate_buckets[user_id] = [t for t in bucket if now - t < window]
    if len(_rate_buckets[user_id]) >= limit:
        return False
    _rate_buckets[user_id].append(now)
    return True


async def rate_limit_dep(
    request: Request,
    token: TokenData = Depends(get_current_user),
) -> TokenData:
    """FastAPI 依赖：先鉴权再限流"""
    if not check_rate_limit(token.user_id):
        raise HTTPException(
            status_code=429,
            detail=f"请求过于频繁，每分钟限 {_settings.RATE_LIMIT_RPM} 次",
            headers={"Retry-After": "60"},
        )
    return token


# ── Token 签发端点（注册到 server.py）────────────────────────
from fastapi import APIRouter
from pydantic import BaseModel

auth_router = APIRouter(prefix="/api/auth", tags=["auth"])

class TokenRequest(BaseModel):
    user_id: str
    password: str = ""    # 演示用；生产接 DB
    tenant_id: str = "default"
    roles: list[str] = ["user"]

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int   # seconds
    user_id: str
    roles: list[str]

@auth_router.post("/token", response_model=TokenResponse)
async def issue_token(req: TokenRequest):
    """
    签发访问令牌（演示版：无密码验证）。
    生产环境需接入用户数据库和密码哈希验证。
    """
    if _settings.AUTH_ENABLED and req.password != "demo":
        raise HTTPException(status_code=401, detail="密码错误")
    token = create_access_token(
        user_id=req.user_id,
        tenant_id=req.tenant_id,
        roles=req.roles,
    )
    return TokenResponse(
        access_token=token,
        expires_in=_settings.JWT_EXPIRE_MINUTES * 60,
        user_id=req.user_id,
        roles=req.roles,
    )

@auth_router.get("/me")
async def get_me(token: TokenData = Depends(get_current_user)):
    """返回当前用户信息"""
    return {
        "user_id":   token.user_id,
        "tenant_id": token.tenant_id,
        "roles":     token.roles,
    }


if __name__ == "__main__":
    # 自测
    t = create_access_token("user_123", tenant_id="school_a", roles=["teacher"])
    print("Token:", t[:40] + "...")
    td = decode_token(t)
    print(f"Decoded: user={td.user_id}, tenant={td.tenant_id}, roles={td.roles}")
    print(f"is_teacher={td.is_teacher}, is_admin={td.is_admin}")
