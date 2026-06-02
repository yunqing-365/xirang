# infra/invite.py
"""
息壤班级码 / 邀请码系统（P0 - 已升级：持久化到 PostgreSQL）
=====================================
两种码：
  1. 班级码（class_code）：教师创建，6位大写字母，学生扫码加入班级
  2. 邀请码（invite_code）：管理员生成，8位，发给教师用于激活专业版

存储策略：
  - USE_POSTGRES=True：读写 classrooms / classroom_members / invite_codes 表
  - 内存回退：开发/单机模式
"""
from __future__ import annotations

import logging
import random
import string
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from infra.auth import TokenData, get_current_user, require_teacher, require_admin
from infra.quota import upgrade_plan, get_quota_status

_log = logging.getLogger(__name__)

from config import get_settings
_settings = get_settings()
USE_POSTGRES: bool = getattr(_settings, "USE_POSTGRES", False)

# ─────────────────────────────────────────────────────────────────
# 数据结构
# ─────────────────────────────────────────────────────────────────

@dataclass
class ClassRoom:
    class_code: str
    room_id: str
    teacher_id: str
    class_name: str
    era: str
    created_at: float = field(default_factory=time.time)
    expires_at: float = 0.0
    members: dict[str, dict] = field(default_factory=dict)
    is_active: bool = True

    @property
    def is_expired(self) -> bool:
        if self.expires_at == 0:
            return False
        return time.time() > self.expires_at

    def to_dict(self) -> dict:
        return {
            "class_code": self.class_code,
            "room_id": self.room_id,
            "teacher_id": self.teacher_id,
            "class_name": self.class_name,
            "era": self.era,
            "member_count": len(self.members),
            "members": list(self.members.values()),
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "is_active": self.is_active and not self.is_expired,
        }


@dataclass
class InviteCode:
    code: str
    plan: str
    duration_days: int
    created_by: str
    created_at: float = field(default_factory=time.time)
    expires_at: float = 0.0
    used_by: Optional[str] = None
    used_at: Optional[float] = None

    @property
    def is_used(self) -> bool:
        return self.used_by is not None

    @property
    def is_expired(self) -> bool:
        if self.expires_at == 0:
            return False
        return time.time() > self.expires_at

    @property
    def is_valid(self) -> bool:
        return not self.is_used and not self.is_expired

    def to_dict(self) -> dict:
        return {
            "code": self.code,
            "plan": self.plan,
            "duration_days": self.duration_days,
            "is_valid": self.is_valid,
            "used_by": self.used_by,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
        }


# ─────────────────────────────────────────────────────────────────
# 内存回退（开发模式）
# ─────────────────────────────────────────────────────────────────

_classrooms: dict[str, ClassRoom] = {}
_classrooms_by_teacher: dict[str, list[str]] = {}
_invite_codes: dict[str, InviteCode] = {}

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
            _log.info("✅ invite: PostgreSQL 连接池已建立")
        except Exception as e:
            _log.error(f"❌ invite: PostgreSQL 连接失败，回退内存: {e}")
    return _db_pool


# ── 班级 DB 操作 ──────────────────────────────────────────────────

async def _db_create_classroom(classroom: ClassRoom):
    pool = await _get_pool()
    if not pool:
        _classrooms[classroom.class_code] = classroom
        _classrooms_by_teacher.setdefault(classroom.teacher_id, []).append(classroom.class_code)
        return
    from datetime import datetime, timezone
    expires_dt = (
        datetime.fromtimestamp(classroom.expires_at, tz=timezone.utc)
        if classroom.expires_at > 0 else None
    )
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO classrooms (teacher_id, class_code, class_name, era, expires_at, is_active)
            VALUES ($1, $2, $3, $4, $5, TRUE)
            ON CONFLICT (class_code) DO NOTHING
            """,
            classroom.teacher_id, classroom.class_code,
            classroom.class_name, classroom.era, expires_dt
        )


async def _db_get_classroom(class_code: str) -> Optional[ClassRoom]:
    pool = await _get_pool()
    if not pool:
        return _classrooms.get(class_code)
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT * FROM classrooms WHERE class_code = $1", class_code
        )
        if not row:
            return None
        # 加载成员
        members_rows = await conn.fetch(
            "SELECT user_id, display_name, role, EXTRACT(EPOCH FROM joined_at) as joined_ts "
            "FROM classroom_members WHERE class_code = $1", class_code
        )
        members = {
            r["user_id"]: {
                "user_id": r["user_id"],
                "display_name": r["display_name"],
                "role": r["role"],
                "joined_at": float(r["joined_ts"] or 0),
            }
            for r in members_rows
        }
        expires_ts = float(row["expires_at"].timestamp()) if row.get("expires_at") else 0.0
        created_ts = float(row["created_at"].timestamp()) if row.get("created_at") else time.time()
        return ClassRoom(
            class_code=row["class_code"],
            room_id=f"room_{row['class_code'].lower()}",
            teacher_id=row["teacher_id"],
            class_name=row["class_name"] or "",
            era=row["era"] or "北宋·熙宁变法",
            created_at=created_ts,
            expires_at=expires_ts,
            members=members,
            is_active=row["is_active"],
        )


async def _db_join_classroom(class_code: str, user_id: str, display_name: str):
    pool = await _get_pool()
    if not pool:
        return
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO classroom_members (class_code, user_id, display_name, role)
            VALUES ($1, $2, $3, 'student')
            ON CONFLICT (class_code, user_id) DO UPDATE
              SET display_name = EXCLUDED.display_name
            """,
            class_code, user_id, display_name or user_id
        )


async def _db_teacher_classrooms(teacher_id: str) -> list[ClassRoom]:
    pool = await _get_pool()
    if not pool:
        codes = _classrooms_by_teacher.get(teacher_id, [])
        return [_classrooms[c] for c in codes if c in _classrooms]
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT class_code FROM classrooms WHERE teacher_id = $1 ORDER BY created_at DESC",
            teacher_id
        )
        result = []
        for r in rows:
            c = await _db_get_classroom(r["class_code"])
            if c:
                result.append(c)
        return result


async def _db_close_classroom(class_code: str, teacher_id: str):
    pool = await _get_pool()
    if not pool:
        c = _classrooms.get(class_code)
        if c:
            c.is_active = False
        return
    async with pool.acquire() as conn:
        result = await conn.execute(
            "UPDATE classrooms SET is_active = FALSE WHERE class_code = $1 AND teacher_id = $2",
            class_code, teacher_id
        )
        if result == "UPDATE 0":
            raise PermissionError("只有班级创建者可以关闭班级")


# ── 邀请码 DB 操作 ──────────────────────────────────────────────

async def _db_save_invite(invite: InviteCode):
    pool = await _get_pool()
    if not pool:
        _invite_codes[invite.code] = invite
        return
    from datetime import datetime, timezone
    expires_dt = (
        datetime.fromtimestamp(invite.expires_at, tz=timezone.utc)
        if invite.expires_at > 0 else None
    )
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO invite_codes (code, plan, duration_days, created_by, expires_at)
            VALUES ($1, $2, $3, $4, $5)
            ON CONFLICT (code) DO NOTHING
            """,
            invite.code, invite.plan, invite.duration_days, invite.created_by, expires_dt
        )


async def _db_get_invite(code: str) -> Optional[InviteCode]:
    pool = await _get_pool()
    if not pool:
        return _invite_codes.get(code)
    async with pool.acquire() as conn:
        row = await conn.fetchrow("SELECT * FROM invite_codes WHERE code = $1", code)
        if not row:
            return None
        return InviteCode(
            code=row["code"],
            plan=row["plan"],
            duration_days=row["duration_days"],
            created_by=row["created_by"],
            created_at=float(row["created_at"].timestamp()) if row.get("created_at") else time.time(),
            expires_at=float(row["expires_at"].timestamp()) if row.get("expires_at") else 0.0,
            used_by=row["used_by"],
            used_at=float(row["used_at"].timestamp()) if row.get("used_at") else None,
        )


async def _db_redeem_invite(code: str, user_id: str) -> InviteCode:
    pool = await _get_pool()
    if not pool:
        invite = _invite_codes.get(code)
        if invite:
            invite.used_by = user_id
            invite.used_at = time.time()
        return invite
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            UPDATE invite_codes
            SET used_by = $1, used_at = NOW()
            WHERE code = $2 AND used_by IS NULL
              AND (expires_at IS NULL OR expires_at > NOW())
            RETURNING *
            """,
            user_id, code
        )
        if not row:
            raise ValueError("邀请码无效、已过期或已被使用")
        return InviteCode(
            code=row["code"], plan=row["plan"], duration_days=row["duration_days"],
            created_by=row["created_by"], used_by=row["used_by"],
            used_at=float(row["used_at"].timestamp()) if row.get("used_at") else None,
        )


async def _db_list_invites() -> list[InviteCode]:
    pool = await _get_pool()
    if not pool:
        return list(_invite_codes.values())
    async with pool.acquire() as conn:
        rows = await conn.fetch("SELECT * FROM invite_codes ORDER BY created_at DESC")
        return [
            InviteCode(
                code=r["code"], plan=r["plan"], duration_days=r["duration_days"],
                created_by=r["created_by"],
                expires_at=float(r["expires_at"].timestamp()) if r.get("expires_at") else 0.0,
                used_by=r["used_by"],
            )
            for r in rows
        ]


# ─────────────────────────────────────────────────────────────────
# 生成函数
# ─────────────────────────────────────────────────────────────────

_CHARS = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"  # 去掉易混淆字符

def _gen_class_code() -> str:
    for _ in range(30):
        code = "".join(random.choices(_CHARS, k=6))
        if code not in _classrooms:
            return code
    raise RuntimeError("无法生成唯一班级码")


def _gen_invite_code() -> str:
    chars = string.ascii_uppercase + string.digits
    for _ in range(30):
        code = "".join(random.choices(chars, k=8))
        if code not in _invite_codes:
            return code
    raise RuntimeError("无法生成唯一邀请码")


# ─────────────────────────────────────────────────────────────────
# 业务逻辑（异步化）
# ─────────────────────────────────────────────────────────────────

async def create_classroom(
    teacher_id: str, class_name: str,
    era: str = "北宋·熙宁变法", expire_days: int = 0,
) -> ClassRoom:
    code = _gen_class_code()
    room_id = "room_" + uuid.uuid4().hex[:8]
    expires_at = (time.time() + expire_days * 86400) if expire_days > 0 else 0.0
    classroom = ClassRoom(
        class_code=code, room_id=room_id, teacher_id=teacher_id,
        class_name=class_name, era=era, expires_at=expires_at,
    )
    await _db_create_classroom(classroom)
    return classroom


async def join_classroom(class_code: str, user_id: str, display_name: str = "") -> ClassRoom:
    code = class_code.upper().strip()
    classroom = await _db_get_classroom(code)
    if not classroom:
        raise ValueError(f"班级码 {code} 不存在")
    if classroom.is_expired:
        raise ValueError(f"班级码 {code} 已过期，请联系老师刷新")
    if not classroom.is_active:
        raise ValueError("该班级已关闭")
    await _db_join_classroom(code, user_id, display_name)
    classroom.members[user_id] = {
        "user_id": user_id,
        "display_name": display_name or user_id,
        "joined_at": time.time(),
        "role": "student",
    }
    return classroom


async def get_classroom_by_code(class_code: str) -> Optional[ClassRoom]:
    return await _db_get_classroom(class_code.upper().strip())


async def get_teacher_classrooms(teacher_id: str) -> list[ClassRoom]:
    return await _db_teacher_classrooms(teacher_id)


async def close_classroom(class_code: str, teacher_id: str) -> ClassRoom:
    await _db_close_classroom(class_code.upper(), teacher_id)
    classroom = await _db_get_classroom(class_code.upper())
    if not classroom:
        raise ValueError("班级码不存在")
    classroom.is_active = False
    return classroom


async def create_invite_codes(
    admin_id: str, plan: str, count: int = 1,
    duration_days: int = 365, expire_days: int = 30,
) -> list[InviteCode]:
    codes = []
    expires_at = time.time() + expire_days * 86400
    for _ in range(count):
        code = _gen_invite_code()
        invite = InviteCode(
            code=code, plan=plan, duration_days=duration_days,
            created_by=admin_id, expires_at=expires_at,
        )
        await _db_save_invite(invite)
        codes.append(invite)
    return codes


async def redeem_invite_code(code: str, user_id: str) -> InviteCode:
    invite = await _db_get_invite(code.upper().strip())
    if not invite:
        raise ValueError(f"邀请码 {code} 不存在")
    if not invite.is_valid:
        raise ValueError("该邀请码已被使用或已过期")
    invite = await _db_redeem_invite(code.upper().strip(), user_id)
    await upgrade_plan(user_id, invite.plan, duration_days=invite.duration_days)
    return invite


# ─────────────────────────────────────────────────────────────────
# FastAPI Router
# ─────────────────────────────────────────────────────────────────

invite_router = APIRouter(prefix="/api/classroom", tags=["classroom"])
invite_code_router = APIRouter(prefix="/api/invite", tags=["invite"])


class CreateClassRequest(BaseModel):
    class_name: str
    era: str = "北宋·熙宁变法"
    expire_days: int = 0


class JoinClassRequest(BaseModel):
    class_code: str
    display_name: str = ""


@invite_router.post("/create")
async def api_create_classroom(
    req: CreateClassRequest,
    token: TokenData = Depends(require_teacher),
):
    """教师创建班级，返回6位班级码"""
    classroom = await create_classroom(
        teacher_id=token.user_id, class_name=req.class_name,
        era=req.era, expire_days=req.expire_days,
    )
    return {
        "success": True,
        "class_code": classroom.class_code,
        "room_id": classroom.room_id,
        "class_name": classroom.class_name,
        "era": classroom.era,
        "message": f"班级码已生成：{classroom.class_code}，发送给学生即可加入",
    }


@invite_router.post("/join")
async def api_join_classroom(
    req: JoinClassRequest,
    token: TokenData = Depends(get_current_user),
):
    """学生用班级码加入班级"""
    try:
        classroom = await join_classroom(req.class_code, token.user_id, req.display_name)
    except (ValueError, PermissionError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {
        "success": True,
        "class_code": classroom.class_code,
        "room_id": classroom.room_id,
        "class_name": classroom.class_name,
        "era": classroom.era,
        "teacher_id": classroom.teacher_id,
        "message": f"成功加入『{classroom.class_name}』班级",
    }


@invite_router.get("/my_classes")
async def api_my_classrooms(token: TokenData = Depends(require_teacher)):
    """教师查看自己的所有班级"""
    classrooms = await get_teacher_classrooms(token.user_id)
    return {"classrooms": [c.to_dict() for c in classrooms]}


@invite_router.get("/info/{class_code}")
async def api_classroom_info(
    class_code: str,
    token: TokenData = Depends(get_current_user),
):
    classroom = await get_classroom_by_code(class_code)
    if not classroom:
        raise HTTPException(status_code=404, detail="班级码不存在")
    return classroom.to_dict()


@invite_router.post("/close/{class_code}")
async def api_close_classroom(
    class_code: str,
    token: TokenData = Depends(require_teacher),
):
    try:
        classroom = await close_classroom(class_code, token.user_id)
    except (ValueError, PermissionError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"success": True, "message": f"班级 {class_code} 已关闭"}


# ── 邀请码接口 ───────────────────────────────────────────────────

class GenerateInviteRequest(BaseModel):
    plan: str = "teacher_pro"
    count: int = 1
    duration_days: int = 365
    expire_days: int = 30


class RedeemInviteRequest(BaseModel):
    code: str


@invite_code_router.post("/generate")
async def api_generate_invite(
    req: GenerateInviteRequest,
    token: TokenData = Depends(require_admin),
):
    """管理员批量生成邀请码"""
    if req.count > 100:
        raise HTTPException(status_code=400, detail="单次最多生成100个邀请码")
    codes = await create_invite_codes(
        admin_id=token.user_id, plan=req.plan, count=req.count,
        duration_days=req.duration_days, expire_days=req.expire_days,
    )
    return {"success": True, "count": len(codes), "codes": [c.to_dict() for c in codes]}


@invite_code_router.post("/redeem")
async def api_redeem_invite(
    req: RedeemInviteRequest,
    token: TokenData = Depends(get_current_user),
):
    """用户兑换邀请码激活套餐"""
    try:
        invite = await redeem_invite_code(req.code, token.user_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    quota = get_quota_status(token.user_id)
    return {
        "success": True,
        "plan": invite.plan,
        "duration_days": invite.duration_days,
        "message": f"邀请码兑换成功！已激活「{quota.current_plan.name}」",
        "quota": quota.to_dict(),
    }


@invite_code_router.get("/list")
async def api_list_invite_codes(token: TokenData = Depends(require_admin)):
    """管理员查看所有邀请码"""
    codes = await _db_list_invites()
    return {
        "total": len(codes),
        "valid": sum(1 for c in codes if c.is_valid),
        "used": sum(1 for c in codes if c.is_used),
        "codes": [c.to_dict() for c in codes],
    }
