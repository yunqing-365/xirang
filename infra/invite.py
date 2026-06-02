# infra/invite.py
"""
息壤班级码 / 邀请码系统（P0）
=====================================
两种码：
  1. 班级码（class_code）：教师创建，6位大写字母，学生扫码加入班级
  2. 邀请码（invite_code）：管理员生成，8位，发给教师用于激活专业版

班级码功能：
  - 教师创建班级 → 生成6位班级码
  - 学生用班级码加入 → 自动绑定到教师的会话/班级
  - 教师实时看到班级成员列表
  - 班级码默认有效期：7天（可续期）

邀请码功能：
  - 管理员批量生成
  - 一次性使用，用后作废
  - 激活指定套餐（teacher_pro / student / school）
"""
from __future__ import annotations

import random
import string
import time
from dataclasses import dataclass, field
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from infra.auth import TokenData, get_current_user, require_teacher, require_admin
from infra.quota import upgrade_plan, get_quota_status

# ─────────────────────────────────────────────────────────────────
# 数据结构
# ─────────────────────────────────────────────────────────────────

@dataclass
class ClassRoom:
    class_code: str           # 6位大写，对外展示
    room_id: str              # 内部ID，对应 server.py 的 _classrooms
    teacher_id: str
    class_name: str
    era: str                  # 绑定的历史场景
    created_at: float = field(default_factory=time.time)
    expires_at: float = 0.0   # 0=永不过期
    members: dict[str, dict] = field(default_factory=dict)   # user_id → info
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
    code: str                 # 8位字母+数字
    plan: str                 # 激活的套餐
    duration_days: int        # 订阅时长
    created_by: str           # admin user_id
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
# 内存存储（生产替换为 PostgreSQL）
# ─────────────────────────────────────────────────────────────────

_classrooms: dict[str, ClassRoom] = {}          # class_code → ClassRoom
_classrooms_by_teacher: dict[str, list[str]] = {}  # teacher_id → [class_code]
_invite_codes: dict[str, InviteCode] = {}       # code → InviteCode


# ─────────────────────────────────────────────────────────────────
# 生成函数
# ─────────────────────────────────────────────────────────────────

def _gen_class_code() -> str:
    """生成唯一6位大写字母班级码，排除易混淆字符"""
    chars = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"  # 去掉 I O 0 1
    for _ in range(20):  # 重试最多20次
        code = "".join(random.choices(chars, k=6))
        if code not in _classrooms:
            return code
    raise RuntimeError("无法生成唯一班级码，请联系管理员")


def _gen_invite_code() -> str:
    chars = string.ascii_uppercase + string.digits
    for _ in range(20):
        code = "".join(random.choices(chars, k=8))
        if code not in _invite_codes:
            return code
    raise RuntimeError("无法生成唯一邀请码")


def _gen_room_id() -> str:
    import uuid
    return "room_" + uuid.uuid4().hex[:8]


# ─────────────────────────────────────────────────────────────────
# 业务逻辑
# ─────────────────────────────────────────────────────────────────

def create_classroom(
    teacher_id: str,
    class_name: str,
    era: str = "北宋·熙宁变法",
    expire_days: int = 0,    # 0=永不过期
) -> ClassRoom:
    code = _gen_class_code()
    room_id = _gen_room_id()
    expires_at = (time.time() + expire_days * 86400) if expire_days > 0 else 0.0
    classroom = ClassRoom(
        class_code=code,
        room_id=room_id,
        teacher_id=teacher_id,
        class_name=class_name,
        era=era,
        expires_at=expires_at,
    )
    _classrooms[code] = classroom
    _classrooms_by_teacher.setdefault(teacher_id, []).append(code)
    return classroom


def join_classroom(class_code: str, user_id: str, display_name: str = "") -> ClassRoom:
    """学生用班级码加入班级"""
    code = class_code.upper().strip()
    classroom = _classrooms.get(code)
    if not classroom:
        raise ValueError(f"班级码 {code} 不存在")
    if classroom.is_expired:
        raise ValueError(f"班级码 {code} 已过期，请联系老师刷新")
    if not classroom.is_active:
        raise ValueError(f"该班级已关闭")
    classroom.members[user_id] = {
        "user_id": user_id,
        "display_name": display_name or user_id,
        "joined_at": time.time(),
        "role": "student",
    }
    return classroom


def get_classroom_by_code(class_code: str) -> Optional[ClassRoom]:
    return _classrooms.get(class_code.upper().strip())


def get_teacher_classrooms(teacher_id: str) -> list[ClassRoom]:
    codes = _classrooms_by_teacher.get(teacher_id, [])
    return [_classrooms[c] for c in codes if c in _classrooms]


def close_classroom(class_code: str, teacher_id: str) -> ClassRoom:
    classroom = _classrooms.get(class_code.upper())
    if not classroom:
        raise ValueError("班级码不存在")
    if classroom.teacher_id != teacher_id:
        raise PermissionError("只有班级创建者可以关闭班级")
    classroom.is_active = False
    return classroom


# ── 邀请码 ──────────────────────────────────────────────────────

def create_invite_codes(
    admin_id: str,
    plan: str,
    count: int = 1,
    duration_days: int = 365,
    expire_days: int = 30,  # 邀请码本身的有效期
) -> list[InviteCode]:
    codes = []
    expires_at = time.time() + expire_days * 86400
    for _ in range(count):
        code = _gen_invite_code()
        invite = InviteCode(
            code=code,
            plan=plan,
            duration_days=duration_days,
            created_by=admin_id,
            expires_at=expires_at,
        )
        _invite_codes[code] = invite
        codes.append(invite)
    return codes


def redeem_invite_code(code: str, user_id: str) -> InviteCode:
    """兑换邀请码，激活对应套餐"""
    invite = _invite_codes.get(code.upper().strip())
    if not invite:
        raise ValueError(f"邀请码 {code} 不存在")
    if not invite.is_valid:
        if invite.is_used:
            raise ValueError("该邀请码已被使用")
        raise ValueError("该邀请码已过期")
    invite.used_by = user_id
    invite.used_at = time.time()
    # 激活套餐
    upgrade_plan(user_id, invite.plan, duration_days=invite.duration_days)
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


# ── 班级码接口 ───────────────────────────────────────────────────

@invite_router.post("/create")
async def api_create_classroom(
    req: CreateClassRequest,
    token: TokenData = Depends(require_teacher),
):
    """教师创建班级，返回6位班级码"""
    classroom = create_classroom(
        teacher_id=token.user_id,
        class_name=req.class_name,
        era=req.era,
        expire_days=req.expire_days,
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
        classroom = join_classroom(req.class_code, token.user_id, req.display_name)
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
    classrooms = get_teacher_classrooms(token.user_id)
    return {"classrooms": [c.to_dict() for c in classrooms]}


@invite_router.get("/info/{class_code}")
async def api_classroom_info(
    class_code: str,
    token: TokenData = Depends(get_current_user),
):
    classroom = get_classroom_by_code(class_code)
    if not classroom:
        raise HTTPException(status_code=404, detail="班级码不存在")
    return classroom.to_dict()


@invite_router.post("/close/{class_code}")
async def api_close_classroom(
    class_code: str,
    token: TokenData = Depends(require_teacher),
):
    try:
        classroom = close_classroom(class_code, token.user_id)
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
    codes = create_invite_codes(
        admin_id=token.user_id,
        plan=req.plan,
        count=req.count,
        duration_days=req.duration_days,
        expire_days=req.expire_days,
    )
    return {
        "success": True,
        "count": len(codes),
        "codes": [c.to_dict() for c in codes],
    }


@invite_code_router.post("/redeem")
async def api_redeem_invite(
    req: RedeemInviteRequest,
    token: TokenData = Depends(get_current_user),
):
    """用户兑换邀请码激活套餐"""
    try:
        invite = redeem_invite_code(req.code, token.user_id)
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
    return {
        "total": len(_invite_codes),
        "valid": sum(1 for c in _invite_codes.values() if c.is_valid),
        "used": sum(1 for c in _invite_codes.values() if c.is_used),
        "codes": [c.to_dict() for c in _invite_codes.values()],
    }
