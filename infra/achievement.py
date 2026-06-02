# infra/achievement.py
"""
息壤成就系统 + 跨会话学生历史记录（P2-E）
==========================================
成就分三类：
  探究类  ── 引用多少史料、提出多少问题、触碰大概念数量
  创作类  ── 诗词/奏疏/辩论稿创作数量与质量
  社交类  ── 参与班级讨论、帮助同学、连续打卡

存储：内存 dict（生产迁移 PostgreSQL）
跨会话：StudentRecord 聚合所有历史会话数据
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

from fastapi import APIRouter, Depends
from infra.auth import TokenData, get_current_user

# ══════════════════════════════════════════════════════════
# 成就定义
# ══════════════════════════════════════════════════════════

@dataclass
class AchievementDef:
    id: str
    name: str
    icon: str
    desc: str
    category: str          # explore / create / social
    threshold: int         # 达成所需数量
    metric: str            # 对应 StudentRecord 的哪个字段

ACHIEVEMENT_DEFS: list[AchievementDef] = [
    # ── 探究类 ──────────────────────────────────────────
    AchievementDef("first_session",  "初入时光隧道", "🌌", "完成第一次历史探究", "explore", 1,  "total_sessions"),
    AchievementDef("session_5",      "时空常客",     "⏳", "累计完成5次历史探究",  "explore", 5,  "total_sessions"),
    AchievementDef("session_20",     "历史旅行家",   "🗺️", "累计完成20次历史探究", "explore", 20, "total_sessions"),
    AchievementDef("cite_5",         "史料入门",     "📜", "累计引用5次史料",      "explore", 5,  "total_citations"),
    AchievementDef("cite_30",        "史料达人",     "📚", "累计引用30次史料",     "explore", 30, "total_citations"),
    AchievementDef("cite_100",       "史家之风",     "🏛️", "累计引用100次史料",    "explore", 100,"total_citations"),
    AchievementDef("concept_3",      "概念启蒙",     "💡", "触碰3个历史大概念",    "explore", 3,  "unique_concepts"),
    AchievementDef("concept_8",      "概念大师",     "🧠", "触碰8个历史大概念",    "explore", 8,  "unique_concepts"),
    AchievementDef("multi_era",      "穿越千年",     "⚡", "探索3个不同朝代",      "explore", 3,  "unique_eras"),
    AchievementDef("socratic_20",    "苏格拉底门徒", "🦉", "累计完成20轮问答探究", "explore", 20, "total_socratic"),
    # ── 创作类 ──────────────────────────────────────────
    AchievementDef("create_1",       "执笔成史",     "✍️", "完成第一件历史创作",   "create", 1,  "total_creations"),
    AchievementDef("create_10",      "才思涌现",     "🖋️", "完成10件历史创作",     "create", 10, "total_creations"),
    AchievementDef("deep_think_70",  "深度思考者",   "🔍", "单次思维深度评分≥70",  "create", 70, "best_depth"),
    AchievementDef("deep_think_90",  "历史洞察家",   "🔮", "单次思维深度评分≥90",  "create", 90, "best_depth"),
    # ── 社交类 ──────────────────────────────────────────
    AchievementDef("join_class",     "加入班级",     "🏫", "使用班级码加入课堂",   "social", 1,  "classes_joined"),
    AchievementDef("streak_3",       "三日打卡",     "🔥", "连续3天完成探究",      "social", 3,  "streak_days"),
    AchievementDef("streak_7",       "一周不辍",     "💎", "连续7天完成探究",      "social", 7,  "streak_days"),
]

_ACHV_MAP = {a.id: a for a in ACHIEVEMENT_DEFS}


# ══════════════════════════════════════════════════════════
# 跨会话学生记录
# ══════════════════════════════════════════════════════════

@dataclass
class SessionSummary:
    session_id: str
    era: str
    era_key: str
    rounds: int
    citations: int
    concepts: list[str]
    creations: int
    socratic_turns: int
    thinking_depth: int
    emotion_arc: list[str]
    achievement_ids: list[str]
    created_at: float = field(default_factory=time.time)


@dataclass
class StudentRecord:
    user_id: str
    display_name: str = ""

    # 累计统计
    total_sessions:   int = 0
    total_citations:  int = 0
    total_creations:  int = 0
    total_socratic:   int = 0
    best_depth:       int = 0
    unique_concepts:  int = 0
    unique_eras:      int = 0
    classes_joined:   int = 0
    streak_days:      int = 0
    last_active_date: str = ""

    # 集合（用于去重计数）
    _all_concepts: list[str] = field(default_factory=list)
    _all_eras:     list[str] = field(default_factory=list)

    # 历史会话列表
    sessions: list[SessionSummary] = field(default_factory=list)

    # 已解锁成就 ID 集合
    unlocked: list[str] = field(default_factory=list)

    def add_session(self, summary: SessionSummary):
        self.sessions.append(summary)
        self.total_sessions   += 1
        self.total_citations  += summary.citations
        self.total_creations  += summary.creations
        self.total_socratic   += summary.socratic_turns
        self.best_depth        = max(self.best_depth, summary.thinking_depth)

        for c in summary.concepts:
            if c not in self._all_concepts:
                self._all_concepts.append(c)
        self.unique_concepts = len(self._all_concepts)

        if summary.era_key and summary.era_key not in self._all_eras:
            self._all_eras.append(summary.era_key)
        self.unique_eras = len(self._all_eras)

        # 连续打卡
        today = time.strftime("%Y-%m-%d")
        if self.last_active_date == today:
            pass  # 同一天不重复计
        elif self.last_active_date == time.strftime("%Y-%m-%d", time.localtime(time.time()-86400)):
            self.streak_days += 1
        else:
            self.streak_days = 1
        self.last_active_date = today

    def check_new_achievements(self) -> list[AchievementDef]:
        """检查本次新解锁的成就"""
        newly = []
        for adef in ACHIEVEMENT_DEFS:
            if adef.id in self.unlocked:
                continue
            val = getattr(self, adef.metric, 0)
            if val >= adef.threshold:
                self.unlocked.append(adef.id)
                newly.append(adef)
        return newly

    def to_dict(self) -> dict:
        unlocked_defs = [
            {"id": a.id, "name": a.name, "icon": a.icon,
             "desc": a.desc, "category": a.category}
            for aid in self.unlocked
            if (a := _ACHV_MAP.get(aid))
        ]
        locked_defs = [
            {"id": a.id, "name": a.name, "icon": a.icon,
             "desc": a.desc, "category": a.category,
             "progress": min(getattr(self, a.metric, 0), a.threshold),
             "threshold": a.threshold}
            for a in ACHIEVEMENT_DEFS if a.id not in self.unlocked
        ]
        return {
            "user_id":         self.user_id,
            "display_name":    self.display_name,
            "total_sessions":  self.total_sessions,
            "total_citations": self.total_citations,
            "total_creations": self.total_creations,
            "total_socratic":  self.total_socratic,
            "best_depth":      self.best_depth,
            "unique_concepts": self.unique_concepts,
            "unique_eras":     self.unique_eras,
            "streak_days":     self.streak_days,
            "all_concepts":    self._all_concepts,
            "all_eras":        self._all_eras,
            "unlocked_count":  len(self.unlocked),
            "achievements_unlocked": unlocked_defs,
            "achievements_locked":   locked_defs[:6],  # 只返回6个待解锁
            "recent_sessions": [
                {"session_id": s.session_id, "era": s.era, "rounds": s.rounds,
                 "citations": s.citations, "concepts": s.concepts[:3],
                 "created_at": s.created_at}
                for s in sorted(self.sessions, key=lambda x: x.created_at, reverse=True)[:10]
            ],
        }


# ══════════════════════════════════════════════════════════
# 存储
# ══════════════════════════════════════════════════════════

_records: dict[str, StudentRecord] = {}


def get_record(user_id: str, display_name: str = "") -> StudentRecord:
    if user_id not in _records:
        _records[user_id] = StudentRecord(user_id=user_id, display_name=display_name)
    return _records[user_id]


def record_session_end(
    user_id: str,
    session_id: str,
    era: str,
    era_key: str,
    rounds: int,
    citations: int,
    concepts: list[str],
    creations: int,
    socratic_turns: int,
    thinking_depth: int,
    emotion_arc: list[str],
    display_name: str = "",
) -> tuple[StudentRecord, list[AchievementDef]]:
    """会话结束时调用，返回（更新后的记录, 新解锁成就列表）"""
    rec = get_record(user_id, display_name)
    summary = SessionSummary(
        session_id=session_id, era=era, era_key=era_key,
        rounds=rounds, citations=citations, concepts=concepts,
        creations=creations, socratic_turns=socratic_turns,
        thinking_depth=thinking_depth, emotion_arc=emotion_arc,
        achievement_ids=[],
    )
    rec.add_session(summary)
    new_achievements = rec.check_new_achievements()
    summary.achievement_ids = [a.id for a in new_achievements]
    return rec, new_achievements


# ══════════════════════════════════════════════════════════
# FastAPI Router
# ══════════════════════════════════════════════════════════

from pydantic import BaseModel

achievement_router = APIRouter(prefix="/api/achievement", tags=["achievement"])


class SessionEndData(BaseModel):
    era: str = ""
    era_key: str = "default"
    rounds: int = 0
    citations: int = 0
    concepts: list[str] = []
    creations: int = 0
    socratic_turns: int = 0
    thinking_depth: int = 0
    emotion_arc: list[str] = []
    display_name: str = ""


@achievement_router.post("/session_end/{session_id}")
async def api_session_end(
    session_id: str,
    data: SessionEndData,
    token: TokenData = Depends(get_current_user),
):
    """会话结束时前端调用，触发成就检查"""
    rec, new_achievements = record_session_end(
        user_id=token.user_id,
        session_id=session_id,
        era=data.era, era_key=data.era_key,
        rounds=data.rounds, citations=data.citations,
        concepts=data.concepts, creations=data.creations,
        socratic_turns=data.socratic_turns,
        thinking_depth=data.thinking_depth,
        emotion_arc=data.emotion_arc,
        display_name=data.display_name or token.user_id,
    )
    return {
        "success": True,
        "new_achievements": [
            {"id": a.id, "name": a.name, "icon": a.icon, "desc": a.desc}
            for a in new_achievements
        ],
        "stats": {
            "total_sessions":  rec.total_sessions,
            "total_citations": rec.total_citations,
            "unique_concepts": rec.unique_concepts,
            "unique_eras":     rec.unique_eras,
            "streak_days":     rec.streak_days,
            "unlocked_count":  len(rec.unlocked),
        },
    }


@achievement_router.get("/my")
async def api_my_achievements(token: TokenData = Depends(get_current_user)):
    """查看我的成就 + 探究历史"""
    rec = get_record(token.user_id)
    return rec.to_dict()


@achievement_router.get("/all_defs")
async def api_all_achievement_defs():
    """返回全部成就定义（用于前端展示成就墙）"""
    return {
        "total": len(ACHIEVEMENT_DEFS),
        "categories": {
            "explore": [a.__dict__ for a in ACHIEVEMENT_DEFS if a.category == "explore"],
            "create":  [a.__dict__ for a in ACHIEVEMENT_DEFS if a.category == "create"],
            "social":  [a.__dict__ for a in ACHIEVEMENT_DEFS if a.category == "social"],
        }
    }


@achievement_router.get("/leaderboard")
async def api_leaderboard():
    """班级排行榜（按探究轮次 + 史料引用综合排名）"""
    ranked = sorted(
        _records.values(),
        key=lambda r: r.total_sessions * 3 + r.total_citations * 2 + r.unique_concepts * 5,
        reverse=True,
    )[:20]
    return {
        "leaderboard": [
            {
                "rank": i + 1,
                "user_id": r.user_id,
                "display_name": r.display_name or r.user_id,
                "score": r.total_sessions * 3 + r.total_citations * 2 + r.unique_concepts * 5,
                "total_sessions": r.total_sessions,
                "total_citations": r.total_citations,
                "unique_concepts": r.unique_concepts,
                "unlocked_count": len(r.unlocked),
                "streak_days": r.streak_days,
            }
            for i, r in enumerate(ranked)
        ]
    }
