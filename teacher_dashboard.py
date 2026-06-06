# teacher_dashboard.py
"""
息壤 · Phase 15B · 教师驾驶舱

核心理念：
  教师不是旁观者——他们是设计师和导航员。
  驾驶舱让教师实时看见：
    · 全班大概念掌握度热力图
    · 哪个学生在哪里「卡住」了
    · 实时情绪分布（班级整体情绪走向）
    · 创作输出质量概览

数据聚合来源：
  - ConceptEngine (concept_engine.py)     → 大概念触碰统计
  - EmotionEngine (emotion_engine.py)     → 情绪弧线
  - ThinkingEngine (thinking_engine.py)   → 因果链深度
  - InquiryEngine (inquiry_engine.py)     → 问题探究深度
  - UserProfile (user_profile.py)         → 知识掌握度
  - WorkshopSession (source_workshop.py)  → 史料引用
  - CreativeSession (creative_engine.py)  → 创作数量
"""
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from concept_engine import HistoryConcept  # noqa
from inquiry_engine import _inquiry_engines  # noqa
from thinking_engine import _thinking_engines as _thinking_engines_ref  # noqa


# ═══════════════════════════════════════════════════════════════
# 学生快照
# ═══════════════════════════════════════════════════════════════

@dataclass
class StudentSnapshot:
    """单个学生在某时刻的学习状态快照"""
    user_id: str
    session_id: str
    display_name: str

    # 进度
    rounds_completed: int = 0
    last_active_ts: float = field(default_factory=time.time)

    # 大概念
    concepts_touched: List[str] = field(default_factory=list)
    top_concept: str = ""

    # 情绪
    current_emotion: str = "—"
    emotion_intensity: int = 0

    # 探究深度
    inquiry_questions_generated: int = 0
    socratic_turns: int = 0
    bookmarked_questions: int = 0

    # 史料
    citations_count: int = 0
    evidence_score: int = 0

    # 创作
    creations_count: int = 0

    # 卡点检测
    is_stuck: bool = False
    stuck_reason: str = ""

    def to_dict(self) -> dict:
        return {
            "user_id": self.user_id,
            "session_id": self.session_id,
            "name": self.display_name,
            "rounds": self.rounds_completed,
            "last_active": self.last_active_ts,
            "concepts": self.concepts_touched,
            "top_concept": self.top_concept,
            "emotion": self.current_emotion,
            "emotion_intensity": self.emotion_intensity,
            "inquiry_q": self.inquiry_questions_generated,
            "socratic_turns": self.socratic_turns,
            "bookmarked": self.bookmarked_questions,
            "citations": self.citations_count,
            "evidence_score": self.evidence_score,
            "creations": self.creations_count,
            "is_stuck": self.is_stuck,
            "stuck_reason": self.stuck_reason,
        }


# ═══════════════════════════════════════════════════════════════
# 驾驶舱引擎
# ═══════════════════════════════════════════════════════════════

class TeacherDashboard:
    """
    教师驾驶舱数据聚合器。
    读取各引擎的会话状态，汇总为班级视图。
    """

    def __init__(self, room_id: str, teacher_id: str):
        self.room_id = room_id
        self.teacher_id = teacher_id
        self._member_sessions: Dict[str, str] = {}  # user_id → session_id
        self._member_names: Dict[str, str] = {}

    def register_member(self, user_id: str, session_id: str, display_name: str = ""):
        self._member_sessions[user_id] = session_id
        self._member_names[user_id] = display_name or user_id[:8]

    def _get_snapshot(self, user_id: str) -> StudentSnapshot:
        """聚合单个学生的当前状态"""
        session_id = self._member_sessions.get(user_id, "")
        snap = StudentSnapshot(
            user_id=user_id,
            session_id=session_id,
            display_name=self._member_names.get(user_id, user_id[:8]),
        )
        if not session_id:
            return snap

        # ── 大概念数据 ────────────────────────────────────────
        from concept_engine import ConceptEngine, _concept_engine_global
        try:
            tracker = _concept_engine_global._trackers.get(session_id)
            if tracker:
                snap.concepts_touched = [c.value for c in tracker.active_concepts]
                top = tracker.get_top_concepts(1)
                snap.top_concept = top[0].value if top else ""
        except Exception:
            pass

        # ── 情绪数据 ──────────────────────────────────────────
        try:
            from emotion_engine import _emotion_engine_global
            states = _emotion_engine_global.get_all_states_summary()
            if states:
                # 取第一个 NPC 的情绪代表整体场景氛围
                first = next(iter(states.values()), {})
                snap.current_emotion = first.get("current_emotion", "—")
                snap.emotion_intensity = first.get("intensity", 0)
        except Exception:
            pass

        # ── 探究问题数据 ──────────────────────────────────────
        try:
            from inquiry_engine import _inquiry_engines
            engine = _inquiry_engines.get(session_id)
            if engine:
                snap.inquiry_questions_generated = len(engine._current_questions)
                snap.socratic_turns = len(engine.socratic.turns)
                snap.bookmarked_questions = len(engine.notebook.bookmarks)
        except Exception:
            pass

        # ── 史料引用数据 ──────────────────────────────────────
        try:
            from source_workshop import _workshop_sessions
            ws = _workshop_sessions.get(session_id)
            if ws:
                snap.citations_count = len(ws.citation_tracker.citations)
                snap.evidence_score = ws.citation_tracker.total_evidence_score
        except Exception:
            pass

        # ── 创作数据 ──────────────────────────────────────────
        try:
            from creative_engine import _creative_sessions
            cs = _creative_sessions.get(session_id)
            if cs:
                snap.creations_count = len(cs.creations)
        except Exception:
            pass

        # ── 卡点检测 ──────────────────────────────────────────
        idle_minutes = (time.time() - snap.last_active_ts) / 60
        if idle_minutes > 5 and snap.rounds_completed < 3:
            snap.is_stuck = True
            snap.stuck_reason = f"超过{int(idle_minutes)}分钟未推进，可能遇到困难"
        elif snap.socratic_turns == 0 and snap.inquiry_questions_generated > 0:
            snap.is_stuck = True
            snap.stuck_reason = "有探究问题但尚未开始苏格拉底对话"

        return snap

    def get_class_overview(self) -> dict:
        """班级整体概览"""
        snapshots = [self._get_snapshot(uid) for uid in self._member_sessions]

        # 大概念热力图：各概念被多少人触碰
        concept_heatmap: Dict[str, int] = {}
        for snap in snapshots:
            for c in snap.concepts_touched:
                concept_heatmap[c] = concept_heatmap.get(c, 0) + 1

        # 情绪分布
        emotion_dist: Dict[str, int] = {}
        for snap in snapshots:
            if snap.current_emotion and snap.current_emotion != "—":
                emotion_dist[snap.current_emotion] = emotion_dist.get(snap.current_emotion, 0) + 1

        # 卡住的学生
        stuck_students = [s.to_dict() for s in snapshots if s.is_stuck]

        # 活跃学生排行（按回合数）
        top_active = sorted(snapshots, key=lambda s: s.rounds_completed, reverse=True)[:5]

        # 创作者
        creators = [s for s in snapshots if s.creations_count > 0]

        return {
            "room_id": self.room_id,
            "member_count": len(self._member_sessions),
            "snapshots": [s.to_dict() for s in snapshots],
            "concept_heatmap": concept_heatmap,
            "emotion_distribution": emotion_dist,
            "stuck_students": stuck_students,
            "top_active": [s.to_dict() for s in top_active],
            "creators": [s.to_dict() for s in creators],
            "summary": {
                "avg_concepts": round(
                    sum(len(s.concepts_touched) for s in snapshots) / max(len(snapshots), 1), 1
                ),
                "avg_citations": round(
                    sum(s.citations_count for s in snapshots) / max(len(snapshots), 1), 1
                ),
                "total_creations": sum(s.creations_count for s in snapshots),
                "stuck_count": len(stuck_students),
            },
        }

    def get_student_detail(self, user_id: str) -> dict:
        """获取单个学生的详细状态"""
        return self._get_snapshot(user_id).to_dict()

    def suggest_intervention(self) -> List[dict]:
        """
        教师干预建议：基于班级状态给出 2–3 条具体建议。
        """
        snapshots = [self._get_snapshot(uid) for uid in self._member_sessions]
        suggestions = []

        stuck = [s for s in snapshots if s.is_stuck]
        if stuck:
            suggestions.append({
                "type": "intervention",
                "priority": "high",
                "message": f"有 {len(stuck)} 名学生可能遇到困难，建议暂停课堂进行集体讨论。",
                "affected_students": [s.user_id for s in stuck[:3]],
            })

        low_citation = [s for s in snapshots if s.citations_count == 0]
        if len(low_citation) > len(snapshots) * 0.6:
            suggestions.append({
                "type": "pedagogy",
                "priority": "medium",
                "message": "超过60%的学生尚未引用史料，建议引导学生使用「史料直面」功能。",
                "affected_students": [],
            })

        zero_creation = [s for s in snapshots if s.creations_count == 0]
        if len(zero_creation) > len(snapshots) * 0.7:
            suggestions.append({
                "type": "engagement",
                "priority": "low",
                "message": "大部分学生尚未创作，可以向全班布置「写一篇历史日记」任务。",
                "affected_students": [],
            })

        return suggestions


# ── 全局注册表 ────────────────────────────────────────────────

_dashboards: Dict[str, TeacherDashboard] = {}


def get_dashboard(room_id: str, teacher_id: str = "") -> TeacherDashboard:
    if room_id not in _dashboards:
        _dashboards[room_id] = TeacherDashboard(room_id, teacher_id)
    return _dashboards[room_id]


# ── 全局引擎引用（供跨模块访问）──────────────────────────────
# 这些在 server.py 初始化后通过 monkey-patch 注入
_concept_engine_ref = None
_emotion_engine_ref = None
