# user_profile.py
"""
息壤 · 用户成长档案系统

设计目标：让每个用户拥有一张跨会话持久的"人文探索地图"。
存储内容：
  - 探索过的朝代/人物/主题
  - 解锁的"彩蛋"知识点
  - 触发过的人文反思记录
  - 偏好的叙事风格
  - 学习轨迹时间线
"""
import json
import os
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

from config import get_settings

_settings = get_settings()
_PROFILE_DIR = os.path.join(_settings.DATA_DIR, "user_profiles")
os.makedirs(_PROFILE_DIR, exist_ok=True)


@dataclass
class ExplorationRecord:
    session_id: str
    theme: str
    era: str
    agents_met: List[str]
    timestamp: float = field(default_factory=time.time)
    rounds_played: int = 0
    choices_made: List[str] = field(default_factory=list)  # 玩家干预选项记录


@dataclass
class ReflectionRecord:
    session_id: str
    insight: str
    reflection_question: str
    era_fact: str
    player_response: Optional[str] = None   # 玩家对反思问题的回答
    timestamp: float = field(default_factory=time.time)


@dataclass
class UserProfile:
    user_id: str
    created_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)

    # ── 探索地图 ──────────────────────────────────────────────
    explored_eras: List[str] = field(default_factory=list)          # ["北宋", "明朝", ...]
    explored_figures: List[str] = field(default_factory=list)       # ["苏轼", "李白", ...]
    explored_themes: List[str] = field(default_factory=list)        # ["贬谪", "科举", ...]
    unlocked_facts: List[str] = field(default_factory=list)         # 触发过的"彩蛋"知识

    # ── 学习轨迹 ──────────────────────────────────────────────
    explorations: List[ExplorationRecord] = field(default_factory=list)
    reflections: List[ReflectionRecord] = field(default_factory=list)

    # ── 偏好系统 ──────────────────────────────────────────────
    preferred_genres: Dict[str, int] = field(default_factory=dict)  # {"市井烟火": 3, ...}
    total_rounds: int = 0
    total_sessions: int = 0

    # ── 连续打卡 & 成就 ───────────────────────────────────────
    streak_days: int = 0                          # 当前连续探索天数
    streak_best: int = 0                          # 历史最长连续天数
    last_checkin_date: str = ""                   # "YYYY-MM-DD" 格式
    badges: List[str] = field(default_factory=list)  # 已解锁成就列表
    daily_digest_last: str = ""                   # 上次生成每日速递的日期

    # ── 知识掌握度热力图 ──────────────────────────────────────
    # {知识点key: {"score":0-100, "attempts":n, "last_seen":ts, "era":str, "label":str}}
    knowledge_mastery: Dict[str, dict] = field(default_factory=dict)
    # 小测验历史: [{session_id, questions_total, correct, era, timestamp}]
    quiz_history: List[dict] = field(default_factory=list)

    def update_knowledge(self, knowledge_key: str, correct: bool,
                         era: str = "", label: str = ""):
        """
        更新知识点掌握度（遗忘曲线衰减 + 答对/错调整）。
        score: 初始50，答对+15，答错-10，每天衰减3分（底线10）
        """
        import time as _t
        now = _t.time()
        kv = self.knowledge_mastery.get(knowledge_key)
        if kv is None:
            kv = {"score": 50, "attempts": 0, "last_seen": now,
                  "era": era, "label": label or knowledge_key}
            self.knowledge_mastery[knowledge_key] = kv
        days = (now - kv["last_seen"]) / 86400
        kv["score"] = max(10, kv["score"] - min(days * 3, kv["score"] - 10))
        kv["score"] = max(0, min(100, kv["score"] + (15 if correct else -10)))
        kv["attempts"] = kv.get("attempts", 0) + 1
        kv["last_seen"] = now
        if era and not kv.get("era"):
            kv["era"] = era
        if label and not kv.get("label"):
            kv["label"] = label

    def record_quiz(self, session_id: str, total: int,
                    correct: int, era: str = ""):
        """记录一次小测验结果"""
        import time as _t
        self.quiz_history.append({
            "session_id": session_id,
            "questions_total": total,
            "correct": correct,
            "score_pct": round(correct / total * 100) if total else 0,
            "era": era,
            "timestamp": _t.time(),
        })
        self.quiz_history = self.quiz_history[-50:]

    def get_mastery_heatmap(self) -> dict:
        """返回适合前端热力图渲染的结构化数据，按朝代分组。"""
        import time as _t
        now = _t.time()
        by_era: Dict[str, list] = {}
        for key, v in self.knowledge_mastery.items():
            era = v.get("era", "未知")
            days = (now - v.get("last_seen", now)) / 86400
            live_score = max(10, v["score"] - min(days * 3, v["score"] - 10))
            by_era.setdefault(era, []).append({
                "key": key,
                "label": v.get("label", key),
                "score": round(live_score),
                "attempts": v.get("attempts", 0),
                "last_seen_days": round(days, 1),
            })
        for era in by_era:
            by_era[era].sort(key=lambda x: -x["score"])
        return {
            "by_era": by_era,
            "total_points": len(self.knowledge_mastery),
            "avg_score": round(
                sum(v["score"] for v in self.knowledge_mastery.values()) /
                max(len(self.knowledge_mastery), 1)
            ),
            "quiz_count": len(self.quiz_history),
        }

    # ── 统计 ──────────────────────────────────────────────────
    @property
    def exploration_depth(self) -> str:
        """根据探索广度给出称号"""
        n = len(set(self.explored_eras))
        if n == 0:
            return "初入时空"
        elif n < 3:
            return "时空旅人"
        elif n < 6:
            return "史海钩沉"
        elif n < 10:
            return "通古博今"
        else:
            return "时空织梦者"

    def checkin_today(self) -> dict:
        """
        处理每日打卡逻辑。
        返回 {streak, is_new_day, newly_unlocked_badges}
        """
        import datetime
        today = datetime.date.today().isoformat()
        newly_unlocked: List[str] = []

        if self.last_checkin_date == today:
            return {"streak": self.streak_days, "is_new_day": False, "newly_unlocked_badges": []}

        yesterday = (datetime.date.today() - datetime.timedelta(days=1)).isoformat()
        if self.last_checkin_date == yesterday:
            self.streak_days += 1
        else:
            self.streak_days = 1  # 断签重置

        self.streak_best = max(self.streak_best, self.streak_days)
        self.last_checkin_date = today
        self.last_active = time.time()

        # 成就解锁
        streak_badges = {3: "三日不辍", 7: "七日探史", 14: "两周博览", 30: "月旦史海"}
        for days, badge in streak_badges.items():
            if self.streak_days >= days and badge not in self.badges:
                self.badges.append(badge)
                newly_unlocked.append(badge)

        era_badges = {3: "踏遍三朝", 6: "六朝通览", 10: "十代长河"}
        for eras, badge in era_badges.items():
            if len(set(self.explored_eras)) >= eras and badge not in self.badges:
                self.badges.append(badge)
                newly_unlocked.append(badge)

        reflection_badges = {5: "五省吾身", 20: "廿思达观"}
        for count, badge in reflection_badges.items():
            if len(self.reflections) >= count and badge not in self.badges:
                self.badges.append(badge)
                newly_unlocked.append(badge)

        return {"streak": self.streak_days, "is_new_day": True, "newly_unlocked_badges": newly_unlocked}

    def record_exploration(self, record: ExplorationRecord):
        self.explorations.append(record)
        self.total_sessions += 1
        self.total_rounds += record.rounds_played

        # 去重追加
        if record.era not in self.explored_eras:
            self.explored_eras.append(record.era)
        for fig in record.agents_met:
            if fig not in self.explored_figures:
                self.explored_figures.append(fig)
        if record.theme not in self.explored_themes:
            self.explored_themes.append(record.theme)

        # 更新偏好计数（通过主题近似风格）
        self.preferred_genres[record.theme] = self.preferred_genres.get(record.theme, 0) + 1
        self.last_active = time.time()

    def record_reflection(self, record: ReflectionRecord):
        self.reflections.append(record)
        self.last_active = time.time()

    def unlock_fact(self, fact: str):
        if fact not in self.unlocked_facts:
            self.unlocked_facts.append(fact)

    def get_recommended_era(self) -> Optional[str]:
        """根据已探索的内容，推荐下一个可能感兴趣的时代"""
        era_progression = {
            "北宋": ["南宋", "五代十国"],
            "南宋": ["元朝", "北宋"],
            "唐朝": ["五代十国", "宋朝"],
            "明朝": ["清朝", "元朝"],
            "清朝": ["民国", "明朝"],
            "先秦": ["秦朝", "汉朝"],
        }
        for explored in reversed(self.explored_eras):
            suggestions = era_progression.get(explored, [])
            for s in suggestions:
                if s not in self.explored_eras:
                    return s
        return None

    def to_context_summary(self) -> str:
        """生成给 LLM 的用户背景摘要（用于个性化提示词注入）"""
        lines = [
            f"用户称号：{self.exploration_depth}",
            f"已探索朝代：{', '.join(self.explored_eras[-5:]) or '无'}",
            f"已结识历史人物：{', '.join(self.explored_figures[-8:]) or '无'}",
            f"偏好主题：{', '.join(list(self.preferred_genres.keys())[:3]) or '无'}",
            f"共完成探索会话：{self.total_sessions} 次，累计 {self.total_rounds} 回合",
        ]
        if self.reflections:
            last_reflection = self.reflections[-1]
            lines.append(f"上次的人文反思：「{last_reflection.insight}」")
        return "\n".join(lines)


class UserProfileStore:
    """
    用户档案持久化层（SQLite 升级版）

    升级说明：
      原版使用 JSON 文件（每用户一个文件），多进程部署时存在并发写入风险。
      本版本改用 SQLite 单文件数据库：
        - 单表 `profiles`，每行存一个用户的完整 JSON blob
        - 写入通过 WAL 模式（Write-Ahead Logging）保障并发安全
        - 迁移：启动时自动将旧 JSON 文件导入并删除，零停机
      外部接口（load / save）完全不变，调用方无需修改。
    """

    _DB_PATH = os.path.join(_settings.DATA_DIR, "profiles.db")

    def __init__(self):
        self._init_db()
        self._migrate_from_json()

    # ── 数据库初始化 ──────────────────────────────────────────

    def _init_db(self):
        import sqlite3
        os.makedirs(os.path.dirname(self._DB_PATH) if os.path.dirname(self._DB_PATH) else ".", exist_ok=True)
        with sqlite3.connect(self._DB_PATH) as conn:
            conn.execute("PRAGMA journal_mode=WAL")   # 多读单写并发安全
            conn.execute("""
                CREATE TABLE IF NOT EXISTS profiles (
                    user_id    TEXT PRIMARY KEY,
                    data       TEXT NOT NULL,
                    updated_at REAL NOT NULL
                )
            """)
            conn.commit()

    # ── 公开接口（与原版完全相同）────────────────────────────

    def load(self, user_id: str) -> UserProfile:
        import sqlite3
        safe_id = self._safe(user_id)
        with sqlite3.connect(self._DB_PATH) as conn:
            row = conn.execute(
                "SELECT data FROM profiles WHERE user_id = ?", (safe_id,)
            ).fetchone()
        if row is None:
            return UserProfile(user_id=safe_id)
        raw = json.loads(row[0])
        raw["explorations"] = [ExplorationRecord(**e) for e in raw.get("explorations", [])]
        raw["reflections"]  = [ReflectionRecord(**r)  for r in raw.get("reflections", [])]
        return UserProfile(**raw)

    def save(self, profile: UserProfile) -> None:
        import sqlite3
        safe_id = self._safe(profile.user_id)
        data = json.dumps(asdict(profile), ensure_ascii=False)
        with sqlite3.connect(self._DB_PATH) as conn:
            conn.execute("""
                INSERT INTO profiles (user_id, data, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(user_id) DO UPDATE SET
                    data       = excluded.data,
                    updated_at = excluded.updated_at
            """, (safe_id, data, time.time()))
            conn.commit()

    # ── 迁移工具 ──────────────────────────────────────────────

    def _migrate_from_json(self):
        """将旧版 JSON 文件一次性导入 SQLite，成功后删除原文件。"""
        if not os.path.isdir(_PROFILE_DIR):
            return
        migrated = 0
        for fname in os.listdir(_PROFILE_DIR):
            if not fname.endswith(".json"):
                continue
            fpath = os.path.join(_PROFILE_DIR, fname)
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                uid = raw.get("user_id", fname[:-5])
                # 只有 SQLite 中尚无该用户时才导入，避免覆盖更新的数据
                import sqlite3
                with sqlite3.connect(self._DB_PATH) as conn:
                    exists = conn.execute(
                        "SELECT 1 FROM profiles WHERE user_id = ?", (self._safe(uid),)
                    ).fetchone()
                if not exists:
                    profile = UserProfile(user_id=uid)  # 用 load 会触发 SQLite，直接手动构建
                    raw["explorations"] = [ExplorationRecord(**e) for e in raw.get("explorations", [])]
                    raw["reflections"]  = [ReflectionRecord(**r)  for r in raw.get("reflections", [])]
                    self.save(UserProfile(**raw))
                os.remove(fpath)
                migrated += 1
            except Exception as exc:
                print(f"[迁移] 跳过 {fname}: {exc}")
        if migrated:
            print(f"[迁移] 已将 {migrated} 个 JSON 档案导入 SQLite → {self._DB_PATH}")

    @staticmethod
    def _safe(user_id: str) -> str:
        """防路径注入：仅保留 alphanumeric + _ + -，最长 64 字符"""
        return "".join(c for c in user_id if c.isalnum() or c in "_-")[:64]


# ── 单例 ──────────────────────────────────────────────────────
profile_store = UserProfileStore()
