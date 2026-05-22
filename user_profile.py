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
    """用户档案的持久化读写层"""

    def load(self, user_id: str) -> UserProfile:
        path = self._path(user_id)
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            # 反序列化嵌套 dataclass
            raw["explorations"] = [ExplorationRecord(**e) for e in raw.get("explorations", [])]
            raw["reflections"] = [ReflectionRecord(**r) for r in raw.get("reflections", [])]
            return UserProfile(**raw)
        return UserProfile(user_id=user_id)

    def save(self, profile: UserProfile) -> None:
        path = self._path(profile.user_id)
        data = asdict(profile)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _path(self, user_id: str) -> str:
        # user_id 仅允许 alphanumeric + _ + - 防路径注入
        safe_id = "".join(c for c in user_id if c.isalnum() or c in "_-")[:64]
        return os.path.join(_PROFILE_DIR, f"{safe_id}.json")


# ── 单例 ──────────────────────────────────────────────────────
profile_store = UserProfileStore()
