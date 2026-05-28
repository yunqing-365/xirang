# learning_map.py
"""
息壤 · Phase 15A · 学习路径图谱

核心理念：
  每个学生的历史探索轨迹都是独一无二的地图。
  系统根据「已探索」「兴趣偏好」「掌握薄弱点」，
  个性化推荐下一个值得探索的时空节点。

组成：
  1. LearningNode     — 时空节点（时代 × 人物 × 主题 的三元组）
  2. PathGraph        — 节点之间的前置/相关关系图
  3. LearningMapEngine — 个性化推荐引擎（协同过滤 + 规则）
  4. ExplorationMap   — 可视化数据（供前端渲染已探索版图）
"""
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from user_profile import UserProfile


# ═══════════════════════════════════════════════════════════════
# 时空节点定义
# ═══════════════════════════════════════════════════════════════

@dataclass
class LearningNode:
    node_id: str
    era: str                    # 时代（如「北宋」）
    figure: str                 # 代表人物（如「苏轼」）
    theme: str                  # 核心主题（如「贬谪与创作」）
    title: str                  # 显示名称
    description: str            # 简介（60字以内）
    difficulty: int             # 难度 1–5
    concepts: List[str]         # 涉及的历史学大概念
    prerequisites: List[str]    # 前置节点 ID（建议先探索哪些）
    related: List[str]          # 相关节点 ID
    tags: List[str]             # 检索标签
    unlock_condition: str       # 解锁条件描述（如「探索任意宋代节点后」）
    icon: str                   # 前端图标 emoji
    position: Tuple[float, float] = (0.0, 0.0)  # 地图坐标（x, y）

    def to_dict(self, explored: bool = False, recommended: bool = False) -> dict:
        return {
            "id": self.node_id,
            "era": self.era,
            "figure": self.figure,
            "theme": self.theme,
            "title": self.title,
            "description": self.description,
            "difficulty": self.difficulty,
            "concepts": self.concepts,
            "prerequisites": self.prerequisites,
            "related": self.related,
            "tags": self.tags,
            "unlock_condition": self.unlock_condition,
            "icon": self.icon,
            "position": list(self.position),
            "explored": explored,
            "recommended": recommended,
        }


# ═══════════════════════════════════════════════════════════════
# 节点库（内置，涵盖路线图所需的时代范围）
# ═══════════════════════════════════════════════════════════════

ALL_NODES: List[LearningNode] = [

    # ── 北宋 ─────────────────────────────────────────────────
    LearningNode(
        node_id="song_suzhi_banishment",
        era="北宋", figure="苏轼", theme="贬谪与创作",
        title="乌台诗案·黄州",
        description="苏轼因诗获罪，贬谪黄州，在逆境中写下《赤壁赋》《念奴娇》。",
        difficulty=2, concepts=["权力", "后果", "道德判断"],
        prerequisites=[], related=["song_wang_anshi", "song_suzhi_lingnan"],
        tags=["苏轼", "贬谪", "乌台诗案", "黄州", "北宋"],
        unlock_condition="初始可探索", icon="🖊️", position=(3.0, 4.0),
    ),
    LearningNode(
        node_id="song_wang_anshi",
        era="北宋", figure="王安石", theme="变法与反对",
        title="熙宁变法",
        description="王安石推行新法，与司马光、苏轼等旧党激烈对抗，开北宋党争之先。",
        difficulty=3, concepts=["变革", "权力", "因果"],
        prerequisites=[], related=["song_suzhi_banishment", "song_sima_guang"],
        tags=["王安石", "变法", "新旧党争", "北宋", "政治"],
        unlock_condition="初始可探索", icon="⚖️", position=(2.0, 3.0),
    ),
    LearningNode(
        node_id="song_sima_guang",
        era="北宋", figure="司马光", theme="史学与保守",
        title="《资治通鉴》与旧党",
        description="司马光历时19年编成《资治通鉴》，废新法，是北宋最重要的保守派领袖。",
        difficulty=3, concepts=["集体记忆", "证据", "权力"],
        prerequisites=["song_wang_anshi"], related=["song_wang_anshi"],
        tags=["司马光", "资治通鉴", "旧党", "史学"],
        unlock_condition="探索「熙宁变法」后解锁", icon="📜", position=(1.5, 2.5),
    ),
    LearningNode(
        node_id="song_suzhi_lingnan",
        era="北宋", figure="苏轼", theme="岭南流放",
        title="惠州·儋州·归途",
        description="苏轼晚年一贬再贬至岭南，却在蛮荒之地活出「此心安处是吾乡」的豁达。",
        difficulty=2, concepts=["延续", "身份认同", "道德判断"],
        prerequisites=["song_suzhi_banishment"], related=["song_suzhi_banishment"],
        tags=["苏轼", "岭南", "惠州", "儋州", "晚年"],
        unlock_condition="探索「乌台诗案·黄州」后解锁", icon="🌴", position=(4.5, 5.0),
    ),
    LearningNode(
        node_id="song_keju",
        era="北宋", figure="欧阳修", theme="科举与文人政治",
        title="宋代科举与文治",
        description="宋太祖以文人治国，科举大规模扩张，欧阳修主持的嘉祐二年堪称科举史上最强一届。",
        difficulty=2, concepts=["权力", "身份认同", "变革"],
        prerequisites=[], related=["song_wang_anshi", "song_suzhi_banishment"],
        tags=["科举", "文人", "欧阳修", "北宋", "制度"],
        unlock_condition="初始可探索", icon="🎓", position=(1.0, 5.0),
    ),

    # ── 唐代 ─────────────────────────────────────────────────
    LearningNode(
        node_id="tang_libai",
        era="唐代", figure="李白", theme="盛唐气象",
        title="李白与盛唐",
        description="李白的诗是盛唐气象的缩影：自由、浪漫、对权贵不屈服。",
        difficulty=1, concepts=["身份认同", "文明交流"],
        prerequisites=[], related=["tang_dufu", "tang_anshi_rebellion"],
        tags=["李白", "唐诗", "盛唐", "浪漫主义"],
        unlock_condition="初始可探索", icon="⚡", position=(0.5, 1.0),
    ),
    LearningNode(
        node_id="tang_dufu",
        era="唐代", figure="杜甫", theme="安史之乱与沉郁",
        title="杜甫与安史之乱",
        description="杜甫用「诗史」记录了大唐由盛转衰，个人命运与帝国命运紧密相连。",
        difficulty=2, concepts=["变革", "后果", "集体记忆"],
        prerequisites=["tang_libai"], related=["tang_libai", "tang_anshi_rebellion"],
        tags=["杜甫", "安史之乱", "沉郁顿挫", "唐诗"],
        unlock_condition="探索「李白与盛唐」后解锁", icon="🕯️", position=(1.5, 1.0),
    ),
    LearningNode(
        node_id="tang_anshi_rebellion",
        era="唐代", figure="唐玄宗/安禄山", theme="帝国转折",
        title="安史之乱：帝国的裂痕",
        description="755年的安史之乱是唐朝由盛而衰的关键节点，折射出中央集权的内在矛盾。",
        difficulty=4, concepts=["因果", "变革", "权力"],
        prerequisites=[], related=["tang_dufu"],
        tags=["安史之乱", "唐玄宗", "节度使", "藩镇"],
        unlock_condition="初始可探索", icon="⚔️", position=(0.0, 0.5),
    ),

    # ── 明代 ─────────────────────────────────────────────────
    LearningNode(
        node_id="ming_zheng_he",
        era="明代", figure="郑和", theme="大航海与文明交流",
        title="郑和下西洋",
        description="七下西洋，最远至非洲——与欧洲大航海同期，为何走向了不同的历史轨迹？",
        difficulty=3, concepts=["文明交流", "视角", "后果"],
        prerequisites=[], related=["ming_haijin"],
        tags=["郑和", "下西洋", "海上丝路", "明代", "比较史"],
        unlock_condition="初始可探索", icon="⛵", position=(5.0, 2.0),
    ),
    LearningNode(
        node_id="ming_haijin",
        era="明代", figure="朱元璋/朱棣", theme="海禁与内向",
        title="海禁政策与内向转型",
        description="从开放的郑和时代到严厉的海禁，明代为何选择封闭？这个选择如何影响了此后三百年？",
        difficulty=4, concepts=["延续", "后果", "权力"],
        prerequisites=["ming_zheng_he"], related=["ming_zheng_he"],
        tags=["海禁", "明代", "朝贡体系", "内向"],
        unlock_condition="探索「郑和下西洋」后解锁", icon="🚧", position=(5.5, 3.0),
    ),

    # ── 近代 ─────────────────────────────────────────────────
    LearningNode(
        node_id="modern_opium_war",
        era="晚清", figure="林则徐/道光帝", theme="冲击与应对",
        title="鸦片战争：两个世界的碰撞",
        description="1840年，工业文明与农耕帝国的第一次正面交锋，清朝的世界观在炮火中崩塌。",
        difficulty=3, concepts=["变革", "视角", "因果"],
        prerequisites=[], related=["modern_self_strengthen"],
        tags=["鸦片战争", "林则徐", "道光", "晚清", "近代史"],
        unlock_condition="初始可探索", icon="💥", position=(7.0, 1.0),
    ),
    LearningNode(
        node_id="modern_self_strengthen",
        era="晚清", figure="李鸿章/张之洞", theme="自强运动",
        title="洋务运动：器物层面的现代化",
        description="「中体西用」——用西方技术保存中国制度，这条路走得通吗？",
        difficulty=4, concepts=["变革", "延续", "后果"],
        prerequisites=["modern_opium_war"], related=["modern_opium_war"],
        tags=["洋务运动", "李鸿章", "中体西用", "近代化"],
        unlock_condition="探索「鸦片战争」后解锁", icon="🏭", position=(7.5, 2.0),
    ),
]

# 快速索引
_NODE_INDEX: Dict[str, LearningNode] = {n.node_id: n for n in ALL_NODES}


# ═══════════════════════════════════════════════════════════════
# 推荐引擎
# ═══════════════════════════════════════════════════════════════

class LearningMapEngine:
    """
    个性化学习路径推荐引擎。
    综合以下维度打分：
      - 前置节点已探索（可达性）
      - 与用户兴趣主题相近（偏好）
      - 与薄弱大概念相关（补强）
      - 难度适配（不太难也不太简单）
      - 多样性（避免连续推同一时代）
    """

    def recommend(
        self,
        profile: UserProfile,
        n: int = 3,
        exclude_explored: bool = True,
    ) -> List[LearningNode]:
        explored_ids = set(profile.explored_eras)  # 已探索时代作为代理
        explored_figures = set(profile.explored_figures)
        explored_themes = set(profile.explored_themes)

        # 判断节点是否已探索
        def _is_explored(node: LearningNode) -> bool:
            return (
                node.era in profile.explored_eras
                and node.figure in profile.explored_figures
            )

        # 判断节点是否可达（前置节点已探索）
        def _is_reachable(node: LearningNode) -> bool:
            if not node.prerequisites:
                return True
            for pre_id in node.prerequisites:
                pre = _NODE_INDEX.get(pre_id)
                if pre and _is_explored(pre):
                    return True
            return False

        candidates = [
            n for n in ALL_NODES
            if (not exclude_explored or not _is_explored(n)) and _is_reachable(n)
        ]
        if not candidates:
            candidates = [n for n in ALL_NODES if not _is_explored(n)]

        # 薄弱概念（知识掌握度最低的概念）
        weak_concepts: Set[str] = set()
        if profile.knowledge_mastery:
            sorted_kv = sorted(
                profile.knowledge_mastery.items(),
                key=lambda x: x[1].get("score", 50),
            )
            weak_concepts = {k for k, _ in sorted_kv[:4]}

        def _score(node: LearningNode) -> float:
            score = 0.0
            # 偏好：已探索主题/人物附近
            if node.theme in explored_themes:
                score += 1.5
            if node.era in profile.explored_eras:
                score += 1.0
            # 薄弱补强
            for c in node.concepts:
                if c in weak_concepts:
                    score += 2.0
            # 难度适配（偏向中等难度）
            total_sessions = max(1, profile.total_sessions)
            ideal_diff = min(5, 1 + total_sessions // 3)
            score -= abs(node.difficulty - ideal_diff) * 0.5
            # 多样性：优先不同时代
            if node.era not in profile.explored_eras:
                score += 1.2
            return score

        ranked = sorted(candidates, key=_score, reverse=True)
        return ranked[:n]

    def get_map_data(self, profile: UserProfile) -> dict:
        """
        返回完整地图数据（所有节点 + 探索状态 + 推荐标记）。
        """
        recommended = self.recommend(profile, n=3)
        rec_ids = {n.node_id for n in recommended}

        def _is_explored(node: LearningNode) -> bool:
            return (
                node.era in profile.explored_eras
                and node.figure in profile.explored_figures
            )

        nodes_data = [
            n.to_dict(
                explored=_is_explored(n),
                recommended=n.node_id in rec_ids,
            )
            for n in ALL_NODES
        ]
        edges = []
        for node in ALL_NODES:
            for pre_id in node.prerequisites:
                edges.append({"source": pre_id, "target": node.node_id, "type": "prerequisite"})
            for rel_id in node.related:
                if rel_id > node.node_id:  # 避免重复
                    edges.append({"source": node.node_id, "target": rel_id, "type": "related"})

        return {
            "nodes": nodes_data,
            "edges": edges,
            "recommended": [n.to_dict() for n in recommended],
            "stats": {
                "total": len(ALL_NODES),
                "explored": sum(1 for n in ALL_NODES if _is_explored(n)),
                "eras_explored": list(set(profile.explored_eras)),
            },
        }


# 全局单例
_map_engine = LearningMapEngine()


def get_map_engine() -> LearningMapEngine:
    return _map_engine
