# thinking_engine.py
"""
息壤 · Phase 13A · 历史思维可视化工具

核心理念：
  帮学生「看见自己的思维方式」——
  因果不是教出来的，是让学生在选择中自己感受到的。

四个工具模块：
  1. CausalChainBuilder   — 因果链图谱（玩家每次选择自动生成因果推演树）
  2. MultiPerspectiveMap  — 多视角对比（同一事件从不同身份的不同解读）
  3. SourceCredibilityRater — 史料可信度评级器（一手/二手/文学加工）
  4. ChronologyPuzzle     — 时序重建（打乱史事顺序让学生重新排列）

数据流：
  server.py → thinking_engine → 结构化 JSON → SSE / API → 前端可视化
"""
import asyncio
import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

from openai import AsyncOpenAI
from infra.resilience import llm_guard, result_cache, annotation_cache, cross_link_cache, perspective_cache, graceful_degradation

from config import get_settings

_settings = get_settings()
_client = AsyncOpenAI(api_key=_settings.API_KEY, base_url=_settings.BASE_URL)


# ═══════════════════════════════════════════════════════════════
# 1. 因果链图谱
# ═══════════════════════════════════════════════════════════════

@dataclass
class CausalNode:
    """因果链中的一个节点"""
    id: str
    text: str                          # 事件/决策描述
    node_type: str                     # "player_choice" | "npc_reaction" | "historical_outcome" | "root"
    round_num: int
    parent_id: Optional[str] = None
    children_ids: List[str] = field(default_factory=list)
    probability: Optional[float] = None   # 该结果的估算概率（0–1）
    historical_parallel: Optional[str] = None  # 真实历史中的类似节点

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "text": self.text,
            "type": self.node_type,
            "round": self.round_num,
            "parent_id": self.parent_id,
            "children": self.children_ids,
            "probability": self.probability,
            "parallel": self.historical_parallel,
        }


class CausalChainBuilder:
    """
    因果链图谱构建器。
    每次玩家选择 → 自动生成「这个选择会导致什么」的因果推演树。
    每次 NPC 响应 → 记录实际发生的结果。
    最终形成一棵完整的「决策树 + 实际历史路径」对比图。
    """

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.nodes: Dict[str, CausalNode] = {}
        self._counter = 0
        # 创建根节点
        root = CausalNode("root", "故事开始", "root", 0)
        self.nodes["root"] = root
        self._last_actual_id = "root"

    def _new_id(self, prefix: str = "n") -> str:
        self._counter += 1
        return f"{prefix}_{self._counter}"

    def add_player_choice(self, choice_text: str, round_num: int) -> str:
        """记录玩家做出的选择，返回节点 ID"""
        nid = self._new_id("choice")
        node = CausalNode(
            id=nid,
            text=choice_text,
            node_type="player_choice",
            round_num=round_num,
            parent_id=self._last_actual_id,
        )
        self.nodes[nid] = node
        # 挂到父节点
        if self._last_actual_id in self.nodes:
            self.nodes[self._last_actual_id].children_ids.append(nid)
        self._last_actual_id = nid
        return nid

    def add_outcome(
        self,
        outcome_text: str,
        round_num: int,
        parent_id: str,
        node_type: str = "historical_outcome",
        probability: Optional[float] = None,
    ) -> str:
        """记录一个结果节点（实际发生的或推演的）"""
        nid = self._new_id("out")
        node = CausalNode(
            id=nid,
            text=outcome_text,
            node_type=node_type,
            round_num=round_num,
            parent_id=parent_id,
            probability=probability,
        )
        self.nodes[nid] = node
        if parent_id in self.nodes:
            self.nodes[parent_id].children_ids.append(nid)
        self._last_actual_id = nid
        return nid

    async def generate_counterfactual_branches(
        self,
        choice_text: str,
        scene_desc: str,
        alternative_choices: List[str],
        round_num: int,
        parent_id: str,
    ) -> List[CausalNode]:
        """
        LLM 推演：如果选择了不同的选项，会发生什么？
        生成 2–3 个反事实分支（虚线节点）。
        """
        if not alternative_choices:
            return []

        alt_list = "\n".join(f"- {a}" for a in alternative_choices[:3])
        prompt = (
            f"历史场景：{scene_desc[:200]}\n"
            f"玩家实际选择：{choice_text}\n"
            f"其他可能的选择：\n{alt_list}\n\n"
            f"请为每个【其他选择】推演出最可能的直接后果（1–2句话），"
            f"以及与真实历史的关联。\n"
            f"以纯JSON数组输出，每项：\n"
            '{"choice": "选项文本", "outcome": "推演后果", '
            '"probability": 0.0-1.0, "historical_note": "真实历史中的类似情况（若有）"}'
        )
        try:
            resp = await _client.chat.completions.create(
                model=_settings.MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                timeout=20,
                max_tokens=500,
            )
            raw = _strip_json(resp.choices[0].message.content)
            items = json.loads(raw)
            branches = []
            for item in items:
                nid = self._new_id("cf")  # counterfactual
                node = CausalNode(
                    id=nid,
                    text=f"[反事实] {item.get('outcome', '')}",
                    node_type="counterfactual",
                    round_num=round_num,
                    parent_id=parent_id,
                    probability=item.get("probability"),
                    historical_parallel=item.get("historical_note"),
                )
                self.nodes[nid] = node
                if parent_id in self.nodes:
                    self.nodes[parent_id].children_ids.append(nid)
                branches.append(node)
            return branches
        except Exception as e:
            print(f"⚠️ [因果链] 反事实推演失败: {e}")
            return []

    def to_graph_data(self) -> dict:
        """返回可供前端 D3/ECharts 渲染的图数据"""
        return {
            "session_id": self.session_id,
            "nodes": [n.to_dict() for n in self.nodes.values()],
            "edges": [
                {"source": n.parent_id, "target": n.id}
                for n in self.nodes.values()
                if n.parent_id
            ],
            "actual_path": self._trace_actual_path(),
        }

    def _trace_actual_path(self) -> List[str]:
        """追踪实际发生的路径（从root到当前节点）"""
        path = []
        current = self._last_actual_id
        visited = set()
        while current and current not in visited:
            path.append(current)
            visited.add(current)
            node = self.nodes.get(current)
            if not node or not node.parent_id:
                break
            current = node.parent_id
        return list(reversed(path))


# ═══════════════════════════════════════════════════════════════
# 2. 多视角对比
# ═══════════════════════════════════════════════════════════════

@dataclass
class PerspectiveView:
    """单个身份视角对同一事件的解读"""
    identity: str          # 身份（如：苏轼、变法派官员、普通百姓、史书编者）
    stance: str            # 立场关键词
    interpretation: str    # 对事件的解读（100字以内）
    emotional_tone: str    # 情感基调
    key_concern: str       # 最关心的是什么

    def to_dict(self) -> dict:
        return {
            "identity": self.identity,
            "stance": self.stance,
            "interpretation": self.interpretation,
            "emotional_tone": self.emotional_tone,
            "key_concern": self.key_concern,
        }


class MultiPerspectiveMap:
    """多视角对比工具：同一事件，不同身份的不同解读"""

    async def generate(
        self,
        event_desc: str,
        scene_era: str,
        identities: Optional[List[str]] = None,
        n: int = 4,
    ) -> List[PerspectiveView]:
        """
        为同一历史事件生成 n 个不同身份的视角解读。
        identities: 指定身份列表（可选，不指定则 LLM 自行生成）
        """
        identity_str = (
            "、".join(identities) if identities
            else f"请自行选取{n}个在{scene_era}时代与此事件最相关的不同身份"
        )

        # 查缓存
        _cache_key = ResultCache.make_key("perspective", event_desc[:60], scene_era)
        _cached = await perspective_cache.get(_cache_key)
        if _cached is not None:
            return _cached

        prompt = (
            f"历史事件：{event_desc[:300]}\n"
            f"时代背景：{scene_era}\n"
            f"请从以下不同身份出发，各自解读这件事：{identity_str}\n\n"
            f"每个视角必须真实体现该身份的利益、情感和认知局限，\n"
            f"不要让所有人说差不多的话。\n\n"
            f"以纯JSON数组输出，每项：\n"
            '{"identity": "身份", "stance": "立场关键词（2-4字）", '
            '"interpretation": "解读（60字以内）", '
            '"emotional_tone": "情感基调（2-4字）", '
            '"key_concern": "最关心的是（10字以内）"}'
        )
        try:
            resp = await llm_guard.call(
                _client.chat.completions.create,
                model=_settings.MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.8,
                timeout=25,
                max_tokens=800,
                fallback=None,
            )
            if resp is None:
                return []
            raw = _strip_json(resp.choices[0].message.content)
            items = json.loads(raw)
            views = [
                PerspectiveView(
                    identity=item["identity"],
                    stance=item.get("stance", ""),
                    interpretation=item.get("interpretation", ""),
                    emotional_tone=item.get("emotional_tone", ""),
                    key_concern=item.get("key_concern", ""),
                )
                for item in items
            ]
            await perspective_cache.set(_cache_key, views)
            return views
        except Exception as e:
            print(f"⚠️ [多视角] 生成失败: {e}")
            return []


# ═══════════════════════════════════════════════════════════════
# 3. 史料可信度评级器
# ═══════════════════════════════════════════════════════════════

class SourceType(str, Enum):
    PRIMARY    = "一手史料"    # 当事人直接记录
    SECONDARY  = "二手史料"    # 后人整理转述
    LITERARY   = "文学加工"    # 艺术化处理，史实成分不确定
    ORAL       = "口述传说"    # 民间流传，真实性存疑
    UNKNOWN    = "来源不明"


@dataclass
class SourceCredibilityResult:
    source_text: str           # 被评估的史料文本
    source_type: SourceType
    credibility_score: int     # 0–100
    time_gap_years: Optional[int]  # 记录距事件发生的时间差
    author_stance: str         # 作者立场分析
    strengths: List[str]       # 可信之处
    weaknesses: List[str]      # 存疑之处
    study_questions: List[str] # 使用这份史料前应追问的问题

    def to_dict(self) -> dict:
        return {
            "source_type": self.source_type.value,
            "credibility_score": self.credibility_score,
            "time_gap_years": self.time_gap_years,
            "author_stance": self.author_stance,
            "strengths": self.strengths,
            "weaknesses": self.weaknesses,
            "study_questions": self.study_questions,
        }


class SourceCredibilityRater:
    """史料可信度评级器"""

    # 本地快速评级规则（免 LLM，即时响应）
    _QUICK_RULES = {
        "奏折": (SourceType.PRIMARY, 70, "官方记录，有政治立场"),
        "日记": (SourceType.PRIMARY, 80, "私人记录，较为真实但视角单一"),
        "实录": (SourceType.SECONDARY, 65, "官方修撰，有选择性"),
        "正史": (SourceType.SECONDARY, 60, "后代编纂，距事件有时间差"),
        "笔记": (SourceType.SECONDARY, 55, "私人笔记，真实性参差"),
        "诗词": (SourceType.LITERARY, 40, "文学表达，情感真实但史实需辨别"),
        "小说": (SourceType.LITERARY, 20, "艺术创作，史实成分极低"),
        "民间": (SourceType.ORAL, 30, "口耳相传，变形较大"),
        "传说": (SourceType.ORAL, 25, "民间传说，象征意义大于史实"),
    }

    def quick_rate(self, source_text: str, source_label: str = "") -> SourceCredibilityResult:
        """快速本地评级（关键词匹配，毫秒级）"""
        combined = source_text + source_label
        for keyword, (stype, score, stance) in self._QUICK_RULES.items():
            if keyword in combined:
                return SourceCredibilityResult(
                    source_text=source_text[:100],
                    source_type=stype,
                    credibility_score=score,
                    time_gap_years=None,
                    author_stance=stance,
                    strengths=self._default_strengths(stype),
                    weaknesses=self._default_weaknesses(stype),
                    study_questions=self._default_questions(stype),
                )
        return SourceCredibilityResult(
            source_text=source_text[:100],
            source_type=SourceType.UNKNOWN,
            credibility_score=50,
            time_gap_years=None,
            author_stance="来源不明，需进一步考证",
            strengths=["保留了某一时期的信息"],
            weaknesses=["作者身份不明", "记录时间不明", "无法交叉验证"],
            study_questions=["这份资料从何而来？", "谁记录了它？", "目的是什么？"],
        )

    async def deep_rate(self, source_text: str, era: str) -> SourceCredibilityResult:
        """LLM 深度评级（10–15秒，但更精准）"""
        prompt = (
            f"请对以下历史资料进行史学评级分析：\n\n"
            f"时代背景：{era}\n"
            f"资料内容：{source_text[:400]}\n\n"
            f"请以纯JSON输出：\n"
            '{"source_type": "一手史料/二手史料/文学加工/口述传说/来源不明", '
            '"credibility_score": 0-100, '
            '"time_gap_years": 数字或null, '
            '"author_stance": "作者立场分析（30字以内）", '
            '"strengths": ["可信之处1", "可信之处2"], '
            '"weaknesses": ["存疑之处1", "存疑之处2"], '
            '"study_questions": ["使用前应追问的问题1", "问题2", "问题3"]}'
        )
        try:
            resp = await _client.chat.completions.create(
                model=_settings.MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                timeout=20,
                max_tokens=600,
            )
            raw = _strip_json(resp.choices[0].message.content)
            data = json.loads(raw)
            stype_map = {v.value: v for v in SourceType}
            stype = stype_map.get(data.get("source_type", ""), SourceType.UNKNOWN)
            return SourceCredibilityResult(
                source_text=source_text[:100],
                source_type=stype,
                credibility_score=int(data.get("credibility_score", 50)),
                time_gap_years=data.get("time_gap_years"),
                author_stance=data.get("author_stance", ""),
                strengths=data.get("strengths", []),
                weaknesses=data.get("weaknesses", []),
                study_questions=data.get("study_questions", []),
            )
        except Exception as e:
            print(f"⚠️ [史料评级] 深度分析失败: {e}")
            return self.quick_rate(source_text)

    @staticmethod
    def _default_strengths(stype: SourceType) -> List[str]:
        return {
            SourceType.PRIMARY:   ["当事人亲历，细节丰富", "时间最接近事件本身"],
            SourceType.SECONDARY: ["经过整理，结构清晰", "可能综合多个来源"],
            SourceType.LITERARY:  ["保留了时代的情感氛围", "反映了当时人的心态"],
            SourceType.ORAL:      ["承载集体记忆", "可能保留官方记录遗漏的民间视角"],
            SourceType.UNKNOWN:   ["保留了某一时期的信息"],
        }.get(stype, [])

    @staticmethod
    def _default_weaknesses(stype: SourceType) -> List[str]:
        return {
            SourceType.PRIMARY:   ["视角单一", "可能受个人情感影响"],
            SourceType.SECONDARY: ["距事件有时间差", "编撰者有选择取舍"],
            SourceType.LITERARY:  ["史实细节可能被艺术化", "情节可能虚构"],
            SourceType.ORAL:      ["长期传播中细节变形", "缺乏文字记录佐证"],
            SourceType.UNKNOWN:   ["作者不明", "时间不明", "无法交叉验证"],
        }.get(stype, [])

    @staticmethod
    def _default_questions(stype: SourceType) -> List[str]:
        return {
            SourceType.PRIMARY:   ["作者当时的处境是什么？", "他有没有隐瞒或夸大的动机？"],
            SourceType.SECONDARY: ["编撰者距事件多少年？", "他依据了哪些更早的资料？"],
            SourceType.LITERARY:  ["哪些是真实史实，哪些是虚构？", "作者想传达什么价值观？"],
            SourceType.ORAL:      ["这个故事在什么群体中流传？", "传播中发生了哪些变化？"],
            SourceType.UNKNOWN:   ["这份资料从何而来？", "谁记录了它？目的是什么？"],
        }.get(stype, [])


# ═══════════════════════════════════════════════════════════════
# 4. 时序重建谜题
# ═══════════════════════════════════════════════════════════════

@dataclass
class ChronologyPuzzle:
    """时序重建练习：打乱史事顺序，让学生重新排列"""
    puzzle_id: str
    title: str
    era: str
    events: List[dict]        # [{"id", "text", "correct_order", "year_hint"}]
    shuffled_events: List[dict]
    difficulty: str           # "easy" | "medium" | "hard"
    time_limit_seconds: int = 120

    def check_answer(self, student_order: List[str]) -> dict:
        """
        检查学生的排序答案。
        student_order: 事件 id 列表（学生排列的顺序）
        """
        correct = sorted(self.events, key=lambda e: e["correct_order"])
        correct_ids = [e["id"] for e in correct]
        if student_order == correct_ids:
            return {"correct": True, "score": 100, "feedback": "完全正确！"}
        # 计算相对位置准确率
        score = sum(
            1 for i, eid in enumerate(student_order)
            if i < len(correct_ids) and eid == correct_ids[i]
        ) / len(correct_ids) * 100

        # 找出第一个错误
        for i, (s, c) in enumerate(zip(student_order, correct_ids)):
            if s != c:
                wrong_event = next((e for e in self.events if e["id"] == s), None)
                correct_event = next((e for e in self.events if e["id"] == c), None)
                feedback = (
                    f"第{i+1}位置放的是『{wrong_event['text'][:20]}』，"
                    f"但正确的应该是『{correct_event['text'][:20]}』。"
                    f"思考一下：为什么时序很重要？"
                )
                break
        else:
            feedback = f"大部分顺序正确，还有{len(correct_ids) - int(score/100*len(correct_ids))}个需要调整。"

        return {"correct": False, "score": round(score), "feedback": feedback}

    def to_dict(self) -> dict:
        return {
            "puzzle_id": self.puzzle_id,
            "title": self.title,
            "era": self.era,
            "shuffled_events": self.shuffled_events,
            "difficulty": self.difficulty,
            "time_limit_seconds": self.time_limit_seconds,
        }


class ChronologyPuzzleFactory:
    """时序谜题工厂：从当前场景自动生成谜题"""

    async def create_from_scene(
        self,
        scene_desc: str,
        era: str,
        known_milestones: List[str],
        difficulty: str = "medium",
    ) -> Optional[ChronologyPuzzle]:
        """从当前场景的里程碑事件自动生成时序谜题"""
        if len(known_milestones) < 3:
            return None   # 事件不够，无法生成有意义的谜题

        n_events = {"easy": 4, "medium": 6, "hard": 8}.get(difficulty, 6)
        milestones_text = "\n".join(f"- {m}" for m in known_milestones[-10:])

        prompt = (
            f"时代背景：{era}\n"
            f"场景：{scene_desc[:150]}\n"
            f"已知历史里程碑：\n{milestones_text}\n\n"
            f"请从以上事件中选取{min(n_events, len(known_milestones))}个，"
            f"或补充相关历史事件，生成一个时序排列谜题。\n"
            f"每个事件给出准确的历史顺序（correct_order从1开始）和模糊年份提示。\n"
            f"以纯JSON输出：\n"
            '{"title": "谜题标题", "events": [{"id": "e1", "text": "事件描述（20字以内）", '
            '"correct_order": 1, "year_hint": "模糊时间提示如\'熙宁初年\'"}]}'
        )
        try:
            resp = await _client.chat.completions.create(
                model=_settings.MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5,
                timeout=20,
                max_tokens=600,
            )
            raw = _strip_json(resp.choices[0].message.content)
            data = json.loads(raw)
            events = data.get("events", [])
            if len(events) < 3:
                return None

            import random
            shuffled = events.copy()
            random.shuffle(shuffled)

            return ChronologyPuzzle(
                puzzle_id=f"puzzle_{int(time.time())}",
                title=data.get("title", "历史时序重建"),
                era=era,
                events=events,
                shuffled_events=shuffled,
                difficulty=difficulty,
            )
        except Exception as e:
            print(f"⚠️ [时序谜题] 生成失败: {e}")
            return None


# ═══════════════════════════════════════════════════════════════
# 统一入口：ThinkingEngine
# ═══════════════════════════════════════════════════════════════

class ThinkingEngine:
    """
    会话级历史思维工具集管理器。
    server.py 通过此类访问所有 Phase 13A 功能。
    """

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.causal_builder = CausalChainBuilder(session_id)
        self.perspective_map = MultiPerspectiveMap()
        self.source_rater = SourceCredibilityRater()
        self.puzzle_factory = ChronologyPuzzleFactory()
        self._active_puzzle: Optional[ChronologyPuzzle] = None
        self._perspectives_cache: Dict[str, List[PerspectiveView]] = {}

    # ── 因果链 ────────────────────────────────────────────────

    def on_player_choice(self, choice_text: str, round_num: int) -> str:
        return self.causal_builder.add_player_choice(choice_text, round_num)

    def on_npc_outcome(self, outcome_text: str, round_num: int, parent_id: str) -> str:
        return self.causal_builder.add_outcome(outcome_text, round_num, parent_id)

    async def expand_counterfactuals(
        self,
        choice_text: str,
        scene_desc: str,
        alternatives: List[str],
        round_num: int,
        parent_id: str,
    ) -> List[dict]:
        branches = await self.causal_builder.generate_counterfactual_branches(
            choice_text, scene_desc, alternatives, round_num, parent_id
        )
        return [b.to_dict() for b in branches]

    def get_causal_graph(self) -> dict:
        return self.causal_builder.to_graph_data()

    # ── 多视角 ────────────────────────────────────────────────

    async def get_perspectives(
        self,
        event_desc: str,
        era: str,
        identities: Optional[List[str]] = None,
    ) -> List[dict]:
        cache_key = event_desc[:50]
        if cache_key in self._perspectives_cache:
            return [p.to_dict() for p in self._perspectives_cache[cache_key]]
        views = await self.perspective_map.generate(event_desc, era, identities)
        self._perspectives_cache[cache_key] = views
        return [v.to_dict() for v in views]

    # ── 史料评级 ──────────────────────────────────────────────

    def rate_source_quick(self, source_text: str, label: str = "") -> dict:
        result = self.source_rater.quick_rate(source_text, label)
        return result.to_dict()

    async def rate_source_deep(self, source_text: str, era: str) -> dict:
        result = await self.source_rater.deep_rate(source_text, era)
        return result.to_dict()

    # ── 时序谜题 ──────────────────────────────────────────────

    async def generate_puzzle(
        self,
        scene_desc: str,
        era: str,
        milestones: List[str],
        difficulty: str = "medium",
    ) -> Optional[dict]:
        puzzle = await self.puzzle_factory.create_from_scene(
            scene_desc, era, milestones, difficulty
        )
        if puzzle:
            self._active_puzzle = puzzle
            return puzzle.to_dict()
        return None

    def check_puzzle_answer(self, student_order: List[str]) -> dict:
        if not self._active_puzzle:
            return {"error": "没有活动谜题"}
        return self._active_puzzle.check_answer(student_order)


# ── 全局会话注册表 ────────────────────────────────────────────

_thinking_engines: Dict[str, ThinkingEngine] = {}


def get_thinking_engine(session_id: str) -> ThinkingEngine:
    if session_id not in _thinking_engines:
        _thinking_engines[session_id] = ThinkingEngine(session_id)
    return _thinking_engines[session_id]


# ── 工具函数 ──────────────────────────────────────────────────

def _strip_json(text: str) -> str:
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()
