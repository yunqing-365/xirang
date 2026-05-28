# community_engine.py
"""
息壤 · Phase 15C + 15D

Phase 15C · 社区与协作
  - 同一时空，不同学生的选择对比
  - 学生评论彼此的历史日记（匿名或署名）
  - 「集体因果树」：全班选择汇聚成一棵决策树

Phase 15D · 课程包 & 认证
  - 一键导出教案（Markdown 结构化格式）
  - 学生历史探究报告生成（AI辅助摘要 + 真实数据）
  - 探索成就认证（JSON Badge）
"""
import json
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from openai import AsyncOpenAI
from infra.resilience import llm_guard, result_cache, annotation_cache, cross_link_cache, perspective_cache, graceful_degradation
from config import get_settings

_settings = get_settings()
_client = AsyncOpenAI(api_key=_settings.API_KEY, base_url=_settings.BASE_URL)


# ════════════════════════════════════════════════════════════════
# Phase 15C · 社区协作
# ════════════════════════════════════════════════════════════════

@dataclass
class ChoiceRecord:
    """单个学生在某一分叉点的选择"""
    user_id: str
    display_name: str
    round_num: int
    choice_text: str
    npc_response_summary: str   # NPC 的回应摘要
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "user_id": self.user_id,
            "name": self.display_name,
            "round": self.round_num,
            "choice": self.choice_text,
            "response": self.npc_response_summary,
        }


@dataclass
class DiaryComment:
    """对历史日记的评论"""
    comment_id: str
    creation_id: str
    author_id: str
    author_name: str
    content: str
    is_anonymous: bool = False
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "id": self.comment_id,
            "creation_id": self.creation_id,
            "author": "匿名同学" if self.is_anonymous else self.author_name,
            "content": self.content,
            "timestamp": self.timestamp,
        }


class CommunityHub:
    """
    班级社区协作中心。
    共享：选择记录、日记评论、集体因果树。
    """

    def __init__(self, room_id: str):
        self.room_id = room_id
        # 所有学生在各回合的选择记录
        self._choice_records: List[ChoiceRecord] = []
        # 日记评论 {creation_id: [comments]}
        self._comments: Dict[str, List[DiaryComment]] = {}
        # 公开的创作 {creation_id: creation_dict}
        self._public_creations: Dict[str, dict] = {}
        self._counter = 0

    def _new_id(self) -> str:
        self._counter += 1
        return f"cmt_{self.room_id}_{self._counter}"

    # ── 选择对比 ──────────────────────────────────────────────

    def record_choice(
        self,
        user_id: str,
        display_name: str,
        round_num: int,
        choice_text: str,
        npc_response_summary: str = "",
    ) -> ChoiceRecord:
        record = ChoiceRecord(
            user_id=user_id,
            display_name=display_name,
            round_num=round_num,
            choice_text=choice_text,
            npc_response_summary=npc_response_summary,
        )
        self._choice_records.append(record)
        return record

    def get_choices_comparison(self, round_num: Optional[int] = None) -> dict:
        """
        获取某回合（或全部）所有学生的选择对比。
        """
        records = (
            [r for r in self._choice_records if r.round_num == round_num]
            if round_num is not None
            else self._choice_records
        )
        # 按选择文本聚合
        grouped: Dict[str, List[dict]] = {}
        for r in records:
            key = r.choice_text[:50]
            if key not in grouped:
                grouped[key] = []
            grouped[key].append(r.to_dict())

        return {
            "room_id": self.room_id,
            "round": round_num,
            "total_responses": len(records),
            "choices": [
                {
                    "choice": k,
                    "count": len(v),
                    "students": v,
                    "pct": round(len(v) / max(len(records), 1) * 100),
                }
                for k, v in sorted(grouped.items(), key=lambda x: -len(x[1]))
            ],
        }

    def get_collective_causal_tree(self) -> dict:
        """
        集体因果树：将全班所有选择汇聚成一棵决策树。
        节点=选择，边=时序，宽度=选择人数。
        """
        rounds = sorted(set(r.round_num for r in self._choice_records))
        nodes = []
        edges = []
        prev_node_ids: List[str] = ["root"]

        nodes.append({"id": "root", "label": "故事开始", "count": 1, "round": 0})

        for rnum in rounds:
            round_records = [r for r in self._choice_records if r.round_num == rnum]
            # 按选择聚合
            choice_groups: Dict[str, List[ChoiceRecord]] = {}
            for r in round_records:
                k = r.choice_text[:40]
                if k not in choice_groups:
                    choice_groups[k] = []
                choice_groups[k].append(r)

            new_ids = []
            for choice_text, members in choice_groups.items():
                nid = f"r{rnum}_{len(nodes)}"
                nodes.append({
                    "id": nid,
                    "label": choice_text,
                    "count": len(members),
                    "round": rnum,
                    "students": [m.display_name for m in members[:5]],
                })
                for prev_id in prev_node_ids:
                    edges.append({"source": prev_id, "target": nid, "weight": len(members)})
                new_ids.append(nid)
            prev_node_ids = new_ids

        return {"room_id": self.room_id, "nodes": nodes, "edges": edges}

    # ── 日记评论 ──────────────────────────────────────────────

    def publish_creation(self, creation_dict: dict, user_id: str):
        """将创作发布到班级社区"""
        cid = creation_dict.get("id", f"pub_{user_id}_{int(time.time())}")
        self._public_creations[cid] = {**creation_dict, "publisher": user_id}

    def add_comment(
        self,
        creation_id: str,
        author_id: str,
        author_name: str,
        content: str,
        anonymous: bool = False,
    ) -> DiaryComment:
        comment = DiaryComment(
            comment_id=self._new_id(),
            creation_id=creation_id,
            author_id=author_id,
            author_name=author_name,
            content=content,
            is_anonymous=anonymous,
        )
        if creation_id not in self._comments:
            self._comments[creation_id] = []
        self._comments[creation_id].append(comment)
        return comment

    def get_public_creations(self) -> List[dict]:
        return list(self._public_creations.values())

    def get_comments(self, creation_id: str) -> List[dict]:
        return [c.to_dict() for c in self._comments.get(creation_id, [])]


# ════════════════════════════════════════════════════════════════
# Phase 15D · 课程包 & 认证
# ════════════════════════════════════════════════════════════════

class CurriculumExporter:
    """
    课程包导出器：将一次历史体验转化为可交付的教育产品。
    """

    async def generate_lesson_plan(
        self,
        era: str,
        scene_desc: str,
        concepts_covered: List[str],
        cross_links: List[dict],
        inquiry_questions: List[dict],
        duration_minutes: int = 45,
    ) -> str:
        """
        生成结构化教案（Markdown 格式）。
        """
        concepts_str = "、".join(concepts_covered[:5])
        cross_str = "\n".join(
            f"- **{l.get('subject','学科')}**：{l.get('topic','')} — {l.get('hook','')}"
            for l in cross_links[:4]
        )
        inquiry_str = "\n".join(
            f"- （L{q.get('bloom_level',4)} {q.get('bloom_label','分析')}）{q.get('text','')}"
            for q in inquiry_questions[:3]
        )

        prompt = (
            f"请为以下历史沉浸体验生成一份完整教案（Markdown格式）：\n\n"
            f"时代：{era}\n场景：{scene_desc[:150]}\n"
            f"核心大概念：{concepts_str}\n"
            f"课时：{duration_minutes}分钟\n\n"
            f"跨学科连接点：\n{cross_str}\n\n"
            f"探究问题（供教师参考）：\n{inquiry_str}\n\n"
            f"教案结构要求：\n"
            f"1. 教学目标（知识/能力/情感三维）\n"
            f"2. 课前准备\n"
            f"3. 教学流程（含时间分配）\n"
            f"4. 核心问题设计\n"
            f"5. 评估方式\n"
            f"6. 延伸资源\n\n"
            f"风格：专业、实用，符合中国中学历史课标。"
        )
        try:
            resp = await _client.chat.completions.create(
                model=_settings.MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.4,
                timeout=40,
                max_tokens=1500,
            )
            return resp.choices[0].message.content
        except Exception as e:
            return f"# 教案生成失败\n\n错误：{e}\n\n请重试或手动编写教案。"

    async def generate_student_report(
        self,
        user_id: str,
        display_name: str,
        session_summary: dict,
        profile_summary: dict,
    ) -> str:
        """
        生成学生历史探究报告（AI摘要 + 真实数据）。
        """
        concepts = ", ".join(session_summary.get("concepts", []))
        creations = session_summary.get("creations_count", 0)
        citations = session_summary.get("citations_count", 0)
        socratic_turns = session_summary.get("socratic_turns", 0)
        evidence_score = session_summary.get("evidence_score", 0)
        bookmarked = session_summary.get("bookmarked", 0)

        prompt = (
            f"为学生「{display_name}」生成一份历史探究学习报告（约200字）。\n\n"
            f"本次探究数据：\n"
            f"- 触碰的历史学大概念：{concepts or '暂无记录'}\n"
            f"- 史料引用次数：{citations}次，实证积分：{evidence_score}分\n"
            f"- 苏格拉底对话轮次：{socratic_turns}轮\n"
            f"- 收藏的好问题：{bookmarked}道\n"
            f"- 创作作品：{creations}件\n\n"
            f"报告要求：\n"
            f"1. 肯定学生的探究过程（具体，不能空泛）\n"
            f"2. 指出1–2个值得继续深入的方向\n"
            f"3. 提出1个给学生的学习建议\n"
            f"语气：温暖、鼓励、专业。不要评分，不要排名。"
        )
        try:
            resp = await _client.chat.completions.create(
                model=_settings.MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.6,
                timeout=25,
                max_tokens=400,
            )
            body = resp.choices[0].message.content
        except Exception:
            body = "本次探究表现积极，建议继续深入探索更多历史时空。"

        # 结构化报告
        now = time.strftime("%Y年%m月%d日")
        return (
            f"# 历史探究学习报告\n\n"
            f"**学生**：{display_name}　　**日期**：{now}\n\n"
            f"---\n\n"
            f"## 本次探究概览\n\n"
            f"| 维度 | 数据 |\n|---|---|\n"
            f"| 触碰大概念 | {concepts or '—'} |\n"
            f"| 史料引用 | {citations} 次（实证积分 {evidence_score} 分）|\n"
            f"| 探究对话 | {socratic_turns} 轮苏格拉底追问 |\n"
            f"| 问题收藏 | {bookmarked} 道好问题 |\n"
            f"| 创作产出 | {creations} 件 |\n\n"
            f"## 评语\n\n{body}\n\n"
            f"---\n\n*由息壤历史沉浸学习平台生成*"
        )

    def generate_badge(
        self,
        user_id: str,
        display_name: str,
        era: str,
        achievements: List[str],
    ) -> dict:
        """
        生成探索成就徽章（JSON Badge 格式，兼容 Open Badges 标准）。
        """
        badge_id = f"badge_{user_id}_{int(time.time())}"
        return {
            "@context": "https://w3id.org/openbadges/v2",
            "type": "Assertion",
            "id": badge_id,
            "recipient": {"type": "id", "identity": user_id},
            "badge": {
                "type": "BadgeClass",
                "name": f"息壤{era}探索者",
                "description": f"完成了{era}时空的历史沉浸探索，触碰了真实的历史学思维。",
                "criteria": {"narrative": "完成至少5回合探索，触碰至少2个历史学大概念"},
                "issuer": {
                    "type": "Issuer",
                    "name": "息壤历史沉浸学习平台",
                    "url": "https://xirang.edu",
                },
            },
            "issuedOn": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "evidence": [
                {"type": "Evidence", "narrative": a}
                for a in achievements
            ],
            "recipient_name": display_name,
            "era": era,
        }


# ── 全局注册表 ────────────────────────────────────────────────

_community_hubs: Dict[str, CommunityHub] = {}
_curriculum_exporter = CurriculumExporter()


def get_community_hub(room_id: str) -> CommunityHub:
    if room_id not in _community_hubs:
        _community_hubs[room_id] = CommunityHub(room_id)
    return _community_hubs[room_id]


def get_curriculum_exporter() -> CurriculumExporter:
    return _curriculum_exporter
