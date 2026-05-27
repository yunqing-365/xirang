# creative_engine.py
"""
息壤 · Phase 14B + 14C

Phase 14B · 学科跨界连接层
  历史 × 语文 × 地理 × 经济
  苏轼贬谪 → 诗词鉴赏 + 宋代地理 + 政治制度
  每个场景自动标注跨学科连接点，一键跳入

Phase 14C · 学生创作输出系统
  理解的最高证明是创作。
  - 历史日记（第一人称，AI辅助润色）
  - 给 NPC 写一封信或一首词
  - 历史报道（用现代媒体形式报道史事）
  - 创作成果导出（Markdown / 可分享卡片）
"""
import asyncio
import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, AsyncGenerator

from openai import AsyncOpenAI

from config import get_settings

_settings = get_settings()
_client = AsyncOpenAI(api_key=_settings.API_KEY, base_url=_settings.BASE_URL)


# ════════════════════════════════════════════════════════════════
# Phase 14B · 学科跨界连接层
# ════════════════════════════════════════════════════════════════

class Subject(str, Enum):
    HISTORY  = "历史"
    CHINESE  = "语文"
    GEOGRAPHY = "地理"
    POLITICS = "政治"
    ECONOMICS = "经济"
    PHILOSOPHY = "哲学"
    ART      = "艺术"


@dataclass
class CrossLink:
    """一个跨学科连接点"""
    subject: Subject
    topic: str                  # 连接的具体知识点（如「词的格律与意境」）
    hook: str                   # 触发句（为什么此刻联系到这个学科）
    content: str                # 核心内容（150字以内）
    inquiry: str                # 配套探究问题
    curriculum_note: str        # 课标对应说明（可选）
    icon: str                   # 前端显示图标

    def to_dict(self) -> dict:
        return {
            "subject": self.subject.value,
            "topic": self.topic,
            "hook": self.hook,
            "content": self.content,
            "inquiry": self.inquiry,
            "curriculum_note": self.curriculum_note,
            "icon": self.icon,
        }


# 预置连接点库（北宋·苏轼时代，可按时代扩展）
_PRESET_LINKS: Dict[str, List[CrossLink]] = {
    "苏轼贬谪": [
        CrossLink(
            subject=Subject.CHINESE,
            topic="词的格律与情感表达",
            hook="苏轼在黄州创作了《念奴娇·赤壁怀古》，这是豪放词的里程碑。",
            content=(
                "词有固定格律（词牌），每个词牌规定了字数、平仄、押韵。"
                "但苏轼突破了词「婉约」的传统，在《念奴娇》中用豪放的语言写政治失意，"
                "开创「豪放派」。他的情感转化——从愤懑到旷达——就在这首词的意象变化中。"
            ),
            inquiry="词中的「人生如梦，一尊还酹江月」表达了什么情感？这是真的豁达还是一种自我安慰？",
            curriculum_note="对应人教版语文必修三《念奴娇·赤壁怀古》",
            icon="📖",
        ),
        CrossLink(
            subject=Subject.GEOGRAPHY,
            topic="黄州·赤壁·长江地理",
            hook="苏轼在黄州两次游览赤壁，写下千古名篇。「赤壁」在哪里？",
            content=(
                "黄州（今湖北黄冈）位于长江中游北岸。苏轼所游的「赤壁」"
                "其实是黄州赤鼻矶，并非真正的三国赤壁之战发生地（在今湖北赤壁市）。"
                "长江在此地拐弯，形成独特的地貌。苏轼的「误认」恰好成就了两篇赤壁赋。"
            ),
            inquiry="苏轼知道这里可能不是真正的赤壁战场，为什么他还要在这里「怀古」？",
            curriculum_note="对应初中历史地理：长江流域 / 三国史地",
            icon="🗺️",
        ),
        CrossLink(
            subject=Subject.POLITICS,
            topic="北宋官员贬谪制度与党争",
            hook="苏轼被贬，是「乌台诗案」，也是新旧党争的结果。",
            content=(
                "北宋中期，以王安石为首的「新党」推行变法，"
                "以苏轼为代表的「旧党」（保守派）反对。"
                "党争导致政策随皇帝更迭而剧烈摇摆，官员在贬谪和复起之间反复。"
                "苏轼一生经历神宗（被贬）→ 哲宗前期（复起）→ 哲宗后期（再贬）的循环。"
            ),
            inquiry="为什么一个文人写诗会引发政治危机？皇权时代，文字的力量边界在哪里？",
            curriculum_note="对应人教版历史必修一：北宋政治制度 / 王安石变法",
            icon="⚖️",
        ),
    ],
    "科举制度": [
        CrossLink(
            subject=Subject.POLITICS,
            topic="科举制的社会流动功能",
            hook="苏轼通过科举入仕——这在宋代意味着什么？",
            content=(
                "宋代科举不限制门第，理论上任何男性都可参考。"
                "这创造了历史上少见的社会流动渠道：寒门子弟可凭才学进入统治阶层。"
                "苏轼的父亲苏洵是晚年才学成的文人，苏轼和苏辙兄弟都在21岁前中进士。"
                "宋代科举人数远超前朝，官员队伍的文人化程度也随之提高。"
            ),
            inquiry="科举制是否真的实现了社会公平？有哪些人被它排除在外？",
            curriculum_note="对应历史：科举制度演变 / 宋代社会阶层",
            icon="🎓",
        ),
    ],
    "王安石变法": [
        CrossLink(
            subject=Subject.ECONOMICS,
            topic="青苗法与货币经济",
            hook="王安石变法的核心是财政改革——他在解决什么问题？",
            content=(
                "北宋国家财政面临「冗官、冗兵、冗费」三大问题。"
                "王安石的青苗法让政府向农民提供低息贷款，"
                "打破民间高利贷的垄断。这实质上是一种国家信贷制度的尝试。"
                "但执行过程中，地方官员强制摊派，反而加重了农民负担。"
                "这是「好制度」遭遇「坏执行」的历史案例。"
            ),
            inquiry="一个政策在理论上合理，在执行中失败，问题出在哪里？今天有类似的例子吗？",
            curriculum_note="对应历史：王安石变法 / 经济政策与社会影响",
            icon="💰",
        ),
    ],
}


class CrossDisciplineEngine:
    """
    学科跨界连接引擎。
    从当前历史场景自动识别跨学科连接点，并提供深度内容。
    """

    def get_preset_links(self, scene_keywords: List[str]) -> List[CrossLink]:
        """基于场景关键词返回预置连接点"""
        links = []
        for keyword in scene_keywords:
            for key, preset_links in _PRESET_LINKS.items():
                if keyword in key or key in keyword:
                    links.extend(preset_links)
        # 去重（按学科）
        seen = set()
        unique = []
        for link in links:
            if link.subject not in seen:
                seen.add(link.subject)
                unique.append(link)
        return unique

    async def generate_links(
        self,
        scene_desc: str,
        era: str,
        dialogue_summary: str,
        focus_subjects: Optional[List[str]] = None,
    ) -> List[CrossLink]:
        """
        LLM 动态生成跨学科连接点（当预置库无匹配时调用）。
        """
        subject_list = focus_subjects or [s.value for s in Subject]
        prompt = (
            f"当前历史场景：{scene_desc[:200]}\n"
            f"时代：{era}\n"
            f"对话摘要：{dialogue_summary[:200]}\n\n"
            f"请为这个历史场景识别3个最有价值的跨学科连接点，"
            f"从以下学科中选：{'/'.join(subject_list)}\n\n"
            f"每个连接点必须：真实关联当前场景，有具体知识内容，不能空泛。\n"
            f"以纯JSON数组输出：\n"
            '[{{"subject": "学科", "topic": "具体知识点（10字以内）", '
            '"hook": "触发句（20字以内）", '
            '"content": "核心内容（100字以内）", '
            '"inquiry": "探究问题（1句）", '
            '"curriculum_note": "课标对应（可选）", '
            '"icon": "emoji图标"}}]'
        )
        try:
            resp = await _client.chat.completions.create(
                model=_settings.MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.6,
                timeout=25,
                max_tokens=700,
            )
            items = json.loads(_strip_json(resp.choices[0].message.content))
            links = []
            for item in items:
                subj_map = {s.value: s for s in Subject}
                subj = subj_map.get(item.get("subject", "历史"), Subject.HISTORY)
                links.append(CrossLink(
                    subject=subj,
                    topic=item.get("topic", ""),
                    hook=item.get("hook", ""),
                    content=item.get("content", ""),
                    inquiry=item.get("inquiry", ""),
                    curriculum_note=item.get("curriculum_note", ""),
                    icon=item.get("icon", "📚"),
                ))
            return links
        except Exception as e:
            print(f"⚠️ [跨学科] LLM生成失败: {e}")
            return []

    async def smart_links(
        self,
        scene_desc: str,
        era: str,
        dialogue_summary: str,
        scene_keywords: Optional[List[str]] = None,
        focus_subjects: Optional[List[str]] = None,
    ) -> List[dict]:
        """
        智能连接：先查预置库，不足则 LLM 补充。
        """
        links = self.get_preset_links(scene_keywords or [])
        if len(links) < 2:
            llm_links = await self.generate_links(
                scene_desc, era, dialogue_summary, focus_subjects
            )
            # 合并，按学科去重
            existing_subjects = {l.subject for l in links}
            for ll in llm_links:
                if ll.subject not in existing_subjects:
                    links.append(ll)
                    existing_subjects.add(ll.subject)
        return [l.to_dict() for l in links[:5]]


# ════════════════════════════════════════════════════════════════
# Phase 14C · 学生创作输出系统
# ════════════════════════════════════════════════════════════════

class CreationType(str, Enum):
    DIARY      = "历史日记"      # 以第一人称写日记
    LETTER     = "历史书信"      # 给 NPC 写信
    POEM       = "仿古词作"      # 写一首词
    NEWS       = "历史报道"      # 用现代媒体形式报道
    ESSAY      = "历史短论"      # 短篇历史分析


CREATION_ICONS = {
    CreationType.DIARY:  "📔",
    CreationType.LETTER: "✉️",
    CreationType.POEM:   "🖊️",
    CreationType.NEWS:   "📰",
    CreationType.ESSAY:  "📝",
}


@dataclass
class StudentCreation:
    """学生的一件创作"""
    creation_id: str
    creation_type: CreationType
    title: str
    draft: str              # 学生原稿
    ai_polished: str        # AI润色版（供参考）
    era: str
    related_npc: str
    session_id: str
    user_id: str
    created_at: float = field(default_factory=time.time)
    word_count: int = 0
    share_card_url: str = ""    # 生成分享卡后的地址

    def to_dict(self) -> dict:
        return {
            "id": self.creation_id,
            "type": self.creation_type.value,
            "icon": CREATION_ICONS[self.creation_type],
            "title": self.title,
            "draft": self.draft,
            "ai_polished": self.ai_polished,
            "era": self.era,
            "npc": self.related_npc,
            "word_count": self.word_count,
            "created_at": self.created_at,
        }

    def to_markdown(self) -> str:
        return (
            f"# {self.title}\n\n"
            f"> 类型：{self.creation_type.value} · 时代：{self.era} · 角色：{self.related_npc}\n\n"
            f"## 我的创作\n\n{self.draft}\n\n"
            f"---\n\n## AI 润色参考\n\n{self.ai_polished}\n\n"
            f"*创作于息壤历史沉浸体验*"
        )


class CreationEngine:
    """
    学生创作输出引擎。
    支持多种体裁，AI 辅助润色（不替代，只参考）。
    """

    # 体裁提示词模板
    _PROMPTS = {
        CreationType.DIARY: (
            "你是一位{era}时期亲历了{event}的人。\n"
            "请以第一人称写一篇当天的日记（150字以内），"
            "语气要真实、私密，包含具体细节和内心感受。\n"
            "不要太「文学」，就像一个真实的人记录当天的事。\n"
            "参考学生原稿：\n{draft}"
        ),
        CreationType.LETTER: (
            "你要帮助学生改进这封写给{npc}的信。\n"
            "时代背景：{era}，写信人的立场是：{perspective}。\n"
            "要求：符合时代礼节，情感真实，有具体内容而非泛泛而谈。\n"
            "学生原稿：\n{draft}\n\n"
            "给出润色版（保留学生的核心想法，改进表达和历史感）："
        ),
        CreationType.POEM: (
            "参考学生这首仿{era}词/诗的草稿，帮助润色：\n{draft}\n\n"
            "要求：保留学生的意象和情感核心，改进平仄/押韵/词汇的历史感。\n"
            "给出润色版，并简短说明改动原因（3句话以内）："
        ),
        CreationType.NEWS: (
            "学生要用现代媒体风格报道这段历史事件：{event}\n"
            "时代原背景：{era}\n"
            "学生草稿：\n{draft}\n\n"
            "请润色为一篇现代新闻报道风格的文章（标题+200字以内正文），"
            "保留核心史实，但用今天的新闻语言表达。"
        ),
        CreationType.ESSAY: (
            "学生写了一篇关于{event}的历史短论：\n{draft}\n\n"
            "请给出润色建议版（150字以内），"
            "保留学生的论点，改进论据的充分性和表达的清晰度。"
        ),
    }

    async def polish_stream(
        self,
        creation_type: CreationType,
        draft: str,
        era: str,
        npc: str = "",
        event: str = "",
        perspective: str = "旁观者",
    ) -> AsyncGenerator[str, None]:
        """
        AI 润色流式输出（打字机效果）。
        """
        template = self._PROMPTS.get(creation_type, self._PROMPTS[CreationType.DIARY])
        prompt = template.format(
            era=era, event=event or "这段历史",
            npc=npc, draft=draft, perspective=perspective,
        )
        full = ""
        try:
            stream = await _client.chat.completions.create(
                model=_settings.MODEL_NAME,
                messages=[
                    {"role": "system", "content": "你是一位温和的历史写作辅导老师，尊重学生的原创，只做锦上添花的改进。"},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
                stream=True,
                timeout=40,
                max_tokens=500,
            )
            async for chunk in stream:
                delta = chunk.choices[0].delta.content or ""
                if delta:
                    full += delta
                    yield delta
        except Exception as e:
            err = f"（润色引擎暂时离线：{e}）"
            yield err

    async def generate_title(self, creation_type: CreationType, draft: str, era: str) -> str:
        """为创作自动生成标题"""
        prompt = (
            f"为以下{creation_type.value}生成一个简短的标题（10字以内，有文学感）：\n{draft[:200]}"
        )
        try:
            resp = await _client.chat.completions.create(
                model=_settings.MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.8,
                timeout=10,
                max_tokens=30,
            )
            return resp.choices[0].message.content.strip().strip('「』「』《》')
        except Exception:
            return f"{era}{creation_type.value}"

    def build_share_card_data(self, creation: StudentCreation) -> dict:
        """构建分享卡所需数据（前端渲染）"""
        return {
            "type": creation.creation_type.value,
            "icon": CREATION_ICONS[creation.creation_type],
            "title": creation.title,
            "text": creation.draft[:200],
            "era": creation.era,
            "npc": creation.related_npc,
            "word_count": creation.word_count,
            "watermark": "息壤历史沉浸体验",
        }


# ════════════════════════════════════════════════════════════════
# 会话级创作管理器
# ════════════════════════════════════════════════════════════════

class CreativeSession:
    """会话级创作容器：管理当前学生的所有创作"""

    def __init__(self, session_id: str, user_id: str):
        self.session_id = session_id
        self.user_id = user_id
        self.engine = CreationEngine()
        self.cross_engine = CrossDisciplineEngine()
        self.creations: List[StudentCreation] = []
        self._counter = 0

    def _new_id(self) -> str:
        self._counter += 1
        return f"creation_{self.session_id}_{self._counter}"

    async def start_creation(
        self,
        creation_type: CreationType,
        draft: str,
        era: str,
        npc: str = "",
        event: str = "",
    ) -> StudentCreation:
        """
        接收学生草稿，异步润色，存入创作列表。
        """
        title = await self.engine.generate_title(creation_type, draft, era)
        # 收集润色结果
        polished = ""
        async for token in self.engine.polish_stream(creation_type, draft, era, npc, event):
            polished += token

        creation = StudentCreation(
            creation_id=self._new_id(),
            creation_type=creation_type,
            title=title,
            draft=draft,
            ai_polished=polished,
            era=era,
            related_npc=npc,
            session_id=self.session_id,
            user_id=self.user_id,
            word_count=len(draft),
        )
        self.creations.append(creation)
        return creation

    async def polish_stream(
        self,
        creation_type: CreationType,
        draft: str,
        era: str,
        npc: str = "",
        event: str = "",
    ) -> AsyncGenerator[str, None]:
        """直接流式润色（不保存）"""
        async for token in self.engine.polish_stream(creation_type, draft, era, npc, event):
            yield token

    async def get_cross_links(
        self,
        scene_desc: str,
        era: str,
        dialogue_summary: str,
        scene_keywords: Optional[List[str]] = None,
        focus_subjects: Optional[List[str]] = None,
    ) -> List[dict]:
        return await self.cross_engine.smart_links(
            scene_desc, era, dialogue_summary, scene_keywords, focus_subjects
        )

    def get_all_creations(self) -> List[dict]:
        return [c.to_dict() for c in self.creations]

    def get_creation(self, creation_id: str) -> Optional[StudentCreation]:
        return next((c for c in self.creations if c.creation_id == creation_id), None)

    def export_markdown(self, creation_id: str) -> Optional[str]:
        creation = self.get_creation(creation_id)
        return creation.to_markdown() if creation else None


# ── 全局会话注册表 ────────────────────────────────────────────

_creative_sessions: Dict[str, CreativeSession] = {}


def get_creative_session(session_id: str, user_id: str = "anonymous") -> CreativeSession:
    if session_id not in _creative_sessions:
        _creative_sessions[session_id] = CreativeSession(session_id, user_id)
    return _creative_sessions[session_id]


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
