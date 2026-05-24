# narrative/historical_triggers.py
"""
历史事件触发器
将真实历史节点编入数据库，当叙事推进到相应时间/地点/人物时自动引爆关键剧情。

触发机制：
  每次 stream_next 时调用 check_triggers()，扫描当前
  session 的 {year, location, agents, milestones}，
  匹配未触发的历史事件，返回触发事件列表。

触发效果：
  - 注入 narrator 消息（历史背景提示）
  - 推送 world_mood_change 事件
  - 将事件写入 milestones
  - 可选：触发 reflection 事件（人文思考）
"""
from __future__ import annotations
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

# ── 触发条件 ─────────────────────────────────────────────────
@dataclass
class TriggerCondition:
    year_min: Optional[int] = None    # 公元年范围（含）
    year_max: Optional[int] = None
    location_keywords: list[str] = field(default_factory=list)   # 地点关键词（任一匹配）
    agent_names: list[str] = field(default_factory=list)          # 必须在场的人物（任一）
    milestone_required: Optional[str] = None   # 前置里程碑（子串匹配）
    dialogue_keywords: list[str] = field(default_factory=list)    # 对话中出现的关键词

@dataclass
class HistoricalTrigger:
    id: str                        # 唯一标识
    era: str                       # 朝代（song/tang/ming…）
    event_name: str                # 事件名称
    condition: TriggerCondition
    # 触发效果
    narrator_text: str             # 向玩家展示的历史背景叙述
    mood_change: Optional[str] = None   # 触发后的世界情绪（TENSE/SOLEMN 等）
    reflection_insight: Optional[str] = None  # 人文反思洞见（可选）
    reflection_question: Optional[str] = None # 自省问题（可选）
    era_fact: Optional[str] = None    # 历史彩蛋
    priority: int = 5             # 1=必触发, 5=普通, 10=装饰性
    one_shot: bool = True         # 是否只触发一次
    tags: list[str] = field(default_factory=list)


# ── 历史事件数据库 ────────────────────────────────────────────
# 每个朝代精心设计若干高质量触发器
TRIGGER_DATABASE: list[HistoricalTrigger] = [

    # ━━━━━━ 北宋 / 苏轼黄州 ━━━━━━
    HistoricalTrigger(
        id="song_wutai_case",
        era="song",
        event_name="乌台诗案余波",
        condition=TriggerCondition(
            year_min=1079, year_max=1082,
            agent_names=["苏轼", "苏子瞻"],
            dialogue_keywords=["诗", "被贬", "谪", "罪"],
        ),
        narrator_text=(
            "【历史回响】元丰二年（1079），苏轼因诗被御史台构陷，身陷囹圄一百三十日。"
            "史称「乌台诗案」——台中多柏树，乌鸦聚居，故名。"
            "此案株连二十余人，苏轼死里逃生，贬黄州团练副使，"
            "实为软禁。彼时他四十四岁，人生正值最深的低谷。"
        ),
        mood_change="MELANCHOLY",
        reflection_insight="政治迫害与文学创作的悖论：最苦的流放，往往成就最伟大的作品。",
        reflection_question="若你身陷苏轼的处境，会选择沉默自保，还是继续言说？",
        era_fact='《赤壁赋》正是苏轼被贬黄州后所作，「乌台诗案」反而成就了他文学生涯的巅峰。',
        priority=1,
        tags=["政治", "文学", "苏轼"],
    ),

    HistoricalTrigger(
        id="song_cold_food_festival",
        era="song",
        event_name="寒食节",
        condition=TriggerCondition(
            year_min=1078, year_max=1090,
            location_keywords=["黄州", "东坡", "临皋亭"],
            dialogue_keywords=["寒食", "清明", "扫墓", "禁火"],
        ),
        narrator_text=(
            "【节气提示】寒食节，冬至后一百零五日，禁火三日。"
            "苏轼在黄州写下《寒食帖》：「今年又苦雨，两月秋萧瑟……」"
            "此帖被后世誉为「天下第三行书」，笔意沉郁，力透纸背。"
        ),
        mood_change="SOLEMN",
        reflection_insight="寒食节源自介子推之死，禁火悼念忠义。节日往往是集体记忆的锚点。",
        era_fact="苏轼《黄州寒食诗帖》现藏台北故宫博物院，是书法史上的旷世珍品。",
        priority=3,
        tags=["节日", "书法", "黄州"],
    ),

    HistoricalTrigger(
        id="song_red_cliff",
        era="song",
        event_name="赤壁之游",
        condition=TriggerCondition(
            year_min=1082, year_max=1083,
            location_keywords=["赤壁", "黄州", "江", "赤鼻矶"],
            agent_names=["苏轼", "苏子瞻"],
        ),
        narrator_text=(
            "【历史现场】元丰五年七月，苏轼与友人泛舟赤鼻矶，"
            "写下前《赤壁赋》；同年十月再游，作后赋。"
            "此地虽非真赤壁——真赤壁在今湖北蒲圻——"
            "然苏轼以文学之力，使黄州赤壁名垂千古。"
        ),
        mood_change="SERENE",
        reflection_insight="苏轼在赤壁悟出「物与我皆无尽也」——历史的沧桑感化为个人的超脱。",
        reflection_question="面对「哀吾生之须臾，羡长江之无穷」的感慨，你会如何回应？",
        era_fact="《前赤壁赋》中「客有吹洞箫者」据考证是苏轼的朋友道潜（参寥子）。",
        priority=2,
        tags=["赤壁", "文学", "苏轼"],
    ),

    HistoricalTrigger(
        id="song_wang_anshi_reform",
        era="song",
        event_name="王安石变法余波",
        condition=TriggerCondition(
            year_min=1069, year_max=1086,
            dialogue_keywords=["变法", "新法", "王安石", "青苗", "免役", "保甲"],
        ),
        narrator_text=(
            "【政治背景】熙宁年间（1068-1077），王安石主导变法，"
            "推行青苗法、募役法、保甲法等，意图富国强兵。"
            "苏轼因反对新法被迫出京，司马光则领导旧党。"
            "这场变法撕裂了整个北宋士大夫阶层，党争延续数十年。"
        ),
        mood_change="TENSE",
        reflection_insight="改革者与保守者之争，往往不是对错之争，而是节奏与代价之争。",
        era_fact="王安石与苏轼私交甚好，政见相左却互相敬重，是中国历史上少有的政敌惺惺相惜。",
        priority=3,
        tags=["政治", "变法", "党争"],
    ),

    # ━━━━━━ 唐代 ━━━━━━
    HistoricalTrigger(
        id="tang_anlushan_shadow",
        era="tang",
        event_name="安史之乱阴影",
        condition=TriggerCondition(
            year_min=755, year_max=800,
            dialogue_keywords=["安禄山", "史思明", "叛乱", "长安破", "玄宗", "流亡"],
        ),
        narrator_text=(
            "【历史震荡】天宝十四载（755），安禄山、史思明举兵叛乱，"
            "玄宗仓皇出奔，长安沦陷。历时八年的安史之乱，"
            "使大唐人口锐减近三分之一，从此盛唐气象一去不返。"
            "杜甫《春望》：「国破山河在，城春草木深。」"
        ),
        mood_change="CHAOTIC",
        reflection_insight="盛世的崩溃往往猝不及防——繁华与动荡之间，只隔一个决策的距离。",
        reflection_question="目睹时代的崩塌，诗人杜甫选择记录苦难。你会如何应对乱世？",
        era_fact="安史之乱后，唐朝藩镇割据局面形成，中央集权大幅削弱，直至唐亡。",
        priority=2,
        tags=["战乱", "唐代", "盛衰"],
    ),

    HistoricalTrigger(
        id="tang_keju_exam",
        era="tang",
        event_name="科举放榜",
        condition=TriggerCondition(
            year_min=618, year_max=907,
            location_keywords=["长安", "礼部", "贡院"],
            dialogue_keywords=["科举", "进士", "放榜", "金榜题名", "落第"],
        ),
        narrator_text=(
            "【制度背景】唐代科举以进士科最受重视，录取率不足百分之一。"
            "「三十老明经，五十少进士」——五十岁中进士仍算年轻。"
            "孟郊《登科后》：「春风得意马蹄疾，一日看尽长安花。」"
            "落第者则「年年下第东归去，文字才堪作鸟笼」。"
        ),
        mood_change="JOYFUL",
        era_fact="唐代270余年间，共举行进士科约260次，录取进士约6600人，平均每次约25人。",
        priority=4,
        tags=["科举", "制度", "唐代"],
    ),

    # ━━━━━━ 明代 ━━━━━━
    HistoricalTrigger(
        id="ming_donglin_party",
        era="ming",
        event_name="东林党争",
        condition=TriggerCondition(
            year_min=1604, year_max=1644,
            dialogue_keywords=["东林", "阉党", "魏忠贤", "党争", "清流"],
        ),
        narrator_text=(
            "【政治漩涡】万历年间，东林书院讲学，形成东林党，"
            "主张廉正奉公，力图匡正时弊。与之对立的魏忠贤阉党，"
            "把持朝政，大肆迫害东林人士。《五人墓碑记》所载，"
            "正是这场浩劫的缩影。明朝在党争中走向衰亡。"
        ),
        mood_change="TENSE",
        reflection_insight="党争的悲剧在于：双方都自认为道义在手，却共同耗尽了王朝的元气。",
        era_fact="东林党的名言「风声雨声读书声声声入耳，家事国事天下事事事关心」至今广为流传。",
        priority=3,
        tags=["政治", "明代", "党争"],
    ),

    HistoricalTrigger(
        id="ming_zhenghe_voyages",
        era="ming",
        event_name="郑和下西洋",
        condition=TriggerCondition(
            year_min=1405, year_max=1433,
            dialogue_keywords=["郑和", "下西洋", "宝船", "永乐", "海禁", "番邦"],
        ),
        narrator_text=(
            "【航海壮举】永乐三年至宣德八年（1405-1433），郑和七下西洋，"
            "统率最多两百余艘、官兵两万七千余人的舰队，"
            "最远到达东非海岸。宝船最大者长四十四丈，"
            "是当时世界上最大的木制船只。此后海禁重开，壮举湮没史册。"
        ),
        mood_change="JOYFUL",
        reflection_insight="郑和之后，中国走向内缩；哥伦布之后，欧洲走向扩张。历史的岔路口往往无声无息。",
        era_fact="郑和船队携带的「永乐通宝」铜钱至今仍在东非、印度沿海出土，是那段历史的无声证人。",
        priority=3,
        tags=["航海", "明代", "对外"],
    ),

    # ━━━━━━ 清代 ━━━━━━
    HistoricalTrigger(
        id="qing_qianlong_siku",
        era="qing",
        event_name="四库全书编纂",
        condition=TriggerCondition(
            year_min=1772, year_max=1782,
            dialogue_keywords=["四库", "全书", "纪昀", "乾隆", "文字狱", "禁书"],
        ),
        narrator_text=(
            "【文化工程】乾隆三十八年（1773），开馆编纂《四库全书》，"
            "历时十年，收录典籍三千四百余种，七万九千余卷。"
            "然其背后，是规模空前的文字狱与禁书运动——"
            "被销毁或删改的书籍，据估计超过收录数量的数倍。"
        ),
        mood_change="SOLEMN",
        reflection_insight="最宏大的文化整理工程，同时是最彻底的文化审查行动——两者并行不悖。",
        reflection_question="当「保存」与「删除」同时发生，我们得到的是历史，还是历史的一个版本？",
        era_fact="《四库全书》共抄写七套，现存最完整的是文渊阁本（台北故宫）和文津阁本（国家图书馆）。",
        priority=2,
        tags=["文化", "清代", "乾隆"],
    ),

    HistoricalTrigger(
        id="qing_opium_war_shadow",
        era="qing",
        event_name="鸦片战争前夕",
        condition=TriggerCondition(
            year_min=1820, year_max=1842,
            dialogue_keywords=["鸦片", "洋货", "英吉利", "海防", "禁烟", "虎门"],
        ),
        narrator_text=(
            "【历史前夜】道光年间，鸦片大量流入，白银外流，"
            "民间「十室九室」吸食成瘾。林则徐受命赴粤禁烟，"
            "虎门销烟二百余万斤。然列强炮舰随即叩关，"
            "中国近代史最深的伤口，从此裂开。"
        ),
        mood_change="TENSE",
        reflection_insight="林则徐看到了危机，却没有看到危机背后更深的结构性困境——这是那个时代无法逾越的认知边界。",
        era_fact="林则徐在虎门销烟后曾预言「终为中国之患者，其俄罗斯乎」，被认为是中国睁眼看世界的第一人。",
        priority=2,
        tags=["近代", "清代", "危机"],
    ),
]

# ── 索引 ─────────────────────────────────────────────────────
_BY_ERA: dict[str, list[HistoricalTrigger]] = {}
for _t in TRIGGER_DATABASE:
    _BY_ERA.setdefault(_t.era, []).append(_t)
_BY_ERA["unknown"] = TRIGGER_DATABASE  # 未知朝代扫全库


def get_triggers_for_era(era: str) -> list[HistoricalTrigger]:
    """返回指定朝代+通用触发器列表"""
    era_key = era.lower().replace(" ", "_")
    return _BY_ERA.get(era_key, TRIGGER_DATABASE)


# ── 触发器检查器 ─────────────────────────────────────────────
class TriggerChecker:
    """
    在每轮推演时检查是否有历史事件应该被触发。
    维护已触发事件集合（per-session），避免重复。
    """

    def __init__(self, era: str, triggered_ids: set[str] | None = None):
        self.era = era
        self.triggered_ids: set[str] = triggered_ids or set()
        self.triggers = get_triggers_for_era(era)
        # 按优先级排序
        self.triggers.sort(key=lambda t: t.priority)

    def check(
        self,
        year: int,
        location: str = "",
        agent_names: list[str] | None = None,
        dialogue: str = "",
        milestones: list[str] | None = None,
    ) -> list[HistoricalTrigger]:
        """
        返回本轮应触发的事件列表（已触发的自动跳过）。
        """
        fired: list[HistoricalTrigger] = []
        agent_names = agent_names or []
        milestones = milestones or []
        milestones_text = " ".join(milestones)

        for trigger in self.triggers:
            if trigger.id in self.triggered_ids:
                continue

            c = trigger.condition

            # 年份窗口
            if c.year_min and year < c.year_min:
                continue
            if c.year_max and year > c.year_max:
                continue

            # 地点关键词（任一匹配）
            if c.location_keywords:
                if not any(kw in location for kw in c.location_keywords):
                    continue

            # 必须在场人物（任一匹配）
            if c.agent_names:
                if not any(name in agent_names for name in c.agent_names):
                    continue

            # 前置里程碑
            if c.milestone_required:
                if c.milestone_required not in milestones_text:
                    continue

            # 对话关键词（任一匹配）
            if c.dialogue_keywords:
                if not any(kw in dialogue for kw in c.dialogue_keywords):
                    continue

            fired.append(trigger)
            if trigger.one_shot:
                self.triggered_ids.add(trigger.id)

        return fired

    def to_dict(self) -> dict:
        return {"era": self.era, "triggered_ids": list(self.triggered_ids)}

    @classmethod
    def from_dict(cls, d: dict) -> "TriggerChecker":
        return cls(era=d.get("era",""), triggered_ids=set(d.get("triggered_ids",[])))


if __name__ == "__main__":
    # 自测
    checker = TriggerChecker(era="song")
    fired = checker.check(
        year=1082,
        location="黄州赤壁",
        agent_names=["苏轼"],
        dialogue="今日与友泛舟赤壁，感慨万千，诗兴大发",
        milestones=[],
    )
    print(f"=== 触发器自测 | 触发 {len(fired)} 个事件 ===")
    for t in fired:
        print(f"\n  [{t.event_name}]")
        print(f"  {t.narrator_text[:60]}...")
        print(f"  情绪变化→ {t.mood_change}")
