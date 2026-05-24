# ingestion/year_normalizer.py
"""
中国历史纪年标准化模块
将各种历史纪年格式统一转换为公元年整数，供 RAG 时间围栏使用。

支持格式：
  - 干支纪年：甲子、乙丑、丙寅……
  - 年号纪年：元丰三年、洪武元年、康熙二十五年……
  - 朝代模糊定位：宋代、明初、唐末……
  - 公元直接表达：公元1080年、1644年……
"""
import re
from typing import Optional, Tuple

# ── 天干地支 ──────────────────────────────────────────────────
_TIAN_GAN = "甲乙丙丁戊己庚辛壬癸"
_DI_ZHI   = "子丑寅卯辰巳午未申酉戌亥"

# 干支六十甲子序号 → 公元换算基准
# 甲子 = 0；公元4年是甲子年（可整除60）
_GANZHI_BASE_CE = 4

def ganzhi_to_ce(ganzhi: str, dynasty_hint_ce: Optional[int] = None) -> Optional[int]:
    """
    将干支年份转换为最近匹配的公元年。
    dynasty_hint_ce: 朝代大致公元年，用于在多个候选年中选最近的。
    """
    if len(ganzhi) < 2:
        return None
    gan = ganzhi[0]
    zhi = ganzhi[1]
    if gan not in _TIAN_GAN or zhi not in _DI_ZHI:
        return None
    gan_idx = _TIAN_GAN.index(gan)   # 0-9
    zhi_idx = _DI_ZHI.index(zhi)     # 0-11
    # 在六十甲子中的序号
    cycle_pos = None
    for i in range(60):
        if i % 10 == gan_idx and i % 12 == zhi_idx:
            cycle_pos = i
            break
    if cycle_pos is None:
        return None
    # 基准：公元4年是甲子(0)年
    # CE = _GANZHI_BASE_CE + cycle_pos + 60 * n
    hint = dynasty_hint_ce or 1000  # 默认宋代附近
    # 找最近的候选
    candidates = []
    for n in range(-30, 35):
        ce = _GANZHI_BASE_CE + cycle_pos + 60 * n
        candidates.append(ce)
    return min(candidates, key=lambda x: abs(x - hint))


# ── 年号 → 公元起始年 对照表 ──────────────────────────────────
# 格式：年号名称 → (起始公元年, 结束公元年)
# 收录主要朝代常见年号（可持续扩充）
REIGN_YEAR_TABLE: dict[str, Tuple[int, int]] = {
    # 唐
    "贞观": (627, 649), "永徽": (650, 655), "开元": (713, 741), "天宝": (742, 756),
    "元和": (806, 820), "会昌": (841, 846), "大中": (847, 859), "乾符": (874, 879),
    # 五代
    "同光": (923, 926), "天福": (936, 947),
    # 北宋
    "建隆": (960, 963), "开宝": (968, 976), "太平兴国": (976, 984), "雍熙": (984, 987),
    "淳化": (990, 994), "至道": (995, 997), "咸平": (998, 1003), "景德": (1004, 1007),
    "大中祥符": (1008, 1016), "天禧": (1017, 1021), "乾兴": (1022, 1022),
    "天圣": (1023, 1032), "明道": (1032, 1033), "景祐": (1034, 1038),
    "宝元": (1038, 1040), "康定": (1040, 1041), "庆历": (1041, 1048),
    "皇祐": (1049, 1054), "至和": (1054, 1056), "嘉祐": (1056, 1063),
    "治平": (1064, 1067), "熙宁": (1068, 1077), "元丰": (1078, 1085),
    "元祐": (1086, 1094), "绍圣": (1094, 1098), "元符": (1098, 1100),
    "建中靖国": (1101, 1101), "崇宁": (1102, 1106), "大观": (1107, 1110),
    "政和": (1111, 1118), "重和": (1118, 1119), "宣和": (1119, 1125),
    # 南宋
    "建炎": (1127, 1130), "绍兴": (1131, 1162), "隆兴": (1163, 1164),
    "乾道": (1165, 1173), "淳熙": (1174, 1189), "绍熙": (1190, 1194),
    "庆元": (1195, 1200), "嘉泰": (1201, 1204), "开禧": (1205, 1207),
    "嘉定": (1208, 1224), "宝庆": (1225, 1227), "绍定": (1228, 1233),
    "端平": (1234, 1236), "嘉熙": (1237, 1240), "淳祐": (1241, 1252),
    "宝祐": (1253, 1258), "开庆": (1259, 1259), "景定": (1260, 1264),
    "咸淳": (1265, 1274), "德祐": (1275, 1276), "景炎": (1276, 1278),
    "祥兴": (1278, 1279),
    # 元
    "至元": (1264, 1294), "元贞": (1295, 1297), "大德": (1297, 1307),
    "至大": (1308, 1311), "皇庆": (1312, 1313), "延祐": (1314, 1320),
    "至治": (1321, 1323), "泰定": (1324, 1328), "至顺": (1330, 1333),
    "元统": (1333, 1335), "至元后": (1335, 1340), "至正": (1341, 1368),
    # 明
    "洪武": (1368, 1398), "建文": (1399, 1402), "永乐": (1403, 1424),
    "洪熙": (1425, 1425), "宣德": (1426, 1435), "正统": (1436, 1449),
    "景泰": (1450, 1457), "天顺": (1457, 1464), "成化": (1465, 1487),
    "弘治": (1488, 1505), "正德": (1506, 1521), "嘉靖": (1522, 1566),
    "隆庆": (1567, 1572), "万历": (1573, 1620), "泰昌": (1620, 1620),
    "天启": (1621, 1627), "崇祯": (1628, 1644),
    # 清
    "顺治": (1644, 1661), "康熙": (1662, 1722), "雍正": (1723, 1735),
    "乾隆": (1736, 1795), "嘉庆": (1796, 1820), "道光": (1821, 1850),
    "咸丰": (1851, 1861), "同治": (1862, 1874), "光绪": (1875, 1908),
    "宣统": (1909, 1912),
    # 先秦/秦汉（粗粒度）
    "始皇": (-221, -210), "汉初": (-206, -180), "文景": (-180, -141),
    "元狩": (-122, -117), "元封": (-110, -105), "太初": (-104, -101),
    "永元": (89, 105), "建安": (196, 220), "黄初": (220, 226),
}

# 朝代 → 中心公元年（用于干支校正）
DYNASTY_CENTER: dict[str, int] = {
    "先秦": -500, "秦": -220, "汉": 100, "西汉": -50, "东汉": 150,
    "三国": 240, "魏": 240, "蜀": 240, "吴": 240, "晋": 300,
    "南北朝": 480, "隋": 605, "唐": 750, "五代": 940,
    "宋": 1100, "北宋": 1050, "南宋": 1200,
    "元": 1320, "明": 1520, "清": 1780,
    "民国": 1925, "近代": 1900, "现代": 1980,
}

# 汉字数字
_HAN_NUM = {
    "零": 0, "○": 0, "〇": 0,
    "一": 1, "二": 2, "三": 3, "四": 4, "五": 5,
    "六": 6, "七": 7, "八": 8, "九": 9, "十": 10,
    "百": 100, "千": 1000,
    "元": 1, "初": 1, "末": -1,  # 特殊
}

def han_to_int(s: str) -> Optional[int]:
    """将汉字数字转为整数（支持'二十三'/'廿三'/'三十'/'百二十'等）"""
    if not s:
        return None
    # 直接阿拉伯数字
    if re.fullmatch(r"\d+", s):
        return int(s)
    s = s.replace("廿", "二十").replace("卅", "三十").replace("卌", "四十")
    # 处理 "元" / "初" → 1
    if s in ("元", "初"):
        return 1
    result = 0
    current = 0
    for ch in s:
        v = _HAN_NUM.get(ch)
        if v is None:
            return None
        if v == 100 or v == 1000:
            if current == 0:
                current = 1
            result += current * v
            current = 0
        elif v == 10:
            if current == 0:
                current = 1
            result += current * 10
            current = 0
        else:
            current = v
    result += current
    return result if result > 0 else None


def reign_year_to_ce(reign: str, year_str: str) -> Optional[int]:
    """将年号+年数转为公元年。例：元丰 三 → 1080"""
    span = REIGN_YEAR_TABLE.get(reign)
    if span is None:
        return None
    n = han_to_int(year_str)
    if n is None:
        return None
    ce = span[0] + n - 1
    # 范围校验（允许稍微溢出，史料有时有误）
    if ce > span[1] + 2:
        return span[1]
    return ce


# ── 正则提取器 ────────────────────────────────────────────────
_PATTERNS = [
    # 公元年直接表达
    (r"公元前?(\d{1,4})年",           "ce_explicit"),
    (r"(?<!\d)(\d{3,4})年(?!\d)",     "ce_bare"),
    # 年号纪年（含"年间"、"间"、"年"）
    (r"([^\s，。,\.]{2,6}?)(元|[零○〇一二三四五六七八九十百廿卅]+)年", "reign"),
    # 年号不带年数（"万历年间"、"康熙朝"）
    (r"([^\s，。,\.]{2,4})(?:年间|朝|时期|年代)",  "reign_bare"),
    # 干支纪年
    (r"([甲乙丙丁戊己庚辛壬癸][子丑寅卯辰巳午未申酉戌亥])年", "ganzhi"),
    # 朝代粗定位
    (r"(先秦|秦|西汉|东汉|汉|三国|魏|蜀|吴|晋|南北朝|隋|唐|五代|北宋|南宋|宋|元|明|清|民国|近代|现代)(?:初|中|末|代|朝|时期|时代|年间)?", "dynasty"),
]

def extract_year_from_text(text: str, dynasty_hint: Optional[str] = None) -> Optional[int]:
    """
    从文本片段中提取最可信的公元年整数。
    优先级：公元显式 > 年号 > 干支 > 朝代粗定位
    """
    hint_ce = DYNASTY_CENTER.get(dynasty_hint or "", None)

    results = []  # (priority, ce)

    for pattern, kind in _PATTERNS:
        for m in re.finditer(pattern, text):
            if kind == "ce_explicit":
                sign = -1 if "前" in m.group(0) else 1
                results.append((0, sign * int(m.group(1))))
            elif kind == "ce_bare":
                v = int(m.group(1))
                if 100 <= v <= 2100:
                    results.append((1, v))
            elif kind == "reign":
                reign, year_str = m.group(1), m.group(2)
                ce = reign_year_to_ce(reign, year_str)
                if ce:
                    results.append((2, ce))
            elif kind == "reign_bare":
                reign = m.group(1)
                span = REIGN_YEAR_TABLE.get(reign)
                if span:
                    results.append((2, (span[0] + span[1]) // 2))

                gz = m.group(1)
                ce = ganzhi_to_ce(gz, dynasty_hint_ce=hint_ce)
                if ce:
                    results.append((3, ce))
            elif kind == "dynasty":
                dyn = m.group(1)
                ce = DYNASTY_CENTER.get(dyn)
                if ce:
                    results.append((4, ce))

    if not results:
        return hint_ce  # 回退到朝代中心年
    # 取优先级最高（数字最小）的
    results.sort(key=lambda x: x[0])
    return results[0][1]


def normalize_era_name(raw: str) -> str:
    """
    将用户输入的朝代别称统一为标准存储名。
    例：北宋 / 宋朝 / 宋代 → song
    """
    mapping = {
        "song": ["宋", "北宋", "南宋", "宋代", "宋朝"],
        "tang": ["唐", "唐代", "唐朝", "大唐"],
        "ming": ["明", "明代", "明朝", "大明"],
        "qing": ["清", "清代", "清朝", "大清"],
        "han":  ["汉", "汉代", "西汉", "东汉", "汉朝"],
        "yuan": ["元", "元代", "元朝", "大元"],
        "sui":  ["隋", "隋代", "隋朝"],
        "tang_five": ["五代", "五代十国"],
        "pre_qin": ["先秦", "春秋", "战国", "周"],
        "qin": ["秦", "秦朝", "大秦"],
        "modern": ["民国", "近代", "现代", "当代"],
    }
    for key, aliases in mapping.items():
        for alias in aliases:
            if alias in raw:
                return key
    return raw.lower().replace(" ", "_")


if __name__ == "__main__":
    # 自测
    tests = [
        ("元丰三年，苏轼被贬至黄州", None),
        ("康熙二十五年，传教士南怀仁去世", None),
        ("洪武元年朱元璋建立大明", None),
        ("乾隆甲子年修四库全书", "清"),
        ("崇祯十七年李自成攻入北京", None),
        ("公元前221年秦始皇统一六国", None),
        ("万历年间，东林党争激烈", None),
        ("宋代市井繁华，汴京商铺林立", None),
    ]
    print("=== 纪年标准化自测 ===")
    for text, hint in tests:
        ce = extract_year_from_text(text, dynasty_hint=hint)
        print(f"  [{str(ce):>6}] {text[:20]}")
