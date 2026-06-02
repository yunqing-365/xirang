# infra/share_card.py
"""
息壤微信分享卡片生成器（P1）
=====================================
输出两种格式：
  1. HTML（内联，分享预览页，可直接打开分享）
  2. OpenGraph meta tags（微信/QQ分享缩略图适配）

卡片内容：
  - 用户昵称 + 成就标题
  - 探索的历史场景
  - 关键数据徽章（史料引用、探究问题、大概念）
  - 二维码区域（指向场景入口）
  - 底部品牌标识

技术路线：
  - 纯 HTML/CSS → 服务端渲染成字符串返回
  - 前端截图（html2canvas）→ 保存图片分享
  - TODO 生产：用 playwright headless 截图生成 PNG

"""
from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from infra.auth import TokenData, get_current_user

share_router = APIRouter(prefix="/api/share", tags=["share"])

# ── 成就等级定义 ─────────────────────────────────────────────────

def _calc_achievement(data: dict) -> tuple[str, str]:
    """根据探究数据计算成就等级和标题"""
    score = 0
    score += min(data.get("citations_count", 0), 10) * 3
    score += min(data.get("socratic_turns", 0), 20) * 2
    score += min(len(data.get("concepts", [])), 8) * 5
    score += min(data.get("creations_count", 0), 5) * 4

    if score >= 80:
        return "🏆", "历史学家"
    elif score >= 50:
        return "📜", "史料探究者"
    elif score >= 25:
        return "🔍", "历史侦探"
    else:
        return "🌱", "历史探索者"


def _era_color(era: str) -> tuple[str, str]:
    """根据朝代返回主色调"""
    mapping = {
        "song": ("#b45309", "#fef3c7"),
        "tang": ("#7c3aed", "#ede9fe"),
        "ming": ("#dc2626", "#fee2e2"),
        "qing": ("#0369a1", "#e0f2fe"),
    }
    for key, colors in mapping.items():
        if key in era.lower():
            return colors
    return ("#1e40af", "#dbeafe")


def generate_share_card_html(
    user_id: str,
    display_name: str,
    era: str,
    data: dict,
) -> str:
    """生成完整的分享卡片 HTML 页面"""
    badge_icon, badge_title = _calc_achievement(data)
    accent, bg_light = _era_color(era)
    concepts = data.get("concepts", [])[:3]
    citations = data.get("citations_count", 0)
    socratic = data.get("socratic_turns", 0)
    creations = data.get("creations_count", 0)

    # 概念标签
    concept_tags = "".join(
        f'<span style="background:{bg_light};color:{accent};padding:2px 10px;'
        f'border-radius:12px;font-size:12px;margin:2px;">{c}</span>'
        for c in concepts
    ) or '<span style="color:#94a3b8;font-size:12px">探索进行中</span>'

    share_url = f"https://xirang.ai?era={era}"  # 生产替换为真实域名

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1,user-scalable=no">
<title>{display_name}的历史探究成就 · 息壤</title>
<!-- Open Graph（微信分享缩略图） -->
<meta property="og:title" content="{display_name}在息壤完成了「{era}」历史探究">
<meta property="og:description" content="引用史料{citations}次 · 探究问答{socratic}轮 · 触碰大概念{len(concepts)}个">
<meta property="og:type" content="website">
<meta property="og:url" content="{share_url}">
<style>
* {{ margin:0;padding:0;box-sizing:border-box; }}
body {{
  font-family: 'PingFang SC','Hiragino Sans GB','Microsoft YaHei',sans-serif;
  background: #f1f5f9;
  min-height: 100vh;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 20px;
}}
.card {{
  width: 360px;
  background: #fff;
  border-radius: 20px;
  overflow: hidden;
  box-shadow: 0 20px 60px rgba(0,0,0,.12);
}}
.card-header {{
  background: linear-gradient(135deg, {accent}, {accent}cc);
  color: #fff;
  padding: 28px 24px 20px;
  position: relative;
}}
.era-name {{
  font-size: 13px;
  opacity: .8;
  letter-spacing: .1em;
  margin-bottom: 6px;
}}
.user-name {{
  font-size: 22px;
  font-weight: 700;
  margin-bottom: 4px;
}}
.badge-row {{
  display: flex;
  align-items: center;
  gap: 8px;
  margin-top: 10px;
}}
.badge {{
  background: rgba(255,255,255,.2);
  border-radius: 20px;
  padding: 4px 12px;
  font-size: 13px;
  display: flex;
  align-items: center;
  gap: 5px;
}}
.deco {{
  position: absolute;
  right: 20px;
  top: 20px;
  font-size: 48px;
  opacity: .15;
}}
.card-body {{
  padding: 20px 24px;
}}
.section-title {{
  font-size: 11px;
  font-weight: 600;
  letter-spacing: .1em;
  text-transform: uppercase;
  color: #94a3b8;
  margin-bottom: 10px;
}}
.stats-row {{
  display: flex;
  gap: 0;
  margin-bottom: 20px;
  border: 1px solid #e2e8f0;
  border-radius: 10px;
  overflow: hidden;
}}
.stat {{
  flex: 1;
  text-align: center;
  padding: 12px 8px;
  border-right: 1px solid #e2e8f0;
}}
.stat:last-child {{ border-right: none; }}
.stat-num {{
  font-size: 20px;
  font-weight: 700;
  color: {accent};
}}
.stat-lbl {{
  font-size: 10px;
  color: #94a3b8;
  margin-top: 3px;
}}
.concepts-wrap {{
  display: flex;
  flex-wrap: wrap;
  gap: 4px;
  margin-bottom: 20px;
}}
.card-footer {{
  border-top: 1px solid #f1f5f9;
  padding: 14px 24px;
  display: flex;
  align-items: center;
  justify-content: space-between;
}}
.brand {{
  font-size: 14px;
  font-weight: 700;
  color: {accent};
}}
.brand-sub {{
  font-size: 10px;
  color: #94a3b8;
}}
.qr-placeholder {{
  width: 52px;
  height: 52px;
  background: {bg_light};
  border-radius: 8px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 22px;
}}
.save-hint {{
  text-align: center;
  font-size: 12px;
  color: #94a3b8;
  padding: 10px;
  background: #f8fafc;
}}
</style>
</head>
<body>
<div>
  <div class="card" id="share-card">
    <div class="card-header">
      <div class="deco">📜</div>
      <div class="era-name">息壤历史探究 · {era}</div>
      <div class="user-name">{display_name}</div>
      <div class="badge-row">
        <div class="badge">{badge_icon} {badge_title}</div>
      </div>
    </div>
    <div class="card-body">
      <div class="section-title">探究成果</div>
      <div class="stats-row">
        <div class="stat">
          <div class="stat-num">{citations}</div>
          <div class="stat-lbl">史料引用</div>
        </div>
        <div class="stat">
          <div class="stat-num">{socratic}</div>
          <div class="stat-lbl">问答轮次</div>
        </div>
        <div class="stat">
          <div class="stat-num">{creations}</div>
          <div class="stat-lbl">创作作品</div>
        </div>
      </div>
      <div class="section-title">触碰大概念</div>
      <div class="concepts-wrap">{concept_tags}</div>
    </div>
    <div class="card-footer">
      <div>
        <div class="brand">息壤</div>
        <div class="brand-sub">AI历史沉浸探究平台</div>
      </div>
      <div class="qr-placeholder">📱</div>
    </div>
  </div>
  <div class="save-hint">长按卡片保存图片，分享给朋友</div>
</div>

<script>
// 长按保存（移动端）
document.getElementById('share-card').addEventListener('contextmenu', e => e.preventDefault());
// 可接入 html2canvas 截图
</script>
</body>
</html>"""


# ── FastAPI 路由 ─────────────────────────────────────────────────

from fastapi.responses import HTMLResponse


@share_router.get("/card/{session_id}", response_class=HTMLResponse)
async def get_share_card(
    session_id: str,
    user_id: str = Query(default="anonymous"),
    display_name: str = Query(default="探索者"),
    era: str = Query(default="历史"),
    citations: int = Query(default=0),
    socratic: int = Query(default=0),
    creations: int = Query(default=0),
    concepts: str = Query(default=""),
):
    """
    生成可分享的成就卡片页面（直接渲染 HTML）。
    前端调用：window.open('/api/share/card/{session_id}?...')
    微信内长按卡片图片即可保存并分享。
    """
    concept_list = [c.strip() for c in concepts.split(",") if c.strip()]
    data = {
        "citations_count": citations,
        "socratic_turns": socratic,
        "creations_count": creations,
        "concepts": concept_list,
    }
    html = generate_share_card_html(user_id, display_name, era, data)
    return HTMLResponse(content=html)


@share_router.get("/card_data/{session_id}")
async def get_share_card_data(
    session_id: str,
    token: TokenData = Depends(get_current_user),
):
    """
    返回分享卡片所需的聚合数据（JSON），供前端自行渲染。
    """
    return {
        "share_url": f"/api/share/card/{session_id}?user_id={token.user_id}",
        "session_id": session_id,
        "tip": "将 share_url 中的参数填入探究数据，传给前端后用 html2canvas 截图分享",
    }
