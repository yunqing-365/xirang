# infra/report_pdf.py
"""
息壤学生历史探究报告 · PDF生成模块（P1）
==========================================
技术：xhtml2pdf (reportlab) + WQY Zen Hei TrueType 字体
报告共 4 页（A4竖版）：封面 | 概要+概念+史料 | 推理+创作 | 评语
"""
from __future__ import annotations

import io
import html as _html
import os
from datetime import datetime
from textwrap import shorten

from xhtml2pdf import pisa

# ── 字体路径 ───────────────────────────────────────────────────
_BASE      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_FONT_DIR  = os.path.join(_BASE, "assets", "fonts")
_FONT_WQY  = os.path.join(_FONT_DIR, "WQYZenHei.ttf")
# 系统回退
if not os.path.exists(_FONT_WQY):
    _FONT_WQY = "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"

# ── 朝代主题色 ─────────────────────────────────────────────────
_THEMES = {
    "song":    {"a": "#92400e", "li": "#fef3c7", "g": "#b45309"},
    "tang":    {"a": "#5b21b6", "li": "#ede9fe", "g": "#7c3aed"},
    "ming":    {"a": "#991b1b", "li": "#fee2e2", "g": "#dc2626"},
    "qing":    {"a": "#075985", "li": "#e0f2fe", "g": "#0369a1"},
    "default": {"a": "#1e40af", "li": "#dbeafe", "g": "#1d4ed8"},
}

def _theme(k: str) -> dict:
    for key, v in _THEMES.items():
        if key in k.lower():
            return v
    return _THEMES["default"]

def _achv(data: dict) -> tuple[str, str]:
    s = (min(len(data.get("citations", [])), 10) * 3
         + min(data.get("socratic_turns", 0), 20) * 2
         + min(len(data.get("concepts", [])), 8) * 5
         + min(len(data.get("creations", [])), 5) * 4)
    if s >= 80: return "历史学家",   "★★★★★"
    if s >= 50: return "史料探究者", "★★★★☆"
    if s >= 25: return "历史侦探",   "★★★☆☆"
    return "历史探索者", "★★☆☆☆"

def _e(s) -> str:
    return _html.escape(str(s or ""))


# ═══════════════════════════════════════════════════════════════
# HTML 模板（xhtml2pdf 兼容：无 flex，无 inline-block，无 @page counter）
# ═══════════════════════════════════════════════════════════════

def _build_html(session_id: str, user_id: str, display_name: str, data: dict) -> str:
    t          = _theme(data.get("era_key", "default"))
    era        = _e(data.get("era", "历史探究"))
    achv, stars = _achv(data)
    now        = datetime.now().strftime("%Y年%m月%d日")
    a, li, g   = t["a"], t["li"], t["g"]

    citations  = data.get("citations", [])
    concepts   = data.get("concepts", [])
    cdetails   = data.get("concept_details", [])
    chains     = data.get("causal_chains", [])
    creations  = data.get("creations", [])
    emo_arc    = data.get("emotion_arc", [])
    socratic   = data.get("socratic_turns", 0)
    rounds     = data.get("rounds_completed", 0)
    depth      = data.get("thinking_depth", 0)
    comment    = _e(data.get("teacher_comment", ""))

    # ── 统计表格 ──────────────────────────────────────────────
    def td(label, val):
        return (f'<td style="background:{li};border:1.5pt solid {a};'
                f'padding:10px 4px;text-align:center;">'
                f'<div style="font-size:20pt;font-weight:bold;color:{a}">{_e(str(val))}</div>'
                f'<div style="font-size:8pt;color:#64748b;margin-top:2px">{_e(label)}</div></td>')

    stats = (f'<table width="100%" cellspacing="4" cellpadding="0" style="margin:12px 0 18px">'
             f'<tr>{td("探究轮次",rounds)}{td("史料引用",len(citations))}'
             f'{td("问答轮次",socratic)}{td("创作作品",len(creations))}'
             f'{td("大概念",len(concepts))}</tr></table>')

    # ── 情绪弧线 ──────────────────────────────────────────────
    emo_html = ""
    if emo_arc:
        emo_html = (f'<p style="background:{li};border-left:3px solid {a};'
                    f'padding:8px 12px;margin:8px 0;color:#475569">'
                    f'{"&nbsp;→&nbsp;".join(_e(x) for x in emo_arc[:8])}</p>')

    # ── 大概念 ────────────────────────────────────────────────
    if cdetails:
        cd_html = "".join(
            f'<div style="border-left:3px solid {a};padding:8px 12px;margin:8px 0;background:#f8fafc">'
            f'<div style="font-weight:bold;color:{a};font-size:11pt">{_e(cd.get("name",""))}</div>'
            f'<div style="font-size:8.5pt;color:#64748b;margin:2px 0">触碰 {cd.get("count",1)} 次</div>'
            f'<div style="font-size:9.5pt;color:#475569">{_e(cd.get("summary",""))}</div>'
            f'</div>'
            for cd in cdetails
        )
    elif concepts:
        cd_html = "<p>" + "&nbsp;&nbsp;".join(
            f'<span style="background:{li};color:{a};padding:2px 8px;'
            f'border:1px solid {a};border-radius:10px">{_e(c)}</span>'
            for c in concepts
        ) + "</p>"
    else:
        cd_html = f'<p style="color:#94a3b8;font-style:italic">本次探究未触碰到已识别的大概念。</p>'

    # ── 史料引用 ──────────────────────────────────────────────
    if citations:
        cite_html = "".join(
            f'<div style="border:1px solid #e2e8f0;border-radius:6px;padding:10px 12px;margin:8px 0">'
            f'<div style="font-weight:bold;color:{a};margin-bottom:5px">[{i+1}] {_e(c.get("title","史料"))}</div>'
            f'<div style="font-size:9.5pt;background:{li};padding:6px 10px;border-radius:4px;font-style:italic">'
            f'{_e(shorten(c.get("content",""), 200, placeholder="…"))}</div>'
            + (f'<div style="font-size:8pt;color:#94a3b8;margin-top:4px">来源：{_e(c.get("source",""))}</div>'
               if c.get("source") else "")
            + '</div>'
            for i, c in enumerate(citations)
        )
    else:
        cite_html = f'<p style="color:#94a3b8;font-style:italic">本次探究未记录到明确的史料引用。</p>'

    # ── 因果推理 ──────────────────────────────────────────────
    if chains:
        chain_html = "".join(
            f'<p style="margin:8px 0;padding:6px 12px;background:#f8fafc;'
            f'border-left:3px solid {a}">'
            f'<strong style="color:{a};margin-right:8px">{i+1}.</strong>'
            f'{_e(ch)}</p>'
            for i, ch in enumerate(chains)
        )
        if depth:
            chain_html += f'<p style="color:#475569;margin-top:8px">思维深度评分：<strong style="color:{a}">{depth} / 100</strong></p>'
    else:
        chain_html = f'<p style="color:#94a3b8;font-style:italic">建议下次追问：为什么会发生这件事？如果做出不同选择，结果会如何？</p>'

    # ── 创作作品 ──────────────────────────────────────────────
    if creations:
        cr_html = "".join(
            f'<div style="border:1.5px solid {a};border-radius:8px;padding:12px 14px;'
            f'margin:10px 0;background:{li}">'
            f'<div style="font-weight:bold;font-size:11pt;color:{a};margin-bottom:6px">'
            f'《{_e(c.get("title",""))}》&nbsp;<span style="font-size:8.5pt;font-weight:normal;color:#64748b">[{_e(c.get("type","创作"))}]</span></div>'
            f'<div style="border-left:2px solid {a};padding-left:10px;color:#334155;line-height:1.8">'
            f'{_e(c.get("content",""))}</div></div>'
            for c in creations
        )
    else:
        cr_html = f'<p style="color:#94a3b8;font-style:italic">本次探究未留下创作作品，建议下次尝试诗词或辩论稿创作。</p>'

    # ── 章节标题 helper ───────────────────────────────────────
    def ch(title):
        return (f'<div style="background:{g};color:white;padding:8px 14px;'
                f'border-radius:6px;font-size:13pt;font-weight:bold;margin:24px 0 14px">'
                f'{title}</div>')

    def sh(title):
        return f'<p style="font-size:11pt;font-weight:bold;color:{a};margin:14px 0 8px">{title}</p>'

    def kv(key, val, accent=False):
        vc = f'color:{a};font-weight:bold' if accent else 'color:#334155'
        return (f'<p style="margin:4px 0">'
                f'<strong style="color:{a}">{_e(key)}</strong>'
                f'<span style="{vc}">{_e(val)}</span></p>')

    def hr():
        return '<hr style="border:none;border-top:1px solid #e2e8f0;margin:12px 0"/>'

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<style>
@font-face {{
    font-family: SC;
    src: url('{_FONT_WQY}');
}}
@page {{
    size: A4;
    margin: 18mm 18mm 20mm 18mm;
}}
* {{ margin: 0; padding: 0; }}
body {{
    font-family: SC, sans-serif;
    font-size: 10pt;
    line-height: 1.7;
    color: #1e293b;
}}
table {{ color: #1e293b; }}
td {{ color: #1e293b; }}
th {{ color: #1e293b; }}
.page-break {{ page-break-before: always; }}
</style>
</head>
<body>

<!-- ══ 封面 ══ -->
<div style="background:{g};color:white;padding:32px 20px;text-align:center;margin:-18mm -18mm 0">
    <div style="font-size:26pt;font-weight:bold;letter-spacing:6pt">息　壤</div>
    <div style="font-size:11pt;opacity:0.85;margin-top:6px">AI 历史沉浸探究平台</div>
</div>
<div style="text-align:center;margin:50px 0 30px">
    <div style="font-size:17pt;font-weight:bold;color:{a};margin-bottom:8px">历史探究报告</div>
    <div style="font-size:13pt;color:#475569;margin-bottom:20px">{era}</div>
    <hr style="border:none;border-top:2px solid {a};width:80px;margin:0 auto 20px"/>
    <div style="font-size:22pt;font-weight:bold;margin-bottom:8px">{_e(display_name)}</div>
    <div style="font-size:13pt;color:{a}">{_e(achv)}&nbsp;&nbsp;{_e(stars)}</div>
</div>
<div style="border:1.5px solid {a};border-radius:8px;background:{li};padding:14px 24px;
            margin:30px auto 0;text-align:center;font-size:9.5pt;color:#475569">
    生成日期：{now}&nbsp;&nbsp;|&nbsp;&nbsp;会话编号：{_e(session_id[:12])}…<br/>
    本报告由息壤 AI 平台自动生成，供教师评阅参考
</div>
<div style="text-align:center;font-size:8pt;color:#94a3b8;margin-top:100px">
    xirang.ai · 让历史课活起来
</div>

<!-- ══ 第一章 ══ -->
<div class="page-break"></div>
{ch("第一章&emsp;探究概要")}
{stats}
{sh("基本信息")}
{kv("学生姓名：", display_name, accent=True)}
{kv("探究场景：", data.get("era",""))}
{kv("成就等级：", achv + " " + stars, accent=True)}
{kv("生成时间：", now)}
{(sh("情绪探究弧线") + emo_html) if emo_arc else ""}
{hr()}
{ch("第二章&emsp;大概念触碰记录")}
{cd_html}
{hr()}
{ch("第三章&emsp;史料引用清单")}
{cite_html}

<!-- ══ 第四章 ══ -->
<div class="page-break"></div>
{ch("第四章&emsp;思维链 &amp; 因果推理")}
{chain_html}
{hr()}
{ch("第五章&emsp;创作作品集")}
{cr_html}

<!-- ══ 教师评语 ══ -->
<div class="page-break"></div>
{ch("教师评语")}
<div style="border:1.5px solid {a};border-radius:8px;min-height:60mm;
            padding:14px;background:{li};color:#334155;line-height:1.8">
    {comment}
</div>
<table width="100%" cellspacing="0" cellpadding="0" style="margin:16px 0;font-size:10pt">
    <tr>
        <td style="padding:6px 0;border-bottom:1px solid #94a3b8;width:50%">
            教师签名：___________________
        </td>
        <td style="padding:6px 0;border-bottom:1px solid #94a3b8;padding-left:20px">
            评阅日期：___________________
        </td>
    </tr>
</table>
<div style="background:{g};color:white;padding:18px;text-align:center;
            border-radius:10px;margin-top:40px">
    <div style="font-size:14pt;font-weight:bold;letter-spacing:4pt">息　壤</div>
    <div style="font-size:9pt;opacity:0.8;margin-top:4px">
        xirang.ai · AI 历史沉浸探究平台 · 让历史课活起来
    </div>
</div>

</body>
</html>"""


# ═══════════════════════════════════════════════════════════════
# 主函数
# ═══════════════════════════════════════════════════════════════

def build_student_report_pdf(
    session_id: str,
    user_id: str,
    display_name: str,
    data: dict,
) -> bytes:
    """
    生成学生历史探究报告 PDF。
    data 字段：era, era_key, citations, concepts, concept_details,
               causal_chains, creations, socratic_turns, rounds_completed,
               thinking_depth, emotion_arc, teacher_comment
    """
    html_str = _build_html(session_id, user_id, display_name, data)
    buf = io.BytesIO()
    result = pisa.pisaDocument(
        io.BytesIO(html_str.encode("utf-8")),
        buf,
        encoding="utf-8",
    )
    if result.err:
        raise RuntimeError(f"PDF渲染失败 (xhtml2pdf error={result.err})")
    return buf.getvalue()


# ═══════════════════════════════════════════════════════════════
# FastAPI Router
# ═══════════════════════════════════════════════════════════════

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
from infra.auth import TokenData, get_current_user
from infra.quota import require_pdf_export, QuotaState

pdf_router = APIRouter(prefix="/api/report", tags=["report"])


class ReportData(BaseModel):
    display_name: str = "历史探究者"
    era: str = "历史探究"
    era_key: str = "default"
    citations: list[dict] = []
    concepts: list[str] = []
    concept_details: list[dict] = []
    causal_chains: list[str] = []
    creations: list[dict] = []
    socratic_turns: int = 0
    rounds_completed: int = 0
    thinking_depth: int = 0
    emotion_arc: list[str] = []
    teacher_comment: str = ""


@pdf_router.post("/pdf/{session_id}")
async def export_pdf_report(
    session_id: str,
    req: ReportData,
    token: TokenData = Depends(get_current_user),
    _quota: QuotaState = Depends(require_pdf_export),
):
    """
    生成学生历史探究报告 PDF（下载）。
    需要教师专业版及以上套餐。
    前端：fetch → blob → <a download>
    """
    try:
        pdf_bytes = build_student_report_pdf(
            session_id, token.user_id, req.display_name, req.model_dump()
        )
    except Exception as ex:
        raise HTTPException(status_code=500, detail=f"PDF生成失败：{ex}")
    filename = f"息壤探究报告_{req.display_name}_{session_id[:8]}.pdf"
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "Content-Length": str(len(pdf_bytes)),
        },
    )


@pdf_router.post("/pdf/{session_id}/preview")
async def preview_pdf_report(
    session_id: str,
    req: ReportData,
    token: TokenData = Depends(get_current_user),
    _quota: QuotaState = Depends(require_pdf_export),
):
    """浏览器内联预览"""
    pdf_bytes = build_student_report_pdf(
        session_id, token.user_id, req.display_name, req.model_dump()
    )
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": "inline"},
    )
