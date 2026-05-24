# narrative/offline_evolution.py
"""
NPC 离线自主推演引擎
玩家离线期间，世界依然按照 NPC 的性格、关系网络、历史进程持续演化。
玩家返回时，看到的是一个已经"活过"的世界，而非静止的存档。

设计原则：
  - 轻量：不触发完整的 stream_next 推演，只做简化快进
  - 有意义：产生对玩家有实质影响的变化（关系值、里程碑、物品）
  - 有感知：玩家重新进入时收到"蝴蝶效应摘要"
  - 可撤销：所有离线变化写入独立的 offline_log，不直接覆盖主存档

离线推演触发条件：
  - 玩家离线超过 OFFLINE_THRESHOLD_HOURS 小时（默认2小时）
  - 叙事阶段不在 DENOUEMENT（收束后不再推进）
"""
from __future__ import annotations
import asyncio
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import get_settings
from openai import AsyncOpenAI

_settings = get_settings()
_client = AsyncOpenAI(api_key=_settings.API_KEY, base_url=_settings.BASE_URL)

# ── 配置 ─────────────────────────────────────────────────────
OFFLINE_THRESHOLD_HOURS = 2     # 超过此时长视为"离线"
MAX_OFFLINE_ROUNDS = 4          # 最多快进回合数（避免世界跑太远）
OFFLINE_LOG_KEY = "offline_log" # session data 中的日志键

# ── 快进推演 Prompt ───────────────────────────────────────────
_FAST_FORWARD_PROMPT = """\
你是息壤时空底座的"历史惯性引擎"。玩家（异乡人）暂时离开了这个世界。
请根据以下信息，模拟在玩家缺席的{hours}小时内，这个世界自然发生的事情。

【当前场景】
{scene_desc}

【在场人物及关系】
{agents_summary}

【最近对话摘要】
{recent_dialogue}

【叙事阶段】{narrative_phase}
【世界情绪】{world_mood}

请生成 {rounds} 个时间快进片段，每个片段描述世界在玩家缺席时自然发生的一件事。
要求：
1. 符合人物性格与历史情境
2. 有真实的因果逻辑（如：苏轼写诗 → 佛印来访品评）
3. 产生可量化的影响（关系值变化、物品出现、地点变化）
4. 语气如旁观者叙事，文言白话混用，50字内每条

严格输出 JSON（不加任何其他文字）：
{{
  "events": [
    {{
      "time_offset": "2小时后",
      "description": "叙事描述（50字内）",
      "relationship_changes": [{{"from": "人物A", "to": "人物B", "affinity_delta": 5, "trust_delta": 3}}],
      "workspace_append": "对时空产物的补充（可为空字符串）",
      "new_milestone": "新的叙事里程碑（可为空字符串）",
      "mood_hint": "SERENE|MELANCHOLY|TENSE|JOYFUL|SOLEMN|CHAOTIC 或空字符串"
    }}
  ],
  "butterfly_summary": "给玩家的蝴蝶效应摘要（100字内，说明世界发生了什么变化）"
}}
"""

_AGENTS_SUMMARY_PROMPT = """\
请用一句话（不超过30字）概括以下人物的当前状态和与其他人的关系：
{agent_json}
"""


async def _build_agents_summary(agents: list) -> str:
    """将 Agent 列表压缩为摘要字符串（用于 fast-forward prompt）"""
    lines = []
    for agent in agents:
        rels = agent.memory.data.get("relationships", {})
        rel_str = "、".join(
            f"与{k}{'亲近' if v.get('affinity',50)>60 else '疏远'}"
            for k, v in list(rels.items())[:3]
        )
        lines.append(f"{agent.name}（{agent.identity}）：{rel_str or '关系尚浅'}")
    return "\n".join(lines)


async def fast_forward(
    session_data: dict,
    offline_hours: float,
    rounds: int = 2,
) -> dict:
    """
    执行离线快进推演。
    返回 fast_forward_result dict，包含事件列表和蝴蝶效应摘要。
    """
    manager    = session_data.get("manager")
    agents     = manager.agents if manager else []
    ns         = session_data.get("narrative_state")
    world_env  = manager.world_env if manager else None

    if not agents:
        return {"events": [], "butterfly_summary": "世界在等待着你归来..."}

    # 跳过已收束的叙事
    phase = ns.phase.value if ns else "开端"
    if phase == "收束":
        return {"events": [], "butterfly_summary": "故事已趋于平静，等待你的续写。"}

    agents_summary = await _build_agents_summary(agents)
    recent_dialogue = (manager.current_dialogue or "")[-600:] if manager else ""
    world_mood = world_env.mood.value if world_env and hasattr(world_env, "mood") else "SERENE"

    prompt = _FAST_FORWARD_PROMPT.format(
        hours=round(offline_hours, 1),
        scene_desc=(manager.scene_desc or "")[:300] if manager else "",
        agents_summary=agents_summary,
        recent_dialogue=recent_dialogue,
        narrative_phase=phase,
        world_mood=world_mood,
        rounds=rounds,
    )

    try:
        resp = await _client.chat.completions.create(
            model=_settings.MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000,
            temperature=0.7,
            timeout=25,
        )
        raw = resp.choices[0].message.content.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        result = json.loads(raw)
    except Exception as e:
        print(f"  ⚠️  离线推演失败: {e}")
        return {
            "events": [],
            "butterfly_summary": f"世界在悄然流逝，{round(offline_hours,1)}小时已过。"
        }

    # 应用效果到 session 数据
    await _apply_fast_forward(result, session_data)
    return result


async def _apply_fast_forward(result: dict, session_data: dict):
    """将快进结果应用到 session 内部状态"""
    manager = session_data.get("manager")
    ns      = session_data.get("narrative_state")
    events  = result.get("events", [])

    for event in events:
        # 1. 更新 NPC 关系值
        rel_changes = event.get("relationship_changes", [])
        for change in rel_changes:
            src_name = change.get("from", "")
            tgt_name = change.get("to", "")
            aff_d    = int(change.get("affinity_delta", 0))
            trust_d  = int(change.get("trust_delta", 0))
            if manager:
                for agent in manager.agents:
                    if agent.name == src_name:
                        agent.memory.update_relationship(tgt_name, aff_d, trust_d)
                        agent.memory.save()

        # 2. 追加 workspace
        ws_append = event.get("workspace_append", "")
        if ws_append and manager:
            manager.shared_workspace += f"\n[离线] {ws_append}"

        # 3. 写入叙事里程碑
        milestone = event.get("new_milestone", "")
        if milestone and ns:
            ns.add_milestone(milestone)

    # 4. 更新世界情绪（取最后一个有效 mood_hint）
    world_env = manager.world_env if manager else None
    if world_env:
        for event in reversed(events):
            mood_hint = event.get("mood_hint", "")
            if mood_hint and hasattr(world_env, "mood"):
                from world_engine import WorldMood
                try:
                    world_env.mood = WorldMood(mood_hint)
                    break
                except ValueError:
                    pass

    # 5. 持久化到 offline_log
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "events": events,
        "butterfly_summary": result.get("butterfly_summary", ""),
    }
    existing_log = session_data.get(OFFLINE_LOG_KEY, [])
    existing_log.append(log_entry)
    session_data[OFFLINE_LOG_KEY] = existing_log[-10:]  # 最多保留10条


# ── 离线检测 ─────────────────────────────────────────────────
def should_fast_forward(session_entry_last_accessed: float) -> tuple[bool, float]:
    """
    判断是否需要离线推演。
    返回 (should_run: bool, offline_hours: float)
    """
    now = time.time()
    offline_seconds = now - session_entry_last_accessed
    offline_hours = offline_seconds / 3600
    should = offline_hours >= OFFLINE_THRESHOLD_HOURS
    return should, offline_hours


def get_offline_log(session_data: dict) -> list[dict]:
    """获取并清空离线日志（玩家已读取，下次不再重复展示）"""
    log = session_data.get(OFFLINE_LOG_KEY, [])
    session_data[OFFLINE_LOG_KEY] = []  # 清空，避免重复
    return log


# ── 蝴蝶效应摘要 SSE 格式化 ─────────────────────────────────
def format_butterfly_sse(fast_forward_result: dict) -> list[dict]:
    """
    将快进结果格式化为一组 SSE 消息，供 stream_next 发送给前端。
    """
    events = fast_forward_result.get("events", [])
    summary = fast_forward_result.get("butterfly_summary", "")
    messages = []

    if not events:
        return messages

    # 开场白
    messages.append({
        "type": "narrator",
        "content": f"〔时空快进：你离开的时光里，世界悄然流转……〕",
    })

    # 各事件摘要
    for event in events:
        desc = event.get("description", "")
        offset = event.get("time_offset", "")
        if desc:
            messages.append({
                "type": "historical_echo",
                "content": f"【{offset}】{desc}",
            })
        mood = event.get("mood_hint", "")
        if mood:
            messages.append({"type": "world_mood_change", "new_mood": mood})

    # 蝴蝶效应总结
    if summary:
        messages.append({
            "type": "narrator",
            "content": f"〔世界已在你缺席时发生了变化：{summary}〕",
        })

    return messages


if __name__ == "__main__":
    print("离线推演引擎模块加载成功")
    print(f"  离线阈值: {OFFLINE_THRESHOLD_HOURS}小时")
    print(f"  最大快进回合: {MAX_OFFLINE_ROUNDS}")
