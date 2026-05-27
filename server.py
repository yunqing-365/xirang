# server.py  ── 硬核升级版
"""
本轮升级：
  1. EventBus 接入：所有组件通过事件通信，server 订阅 PERSONA_VIOLATION 事件
  2. WorldEngine 替换 environment.py：FSM 状态转移在 stream_next 中触发
  3. Agent 初始化改为并发（await manager.initialize_agents()）
  4. stream_next 将 current_emotion 传递给 Agent（记忆情绪相似加权）
  5. 新增 /api/events/{session_id} 端点：返回会话事件回放（用于断线重连）
  6. 新增 /api/world_state/{session_id} 端点：返回当前世界状态
"""
import asyncio
import json
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, Query
from openai import AsyncOpenAI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from config import get_settings
from director import SpatiotemporalDirector
from event_bus import bus, Event, EventType
from narrative_engine import NarrativeEngine, NarrativeState
from narrative.historical_triggers import TriggerChecker
from narrative.offline_evolution import (
    should_fast_forward, fast_forward, get_offline_log,
    format_butterfly_sse, MAX_OFFLINE_ROUNDS,
)
from infra.auth import auth_router, get_current_user, rate_limit_dep
from infra.observability import (
    setup_observability, metrics_router,
    record_narrative_event, record_trigger_fire,
    record_offline_evolution, update_active_sessions,
)
from reflection_engine import ReflectionEngine
from scenario_manager import ScenarioManager
from session_manager import session_mgr
from user_profile import profile_store
# Phase 12A + 12B
from emotion_engine import EmotionEngine, Emotion, SUSHI_MONOLOGUES
from concept_engine import ConceptEngine
# Phase 13A + 13B
from thinking_engine import get_thinking_engine, ThinkingEngine
from inquiry_engine import get_inquiry_engine, InquiryEngine, BloomLevel
# Phase 14A + 14B + 14C
from source_workshop import get_workshop, get_source_by_id, get_all_sources, BUILTIN_SOURCES
from creative_engine import (
    get_creative_session, CreationType, CrossDisciplineEngine,
    CreativeSession,
)

_settings = get_settings()
_narrative_engine = NarrativeEngine()
_reflection_engine = ReflectionEngine()
# Phase 12A + 12B: 全局情绪引擎和概念引擎（每个会话共享，按 session_id 隔离状态）
_emotion_engine = EmotionEngine()
_concept_engine = ConceptEngine()


# ═══════════════════════════════════════════════════════════════
# 全局 EventBus 订阅（服务启动时注册）
# ═══════════════════════════════════════════════════════════════

async def _on_persona_violation(event: Event):
    """人格违规时记录到会话，前端可通过 /api/events 查看"""
    pass   # 已由 event_bus 的 console_monitor 打印；此处可扩展告警

async def _on_session_expired(event: Event):
    await session_mgr.delete(event.session_id)


@asynccontextmanager
async def lifespan(app: FastAPI):
    bus.subscribe(EventType.PERSONA_VIOLATION, _on_persona_violation)
    bus.subscribe(EventType.SESSION_EXPIRED,   _on_session_expired)
    await bus.emit(Event(EventType.WORLD_CREATED, session_id="system",
                         payload={"msg": "息壤引擎启动"}))
    print("🚀 息壤 v2.1 启动，访问 http://localhost:8000")
    yield
    print("🌙 息壤引擎休眠…")


app = FastAPI(title="息壤", version="2.1.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])
app.mount("/images", StaticFiles(directory="data/raw_documents"), name="images")

# 注册鉴权路由和可观测性
app.include_router(auth_router)
setup_observability(app)


# ═══════════════════════════════════════════════════════════════
# 请求/响应模型
# ═══════════════════════════════════════════════════════════════

class WorldCreationRequest(BaseModel):
    theme: str
    genre: str = "历史客观写实"
    user_id: str = "anonymous"

class InterventionRequest(BaseModel):
    message: str
    session_id: str = "default"

class ChoiceRequest(BaseModel):
    session_id: str
    choice_id: int
    user_id: str = "anonymous"

class ReflectRequest(BaseModel):
    session_id: str
    user_id: str = "anonymous"
    player_context: str = ""


# ═══════════════════════════════════════════════════════════════
# 工具
# ═══════════════════════════════════════════════════════════════

async def _get_session(session_id: str) -> dict:
    data = await session_mgr.get(session_id)
    if data is None:
        raise HTTPException(status_code=404, detail=f"会话 {session_id} 不存在或已过期")
    return data

def _sse(data: dict) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


# ═══════════════════════════════════════════════════════════════
# 基础路由
# ═══════════════════════════════════════════════════════════════

@app.get("/")
async def get_index():
    return FileResponse("index.html")

@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "version": "2.1.0",
        "active_sessions": await session_mgr.active_count(),
        "model": _settings.MODEL_NAME,
    }


# ═══════════════════════════════════════════════════════════════
# 世界创建
# ═══════════════════════════════════════════════════════════════

@app.post("/api/create_world")
async def create_world(req: WorldCreationRequest):
    session_id = f"session_{uuid.uuid4().hex[:8]}"
    manager = ScenarioManager()

    result = await asyncio.to_thread(
        manager.generate_dynamic_scenario, req.theme, req.genre, session_id
    )
    if not result:
        raise HTTPException(status_code=500, detail="世界生成失败")

    await asyncio.to_thread(manager.load_era, result)

    # 并发初始化所有 Agent 的人格指纹
    await manager.initialize_agents()

    director = SpatiotemporalDirector(manager.agents)
    narrative_state = NarrativeState()

    await session_mgr.create(session_id, {
        "manager": manager,
        "agents": manager.agents,
        "director": director,
        "narrative_state": narrative_state,
        "user_id": req.user_id,
        "theme": req.theme,
        "trigger_checker": TriggerChecker(
            era=getattr(manager, "_era_name", "song")
        ),
    })

    await bus.emit(Event(EventType.WORLD_CREATED, session_id=session_id,
                         payload={"theme": req.theme, "genre": req.genre}))

    profile = await asyncio.to_thread(profile_store.load, req.user_id)
    profile.total_sessions += 1
    await asyncio.to_thread(profile_store.save, profile)

    return {
        "status": "success",
        "session_id": session_id,
        "scene_desc": manager.scene_desc,
        "initial_dialogue": manager.current_dialogue,
        "agents": [a.name for a in manager.agents],
        "world_mood": manager.world_env.mood.value,
        "user_title": profile.exploration_depth,
    }


# ═══════════════════════════════════════════════════════════════
# 干预 / 叙事选项
# ═══════════════════════════════════════════════════════════════

@app.post("/api/intervene")
async def post_intervention(req: InterventionRequest):
    await _get_session(req.session_id)

    # ── Phase 14A：检测玩家是否引用了史料 ────────────────────
    workshop = get_workshop(req.session_id)
    data = await _get_session(req.session_id)
    ns_data: NarrativeState = data.get("narrative_state")
    round_num = ns_data.rounds if ns_data else 0
    citation_result = workshop.on_player_input(req.message, round_num)

    message = req.message
    if citation_result and citation_result.get("agent_awareness"):
        # 将史料引用感知追加到干预指令中
        message = message + "\n\n" + citation_result["agent_awareness"]

    await session_mgr.set_intervention(req.session_id, message)
    return {
        "status": "success",
        "citation": citation_result,
    }

@app.get("/api/choices/{session_id}")
async def get_choices(session_id: str):
    data = await _get_session(session_id)
    manager: ScenarioManager = data["manager"]
    ns: NarrativeState = data["narrative_state"]

    if not ns.should_offer_choices():
        return {"choices": [], "phase": ns.phase.value}

    choices = await _narrative_engine.generate_choices(
        manager.scene_desc, manager.current_dialogue,
        [a.name for a in manager.agents], ns, n=3,
    )
    ns.pending_choices = choices
    return {"choices": [c.to_dict() for c in choices], "phase": ns.phase.value}

@app.post("/api/choose")
async def submit_choice(req: ChoiceRequest):
    data = await _get_session(req.session_id)
    ns: NarrativeState = data["narrative_state"]
    manager: ScenarioManager = data["manager"]
    chosen = next((c for c in (ns.pending_choices or []) if c.id == req.choice_id), None)
    if not chosen:
        raise HTTPException(status_code=400, detail="无效选项 ID")
    directive = _narrative_engine.choice_to_intervention(chosen, ns)
    await session_mgr.set_intervention(req.session_id, directive)
    await bus.emit(Event(EventType.PLAYER_CHOSE, session_id=req.session_id,
                         payload={"choice": chosen.text}))

    # ── Phase 13A：记录玩家选择到因果链 ──────────────────────
    te = get_thinking_engine(req.session_id)
    choice_node_id = te.on_player_choice(chosen.text, ns.rounds)

    # 反事实推演（其他未选择的选项）
    alt_choices = [
        c.text for c in (ns.pending_choices or []) if c.id != req.choice_id
    ]
    if alt_choices:
        asyncio.create_task(te.expand_counterfactuals(
            chosen.text, manager.scene_desc, alt_choices,
            ns.rounds, choice_node_id,
        ))

    return {"status": "success", "directive": directive, "causal_node_id": choice_node_id}


# ═══════════════════════════════════════════════════════════════
# 核心流式推演（SSE）
# ═══════════════════════════════════════════════════════════════

@app.get("/api/stream_next/{session_id}")
async def stream_next(session_id: str, user_id: str = Query(default="anonymous")):

    session_data = await session_mgr.get(session_id)
    if session_data is None:
        # 会话恢复：尝试从磁盘加载 song 默认场景
        manager = ScenarioManager()
        try:
            await asyncio.to_thread(manager.load_era, session_id)
        except Exception:
            await asyncio.to_thread(manager.load_era, "song")
        await manager.initialize_agents()
        director = SpatiotemporalDirector(manager.agents)
        narrative_state = NarrativeState()
        await session_mgr.create(session_id, {
            "manager": manager, "agents": manager.agents,
            "director": director, "narrative_state": narrative_state,
            "user_id": user_id, "theme": "默认",
            "trigger_checker": TriggerChecker(era=session_id.split("_")[0] if "_" in session_id else "song"),
        })
        session_data = await session_mgr.get(session_id)

    manager: ScenarioManager        = session_data["manager"]
    agents                          = session_data["agents"]
    director: SpatiotemporalDirector= session_data["director"]
    ns: NarrativeState              = session_data["narrative_state"]
    session_user_id                 = session_data.get("user_id", user_id)

    profile = await asyncio.to_thread(profile_store.load, session_user_id)
    user_context = profile.to_context_summary()

    # ── Phase 12A：会话级事件队列，捕获情绪/独白 EventBus 事件 ──
    import asyncio as _asyncio
    _emotion_event_queue: _asyncio.Queue = _asyncio.Queue()

    async def _on_emotion_event(event: Event):
        if event.session_id == session_id:
            await _emotion_event_queue.put(event)

    bus.subscribe(EventType.MONOLOGUE_UNLOCKED, _on_emotion_event)
    bus.subscribe(EventType.CONCEPT_TOUCHED,    _on_emotion_event)

    async def event_generator():
        # ── 离线快进检测（蝴蝶效应）────────────────────────────
        entry = session_mgr._sessions.get(session_id)
        if entry:
            do_ff, offline_hours = should_fast_forward(entry.last_accessed)
            # 先检查是否有上次已生成但未读的 offline_log
            pending_log = get_offline_log(session_data)
            if pending_log:
                for log_entry in pending_log:
                    for msg in format_butterfly_sse({
                        "events": log_entry.get("events", []),
                        "butterfly_summary": log_entry.get("butterfly_summary", ""),
                    }):
                        yield _sse(msg)
                        await asyncio.sleep(0.3)
            elif do_ff:
                yield _sse({"type": "narrator",
                             "content": f"〔你已离开 {offline_hours:.1f} 小时，时空在流转……〕"})
                ff_result = await fast_forward(
                    session_data, offline_hours,
                    rounds=min(int(offline_hours // 2) + 1, MAX_OFFLINE_ROUNDS),
                )
                for msg in format_butterfly_sse(ff_result):
                    yield _sse(msg)
                    await asyncio.sleep(0.4)

        # ── 获取/初始化历史触发器 ─────────────────────────────
        trigger_checker: TriggerChecker = session_data.get("trigger_checker")
        if trigger_checker is None:
            era_guess = session_data.get("theme", "song")
            trigger_checker = TriggerChecker(era=era_guess)
            session_data["trigger_checker"] = trigger_checker

        # ── 当前世界情绪（供记忆检索情绪加权）────────────────
        current_emotion = manager.world_env.mood.value if manager.world_env else ""

        # ── 电影级环境描述（异步 LLM 生成）──────────────────
        env_cinematic = await manager.world_env.get_cinematic_description()
        env_text = manager.world_env.get_current_state_text()
        env_text_with_cinematic = f"{env_text}\n【本幕电影意象】: {env_cinematic}"

        # ── 消费干预指令 ──────────────────────────────────────
        intervention = await session_mgr.pop_intervention(session_id)
        if intervention:
            env_text_with_cinematic += f"\n\n【⚠️来自高维时空的低语⚠️】: {intervention}"

        # ── 导演决策 ──────────────────────────────────────────
        direction = await director.direct_next_scene(
            manager.scene_desc, manager.current_dialogue, env_text_with_cinematic
        )
        next_speaker_name = direction.get("next_speaker")
        narrator_event    = direction.get("narrator_event", "无")
        historical_echo   = direction.get("historical_echo", "无")

        if narrator_event and narrator_event != "无":
            manager.current_dialogue += f"\n【旁白】: {narrator_event}"
            yield _sse({"type": "narrator", "content": narrator_event})
            await bus.emit(Event(EventType.NARRATOR_EVENT, session_id,
                                 payload={"content": narrator_event}))
            await asyncio.sleep(0.5)

        if historical_echo and historical_echo != "无":
            yield _sse({"type": "historical_echo", "content": historical_echo})
            await bus.emit(Event(EventType.HISTORICAL_ECHO, session_id,
                                 payload={"content": historical_echo}))
            await asyncio.sleep(1.0)

        current_agent = next(
            (a for a in agents if a.name == next_speaker_name), agents[0]
        )
        yield _sse({"type": "thinking", "name": current_agent.name})

        # ── Phase 12A：确保 NPC 挂载情绪引擎 ─────────────────
        if current_agent._emotion_engine is None:
            current_agent.mount_emotion_engine(_emotion_engine)
        # ── Agent 流式生成 ────────────────────────────────────
        async for chunk in current_agent.generate_response_stream(
            manager.scene_desc, manager.current_task, manager.shared_workspace,
            manager.current_dialogue, env_text_with_cinematic,
            session_id=session_id,
            user_context=user_context,
            current_emotion=current_emotion,
        ):
            if chunk["type"] == "token":
                yield _sse({"type": "stream_token", "name": current_agent.name,
                             "content": chunk["content"]})

            elif chunk["type"] == "done":
                res = chunk["parsed_data"]
                action      = res.get("action",      "静坐")
                dialogue    = res.get("dialogue",    "…")
                contribution= res.get("contribution","无")
                show_image  = res.get("show_image",  "无")
                emotion     = res.get("emotion_keyword", "平静")
                env_impact  = res.get("env_impact")

                # ── FSM 状态转移（异步，基于本回合情绪+旁白事件）
                trigger = f"{narrator_event}; {dialogue[:100]}"
                state_changed = await manager.world_env.try_transition(trigger, emotion)
                if state_changed:
                    await bus.emit(Event(EventType.WORLD_STATE_CHANGED, session_id,
                                         payload={"new_mood": manager.world_env.mood.value}))

                if env_impact and isinstance(env_impact, dict):
                    manager.world_env.apply_impact(current_agent.name, env_impact)

                manager.current_dialogue += f"\n{current_agent.name}（{action}）：{dialogue}"
                if contribution != "无" and contribution not in manager.shared_workspace:
                    manager.shared_workspace += f"\n\n[{current_agent.name} 补充]: {contribution}"

                ns.record_milestone(historical_echo if historical_echo != "无" else "")

                # ── Phase 12B：大概念扫描 ─────────────────────
                scan_text = f"{dialogue} {action}"
                touched_concepts = _concept_engine.scan_text(session_id, scan_text, ns.rounds)
                if touched_concepts:
                    await bus.emit(Event(
                        type=EventType.CONCEPT_TOUCHED,
                        session_id=session_id,
                        payload={
                            "concepts": [c.value for c in touched_concepts],
                            "round": ns.rounds,
                        },
                    ))

                # ── Phase 12A：情绪状态快照（用于前端弧线） ──
                emotion_state = _emotion_engine.get_state(current_agent.name)
                emotion_payload = {}
                if emotion_state:
                    emotion_payload = {
                        "name": current_agent.name,
                        "emotion": emotion_state.current_emotion.value,
                        "intensity": emotion_state.intensity,
                    }

                yield _sse({
                    "type": "agent_action",
                    "name": current_agent.name,
                    "action": action,
                    "dialogue": dialogue,
                    "contribution": contribution,
                    "show_image": show_image,
                    "workspace": manager.shared_workspace,
                    "world_mood": manager.world_env.mood.value,
                    "consistency_score": res.get("consistency_score", 100),
                    # Phase 12A: 情绪状态随每次 agent_action 下发
                    "emotion_state": emotion_payload,
                    # Phase 12B: 本回合触碰的概念
                    "concepts_touched": [c.value for c in touched_concepts],
                })
                record_narrative_event("agent_action")
                update_active_sessions(await session_mgr.active_count())

                # ── Phase 13B：生成本回合探究问题（异步，不阻塞 SSE）──
                asyncio.create_task(_generate_and_cache_inquiry_questions(
                    session_id=session_id,
                    scene_desc=manager.scene_desc,
                    event_summary=f"{current_agent.name}：{dialogue[:80]}",
                    era=getattr(manager.world_env, "era", "宋代"),
                    concepts=[c.value for c in touched_concepts],
                    user_id=session_user_id,
                ))

            elif chunk["type"] == "error":
                yield _sse({"type": "error", "content": chunk["content"]})

        # ── 叙事推进 ──────────────────────────────────────────
        ns.advance_round()
        manager.world_env.advance_time()

        # ── 历史触发器检查 ────────────────────────────────────
        current_year = getattr(manager.world_env, "current_year",
                               getattr(manager.world_env, "year", 1080))
        current_location = manager.scene_desc[:80]
        agent_name_list = [a.name for a in agents]
        fired_triggers = trigger_checker.check(
            year=current_year,
            location=current_location,
            agent_names=agent_name_list,
            dialogue=manager.current_dialogue[-400:],
            milestones=ns.milestones,
        )
        for trigger in fired_triggers:
            # 旁白推送历史背景
            yield _sse({"type": "narrator", "content": trigger.narrator_text})
            await asyncio.sleep(0.6)
            # 情绪变化
            if trigger.mood_change:
                try:
                    from world_engine import WorldMood
                    manager.world_env.mood = WorldMood(trigger.mood_change)
                    yield _sse({"type": "world_mood_change",
                                 "new_mood": trigger.mood_change})
                    await asyncio.sleep(0.3)
                except (ValueError, AttributeError):
                    pass
            # 人文反思
            if trigger.reflection_insight:
                yield _sse({
                    "type": "reflection",
                    "insight": trigger.reflection_insight,
                    "reflection_question": trigger.reflection_question or "",
                    "era_fact": trigger.era_fact or "无",
                })
                await asyncio.sleep(0.3)
            # 写入里程碑 + 记录指标
            ns.record_milestone(trigger.event_name)
            record_trigger_fire(trigger.id, trigger.era)

        if ns.should_offer_choices():
            # ── Phase 12A：flush 情绪/独白 EventBus 事件到前端 SSE ──
            while not _emotion_event_queue.empty():
                evt: Event = _emotion_event_queue.get_nowait()
                if evt.type == EventType.MONOLOGUE_UNLOCKED:
                    yield _sse({
                        "type": "monologue_unlocked",
                        "name": evt.payload.get("name"),
                        "monologue": evt.payload.get("monologue"),
                        "emotion_context": evt.payload.get("emotion_context"),
                    })
                elif evt.type == EventType.CONCEPT_TOUCHED:
                    yield _sse({
                        "type": "concept_touched",
                        "concepts": evt.payload.get("concepts", []),
                        "round": evt.payload.get("round", 0),
                    })

            yield _sse({"type": "choices_ready", "round": ns.rounds})
            await bus.emit(Event(EventType.CHOICES_READY, session_id,
                                 payload={"round": ns.rounds}))

        # ── 人文反思 ──────────────────────────────────────────
        if _reflection_engine.should_trigger(ns.rounds):
            reflection = await _reflection_engine.generate(
                scene_desc=manager.scene_desc,
                dialogue_summary=manager.current_dialogue,
                player_choices=ns.player_choices,
            )
            if reflection:
                yield _sse({"type": "reflection", **reflection.to_dict()})
                await bus.emit(Event(EventType.REFLECTION_DONE, session_id,
                                     payload=reflection.to_dict()))
                profile.record_reflection(reflection.to_profile_record(session_id))
                await asyncio.to_thread(profile_store.save, profile)

        # ── 存档 ──────────────────────────────────────────────
        await asyncio.to_thread(manager.save_state, session_id)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# ═══════════════════════════════════════════════════════════════
# 新增端点
# ═══════════════════════════════════════════════════════════════

@app.get("/api/world_state/{session_id}")
async def get_world_state(session_id: str):
    """返回当前世界 FSM 状态（可用于前端氛围 UI）"""
    data = await _get_session(session_id)
    manager: ScenarioManager = data["manager"]
    we = manager.world_env
    return {
        "mood": we.mood.value,
        "time_passed": we.time_passed,
        "state_vars": we.state,
        "change_log": we.get_change_log_text(),
    }

@app.get("/api/events/{session_id}")
async def get_session_events(session_id: str, limit: int = 30):
    """返回会话事件回放（断线重连/调试用）"""
    events = bus.get_session_events(session_id, limit=limit)
    return {
        "events": [
            {"type": e.type.name, "payload": e.payload, "timestamp": e.timestamp}
            for e in events
        ]
    }

@app.post("/api/reflect")
async def trigger_reflection(req: ReflectRequest):
    data = await _get_session(req.session_id)
    manager: ScenarioManager = data["manager"]
    ns: NarrativeState = data["narrative_state"]
    reflection = await _reflection_engine.generate(
        manager.scene_desc, manager.current_dialogue,
        ns.player_choices, req.player_context,
    )
    if not reflection:
        return {"status": "skip", "message": "对话尚不足以触发深度反思"}
    profile = await asyncio.to_thread(profile_store.load, req.user_id)
    profile.record_reflection(reflection.to_profile_record(req.session_id))
    await asyncio.to_thread(profile_store.save, profile)
    return {"status": "success", **reflection.to_dict()}

@app.get("/api/profile/{user_id}")
async def get_profile(user_id: str):
    p = await asyncio.to_thread(profile_store.load, user_id)
    return {
        "user_id": user_id,
        "title": p.exploration_depth,
        "explored_eras": p.explored_eras,
        "explored_figures": p.explored_figures,
        "total_sessions": p.total_sessions,
        "total_rounds": p.total_rounds,
        "reflections_count": len(p.reflections),
        "recommended_era": p.get_recommended_era(),
        "recent_reflections": [
            {"insight": r.insight, "question": r.reflection_question}
            for r in p.reflections[-3:]
        ],
    }



# ── 知识图谱 API ─────────────────────────────────────────────
@app.get("/api/graph/{session_id}")
async def get_knowledge_graph(session_id: str):
    """
    返回当前 session 的人物关系图谱数据。
    节点 = agents，边 = 双向关系（affinity / trust）。
    """
    data = await _get_session(session_id)
    agents = data.get("agents", [])
    manager = data.get("manager")

    nodes = []
    edges = []
    seen_edges: set = set()

    for agent in agents:
        nodes.append({
            "id": agent.name,
            "label": agent.name,
            "identity": agent.identity,
            "era": getattr(agent, "era", ""),
            "task_role": agent.task_role,
        })

    for agent in agents:
        rels = agent.memory.data.get("relationships", {})
        for target_name, scores in rels.items():
            key = tuple(sorted([agent.name, target_name]))
            if key in seen_edges:
                continue
            seen_edges.add(key)
            edges.append({
                "source": agent.name,
                "target": target_name,
                "affinity": scores.get("affinity", 50),
                "trust": scores.get("trust", 50),
            })

    # 加入用户探索档案里的历史人物节点（如果不在 agents 中）
    milestones = []
    ns = data.get("narrative_state")
    if ns:
        milestones = ns.milestones[-10:]

    return {
        "nodes": nodes,
        "edges": edges,
        "milestones": milestones,
        "phase": data.get("narrative_state").phase.value if data.get("narrative_state") else "OPENING",
    }


# ── 会话后小测验 API ─────────────────────────────────────────
class QuizRequest(BaseModel):
    session_id: str
    user_id: str = "anonymous"

_quiz_client = AsyncOpenAI(api_key=_settings.API_KEY, base_url=_settings.BASE_URL)

@app.post("/api/quiz")
async def generate_quiz(req: QuizRequest):
    """
    根据本次会话的对话历史与里程碑，用 LLM 生成 3 道选择题进行记忆强化。
    """
    data = await _get_session(req.session_id)
    manager = data.get("manager")
    ns = data.get("narrative_state")

    dialogue_excerpt = (manager.current_dialogue or "")[-800:]
    milestones_text  = "; ".join(ns.milestones[-8:]) if ns else ""
    workspace_text   = (manager.shared_workspace or "")[-400:]

    prompt = f"""你是一位博学的历史教育者。根据以下时空情境，出 3 道单选题，测试玩家对本次历史体验的记忆与理解。

【本次对话摘要】
{dialogue_excerpt}

【重要叙事事件】
{milestones_text or "无"}

【时空产物】
{workspace_text or "无"}

要求：
- 每题紧扣本次剧情或历史背景，不出泛泛常识题
- 每题 4 个选项（A/B/C/D），只有 1 个正确答案
- 正确答案后给出 1-2 句简短解析
- 输出严格 JSON，格式如下，不要加任何其他文字：
[
  {{
    "q": "题目文字",
    "options": {{"A":"...","B":"...","C":"...","D":"..."}},
    "answer": "A",
    "explanation": "解析文字"
  }},
  ...
]"""

    try:
        resp = await _quiz_client.chat.completions.create(
            model=_settings.MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=800,
            temperature=0.5,
            timeout=20,
        )
        raw = resp.choices[0].message.content.strip()
        # 去掉可能的 markdown fence
        raw = raw.replace("```json", "").replace("```", "").strip()
        questions = json.loads(raw)
        return {"status": "success", "questions": questions[:3]}
    except Exception as e:
        return {"status": "error", "message": str(e), "questions": []}


# ── 情境知识气泡 API ─────────────────────────────────────────
class ExplainRequest(BaseModel):
    term: str           # 需要解释的历史名词
    context: str = ""  # 当前对话场景（可选，提升解释相关性）

_explain_client = AsyncOpenAI(api_key=_settings.API_KEY, base_url=_settings.BASE_URL)

@app.post("/api/explain")
async def explain_term(req: ExplainRequest):
    """为历史名词生成简短的情境知识气泡解释"""
    term = req.term.strip()[:30]
    if not term:
        raise HTTPException(status_code=400, detail="term 不能为空")

    prompt = (
        f"你是一位古典文化顾问。请用50字以内，简洁解释历史名词「{term}」。"
        f"{'当前场景：' + req.context[:200] if req.context else ''}"
        "要求：语言文雅，通俗易懂，如有朝代/人物/制度背景请提及。只输出解释，不要加引号或前缀。"
    )
    try:
        resp = await _explain_client.chat.completions.create(
            model=_settings.MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=120,
            temperature=0.3,
            timeout=8,
        )
        explanation = resp.choices[0].message.content.strip()
        return {"term": term, "explanation": explanation}
    except Exception as e:
        return {"term": term, "explanation": f"（暂无解释：{str(e)[:40]}）"}



# ── Phase 6: 探索连续打卡 ─────────────────────────────────
class CheckinRequest(BaseModel):
    user_id: str = "anonymous"

@app.post("/api/checkin")
async def checkin(req: CheckinRequest):
    """每日打卡：更新 streak，返回成就解锁信息"""
    profile = await asyncio.to_thread(profile_store.load, req.user_id)
    result = profile.checkin_today()
    await asyncio.to_thread(profile_store.save, profile)
    return {
        "streak": result["streak"],
        "streak_best": profile.streak_best,
        "is_new_day": result["is_new_day"],
        "newly_unlocked_badges": result["newly_unlocked_badges"],
        "all_badges": profile.badges,
        "title": profile.exploration_depth,
    }

@app.get("/api/checkin/{user_id}")
async def get_checkin_status(user_id: str):
    """获取当前打卡状态（不写入）"""
    import datetime
    profile = await asyncio.to_thread(profile_store.load, user_id)
    today = datetime.date.today().isoformat()
    return {
        "streak": profile.streak_days,
        "streak_best": profile.streak_best,
        "checked_in_today": profile.last_checkin_date == today,
        "all_badges": profile.badges,
        "title": profile.exploration_depth,
    }


# ── Phase 6: 每日时空速递 ─────────────────────────────────
class DigestRequest(BaseModel):
    user_id: str = "anonymous"

_digest_client = AsyncOpenAI(api_key=_settings.API_KEY, base_url=_settings.BASE_URL)

@app.post("/api/daily_digest")
async def generate_daily_digest(req: DigestRequest):
    """
    根据用户的探索档案，LLM 生成一段 2 分钟可读的每日历史情境速递。
    包含：一则今日史事、一个古今镜像类比、一句人文金句。
    """
    import datetime
    profile = await asyncio.to_thread(profile_store.load, req.user_id)
    today = datetime.date.today().isoformat()

    eras_text  = "、".join(profile.explored_eras[-4:]) or "中国历史"
    themes_text = "、".join(list(profile.preferred_genres.keys())[:3]) or "人文历史"
    figures_text = "、".join(profile.explored_figures[-4:]) or "古代人物"

    prompt = f"""你是「息壤」人文时空底座的每日速递主编。根据用户的探索偏好，生成一篇约 200 字的今日历史速递。

用户偏好：
- 已探索朝代：{eras_text}
- 感兴趣主题：{themes_text}
- 结识人物：{figures_text}
- 今日日期：{today}

要求输出严格 JSON（不加任何其他文字）：
{{
  "title": "速递标题（10字内，引人入胜）",
  "today_event": "今日史事（60字内，选与用户偏好朝代/主题相关的真实历史事件或节气文化）",
  "mirror": "古今镜像（50字内，将今日史事与当代生活做一类比，启发思考）",
  "quote": "人文金句（来自该朝代文人或史书，20字内，附出处）",
  "quote_source": "出处（书名或人名）",
  "invitation": "邀请语（30字内，邀请用户今日探索某个相关场景）"
}}"""

    try:
        resp = await _digest_client.chat.completions.create(
            model=_settings.MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0.7,
            timeout=20,
        )
        raw = resp.choices[0].message.content.strip().replace("```json","").replace("```","").strip()
        digest = json.loads(raw)
        # 记录生成日期
        profile.daily_digest_last = today
        await asyncio.to_thread(profile_store.save, profile)
        return {"status": "success", "date": today, "digest": digest}
    except Exception as e:
        return {"status": "error", "message": str(e)}


# ── Phase 6: 金句分享卡数据 ──────────────────────────────
class ShareCardRequest(BaseModel):
    session_id: str
    quote: str          # 要分享的对话金句
    speaker: str = ""   # 说话人
    era: str = ""       # 朝代/时代

@app.post("/api/share_card")
async def generate_share_card(req: ShareCardRequest):
    """
    返回分享卡所需的结构化数据，供前端 Canvas 渲染成精美图片。
    额外用 LLM 生成一句当代导读，让金句更有传播价值。
    """
    _sc_client = AsyncOpenAI(api_key=_settings.API_KEY, base_url=_settings.BASE_URL)
    prompt = (
        f"以下是历史情境中「{req.speaker or '古人'}」的一句话：\n「{req.quote}」\n"
        f"朝代：{req.era or '不详'}\n"
        "请用20字以内写一句当代导读，让现代读者感同身受。只输出导读，不加引号或其他文字。"
    )
    try:
        resp = await _sc_client.chat.completions.create(
            model=_settings.MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=60, temperature=0.6, timeout=10,
        )
        modern_note = resp.choices[0].message.content.strip()
    except Exception:
        modern_note = "历史深处的回响，穿越千年而来"

    return {
        "quote": req.quote,
        "speaker": req.speaker or "历史人物",
        "era": req.era or "",
        "modern_note": modern_note,
        "watermark": "息壤 · 人文时空",
    }


# ── Phase 7: 历史人物 TTS 语音 ───────────────────────────────
import io
import os

# 音色映射：不同角色类型 → OpenAI TTS voice
_VOICE_MAP = {
    "默认":   "onyx",    # 沉稳男声
    "文人":   "fable",   # 儒雅叙事
    "女性":   "nova",    # 温婉女声
    "少年":   "shimmer", # 清亮年轻
    "权贵":   "echo",    # 威严男声
    "市井":   "alloy",   # 亲切平民
}

# 时代语调前缀（注入到 TTS 文本前，引导语气）
_ERA_TONE_PREFIX = {
    "唐朝": "（以盛唐豪放之气朗声道）",
    "宋朝": "（以宋人温雅之风徐徐说道）",
    "北宋": "（以宋人温雅之风徐徐说道）",
    "明朝": "（以明代文士从容口吻说道）",
    "清朝": "（以清人严谨之态缓缓言道）",
    "先秦": "（以古雅庄重之声说道）",
}

class TTSRequest(BaseModel):
    text: str
    speaker: str = ""
    era: str = ""
    voice_type: str = "默认"   # 默认/文人/女性/少年/权贵/市井

_tts_client = AsyncOpenAI(api_key=_settings.API_KEY, base_url=_settings.BASE_URL)

@app.post("/api/tts")
async def text_to_speech(req: TTSRequest):
    """
    将对话文本转为历史人物语音。
    返回 audio/mpeg 音频流。
    """
    text = req.text.strip()[:300]
    if not text:
        raise HTTPException(status_code=400, detail="text 不能为空")

    voice = _VOICE_MAP.get(req.voice_type, "onyx")
    tone_prefix = _ERA_TONE_PREFIX.get(req.era, "")
    tts_text = tone_prefix + text if tone_prefix else text

    try:
        response = await _tts_client.audio.speech.create(
            model="tts-1",
            voice=voice,
            input=tts_text,
            speed=0.9,          # 略慢，增添古韵感
        )
        audio_bytes = response.content
        return StreamingResponse(
            io.BytesIO(audio_bytes),
            media_type="audio/mpeg",
            headers={"Content-Disposition": "inline; filename=speech.mp3"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS 生成失败：{str(e)[:100]}")


# ── Phase 7: 场景图像生成 ─────────────────────────────────────
class SceneImageRequest(BaseModel):
    session_id: str
    scene_hint: str = ""   # 可选：当前场景关键词

_img_client = AsyncOpenAI(api_key=_settings.API_KEY, base_url=_settings.BASE_URL)

# 水墨风格提示词模板
_INK_STYLE = (
    "traditional Chinese ink wash painting style, "
    "Song dynasty aesthetics, monochromatic with subtle warm tones, "
    "loose brushwork, atmospheric perspective, "
    "scholarly and poetic mood, no text, no watermark"
)

@app.post("/api/scene_image")
async def generate_scene_image(req: SceneImageRequest):
    """
    根据当前场景描述生成水墨风场景图。
    返回图像 URL 或 base64（取决于接口支持）。
    """
    data = await _get_session(req.session_id)
    manager = data.get("manager")

    # 构建场景描述
    scene_raw = getattr(manager, "current_dialogue", "") or ""
    scene_excerpt = scene_raw[-300:] if scene_raw else req.scene_hint or "古代中国文人茶室"

    # 先用 LLM 将对话场景提炼成图像提示词（英文）
    _prompt_client = AsyncOpenAI(api_key=_settings.API_KEY, base_url=_settings.BASE_URL)
    try:
        prompt_resp = await _prompt_client.chat.completions.create(
            model=_settings.MODEL_NAME,
            messages=[{
                "role": "user",
                "content": (
                    f"将以下中文历史场景描述提炼成 20 个英文关键词，用于文生图提示词。"
                    f"只输出英文关键词，逗号分隔，不加其他文字：\n{scene_excerpt}"
                )
            }],
            max_tokens=80, temperature=0.3, timeout=8,
        )
        scene_keywords = prompt_resp.choices[0].message.content.strip()
    except Exception:
        scene_keywords = req.scene_hint or "ancient Chinese scholar room, candlelight"

    full_prompt = f"{scene_keywords}, {_INK_STYLE}"

    try:
        img_resp = await _img_client.images.generate(
            model="dall-e-3",
            prompt=full_prompt,
            size="1024x576",
            quality="standard",
            n=1,
        )
        image_url = img_resp.data[0].url
        return {"status": "success", "url": image_url, "prompt": full_prompt}
    except Exception as e:
        # 如果 DALL-E 不可用，返回错误信息供前端优雅降级
        return {"status": "error", "message": str(e)[:120], "url": None}



# ══════════════════════════════════════════════════════════════
# Phase 8: 多模态知识库摄入管理 API
# ══════════════════════════════════════════════════════════════
import shutil
from fastapi import UploadFile, File, Form, BackgroundTasks

_DATA_DIR = Path(_settings.DATA_DIR)

# ── 文档上传 ──────────────────────────────────────────────────
@app.post("/api/ingest/upload")
async def upload_document(
    file: UploadFile = File(...),
    era: str = Form(...),          # 目标朝代（song/tang/ming…）
):
    """
    上传史料文档到对应朝代的 raw_documents 目录。
    支持格式：.txt / .md / .pdf
    """
    suffix = Path(file.filename).suffix.lower()
    if suffix not in {".txt", ".md", ".pdf", ".text"}:
        raise HTTPException(status_code=400, detail=f"不支持的文件格式：{suffix}")

    from ingestion.year_normalizer import normalize_era_name
    era_key = normalize_era_name(era)
    dest_dir = _DATA_DIR / "raw_documents" / era_key
    dest_dir.mkdir(parents=True, exist_ok=True)

    dest = dest_dir / file.filename
    content = await file.read()
    dest.write_bytes(content)

    return {
        "status": "uploaded",
        "filename": file.filename,
        "era": era_key,
        "size_kb": round(len(content) / 1024, 1),
        "path": str(dest.relative_to(_DATA_DIR)),
    }


# ── 后台摄入任务 ──────────────────────────────────────────────
_ingest_jobs: dict[str, dict] = {}   # job_id → status dict

def _run_ingest_job(job_id: str, era: str, skip_graph: bool, skip_index: bool):
    """后台线程：运行摄入流水线并更新任务状态"""
    _ingest_jobs[job_id]["status"] = "running"
    try:
        from ingestion.pipeline import run_ingestion
        summary = run_ingestion(
            era=era,
            skip_graph=skip_graph,
            skip_index=skip_index,
        )
        _ingest_jobs[job_id].update({
            "status": "done",
            "summary": summary,
        })
    except Exception as e:
        _ingest_jobs[job_id].update({
            "status": "error",
            "error": str(e)[:200],
        })

@app.post("/api/ingest/run")
async def trigger_ingestion(
    background_tasks: BackgroundTasks,
    era: str,
    skip_graph: bool = False,
    skip_index: bool = False,
):
    """
    触发指定朝代的摄入流水线（后台异步执行）。
    返回 job_id，通过 /api/ingest/status/{job_id} 轮询进度。
    """
    import uuid as _uuid
    job_id = _uuid.uuid4().hex[:8]
    _ingest_jobs[job_id] = {
        "job_id": job_id,
        "era": era,
        "status": "queued",
        "created_at": datetime.now().isoformat(),
    }
    background_tasks.add_task(
        _run_ingest_job, job_id, era, skip_graph, skip_index
    )
    return {"job_id": job_id, "era": era, "status": "queued"}

@app.get("/api/ingest/status/{job_id}")
async def get_ingest_job(job_id: str):
    job = _ingest_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="任务不存在")
    return job

@app.get("/api/ingest/status")
async def get_ingest_status():
    """获取知识库全局摄入状态 + 各朝代图谱概览"""
    from ingestion.pipeline import _load_status
    status = _load_status()

    # 图谱概览
    graph_summary = []
    for graph_file in sorted((_DATA_DIR / "knowledge").glob("*/graph_network.json")):
        era_name = graph_file.parent.name
        try:
            g = json.loads(graph_file.read_text(encoding="utf-8"))
            graph_summary.append({
                "era": era_name,
                "entities": len(g.get("entities", [])),
                "relations": len(g.get("relationships", [])),
            })
        except Exception:
            pass

    # raw_documents 目录概览
    raw_docs = []
    raw_base = _DATA_DIR / "raw_documents"
    if raw_base.exists():
        for era_dir in sorted(raw_base.iterdir()):
            if era_dir.is_dir():
                files = list(era_dir.glob("*"))
                doc_files = [f for f in files if f.suffix.lower() in {".txt",".md",".pdf"}]
                raw_docs.append({
                    "era": era_dir.name,
                    "file_count": len(doc_files),
                    "files": [f.name for f in doc_files[:10]],
                })

    return {
        "last_run": status.get("last_run"),
        "total_chunks": status.get("total_chunks", 0),
        "total_entities": status.get("total_entities", 0),
        "processed_files": len(status.get("processed", {})),
        "graphs": graph_summary,
        "raw_documents": raw_docs,
        "active_jobs": [j for j in _ingest_jobs.values() if j["status"] in ("queued","running")],
    }



# ══════════════════════════════════════════════════════════════
# Phase 10: 教育考评闭环
# ══════════════════════════════════════════════════════════════

# ── 知识掌握度热力图 ──────────────────────────────────────────
@app.get("/api/knowledge/heatmap/{user_id}")
async def get_knowledge_heatmap(user_id: str):
    """返回用户的知识掌握度热力图数据（按朝代分组）"""
    profile = await asyncio.to_thread(profile_store.load, user_id)
    heatmap = profile.get_mastery_heatmap()
    # 若尚无数据，返回空结构并给出提示
    if not heatmap["total_points"]:
        heatmap["hint"] = "完成小测验后，知识点将自动追踪于此。"
    return heatmap


class QuizResultRequest(BaseModel):
    user_id: str = "anonymous"
    session_id: str = ""
    era: str = ""
    answers: list[dict]   # [{question_key, question_label, correct: bool}]

@app.post("/api/knowledge/quiz_result")
async def submit_quiz_result(req: QuizResultRequest):
    """
    小测验结果提交：更新知识掌握度 + 记录测验历史。
    answers 格式：[{"question_key":"乌台诗案","question_label":"乌台诗案的起因","correct":true}]
    """
    profile = await asyncio.to_thread(profile_store.load, req.user_id)
    correct_count = 0
    for ans in req.answers:
        key    = ans.get("question_key", ans.get("question_label", "unknown"))
        label  = ans.get("question_label", key)
        correct = bool(ans.get("correct", False))
        if correct:
            correct_count += 1
        profile.update_knowledge(key, correct, era=req.era, label=label)

    profile.record_quiz(
        session_id=req.session_id,
        total=len(req.answers),
        correct=correct_count,
        era=req.era,
    )
    # 知识成就徽章
    newly_unlocked = []
    mastery_badges = {10: "博闻强识", 30: "学富五车", 60: "通古博今·学"}
    for count, badge in mastery_badges.items():
        if len(profile.knowledge_mastery) >= count and badge not in profile.badges:
            profile.badges.append(badge)
            newly_unlocked.append(badge)

    await asyncio.to_thread(profile_store.save, profile)
    return {
        "total": len(req.answers),
        "correct": correct_count,
        "score_pct": round(correct_count / max(len(req.answers), 1) * 100),
        "newly_unlocked_badges": newly_unlocked,
        "knowledge_points_tracked": len(profile.knowledge_mastery),
    }


# ── 班级共探模式 ──────────────────────────────────────────────
# 房间结构: {room_id: {"session_id", "teacher_id", "members": {user_id: role},
#                      "broadcast_log": [...最近50条], "created_at"}}
_classrooms: dict[str, dict] = {}

class ClassroomCreateRequest(BaseModel):
    teacher_id: str
    session_id: str   # 绑定的时空会话
    room_name: str = ""

class ClassroomJoinRequest(BaseModel):
    room_id: str
    user_id: str
    role: str = "student"   # student / observer

@app.post("/api/classroom/create")
async def create_classroom(req: ClassroomCreateRequest):
    """教师创建班级共探房间，绑定一个已存在的时空会话"""
    await _get_session(req.session_id)   # 验证 session 存在
    import uuid as _u
    room_id = "room_" + _u.uuid4().hex[:6]
    _classrooms[room_id] = {
        "room_id":    room_id,
        "room_name":  req.room_name or f"{req.teacher_id}的时空课堂",
        "session_id": req.session_id,
        "teacher_id": req.teacher_id,
        "members":    {req.teacher_id: "teacher"},
        "broadcast_log": [],
        "created_at": datetime.now().isoformat(),
        "paused":     False,
        "annotation": "",   # 教师暂停注释
    }
    return {"room_id": room_id, **_classrooms[room_id]}

@app.post("/api/classroom/join")
async def join_classroom(req: ClassroomJoinRequest):
    """学生加入班级房间"""
    room = _classrooms.get(req.room_id)
    if not room:
        raise HTTPException(status_code=404, detail="房间不存在")
    room["members"][req.user_id] = req.role
    return {"room_id": req.room_id, "role": req.role,
            "session_id": room["session_id"],
            "member_count": len(room["members"])}

@app.get("/api/classroom/{room_id}")
async def get_classroom(room_id: str):
    """获取班级房间状态"""
    room = _classrooms.get(room_id)
    if not room:
        raise HTTPException(status_code=404, detail="房间不存在")
    return {**room, "member_count": len(room["members"])}

class TeacherAnnotationRequest(BaseModel):
    room_id: str
    teacher_id: str
    annotation: str   # 暂停注释内容
    paused: bool = True

@app.post("/api/classroom/annotate")
async def teacher_annotate(req: TeacherAnnotationRequest):
    """
    教师暂停并发布注释（历史细节讲解）。
    前端通过 /api/classroom/stream/{room_id} 实时接收。
    """
    room = _classrooms.get(req.room_id)
    if not room:
        raise HTTPException(status_code=404, detail="房间不存在")
    if room["teacher_id"] != req.teacher_id:
        raise HTTPException(status_code=403, detail="仅教师可操作")
    room["paused"] = req.paused
    room["annotation"] = req.annotation
    # 写入广播日志
    room["broadcast_log"].append({
        "type": "teacher_annotation",
        "content": req.annotation,
        "paused": req.paused,
        "timestamp": datetime.now().isoformat(),
    })
    room["broadcast_log"] = room["broadcast_log"][-50:]
    return {"status": "ok", "paused": req.paused}

@app.get("/api/classroom/stream/{room_id}")
async def classroom_stream(room_id: str, user_id: str = Query(default="student"),
                           since: int = Query(default=0)):
    """
    班级共探 SSE 流：将教室广播日志推送给所有成员。
    since: 从第几条日志开始推送（增量拉取）
    """
    room = _classrooms.get(room_id)
    if not room:
        raise HTTPException(status_code=404, detail="房间不存在")

    async def classroom_generator():
        # 先推送未读的广播日志
        log = room["broadcast_log"]
        for entry in log[since:]:
            yield _sse(entry)
            await asyncio.sleep(0.1)

        # 长轮询：等待新广播（简单轮询，30s timeout）
        last_len = len(log)
        for _ in range(60):   # 最多等30秒
            await asyncio.sleep(0.5)
            current_log = room["broadcast_log"]
            if len(current_log) > last_len:
                for entry in current_log[last_len:]:
                    yield _sse(entry)
                last_len = len(current_log)
            # 推送心跳
            yield _sse({"type": "heartbeat", "members": len(room["members"]),
                         "paused": room["paused"]})

    return StreamingResponse(classroom_generator(), media_type="text/event-stream")

@app.get("/api/classroom/list")
async def list_classrooms():
    """列出所有活跃班级房间"""
    return [
        {"room_id": r["room_id"], "room_name": r["room_name"],
         "member_count": len(r["members"]), "created_at": r["created_at"]}
        for r in _classrooms.values()
    ]



# ════════════════════════════════════════════════════════════════
# Phase 12A · 情绪弧线 API
# ════════════════════════════════════════════════════════════════

@app.get("/api/emotion/arc/{npc_name}")
async def get_emotion_arc(npc_name: str):
    """
    获取指定 NPC 的情感弧线数据（供前端绘制弧线图）。
    返回情绪历史快照列表。
    """
    arc = _emotion_engine.get_arc_data(npc_name)
    state = _emotion_engine.get_state(npc_name)
    return {
        "npc": npc_name,
        "current_emotion": state.current_emotion.value if state else "未知",
        "intensity": state.intensity if state else 0,
        "resonance_points": state.resonance_points if state else 0,
        "arc": arc,
        "unlocked_monologues": [
            m.to_dict() for m in state.get_unlocked_monologues()
        ] if state else [],
    }


@app.get("/api/emotion/all_states")
async def get_all_emotion_states():
    """获取当前场景中所有 NPC 的情绪状态（用于教师驾驶舱）"""
    return _emotion_engine.get_all_states_summary()


# ════════════════════════════════════════════════════════════════
# Phase 12B · 大概念 API
# ════════════════════════════════════════════════════════════════

@app.get("/api/concepts/summary/{session_id}")
async def get_concept_summary(session_id: str):
    """
    获取会话的大概念探索总结卡。
    包含：触碰的概念列表、高频概念、对应探究问题。
    """
    summary = _concept_engine.get_session_summary(session_id)
    if not summary:
        return {"message": "该会话尚无概念记录"}
    return summary


@app.post("/api/concepts/analyze/{session_id}")
async def analyze_concepts_deep(session_id: str):
    """
    触发 LLM 对当前会话进行深度概念分析（会话结束时调用）。
    """
    session_data = await session_mgr.get(session_id)
    if not session_data:
        raise HTTPException(status_code=404, detail="Session not found")
    manager = session_data.get("manager")
    if not manager:
        raise HTTPException(status_code=404, detail="Manager not found")
    dialogue_summary = manager.current_dialogue[-1000:]
    concepts = await _concept_engine.analyze_with_llm(session_id, dialogue_summary)
    return {
        "session_id": session_id,
        "llm_identified_concepts": [c.value for c in concepts],
        "summary": _concept_engine.get_session_summary(session_id),
    }


@app.get("/api/concepts/questions/{session_id}")
async def get_inquiry_questions(session_id: str, n: int = 3):
    """
    获取本会话当前的探究式问题（Phase 13B 接口）。
    """
    questions = _concept_engine.get_round_questions(session_id, n=n)
    return {"session_id": session_id, "questions": questions}




# ════════════════════════════════════════════════════════════════
# Phase 13A · 历史思维可视化 API
# ════════════════════════════════════════════════════════════════

@app.get("/api/thinking/causal_graph/{session_id}")
async def get_causal_graph(session_id: str):
    """
    获取当前会话的因果链图谱数据。
    返回节点和边的结构，供前端 D3/ECharts 渲染。
    """
    te = get_thinking_engine(session_id)
    return te.get_causal_graph()


@app.post("/api/thinking/perspectives")
async def get_perspectives(
    session_id: str = Query(...),
    event_desc: str = Query(...),
):
    """生成多视角对比：同一事件，不同身份的解读"""
    data = await _get_session(session_id)
    manager: ScenarioManager = data["manager"]
    era = getattr(manager.world_env, "era", "宋代")
    te = get_thinking_engine(session_id)
    views = await te.get_perspectives(event_desc, era)
    return {"session_id": session_id, "event": event_desc, "perspectives": views}


@app.post("/api/thinking/rate_source")
async def rate_source(
    session_id: str = Query(...),
    source_text: str = Query(...),
    deep: bool = Query(default=False),
):
    """史料可信度评级（fast=关键词匹配，deep=LLM分析）"""
    te = get_thinking_engine(session_id)
    if deep:
        data = await _get_session(session_id)
        manager: ScenarioManager = data["manager"]
        era = getattr(manager.world_env, "era", "宋代")
        result = await te.rate_source_deep(source_text, era)
    else:
        result = te.rate_source_quick(source_text)
    return {"session_id": session_id, "rating": result}


@app.post("/api/thinking/puzzle/generate")
async def generate_puzzle(
    session_id: str = Query(...),
    difficulty: str = Query(default="medium"),
):
    """生成时序重建谜题"""
    data = await _get_session(session_id)
    manager: ScenarioManager = data["manager"]
    ns: NarrativeState = data["narrative_state"]
    era = getattr(manager.world_env, "era", "宋代")
    te = get_thinking_engine(session_id)
    puzzle = await te.generate_puzzle(
        manager.scene_desc, era, ns.milestones, difficulty
    )
    if not puzzle:
        raise HTTPException(status_code=422, detail="里程碑事件不足，无法生成谜题（需至少3个）")
    return puzzle


@app.post("/api/thinking/puzzle/check")
async def check_puzzle(session_id: str = Query(...), body: dict = None):
    """检查时序谜题答案"""
    te = get_thinking_engine(session_id)
    student_order = (body or {}).get("order", [])
    return te.check_puzzle_answer(student_order)


# ════════════════════════════════════════════════════════════════
# Phase 13B · 探究式问题 API
# ════════════════════════════════════════════════════════════════

# 后台任务：异步生成并缓存探究问题
async def _generate_and_cache_inquiry_questions(
    session_id: str,
    scene_desc: str,
    event_summary: str,
    era: str,
    concepts: List[str],
    user_id: str,
):
    try:
        engine = get_inquiry_engine(session_id, scene_desc, era, user_id)
        await engine.generate_round_questions(
            scene_desc, event_summary, era, concepts, n=3
        )
    except Exception as e:
        print(f"⚠️ [探究问题] 后台生成失败: {e}")


@app.get("/api/inquiry/questions/{session_id}")
async def get_inquiry_questions(session_id: str, n: int = 3):
    """获取本回合的 Bloom 分层探究问题"""
    data = await _get_session(session_id)
    manager: ScenarioManager = data["manager"]
    ns: NarrativeState = data["narrative_state"]
    era = getattr(manager.world_env, "era", "宋代")
    engine = get_inquiry_engine(session_id, manager.scene_desc, era)
    if engine._current_questions:
        return {"questions": [q.to_dict() for q in engine._current_questions]}
    # 尚未缓存，即时生成
    questions = await engine.generate_round_questions(
        manager.scene_desc,
        manager.current_dialogue[-200:],
        era,
        n=n,
    )
    return {"questions": questions}


@app.post("/api/inquiry/socratic/start")
async def start_socratic(session_id: str = Query(...), question_id: str = Query(...)):
    """开始一段苏格拉底式追问对话"""
    data = await _get_session(session_id)
    manager: ScenarioManager = data["manager"]
    era = getattr(manager.world_env, "era", "宋代")
    engine = get_inquiry_engine(session_id, manager.scene_desc, era)
    opening = engine.start_socratic(question_id)
    return {"opening": opening, "question_id": question_id}


@app.get("/api/inquiry/socratic/stream/{session_id}")
async def socratic_stream(session_id: str, student_input: str = Query(...)):
    """苏格拉底追问流式响应（SSE）"""
    data = await _get_session(session_id)
    manager: ScenarioManager = data["manager"]
    era = getattr(manager.world_env, "era", "宋代")
    engine = get_inquiry_engine(session_id, manager.scene_desc, era)

    async def generator():
        async for token in engine.socratic_stream(student_input):
            yield _sse({"type": "socratic_token", "token": token})
        yield _sse({"type": "socratic_done"})

    return StreamingResponse(generator(), media_type="text/event-stream")


@app.post("/api/inquiry/bookmark")
async def bookmark_question(session_id: str = Query(...), question_id: str = Query(...)):
    """收藏一道探究问题到学生问题本"""
    data = await _get_session(session_id)
    manager: ScenarioManager = data["manager"]
    era = getattr(manager.world_env, "era", "宋代")
    engine = get_inquiry_engine(session_id, manager.scene_desc, era)
    return engine.bookmark_question(question_id)


@app.get("/api/inquiry/notebook/{session_id}")
async def get_notebook(session_id: str):
    """获取学生的问题收藏本"""
    data = await _get_session(session_id)
    manager: ScenarioManager = data["manager"]
    era = getattr(manager.world_env, "era", "宋代")
    engine = get_inquiry_engine(session_id, manager.scene_desc, era)
    return engine.get_notebook()


@app.get("/api/inquiry/socratic/history/{session_id}")
async def get_socratic_history(session_id: str):
    """获取当前苏格拉底对话历史"""
    data = await _get_session(session_id)
    manager: ScenarioManager = data["manager"]
    era = getattr(manager.world_env, "era", "宋代")
    engine = get_inquiry_engine(session_id, manager.scene_desc, era)
    return {"history": engine.get_dialogue_history()}



# ════════════════════════════════════════════════════════════════
# Phase 14A · 史料直面工作坊 API
# ════════════════════════════════════════════════════════════════

@app.get("/api/workshop/sources")
async def list_sources(era: str = Query(default="")):
    """列出内置史料库（可按时代过滤）"""
    return get_all_sources(era)


@app.post("/api/workshop/present/{session_id}")
async def present_source(session_id: str, fragment_id: str = Query(...)):
    """
    向学生呈现一条史料片段（自动生成注释）。
    """
    await _get_session(session_id)
    fragment = get_source_by_id(fragment_id)
    if not fragment:
        raise HTTPException(status_code=404, detail=f"史料 {fragment_id} 不存在")
    workshop = get_workshop(session_id)
    data = await _get_session(session_id)
    manager: ScenarioManager = data["manager"]
    era = getattr(manager.world_env, "era", fragment.era)
    result = await workshop.present_fragment(fragment, era)
    return result


@app.get("/api/workshop/search/{session_id}")
async def search_sources(session_id: str, query: str = Query(...), era: str = Query(default="")):
    """关键词搜索史料"""
    workshop = get_workshop(session_id)
    return {"results": workshop.search_sources(query, era)}


@app.post("/api/workshop/compare/{session_id}")
async def compare_sources(
    session_id: str,
    fragment_id_a: str = Query(...),
    fragment_id_b: str = Query(...),
    event_desc: str = Query(default="当前历史事件"),
):
    """对比两条史料"""
    workshop = get_workshop(session_id)
    return await workshop.compare_fragments(fragment_id_a, fragment_id_b, event_desc)


@app.get("/api/workshop/citations/{session_id}")
async def get_citations(session_id: str):
    """获取学生引用史料的统计"""
    workshop = get_workshop(session_id)
    return workshop.get_citation_summary()


# ════════════════════════════════════════════════════════════════
# Phase 14B · 学科跨界连接 API
# ════════════════════════════════════════════════════════════════

@app.get("/api/cross/links/{session_id}")
async def get_cross_links(
    session_id: str,
    focus: str = Query(default=""),
):
    """
    获取当前场景的跨学科连接点。
    focus: 指定聚焦的学科（逗号分隔，如 '语文,地理'）
    """
    data = await _get_session(session_id)
    manager: ScenarioManager = data["manager"]
    era = getattr(manager.world_env, "era", "宋代")
    focus_subjects = [s.strip() for s in focus.split(",") if s.strip()] or None

    cs = get_creative_session(session_id)
    # 从场景和对话中提取关键词
    scene_kw = []
    for kw in ["苏轼", "贬谪", "王安石", "变法", "科举", "朝廷", "赤壁"]:
        if kw in manager.scene_desc or kw in manager.current_dialogue:
            scene_kw.append(kw)

    links = await cs.get_cross_links(
        manager.scene_desc,
        era,
        manager.current_dialogue[-300:],
        scene_keywords=scene_kw,
        focus_subjects=focus_subjects,
    )
    return {"session_id": session_id, "era": era, "links": links}


# ════════════════════════════════════════════════════════════════
# Phase 14C · 学生创作输出 API
# ════════════════════════════════════════════════════════════════

class CreationRequest(BaseModel):
    session_id: str
    creation_type: str       # "历史日记" | "历史书信" | "仿古词作" | "历史报道" | "历史短论"
    draft: str
    npc: str = ""
    event: str = ""
    user_id: str = "anonymous"


@app.post("/api/create/submit")
async def submit_creation(req: CreationRequest):
    """
    提交学生创作草稿，返回 AI 润色版本（非流式）。
    """
    await _get_session(req.session_id)
    data = await _get_session(req.session_id)
    manager: ScenarioManager = data["manager"]
    era = getattr(manager.world_env, "era", "宋代")

    type_map = {ct.value: ct for ct in CreationType}
    ctype = type_map.get(req.creation_type, CreationType.DIARY)

    cs = get_creative_session(req.session_id, req.user_id)
    creation = await cs.start_creation(ctype, req.draft, era, req.npc, req.event)
    return creation.to_dict()


@app.get("/api/create/polish/stream/{session_id}")
async def polish_stream(
    session_id: str,
    creation_type: str = Query(...),
    draft: str = Query(...),
    npc: str = Query(default=""),
    event: str = Query(default=""),
    user_id: str = Query(default="anonymous"),
):
    """
    AI 润色流式接口（SSE）。
    适合实时打字机效果展示润色过程。
    """
    data = await _get_session(session_id)
    manager: ScenarioManager = data["manager"]
    era = getattr(manager.world_env, "era", "宋代")

    type_map = {ct.value: ct for ct in CreationType}
    ctype = type_map.get(creation_type, CreationType.DIARY)
    cs = get_creative_session(session_id, user_id)

    async def generator():
        async for token in cs.polish_stream(ctype, draft, era, npc, event):
            yield _sse({"type": "polish_token", "token": token})
        yield _sse({"type": "polish_done"})

    return StreamingResponse(generator(), media_type="text/event-stream")


@app.get("/api/create/list/{session_id}")
async def list_creations(session_id: str, user_id: str = Query(default="anonymous")):
    """获取当前会话的所有创作"""
    cs = get_creative_session(session_id, user_id)
    return {"creations": cs.get_all_creations()}


@app.get("/api/create/export/{session_id}/{creation_id}")
async def export_creation(session_id: str, creation_id: str):
    """导出创作为 Markdown 文本"""
    cs = get_creative_session(session_id)
    md = cs.export_markdown(creation_id)
    if not md:
        raise HTTPException(status_code=404, detail="创作不存在")
    return {"markdown": md}


@app.get("/api/create/card/{session_id}/{creation_id}")
async def get_share_card(session_id: str, creation_id: str):
    """获取分享卡数据"""
    cs = get_creative_session(session_id)
    creation = cs.get_creation(creation_id)
    if not creation:
        raise HTTPException(status_code=404, detail="创作不存在")
    return cs.engine.build_share_card_data(creation)


if __name__ == "__main__":
    uvicorn.run("server:app", host=_settings.HOST, port=_settings.PORT,
                reload=_settings.DEBUG)
