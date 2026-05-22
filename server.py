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

import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from config import get_settings
from director import SpatiotemporalDirector
from event_bus import bus, Event, EventType
from narrative_engine import NarrativeEngine, NarrativeState
from reflection_engine import ReflectionEngine
from scenario_manager import ScenarioManager
from session_manager import session_mgr
from user_profile import profile_store

_settings = get_settings()
_narrative_engine = NarrativeEngine()
_reflection_engine = ReflectionEngine()


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
    await session_mgr.set_intervention(req.session_id, req.message)
    return {"status": "success"}

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
    chosen = next((c for c in (ns.pending_choices or []) if c.id == req.choice_id), None)
    if not chosen:
        raise HTTPException(status_code=400, detail="无效选项 ID")
    directive = _narrative_engine.choice_to_intervention(chosen, ns)
    await session_mgr.set_intervention(req.session_id, directive)
    await bus.emit(Event(EventType.PLAYER_CHOSE, session_id=req.session_id,
                         payload={"choice": chosen.text}))
    return {"status": "success", "directive": directive}


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
        })
        session_data = await session_mgr.get(session_id)

    manager: ScenarioManager        = session_data["manager"]
    agents                          = session_data["agents"]
    director: SpatiotemporalDirector= session_data["director"]
    ns: NarrativeState              = session_data["narrative_state"]
    session_user_id                 = session_data.get("user_id", user_id)

    profile = await asyncio.to_thread(profile_store.load, session_user_id)
    user_context = profile.to_context_summary()

    async def event_generator():
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
                })

            elif chunk["type"] == "error":
                yield _sse({"type": "error", "content": chunk["content"]})

        # ── 叙事推进 ──────────────────────────────────────────
        ns.advance_round()
        manager.world_env.advance_time()

        if ns.should_offer_choices():
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


if __name__ == "__main__":
    uvicorn.run("server:app", host=_settings.HOST, port=_settings.PORT,
                reload=_settings.DEBUG)
