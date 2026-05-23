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
from openai import AsyncOpenAI
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


if __name__ == "__main__":
    uvicorn.run("server:app", host=_settings.HOST, port=_settings.PORT,
                reload=_settings.DEBUG)
