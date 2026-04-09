# server.py
import uvicorn
import json
import asyncio
import uuid
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from scenario_manager import ScenarioManager
from director import SpatiotemporalDirector

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/images", StaticFiles(directory="data/raw_documents"), name="images")

active_sessions = {}
current_intervention = {} 

class WorldCreationRequest(BaseModel):
    theme: str
    genre: str = "历史客观写实" # 默认风格

class Intervention(BaseModel):
    message: str
    session_id: str = "default" 

@app.get("/")
async def get_index():
    return FileResponse("index.html")

@app.post("/api/create_world")
async def create_world(req: WorldCreationRequest):
    """根据主题和风格，动态生成一个全新的沙盒宇宙"""
    session_id = f"session_{uuid.uuid4().hex[:8]}"
    
    manager = ScenarioManager()
    result_session = manager.generate_dynamic_scenario(req.theme, req.genre, session_id)
    
    if not result_session:
        return {"status": "error", "message": "世界生成失败"}
        
    agents = manager.load_era(result_session)
    director = SpatiotemporalDirector(agents)
    
    active_sessions[session_id] = {
        "manager": manager,
        "agents": agents,
        "director": director
    }
    
    env_text = manager.world_env.get_current_state_text()
    
    return {
        "status": "success", 
        "session_id": session_id,
        "scene_desc": manager.scene_desc,
        "initial_dialogue": manager.current_dialogue,
        "agents": [a.name for a in agents]
    }

@app.post("/api/intervene")
async def post_intervention(intervention: Intervention):
    """接收用户从前端发来的干预指令"""
    global current_intervention
    current_intervention[intervention.session_id] = intervention.message
    print(f"⚡ [系统警报] 接收到高维观察者的干预 (Session: {intervention.session_id}): {intervention.message}")
    return {"status": "success", "message": "干预指令已注册，将在下一幕生效"}

@app.get("/api/stream_next/{session_id}")
async def stream_next_round(session_id: str):
    """
    核心升级：支持断线重连读档与自动存档，并渲染「时空回响」
    """
    if session_id not in active_sessions:
        manager = ScenarioManager()
        try:
            print(f"📡 内存无活跃会话，尝试从硬盘唤醒存档: {session_id}")
            agents = manager.load_era(session_id)
        except Exception as e:
            print(f"⚠️ 找不到存档，回退到默认剧本: {e}")
            agents = manager.load_era("song")
            
        director = SpatiotemporalDirector(agents)
        active_sessions[session_id] = {
            "manager": manager,
            "agents": agents,
            "director": director
        }

    session_data = active_sessions[session_id]
    manager = session_data["manager"]
    agents = session_data["agents"]
    director = session_data["director"]

    global current_intervention
    
    async def event_generator():
        env_text = manager.world_env.get_current_state_text()
        
        intervention_msg = current_intervention.get(session_id)
        if intervention_msg:
            env_text += f"\n\n【⚠️来自高维时空（神）的低语⚠️】: {intervention_msg}"
            
        direction = director.direct_next_scene(
            manager.scene_desc, 
            manager.current_dialogue, 
            env_text
        )
        
        next_speaker_name = direction.get("next_speaker")
        narrator_event = direction.get("narrator_event", "无")
        historical_echo = direction.get("historical_echo", "无") # ⚡ 提取时空回响
        
        # 1. 渲染普通旁白或突发事件
        if narrator_event and narrator_event != "无":
            manager.current_dialogue += f"\n【旁白】: {narrator_event}"
            yield f"data: {json.dumps({'type': 'narrator', 'content': narrator_event})}\n\n"
            await asyncio.sleep(1) 
            
        # 2. ⚡ 渲染极具沉浸感的「时空回响」
        if historical_echo and historical_echo != "无":
            # 同样作为旁白推给前端展示，达到“润物细无声”的文化引导效果
            yield f"data: {json.dumps({'type': 'narrator', 'content': historical_echo})}\n\n"
            await asyncio.sleep(1.5)
            
        current_agent = next((a for a in agents if a.name == next_speaker_name), agents[0])
        
        yield f"data: {json.dumps({'type': 'thinking', 'name': current_agent.name})}\n\n"
        
        response_stream = current_agent.generate_response_stream(
            manager.scene_desc, 
            manager.current_task, 
            manager.shared_workspace, 
            manager.current_dialogue,
            env_text
        )
        
        for chunk in response_stream:
            if chunk["type"] == "token":
                yield f"data: {json.dumps({'type': 'stream_token', 'name': current_agent.name, 'content': chunk['content']})}\n\n"
                
            elif chunk["type"] == "done":
                res = chunk["parsed_data"]
                action = res.get("action", "静坐")
                dialogue = res.get("dialogue", "...")
                contribution = res.get("contribution", "无")
                show_image = res.get("show_image", "无") 
                env_impact = res.get("env_impact")
                
                # 触发天人合一环境变化 (由 environment.py 处理)
                emotion = res.get("emotion_keyword")
                if emotion:
                    manager.world_env.resonate_with_emotion(emotion)
                
                if env_impact and isinstance(env_impact, dict):
                    manager.world_env.apply_impact(current_agent.name, env_impact)
                
                manager.current_dialogue += f"\n{current_agent.name}（{action}）：{dialogue}"
                if contribution != "无" and contribution not in manager.shared_workspace:
                    manager.shared_workspace += f"\n\n[{current_agent.name} 补充]: {contribution}"

                payload = {
                    "type": "agent_action",
                    "name": current_agent.name,
                    "action": action,
                    "dialogue": dialogue,
                    "contribution": contribution,
                    "show_image": show_image,
                    "workspace": manager.shared_workspace
                }
                yield f"data: {json.dumps(payload)}\n\n"
                
            elif chunk["type"] == "error":
                yield f"data: {json.dumps({'type': 'error', 'content': chunk['content']})}\n\n"
            
        manager.world_env.advance_time()
        
        # 演进完毕，触发自动存档
        manager.save_state(session_id)
        
        if session_id in current_intervention:
            del current_intervention[session_id]

    return StreamingResponse(event_generator(), media_type="text/event-stream")

if __name__ == "__main__":
    print("🚀 息壤 MMO 引擎后端启动！请在浏览器访问: http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)