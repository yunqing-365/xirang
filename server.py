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

# 移除全局的 manager 和 agents，改为基于 session 管理
# manager = ScenarioManager()
# agents = manager.load_era("song")
# director = SpatiotemporalDirector(agents)

# 临时存储活跃的会话和干预信息
active_sessions = {}
current_intervention = {} # 改为字典，支持多 session

# 新增：用于接收前端创建世界请求的模型
class WorldCreationRequest(BaseModel):
    theme: str

class Intervention(BaseModel):
    message: str
    session_id: str = "default" # 默认 session

@app.get("/")
async def get_index():
    return FileResponse("index.html")

# ====== 新增：创世 API ======
@app.post("/api/create_world")
async def create_world(req: WorldCreationRequest):
    """根据主题，动态生成一个全新的沙盒宇宙"""
    session_id = f"session_{uuid.uuid4().hex[:8]}"
    
    manager = ScenarioManager()
    # 调用我们刚刚在 scenario_manager 中写的动态生成逻辑
    result_session = manager.generate_dynamic_scenario(req.theme, session_id)
    
    if not result_session:
        return {"status": "error", "message": "世界生成失败"}
        
    # 加载这个新生成的世界
    agents = manager.load_era(result_session)
    director = SpatiotemporalDirector(agents)
    
    # 存入活跃会话
    active_sessions[session_id] = {
        "manager": manager,
        "agents": agents,
        "director": director
    }
    
    # 初始化环境并返回给前端展示
    env_text = manager.world_env.get_current_state_text()
    
    return {
        "status": "success", 
        "session_id": session_id,
        "scene_desc": manager.scene_desc,
        "initial_dialogue": manager.current_dialogue,
        "agents": [a.name for a in agents]
    }
# ============================

@app.post("/api/intervene")
async def post_intervention(intervention: Intervention):
    """接收用户从前端发来的干预指令"""
    global current_intervention
    current_intervention[intervention.session_id] = intervention.message
    print(f"⚡ [系统警报] 接收到高维观察者的干预 (Session: {intervention.session_id}): {intervention.message}")
    return {"status": "success", "message": "干预指令已注册，将在下一幕生效"}

# 修改流式接口以支持 Session
@app.get("/api/stream_next/{session_id}")
async def stream_next_round(session_id: str):
    """
    升级版：支持多世界的流式推送。
    """
    if session_id not in active_sessions:
        # 为了兼容之前的测试，如果找不到 session，尝试加载默认的宋代剧本
        manager = ScenarioManager()
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
        
        # 获取该 session 的干预指令
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
        
        if narrator_event and narrator_event != "无":
            manager.current_dialogue += f"\n【旁白】: {narrator_event}"
            yield f"data: {json.dumps({'type': 'narrator', 'content': narrator_event})}\n\n"
            await asyncio.sleep(1) 
            
        current_agent = next((a for a in agents if a.name == next_speaker_name), agents[0])
        
        yield f"data: {json.dumps({'type': 'thinking', 'name': current_agent.name})}\n\n"
        
        res = current_agent.generate_response(
            manager.scene_desc, 
            manager.current_task, 
            manager.shared_workspace, 
            manager.current_dialogue,
            env_text
        )
        
        if res:
            action = res.get("action", "静坐")
            dialogue = res.get("dialogue", "...")
            contribution = res.get("contribution", "无")
            show_image = res.get("show_image", "无") 
            env_impact = res.get("env_impact")
            
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
            
        manager.world_env.advance_time()
        
        # 清空该 session 的干预
        if session_id in current_intervention:
            del current_intervention[session_id]

    return StreamingResponse(event_generator(), media_type="text/event-stream")

if __name__ == "__main__":
    print("🚀 息壤引擎流式后端启动！请在浏览器访问: http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)