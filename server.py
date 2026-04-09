# server.py
import uvicorn
import json
import asyncio
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

manager = ScenarioManager()
agents = manager.load_era("song")
director = SpatiotemporalDirector(agents)

# 新增：用于接收前端“神之旨意”（用户干预）的模型
class Intervention(BaseModel):
    message: str

# 临时存储用户的干预信息
current_intervention = None

@app.get("/")
async def get_index():
    return FileResponse("index.html")

@app.post("/api/intervene")
async def post_intervention(intervention: Intervention):
    """接收用户从前端发来的干预指令"""
    global current_intervention
    current_intervention = intervention.message
    print(f"⚡ [系统警报] 接收到高维观察者的干预: {current_intervention}")
    return {"status": "success", "message": "干预指令已注册，将在下一幕生效"}

@app.get("/api/stream_next")
async def stream_next_round():
    """
    升级版：使用 Server-Sent Events (SSE) 机制流式推送剧情。
    这样前端就可以实现一个字一个字蹦出来的“打字机”效果。
    """
    global current_intervention
    
    async def event_generator():
        env_text = manager.world_env.get_current_state_text()
        
        # 将用户的干预信息秘密注入到环境文本中，让导演和智能体都能感知到
        if current_intervention:
            env_text += f"\n\n【⚠️来自高维时空（神）的低语⚠️】: {current_intervention}"
            
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
            await asyncio.sleep(1) # 旁白后稍微停顿
            
        current_agent = next((a for a in agents if a.name == next_speaker_name), agents[0])
        
        # 通知前端：某某智能体正在思考
        yield f"data: {json.dumps({'type': 'thinking', 'name': current_agent.name})}\n\n"
        
        # （由于你底层的 OpenAI 调用目前是阻塞的，真正的流式文本需要改底层。
        # 这里我们模拟：先等完整结果出来，然后把结果按块推送给前端）
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

            # 打包最终数据推给前端
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
        
        # 一幕演完，清空当前的干预指令
        global current_intervention
        current_intervention = None

    return StreamingResponse(event_generator(), media_type="text/event-stream")

if __name__ == "__main__":
    print("🚀 息壤引擎流式后端启动！请在浏览器访问: http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)