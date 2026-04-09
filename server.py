import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles  # 新增：用于提供图片文件
from scenario_manager import ScenarioManager

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 【关键】将我们的私有文献目录挂载到 /images 路由下，让网页可以读取图片
app.mount("/images", StaticFiles(directory="data/raw_documents"), name="images")

# 实例化全局引擎管理器并加载宋代世界
manager = ScenarioManager()
agents = manager.load_era("song")

@app.get("/")
async def get_index():
    return FileResponse("index.html")

@app.post("/api/next")
async def generate_next_round():
    results = []
    
    for agent in agents:
        # 1. 获取动态物理环境
        env_text = manager.world_env.get_current_state_text()
        
        # 2. 智能体思考
        res = agent.generate_response(
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
            show_image = res.get("show_image", "无")  # 新增：提取图片名字
            env_impact = res.get("env_impact")
            
            # 3. 改变世界物理状态
            if env_impact and isinstance(env_impact, dict):
                manager.world_env.apply_impact(agent.name, env_impact)
            
            # 4. 更新公共对话与协同产物
            manager.current_dialogue = f"{agent.name}（{action}）：{dialogue}"
            if contribution != "无" and contribution not in manager.shared_workspace:
                manager.shared_workspace += f"\n\n[{agent.name} 补充]: {contribution}"

            # 5. 打包返回给 HTML 前端
            results.append({
                "name": agent.name,
                "action": action,
                "dialogue": dialogue,
                "contribution": contribution,
                "show_image": show_image,  # 把图片名发给前端
                "workspace": manager.shared_workspace
            })
            
    # 一轮结束，时间流逝
    manager.world_env.advance_time()
            
    return {"status": "success", "results": results}

if __name__ == "__main__":
    print("🚀 息壤引擎后端启动！请在浏览器访问: http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)