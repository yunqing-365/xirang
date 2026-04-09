import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from agent import SocialAgent # 引入你之前写好的智能体大脑

app = FastAPI()

# 允许跨域，方便本地 HTML 调试
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# === 初始化全局状态 ===
scene = "元丰三年冬至夜。黄州临皋亭。红泥小火炉上炖着肉。三人准备协同创作《临皋雪夜图》。"
current_task = "三人合力创作并题词，留下千古名篇。"
shared_workspace = "（画纸空白，笔墨已具。）"
current_dialogue = "（佛印铺宣纸，苏轼端酒杯，王朝云研墨。大家准备开始创作。）"

sushi = SocialAgent("苏轼", "主创文人", "豁达洒脱", {"灵感": 80}, "主笔作词。")
foyin = SocialAgent("佛印", "禅师画师", "睿智通透", {"禅定": 90}, "画家点评。")
chaoyun = SocialAgent("王朝云", "红颜知己", "温柔聪慧", {"爱意": 90}, "研墨情绪调节。")
agents = [sushi, foyin, chaoyun]
# =====================

# 路由 1：提供 HTML 页面
@app.get("/")
async def get_index():
    return FileResponse("index.html")

# 路由 2：接收前端的“下一回合”指令，调用 DeepSeek
@app.post("/api/next")
async def generate_next_round():
    global shared_workspace, current_dialogue
    
    results = []
    for agent in agents:
        res = agent.generate_response(scene, current_task, shared_workspace, current_dialogue)
        if res:
            action = res.get("action", "静坐")
            dialogue = res.get("dialogue", "...")
            contribution = res.get("contribution", "无")
            
            # 更新上下文
            current_dialogue = f"{agent.name}（{action}）：{dialogue}"
            if contribution != "无" and contribution not in shared_workspace:
                shared_workspace += f"\n\n[{agent.name}]: {contribution}"

            # 收集该智能体的数据返回给 HTML
            results.append({
                "name": agent.name,
                "action": action,
                "dialogue": dialogue,
                "contribution": contribution,
                "workspace": shared_workspace
            })
            
    return {"status": "success", "results": results}

if __name__ == "__main__":
    print("🚀 息壤引擎后端启动！请在浏览器访问: http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)