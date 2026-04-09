# main_task.py
import time
from agent import SocialAgent

def collaborate_task_loop():
    print("🚀 息壤 4.0 模块化启动: 协同创作网络\n")
    
    # 1. 设定全局场景与任务
    scene = "大雪夜，临皋亭。红泥小火炉上炖着肉。三人围坐，桌上铺着笔墨纸砚。"
    current_task = "三人共同创作一幅《临皋雪夜图》并题上一首词，以此排解被贬的苦闷，留下千古名篇。"
    shared_workspace = "【空白的画纸】目前还没有任何人下笔。" # 这是大家共享的“黑板”

    # 2. 实例化智能体（分配不同的团队角色）
    sushi = SocialAgent(
        name="苏轼", 
        identity="主创，被贬文人", 
        personality="豁达洒脱，文采飞扬，急公好义", 
        initial_metrics={"创作灵感": 80, "快乐": 50},
        task_role="主笔/作词。负责定下诗词的基调，写出核心句子。"
    )
    
    foyin = SocialAgent(
        name="佛印", 
        identity="精神导师/画师，苏轼损友", 
        personality="睿智通透，爱调侃", 
        initial_metrics={"禅定": 90, "诙谐": 80},
        task_role="画家/点评。负责在纸上画图，并用佛法禅意修改苏轼诗词中过于悲凉的句子。"
    )
    
    chaoyun = SocialAgent(
        name="王朝云", 
        identity="情绪后勤/品鉴者", 
        personality="温柔聪慧，极具艺术品鉴力", 
        initial_metrics={"爱意": 90, "敏锐度": 85},
        task_role="研墨/情绪调节。负责给苏轼倒酒研墨，如果苏轼和佛印吵起来负责劝架，并补充细腻的字眼。"
    )

    agents = [sushi, foyin, chaoyun]
    current_dialogue = "（佛印把宣纸铺好，苏轼端起酒杯，王朝云开始在一旁研墨。大家准备开始创作。）"
    
    # 3. 开启协同创作回合
    for r in range(1, 4):
        print(f"\n{'='*20} 创作推进: 第 {r} 回合 {'='*20}")
        
        for agent in agents:
            res = agent.generate_response(scene, current_task, shared_workspace, current_dialogue)
            if res:
                print(f"\n👤 [{agent.name}] (动作: {res.get('action')})")
                print(f"🗣️  台词: 「{res.get('dialogue')}」")
                print(f"✨  对任务的贡献: {res.get('contribution')}")
                
                # 提取出智能体的贡献，更新到全局的 Shared Workspace 中
                contribution = res.get('contribution')
                if contribution and contribution != "无":
                    shared_workspace += f"\n[{agent.name} 补充]: {contribution}"
                
                # 更新公共对话流，让下一个人知道刚才发生了什么
                current_dialogue = f"{agent.name}（{res['action']}）：{res['dialogue']}"
                time.sleep(1.5)
                
    print(f"\n{'='*20} 最终的协同产出物 {'='*20}")
    print(shared_workspace)
    print("="*60)

if __name__ == "__main__":
    collaborate_task_loop()