# main_task.py
import time
from scenario_manager import ScenarioManager

def collaborate_task_loop(era_name):
    print(f"🚀 息壤引擎启动: 正在跨越时空前往 '{era_name}'\n")
    
    manager = ScenarioManager()
    agents = manager.load_era(era_name)
    
    if not agents:
        print("❌ 当前时空没有检测到任何智能体存在。")
        return

    for r in range(1, 4):
        print(f"\n{'='*20} 时空推演: 第 {r} 回合 {'='*20}")
        
        for agent in agents:
            # 获取当前的物理世界状态文本
            env_text = manager.world_env.get_current_state_text()
            
            res = agent.generate_response(
                manager.scene_desc, 
                manager.current_task, 
                manager.shared_workspace, 
                manager.current_dialogue,
                env_text   # 传入环境
            )
            
            if res:
                print(f"\n👤 [{agent.name}] (动作: {res.get('action')})")
                print(f"💡 内心: {res.get('thought')}")
                print(f"🗣️ 台词: 「{res.get('dialogue')}」")
                
                # 处理环境的改变
                env_impact = res.get('env_impact')
                if env_impact and isinstance(env_impact, dict):
                    manager.world_env.apply_impact(agent.name, env_impact)
                
                # 提取贡献并更新到公共空间
                contribution = res.get('contribution')
                if contribution and contribution != "无":
                    manager.shared_workspace += f"\n[{agent.name} 补充]: {contribution}"
                    print(f"✨ 贡献: {contribution}")
                
                manager.current_dialogue = f"{agent.name}（{res.get('action')}）：{res.get('dialogue')}"
                time.sleep(1.5)
                
        # 每过一回合，时间流逝
        manager.world_env.advance_time()
                
    print(f"\n{'='*20} 最终的时空产出物 {'='*20}")
    print(manager.shared_workspace)
    print("="*60)

if __name__ == "__main__":
    collaborate_task_loop("song")