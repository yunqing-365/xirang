# environment.py
class WorldEnvironment:
    def __init__(self, initial_vars):
        """初始化世界的动态物理参数"""
        # 如果剧本里没有定义，就给个空的字典
        self.state = initial_vars if initial_vars else {}
        self.time_passed = 0

    def apply_impact(self, agent_name, impacts):
        """接收智能体动作对环境造成的改变"""
        if not impacts or impacts == "无":
            return
            
        print(f"\n🌍 [世界引擎] {agent_name} 的行为改变了环境！")
        for key, new_value in impacts.items():
            if key in self.state:
                old_value = self.state[key]
                self.state[key] = new_value
                print(f"   ↳ {key}: [{old_value}] -> [{new_value}]")
            else:
                # 允许智能体创造新的环境参数
                self.state[key] = new_value
                print(f"   ↳ 新增环境状态: {key} -> [{new_value}]")

    def advance_time(self):
        """推动时间流逝"""
        self.time_passed += 1

    def get_current_state_text(self):
        """将环境参数翻译为大模型能懂的 prompt"""
        if not self.state:
            return "环境无特殊动态变化。"
        
        state_strs = [f"{k}: {v}" for k, v in self.state.items()]
        return f"当前经过了 {self.time_passed} 个演化回合。\n【世界物理参数】: " + ", ".join(state_strs)