# environment.py
import random

class WorldEnvironment:
    def __init__(self, initial_vars):
        """初始化世界的动态物理参数与文学意象"""
        self.state = initial_vars if initial_vars else {}
        self.time_passed = 0
        self.current_motif = "平淡如水" # 当前的文学意象

    def apply_impact(self, agent_name, impacts):
        """接收智能体动作对环境造成的改变"""
        if not impacts or impacts == "无":
            return
            
        print(f"\n🌍 [世界引擎] {agent_name} 的行为扰动了时空法则！")
        for key, new_value in impacts.items():
            if key in self.state:
                old_value = self.state[key]
                self.state[key] = new_value
                print(f"   ↳ {key}: [{old_value}] -> [{new_value}]")
            else:
                self.state[key] = new_value
                print(f"   ↳ 新增环境状态: {key} -> [{new_value}]")

    def resonate_with_emotion(self, emotion_keyword):
        """核心升级：天人合一！根据角色的核心情绪，扭转时空环境的文学意象"""
        motifs = {
            "悲凉": ["秋风萧瑟，落叶无声", "残烛明灭，暗影浮动", "冷雨敲窗，寒意彻骨"],
            "豁达": ["清风徐来，水波不兴", "明月高悬，万里无云", "孤云出岫，悠然自得"],
            "紧张": ["乌云压顶，风雨欲来", "四周死寂，落针可闻", "惊鸟飞林，寒鸦夜啼"],
            "喜悦": ["春和景明，鸟语花香", "晨光微露，暖意融融", "微风拂柳，暗香浮动"]
        }
        
        # 简单的情感词匹配，如果没有匹配到，就维持原样或随机给一个微小的扰动
        for key, motif_list in motifs.items():
            if key in emotion_keyword:
                self.current_motif = random.choice(motif_list)
                print(f"🌫️ [美学引擎] 天人交感触发！环境意象已扭转为: {self.current_motif}")
                return
                
    def advance_time(self):
        """推动时间流逝"""
        self.time_passed += 1

    def get_current_state_text(self):
        """将环境参数翻译为大模型能懂的 prompt"""
        state_strs = [f"{k}: {v}" for k, v in self.state.items()] if self.state else ["无特殊物理变化"]
        
        return (
            f"当前经过了 {self.time_passed} 个演化回合。\n"
            f"【世界物理参数】: {', '.join(state_strs)}\n"
            f"【当前环境意象（天人交感）】: {self.current_motif}（请在对话中巧妙化用此景来借景抒情）"
        )