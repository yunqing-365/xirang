# director.py
import json
import re
from openai import OpenAI
from config import API_KEY, BASE_URL, MODEL_NAME

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

class SpatiotemporalDirector:
    def __init__(self, agents):
        # 获取所有在场智能体的名字
        self.agent_names = [a.name for a in agents]

    def direct_next_scene(self, scene_desc, current_dialogue, env_text):
        """导演评估局势，决定下一个谁发言，并偶尔注入环境突发事件"""
        
        prompt = f"""
        你是一个历史情境的“隐形导演”。
        【当前场景】: {scene_desc}
        【动态环境】: {env_text}
        【最新对话进展】: {current_dialogue[-600:]} # 只看最近的对话
        【在场人物】: {self.agent_names}

        任务：
        1. 根据当前对话的语境和冲突，决定下一个最应该接话或做出反应的人是谁？（必须是在场人物之一）
        2. 像剧本旁白一样，随机注入一个微小的环境突发事件（例如：窗外传来打更声、火炉突然爆出火星、一阵江风吹过）。这能增加世界的真实感。如果觉得当前不需要突发事件，填“无”。

        必须以纯 JSON 格式输出：
        {{
            "next_speaker": "名字",
            "narrator_event": "旁白事件描述（或 无）"
        }}
        """

        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.6, # 导演需要一定的随机性和创造力
                timeout=15
            )
            raw = response.choices[0].message.content
            match = re.search(r'\{.*\}', raw, re.DOTALL)
            if match:
                return json.loads(match.group(0))
        except Exception as e:
            print(f"🎬 [导演引擎出错]: {e}")
            
        # 异常时的降级方案：随机点名
        import random
        return {"next_speaker": random.choice(self.agent_names), "narrator_event": "无"}