# director.py
import json
import re
from openai import OpenAI
from config import API_KEY, BASE_URL, MODEL_NAME

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

class SpatiotemporalDirector:
    def __init__(self, agents):
        self.agent_names = [a.name for a in agents]

    def direct_next_scene(self, scene_desc, current_dialogue, env_text):
        """核心升级：剥离说教感，化身为沉浸式的'时空潜意识'"""
        
        prompt = f"""
        你是一个沉浸式历史沙盒的“时空潜意识”与“无形编剧”。
        【当前场景】: {scene_desc}
        【环境意象】: {env_text}
        【最新对话进展】: {current_dialogue[-800:]} 
        【在场人物】: {self.agent_names}

        任务：
        1. 决定下一个最应该接话或做出反应的人物是谁？（必须是在场人物）
        2. 像剧本旁白一样，生成一个微小的环境突发事件（如：一阵风吹灭了蜡烛）。
        3. 【潜移默化的文化引导】：仔细审视刚刚的对话。如果其中暗藏了诗词典故、历史宿命或极深的“潜台词”，请你以“时空回响”的形式，用充满文学美感、宿命感或历史沧桑感的文字，向玩家点破这层底蕴。
        ⚠️警告：绝对不要使用“这里化用了…”、“导师点评”等第三人称说教口吻！你要像一部史诗电影的画外音。
        正确示范：“（时空回响）千年后的异乡人或许不知，他此刻轻描淡写的‘微雨’，正是他半生漂泊的缩影。”
        如果没有值得点破的深意，填“无”。

        必须以纯 JSON 格式输出：
        {{
            "next_speaker": "名字",
            "narrator_event": "旁白事件描述（或 无）",
            "historical_echo": "沉浸式文化底蕴揭示（或 无）"
        }}
        """

        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.6, 
                timeout=15
            )
            raw = response.choices[0].message.content
            
            md_json = "`" * 3 + "json"
            md_empty = "`" * 3
            if raw.startswith(md_json): raw = raw[7:-3].strip()
            elif raw.startswith(md_empty): raw = raw[3:-3].strip()
            
            match = re.search(r'\{.*\}', raw, re.DOTALL)
            if match:
                return json.loads(match.group(0))
        except Exception as e:
            print(f"🎬 [导演引擎出错]: {e}")
            
        import random
        return {
            "next_speaker": random.choice(self.agent_names), 
            "narrator_event": "无",
            "historical_echo": "无"
        }