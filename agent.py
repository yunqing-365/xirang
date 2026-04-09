# agent.py
import json
import re
from openai import OpenAI
from config import API_KEY, BASE_URL, MODEL_NAME
from memory import SocialMemory

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

class SocialAgent:
    def __init__(self, name, identity, personality, initial_metrics, task_role):
        self.name = name
        self.identity = identity
        self.personality = personality
        self.metrics = initial_metrics
        self.task_role = task_role  # 【新增】在团队协同中的具体职责
        self.memory = SocialMemory(name)

    def generate_response(self, scene_desc, current_task, shared_workspace, current_dialogue):
        relationships_str = json.dumps(self.memory.data["relationships"], ensure_ascii=False)
        
        system_prompt = f"""
你是【{self.name}】。
身份：{self.identity}
性格：{self.personality}
精神状态：{json.dumps(self.metrics, ensure_ascii=False)}

【社会关系记忆】: {relationships_str}
【当前场景】: {scene_desc}

=== 协同任务系统 ===
【团队共同任务】: {current_task}
【你在团队中的职责】: {self.task_role}
【当前的团队协作产出物(如草稿/计划)】: {shared_workspace}

任务要求：
根据你的性格、对他人的印象以及你的【职责】，与其他智能体协同推进任务。
你可以提出修改意见、直接贡献内容，或者给予情绪支持。

必须以纯 JSON 格式输出：
{{
    "thought": "对当前任务进度和社交局势的分析",
    "target": "你话语的对象（所有人或特定某人）",
    "action": "动作描写",
    "dialogue": "台词",
    "contribution": "你对协同任务做出的实质性补充或修改建议（如果没有则填无）",
    "social_impact": {{
        "对象名": {{"affinity": 变化值, "trust": 变化值}}
    }}
}}
"""
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": current_dialogue}
                ],
                temperature=0.7,
                timeout=30
            )
            raw = response.choices[0].message.content
            match = re.search(r'\{.*\}', raw, re.DOTALL)
            if match:
                res = json.loads(match.group(0))
                
                # 持久化社会关系更新
                impact = res.get("social_impact", {})
                for target, changes in impact.items():
                    self.memory.update_relationship(target, changes.get("affinity", 0), changes.get("trust", 0))
                self.memory.save()
                return res
        except Exception as e:
            print(f"[{self.name}] 引擎出错: {e}")
        return None