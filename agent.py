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
        self.task_role = task_role
        self.memory = SocialMemory(name)
        self.rag_engine = None # 预留外接知识引擎的接口

    def mount_knowledge(self, rag_engine):
        self.rag_engine = rag_engine

    def generate_response(self, scene_desc, current_task, shared_workspace, current_dialogue, env_state_text):
        relationships_str = json.dumps(self.memory.data["relationships"], ensure_ascii=False)
        
        # 1. 检索全局知识 (RAG)
        search_query = f"{current_task} {current_dialogue}"
        reference_knowledge = "无"
        if self.rag_engine:
            reference_knowledge = self.rag_engine.retrieve(search_query)
            
        # 2. 唤醒个人专属情境记忆 (Vector Memory)
        past_memories = self.memory.retrieve_episodic_memory(current_dialogue)

        system_prompt = f"""
你是【{self.name}】。
身份：{self.identity}
性格：{self.personality}
精神状态：{json.dumps(self.metrics, ensure_ascii=False)}

【社会关系记忆】: {relationships_str}
【当前场景】: {scene_desc}

=== 动态物理环境 ===
{env_state_text}

=== 专属时空知识检索结果 ===
（以下是系统为你检索到的相关记忆或知识。💡提示：如果其中包含类似【视觉文献来源：xxx.jpg】的标记，说明这是你可以直接向大家展示的画作或实物！）
{reference_knowledge}

=== 【你的往昔记忆涌上心头】 ===
以下是你过去经历过的相关情境，可作为你当前对话的情感和事实参考：
{past_memories}

=== 协同任务系统 ===
【团队共同任务】: {current_task}
【你在团队中的职责】: {self.task_role}
【当前的团队协作产出物】: {shared_workspace}

必须以纯 JSON 格式输出：
{{
    "perception_of_others": "【心智理论】推测当前环境中其他人的心理状态、情绪或潜在意图",
    "thought": "结合他人意图、你的往昔记忆以及感官环境，进行下一步的行动分析",
    "target": "你话语的对象",
    "action": "动作描写",
    "dialogue": "台词（可以适当感叹或呼应记忆中的往事）",
    "contribution": "对协作产出的实质性补充（无则填无）",
    "show_image": "如果你想展示知识库中提到的视觉文献，请提取其文件名（如 xxx.jpg），如果没有则填 无",
    "env_impact": {{"环境参数名": "改变后的状态"}},
    "social_impact": {{"对象名": {{"affinity": 变化值, "trust": 变化值}}}}
}}
"""

        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": current_dialogue[-500:]} # 限制一下长度
                ],
                temperature=0.7,
                timeout=30
            )
            raw = response.choices[0].message.content
            
            # 处理 markdown 包裹问题
            if raw.startswith("```json"): 
                raw = raw[7:-3].strip()
            elif raw.startswith("```"): 
                raw = raw[3:-3].strip()
                
            match = re.search(r'\{.*\}', raw, re.DOTALL)
            if match:
                res = json.loads(match.group(0))
                
                # 持久化社会关系更新
                impact = res.get("social_impact", {})
                for target, changes in impact.items():
                    self.memory.update_relationship(target, changes.get("affinity", 0), changes.get("trust", 0))
                self.memory.save()
                
                # ====== 新增：动作发生后，写入长期记忆库 ======
                self.memory.add_episodic_memory(
                    env_state=env_state_text,
                    action=res.get("action", ""),
                    dialogue=res.get("dialogue", "")
                )
                
                # 你可以在控制台打印一下心智理论的推理过程，非常有趣！
                print(f"🧩 [{self.name} 的揣测]: {res.get('perception_of_others', '')}")
                
                return res
        except Exception as e:
            print(f"[{self.name}] 引擎出错: {e}")
        return None