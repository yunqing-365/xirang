# scenario_manager.py
import json
import os
import re
from openai import OpenAI
from agent import SocialAgent
from config import DATA_DIR, API_KEY, BASE_URL, MODEL_NAME
from environment import WorldEnvironment
from rag_engine import KnowledgeRetriever 

# 初始化世界架构师的客户端
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

class ScenarioManager:
    def __init__(self):
        self.scenarios_base_dir = os.path.join(DATA_DIR, "scenarios")
        # 确保基础剧本目录存在
        os.makedirs(self.scenarios_base_dir, exist_ok=True)
        
        self.agents = []
        self.world_env = None  # 物理世界
        
        self.scene_desc = ""
        self.current_task = ""
        self.shared_workspace = ""
        self.current_dialogue = ""

    def generate_dynamic_scenario(self, theme: str, session_id: str):
        """
        [沙盒核心]：基于用户输入的主题，利用大模型动态生成场景和人物配置，实现真正的生成式宇宙。
        """
        print(f"\n🌌 [世界架构师] 正在根据主题「{theme}」开辟新的平行时空 (Session: {session_id})...")
        
        prompt = f"""
        你是一个历史数字人文沙盒的“世界架构师”。
        请根据用户提供的主题：【{theme}】，自动生成一个符合历史逻辑的场景剧本以及 2-3 位在场人物（可以包含历史名人或虚构的典型NPC）。
        
        必须以纯 JSON 格式输出，不要有任何多余的解释，结构如下：
        {{
            "scene": {{
                "era": "时代背景（如：明朝万历年间）",
                "location": "具体地点",
                "scene_desc": "环境与氛围的细节描写",
                "current_task": "他们聚在一起的共同目的、讨论的话题或潜在冲突",
                "initial_workspace": "当前场景的物理实体或协同资产（如：桌上的一张白纸、一壶刚泡好的茶、一份情报）",
                "initial_dialogue": "第一句开场白（带上动作）",
                "env_variables": {{"温度": "寒冷", "光线": "昏暗", "时间": "子时"}}
            }},
            "agents": [
                {{
                    "name": "姓名",
                    "identity": "身份背景",
                    "personality": "性格特点",
                    "initial_metrics": {{"心情": 80, "疲劳度": 20}},
                    "task_role": "在这个场景中的角色/动机"
                }}
            ]
        }}
        """
        
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.8 # 稍微调高温度，增加场景的创意和随机性
            )
            raw = response.choices[0].message.content.strip()
            
            # 安全解析 JSON
            md_json = "`" * 3 + "json"
            md_empty = "`" * 3
            if raw.startswith(md_json): raw = raw[7:-3].strip()
            elif raw.startswith(md_empty): raw = raw[3:-3].strip()
            
            match = re.search(r'\{.*\}', raw, re.DOTALL)
            if not match:
                raise ValueError("未能提取到有效的 JSON 结构")
                
            world_data = json.loads(match.group(0))
            
            # 创建新的时空文件夹
            target_dir = os.path.join(self.scenarios_base_dir, session_id)
            os.makedirs(target_dir, exist_ok=True)
            
            # 保存场景配置
            with open(os.path.join(target_dir, "scene.json"), 'w', encoding='utf-8') as f:
                json.dump(world_data["scene"], f, ensure_ascii=False, indent=4)
                
            # 保存人物配置
            for agent in world_data["agents"]:
                agent_filename = f"{agent['name']}.json"
                with open(os.path.join(target_dir, agent_filename), 'w', encoding='utf-8') as f:
                    json.dump(agent, f, ensure_ascii=False, indent=4)
                    
            print(f"✅ 时空创世成功！新世界已落盘至: {target_dir}")
            return session_id
            
        except Exception as e:
            print(f"❌ 动态场景生成失败: {e}")
            return None

    def load_era(self, era_folder_name):
        """加载指定的时空节点（支持加载动态生成的 session）"""
        era_path = os.path.join(self.scenarios_base_dir, era_folder_name)
        if not os.path.exists(era_path):
            raise FileNotFoundError(f"❌ 找不到时代文件夹: {era_path}")

        # 1. 加载场景与物理环境
        scene_file = os.path.join(era_path, "scene.json")
        with open(scene_file, 'r', encoding='utf-8') as f:
            scene_data = json.load(f)
            
        self.scene_desc = f"【所处时代】{scene_data.get('era')}\n【地点】{scene_data.get('location')}\n【环境】{scene_data.get('scene_desc')}"
        self.current_task = scene_data.get('current_task', '')
        self.shared_workspace = scene_data.get('initial_workspace', '')
        self.current_dialogue = scene_data.get('initial_dialogue', '')
        
        # 初始化物理世界
        env_vars = scene_data.get('env_variables', {})
        self.world_env = WorldEnvironment(env_vars)

        # 2. 加载人物剧本
        self.agents = []
        for file_name in os.listdir(era_path):
            if file_name.endswith('.json') and file_name != 'scene.json':
                agent_file = os.path.join(era_path, file_name)
                with open(agent_file, 'r', encoding='utf-8') as f:
                    try:
                        agent_data = json.load(f)
                        
                        # 【终极防御加强版】：检查是否为字典，且确保所有必需字段都完全具备！
                        required_keys = ['name', 'identity', 'personality', 'initial_metrics', 'task_role']
                        if not isinstance(agent_data, dict) or not all(k in agent_data for k in required_keys):
                            continue
                            
                        agent = SocialAgent(
                            name=agent_data['name'],
                            identity=agent_data['identity'],
                            personality=agent_data['personality'],
                            initial_metrics=agent_data['initial_metrics'],
                            task_role=agent_data['task_role']
                        )
                        self.agents.append(agent)
                    except Exception as e:
                        print(f"⚠️ 跳过无法解析的文件 {file_name}: {e}")
                    
        # 3. 挂载 RAG 引擎
        # 这里动态场景默认复用其时代大背景的知识库，如果 era_folder_name 是具体朝代则加载该朝代
        # 如果是沙盒 session，为了演示，我们可以先不挂载知识库或挂载一个通用的
        try:
            knowledge_engine = KnowledgeRetriever(era_folder_name)
            for agent in self.agents:
                agent.mount_knowledge(knowledge_engine)
        except Exception as e:
            print(f"ℹ️ 当前场景为动态沙盒，暂未绑定专属离线 RAG 知识库。")

        print(f"✅ 成功加载时空节点: [{scene_data.get('era')}] - 共唤醒了 {len(self.agents)} 位数字生命")
        return self.agents