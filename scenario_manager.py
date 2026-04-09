# scenario_manager.py
import json
import os
from agent import SocialAgent
from config import DATA_DIR
from environment import WorldEnvironment
from rag_engine import KnowledgeRetriever 

class ScenarioManager:
    def __init__(self):
        self.scenarios_base_dir = os.path.join(DATA_DIR, "scenarios")
        self.agents = []
        self.world_env = None  # 物理世界
        
        self.scene_desc = ""
        self.current_task = ""
        self.shared_workspace = ""
        self.current_dialogue = ""

    def load_era(self, era_folder_name):
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
                        
                        # 终极防御：如果 JSON 里根本没有 'name' 字段，说明它绝不是人物设定文件，直接跳过！
                        if 'name' not in agent_data:
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
        knowledge_engine = KnowledgeRetriever(era_folder_name)
        for agent in self.agents:
            agent.mount_knowledge(knowledge_engine)

        print(f"✅ 成功加载时空节点: [{scene_data.get('era')}] - 共唤醒了 {len(self.agents)} 位数字生命")
        return self.agents