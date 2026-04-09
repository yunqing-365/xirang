# scenario_manager.py
import json
import os
import re
from openai import OpenAI
from agent import SocialAgent
from config import DATA_DIR, API_KEY, BASE_URL, MODEL_NAME
from environment import WorldEnvironment
from rag_engine import KnowledgeRetriever 

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

class ScenarioManager:
    def __init__(self):
        self.scenarios_base_dir = os.path.join(DATA_DIR, "scenarios")
        os.makedirs(self.scenarios_base_dir, exist_ok=True)
        
        self.agents = []
        self.world_env = None  
        
        self.scene_desc = ""
        self.current_task = ""
        self.shared_workspace = ""
        self.current_dialogue = ""

    # =================================================================
    # 核心升级 1：生态扩容，加入 genre（风格/题材）参数，拒绝写死！
    # =================================================================
    def generate_dynamic_scenario(self, theme: str, genre: str, session_id: str):
        print(f"\n🌌 [世界架构师] 正在根据主题「{theme}」(世界观: {genre}) 开辟平行时空 (Session: {session_id})...")
        
        prompt = f"""
        你是一个历史数字人文沙盒的“世界架构师”。
        请根据用户提供的主题：【{theme}】和 世界观风格：【{genre}】，自动生成一个符合该逻辑的场景剧本以及 2-3 位在场人物。
        
        【风格动态约束提示】：
        - 如果风格偏向“宫廷权谋/政治”，请强调阶级森严、暗藏杀机，人物对话需谨慎克制。
        - 如果风格偏向“市井烟火”，请多用俗语俚语，关注柴米油盐和银钱交易。
        - 如果风格偏向“凄美宿命/红楼梦”，请强调细腻的心理描写、草木枯荣和宿命感。
        - 如果没有特定风格，请保持历史客观写实。

        必须以纯 JSON 格式输出，不要有任何多余解释，结构如下：
        {{
            "scene": {{
                "era": "时代背景（如：明朝万历年间）",
                "location": "具体地点",
                "scene_desc": "环境与氛围的细节描写",
                "current_task": "他们聚在一起的共同目的、讨论的话题或潜在冲突",
                "initial_workspace": "当前场景的物理实体或协同资产",
                "initial_dialogue": "第一句开场白（带上动作）",
                "env_variables": {{"温度": "寒冷", "光线": "昏暗", "时间": "子时"}}
            }},
            "agents": [
                {{
                    "name": "姓名",
                    "identity": "身份背景",
                    "personality": "性格特点",
                    "initial_metrics": {{"心情": 80, "疲劳度": 20}},
                    "task_role": "在这个场景中的动机"
                }}
            ]
        }}
        """
        
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.8
            )
            raw = response.choices[0].message.content.strip()
            
            md_json = "`" * 3 + "json"
            md_empty = "`" * 3
            if raw.startswith(md_json): raw = raw[7:-3].strip()
            elif raw.startswith(md_empty): raw = raw[3:-3].strip()
            
            match = re.search(r'\{.*\}', raw, re.DOTALL)
            if not match:
                raise ValueError("未能提取到有效的 JSON 结构")
                
            world_data = json.loads(match.group(0))
            
            target_dir = os.path.join(self.scenarios_base_dir, session_id)
            os.makedirs(target_dir, exist_ok=True)
            
            with open(os.path.join(target_dir, "scene.json"), 'w', encoding='utf-8') as f:
                json.dump(world_data["scene"], f, ensure_ascii=False, indent=4)
                
            for agent in world_data["agents"]:
                agent_filename = f"{agent['name']}.json"
                with open(os.path.join(target_dir, agent_filename), 'w', encoding='utf-8') as f:
                    json.dump(agent, f, ensure_ascii=False, indent=4)
                    
            print(f"✅ 时空创世成功！新世界已落盘至: {target_dir}")
            return session_id
            
        except Exception as e:
            print(f"❌ 动态场景生成失败: {e}")
            return None

    # =================================================================
    # 核心升级 2：状态唤醒 (Awaken) - 读取动态进度
    # =================================================================
    def load_era(self, era_folder_name):
        era_path = os.path.join(self.scenarios_base_dir, era_folder_name)
        if not os.path.exists(era_path):
            raise FileNotFoundError(f"❌ 找不到时代文件夹: {era_path}")

        scene_file = os.path.join(era_path, "scene.json")
        state_file = os.path.join(era_path, "state.json") # 动态进度存档文件
        
        with open(scene_file, 'r', encoding='utf-8') as f:
            scene_data = json.load(f)
            
        self.scene_desc = f"【所处时代】{scene_data.get('era')}\n【地点】{scene_data.get('location')}\n【环境】{scene_data.get('scene_desc')}"

        # 检查是否存在“进度存档”
        if os.path.exists(state_file):
            print(f"🔄 发现休眠存档，正在唤醒 [{era_folder_name}] 的世界进度...")
            with open(state_file, 'r', encoding='utf-8') as f:
                state_data = json.load(f)
            # 继承存档进度
            self.current_task = state_data.get('current_task', scene_data.get('current_task', ''))
            self.shared_workspace = state_data.get('shared_workspace', scene_data.get('initial_workspace', ''))
            self.current_dialogue = state_data.get('current_dialogue', scene_data.get('initial_dialogue', ''))
            
            env_vars = state_data.get('env_variables', scene_data.get('env_variables', {}))
            self.world_env = WorldEnvironment(env_vars)
            self.world_env.time_passed = state_data.get('time_passed', 0)
        else:
            print(f"🌱 这是一个全新的世界，正在初始化...")
            self.current_task = scene_data.get('current_task', '')
            self.shared_workspace = scene_data.get('initial_workspace', '')
            self.current_dialogue = scene_data.get('initial_dialogue', '')
            env_vars = scene_data.get('env_variables', {})
            self.world_env = WorldEnvironment(env_vars)

        # 加载人物剧本
        self.agents = []
        for file_name in os.listdir(era_path):
            if file_name.endswith('.json') and file_name not in ['scene.json', 'state.json']:
                agent_file = os.path.join(era_path, file_name)
                with open(agent_file, 'r', encoding='utf-8') as f:
                    try:
                        agent_data = json.load(f)
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
                    
        # 挂载 RAG 引擎
        try:
            knowledge_engine = KnowledgeRetriever(era_folder_name)
            for agent in self.agents:
                agent.mount_knowledge(knowledge_engine)
        except Exception as e:
            print(f"ℹ️ 当前场景暂无离线 RAG 知识库，将全凭大模型基础能力演化。")

        print(f"✅ 成功加载时空节点: [{scene_data.get('era')}] - 共唤醒了 {len(self.agents)} 位数字生命")
        return self.agents

    # =================================================================
    # 核心升级 3：世界休眠 (Hibernate) - 保存动态进度
    # =================================================================
    def save_state(self, session_id):
        """保存当前世界的演化进度存档"""
        target_dir = os.path.join(self.scenarios_base_dir, session_id)
        if not os.path.exists(target_dir):
            return
            
        state_data = {
            "current_task": self.current_task,
            "shared_workspace": self.shared_workspace,
            # 防止上下文无限膨胀，保留最近的 2000 个字符的对话作为短期回溯
            "current_dialogue": self.current_dialogue[-2000:], 
            "env_variables": self.world_env.state if self.world_env else {},
            "time_passed": self.world_env.time_passed if self.world_env else 0
        }
        
        state_file = os.path.join(target_dir, "state.json")
        with open(state_file, 'w', encoding='utf-8') as f:
            json.dump(state_data, f, ensure_ascii=False, indent=4)
        print(f"💾 世界 [{session_id}] 状态已固化至存档！")