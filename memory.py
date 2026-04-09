# memory.py
import json
import os
import time
import chromadb
from config import DATA_DIR, API_KEY, BASE_URL, MODEL_NAME
from openai import OpenAI

# 初始化大模型客户端，用于“潜意识反思”
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

class SocialMemory:
    def __init__(self, agent_name):
        self.agent_name = agent_name
        self.file_path = os.path.join(DATA_DIR, f"memory_{agent_name}.json")
        self.data = self._load()
        
        # ====== 挂载双轨向量记忆体 ======
        db_path = os.path.join(DATA_DIR, "chroma_db_memory")
        self.chroma_client = chromadb.PersistentClient(path=db_path)
        
        # 1. 短期情境记忆 (流水账)
        self.episodic_memory = self.chroma_client.get_or_create_collection(name=f"episodic_{self.agent_name}")
        
        # 2. 长期语义记忆 (折叠升华后的核心认知)
        self.semantic_memory = self.chroma_client.get_or_create_collection(name=f"semantic_{self.agent_name}")
        
        # 反思触发器配置
        self.memory_counter = 0 
        self.REFLECTION_THRESHOLD = 5 # 每积累 5 条短期记忆，触发一次深夜反思

    def _load(self):
        if os.path.exists(self.file_path):
            with open(self.file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {"relationships": {}}

    def save(self):
        with open(self.file_path, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, ensure_ascii=False, indent=4)

    def update_relationship(self, target_name, affinity_change, trust_change):
        rel = self.data["relationships"].get(target_name, {"affinity": 50, "trust": 50})
        # 限制在 0-100 之间
        rel["affinity"] = max(0, min(100, rel["affinity"] + affinity_change))
        rel["trust"] = max(0, min(100, rel["trust"] + trust_change))
        self.data["relationships"][target_name] = rel

    def add_episodic_memory(self, env_state, action, dialogue):
        """记录回合结束后的情境记忆，并触发记忆衰退与折叠机制"""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        memory_text = f"【环境】: {env_state}\n【我做出的动作】: {action}\n【我说的话】: {dialogue}"
        
        doc_id = f"mem_epi_{int(time.time() * 1000)}"
        self.episodic_memory.add(
            documents=[memory_text], 
            metadatas=[{"time": timestamp}], 
            ids=[doc_id]
        )
        
        # 记忆累积计数
        self.memory_counter += 1
        if self.memory_counter >= self.REFLECTION_THRESHOLD:
            self.consolidate_memories()
            self.memory_counter = 0

    def consolidate_memories(self):
        """核心升级：记忆折叠与衰退 (Memory Consolidation)"""
        if self.episodic_memory.count() == 0:
            return
            
        print(f"\n🧠 [{self.agent_name}] 的潜意识正在进行记忆折叠与反思...")
        
        # 提取所有未被折叠的短期记忆
        all_episodes = self.episodic_memory.get()
        docs = all_episodes.get("documents", [])
        ids = all_episodes.get("ids", [])
        
        if not docs:
            return
            
        episodes_text = "\n\n".join(docs)
        
        # 让大模型作为角色的“潜意识”进行提炼
        prompt = f"""
        你现在是角色【{self.agent_name}】的潜意识。以下是你最近经历的一系列流水账事件：
        {episodes_text}
        
        请你进行深度的“反思与记忆折叠”：
        1. 提取出最重要的 1-2 条长效认知（例如：发现了某人的秘密、对当前局势的深刻感悟、掌握的关键信息）。
        2. 彻底抛弃那些无关紧要的日常动作细节（如吃饭、走路、简单的附和）。
        直接输出总结出的核心认知（作为一段完整的话），不要有任何前缀或多余解释。
        """
        
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )
            insight = response.choices[0].message.content.strip()
            
            if insight:
                print(f"✨ [{self.agent_name}] 产生了新的长效认知: {insight}")
                
                # 1. 存入长期语义记忆库
                sem_id = f"mem_sem_{int(time.time() * 1000)}"
                self.semantic_memory.add(
                    documents=[insight],
                    metadatas=[{"time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}],
                    ids=[sem_id]
                )
                
                # 2. 遗忘机制：清理已经被总结过的流水账，防止向量库爆炸
                self.episodic_memory.delete(ids=ids)
                
        except Exception as e:
            print(f"⚠️ [{self.agent_name}] 潜意识反思失败: {e}")

    def retrieve_episodic_memory(self, current_context, top_k=2):
        """升级版唤醒机制：混合检索长期认知与短期事件"""
        past_memories = []
        
        # 1. 唤醒长期的深层认知 (Semantic Memory)
        if self.semantic_memory.count() > 0:
            sem_results = self.semantic_memory.query(
                query_texts=[current_context], 
                n_results=1 # 取最深刻的一条认知
            )
            sem_docs = sem_results.get("documents", [[]])[0]
            if sem_docs:
                past_memories.append("【深层长期认知】: " + sem_docs[0])
                
        # 2. 唤醒近期的情境碎片 (Episodic Memory)
        if self.episodic_memory.count() > 0:
            # 动态调整：如果情境记忆不够，就取全部
            k = min(top_k, self.episodic_memory.count())
            epi_results = self.episodic_memory.query(
                query_texts=[current_context], 
                n_results=k
            )
            epi_docs = epi_results.get("documents", [[]])[0]
            for doc in epi_docs:
                past_memories.append("【近期发生的事】: " + doc)
                
        if not past_memories:
            return "（脑海中暂无相关往事）"
            
        return "\n\n".join(past_memories)