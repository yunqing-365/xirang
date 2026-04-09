# memory.py
import json
import os
import time
import chromadb
from config import DATA_DIR

class SocialMemory:
    def __init__(self, agent_name):
        self.agent_name = agent_name
        self.file_path = os.path.join(DATA_DIR, f"memory_{agent_name}.json")
        self.data = self._load()
        
        # ====== 新增：挂载专属向量记忆体 ======
        db_path = os.path.join(DATA_DIR, "chroma_db_memory")
        self.chroma_client = chromadb.PersistentClient(path=db_path)
        # 为每个智能体创建一个独立的 Collection
        self.episodic_memory = self.chroma_client.get_or_create_collection(name=f"episodic_{self.agent_name}")
        # =====================================

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

    # ====== 新增：长期记忆读写接口 ======
    def add_episodic_memory(self, env_state, action, dialogue):
        """记录回合结束后的情境记忆"""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        memory_text = f"【环境】: {env_state}\n【我做出的动作】: {action}\n【我说的话】: {dialogue}"
        
        doc_id = f"mem_{int(time.time() * 1000)}"
        self.episodic_memory.add(
            documents=[memory_text], 
            metadatas=[{"time": timestamp}], 
            ids=[doc_id]
        )

    def retrieve_episodic_memory(self, current_context, top_k=2):
        """根据当前对话上下文，唤醒最相关的过去记忆"""
        if self.episodic_memory.count() == 0:
            return "（脑海中暂无相关往事）"
            
        results = self.episodic_memory.query(
            query_texts=[current_context], 
            n_results=min(top_k, self.episodic_memory.count())
        )
        
        memories = results.get("documents", [[]])[0]
        if not memories:
            return "（脑海中暂无相关往事）"
            
        return "\n\n".join(memories)
    # =====================================