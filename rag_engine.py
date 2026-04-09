import os
import json
import chromadb
from chromadb.utils import embedding_functions
from config import DATA_DIR

class KnowledgeRetriever:
    def __init__(self, era_name):
        self.era_name = era_name
        self.kb_path = os.path.join(DATA_DIR, "knowledge", era_name)
        
        # 1. 初始化 ChromaDB (代码保持不变)
        db_path = os.path.join(DATA_DIR, "chroma_db")
        self.chroma_client = chromadb.PersistentClient(path=db_path)
        self.ef = embedding_functions.DefaultEmbeddingFunction()
        self.collection = self.chroma_client.get_or_create_collection(
            name=f"era_{self.era_name}", 
            embedding_function=self.ef
        )
        self._build_vector_index()
        
        # === 【新增：加载全局图谱网络】 ===
        self.graph_network = []
        graph_path = os.path.join(self.kb_path, "graph_network.json")
        if os.path.exists(graph_path):
            with open(graph_path, 'r', encoding='utf-8') as f:
                graph_data = json.load(f)
                self.graph_network = graph_data.get("relationships", [])

    # ... _build_vector_index() 方法保持不变 ...

    def retrieve(self, query, top_k=2):
        """
        进行真正的混合检索：语义向量搜索 (Dense) + 图谱关系网检索 (Graph)
        """
        result_text = ""

        # 1. 向量数据库检索 (代码保持不变)
        if self.collection.count() > 0:
            results = self.collection.query(
                query_texts=[query],
                n_results=top_k
            )
            retrieved_docs = results['documents'][0]
            if retrieved_docs:
                result_text += "【详细史料】\n" + "\n".join(retrieved_docs) + "\n\n"

        # === 【新增：知识图谱过滤】 ===
        # 遍历图谱，寻找与当前查询（query）相关的三元组关系
        relevant_triplets = []
        for triplet in self.graph_network:
            if triplet["source"] in query or triplet["target"] in query:
                relevant_triplets.append(f"{triplet['source']} --[{triplet['relation']}]--> {triplet['target']}")
        
        if relevant_triplets:
            result_text += "【全局历史羁绊(GraphRAG)】\n" + "\n".join(relevant_triplets[:5]) # 拿最相关的前5条

        if not result_text:
            return "未能检索到相关时空记忆。"
            
        return result_text