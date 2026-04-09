# rag_engine.py
import os
import json
import re
import jieba
import requests
import chromadb
from chromadb.utils import embedding_functions
from chromadb.utils.data_loaders import ImageLoader
import networkx as nx
from rank_bm25 import BM25Okapi
from openai import OpenAI
from config import DATA_DIR, API_KEY, BASE_URL, MODEL_NAME

# 初始化统一的 LLM 客户端
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

class KnowledgeRetriever:
    def __init__(self, era_name):
        self.era_name = era_name
        self.kb_path = os.path.join(DATA_DIR, "knowledge", era_name)
        self.raw_path = os.path.join(DATA_DIR, "raw_documents", era_name)
        
        db_path = os.path.join(DATA_DIR, "chroma_db")
        self.chroma_client = chromadb.PersistentClient(path=db_path)
        
        # ====== 双轨嵌入器 ======
        # 1. 文本档案轨：使用 BGE-M3
        self.text_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="BAAI/bge-m3")
        self.text_collection = self.chroma_client.get_or_create_collection(
            name=f"era_{self.era_name}_text", 
            embedding_function=self.text_ef
        )
        
        # 2. 视觉文物轨：使用 CLIP
        self.vision_ef = embedding_functions.OpenCLIPEmbeddingFunction()
        self.image_loader = ImageLoader()
        self.image_collection = self.chroma_client.get_or_create_collection(
            name=f"era_{self.era_name}_vision", 
            embedding_function=self.vision_ef,
            data_loader=self.image_loader
        )
        
        self.graph = nx.Graph()
        self._load_graph_network()
        
        # 检查是否需要重建索引
        if self.text_collection.count() == 0 or self.image_collection.count() == 0:
            self._build_multimodal_index()

        self.bm25 = None
        self.bm25_docs = [] 
        self.bm25_metadatas = []
        self._init_bm25()

    def _init_bm25(self):
        print("🧮 正在构建 BM25 精确匹配倒排索引...")
        if self.text_collection.count() > 0:
            all_data = self.text_collection.get(include=["documents", "metadatas"])
            docs = all_data.get("documents", [])
            metas = all_data.get("metadatas", [])
            
            tokenized_corpus = []
            for i, doc in enumerate(docs):
                if doc:
                    self.bm25_docs.append(doc)
                    self.bm25_metadatas.append(metas[i] if metas else {})
                    tokenized_corpus.append(jieba.lcut(doc))
            
            if tokenized_corpus:
                self.bm25 = BM25Okapi(tokenized_corpus)
                print(f"✅ BM25 索引就绪，共装载 {len(tokenized_corpus)} 个文本知识块。")

    def _load_graph_network(self):
        graph_path = os.path.join(self.kb_path, "graph_network.json")
        if os.path.exists(graph_path):
            with open(graph_path, 'r', encoding='utf-8') as f:
                graph_data = json.load(f)
                for triplet in graph_data.get("relationships", []):
                    self.graph.add_edge(
                        triplet["source"], 
                        triplet["target"], 
                        relation=triplet["relation"]
                    )
            print(f"🕸️ GraphRAG 引擎就绪：加载了 {len(self.graph.nodes)} 个实体节点。")

    # ========================================================
    # [核心升级 1] 绝对年份提取器 (用于时间围栏)
    # ========================================================
    def _extract_year_with_llm(self, text_chunk):
        """提取文本对应的绝对年份（如公元1080年返回1080）"""
        prompt = f"""
        你是一个严谨的历史纪年专家。请从以下历史文本中提取事件发生的绝对年份（公元纪年）。
        例如，如果文本提到了“北宋元丰三年”，请换算返回 1080。
        如果不确定或没有明显时间信息，请返回 0。
        必须只返回一个纯数字！
        文本：{text_chunk[:400]}
        """
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                timeout=10
            )
            year_str = response.choices[0].message.content.strip()
            # 过滤非数字字符
            year_match = re.search(r'\d+', year_str)
            return int(year_match.group(0)) if year_match else 0
        except Exception:
            return 0

    def _build_multimodal_index(self):
        print(f"📚 正在为 [{self.era_name}] 构建双轨向量索引 (BGE-M3 + CLIP)...")
        
        # 1. 灌入纯文本数据
        documents, metadatas, ids = [], [], []
        doc_idx = 0
        if os.path.exists(self.kb_path):
            for file_name in os.listdir(self.kb_path):
                if file_name.endswith('.txt') and file_name != "graph_network.json":
                    file_path = os.path.join(self.kb_path, file_name)
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        
                    chunks = content.split('\n\n')
                    for chunk in chunks:
                        chunk = chunk.strip()
                        if not chunk: continue
                        
                        metadata = {"source": file_name, "type": "text"}
                        
                        # 识别视觉线索
                        match = re.search(r'【视觉文献来源：(.*?)】', chunk)
                        if match:
                            metadata["image_target"] = match.group(1).strip()
                            
                        # [核心升级] 注入绝对时间戳
                        extracted_year = self._extract_year_with_llm(chunk)
                        if extracted_year > 0:
                            metadata["year"] = extracted_year
                            
                        documents.append(chunk)
                        metadatas.append(metadata)
                        ids.append(f"doc_{self.era_name}_{doc_idx}")
                        doc_idx += 1
                        
            if documents:
                self.text_collection.add(documents=documents, metadatas=metadatas, ids=ids)

        # 2. 灌入视觉图片数据
        image_uris, image_metadatas, image_ids = [], [], []
        img_idx = 0
        if os.path.exists(self.raw_path):
            valid_exts = {'.jpg', '.jpeg', '.png', '.webp'}
            for file_name in os.listdir(self.raw_path):
                if os.path.splitext(file_name)[1].lower() in valid_exts:
                    img_path = os.path.join(self.raw_path, file_name)
                    image_uris.append(img_path)
                    image_metadatas.append({"source": file_name, "type": "image"})
                    image_ids.append(f"img_{self.era_name}_{img_idx}")
                    img_idx += 1
                    
            if image_uris:
                self.image_collection.add(uris=image_uris, metadatas=image_metadatas, ids=image_ids)
                
        print(f"✅ 双轨索引构建完成！BGE-M3入库 {len(documents)} 个文本块，CLIP入库 {len(image_uris)} 张原生图片。")

    def _extract_and_normalize_entities(self, query):
        if not self.graph.nodes:
            return []

        valid_nodes = list(self.graph.nodes)
        prompt = f"""
        你是一个知识图谱实体识别与消歧专家。
        查询语句："{query}"
        数据库标准实体名单：{valid_nodes[:100]} 
        
        请提取查询中的核心实体，并映射为名单中的标准实体。
        必须只返回纯 JSON 数组格式，不要多余文字。例如：["苏轼", "黄州"]。若无匹配则返回 []。
        """
        
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                timeout=10
            )
            raw = response.choices[0].message.content.strip()
            
            md_json = "`" * 3 + "json"
            md_empty = "`" * 3
            if raw.startswith(md_json): raw = raw[7:-3].strip()
            elif raw.startswith(md_empty): raw = raw[3:-3].strip()
            
            extracted_entities = json.loads(raw)
            normalized_entities = [e for e in extracted_entities if e in self.graph.nodes]
            
            if normalized_entities:
                print(f"🕸️ [图谱雷达] 成功消歧为标准节点: {normalized_entities}")
            return normalized_entities
            
        except Exception as e:
            print(f"⚠️ 实体消歧降级: {e}")
            return [node for node in self.graph.nodes if node in query]

    # ========================================================
    # [核心升级 2] CRAG 自我纠错与相关性评估
    # ========================================================
    def _evaluate_relevance(self, query, retrieved_text):
        """评估检索到的知识是否能回答当前的 Query"""
        if not retrieved_text.strip():
            return False
            
        prompt = f"""
        你是一个严格的史料审核员。
        用户/数字生命的当前语境与意图是："{query}"
        系统从资料库中检索到的参考知识是：
        {retrieved_text}
        
        请判断这些参考知识是否对当前的情境有实质性的帮助或逻辑关联？
        如果有帮助，请回复 "YES"；如果完全无关、纯属牵强附会，请回复 "NO"。
        只允许回复 "YES" 或 "NO"。
        """
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                timeout=10
            )
            return "YES" in response.choices[0].message.content.upper()
        except Exception:
            return True # 降级：如果判断引擎卡顿，默认放行

    # ========================================================
    # [核心升级 3] 带时间过滤的统一 retrieve 方法
    # ========================================================
    def retrieve(self, query, current_year=None, top_k=3):
        result_text = ""
        dense_docs = []
        text_metas = []
        
        # 1. 构造时间围栏
        where_filter = None
        if current_year:
            # 过滤掉大于 current_year 的知识（防剧透/时空穿梭）
            where_filter = {"year": {"$lte": current_year}}
            print(f"⏳ [时间围栏启动] 锁定知识库时间至公元 {current_year} 年...")

        # 2. BGE-M3 纯文本深度检索 (带时间过滤)
        if self.text_collection.count() > 0:
            try:
                vector_results = self.text_collection.query(
                    query_texts=[query],
                    n_results=top_k * 2,
                    where=where_filter,
                    include=["documents", "metadatas"]
                )
                dense_docs = vector_results.get('documents', [[]])[0]
                text_metas = vector_results.get('metadatas', [[]])[0]
            except Exception as e:
                print(f"⚠️ 稠密检索遇到问题 (可能是空过滤结果): {e}")
            
        # 3. BM25 稀疏检索融合 (增加时间顾虑)
        tokenized_query = jieba.lcut(query)
        sparse_scores = self.bm25.get_scores(tokenized_query) if self.bm25 else []
        
        # 获取 Top K 稀疏结果，并手动做时间过滤
        sparse_candidates = []
        for i in sorted(range(len(sparse_scores)), key=lambda x: sparse_scores[x], reverse=True):
            if len(sparse_candidates) >= top_k * 2:
                break
            meta = self.bm25_metadatas[i] if i < len(self.bm25_metadatas) else {}
            # 检查这篇文档的年份
            doc_year = meta.get("year", 0)
            if not current_year or doc_year == 0 or doc_year <= current_year:
                sparse_candidates.append(i)
        
        # RRF (倒数秩融合)
        rrf_k = 60
        rrf_scores = {}
        
        for rank, doc in enumerate(dense_docs):
            if doc not in rrf_scores: 
                rrf_scores[doc] = {"score": 0, "meta": text_metas[rank] if text_metas else None}
            rrf_scores[doc]["score"] += 1.0 / (rrf_k + rank + 1)
            
        for rank, doc_idx in enumerate(sparse_candidates):
            if doc_idx < len(self.bm25_docs):
                doc = self.bm25_docs[doc_idx]
                if doc not in rrf_scores: 
                    rrf_scores[doc] = {"score": 0, "meta": self.bm25_metadatas[doc_idx]}
                rrf_scores[doc]["score"] += 1.0 / (rrf_k + rank + 1)
            
        fused_text_results = sorted(rrf_scores.items(), key=lambda x: x[1]["score"], reverse=True)[:top_k]

        for doc, data in fused_text_results:
            meta = data["meta"]
            doc_text = doc
            if meta and meta.get("type") == "text":
                if "image_target" in meta:
                    doc_text += f"\n(💡视觉线索：对应本地图片 {meta['image_target']})"
            result_text += f"【史料/档案 (混合检索)】\n{doc_text}\n\n"

        # 4. CLIP 跨模态图片检索
        if self.image_collection.count() > 0:
            vision_results = self.image_collection.query(
                query_texts=[query],
                n_results=1,
                include=["uris"]
            )
            uris = vision_results.get('uris', [[]])[0]
            if uris:
                img_name = os.path.basename(uris[0])
                result_text += f"【视觉文物 (CLIP 直觉)】\n(💡视觉线索：意境古画 {img_name})\n\n"

        # 5. GraphRAG 图谱羁绊检索
        relevant_triplets = []
        detected_entities = self._extract_and_normalize_entities(query)
        
        for entity in detected_entities:
            subgraph = nx.ego_graph(self.graph, entity, radius=2)
            for u, v, data in subgraph.edges(data=True):
                relevant_triplets.append(f"{u} --[{data['relation']}]--> {v}")
        
        unique_triplets = list(set(relevant_triplets))
        if unique_triplets:
            result_text += "【全局羁绊(GraphRAG - 2跳推理)】\n" + "\n".join(unique_triplets[:8])

        # ========================================================
        # [核心升级] 最终的 CRAG 拦截审查
        # ========================================================
        if not result_text.strip():
            return "未能检索到相关时空记忆。"
            
        is_relevant = self._evaluate_relevance(query, result_text)
        
        if not is_relevant:
            print(f"🛑 [CRAG拦截] 检索到的知识与当前情境无关，已强行阻断 (防胡编乱造)。")
            return "未能检索到相关时空记忆。"
            
        return result_text.strip()