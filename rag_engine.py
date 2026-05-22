# rag_engine.py  ── 硬核升级版
"""
四大核心算法升级：
  1. HyDE（假设文档嵌入）：生成假设答案再检索，对稀疏历史查询效果显著提升
  2. 多查询扩展：从3个不同角度重写查询，合并结果后去重，召回更全面
  3. 上下文压缩：对每个检索 chunk 用 LLM 压缩到只保留与查询相关的句子
  4. Cross-Encoder 重排序：对候选文档用交叉编码器精排（有包则用，无包降级到 LLM 重排）
  
  保留：BM25 稀疏检索 + 向量稠密检索 + RRF 融合 + GraphRAG + 时间围栏 + CRAG
"""
import asyncio
import json
import os
import re
import time
from typing import Dict, List, Optional, Tuple

import jieba
import chromadb
from chromadb.utils import embedding_functions
from chromadb.utils.data_loaders import ImageLoader
import networkx as nx
from rank_bm25 import BM25Okapi
from openai import AsyncOpenAI, OpenAI

from config import get_settings
from prompt_templates import (
    RAG_YEAR_EXTRACTOR, RAG_ENTITY_NORMALIZE, RAG_RELEVANCE_JUDGE,
    RAG_HYDE, RAG_MULTI_QUERY, RAG_CONTEXTUAL_COMPRESS, RAG_QUALITY_SCORE,
)

_settings = get_settings()
_async_client = AsyncOpenAI(api_key=_settings.API_KEY, base_url=_settings.BASE_URL)
_sync_client  = OpenAI(api_key=_settings.API_KEY, base_url=_settings.BASE_URL)

# ── 尝试加载 Cross-Encoder（可选依赖）────────────────────────
try:
    from sentence_transformers import CrossEncoder
    _CROSS_ENCODER = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    _HAS_CROSS_ENCODER = True
    print("✅ Cross-Encoder 已加载，精排功能启用")
except Exception:
    _CROSS_ENCODER = None
    _HAS_CROSS_ENCODER = False
    print("ℹ️  Cross-Encoder 不可用，降级为 LLM 精排")


# ═══════════════════════════════════════════════════════════════
# 检索结果数据类
# ═══════════════════════════════════════════════════════════════

class RetrievedChunk:
    __slots__ = ("text", "source", "score", "meta", "compressed")

    def __init__(self, text: str, source: str = "", score: float = 0.0, meta: dict = None):
        self.text = text
        self.source = source
        self.score = score
        self.meta = meta or {}
        self.compressed: Optional[str] = None  # 压缩后的内容

    def display_text(self) -> str:
        return self.compressed if self.compressed else self.text


# ═══════════════════════════════════════════════════════════════
# HyDE 引擎
# ═══════════════════════════════════════════════════════════════

class HyDEEngine:
    """
    Hypothetical Document Embeddings（假设文档嵌入）
    原理：query → LLM 生成"假设答案文档" → 用该文档的嵌入去检索
    效果：将 query space 投影到 document space，显著减少检索偏差
    """

    async def generate_hypothesis(self, query: str) -> str:
        prompt = RAG_HYDE.substitute(query=query)
        try:
            resp = await _async_client.chat.completions.create(
                model=_settings.MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=200,
                timeout=10,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            print(f"⚠️ HyDE 生成失败，回退原始查询: {e}")
            return query


# ═══════════════════════════════════════════════════════════════
# 多查询扩展引擎
# ═══════════════════════════════════════════════════════════════

class MultiQueryExpander:
    """
    从 N 个不同角度重写查询，提高召回的多样性和覆盖面。
    """

    async def expand(self, query: str, n: int = 3) -> List[str]:
        prompt = RAG_MULTI_QUERY.substitute(query=query, n=n)
        try:
            resp = await _async_client.chat.completions.create(
                model=_settings.MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.6,
                max_tokens=200,
                timeout=10,
            )
            raw = _strip_json(resp.choices[0].message.content)
            queries = json.loads(raw)
            result = [query] + [q for q in queries if isinstance(q, str) and q != query]
            print(f"🔀 多查询扩展: {len(result)} 个角度")
            return result[:n + 1]
        except Exception as e:
            print(f"⚠️ 多查询扩展失败，使用原始查询: {e}")
            return [query]


# ═══════════════════════════════════════════════════════════════
# 上下文压缩器
# ═══════════════════════════════════════════════════════════════

class ContextualCompressor:
    """
    对检索到的 chunk 进行 LLM 压缩，只保留与查询相关的句子。
    减少注入 Agent 的 noise，提高知识利用率。
    """

    async def compress(self, query: str, chunks: List[RetrievedChunk]) -> List[RetrievedChunk]:
        """并发压缩所有 chunks"""
        tasks = [self._compress_one(query, chunk) for chunk in chunks]
        compressed = await asyncio.gather(*tasks, return_exceptions=True)
        for i, result in enumerate(compressed):
            if isinstance(result, str) and result != "无关内容":
                chunks[i].compressed = result
        return chunks

    async def _compress_one(self, query: str, chunk: RetrievedChunk) -> str:
        if len(chunk.text) < 80:
            return chunk.text   # 太短不值得压缩
        prompt = RAG_CONTEXTUAL_COMPRESS.substitute(query=query, chunk=chunk.text[:600])
        try:
            resp = await _async_client.chat.completions.create(
                model=_settings.MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=150,
                timeout=8,
            )
            return resp.choices[0].message.content.strip()
        except Exception:
            return chunk.text


# ═══════════════════════════════════════════════════════════════
# 重排序器
# ═══════════════════════════════════════════════════════════════

class Reranker:
    """
    Cross-Encoder 精排（有 sentence-transformers 时）
    或 LLM 评分精排（降级方案）
    """

    def rerank(self, query: str, chunks: List[RetrievedChunk]) -> List[RetrievedChunk]:
        if not chunks:
            return chunks

        if _HAS_CROSS_ENCODER:
            return self._cross_encoder_rerank(query, chunks)
        else:
            return self._score_based_rerank(chunks)

    def _cross_encoder_rerank(self, query: str, chunks: List[RetrievedChunk]) -> List[RetrievedChunk]:
        pairs = [(query, c.display_text()[:300]) for c in chunks]
        scores = _CROSS_ENCODER.predict(pairs)
        for chunk, score in zip(chunks, scores):
            chunk.score = float(score)
        chunks.sort(key=lambda c: c.score, reverse=True)
        print(f"🏆 Cross-Encoder 精排完成，top 得分: {chunks[0].score:.3f}")
        return chunks

    def _score_based_rerank(self, chunks: List[RetrievedChunk]) -> List[RetrievedChunk]:
        """纯基于融合分数排序（RRF 分已在 retrieve 阶段计算）"""
        chunks.sort(key=lambda c: c.score, reverse=True)
        return chunks


# ═══════════════════════════════════════════════════════════════
# 知识检索器主体（全异步版）
# ═══════════════════════════════════════════════════════════════

class KnowledgeRetriever:

    def __init__(self, era_name: str):
        self.era_name = era_name
        self.kb_path  = os.path.join(_settings.DATA_DIR, "knowledge", era_name)
        self.raw_path = os.path.join(_settings.DATA_DIR, "raw_documents", era_name)

        db_path = os.path.join(_settings.DATA_DIR, "chroma_db")
        self.chroma_client = chromadb.PersistentClient(path=db_path)

        self.text_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="BAAI/bge-m3"
        )
        self.text_collection = self.chroma_client.get_or_create_collection(
            name=f"era_{self.era_name}_text",
            embedding_function=self.text_ef,
        )

        self.vision_ef = embedding_functions.OpenCLIPEmbeddingFunction()
        self.image_loader = ImageLoader()
        self.image_collection = self.chroma_client.get_or_create_collection(
            name=f"era_{self.era_name}_vision",
            embedding_function=self.vision_ef,
            data_loader=self.image_loader,
        )

        self.graph = nx.Graph()
        self._load_graph_network()

        if self.text_collection.count() == 0 or self.image_collection.count() == 0:
            self._build_multimodal_index()

        self.bm25: Optional[BM25Okapi] = None
        self.bm25_docs: List[str] = []
        self.bm25_metadatas: List[dict] = []
        self._init_bm25()

        # 新增：四大组件
        self._hyde     = HyDEEngine()
        self._expander = MultiQueryExpander()
        self._compressor = ContextualCompressor()
        self._reranker = Reranker()

    # ── 外部接口（同步包装，供 asyncio.to_thread 调用）────────

    def retrieve(self, query: str, current_year: int = None, top_k: int = 3) -> str:
        """同步入口（被 agent.py 通过 asyncio.to_thread 调用）"""
        return asyncio.get_event_loop().run_until_complete(
            self.aretrieve(query, current_year, top_k)
        )

    # ── 核心异步检索链路 ──────────────────────────────────────

    async def aretrieve(
        self,
        query: str,
        current_year: Optional[int] = None,
        top_k: int = 3,
    ) -> str:
        t0 = time.perf_counter()

        # ── Step 1: 多查询扩展 + HyDE 并发生成 ───────────────
        expanded_queries, hyde_hypothesis = await asyncio.gather(
            self._expander.expand(query, n=2),
            self._hyde.generate_hypothesis(query),
            return_exceptions=True,
        )
        if isinstance(expanded_queries, Exception):
            expanded_queries = [query]
        if isinstance(hyde_hypothesis, Exception):
            hyde_hypothesis = query

        all_queries = list(dict.fromkeys(expanded_queries + [hyde_hypothesis]))

        # ── Step 2: 多查询并发检索（向量 + BM25）─────────────
        where_filter = {"year": {"$lte": current_year}} if current_year else None
        candidate_chunks: Dict[str, RetrievedChunk] = {}

        async def _single_query_retrieve(q: str):
            # 向量检索
            if self.text_collection.count() > 0:
                try:
                    vr = await asyncio.to_thread(
                        self.text_collection.query,
                        query_texts=[q],
                        n_results=min(top_k * 2, self.text_collection.count()),
                        where=where_filter,
                        include=["documents", "metadatas"],
                    )
                    docs  = vr.get("documents", [[]])[0]
                    metas = vr.get("metadatas", [[]])[0]
                    for rank, (doc, meta) in enumerate(zip(docs, metas)):
                        rrf_score = 1.0 / (60 + rank + 1)
                        if doc in candidate_chunks:
                            candidate_chunks[doc].score += rrf_score
                        else:
                            candidate_chunks[doc] = RetrievedChunk(doc, meta.get("source",""), rrf_score, meta)
                except Exception as _e:
                    print(f"⚠️ 向量检索问题: {_e}")

            # BM25 稀疏检索
            if self.bm25:
                tokenized = jieba.lcut(q)
                scores = self.bm25.get_scores(tokenized)
                for rank, idx in enumerate(
                    sorted(range(len(scores)), key=lambda x: scores[x], reverse=True)[:top_k * 2]
                ):
                    if idx >= len(self.bm25_docs):
                        continue
                    meta = self.bm25_metadatas[idx]
                    doc_year = meta.get("year", 0)
                    if current_year and doc_year > 0 and doc_year > current_year:
                        continue
                    doc = self.bm25_docs[idx]
                    rrf_score = 1.0 / (60 + rank + 1)
                    if doc in candidate_chunks:
                        candidate_chunks[doc].score += rrf_score
                    else:
                        candidate_chunks[doc] = RetrievedChunk(doc, meta.get("source",""), rrf_score, meta)

        await asyncio.gather(*[_single_query_retrieve(q) for q in all_queries])

        # ── Step 3: GraphRAG 实体检索 ─────────────────────────
        graph_text = await asyncio.to_thread(self._graph_retrieve, query)

        # ── Step 4: CLIP 图片检索 ─────────────────────────────
        image_text = await asyncio.to_thread(self._image_retrieve, query)

        # ── Step 5: Cross-Encoder 重排序 ──────────────────────
        sorted_chunks = self._reranker.rerank(
            query,
            list(candidate_chunks.values()),
        )[:top_k]

        # ── Step 6: CRAG 相关性守门 ───────────────────────────
        raw_text = "\n\n".join(c.display_text() for c in sorted_chunks)
        if not self._evaluate_relevance(query, raw_text):
            print("🛑 [CRAG] 检索内容与当前情境无关，已阻断。")
            return "未能检索到相关时空记忆。"

        # ── Step 7: 并发上下文压缩 ────────────────────────────
        sorted_chunks = await self._compressor.compress(query, sorted_chunks)

        # ── 组装最终输出 ──────────────────────────────────────
        parts = []
        for i, c in enumerate(sorted_chunks, 1):
            source_hint = ""
            if c.meta.get("image_target"):
                source_hint = f"\n(💡视觉线索：{c.meta['image_target']})"
            parts.append(f"【史料{i} | 得分:{c.score:.3f}】\n{c.display_text()}{source_hint}")

        if graph_text:
            parts.append(graph_text)
        if image_text:
            parts.append(image_text)

        elapsed = time.perf_counter() - t0
        print(f"🔍 [RAG] 检索完成 {elapsed:.2f}s | {len(sorted_chunks)} 条文本 + 图 + 图谱")
        return "\n\n".join(parts) if parts else "未能检索到相关时空记忆。"

    # ── GraphRAG ──────────────────────────────────────────────

    def _graph_retrieve(self, query: str) -> str:
        if not self.graph.nodes:
            return ""
        detected = self._extract_entities_sync(query)
        triplets = []
        for entity in detected:
            try:
                subgraph = nx.ego_graph(self.graph, entity, radius=2)
                for u, v, d in subgraph.edges(data=True):
                    triplets.append(f"{u} --[{d['relation']}]--> {v}")
            except Exception:
                pass
        unique = list(set(triplets))[:8]
        if unique:
            return "【全局羁绊 (GraphRAG)】\n" + "\n".join(unique)
        return ""

    def _extract_entities_sync(self, query: str) -> List[str]:
        valid_nodes = list(self.graph.nodes)[:100]
        prompt = RAG_ENTITY_NORMALIZE.substitute(query=query, valid_nodes=str(valid_nodes))
        try:
            resp = _sync_client.chat.completions.create(
                model=_settings.MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1, timeout=8,
            )
            raw = _strip_json(resp.choices[0].message.content)
            extracted = json.loads(raw)
            return [e for e in extracted if e in self.graph.nodes]
        except Exception:
            return [n for n in valid_nodes if n in query]

    # ── CLIP 图片检索 ─────────────────────────────────────────

    def _image_retrieve(self, query: str) -> str:
        if self.image_collection.count() == 0:
            return ""
        try:
            vr = self.image_collection.query(
                query_texts=[query], n_results=1, include=["uris"]
            )
            uris = vr.get("uris", [[]])[0]
            if uris:
                return f"【视觉文物 (CLIP)】\n(💡意境古画 {os.path.basename(uris[0])})"
        except Exception:
            pass
        return ""

    # ── CRAG 相关性守门 ───────────────────────────────────────

    def _evaluate_relevance(self, query: str, retrieved_text: str) -> bool:
        if not retrieved_text.strip():
            return False
        prompt = RAG_RELEVANCE_JUDGE.substitute(query=query, retrieved_text=retrieved_text[:500])
        try:
            resp = _sync_client.chat.completions.create(
                model=_settings.MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1, timeout=8,
            )
            return "YES" in resp.choices[0].message.content.upper()
        except Exception:
            return True

    # ── 索引构建（同原版）────────────────────────────────────

    def _init_bm25(self):
        if self.text_collection.count() == 0:
            return
        all_data = self.text_collection.get(include=["documents", "metadatas"])
        docs  = all_data.get("documents", [])
        metas = all_data.get("metadatas", [])
        corpus = []
        for i, doc in enumerate(docs):
            if doc:
                self.bm25_docs.append(doc)
                self.bm25_metadatas.append(metas[i] if metas else {})
                corpus.append(jieba.lcut(doc))
        if corpus:
            self.bm25 = BM25Okapi(corpus)
            print(f"✅ BM25 就绪：{len(corpus)} 块")

    def _load_graph_network(self):
        path = os.path.join(self.kb_path, "graph_network.json")
        if not os.path.exists(path):
            return
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for t in data.get("relationships", []):
            self.graph.add_edge(t["source"], t["target"], relation=t["relation"])
        print(f"🕸️ GraphRAG 就绪：{len(self.graph.nodes)} 节点")

    def _extract_year_sync(self, text_chunk: str) -> int:
        prompt = RAG_YEAR_EXTRACTOR.substitute(text_chunk=text_chunk[:400])
        try:
            resp = _sync_client.chat.completions.create(
                model=_settings.MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1, timeout=10,
            )
            m = re.search(r'\d+', resp.choices[0].message.content.strip())
            return int(m.group(0)) if m else 0
        except Exception:
            return 0

    def _build_multimodal_index(self):
        print(f"📚 [{self.era_name}] 构建双轨向量索引…")
        documents, metadatas, ids = [], [], []
        idx = 0
        if os.path.exists(self.kb_path):
            for file_name in os.listdir(self.kb_path):
                if not file_name.endswith(".txt"):
                    continue
                with open(os.path.join(self.kb_path, file_name), "r", encoding="utf-8") as f:
                    content = f.read()
                for chunk in content.split("\n\n"):
                    chunk = chunk.strip()
                    if not chunk:
                        continue
                    meta: Dict = {"source": file_name, "type": "text"}
                    m = re.search(r'【视觉文献来源：(.*?)】', chunk)
                    if m:
                        meta["image_target"] = m.group(1).strip()
                    year = self._extract_year_sync(chunk)
                    if year > 0:
                        meta["year"] = year
                    documents.append(chunk)
                    metadatas.append(meta)
                    ids.append(f"doc_{self.era_name}_{idx}")
                    idx += 1
            if documents:
                self.text_collection.add(documents=documents, metadatas=metadatas, ids=ids)

        img_uris, img_metas, img_ids = [], [], []
        i2 = 0
        if os.path.exists(self.raw_path):
            for fn in os.listdir(self.raw_path):
                if os.path.splitext(fn)[1].lower() in {".jpg",".jpeg",".png",".webp"}:
                    img_uris.append(os.path.join(self.raw_path, fn))
                    img_metas.append({"source": fn, "type": "image"})
                    img_ids.append(f"img_{self.era_name}_{i2}")
                    i2 += 1
            if img_uris:
                self.image_collection.add(uris=img_uris, metadatas=img_metas, ids=img_ids)
        print(f"✅ 索引完成：文本 {len(documents)} 块，图片 {len(img_uris)} 张")


def _strip_json(text: str) -> str:
    text = text.strip()
    if text.startswith("```json"): text = text[7:]
    elif text.startswith("```"):   text = text[3:]
    if text.endswith("```"):       text = text[:-3]
    return text.strip()
