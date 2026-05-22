# memory.py  ── 硬核升级版
"""
三大核心升级：
  1. 重要性评分：每条记忆入库时，LLM 打分 1-10，高分记忆优先保留/检索
  2. 时序衰减函数：relevance = importance × exp(-λ × Δt) + query_similarity
     - 重要度高的记忆衰减慢（半衰期长）
     - 日常琐事快速淡出，关键事件久久难忘
  3. 情绪标注：每条记忆附带主导情绪标签，支持"情绪相似检索"
     （悲凉的当下容易唤起悲凉的往事）

  保留：双轨向量记忆（情境记忆/语义记忆）+ 记忆折叠
"""
import asyncio
import json
import math
import os
import time
from dataclasses import dataclass, field
from typing import List, Optional

import chromadb
from openai import AsyncOpenAI, OpenAI

from config import get_settings
from prompt_templates import (
    MEMORY_CONSOLIDATION, MEMORY_IMPORTANCE_SCORE, MEMORY_EMOTIONAL_TAG
)

_settings = get_settings()
_async_client = AsyncOpenAI(api_key=_settings.API_KEY, base_url=_settings.BASE_URL)
_sync_client  = OpenAI(api_key=_settings.API_KEY, base_url=_settings.BASE_URL)

# 衰减速率（每回合）：重要度越低衰减越快
_DECAY_RATES = {
    range(1,  4): 0.20,   # 低重要（1-3分）：快速遗忘
    range(4,  7): 0.08,   # 中重要（4-6分）：正常遗忘
    range(7, 11): 0.02,   # 高重要（7-10分）：持久记忆
}


def _decay_rate(importance: int) -> float:
    for r, rate in _DECAY_RATES.items():
        if importance in r:
            return rate
    return 0.08


def _memory_score(importance: int, time_delta: int, query_sim: float = 0.5) -> float:
    """
    综合记忆评分公式：
      score = importance × exp(-λ × Δt) + query_sim × 0.3
    """
    lam = _decay_rate(importance)
    decay = importance * math.exp(-lam * time_delta)
    return decay + query_sim * 0.3


# ═══════════════════════════════════════════════════════════════
# 重要性评分器（异步）
# ═══════════════════════════════════════════════════════════════

class ImportanceScorer:
    async def score(self, agent_name: str, env_state: str, action: str, dialogue: str) -> int:
        prompt = MEMORY_IMPORTANCE_SCORE.substitute(
            agent_name=agent_name,
            env_state=env_state[:200],
            action=action,
            dialogue=dialogue,
        )
        try:
            resp = await _async_client.chat.completions.create(
                model=_settings.MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=5,
                timeout=6,
            )
            raw = resp.choices[0].message.content.strip()
            score = int(''.join(filter(str.isdigit, raw))[:1] or '5')
            return max(1, min(10, score))
        except Exception:
            return 5   # 默认中等重要


# ═══════════════════════════════════════════════════════════════
# 情绪标注器（异步）
# ═══════════════════════════════════════════════════════════════

class EmotionalTagger:
    VALID_TAGS = {"喜悦","悲凉","愤怒","恐惧","豁达","孤独","渴望","平静","紧张","感恩"}

    async def tag(self, text: str) -> str:
        prompt = MEMORY_EMOTIONAL_TAG.substitute(text=text[:300])
        try:
            resp = await _async_client.chat.completions.create(
                model=_settings.MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=10,
                timeout=5,
            )
            tag = resp.choices[0].message.content.strip()
            return tag if tag in self.VALID_TAGS else "平静"
        except Exception:
            return "平静"


# ═══════════════════════════════════════════════════════════════
# 社会记忆主体
# ═══════════════════════════════════════════════════════════════

class SocialMemory:
    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        self.file_path = os.path.join(_settings.DATA_DIR, f"memory_{agent_name}.json")
        self.data = self._load()

        db_path = os.path.join(_settings.DATA_DIR, "chroma_db_memory")
        self.chroma_client = chromadb.PersistentClient(path=db_path)

        self.episodic_memory = self.chroma_client.get_or_create_collection(
            name=f"episodic_{agent_name}"
        )
        self.semantic_memory = self.chroma_client.get_or_create_collection(
            name=f"semantic_{agent_name}"
        )

        self.memory_counter = 0
        self.current_round = 0
        self.REFLECTION_THRESHOLD = _settings.MEMORY_REFLECTION_THRESHOLD

        self._importance_scorer = ImportanceScorer()
        self._emotion_tagger = EmotionalTagger()

    # ── 持久化 ────────────────────────────────────────────────

    def _load(self) -> dict:
        if os.path.exists(self.file_path):
            with open(self.file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {"relationships": {}}

    def save(self) -> None:
        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=4)

    # ── 关系图谱 ──────────────────────────────────────────────

    def update_relationship(self, target: str, affinity: int, trust: int):
        rel = self.data["relationships"].get(target, {"affinity": 50, "trust": 50})
        rel["affinity"] = max(0, min(100, rel["affinity"] + affinity))
        rel["trust"]    = max(0, min(100, rel["trust"]    + trust))
        self.data["relationships"][target] = rel

    # ── 记忆写入（异步，带重要性评分和情绪标注）─────────────

    async def add_episodic_memory_async(
        self,
        env_state: str,
        action: str,
        dialogue: str,
    ) -> None:
        """
        异步版记忆写入：并发完成重要性评分 + 情绪标注，再入库。
        由 agent.py 直接 await，不需要 to_thread 包裹。
        """
        memory_text = (
            f"【环境】: {env_state}\n"
            f"【我做出的动作】: {action}\n"
            f"【我说的话】: {dialogue}"
        )

        # 并发获取重要性分数和情绪标签
        importance, emotion = await asyncio.gather(
            self._importance_scorer.score(self.agent_name, env_state, action, dialogue),
            self._emotion_tagger.tag(memory_text),
        )

        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        doc_id = f"mem_epi_{int(time.time() * 1000)}"

        await asyncio.to_thread(
            self.episodic_memory.add,
            documents=[memory_text],
            metadatas=[{
                "time": timestamp,
                "importance": importance,
                "emotion": emotion,
                "round": self.current_round,
            }],
            ids=[doc_id],
        )

        print(f"  💾 [{self.agent_name}] 记忆落盘 | 重要性:{importance}/10 | 情绪:{emotion}")

        self.memory_counter += 1
        if self.memory_counter >= self.REFLECTION_THRESHOLD:
            await asyncio.to_thread(self._consolidate_sync)
            self.memory_counter = 0

    # 向后兼容的同步包装
    def add_episodic_memory(self, env_state: str, action: str, dialogue: str):
        asyncio.get_event_loop().run_until_complete(
            self.add_episodic_memory_async(env_state, action, dialogue)
        )

    # ── 记忆检索（带时序衰减评分）───────────────────────────

    def retrieve_episodic_memory(
        self,
        current_context: str,
        current_emotion: str = "",
        top_k: int = 2,
    ) -> str:
        """
        检索逻辑升级：
          1. 先做向量相似度检索拿 top 2K 候选
          2. 对每条候选用 _memory_score 重新排序（融合重要性+衰减+情绪相似）
          3. 取最终 top_k 条
        """
        past_memories = []

        # ── 语义记忆（长效认知）────────────────────────────
        if self.semantic_memory.count() > 0:
            sem = self.semantic_memory.query(
                query_texts=[current_context], n_results=1
            )
            docs = sem.get("documents", [[]])[0]
            if docs:
                past_memories.append("【深层长效认知】: " + docs[0])

        # ── 情境记忆（带衰减重排）──────────────────────────
        if self.episodic_memory.count() == 0:
            return "\n\n".join(past_memories) if past_memories else "（脑海中暂无相关往事）"

        k_fetch = min(top_k * 4, self.episodic_memory.count())
        epi = self.episodic_memory.query(
            query_texts=[current_context],
            n_results=k_fetch,
            include=["documents", "metadatas", "distances"],
        )
        docs      = epi.get("documents",  [[]])[0]
        metas     = epi.get("metadatas",  [[]])[0]
        distances = epi.get("distances",  [[]])[0]

        scored = []
        for doc, meta, dist in zip(docs, metas, distances):
            importance  = meta.get("importance", 5)
            stored_round= meta.get("round", 0)
            emotion_tag = meta.get("emotion", "")
            delta_t     = max(0, self.current_round - stored_round)
            query_sim   = max(0.0, 1.0 - dist)   # chromadb 返回的是 L2 距离，近似转相似度

            # 情绪相似奖励
            emotion_bonus = 0.2 if (current_emotion and current_emotion == emotion_tag) else 0.0

            score = _memory_score(importance, delta_t, query_sim) + emotion_bonus
            scored.append((score, doc, emotion_tag, importance))

        scored.sort(reverse=True)
        for score, doc, emotion_tag, importance in scored[:top_k]:
            past_memories.append(
                f"【近期往事 | 重要度:{importance}/10 | 情绪:{emotion_tag}】:\n{doc}"
            )

        return "\n\n".join(past_memories) if past_memories else "（脑海中暂无相关往事）"

    # ── 记忆折叠（同步，由 to_thread 调用）──────────────────

    def _consolidate_sync(self):
        if self.episodic_memory.count() == 0:
            return
        print(f"\n🧠 [{self.agent_name}] 记忆折叠中（重要性筛选）…")
        all_ep = self.episodic_memory.get(include=["documents", "metadatas", "ids"])
        docs   = all_ep.get("documents", [])
        metas  = all_ep.get("metadatas", [])
        ids    = all_ep.get("ids", [])

        if not docs:
            return

        # 只折叠重要度低的记忆（≤ 5分），保留高重要度记忆
        low_imp_docs = []
        low_imp_ids  = []
        keep_docs    = []
        for doc, meta, mid in zip(docs, metas, ids):
            imp = meta.get("importance", 5)
            if imp <= 5:
                low_imp_docs.append(doc)
                low_imp_ids.append(mid)
            else:
                keep_docs.append(doc)

        if not low_imp_docs:
            print(f"  ✨ [{self.agent_name}] 所有记忆重要度较高，跳过折叠")
            return

        prompt = MEMORY_CONSOLIDATION.substitute(
            agent_name=self.agent_name,
            episodes_text="\n\n".join(low_imp_docs),
        )
        try:
            resp = _sync_client.chat.completions.create(
                model=_settings.MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
            )
            insight = resp.choices[0].message.content.strip()
            if insight:
                print(f"  ✨ 折叠成新认知: {insight[:60]}…")
                self.semantic_memory.add(
                    documents=[insight],
                    metadatas=[{
                        "time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                        "source_count": len(low_imp_docs),
                    }],
                    ids=[f"sem_{int(time.time()*1000)}"],
                )
                self.episodic_memory.delete(ids=low_imp_ids)
                print(f"  🗑️  清理了 {len(low_imp_ids)} 条低重要度记忆，保留 {len(keep_docs)} 条")
        except Exception as e:
            print(f"⚠️ [{self.agent_name}] 记忆折叠失败: {e}")
