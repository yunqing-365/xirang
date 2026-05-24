# ingestion/entity_extractor.py
"""
实体抽取与知识图谱自动扩展模块

从文本块中提取：
  - 人物（姓名、字号、官职）
  - 地点（地名、古今映射）
  - 事件（时间+动词+主体）
  - 典籍（书名、引文来源）
  - 物品（文物、器具、食物）

并将三元组 (主体, 关系, 客体) 自动合并到
data/knowledge/{era}/graph_network.json 中。
"""
from __future__ import annotations
import json
import re
import sys
import os
from pathlib import Path
from typing import Any

# 兼容直接运行和作为模块导入
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import get_settings

_settings = get_settings()

from openai import OpenAI

_client = OpenAI(api_key=_settings.API_KEY, base_url=_settings.BASE_URL)

# ── 提示词模板 ────────────────────────────────────────────────
_EXTRACT_PROMPT = """\
你是一位专业的历史数字人文研究员，擅长从古籍文本中提取结构化知识。

请从以下【文本片段】中抽取所有有意义的实体和关系，输出严格 JSON，不加任何其他文字。

【文本片段】
{text}

【朝代背景】{era}  【参考年份】{year}

输出格式（严格遵守，不加注释）：
{{
  "entities": [
    {{
      "id": "唯一标识（人名/地名/事件名，英文下划线连接）",
      "name": "标准名称",
      "type": "PERSON|PLACE|EVENT|ARTIFACT|TEXT|CONCEPT",
      "aliases": ["字", "号", "谥号等别称"],
      "year": 公元年整数或null,
      "era": "朝代",
      "desc": "一句话简介（30字内）"
    }}
  ],
  "relations": [
    {{
      "source": "实体id",
      "target": "实体id",
      "type": "关系类型（师友/上下级/夫妻/著作/发生于/使用/参与等）",
      "desc": "关系说明（20字内）",
      "year": 公元年整数或null
    }}
  ]
}}

注意：
1. 只抽取文中明确出现的实体，不要推断文中没有的
2. 人物必须给出 type=PERSON，地点 type=PLACE
3. 同一实体可能用不同称谓出现，统一用最常见的名字作为 id
4. id 用中文即可，保持简洁
"""

_COREF_PROMPT = """\
以下两批实体来自同一朝代的不同文本，请找出指向同一历史人物/地点的实体对，
输出合并建议，JSON 格式：
{{"merges": [{{"keep": "保留的id", "remove": "要合并掉的id", "reason": "理由"}}]}}

批次A：{batch_a}
批次B：{batch_b}

只输出需要合并的对，没有则输出 {{"merges": []}}
"""


def extract_entities(text: str, era: str = "", year: int = 1000) -> dict:
    """调用 LLM 从单个文本块提取实体和关系"""
    prompt = _EXTRACT_PROMPT.format(
        text=text[:1500],
        era=era or "不明",
        year=year,
    )
    try:
        resp = _client.chat.completions.create(
            model=_settings.MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1500,
            temperature=0.1,
            timeout=25,
        )
        raw = resp.choices[0].message.content.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        return json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"  ⚠️  JSON 解析失败: {e}")
        return {"entities": [], "relations": []}
    except Exception as e:
        print(f"  ⚠️  LLM 调用失败: {e}")
        return {"entities": [], "relations": []}


def coref_resolution(batch_a: list, batch_b: list) -> list[dict]:
    """
    跨批次实体共指消解：找出两批实体中指向同一真实历史存在的条目。
    返回合并建议列表。
    """
    if not batch_a or not batch_b:
        return []
    a_str = json.dumps([{"id": e["id"], "name": e["name"], "desc": e.get("desc","")} for e in batch_a[:15]], ensure_ascii=False)
    b_str = json.dumps([{"id": e["id"], "name": e["name"], "desc": e.get("desc","")} for e in batch_b[:15]], ensure_ascii=False)
    prompt = _COREF_PROMPT.format(batch_a=a_str, batch_b=b_str)
    try:
        resp = _client.chat.completions.create(
            model=_settings.MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0.0,
            timeout=15,
        )
        raw = resp.choices[0].message.content.strip().replace("```json","").replace("```","").strip()
        return json.loads(raw).get("merges", [])
    except Exception:
        return []


class GraphUpdater:
    """
    增量更新 data/knowledge/{era}/graph_network.json。
    支持：新增实体/关系、共指消解合并、重复检测。
    """

    def __init__(self, era: str, data_dir: str):
        self.era = era
        self.graph_path = Path(data_dir) / "knowledge" / era / "graph_network.json"
        self.graph_path.parent.mkdir(parents=True, exist_ok=True)
        self.graph = self._load()

    def _load(self) -> dict:
        if self.graph_path.exists():
            try:
                return json.loads(self.graph_path.read_text(encoding="utf-8"))
            except Exception:
                pass
        return {"entities": [], "relationships": [], "meta": {"era": self.era, "version": 1}}

    def save(self):
        self.graph_path.write_text(
            json.dumps(self.graph, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    # ── 实体层 ──────────────────────────────────────────────
    def _entity_index(self) -> dict[str, int]:
        return {e["id"]: i for i, e in enumerate(self.graph["entities"])}

    def upsert_entity(self, entity: dict) -> str:
        """插入或更新实体，返回最终使用的 id"""
        idx = self._entity_index()
        eid = entity.get("id", "").strip()
        if not eid:
            return ""

        if eid in idx:
            # 更新：补充别称、完善描述
            existing = self.graph["entities"][idx[eid]]
            new_aliases = entity.get("aliases") or []
            existing_aliases = existing.get("aliases") or []
            existing["aliases"] = list(set(existing_aliases + new_aliases))
            # 补充年份（原来没有时）
            if existing.get("year") is None and entity.get("year"):
                existing["year"] = entity["year"]
            return eid
        else:
            self.graph["entities"].append({
                "id":      eid,
                "name":    entity.get("name", eid),
                "type":    entity.get("type", "CONCEPT"),
                "aliases": entity.get("aliases") or [],
                "year":    entity.get("year"),
                "era":     entity.get("era", self.era),
                "desc":    entity.get("desc", ""),
            })
            return eid

    def merge_entities(self, keep_id: str, remove_id: str):
        """共指消解：将 remove_id 合并到 keep_id，更新所有关系引用"""
        idx = self._entity_index()
        if remove_id not in idx or keep_id not in idx:
            return
        # 将 remove 的别称迁移给 keep
        keep_e = self.graph["entities"][idx[keep_id]]
        rem_e  = self.graph["entities"][idx[remove_id]]
        keep_e["aliases"] = list(set(
            (keep_e.get("aliases") or []) +
            (rem_e.get("aliases") or []) +
            [rem_e["name"]]
        ))
        # 更新关系中所有对 remove_id 的引用
        for rel in self.graph["relationships"]:
            if rel["source"] == remove_id:
                rel["source"] = keep_id
            if rel["target"] == remove_id:
                rel["target"] = keep_id
        # 删除 remove 实体
        self.graph["entities"] = [
            e for e in self.graph["entities"] if e["id"] != remove_id
        ]
        print(f"  🔗 合并实体：{remove_id} → {keep_id}")

    # ── 关系层 ──────────────────────────────────────────────
    def _rel_key(self, rel: dict) -> tuple:
        return (rel["source"], rel["target"], rel.get("type",""))

    def upsert_relation(self, rel: dict):
        """插入关系（去重）"""
        existing_keys = {self._rel_key(r) for r in self.graph["relationships"]}
        key = self._rel_key(rel)
        if key not in existing_keys:
            self.graph["relationships"].append({
                "source": rel["source"],
                "target": rel["target"],
                "type":   rel.get("type", "相关"),
                "desc":   rel.get("desc", ""),
                "year":   rel.get("year"),
            })

    # ── 批量摄入 ────────────────────────────────────────────
    def ingest_extraction(self, extraction: dict):
        """将一次 extract_entities() 的结果增量写入图谱"""
        valid_ids: set[str] = set()

        for entity in extraction.get("entities", []):
            eid = self.upsert_entity(entity)
            if eid:
                valid_ids.add(eid)

        for rel in extraction.get("relations", []):
            src, tgt = rel.get("source",""), rel.get("target","")
            # 只保留源/目标都在当前图谱中的关系
            if src and tgt:
                self.upsert_entity({"id": src, "name": src, "type": "CONCEPT"})
                self.upsert_entity({"id": tgt, "name": tgt, "type": "CONCEPT"})
                self.upsert_relation(rel)

    def stats(self) -> dict:
        types: dict[str,int] = {}
        for e in self.graph["entities"]:
            t = e.get("type","?")
            types[t] = types.get(t,0) + 1
        return {
            "entities": len(self.graph["entities"]),
            "relations": len(self.graph["relationships"]),
            "by_type": types,
        }


# ── 批量处理入口 ─────────────────────────────────────────────
def process_chunks_to_graph(
    chunks_with_meta: list[tuple[str, dict]],
    era: str,
    data_dir: str,
    coref_every_n: int = 20,
    verbose: bool = True,
) -> GraphUpdater:
    """
    将解析好的文本块列表批量提取实体并写入图谱。

    coref_every_n: 每处理 N 个块后做一次共指消解
    """
    updater = GraphUpdater(era=era, data_dir=data_dir)
    prev_batch_entities: list = []

    for i, (chunk, meta) in enumerate(chunks_with_meta):
        if verbose:
            print(f"  [{i+1:3d}/{len(chunks_with_meta)}] 抽取实体 | {meta.get('source','?')[:25]} | {len(chunk)}字")

        extraction = extract_entities(
            text=chunk,
            era=meta.get("era", era),
            year=meta.get("year", 1000),
        )
        updater.ingest_extraction(extraction)

        # 共指消解（每 coref_every_n 块）
        if i > 0 and i % coref_every_n == 0 and prev_batch_entities:
            curr_entities = [e for e in updater.graph["entities"][-coref_every_n:]]
            merges = coref_resolution(prev_batch_entities, curr_entities)
            for merge in merges:
                updater.merge_entities(merge["keep"], merge["remove"])
        
        if i % coref_every_n == 0:
            prev_batch_entities = list(updater.graph["entities"][-coref_every_n:])

    updater.save()
    if verbose:
        s = updater.stats()
        print(f"\n  ✅ 图谱更新完成 | 实体 {s['entities']} | 关系 {s['relations']} | 分布: {s['by_type']}")
    return updater


if __name__ == "__main__":
    # 自测：用示例文本验证提取效果
    sample_chunks = [
        (
            "元丰三年，苏轼以乌台诗案被贬黄州团练副使。"
            "黄州僻陋，苏轼生活拮据，躬耕东坡，自号东坡居士。"
            "佛印禅师居金山寺，与苏轼诗文往来，互为知己。",
            {"source": "test.txt", "era": "song", "year": 1080}
        ),
        (
            "王朝云随侍苏轼于黄州，伺候起居，情谊深厚。"
            "朝云常言：学士满腹皆不合时宜。"
            "元丰五年，苏轼作《赤壁赋》，寄情山水，抒发胸中块垒。",
            {"source": "test.txt", "era": "song", "year": 1082}
        ),
    ]

    import tempfile, os
    with tempfile.TemporaryDirectory() as tmpdir:
        os.makedirs(os.path.join(tmpdir, "knowledge", "song"), exist_ok=True)
        updater = process_chunks_to_graph(
            sample_chunks, era="song", data_dir=tmpdir, verbose=True
        )
        print("\n实体列表：")
        for e in updater.graph["entities"]:
            print(f"  {e['id']:12} [{e['type']:7}] {e.get('desc','')[:30]}")
        print("\n关系列表：")
        for r in updater.graph["relationships"]:
            print(f"  {r['source']} --[{r['type']}]--> {r['target']}")
