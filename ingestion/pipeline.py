# ingestion/pipeline.py
"""
息壤多模态知识库摄入流水线（主控器）

使用方式：

  # 处理单个文件
  python -m ingestion.pipeline ingest --file data/raw_documents/song/东坡志林.txt --era song

  # 处理整个朝代目录
  python -m ingestion.pipeline ingest --era song

  # 仅重建图谱（跳过文本块生成）
  python -m ingestion.pipeline build-graph --era song

  # 查看摄入状态
  python -m ingestion.pipeline status

流水线五阶段：
  ① PARSE     原始文件 → 清洗文本块
  ② NORMALIZE 繁简转换 + 纪年标准化
  ③ EXTRACT   LLM 实体/关系抽取
  ④ INDEX     写入 ChromaDB 向量索引
  ⑤ GRAPH     增量更新知识图谱
"""
from __future__ import annotations
import argparse
import json
import sys
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

# 项目根目录加入 sys.path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from config import get_settings
_settings = get_settings()

from ingestion.text_parser    import AncientTextParser
from ingestion.entity_extractor import process_chunks_to_graph
from ingestion.year_normalizer  import normalize_era_name

DATA_DIR = Path(_settings.DATA_DIR)

# ── 摄入状态日志 ─────────────────────────────────────────────
STATUS_FILE = DATA_DIR / "ingestion_status.json"

def _load_status() -> dict:
    if STATUS_FILE.exists():
        try:
            return json.loads(STATUS_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"processed": {}, "last_run": None, "total_chunks": 0, "total_entities": 0}

def _save_status(status: dict):
    STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATUS_FILE.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")

def _file_key(path: Path) -> str:
    return str(path.relative_to(DATA_DIR) if path.is_relative_to(DATA_DIR) else path)

# ── 阶段 ①②：解析 + 标准化 ───────────────────────────────────
def stage_parse(
    raw_path: Path,
    era: str,
    chunk_size: int = 400,
    force: bool = False,
) -> list[tuple[str, dict]]:
    """
    解析原始文件目录，返回 [(chunk_text, metadata), ...]。
    跳过已处理的文件（除非 force=True）。
    """
    status = _load_status()
    parser = AncientTextParser(era=era, target_chunk_size=chunk_size)
    all_chunks: list[tuple[str, dict]] = []
    skipped = 0

    files = sorted(raw_path.rglob("*"))
    supported = {".txt", ".md", ".pdf", ".text"}
    doc_files = [f for f in files if f.is_file() and f.suffix.lower() in supported]

    if not doc_files:
        print(f"  ℹ️  {raw_path} 下没有找到支持的文档文件")
        return []

    print(f"  📂 发现 {len(doc_files)} 个文档文件")

    for f in doc_files:
        fkey = _file_key(f)
        mtime = f.stat().st_mtime

        if not force and fkey in status["processed"]:
            if status["processed"][fkey].get("mtime") == mtime:
                skipped += 1
                continue

        print(f"    ▸ 解析: {f.name}")
        try:
            chunks = parser.parse(f)
            all_chunks.extend(chunks)
            status["processed"][fkey] = {
                "mtime": mtime,
                "chunks": len(chunks),
                "era": era,
                "parsed_at": datetime.now().isoformat(),
            }
        except Exception as e:
            print(f"    ⚠️  解析失败: {e}")

    if skipped:
        print(f"  ⏭️  跳过 {skipped} 个未变更文件（使用 --force 重新处理）")

    _save_status(status)
    return all_chunks


# ── 阶段 ④：写入 ChromaDB ─────────────────────────────────────
def stage_index(
    chunks_with_meta: list[tuple[str, dict]],
    era: str,
    rebuild: bool = False,
) -> int:
    """
    将文本块写入 ChromaDB 向量索引。
    使用项目现有的 RagEngine（双轨索引：文本 + 视觉）。
    返回新增的块数量。
    """
    if not chunks_with_meta:
        return 0

    try:
        from rag_engine import RagEngine
        engine = RagEngine(era_name=era)

        # 如果重建，先清空
        if rebuild:
            print(f"  🗑️  清空 [{era}] 现有向量索引...")
            try:
                engine.text_collection.delete(
                    where={"era": era}
                )
            except Exception:
                pass

        texts  = [c[0] for c in chunks_with_meta]
        metas  = [c[1] for c in chunks_with_meta]
        ids    = [f"chunk_{era}_{i}_{int(time.time())}" for i in range(len(texts))]

        # 批量写入（每批 50 条，避免超时）
        batch_size = 50
        total_added = 0
        for start in range(0, len(texts), batch_size):
            end = min(start + batch_size, len(texts))
            engine.text_collection.add(
                documents=texts[start:end],
                metadatas=metas[start:end],
                ids=ids[start:end],
            )
            total_added += end - start
            print(f"  📥 索引写入 {total_added}/{len(texts)} 块...")

        return total_added

    except ImportError:
        print("  ⚠️  rag_engine 未找到，跳过向量索引阶段")
        return 0
    except Exception as e:
        print(f"  ⚠️  向量索引写入失败: {e}")
        return 0


# ── 阶段 ③⑤：实体抽取 + 图谱更新 ────────────────────────────
def stage_graph(
    chunks_with_meta: list[tuple[str, dict]],
    era: str,
    coref_every_n: int = 20,
) -> dict:
    """
    LLM 实体抽取 + 增量更新 graph_network.json。
    返回图谱统计信息。
    """
    if not chunks_with_meta:
        return {}

    updater = process_chunks_to_graph(
        chunks_with_meta,
        era=era,
        data_dir=str(DATA_DIR),
        coref_every_n=coref_every_n,
        verbose=True,
    )
    return updater.stats()


# ── 主流水线 ──────────────────────────────────────────────────
def run_ingestion(
    era: str,
    file_path: Optional[Path] = None,
    chunk_size: int = 400,
    force: bool = False,
    skip_graph: bool = False,
    skip_index: bool = False,
    rebuild_index: bool = False,
) -> dict:
    """
    完整摄入流程：PARSE → INDEX → GRAPH

    era:          朝代标识（如 song/tang/ming）
    file_path:    指定单个文件（None 则处理整个 era 目录）
    force:        强制重新处理已缓存的文件
    skip_graph:   跳过实体抽取（节省 API 调用，仅更新向量索引）
    skip_index:   跳过向量索引
    rebuild_index: 重建向量索引（清空后重写）
    """
    era = normalize_era_name(era)
    raw_dir = DATA_DIR / "raw_documents" / era
    raw_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    print(f"\n{'='*55}")
    print(f"🏛️  息壤知识摄入流水线 | 朝代: {era}")
    print(f"{'='*55}")

    # ① 确定输入路径
    if file_path:
        if not file_path.exists():
            print(f"❌ 文件不存在: {file_path}")
            return {}
        # 将单文件临时解析
        parser = AncientTextParser(era=era, target_chunk_size=chunk_size)
        print(f"\n📄 阶段①② 解析文件: {file_path.name}")
        chunks = parser.parse(file_path)
        print(f"  ✅ 解析完成: {len(chunks)} 块")
    else:
        print(f"\n📁 阶段①② 批量解析目录: {raw_dir}")
        chunks = stage_parse(raw_dir, era=era, chunk_size=chunk_size, force=force)
        print(f"  ✅ 解析完成: {len(chunks)} 块")

    if not chunks:
        print(f"\n⚠️  没有可处理的文本块")
        print(f"   请将文档放入: {raw_dir}/")
        _print_supported_formats()
        return {}

    # ② 向量索引
    if not skip_index:
        print(f"\n📥 阶段④ 写入向量索引（共 {len(chunks)} 块）")
        indexed = stage_index(chunks, era=era, rebuild=rebuild_index)
        print(f"  ✅ 索引完成: {indexed} 块")
    else:
        print(f"\n⏭️  跳过向量索引阶段")
        indexed = 0

    # ③ 实体抽取 + 图谱
    if not skip_graph:
        print(f"\n🕸️  阶段③⑤ 实体抽取 + 知识图谱（共 {len(chunks)} 块）")
        print(f"  ⚡ 此阶段调用 LLM，预计耗时 {len(chunks) * 3} 秒...")
        graph_stats = stage_graph(chunks, era=era)
    else:
        print(f"\n⏭️  跳过图谱构建阶段")
        graph_stats = {}

    # ④ 更新全局状态
    elapsed = time.time() - start_time
    status = _load_status()
    status["last_run"] = datetime.now().isoformat()
    status["total_chunks"] = status.get("total_chunks", 0) + len(chunks)
    status["total_entities"] = graph_stats.get("entities", 0)
    _save_status(status)

    summary = {
        "era": era,
        "chunks_parsed": len(chunks),
        "chunks_indexed": indexed,
        "graph": graph_stats,
        "elapsed_sec": round(elapsed, 1),
    }
    print(f"\n{'='*55}")
    print(f"✨ 摄入完成 | 耗时 {elapsed:.1f}s")
    print(f"   文本块: {len(chunks)} | 索引: {indexed} | 实体: {graph_stats.get('entities','-')}")
    print(f"{'='*55}\n")
    return summary


def _print_supported_formats():
    print("""
  支持的文档格式：
    📄 .txt / .md   — 纯文本，UTF-8 编码（推荐）
    📕 .pdf         — 自动提取文字层，竖排OCR去噪
  
  目录结构示例：
    data/raw_documents/
      song/          ← 北宋/南宋
        东坡志林.txt
        黄州寒食帖.txt
        地方志_黄州府志.pdf
      tang/          ← 唐朝
        全唐诗摘录.txt
      ming/          ← 明朝
        ...
""")


# ── 状态查看 ──────────────────────────────────────────────────
def show_status():
    status = _load_status()
    print("\n📊 息壤知识库摄入状态")
    print(f"  最后运行: {status.get('last_run','从未')}")
    print(f"  累计文本块: {status.get('total_chunks',0)}")
    print(f"  图谱实体: {status.get('total_entities',0)}")
    processed = status.get("processed", {})
    print(f"  已处理文件: {len(processed)}")

    # 按朝代分组
    by_era: dict[str, list] = {}
    for fkey, info in processed.items():
        era = info.get("era", "unknown")
        by_era.setdefault(era, []).append(info)
    for era, files in sorted(by_era.items()):
        total_chunks = sum(f.get("chunks", 0) for f in files)
        print(f"    [{era:10}] {len(files)} 文件 / {total_chunks} 块")

    # 知识图谱概览
    for era_dir in sorted((DATA_DIR / "knowledge").glob("*/graph_network.json")):
        era_name = era_dir.parent.name
        try:
            g = json.loads(era_dir.read_text(encoding="utf-8"))
            e_count = len(g.get("entities", []))
            r_count = len(g.get("relationships", []))
            print(f"  📊 [{era_name:10}] 图谱: {e_count} 实体 / {r_count} 关系")
        except Exception:
            pass


# ── CLI 入口 ──────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="息壤多模态知识库摄入流水线",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command")

    # ingest 命令
    p_ingest = sub.add_parser("ingest", help="摄入文档到知识库")
    p_ingest.add_argument("--era",          required=True,  help="朝代（如 song/tang/ming）")
    p_ingest.add_argument("--file",         default=None,   help="指定单个文件路径")
    p_ingest.add_argument("--chunk-size",   type=int, default=400, help="每块目标字数（默认400）")
    p_ingest.add_argument("--force",        action="store_true", help="强制重新处理已缓存文件")
    p_ingest.add_argument("--skip-graph",   action="store_true", help="跳过实体抽取与图谱更新")
    p_ingest.add_argument("--skip-index",   action="store_true", help="跳过向量索引写入")
    p_ingest.add_argument("--rebuild-index",action="store_true", help="重建向量索引（清空重写）")

    # build-graph 命令（仅重建图谱，不重新解析文本）
    p_graph = sub.add_parser("build-graph", help="仅重建知识图谱")
    p_graph.add_argument("--era", required=True, help="朝代")

    # status 命令
    sub.add_parser("status", help="查看摄入状态")

    args = parser.parse_args()

    if args.command == "ingest":
        run_ingestion(
            era=args.era,
            file_path=Path(args.file) if args.file else None,
            chunk_size=args.chunk_size,
            force=args.force,
            skip_graph=args.skip_graph,
            skip_index=args.skip_index,
            rebuild_index=args.rebuild_index,
        )

    elif args.command == "build-graph":
        era = normalize_era_name(args.era)
        kb_dir = DATA_DIR / "knowledge" / era
        # 读取已有的 .txt 知识文件重新抽取
        chunks = []
        for f in sorted(kb_dir.glob("*.txt")):
            text = f.read_text(encoding="utf-8")
            for i, chunk in enumerate(text.split("\n\n")):
                if chunk.strip():
                    chunks.append((chunk.strip(), {"source": f.name, "era": era, "year": 1000}))
        if chunks:
            stage_graph(chunks, era=era)
        else:
            print(f"⚠️  {kb_dir} 下没有文本文件")

    elif args.command == "status":
        show_status()

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
