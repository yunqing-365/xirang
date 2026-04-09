# tools/build_graph.py
import os
import sys
import json
import networkx as nx
from networkx.algorithms import community
from openai import OpenAI

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import API_KEY, BASE_URL, MODEL_NAME, DATA_DIR

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

class GraphRAGBuilder:
    def __init__(self):
        self.knowledge_dir = os.path.join(DATA_DIR, "knowledge")

    def _normalize_entities_with_llm(self, entities):
        print(f"\n🧠 正在启动实体共指消解，总计 {len(entities)} 个待核查实体...")
        if not entities:
            return {}
            
        prompt = f"""
        作为历史图谱专家，请找出以下实体列表中指向同一人/物的别名，并输出为规范映射字典。
        合并原则：将字号、别称映射为最广为人知的标准真名/全称。例如：{{"苏东坡": "苏轼", "苏子瞻": "苏轼"}}。没有别名的实体可以直接略过或映射为自身。
        必须只返回纯 JSON 字典格式，不要有任何 markdown 标记或其他说明。
        实体列表：{list(entities)}
        """
        
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
            raw = response.choices[0].message.content.strip()
            
            md_json = "`" * 3 + "json"
            md_empty = "`" * 3
            if raw.startswith(md_json): 
                raw = raw[7:-3].strip()
            elif raw.startswith(md_empty): 
                raw = raw[3:-3].strip()
            
            mapping = json.loads(raw)
            print(f"✅ 消解完成，发现 {len(mapping)} 条映射规则。")
            return mapping
        except Exception as e:
            print(f"⚠️ 实体消解失败，将降级使用原始实体: {e}")
            return {}

    def build_era_graph(self, era_name):
        era_path = os.path.join(self.knowledge_dir, era_name)
        if not os.path.exists(era_path):
            print(f"❌ 找不到时代文件夹: {era_path}")
            return

        print(f"\n🕸️ 正在启动 GraphRAG 关系网抽取引擎，扫描 [{era_name}] 时代的史料...")
        
        raw_triplets = []
        all_entities = set()

        # 1. 第一阶段：全量抽取并暂存
        for file_name in os.listdir(era_path):
            if file_name.endswith('.txt'):
                file_path = os.path.join(era_path, file_name)
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                triplets = self._extract_triplets_with_llm(content, file_name)
                
                for t in triplets:
                    raw_triplets.append(t)
                    all_entities.add(t["source"])
                    all_entities.add(t["target"])

        # 2. 第二阶段：全局实体归一化
        entity_mapping = self._normalize_entities_with_llm(all_entities)
        
        # 3. 第三阶段：构建标准图谱
        era_graph = {"entities": set(), "relationships": []}
        
        # 准备一个 networkx 图对象，用于计算社区
        G = nx.Graph()
        
        for t in raw_triplets:
            norm_source = entity_mapping.get(t["source"], t["source"])
            norm_target = entity_mapping.get(t["target"], t["target"])
            
            if norm_source != norm_target:
                era_graph["entities"].add(norm_source)
                era_graph["entities"].add(norm_target)
                era_graph["relationships"].append({
                    "source": norm_source,
                    "target": norm_target,
                    "relation": t["relation"]
                })
                # 将节点和边加入到 networkx 图中
                G.add_edge(norm_source, norm_target, relation=t["relation"])

        era_graph["entities"] = list(era_graph["entities"])
        
        out_file = os.path.join(era_path, "graph_network.json")
        with open(out_file, 'w', encoding='utf-8') as f:
            json.dump(era_graph, f, ensure_ascii=False, indent=4)
            
        print(f"✅ 基础图谱构建完成！共 {len(era_graph['entities'])} 个实体，{len(era_graph['relationships'])} 条连线。")

        # ========================================================
        # [核心升级] 第四阶段：宏观社区发现 (Global GraphRAG)
        # ========================================================
        self._detect_and_summarize_communities(G, era_path)

    def _detect_and_summarize_communities(self, G, era_path):
        """利用算法进行社区划分，并让大模型生成社区宏观摘要"""
        if len(G.nodes) < 3:
            return
            
        print("\n🌍 正在进行宏观社区发现与派系总结 (Global GraphRAG)...")
        
        # 使用贪婪模块度算法找出网络中的“社区”（即历史派系、关系网圈子）
        communities = community.greedy_modularity_communities(G)
        community_data = []
        
        for i, comm in enumerate(communities):
            # 只有当一个圈子包含 3 个以上的实体时，才值得被作为“宏观特征”进行总结
            if len(comm) >= 3:
                comm_nodes = list(comm)
                print(f"   -> 发现第 {i+1} 号社区，包含成员：{comm_nodes[:5]}...")
                
                # 让大模型为这个群体写传
                prompt = f"""
                你是一个历史社会学家。以下是通过图论算法，在历史资料中挖掘出的一个联系非常紧密的“人物/实体关系网”。
                请你分析这些实体为什么会聚集在一起？请用大约 150 字，总结这个“圈子”的核心特征、政治倾向或历史标签。
                直接输出总结文本，不要有任何前缀。
                实体名单：{comm_nodes}
                """
                
                try:
                    response = client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.3
                    )
                    summary = response.choices[0].message.content.strip()
                    community_data.append({
                        "community_id": i + 1,
                        "members": comm_nodes,
                        "summary": summary
                    })
                    print(f"      [社区画像]: {summary[:30]}...")
                except Exception as e:
                    print(f"      ⚠️ 社区生成失败: {e}")

        if community_data:
            out_file = os.path.join(era_path, "community_summaries.json")
            with open(out_file, 'w', encoding='utf-8') as f:
                json.dump(community_data, f, ensure_ascii=False, indent=4)
            print(f"✅ 宏观社区总结完成！共划分为 {len(community_data)} 个派系，资产已保存至: {out_file}")

    def _extract_triplets_with_llm(self, text, source_file):
        print(f"   -> 正在从 {source_file} 中提炼历史羁绊...")
        
        prompt = f"""
        你是一个知识图谱构建专家。请从以下历史文献中，提取出关键的“三元组”关系网络。
        格式要求：严格以 JSON 数组格式输出，不要有任何其他解释文字。
        示例:
        [
            {{"source": "苏轼", "target": "黄州", "relation": "被贬谪至"}},
            {{"source": "东坡肉", "target": "苏轼", "relation": "由其发明"}}
        ]
        
        待抽取的文献文本：
        {text[:1500]}
        """

        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
            raw_output = response.choices[0].message.content.strip()
            
            md_json = "`" * 3 + "json"
            md_empty = "`" * 3
            if raw_output.startswith(md_json):
                raw_output = raw_output[7:-3].strip()
            elif raw_output.startswith(md_empty):
                raw_output = raw_output[3:-3].strip()
                
            return json.loads(raw_output)
        except Exception as e:
            print(f"   ❌ 抽取该文件图谱时出错: {e}")
            return []

if __name__ == "__main__":
    builder = GraphRAGBuilder()
    builder.build_era_graph("song")