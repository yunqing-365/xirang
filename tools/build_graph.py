# tools/build_graph.py
import os
import sys
import json
from openai import OpenAI

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import API_KEY, BASE_URL, MODEL_NAME, DATA_DIR

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

class GraphRAGBuilder:
    def __init__(self):
        self.knowledge_dir = os.path.join(DATA_DIR, "knowledge")

    def build_era_graph(self, era_name):
        era_path = os.path.join(self.knowledge_dir, era_name)
        if not os.path.exists(era_path):
            print(f"❌ 找不到时代文件夹: {era_path}")
            return

        print(f"\n🕸️ 正在启动 GraphRAG 关系网抽取引擎，扫描 [{era_name}] 时代的史料...")
        
        era_graph = {"entities": set(), "relationships": []}

        # 遍历该时代所有的 txt 史料文件
        for file_name in os.listdir(era_path):
            if file_name.endswith('.txt'):
                file_path = os.path.join(era_path, file_name)
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                # 提取图谱
                triplets = self._extract_triplets_with_llm(content, file_name)
                
                # 合并到全局图谱中
                for triplet in triplets:
                    era_graph["entities"].add(triplet["source"])
                    era_graph["entities"].add(triplet["target"])
                    era_graph["relationships"].append(triplet)

        # 将 set 转换为 list 以便 JSON 序列化
        era_graph["entities"] = list(era_graph["entities"])
        
        # 将最终图谱保存为 JSON 文件
        out_file = os.path.join(era_path, "graph_network.json")
        with open(out_file, 'w', encoding='utf-8') as f:
            json.dump(era_graph, f, ensure_ascii=False, indent=4)
            
        print(f"✅ 图谱构建完成！共提取出 {len(era_graph['entities'])} 个历史实体，{len(era_graph['relationships'])} 条羁绊连线。")
        print(f"📁 图谱资产已保存至: {out_file}")

    def _extract_triplets_with_llm(self, text, source_file):
        """核心壁垒：利用大模型执行 NER (命名实体识别) 和关系抽取"""
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
        {text[:1500]} # 限制长度避免超载
        """

        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
            raw_output = response.choices[0].message.content.strip()
            
            # 清理 Markdown 代码块包裹
            if raw_output.startswith("```json"):
                raw_output = raw_output[7:-3].strip()
            elif raw_output.startswith("```"):
                raw_output = raw_output[3:-3].strip()
                
            return json.loads(raw_output)
        except Exception as e:
            print(f"   ❌ 抽取该文件图谱时出错: {e}")
            return []

if __name__ == "__main__":
    builder = GraphRAGBuilder()
    builder.build_era_graph("song")