# tools/build_knowledge.py
import os
import sys
from openai import OpenAI

# 将根目录加入系统路径，以便引入 config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import API_KEY, BASE_URL, MODEL_NAME, DATA_DIR

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

class KnowledgeBuilder:
    def __init__(self):
        self.knowledge_base_dir = os.path.join(DATA_DIR, "knowledge")

    def auto_generate_corpus(self, era_name, topic, num_entries=5):
        """
        利用大模型自动提取特定话题的史料知识，生成本地知识库
        """
        print(f"🔍 正在从大模型中蒸馏关于【{topic}】的 {num_entries} 条深度历史知识...")
        
        prompt = f"""
        你现在是一个严谨的历史学家和数字人文数据工程师。
        请围绕主题“{topic}”（属于时代：{era_name}），为我的 RAG 知识库生成 {num_entries} 条具体的、有细节的、带有趣味性的历史背景资料。
        
        要求：
        1. 必须包含具体的名词、事件、物品或古文典故。
        2. 每条知识不要太长，直接输出事实。
        3. 必须以纯文本返回，每条知识之间用两个换行符 (\n\n) 隔开，不要任何额外的 markdown 标记或编号。
        """

        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3 # 温度调低，保证知识的客观准确性
            )
            knowledge_text = response.choices[0].message.content.strip()
            
            # 保存到本地文件
            self._save_to_disk(era_name, topic, knowledge_text)
            
        except Exception as e:
            print(f"❌ 知识蒸馏失败: {e}")

    def _save_to_disk(self, era_name, topic, content):
        era_dir = os.path.join(self.knowledge_base_dir, era_name)
        os.makedirs(era_dir, exist_ok=True)
        
        # 将主题转为合法的文件名
        safe_topic = topic.replace(" ", "_").replace("/", "_")
        file_path = os.path.join(era_dir, f"{safe_topic}.txt")
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
            
        print(f"✅ 成功生成并保存知识语料: {file_path}")

if __name__ == "__main__":
    builder = KnowledgeBuilder()
    
    # 只需要在这里配置你需要的知识点，脚本会自动帮你建库！
    topics_to_build = [
        {"era": "song", "topic": "苏轼在黄州时期的饮食习惯与猪肉颂"},
        {"era": "song", "topic": "苏轼与王朝云在黄州的感情与生活细节"},
        {"era": "song", "topic": "北宋时期文人画画与研墨的规矩和工具"},
        {"era": "modern", "topic": "2024年硅谷创业公司的常见黑话与工作节奏"}
    ]
    
    print("🚀 息壤自动化知识构建流水线启动...\n")
    for item in topics_to_build:
        builder.auto_generate_corpus(item["era"], item["topic"], num_entries=3)