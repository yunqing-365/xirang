# tools/vlm_image_parser.py

import os
import sys
from openai import OpenAI

# 引入统一配置
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import VISION_API_KEY, VISION_BASE_URL, VISION_MODEL_NAME, DATA_DIR

# 使用配置好的变量初始化客户端
client = OpenAI(api_key=VISION_API_KEY, base_url=VISION_BASE_URL)

class VisualKnowledgeParser:
    # ... 之前的逻辑保持不变 ...
    def _analyze_image_with_vlm(self, image_path, file_name, out_file_path, era_name):
        # 这里的调用现在是绝对安全的
        response = client.chat.completions.create(
            model=VISION_MODEL_NAME,
            # ... 其余参数 ...
        )

class VisualKnowledgeParser:
    def __init__(self):
        self.raw_dir = os.path.join(DATA_DIR, "raw_documents")
        self.knowledge_dir = os.path.join(DATA_DIR, "knowledge")
        self.supported_formats = ['.png', '.jpg', '.jpeg', '.webp']

    def encode_image_to_base64(self, image_path):
        """将图片转换为 Base64 编码"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def parse_all_images(self):
        """扫描所有时代文件夹中的图片并进行解析"""
        if not os.path.exists(self.raw_dir):
            print(f"❌ 找不到原始文件目录: {self.raw_dir}")
            return

        for era_name in os.listdir(self.raw_dir):
            era_path = os.path.join(self.raw_dir, era_name)
            if os.path.isdir(era_path):
                self._process_era_images(era_name, era_path)

    def _process_era_images(self, era_name, era_path):
        print(f"\n👁️ 正在启动 VLM 视觉雷达，扫描 [{era_name}] 时代的视觉资产...")
        
        out_era_dir = os.path.join(self.knowledge_dir, era_name)
        os.makedirs(out_era_dir, exist_ok=True)

        for file_name in os.listdir(era_path):
            ext = os.path.splitext(file_name)[1].lower()
            if ext in self.supported_formats:
                file_path = os.path.join(era_path, file_name)
                
                # 检查是否已经解析过，避免重复烧钱
                safe_name = file_name.rsplit('.', 1)[0]
                out_file_path = os.path.join(out_era_dir, f"VISUAL_{safe_name}.txt")
                if os.path.exists(out_file_path):
                    print(f"⏭️ 图片已解析过，跳过: {file_name}")
                    continue

                self._analyze_image_with_vlm(file_path, file_name, out_file_path, era_name)

    def _analyze_image_with_vlm(self, image_path, file_name, out_file_path, era_name):
        print(f"🖼️ 正在凝视古画/文物: {file_name} ...")
        
        base64_image = self.encode_image_to_base64(image_path)
        
        # 精心设计的 Prompt，要求模型以“史料记载”的口吻输出
        prompt = f"""
        你是一位顶级的数字人文研究员、艺术史学家和文物鉴定专家。
        这是属于【{era_name}】时代的一份视觉资料（古画、遗址、手稿或器物）。
        
        请仔细观察，并用大约400字撰写一份详尽的学术描述。
        要求：
        1. 描述画面的构图、色彩、笔触或器物纹理。
        2. 推测它所蕴含的情感氛围或历史背景。
        3. 将你的输出直接写成一段严谨的背景知识文本，不要有任何多余的开头（如“好的，这是描述...”），直接输出事实。
        """

        try:
            response = client.chat.completions.create(
                model=VISION_MODEL_NAME,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=800,
                temperature=0.4
            )
            
            description = response.choices[0].message.content.strip()
            
            # 将视觉描述固化为文本知识资产
            with open(out_file_path, 'w', encoding='utf-8') as f:
                # 在开头标注来源，方便后续智能体溯源
                f.write(f"【视觉文献来源：{file_name}】\n\n{description}")
                
            print(f"✅ 解析成功！提取的视觉意境已转化为高维语义特征并固化。")
            
        except Exception as e:
            print(f"❌ 视觉解析失败 (请检查你的 API_KEY 是否支持 {VISION_MODEL_NAME} 模型): {e}")

if __name__ == "__main__":
    parser = VisualKnowledgeParser()
    parser.parse_all_images()