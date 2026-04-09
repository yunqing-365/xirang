# tools/import_local_docs.py
import os
import sys

# 如果你安装了 PyPDF2，解除这里的注释
try:
    from PyPDF2 import PdfReader
    HAS_PDF = True
except ImportError:
    HAS_PDF = False

# 将根目录加入系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_DIR

class LocalDocumentImporter:
    def __init__(self):
        self.raw_dir = os.path.join(DATA_DIR, "raw_documents")
        self.knowledge_dir = os.path.join(DATA_DIR, "knowledge")

    def process_all_eras(self):
        """遍历 raw_documents 下的所有时代文件夹，并处理其中的文件"""
        if not os.path.exists(self.raw_dir):
            os.makedirs(self.raw_dir)
            print(f"📁 已创建私有文件上传目录: {self.raw_dir}，请将你的私密资料放入对应的时代文件夹中。")
            return

        for era_name in os.listdir(self.raw_dir):
            era_path = os.path.join(self.raw_dir, era_name)
            if os.path.isdir(era_path):
                self._process_era_folder(era_name, era_path)

    def _process_era_folder(self, era_name, era_path):
        """处理特定时代文件夹下的所有私密文件"""
        print(f"\n🔍 正在扫描 [{era_name}] 时代的私有文献...")
        
        # 确保输出目录存在
        out_era_dir = os.path.join(self.knowledge_dir, era_name)
        os.makedirs(out_era_dir, exist_ok=True)

        for file_name in os.listdir(era_path):
            file_path = os.path.join(era_path, file_name)
            
            # 1. 提取文本
            text_content = ""
            if file_name.endswith(".txt") or file_name.endswith(".md"):
                with open(file_path, 'r', encoding='utf-8') as f:
                    text_content = f.read()
            elif file_name.endswith(".pdf"):
                if HAS_PDF:
                    text_content = self._extract_text_from_pdf(file_path)
                else:
                    print(f"⚠️ 无法处理 {file_name}，请先运行: pip install PyPDF2")
                    continue
            else:
                print(f"⏭️ 暂不支持的文件格式，已跳过: {file_name}")
                continue
            
            if not text_content.strip():
                continue

            # 2. 文本清洗与智能切块 (Chunking)
            # 在真实的 RAG 中，如果把 10 万字的 PDF 直接丢进向量库会崩溃。
            # 我们必须把它切成一段段 500 字左右的“知识块”。
            chunks = self._chunk_text(text_content, chunk_size=500, overlap=50)
            
            # 3. 将切片后的私有资产固化到标准知识库中
            safe_name = file_name.rsplit('.', 1)[0]
            out_file_path = os.path.join(out_era_dir, f"PRIVATE_{safe_name}.txt")
            
            with open(out_file_path, 'w', encoding='utf-8') as f:
                # 用双换行符隔开每个切块，这是我们 rag_engine.py 识别的标志
                f.write("\n\n".join(chunks))
            
            print(f"✅ 成功解析私密文件: {file_name} -> 提取了 {len(chunks)} 个高维知识块。")

    def _extract_text_from_pdf(self, file_path):
        """从 PDF 中提取文本"""
        text = ""
        try:
            reader = PdfReader(file_path)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        except Exception as e:
            print(f"❌ 读取 PDF 失败 {file_path}: {e}")
        return text

    def _chunk_text(self, text, chunk_size=500, overlap=50):
        """
        滑窗切块算法：保证上下文语义不会在截断处丢失
        """
        # 清洗掉多余的换行和空格
        text = text.replace('\n', ' ').replace('\r', '')
        words = text.split(' ') # 中文其实按字切更好，这里用简化的空格/字元混排策略
        
        # 简单暴力的按字符长度切分（加入 overlap 保证语义连贯）
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk = text[start:end]
            chunks.append(chunk.strip())
            start += (chunk_size - overlap) # 回退 overlap 个字符，实现滑窗
            
        return chunks

if __name__ == "__main__":
    print("🚀 息壤私有文献解析管道启动...")
    importer = LocalDocumentImporter()
    importer.process_all_eras()