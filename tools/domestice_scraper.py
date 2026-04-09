# tools/domestic_scraper.py
import os
import requests
import time
import json
from bs4 import BeautifulSoup
import urllib.parse
import sys

# 引入统一配置
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_DIR

class DomesticHeritageAggregator:
    def __init__(self):
        self.knowledge_dir = os.path.join(DATA_DIR, "knowledge")
        
    def _ensure_dir(self, era_name):
        era_path = os.path.join(self.knowledge_dir, era_name)
        os.makedirs(era_path, exist_ok=True)
        return era_path

    # ==========================================
    # 数据源 1：CText (中国哲学书电子化计划 API)
    # 提取古籍、正史中的第一手文言文史料
    # ==========================================
    def fetch_ancient_texts(self, era_name, keyword, limit=3):
        print(f"\n📜 [古籍检索] 正在从 CText 寻找关于【{keyword}】的古代文献...")
        save_dir = self._ensure_dir(era_name)
        
        # CText 开放搜索 API
        url = f"https://api.ctext.org/searchtext?pattern={urllib.parse.quote(keyword)}&remap=gb"
        
        try:
            res = requests.get(url, timeout=10).json()
            results = res.get("results", [])
            
            if not results:
                print("   ❌ 未能找到相关古籍文献。")
                return

            # 将多条结果合并为一个 TXT 知识块
            out_file = os.path.join(save_dir, f"DOMESTIC_古籍_{keyword}.txt")
            with open(out_file, 'w', encoding='utf-8') as f:
                f.write(f"【本土典籍文献：{keyword}相关检索】\n\n")
                count = 0
                for item in results:
                    if count >= limit: break
                    title = item.get('title', '佚名典籍')
                    text_match = item.get('text', '').replace('<b>', '').replace('</b>', '')
                    
                    f.write(f"《{title}》记载：\n{text_match}\n\n")
                    print(f"   -> 成功摘录《{title}》中的相关史料")
                    count += 1
                    
        except Exception as e:
            print(f"   ❌ CText 接口请求失败: {e}")

    # ==========================================
    # 数据源 2：开源古诗词 API
    # 提取苏轼等文人的诗词、翻译与赏析
    # ==========================================
    def fetch_poetry(self, era_name, author, keyword=""):
        print(f"\n🏮 [诗词赋库] 正在检索【{author}】的诗词意境...")
        save_dir = self._ensure_dir(era_name)
        
        # 使用一个稳定的开源古诗词检索接口示例 (具体可用接口可能变动，此为结构演示)
        url = f"https://v1.alapi.cn/api/shici?author={urllib.parse.quote(author)}"
        
        try:
            res = requests.get(url, timeout=10).json()
            if res.get("code") == 200:
                data = res.get("data", {})
                title = data.get("title", "无题")
                content = data.get("content", "")
                
                print(f"   -> 成功提取诗词：《{title}》")
                
                out_file = os.path.join(save_dir, f"DOMESTIC_诗词_{title}.txt")
                with open(out_file, 'w', encoding='utf-8') as f:
                    f.write(f"【本土文学资产：{author}的诗词】\n")
                    f.write(f"标题：{title}\n")
                    f.write(f"作者：{author}\n")
                    f.write(f"正文：\n{content}\n")
            else:
                print("   ❌ 未能命中诗词接口。")
                
        except Exception as e:
            print(f"   ❌ 诗词接口请求失败: {e}")

    # ==========================================
    # 数据源 3：定向百科实体抽取 (BeautifulSoup 爬虫)
    # 用于抓取生平事迹、物品介绍，喂给 GraphRAG
    # ==========================================
    def fetch_baike_entity(self, era_name, entity_name):
        print(f"\n🔍 [百科抽取] 正在扫描实体【{entity_name}】的背景知识...")
        save_dir = self._ensure_dir(era_name)
        
        url = f"https://baike.baidu.com/item/{urllib.parse.quote(entity_name)}"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        
        try:
            res = requests.get(url, headers=headers, timeout=10)
            res.encoding = 'utf-8' # 解决中文乱码
            soup = BeautifulSoup(res.text, 'html.parser')
            
            # 提取百度百科的摘要部分 (通常是 class="lemma-summary" 或类似)
            summary_div = soup.find('div', class_='lemma-summary')
            if summary_div:
                summary_text = summary_div.get_text(strip=True)
                
                print(f"   -> 成功提取实体摘要 ({len(summary_text)}字)")
                
                out_file = os.path.join(save_dir, f"DOMESTIC_百科_{entity_name}.txt")
                with open(out_file, 'w', encoding='utf-8') as f:
                    f.write(f"【本土实体知识百科：{entity_name}】\n\n")
                    f.write(summary_text)
            else:
                print(f"   ❌ 页面未找到【{entity_name}】的摘要内容。")
                
        except Exception as e:
            print(f"   ❌ 百科网页解析失败: {e}")

if __name__ == "__main__":
    scraper = DomesticHeritageAggregator()
    
    # 模拟为“息壤”宋代剧本补充弹药
    scraper.fetch_ancient_texts("song", "苏轼 黄州", limit=5)
    scraper.fetch_poetry("song", "苏轼")
    scraper.fetch_baike_entity("song", "临皋亭")
    scraper.fetch_baike_entity("song", "东坡肉")
    
    print("\n✅ 国内本土数据源汲取完成！请运行 RAG 索引脚本将它们向量化。")