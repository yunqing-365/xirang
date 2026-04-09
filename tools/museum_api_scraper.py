# tools/museum_api_scraper.py
import os
import sys
import requests
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_DIR

class MuseumDataAggregator:
    def __init__(self):
        self.raw_dir = os.path.join(DATA_DIR, "raw_documents")
        # 大都会博物馆免费开放 API
        self.search_url = "https://collectionapi.metmuseum.org/public/collection/v1/search"
        self.object_url = "https://collectionapi.metmuseum.org/public/collection/v1/objects/"

    def fetch_era_artifacts(self, era_name, search_keyword, limit=3):
        print(f"\n🏛️ 正在启动全球数字遗产采集协议...")
        print(f"🔍 目标: [{era_name}] 时代 | 关键词: [{search_keyword}]")
        
        era_path = os.path.join(self.raw_dir, era_name)
        os.makedirs(era_path, exist_ok=True)

        # 1. 搜索具有图片的文物 ID
        params = {
            "q": search_keyword,
            "hasImages": "true"
        }
        
        try:
            response = requests.get(self.search_url, params=params)
            response.raise_for_status()
            data = response.json()
            
            object_ids = data.get("objectIDs", [])
            if not object_ids:
                print("❌ 未在博物馆数据库中找到相关文物。")
                return
                
            print(f"✅ 检索到 {len(object_ids)} 件相关文物，准备下载前 {limit} 件...")
            
            # 2. 获取每件文物的详细信息和图片
            count = 0
            for obj_id in object_ids:
                if count >= limit:
                    break
                    
                self._download_artifact(obj_id, era_path)
                count += 1
                time.sleep(1) # 礼貌性延时，防止被 API 封禁
                
        except Exception as e:
            print(f"❌ 采集请求失败: {e}")

    def _download_artifact(self, obj_id, save_dir):
        try:
            res = requests.get(self.object_url + str(obj_id))
            res.raise_for_status()
            obj_data = res.json()
            
            title = obj_data.get("title", f"未命名文物_{obj_id}")
            # 清理文件名中的非法字符
            safe_title = "".join(c for c in title if c.isalnum() or c in (' ', '_')).rstrip()
            
            image_url = obj_data.get("primaryImageSmall") or obj_data.get("primaryImage")
            if not image_url:
                return

            print(f"   -> 正在入库文物: {title}")
            
            # 1. 下载图片到 raw_documents
            img_ext = image_url.split('.')[-1]
            img_filename = f"{safe_title}.{img_ext}"
            img_path = os.path.join(save_dir, img_filename)
            
            if not os.path.exists(img_path):
                img_res = requests.get(image_url)
                with open(img_path, 'wb') as f:
                    f.write(img_res.content)
            
            # 2. 提取官方英文描述（如果有），保存为 txt 辅助资料
            metadata_str = f"【官方数字遗产元数据】\n名称: {title}\n"
            metadata_str += f"朝代/时期: {obj_data.get('dynasty', '未知')}\n"
            metadata_str += f"材质: {obj_data.get('medium', '未知')}\n"
            metadata_str += f"博物馆分类: {obj_data.get('department', '未知')}\n"
            
            txt_path = os.path.join(save_dir, f"{safe_title}.txt")
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(metadata_str)
                
        except Exception as e:
            print(f"   ❌ 获取文物 {obj_id} 失败: {e}")

if __name__ == "__main__":
    aggregator = MuseumDataAggregator()
    # 搜索宋代相关的文物，下载 2 件作为 Demo 测试
    aggregator.fetch_era_artifacts("song", "Song Dynasty", limit=2)