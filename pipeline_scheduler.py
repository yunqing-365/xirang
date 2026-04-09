# pipeline_scheduler.py
from apscheduler.schedulers.blocking import BlockingScheduler
from tools.museum_api_scraper import MuseumDataAggregator
from tools.import_local_docs import LocalDocumentImporter
from tools.vlm_image_parser import VisualKnowledgeParser

def run_heritage_pipeline():
    print("⏳ 触发定期文化遗产采集任务...")
    
    # 1. 从 API 获取新数据存入 raw_documents
    aggregator = MuseumDataAggregator()
    aggregator.fetch_era_artifacts("song", "Song Dynasty calligraphy", limit=5)
    
    # 2. 将官方 txt 元数据切块入库
    importer = LocalDocumentImporter()
    importer.process_all_eras()
    
    # 3. 让 VLM 视觉雷达自动解析新下载的图片
    parser = VisualKnowledgeParser()
    parser.parse_all_images()
    
    print("✅ 全网文化遗产自运转收录完成！新知识已注入息壤底座。")

if __name__ == "__main__":
    scheduler = BlockingScheduler()
    # 设定每周日凌晨 2 点自动去博物馆拉取一次数据
    scheduler.add_job(run_heritage_pipeline, 'cron', day_of_week='sun', hour=2)
    print("🚀 息壤数字资产自动化收集守护进程已启动...")
    scheduler.start()