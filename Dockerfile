# ─────────────────────────────────────────────
# 息壤 Dockerfile  ·  生产镜像
# 构建：docker build -t xirang:latest .
# 运行：docker compose up -d
# ─────────────────────────────────────────────

FROM python:3.12-slim-bookworm

# 系统依赖（CJK字体 + 文档处理 + 编译工具）
RUN apt-get update && apt-get install -y --no-install-recommends \
    # CJK 字体（PDF报告中文渲染）
    fonts-noto-cjk \
    fonts-wqy-zenhei \
    fonts-wqy-microhei \
    # PDF/图像处理
    libpango-1.0-0 \
    libpangocairo-1.0-0 \
    libcairo2 \
    libgdk-pixbuf2.0-0 \
    shared-mime-info \
    # tesseract OCR
    tesseract-ocr \
    tesseract-ocr-chi-sim \
    # 编译工具（某些 Python 包需要）
    gcc \
    g++ \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 安装 Python 依赖（分层缓存：先装依赖，再复制源码）
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制源码
COPY . .

# 准备字体（从 WQY TTC 提取简体中文 TTF 供 xhtml2pdf 使用）
RUN python3 scripts/extract_fonts.py 2>/dev/null || true

# 创建数据目录
RUN mkdir -p /app/data/raw_documents /app/data/scenarios \
             /app/data/knowledge /app/assets/fonts

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

# 启动
EXPOSE 8000
CMD ["gunicorn", "server:app", \
     "--worker-class", "uvicorn.workers.UvicornWorker", \
     "--workers", "2", \
     "--bind", "0.0.0.0:8000", \
     "--timeout", "120", \
     "--keep-alive", "5", \
     "--access-logfile", "-", \
     "--error-logfile", "-"]
