# deploy/Dockerfile
# 息壤生产镜像
# 多阶段构建：依赖层 + 应用层分离，利用层缓存加速 CI

# ── 阶段 1：依赖构建 ─────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# 系统依赖（OCR、PDF 处理）
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-chi-sim \
    tesseract-ocr-chi-tra \
    libpoppler-cpp-dev \
    poppler-utils \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Python 依赖
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# ── 阶段 2：运行时镜像 ───────────────────────────────────────
FROM python:3.11-slim AS runtime

# 系统运行时库
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-chi-sim \
    tesseract-ocr-chi-tra \
    libpoppler-cpp-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 非 root 用户运行
RUN useradd -m -u 1000 xirang
WORKDIR /app

# 从 builder 阶段复制已安装的 Python 包
COPY --from=builder /root/.local /home/xirang/.local

# 复制项目代码（排除 .git、数据目录、虚拟环境）
COPY --chown=xirang:xirang . .

# 数据目录挂载点
RUN mkdir -p /data/raw_documents /data/knowledge /data/scenarios /data/chroma_db \
    && chown -R xirang:xirang /data

USER xirang

# 环境变量默认值（可被 docker-compose 或 k8s 覆盖）
ENV XIRANG_ENV=prod \
    XIRANG_HOST=0.0.0.0 \
    XIRANG_PORT=8000 \
    XIRANG_WORKERS=2 \
    XIRANG_DATA_DIR=/data \
    XIRANG_USE_REDIS=true \
    XIRANG_LOG_JSON=true \
    PATH="/home/xirang/.local/bin:$PATH"

EXPOSE 8000

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# 启动命令（Gunicorn + Uvicorn workers）
CMD ["gunicorn", "server:app", \
     "--worker-class", "uvicorn.workers.UvicornWorker", \
     "--workers", "2", \
     "--bind", "0.0.0.0:8000", \
     "--timeout", "120", \
     "--keepalive", "5", \
     "--access-logfile", "-", \
     "--error-logfile", "-"]
