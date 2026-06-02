#!/bin/bash
# deploy/start.sh
# 息壤启动 & 运维脚本
# 用法：
#   ./deploy/start.sh dev          # 本地开发
#   ./deploy/start.sh prod         # 生产（Docker Compose）
#   ./deploy/start.sh stop         # 停止所有服务
#   ./deploy/start.sh logs         # 查看日志
#   ./deploy/start.sh ingest song  # 手动触发史料摄入
#   ./deploy/start.sh migrate      # 运行数据库迁移
#   ./deploy/start.sh backup       # 备份数据
#   ./deploy/start.sh status       # 查看运行状态

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
COMPOSE_FILE="$SCRIPT_DIR/docker-compose.yml"
ENV_FILE="$PROJECT_ROOT/.env"

# ── 颜色输出 ──────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; NC='\033[0m'

info()    { echo -e "${BLUE}ℹ  $*${NC}"; }
success() { echo -e "${GREEN}✓  $*${NC}"; }
warn()    { echo -e "${YELLOW}⚠  $*${NC}"; }
error()   { echo -e "${RED}✗  $*${NC}" >&2; exit 1; }

banner() {
    echo -e "${BLUE}"
    echo "╔═══════════════════════════════════════╗"
    echo "║        息 壤 · 人文时空底座            ║"
    echo "║        生成式叙事引擎 v2.1             ║"
    echo "╚═══════════════════════════════════════╝"
    echo -e "${NC}"
}

# ── 前置检查 ──────────────────────────────────────────────────
check_env() {
    if [ ! -f "$ENV_FILE" ]; then
        warn ".env 文件不存在，从模板创建..."
        cp "$PROJECT_ROOT/.env.example" "$ENV_FILE"
        error "请先编辑 .env 填入 XIRANG_API_KEY 等配置，再重新启动"
    fi
    # 检查必填项
    source "$ENV_FILE" 2>/dev/null || true
    if [ -z "${XIRANG_API_KEY:-}" ] || [ "$XIRANG_API_KEY" = "sk-your-api-key-here" ]; then
        error "请在 .env 中设置真实的 XIRANG_API_KEY"
    fi
    success "环境配置检查通过"
}

check_docker() {
    command -v docker >/dev/null 2>&1 || error "Docker 未安装，请先安装 Docker"
    command -v docker compose >/dev/null 2>&1 || \
        command -v docker-compose >/dev/null 2>&1 || \
        error "Docker Compose 未安装"
    success "Docker 环境检查通过"
}

docker_compose() {
    if command -v docker compose >/dev/null 2>&1; then
        docker compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" "$@"
    else
        docker-compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" "$@"
    fi
}

# ── 开发模式 ──────────────────────────────────────────────────
cmd_dev() {
    banner
    info "启动开发服务器（热重载）..."
    cd "$PROJECT_ROOT"

    # 创建数据目录
    mkdir -p data/raw_documents data/knowledge data/scenarios data/chroma_db

    # 检查依赖
    if ! python3 -c "import fastapi" 2>/dev/null; then
        info "安装 Python 依赖..."
        pip install -r requirements.txt --break-system-packages -q
    fi

    export XIRANG_ENV=dev
    export XIRANG_LOG_JSON=false
    export XIRANG_METRICS=true

    info "访问地址: http://localhost:8000"
    info "指标地址: http://localhost:8000/metrics"
    info "按 Ctrl+C 停止"
    echo ""

    python3 -m uvicorn server:app \
        --host 0.0.0.0 \
        --port 8000 \
        --reload \
        --reload-dir . \
        --log-level info
}

# ── 生产模式 ──────────────────────────────────────────────────
cmd_prod() {
    banner
    check_env
    check_docker
    info "启动生产服务栈..."

    # 构建镜像
    info "构建应用镜像..."
    docker_compose build --no-cache app

    # 启动服务
    info "启动所有服务..."
    docker_compose up -d

    # 等待健康检查
    info "等待服务就绪..."
    local retries=0
    until docker_compose exec -T app curl -sf http://localhost:8000/health >/dev/null 2>&1; do
        retries=$((retries + 1))
        if [ $retries -gt 20 ]; then
            error "服务启动超时，请检查日志: ./deploy/start.sh logs"
        fi
        sleep 3
        echo -n "."
    done
    echo ""

    success "息壤生产服务已启动！"
    echo ""
    echo "  📡 应用:    http://localhost"
    echo "  📊 Grafana: http://localhost:3000  (admin / \${GRAFANA_PASSWORD})"
    echo "  🔍 Prometheus: http://localhost:9090"
    echo ""
    info "查看日志: ./deploy/start.sh logs"
    info "停止服务: ./deploy/start.sh stop"
}

# ── 停止 ──────────────────────────────────────────────────────
cmd_stop() {
    info "停止所有服务..."
    docker_compose down
    success "服务已停止"
}

# ── 日志 ──────────────────────────────────────────────────────
cmd_logs() {
    local service="${1:-app}"
    docker_compose logs -f --tail=100 "$service"
}

# ── 史料摄入 ──────────────────────────────────────────────────
cmd_ingest() {
    local era="${1:-song}"
    local skip_graph="${2:-}"
    info "触发史料摄入流水线: 朝代=$era"

    cd "$PROJECT_ROOT"

    local args="--era $era"
    [ "$skip_graph" = "--skip-graph" ] && args="$args --skip-graph"

    if [ "${XIRANG_ENV:-dev}" = "prod" ]; then
        # 生产：在容器内运行
        docker_compose exec app python3 -m ingestion.pipeline ingest $args
    else
        # 开发：直接运行
        python3 -m ingestion.pipeline ingest $args
    fi
}

# ── 数据库迁移 ────────────────────────────────────────────────
cmd_migrate() {
    info "运行数据库迁移..."
    if [ "${XIRANG_ENV:-dev}" = "prod" ]; then
        docker_compose exec postgres psql -U xirang -d xirang \
            -f /docker-entrypoint-initdb.d/init.sql
    else
        warn "开发模式使用 JSON 文件存储，无需迁移"
    fi
}

# ── 数据备份 ──────────────────────────────────────────────────
cmd_backup() {
    local ts
    ts=$(date +%Y%m%d_%H%M%S)
    local backup_dir="$PROJECT_ROOT/backups/$ts"
    mkdir -p "$backup_dir"

    info "备份数据到 $backup_dir ..."

    # 备份知识库和会话数据
    if [ -d "$PROJECT_ROOT/data" ]; then
        cp -r "$PROJECT_ROOT/data/knowledge" "$backup_dir/" 2>/dev/null || true
        cp -r "$PROJECT_ROOT/data/scenarios" "$backup_dir/" 2>/dev/null || true
        info "知识库和会话数据已备份"
    fi

    # 备份 PostgreSQL（生产）
    if docker_compose ps postgres 2>/dev/null | grep -q "Up"; then
        docker_compose exec postgres pg_dump -U xirang xirang \
            > "$backup_dir/postgres_$ts.sql" 2>/dev/null && \
            info "PostgreSQL 数据已备份" || warn "PostgreSQL 备份跳过"
    fi

    success "备份完成: $backup_dir"
}

# ── 状态查看 ──────────────────────────────────────────────────
cmd_status() {
    banner
    echo "【知识库状态】"
    cd "$PROJECT_ROOT"
    python3 -m ingestion.pipeline status 2>/dev/null || warn "ingestion 模块未加载"
    echo ""

    if command -v docker >/dev/null 2>&1; then
        echo "【容器状态】"
        docker_compose ps 2>/dev/null || warn "Docker Compose 未运行"
    fi
}

# ── SSL 证书管理 ─────────────────────────────────────────────

# 生成自签证书（开发/局域网测试用）
cmd_ssl() {
    local cert_dir="$SCRIPT_DIR/certs"
    mkdir -p "$cert_dir"
    info "生成自签 SSL 证书（仅开发测试用）..."
    openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
        -keyout "$cert_dir/privkey.pem" \
        -out "$cert_dir/fullchain.pem" \
        -subj "/C=CN/ST=Beijing/L=Beijing/O=Xirang/CN=localhost" 2>/dev/null
    success "证书已生成: $cert_dir/"
    warn "生产环境请用: ./start.sh certbot <your-domain.com> <your@email.com>"
}

# Let's Encrypt 自动申请/续期（生产用）
# 用法：./deploy/start.sh certbot <域名> <邮箱>
# 示例：./deploy/start.sh certbot xirang.ai admin@xirang.ai
cmd_certbot() {
    local domain="${1:-}"
    local email="${2:-}"

    if [ -z "$domain" ] || [ -z "$email" ]; then
        error "用法: ./start.sh certbot <域名> <邮箱>\n示例: ./start.sh certbot xirang.ai admin@xirang.ai"
        exit 1
    fi

    info "申请 Let's Encrypt 证书: $domain"

    # 确保 certbot webroot 目录存在
    mkdir -p "$SCRIPT_DIR/certbot-webroot/.well-known/acme-challenge"

    # 启动 nginx（提供 ACME 验证路径）
    if ! docker_compose ps nginx 2>/dev/null | grep -q "running"; then
        info "启动 Nginx 以完成域名验证..."
        docker_compose up -d nginx
        sleep 3
    fi

    # 用 certbot docker 申请证书
    docker run --rm \
        -v "$SCRIPT_DIR/certs/letsencrypt:/etc/letsencrypt" \
        -v "$SCRIPT_DIR/certbot-webroot:/var/www/certbot" \
        certbot/certbot certonly \
        --webroot --webroot-path=/var/www/certbot \
        --email "$email" --agree-tos --no-eff-email \
        -d "$domain" -d "www.$domain" 2>&1

    if [ $? -eq 0 ]; then
        # 将证书软链接到 Nginx 期望的位置
        local cert_dir="$SCRIPT_DIR/certs"
        mkdir -p "$cert_dir"
        ln -sf "/etc/letsencrypt/live/$domain/fullchain.pem" "$cert_dir/fullchain.pem"
        ln -sf "/etc/letsencrypt/live/$domain/privkey.pem"   "$cert_dir/privkey.pem"
        success "证书申请成功！域名: $domain"
        info "重启 Nginx 使证书生效..."
        docker_compose restart nginx
        success "HTTPS 已启用：https://$domain"
    else
        error "证书申请失败，请检查：\n  1. 域名 DNS 已解析到本服务器 IP\n  2. 80 端口已开放\n  3. 邮箱地址正确"
        exit 1
    fi
}

# Let's Encrypt 自动续期（建议加入 crontab）
# crontab 示例：0 3 * * 0 /path/to/deploy/start.sh certbot-renew
cmd_certbot_renew() {
    info "检查并续期 Let's Encrypt 证书..."
    docker run --rm \
        -v "$SCRIPT_DIR/certs/letsencrypt:/etc/letsencrypt" \
        -v "$SCRIPT_DIR/certbot-webroot:/var/www/certbot" \
        certbot/certbot renew --quiet 2>&1

    if [ $? -eq 0 ]; then
        docker_compose restart nginx
        success "证书续期检查完成"
    else
        warn "证书续期失败或未到续期时间（Let's Encrypt 证书剩余30天内才会续期）"
    fi
}

# ── 入口 ──────────────────────────────────────────────────────
CMD="${1:-help}"
shift || true

case "$CMD" in
    dev)      cmd_dev ;;
    prod)     cmd_prod ;;
    stop)     cmd_stop ;;
    logs)     cmd_logs "$@" ;;
    ingest)   cmd_ingest "$@" ;;
    migrate)  cmd_migrate ;;
    backup)   cmd_backup ;;
    status)   cmd_status ;;
    ssl)            cmd_ssl ;;
    certbot)        cmd_certbot "$@" ;;
    certbot-renew)  cmd_certbot_renew ;;
    help|*)
        banner
        echo "用法: ./deploy/start.sh <命令> [参数]"
        echo ""
        echo "命令:"
        echo "  dev                   本地开发（热重载）"
        echo "  prod                  生产部署（Docker Compose）"
        echo "  stop                  停止所有服务"
        echo "  logs [service]        查看日志（默认 app）"
        echo "  ingest <era>          触发史料摄入（如 song/tang/ming）"
        echo "  migrate               运行数据库迁移"
        echo "  backup                备份数据"
        echo "  status                查看运行状态"
        echo "  ssl                   生成自签 SSL 证书（开发用）
  certbot <domain> <email>  申请 Let's Encrypt 证书（生产用）
  certbot-renew         续期 Let's Encrypt 证书（建议加入 crontab）"
        echo ""
        echo "快速开始:"
        echo "  cp .env.example .env && vim .env"
        echo "  ./deploy/start.sh dev"
        ;;
esac
