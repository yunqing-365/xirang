# Makefile
# 息壤常用开发命令
# 用法：make <target>

.PHONY: help dev prod stop logs ingest status test lint clean backup

# 默认目标
help:
	@echo ""
	@echo "息壤 · 开发命令"
	@echo "──────────────────────────────────────"
	@echo "  make dev              启动开发服务器"
	@echo "  make prod             生产部署"
	@echo "  make stop             停止服务"
	@echo "  make logs             查看应用日志"
	@echo "  make status           查看知识库状态"
	@echo ""
	@echo "  make ingest ERA=song  摄入史料（指定朝代）"
	@echo "  make test             运行测试"
	@echo "  make lint             代码检查"
	@echo "  make clean            清理缓存"
	@echo "  make backup           备份数据"
	@echo ""

# ── 开发 ─────────────────────────────────────────────────────
dev:
	@bash deploy/start.sh dev

prod:
	@bash deploy/start.sh prod

stop:
	@bash deploy/start.sh stop

logs:
	@bash deploy/start.sh logs $(SERVICE)

status:
	@bash deploy/start.sh status

# ── 史料摄入 ──────────────────────────────────────────────────
ERA ?= song
ingest:
	@python3 -m ingestion.pipeline ingest --era $(ERA)

ingest-graph:
	@python3 -m ingestion.pipeline build-graph --era $(ERA)

# ── 代码质量 ──────────────────────────────────────────────────
lint:
	@echo "运行代码检查..."
	@python3 -m py_compile server.py config.py user_profile.py && echo "✓ 语法检查通过"
	@python3 -m py_compile infra/auth.py infra/cache.py infra/observability.py && echo "✓ infra 模块通过"
	@python3 -m py_compile ingestion/year_normalizer.py ingestion/text_parser.py ingestion/pipeline.py && echo "✓ ingestion 模块通过"
	@python3 -m py_compile narrative/historical_triggers.py narrative/offline_evolution.py && echo "✓ narrative 模块通过"

# ── 测试 ─────────────────────────────────────────────────────
test:
	@echo "运行自测..."
	@python3 -m ingestion.year_normalizer
	@python3 -m ingestion.text_parser
	@python3 -m narrative.historical_triggers
	@python3 -m infra.auth
	@python3 -m infra.observability
	@python3 -m infra.cache
	@echo "✅ 全部自测通过"

# ── 运维 ─────────────────────────────────────────────────────
backup:
	@bash deploy/start.sh backup

ssl:
	@bash deploy/start.sh ssl

# ── 清理 ─────────────────────────────────────────────────────
clean:
	@find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	@find . -name "*.pyc" -delete 2>/dev/null || true
	@find . -name "*.pyo" -delete 2>/dev/null || true
	@echo "✓ 缓存已清理"

# ── 依赖 ─────────────────────────────────────────────────────
install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements.txt
	pip install pytest ruff mypy
