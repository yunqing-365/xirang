"""
息壤生产配置模块
所有配置项均可通过环境变量覆盖，支持 .env 文件（python-dotenv 可选）。

部署时设置的关键环境变量：
  XIRANG_API_KEY        LLM API 密钥
  XIRANG_BASE_URL       LLM API 地址
  XIRANG_MODEL          模型名称
  XIRANG_DATA_DIR       数据目录
  XIRANG_REDIS_URL      Redis 连接串（生产必须）
  XIRANG_DB_URL         PostgreSQL 连接串（生产必须）
  XIRANG_JWT_SECRET     JWT 签名密钥（生产必须随机生成）
  XIRANG_ENV            dev / staging / prod
"""
import os
import secrets
from dataclasses import dataclass, field
from pathlib import Path

# 尝试加载 .env 文件（可选依赖）
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env", override=False)
except ImportError:
    pass


def _env(key: str, default: str = "") -> str:
    return os.environ.get(key, default)

def _env_int(key: str, default: int) -> int:
    try:
        return int(os.environ.get(key, default))
    except (ValueError, TypeError):
        return default

def _env_bool(key: str, default: bool = False) -> bool:
    v = os.environ.get(key, "")
    if not v:
        return default
    return v.lower() in ("1", "true", "yes", "on")


@dataclass
class Settings:
    # ── LLM ─────────────────────────────────────────────────
    API_KEY:    str = field(default_factory=lambda: _env("XIRANG_API_KEY", "sk-placeholder"))
    BASE_URL:   str = field(default_factory=lambda: _env("XIRANG_BASE_URL", "https://api.openai.com/v1"))
    MODEL_NAME: str = field(default_factory=lambda: _env("XIRANG_MODEL", "gpt-4o-mini"))

    # ── 数据目录 ─────────────────────────────────────────────
    DATA_DIR: str = field(default_factory=lambda: _env(
        "XIRANG_DATA_DIR",
        str(Path(__file__).parent / "data")
    ))

    # ── 环境 ────────────────────────────────────────────────
    ENV: str = field(default_factory=lambda: _env("XIRANG_ENV", "dev"))

    @property
    def is_production(self) -> bool:
        return self.ENV == "prod"

    @property
    def is_dev(self) -> bool:
        return self.ENV == "dev"

    # ── 会话管理 ─────────────────────────────────────────────
    SESSION_TTL_SECONDS: int = field(
        default_factory=lambda: _env_int("XIRANG_SESSION_TTL", 7200)  # 2h
    )
    MAX_ACTIVE_SESSIONS: int = field(
        default_factory=lambda: _env_int("XIRANG_MAX_SESSIONS", 200)
    )

    # ── Redis（会话持久化 & 限流）────────────────────────────
    REDIS_URL: str = field(
        default_factory=lambda: _env("XIRANG_REDIS_URL", "redis://localhost:6379/0")
    )
    USE_REDIS: bool = field(
        default_factory=lambda: _env_bool("XIRANG_USE_REDIS", False)
    )
    REDIS_SESSION_PREFIX: str = "xirang:session:"
    REDIS_RATELIMIT_PREFIX: str = "xirang:rl:"

    # ── PostgreSQL（用户档案持久化）──────────────────────────
    DB_URL: str = field(
        default_factory=lambda: _env(
            "XIRANG_DB_URL",
            "postgresql://xirang:xirang@localhost:5432/xirang"
        )
    )
    USE_POSTGRES: bool = field(
        default_factory=lambda: _env_bool("XIRANG_USE_POSTGRES", False)
    )

    # ── 鉴权 & 多租户 ────────────────────────────────────────
    JWT_SECRET: str = field(
        default_factory=lambda: _env("XIRANG_JWT_SECRET", secrets.token_hex(32))
    )
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRE_MINUTES: int = field(
        default_factory=lambda: _env_int("XIRANG_JWT_EXPIRE", 60 * 24 * 7)  # 7天
    )
    # 内置 API Keys（逗号分隔）；生产用 DB 存储
    STATIC_API_KEYS: list[str] = field(
        default_factory=lambda: [
            k.strip() for k in _env("XIRANG_API_KEYS", "").split(",") if k.strip()
        ]
    )
    AUTH_ENABLED: bool = field(
        default_factory=lambda: _env_bool("XIRANG_AUTH_ENABLED", False)
    )

    # ── 限流 ─────────────────────────────────────────────────
    RATE_LIMIT_RPM: int = field(
        default_factory=lambda: _env_int("XIRANG_RATE_LIMIT_RPM", 60)
    )
    RATE_LIMIT_ENABLED: bool = field(
        default_factory=lambda: _env_bool("XIRANG_RATE_LIMIT", False)
    )

    # ── 可观测性 ─────────────────────────────────────────────
    METRICS_ENABLED: bool = field(
        default_factory=lambda: _env_bool("XIRANG_METRICS", True)
    )
    LOG_LEVEL: str = field(
        default_factory=lambda: _env("XIRANG_LOG_LEVEL", "INFO")
    )
    LOG_JSON: bool = field(
        default_factory=lambda: _env_bool("XIRANG_LOG_JSON", False)
    )

    # ── Gunicorn / Uvicorn ───────────────────────────────────
    HOST: str = field(default_factory=lambda: _env("XIRANG_HOST", "0.0.0.0"))
    PORT: int = field(default_factory=lambda: _env_int("XIRANG_PORT", 8000))
    WORKERS: int = field(default_factory=lambda: _env_int("XIRANG_WORKERS", 1))


_instance: Settings | None = None

def get_settings() -> Settings:
    global _instance
    if _instance is None:
        _instance = Settings()
    return _instance

# 旧式兼容
_s = get_settings()
API_KEY    = _s.API_KEY
BASE_URL   = _s.BASE_URL
MODEL_NAME = _s.MODEL_NAME
DATA_DIR   = _s.DATA_DIR
