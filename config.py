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


import logging
_log = logging.getLogger(__name__)


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
    DIALOGUE_CONTEXT_WINDOW: int = field(
        default_factory=lambda: _env_int("XIRANG_DIALOGUE_WINDOW", 8000)  # 保留最近 8000 字符
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

    # ── CORS ─────────────────────────────────────────────────
    CORS_ORIGINS: list[str] = field(
        default_factory=lambda: [
            o.strip()
            for o in _env("XIRANG_CORS_ORIGINS", "http://localhost:3000,http://localhost:8000").split(",")
            if o.strip()
        ]
    )

    # ── 短信 ─────────────────────────────────────────────────
    SMS_SANDBOX: bool = field(default_factory=lambda: _env_bool("XIRANG_SMS_SANDBOX", True))
    ALIYUN_ACCESS_KEY_ID:     str = field(default_factory=lambda: _env("ALIYUN_ACCESS_KEY_ID", ""))
    ALIYUN_ACCESS_KEY_SECRET: str = field(default_factory=lambda: _env("ALIYUN_ACCESS_KEY_SECRET", ""))
    SMS_SIGN_NAME:    str = field(default_factory=lambda: _env("XIRANG_SMS_SIGN_NAME", "息壤历史"))
    SMS_TEMPLATE_CODE: str = field(default_factory=lambda: _env("XIRANG_SMS_TEMPLATE_CODE", ""))

    # ── 支付 ─────────────────────────────────────────────────
    PAYMENT_SANDBOX:    bool = field(default_factory=lambda: _env_bool("XIRANG_PAYMENT_SANDBOX", True))
    WECHAT_APP_ID:      str  = field(default_factory=lambda: _env("WECHAT_APP_ID", ""))
    WECHAT_APP_SECRET:  str  = field(default_factory=lambda: _env("WECHAT_APP_SECRET", ""))
    WECHAT_MCH_ID:      str  = field(default_factory=lambda: _env("WECHAT_MCH_ID", ""))
    WECHAT_API_KEY:     str  = field(default_factory=lambda: _env("WECHAT_API_KEY", ""))
    ALIPAY_APP_ID:      str  = field(default_factory=lambda: _env("ALIPAY_APP_ID", ""))
    ALIPAY_PRIVATE_KEY: str  = field(default_factory=lambda: _env("ALIPAY_PRIVATE_KEY", ""))

    # ── 鉴权 & 多租户 ────────────────────────────────────────
    JWT_SECRET: str = field(default_factory=lambda: _env("XIRANG_JWT_SECRET", ""))
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
    DEBUG: bool = field(default_factory=lambda: _env_bool("XIRANG_DEBUG", False))

    def __post_init__(self):
        # JWT_SECRET 校验：生产必须显式配置，开发模式随机生成并警告
        if not self.JWT_SECRET:
            if self.ENV == "prod":
                raise RuntimeError(
                    "生产环境必须设置 XIRANG_JWT_SECRET 环境变量（至少32位随机字符串）\n"
                    "生成命令：python -c \"import secrets; print(secrets.token_hex(32))\""
                )
            _log.warning(
                "⚠️  XIRANG_JWT_SECRET 未设置，使用随机密钥"
                "（重启后所有用户需重新登录）——开发/测试模式可接受，生产请设置固定密钥"
            )
            self.JWT_SECRET = secrets.token_hex(32)


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
