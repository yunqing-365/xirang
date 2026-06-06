# infra/llm_client.py
"""
息壤 · 统一 LLM 客户端

所有引擎统一从此模块导入 `llm_chat` / `llm_chat_stream`，
内置：
  - 指数退避重试（最多3次）
  - 超时保护（非流式 45s，流式 90s）
  - 熔断器（连续5次失败后熔断60s）
  - DeepSeek / OpenAI 兼容

用法：
  from infra.llm_client import llm_chat, llm_chat_stream, aclient

  # 普通调用（返回 message 对象）
  msg = await llm_chat(messages=[...], temperature=0.7)
  text = msg.content

  # 流式调用（返回 AsyncGenerator[str]）
  async for token in llm_chat_stream(messages=[...]):
      print(token, end="")
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import AsyncGenerator, Optional

from openai import AsyncOpenAI, APITimeoutError, APIConnectionError, RateLimitError

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import get_settings

_settings = get_settings()
_log = logging.getLogger("xirang.llm_client")

# ── 全局单例客户端 ────────────────────────────────────────────
aclient = AsyncOpenAI(
    api_key=_settings.API_KEY,
    base_url=_settings.BASE_URL,
    timeout=90.0,   # 连接超时；具体调用超时在 call 层控制
)

# ── 熔断器状态 ────────────────────────────────────────────────
_circuit_failures   = 0
_circuit_open_until = 0.0
_FAILURE_THRESHOLD  = 5
_RECOVERY_SECONDS   = 60.0
_circuit_lock       = asyncio.Lock()

def _is_open() -> bool:
    if _circuit_open_until and time.time() < _circuit_open_until:
        return True
    return False

async def _record_failure():
    global _circuit_failures, _circuit_open_until
    async with _circuit_lock:
        _circuit_failures += 1
        if _circuit_failures >= _FAILURE_THRESHOLD:
            _circuit_open_until = time.time() + _RECOVERY_SECONDS
            _log.error(f"🔴 LLM 熔断触发，{_RECOVERY_SECONDS}s 后自动恢复")

async def _record_success():
    global _circuit_failures, _circuit_open_until
    async with _circuit_lock:
        _circuit_failures = 0
        _circuit_open_until = 0.0

# ── 可重试的异常类型 ──────────────────────────────────────────
_RETRYABLE = (APITimeoutError, APIConnectionError, RateLimitError)

# ── 普通调用（带重试 + 熔断）─────────────────────────────────
async def llm_chat(
    messages: list[dict],
    model: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 1500,
    timeout: float = 45.0,
    retries: int = 3,
    fallback_text: str = "【系统繁忙，请稍后重试】",
) -> str:
    """
    普通 LLM 调用，返回字符串内容。
    失败时返回 fallback_text，不抛出异常。
    """
    if _is_open():
        _log.warning("⚡ LLM 熔断中，返回兜底文本")
        return fallback_text

    model = model or _settings.MODEL_NAME
    last_exc: Optional[Exception] = None

    for attempt in range(retries + 1):
        try:
            resp = await asyncio.wait_for(
                aclient.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream=False,
                ),
                timeout=timeout,
            )
            await _record_success()
            return resp.choices[0].message.content or ""

        except _RETRYABLE as exc:
            last_exc = exc
            await _record_failure()
            if attempt < retries:
                delay = min(2.0 ** attempt, 16.0)
                _log.warning(f"⚠️ LLM 调用失败(第{attempt+1}次)，{delay:.0f}s 后重试: {exc}")
                await asyncio.sleep(delay)

        except asyncio.TimeoutError:
            last_exc = asyncio.TimeoutError(f"LLM 调用超时 ({timeout}s)")
            await _record_failure()
            if attempt < retries:
                _log.warning(f"⚠️ LLM 超时(第{attempt+1}次)，重试中…")

        except Exception as exc:
            # 非重试类错误（如参数错误）直接返回兜底
            _log.error(f"❌ LLM 调用错误(非重试): {exc}")
            return fallback_text

    _log.error(f"❌ LLM 调用最终失败({retries+1}次): {last_exc}")
    return fallback_text


# ── 流式调用（带超时 + 熔断，重试1次）───────────────────────
async def llm_chat_stream(
    messages: list[dict],
    model: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 2000,
    timeout: float = 90.0,
) -> AsyncGenerator[str, None]:
    """
    流式 LLM 调用，yield token 字符串。
    连接失败时 yield 一个错误提示字符串后结束。
    """
    if _is_open():
        _log.warning("⚡ LLM 熔断中，流式调用跳过")
        yield "【系统繁忙，请稍后重试】"
        return

    model = model or _settings.MODEL_NAME

    for attempt in range(2):  # 流式最多重试1次
        try:
            stream = await asyncio.wait_for(
                aclient.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream=True,
                ),
                timeout=15.0,  # 首包超时15s
            )
            await _record_success()
            async for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta
            return  # 正常结束

        except _RETRYABLE as exc:
            await _record_failure()
            if attempt == 0:
                _log.warning(f"⚠️ 流式LLM连接失败，重试: {exc}")
                await asyncio.sleep(2.0)
            else:
                _log.error(f"❌ 流式LLM最终失败: {exc}")
                yield "【网络波动，请重新推进时空】"

        except asyncio.TimeoutError:
            await _record_failure()
            _log.error("❌ 流式LLM首包超时")
            if attempt == 0:
                await asyncio.sleep(2.0)
            else:
                yield "【响应超时，请重新推进时空】"

        except Exception as exc:
            _log.error(f"❌ 流式LLM错误: {exc}")
            yield "【系统错误，请重新推进时空】"
            return


# ── 健康状态 ──────────────────────────────────────────────────
def get_llm_status() -> dict:
    return {
        "circuit_open": _is_open(),
        "failures": _circuit_failures,
        "open_until": _circuit_open_until,
        "base_url": _settings.BASE_URL,
        "model": _settings.MODEL_NAME,
    }
