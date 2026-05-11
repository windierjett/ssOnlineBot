import os
import threading
import time
from openai import OpenAI

from bot_core.config import get_secret


# 支持环境变量优先，其次读取统一配置文件。
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY") or get_secret("dashscope_api_key", "")
DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DASHSCOPE_MODEL = "qwen-plus"
MAX_HISTORY_ROUNDS = 10

_history_lock = threading.Lock()
_history_messages: list[dict[str, str]] = []

# Rate limiting / caching for model calls
RATE_LIMIT_SECONDS = 3
MAX_REPLY_CHARS = 50
_rate_lock = threading.Lock()
_last_request_time: float = 0.0
_last_reply: str = ""
_inflight: bool = False


def _trim_history(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    max_messages = MAX_HISTORY_ROUNDS * 2
    if len(messages) <= max_messages:
        return messages
    return messages[-max_messages:]


def clear_conversation_history() -> None:
    """清空会话历史。"""
    with _history_lock:
        _history_messages.clear()


def _build_messages(user_text: str) -> list[dict[str, str]]:
    with _history_lock:
        history = list(_trim_history(_history_messages))

    return [
        {"role": "system", "content": "You are a helpful assistant."},
        *history,
        {"role": "user", "content": user_text},
    ]


def ask_llm(message: str) -> str:
    """向大模型发送一段文本并返回回复内容。

    Args:
        message: 需要发送给模型的用户消息。

    Returns:
        模型返回的文本内容。
    """
    text = str(message or "").strip()
    if not text:
        raise ValueError("message 不能为空")

    if text == "终止对话":
        clear_conversation_history()
        return "已终止对话，历史消息已清空。"

    # Rate-limit: if a recent request (within RATE_LIMIT_SECONDS) was made,
    # discard the new request (do not wait) by returning empty string.
    global _last_request_time, _last_reply, _inflight
    now = time.time()
    with _rate_lock:
        if now - _last_request_time < RATE_LIMIT_SECONDS:
            # drop new request
            return ""
        if _inflight:
            # Another thread is calling the model; drop this request as well
            return ""
        # mark as in-flight for this caller
        _inflight = True

    if not DASHSCOPE_API_KEY:
        raise RuntimeError("未配置 DashScope API Key，请在 app_secrets.json 或 DASHSCOPE_API_KEY 中设置")

    client = OpenAI(api_key=DASHSCOPE_API_KEY, base_url=DASHSCOPE_BASE_URL)

    # Build messages with strict system instruction to limit reply length
    with _history_lock:
        history = list(_trim_history(_history_messages))

    messages = [
        {"role": "system", "content": f"You are a helpful assistant. Please answer in at most {MAX_REPLY_CHARS} characters."},
        *history,
        {"role": "user", "content": text},
    ]

    completion = client.chat.completions.create(
        model=DASHSCOPE_MODEL,
        messages=messages,
    )

    raw_reply = completion.choices[0].message.content or ""
    # Trim reply to MAX_REPLY_CHARS characters as a safety net
    reply = (raw_reply[:MAX_REPLY_CHARS]) if len(raw_reply) > MAX_REPLY_CHARS else raw_reply

    with _rate_lock:
        _last_reply = reply
        _last_request_time = time.time()
        _inflight = False

    # 更新会话历史（不在 rate-limit 直接返回缓存的路径中重复写入历史）
    with _history_lock:
        _history_messages.append({"role": "user", "content": text})
        _history_messages.append({"role": "assistant", "content": reply})
        trimmed = _trim_history(_history_messages)
        if len(trimmed) != len(_history_messages):
            _history_messages[:] = trimmed

    return reply


if __name__ == "__main__":
    print(ask_llm("你是谁？"))
