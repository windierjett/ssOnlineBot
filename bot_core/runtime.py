"""运行时对象注册表。"""

from __future__ import annotations

from threading import Lock
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .app_runner import ApplicationRunner
    from .ws_client import GameWebSocketClient


_lock = Lock()
_active_runner: Optional["ApplicationRunner"] = None
_active_client: Optional["GameWebSocketClient"] = None


def register_runner(runner: "ApplicationRunner") -> None:
    """注册当前运行中的应用实例。"""
    global _active_runner
    with _lock:
        _active_runner = runner


def unregister_runner(runner: "ApplicationRunner") -> None:
    """取消注册当前运行中的应用实例。"""
    global _active_runner, _active_client
    with _lock:
        if _active_runner is runner:
            _active_runner = None
        if _active_client is not None and getattr(_active_client, "runner", None) is runner:
            _active_client = None


def set_active_client(client: Optional["GameWebSocketClient"]) -> None:
    """注册或清空当前活跃的 WebSocket 客户端。"""
    global _active_client
    with _lock:
        _active_client = client


def get_active_runner() -> Optional["ApplicationRunner"]:
    with _lock:
        return _active_runner


def get_active_client() -> Optional["GameWebSocketClient"]:
    with _lock:
        return _active_client


def get_active_socket():
    client = get_active_client()
    if client is None:
        return None
    return client.get_active_socket()


def send_room_message(message: str) -> bool:
    """通过当前活跃 socket 发送房间消息。"""
    client = get_active_client()
    if client is None:
        return False
    return client.send_room_message(message)
