import time

import requests

from .auth_service import AuthService
from .config import AppConfig, ReconnectContext
from .runtime import get_active_client, get_active_socket, register_runner, unregister_runner, send_room_message
from .ws_client import GameWebSocketClient


class ApplicationRunner:
    """应用运行器（Facade）。

    对外只暴露 run()，内部协调：
    1. 登录与 token 缓存
    2. WebSocket 会话启动
    3. 异常与重连策略
    """

    def __init__(self, config: AppConfig | None = None) -> None:
        self.config = config or AppConfig()
        self.reconnect = ReconnectContext()
        self.auth_service = AuthService(self.config)
        register_runner(self)

    def _acquire_fv(self, session: requests.Session) -> dict:
        """根据当前模式决定是否复用缓存 token。"""
        if self.reconnect.fast_reconnect_once and self.reconnect.fv_cache:
            if self.reconnect.cookie_cache:
                session.cookies.update(self.reconnect.cookie_cache)
            print("平滑线路切换，使用已有 token...")
            self.reconnect.fast_reconnect_once = False
            return self.reconnect.fv_cache

        fv = self.auth_service.login(session)
        self.reconnect.fv_cache = fv
        self.reconnect.cookie_cache = session.cookies.get_dict()
        return fv

    def run(self) -> None:
        """主循环。

        该循环负责维持服务长期运行，并在连接断开后自动恢复。
        """
        while True:
            session = requests.Session()
            try:
                used_fast_reconnect = self.reconnect.fast_reconnect_once and bool(self.reconnect.fv_cache)
                fv = self._acquire_fv(session)
                ws_client = GameWebSocketClient(self.config, self.reconnect, session, fv)
                ws_client.run_forever()

                wait_seconds = 0 if used_fast_reconnect else self.config.reconnect_seconds
                if used_fast_reconnect:
                    print("平滑切换完成，立即重连...")
                else:
                    print(f"连接已断开，{self.config.reconnect_seconds}秒后自动重连...")
            except KeyboardInterrupt:
                print("收到中断信号，程序退出")
                break
            except Exception as exc:
                print(f"main_error={exc}")
                print(f"{self.config.reconnect_seconds}秒后重试登录与连接...")
                wait_seconds = self.config.reconnect_seconds
                self.reconnect.fast_reconnect_once = False
                self.reconnect.jump_reconnect_count = 0

            time.sleep(wait_seconds)

    def get_active_client(self):
        """获取当前活跃的 WebSocket 客户端。"""
        return get_active_client()

    def get_active_socket(self):
        """获取当前活跃的 WebSocket 连接实例。"""
        return get_active_socket()

    def send_room_message(self, message: str) -> bool:
        """通过当前活跃连接发送房间消息。"""
        return send_room_message(message)

    def close(self) -> None:
        """取消注册当前运行器。"""
        unregister_runner(self)
