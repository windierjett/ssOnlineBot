"""管理模式入口。

启动后会运行机器人主循环，同时提供 HTTP 管理接口。
"""

from bot_core.admin_api_runner import AdminApiRunner


def main() -> None:
    app = AdminApiRunner()
    app.run()


if __name__ == "__main__":
    main()
