
"""程序入口。

重构后 main.py 仅负责启动应用，
业务逻辑集中在 bot_core 包，便于分层迭代开发。
"""

from bot_core.app_runner import ApplicationRunner


def main() -> None:
    """启动应用。"""
    runner = ApplicationRunner()
    runner.run()


if __name__ == "__main__":
    main()
