import json
import re
from typing import Dict

import requests

from .config import AppConfig
from .parsers import parse_fv_params


def build_cookie_header(session: requests.Session) -> str:
    """将 Session Cookie 展平为 WS Header 需要的格式。"""
    return "; ".join([f"{k}={v}" for k, v in session.cookies.get_dict().items()])


class AuthService:
    """认证服务。

    统一封装登录流程，避免 main 流程直接关心 HTTP 细节。
    """

    def __init__(self, config: AppConfig) -> None:
        self.config = config

    def login(self, session: requests.Session) -> Dict[str, str]:
        session.headers.update(
            {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/147.0.0.0 Safari/537.36 Edg/147.0.0.0"
            }
        )

        session.get(self.config.home_url, timeout=10)
        data = {
            "txt_Name": self.config.login_user,
            "txt_Pwd": self.config.login_pwd,
            "url": "/",
        }

        res = session.post(self.config.login_url, data=data, timeout=10)
        body = res.text.strip()

        print(f"login_http_status={res.status_code}, url={res.url}")
        msg_match = re.search(r'var\\s+msg\\s*=\\s*"([^"]*)"', body)
        if msg_match:
            login_msg = msg_match.group(1)
            print(f"login_msg={login_msg}")
            if login_msg != "OK":
                raise RuntimeError(f"官网登录失败: {login_msg}")

        game_page = session.get(self.config.game_url, timeout=10)
        print(f"game_page_status={game_page.status_code}, url={game_page.url}")

        fv = parse_fv_params(game_page.text)
        print(
            "game_fv="
            + json.dumps(
                {
                    "u": fv.get("u", ""),
                    "i": fv.get("i", ""),
                    "IP": fv.get("IP", ""),
                    "UUID": fv.get("UUID", ""),
                },
                ensure_ascii=False,
                separators=(",", ":"),
            )
        )
        return fv
