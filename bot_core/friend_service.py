import json
from typing import Dict, List

import requests

from .config import AppConfig


def normalize_online(value) -> bool:
    """将不同类型的在线字段统一归一化为布尔值。"""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value > 0
    if isinstance(value, str):
        text = value.strip().lower()
        return text in {"1", "true", "online", "on"}
    return False


def collect_friend_records(payload) -> List[Dict[str, object]]:
    """从多种协议结构提取好友记录。

    返回结构统一为：
    [{"user_id": str, "name": str, "online": bool}, ...]
    """
    records: List[Dict[str, object]] = []

    if isinstance(payload, dict):
        compressed = payload.get("compress")
        data = payload.get("data")
        if isinstance(compressed, list) and "data" in compressed and isinstance(data, list) and data:
            header = data[0]
            if isinstance(header, list):
                for row in data[1:]:
                    if not isinstance(row, list):
                        continue

                    row_obj = {}
                    for idx, key in enumerate(header):
                        if idx < len(row) and isinstance(key, str):
                            row_obj[key] = row[idx]

                    user_id = row_obj.get("userId")
                    if user_id is None:
                        continue

                    name = row_obj.get("userName") or row_obj.get("nickName") or str(user_id)
                    records.append(
                        {
                            "user_id": str(user_id),
                            "name": str(name),
                            "online": normalize_online(row_obj.get("online", 0)),
                        }
                    )

                if records:
                    return records

    def visit(node):
        if isinstance(node, list):
            for item in node:
                visit(item)
            return

        if not isinstance(node, dict):
            return

        user_id = node.get("UserId")
        if user_id is None:
            user_id = node.get("userId")

        if user_id is not None:
            online_raw = (
                node.get("Online")
                if "Online" in node
                else node.get("online", node.get("onLine", node.get("IsOnline", 0)))
            )
            name = (
                node.get("UserName")
                or node.get("userName")
                or node.get("nickName")
                or node.get("Name")
                or node.get("uuid")
                or str(user_id)
            )
            records.append(
                {
                    "user_id": str(user_id),
                    "name": str(name),
                    "online": normalize_online(online_raw),
                }
            )

        for value in node.values():
            if isinstance(value, (dict, list)):
                visit(value)

    visit(payload)
    return records


class FriendHttpService:
    """好友列表 HTTP 拉取服务。"""

    def __init__(self, config: AppConfig) -> None:
        self.config = config

    def fetch_friends_via_http(self, session: requests.Session, u_token: str = "") -> dict:
        hosts = ["https://t1.ss911.cn"]
        query_candidates = [
            {"p": 1, "ps": 200, "t": 0, "group": 0},
            {"p": 1, "ps": 200, "t": 3, "group": 0},
            {"p": 1, "ps": 200, "t": 4, "group": 0},
            {"p": 1, "ps": 200},
        ]

        last_error = ""
        first_payload = None

        for host in hosts:
            for params in query_candidates:
                url = host + self.config.friend_http_path
                req_params = dict(params)

                if u_token:
                    req_params["u"] = u_token
                else:
                    cookie_u = session.cookies.get("uservalues", "")
                    if cookie_u:
                        req_params["u"] = cookie_u

                try:
                    res = session.get(url, params=req_params, timeout=10)
                    print(f"friend_http_status={res.status_code} url={res.url}")
                    if res.status_code != 200:
                        continue

                    try:
                        data = res.json()
                    except Exception:
                        text = res.text.strip()
                        data = json.loads(text) if text.startswith("{") or text.startswith("[") else {}

                    if isinstance(data, dict):
                        keys = list(data.keys())[:8]
                        print(f"friend_http_keys={keys}")
                        if first_payload is None:
                            first_payload = data

                        rows = data.get("data")
                        row_len = len(rows) if isinstance(rows, list) else -1
                        print(f"friend_http_rows={row_len} params={params}")
                        if isinstance(rows, list) and rows:
                            return data
                        continue

                    if isinstance(data, list):
                        print(f"friend_http_rows={len(data)} params={params}")
                        if data:
                            return {"data": data}
                except Exception as exc:
                    last_error = str(exc)
                    print(f"friend_http_error host={host} params={params} err={exc}")

        if last_error:
            raise RuntimeError(f"好友HTTP请求失败: {last_error}")
        if first_payload is not None:
            return first_payload
        raise RuntimeError("好友HTTP请求失败: t1域名未返回可用数据")
