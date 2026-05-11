import json
import re
import time
from typing import Dict, Tuple
from urllib.parse import unquote
from urllib.parse import urlparse


def now_str() -> str:
    """返回统一格式的当前时间字符串。"""
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def parse_fv_params(html: str) -> Dict[str, str]:
    """从游戏页面 HTML 中提取 fv 参数。

    页面中的 token 经常是 URL 编码形式，这里统一解码，
    避免后续 WS 认证时出现 NotReg。
    """
    params: Dict[str, str] = {}
    pattern = r"fv\.([A-Za-z0-9_]+)\s*=\s*['\"]([^'\"]*)['\"]"
    for key, value in re.findall(pattern, html):
        params[key] = unquote(value)
    return params


def parse_invite_msg(msg: str) -> Tuple[str, str]:
    """从邀请文本中提取邀请者和房号。"""
    m = re.search(r"\[([^\]]+)\].*?【(\d+)房】", msg)
    if not m:
        return "", ""
    return m.group(1), m.group(2)


def parse_password_from_msg(msg: str) -> str:
    """从文本中提取邀请密码，兼容中英文冒号写法。"""
    m = re.search(r"密码\s*[:：]?\s*([A-Za-z0-9]+)", msg)
    if not m:
        return ""
    return m.group(1)


def extract_room_state_from_obj(cmd: str, obj: dict) -> Tuple[str, str]:
    """从协议对象中提取 room_id 与 line_id。"""
    room_id = ""
    line_id = ""

    if not isinstance(obj, dict):
        return room_id, line_id

    for key in ("RoomId", "roomId", "roomid", "ToRoomId"):
        value = obj.get(key)
        if value not in (None, ""):
            room_id = str(value)
            break

    if cmd == "jump":
        line_value = obj.get("l")
        if line_value not in (None, ""):
            line_id = str(line_value)

        room_value = obj.get("r")
        if room_value not in (None, ""):
            room_id = str(room_value)
    else:
        for key in ("LineId", "lineId", "lineid", "l"):
            value = obj.get(key)
            if value not in (None, ""):
                line_id = str(value)
                break

    return room_id, line_id


def parse_cmd_and_obj(text: str) -> Tuple[str, dict]:
    """解析服务端消息，兼容 cmd{...} 和标准 JSON 两种格式。"""
    m = re.match(r"^([A-Za-z0-9_]+)\{(.*)\}$", text)
    if m:
        cmd = m.group(1)
        obj_text = "{" + m.group(2) + "}"
        try:
            return cmd, json.loads(obj_text)
        except Exception:
            return cmd, {}

    if text.startswith("{") and text.endswith("}"):
        try:
            obj = json.loads(text)
            if isinstance(obj, dict):
                return str(obj.get("cmd") or obj.get("c") or ""), obj
        except Exception:
            pass

    return "", {}


def build_ws_url_by_line(default_ws_url: str, line_id: str) -> str:
    """根据线路 ID 构造目标 WS 地址。"""
    lid = str(line_id or "").strip()
    if not lid.isdigit():
        return default_ws_url

    port = _normalize_ws_port(str(9100 + int(lid)))
    return f"wss://kg{lid}.ss911.cn:{port}/"


def build_ws_url_by_host_port(default_ws_url: str, host: str, port: str) -> str:
    """根据服务端下发的 host/port 构造目标 WS 地址。"""
    host_text = str(host or "").strip()
    port_text = str(port or "").strip()
    if not host_text or not port_text.isdigit():
        return default_ws_url

    scheme = "wss"
    try:
        parsed = urlparse(default_ws_url)
        if parsed.scheme in {"ws", "wss"}:
            scheme = parsed.scheme
    except Exception:
        pass

    return f"{scheme}://{host_text}:{_normalize_ws_port(port_text)}/"


def _normalize_ws_port(port_text: str) -> str:
    """将线路接口返回的 91xx 端口映射为前端实际使用的 61xx 端口。"""
    text = str(port_text or "").strip()
    if len(text) == 4 and text.startswith("9"):
        return "6" + text[1:]
    return text
