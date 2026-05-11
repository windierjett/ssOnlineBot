import os
import re
from typing import Tuple

from .config import AppConfig
from .parsers import now_str


def append_log_line(file_path: str, line: str) -> None:
    """向文件追加一行日志。"""
    with open(file_path, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def sanitize_filename(name: str) -> str:
    """将昵称转换成安全文件名。"""
    cleaned = re.sub(r'[\\/:*?"<>|]+', "_", str(name or "").strip())
    cleaned = re.sub(r"\s+", "_", cleaned)
    return cleaned or "unknown"


def parse_room_say_payload(obj: dict) -> Tuple[str, str]:
    """提取 RoomSay 的发言昵称和内容。"""
    if not isinstance(obj, dict):
        return "", ""

    user_arr = obj.get("u")
    sender_name = ""
    if isinstance(user_arr, list) and len(user_arr) > 2:
        sender_name = str(user_arr[2] or "")

    msg_text = str(obj.get("m", "") or "")
    return sender_name, msg_text


class RoomSayLogger:
    """房间发言日志服务。

    设计要点：
    1. 单一职责：只处理 RoomSay 日志，不混入协议控制逻辑。
    2. 可扩展：后续可替换为数据库落盘，不影响上层调用点。
    """

    def __init__(self, config: AppConfig) -> None:
        self.config = config

    def _get_room_say_log_file(self, sender_name: str) -> str:
        mapped = self.config.room_say_name_to_file.get(sender_name)
        file_name = mapped if mapped else f"room_say_{sanitize_filename(sender_name)}.txt"
        return os.path.join(self.config.room_say_log_dir, file_name)

    def log_room_say_message(self, obj: dict, room_id: str = "") -> None:
        if not self.config.room_say_log_enabled:
            return

        sender_name, msg_text = parse_room_say_payload(obj)
        if not sender_name:
            return

        os.makedirs(self.config.room_say_log_dir, exist_ok=True)
        log_file = self._get_room_say_log_file(sender_name)

        if room_id:
            full_line = f"[{now_str()}] [{room_id}] {sender_name}: {msg_text}"
        else:
            full_line = f"[{now_str()}] {sender_name}: {msg_text}"

        append_log_line(log_file, full_line)
        append_log_line(self.config.room_say_log_file, full_line)


def make_self_room_status_line(room_id: str, line_id: str, state_text: str) -> str:
    """生成统一格式的房间状态日志行。"""
    room_text = room_id if room_id else "未知"
    line_text = line_id if line_id else "未知"
    return f"[{now_str()}] 房间={room_text} 线路={line_text} 状态={state_text}"
