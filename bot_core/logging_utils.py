import os
import re
from typing import Tuple

from .config import AppConfig
from .parsers import now_str


def append_log_line(file_path: str, line: str) -> None:
    """向文件追加一行日志。"""
    with open(file_path, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def _extract_message_body(line: str) -> str:
    """从日志行中提取消息正文。"""
    text = str(line or "").strip()
    if not text:
        return ""

    parts = text.rsplit(":", 1)
    if len(parts) == 2:
        return parts[1].strip()

    return text


def _normalize_message(text: str) -> str:
    """用于重复与垃圾检测的归一化文本。"""
    value = str(text or "").strip().lower()
    if not value:
        return ""

    value = re.sub(r"\s+", "", value)
    value = re.sub(r"[\W_]+", "", value, flags=re.UNICODE)
    return value


def _is_garbled_or_spam(text: str) -> bool:
    """判断是否为乱码、刷屏或无意义输入。"""
    value = str(text or "").strip()
    if not value:
        return True

    if value.isdigit():
        return True

    normalized = _normalize_message(value)
    if not normalized:
        return True

    # 仅由少量重复字符组成的刷屏文本，例如 "哈哈哈哈哈"、"11111"、"。。。。。"
    if len(set(normalized)) <= 2 and len(normalized) >= 5:
        return True

    # 明显乱码：大部分字符不可读或无字母/数字/中文
    readable = re.findall(r"[A-Za-z0-9\u4e00-\u9fff]", value)
    if not readable:
        return True

    # 过短且缺少有效字符的噪声，通常没有存储价值
    if len(normalized) <= 1:
        return True

    return False


def _contains_abusive_language(text: str) -> bool:
    """判断是否包含常见辱骂或垃圾话。"""
    value = str(text or "").lower()
    if not value:
        return False

    blocked_terms = (
        "傻逼",
        "sb",
        "垃圾",
        "滚",
        "去死",
        "废物",
        "脑残",
        "傻叉",
        "狗东西",
    )
    return any(term in value for term in blocked_terms)


def _should_store_room_say_message(msg_text: str, previous_msg_text: str = "") -> bool:
    """判断是否应写入房间聊天日志。"""
    current = str(msg_text or "").strip()
    if not current:
        return False

    if _is_garbled_or_spam(current):
        return False

    if _contains_abusive_language(current):
        return False

    current_norm = _normalize_message(current)
    previous_norm = _normalize_message(previous_msg_text)
    if current_norm and previous_norm and current_norm == previous_norm:
        return False

    return True


def sanitize_filename(name: str) -> str:
    """将昵称转换成安全文件名。"""
    cleaned = re.sub(r'[\\/:*?"<>|]+', "_", str(name or "").strip())
    cleaned = re.sub(r"\s+", "_", cleaned)
    return cleaned or "unknown"


def parse_room_say_payload(obj: dict) -> Tuple[str, str,str]:
    """提取 RoomSay 的发言昵称和内容。"""
    if not isinstance(obj, dict):
        return "", "",""

    user_arr = obj.get("u")
    sender_name = ""
    if isinstance(user_arr, list) and len(user_arr) > 2:
        sender_name = str(user_arr[2] or "")

    msg_text = str(obj.get("m", "") or "")
    position = str(obj.get("s", "") or "")
    return sender_name, msg_text, position



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

    def _get_last_room_say_message(self) -> str:
        """读取主日志的最后一条消息正文，用于重复过滤。"""
        log_file = self.config.room_say_log_file
        try:
            with open(log_file, "r", encoding="utf-8") as file_handle:
                last_line = ""
                for line in file_handle:
                    stripped = line.rstrip("\r\n")
                    if stripped:
                        last_line = stripped
                return _extract_message_body(last_line)
        except FileNotFoundError:
            return ""
        except OSError:
            return ""

    def log_room_say_message(self, obj: dict, room_id: str = "") -> None:
        if not self.config.room_say_log_enabled:
            return

        sender_name, msg_text,position = parse_room_say_payload(obj)
        if not sender_name:
            return

        os.makedirs(self.config.room_say_log_dir, exist_ok=True)
        log_file = self._get_room_say_log_file(sender_name)

        previous_message = self._get_last_room_say_message()
        if not _should_store_room_say_message(msg_text, previous_message):
            return

        # 构建日志行，包含位置信息
        if room_id:
            if position:
                full_line = f"[{now_str()}] [{room_id}] [位置:{position}] {sender_name}: {msg_text}"
            else:
                full_line = f"[{now_str()}] [{room_id}] {sender_name}: {msg_text}"
        else:
            if position:
                full_line = f"[{now_str()}] [位置:{position}] {sender_name}: {msg_text}"
            else:
                full_line = f"[{now_str()}] {sender_name}: {msg_text}"

        append_log_line(log_file, full_line)
        append_log_line(self.config.room_say_log_file, full_line)


def make_self_room_status_line(room_id: str, line_id: str, state_text: str) -> str:
    """生成统一格式的房间状态日志行。"""
    room_text = room_id if room_id else "未知"
    line_text = line_id if line_id else "未知"
    return f"[{now_str()}] 房间={room_text} 线路={line_text} 状态={state_text}"
