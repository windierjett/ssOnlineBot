import json
import os
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Tuple
from urllib.parse import unquote
from urllib.parse import urljoin, urlparse
import socket
import traceback

import requests
import websocket
from llm_client import ask_llm_chunks
from map import get_weather

from .auth_service import build_cookie_header
from .config import AppConfig, ReconnectContext, load_app_secrets, save_app_secrets
from .friend_service import FriendHttpService, collect_friend_records
from .logging_utils import (
    RoomSayLogger,
    append_log_line,
    make_self_room_status_line,
    parse_room_say_payload,
)
from .parsers import (
    build_ws_url_by_host_port,
    build_ws_url_by_line,
    extract_room_state_from_obj,
    now_str,
    parse_cmd_and_obj,
)
from .runtime import set_active_client


@dataclass
class WsState:
    """WebSocket 会话运行态。"""

    auth_idx: int = 0
    login_ok: bool = False
    join_hall_sent: bool = False
    success: bool = False
    retried_on_notreg: bool = False
    joined_hall: bool = False
    room_id: str = ""
    create_room_sent: bool = False

    friend_status: Dict[str, Dict[str, object]] = field(default_factory=dict)
    friend_seen_once: bool = False
    friend_poll_round: int = 0
    friend_resp_count: int = 0
    friend_empty_logged: bool = False

    current_room_id: str = ""
    current_line_id: str = ""
    current_room_site: str = ""
    last_room_status_log: str = ""

    bootstrap_room_id: str = ""
    bootstrap_room_pwd: str = ""
    rejoin_watch_version: int = 0
    connected_ws_line_id: str = ""
    llm_recent_chat_sent_at: float = 0.0

    # 新增：维护用户名到位置的映射 {username: position}
    user_position_map: Dict[str, str] = field(default_factory=dict)
    last_position_update_time: float = 0.0
    # 位置映射表的版本号，每次更新时递增
    position_map_version: int = 0
    # LLM 上次使用的位置映射表版本号
    llm_last_position_map_version: int = 0

class GameWebSocketClient:
    """WebSocket 客户端。

    使用策略分发表（cmd -> handler）处理服务端命令，
    相比巨型 if/elif 更利于迭代维护。
    """

    def __init__(
        self,
        config: AppConfig,
        reconnect: ReconnectContext,
        session: requests.Session,
        fv: Dict[str, str],
    ) -> None:
        self.config = config
        self.reconnect = reconnect
        self.session = session
        self.fv = fv

        self.room_say_logger = RoomSayLogger(config)
        self.friend_http_service = FriendHttpService(config)

        self.state = WsState(
            bootstrap_room_id=reconnect.next_join_room_id,
            bootstrap_room_pwd=reconnect.next_join_room_pwd,
        )

        self.stop_keepalive = threading.Event()
        self.send_lock = threading.Lock()
        self.friend_http_lock = threading.Lock()
        self._active_ws = None
        self._active_ws_lock = threading.Lock()

        self.auth_candidates = self._build_auth_candidates()

        self.cmd_handlers = {
            "Login": self._handle_login,
            "RoomSay": self._handle_room_say,
            "JoinHall": self._handle_join_hall,
            "AlertMsg": self._handle_alert_msg,
            "jump": self._handle_jump,
            "JoinRoom": self._handle_join_room,
            "Invite": self._handle_invite,
            "InviteGame": self._handle_invite,
        }

    def _build_auth_candidates(self) -> List[Dict[str, object]]:
        ip = self.fv.get("IP", "")
        token_i = self.fv.get("i", "")
        token_u = self.fv.get("u", "")
        token_i_raw = unquote(token_i)
        token_u_raw = unquote(token_u)
        device = self.fv.get("device", self.config.ws_device)
        p_token = self.fv.get("p", ip)

        def push_auth(candidates: List[Dict[str, object]], payload: Dict[str, object]) -> None:
            if payload not in candidates:
                candidates.append(payload)

        auth_candidates: List[Dict[str, object]] = []

        for tk in [token_i, token_i_raw]:
            if tk:
                push_auth(auth_candidates, {"i": tk, "device": device, "p": p_token})
                push_auth(auth_candidates, {"u": tk, "device": device, "p": p_token})

        for tk in [token_u, token_u_raw]:
            if tk:
                push_auth(auth_candidates, {"uv": tk, "device": device, "p": p_token})
                push_auth(auth_candidates, {"u": tk, "device": device, "p": p_token})

        if not auth_candidates:
            auth_candidates.append({"device": "", "p": ip})

        return auth_candidates

    def _send_json(self, ws, payload: Dict[str, object], tag: str) -> None:
        text = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        with self.send_lock:
            ws.send(text)
        print(f"ws_send[{tag}]={text}")

    def _set_active_socket(self, ws) -> None:
        with self._active_ws_lock:
            self._active_ws = ws

    def get_active_socket(self):
        """获取当前活跃的 WebSocket 连接实例。"""
        with self._active_ws_lock:
            return self._active_ws

    def send_raw(self, payload: Dict[str, object], tag: str = "admin") -> bool:
        """通过当前活跃 socket 发送任意 JSON。"""
        ws = self.get_active_socket()
        if ws is None:
            return False
        try:
            self._send_json(ws, payload, tag)
            return True
        except Exception as exc:
            print(f"admin_send_error tag={tag} err={exc}")
            return False

    def send_room_message(self, message: str) -> bool:
        """向房间发送一条普通消息。"""
        text = str(message or "").strip()
        if not text:
            return False
        return self.send_raw(
            {
                "Act": "",
                "Color": "#FFFFFF",
                "Msg": text,
                "c": "SayInRoom",
            },
            "adminSayInRoom",
        )

    def _should_suppress_ws_log(self, cmd: str) -> bool:
        if not self.config.filter_loudspeaker_logs:
            return False
        return cmd in {"SM", "Speaker"}

    @staticmethod
    def _is_friend_cmd(cmd: str) -> bool:
        return "friend" in (cmd or "").lower()

    def _handle_friend_status_update(self, obj: dict) -> None:
        records = collect_friend_records(obj)
        if not records:
            if isinstance(obj, dict) and isinstance(obj.get("data"), list) and len(obj.get("data")) == 0:
                print("friend_list_empty=1")
                if not self.state.friend_empty_logged:
                    append_log_line(
                        self.config.friend_monitor_log_file,
                        f"[{now_str()}] 好友列表为空，暂无可监控对象",
                    )
                    self.state.friend_empty_logged = True
            else:
                print("friend_monitor_parse_empty=1")
            return

        print(f"friend_monitor_records={len(records)}")
        latest: Dict[str, Dict[str, object]] = {}
        for item in records:
            latest[str(item["user_id"])] = {
                "name": str(item["name"]),
                "online": bool(item["online"]),
            }

        if not self.state.friend_seen_once:
            self.state.friend_status = latest
            self.state.friend_seen_once = True
            print(f"friend_monitor_init_ok count={len(latest)}")
            return

        for uid, new_item in latest.items():
            old_item = self.state.friend_status.get(uid)
            if old_item is None:
                status_text = "在线" if new_item["online"] else "离线"
                append_log_line(
                    self.config.friend_monitor_log_file,
                    f"[{now_str()}] 新好友 {new_item['name']}({uid}) 当前{status_text}",
                )
                continue

            if old_item["online"] != new_item["online"]:
                status_text = "上线" if new_item["online"] else "下线"
                append_log_line(
                    self.config.friend_monitor_log_file,
                    f"[{now_str()}] {new_item['name']}({uid}) {status_text}",
                )

        self.state.friend_status = latest

    def _poll_friends_via_http(self, ws, tag: str) -> None:
        with self.friend_http_lock:
            try:
                payload = self.friend_http_service.fetch_friends_via_http(
                    self.session,
                    self.fv.get("u", ""),
                )
                print(f"friend_http_poll_ok tag={tag}")
                if self.config.monitor_friends_enabled:
                    self._handle_friend_status_update(payload)
            except Exception as exc:
                print(f"friend_http_poll_error tag={tag} err={exc}")

    def _send_flow(self, ws) -> None:
        idx = self.state.auth_idx
        if idx >= len(self.auth_candidates):
            return

        payload = dict(self.auth_candidates[idx])
        if self.state.bootstrap_room_id:
            payload["R"] = int(self.state.bootstrap_room_id)
        if self.state.bootstrap_room_pwd:
            payload["RP"] = self.state.bootstrap_room_pwd
        self._send_json(ws, payload, f"auth#{idx + 1}")

    def _queue_line_reconnect(self, ws, line_id: str, room_id: str = "", room_pwd: str = "") -> None:
        target_line_id = str(line_id or "").strip()
        if not target_line_id:
            return

        target_room_id = str(room_id or "").strip()
        target_room_pwd = str(room_pwd or "").strip()

        self.reconnect.next_ws_url = self._resolve_ws_url_by_line(target_line_id)
        self.reconnect.next_line_id = target_line_id
        self.reconnect.next_join_room_id = target_room_id
        self.reconnect.next_join_room_pwd = target_room_pwd
        self.reconnect.fast_reconnect_once = True

        print(
            "smooth_line_switch "
            + f"url={self.reconnect.next_ws_url} room={target_room_id or 'current'} "
            + f"line={target_line_id} pwd={'yes' if target_room_pwd else 'no'}"
        )
        try:
            ws.close()
        except Exception:
            pass

    def _resolve_ws_url_by_line(self, line_id: str) -> str:
        target_line_id = str(line_id or "").strip()
        if not target_line_id:
            return self.config.ws_url

        try:
            line_info = self._fetch_line_info()
            for item in line_info:
                if str(item.get("Id") or "") != target_line_id:
                    continue

                ip_port = str(item.get("IpPort") or "").strip()
                if not ip_port:
                    break

                # 尝试 IpPort 列表中的每个候选地址，优先返回可 TCP 连接的地址
                for addr in ip_port.split("|"):
                    addr = addr.strip()
                    host, port = self._split_host_port(addr)
                    if not host or not port:
                        continue
                    # 测试 TCP 可达性
                    try:
                        sock_timeout = 2
                        with socket.create_connection((host, int(port)), timeout=sock_timeout):
                            ws_url = build_ws_url_by_host_port(self.config.ws_url, host, port)
                            print(f"line_resolve_ok line={target_line_id} host={host} port={port}")
                            return ws_url
                    except Exception as exc:
                        print(f"line_addr_unreachable line={target_line_id} addr={host}:{port} err={exc}")
                        continue
                # 若所有候选地址均不可达，则尝试使用 Server 字段
                server = str(item.get("Server") or "").strip()
                if server:
                    host, port = self._split_host_port(server)
                    if host and port:
                        try:
                            with socket.create_connection((host, int(port)), timeout=2):
                                ws_url = build_ws_url_by_host_port(self.config.ws_url, host, port)
                                print(f"line_resolve_ok_via_server line={target_line_id} host={host} port={port}")
                                return ws_url
                        except Exception as exc:
                            print(f"server_unreachable line={target_line_id} server={server} err={exc}")
                break
        except Exception as exc:
            print(f"line_resolve_error line={target_line_id} err={exc}")

        ws_url = build_ws_url_by_line(self.config.ws_url, target_line_id)
        print(f"line_resolve_fallback line={target_line_id} ws={ws_url}")
        return ws_url

    def _fetch_line_info(self) -> List[dict]:
        parsed = urlparse(self.config.game_url)
        base_url = f"{parsed.scheme}://{parsed.netloc}"
        line_url = urljoin(base_url, "/Conn/GetLines.ss")
        # 前端传 ssl 由页面协议决定（https -> 1），不要使用 ws_url 判断
        ssl_flag = 1 if parsed.scheme == "https" else 0
        try:
            cookie_count = len(self.session.cookies.get_dict())
        except Exception:
            cookie_count = 0
        print(f"fetch_lines_cookie_count={cookie_count}")

        headers = {
            "Referer": self.config.game_url,
            "Origin": base_url,
            "X-Requested-With": "XMLHttpRequest",
            "Accept": "application/json, text/plain, */*",
        }

        # 前端在请求线路列表时会带上 fv 中的 token（例如 u 或 i）作为 query 参数，
        # 如果不带这些参数接口可能返回空的 lineInfo。这里按可用项一并传入。
        params = {"ssl": ssl_flag}
        if isinstance(self.fv, dict):
            if self.fv.get("u"):
                params["u"] = self.fv.get("u")
            if self.fv.get("i"):
                params["i"] = self.fv.get("i")
            if self.fv.get("p"):
                params["p"] = self.fv.get("p")

        print(f"fetch_lines_request_params={params}")
        try:
            res = self.session.get(line_url, params=params, headers=headers, timeout=10)
        except Exception as exc:
            print(f"fetch_lines_request_error url={line_url} err={exc}")
            raise

        # Diagnostic logging to help debug why line list may be empty
        body_preview = (res.text or "")[:1024]
        print(
            "fetch_lines_http_status="
            f"{res.status_code} url={res.url} referer={headers['Referer']} origin={headers['Origin']} preview={body_preview}"
        )

        if res.status_code != 200:
            raise RuntimeError(f"fetch_lines_http_status={res.status_code}")

        try:
            payload = res.json()
        except Exception as exc:
            print(f"fetch_lines_json_error err={exc} body_preview={body_preview}")
            raise

        if not isinstance(payload, dict):
            print(f"fetch_lines_payload_not_dict type={type(payload)} keys_preview={list(payload)[:8]}")
            raise RuntimeError("fetch_lines_payload_not_dict")

        lines = payload.get("lineInfo")
        if not isinstance(lines, list):
            print(f"fetch_lines_no_lineInfo payload_keys={list(payload.keys())}")
            raise RuntimeError("未获取到可用线路列表: no lineInfo")

        return lines

    @staticmethod
    def _pick_best_line(line_info: List[dict]) -> dict:
        if not line_info:
            return {}

        sorted_lines = sorted(line_info, key=lambda item: int(item.get("Id") or 0))
        for item in sorted_lines:
            max_player = int(item.get("MaxPlayerNum") or 0)
            now_player = int(item.get("NowPlayerNum") or 0)
            if max_player <= 0 or now_player < max_player:
                return item
        return sorted_lines[0]

    def _bootstrap_ws_url(self) -> str:
        if self.reconnect.next_ws_url:
            return self.reconnect.next_ws_url

        if self.config.ws_url:
            return self.config.ws_url

        line_info = self._fetch_line_info()
        if not line_info:
            raise RuntimeError("未获取到可用线路列表")

        selected = self._pick_best_line(line_info)
        line_id = str(selected.get("Id") or "").strip()
        if not line_id:
            raise RuntimeError("线路列表中缺少有效 Id")

        self.reconnect.next_line_id = line_id
        return self._resolve_ws_url_by_line(line_id)

    @staticmethod
    def _split_host_port(addr: str) -> Tuple[str, str]:
        text = str(addr or "").strip()
        if not text:
            return "", ""

        if "://" in text:
            text = text.split("://", 1)[1]
        text = text.strip("/")
        if ":" not in text:
            return text, ""

        host, port = text.rsplit(":", 1)
        return host.strip(), port.strip()

    def _send_create_room(self, ws, reason: str) -> None:
        if self.state.current_room_id or self.state.room_id:
            return
        if self.state.create_room_sent:
            return

        self._send_json(
            ws,
            {
                "cmd": "createRoom",
                "a": self.config.room_area,
                "gt": self.config.room_gt,
                "name": self.config.room_name,
                "limit": 0,
                "pwd": "",
                "lip": 0,
                "FBlack": 0,
            },
            f"createRoom[{reason}]",
        )
        self.state.create_room_sent = True

    def _update_self_room_state(self, cmd: str, obj: dict, source: str) -> None:
        # 只用“自身位置确认类”消息更新当前房间，避免 Invite 携带 RoomId 污染状态。
        if cmd not in {"JoinRoom", "JoinHall", "jump", "LeaveRoom"}:
            return

        room_id, line_id = extract_room_state_from_obj(cmd, obj)
        changed = False

        if cmd == "JoinHall" and self.state.current_room_id:
            self.state.current_room_id = ""
            changed = True

        if room_id and room_id != self.state.current_room_id:
            self.state.current_room_id = room_id
            changed = True

        if line_id and line_id != self.state.current_line_id:
            self.state.current_line_id = line_id
            changed = True

        if changed:
            line = make_self_room_status_line(
                self.state.current_room_id,
                self.state.current_line_id,
                source,
            )
            self.state.last_room_status_log = line
            print(f"self_room_status_update={line}")
            append_log_line(self.config.self_status_log_file, line)

    def _report_self_room_status(self, tag: str) -> None:
        state_text = "大厅" if not self.state.current_room_id else "房间内"
        line = make_self_room_status_line(
            self.state.current_room_id,
            self.state.current_line_id,
            f"{tag},{state_text}",
        )
        self.state.last_room_status_log = line
        print(f"self_room_status={line}")
        append_log_line(self.config.self_status_log_file, line)

    def _arm_rejoin_fallback(self, ws, target_room_id: str, timeout_seconds: int = 8) -> None:
        """自动回房后启用兜底：超时仍未入房则创建新房。"""
        self.state.rejoin_watch_version += 1
        watch_version = self.state.rejoin_watch_version

        def worker() -> None:
            # 连接已关闭时 stop_keepalive 会置位，此处直接退出。
            if self.stop_keepalive.wait(timeout_seconds):
                return

            if watch_version != self.state.rejoin_watch_version:
                return

            if self.state.current_room_id:
                return

            if self.state.room_id != target_room_id:
                return

            print(f"rejoin_timeout_fallback room={target_room_id} action=createRoom")
            self.state.room_id = ""
            self.state.create_room_sent = False
            try:
                self._send_create_room(ws, "rejoinTimeoutFallback")
            except Exception as exc:
                print(f"rejoin_timeout_fallback_error={exc}")

        threading.Thread(target=worker, daemon=True).start()

    @staticmethod
    def _extract_jump_host_port(obj: dict) -> Tuple[str, str]:
        if not isinstance(obj, dict):
            return "", ""

        host = ""
        for key in ("host", "h", "ip", "wsHost"):
            value = obj.get(key)
            if value not in (None, ""):
                host = str(value).strip()
                break

        port = ""
        for key in ("port", "n", "wsPort", "p"):
            value = obj.get(key)
            if value in (None, ""):
                continue
            text = str(value).strip()
            if text.isdigit():
                port = text
                break

        return host, port

    @staticmethod
    def _extract_line_id_from_ws_url(ws_url: str) -> str:
        text = str(ws_url or "")
        match = re.search(r"kg(\d+)\.ss911\.cn", text)
        if not match:
            return ""
        return match.group(1)

    def _handle_login(self, ws, _obj: dict, _text: str) -> None:
        self.state.login_ok = True
        if self.state.bootstrap_room_id:
            # 前端首包已带 R 时，等待服务端直接回 JoinRoom，不再主动发 JoinHall/JoinRoom。
            self.state.join_hall_sent = True
            print(f"bootstrap_wait_join_room={self.state.bootstrap_room_id}")
            return

        if not self.state.join_hall_sent:
            self._send_json(ws, {"cmd": "UserInfo"}, "UserInfo")
            self._send_json(ws, {"cmd": "JoinHall"}, "JoinHall")
            self.state.join_hall_sent = True

    @staticmethod
    def _extract_sender_from_room_log_line(line: str) -> str:
        """从 room_say 日志行中提取发送者昵称。"""
        text = str(line or "").strip()
        if not text:
            return ""
        # 匹配包含位置信息的格式：[时间] [房间ID] [位置:X] 昵称: 消息
        match = re.match(r"^\[[^\]]+\]\s*(?:\[[^\]]+\]\s*)?(?:\[位置:[^\]]+\]\s*)?([^:]+):\s*.*$", text)
        if not match:
            # 回退到旧格式：[时间] [房间ID] 昵称: 消息
            match = re.match(r"^\[[^\]]+\]\s*(?:\[[^\]]+\]\s*)?([^:]+):\s*.*$", text)
            if not match:
                return ""

        return str(match.group(1) or "").strip()

    def _get_recent_room_messages_for_llm(self, limit: int = 20, exclude_sender: str = "夏凌依") -> list[str]:
        """读取 room_say.txt 最近若干条，按发送者过滤。"""
        file_path = self.config.room_say_log_file
        if not file_path or not os.path.exists(file_path):
            return []

        kept: list[str] = []
        try:
            with open(file_path, "r", encoding="utf-8") as file_handle:
                for raw_line in file_handle:
                    line = raw_line.rstrip("\r\n")
                    if not line:
                        continue
                    sender = self._extract_sender_from_room_log_line(line)
                    if sender == exclude_sender:
                        continue
                    kept.append(line)
        except OSError:
            return []

        if limit <= 0:
            return []
        return kept[-limit:]

    def _dispatch_room_say_to_llm(self, ws, obj: dict) -> None:
        """将 RoomSay 文本异步转发给大模型，并把结果按 SayInRoom 格式发送。"""
        sender_name, msg_text,position = parse_room_say_payload(obj)

        allowlist = self.config.get_room_say_llm_allowlist()
        if sender_name not in allowlist:
            print(f"llm_skip_not_allowed_sender sender={sender_name}")
            return

        # 服务器会回推自己发出的消息，这里按昵称过滤，避免进入自触发循环。
        if sender_name == self.config.self_nickname:
            print(f"llm_skip_self_room_say sender={sender_name}")
            return

        # 仅当消息的第一个字符与机器人当前的 RoomSite 相同时，才发送给大模型。
        # 例如机器人位置为 '5'，消息为 '5，你喜欢谁' 时才会触发。
        site = str(self.state.current_room_site or "").strip()
        if site:
            txt = str(msg_text or "").lstrip()
            if not txt:
                print("llm_skip_empty_msg=1")
                return
            first_char = txt[0]
            if first_char != site:
                print(f"llm_skip_site_mismatch site={site} first_char={first_char}")
                return

        content = str(msg_text or "").strip()
        if not content:
            return
        if content.isdigit():
            print("llm_skip_pure_number")
            return
        # 特殊指令处理：如果消息格式为 "<roomsite>[ ,]?添加权限[ ,]?<username>"，
        # 则在执行调用大模型前把 <username> 添加到 app_secrets.json 的 room_say_llm_allowlist。
        # 支持分隔符为空格或逗号或无分隔符。
        cmd_marker = "添加权限"
        idx = content.find(cmd_marker)
        if idx != -1:
            left = content[:idx].strip(" ,")
            right = content[idx + len(cmd_marker) :].strip(" ,")
            new_name = right or ""
            # 如果没有提取到用户名，则不处理
            if new_name:
                # 仅当发送者在当前白名单内才允许操作
                current_allowed = self.config.get_room_say_llm_allowlist()
                if sender_name in current_allowed:
                    # 读取当前 secrets，确保类型正确并更新
                    current = load_app_secrets(force_reload=True)
                    if not isinstance(current, dict):
                        current = {}

                    raw_list = current.get("room_say_llm_allowlist", [])
                    if isinstance(raw_list, list):
                        allowlist = [str(i).strip() for i in raw_list if str(i).strip()]
                    elif isinstance(raw_list, str):
                        allowlist = [p.strip() for p in raw_list.split(",") if p.strip()]
                    else:
                        allowlist = []

                    if new_name not in allowlist:
                        allowlist.append(new_name)
                        current["room_say_llm_allowlist"] = allowlist
                        ok = save_app_secrets(current)
                        print(f"permission_cmd_by={sender_name} target={new_name} saved={ok}")
                        self.send_room_message("添加权限成功~")
                    else:
                        print(f"permission_already_exists target={new_name}")
                        self.send_room_message("添加权限已存在~")
                    # 不继续调用大模型
                    return


        # 为避免模型直接复述问题，构造更明确的提示：
        # 说明提问者并要求模型直接回答且不要复述问题。
        # 智能判断是否需要长回复
        long_response_keywords = ["故事", "小说", "长篇", "详细", "完整", "写一篇", "创作", "讲一个"]
        is_long_request = any(keyword in content for keyword in long_response_keywords)
        if is_long_request:
            # 故事模式：只传递用户输入，不添加任何上下文
            llm_input = content
            print(f"llm_story_mode sender={sender_name} input={content[:30]}...")
        else:
            recent_context = self._get_full_llm_context()
            # 构建包含位置信息的提示
            position_info = f"（位置：{position}）" if position else ""
            if sender_name:
                base_input = (
                    f"玩家 {sender_name},位置{position_info} 提问：{content}\n请作为友好的玩家直接回答这个问题，"
                    "不要复述问题，回答简短。"
                )
            else:
                base_input = f"请直接回答：{content}，不要复述问题。"

            if recent_context:
                llm_input = f"{recent_context}\n\n{base_input}"
            else:
                llm_input = base_input

        def worker() -> None:
            from llm_client import ask_llm_stream
            chunk_count = 0
            if is_long_request:
                for reply_text in ask_llm_stream(llm_input, chunk_size=50):
                    reply_text = str(reply_text or "").strip()
                    if not reply_text:
                        continue
                    chunk_count += 1
                    self._send_json(
                        ws,
                        {
                            "Act": "",
                            "Color": "#FFFFFF",
                            "Msg": reply_text,
                            "c": "SayInRoom",
                        },
                        f"llmSayInRoom#{chunk_count}",
                    )
                    # 在每个 chunk 发送后添加延时，防止刷屏被踢
                    if chunk_count > 1:  # 第一条不延时，快速响应
                        time.sleep(0.8)  # 每条消息间隔 0.8 秒
            else:
                try:
                    reply_chunks = ask_llm_chunks(llm_input, chunk_size=50)
                    if not reply_chunks:
                        print("llm_reply_empty=1")
                        return

                    for idx, chunk in enumerate(reply_chunks, start=1):
                        reply_text = str(chunk or "").strip()
                        if not reply_text:
                            continue

                        self._send_json(
                            ws,
                            {
                                "Act": "",
                                "Color": "#FFFFFF",
                                "Msg": reply_text,
                                "c": "SayInRoom",
                            },
                            f"llmSayInRoom#{idx}",
                        )
                    print(f"llm_reply_chunks={len(reply_chunks)}")
                except Exception as exc:
                    print(f"llm_call_error={exc}")

        threading.Thread(target=worker, daemon=True).start()

    def _handle_room_say(self, ws, obj: dict, _text: str) -> None:
        self.room_say_logger.log_room_say_message(obj, self.state.current_room_id)
        # 更新用户位置映射
        self._update_user_position_map(obj)
        # 先检查是否是天气查询
        sender_name, msg_text,position = parse_room_say_payload(obj)

        # 检查消息是否以RoomSite开头
        site = str(self.state.current_room_site or "").strip()
        if site and msg_text:
            txt = str(msg_text or "").lstrip()
            if txt and txt[0] == site:
                # 提取去除RoomSite后的消息内容
                content = txt[1:].strip()
                # 尝试解析天气查询
                city, is_forecast = self._parse_weather_query(content)
                if city:
                    # 拦截天气查询，不调用LLM
                    query_type = "预报" if is_forecast else "实况"
                    print(f"weather_intercepted sender={sender_name} city={city} type={query_type}")
                    self._dispatch_weather_query(ws, city, is_forecast)
                    return

        # 非天气查询，正常调用LLM
        self._dispatch_room_say_to_llm(ws, obj)

    def _handle_join_hall(self, ws, _obj: dict, _text: str) -> None:
        self.state.joined_hall = True
        self.state.current_room_id = ""

        # bootstrap 首包已带 R，等待服务端回 JoinRoom，避免重复发 JoinRoom/createRoom。
        if self.state.bootstrap_room_id:
            print(f"bootstrap_wait_join_room={self.state.bootstrap_room_id}")
            return

        # 若已记录自己的房间号但当前在大厅（例如被踢出），优先自动回到原房间。
        if self.state.room_id:
            target_room_id = self.state.room_id
            self._send_json(
                ws,
                {
                    "cmd": "JoinRoom",
                    "RoomId": target_room_id,
                    "Password": "",
                },
                "autoRejoinRoom",
            )
            print(f"auto_rejoin_room room={target_room_id}")
            self._arm_rejoin_fallback(ws, target_room_id)
            return

        # 登录成功进入大厅后，默认立即创建房间。
        self._send_create_room(ws, "joinHall")

    def _handle_alert_msg(self, ws, obj: dict, _text: str) -> None:
        msg_text = str(obj.get("Msg", "") or "")
        if "房间不存在" not in msg_text:
            return

        print("current_room_missing_recreate=1")
        self.state.room_id = ""
        self.state.current_room_id = ""
        self.state.create_room_sent = False
        if self.state.joined_hall:
            self._send_create_room(ws, "roomMissing")

    def _handle_jump(self, ws, obj: dict, _text: str) -> None:
        jump_room = str(obj.get("r", "") or "")
        jump_line = str(obj.get("l", "") or "")
        jump_host, jump_port = self._extract_jump_host_port(obj)

        # 如果服务端推送的 jump 指向当前已连接的线路且没有指定新的主机，
        # 并且也没有携带目标房间号，则忽略该 jump（与前端行为一致）。
        # 但如果 jump 携带目标房间号（r），即使是同一线路也应当处理，
        # 因为这通常意味着需要进入指定房间。
        if (
            jump_line
            and self.state.connected_ws_line_id
            and jump_line == self.state.connected_ws_line_id
            and not jump_host
            and not jump_room
        ):
            print(
                "jump_ignore_same_line "
                + f"line={jump_line} room={self.state.room_id or self.state.bootstrap_room_id or 'current'}"
            )
            return

        target_room = jump_room or self.state.bootstrap_room_id
        if not (target_room or jump_line or jump_host):
            return

        ws_url = build_ws_url_by_host_port(self.config.ws_url, jump_host, jump_port)
        if ws_url == self.config.ws_url and jump_line:
            ws_url = build_ws_url_by_line(self.config.ws_url, jump_line)

        self.reconnect.next_ws_url = ws_url
        self.reconnect.next_line_id = jump_line
        self.reconnect.next_join_room_id = target_room

        if target_room and target_room == self.state.bootstrap_room_id:
            self.reconnect.next_join_room_pwd = self.state.bootstrap_room_pwd
        else:
            self.reconnect.next_join_room_pwd = ""

        self.reconnect.fast_reconnect_once = True

        print(
            "smooth_line_switch "
            + f"url={self.reconnect.next_ws_url} room={self.reconnect.next_join_room_id or 'current'} "
            + f"line={jump_line or 'unknown'} host={jump_host or 'none'} port={jump_port or 'none'}"
        )
        try:
            ws.close()
        except Exception:
            pass

    def _handle_join_room(self, _ws, obj: dict, _text: str) -> None:
        room_id = obj.get("RoomId")
        line_id = str(obj.get("LineId") or obj.get("lineId") or "").strip()
        active_line_id = str(self.state.current_line_id or self.state.connected_ws_line_id or "").strip()

        if line_id and active_line_id and line_id != active_line_id:
            target_room_id = str(room_id or self.state.bootstrap_room_id or self.state.room_id or "").strip()
            target_room_pwd = self.state.bootstrap_room_pwd or self.reconnect.next_join_room_pwd or ""
            self._queue_line_reconnect(_ws, line_id, target_room_id, target_room_pwd)
            return

        if room_id is not None:
            self.state.rejoin_watch_version += 1
            self.state.current_room_id = str(room_id)
            self.state.room_id = str(room_id)
            if line_id:
                self.state.current_line_id = line_id
            elif active_line_id:
                self.state.current_line_id = active_line_id
            self.state.bootstrap_room_id = ""
            self.state.bootstrap_room_pwd = ""

            self.reconnect.next_join_room_id = ""
            self.reconnect.next_join_room_pwd = ""
            self.reconnect.jump_reconnect_count = 0
            self.state.create_room_sent = False

            self._subscribe_room_so(_ws, str(room_id))

        self.state.success = True
        self._report_self_room_status("joinRoomAck")

    def _subscribe_room_so(self, ws, room_id: str) -> None:
        rid = str(room_id or "").strip()
        if not rid:
            return

        # 对齐前端入房后的 SO 订阅顺序，确保房间基础同步可达。
        for so_name in (f"Room{rid}", f"Game{rid}", f"Player{rid}", f"RoomUsers{rid}"):
            self._send_json(ws, {"c": "SO_o", "n": so_name}, f"SO_o[{so_name}]")

    def _handle_invite(self, ws, obj: dict, _text: str) -> None:
        room_id = str(obj.get("RoomId") or obj.get("roomId") or obj.get("ToRoomId") or "").strip()
        line_id = str(obj.get("LineId") or obj.get("lineId") or obj.get("l") or "").strip()
        room_pwd = str(obj.get("RoomPwd") or obj.get("roomPwd") or obj.get("pwd") or "").strip()
        from_user = str(obj.get("FromUserName") or obj.get("fromUserName") or obj.get("UserName") or "").strip()

        if not room_id:
            print("invite_ignore_missing_room=1")
            return

        current_room_id = str(self.state.current_room_id or self.state.room_id or "").strip()
        print(
            "invite_recv "
            + f"from={from_user or 'unknown'} room={room_id} line={line_id or 'unknown'} "
            + f"current_room={current_room_id or 'hall'}"
        )

        active_line_id = str(self.state.current_line_id or self.state.connected_ws_line_id or "").strip()
        if line_id and line_id != active_line_id:
            self._queue_line_reconnect(ws, line_id, room_id, room_pwd)
            return

        if current_room_id == room_id:
            print("invite_same_room_ignore=1")
            return

        payload = {
            "cmd": "LeaveRoom",
            "ToRoomId": room_id,
            "FromUserName": from_user,
        }
        if room_pwd:
            payload["RoomPwd"] = room_pwd

        self._send_json(ws, payload, "acceptInviteLeaveRoom")

    def _on_open(self, ws) -> None:
        self._set_active_socket(ws)
        print("ws_connected=1")
        self._send_flow(ws)

        def keepalive_loop() -> None:
            while not self.stop_keepalive.wait(self.config.keepalive_seconds):
                try:
                    self._send_json(ws, {"cmd": "UserInfo"}, "keepalive")
                except Exception as exc:
                    print(f"ws_keepalive_error={exc}")
                    break

        def self_status_loop() -> None:
            while not self.stop_keepalive.wait(self.config.self_status_interval_seconds):
                try:
                    self._send_json(ws, {"cmd": "UserInfo"}, "selfStatusProbe")
                    self._report_self_room_status("periodic")
                except Exception as exc:
                    print(f"self_status_error={exc}")
                    break

        threading.Thread(target=keepalive_loop, daemon=True).start()
        threading.Thread(target=self_status_loop, daemon=True).start()

        if self.config.monitor_friends_enabled:
            abs_log_file = os.path.abspath(self.config.friend_monitor_log_file)
            print(f"friend_monitor_started log={abs_log_file}")

            def friend_monitor_loop() -> None:
                while not self.stop_keepalive.wait(self.config.friend_monitor_interval_seconds):
                    try:
                        self.state.friend_poll_round += 1
                        self._poll_friends_via_http(ws, f"round-{self.state.friend_poll_round}")
                        if self.config.friend_ws_fallback_enabled:
                            self._send_json(ws, {"cmd": "MyFriends"}, "friendPoll")
                        print(f"friend_poll_round={self.state.friend_poll_round}")
                    except Exception as exc:
                        print(f"friend_poll_error={exc}")
                        break

            threading.Thread(target=friend_monitor_loop, daemon=True).start()
            self.state.friend_poll_round += 1
            self._poll_friends_via_http(ws, "init")
            if self.config.friend_ws_fallback_enabled:
                self._send_json(ws, {"cmd": "MyFriends"}, "friendPollInit")
            print(f"friend_poll_round={self.state.friend_poll_round}")

    def _on_message(self, ws, message) -> None:
        text = message if isinstance(message, str) else str(message)
        cmd, obj = parse_cmd_and_obj(text)

        if not self._should_suppress_ws_log(cmd):
            print(f"ws_recv={text}")

        # 处理 SO_Sync 中包含的房间用户列表，记录机器人自己的 RoomSite（位置）。
        # SO_Sync 的结构示例: {"l":[[1,"15315766",{...}], [1,"19348971",{...}]], "n":"RoomUsers3155"}
        if cmd == "SO_Sync" and isinstance(obj, dict):
            try:
                rows = obj.get("l") or obj.get("data") or []
                has_position_change = False
                for row in rows:
                    if not isinstance(row, (list, tuple)) or len(row) < 3:
                        continue
                    payload = row[2]
                    if not isinstance(payload, dict):
                        continue
                    name = str(payload.get("UserName") or "").strip()
                    site = payload.get("RoomSite")
                    if name and site is not None:
                        site_str = str(site)
                        old_site = self.state.user_position_map.get(name)

                        # 更新所有用户的位置映射
                        self.state.user_position_map[name] = site_str
                        # 如果位置发生变化，标记需要更新版本号
                        if old_site != site_str:
                            has_position_change = True
                        # 如果是自己，也更新 current_room_site
                        if name == self.config.self_nickname:
                            self.state.current_room_site = str(site)
                            print(f"self_roomsite_update site={self.state.current_room_site} name={name}")
                            # 如果有位置变化，递增版本号
                if has_position_change:
                    self.state.position_map_version += 1
                    print(f"-------------game_position_info_updated-----------------")

            except Exception as exc:
                print(f"so_sync_parse_error={exc}")

        self._update_self_room_state(cmd, obj, cmd or "recv")

        handler = self.cmd_handlers.get(cmd)
        if handler is not None:
            handler(ws, obj, text)

        if self.config.monitor_friends_enabled and self._is_friend_cmd(cmd):
            self.state.friend_resp_count += 1
            print(f"friend_cmd_recv cmd={cmd} count={self.state.friend_resp_count}")
            self._handle_friend_status_update(obj)

        if "JoinRoom" in text or '"cmd":"JoinRoom"' in text:
            self.state.success = True

        if "NotReg" in text and not self.state.success:
            if self.state.auth_idx + 1 < len(self.auth_candidates):
                self.state.auth_idx += 1
                self.state.retried_on_notreg = True
                print(f"ws_retry_auth={self.state.auth_idx + 1}/{len(self.auth_candidates)}")
                self._send_flow(ws)

    def _on_error(self, _ws, error) -> None:
        print(f"ws_error={error}")

    def _on_close(self, _ws, status_code, close_msg) -> None:
        self.stop_keepalive.set()
        self._set_active_socket(None)
        print(f"ws_closed code={status_code} msg={close_msg}")
        if self.state.retried_on_notreg and not self.state.success:
            print("提示: 已尝试多组认证参数，仍然 NotReg，可继续抓包首包进一步确定登录字段")

    def run_forever(self) -> None:
        cookie_header = build_cookie_header(self.session)
        if not cookie_header:
            raise RuntimeError("未获取到 Cookie，无法建立游戏 WebSocket 会话")

        set_active_client(self)
        try:
            ws_url = self._bootstrap_ws_url()
            self.reconnect.next_ws_url = ""
            self.state.connected_ws_line_id = self._extract_line_id_from_ws_url(ws_url) or self.reconnect.next_line_id
            self.reconnect.next_line_id = ""
            print(f"ws_target={ws_url}")

            headers = [
                "Origin: https://t1.ss911.cn",
                "Pragma: no-cache",
                "Cache-Control: no-cache",
                "Accept-Language: zh-CN,zh;q=0.9,en;q=0.8",
                "User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/147.0.0.0 Safari/537.36 Edg/147.0.0.0",
                f"Cookie: {cookie_header}",
            ]

            ws_app = websocket.WebSocketApp(
                ws_url,
                header=headers,
                on_open=self._on_open,
                on_message=self._on_message,
                on_error=self._on_error,
                on_close=self._on_close,
            )
            try:
                ws_app.run_forever(ping_interval=20, ping_timeout=10)
            except Exception as exc:
                print(f"ws_run_forever_error={exc}")
                traceback.print_exc()
                raise
        finally:
            set_active_client(None)

    def _parse_weather_query(self, msg_text: str) -> tuple:
        """
        解析天气查询消息，提取城市名称和查询类型

        Args:
            msg_text: 消息文本，格式如 "南宁天气" 或 "南宁天气预报"

        Returns:
            tuple: (城市名称, 是否预报), 如果不是天气查询则返回 ("", False)
        """
        if not msg_text:
            return "", False

        msg_text = msg_text.strip()

        # 匹配模式1：城市名（2-4个汉字）+ "天气预报"
        # 示例：南宁天气预报, 武汉天气预报
        pattern_forecast = r'^([\u4e00-\u9fa5]{2,4})天气预报$'
        match_forecast = re.match(pattern_forecast, msg_text)

        if match_forecast:
            city = match_forecast.group(1)
            print(f"weather_forecast_query_detected city={city}")
            return city, True

        # 匹配模式2：城市名（2-4个汉字）+ "天气"
        # 示例：南宁天气, 武汉天气
        pattern_current = r'^([\u4e00-\u9fa5]{2,4})天气$'
        match_current = re.match(pattern_current, msg_text)

        if match_current:
            city = match_current.group(1)
            print(f"weather_query_detected city={city}")
            return city, False

        return "", False

    def _dispatch_weather_query(self, ws, city: str, is_forecast: bool = False) -> None:
        """
        处理天气查询，调用天气API并发送结果

        Args:
            ws: WebSocket连接
            city: 城市名称
            is_forecast: 是否查询天气预报
        """

        def worker() -> None:
            try:
                weather_data = get_weather(city, forecast=is_forecast)

                if not weather_data:
                    reply_text = f"抱歉，查询{city}天气失败，请稍后重试。"
                else:
                    # 解析天气数据
                    status = weather_data.get("status")
                    if status != "1":
                        reply_text = f"抱歉，查询{city}天气失败。"
                    else:
                        if is_forecast:
                            # 解析天气预报数据
                            forecasts = weather_data.get("forecasts", [])
                            if not forecasts:
                                reply_text = f"抱歉，未找到{city}的天气预报信息。"
                            else:
                                forecast = forecasts[0]
                                province = forecast.get("province", "")
                                city_name = forecast.get("city", "")
                                casts = forecast.get("casts", [])

                                if not casts:
                                    reply_text = f"抱歉，未找到{city}的天气预报信息。"
                                else:
                                    # 整理天气预报信息
                                    forecast_lines = []
                                    for cast in casts[:4]:  # 最多显示4天
                                        date = cast.get("date", "")
                                        week = cast.get("week", "")
                                        dayweather = cast.get("dayweather", "")
                                        nightweather = cast.get("nightweather", "")
                                        daytemp = cast.get("daytemp", "")
                                        nighttemp = cast.get("nighttemp", "")
                                        daywind = cast.get("daywind", "")
                                        nightwind = cast.get("nightwind", "")
                                        daypower = cast.get("daypower", "")
                                        nightpower = cast.get("nightpower", "")

                                        # 格式化日期（去掉年份）
                                        if len(date) >= 5:
                                            date_short = date[5:]  # MM-DD
                                        else:
                                            date_short = date

                                        day_line = f"白天：{dayweather} {daytemp}℃ {daywind}风{daypower}"
                                        night_line = f"夜间：{nightweather} {nighttemp}℃ {nightwind}风{nightpower}"

                                        forecast_lines.append(f"【{date_short} 周{week}】\n{day_line}\n{night_line}")

                                    reply_text = (
                                            f"【{province}{city_name} 天气预报】\n"
                                            + "\n".join(forecast_lines)
                                    )
                        else:
                            # 解析实况天气数据
                            lives = weather_data.get("lives", [])
                            if not lives:
                                reply_text = f"抱歉，未找到{city}的天气信息。"
                            else:
                                life = lives[0]
                                province = life.get("province", "")
                                city_name = life.get("city", "")
                                weather = life.get("weather", "")
                                temperature = life.get("temperature", "")
                                winddirection = life.get("winddirection", "")
                                windpower = life.get("windpower", "")
                                humidity = life.get("humidity", "")
                                reporttime = life.get("reporttime", "")

                                # 风力等级转换为描述
                                wind_power_desc = {
                                    "0": "无风",
                                    "1": "软风",
                                    "2": "轻风",
                                    "3": "微风",
                                    "4": "和风",
                                    "5": "清风",
                                    "6": "强风",
                                    "7": "疾风",
                                    "8": "大风",
                                    "9": "烈风",
                                    "10": "狂风",
                                    "11": "暴风",
                                    "12": "飓风"
                                }
                                wind_desc = wind_power_desc.get(windpower, f"{windpower}级")

                                # 整理天气信息
                                reply_text = (
                                    f"【{province}{city_name} 天气】\n"
                                    f"天气状况：{weather}\n"
                                    f"当前温度：{temperature}℃\n"
                                    f"风向风力：{winddirection}风 {wind_desc}\n"
                                    f"相对湿度：{humidity}%\n"
                                    f"更新时间：{reporttime}"
                                )

                # 发送天气查询结果到房间
                self._send_json(
                    ws,
                    {
                        "Act": "",
                        "Color": "#FFFFFF",
                        "Msg": reply_text,
                        "c": "SayInRoom",
                    },
                    "weatherSayInRoom",
                )
                print(f"weather_reply={reply_text}")

            except Exception as exc:
                print(f"weather_call_error={exc}")
                # 发送错误信息
                self._send_json(
                    ws,
                    {
                        "Act": "",
                        "Color": "#FFFFFF",
                        "Msg": f"抱歉，查询天气时出错：{str(exc)}",
                        "c": "SayInRoom",
                    },
                    "weatherErrorSayInRoom",
                )
        threading.Thread(target=worker, daemon=True).start()

    def _update_user_position_map(self, obj: dict) -> None:
        """从 RoomSay 消息中更新用户位置映射。"""
        if not isinstance(obj, dict):
            return

        user_arr = obj.get("u")
        if not isinstance(user_arr, list) or len(user_arr) < 3:
            return

        username = str(user_arr[2] or "").strip()
        position = str(obj.get("s", "") or "").strip()

        if username and position:
            old_position = self.state.user_position_map.get(username)
            # 只有当位置发生变化或是新用户时才更新版本号
            if old_position != position:
                self.state.user_position_map[username] = position
                self.state.position_map_version += 1

    def _build_position_info_string(self) -> str:
        """构建用户位置信息的字符串表示。"""
        if not self.state.user_position_map:
            return ""

        # 按位置排序，生成易读的格式
        position_list = []
        for username, position in sorted(self.state.user_position_map.items(),
                                         key=lambda x: int(x[1]) if x[1].isdigit() else 999):
            position_list.append(f"位置{position}: {username}")

        return "，".join(position_list)

    def _build_recent_chat_context_for_llm(self) -> str:
        """每 1 分钟最多追加一次最近 20 条聊天上下文（仅处理聊天记录）。"""
        now = time.time()
        if now - float(self.state.llm_recent_chat_sent_at or 0.0) < 60:
            return ""

        recent_lines = self._get_recent_room_messages_for_llm(limit=20, exclude_sender="夏凌依")
        if not recent_lines:
            return ""

        self.state.llm_recent_chat_sent_at = now
        context_body = "\n".join(recent_lines)

        result = (
            "以下是 room_say 最近20条聊天记录（已过滤发送者’夏凌依‘，“夏凌依是你的名字”），仅供你理解上下文：\n"
        f"{context_body}\n"
        )

        result += "请基于这些上下文回答当前问题，回答仍需简短。"
        return result

    def _build_position_context_for_llm(self) -> str:
        """构建位置信息上下文（独立于聊天记录）。"""
        # 检查位置映射表是否有更新
        if self.state.position_map_version > self.state.llm_last_position_map_version:
            position_info = self._build_position_info_string()
            # 更新 LLM 上次使用的位置映射表版本号
            self.state.llm_last_position_map_version = self.state.position_map_version

            if position_info:
                print(f"llm_position_info_updated version={self.state.position_map_version}")
                return (
                    f"\n【重要-最高优先级】当前房间内用户的最新位置信息（这是权威数据，必须以此为准）：\n"
                    f"{position_info}\n"
                    f"注意：如果聊天记录中的位置与此处冲突，请以本位置信息为准！\n"
                )

        return ""

    def _get_full_llm_context(self) -> str:
        """获取完整的 LLM 上下文（聊天记录 + 位置信息）。"""
        chat_context = self._build_recent_chat_context_for_llm()
        position_context = self._build_position_context_for_llm()

        if not chat_context and not position_context:
            return ""

         # 组合两个上下文：位置信息放在最前面，强调其权威性
        if position_context and chat_context:
            # 位置信息在前，聊天记录在后，确保 LLM 优先参考最新位置
            result = position_context + "\n" + chat_context
            return result
        elif position_context:
            return position_context
        else:
            return chat_context