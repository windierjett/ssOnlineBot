import json
from collections import deque
from pathlib import Path
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlparse

from .app_runner import ApplicationRunner
from .config import load_app_secrets
from .logging_utils import sanitize_filename


_SECRETS_FILE = Path(__file__).resolve().parent.parent / "app_secrets.json"


class _AdminHttpServer(ThreadingHTTPServer):
    def __init__(self, server_address: tuple[str, int], app: "AdminApiRunner") -> None:
        super().__init__(server_address, _AdminHttpHandler)
        self.app = app


class _AdminHttpHandler(BaseHTTPRequestHandler):
    server: _AdminHttpServer

    def log_message(self, _format: str, *_args: Any) -> None:
        # 避免默认 HTTP 日志刷屏，业务日志由应用统一输出。
        return

    def _send_cors_headers(self) -> None:
        origin = self.headers.get("Origin")
        self.send_header("Access-Control-Allow-Origin", origin or "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization, X-Requested-With, zone")
        self.send_header("Access-Control-Max-Age", "86400")
        self.send_header("Access-Control-Expose-Headers", "Content-Type")

    def _send_json(self, status_code: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self._send_cors_headers()
        self.end_headers()
        self.wfile.write(body)

    def _send_no_content(self, status_code: int = 204) -> None:
        self.send_response(status_code)
        self._send_cors_headers()
        self.end_headers()

    def _request_params(self) -> dict[str, str]:
        cache = getattr(self, "_request_params_cache", None)
        if isinstance(cache, dict):
            return cache

        params: dict[str, str] = {}
        if self.command == "GET":
            query = parse_qs(urlparse(self.path).query)
            params = {key: str((values or [""])[0]).strip() for key, values in query.items()}
        else:
            content_length = int(self.headers.get("Content-Length") or 0)
            raw_body = self.rfile.read(content_length) if content_length > 0 else b""

            if raw_body:
                content_type = (self.headers.get("Content-Type") or "").lower()

                if "application/json" in content_type:
                    try:
                        data = json.loads(raw_body.decode("utf-8"))
                    except Exception:
                        data = {}
                    if isinstance(data, dict):
                        params = {str(key): str(value).strip() for key, value in data.items()}
                elif "application/x-www-form-urlencoded" in content_type:
                    try:
                        form = parse_qs(raw_body.decode("utf-8"))
                    except Exception:
                        form = {}
                    params = {key: str((values or [""])[0]).strip() for key, values in form.items()}
                else:
                    params = {"msg": raw_body.decode("utf-8", errors="ignore").strip()}

        self._request_params_cache = params
        return params

    def _extract_msg(self) -> str:
        return str(self._request_params().get("msg") or "").strip()

    def _extract_param(self, name: str) -> str:
        return str(self._request_params().get(name) or "").strip()

    @staticmethod
    def _parse_bool(raw_value: str) -> bool | None:
        text = str(raw_value or "").strip().lower()
        if text in {"1", "true", "yes", "y", "on"}:
            return True
        if text in {"0", "false", "no", "n", "off"}:
            return False
        return None

    @staticmethod
    def _tail_lines(file_path: str, limit: int = 20) -> list[str]:
        tail = deque(maxlen=max(int(limit), 0))
        try:
            with open(file_path, "r", encoding="utf-8") as file_handle:
                for line in file_handle:
                    stripped = line.rstrip("\r\n")
                    if stripped:
                        tail.append(stripped)
        except FileNotFoundError:
            return []
        except OSError:
            return []
        return list(tail)

    def _handle_room_messages(self) -> None:
        snapshot = self.server.app.get_room_snapshot()
        log_file = snapshot.get("room_say_log_file") or self.server.app.runner.config.room_say_log_file
        messages = self._tail_lines(str(log_file), limit=20)
        snapshot["recent_messages"] = messages
        snapshot["recent_message_count"] = len(messages)
        self._send_json(200, {"ok": True, "data": snapshot})

    def _handle_friends(self) -> None:
        result = self.server.app.get_friends()
        if not result.get("ok"):
            self._send_json(503, result)
            return
        self._send_json(200, result)

    def _handle_friend_status_history(self) -> None:
        raw_limit = self._extract_param("limit")
        limit = 5000
        if raw_limit:
            try:
                limit = max(0, min(int(raw_limit), 5000))
            except ValueError:
                limit = 5000

        log_file = self.server.app.runner.config.friend_monitor_log_file
        messages = self._tail_lines(str(log_file), limit=limit)
        self._send_json(200, {"ok": True, "friend_status_history": messages, "data": messages})

    def _handle_room_message_history(self) -> None:
        raw_limit = self._extract_param("limit")
        limit = 5000
        if raw_limit:
            try:
                limit = max(0, min(int(raw_limit), 5000))
            except ValueError:
                limit = 5000

        log_file = self.server.app.runner.config.room_say_log_file
        messages = self._tail_lines(str(log_file), limit=limit)
        self._send_json(200, {"ok": True, "room_message_history": messages, "count": len(messages), "data": messages})

    def _handle_all_friend_messages(self) -> None:
        log_dir = Path(self.server.app.runner.config.room_say_log_dir)
        friends: list[str] = []
        if log_dir.is_dir():
            for log_file in sorted(log_dir.glob("room_say_*.txt")):
                if log_file.name == "room_say.txt":
                    continue
                friend_name = log_file.stem.removeprefix("room_say_")
                friends.append(friend_name)
        self._send_json(200, {"ok": True, "friends": friends, "data": friends})

    def _handle_friend_messages(self) -> None:
        friend_name = str(self._extract_param("friendName") or "").strip()
        if not friend_name:
            self._send_json(400, {"ok": False, "error": "friendName 不能为空"})
            return

        config = self.server.app.runner.config
        file_name = config.room_say_name_to_file.get(friend_name)
        if not file_name:
            file_name = f"room_say_{sanitize_filename(friend_name)}.txt"

        log_file = Path(config.room_say_log_dir) / file_name
        messages = self._tail_lines(str(log_file), limit=5000)
        self._send_json(
            200,
            {
                "ok": True,
                "friendName": friend_name,
                "friend_messages": messages,
                "count": len(messages),
                "data": messages,
            },
        )

    def _handle_permission(self) -> None:
        name = self._extract_param("name")
        enabled_raw = self._extract_param("enabled") or self._extract_param("boolean")
        enabled = self._parse_bool(enabled_raw)

        if not name:
            self._send_json(400, {"ok": False, "error": "name 不能为空"})
            return

        if enabled is None:
            self._send_json(400, {"ok": False, "error": "enabled 必须是布尔值"})
            return

        result = self.server.app.update_room_say_permission(name, enabled)
        if not result.get("ok"):
            self._send_json(500, result)
            return

        self._send_json(200, result)

    def _handle_permission_list(self) -> None:
        result = self.server.app.get_room_say_allowlist()
        self._send_json(200, {"ok": True, "room_say_llm_allowlist": result, "data": result})

    def _handle_roomsay(self) -> None:
        msg = self._extract_msg()
        if not msg:
            self._send_json(400, {"ok": False, "error": "msg 不能为空"})
            return

        ok = self.server.app.send_room_message(msg)
        if not ok:
            self._send_json(503, {"ok": False, "error": "当前无可用连接，消息发送失败"})
            return

        self._send_json(200, {"ok": True, "action": "roomsay", "msg": msg})

    def do_GET(self) -> None:
        route = urlparse(self.path).path
        if route == "/api/roomsay":
            self._handle_roomsay()
            return

        if route == "/api/room-messages":
            self._handle_room_messages()
            return

        if route == "/api/friends":
            self._handle_friends()
            return

        if route == "/api/friend-status-history":
            self._handle_friend_status_history()
            return

        if route == "/api/room-message-history":
            self._handle_room_message_history()
            return

        if route == "/api/all-friend-messages":
            self._handle_all_friend_messages()
            return

        if route == "/api/friend-messages":
            self._handle_friend_messages()
            return

        if route == "/api/room-permission":
            self._handle_permission()
            return

        if route == "/api/room-permission-list":
            self._handle_permission_list()
            return

        self._send_json(404, {"ok": False, "error": "not found"})

    def do_POST(self) -> None:
        route = urlparse(self.path).path
        if route == "/api/roomsay":
            self._handle_roomsay()
            return

        if route == "/api/room-messages":
            self._handle_room_messages()
            return

        if route == "/api/friends":
            self._handle_friends()
            return

        if route == "/api/friend-status-history":
            self._handle_friend_status_history()
            return

        if route == "/api/room-message-history":
            self._handle_room_message_history()
            return

        if route == "/api/all-friend-messages":
            self._handle_all_friend_messages()
            return

        if route == "/api/friend-messages":
            self._handle_friend_messages()
            return

        if route == "/api/room-permission":
            self._handle_permission()
            return

        if route == "/api/room-permission-list":
            self._handle_permission_list()
            return

        self._send_json(404, {"ok": False, "error": "not found"})

    def do_OPTIONS(self) -> None:
        route = urlparse(self.path).path
        if route in {
            "/api/roomsay",
            "/api/room-messages",
            "/api/friends",
            "/api/friend-status-history",
            "/api/room-message-history",
            "/api/all-friend-messages",
            "/api/friend-messages",
            "/api/room-permission",
            "/api/room-permission-list",
        }:
            self._send_no_content()
            return

        self._send_no_content()


class AdminApiRunner:
    """启动类：运行机器人并对外暴露管理接口。"""

    def __init__(self, host: str = "0.0.0.0", port: int = 8081) -> None:
        self.host = host
        self.port = port
        self.runner = ApplicationRunner()
        self._server: _AdminHttpServer | None = None

    def send_room_message(self, msg: str) -> bool:
        """通过当前活跃连接发送 RoomSay。"""
        return self.runner.send_room_message(msg)

    def get_friends(self) -> dict[str, Any]:
        """调用项目中已有的好友列表接口，返回当前好友数据。"""
        client = self.runner.get_active_client()
        if client is None:
            return {"ok": False, "error": "当前无可用连接，无法获取好友列表"}

        try:
            data = client.friend_http_service.fetch_friends_via_http(
                client.session,
                client.fv.get("u", ""),
            )
            return {"ok": True, "data": data}
        except Exception as exc:
            return {"ok": False, "error": f"获取好友列表失败: {exc}"}

    def get_room_snapshot(self) -> dict[str, Any]:
        """返回当前活跃连接和房间状态快照。"""
        client = self.runner.get_active_client()
        active_socket = self.runner.get_active_socket()

        snapshot: dict[str, Any] = {
            "socket_active": active_socket is not None,
            "socket_id": id(active_socket) if active_socket is not None else None,
            "room_say_log_file": self.runner.config.room_say_log_file,
            "room_say_log_dir": self.runner.config.room_say_log_dir,
        }

        if client is None:
            snapshot.update(
                {
                    "client_active": False,
                    "current_room_id": "",
                    "current_line_id": "",
                    "current_room_site": "",
                    "connected_ws_line_id": "",
                    "bootstrap_room_id": "",
                    "bootstrap_room_pwd": "",
                }
            )
            return snapshot

        state = client.state
        snapshot.update(
            {
                "client_active": True,
                "current_room_id": str(state.current_room_id or ""),
                "current_line_id": str(state.current_line_id or ""),
                "current_room_site": str(state.current_room_site or ""),
                "connected_ws_line_id": str(state.connected_ws_line_id or ""),
                "bootstrap_room_id": str(state.bootstrap_room_id or ""),
                "bootstrap_room_pwd": str(state.bootstrap_room_pwd or ""),
                "joined_hall": bool(state.joined_hall),
                "success": bool(state.success),
                "auth_idx": int(state.auth_idx),
            }
        )
        return snapshot

    def update_room_say_permission(self, name: str, enabled: bool) -> dict[str, Any]:
        """更新 room_say_llm_allowlist 并写回 app_secrets.json。"""
        cleaned_name = str(name or "").strip()
        if not cleaned_name:
            return {"ok": False, "error": "name 不能为空"}

        try:
            current = load_app_secrets(force_reload=True)
            if not isinstance(current, dict):
                current = {}

            allowlist_raw = current.get("room_say_llm_allowlist", [])
            if isinstance(allowlist_raw, list):
                allowlist = [str(item).strip() for item in allowlist_raw if str(item).strip()]
            elif isinstance(allowlist_raw, str):
                allowlist = [part.strip() for part in allowlist_raw.split(",") if part.strip()]
            else:
                allowlist = []

            if enabled:
                if cleaned_name not in allowlist:
                    allowlist.append(cleaned_name)
            else:
                allowlist = [item for item in allowlist if item != cleaned_name]

            current["room_say_llm_allowlist"] = allowlist
            _SECRETS_FILE.write_text(json.dumps(current, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            load_app_secrets(force_reload=True)

            return {
                "ok": True,
                "name": cleaned_name,
                "enabled": bool(enabled),
                "room_say_llm_allowlist": allowlist,
            }
        except Exception as exc:
            return {"ok": False, "error": f"更新权限失败: {exc}"}

    def get_room_say_allowlist(self) -> list[str]:
        """读取 app_secrets.json 中的 room_say_llm_allowlist。"""
        current = load_app_secrets(force_reload=True)
        if not isinstance(current, dict):
            return []

        allowlist_raw = current.get("room_say_llm_allowlist", [])
        if isinstance(allowlist_raw, list):
            return [str(item).strip() for item in allowlist_raw if str(item).strip()]
        if isinstance(allowlist_raw, str):
            return [part.strip() for part in allowlist_raw.split(",") if part.strip()]
        return []

    def run(self) -> None:
        """启动机器人主循环和管理接口。"""
        bot_thread = threading.Thread(target=self.runner.run, daemon=True, name="bot-main-loop")
        bot_thread.start()

        self._server = _AdminHttpServer((self.host, self.port), self)
        print(f"admin_api_started=http://{self.host}:{self.port}/api/roomsay")
        try:
            self._server.serve_forever()
        except KeyboardInterrupt:
            print("admin_api_stopped=keyboard_interrupt")
        finally:
            self._server.server_close()
            self.runner.close()
