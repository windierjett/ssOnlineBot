import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict, Set


_SECRETS_FILE = Path(__file__).resolve().parent.parent / "app_secrets.json"
_secrets_cache: Dict[str, Any] = {}
_secrets_mtime_ns: int | None = None


def _default_room_say_allowlist() -> Set[str]:
    return {
        "风儿吹吹",
        "安若素",
        "李洛",
        "折霜枝",
        "山羊绰菀",
        "提灯映桃花",
    }


def load_app_secrets(force_reload: bool = False) -> Dict[str, Any]:
    """加载项目根目录的统一敏感配置文件。支持按文件变更自动刷新。"""
    global _secrets_cache, _secrets_mtime_ns

    if not _SECRETS_FILE.exists():
        _secrets_cache = {}
        _secrets_mtime_ns = None
        return {}

    try:
        current_mtime_ns = _SECRETS_FILE.stat().st_mtime_ns
    except OSError:
        _secrets_cache = {}
        _secrets_mtime_ns = None
        return {}

    if not force_reload and _secrets_mtime_ns == current_mtime_ns:
        return _secrets_cache

    try:
        data = json.loads(_SECRETS_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        _secrets_cache = {}
        _secrets_mtime_ns = current_mtime_ns
        return {}

    if not isinstance(data, dict):
        _secrets_cache = {}
        _secrets_mtime_ns = current_mtime_ns
        return {}

    _secrets_cache = data
    _secrets_mtime_ns = current_mtime_ns
    return _secrets_cache


def save_app_secrets(data: Dict[str, Any]) -> bool:
    """将敏感配置写回到 app_secrets.json 并刷新缓存。

    返回 True 表示写入成功，False 表示失败。
    """
    global _secrets_cache, _secrets_mtime_ns
    try:
        _SECRETS_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        # force reload to update in-memory cache and mtime
        _secrets_cache = data
        try:
            _secrets_mtime_ns = _SECRETS_FILE.stat().st_mtime_ns
        except OSError:
            _secrets_mtime_ns = None
        return True
    except Exception:
        return False


def get_secret(name: str, default: str = "") -> str:
    """读取指定敏感配置项，缺失时返回默认值。"""
    value = load_app_secrets().get(name, default)
    return str(value) if value is not None else default


def get_secret_set(name: str, default: Set[str] | None = None) -> Set[str]:
    """读取字符串集合配置，支持数组或逗号分隔字符串。"""
    fallback = set(default or set())
    value = load_app_secrets().get(name)

    if value is None:
        return fallback

    if isinstance(value, list):
        return {str(item).strip() for item in value if str(item).strip()}

    if isinstance(value, str):
        return {part.strip() for part in value.split(",") if part.strip()}

    return fallback


@dataclass
class AppConfig:
    """应用配置对象。

    采用集中配置的方式，避免散落的全局常量难以维护。
    后续如果要切换不同环境（测试/生产），可以基于该对象做分层覆盖。
    """

    login_user: str = field(default_factory=lambda: get_secret("login_user", ""))
    login_pwd: str = field(default_factory=lambda: get_secret("login_pwd", ""))

    room_name: str = "Test"
    room_area: int = 3000
    room_gt: int = 10001

    login_url: str = "https://www.ss911.cn/NewLogin_p.ss"
    home_url: str = "https://www.ss911.cn/"
    game_url: str = "https://t1.ss911.cn/IndexH5.ss"
    ws_url: str = ""
    ws_device: str = "html5:1447"

    friend_http_path: str = "/User/MyF.ss"

    keepalive_seconds: int = 60
    reconnect_seconds: int = 1

    # 仅这些昵称的 RoomSay 会转发给大模型。
    room_say_llm_allowlist: Set[str] = field(
        default_factory=_default_room_say_allowlist
    )

    monitor_friends_enabled: bool = True
    friend_monitor_interval_seconds: int = 60
    friend_monitor_log_file: str = "friend_status_changes.txt"
    friend_ws_fallback_enabled: bool = False

    self_status_interval_seconds: int = 180
    self_status_log_file: str = "self_room_status.txt"

    room_say_log_enabled: bool = True
    room_say_log_dir: str = "room_say_logs"
    room_say_name_to_file: Dict[str, str] = field(
        default_factory=lambda: {
            "风儿吹吹": "room_say_fenger.txt",
            "安若素": "room_say_anruosu.txt",
        }
    )
    room_say_log_file: str = "room_say.txt"

    # 当前账号昵称：用于过滤“自己发送后服务器回推”的 RoomSay，避免重复触发大模型。
    self_nickname: str = "夏凌依"

    filter_loudspeaker_logs: bool = True

    def get_room_say_llm_allowlist(self) -> Set[str]:
        """运行时动态读取允许触发 LLM 的昵称白名单。"""
        return get_secret_set("room_say_llm_allowlist", self.room_say_llm_allowlist)


@dataclass
class ReconnectContext:
    """跨连接上下文。

    该对象承载“一次性重连参数”和“登录 token 缓存”，
    用于实现跨线路平滑切换时不重新登录。
    """

    next_ws_url: str = ""
    next_line_id: str = ""
    next_join_room_id: str = ""
    next_join_room_pwd: str = ""
    fast_reconnect_once: bool = False

    jump_reconnect_count: int = 0
    max_jump_reconnect: int = 4

    # 缓存登录产生的 fv 参数，供平滑重连复用。
    fv_cache: Dict[str, str] = field(default_factory=dict)

    # 缓存登录后的 Cookie，供跳线重连时复用同一会话身份。
    cookie_cache: Dict[str, str] = field(default_factory=dict)
