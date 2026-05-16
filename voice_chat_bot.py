# -*- coding: utf-8 -*-
"""
Agora 语音房间机器人 - 开麦播放指定音频

功能：
1. 获取 Agora Token（通过 t1.ss911.cn）
2. 发现 Agora 边缘节点（WebSocket 信令服务器）
3. 通过 WebSocket 信令加入频道（join_v3）
4. 发布本地音频流（publish），播放 WAV/MP3 文件
5. 频道内其他人可听到播放的音频

协议说明：
- 信令层：WebSocket 连接到 Agora 边缘节点，使用自定义 JSON 协议
  (_type: join_v3 / publish / ping / leave 等)
- 媒体层：WebRTC (SRTP/UDP)，通过 aiortc 实现音频编码和推流
- 不需要接收远端音频，仅需发送
"""

import requests
import json
import time
import threading
import os
import sys
import uuid
import struct
import hashlib
import hmac
import base64
import wave
import argparse
import subprocess
from typing import Optional, Dict, Any, List, Callable

# ============================================================================
# Playwright headless browser（备选方案：当 REST API 不可用时）
# ============================================================================
try:
    from playwright.sync_api import sync_playwright
    HAS_PLAYWRIGHT = True
except ImportError:
    HAS_PLAYWRIGHT = False

# ============================================================================
# 轻量级 WebSocket 信令客户端（基于 websocket-client，同步 API）
# ============================================================================
try:
    import websocket
    HAS_WS = True
except ImportError:
    HAS_WS = False
    print("⚠️ 请安装 websocket-client: pip install websocket-client")

# ============================================================================
# WebRTC 媒体层（基于 aiortc，异步 API）
# ============================================================================
try:
    import asyncio
    HAS_ASYNCIO = True
except ImportError:
    HAS_ASYNCIO = False

try:
    from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack
    from aiortc.contrib.media import MediaPlayer, MediaRelay
    from aiortc.mediastreams import AudioStreamTrack
    import av
    HAS_AIORTC = True
except ImportError:
    HAS_AIORTC = False
    print("⚠️ 请安装 aiortc: pip install aiortc av")


# ============================================================================
# 工具函数
# ============================================================================

def gen_random_hex(length: int = 6) -> str:
    """生成随机 hex 字符串（用于 _id）"""
    return uuid.uuid4().hex[:length]

def gen_session_id() -> str:
    """生成 session_id（32位大写 hex）"""
    return uuid.uuid4().hex.upper()

def gen_process_id() -> str:
    """生成 process_id"""
    return f"process-{uuid.uuid4()}"

def now_ms() -> int:
    """当前毫秒时间戳"""
    return int(time.time() * 1000)


# ============================================================================
# Agora 信令客户端 —— 通过 WebSocket 与 Agora 边缘节点通信
# ============================================================================

class AgoraSignalingClient:
    """
    Agora WebSocket 信令客户端
    
    负责与 Agora 边缘节点建立 WebSocket 连接，发送/接收信令消息。
    支持的 _type：
      - join_v3:   加入频道
      - publish:   发布/取消发布媒体流
      - ping:      心跳保活
      - leave:     离开频道
    """

    def __init__(self, app_id: str, channel_name: str, uid: int,
                 token: str, codec: str = "vp8"):
        self.app_id = app_id
        self.channel_name = channel_name
        self.uid = uid
        self.token = token
        self.codec = codec
        self.session_id = gen_session_id()
        self.process_id = gen_process_id()

        self.ws: Optional[websocket.WebSocketApp] = None
        self.ws_thread: Optional[threading.Thread] = None
        self._connected = threading.Event()
        self._join_done = threading.Event()
        self._join_response: Optional[Dict] = None
        self._stop = threading.Event()
        self._ping_thread: Optional[threading.Thread] = None

        # 消息回调
        self._msg_callbacks: Dict[str, Callable] = {}
        self._response_futures: Dict[str, threading.Event] = {}

        # join_v3 响应中的关键数据
        self.cid: Optional[int] = None
        self.edge_uid: Optional[int] = None
        self.cert: Optional[str] = None
        self.ticket: Optional[str] = None
        self.ice_ufrag: Optional[str] = None
        self.ice_pwd: Optional[str] = None
        self.dtls_fingerprints: Optional[List[Dict]] = None

    # ---- 消息路由 ----

    def _on_ws_open(self, ws):
        print(f"🔗 WebSocket 已连接")
        self._connected.set()

    def _on_ws_message(self, ws, raw_text: str):
        try:
            msg = json.loads(raw_text)
            msg_type = msg.get("_type", "")
            msg_id = msg.get("_id", "")

            # 打印关键消息
            if msg_type in ("join_v3", "publish_result", "ping_back", "leave_result"):
                print(f"📩 收到 [{msg_type}] id={msg_id}")
            elif msg_type in ("wrtc_stats", "traffic_stats", "ping"):
                pass  # 静默高频消息
            else:
                print(f"📩 收到 [{msg_type}] id={msg_id}")

            # 触发 join_v3 响应
            if msg_type == "join_v3" and "_message" in msg:
                ap_resp = msg["_message"].get("ap_response", {})
                if ap_resp.get("code") == 0:
                    self._join_response = msg["_message"]
                    self.cid = ap_resp.get("cid")
                    self.edge_uid = ap_resp.get("uid")
                    self.cert = ap_resp.get("cert")
                    self.ticket = ap_resp.get("ticket")
                    # 解析 ORTC/ICE 参数
                    ortc = msg["_message"].get("ortc", {})
                    ice_params = ortc.get("iceParameters", {})
                    self.ice_ufrag = ice_params.get("iceUfrag", "")
                    self.ice_pwd = ice_params.get("icePwd", "")
                    dtls = ortc.get("dtlsParameters", {})
                    self.dtls_fingerprints = dtls.get("fingerprints", [])
                    print(f"✅ 加入频道成功! uid={self.edge_uid}, cid={self.cid}")
                    self._join_done.set()
                else:
                    print(f"❌ 加入频道失败: code={ap_resp.get('code')}")

            # 触发回调
            cb = self._msg_callbacks.get(msg_type)
            if cb:
                cb(msg)

            # 触发等待中的 future
            if msg_id and msg_id in self._response_futures:
                self._response_futures[msg_id].set()

        except json.JSONDecodeError:
            pass

    def _on_ws_error(self, ws, error):
        print(f"❌ WebSocket 错误: {error}")

    def _on_ws_close(self, ws, close_status_code, close_msg):
        print(f"🔌 WebSocket 已关闭 code={close_status_code}")
        self._connected.clear()

    # ---- 对外 API ----

    def connect(self, ws_url: str, timeout: float = 15.0) -> bool:
        """连接到 Agora 边缘节点"""
        if not HAS_WS:
            print("❌ websocket-client 未安装")
            return False

        print(f"🔌 连接信令服务器: {ws_url}")
        self.ws = websocket.WebSocketApp(
            ws_url,
            on_open=self._on_ws_open,
            on_message=self._on_ws_message,
            on_error=self._on_ws_error,
            on_close=self._on_ws_close,
        )

        self.ws_thread = threading.Thread(
            target=self.ws.run_forever,
            kwargs={"ping_interval": 30, "ping_timeout": 10},
            daemon=True
        )
        self.ws_thread.start()

        if not self._connected.wait(timeout=timeout):
            print("❌ WebSocket 连接超时")
            return False

        return True

    def send_raw(self, payload: Dict) -> bool:
        """发送原始 JSON 消息"""
        if not self.ws or not self._connected.is_set():
            return False
        try:
            text = json.dumps(payload, ensure_ascii=False)
            self.ws.send(text)
            return True
        except Exception as e:
            print(f"❌ 发送失败: {e}")
            return False

    def join_channel(self, timeout: float = 20.0) -> bool:
        """
        发送 join_v3 消息加入频道
        
        消息格式参考前端 Agora SDK 4.24.2 内部信令协议
        """
        self._join_done.clear()

        msg_id = gen_random_hex(6)
        join_msg = {
            "_id": msg_id,
            "_type": "join_v3",
            "_message": {
                "p2p_id": 2,
                "session_id": self.session_id,
                "app_id": self.app_id,
                "channel_name": self.channel_name,
                "channel_key": self.token,
                "codec": self.codec,
                "mode": "rtc",
                "role": "host",
                "sdk_version": "4.24.2",
                "process_id": self.process_id,
                "join_ts": now_ms(),
                "has_changed_gateway": True,
                "features": {"rejoin": True},
                "details": {},
                "extend": "",
                "ortc": {
                    "version": "2",
                    "iceParameters": {
                        "iceUfrag": "",
                        "icePwd": ""
                    },
                    "dtlsParameters": {
                        "fingerprints": [
                            {
                                "hashFunction": "sha-256",
                                "value": "00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:"
                                        "00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00"
                            }
                        ]
                    },
                    "rtpCapabilities": {
                        "send": {
                            "audioCodecs": [],
                            "audioExtensions": [],
                            "videoCodecs": [],
                            "videoExtensions": []
                        }
                    }
                },
                "attributes": {
                    "userAttributes": {
                        "enableAudioMetadata": False,
                        "enableAudioPts": False,
                        "enableAutCC": True,
                        "enableAutFeedback": True,
                        "enableDataStream2": False,
                        "enableDualStreamFlag": False,
                        "enableInstantVideo": False,
                        "enableLossbasedBwe": True,
                        "enableNetworkQualityProbe": False,
                        "enablePreallocPC": False,
                        "enablePubRTX": True,
                        "enablePubTWCC": False,
                        "enablePublishedUserList": True,
                        "enableQualityFallback": False,
                        "enableRTX": True,
                        "enableSubRTX": True,
                        "enableSubTWCC": True,
                        "enableUserAutoRebalanceCheck": True,
                        "enableUserLicenseCheck": True,
                        "enableVosFallback": False,
                        "enableXR": True,
                        "maxSubscription": 50
                    }
                },
                "browser": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/147.0.0.0 Safari/537.36 Edg/147.0.0.0"
                )
            }
        }

        print(f"📤 发送 join_v3: channel={self.channel_name}, uid={self.uid}")
        if not self.send_raw(join_msg):
            return False

        if not self._join_done.wait(timeout=timeout):
            print("❌ join_v3 响应超时")
            return False

        # 启动心跳
        self._start_ping()
        return True

    def publish_audio(self, state: str = "offer", sdp: str = "") -> bool:
        """
        发送 publish 消息（开麦）
        
        参数:
            state: "offer" 发布 / "close" 取消发布
            sdp: WebRTC SDP（首次发布时提供 localDescription.sdp）
        """
        msg_id = gen_random_hex(6)
        pub_msg = {
            "_id": msg_id,
            "_type": "publish",
            "_message": {
                "p2p_id": 2,
                "session_id": self.session_id,
                "state": state,
                "ortc": {
                    "stream_type": "audio",
                    "attributes": {
                        "hq": False,
                        "lq": False,
                        "stereo": False,
                        "speech": False
                    }
                }
            }
        }

        # 如果有 SDP，附加上
        if sdp:
            pub_msg["_message"]["ortc"]["sdp"] = sdp

        print(f"📤 发送 publish (state={state})")
        return self.send_raw(pub_msg)

    def leave_channel(self) -> bool:
        """发送 leave 消息"""
        msg_id = gen_random_hex(6)
        leave_msg = {
            "_id": msg_id,
            "_type": "leave",
            "_message": {
                "session_id": self.session_id
            }
        }
        print(f"📤 发送 leave")
        return self.send_raw(leave_msg)

    # ---- 心跳 ----

    def _start_ping(self):
        """启动心跳线程"""
        if self._ping_thread and self._ping_thread.is_alive():
            return
        self._stop.clear()
        self._ping_thread = threading.Thread(target=self._ping_loop, daemon=True)
        self._ping_thread.start()

    def _ping_loop(self):
        """每 5 秒发送 ping"""
        while not self._stop.is_set():
            time.sleep(5)
            if not self._connected.is_set():
                break
            ping_msg = {
                "_id": gen_random_hex(6),
                "_type": "ping"
            }
            self.send_raw(ping_msg)

    def disconnect(self):
        """断开连接"""
        self._stop.set()
        if self._ping_thread:
            self._ping_thread.join(timeout=2)
        try:
            self.leave_channel()
        except:
            pass
        if self.ws:
            self.ws.close()
        self._connected.clear()

    def is_connected(self) -> bool:
        return self._connected.is_set()

    def is_joined(self) -> bool:
        return self._join_done.is_set()


# ============================================================================
# 音频文件音轨（aiortc）- 将 WAV/MP3 文件转为 WebRTC AudioTrack
# ============================================================================

if HAS_AIORTC:

    class AudioFileTrack(AudioStreamTrack):
        """
        从音频文件读取帧，作为 WebRTC AudioStreamTrack 输出
        
        支持的格式: WAV (通过 wave 模块), 其他格式通过 av (ffmpeg)
        音频会被重采样为 48000Hz / 1ch / s16（Opus 编码要求的格式）
        """

        kind = "audio"

        def __init__(self, file_path: str, volume: float = 1.0):
            super().__init__()
            self.file_path = file_path
            self.volume = max(0.0, min(2.0, volume))
            self._start_time: Optional[float] = None
            self._sample_rate = 48000
            self._samples: Optional[bytes] = None
            self._sample_width = 2  # 16-bit
            self._channels = 1
            self._position = 0
            self._frame_size = 960  # 20ms @ 48kHz (Opus 标准帧大小)
            self._finished = False
            self._load_file()

        def _load_file(self):
            """加载音频文件并转换为 48kHz/1ch/s16 PCM"""
            ext = os.path.splitext(self.file_path)[1].lower()

            if ext == '.wav':
                self._load_wav()
            else:
                self._load_via_av()

            print(f"🎵 音频已加载: {len(self._samples)} 采样 @ {self._sample_rate}Hz "
                  f"(≈{len(self._samples)/self._sample_rate:.1f}s)")

        def _load_wav(self):
            """通过 wave 模块加载 WAV"""
            with wave.open(self.file_path, 'rb') as wf:
                self._sample_rate = wf.getframerate()
                self._channels = wf.getnchannels()
                self._sample_width = wf.getsampwidth()
                raw = wf.readframes(wf.getnframes())

            # 转换为单声道 48kHz s16
            self._samples = self._resample_to_48k_mono(raw)

        def _load_via_av(self):
            """通过 av (PyAV/ffmpeg) 加载任意格式"""
            container = av.open(self.file_path)
            audio_stream = container.streams.audio[0]

            # 重采样器: -> s16, 48000Hz, mono
            resampler = av.AudioResampler(
                format='s16',
                layout='mono',
                rate=48000
            )

            frames = []
            for frame in container.decode(audio=0):
                for resampled in resampler.resample(frame):
                    frames.append(resampled.to_ndarray().tobytes())

            self._sample_rate = 48000
            self._channels = 1
            self._sample_width = 2
            self._samples = b''.join(frames)
            container.close()

        def _resample_to_48k_mono(self, raw: bytes) -> bytes:
            """将 PCM 数据重采样为 48kHz 单声道 s16（简易线性插值）"""
            if self._sample_width == 1:
                fmt = f"<{len(raw)}b"
                arr = struct.unpack(fmt, raw)
                arr = [s * 256 for s in arr]  # 8-bit -> 16-bit
            elif self._sample_width == 2:
                fmt = f"<{len(raw)//2}h"
                arr = list(struct.unpack(fmt, raw))
            else:
                return raw

            # 立体声 -> 单声道
            if self._channels == 2:
                arr = [(arr[i] + arr[i+1]) // 2 for i in range(0, len(arr)-1, 2)]

            # 简单重采样到 48kHz（取最近邻）
            if self._sample_rate != 48000:
                ratio = 48000 / self._sample_rate
                new_len = int(len(arr) * ratio)
                arr = [arr[min(int(i / ratio), len(arr)-1)] for i in range(new_len)]

            # 应用音量
            if self.volume != 1.0:
                arr = [max(-32768, min(32767, int(s * self.volume))) for s in arr]

            return struct.pack(f"<{len(arr)}h", *arr)

        async def recv(self):
            """获取下一帧音频数据（20ms @ 48kHz = 960 samples）"""
            if self._finished:
                # 返回静音帧表示结束
                from aiortc.mediastreams import AudioFrame
                import fractions
                frame = AudioFrame(format='s16', layout='mono', samples=0)
                frame.sample_rate = 48000
                frame.pts = 0
                frame.time_base = fractions.Fraction(1, 48000)
                return frame

            if self._start_time is None:
                self._start_time = time.time()

            offset = self._position * self._frame_size * self._sample_width
            remaining = len(self._samples) - offset

            if remaining <= 0:
                self._finished = True
                from aiortc.mediastreams import AudioFrame
                import fractions
                frame = AudioFrame(format='s16', layout='mono', samples=0)
                frame.sample_rate = 48000
                frame.pts = 0
                frame.time_base = fractions.Fraction(1, 48000)
                return frame

            # 读取一帧
            frame_samples = min(self._frame_size, remaining // self._sample_width)
            frame_bytes = self._samples[offset:offset + frame_samples * self._sample_width]

            # 补齐到 960 samples
            if frame_samples < self._frame_size:
                pad_bytes = b'\x00' * ((self._frame_size - frame_samples) * self._sample_width)
                frame_bytes += pad_bytes

            self._position += 1

            from aiortc.mediastreams import AudioFrame
            import fractions
            pts = int((time.time() - self._start_time) * 48000)
            frame = AudioFrame(format='s16', layout='mono', samples=self._frame_size)
            frame.planes[0].update(frame_bytes)
            frame.sample_rate = 48000
            frame.pts = pts
            frame.time_base = fractions.Fraction(1, 48000)
            return frame

        def is_finished(self) -> bool:
            return self._finished


# ============================================================================
# Agora 语音机器人 —— 整合信令 + 媒体
# ============================================================================

class AgoraVoiceBot:
    """
    Agora 语音房间机器人
    
    使用流程:
        bot = AgoraVoiceBot()
        bot.get_token(channel, uid, u, i)
        bot.get_edge_servers()
        bot.join_and_publish("path/to/audio.wav")
    """

    def __init__(self, app_id: str = "14afb711d60047e899beaca69932a16a"):
        self.app_id = app_id
        self.token_url = "https://t1.ss911.cn/Conn/VoiceKey.ss"

        # 状态
        self.token: Optional[str] = None
        self.channel: Optional[str] = None
        self.uid: Optional[int] = None
        self.webrtc_nodes: List[Dict] = []
        self.webrtc_cert: Optional[str] = None
        self.webrtc_cid: Optional[int] = None

        # 信令客户端
        self.signaling: Optional[AgoraSignalingClient] = None

        # 媒体（aiortc）
        self._pc = None  # type: ignore
        self._audio_track = None  # type: ignore
        self._publish_done = threading.Event()
        self._playing = threading.Event()
        self._loop = None  # type: ignore
        self._loop_thread: Optional[threading.Thread] = None
        self._pw_thread: Optional[threading.Thread] = None
        self._pw_success = False

        # 游戏 WebSocket 回调（用于发送 JoinVoice/LeaveVoice 等协议消息）
        self._game_ws_send: Optional[Callable[[Dict], bool]] = None

    # ========================================================================
    # 设置游戏 WebSocket 回调
    # ========================================================================

    def set_game_ws_callback(self, sender: Callable[[Dict], bool]):
        """
        设置游戏 WebSocket 发送回调
        
        用于在加入/离开语音频道时发送 JoinVoice 协议消息
        """
        self._game_ws_send = sender

    def _send_game_ws(self, payload: Dict):
        """通过游戏 WebSocket 发送消息"""
        if self._game_ws_send:
            try:
                self._game_ws_send(payload)
            except Exception as e:
                print(f"⚠️  游戏 WS 发送失败: {e}")

    # ========================================================================
    # 步骤 1: 获取 Token
    # ========================================================================

    def get_token(self, channel: str, uid: int, u: str, i: str) -> Optional[str]:
        """从 t1.ss911.cn 获取 Agora Token"""
        print(f"🔄 获取Token... channel={channel}, uid={uid}")
        self.channel = channel
        self.uid = uid

        try:
            resp = requests.get(self.token_url, params={
                "channel": channel, "uid": uid, "u": u, "i": i,
            }, timeout=10)
            resp.raise_for_status()
            data = resp.json()

            if "key" in data:
                self.token = data["key"]
                print(f"✅ Token 获取成功: {self.token[:20]}...")
                return self.token
            else:
                print(f"❌ Token 响应无 key 字段: {data}")
                return None
        except Exception as e:
            print(f"❌ 获取Token失败: {e}")
            return None

    # ========================================================================
    # 步骤 2: 发现边缘节点
    # ========================================================================

    def get_edge_servers(self) -> bool:
        """
        获取 Agora 边缘节点列表（先 agora.io，失败则回退 sd-rtn.com）

        参考前端 Agora SDK 内部的 transpond API：
          https://webrtc2-ap-web-1.agora.io/api/v2/transpond/webrtc?v=2
          https://webrtc2-2.ap.sd-rtn.com/api/v2/transpond/webrtc?v=2

        WebSocket 信令地址格式: wss://{ip}.edge.{domain}:{port}/
        """
        if not self.token or not self.channel or not self.uid:
            print("❌ 请先获取 token")
            return False

        # 两个 API 端点，依次尝试
        endpoints = [
            ("agora.io", "https://webrtc2-ap-web-1.agora.io/api/v2/transpond/webrtc?v=2", "edge.agora.io"),
            ("sd-rtn.com", "https://webrtc2-2.ap.sd-rtn.com/api/v2/transpond/webrtc?v=2", "edge.sd-rtn.com"),
        ]

        # sid 使用固定格式（参考前端 Agora SDK 内部生成逻辑，32位 hex）
        sid = "065C010D12864CE9A911F73907AEF9EE"
        ts = now_ms()

        payload = {
            "appid": self.app_id,
            "client_ts": ts,
            "opid": ts % 10 ** 12,
            "sid": sid,
            "request_bodies": [{
                "uri": 22,
                "buffer": {
                    "cname": self.channel,
                    "detail": {
                        "11": "CN,GLOBAL",
                        "17": "1",
                        "22": "CN,GLOBAL"
                    },
                    "key": self.token,
                    "service_ids": [11, 26],
                    "uid": self.uid
                }
            }]
        }

        headers = {
            "Content-Type": "application/json",
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/147.0.0.0 Safari/537.36 Edg/147.0.0.0"
            ),
        }

        for label, api_url, edge_domain in endpoints:
            try:
                print(f"🔌 尝试 {label} 发现边缘节点...")
                resp = requests.post(api_url, json=payload,
                                    headers=headers, timeout=15)
                resp.raise_for_status()
                data = resp.json()

                nodes = []
                for item in data.get("response_body", []):
                    buf = item.get("buffer", {})
                    self.webrtc_cert = buf.get("cert")
                    self.webrtc_cid = buf.get("cid")
                    for edge in buf.get("edges_services", []):
                        ip = edge["ip"]
                        port = edge["port"]
                        # 格式: wss://171-43-252-19.edge.agora.io:4705/
                        # DNS 标签中不能用点号，必须转为横线
                        ip_dns = ip.replace(".", "-")
                        ws_url = f"wss://{ip_dns}.{edge_domain}:{port}/"
                        nodes.append({
                            "ip": ip,
                            "ip_dns": ip_dns,
                            "port": port,
                            "ws_url": ws_url,
                            "domain": edge_domain,
                        })

                if nodes:
                    self.webrtc_nodes = nodes
                    print(f"✅ 发现 {len(nodes)} 个边缘节点:")
                    for n in nodes[:5]:
                        print(f"   - {n['ws_url']}")
                    return True
                else:
                    print(f"⚠️  {label} 返回空节点列表")

            except requests.exceptions.HTTPError as e:
                print(f"⚠️  {label} HTTP {e.response.status_code}: {e.response.text[:100] if e.response.text else 'N/A'}")
            except requests.exceptions.ConnectionError as e:
                print(f"⚠️  {label} 连接失败: {e}")
            except requests.exceptions.Timeout:
                print(f"⚠️  {label} 请求超时")
            except Exception as e:
                print(f"⚠️  {label} 异常: {e}")

        print("❌ 所有端点均无法获取边缘节点")
        return False

    # ========================================================================
    # 步骤 3: 加入频道 + 开麦 + 播放音频
    # ========================================================================

    def join_and_publish(self, audio_file: Optional[str] = None,
                         volume: float = 1.0) -> bool:
        """
        主流程：
          优先尝试 REST API 发现边缘节点 + WebSocket 信令 + aiortc 推流
          如果节点发现失败，回退到 Playwright headless 浏览器方案
        """
        if not self.token or not self.channel or not self.uid:
            print("❌ 缺少必要参数 (token/channel/uid)")
            return False

        # 发送 JoinVoice 消息到游戏服务器（格式参考前端 AgoraCp.record/stop）
        self._send_game_ws({"cmd": "JoinVoice", "join": "1"})

        # ---- 尝试路径 A: REST API 边缘发现 + WebSocket 信令 ----
        if self.get_edge_servers():
            return self._join_via_signaling(audio_file, volume)

        # ---- 路径 B: Playwright headless 浏览器 ----
        print("\n⚠️  REST API 节点发现失败，启用 Playwright headless 浏览器方案...")
        return self._join_via_playwright(audio_file, volume)

    def _join_via_signaling(self, audio_file: Optional[str] = None,
                            volume: float = 1.0) -> bool:
        """通过 WebSocket 信令加入频道"""
        node = self.webrtc_nodes[0]
        ws_url = node["ws_url"]

        self.signaling = AgoraSignalingClient(
            app_id=self.app_id,
            channel_name=self.channel,
            uid=self.uid,
            token=self.token,
        )

        if not self.signaling.connect(ws_url):
            return False

        if not self.signaling.join_channel():
            print("❌ 加入频道失败")
            self.signaling.disconnect()
            return False

        print(f"✅ 已加入频道! cid={self.signaling.cid}, "
              f"edge_uid={self.signaling.edge_uid}")

        if not audio_file:
            print("ℹ️ 未指定音频文件，仅加入频道（已开麦）")
            return True

        if not HAS_AIORTC:
            print("=" * 50)
            print("⚠️  aiortc 未安装，无法推流音频")
            print("   安装: pip install aiortc av")
            print("   已成功加入频道（信令层），但无媒体流")
            print("=" * 50)
            return True

        return self._publish_audio_file(audio_file, volume)

    # ========================================================================
    # 路径 B: Playwright headless 浏览器（当 REST API 不可用时）
    # ========================================================================

    def _join_via_playwright(self, audio_file: Optional[str] = None,
                             volume: float = 1.0) -> bool:
        """通过 Playwright 控制 headless Chrome，加载 Agora Web SDK 加入频道并推流"""
        if not HAS_PLAYWRIGHT:
            print("=" * 50)
            print("❌ Playwright 未安装，且边缘节点 REST API 也不可用")
            print("   请安装: pip install playwright && playwright install chromium")
            print("=" * 50)
            return False

        # 构建本地 HTML 页面的 file:// URL
        page_file = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "voice", "agora_bot_page.html")
        if not os.path.exists(page_file):
            print(f"❌ 找不到 Agora 页面文件: {page_file}")
            return False

        page_url = "file:///" + page_file.replace("\\", "/")

        print(f"🌐 启动 headless 浏览器...")
        self._playing.clear()
        self._publish_done.clear()
        self._pw_success = False  # 线程内标记真正成功

        self._pw_thread = threading.Thread(
            target=self._run_playwright,
            args=(page_url, audio_file, volume),
            daemon=True
        )
        self._pw_thread.start()

        # 等待加入成功或线程结束
        while not self._playing.is_set() and not self._publish_done.is_set():
            time.sleep(0.2)

        if not self._playing.is_set() and self._publish_done.is_set():
            # 线程已结束但 _playing 没设置 = 失败
            print("❌ Playwright 线程异常退出")
            return False

        return self._playing.is_set()

    def _run_playwright(self, page_url: str, audio_file: Optional[str],
                        volume: float):
        """在后台线程中运行 Playwright"""
        joined_ok = False  # 跟踪是否真正加入成功
        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(
                    headless=True,
                    args=[
                        "--no-sandbox",
                        "--disable-setuid-sandbox",
                        "--use-fake-device-for-media-stream",
                        "--disable-web-security",
                        "--allow-file-access-from-files",
                        "--autoplay-policy=no-user-gesture-required",
                    ]
                )
                context = browser.new_context(
                    permissions=["microphone"],
                    user_agent=(
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/147.0.0.0 Safari/537.36 Edg/147.0.0.0"
                    )
                )
                page = context.new_page()

                # 监听 console
                page.on("console", lambda msg: print(f"  [browser] {msg.text}"))

                print(f"  📄 加载页面: {page_url}")
                page.goto(page_url, timeout=30000)

                # 等待 AgoraRTC SDK 加载
                page.wait_for_function("typeof AgoraRTC !== 'undefined'", timeout=15000)
                print("  ✅ AgoraRTC SDK 已加载")

                # 加入频道
                print(f"  🔗 加入频道 channel={self.channel} uid={self.uid}")
                result = page.evaluate("""([appId, channel, token, uid]) => {
                    return joinChannel(appId, channel, token, uid);
                }""", [self.app_id, self.channel, self.token, self.uid])

                result = json.loads(json.dumps(result))
                if not result.get("ok"):
                    print(f"  ❌ 加入频道失败: {result.get('error')}")
                    browser.close()
                    return

                joined_ok = True  # ← 标记成功
                print("  ✅ 已加入频道")
                self._playing.set()

                if not audio_file:
                    print("  ℹ️ 无音频文件，保持频道连接...")
                    while not self._publish_done.is_set():
                        time.sleep(1)
                    browser.close()
                    return

                # 读取音频文件为 base64（绕过 file:// URL 限制）
                abs_file = os.path.abspath(audio_file)
                if not os.path.exists(abs_file):
                    print(f"  ❌ 音频文件不存在: {abs_file}")
                    browser.close()
                    return

                with open(abs_file, "rb") as f:
                    raw_bytes = f.read()
                b64_data = base64.b64encode(raw_bytes).decode("ascii")

                # 判断 MIME 类型
                ext = os.path.splitext(abs_file)[1].lower()
                mime_map = {".mp3": "audio/mpeg", ".wav": "audio/wav",
                            ".ogg": "audio/ogg", ".m4a": "audio/mp4",
                            ".aac": "audio/aac", ".flac": "audio/flac"}
                mime_type = mime_map.get(ext, "audio/mpeg")

                print(f"  🔊 开始播放: {os.path.basename(abs_file)} "
                      f"({len(raw_bytes)} bytes, {mime_type})")
                result = page.evaluate("""([b64, mime, vol]) => {
                    return playAudioFile(b64, mime, vol);
                }""", [b64_data, mime_type, volume])

                result = json.loads(json.dumps(result))
                if not result.get("ok"):
                    print(f"  ❌ 播放启动失败: {result.get('error')}")
                    browser.close()
                    return

                dur = result.get("duration", 0)
                print(f"  🔊 播放中 ({dur:.1f}s)...")

                # 轮询等待播放完成（可被 stop() 中断）
                while not self._publish_done.is_set():
                    time.sleep(0.5)
                    try:
                        if page.evaluate("isAudioFinished()"):
                            break
                    except Exception:
                        break

                # 停止并清理音频 track
                try:
                    page.evaluate("stopAudio()")
                except Exception:
                    pass

                if dur > 0:
                    print(f"  ✅ 音频播放完毕 (时长 {dur:.1f}s)")
                else:
                    print(f"  ✅ 音频播放完毕")

                # 离开频道
                page.evaluate("leaveChannel()")
                browser.close()

        except Exception as e:
            print(f"❌ Playwright 异常: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self._publish_done.set()
            # 只有真正加入成功时 _playing 才已经被设置
            # 如果异常发生在加入之前，_playing 不会被设置 → 上层会正确感知失败
            if not joined_ok and not self._playing.is_set():
                pass  # 加入失败，不设置 _playing，让上层正确报告失败

    def _publish_audio_file(self, audio_file: str, volume: float) -> bool:
        """通过 aiortc 发布音频文件到频道"""

        if not os.path.exists(audio_file):
            print(f"❌ 音频文件不存在: {audio_file}")
            return False

        # 在后台线程中运行 asyncio event loop
        self._publish_done.clear()
        self._playing.clear()

        self._loop_thread = threading.Thread(
            target=self._run_async_media,
            args=(audio_file, volume),
            daemon=True
        )
        self._loop_thread.start()

        # 等待音频开始播放
        self._playing.wait(timeout=10)

        return True

    def _run_async_media(self, audio_file: str, volume: float):
        """异步媒体线程入口"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        try:
            loop.run_until_complete(self._async_publish(audio_file, volume))
        except Exception as e:
            print(f"❌ 媒体线程异常: {e}")
            import traceback
            traceback.print_exc()
        finally:
            loop.close()

    async def _async_publish(self, audio_file: str, volume: float):
        """异步发布音频流"""
        # 创建 PeerConnection
        pc = RTCPeerConnection()
        self._pc = pc

        # 创建音频文件音轨
        audio_track = AudioFileTrack(audio_file, volume)
        self._audio_track = audio_track

        # 添加到 PeerConnection
        pc.addTrack(audio_track)

        # 创建 Offer
        offer = await pc.createOffer()
        await pc.setLocalDescription(offer)

        print(f"📤 发送 publish (SDP offer)")
        if self.signaling:
            self.signaling.publish_audio(
                state="offer",
                sdp=pc.localDescription.sdp
            )

        self._playing.set()
        print(f"🔊 开始播放: {os.path.basename(audio_file)}")

        # 等待播放完成
        while not audio_track.is_finished():
            await asyncio.sleep(0.1)

        print(f"✅ 音频播放完毕")
        self._publish_done.set()

        # 取消发布
        if self.signaling:
            self.signaling.publish_audio(state="close")

        await pc.close()

    # ========================================================================
    # 控制 API
    # ========================================================================

    def is_playing(self) -> bool:
        return self._playing.is_set() and not self._publish_done.is_set()

    def wait_for_completion(self, timeout: Optional[float] = None):
        """等待音频播放完成"""
        if self._publish_done.wait(timeout=timeout):
            print("✅ 播放完成")

    def stop(self):
        """停止播放并断开"""
        print("🛑 正在停止...")
        self._playing.clear()
        self._publish_done.set()

        # 发送 LeaveVoice 消息到游戏服务器（格式参考前端 AgoraCp.stop）
        self._send_game_ws({"cmd": "JoinVoice", "join": "0"})

        if self._audio_track:
            self._audio_track._finished = True

        if self.signaling:
            self.signaling.disconnect()
            self.signaling = None

        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)

        # Playwright 线程由 _publish_done 信号自然退出
        if hasattr(self, '_pw_thread') and self._pw_thread and self._pw_thread.is_alive():
            self._pw_thread.join(timeout=5)

    def status(self) -> Dict[str, Any]:
        """获取当前状态"""
        return {
            "channel": self.channel,
            "uid": self.uid,
            "token": (self.token[:30] + "...") if self.token else None,
            "signaling_connected": self.signaling.is_connected() if self.signaling else False,
            "joined": self.signaling.is_joined() if self.signaling else False,
            "edge_uid": self.signaling.edge_uid if self.signaling else None,
            "cid": self.signaling.cid if self.signaling else None,
            "playing": self.is_playing(),
            "nodes": len(self.webrtc_nodes)
        }
    
# ============================================================================
# 命令行入口
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Agora 语音房间机器人 - 开麦播放指定音频',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 仅加入频道（开麦但不播放）
  python voice_chat_bot.py -c 3089 -u 19348971 -U "your_u" -I "your_i"

  # 加入频道并播放 WAV 文件
  python voice_chat_bot.py -c 3089 -u 19348971 -U "your_u" -I "your_i" -f test.wav

  # 加入频道并播放 MP3 文件，音量 80%
  python voice_chat_bot.py -c 3089 -u 19348971 -U "your_u" -I "your_i" -f music.mp3 -v 0.8
        """
    )
    parser.add_argument('-c', '--channel', type=str, required=True,
                        help='频道名称 (如 3089)')
    parser.add_argument('-u', '--uid', type=int, required=True,
                        help='用户 UID (如 19348971)')
    parser.add_argument('-U', '--token-u', type=str, required=True,
                        help='用户认证参数 u (从登录信息获取)')
    parser.add_argument('-I', '--token-i', type=str, required=True,
                        help='用户认证参数 i (从登录信息获取)')
    parser.add_argument('-f', '--file', type=str, default=None,
                        help='要播放的音频文件路径 (WAV/MP3/等)')
    parser.add_argument('-v', '--volume', type=float, default=1.0,
                        help='播放音量 (0.0-2.0, 默认 1.0)')
    parser.add_argument('--app-id', type=str,
                        default='14afb711d60047e899beaca69932a16a',
                        help='Agora App ID')

    args = parser.parse_args()

    print("=" * 60)
    print("🎙️  Agora 语音房间机器人")
    print("=" * 60)
    print(f"  App ID:   {args.app_id}")
    print(f"  频  道:   {args.channel}")
    print(f"  用  户:   {args.uid}")
    print(f"  音  频:   {args.file if args.file else '(不开麦，仅加入房间)'}")
    print(f"  音  量:   {args.volume}")
    print("=" * 60)

    bot = AgoraVoiceBot(app_id=args.app_id)

    try:
        # 步骤 1: 获取 Token
        print("\n[1/3] 获取 Token...")
        token = bot.get_token(
            channel=args.channel,
            uid=args.uid,
            u=args.token_u,
            i=args.token_i,
        )
        if not token:
            print("❌ 无法获取 Token，退出")
            sys.exit(1)

        # 步骤 2+3: 发现节点 + 加入频道 + 开麦 + 播放音频
        print("\n[2/3] 连接 Agora 并开麦...")
        if not bot.join_and_publish(audio_file=args.file, volume=args.volume):
            print("❌ 加入频道失败")
            sys.exit(1)

        print("\n" + "=" * 60)
        print("📊 状态:", json.dumps(bot.status(), ensure_ascii=False, indent=2))
        print("=" * 60)

        if bot.is_playing():
            print("\n🔊 正在播放音频，等待完成... (按 Ctrl+C 中断)")
            try:
                bot.wait_for_completion()
            except KeyboardInterrupt:
                print("\n⚠️ 用户中断播放")

        elif args.file is None:
            print("\n📡 已加入频道（无音频播放），保持连接... (按 Ctrl+C 退出)")
            try:
                while bot.signaling and bot.signaling.is_connected():
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\n👋 用户退出")

        print("\n✅ 流程完成!")

    except KeyboardInterrupt:
        print("\n\n👋 用户中断")
    except Exception as e:
        print(f"\n❌ 异常: {e}")
        import traceback
        traceback.print_exc()
    finally:
        bot.stop()
        print("👋 已清理资源")


if __name__ == "__main__":
    main()
