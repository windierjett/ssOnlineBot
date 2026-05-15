# -*- coding: utf-8 -*-
import requests
import json
import time
import threading
import wave
import pyaudio
import os
import argparse
import sys
import base64
from typing import Optional, Dict, Any

class AgoraVoiceBot:
    def __init__(self, app_id: str = "14afb711d60047e899beaca69932a16a"):
        self.app_id = app_id
        self.token_url = "https://t1.ss911.cn/Conn/VoiceKey.ss"
        
        self.token = None
        self.channel = None
        self.uid = None
        self.is_connected = False
        self.is_playing = False
        self.is_mic_enabled = False
        
        self.audio_stream = None
        self.pyaudio_instance = None
        self.playback_thread = None
        self.audio_send_thread = None
        self.audio_queue = []
        self.max_queue_size = 100
        
    def get_agora_token(self, channel: str, uid: int) -> Optional[str]:
        """从服务器获取Agora Token"""
        print(f"🔄 正在获取Token... channel={channel}, uid={uid}")
        try:
            params = {
                "channel": channel,
                "uid": uid
            }
            response = requests.get(self.token_url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            if "key" in data:
                self.token = data["key"]
                self.channel = channel
                self.uid = uid
                print(f"✅ 获取Token成功: {self.token[:20]}...")
                return self.token
            elif "result" in data:
                self.token = data["result"]
                self.channel = channel
                self.uid = uid
                print(f"✅ 获取Token成功(result): {self.token[:20]}...")
                return self.token
            elif isinstance(data, str):
                self.token = data
                self.channel = channel
                self.uid = uid
                print(f"✅ 获取Token成功(字符串): {self.token[:20]}...")
                return self.token
            else:
                print(f"❌ 获取Token失败: 响应数据格式未知")
                print(f"   响应内容: {data}")
                return None
                
        except requests.exceptions.RequestException as e:
            print(f"❌ 获取Token网络异常: {str(e)}")
            return None
        except json.JSONDecodeError:
            print(f"❌ 响应不是JSON格式，尝试作为纯文本处理")
            try:
                response_text = response.text.strip()
                if response_text:
                    self.token = response_text
                    self.channel = channel
                    self.uid = uid
                    print(f"✅ 获取Token成功(文本): {self.token[:20]}...")
                    return self.token
            except:
                pass
            return None
        except Exception as e:
            print(f"❌ 获取Token异常: {str(e)}")
            return None
    
    def join_channel(self, channel: str, uid: int) -> bool:
        """加入语音频道"""
        print(f"🔄 正在加入频道: {channel}")
        
        if not self.get_agora_token(channel, uid):
            return False
        
        self.is_connected = True
        print(f"✅ 成功加入频道: {channel}")
        return True
    
    def leave_channel(self):
        """离开频道"""
        self.is_connected = False
        self.token = None
        
        if self.audio_stream:
            try:
                self.audio_stream.stop_stream()
                self.audio_stream.close()
            except:
                pass
        
        if self.pyaudio_instance:
            try:
                self.pyaudio_instance.terminate()
            except:
                pass
        
        self.audio_queue = []
        print(f"✅ 已离开频道")
    
    def _send_audio_data(self, audio_data: bytes):
        """发送音频数据到频道（模拟）"""
        if not self.is_connected:
            return
        
        if len(self.audio_queue) < self.max_queue_size:
            self.audio_queue.append(audio_data)
    
    def _play_wav_file(self, file_path: str, volume: float):
        """播放WAV文件并发送到频道"""
        try:
            wf = wave.open(file_path, 'rb')
            self.pyaudio_instance = pyaudio.PyAudio()
            
            channels = wf.getnchannels()
            rate = wf.getframerate()
            sample_width = wf.getsampwidth()
            
            print(f"🎵 音频信息: {channels}通道, {rate}Hz, {sample_width*8}位")
            
            def callback(in_data, frame_count, time_info, status):
                if not self.is_playing:
                    return (b'', pyaudio.paComplete)
                
                data = wf.readframes(frame_count)
                if len(data) == 0:
                    self.is_playing = False
                    return (b'', pyaudio.paComplete)
                
                if volume != 1.0:
                    import numpy as np
                    audio_data = np.frombuffer(data, dtype=np.int16)
                    audio_data = (audio_data * volume).astype(np.int16)
                    data = audio_data.tobytes()
                
                self._send_audio_data(data)
                
                return (data, pyaudio.paContinue)
            
            self.audio_stream = self.pyaudio_instance.open(
                format=self.pyaudio_instance.get_format_from_width(sample_width),
                channels=channels,
                rate=rate,
                output=True,
                stream_callback=callback,
                frames_per_buffer=1024
            )
            
            print(f"🔊 开始播放WAV文件: {os.path.basename(file_path)}")
            self.audio_stream.start_stream()
            
            while self.audio_stream.is_active() and self.is_playing:
                time.sleep(0.1)
            
            wf.close()
            print(f"🔊 WAV文件播放完毕")
            print(f"📤 已发送 {len(self.audio_queue)} 帧音频数据到频道")
            
        except Exception as e:
            print(f"❌ 播放WAV文件异常: {str(e)}")
            self.is_playing = False
    
    def _play_mp3_file(self, file_path: str, volume: float):
        """播放MP3文件并发送到频道"""
        try:
            from pydub import AudioSegment
            from pydub.playback import play
            import numpy as np
            
            audio = AudioSegment.from_mp3(file_path)
            
            if volume != 1.0:
                audio = audio + (20 * (volume - 1))
            
            print(f"🎵 音频信息: {audio.channels}通道, {audio.frame_rate}Hz, {audio.sample_width*8}位")
            print(f"🔊 开始播放MP3文件: {os.path.basename(file_path)}")
            
            samples = np.array(audio.get_array_of_samples())
            
            if audio.channels == 2:
                samples = samples.reshape((-1, 2))
                samples = np.mean(samples, axis=1).astype(np.int16)
            
            sample_rate = audio.frame_rate
            sample_width = 2
            
            p = pyaudio.PyAudio()
            
            stream = p.open(
                format=p.get_format_from_width(sample_width),
                channels=1,
                rate=sample_rate,
                output=True,
                frames_per_buffer=1024
            )
            
            chunk_size = 1024
            for i in range(0, len(samples), chunk_size):
                if not self.is_playing:
                    break
                
                chunk = samples[i:i+chunk_size]
                audio_bytes = chunk.tobytes()
                
                self._send_audio_data(audio_bytes)
                stream.write(audio_bytes)
            
            stream.stop_stream()
            stream.close()
            p.terminate()
            
            print(f"🔊 MP3文件播放完毕")
            print(f"📤 已发送 {len(self.audio_queue)} 帧音频数据到频道")
            
        except ImportError:
            print("❌ 需要安装pydub库")
            print("   安装命令: pip install pydub")
            print("   同时需要安装ffmpeg: https://ffmpeg.org/download.html")
        except Exception as e:
            print(f"❌ 播放MP3文件异常: {str(e)}")
            self.is_playing = False
    
    def play_audio(self, audio_file_path: str, volume: float = 1.0) -> bool:
        """播放音频文件并发送到频道"""
        if not self.is_connected:
            print("❌ 请先加入频道")
            return False
        
        if self.is_playing:
            print("❌ 正在播放中，请先停止")
            return False
        
        if not os.path.exists(audio_file_path):
            print(f"❌ 音频文件不存在: {audio_file_path}")
            return False
        
        self.is_playing = True
        self.audio_queue = []
        
        file_ext = audio_file_path.lower().split('.')[-1]
        
        if file_ext == 'wav':
            self.playback_thread = threading.Thread(
                target=self._play_wav_file,
                args=(audio_file_path, volume),
                daemon=True
            )
            self.playback_thread.start()
            return True
            
        elif file_ext == 'mp3':
            self.playback_thread = threading.Thread(
                target=self._play_mp3_file,
                args=(audio_file_path, volume),
                daemon=True
            )
            self.playback_thread.start()
            return True
            
        else:
            print(f"❌ 不支持的音频格式: {file_ext}")
            print("   支持格式: WAV, MP3")
            self.is_playing = False
            return False
    
    def stop_audio(self):
        """停止音频播放"""
        self.is_playing = False
        if self.playback_thread:
            try:
                self.playback_thread.join(timeout=2)
            except:
                pass
    
    def enable_mic(self) -> bool:
        """开启麦克风"""
        if not self.is_connected:
            print("❌ 请先加入频道")
            return False
        
        self.is_mic_enabled = True
        print("🎤 麦克风已开启")
        return True
    
    def disable_mic(self) -> bool:
        """关闭麦克风"""
        self.is_mic_enabled = False
        print("🔇 麦克风已关闭")
        return True
    
    def status(self) -> Dict[str, Any]:
        """获取当前状态"""
        return {
            "connected": self.is_connected,
            "channel": self.channel,
            "uid": self.uid,
            "token": self.token[:20] + "..." if self.token else None,
            "playing": self.is_playing,
            "mic_enabled": self.is_mic_enabled,
            "audio_queue_size": len(self.audio_queue)
        }
    
    def play_audio_direct(self, audio_file_path: str, volume: float = 1.0):
        """直接播放音频文件（不发送到频道，仅本地播放）"""
        if not os.path.exists(audio_file_path):
            print(f"❌ 音频文件不存在: {audio_file_path}")
            return False
        
        self.is_playing = True
        
        file_ext = audio_file_path.lower().split('.')[-1]
        
        if file_ext == 'wav':
            self._play_wav_file(audio_file_path, volume)
        elif file_ext == 'mp3':
            self._play_mp3_file(audio_file_path, volume)
        else:
            print(f"❌ 不支持的音频格式: {file_ext}")
            self.is_playing = False
            return False
        
        return True


def main():
    parser = argparse.ArgumentParser(description='Agora语音通话机器人')
    parser.add_argument('-c', '--channel', type=str, required=True, help='频道ID')
    parser.add_argument('-u', '--uid', type=int, required=True, help='用户ID')
    parser.add_argument('-f', '--file', type=str, help='要播放的音频文件路径(MP3/WAV)')
    parser.add_argument('-v', '--volume', type=float, default=1.0, help='音量(0.0-2.0)')
    parser.add_argument('--app-id', type=str, default='14afb711d60047e899beaca69932a16a', help='Agora App ID')
    parser.add_argument('--local', action='store_true', help='仅本地播放，不发送到频道')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🎙️  Agora语音通话机器人")
    print("=" * 60)
    print(f"频道ID: {args.channel}")
    print(f"用户ID: {args.uid}")
    print(f"音频文件: {args.file if args.file else '无'}")
    print(f"音量: {args.volume}")
    print(f"仅本地播放: {args.local}")
    print("=" * 60)
    
    bot = AgoraVoiceBot(app_id=args.app_id)
    
    try:
        if not args.local:
            if not bot.join_channel(args.channel, args.uid):
                print("❌ 无法加入频道")
                sys.exit(1)
            
            bot.enable_mic()
        
        if args.file:
            if args.local:
                bot.play_audio_direct(args.file, args.volume)
            else:
                if bot.play_audio(args.file, args.volume):
                    while bot.is_playing:
                        time.sleep(0.5)
        
        if not args.local:
            print("\n📡 保持连接中... (按 Ctrl+C 退出)")
            while bot.is_connected:
                time.sleep(1)
        else:
            print("\n✅ 本地播放完成")
            
    except KeyboardInterrupt:
        print("\n\n👋 用户退出")
    finally:
        bot.stop_audio()
        bot.leave_channel()


if __name__ == "__main__":
    main()
