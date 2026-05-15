import asyncio
import edge_tts
import os
def generate_audio(text: str, voice: str, filename: str, output_dir: str = "wav") -> None:
    """
    传入文本、语音及文件名，生成语音并保存到指定文件夹
    :param text: 需要合成的中文文本
    :param voice: 使用的语音类型，如 'zh-CN-XiaoyiNeural'
    :param filename: 输出的音频文件名（仅文件名，不含路径）
    :param output_dir: 目标文件夹路径，默认为 'wav'
    """
    # 1. 确保目标文件夹存在，如果不存在会自动创建
    os.makedirs(output_dir, exist_ok=True)

    # 2. 拼接完整的文件路径
    output_file = os.path.join(output_dir, filename)

    async def generate_audio_async() -> None:
        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(output_file)
        print(f"音频已保存为 {output_file}")

    asyncio.run(generate_audio_async())


if __name__ == '__main__':
    text = "声音还挺好听吗？一般般吧，毕竟比不上专业的，也就日常消遣，自己听着舒服就够啦。"
    voice = "zh-CN-XiaoyiNeural"
    # 只需要传入文件名即可，不需要写死路径，所有文件都会进入 output_dir 指定的文件夹
    filename = "一般般.mp3"

    generate_audio(text, voice, filename)