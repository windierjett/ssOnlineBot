from bot_core.config import AppConfig
from datetime import datetime
from collections import defaultdict

def analyze_message_frequency():
    config = AppConfig()
    log_file = config.room_say_log_file

    print(f"正在分析日志文件：{log_file}")

    message = []
    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.startswith('['):
                    try:
                        end_bracket = line.find(']')
                        if end_bracket > 0:
                            timestamp = line[1:end_bracket]
                            dt = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
                            message.append(dt)
                    except ValueError:
                        continue
        if not message:
            print("没有找到任何消息")
            return
        message.sort()
        print(f"找到 {len(message)} 条消息")
        total_duration = (message[-1] - message[0]).total_seconds()/60
        print(f"消息总时长为 {total_duration:.2f} 分钟")
        five_minute_groups = defaultdict(int)
        for msg_time in message:
            minutes_from_start = (msg_time - message[0]).total_seconds()/60
            group_index = int(minutes_from_start/5)
            five_minute_groups[group_index] += 1

        total_messages = sum(five_minute_groups.values())
        num_groups = len(five_minute_groups)
        average = total_messages/num_groups
        print(f"每五分钟消息数平均为 {average:.2f} 条")

        # 显示每个5分钟区间的消息数
        for group_index in sorted(five_minute_groups.keys()):
            count = five_minute_groups[group_index]
            start_offset = group_index * 5
            end_offset = start_offset + 5
            print(f"第 {group_index + 1:3d} 个5分钟区间 "
                  f"(+{start_offset:4.0f}~{end_offset:4.0f}分钟): "
                  f"{count:3d} 条消息")

        print(f"\n{'=' * 60}")
        print(f"统计结果:")
        print(f"{'=' * 60}")
        print(f"5分钟区间总数: {num_groups}")
        print(f"消息总数: {total_messages}")
        print(f"平均每5分钟消息数: {average:.2f} 条")
        print(f"平均每分钟消息数: {average / 5:.2f} 条")


    except Exception as e:
        print(f"无法打开日志文件：{log_file}")
        print(e)
        return
if __name__ == "__main__":
    analyze_message_frequency()