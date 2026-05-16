"""
RAG 系统使用示例和工具脚本
"""
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from RagEnhancedRuleLLM import get_rag_llm
from chatEnhancedLLM import get_chat_enhanced_llm


def create_sample_rules():
    """创建示例游戏规则文件"""
    game_rule_dir = project_root / "gameRule"
    game_rule_dir.mkdir(exist_ok=True)

    # 示例：斗地主规则
    doudizhu_rules = """
# 斗地主游戏规则

## 基本玩法
斗地主是一种三人扑克牌游戏，使用一副54张牌（包括大小王）。

## 角色分配
- 地主：1人，获得3张底牌，需要单独对抗两个农民
- 农民：2人，合作对抗地主

## 胜利条件
- 地主先出完所有手牌则地主获胜
- 任一农民先出完所有手牌则农民方获胜

## 牌型大小
1. 单张：3 < 4 < 5 < ... < K < A < 2 < 小王 < 大王
2. 对子：两张相同的牌
3. 三张：三张相同的牌
4. 三带一：三张相同的牌 + 一张单牌
5. 三带二：三张相同的牌 + 一对牌
6. 顺子：5张或更多连续的单牌（如：3-4-5-6-7）
7. 连对：3对或更多连续的对子（如：33-44-55）
8. 飞机：2个或更多连续的三张（如：333-444）
9. 炸弹：四张相同的牌（如：3333），可以压制除王炸外的所有牌型
10. 王炸：大王+小王，是最大的牌型

## 出牌规则
- 首家可以出任意合法牌型
- 后续玩家必须出相同牌型且更大的牌，或者选择不出
- 如果所有其他玩家都不出，则最后出牌者获得下一轮出牌权

## 叫牌规则
- 按顺序轮流叫分：1分、2分、3分或"不叫"
- 叫分最高者成为地主，获得3张底牌
- 如果所有人都不叫，则重新发牌
"""

    with open(game_rule_dir / "doudizhu_rules.md", "w", encoding="utf-8") as f:
        f.write(doudizhu_rules)

    print("✓ 已创建示例游戏规则文件: doudizhu_rules.md")


def test_rag_system():
    """测试游戏规则 RAG 系统"""
    print("=" * 60)
    print("【游戏规则 RAG 系统测试】")
    print("=" * 60)

    # 获取 RAG 实例
    rag_llm = get_rag_llm()

    # 测试1: 检查是否会识别游戏规则查询
    test_queries = [
        "斗地主怎么玩？",
        "什么是炸弹？",
        "今天天气怎么样？",
        "地主的胜利条件是什么？",
        "给我讲个故事"
    ]

    print("\n【测试1】游戏规则查询识别")
    for query in test_queries:
        is_game_rule = rag_llm.rag.is_game_rule_query(query)
        status = "✓ 游戏规则" if is_game_rule else "✗ 普通聊天"
        print(f"  {status} | {query}")

    # 测试2: 检索相关规则
    print("\n【测试2】规则检索测试")
    search_query = "斗地主胜利条件"
    results = rag_llm.search_rule(search_query, top_k=2)

    if results:
        print(f"  找到 {len(results)} 条相关规则:")
        for i, result in enumerate(results, 1):
            filename = result["metadata"].get("filename", "未知")
            score = result.get("relevance_score", 0)
            preview = result["content"][:100].replace("\n", " ")
            print(f"\n  [{i}] 来源: {filename}")
            print(f"      相关度: {score:.4f}")
            print(f"      预览: {preview}...")
    else:
        print("  未找到相关规则（可能还没有添加规则文件）")

    # 测试3: 基于规则的问答
    print("\n【测试3】游戏规则问答测试")
    question = "斗地主的胜利条件是什么？"
    print(f"  问题: {question}")
    answer = rag_llm.ask_with_rules(question)
    if answer:
        print(f"  回答: {answer}")
    else:
        print("  未获取到回答")

    print("\n" + "=" * 60)


# 在 test.py 中添加新的测试函数
def test_intent_recognition():
    """测试意图识别功能"""
    print("\n" + "=" * 60)
    print("【意图识别测试】")
    print("=" * 60)

    chat_llm = get_chat_enhanced_llm()

    # 测试用例：应该使用 RAG 的消息
    should_use_rag = [
        "你还记得我吗？",
        "李洛的性格怎么样？",
        "我之前说过什么？",
        "风儿吹吹有什么特点？",
        "我们上次聊了什么？",
        "你还记得我喜欢什么吗？",
        "[李洛]这个人怎么样？",
        "我的历史记录里有啥？",
        "你对我有什么印象？",
        "我记得你之前说过...",
    ]

    # 测试用例：不应该使用 RAG 的消息
    should_not_use_rag = [
        "你好",
        "今天天气怎么样？",
        "讲个故事",
        "哈哈哈",
        "在吗？",
        "吃饭了吗？",
        "晚安",
        "666",
        "斗地主怎么玩？",  # 这是游戏规则，不是聊天历史
        "随便聊聊",
    ]

    print("\n【应该使用 RAG 的消息】")
    for msg in should_use_rag:
        result = chat_llm.chat_rag.is_memory_related_query(msg)
        status = "✓" if result else "✗"
        print(f"  {status} {msg}")

    print("\n【不应该使用 RAG 的消息】")
    for msg in should_not_use_rag:
        result = chat_llm.chat_rag.is_memory_related_query(msg)
        status = "✗" if result else "✓"
        print(f"  {status} {msg}")

    print("\n" + "=" * 60)

def test_chat_history_system():
    """测试聊天历史 RAG 系统"""
    print("\n" + "=" * 60)
    print("【聊天历史 RAG 系统测试】")
    print("=" * 60)

    # 获取聊天历史 RAG 实例
    chat_llm = get_chat_enhanced_llm()

    # 测试1: 消息解析功能（包含位置信息和机器人前缀）
    print("\n【测试1】消息解析测试")
    test_messages = [
        "[2026-05-13 21:32:07] [3107] [位置:3] 逸尘: 2今天天气这么热",
        "[2026-05-13 21:44:49] [3081] [位置:5] 李洛: 4武汉天气",
        "[2026-05-13 21:47:11] [3134] [位置:1] 风儿吹吹: 2讲个悲伤的故事",
        "[2026-05-13 21:17:13] [3081] [位置:20] 聆伊: 我喜欢你！",
        "[2026-05-13 21:43:54] [3081] [位置:6] 李洛: 老婆"
    ]

    for msg in test_messages:
        result = chat_llm.chat_rag.parse_and_process_message(msg)
        if result['success']:
            print(f"\n  原始消息: {msg}")
            print(f"  用户名: {result['username']}")
            print(f"  位置: {result['position']}")
            print(f"  是否为机器人查询: {result['is_bot_query']}")
            if result['bot_position']:
                print(f"  机器人位置: {result['bot_position']}")
            print(f"  实际消息: {result['actual_message']}")
            if result['mentioned_users']:
                print(f"  提到的用户: {result['mentioned_users']}")
        else:
            print(f"  解析失败: {result.get('error')}")

    # 测试2: 机器人查询处理（过滤数字前缀）
    print("\n\n【测试2】机器人查询处理测试")
    bot_queries = [
        "2李洛的性格怎么样",
        "4讲一个悲伤的故事",
        "1今天天气如何",
        "3你觉得呢",
        "普通消息没有数字前缀"
    ]

    for query in bot_queries:
        bot_pos, actual_msg, mentioned = chat_llm.chat_rag.process_bot_query(query)
        print(f"\n  原始查询: {query}")
        print(f"  机器人位置: {bot_pos if bot_pos else '无'}")
        print(f"  实际消息: {actual_msg}")
        print(f"  提到的用户: {mentioned if mentioned else '无'}")

    # 测试3: 个性化上下文获取（80%当前用户 + 20%其他用户）
    print("\n\n【测试3】个性化上下文获取 - 80%当前用户 + 20%其他用户")
    test_cases = [
        ("风儿吹吹", "哇今天真难受呀"),
        ("逸尘", "讲个悲伤的故事"),
        ("李洛", "今天天气怎么样"),
        ("折霜枝", "你喜欢什么")
    ]

    for username, query in test_cases:
        print(f"\n{'=' * 50}")
        print(f"用户: {username}")
        print(f"查询: {query}")
        print(f"{'=' * 50}")

        context = chat_llm.chat_rag.get_personalized_context(
            username=username,
            query=query
        )

        if context:
            print("获取到的个性化上下文:")
            print(context)

            # 分析上下文组成
            lines = context.split('\n')
            current_user_count = sum(1 for line in lines if username in line and '的聊天记录' in line)
            other_user_count = sum(1 for line in lines if '其他用户的相似对话参考' in line)

            print(f"\n[统计] 当前用户历史段落: {current_user_count}, 其他用户参考段落: {other_user_count}")
        else:
            print("未获取到上下文（可能还没有相关聊天记录）")

    # 测试4: 用户名提取
    print("\n\n【测试4】用户名提取测试")
    test_texts = [
        "[风儿吹吹]：[@李洛]最近怎么样？",
        "听说@张三和@李四关系很好",
        "今天天气不错",
        "[王五] 说：[@赵六]你来了",
        "2李洛你觉得呢",
        "[逸尘]: 4[@夏凌依]讲个故事"
    ]

    for text in test_texts:
        mentions = chat_llm.chat_rag.extract_mentioned_users(text)
        print(f"  输入: {text}")
        print(f"  提取到的用户名: {mentions if mentions else '无'}")
        print()

    # 测试5: 完整流程演示
    print("\n【测试5】完整流程演示 - 从消息到上下文")
    full_message = "[2026-05-13 21:32:07] [3107] [位置:3] 逸尘: 2今天天气这么热"
    print(f"  完整消息: {full_message}")

    # 步骤1: 解析消息
    parsed = chat_llm.chat_rag.parse_and_process_message(full_message)
    if parsed['success']:
        print(f"\n  ✓ 解析成功")
        print(f"    - 发送者: {parsed['username']}")
        print(f"    - 位置: {parsed['position']}")
        print(f"    - 机器人位置: {parsed['bot_position']}")
        print(f"    - 实际问题: {parsed['actual_message']}")

        # 步骤2: 获取个性化上下文（只传入用户名和实际问题）
        context = chat_llm.chat_rag.get_personalized_context(
            username=parsed['username'],
            query=parsed['actual_message']
        )

        if context:
            print(f"\n  ✓ 获取到个性化上下文:")
            print(f"    {context[:400]}")
            if len(context) > 400:
                print("    ...")
        else:
            print(f"\n  ✗ 未获取到上下文")

    print("\n" + "=" * 60)


def main():
    """主函数"""
    if len(sys.argv) > 1:
        command = sys.argv[1]

        if command == "create-sample":
            create_sample_rules()
            print("\n现在可以运行 'python test.py test' 来测试 RAG 系统")

        elif command == "test":
            #test_rag_system()
            #test_chat_history_system()
            test_intent_recognition()

        elif command == "rebuild-rules":
            print("正在重建游戏规则索引...")
            rag_llm = get_rag_llm()
            rag_llm.rebuild_index()
            print("游戏规则索引重建完成！")

        elif command == "rebuild-chat":
            print("正在重建聊天历史索引...")
            chat_llm = get_chat_enhanced_llm()
            chat_llm.rebuild_index()
            print("聊天历史索引重建完成！")

        else:
            print(f"未知命令: {command}")
            print("可用命令: create-sample, test, rebuild-rules, rebuild-chat")
    else:
        print("RAG 系统示例工具")
        print("\n用法:")
        print("  python test.py create-sample    # 创建示例规则文件")
        print("  python test.py test             # 测试 RAG 系统")
        print("  python test.py rebuild-rules    # 重建游戏规则索引")
        print("  python test.py rebuild-chat     # 重建聊天历史索引")
        print("\n提示: 先在 gameRule 目录下放置游戏规则文件，然后运行测试")


if __name__ == "__main__":
    main()
