"""
聊天历史 RAG 系统测试脚本
"""
import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from chatEnhancedLLM import get_chat_enhanced_llm


def test_chat_history_rag():
    """测试聊天历史 RAG 系统"""
    print("=" * 60)
    print("聊天历史 RAG 系统测试")
    print("=" * 60)

    # 获取聊天增强 LLM 实例
    chat_llm = get_chat_enhanced_llm()

    # 测试场景1: 普通聊天
    print("\n【测试1】普通聊天场景")
    print("用户: [风儿吹吹]：哇今天真难受呀")
    context = chat_llm.chat_rag.get_personalized_context(
        username="风儿吹吹",
        query="哇今天真难受呀"
    )
    if context:
        print("获取到的上下文:")
        print(context[:500])
    else:
        print("未获取到上下文（可能还没有该用户的聊天记录）")

    # 测试场景2: 提到其他用户
    print("\n" + "=" * 50)
    print("\n【测试2】提到其他用户的场景")
    print("用户: [风儿吹吹]：[@李洛]的特征是什么？")
    context = chat_llm.chat_rag.get_personalized_context(
        username="风儿吹吹",
        query="李洛的特征是什么？",
        mentioned_users=["李洛"]
    )
    if context:
        print("获取到的上下文:")
        print(context[:500])
    else:
        print("未获取到上下文")

    # 测试场景3: 测试用户名提取
    print("\n" + "=" * 50)
    print("\n【测试3】用户名提取测试")
    test_texts = [
        "[风儿吹吹]：[@李洛]最近怎么样？",
        "听说@张三和@李四关系很好",
        "今天天气不错",
        "[王五] 说：[@赵六]你来了"
    ]

    for text in test_texts:
        mentions = chat_llm.chat_rag.extract_mentioned_users(text)
        print(f"  输入: {text}")
        print(f"  提取到的用户名: {mentions}")
        print()


def main():
    """主函数"""
    if len(sys.argv) > 1:
        command = sys.argv[1]

        if command == "rebuild":
            print("正在重建聊天历史索引...")
            chat_llm = get_chat_enhanced_llm()
            chat_llm.rebuild_index()
            print("索引重建完成！")

        elif command == "test":
            test_chat_history_rag()

        else:
            print(f"未知命令: {command}")
            print("可用命令: test, rebuild")
    else:
        print("聊天历史 RAG 系统测试工具")
        print("\n用法:")
        print("  python chat_history_test.py test    # 测试系统")
        print("  python chat_history_test.py rebuild # 重建索引")


if __name__ == "__main__":
    main()
