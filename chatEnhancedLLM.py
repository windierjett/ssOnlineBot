from typing import Optional, List
import chatHistroyRAG
class ChatEnhancedLLM:
    """聊天历史 RAG 增强的 LLM 客户端"""

    def __init__(
            self,
            chat_logs_dir: str = "room_say_logs",
            persist_dir: str = "chat_rag_index",
            auto_enhance: bool = True
    ):
        """
        初始化聊天历史 RAG 增强的 LLM
        Args:
            chat_logs_dir: 聊天记录目录
            persist_dir: 向量索引持久化目录
            auto_enhance: 是否自动使用聊天历史增强
        """
        self.chat_rag = chatHistroyRAG.ChatHistoryRAG(
            chat_logs_dir=chat_logs_dir,
            persist_dir=persist_dir
        )
        self.auto_enhance = auto_enhance

        # 初始化 LLM（独立的聊天历史 LLM）
        from langchain_openai import ChatOpenAI
        from bot_core.config import get_secret
        import os

        api_key = os.getenv("DASHSCOPE_API_KEY") or get_secret("dashscope_api_key", "")
        self.llm = ChatOpenAI(
            model="qwen-plus",
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            api_key=api_key,
            temperature=0.7,
            top_p=0.85,
            max_tokens=200,
        )

    def rebuild_index(self):
        """重建聊天历史索引"""
        self.chat_rag.rebuild_index()

    def ask_with_chat_history(
        self,
        username: str,
        user_input: str,
        max_chars: int = 100
    ) -> Optional[str]:
        """
        基于聊天历史进行个性化问答

        Args:
            username: 当前用户名
            user_input: 用户输入（可能包含位置信息等）
            max_chars: 回复最大字符数

        Returns:
            个性化回复，如果不需要或不适合使用 RAG 则返回 None
        """
        # 判断是否需要使用 RAG
        if not self.chat_rag.should_use_rag(username, user_input):
            return None  # 不需要 RAG，返回 None 让调用者使用普通 LLM

        # 提取提到的用户
        mentioned_users = self.chat_rag.extract_mentioned_users(user_input)

        # 获取个性化上下文
        context = self.chat_rag.get_personalized_context(
            username=username,
            query=user_input,
            mentioned_users=mentioned_users
        )

        if not context:
            return None  # 没有聊天历史上下文，返回 None 让调用者回退到其他 LLM

        # 构建 Prompt
        from langchain_core.prompts import ChatPromptTemplate

        prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个有记忆力的聊天伙伴'夏凌依'。请根据用户的聊天历史给出个性化回复：
    1. 【保持独立】不盲目附和或迎合任何人，遇到主观评价请给出自己的客观看法。
    2. 【逻辑判断】如果前后说法矛盾，请指出差异并给出中立或温和的结论。
    3. 【玩家口吻】回复要自然简短，像真实玩家一样，控制在 {max_chars} 字以内。
    4. 【风格一致】回复风格要与该用户的历史聊天风格保持一致。
    5. 【参考历史】如果提到了其他用户，参考这些用户的历史特征。
    6. 直接输出回复内容，不要复述问题。

    【聊天历史上下文】
    {chat_history_context}"""),
                ("human", "{input}"),
            ])

        # 调用 LLM
        response = prompt | self.llm
        result = response.invoke({
                "input": user_input,
                "chat_history_context": context,
                "max_chars": max_chars
            })

        reply = result.content
        if len(reply) > max_chars:
            reply = reply[:max_chars]

        return reply

    def search_user_history(self, username: str, query: str, top_k: int = 3):
        """
        搜索特定用户的聊天历史
        Args:
            username: 用户名
            query: 查询内容
            top_k: 返回结果数量

        Returns:
            相关聊天记录列表
        """
        return self.chat_rag.search_user_history(username, query, top_k)


# 全局单例
_chat_enhanced_llm: Optional[ChatEnhancedLLM] = None


def get_chat_enhanced_llm(
        chat_logs_dir: str = "room_say_logs",
        persist_dir: str = "chat_rag_index",
        auto_enhance: bool = True
) -> ChatEnhancedLLM:
    """获取聊天历史 RAG 增强的 LLM 单例"""
    global _chat_enhanced_llm
    if _chat_enhanced_llm is None:
        _chat_enhanced_llm = ChatEnhancedLLM(
            chat_logs_dir=chat_logs_dir,
            persist_dir=persist_dir,
            auto_enhance=auto_enhance
        )
    return _chat_enhanced_llm
