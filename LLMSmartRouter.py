from typing import Optional
from RagEnhancedRuleLLM import get_rag_llm
from chatEnhancedLLM import get_chat_enhanced_llm
import llm_client

class LLMSmartRouter:
    """智能 LLM 路由器 - 根据场景自动选择合适的 LLM"""

    def __init__(self):
        """初始化路由器"""
        self._rag_llm = None
        self._chat_llm = None
        self._base_llm = llm_client

    def _get_rag_llm(self):
        """懒加载游戏规则 LLM"""
        if self._rag_llm is None:
            self._rag_llm = get_rag_llm()
        return self._rag_llm

    def _get_chat_llm(self):
        """懒加载聊天历史 LLM"""
        if self._chat_llm is None:
            self._chat_llm = get_chat_enhanced_llm()
        return self._chat_llm

    def ask(
            self,
            message: str,
            username: str = "",
            use_rules: bool = True,
            use_chat_history: bool = True
    ) -> str:
        """
        智能问答 - 自动选择合适的 LLM

        Args:
            message: 用户消息
            username: 用户名
            use_rules: 是否启用游戏规则 RAG
            use_chat_history: 是否启用聊天历史 RAG

        Returns:
            LLM 回复
        """
        # 1. 优先检查是否是游戏规则问题
        if use_rules:
            rag_llm = self._get_rag_llm()
            if rag_llm.rag.is_game_rule_query(message):
                reply = rag_llm.ask_with_rules(message)
                if reply:
                    return reply

        # 2. 检查是否有聊天历史上下文
        if use_chat_history and username:
            chat_llm = self._get_chat_llm()
            reply = chat_llm.ask_with_chat_history(username, message)
            if reply:
                return reply

        # 3. 默认使用基础 LLM
        return self._base_llm.ask_llm(message, username=username)

    def ask_stream(
            self,
            message: str,
            username: str = "",
            chunk_size: int = 50
    ):
        """
        流式问答 - 使用基础 LLM 流式输出

        Args:
            message: 用户消息
            username: 用户名
            chunk_size: 分块大小

        Yields:
            str: 文本片段
        """
        # 流式目前只支持基础 LLM
        for chunk in self._base_llm.ask_llm_stream(message, chunk_size=chunk_size, username=username):
            yield chunk


# 全局单例
_smart_router: Optional[LLMSmartRouter] = None


def get_smart_router() -> LLMSmartRouter:
    """获取智能 LLM 路由器单例"""
    global _smart_router
    if _smart_router is None:
        _smart_router = LLMSmartRouter()
    return _smart_router
