from typing import Optional
import gameRuleRag


class RagEnhancedLLM:
    """游戏规则 RAG 增强的 LLM 客户端"""

    def __init__(
            self,
            game_rule_dir: str = "gameRule",
            persist_dir: str = "rag_index",
            auto_retrieve: bool = True
    ):
        """
        初始化游戏规则 RAG 增强的 LLM
        Args:
            game_rule_dir: 游戏规则文件目录
            persist_dir: 向量索引持久化目录
            auto_retrieve: 是否自动检测并检索游戏规则
        """
        self.rag = gameRuleRag.GameRuleRag(game_rule_dir, persist_dir)
        self.auto_retrieve = auto_retrieve

        # 初始化 LLM（独立的游戏规则 LLM）
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
        """重建游戏规则索引"""
        self.rag.rebuild_index()

    def add_rule(self, file_path: str):
        """添加游戏规则文件"""
        self.rag.add_document(file_path)

    def ask_with_rules(self, user_input: str, max_chars: int = 100) -> str:
        """
        基于游戏规则进行问答

        Args:
            user_input: 用户输入
            max_chars: 回复最大字符数

        Returns:
            LLM 回复
        """
        # 检查是否是游戏规则相关查询
        if not self.rag.is_game_rule_query(user_input):
            return ""  # 不是游戏规则问题，返回空

        # 检索相关规则
        context = self.rag.get_context_for_query(user_input)

        if not context:
            return ""

        # 构建 Prompt
        from langchain_core.prompts import ChatPromptTemplate

        prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个游戏规则专家。请基于以下游戏规则回答问题：
1. 严格基于提供的规则内容回答，不要编造规则中不存在的内容。
2. 如果规则中没有明确说明，请诚实地告知用户。
3. 回复要简短清晰，控制在 {max_chars} 字以内。

【游戏规则参考】
{rules_context}"""),
            ("human", "{input}"),
        ])

        # 调用 LLM
        response = prompt | self.llm
        result = response.invoke({
            "input": user_input,
            "rules_context": context,
            "max_chars": max_chars
        })

        reply = result.content
        if len(reply) > max_chars:
            reply = reply[:max_chars]

        return reply

    def search_rule(self, query: str, top_k: int = 3):
        """
        直接搜索游戏规则（用于调试或手动查询）
        Args:
            query: 查询文本
            top_k: 返回结果数量

        Returns:
            相关规则列表
        """
        return self.rag.search(query, top_k)


# 全局单例
_rag_llm: Optional[RagEnhancedLLM] = None


def get_rag_llm(
        game_rule_dir: str = "gameRule",
        persist_dir: str = "rag_index",
        auto_retrieve: bool = True
) -> RagEnhancedLLM:
    """获取游戏规则 RAG 增强的 LLM 单例"""
    global _rag_llm
    if _rag_llm is None:
        _rag_llm = RagEnhancedLLM(
            game_rule_dir=game_rule_dir,
            persist_dir=persist_dir,
            auto_retrieve=auto_retrieve
        )
    return _rag_llm
