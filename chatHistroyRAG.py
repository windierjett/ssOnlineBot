import os
import re
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from bot_core.config import get_secret


class ChatHistoryRAG:
    """聊天记录 RAG 系统 - 基于用户聊天历史提供个性化回复"""

    def __init__(
            self,
            chat_logs_dir: str = "room_say_logs",
            persist_dir: str = "chat_rag_index",
            embedding_model: str = "text-embedding-v3",
            chunk_size: int = 300,
            chunk_overlap: int = 50,
            top_k: int = 5
    ):
        """
        初始化聊天记录 RAG 系统

        Args:
            chat_logs_dir: 聊天记录文件目录
            persist_dir: 向量索引持久化目录
            embedding_model: 嵌入模型名称
            chunk_size: 文本分块大小
            chunk_overlap: 分块重叠大小
            top_k: 检索返回的最相关聊天记录数量
        """
        self.chat_logs_dir = Path(chat_logs_dir)
        self.persist_dir = Path(persist_dir)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k

        # 确保目录存在
        self.chat_logs_dir.mkdir(parents=True, exist_ok=True)
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        # 初始化嵌入模型
        api_key = get_secret("DASHSCOPE_API_KEY") or get_secret("dashscope_api_key", "")
        os.environ["DASHSCOPE_API_KEY"] = api_key

        from langchain_community.embeddings import DashScopeEmbeddings
        self.embeddings = DashScopeEmbeddings(model=embedding_model)

        # 文本分割器
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n", "。", "！", "？", "；", "，", " "]
        )

        # 向量存储
        self.vectorstore: Optional[FAISS] = None

        # 加载或创建索引
        self._load_or_create_index()

    def _parse_chat_line(self, line: str) -> Optional[Dict]:
        """
        解析单行聊天记录

        Args:
            line: 聊天记录行

        Returns:
            解析后的字典，包含 timestamp, room_id, position, username, message
            如果解析失败返回 None
        """
        # 匹配格式: [时间戳] [房间ID] [位置:X] 用户名: 消息内容
        pattern = r'\[(.*?)\]\s*\[(\d+)\]\s*\[位置:(\d+)\]\s*(.+?):\s*(.*)'
        match = re.match(pattern, line.strip())

        if match:
            return {
                'timestamp': match.group(1),
                'room_id': match.group(2),
                'position': match.group(3),
                'username': match.group(4).strip(),
                'message': match.group(5).strip()
            }

        # 兼容旧格式（无位置信息）: [时间戳] [房间ID] 用户名: 消息内容
        old_pattern = r'\[(.*?)\]\s*\[(\d+)\]\s*(.+?):\s*(.*)'
        match = re.match(old_pattern, line.strip())

        if match:
            return {
                'timestamp': match.group(1),
                'room_id': match.group(2),
                'position': None,
                'username': match.group(3).strip(),
                'message': match.group(4).strip()
            }

        return None

    def _extract_bot_query(self, message: str) -> Tuple[Optional[str], str]:
        """
        提取针对机器人的查询，过滤掉开头的数字（机器人位置）

        Args:
            message: 原始消息内容

        Returns:
            (bot_position, actual_message) 元组
            - bot_position: 机器人位置（如果有），否则为 None
            - actual_message: 过滤后的实际消息内容
        """
        # 匹配以数字开头的消息，如 "2李洛的性格怎么样" 或 "2 讲一个悲伤的故事"
        pattern = r'^(\d+)\s*(.*)'
        match = re.match(pattern, message)

        if match:
            bot_position = match.group(1)
            actual_message = match.group(2).strip()
            return bot_position, actual_message

        return None, message
    def _load_chat_logs(self) -> List[Document]:
        """加载所有聊天记录文件"""
        if not self.chat_logs_dir.exists():
            print(f"警告: 聊天记录目录 {self.chat_logs_dir} 不存在")
            return []

        documents = []

        for file_path in self.chat_logs_dir.rglob("*.txt"):
            try:
                # 从文件名提取用户名 (格式: room_say_用户名.txt)
                username = file_path.stem.replace("room_say_", "")
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                # 创建文档，添加用户元数据
                doc = Document(
                    page_content=content,
                    metadata={
                        "username": username,
                        "filename": file_path.name,
                        "filepath": str(file_path),
                        "last_modified": os.path.getmtime(file_path)
                    }
                )
                documents.append(doc)

            except Exception as e:
                print(f"加载聊天记录 {file_path} 失败: {e}")

        print(f"成功加载 {len(documents)} 个用户的聊天记录")
        return documents

    def _create_index(self):
        """从聊天记录创建向量索引"""
        documents = self._load_chat_logs()
        print("创建索引中......")
        if not documents:
            print("没有聊天记录可索引，创建空索引")
            self.vectorstore = FAISS.from_texts(
                texts=["聊天记录"],
                embedding=self.embeddings
            )
            return

        # 分割文档
        splits = self.text_splitter.split_documents(documents)
        print(f"聊天记录分割为 {len(splits)} 个片段")

        # 创建向量存储
        self.vectorstore = FAISS.from_documents(
            documents=splits,
            embedding=self.embeddings
        )

        # 持久化索引
        self._save_index()
        print("聊天记录索引创建完成并已保存")

    def _load_or_create_index(self):
        """加载已有索引或创建新索引"""
        index_path = self.persist_dir / "index.faiss"

        if index_path.exists():
            print("正在加载聊天记录索引...")
            try:
                self.vectorstore = FAISS.load_local(
                    str(self.persist_dir),
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                print("已加载现有的聊天记录索引")
            except Exception as e:
                print(f"加载索引失败: {e}，重新创建索引")
                self._create_index()
        else:
            self._create_index()

    def _save_index(self):
        """保存向量索引到磁盘"""
        if self.vectorstore:
            self.vectorstore.save_local(str(self.persist_dir))

    def rebuild_index(self):
        """重建索引（当聊天记录更新时调用）"""
        print("正在重建聊天记录索引...")
        self._create_index()

    def search_user_history(self, username: str, query: str, top_k: Optional[int] = None) -> List[Dict]:
        """
        搜索特定用户的聊天记录

        Args:
            username: 用户名
            query: 查询内容
            top_k: 返回结果数量

        Returns:
            相关聊天记录列表
        """
        if not self.vectorstore:
            return []

        k = top_k or self.top_k

        try:
            # 使用过滤器只搜索特定用户的记录
            results = self.vectorstore.similarity_search_with_score(
                query=query,
                k=k,
                filter={"username": username}
            )

            formatted_results = []
            for doc, score in results:
                formatted_results.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "relevance_score": float(score)
                })

            return formatted_results

        except Exception as e:
            print(f"搜索用户历史失败: {e}")
            return []

    def search_related_users(self, query: str, top_k: Optional[int] = None) -> List[Dict]:
        """
        搜索与查询相关的所有用户聊天记录

        Args:
            query: 查询内容
            top_k: 返回结果数量

        Returns:
            相关聊天记录列表
        """
        if not self.vectorstore:
            return []

        k = top_k or self.top_k

        try:
            results = self.vectorstore.similarity_search_with_score(query, k=k)

            formatted_results = []
            for doc, score in results:
                formatted_results.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "relevance_score": float(score)
                })

            return formatted_results

        except Exception as e:
            print(f"搜索相关用户失败: {e}")
            return []

    def extract_mentioned_users(self, text: str) -> List[str]:
        """
        从文本中提取提到的用户名

        Args:
            text: 输入文本

        Returns:
            提到的用户名列表
        """
        import re

        # 提取 [@用户名] 格式
        at_mentions = re.findall(r'\[(.+?)\]', text)

        # 提取 @用户名 格式
        simple_mentions = re.findall(r'@(\S+)', text)

        # 合并并去重
        mentioned_users = list(set(at_mentions + simple_mentions))

        return mentioned_users

    def process_bot_query(self, message: str) -> Tuple[Optional[str], str, List[str]]:
        """
        处理针对机器人的查询消息

        Args:
            message: 原始消息内容

        Returns:
            (bot_position, actual_message, mentioned_users) 元组
            - bot_position: 机器人位置（如果有）
            - actual_message: 过滤后的实际消息
            - mentioned_users: 消息中提到的其他用户名列表
        """
        # 1. 提取并过滤机器人位置前缀
        bot_position, actual_message = self._extract_bot_query(message)

        # 2. 提取消息中提到的其他用户
        mentioned_users = self.extract_mentioned_users(actual_message)

        return bot_position, actual_message, mentioned_users

    def get_personalized_context(
            self,
            username: str,
            query: str,
            mentioned_users: Optional[List[str]] = None
    ) -> str:
        """
        获取个性化上下文

        Args:
            username: 当前说话的用户名
            query: 用户的查询/消息
            mentioned_users: 提到的其他用户名列表

        Returns:
            格式化的上下文字符串
        """
        context_parts = []

        # 1. 优先搜索当前用户的聊天历史（占 80%）
        user_history = self.search_user_history(username, query, top_k=4)
        if user_history:
            context_parts.append(f"【{username}的聊天记录】")
            for i, result in enumerate(user_history, 1):
                content = result["content"][:200]  # 限制长度
                context_parts.append(f"  [{i}] {content}")

        # 2. 搜索其他用户的相似内容（占 20%，最多1-2条）
        related_chats = self.search_related_users(query, top_k=5)
        if related_chats:
            # 过滤掉当前用户的记录，只保留其他用户
            other_user_chats = [
                chat for chat in related_chats
                if chat["metadata"].get("username") != username
            ]

            # 只取前 1-2 条作为补充参考
            if other_user_chats:
                selected_chats = other_user_chats[:2]

                # 只有在已经有当前用户历史时才添加其他用户内容
                # 或者当没有其他选择时才添加
                if user_history or not context_parts:
                    context_parts.append("\n【其他用户的相似对话参考】")
                    for i, result in enumerate(selected_chats, 1):
                        username_from_meta = result["metadata"].get("username", "未知用户")
                        content = result["content"][:150]
                        context_parts.append(f"  [{i}] {username_from_meta}: {content}")

        # 3. 如果完全没有找到任何信息
        if not context_parts:
            return ""

        return "\n".join(context_parts)

    def parse_and_process_message(self, raw_message: str) -> Dict:
        """
        解析并处理完整的聊天消息（包含元数据和内容）

        Args:
            raw_message: 完整的聊天消息行，如 "[2026-05-13 21:32:07] [3107] [位置:3] 逸尘: 2今天天气这么热"

        Returns:
            包含解析结果和处理后信息的字典
        """
        # 1. 解析消息行
        parsed = self._parse_chat_line(raw_message)
        if not parsed:
            return {
                'success': False,
                'error': '消息格式无法解析',
                'raw_message': raw_message
            }

        # 2. 处理机器人查询（过滤数字前缀）
        bot_position, actual_message, mentioned_users = self.process_bot_query(parsed['message'])

        return {
            'success': True,
            'timestamp': parsed['timestamp'],
            'room_id': parsed['room_id'],
            'position': parsed['position'],
            'username': parsed['username'],
            'original_message': parsed['message'],
            'bot_position': bot_position,
            'actual_message': actual_message,
            'mentioned_users': mentioned_users,
            'is_bot_query': bot_position is not None
        }

    def is_memory_related_query(self, text: str) -> bool:
        """
        判断用户消息是否与记忆/历史相关

        Args:
            text: 用户消息文本

        Returns:
            True 如果消息与记忆/历史相关，需要调用 RAG
        """
        if not text or not text.strip():
            return False

        text_lower = text.lower()

        # 定义记忆相关的关键词
        memory_keywords = [
            # 直接提及记忆/历史
            "记忆", "回忆", "历史", "过去", "以前", "之前","印象",

            # 询问是否记得
            "记得", "还记得", "忘了", "忘记", "忘掉",

            # 询问特征/性格/喜好
            "性格", "特征", "特点", "喜好", "喜欢", "讨厌",
            "爱好", "习惯", "风格",

            # 询问关系/互动
            "关系", "认识", "熟悉", "了解",

            # 时间相关
            "上次", "上回", "曾经", "以前说过", "之前说过",

            # RAG 相关
            "rag", "检索", "查找", "搜索",

            # 对比/变化
            "变化", "改变", "不同", "区别",

            # 评价类
            "怎么样", "如何", "怎样", "什么人", "什么样的人"
        ]

        # 检查是否包含记忆相关关键词
        for keyword in memory_keywords:
            if keyword in text_lower:
                return True

        # 检查是否有 @用户名 或 [用户名] 的提及模式，并且伴随疑问
        mention_pattern = r'[@\[]([^\]@]+)[\]@]'
        has_mention = re.search(mention_pattern, text)

        # 如果有提及用户，并且包含疑问词，可能需要查询该用户的历史
        if has_mention:
            question_words = ["吗", "呢", "什么", "怎么", "如何", "为什么", "？", "?"]
            if any(word in text for word in question_words):
                return True

        return False

    def should_use_rag(self, username: str, message: str) -> bool:
        """
        综合判断是否应该使用 RAG

        Args:
            username: 用户名
            message: 用户消息

        Returns:
            True 如果应该使用 RAG
        """
        # 1. 首先检查是否是记忆相关的查询
        if self.is_memory_related_query(message):
            return True

        # 2. 检查用户是否有足够的历史记录（可选优化）
        # 如果用户历史记录很少，可能不需要 RAG
        try:
            user_history = self.search_user_history(username, "", top_k=1)
            if not user_history:
                # 用户没有历史记录，不需要 RAG
                return False
        except Exception:
            pass

        # 3. 默认不使用 RAG，保持快速响应
        return False