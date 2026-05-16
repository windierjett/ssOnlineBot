import os
import json
from typing import List, Optional, Dict
from pathlib import Path

from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from bot_core.config import get_secret


class GameRuleRag:
    def __init__(
            self,
            game_rule_dir: str = "gameRule",
            persist_dir: str = "rag_index",
            embedding_model: str = "text-embedding-v3",
            chunk_size: int = 400,
            chunk_overlap: int = 50,
            top_k: int = 5
    ):
        """
        初始化 RAG 系统
        Args:
            game_rule_dir: 游戏规则文件目录
            persist_dir: 向量索引持久化目录
            embedding_model: 嵌入模型名称
            chunk_size: 文本分块大小（400）
            chunk_overlap: 分块重叠大小（50）
            top_k: 检索返回的最相关文档数量（增加到5以提高召回率）
        """
        self.game_rule_dir = Path(game_rule_dir)
        self.persist_dir = Path(persist_dir)
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k

        # 确保目录存在
        self.game_rule_dir.mkdir(parents=True, exist_ok=True)
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        api_key = get_secret("DASHSCOPE_API_KEY") or get_secret("dashscope_api_key", "")
        os.environ["DASHSCOPE_API_KEY"] = api_key

        self.embeddings = DashScopeEmbeddings(
            model="text-embedding-v3",
        )

        # 文本分割器 - 优化分割策略
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=[
                "\n\n",  # 段落分隔
                "\n# ",  # Markdown 一级标题
                "\n## ",  # Markdown 二级标题
                "\n### ",  # Markdown 三级标题
                "\n",  # 换行
                "。",  # 句号
                "！",  # 感叹号
                "？",  # 问号
                "；",  # 分号
                "，",  # 逗号
                " ",  # 空格
                ""  # 字符
            ]
        )

        # Markdown 标题分割器（用于保留文档结构）
        self.markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "header_1"),
                ("##", "header_2"),
                ("###", "header_3"),
            ]
        )

        # 向量存储
        self.vectorstore: Optional[FAISS] = None

        # 加载或创建索引
        self._load_or_create_index()

    def _load_documents(self) -> List[Document]:
        """加载游戏规则目录下的所有文档"""
        if not self.game_rule_dir.exists():
            raise FileNotFoundError(f"游戏规则目录 {self.game_rule_dir} 不存在")
        documents = []

        # 遍历目录下所有文件
        for file_path in self.game_rule_dir.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in ['.txt', '.md', '.json']:
                try:
                    if file_path.suffix.lower() == '.json':
                        # JSON 文件特殊处理
                        with open(file_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            content = json.dumps(data, ensure_ascii=False, indent=2)
                            doc = Document(
                                page_content=content,
                                metadata={
                                    "source": str(file_path),
                                    "filename": file_path.name,
                                    "type": "json"
                                }
                            )
                            documents.append(doc)
                    else:
                        # 文本文件
                        loader = TextLoader(
                            str(file_path),
                            encoding='utf-8',
                            autodetect_encoding=True
                        )
                        docs = loader.load()
                        # 添加元数据
                        for doc in docs:
                            doc.metadata.update({
                                "filename": file_path.name,
                                "type": file_path.suffix.lower(),
                                "filepath": str(file_path)
                            })
                        documents.extend(docs)
                except Exception as e:
                    print(f"加载文件 {file_path} 失败: {e}")

        print(f"成功加载 {len(documents)} 个文档")
        return documents

    def _create_index(self):
        """创建索引"""
        documents = self._load_documents()
        print("创建索引中......")
        if not documents:
            print("没有文档可索引，创建空索引")
            self.vectorstore = FAISS.from_texts(
                texts=["游戏规则"],
                embedding=self.embeddings
            )
            return

        # 分割文档
        splits = self.text_splitter.split_documents(documents)
        print(f"文档分割为 {len(splits)} 个片段")

        # 为每个片段添加标题信息（从内容中提取）
        enhanced_splits = []
        for split in splits:
            # 尝试从内容中提取标题作为元数据
            content = split.page_content
            # 查找 Markdown 标题
            import re
            headers = re.findall(r'^(#+)\s+(.+)$', content, re.MULTILINE)
            if headers:
                split.metadata["headers"] = [h[1] for h in headers[:3]]  # 最多保留3个标题

            enhanced_splits.append(split)

        # 创建向量存储
        self.vectorstore = FAISS.from_documents(
            documents=enhanced_splits,
            embedding=self.embeddings
        )

        # 持久化索引
        self._save_index()
        print("向量索引创建完成并已保存")

    def _load_or_create_index(self):
        """加载已有索引或创建新索引"""
        index_path = self.persist_dir / "index.faiss"

        if index_path.exists():
            print("正在加载向量索引...")
            try:
                self.vectorstore = FAISS.load_local(
                    str(self.persist_dir),
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                print("已加载现有的向量索引")
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
        """重建索引（当游戏规则文件更新时调用）"""
        print("正在重建向量索引...")
        self._create_index()

    def add_document(self, file_path: str):
        """添加单个文档到索引"""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")

        # 加载新文档
        if path.suffix.lower() == '.json':
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                content = json.dumps(data, ensure_ascii=False, indent=2)
                doc = Document(
                    page_content=content,
                    metadata={
                        "source": str(path),
                        "filename": path.name,
                        "type": "json"
                    }
                )
                documents = [doc]
        else:
            loader = TextLoader(str(path), encoding='utf-8', autodetect_encoding=True)
            documents = loader.load()
            for doc in documents:
                doc.metadata.update({
                    "filename": path.name,
                    "type": path.suffix.lower()
                })

        # 分割并添加到索引
        splits = self.text_splitter.split_documents(documents)

        if self.vectorstore:
            self.vectorstore.add_documents(splits)
            self._save_index()
            print(f"成功添加文档 {path.name} 到索引")
        else:
            self.vectorstore = FAISS.from_documents(
                documents=splits,
                embedding=self.embeddings
            )
            self._save_index()

    def search(self, query: str, top_k: Optional[int] = None) -> List[Dict]:
        """
        检索与查询最相关的文档

        Args:
            query: 查询文本
            top_k: 返回结果数量（默认使用初始化时的设置）

        Returns:
            相关文档列表，每个文档包含内容和元数据
        """
        if not self.vectorstore:
            return []

        k = top_k or self.top_k

        try:
            # 使用 MMR (Max Marginal Relevance) 检索，增加结果多样性
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
            print(f"检索失败: {e}")
            return []

    def search_with_keywords(self, query: str, top_k: Optional[int] = None) -> List[Dict]:
        """
        使用关键词增强的检索（提取查询中的关键词进行多轮检索）

        Args:
            query: 查询文本
            top_k: 返回结果数量

        Returns:
            相关文档列表
        """
        # 提取关键词
        keywords = self._extract_keywords(query)

        all_results = []
        seen_contents = set()

        # 先用原始查询检索
        results = self.search(query, top_k=top_k)
        for r in results:
            content_hash = hash(r["content"][:100])
            if content_hash not in seen_contents:
                all_results.append(r)
                seen_contents.add(content_hash)

        # 用每个关键词检索
        for keyword in keywords[:3]:  # 最多用3个关键词
            keyword_results = self.search(keyword, top_k=2)
            for r in keyword_results:
                content_hash = hash(r["content"][:100])
                if content_hash not in seen_contents:
                    all_results.append(r)
                    seen_contents.add(content_hash)

        # 按相关度排序
        all_results.sort(key=lambda x: x["relevance_score"])

        return all_results[:top_k or self.top_k]

    def _extract_keywords(self, query: str) -> List[str]:
        """从查询中提取关键词"""
        # 简单的关键词提取：去除停用词
        stop_words = {"的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都", "一", "一个", "怎么", "什么",
                      "如何"}

        # 分词（简单按标点分割）
        import re
        words = re.split(r'[，。！？；、\s]+', query)

        # 过滤停用词和空字符串
        keywords = [w for w in words if w and w not in stop_words and len(w) >= 2]

        return keywords

    def get_context_for_query(self, query: str, top_k: Optional[int] = None, use_keyword_search: bool = True) -> str:
        """
        为查询获取格式化的上下文文本

        Args:
            query: 查询文本
            top_k: 返回结果数量
            use_keyword_search: 是否使用关键词增强检索

        Returns:
            格式化的上下文字符串
        """
        # 使用关键词增强检索
        if use_keyword_search:
            results = self.search_with_keywords(query, top_k)
        else:
            results = self.search(query, top_k)

        if not results:
            return ""

        context_parts = []
        for i, result in enumerate(results, 1):
            filename = result["metadata"].get("filename", "未知文件")
            content = result["content"]
            headers = result["metadata"].get("headers", [])

            # 如果有标题信息，添加到上下文中
            header_info = ""
            if headers:
                header_info = f"（来自章节：{' > '.join(headers)}）"

            context_parts.append(f"[规则来源 {i}: {filename}]{header_info}\n{content}")

        return "\n\n".join(context_parts)

    def is_game_rule_query(self, text: str) -> bool:
        """
        判断查询是否涉及游戏规则

        Args:
            text: 用户输入文本

        Returns:
            是否为游戏规则相关查询
        """
        game_keywords = [
            "规则", "玩法", "怎么赢", "胜利条件", "游戏机制",
            "积分", "得分", "惩罚", "奖励", "回合",
            "出牌", "手牌", "牌型", "炸弹", "顺子",
            "地主", "农民", "叫牌", "抢地主",
            "怎么玩", "什么意思", "什么是", "如何",
            "获胜", "胜利", "赢", "输", "失败",
            "条件", "要求", "需要"
        ]

        content = str(text or "").strip().lower()
        return any(keyword in content for keyword in game_keywords)
