import os
import json
from typing import List, Optional, Dict
from pathlib import Path

from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from openai import api_key

from bot_core.config import get_secret

class GameRuleRag:
    def __init__(
            self,
            game_rule_dir: str = "gameRule",
            persist_dir: str = "rag_index",
            embedding_model: str =  "text-embedding-v3",
            chunk_size: int = 500,
            chunk_overlap: int = 50,
            top_k: int = 3
    ):
        """
        初始化 RAG 系统
        Args:
            game_rule_dir: 游戏规则文件目录
            persist_dir: 向量索引持久化目录
            embedding_model: 嵌入模型名称
            chunk_size: 文本分块大小
            chunk_overlap: 分块重叠大小
            top_k: 检索返回的最相关文档数量
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

        # 初始化嵌入模型
        api_key = get_secret("DASHSCOPE_API_KEY") or get_secret("dashscope_api_key", "")
        self.embeddings = OpenAIEmbeddings(
            model = embedding_model,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            api_key=api_key,
            dimensions=1536
        )

        # 文本分割器
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n","\n", " ", "！", "。", "？", ",", ".", "?", "!", ":", ";", "；", "：", "，"]
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
                                "type": file_path.suffix.lower()
                            })
                        documents.extend(docs)
                except Exception as e:
                    print(f"加载文件 {file_path} 失败: {e}")

        print(f"成功加载 {len(documents)} 个文档")
        return documents
    def _create_index(self):
        """创建索引"""
        documents = self._load_documents()
        if not documents:
            print("没有文档可索引，创建空索引")
            self.vectorstore = FAISS.from_texts(
                texts=[""],
                embedding=self.embeddings
            )
            return

         # 分割文档
        splits = self.text_splitter.split_documents(documents)
        print(f"文档分割为 {len(splits)} 个片段")

        # 创建向量存储
        self.vectorstore = FAISS.from_documents(
            documents=splits,
            embedding=self.embeddings
        )

        # 持久化索引
        self._save_index()
        print("向量索引创建完成并已保存")

    def _load_or_create_index(self):
        """加载已有索引或创建新索引"""
        index_path = self.persist_dir / "faiss_index"

        if index_path.exists():
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

    def get_context_for_query(self, query: str, top_k: Optional[int] = None) -> str:
        """
        为查询获取格式化的上下文文本

        Args:
            query: 查询文本
            top_k: 返回结果数量

        Returns:
            格式化的上下文字符串
        """
        results = self.search(query, top_k)

        if not results:
            return ""

        context_parts = []
        for i, result in enumerate(results, 1):
            filename = result["metadata"].get("filename", "未知文件")
            content = result["content"]
            context_parts.append(f"[规则来源 {i}: {filename}]\n{content}")

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
            "怎么玩", "什么意思", "什么是", "如何"
        ]

        content = str(text or "").strip().lower()
        return any(keyword in content for keyword in game_keywords)