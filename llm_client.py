import os
import threading
import time


from openai import OpenAI
from typing import Optional
from bot_core.config import get_secret

# 1. 核心组件现在位于 langchain_core
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
# 2. OpenAI 适配器位于 langchain_openai
from langchain_openai import ChatOpenAI
# 3. 记忆模块：优先使用 LangChain 新的 Store/create_agent API（若可用），否则回退到本地实现。
import warnings

# 尝试导入 LangChain 的 InMemory Store（不同版本放在不同包下）
LCInMemoryStore = None
try:
    from langchain.stores import InMemoryStore as LCInMemoryStore  # type: ignore
    LCInMemoryStore = LCInMemoryStore
except Exception:
    try:
        from langchain_core.stores import InMemoryStore as LCInMemoryStore  # type: ignore
        LCInMemoryStore = LCInMemoryStore
    except Exception:
        LCInMemoryStore = None


if LCInMemoryStore is not None:
    class ConversationBufferMemory:
        """基于 LangChain 新 Store 的适配器，兼容原有接口。"""
        def __init__(self, memory_key: str = "history", return_messages: bool = True, max_token_limit: int = 3000):
            try:
                self.store = LCInMemoryStore()
            except Exception:
                # 如果构造失败，回退到内存列表
                self.store = None
            self._history = []

        def clear(self) -> None:
            if getattr(self, "store", None) is not None:
                try:
                    # 不同实现可能有不同方法，尽力调用常见的 clear/remove_all
                    if hasattr(self.store, "clear"):
                        self.store.clear()
                        return
                    if hasattr(self.store, "remove_all"):
                        self.store.remove_all()
                        return
                except Exception:
                    pass
            self._history.clear()

        def load_memory_variables(self, _: dict) -> dict:
            if getattr(self, "store", None) is not None:
                try:
                    if hasattr(self.store, "get_all"):
                        items = list(self.store.get_all() or [])
                        # 保证 items 为 [{'role':..., 'content':...}, ...]
                        return {"history": items}
                    if hasattr(self.store, "items"):
                        items = list(self.store.items() or [])
                        return {"history": items}
                except Exception:
                    pass
            return {"history": list(self._history)}

        def save_context(self, inputs: dict, outputs: dict) -> None:
            inp = inputs.get("input")
            out = outputs.get("output")
            # 使用 LangChain 聊天消息格式
            entry_user = {"role": "user", "content": inp}
            entry_assistant = {"role": "assistant", "content": out}
            if getattr(self, "store", None) is not None:
                try:
                    if hasattr(self.store, "add"):
                        # 尝试以两条消息的形式写入 store
                        try:
                            self.store.add(entry_user)
                        except Exception:
                            pass
                        try:
                            self.store.add(entry_assistant)
                        except Exception:
                            pass
                        return
                except Exception:
                    pass
            # 回退：保持本地列表
            if inp:
                self._history.append(entry_user)
            if out:
                self._history.append(entry_assistant)

else:
    # 回退实现（当系统上没有 LangChain Store）
    class ConversationBufferMemory:
        def __init__(self, memory_key: str = "history", return_messages: bool = True, max_token_limit: int = 3000):
            self._history = []

        def clear(self) -> None:
            self._history.clear()

        def load_memory_variables(self, _: dict) -> dict:
            return {"history": list(self._history)}

        def save_context(self, inputs: dict, outputs: dict) -> None:
            inp = inputs.get("input")
            out = outputs.get("output")
            if inp:
                self._history.append({"role": "user", "content": inp})
            if out:
                self._history.append({"role": "assistant", "content": out})

# 静默原先 ConversationBufferMemory 的弃用提示，避免启动时刷屏
warnings.filterwarnings("ignore", message=r".*ConversationBufferMemory.*")



# 支持环境变量优先，其次读取统一配置文件。
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY") or get_secret("dashscope_api_key", "")
DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DASHSCOPE_MODEL = "qwen-plus"
MAX_HISTORY_ROUNDS = 10
RATE_LIMIT_SECONDS = 3
MAX_REPLY_CHARS = 50
LONG_REPLY_MAX_TOKENS = 900
SEND_CHUNK_CHARS = 50


def _is_story_request(text: str) -> bool:
    content = str(text or "").strip()
    if not content:
        return False
    story_markers = ("讲故事", "说个故事", "编个故事", "故事", "小说", "续写")
    return any(marker in content for marker in story_markers)


def _split_text_by_chars(text: str, chunk_size: int = SEND_CHUNK_CHARS) -> list[str]:
    raw = str(text or "").strip()
    if not raw:
        return []
    size = max(int(chunk_size), 1)
    return [raw[i : i + size] for i in range(0, len(raw), size)]

# --- LangChain 组件初始化 ---
# 1. 定义 Prompt 模板
prompt = ChatPromptTemplate.from_messages([
    ("system", """你是一个有独立见解的聊天伙伴，性格自然不机械。请遵守以下原则：
1. 【保持独立】不盲目附和或迎合任何人，遇到主观评价请给出自己的客观看法。
2. 【逻辑判断】如果前后说法矛盾，请指出差异并给出中立或温和的结论，不要简单重复对方的话。
3. 【玩家口吻】回复像真实玩家一样简短自然，避免说教、AI腔或过度礼貌。
4. 【信息不足】遇到模糊话题可委婉表达不确定，或分享自己的视角，不强行站队。
5. 严格控制在 {max_chars} 字以内，直接输出回复内容。"""),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
])

prompt_long = ChatPromptTemplate.from_messages([
    ("system", """你是一个有独立见解的聊天伙伴，性格自然不机械。请遵守以下原则：
1. 【保持独立】不盲目附和或迎合任何人，遇到主观评价请给出自己的客观看法。
2. 【逻辑判断】如果前后说法矛盾，请指出差异并给出中立或温和的结论，不要简单重复对方的话。
3. 【玩家口吻】回复像真实玩家一样自然，避免说教、AI腔或过度礼貌。
4. 【信息不足】遇到模糊话题可委婉表达不确定，或分享自己的视角，不强行站队。
5. 当前用户在请求故事类内容，你可以给出较完整回复。"""),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
])
# 2. 定义 LLM 模型
llm = ChatOpenAI(
    model=DASHSCOPE_MODEL,
    base_url=DASHSCOPE_BASE_URL,
    api_key=DASHSCOPE_API_KEY,
    temperature=0.7,
    top_p=0.85,
    max_tokens=MAX_REPLY_CHARS * 2,  # 在 API 层面限制最大 token
)

llm_long = ChatOpenAI(
    model=DASHSCOPE_MODEL,
    base_url=DASHSCOPE_BASE_URL,
    api_key=DASHSCOPE_API_KEY,
    temperature=0.8,
    top_p=0.9,
    max_tokens=LONG_REPLY_MAX_TOKENS,
)

# 3. 定义 Memory（对话历史）
memory = ConversationBufferMemory(
    memory_key="history",
    return_messages=True,
    max_token_limit=3000  # 限制历史记录的总 token 数，超出自动遗忘
)
# --- 业务逻辑封装 ---
_lock = threading.Lock()
_last_request_time: float = 0.0
_inflight: bool = False
def clear_conversation_history() -> None:
    """清空会话历史。"""
    memory.clear()
def ask_llm(message: str) -> str:
    """向大模型发送文本并返回回复。"""
    text = str(message or "").strip()
    if not text:
        raise ValueError("message 不能为空")

    if text == "终止对话":
        clear_conversation_history()
        return "已终止对话，历史消息已清空。"

    # 限流逻辑（保持原有设计）
    global _last_request_time, _inflight
    now = time.time()
    with _lock:
        if now - _last_request_time < RATE_LIMIT_SECONDS:
            return ""  # 丢弃频繁请求
        if _inflight:
            return ""  # 丢弃并发请求
        _inflight = True
        _last_request_time = now

    try:
        story_mode = _is_story_request(text)
        # 构建 Chain
        # 使用 RunnablePassthrough 将 history 注入到 Prompt 中
        if story_mode:
            chain = (
                {
                    "input": RunnablePassthrough(),
                    "history": lambda _: memory.load_memory_variables({})["history"],
                }
                | prompt_long
                | llm_long
            )
        else:
            chain = (
                {
                    "input": RunnablePassthrough(),
                    "history": lambda _: memory.load_memory_variables({})["history"],
                    "max_chars": lambda _: MAX_REPLY_CHARS
                }
                | prompt
                | llm
            )

        # 调用并获取结果
        response = chain.invoke(text)
        reply = response.content

        # 更新 Memory
        memory.save_context({"input": text}, {"output": reply})

        # 非故事模式下安全截断（防止模型忽略指令）
        if (not story_mode) and len(reply) > MAX_REPLY_CHARS:
            reply = reply[:MAX_REPLY_CHARS]

        return reply

    except Exception as e:
        print(f"LLM call error: {e}")
        return ""
    finally:
        with _lock:
            _inflight = False


def ask_llm_chunks(message: str, chunk_size: int = SEND_CHUNK_CHARS) -> list[str]:
    """调用 LLM 并按固定长度切分，适合分条发送到房间。"""
    reply = ask_llm(message)
    if not reply:
        return []
    return _split_text_by_chars(reply, chunk_size=chunk_size)


if __name__ == "__main__":
    print(ask_llm("今天天气怎么样？"))
    time.sleep(4)
    print(ask_llm("我刚刚问了你什么？"))
    time.sleep(4)
    print(ask_llm("1+1="))
    time.sleep(4)
    print(ask_llm("2+2="))
    time.sleep(4)
    print(ask_llm("3+3="))
    time.sleep(4)
    print(ask_llm("按照规律，下一个问题是？"))
