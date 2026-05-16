
import threading
import time
from bot_core.config import get_secret, get_current_llm_config
from RagEnhancedRuleLLM import get_rag_llm
# 1. 核心组件现在位于 langchain_core
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
# 2. OpenAI 适配器位于 langchain_openai
from langchain_openai import ChatOpenAI
# 3. 记忆模块：优先使用 LangChain 新的 Store/create_agent API（若可用），否则回退到本地实现。
import warnings
from chatEnhancedLLM import get_chat_enhanced_llm

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



# 从配置文件加载大模型配置
llm_config = get_current_llm_config()
DASHSCOPE_API_KEY = llm_config["api_key"]
DASHSCOPE_BASE_URL = llm_config["base_url"]
DASHSCOPE_MODEL = llm_config["model"]

MAX_HISTORY_ROUNDS = 10
RATE_LIMIT_SECONDS = 3
MAX_REPLY_CHARS = 100
LONG_REPLY_MAX_TOKENS = 900
SEND_CHUNK_CHARS = 50

def _is_story_request(text: str) -> bool:
    content = str(text or "").strip()
    if not content:
        return False
    story_markers = ("讲故事", "说个故事", "编个故事", "故事", "小说", "续写")
    return any(marker in content for marker in story_markers)


def _split_text_by_chars(text: str, chunk_size: int = SEND_CHUNK_CHARS) -> list[str]:
    """智能分割文本，优先在标点符号处断句。
    Args:
        text: 要分割的文本
        chunk_size: 每个片段的目标字符数

    Returns:
        分割后的文本片段列表
    """
    raw = str(text or "").strip()
    if not raw:
        return []
    # 定义可以断句的标点符号（优先级从高到低）
    strong_breaks = {'。', '！', '？', '…', '\n'}  # 强断点：句号、感叹号、问号等
    medium_breaks = {'，', '；', '、', ')', '）', '"', '"'}  # 中断点：逗号、分号等
    weak_breaks = {' ', '—', '-', '(', '（'}  # 弱断点：空格、破折号等
    chunks = []
    start = 0
    text_len = len(raw)
    while start < text_len:
        # 计算当前片段的结束位置
        end = min(start + chunk_size, text_len)

        # 如果已经到达文本末尾，直接截取
        if end >= text_len:
            chunks.append(raw[start:])
            break

        # 在 [start, end] 范围内寻找最佳断点
        best_end = end
        best_priority = -1  # 3=强断点, 2=中断点, 1=弱断点, 0=无断点

        # 从后往前搜索，优先找靠近 chunk_size 的断点
        search_start = max(start, end - chunk_size // 2)  # 至少保留一半长度

        for i in range(end, search_start - 1, -1):
            char = raw[i]

            if char in strong_breaks:
                best_end = i + 1  # 包含标点符号
                best_priority = 3
                break
            elif char in medium_breaks and best_priority < 2:
                best_end = i + 1
                best_priority = 2
            elif char in weak_breaks and best_priority < 1:
                best_end = i + 1
                best_priority = 1

        # 如果没找到合适的断点，就强制在 chunk_size 处切割
        if best_priority == -1:
            best_end = end
        # 提取片段并去除首尾空白
        chunk = raw[start:best_end].strip()
        if chunk:
            chunks.append(chunk)
        start = best_end
    return chunks
# --- LangChain 组件初始化 ---
# 1. 定义基础 Prompt 模板
base_system_prompt = """你是一个有独立见解的聊天伙伴，请牢记你的名字是‘夏凌依’，性格自然不机械。请遵守以下原则：
1. 【保持独立】不盲目附和或迎合任何人，遇到主观评价请给出自己的客观看法。
2. 【逻辑判断】如果前后说法矛盾，请指出差异并给出中立或温和的结论，不要简单重复对方的话。
3. 【玩家口吻】回复像真实玩家一样简短自然，避免说教、AI腔或过度礼貌。
4. 【信息不足】遇到模糊话题可委婉表达不确定，或分享自己的视角，不强行站队。
5. 严格控制在 {max_chars} 字以内，直接输出回复内容。"""

# 普通聊天 Prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", base_system_prompt),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
])

# 故事模式 Prompt
prompt_long = ChatPromptTemplate.from_messages([
    ("system", """发挥你的想象力，讲一个生动有趣且不重复的故事吧！"""),
    ("human", "{input}"),
])

# 游戏规则增强 Prompt
prompt_with_rules = ChatPromptTemplate.from_messages([
    ("system", """你是一个有独立见解的聊天伙伴，性格自然不机械。请遵守以下原则：
1. 【保持独立】不盲目附和或迎合任何人，遇到主观评价请给出自己的客观看法。
2. 【逻辑判断】如果前后说法矛盾，请指出差异并给出中立或温和的结论，不要简单重复对方的话。
3. 【玩家口吻】回复像真实玩家一样简短自然，避免说教、AI腔或过度礼貌。
4. 【信息不足】遇到模糊话题可委婉表达不确定，或分享自己的视角，不强行站队。
5. 严格控制在 {max_chars} 字以内，直接输出回复内容。
6. 【规则遵循】如果提供了游戏规则，请严格基于规则回答，不要编造规则中不存在的内容。

【游戏规则参考】
{rules_context}"""),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
])

# 聊天历史增强 Prompt
prompt_with_chat_history = ChatPromptTemplate.from_messages([
    ("system", """你是一个有独立见解的聊天伙伴，性格自然不机械。请遵守以下原则：
1. 【保持独立】不盲目附和或迎合任何人，遇到主观评价请给出自己的客观看法。
2. 【逻辑判断】如果前后说法矛盾，请指出差异并给出中立或温和的结论，不要简单重复对方的话。
3. 【玩家口吻】回复像真实玩家一样简短自然，避免说教、AI腔或过度礼貌。
4. 【信息不足】遇到模糊话题可委婉表达不确定，或分享自己的视角，不强行站队。
5. 严格控制在 {max_chars} 字以内，直接输出回复内容。
6. 【个性化】根据用户的聊天历史，给出符合其风格和语境的针对性回复。

【聊天历史上下文】
{chat_history_context}"""),
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
    temperature=0.98,
    top_p=0.95,
    max_tokens=LONG_REPLY_MAX_TOKENS,
    presence_penalty=1.2,  # 抑制重复剧情
    frequency_penalty=0.95,  # 抑制重复词句
)

# 3. 定义 Memory（对话历史）
memory = ConversationBufferMemory(
    memory_key="history",
    return_messages=True,
    max_token_limit=10000  # 限制历史记录的总 token 数，超出自动遗忘
)

# 4. 初始化 RAG 系统（延迟加载，避免启动时阻塞）
_rag_llm_instance = None
def _get_rag_llm():
    """懒加载 RAG 实例"""
    global _rag_llm_instance
    if _rag_llm_instance is None:
        try:
            _rag_llm_instance = get_rag_llm()
        except Exception as e:
            print(f"RAG 系统初始化失败: {e}")
    return _rag_llm_instance

# 5. 初始化聊天历史 RAG
_chat_enhanced_llm_instance = None
def _get_chat_enhanced_llm():
    """懒加载聊天增强 LLM 实例"""
    global _chat_enhanced_llm_instance
    if _chat_enhanced_llm_instance is None:
        try:
            _chat_enhanced_llm_instance = get_chat_enhanced_llm()
        except Exception as e:
            print(f"聊天历史 RAG 系统初始化失败: {e}")
    return _chat_enhanced_llm_instance

# --- 业务逻辑封装 ---
_lock = threading.Lock()
_last_request_time: float = 0.0
_inflight: bool = False
def clear_conversation_history() -> None:
    """清空会话历史。"""
    memory.clear()


def ask_llm(message: str, username: str = "") -> str:
    """向大模型发送文本并返回回复。
    Args:
        message: 用户消息
        username: 用户名（可选，用于日志记录）
    """
    text = str(message or "").strip()
    if not text:
        raise ValueError("message 不能为空")
    # 检查是否为终止对话指令（支持多种表达）
    stop_commands = ["终止对话", "清空历史", "重置对话", "清除记忆"]
    if any(cmd in text for cmd in stop_commands):
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
        # 默认使用聊天增强 LLM（带 RAG 个性化上下文）
        chat_enhanced = _get_chat_enhanced_llm()
        if chat_enhanced and username:
            reply = chat_enhanced.ask_with_chat_history(username, text)
            if reply:
                print("chat has enhanced!")
                print(f"[LLM] {username}: {text}")
                # 更新 Memory
                memory.save_context({"input": text}, {"output": reply})

                # 安全截断（防止模型忽略指令）
                if len(reply) > MAX_REPLY_CHARS:
                    reply = reply[:MAX_REPLY_CHARS]

                return reply

        # 如果没有用户名或聊天增强失败，回退到普通 Chain
        print("[LLM] 正在调用 普通 LLM...")
        chain_input = {
            "input": RunnablePassthrough(),
            "history": lambda _: memory.load_memory_variables({})["history"],
            "max_chars": lambda _: MAX_REPLY_CHARS
        }
        chain = chain_input | prompt | llm
        # 调用并获取结果
        response = chain.invoke(text)
        reply = response.content
        # 更新 Memory
        memory.save_context({"input": text}, {"output": reply})

        # 安全截断（防止模型忽略指令）
        if len(reply) > MAX_REPLY_CHARS:
            reply = reply[:MAX_REPLY_CHARS]

        return reply
    except Exception as e:
        print(f"LLM call error: {e}")
        return ""
    finally:
        with _lock:
            _inflight = False


def ask_llm_chunks(message: str,username: str = "", chunk_size: int = SEND_CHUNK_CHARS) -> list[str]:
    """调用 LLM 并按固定长度切分，适合分条发送到房间。"""
    reply = ask_llm(message, username)
    if not reply:
        return []
    return _split_text_by_chars(reply, chunk_size=chunk_size)


def ask_llm_stream(message: str, chunk_size: int = SEND_CHUNK_CHARS):
    """流式调用 LLM，生成一个 chunk 就 yield 一个。
    Args:
        message: 用户输入
        chunk_size: 每个片段的字符数

    Yields:
        str: 每次生成足够字符后返回一个片段
    """
    text = str(message or "").strip()
    if not text:
        raise ValueError("message 不能为空")

    if text == "终止对话":
        clear_conversation_history()
        yield "已终止对话，历史消息已清空。"
        return

    # 限流逻辑
    global _last_request_time, _inflight
    now = time.time()
    with _lock:
        if now - _last_request_time < RATE_LIMIT_SECONDS:
            return  # 丢弃频繁请求
        if _inflight:
            return  # 丢弃并发请求
        _inflight = True
        _last_request_time = now

    try:
        story_mode = _is_story_request(text)

        # 选择对应的 LLM
        selected_llm = llm_long if story_mode else llm

        # 构建 Chain（故事模式不使用历史记忆）
        # 故事模式：直接使用 prompt_long，不注入 history
        chain_input = {
            "input": RunnablePassthrough(),
            "history": lambda _: memory.load_memory_variables({})["history"],
        }
        chain = chain_input | prompt_long | selected_llm
        stream_input = {"input": text}
        # 流式调用 LLM
        buffer = ""
        full_reply = ""

        for chunk in chain.stream(stream_input):
            # chunk.content 是每次生成的文本片段
            if hasattr(chunk, 'content') and chunk.content:
                token = chunk.content
                full_reply += token
                buffer += token

                # 当缓冲区达到 chunk_size 时，yield 出去
                if len(buffer) >= chunk_size:
                    time.sleep(0.5)
                    yield buffer[:chunk_size]
                    buffer = buffer[chunk_size:]

        # 发送剩余的缓冲区内容
        if buffer:
            yield buffer

        # 更新 Memory（流式完成后）
        if full_reply:
            memory.save_context({"input": text}, {"output": full_reply})

            # 非故事模式下安全截断
            if (not story_mode) and len(full_reply) > MAX_REPLY_CHARS:
                full_reply = full_reply[:MAX_REPLY_CHARS]

    except Exception as e:
        print(f"LLM stream error: {e}")
    finally:
        with _lock:
            _inflight = False

def rebuild_rag_index():
    """重建 RAG 索引（当游戏规则文件更新时调用）"""
    rag_llm = _get_rag_llm()
    if rag_llm:
        rag_llm.rebuild_index()
        return True
    return False
def rebuild_chat_history_index():
    """重建聊天历史索引（当聊天记录更新时调用）"""
    chat_llm = _get_chat_enhanced_llm()
    if chat_llm:
        chat_llm.rebuild_index()
        return True
    return False

def add_rule_file(file_path: str):
    """添加新的游戏规则文件到索引"""
    rag_llm = _get_rag_llm()
    if rag_llm:
        rag_llm.add_rule(file_path)
        return True
    return False
if __name__ == "__main__":
    print(ask_llm("你还记得我吗？","风儿吹吹"))
    time.sleep(4)
    print(ask_llm("你叫什么名字rag？","风儿吹吹"))
    time.sleep(4)
    print(ask_llm("你叫什么名字？","风儿吹吹"))
    time.sleep(4)
    print(ask_llm("你好呀","风儿吹吹"))