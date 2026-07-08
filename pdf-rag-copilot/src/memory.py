import re
import sqlite3
from pathlib import Path
from typing import List, Dict

from langchain_classic.memory import ConversationSummaryBufferMemory

DB_DIR = Path("./conversations")

# ── CJK 字符范围 ──
_CJK_RE = re.compile(r"[一-鿿㐀-䶿豈-﫿]")


def _approx_tokens(text: str) -> int:
    """近似 token 计数：中文字符 ≈ 1 token，英文 ≈ 0.25 token/char"""
    cjk = len(_CJK_RE.findall(text))
    other = len(text) - cjk
    return cjk + max(1, other // 4)


def _patch_llm_token_counter(llm):
    """给非 OpenAI 模型打补丁，提供近似 token 计数。

    ConversationSummaryBufferMemory 内部会调用
    llm.get_num_tokens_from_messages() 判断是否触发总结，
    而 qwen-max 等模型不在 tiktoken 注册表中，会抛 NotImplementedError。

    使用 object.__setattr__ 绕过 Pydantic v2 的字段验证。
    """
    def get_num_tokens_from_messages(messages):
        total = 0
        for msg in messages:
            content = msg.content if hasattr(msg, "content") else str(msg)
            if isinstance(content, str):
                total += _approx_tokens(content)
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and "text" in block:
                        total += _approx_tokens(block["text"])
        return total

    object.__setattr__(llm, "get_num_tokens_from_messages", get_num_tokens_from_messages)


def _ensure_dir():
    DB_DIR.mkdir(exist_ok=True)


def _db_path(session_id: str) -> str:
    _ensure_dir()
    return str(DB_DIR / f"{session_id}.db")


def add_message(session_id: str, role: str, content: str):
    conn = sqlite3.connect(_db_path(session_id))
    conn.execute(
        "CREATE TABLE IF NOT EXISTS messages ("
        "  id INTEGER PRIMARY KEY AUTOINCREMENT,"
        "  role TEXT NOT NULL,"
        "  content TEXT NOT NULL,"
        "  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
        ")"
    )
    conn.execute("INSERT INTO messages (role, content) VALUES (?, ?)", (role, content))
    conn.commit()
    conn.close()


def get_messages(session_id: str, limit: int = 50) -> List[Dict[str, str]]:
    db_path = _db_path(session_id)
    if not Path(db_path).exists():
        return []
    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        "SELECT role, content FROM messages ORDER BY id ASC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    return [{"role": r, "content": c} for r, c in rows]


def clear_messages(session_id: str):
    db_path = _db_path(session_id)
    if Path(db_path).exists():
        conn = sqlite3.connect(db_path)
        conn.execute("DELETE FROM messages")
        conn.commit()
        conn.close()


def get_conversation_memory(session_id: str, llm, max_token_limit: int = 2000):
    """创建或恢复 ConversationSummaryBufferMemory，底层由 SQLite 持久化。

    每次启动时从 SQLite 恢复最近的聊天记录到内存 buffer，
    对话过程中 LangChain 链自动管理 summarization 与 buffer 裁剪。
    """
    _patch_llm_token_counter(llm)

    memory = ConversationSummaryBufferMemory(
        llm=llm,
        max_token_limit=max_token_limit,
        return_messages=True,
        memory_key="chat_history",
    )
    # 兼容不同 langchain_classic 版本的 output_key 设置方式
    memory.output_key = "answer"
    memory.input_key = "question"

    messages = get_messages(session_id)
    for msg in messages:
        if msg["role"] == "user":
            memory.chat_memory.add_user_message(msg["content"])
        elif msg["role"] == "assistant":
            memory.chat_memory.add_ai_message(msg["content"])

    return memory
