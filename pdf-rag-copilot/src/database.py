import os
from typing import List

from dashscope import TextEmbedding
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document


class DashScopeEmbeddings(Embeddings):
    """DashScope 原生 Embedding，兼容 LangChain"""

    def __init__(self, model: str = "text-embedding-v4", api_key: str | None = None):
        self.model = model
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        result = []
        for text in texts:
            resp = TextEmbedding.call(model=self.model, input=text, api_key=self.api_key)
            if resp.status_code != 200:
                raise RuntimeError(f"Embedding 失败: {resp.message}")
            result.append(resp.output["embeddings"][0]["embedding"])
        return result

    def embed_query(self, text: str) -> List[float]:
        resp = TextEmbedding.call(model=self.model, input=text, api_key=self.api_key)
        if resp.status_code != 200:
            raise RuntimeError(f"Embedding 失败: {resp.message}")
        return resp.output["embeddings"][0]["embedding"]


def create_vector_store(chunks, persist_directory: str = "vector_store"):
    """创建向量数据库并持久化"""
    embeddings = DashScopeEmbeddings()
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory,
    )
    return vector_store


def load_vector_store(persist_directory: str = "vector_store"):
    """加载已有向量数据库"""
    embeddings = DashScopeEmbeddings()
    return Chroma(persist_directory=persist_directory, embedding_function=embeddings)


def list_file_stats(persist_directory: str = "vector_store") -> List[dict]:
    """列出向量库中每个文件的统计信息。

    Returns:
        [{ "source": "file.pdf", "chunks": 42, "images": 3 }, ...]
    """
    from pathlib import Path
    d = Path(persist_directory)
    if not d.exists() or not any(d.iterdir()):
        return []

    try:
        vs = load_vector_store(persist_directory)
        collection = vs._collection
        results = collection.get(include=["metadatas"])
        if not results["ids"]:
            return []

        stats: dict[str, dict] = {}
        for meta in results["metadatas"]:
            if not meta:
                continue
            src = meta.get("source", "unknown")
            if src not in stats:
                stats[src] = {"source": src, "chunks": 0, "images": 0}
            stats[src]["chunks"] += 1
            if meta.get("type") == "image":
                stats[src]["images"] += 1

        return list(stats.values())
    except Exception:
        return []


def delete_by_source(source: str, persist_directory: str = "vector_store") -> int:
    """按源文件删除向量，返回删除数量。"""
    try:
        vs = load_vector_store(persist_directory)
        collection = vs._collection
        results = collection.get(include=["metadatas"])
        ids_to_delete = [
            id_ for id_, meta in zip(results["ids"], results["metadatas"])
            if meta and meta.get("source") == source
        ]
        if ids_to_delete:
            collection.delete(ids=ids_to_delete)
        return len(ids_to_delete)
    except Exception:
        return 0


def delete_all(persist_directory: str = "vector_store") -> int:
    """清空整个向量库，返回删除数量。"""
    try:
        vs = load_vector_store(persist_directory)
        collection = vs._collection
        count = collection.count()
        if count > 0:
            results = collection.get()
            collection.delete(ids=results["ids"])
        return count
    except Exception:
        return 0


def get_chunk_count(persist_directory: str = "vector_store") -> int:
    """返回向量库中的 chunk 总数。"""
    try:
        from pathlib import Path
        if not Path(persist_directory).exists():
            return 0
        vs = load_vector_store(persist_directory)
        return vs._collection.count()
    except Exception:
        return 0
