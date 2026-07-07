"""Reranker: 用 Cross-Encoder 对初检文档进行重排序，提升检索精度。

模型优先级（自动降级）：
  1. BAAI/bge-reranker-v2-m3（中文最佳，需下载 ~2GB）
  2. BAAI/bge-reranker-base（轻量，需下载 ~1GB）
  3. cross-encoder/mmarco-mMiniLMv2-L12-H384-v1（本地缓存，英文）
  4. 降级为无重排序模式
"""
import os
import traceback
from typing import List

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "15")

FALLBACK_MODELS = [
    "BAAI/bge-reranker-v2-m3",
    "BAAI/bge-reranker-base",
    "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
]


def _try_load_model(model_name: str):
    """尝试加载一个 CrossEncoder 模型，优先使用本地缓存。"""
    from sentence_transformers import CrossEncoder

    def _load(local_only: bool):
        return CrossEncoder(model_name, trust_remote_code=True, local_files_only=local_only)

    # 先尝试本地缓存（避免连接 mirror 超时）
    try:
        m = _load(local_only=True)
        print(f"[Reranker] 模型加载成功（本地缓存）: {model_name}")
        return m
    except Exception:
        pass

    # 本地没有则尝试下载
    try:
        m = _load(local_only=False)
        print(f"[Reranker] 模型加载成功（下载）: {model_name}")
        return m
    except Exception:
        pass

    return None


class RerankerRetriever(BaseRetriever):
    """包装一个 base retriever，先多召回 (fetch_k)，再经 Reranker 排序后返回 top_n。

    如果首选模型加载失败，自动尝试备选模型；全部失败则直接返回原检索结果。
    """

    def __init__(self, base_retriever: BaseRetriever, model_name: str = "BAAI/bge-reranker-v2-m3", top_n: int = 3, fetch_k: int = 20):
        super().__init__()
        self._base = base_retriever
        self._top_n = top_n
        self._fetch_k = fetch_k
        self._model_name = model_name
        self._model = None
        self._model_loaded: str | None = None

    def _load_model(self):
        if self._model is not None:
            return

        # 首选模型
        m = _try_load_model(self._model_name)
        if m is not None:
            self._model = m
            self._model_loaded = self._model_name
            return

        # 备选模型
        for fb_name in FALLBACK_MODELS:
            if fb_name == self._model_name:
                continue
            m = _try_load_model(fb_name)
            if m is not None:
                self._model = m
                self._model_loaded = fb_name
                print(f"[Reranker] 降级使用备选模型: {fb_name}")
                return

        print("[Reranker] 所有模型加载失败，降级为无重排序模式")

    def _get_relevant_documents(self, query: str, *, run_manager=None) -> List[Document]:
        # 1. 多召回
        original_kwargs = self._base.search_kwargs.copy()
        self._base.search_kwargs["k"] = self._fetch_k
        docs = self._base.invoke(query)
        self._base.search_kwargs = original_kwargs

        # 2. 尝试加载模型
        self._load_model()

        # 3. 不可用或文档太少 → 直接返回
        if self._model is None or len(docs) <= self._top_n:
            return docs[:self._top_n]

        # 4. 用 CrossEncoder 计算相关性分数
        pairs = [[query, doc.page_content] for doc in docs]
        scores = self._model.predict(pairs, show_progress_bar=False)

        # 5. 按分数降序排序，取 top_n
        scored = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in scored[:self._top_n]]

    async def _aget_relevant_documents(self, query: str, *, run_manager=None) -> List[Document]:
        return self._get_relevant_documents(query, run_manager=run_manager)
