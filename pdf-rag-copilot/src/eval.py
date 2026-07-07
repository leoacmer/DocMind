"""评估模块：自动生成测试问题并评估 RAG 流水线质量。

指标说明：
  - faithfulness (忠实度): 答案中的陈述是否都能在检索上下文中找到依据
  - answer_relevancy (答案相关性): 答案与问题的语义相关程度
  - context_precision (上下文精确度): 检索到的文档中有多少是对问题有用的
  - context_recall (上下文召回率): 参考答案中的信息有多少被检索到
"""
import os
import re
from typing import List

import numpy as np
from langchain_core.documents import Document

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")


def generate_test_questions(chunks: List[Document], n: int = 5) -> List[str]:
    """基于文档块自动生成测试问题。"""
    import random
    from src.llm import get_llm

    llm = get_llm()
    sample = random.sample(chunks, min(n * 2, len(chunks)))

    questions = []
    for ctx in sample:
        prompt = (
            "根据以下文档内容，生成一个中文问题，要求该问题能够通过文档内容回答。"
            "只输出问题本身，不要加任何前缀或解释。\n\n"
            f"文档内容：\n{ctx.page_content[:800]}"
        )
        resp = llm.invoke(prompt)
        q = resp.content.strip()
        if q and len(q) > 2 and q not in questions:
            questions.append(q)
        if len(questions) >= n:
            break

    return questions


def _cosine_sim(a, b):
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def _eval_faithfulness(answer: str, contexts: List[str], llm) -> float:
    """检测答案陈述是否有上下文支撑。"""
    if not answer or not contexts:
        return 0.0

    ctx_text = "\n---\n".join(contexts)
    prompt = (
        "你的任务是判断「答案」中的每一句话是否都能从「上下文」中找到依据。\n"
        "请按以下规则评分：\n"
        "- 1.0：答案中所有陈述都能在上下文中找到直接支持\n"
        "- 0.7：答案大部分陈述有依据，但有少量细节不在上下文中\n"
        "- 0.4：答案只有少部分能在上下文中找到依据\n"
        "- 0.0：答案完全无法从上下文中推断\n\n"
        f"【上下文】:\n{ctx_text}\n\n"
        f"【答案】:\n{answer}\n\n"
        "请只输出一个 0.0 到 1.0 之间的数字，代表忠实度分数："
    )
    resp = llm.invoke(prompt)
    try:
        return max(0.0, min(1.0, float(resp.content.strip())))
    except ValueError:
        numbers = re.findall(r'[\d.]+', resp.content)
        return float(numbers[0]) if numbers else 0.5


def _eval_answer_relevancy(question: str, answer: str, llm) -> float:
    """检测答案是否紧扣问题。"""
    if not answer or not question:
        return 0.0

    prompt = (
        "你的任务是评估「答案」与「问题」的相关程度。\n"
        "按以下规则评分：\n"
        "- 1.0：答案完全针对问题，直接且准确地回应了问题核心\n"
        "- 0.7：答案大体相关，但有少量偏离\n"
        "- 0.4：答案部分相关，但包含大量无关内容或未触及问题核心\n"
        "- 0.0：答案与问题完全无关\n\n"
        f"【问题】:\n{question}\n\n"
        f"【答案】:\n{answer}\n\n"
        "请只输出一个 0.0 到 1.0 之间的数字，代表相关性分数："
    )
    resp = llm.invoke(prompt)
    try:
        return max(0.0, min(1.0, float(resp.content.strip())))
    except ValueError:
        numbers = re.findall(r'[\d.]+', resp.content)
        return float(numbers[0]) if numbers else 0.5


def _eval_context_precision(question: str, contexts: List[str], llm) -> float:
    """检测检索到的文档中有多少对回答问题有用。"""
    if not contexts:
        return 0.0

    scores = []
    for ctx in contexts:
        prompt = (
            "判断以下文档片段是否包含有助于回答问题的信息。\n"
            "只输出 1（有用）或 0（无用）。\n\n"
            f"【问题】:\n{question}\n\n"
            f"【文档片段】:\n{ctx[:600]}\n\n"
            "分数："
        )
        resp = llm.invoke(prompt)
        try:
            scores.append(float(resp.content.strip()))
        except ValueError:
            scores.append(0.0)

    return float(np.mean(scores)) if scores else 0.0


def _eval_context_recall(contexts: List[str], ground_truth: str, llm) -> float:
    """检测参考答案中的关键信息有多少被检索到。"""
    if not contexts or not ground_truth:
        return 0.0

    ctx_text = "\n---\n".join(contexts)
    prompt = (
        "评估「检索内容」是否覆盖了「参考答案」中的关键信息。\n"
        "按以下规则评分：\n"
        "- 1.0：参考答案的所有关键信息都能在检索内容中找到\n"
        "- 0.7：大部分关键信息被覆盖，少量缺失\n"
        "- 0.4：只有少部分关键信息被覆盖\n"
        "- 0.0：检索内容几乎完全不相关\n\n"
        f"【参考答案】:\n{ground_truth}\n\n"
        f"【检索内容】:\n{ctx_text}\n\n"
        "请只输出一个 0.0 到 1.0 之间的数字："
    )
    resp = llm.invoke(prompt)
    try:
        return max(0.0, min(1.0, float(resp.content.strip())))
    except ValueError:
        numbers = re.findall(r'[\d.]+', resp.content)
        return float(numbers[0]) if numbers else 0.5


def evaluate_rag(rag_chain, questions: List[str], ground_truths: List[str] | None = None) -> dict:
    """运行 RAG 流水线并计算评估指标。

    Args:
        rag_chain: LangChain RAG chain
        questions: 测试问题列表
        ground_truths: 参考答案（可选）

    Returns:
        dict: {"metrics": {...}, "details": [...]}
    """
    from src.llm import get_llm

    llm = get_llm()

    answers, contexts_list = [], []
    for q in questions:
        try:
            resp = rag_chain.invoke({"input": q})
            answers.append(resp.get("answer", ""))
            ctxs = [d.page_content for d in resp.get("context", [])]
            contexts_list.append(ctxs)
        except Exception as e:
            answers.append(f"[ERROR] {e}")
            contexts_list.append([])

    # 计算各项指标
    faithfulness_scores = [
        _eval_faithfulness(a, c, llm) for a, c in zip(answers, contexts_list)
    ]
    relevancy_scores = [
        _eval_answer_relevancy(q, a, llm) for q, a in zip(questions, answers)
    ]
    precision_scores = [
        _eval_context_precision(q, c, llm) for q, c in zip(questions, contexts_list)
    ]

    metrics = {
        "faithfulness": round(float(np.mean(faithfulness_scores)), 4),
        "answer_relevancy": round(float(np.mean(relevancy_scores)), 4),
        "context_precision": round(float(np.mean(precision_scores)), 4),
    }

    # context_recall 需要 ground_truth
    if ground_truths and len(ground_truths) == len(questions):
        recall_scores = [
            _eval_context_recall(c, g, llm)
            for c, g in zip(contexts_list, ground_truths)
        ]
        metrics["context_recall"] = round(float(np.mean(recall_scores)), 4)

    return {
        "metrics": metrics,
        "details": [
            {"question": q, "answer": a, "contexts": c}
            for q, a, c in zip(questions, answers, contexts_list)
        ],
    }
