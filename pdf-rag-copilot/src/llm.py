import os

from langchain_openai import ChatOpenAI
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


def get_rag_chain(retriever):
    """构建 RAG 链（retriever 可以是普通 retriever 或 RerankerRetriever）"""
    llm = ChatOpenAI(
        model=os.getenv("LLM_MODEL_NAME", "qwen-max"),
        openai_api_base=os.getenv("OPENAI_BASE_URL", DASHSCOPE_BASE_URL),
        openai_api_key=os.getenv("DASHSCOPE_API_KEY"),
        temperature=0.2,
    )

    system_prompt = (
        "你是一个专业的文档分析助手。根据以下检索到的上下文内容回答问题。\n"
        "上下文可能包含文本片段和图片描述（以 [图片描述] 开头）。\n"
        "如果上下文中包含图片描述，请在回答中明确指出图片内容的要点，\n"
        "但不要使用'图X'这种编号（因为你不知道前端如何编号），\n"
        "而是直接描述图片内容，如'根据文档中的架构图...'。\n"
        "如果你不知道答案，直接说不知道，请勿捏造事实。\n\n"
        "【上下文信息】:\n{context}"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    return rag_chain


def get_llm():
    """返回底层 LLM 实例（供评估等场景使用）"""
    return ChatOpenAI(
        model=os.getenv("LLM_MODEL_NAME", "qwen-max"),
        openai_api_base=os.getenv("OPENAI_BASE_URL", DASHSCOPE_BASE_URL),
        openai_api_key=os.getenv("DASHSCOPE_API_KEY"),
        temperature=0,
    )
