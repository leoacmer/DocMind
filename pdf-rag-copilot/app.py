import os
import shutil
import time
from pathlib import Path

# 必须在其他 import 之前，避免 Hugging Face 下载模型时走代理失败
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ── 页面配置 ──────────────────────────────────────────
st.set_page_config(
    page_title="DocMind — 智能文档助手",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── 自定义 CSS ────────────────────────────────────────
st.markdown("""
<style>
    /* ── 标题 ── */
    .main-title {
        font-size: 2rem; font-weight: 700; color: #4f46e5; margin-bottom: 0;
    }
    .main-subtitle { color: #6b7280; font-size: 0.9rem; }

    /* ── 统计卡片 ── */
    .stat-card {
        border: 1px solid #e5e7eb; border-radius: 12px;
        padding: 16px 20px; text-align: center;
        transition: box-shadow 0.2s ease;
    }
    .stat-card:hover { box-shadow: 0 2px 12px rgba(79,70,229,0.1); }
    .stat-number { font-size: 1.6rem; font-weight: 700; color: #4f46e5; }
    .stat-label { color: #6b7280; font-size: 0.8rem; }

    /* ── 文件标签 ── */
    .file-chip {
        display: inline-flex; align-items: center; gap: 5px;
        border: 1px solid #e5e7eb; border-radius: 16px;
        padding: 4px 12px; margin: 2px; font-size: 0.8rem;
        transition: border-color 0.2s;
    }
    .file-chip:hover { border-color: #4f46e5; }
    .file-chip .ext-badge {
        background: #4f46e5; color: #fff; border-radius: 8px;
        padding: 1px 7px; font-size: 0.68rem; font-weight: 600;
        text-transform: uppercase;
    }

    /* ── 按钮圆角 ── */
    .stButton > button { border-radius: 10px; font-weight: 500; }
</style>
""", unsafe_allow_html=True)


# ── 初始化 Session State ──────────────────────────────
def _vector_store_exists() -> bool:
    d = Path("vector_store")
    return d.exists() and any(d.iterdir())

def _restore_state_from_chroma():
    """从持久化的 Chroma 向量库恢复统计信息"""
    try:
        from src.database import list_file_stats, get_chunk_count
        count = get_chunk_count()
        if count == 0:
            return 0, [], 0
        file_stats = list_file_stats()
        files = [fs["source"] for fs in file_stats]
        image_count = sum(fs["images"] for fs in file_stats)
        return count, files, image_count
    except Exception:
        return 0, [], 0

if "vector_store_ready" not in st.session_state:
    exists = _vector_store_exists()
    st.session_state.vector_store_ready = exists
    if exists:
        cnt, files, imgs = _restore_state_from_chroma()
        st.session_state.chunk_count = cnt
        st.session_state.processed_files = files
        st.session_state.total_images = imgs
    else:
        st.session_state.chunk_count = 0
        st.session_state.processed_files = []
        st.session_state.total_images = 0

for k, v in {"messages": [], "total_queries": 0}.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ═══════════════════════════════════════════════════════
#  侧边栏
# ═══════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### ⚙️ 设置")

    # ── API 配置 ──
    with st.expander("🔑 API 配置", expanded=False):
        api_key = st.text_input(
            "DashScope API Key",
            type="password",
            value=os.getenv("DASHSCOPE_API_KEY", ""),
            placeholder="sk-...",
        )
        if api_key:
            os.environ["DASHSCOPE_API_KEY"] = api_key

    # ── 模型 ──
    with st.expander("🧠 模型选择", expanded=False):
        llm_model = st.selectbox("对话模型", ["qwen-max", "qwen-plus", "qwen-turbo"], index=0)
        os.environ["LLM_MODEL_NAME"] = llm_model
        embedding_model = st.selectbox("Embedding 模型", ["text-embedding-v4", "text-embedding-v3"], index=0)
        os.environ["EMBEDDING_MODEL_NAME"] = embedding_model

    # ── 切分参数 ──
    with st.expander("📐 文本切分", expanded=False):
        chunk_size = st.slider("Chunk Size", 200, 2000, 600, 100)
        chunk_overlap = st.slider("Chunk Overlap", 0, 500, 60, 10)

    # ── 检索参数 ──
    with st.expander("🔍 检索设置", expanded=False):
        top_k = st.slider("检索数量 (Top-K)", 1, 10, 3, 1)
        use_reranker = st.checkbox("启用 Reranker 重排序", value=True, help="使用 BGE-Reranker 对初检文档进行语义重排序，提升检索精度")
        if use_reranker:
            reranker_model = st.selectbox("Reranker 模型", ["BAAI/bge-reranker-v2-m3", "BAAI/bge-reranker-base", "BAAI/bge-reranker-large"], index=0)
            fetch_k = st.slider("初检数量 (Fetch-K)", top_k, 50, max(top_k, 20), 5, help="先多召回，再经 Reranker 筛选")
        else:
            fetch_k = top_k
            reranker_model = "BAAI/bge-reranker-v2-m3"

    # ── 文件上传 ──
    st.divider()
    st.markdown("### 📤 上传文档")
    st.caption("支持 PDF / DOCX / PPTX / XLSX / HTML / MD / TXT / 图片")
    uploaded_files = st.file_uploader(
        "拖拽或点击上传",
        type=["pdf", "docx", "pptx", "xlsx", "html", "htm", "md", "txt", "png", "jpg", "jpeg"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded_files and st.button("🔨 解析并构建向量库", type="primary", use_container_width=True):
        if not api_key:
            st.error("请先填写 API Key")
        else:
            all_chunks = []
            progress = st.progress(0, text="正在解析...")
            file_names = []

            for i, file in enumerate(uploaded_files):
                progress.progress((i) / len(uploaded_files), text=f"解析中: {file.name}")
                data_dir = Path("data")
                data_dir.mkdir(exist_ok=True)
                file_path = data_dir / file.name
                file_path.write_bytes(file.getvalue())

                from src.parser import load_and_split_document
                chunks = load_and_split_document(str(file_path), chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                all_chunks.extend(chunks)
                file_names.append(file.name)

            progress.progress(0.9, text="正在构建向量库...")
            from src.database import create_vector_store
            create_vector_store(all_chunks)

            st.session_state.vector_store_ready = True
            st.session_state.chunk_count = len(all_chunks)
            st.session_state.processed_files = file_names
            st.session_state.total_images = sum(
                1 for c in all_chunks if c.metadata.get("type") == "image"
            )

            progress.progress(1.0, text="完成！")
            time.sleep(0.5)
            progress.empty()
            st.toast(f"✅ 解析完成 — {len(all_chunks)} 个文本块, {len(file_names)} 个文件")
            st.rerun()

    # ── 向量库状态 ──
    st.divider()
    st.markdown("### 📊 向量库状态")
    if st.session_state.vector_store_ready:
        total_chunks = 0
        total_imgs = 0
        try:
            from src.database import list_file_stats
            file_stats = list_file_stats()
        except Exception:
            file_stats = []

        if file_stats:
            total_chunks = sum(fs["chunks"] for fs in file_stats)
            total_imgs = sum(fs["images"] for fs in file_stats)
            st.success(f"🟢 {total_chunks} 文本块 / {total_imgs} 图 / {len(file_stats)} 文件")

            for fs in file_stats:
                fname = Path(fs["source"]).name
                col_info, col_btn = st.columns([4, 1])
                with col_info:
                    st.caption(f"📄 {fname} — {fs['chunks']}块 {fs['images']}图")
                with col_btn:
                    if st.button("🗑️", key=f"del_{fs['source']}", help=f"删除 {fname} 的向量"):
                        from src.database import delete_by_source, get_chunk_count
                        n = delete_by_source(fs["source"])
                        st.toast(f"已删除 {n} 条向量: {fname}")
                        # 刷新 session state
                        remaining = get_chunk_count()
                        if remaining == 0:
                            st.session_state.vector_store_ready = False
                            st.session_state.chunk_count = 0
                            st.session_state.total_images = 0
                            st.session_state.processed_files = []
                        else:
                            st.session_state.chunk_count = remaining
                            st.session_state.processed_files = [
                                s["source"] for s in list_file_stats()
                            ]
                            st.session_state.total_images = sum(
                                s["images"] for s in list_file_stats()
                            )
                        st.rerun()
        else:
            st.warning("⚠️ 无法读取文件列表")

        # 清空全部
        if st.button("🗑️ 清空全部向量库", use_container_width=True, type="secondary"):
            from src.database import delete_all
            n = delete_all()
            shutil.rmtree("vector_store", ignore_errors=True)
            shutil.rmtree("extracted_images", ignore_errors=True)
            shutil.rmtree("data", ignore_errors=True)
            for k in ["vector_store_ready", "chunk_count", "processed_files", "messages", "total_queries", "total_images"]:
                st.session_state[k] = [] if k in ("processed_files", "messages") else 0 if k in ("chunk_count", "total_queries", "total_images") else False
            st.toast(f"已清空全部 {n} 条向量")
            st.rerun()
    else:
        st.info("⚪ 尚未构建向量库")

    # ── 评估 ──
    st.divider()
    st.markdown("### 📈 质量评估")
    if not st.session_state.vector_store_ready:
        st.caption("请先构建向量库")
    else:
        eval_n = st.slider("生成测试问题数", 3, 10, 5, 1, key="eval_n")
        if st.button("🚀 运行评估", use_container_width=True):
            with st.spinner("正在生成测试问题并评估..."):
                from src.eval import generate_test_questions, evaluate_rag
                from src.database import load_vector_store
                from langchain_core.documents import Document

                vs = load_vector_store()
                all_docs = vs.get(include=["documents", "metadatas"])
                # 只取文本类型 chunk 用于生成问题
                text_docs = [
                    Document(page_content=all_docs["documents"][i], metadata=all_docs["metadatas"][i] or {})
                    for i in range(len(all_docs["documents"]))
                    if all_docs["metadatas"][i] and all_docs["metadatas"][i].get("type") != "image"
                ]

                if not text_docs:
                    st.warning("没有可用于生成问题的文本块（当前向量库可能只有图片）")
                    st.stop()

                questions = generate_test_questions(text_docs, n=eval_n)

                if not questions:
                    st.warning("无法自动生成测试问题，请检查文档内容")
                    st.stop()

                # 构建 RAG chain（带/不带 Reranker）
                retriever = vs.as_retriever(search_kwargs={"k": fetch_k})
                if use_reranker:
                    from src.reranker import RerankerRetriever
                    retriever = RerankerRetriever(retriever, model_name=reranker_model, top_n=top_k, fetch_k=fetch_k)
                from src.llm import get_rag_chain
                rag_chain = get_rag_chain(retriever)

                result = evaluate_rag(rag_chain, questions)

            st.success("评估完成！")
            st.markdown("#### 📊 评估指标")
            cols = st.columns(len(result["metrics"]))
            metric_labels = {
                "faithfulness": ("📝 忠实度", "答案是否基于检索内容"),
                "answer_relevancy": ("🎯 答案相关性", "答案与问题的相关程度"),
                "context_precision": ("🔍 上下文精确度", "检索内容是否精准"),
                "context_recall": ("📋 上下文召回率", "是否检索到所有相关内容"),
            }
            for i, (k, v) in enumerate(result["metrics"].items()):
                with cols[i % len(cols)]:
                    label, desc = metric_labels.get(k, (k, ""))
                    color = "#22c55e" if v >= 0.7 else "#f59e0b" if v >= 0.4 else "#ef4444"
                    st.markdown(f"""
                    <div class="stat-card">
                        <div class="stat-number" style="color:{color}">{v:.3f}</div>
                        <div class="stat-label">{label}</div>
                        <div style="font-size:0.7rem;color:#9ca3af">{desc}</div>
                    </div>
                    """, unsafe_allow_html=True)

            with st.expander("📋 评估详情", expanded=False):
                st.caption(f"基于 {len(questions)} 个自动生成的问题")
                for i, d in enumerate(result["details"], 1):
                    st.markdown(f"**Q{i}** {d['question']}")
                    st.caption(f"A: {d['answer'][:300]}")


# ═══════════════════════════════════════════════════════
#  主区域 — 顶部信息栏
# ═══════════════════════════════════════════════════════
col_title, col_stats = st.columns([3, 2])
with col_title:
    st.markdown('<p class="main-title">🧠 DocMind</p>', unsafe_allow_html=True)
    st.markdown('<p class="main-subtitle">智能文档助手 — 上传文档，自由问答</p>', unsafe_allow_html=True)

with col_stats:
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-number">{st.session_state.chunk_count}</div>
            <div class="stat-label">📄 文本块</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-number">{st.session_state.total_images}</div>
            <div class="stat-label">🖼️ 图片</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-number">{st.session_state.total_queries}</div>
            <div class="stat-label">💬 问答次数</div>
        </div>
        """, unsafe_allow_html=True)

# ── 已处理文件 ──
if st.session_state.processed_files:
    chips_html = ""
    for f in st.session_state.processed_files:
        ext = Path(f).suffix.lstrip(".")
        chips_html += f'<span class="file-chip"><span class="ext-badge">{ext}</span> {Path(f).stem[:30]}</span>'
    st.markdown(f'<div style="margin: 8px 0 16px;">{chips_html}</div>', unsafe_allow_html=True)

st.divider()

# ═══════════════════════════════════════════════════════
#  对话区域
# ═══════════════════════════════════════════════════════
st.markdown('<div class="chat-container">', unsafe_allow_html=True)

for idx, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        # 展示关联图片
        if msg.get("images"):
            existing = [p for p in msg["images"] if Path(p).exists()]
            if existing:
                cols = st.columns(min(len(existing), 3))
                for i, img_path in enumerate(existing, 1):
                    with cols[(i - 1) % 3]:
                        st.image(img_path, use_container_width=True)
                        # 尝试从 sources 中匹配描述
                        caption = f"图{i}"
                        if msg.get("image_descs") and i <= len(msg["image_descs"]):
                            desc = msg["image_descs"][i - 1]
                            caption = f"图{i}：{desc[:100]}{'...' if len(desc) > 100 else ''}"
                        st.caption(caption)

        # 参考来源
        if msg.get("sources"):
            with st.expander("📎 参考来源", expanded=False):
                for i, src in enumerate(msg["sources"], 1):
                    st.caption(f"来源 {i}")
                    st.text(src[:500])

        # 操作按钮
        btn_cols = st.columns([1, 1, 10])
        with btn_cols[0]:
            if st.button("👍", key=f"like_{idx}", help="有帮助"):
                st.toast("感谢反馈！", icon="💜")
        with btn_cols[1]:
            if st.button("👎", key=f"dislike_{idx}", help="不准确"):
                st.toast("已记录，我们会持续改进", icon="🙏")

st.markdown('</div>', unsafe_allow_html=True)

# ── 底部操作栏 ──
action_col1, action_col2, action_col3 = st.columns([1, 1, 6])
with action_col1:
    if st.session_state.messages and st.button("🗑️ 清空对话", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
with action_col2:
    if st.session_state.messages and st.button("📥 导出对话", use_container_width=True):
        md = "\n\n---\n\n".join(
            f"**{m['role'].upper()}**\n{m['content']}" for m in st.session_state.messages
        )
        st.download_button("下载对话记录", md, "conversation.md", "text/markdown", key="download_conv")

# ── 输入框 ──
placeholder = (
    "输入你的问题，按 Enter 发送..."
    if st.session_state.vector_store_ready
    else "请先上传文档并构建向量库"
)
if question := st.chat_input(placeholder, disabled=not st.session_state.vector_store_ready):
    st.session_state.total_queries += 1
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("🤔 正在检索和分析..."):
            from src.database import load_vector_store
            from src.llm import get_rag_chain

            vector_store = load_vector_store()
            retriever = vector_store.as_retriever(search_kwargs={"k": fetch_k})

            if use_reranker:
                from src.reranker import RerankerRetriever
                retriever = RerankerRetriever(
                    retriever, model_name=reranker_model, top_n=top_k, fetch_k=fetch_k
                )

            rag_chain = get_rag_chain(retriever)

            response = rag_chain.invoke({"input": question})
            answer = response["answer"]
            source_docs = response.get("context", [])

            st.markdown(answer)

            image_sources = [
                d for d in source_docs
                if d.metadata.get("type") == "image"
                and d.metadata.get("image_path")
                and Path(d.metadata["image_path"]).exists()
            ]
            if image_sources:
                cols = st.columns(min(len(image_sources), 3))
                for i, doc in enumerate(image_sources, 1):
                    with cols[(i - 1) % 3]:
                        # 从 page_content 提取描述（去掉 [图片描述] 前缀）
                        desc = doc.page_content
                        if desc.startswith("[图片描述]"):
                            desc = desc[len("[图片描述]"):].strip()
                        st.image(doc.metadata["image_path"], use_container_width=True)
                        st.caption(f"图{i}：{desc[:120]}{'...' if len(desc) > 120 else ''}")

            if source_docs:
                with st.expander("📎 参考来源", expanded=False):
                    img_idx = 0
                    ref_idx = 0
                    for doc in source_docs:
                        doc_type = doc.metadata.get("type", "text")
                        if doc_type == "image":
                            img_idx += 1
                            desc = doc.page_content
                            if desc.startswith("[图片描述]"):
                                desc = desc[len("[图片描述]"):].strip()
                            st.markdown(f"**图{img_idx}**（🖼️ 图片描述）")
                            st.caption(desc[:500])
                        else:
                            ref_idx += 1
                            st.markdown(f"**来源 {ref_idx}**")
                            st.text(doc.page_content[:500])

            # 检索统计
            st.caption(f"🔍 检索 {len(source_docs)} 条参考 · Top-{top_k} · {st.session_state.chunk_count} 个文本块")

    image_paths = [
        d.metadata["image_path"] for d in source_docs
        if d.metadata.get("type") == "image" and d.metadata.get("image_path")
    ]
    image_descs = []
    for d in source_docs:
        if d.metadata.get("type") == "image":
            desc = d.page_content
            if desc.startswith("[图片描述]"):
                desc = desc[len("[图片描述]"):].strip()
            image_descs.append(desc)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": [d.page_content for d in source_docs] if source_docs else [],
        "images": image_paths,
        "image_descs": image_descs,
    })
    st.rerun()
