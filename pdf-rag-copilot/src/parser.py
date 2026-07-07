import os
import base64
from pathlib import Path

# ⚠️ 必须在 import docling 之前设置，否则下载模型时可能走代理失败
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.document import PictureItem
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document as LCDocument
from dashscope import MultiModalConversation

IMAGE_DIR = Path("./extracted_images")
IMAGE_DIR.mkdir(parents=True, exist_ok=True)

# 支持的文件格式
SUPPORTED_FORMATS = {".pdf", ".docx", ".pptx", ".xlsx", ".html", ".htm", ".md", ".txt", ".png", ".jpg", ".jpeg"}

# 带图片提取的格式（PDF/DOCX/PPTX 内部可能嵌图）
IMAGE_FORMATS = {".pdf", ".docx", ".pptx"}


def _encode_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _describe_image(image_path: str) -> str | None:
    """用千问多模态模型为图片生成描述"""
    try:
        b64 = _encode_image(image_path)
        prompt = (
            "请详细描述这张图片的内容，用于后续的 RAG 检索。\n"
            "描述中必须包含：\n"
            "1. 图片中出现的所有文字、标签、标题、图号（如'图1-1'、'表2-3'等）\n"
            "2. 图表的数据趋势或架构关系\n"
            "3. 图片的类型（流程图/架构图/数据图表/截图/照片等）\n"
            "如果图片中有图号或标题，请在描述开头明确写出，格式为'[图号: xxx]'。"
        )
        messages = [{
            "role": "user",
            "content": [
                {"text": prompt},
                {"image": f"data:image/png;base64,{b64}"},
            ]
        }]
        resp = MultiModalConversation.call(
            model="qwen-vl-max",
            messages=messages,
            api_key=os.getenv("DASHSCOPE_API_KEY"),
        )
        return resp.output.choices[0].message.content[0]["text"]
    except Exception as e:
        print(f"  [警告] 图片描述生成失败: {e}")
        return None


def _extract_images(doc) -> list[str]:
    """从文档中提取图片，返回保存路径列表"""
    image_paths = []
    for element, level in doc.iterate_items():
        if isinstance(element, PictureItem):
            image_id = element.self_ref.replace("/", "_")
            save_path = IMAGE_DIR / f"{image_id}.png"
            pil_img = element.get_image(doc)
            if pil_img is None:
                continue
            pil_img.save(save_path)
            image_paths.append(str(save_path))
    return image_paths


def load_and_split_document(
    file_path: str,
    chunk_size: int = 600,
    chunk_overlap: int = 60,
    extract_images: bool = True,
):
    """加载任意支持格式的文档，切分为 LangChain Document 块（文本 + 图片描述）"""
    ext = Path(file_path).suffix.lower()
    if ext not in SUPPORTED_FORMATS:
        raise ValueError(f"不支持的文件格式: {ext}，支持: {', '.join(SUPPORTED_FORMATS)}")

    # 根据格式选择解析器
    if ext == ".pdf":
        pipeline_options = PdfPipelineOptions()
        pipeline_options.images_scale = 2.0
        pipeline_options.generate_page_images = True
        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )
    else:
        # DOCX / PPTX / XLSX / HTML / MD / TXT / 图片 → 默认转换器
        converter = DocumentConverter()

    result = converter.convert(file_path)
    doc = result.document

    # 提取图片（仅支持的格式）
    image_paths = []
    if extract_images and ext in IMAGE_FORMATS:
        image_paths = _extract_images(doc)

    # 导出 HTML（保留结构）
    html_content = doc.export_to_html()
    print(f"解析完成：{file_path} → HTML ({len(html_content)} 字符)")

    # 构建 LangChain Documents
    lc_docs = []

    # HTML 文本切分
    if html_content.strip():
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", "。", ".", " ", ""],
        )
        html_doc = LCDocument(page_content=html_content, metadata={"source": file_path, "type": "text"})
        lc_docs.extend(text_splitter.split_documents([html_doc]))

    # 图片描述（多模态）
    if extract_images and image_paths:
        print(f"正在使用千问多模态模型为 {len(image_paths)} 张图片生成语义描述...")
        for img_path in image_paths:
            description = _describe_image(img_path)
            if description:
                lc_docs.append(LCDocument(
                    page_content=f"[图片描述] {description}",
                    metadata={"type": "image", "image_path": img_path, "source": file_path},
                ))
        print("图片描述生成完毕")

    return lc_docs
