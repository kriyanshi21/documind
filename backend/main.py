from __future__ import annotations

import csv
import io
import logging
import os
import re
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Final
from xml.etree import ElementTree as ET

import fitz  # PyMuPDF
import google.generativeai as genai
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field

load_dotenv()

logger = logging.getLogger(__name__)

app = FastAPI(title="DocuMind API", version="1.0.0")

# Needed when the frontend is served from a different origin (e.g. Vite dev server).
_cors_origins = os.getenv(
    "CORS_ORIGINS",
    "http://localhost:5173,http://127.0.0.1:5173",
)
ALLOWED_ORIGINS: Final[list[str]] = [
    origin.strip()
    for origin in _cors_origins.split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_NAME: Final[str] = "gemini-2.5-flash"
EMBEDDING_MODEL: Final[str] = "models/embedding-001"
EMBEDDING_MODEL_FALLBACK: Final[str] = "models/text-embedding-004"
MAX_CONTEXT_CHARS: Final[int] = 50_000

CHROMA_DIR: Final[Path] = Path(tempfile.gettempdir()) / "documind_chroma_db"
COLLECTION_NAME: Final[str] = "documind_chunks"

CHUNK_SIZE: Final[int] = 1000
CHUNK_OVERLAP: Final[int] = 200
EMBEDDING_MODEL_CANDIDATES: Final[tuple[str, ...]] = (
    EMBEDDING_MODEL,
    EMBEDDING_MODEL_FALLBACK,
    "models/embedding-002",
    "embedding-001",
    "text-embedding-004",
)
CHAT_MODEL_CANDIDATES: Final[tuple[str, ...]] = (
    MODEL_NAME,
    "models/gemini-2.0-flash",
    "models/gemini-1.5-flash",
)
IMAGE_MIME_BY_EXTENSION: Final[dict[str, str]] = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
}
SUPPORTED_EXTENSIONS: Final[set[str]] = {
    ".pdf",
    ".docx",
    ".csv",
    ".txt",
    ".xlsx",
    ".pptx",
    *IMAGE_MIME_BY_EXTENSION.keys(),
}


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Question to ask about the uploaded PDF")


class ChatResponse(BaseModel):
    answer: str
    sources: list["SourceChunk"] = []


class UploadResponse(BaseModel):
    filename: str
    pages: int
    characters: int
    message: str


class SourceChunk(BaseModel):
    source: str
    chunk: str
    score: float | None = None


class ExtractedContent(BaseModel):
    text: str
    pages: int


def _require_gemini_api_key() -> str:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="Missing GEMINI_API_KEY. Add it to your environment or .env file.",
        )
    return api_key


def get_gemini_model() -> genai.GenerativeModel:
    api_key = _require_gemini_api_key()
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(MODEL_NAME)


def _discover_chat_model_names() -> list[str]:
    available_names: list[str] = []
    try:
        for model in genai.list_models():
            methods = getattr(model, "supported_generation_methods", []) or []
            if "generateContent" in methods:
                name = getattr(model, "name", "")
                if name:
                    available_names.append(name)
    except Exception as exc:
        logger.warning("Failed to list Gemini chat models: %s", exc)
        return [name if name.startswith("models/") else f"models/{name}" for name in CHAT_MODEL_CANDIDATES]

    available_set = set(available_names)
    selected: list[str] = []
    for candidate in CHAT_MODEL_CANDIDATES:
        prefixed = candidate if candidate.startswith("models/") else f"models/{candidate}"
        short = candidate.removeprefix("models/")
        if prefixed in available_set or short in available_set:
            selected.append(prefixed)

    if selected:
        return selected

    # If none of our preferred text-capable models are listed, keep a safe default.
    return [MODEL_NAME if MODEL_NAME.startswith("models/") else f"models/{MODEL_NAME}"]


def _discover_embedding_model_name() -> str:
    available_names: list[str] = []
    try:
        for model in genai.list_models():
            methods = getattr(model, "supported_generation_methods", []) or []
            if "embedContent" in methods:
                available_names.append(getattr(model, "name", ""))
    except Exception as exc:
        logger.warning("Failed to list Gemini models: %s", exc)

    available_set = {name for name in available_names if name}

    for candidate in EMBEDDING_MODEL_CANDIDATES:
        # Support both prefixed and non-prefixed names in candidate matching.
        prefixed = candidate if candidate.startswith("models/") else f"models/{candidate}"
        short = candidate.removeprefix("models/")
        if prefixed in available_set or short in available_set:
            return prefixed

    if available_names:
        return available_names[0]

    # Last-resort default if model discovery fails completely.
    return EMBEDDING_MODEL


def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    api_key = _require_gemini_api_key()
    genai.configure(api_key=api_key)
    selected_model = _discover_embedding_model_name()
    embeddings = GoogleGenerativeAIEmbeddings(
        model=selected_model,
        google_api_key=api_key,
    )
    try:
        embeddings.embed_query("health-check")
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=(
                "No embedding model is available for this API key/project. "
                "Enable a Gemini embedding model that supports embedContent."
            ),
        ) from exc
    else:
        logger.info("Using embedding model: %s", selected_model)
        return embeddings


def _extract_text_from_pdf(file_bytes: bytes) -> ExtractedContent:
    try:
        document = fitz.open(stream=file_bytes, filetype="pdf")
    except Exception as exc:
        logger.exception("Failed to open PDF")
        raise HTTPException(status_code=400, detail="Invalid or corrupted PDF file.") from exc

    chunks: list[str] = []
    page_count = len(document)
    for page in document:
        page_text = page.get_text("text")
        if page_text:
            chunks.append(page_text.strip())
    document.close()

    text = "\n\n".join(chunk for chunk in chunks if chunk).strip()
    return ExtractedContent(text=text, pages=page_count)


def _extract_text_from_docx(file_bytes: bytes) -> ExtractedContent:
    try:
        with zipfile.ZipFile(io.BytesIO(file_bytes)) as archive:
            xml_bytes = archive.read("word/document.xml")
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid or unsupported DOCX file.") from exc

    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError as exc:
        raise HTTPException(status_code=400, detail="Failed to parse DOCX content.") from exc

    namespace = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    pieces = [node.text.strip() for node in root.findall(".//w:t", namespace) if node.text and node.text.strip()]
    return ExtractedContent(text="\n".join(pieces).strip(), pages=1)


def _extract_text_from_xlsx(file_bytes: bytes) -> ExtractedContent:
    try:
        with zipfile.ZipFile(io.BytesIO(file_bytes)) as archive:
            shared_strings: list[str] = []
            if "xl/sharedStrings.xml" in archive.namelist():
                shared_xml = ET.fromstring(archive.read("xl/sharedStrings.xml"))
                ns = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
                for node in shared_xml.findall(".//s:si", ns):
                    text_parts = [t.text or "" for t in node.findall(".//s:t", ns)]
                    value = "".join(text_parts).strip()
                    if value:
                        shared_strings.append(value)

            sheet_names = sorted(
                name for name in archive.namelist() if name.startswith("xl/worksheets/sheet") and name.endswith(".xml")
            )
            lines: list[str] = []
            ns = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
            for sheet_name in sheet_names:
                sheet_xml = ET.fromstring(archive.read(sheet_name))
                for row in sheet_xml.findall(".//s:row", ns):
                    row_values: list[str] = []
                    for cell in row.findall("s:c", ns):
                        cell_type = cell.attrib.get("t")
                        value_node = cell.find("s:v", ns)
                        if value_node is None or value_node.text is None:
                            continue
                        raw_value = value_node.text.strip()
                        if not raw_value:
                            continue
                        if cell_type == "s":
                            try:
                                row_values.append(shared_strings[int(raw_value)])
                            except (ValueError, IndexError):
                                row_values.append(raw_value)
                        else:
                            row_values.append(raw_value)
                    if row_values:
                        lines.append(" | ".join(row_values))
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid or unsupported XLSX file.") from exc

    return ExtractedContent(text="\n".join(lines).strip(), pages=1)


def _extract_text_from_pptx(file_bytes: bytes) -> ExtractedContent:
    try:
        with zipfile.ZipFile(io.BytesIO(file_bytes)) as archive:
            slide_names = sorted(
                name for name in archive.namelist() if name.startswith("ppt/slides/slide") and name.endswith(".xml")
            )
            ns = {"a": "http://schemas.openxmlformats.org/drawingml/2006/main"}
            slide_texts: list[str] = []
            for slide_name in slide_names:
                slide_xml = ET.fromstring(archive.read(slide_name))
                texts = [node.text.strip() for node in slide_xml.findall(".//a:t", ns) if node.text and node.text.strip()]
                if texts:
                    slide_texts.append("\n".join(texts))
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid or unsupported PPTX file.") from exc

    text = "\n\n".join(slide_texts).strip()
    return ExtractedContent(text=text, pages=max(len(slide_texts), 1))


def _extract_text_from_csv(file_bytes: bytes) -> ExtractedContent:
    decoded = file_bytes.decode("utf-8", errors="ignore")
    reader = csv.reader(io.StringIO(decoded))
    rows = [", ".join(cell.strip() for cell in row if cell and cell.strip()) for row in reader]
    text = "\n".join(row for row in rows if row).strip()
    return ExtractedContent(text=text, pages=1)


def _extract_text_from_plaintext(file_bytes: bytes) -> ExtractedContent:
    text = file_bytes.decode("utf-8", errors="ignore").strip()
    return ExtractedContent(text=text, pages=1)


def _extract_text_from_image(file_bytes: bytes, mime_type: str) -> ExtractedContent:
    api_key = _require_gemini_api_key()
    genai.configure(api_key=api_key)
    chat_models = _discover_chat_model_names()
    prompt = (
        "Extract all readable text from this image. "
        "Return only the extracted text without extra commentary."
    )

    for chat_model in chat_models:
        try:
            response = genai.GenerativeModel(chat_model).generate_content(
                [
                    prompt,
                    {"mime_type": mime_type, "data": file_bytes},
                ]
            )
            text = (response.text or "").strip()
            if text:
                return ExtractedContent(text=text, pages=1)
        except Exception:
            logger.warning("Image text extraction failed on model %s", chat_model)
            continue

    raise HTTPException(
        status_code=500,
        detail="Failed to extract text from image with available Gemini models.",
    )


def extract_text_from_file(file: UploadFile, file_bytes: bytes) -> ExtractedContent:
    filename = file.filename or ""
    extension = Path(filename).suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        allowed = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{extension or 'unknown'}'. Allowed types: {allowed}",
        )

    if extension == ".pdf":
        return _extract_text_from_pdf(file_bytes)
    if extension == ".docx":
        return _extract_text_from_docx(file_bytes)
    if extension == ".xlsx":
        return _extract_text_from_xlsx(file_bytes)
    if extension == ".pptx":
        return _extract_text_from_pptx(file_bytes)
    if extension == ".csv":
        return _extract_text_from_csv(file_bytes)
    if extension == ".txt":
        return _extract_text_from_plaintext(file_bytes)

    image_mime = IMAGE_MIME_BY_EXTENSION.get(extension)
    if image_mime:
        return _extract_text_from_image(file_bytes, image_mime)

    raise HTTPException(status_code=400, detail="Unsupported file type.")


def get_vectorstore() -> Chroma:
    return Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=str(CHROMA_DIR),
        embedding_function=get_embeddings(),
    )


def ingest_text_into_chroma(extracted_text: str, source_name: str) -> None:
    if CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    splits = text_splitter.split_text(extracted_text)
    documents = [
        Document(
            page_content=chunk,
            metadata={
                "source": source_name,
                "chunk_index": idx,
            },
        )
        for idx, chunk in enumerate(splits)
    ]

    embeddings = get_embeddings()
    Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=str(CHROMA_DIR),
    )


@app.get("/")
async def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)) -> UploadResponse:
    if not file.filename:
        raise HTTPException(status_code=400, detail="A filename is required.")

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    extracted = extract_text_from_file(file, file_bytes)
    extracted_text = extracted.text.strip()
    if not extracted_text:
        raise HTTPException(status_code=400, detail="No extractable text found in this file.")

    try:
        ingest_text_into_chroma(extracted_text, file.filename)
    except Exception as exc:
        logger.exception("Failed to build vector store")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to index document for search: {exc}",
        ) from exc

    return UploadResponse(
        filename=file.filename,
        pages=extracted.pages,
        characters=len(extracted_text),
        message="File processed and indexed for RAG successfully.",
    )


@app.post("/chat", response_model=ChatResponse)
async def chat_with_pdf(payload: ChatRequest) -> ChatResponse:
    if not CHROMA_DIR.exists():
        raise HTTPException(
            status_code=400,
            detail="No indexed document available. Upload a PDF to /upload first.",
        )

    vectorstore = get_vectorstore()
    try:
        scored_docs = vectorstore.similarity_search_with_relevance_scores(payload.question, k=5)
        diverse_docs = vectorstore.max_marginal_relevance_search(payload.question, k=3, fetch_k=10)
    except Exception as exc:
        logger.exception("Vector similarity search failed")
        raise HTTPException(
            status_code=502,
            detail=f"Document search failed: {exc}",
        ) from exc

    merged_docs: list[tuple[Document, float | None]] = []
    seen_keys: set[str] = set()
    for doc, score in scored_docs:
        key = f"{doc.metadata.get('source', '')}:{doc.metadata.get('chunk_index', '')}:{doc.page_content[:64]}"
        if key in seen_keys:
            continue
        seen_keys.add(key)
        merged_docs.append((doc, score))

    for doc in diverse_docs:
        key = f"{doc.metadata.get('source', '')}:{doc.metadata.get('chunk_index', '')}:{doc.page_content[:64]}"
        if key in seen_keys:
            continue
        seen_keys.add(key)
        merged_docs.append((doc, None))

    if not merged_docs:
        raise HTTPException(
            status_code=400,
            detail="No matching context found. Upload a file to /upload first.",
        )

    top_docs = merged_docs[:5]
    context = "\n\n".join(
        f"[Source: {doc.metadata.get('source', 'uploaded-file')}]\n{doc.page_content}"
        for doc, _ in top_docs
    )
    if len(context) > MAX_CONTEXT_CHARS:
        context = context[:MAX_CONTEXT_CHARS]

    prompt = (
        "You are a document QA assistant. Use only the provided context to answer. "
        "If the answer is not present in the context, explicitly say you could not find it.\n"
        "CRITICAL: Do NOT use any markdown formatting. Do not use asterisks (*), bold text, or markdown lists. "
        "Format your response strictly as clean, plain text paragraphs.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {payload.question}"
    )
    api_key = _require_gemini_api_key()
    genai.configure(api_key=api_key)
    chat_models = _discover_chat_model_names()

    response = None
    last_exc: Exception | None = None
    quota_hits = 0
    retry_after_candidates: list[float] = []

    for chat_model in chat_models:
        try:
            response = genai.GenerativeModel(chat_model).generate_content(prompt)
            logger.info("Chat answered using model: %s", chat_model)
            break
        except Exception as exc:
            last_exc = exc
            error_text = str(exc)
            is_quota = "429" in error_text or "quota" in error_text.lower()
            if is_quota:
                quota_hits += 1
                retry_match = re.search(r"retry in ([0-9]+(?:\.[0-9]+)?)s", error_text, flags=re.IGNORECASE)
                if retry_match:
                    retry_after_candidates.append(float(retry_match.group(1)))
                logger.warning("Quota hit on chat model %s, trying next model", chat_model)
                continue

            unsupported_modalities = (
                "response modalities" in error_text.lower()
                and "not supported" in error_text.lower()
            )
            if unsupported_modalities:
                logger.warning(
                    "Model %s does not support text responses, trying next model",
                    chat_model,
                )
                continue

            logger.exception("Gemini API request failed on model %s", chat_model)
            raise HTTPException(
                status_code=502,
                detail=f"Failed to get response from Gemini API: {exc}",
            ) from exc

    if response is None:
        if quota_hits:
            retry_after = max(retry_after_candidates) if retry_after_candidates else None
            detail = (
                f"Gemini quota exceeded across all configured chat models. Retry after about {retry_after:.0f} seconds."
                if retry_after is not None
                else "Gemini quota exceeded across all configured chat models. Please retry shortly."
            )
            raise HTTPException(status_code=429, detail=detail) from last_exc
        raise HTTPException(
            status_code=502,
            detail=f"Failed to get response from Gemini API: {last_exc}",
        ) from last_exc

    try:
        answer = (response.text or "").strip()
    except ValueError as exc:
        logger.exception("Gemini response had no text (blocked or invalid candidates)")
        raise HTTPException(
            status_code=502,
            detail="Gemini returned no text (request may have been blocked by safety filters).",
        ) from exc

    if not answer:
        raise HTTPException(status_code=502, detail="Gemini returned an empty response.")

    sources = [
        SourceChunk(
            source=str(doc.metadata.get("source", "uploaded-file")),
            chunk=doc.page_content[:500],
            score=score,
        )
        for doc, score in top_docs
    ]
    return ChatResponse(answer=answer, sources=sources)
