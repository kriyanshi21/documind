from __future__ import annotations

import logging
import os
import re
import shutil
from pathlib import Path
from typing import Final

import fitz  # PyMuPDF
import google.generativeai as genai
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field

load_dotenv()

logger = logging.getLogger(__name__)

app = FastAPI(title="DocuMind API", version="1.0.0")

MODEL_NAME: Final[str] = "gemini-2.5-flash"
EMBEDDING_MODEL: Final[str] = "models/embedding-001"
EMBEDDING_MODEL_FALLBACK: Final[str] = "models/text-embedding-004"
MAX_CONTEXT_CHARS: Final[int] = 50_000

CHROMA_DIR: Final[Path] = Path(__file__).resolve().parent / "chroma_db"
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


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Question to ask about the uploaded PDF")


class ChatResponse(BaseModel):
    answer: str


class UploadResponse(BaseModel):
    filename: str
    pages: int
    characters: int
    message: str


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


def get_vectorstore() -> Chroma:
    return Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=str(CHROMA_DIR),
        embedding_function=get_embeddings(),
    )


def ingest_text_into_chroma(extracted_text: str) -> None:
    if CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    splits = text_splitter.split_text(extracted_text)
    documents = [Document(page_content=chunk) for chunk in splits]

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
async def upload_pdf(file: UploadFile = File(...)) -> UploadResponse:
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    pdf_bytes = await file.read()
    if not pdf_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        document = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as exc:
        logger.exception("Failed to open PDF")
        raise HTTPException(status_code=400, detail="Invalid or corrupted PDF file.") from exc

    extracted_chunks: list[str] = []
    page_count = len(document)
    for page in document:
        page_text = page.get_text("text")
        if page_text:
            extracted_chunks.append(page_text.strip())

    document.close()

    extracted_text = "\n\n".join(chunk for chunk in extracted_chunks if chunk).strip()
    if not extracted_text:
        raise HTTPException(status_code=400, detail="No extractable text found in this PDF.")

    try:
        ingest_text_into_chroma(extracted_text)
    except Exception as exc:
        logger.exception("Failed to build vector store")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to index document for search: {exc}",
        ) from exc

    return UploadResponse(
        filename=file.filename,
        pages=page_count,
        characters=len(extracted_text),
        message="PDF processed and indexed for RAG successfully.",
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
        relevant_docs = vectorstore.similarity_search(payload.question, k=3)
    except Exception as exc:
        logger.exception("Vector similarity search failed")
        raise HTTPException(
            status_code=502,
            detail=f"Document search failed: {exc}",
        ) from exc

    if not relevant_docs:
        raise HTTPException(
            status_code=400,
            detail="No matching context found. Upload a PDF to /upload first.",
        )

    context = "\n\n".join(doc.page_content for doc in relevant_docs)
    if len(context) > MAX_CONTEXT_CHARS:
        context = context[:MAX_CONTEXT_CHARS]

    prompt = f"Use the following excerpts from the document as context. Answer the question based only on this context when possible.\n\nContext:\n{context}\n\nQuestion: {payload.question}"
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

    return ChatResponse(answer=answer)
