# DocuMind

## About
**DocuMind** is a full-stack, Retrieval-Augmented Generation (RAG) application that allows you to seamlessly upload various document types (PDFs, Images, Spreadsheets, etc.) and chat with them using Google's powerful Gemini models. 

## Live Demo
🚀 **Try it live here:** [https://documind-frontend-t1fi.onrender.com](https://documind-frontend-t1fi.onrender.com)

## Features

- **Multi-Format Support**: Upload PDF, DOCX, XLSX, PPTX, CSV, TXT, and Images (PNG, JPG, WEBP).
- **Advanced Retrieval**: Uses a dual retrieval strategy combining similarity search and max marginal relevance for both precision and diversity.
- **Grounded Answers**: The AI explicitly answers only from the retrieved context and provides source citations (filename, excerpts, and similarity scores).
- **Metadata Tracking**: Chunk indexing ensures accurate source citation and deduplication during the RAG pipeline.

## Project Structure

This repository is split into two main parts:

### `backend/`
A Python FastAPI application responsible for:
- Text extraction from multiple file formats.
- Chunking and embedding generation using LangChain and Google Generative AI Embeddings.
- Local vector storage using ChromaDB.
- Handling the RAG chat pipeline and prompting Gemini.

### `frontend/`
A React application powered by Vite that provides:
- A clean UI for uploading and indexing documents.
- A chat interface to ask questions.
- A dedicated section rendering source citations and excerpts below each answer.

## Getting Started

Please see the respective directories (`backend/` and `frontend/`) for specific setup, installation, and run instructions.
