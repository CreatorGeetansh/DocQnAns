# Processing functions for the FastAPI application

# utils/processing.py (Relevant changes/additions)
import os
import tempfile
import logging
import time
import asyncio # For running sync code in async context
from typing import List, Optional, Tuple
from pathlib import Path

from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS # Using FAISS

import easyocr
from PIL import Image

from . import config
from .logging_config import get_logger

logger = get_logger(__name__)

# --- Initialize Models ---
try:
    embeddings = GoogleGenerativeAIEmbeddings(
        model=config.EMBEDDING_MODEL_NAME, google_api_key=config.GOOGLE_API_KEY
    )
    logger.info(f"Initialized Embeddings: {config.EMBEDDING_MODEL_NAME}")
except Exception as e:
    logger.exception("Failed to initialize Google Embeddings", exc_info=True)
    raise RuntimeError("Could not initialize embeddings model") from e

try:
    logger.info(f"Initializing EasyOCR (CPU) for languages: {config.OCR_LANGUAGES}...")
    # Explicitly use CPU
    ocr_reader = easyocr.Reader(config.OCR_LANGUAGES, gpu=False)
    logger.info("EasyOCR Reader initialized successfully (CPU).")
except Exception as e:
    logger.exception("Failed to initialize EasyOCR Reader", exc_info=True)
    raise RuntimeError("Could not initialize EasyOCR Reader") from e

# --- Helper Functions --- (load_document_text, split_documents remain similar)
# Make sure load_document_text calls the modified get_text_from_image_easyocr

def get_text_from_image_easyocr(file_path: str) -> Optional[str]:
    """Extracts text from an image file using EasyOCR on CPU."""
    try:
        logger.info(f"Performing OCR on image (CPU): {Path(file_path).name}")
        start_time = time.time()
        # EasyOCR readtext might be blocking, run in thread if needed
        results = ocr_reader.readtext(file_path)
        duration = time.time() - start_time
        logger.info(f"EasyOCR readtext took {duration:.2f} seconds (CPU)")
        extracted_text = " ".join([item[1] for item in results])
        if not extracted_text:
            logger.warning(f"OCR found no text in image: {Path(file_path).name}")
            return None
        logger.info(f"OCR successful for {Path(file_path).name}, extracted ~{len(extracted_text)} chars.")
        return extracted_text
    except Exception as e:
        logger.error(f"EasyOCR (CPU) failed for file {Path(file_path).name}", exc_info=True)
        return None

# --- Load Document Text (similar to previous, calls CPU OCR) ---
def load_document_text(file_path: str, file_type: str, file_name: str) -> List[Document]:
    """Loads text from various document types, including OCR for images."""
    docs = []
    try:
        logger.info(f"Loading document: {file_name} ({file_type})")
        if file_type == 'application/pdf':
            loader = PyPDFLoader(file_path)
            docs = loader.load() # PyPDFLoader usually returns docs split by page
        elif file_type == 'text/plain':
            loader = TextLoader(file_path, encoding='utf-8') # Specify encoding
            docs = loader.load()
        elif file_type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
            loader = Docx2txtLoader(file_path)
            docs = loader.load()
        elif file_type.startswith('image/'):
            # Run potentially blocking OCR in a separate thread
            # extracted_text = await asyncio.to_thread(get_text_from_image_easyocr, file_path)
            # For simplicity now, run sync, but be aware it can block
            extracted_text = get_text_from_image_easyocr(file_path)
            if extracted_text:
                docs = [Document(page_content=extracted_text, metadata={"source": file_name, "page": 0})]
            else:
                logger.warning(f"OCR returned no text for image: {file_name}")
                docs = [] # Return empty list if OCR fails or finds nothing
        else:
            logger.warning(f"Unsupported file type '{file_type}' for file: {file_name}")
            return [] # Return empty list for unsupported types

    except Exception as e:
        logger.error(f"Failed to load/process file {file_name} ({file_type})", exc_info=True)
        return [] # Return empty list on error

    logger.info(f"Successfully loaded {len(docs)} initial document sections from {file_name}.")
    # Standardize source metadata
    for doc in docs:
        doc.metadata["source"] = file_name # Use original filename
    return docs

# --- Split Documents (same as before) ---
def split_documents(docs: List[Document]) -> List[Document]:
    """Splits documents into smaller chunks."""
    if not docs:
        return []
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        add_start_index=True,
    )
    splits = text_splitter.split_documents(docs)
    logger.info(f"Split {len(docs)} documents into {len(splits)} chunks.")
    return splits

# --- FAISS Specific Functions ---
def create_and_save_faiss_index(splits: List[Document], document_id: str):
    """Creates FAISS index and saves it locally."""
    if not splits:
        logger.warning(f"No splits provided for {document_id}. Cannot create index.")
        raise ValueError("No document content available to index.")

    index_path = os.path.join(config.FAISS_INDEX_PATH, document_id)
    logger.info(f"Creating FAISS index for {document_id} from {len(splits)} chunks...")
    try:
        start_time = time.time()
        # FAISS creation might be CPU intensive
        vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
        duration = time.time() - start_time
        logger.info(f"FAISS index creation took {duration:.2f} seconds.")

        logger.info(f"Saving FAISS index to: {index_path}")
        vectorstore.save_local(index_path)
        logger.info(f"Successfully saved FAISS index for {document_id}")

    except Exception as e:
        logger.exception(f"Failed to create/save FAISS index for {document_id}", exc_info=True)
        # Clean up potentially partially created folder?
        raise RuntimeError("FAISS index creation/saving failed") from e

def load_faiss_index(document_id: str) -> Optional[FAISS]:
    """Loads a previously saved FAISS index."""
    index_path = os.path.join(config.FAISS_INDEX_PATH, document_id)
    if not os.path.exists(index_path):
        logger.error(f"FAISS index not found at path: {index_path}")
        return None
    try:
        logger.info(f"Loading FAISS index from: {index_path}")
        # Loading might also take some time
        vector_store = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True) # Be cautious with this flag
        logger.info(f"Successfully loaded FAISS index for {document_id}")
        return vector_store
    except Exception as e:
        logger.exception(f"Failed to load FAISS index for {document_id}", exc_info=True)
        return None

# --- Main Processing Pipeline Function ---
async def process_uploaded_file(file: 'UploadFile', document_id: str):
    """Full pipeline: Save temp, Load, Split, Embed, Store FAISS."""
    logger.info(f"Starting processing pipeline for document ID: {document_id}, file: {file.filename}")

    # 1. Save file temporarily (using async file ops)
    file_path = None
    try:
        suffix = Path(file.filename).suffix
        # Use a simple temp name structure for easy cleanup if needed
        temp_dir = tempfile.gettempdir()
        file_path = os.path.join(temp_dir, f"rag_{document_id}{suffix}")

        import aiofiles
        async with aiofiles.open(file_path, 'wb') as out_file:
            content = await file.read()  # Read file content async
            await out_file.write(content)
        logger.debug(f"File saved temporarily to: {file_path}")

        # 2. Load Text (can run sync function in thread)
        docs = await asyncio.to_thread(
            load_document_text, file_path, file.content_type, file.filename
        )
        if not docs:
            raise ValueError("Failed to extract text from document")

        # 3. Split Text (sync, usually fast)
        splits = split_documents(docs)
        if not splits:
            raise ValueError("Failed to split document into chunks")

        # 4. Create and Store Embeddings/FAISS Index (can run sync function in thread)
        await asyncio.to_thread(create_and_save_faiss_index, splits, document_id)

        logger.info(f"Successfully processed document ID: {document_id}")

    except Exception as e:
        logger.error(f"Error during file processing for {document_id}: {e}", exc_info=True)
        # Re-raise to be caught by the endpoint handler
        raise e
    finally:
        # Ensure temporary file is always cleaned up
        if file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
                logger.debug(f"Temporary file {file_path} removed.")
            except OSError as ose:
                logger.error(f"Error removing temporary file {file_path}: {ose}")