# utils/processing.py
import os
import tempfile
import logging
import time
import asyncio # For running sync code in async context
from typing import List, Optional, Tuple
from pathlib import Path

# Langchain imports
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS # Using FAISS

# Imports for OCR with Pytesseract
import pytesseract
from PIL import Image

# Local imports
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

# --- Initialize Tesseract Path (Optional but recommended) ---
# Check if a custom Tesseract path is provided in config.py
if hasattr(config, 'TESSERACT_CMD') and config.TESSERACT_CMD:
    # Check if the path exists, warn if not, but let pytesseract handle the final error
    if os.path.exists(config.TESSERACT_CMD):
        pytesseract.pytesseract.tesseract_cmd = config.TESSERACT_CMD
        logger.info(f"Set Tesseract command path to: {config.TESSERACT_CMD}")
    else:
        logger.warning(f"Tesseract command path specified but not found: {config.TESSERACT_CMD}. Pytesseract might fail.")
# Note: Pytesseract does not require explicit initialization like EasyOCR's Reader object.
# It relies on the tesseract executable being in the system PATH or specified via tesseract_cmd.

# --- Helper Functions ---

def get_text_from_image_pytesseract(file_path: str) -> Optional[str]:
    """Extracts text from an image file using Pytesseract."""
    try:
        logger.info(f"Performing OCR on image (Pytesseract): {Path(file_path).name}")
        start_time = time.time()

        # Use pytesseract to extract text
        # Pass the language codes from config (e.g., 'eng' or 'eng+fra')
        # Ensure TESSERACT_LANG is defined in config.py
        lang_codes = getattr(config, 'TESSERACT_LANG', 'eng') # Default to 'eng' if not set
        logger.info(f"Using Pytesseract languages: {lang_codes}")

        # Open image using PIL before passing to pytesseract
        img = Image.open(file_path)
        extracted_text = pytesseract.image_to_string(img, lang=lang_codes)

        duration = time.time() - start_time
        logger.info(f"Pytesseract OCR took {duration:.2f} seconds")

        # Check if any meaningful text was extracted
        if not extracted_text or extracted_text.isspace():
            logger.warning(f"Pytesseract found no text or only whitespace in image: {Path(file_path).name}")
            return None

        logger.info(f"Pytesseract OCR successful for {Path(file_path).name}, extracted ~{len(extracted_text)} chars.")
        # Optional: Simple cleanup of excessive newlines often produced by OCR
        cleaned_text = "\n".join([line for line in extracted_text.splitlines() if line.strip()])
        return cleaned_text

    except pytesseract.TesseractNotFoundError:
        logger.error(
            "Tesseract executable not found. Ensure Tesseract OCR is installed and "
            "either in the system PATH or the correct path is set in config.TESSERACT_CMD."
        )
        # Propagate the error or handle appropriately
        raise RuntimeError("Tesseract not found. OCR cannot proceed.") from None # Avoid chaining the original exception
    except Exception as e:
        logger.error(f"Pytesseract OCR failed for file {Path(file_path).name}", exc_info=True)
        return None # Return None on other OCR errors


# --- Load Document Text (Updated to call Pytesseract OCR) ---
def load_document_text(file_path: str, file_type: str, file_name: str) -> List[Document]:
    """Loads text from various document types, including OCR for images using Pytesseract."""
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
            extracted_text = get_text_from_image_pytesseract(file_path)

            if extracted_text:
                docs = [Document(page_content=extracted_text, metadata={"source": file_name, "page": 0})] # Assign page 0 for images
            else:
                logger.warning(f"Pytesseract OCR returned no text for image: {file_name}")
                docs = [] # Return empty list if OCR fails or finds nothing
        else:
            logger.warning(f"Unsupported file type '{file_type}' for file: {file_name}")
            return [] # Return empty list for unsupported types

    except RuntimeError as re: # Catch the specific TesseractNotFound error
         logger.error(f"Runtime error during loading/processing of {file_name}: {re}")
         raise # Re-raise to signal critical configuration issue
    except Exception as e:
        logger.error(f"Failed to load/process file {file_name} ({file_type})", exc_info=True)
        return [] # Return empty list on general errors

    logger.info(f"Successfully loaded {len(docs)} initial document sections from {file_name}.")
    # Standardize source metadata
    for doc in docs:
        if "source" not in doc.metadata:
             doc.metadata["source"] = file_name # Use original filename if loader didn't set it
        # Ensure filename consistency if loader used full path
        doc.metadata["source"] = file_name
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

# --- FAISS Specific Functions (same as before) ---
def create_and_save_faiss_index(splits: List[Document], document_id: str):
    """Creates FAISS index and saves it locally."""
    if not splits:
        logger.warning(f"No splits provided for {document_id}. Cannot create index.")
        raise ValueError("No document content available to index.")

    index_path = os.path.join(config.FAISS_INDEX_PATH, document_id)
    # Ensure the directory exists
    os.makedirs(config.FAISS_INDEX_PATH, exist_ok=True)

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
        # Clean up potentially partially created folder? Maybe not automatically.
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
        # Added check for allow_dangerous_deserialization based on your original code
        allow_deserialize = getattr(config, 'FAISS_ALLOW_DANGEROUS_DESERIALIZATION', True) # Default to True if not set
        if allow_deserialize:
             logger.warning("Loading FAISS index with allow_dangerous_deserialization=True. Ensure the index source is trusted.")

        vector_store = FAISS.load_local(
            index_path,
            embeddings,
            allow_dangerous_deserialization=allow_deserialize
        )
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
            # Read file content in chunks to handle large files
            while content := await file.read(1024 * 1024): # Read 1MB chunks
                 await out_file.write(content)
        logger.debug(f"File saved temporarily to: {file_path}")

        # 2. Load Text (Run blocking IO/CPU tasks in thread pool)
        # This includes file parsing (PyPDFLoader, Docx2txtLoader) and OCR (Pytesseract)
        docs = await asyncio.to_thread(
            load_document_text, file_path, file.content_type, file.filename
        )
        if not docs:
            # load_document_text logs warnings/errors, but we might want a clearer error here
            logger.warning(f"No documents extracted from file {file.filename} for ID {document_id}. Processing halted.")
            # Decide if this should be a hard error or allow empty index creation (likely an error)
            raise ValueError(f"Failed to extract text from document: {file.filename}")


        # 3. Split Text (sync, usually fast enough, but can be run in thread too if needed)
        # splits = split_documents(docs)
        # If splitting becomes a bottleneck for very large documents, use to_thread:
        splits = await asyncio.to_thread(split_documents, docs)

        if not splits:
            # This might happen if the document was loaded but contained no splittable text
            logger.warning(f"Document {file.filename} resulted in zero text chunks after splitting for ID {document_id}.")
            raise ValueError(f"Failed to split document into chunks: {file.filename}")

        # 4. Create and Store Embeddings/FAISS Index (Run blocking IO/CPU task in thread pool)
        await asyncio.to_thread(create_and_save_faiss_index, splits, document_id)

        logger.info(f"Successfully processed document ID: {document_id} for file: {file.filename}")

    except (ValueError, RuntimeError, pytesseract.TesseractNotFoundError) as e:
        # Catch specific known errors and potentially provide clearer messages
        logger.error(f"Processing failed for {document_id} ({file.filename}): {type(e).__name__} - {e}", exc_info=False) # Log concisely
        # Re-raise to be caught by the endpoint handler for HTTP response
        raise e
    except Exception as e:
        # Catch any unexpected errors
        logger.error(f"Unexpected error during file processing for {document_id} ({file.filename})", exc_info=True)
        # Re-raise to be caught by the endpoint handler
        raise RuntimeError(f"An unexpected error occurred during processing: {e}") from e
    finally:
        # Ensure temporary file is always cleaned up
        if file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
                logger.debug(f"Temporary file {file_path} removed.")
            except OSError as ose:
                logger.error(f"Error removing temporary file {file_path}: {ose}")