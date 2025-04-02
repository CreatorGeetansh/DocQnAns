# --- FastAPI WebApp for Local RAG with Google Gemini and FAISS ---
# This FastAPI application serves as a backend for a Retrieval-Augmented Generation (RAG) web application.  
import uuid
import asyncio
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, File, UploadFile, HTTPException, Body, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field
# Langchain specific imports
from langchain.chains import RetrievalQA
# from langchain.chains import ConversationalRetrievalChain # Optional for history
from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain.memory import ConversationBufferMemory # Optional for history

# Local imports
from utils.logging_config import setup_logging, get_logger
from utils import config, processing
import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
# --- Setup ---
setup_logging()
logger = get_logger(__name__)

# --- Initialize FastAPI App ---
app = FastAPI(
    title="Local RAG WebApp API",
    description="API for document upload and querying using Gemini and local FAISS.",
    version="1.0.0",
)

# --- CORS Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allow all for simplicity, restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- LLM Initialization ---
try:
    llm = ChatGoogleGenerativeAI(
        model=config.LLM_MODEL_NAME,
        google_api_key=config.GOOGLE_API_KEY,
        temperature=0.6,
        convert_system_message_to_human=True,
    )
    logger.info(f"Initialized LLM: {config.LLM_MODEL_NAME}")
except Exception as e:
    logger.exception("Failed to initialize LLM model", exc_info=True)
    raise RuntimeError("Could not initialize LLM") from e

# --- Pydantic Models ---
class UploadResponse(BaseModel):
    document_id: str
    filename: str
    message: str = "File processing started." # Indicate async nature

class QueryRequest(BaseModel):
    document_id: str
    question: str
    # chat_history: Optional[List[Dict[str, str]]] = None # If using conversational chain

class SourceDocument(BaseModel):
    content: str
    metadata: Dict[str, Any]

class QueryResponse(BaseModel):
    answer: str
    source_documents: Optional[List[SourceDocument]] = None
    # updated_chat_history: List[Dict[str, str]] # If using conversational chain

# --- API Endpoints ---

@app.post("/api/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """
    Uploads a document, processes it (async), saves FAISS index,
    and returns a document_id.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided.")

    document_id = str(uuid.uuid4())
    logger.info(f"Received upload request: {file.filename}, assigned ID: {document_id}")

    try:
        # Run the processing in the background (fire and forget for this simple example)
        # For production: Use background tasks (FastAPI's BackgroundTasks or Celery)
        # Here we run it directly but await it
        await processing.process_uploaded_file(file, document_id)
        message = f"File '{file.filename}' processed successfully."
        status_code = 200
        logger.info(f"Processing completed for {document_id}")

    except ValueError as ve: # Catch specific errors from processing
        logger.error(f"Processing error for {document_id}: {ve}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(ve))
    except RuntimeError as re: # Catch runtime errors (FAISS save etc.)
        logger.error(f"Processing runtime error for {document_id}: {re}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(re))
    except Exception as e:
        logger.exception(f"Unexpected error during processing trigger for {document_id}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to process the file.")

    return UploadResponse(document_id=document_id, filename=file.filename, message=message)


@app.post("/api/query", response_model=QueryResponse)
async def query_document(request: QueryRequest):
    """
    Queries the document associated with document_id using RetrievalQA.
    """
    document_id = request.document_id
    question = request.question
    logger.info(f"Received query for document_id: {document_id}")

    # Load FAISS index (run sync load in thread)
    vector_store = await asyncio.to_thread(processing.load_faiss_index, document_id)

    if vector_store is None:
        logger.error(f"Could not load FAISS index for document_id: {document_id}")
        raise HTTPException(status_code=404, detail="Document not found or index missing.")

    # Create RetrievalQA Chain
    try:
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff", # Simple chain type
            retriever=vector_store.as_retriever(search_kwargs={'k': 3}),
            return_source_documents=True,
        )
    except Exception as e:
        logger.exception("Failed to create QA chain", exc_info=True)
        raise HTTPException(status_code=500, detail="Could not initialize query engine.")

    # Execute Query (run sync chain in thread)
    try:
        logger.debug(f"Executing QA chain for document: {document_id}")
        result = await asyncio.to_thread(qa_chain, {"query": question})
        logger.debug(f"QA chain execution completed for document: {document_id}")

        answer = result.get('result', "Sorry, I could not generate an answer.")
        source_docs_output = []
        if result.get('source_documents'):
            source_docs_output = [
                SourceDocument(content=doc.page_content, metadata=doc.metadata)
                for doc in result['source_documents']
            ]

        return QueryResponse(answer=answer, source_documents=source_docs_output)

    except Exception as e:
        logger.exception(f"Error during query execution for {document_id}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Query failed: {e}")


@app.get("/api/health")
async def health_check():
    """Simple health check endpoint."""
    logger.debug("Health check endpoint called")
    return {"status": "ok", "message": "Backend is running"}

# --- Serve Frontend ---
# Mount the 'frontend' directory to serve static files
app.mount("/static", StaticFiles(directory="../frontend"), name="static")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Serves the main HTML page."""
    try:
        with open("../frontend/index.html", "r") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content)
    except FileNotFoundError:
         logger.error("Frontend index.html not found.")
         raise HTTPException(status_code=500, detail="Frontend not found.")

# --- Run the App (for local development) ---
# Use this block only if running `python main.py` directly
if __name__ == "__main__":
    import uvicorn
    logger.info("Starting FastAPI application with Uvicorn...")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, log_level="info")