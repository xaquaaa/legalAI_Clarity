from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from google import genai
from docx import Document as DocxDocument
from pypdf import PdfReader
import io
import os
import textwrap
from pathlib import Path # Used for reliable path resolution
import mimetypes # Used for robust MIME type detection

# --- CONFIGURATION & PATHS ---

# Resolve the absolute path to the base directory of the running script
BASE_DIR = Path(__file__).resolve().parent

# Define the directory where the built React frontend files are located
BUILD_DIR = BASE_DIR / "build"

# --- FastAPI App Initialization ---
app = FastAPI()

# --- CORS MIDDLEWARE (Allow all for hackathon compatibility) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- GOOGLE GENAI CLIENT ---

def get_gemini_client():
    """Initializes and returns the Gemini client."""
    try:
        return genai.Client() 
    except Exception:
        raise HTTPException(status_code=500, detail="Gemini API Key is missing or invalid. Check environment variables.")

# --- FILE EXTRACTION UTILITIES ---

def extract_text_from_pdf(file_content: bytes) -> str:
    """Extracts text from a PDF file using pypdf."""
    try:
        reader = PdfReader(io.BytesIO(file_content))
        text = "".join([page.extract_text() or "" for page in reader.pages[:10]])
        return text.strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF extraction error: {e}")

def extract_text_from_docx(file_content: bytes) -> str:
    """Extracts text from a DOCX file using python-docx."""
    try:
        document = DocxDocument(io.BytesIO(file_content))
        text = "\n".join([paragraph.text for paragraph in document.paragraphs if paragraph.text.strip()])
        return text.strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DOCX extraction error: {e}")


# --- PYDANTIC MODELS ---

class UploadResponse(BaseModel):
    filename: str
    extracted_text: str
    text_length: int

class ChatRequest(BaseModel):
    document_text: str 
    question: str

class ChatResponse(BaseModel):
    answer: str

class RewriteRequest(BaseModel):
    clause_text: str

class RiskSummaryRequest(BaseModel):
    document_text: str
    user_role: str


# --- CORE AI API ENDPOINTS ---

@app.post("/upload/", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """Handles document upload (PDF/DOCX) and returns the extracted text content."""
    
    file_content = await file.read()
    mime_type = file.content_type
    
    if mime_type == "application/pdf":
        extracted_text = extract_text_from_pdf(file_content)
    elif mime_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        extracted_text = extract_text_from_docx(file_content)
    else:
        raise HTTPException(status_code=400, detail="Invalid file type. Only PDF and DOCX are supported.")

    if not extracted_text:
        raise HTTPException(status_code=400, detail="Could not extract text from the document.")

    return {
        "filename": file.filename,
        "extracted_text": extracted_text,
        "text_length": len(extracted_text)
    }

@app.post("/chat_with_document/", response_model=ChatResponse)
async def chat_with_document(request: ChatRequest):
    """Uses Gemini to answer user questions based on the provided document text (RAG)."""
    
    client = get_gemini_client() 

    prompt = textwrap.dedent(f"""
    You are a professional Legal Assistant, named the 'Conversational Legal Twin'. 
    Your primary function is to simplify complex legal information and answer the user's question 
    based **STRICTLY AND ONLY** on the legal document provided below.
    
    If you cannot find a direct answer or relevant clause in the document, state clearly: 
    "The answer to that question could not be found in the provided document."

    --- LEGAL DOCUMENT TEXT ---
    {request.document_text}
    --- END OF DOCUMENT ---

    USER QUESTION: {request.question}
    """)

    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
        )
        
        return {"answer": response.text}

    except Exception as e:
        print(f"Gemini API Error: {e}")
        raise HTTPException(status_code=500, detail=f"AI processing failed: A communication error occurred with the Gemini API. Error details: {e}")


@app.post("/rewrite_clause/")
async def rewrite_clause(request: RewriteRequest):
    """Rewrites a specific complex clause into simple, plain English."""
    
    client = get_gemini_client()

    rewrite_prompt = textwrap.dedent(f"""
    You are an expert Plain Language Translator. Your task is to rewrite the
    following legal clause into simple, easy-to-understand English.
    The rewritten text must preserve the full original legal meaning and risk, but use
    no legal jargon.

    --- ORIGINAL CLAUSE ---
    {request.clause_text}
    """)

    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=rewrite_prompt,
        )
        return {"simplified_text": response.text}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI rewriting failed: {e}")


@app.post("/generate_risk_summary/")
async def generate_risk_summary(request: RiskSummaryRequest):
    """Generates a structured, exportable summary of only the risks for a role."""
    
    client = get_gemini_client()

    risk_prompt = textwrap.dedent(f"""
    You are a high-level Contract Risk Analyst. Your task is to generate a comprehensive, 
    structured risk report for the **{request.user_role}** based on the document below. 
    The output must be formatted with the main title and section headings in markdown.

    1. **Identify the Top 3 Financial Risks** to the {request.user_role}.
    2. **Identify the Top 3 Legal/Compliance Risks** (e.g., breach of contract, loss of rights).
    3. For each risk, cite the **relevant Section number** and provide a brief **mitigation suggestion** (e.g., "Always pay by the 1st").

    --- LEGAL DOCUMENT TEXT ---
    {request.document_text}
    """)

    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=risk_prompt,
        )
        return {"risk_report": response.text}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI risk report generation failed: {e}")


# --- REACT STATIC FILE SERVING ---

# 1. Mount the /static route to serve assets like JS/CSS/Maps
# This handles requests like /static/js/main.ed54bb15.js
app.mount("/static", StaticFiles(directory=BUILD_DIR / "static"), name="static")

# 2. Main catch-all for the Single-Page Application (SPA) - Must be last!
@app.get("/{file_path:path}")
async def serve_react_app(file_path: str):
    """
    Serves static files from the build directory, falling back to index.html 
    for SPA routing (Fixes the UnicodeDecodeError).
    """
    path_to_file = BUILD_DIR / file_path
    
    # Check if the requested file exists in the build directory
    if path_to_file.is_file():
        # Use mimetypes to automatically detect the file type for correct serving
        mime_type, _ = mimetypes.guess_type(path_to_file)
        
        # Use FileResponse for direct file serving (works for both binary and text)
        return FileResponse(path_to_file, media_type=mime_type)

    # If the file path is a directory, or the file doesn't exist, serve index.html (SPA fallback)
    return await serve_index_html()

# 3. Dedicated function to serve the main index.html file
async def serve_index_html():
    """Reads and serves the main index.html for the React application."""
    index_path = BUILD_DIR / "index.html"
    
    if not index_path.is_file():
        # This means the React build is missing or the path is wrong
        return HTMLResponse(
            status_code=404,
            content="<h1>404 Not Found</h1><p>Frontend 'index.html' not found. Did you run 'npm run build' and place the 'build' folder correctly?</p>"
        )
    
    # Read the file content and serve as HTML
    with open(index_path, 'r', encoding='utf-8') as f:
        html_content = f.read()
    
    return HTMLResponse(html_content, media_type="text/html")
