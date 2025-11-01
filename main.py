from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from google import genai
from docx import Document as DocxDocument
from pypdf import PdfReader
import io
import os
import uvicorn
import textwrap # Utility for cleaning up text

from fastapi.middleware.cors import CORSMiddleware

# --- FastAPI App Initialization ---
app = FastAPI()

# --- ADD THIS BLOCK TO ENABLE CORS ---
origins = [
    "http://localhost",
    "http://localhost:3000", # The address where your React app is running
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Configuration and Utility Functions ---

def extract_text_from_pdf(file_content: bytes) -> str:
    """Extracts text from a PDF file using pypdf."""
    # Note: Only reads the first 10 pages to prevent hitting context limits in the MVP
    try:
        reader = PdfReader(io.BytesIO(file_content))
        text = ""
        for page in reader.pages[:10]: 
            text += page.extract_text() or ""
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

# --- Pydantic Models for Data Structure ---

class UploadResponse(BaseModel):
    filename: str
    extracted_text: str
    text_length: int

class ChatRequest(BaseModel):
    # The frontend will send the extracted text back with each chat message
    document_text: str 
    question: str

class ChatResponse(BaseModel):
    answer: str

# Add this model definition to the top of main.py, next to ChatRequest
class RewriteRequest(BaseModel):
    clause_text: str # This will be a single clause sent from the frontend

@app.post("/rewrite_clause/")
async def rewrite_clause(request: RewriteRequest):
    """
    Rewrites a specific complex clause into simple, plain English.
    """
    try:
        client = genai.Client() 
    except Exception:
        raise HTTPException(status_code=500, detail="Gemini API Key is missing or invalid.")

    rewrite_prompt = f"""
    You are an expert Plain Language Translator. Your task is to rewrite the
    following legal clause into simple, easy-to-understand English.
    The rewritten text must preserve the full original legal meaning and risk, but use
    no legal jargon.

    --- ORIGINAL CLAUSE ---
    {request.clause_text}
    """

    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=rewrite_prompt,
        )
        return {"simplified_text": response.text}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI rewriting failed: {e}")
    
# Add this model definition to the top of main.py
class PersonalizedSummaryRequest(BaseModel):
    document_text: str
    user_role: str # e.g., "Tenant" or "Landlord"

@app.post("/personalized_summary/")
async def personalized_summary(request: PersonalizedSummaryRequest):
    """
    Generates a summary of key obligations and risks relevant to a specific role.
    """
    try:
        client = genai.Client()
    except Exception:
        raise HTTPException(status_code=500, detail="Gemini API Key is missing or invalid.")

    summary_prompt = f"""
    You are a Legal Risk Analyst. Summarize the key rights, obligations, and potential
    risks in the legal document below, focusing **only** on the perspective of the **{request.user_role}**.
    Use clear bullet points and cite the section number for each item.

    --- LEGAL DOCUMENT TEXT ---
    {request.document_text}
    """

    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=summary_prompt,
        )
        return {"summary": response.text}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI summarization failed: {e}")
    
# Add this model definition to the top of main.py
class RiskSummaryRequest(BaseModel):
    document_text: str
    user_role: str

@app.post("/generate_risk_summary/")
async def generate_risk_summary(request: RiskSummaryRequest):
    """
    Generates a structured, exportable summary of only the risks for a role.
    """
    try:
        client = genai.Client()
    except Exception:
        raise HTTPException(status_code=500, detail="Gemini API Key is missing or invalid.")

    risk_prompt = f"""
    You are a high-level Contract Risk Analyst. Your task is to generate a comprehensive, 
    structured risk report for the **{request.user_role}** based on the document below. 
    The output must be formatted with the main title and section headings in markdown.

    1. **Identify the Top 3 Financial Risks** to the {request.user_role}.
    2. **Identify the Top 3 Legal/Compliance Risks** (e.g., breach of contract, loss of rights).
    3. For each risk, cite the **relevant Section number** and provide a brief **mitigation suggestion** (e.g., "Always pay by the 1st").

    --- LEGAL DOCUMENT TEXT ---
    {request.document_text}
    """

    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=risk_prompt,
        )
        return {"risk_report": response.text}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI risk report generation failed: {e}")

# --- API Endpoints ---

@app.post("/upload/", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """
    Handles document upload (PDF/DOCX) and returns the extracted text content.
    """
    file_content = await file.read()
    mime_type = file.content_type
    
    extracted_text = ""
    if mime_type == "application/pdf":
        extracted_text = extract_text_from_pdf(file_content)
    elif mime_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        extracted_text = extract_text_from_docx(file_content)
    else:
        raise HTTPException(status_code=400, detail="Invalid file type. Only PDF and DOCX are supported for this MVP.")

    if not extracted_text:
        raise HTTPException(status_code=400, detail="Could not extract text from the document.")

    return {
        "filename": file.filename,
        "extracted_text": extracted_text,
        "text_length": len(extracted_text)
    }

@app.post("/chat_with_document/", response_model=ChatResponse)
async def chat_with_document(request: ChatRequest):
    """
    Uses Gemini to answer user questions based on the provided document text (RAG).
    """
    # 1. Initialize the Gemini Client securely for this request
    try:
        # Client() automatically picks up the GEMINI_API_KEY environment variable
        client = genai.Client() 
    except Exception:
        raise HTTPException(status_code=500, detail="Gemini API Key is missing or invalid. Check environment variables.")

    # 2. Construct the RAG Prompt
    # We use textwrap.dedent and f-strings for a clean, multi-line prompt
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
        # 3. Call the Gemini API
        response = client.models.generate_content(
            model='gemini-2.5-flash', # Fast, highly capable, and cost-effective for RAG
            contents=prompt,
        )
        
        return {"answer": response.text}

    except Exception as e:
        print(f"Gemini API Error: {e}")
        # Return a user-friendly error if the AI call fails (e.g., context window exceeded, API key issue)
        raise HTTPException(status_code=500, detail=f"AI processing failed: A communication error occurred with the Gemini API. Error details: {e}")

# --- Main Entry Point for Local Testing ---
if __name__ == "__main__":
    # Note: Use uvicorn main:app without the if __name__ == "__main__": block
    # or run the script directly with this block. We use the command below 
    # for the easiest startup.
    print("To run the application, use the command: uvicorn main:app")