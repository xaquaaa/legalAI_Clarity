import requests
import json
import os

# --- Configuration ---
API_URL = "http://127.0.0.1:8000" # Default FastAPI address
DOCUMENT_PATH = "sample_lease.pdf" # Make sure this matches your file name
TEST_QUESTION = "What is the financial penalty if the Tenant moves out in three months?"

def upload_and_extract(file_path: str):
    """
    Tests the /upload/ endpoint.
    """
    print("--- 1. Testing /upload/ endpoint (Document Extraction) ---")
    
    # 1. Prepare the file data for the multipart form request
    try:
        files = {'file': (os.path.basename(file_path), open(file_path, 'rb'), 'application/pdf')}
    except FileNotFoundError:
        print(f"\nERROR: File not found at path: {file_path}")
        print("Please ensure your sample document is named exactly 'sample_lease.pdf' or update the DOCUMENT_PATH variable.")
        return None

    # 2. Send the POST request to FastAPI
    try:
        response = requests.post(f"{API_URL}/upload/", files=files)
        response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)
        
        # 3. Parse the JSON response
        data = response.json()
        print(f"Status: {response.status_code}")
        print(f"Extracted Text Length: {len(data['extracted_text'])} characters.")
        
        # We need the extracted text for the next step
        return data['extracted_text']

    except requests.exceptions.HTTPError as errh:
        print(f"HTTP Error: {errh}")
        print(f"Server Response Detail: {response.json().get('detail', 'N/A')}")
    except requests.exceptions.RequestException as e:
        print(f"Connection Error: Could not connect to FastAPI server at {API_URL}. Is 'uvicorn main:app' running?")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    
    return None


def chat_with_document(document_text: str, question: str):
    """
    Tests the /chat_with_document/ endpoint with the extracted text.
    """
    print("\n--- 2. Testing /chat_with_document/ endpoint (Gemini RAG) ---")

    # 1. Prepare the JSON payload
    payload = {
        "document_text": document_text,
        "question": question
    }

    # 2. Send the POST request to FastAPI
    try:
        response = requests.post(
            f"{API_URL}/chat_with_document/", 
            json=payload,
            timeout=30 # Add a timeout as Gemini can take a few seconds
        )
        response.raise_for_status()
        
        # 3. Parse the JSON response
        data = response.json()
        
        print(f"Question: {question}")
        print("\n--- AI Response ---")
        print(data['answer'])
        print("-------------------\n")

    except requests.exceptions.HTTPError as errh:
        print(f"HTTP Error: {errh}")
        print(f"Server Response Detail: {response.json().get('detail', 'N/A')}")
    except requests.exceptions.Timeout:
        print("Request Timed Out. Gemini took too long to respond.")
    except requests.exceptions.RequestException as e:
        print(f"Connection Error: {e}")

COMPLEX_CLAUSE = "The Tenant MAY NOT terminate this Agreement early without the Landlord's written consent. Should the Tenant breach this clause and vacate the Premises early, the Tenant shall be liable for three (3) months' rent as an early termination fee, in addition to any unpaid rent accrued up to the date of vacation."

def test_rewrite_clause():
    """
    Tests the /rewrite_clause/ endpoint.
    """
    print("\n--- 3. Testing /rewrite_clause/ endpoint ---")
    payload = {"clause_text": COMPLEX_CLAUSE}
    
    try:
        response = requests.post(f"{API_URL}/rewrite_clause/", json=payload)
        response.raise_for_status()
        
        data = response.json()
        print(f"Original Clause: {COMPLEX_CLAUSE[:50]}...")
        print("\n--- Simplified Rewrite ---")
        print(data['simplified_text'])
        print("--------------------------\n")

    except requests.exceptions.RequestException as e:
        print(f"Error testing rewrite endpoint: {e}")

def test_personalized_summary(document_text: str):
    """
    Tests the /personalized_summary/ endpoint for a specific role.
    """
    ROLE = "Tenant"
    print(f"\n--- 4. Testing /personalized_summary/ for role: {ROLE} ---")

    payload = {
        "document_text": document_text,
        "user_role": ROLE
    }
    
    try:
        response = requests.post(f"{API_URL}/personalized_summary/", json=payload, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        print("\n--- Tenant-Specific Summary ---")
        print(data['summary'])
        print("-----------------------------\n")

    except requests.exceptions.RequestException as e:
        print(f"Error testing personalized summary endpoint: {e}")

def test_risk_summary(document_text: str):
    """
    Tests the /generate_risk_summary/ endpoint.
    """
    ROLE = "Tenant"
    print(f"\n--- 5. Testing /generate_risk_summary/ for role: {ROLE} ---")

    payload = {
        "document_text": document_text,
        "user_role": ROLE
    }
    
    try:
        response = requests.post(f"{API_URL}/generate_risk_summary/", json=payload, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        print("\n--- Exportable Risk Report ---")
        print(data['risk_report'])
        print("------------------------------\n")

    except requests.exceptions.RequestException as e:
        print(f"Error testing risk summary endpoint: {e}")

# --- Main Execution ---
if __name__ == "__main__":
    
    # --- Check for API Key ---
    if not os.environ.get("GEMINI_API_KEY"):
        print("!!! WARNING: GEMINI_API_KEY environment variable is NOT set. The chat endpoint will likely fail. !!!")
        print("Please run: export GEMINI_API_KEY=\"YOUR_KEY_HERE\"")
    
    # 1. Run the upload and get the extracted text
    extracted_text = upload_and_extract(DOCUMENT_PATH)
    
    # 2. If extraction was successful, proceed to chat
    if extracted_text and len(extracted_text) > 50: # Check for minimum length
        # Test 2: Conversational Twin
        chat_with_document(extracted_text, TEST_QUESTION)
        
        # Test 3: Clause Rewriting
        test_rewrite_clause()
        
        # Test 4: Personalized Summary
        test_personalized_summary(extracted_text)

        # New Test 5: Risk Summary
        test_risk_summary(extracted_text)
    else:
        print("\nSkipping chat test due to failed or empty text extraction.")