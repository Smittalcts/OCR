import os
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai.embeddings.azure import AzureOpenAIEmbeddings

# --- Environment Setup ---
# Load environment variables from .env file
load_dotenv()

# IMPORTANT: Set these environment variables in your .env file or system
# 1. AZURE_OPENAI_API_KEY: Your Azure OpenAI API key
# 2. AZURE_OPENAI_ENDPOINT: Your Azure OpenAI endpoint (e.g., "https://your-instance.openai.azure.com/")
# 3. OPENAI_API_VERSION: The API version (e.g., "2024-05-01-preview")
# 4. AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME: Your deployment name for "text-embeddings-001"

# --- Configuration ---
SOURCE_FILE_PATH = "induction_data.txt"
PERSIST_DIRECTORY = "./chroma_db"

def check_env_vars():
    """Check if all required Azure environment variables are set."""
    required_vars = [
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_ENDPOINT",
        "OPENAI_API_VERSION",
        "AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME"
    ]
    missing_vars = [var for var in required_vars if not os.environ.get(var)]
    if missing_vars:
        print(f"Error: Missing required environment variables: {', '.join(missing_vars)}")
        print("Please set them in your .env file or system environment.")
        return False
    return True

def create_dummy_file_if_not_exists():
    """Create a sample oncall_data.txt if it doesn't exist."""
    if not os.path.exists(SOURCE_FILE_PATH):
        print(f"'{SOURCE_FILE_PATH}' not found. Creating a dummy file.")
        dummy_data = """
Key metrics for on-call include 'latency' (p95, p99), 'error_rate' (5xx errors), 'throughput' (reqs/sec), and 'saturation' (CPU/memory usage). Alerts should be actionable and trigger on symptoms, not causes (e.g., 'high latency' vs 'high CPU').

The incident response lifecycle is: 1. Detect 2. Respond (triage, assemble team, open war room) 3. Mitigate (stop the bleeding) 4. Remediate (find root cause, deploy fix) 5. Postmortem (document, learn, prevent recurrence).

A 'Pod' is the smallest deployable unit in Kubernetes, holding one or more containers. A 'Service' provides a stable IP address for a set of Pods (e.g., via a 'Deployment'). A 'Readiness Probe' checks if a container is ready to accept traffic, while a 'Liveness Probe' checks if it needs to be restarted.

Triage involves assessing an alert's urgency and impact. An 'urgent' alert (e.g., P0, P1) requires immediate action, day or night. A 'low-priority' alert (e.g., P3) can wait until business hours. First steps are to check dashboards, a runbook, and communicate in the incident channel.
"""
        with open(SOURCE_FILE_PATH, "w", encoding="utf-8") as f:
            f.write(dummy_data)
        print(f"Dummy '{SOURCE_FILE_PATH}' created with sample data.")

def main():
    """Main ingestion pipeline."""
    if not check_env_vars():
        return

    # Create a dummy source file if one doesn't exist
    # create_dummy_file_if_not_exists()

    print(f"Loading documents from '{SOURCE_FILE_PATH}'...")
    loader = TextLoader(SOURCE_FILE_PATH, encoding="utf-8")
    documents = loader.load()

    print("Splitting documents...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, 
        chunk_overlap=200
    )
    docs = text_splitter.split_documents(documents)

    print("Initializing Azure OpenAI Embeddings...")
    try:
        embeddings_model = AzureOpenAIEmbeddings(
            openai_api_version=os.environ.get("OPENAI_API_VERSION"),
            azure_deployment=os.environ.get("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME"),
            # API key and endpoint are read from environment by default
        )
    except Exception as e:
        print(f"Error initializing AzureOpenAIEmbeddings: {e}")
        return

    print(f"Creating and persisting vector store at '{PERSIST_DIRECTORY}'...")
    if os.path.exists(PERSIST_DIRECTORY):
        print(f"Warning: Directory '{PERSIST_DIRECTORY}' already exists and will be overwritten.")
        import shutil
        shutil.rmtree(PERSIST_DIRECTORY)

    try:
        vector_store = Chroma.from_documents(
            documents=docs,
            embedding=embeddings_model,
            persist_directory=PERSIST_DIRECTORY
        )
        print("\n--- Success! ---")
        print(f"Vector store has been created and persisted at '{PERSIST_DIRECTORY}'.")
        print(f"Total documents ingested: {len(docs)}")
        print("You can now run 'agent_server.py'.")
    except Exception as e:
        print(f"Error creating Chroma vector store: {e}")
        print("Please check your Azure credentials and deployment names.")

if __name__ == "__main__":
    main()