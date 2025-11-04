# services.py
import os
import logging
from langchain_openai import AzureChatOpenAI
from vector_store import VectorStoreManager

logger = logging.getLogger(__name__)

llm = None
vector_store_manager = None

try:
    llm = AzureChatOpenAI(
        azure_deployment=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"),
        api_version="2024-08-01-preview",
        temperature=0.7,
        max_tokens=1000,
    )
    vector_store_manager = VectorStoreManager(
        persist_directory="./chroma_db",
        collection_name="induction_docs"
    )
    logger.info("LLM and VectorStoreManager initialized successfully.")
except Exception as e:
    logger.error(f"FATAL: Could not initialize core components. Error: {e}", exc_info=True)
    # The app will fail to build the graph, which is appropriate
