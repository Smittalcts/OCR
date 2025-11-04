import os
from dotenv import load_dotenv
from document_processor import DocumentProcessor
from vector_store import VectorStoreManager
import logging

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    """
    Main function to process documents and add them to the vector store.
    """
    load_dotenv()
    logging.info("Environment variables loaded.")

    # Configuration
    source_file_path = "data.txt"
    persist_directory = "./chroma_db_data"
    collection_name = "interview_data_lg" # Use a distinct name for the LangGraph version

    try:
        doc_processor = DocumentProcessor(chunk_size=500, chunk_overlap=100)
        vector_store_manager = VectorStoreManager(
            persist_directory=persist_directory,
            collection_name=collection_name
        )
        logging.info("DocumentProcessor and VectorStoreManager initialized.")
    except Exception as e:
        logging.error(f"Failed to initialize managers: {e}")
        return

    logging.info(f"Resetting ChromaDB collection: '{collection_name}'")
    vector_store_manager.reset_collection()

    if not os.path.exists(source_file_path):
        logging.error(f"Source document not found: {source_file_path}")
        return

    try:
        logging.info(f"Processing document: '{source_file_path}'")
        documents = doc_processor.process_document(source_file_path)
        
        logging.info(f"Adding {len(documents)} document chunks to the vector store...")
        success = vector_store_manager.add_documents(documents)
        
        if success:
            logging.info("Successfully added documents to ChromaDB.")
            collection_info = vector_store_manager.get_collection_info()
            logging.info(f"Collection Info: {collection_info}")
        else:
            logging.error("Failed to add documents to the vector store.")
            
    except Exception as e:
        logging.error(f"An error occurred during ingestion: {e}")

if __name__ == "__main__":
    main()
