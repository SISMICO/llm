import os
import logging
from dotenv import load_dotenv
import numpy as np
import nltk

from src.faiss_utils import create_faiss_index, save_faiss_index, FAISS_INDEX_PATH, DOC_METADATA_PATH
from src.markdown_processor import find_markdown_files # Keep find_markdown_files

# Langchain components
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
# You might need to install:
# pip install langchain-community ollama "unstructured[md]"
# For UnstructuredMarkdownLoader, you might need: pip install "unstructured[md]"
from langchain_ollama import OllamaEmbeddings
# Previously: from langchain_community.embeddings import HuggingFaceEmbeddings

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
load_dotenv()

nltk.download('averaged_perceptron_tagger')

MARKDOWN_DIR = os.getenv("MARKDOWN_DIR", "./sample_docs")
OLLAMA_EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "llama3.2") # Or your preferred Ollama model
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434") # Default Ollama API URL
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 500))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 50))

def main():
    logging.info("Starting Markdown to FAISS loading process (using Langchain)...")

    try:
        # --- Clean up old index files if they exist ---
        if os.path.exists(FAISS_INDEX_PATH):
            os.remove(FAISS_INDEX_PATH)
            logging.info(f"Removed existing FAISS index file: {FAISS_INDEX_PATH}")
        if os.path.exists(DOC_METADATA_PATH):
            os.remove(DOC_METADATA_PATH)
            logging.info(f"Removed existing document metadata file: {DOC_METADATA_PATH}")


        # --- Initialize Langchain embedding model ---
        logging.info(f"Initializing Ollama embedding model: {OLLAMA_EMBEDDING_MODEL} from {OLLAMA_BASE_URL}...")
        try:
            # Initialize OllamaEmbeddings
            # Ensure your Ollama instance is running and the model (e.g., nomic-embed-text) is pulled.
            # You can pull models with `ollama pull nomic-embed-text`
            embeddings_model = OllamaEmbeddings(
                model=OLLAMA_EMBEDDING_MODEL,
                base_url=OLLAMA_BASE_URL
            )
            # Determine embedding dimension by embedding a test query
            test_embedding = embeddings_model.embed_query("test")
            vector_dimension = len(test_embedding)
            logging.info(f"Embedding model initialized. Dimension: {vector_dimension}")
        except Exception as e:
            logging.error(f"Failed to initialize Ollama embedding model '{OLLAMA_EMBEDDING_MODEL}' from '{OLLAMA_BASE_URL}': {e}. Ensure Ollama is running and the model is available. Exiting.")
            return

        if not vector_dimension:
             logging.error("Could not determine embedding dimension. Exiting.")
             return

        # --- Find Markdown Files ---
        markdown_files = find_markdown_files(MARKDOWN_DIR)
        if not markdown_files:
            logging.warning(f"No markdown files found in {MARKDOWN_DIR}. Exiting.")
            return

        # --- Process Each File and Collect Chunks/Embeddings ---
        all_chunks_content = []
        all_embeddings_list = []
        all_metadata_list = []
        total_chunks_processed = 0

        for filepath in markdown_files:
            logging.info(f"Processing file: {filepath}")
            source_name = os.path.relpath(filepath, MARKDOWN_DIR) # Get relative path for source ID

            # Load and extract text using Langchain's UnstructuredMarkdownLoader
            try:
                loader = UnstructuredMarkdownLoader(filepath)
                docs_from_file = loader.load() # Returns a list of Langchain Document objects

                if not docs_from_file:
                    logging.warning(f"No documents returned by UnstructuredMarkdownLoader for: {filepath}. Skipping file.")
                    continue

                # Concatenate page_content from all loaded documents for this file
                text = "\n\n".join([doc.page_content for doc in docs_from_file if doc.page_content])

                if not text.strip(): # Check if the concatenated text is empty or just whitespace
                    logging.warning(f"Skipping file due to empty content after processing with UnstructuredMarkdownLoader: {filepath}")
                    continue
            except Exception as e:
                logging.error(f"Error loading or processing markdown file {filepath} with UnstructuredMarkdownLoader: {e}")
                continue

            # Chunk text using Langchain's TextSplitter
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP
            )
            chunks_text_list = text_splitter.split_text(text) # This returns a list of strings

            if not chunks_text_list:
                logging.warning(f"No text chunks generated by RecursiveCharacterTextSplitter for: {filepath}")
                continue

            # Generate Embeddings for current file's chunks
            try:
                current_file_embeddings = embeddings_model.embed_documents(chunks_text_list)
            except Exception as e:
                logging.error(f"Embedding generation failed for {filepath} using Langchain model: {e}. Skipping file.")
                continue

            if current_file_embeddings is None or len(current_file_embeddings) != len(chunks_text_list):
                logging.error(
                    f"Embedding generation mismatch for {filepath}. "
                    f"Expected {len(chunks_text_list)} embeddings, "
                    f"got {len(current_file_embeddings) if current_file_embeddings is not None else 'None'}. Skipping file."
                )
                continue

            # Store chunks and their embeddings
            for i, (chunk_content, embedding_vector) in enumerate(zip(chunks_text_list, current_file_embeddings)):
                all_chunks_content.append(chunk_content) # Not strictly needed if metadata has content
                all_embeddings_list.append(embedding_vector)
                all_metadata_list.append({
                    "source": source_name,
                    "chunk_index": i,
                    "content": chunk_content
                })
            total_chunks_processed += len(chunks_text_list)
            logging.info(f"Processed {len(chunks_text_list)} chunks from {source_name}.")

        # --- Create and Save FAISS Index ---
        if all_embeddings_list:
            logging.info(f"Creating FAISS index for {len(all_embeddings_list)} total chunks...")
            faiss_index = create_faiss_index(all_embeddings_list, vector_dimension, normalize=True)
            if faiss_index:
                save_faiss_index(faiss_index, all_metadata_list)
                logging.info(f"Processing complete. FAISS index and metadata saved. Total chunks indexed: {total_chunks_processed}")
            else:
                logging.error("FAISS index creation failed. No index was saved.")
        else:
            logging.info("No embeddings generated. FAISS index not created.")

    except Exception as e:
        logging.error(f"An unexpected error occurred during the main process: {e}")

if __name__ == "__main__":
    main()
