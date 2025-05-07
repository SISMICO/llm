import os
import logging
import argparse
import numpy as np
import faiss # Required for faiss.normalize_L2
from dotenv import load_dotenv

from src.faiss_utils import load_faiss_index, FAISS_INDEX_PATH, DOC_METADATA_PATH
from langchain_ollama import OllamaEmbeddings, ChatOllama # Using Ollama for consistency
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain.chains.combine_documents import create_stuff_documents_chain

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
load_dotenv()

# Configuration for Ollama Embeddings (should match load_data.py)
OLLAMA_EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "llama3.2")
OLLAMA_LLM_MODEL = os.getenv("OLLAMA_LLM_MODEL", "llama3.2") # Model for generation
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

embeddings_model_instance = None
llm_instance = None # For ChatOllama
faiss_index_instance = None
doc_metadata_instance = None

def initialize_query_resources():
    """Initializes embedding model, FAISS index, and metadata."""
    global embeddings_model_instance, faiss_index_instance, doc_metadata_instance, llm_instance

    if embeddings_model_instance is None:
        try:
            logging.info(f"Initializing Ollama embedding model for querying: {OLLAMA_EMBEDDING_MODEL} from {OLLAMA_BASE_URL}...")
            embeddings_model_instance = OllamaEmbeddings(model=OLLAMA_EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)
            # Test query to ensure model is working and to get dimension if needed elsewhere
            embeddings_model_instance.embed_query("test")
            logging.info("Ollama embedding model for querying initialized.")
        except Exception as e:
            logging.error(f"Failed to initialize Ollama embedding model: {e}")
            raise

    if faiss_index_instance is None or doc_metadata_instance is None:
        logging.info("Loading FAISS index and document metadata...")
        faiss_index_instance, doc_metadata_instance = load_faiss_index(FAISS_INDEX_PATH, DOC_METADATA_PATH)
        if faiss_index_instance is None or not doc_metadata_instance:
            logging.warning("FAISS index or metadata not loaded. Similarity search will not work.")
    
    if llm_instance is None:
        try:
            logging.info(f"Initializing Ollama LLM for generation: {OLLAMA_LLM_MODEL} from {OLLAMA_BASE_URL}...")
            llm_instance = ChatOllama(model=OLLAMA_LLM_MODEL, base_url=OLLAMA_BASE_URL)
            # Test LLM to ensure it's working
            llm_instance.invoke("Respond with 'ok' if you are working.")
            logging.info("Ollama LLM for generation initialized.")
        except Exception as e:
            logging.error(f"Failed to initialize Ollama LLM: {e}")
            # LLM initialization failure is not fatal for document search, but generation will not work.

def find_similar_documents(query: str, top_n: int = 5) -> list:
    """Finds documents similar to the query using vector similarity search."""
    global embeddings_model_instance, faiss_index_instance, doc_metadata_instance

    if embeddings_model_instance is None or faiss_index_instance is None or not doc_metadata_instance:
        logging.error("Query resources not initialized. Cannot perform search.")
        return []

    query_embedding_list = embeddings_model_instance.embed_query(query)
    if query_embedding_list is None:
        logging.error("Failed to generate embedding for the query.")
        return []

    query_embedding_np = np.array([query_embedding_list], dtype='float32')
    faiss.normalize_L2(query_embedding_np) # Normalize query embedding as index embeddings were normalized

    try:
        distances, indices = faiss_index_instance.search(query_embedding_np, top_n)
        
        results = []
        if indices.size > 0:
            for i, idx in enumerate(indices[0]):
                if idx != -1: # FAISS can return -1 if fewer than top_n results
                    metadata = doc_metadata_instance[idx]
                    results.append({
                        "source": metadata["source"],
                        "chunk_index": metadata["chunk_index"],
                        "content": metadata["content"],
                        "distance": float(distances[0][i]) # L2 distance
                    })
        logging.info(f"Found {len(results)} similar documents.")
        return results

    except Exception as e:
        logging.error(f"Error during similarity search: {e}")
        return []

def generate_contextual_answer(query: str, documents: list, top_n_for_context: int = 3) -> str:
    """Generates a human-like answer based on the query and retrieved documents."""
    global llm_instance

    if llm_instance is None:
        logging.warning("LLM not initialized. Cannot generate contextual answer.")
        return "The language model is not available to generate an answer."

    if not documents:
        return "I couldn't find any relevant documents to provide a contextual answer."

    # Use a subset of documents for context to avoid overly long prompts
    context_docs_data = documents[:top_n_for_context]
    
    langchain_documents = [
        Document(page_content=doc["content"], metadata={"source": doc["source"], "chunk_index": doc["chunk_index"]})
        for doc in context_docs_data
    ]

    prompt_template_str = """You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question.
If you don't know the answer based on the context, state that you don't know.
Do not make up information outside of the provided context.
Keep the answer concise and directly related to the provided context.

Context:
{context}

Question: {input}

Answer:"""
    prompt = ChatPromptTemplate.from_template(prompt_template_str)
    
    question_answer_chain = create_stuff_documents_chain(llm_instance, prompt)
    
    try:
        response = question_answer_chain.invoke({"input": query, "context": langchain_documents})
        return response
    except Exception as e:
        logging.error(f"Error during answer generation: {e}")
        return "An error occurred while generating the answer."

def main():
    parser = argparse.ArgumentParser(description="Query similar documents from FAISS and generate a response.")
    parser.add_argument("query", type=str, help="The text query to search for.")
    parser.add_argument("-n", "--top_n", type=int, default=5, help="Number of similar documents to retrieve.")
    args = parser.parse_args()

    try:
        initialize_query_resources() # Load models and FAISS index
    except Exception as e:
        logging.error(f"Failed to initialize resources: {e}")
        print("Exiting due to initialization error. Check logs.")
        return

    logging.info(f"Searching for documents similar to: '{args.query}'")
    similar_docs = find_similar_documents(args.query, args.top_n)

    if similar_docs:
        print("\n--- Similar Documents Found ---")
        for i, doc in enumerate(similar_docs):
            print(f"\n[{i+1}] Source: {doc['source']} (Chunk {doc['chunk_index']})")
            print(f"    Distance (L2): {doc['distance']:.4f}") # Lower is more similar for L2
            print(f"    Content: {doc['content'][:200]}...") # Print snippet
        print("\n-----------------------------")

        print("\n--- Generating Human-like Answer ---")
        # Use top 3 retrieved documents for context by default, adjust as needed
        answer = generate_contextual_answer(args.query, similar_docs, top_n_for_context=3)
        print(f"\nAnswer: {answer}")
        print("\n------------------------------------")

    else:
        print("\nNo similar documents found.")
        # Optionally, you could still try to get a generic answer without context:
        # print("\n--- Generating Answer (No Context) ---")
        # answer = generate_contextual_answer(args.query, []) 
        # print(f"\nAnswer: {answer}")

if __name__ == "__main__":
    main()
