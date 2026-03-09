import faiss
import numpy as np
import os
import pickle
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define default paths for FAISS index and metadata
# These can be overridden by environment variables
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "vector_store/index.faiss")
DOC_METADATA_PATH = os.getenv("DOC_METADATA_PATH", "vector_store/doc_metadata.pkl")

def save_faiss_index(index: faiss.Index, metadata_list: list, index_path: str = FAISS_INDEX_PATH, metadata_path: str = DOC_METADATA_PATH):
    """Saves the FAISS index and corresponding document metadata."""
    try:
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        os.makedirs(os.path.dirname(metadata_path), exist_ok=True)

        faiss.write_index(index, index_path)
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata_list, f)
        logging.info(f"FAISS index saved to {index_path}")
        logging.info(f"Document metadata saved to {metadata_path}")
    except Exception as e:
        logging.error(f"Error saving FAISS index or metadata: {e}")
        raise

def load_faiss_index(index_path: str = FAISS_INDEX_PATH, metadata_path: str = DOC_METADATA_PATH):
    """Loads the FAISS index and document metadata."""
    index = None
    metadata_list = []
    if os.path.exists(index_path) and os.path.exists(metadata_path):
        try:
            index = faiss.read_index(index_path)
            with open(metadata_path, 'rb') as f:
                metadata_list = pickle.load(f)
            logging.info(f"FAISS index loaded from {index_path} with {index.ntotal} vectors.")
            logging.info(f"Document metadata loaded from {metadata_path} for {len(metadata_list)} documents.")
        except Exception as e:
            logging.error(f"Error loading FAISS index or metadata: {e}")
            return None, []
    else:
        logging.warning(f"FAISS index ({index_path}) or metadata ({metadata_path}) not found. A new one will be created if loading data.")
    return index, metadata_list

def create_faiss_index(embeddings: list[list[float]], vector_dimension: int, normalize: bool = True):
    """Creates a FAISS index from a list of embedding vectors."""
    if not embeddings or not isinstance(embeddings, list) or not embeddings[0]:
        logging.error("No embeddings provided or embeddings format is incorrect.")
        return None
    
    embeddings_np = np.array(embeddings, dtype='float32')
    if embeddings_np.ndim == 1: # Should not happen if list of lists
        embeddings_np = np.expand_dims(embeddings_np, axis=0)

    if embeddings_np.shape[1] != vector_dimension:
        logging.error(f"Embedding dimension mismatch. Expected {vector_dimension}, got {embeddings_np.shape[1]}")
        return None

    if normalize:
        faiss.normalize_L2(embeddings_np) # Normalize for cosine similarity with L2 distance
    
    index = faiss.IndexFlatL2(vector_dimension)
    index.add(embeddings_np)
    logging.info(f"FAISS index created with {index.ntotal} vectors (dimension: {vector_dimension}).")
    return index
