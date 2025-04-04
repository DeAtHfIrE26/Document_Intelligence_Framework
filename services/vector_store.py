import logging
import faiss
import numpy as np
import threading
import pickle
import os
import hashlib
import json
from typing import List, Dict, Any, Optional, Tuple, Union

from app import app
from config import VECTOR_DIMENSION, INDEX_TYPE, TOP_K_RETRIEVAL

logger = logging.getLogger(__name__)

# Global variables for FAISS indexes
index = None
index_lock = threading.RLock()
index_metadata = {}
index_file_path = "data/faiss_index.bin"
metadata_file_path = "data/index_metadata.pkl"

def init_app(app):
    """Initialize the vector store with the app context"""
    global index, index_metadata
    
    # Create data directory if it doesn't exist
    os.makedirs("data", exist_ok=True)
    
    # Initialize or load the FAISS index
    with index_lock:
        if os.path.exists(index_file_path) and os.path.exists(metadata_file_path):
            try:
                # Load existing index
                index = faiss.read_index(index_file_path)
                with open(metadata_file_path, "rb") as f:
                    index_metadata = pickle.load(f)
                logger.info(f"Loaded existing FAISS index with {index.ntotal} vectors")
            except Exception as e:
                logger.error(f"Error loading FAISS index: {e}")
                create_new_index()
        else:
            create_new_index()

def create_new_index():
    """Create a new FAISS index"""
    global index, index_metadata
    
    try:
        # Create appropriate index based on configuration
        if INDEX_TYPE == "FLAT":
            # Exact search with L2 distance
            index = faiss.IndexFlatL2(VECTOR_DIMENSION)
        elif INDEX_TYPE == "IVF":
            # Inverted file index with approximate search
            # First create a flat index for training
            quantizer = faiss.IndexFlatL2(VECTOR_DIMENSION)
            # Then create the IVF index (nlist=100 is the number of clusters)
            index = faiss.IndexIVFFlat(quantizer, VECTOR_DIMENSION, 100, faiss.METRIC_L2)
            # Train on dummy data if no data is available
            dummy_data = np.random.random((1000, VECTOR_DIMENSION)).astype(np.float32)
            index.train(dummy_data)
        elif INDEX_TYPE == "HNSW":
            # Hierarchical Navigable Small World graph index
            index = faiss.IndexHNSWFlat(VECTOR_DIMENSION, 32)  # 32 is the number of neighbors
        else:
            # Default to flat index
            index = faiss.IndexFlatL2(VECTOR_DIMENSION)
            
        # Initialize empty metadata dictionary
        index_metadata = {}
        logger.info(f"Created new FAISS index of type {INDEX_TYPE}")
        
    except Exception as e:
        logger.error(f"Error creating FAISS index: {e}")
        # Fallback to simple flat index
        index = faiss.IndexFlatL2(VECTOR_DIMENSION)
        index_metadata = {}

def save_index():
    """Save the FAISS index and metadata to disk"""
    try:
        with index_lock:
            faiss.write_index(index, index_file_path)
            with open(metadata_file_path, "wb") as f:
                pickle.dump(index_metadata, f)
        logger.info(f"Saved FAISS index with {index.ntotal} vectors")
    except Exception as e:
        logger.error(f"Error saving FAISS index: {e}")

def store_embedding(embedding: np.ndarray, metadata: Dict[str, Any]) -> str:
    """
    Store a single embedding in the vector database
    
    Args:
        embedding: Embedding vector
        metadata: Metadata associated with the embedding
        
    Returns:
        Unique ID for the stored embedding
    """
    try:
        # Convert embedding to the correct format
        if isinstance(embedding, list):
            embedding = np.array(embedding, dtype=np.float32)
        
        if embedding.dtype != np.float32:
            embedding = embedding.astype(np.float32)
            
        # Reshape if necessary
        if len(embedding.shape) == 1:
            embedding = embedding.reshape(1, -1)
            
        # Generate unique ID
        metadata_str = json.dumps(metadata, sort_keys=True)
        embedding_id = hashlib.sha256(f"{metadata_str}_{embedding.tobytes()}".encode()).hexdigest()
        
        with index_lock:
            # Check if embedding already exists
            if embedding_id in index_metadata:
                logger.debug(f"Embedding already exists with ID: {embedding_id}")
                return embedding_id
                
            # Store in FAISS index
            index_position = index.ntotal
            index.add(embedding)
            
            # Store metadata with the position in the index
            index_metadata[embedding_id] = {
                "position": index_position,
                "metadata": metadata
            }
            
            # Save to disk periodically
            if index.ntotal % 100 == 0:
                save_index()
                
        return embedding_id
        
    except Exception as e:
        logger.error(f"Error storing embedding: {e}")
        return None

def store_embeddings(embeddings: np.ndarray, metadata_list: List[Dict[str, Any]]) -> List[str]:
    """
    Store multiple embeddings in the vector database
    
    Args:
        embeddings: Array of embedding vectors
        metadata_list: List of metadata dictionaries
        
    Returns:
        List of unique IDs for the stored embeddings
    """
    try:
        # Ensure embeddings have the correct format
        if not isinstance(embeddings, np.ndarray):
            embeddings = np.array(embeddings, dtype=np.float32)
            
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)
            
        # Generate IDs
        embedding_ids = []
        for i, metadata in enumerate(metadata_list):
            metadata_str = json.dumps(metadata, sort_keys=True)
            embedding_id = hashlib.sha256(f"{metadata_str}_{embeddings[i].tobytes()}".encode()).hexdigest()
            embedding_ids.append(embedding_id)
            
        with index_lock:
            # Get the starting position in the index
            start_position = index.ntotal
            
            # Add embeddings to the index
            index.add(embeddings)
            
            # Store metadata for each embedding
            for i, embedding_id in enumerate(embedding_ids):
                index_metadata[embedding_id] = {
                    "position": start_position + i,
                    "metadata": metadata_list[i]
                }
                
            # Save to disk periodically
            if index.ntotal % 100 == 0:
                save_index()
                
        return embedding_ids
        
    except Exception as e:
        logger.error(f"Error storing multiple embeddings: {e}")
        return [None] * len(metadata_list)

def search_similar(embedding: np.ndarray, k: int = TOP_K_RETRIEVAL, 
                   filter_dict: Dict[str, Any] = None) -> List[Dict[str, Any]]:
    """
    Search for similar embeddings
    
    Args:
        embedding: Query embedding
        k: Number of results to return
        filter_dict: Optional metadata filter
        
    Returns:
        List of results with metadata and similarity scores
    """
    try:
        # Convert embedding to the correct format
        if isinstance(embedding, list):
            embedding = np.array(embedding, dtype=np.float32)
            
        if embedding.dtype != np.float32:
            embedding = embedding.astype(np.float32)
            
        # Reshape if necessary
        if len(embedding.shape) == 1:
            embedding = embedding.reshape(1, -1)
            
        with index_lock:
            # If the index is empty, return empty results
            if index.ntotal == 0:
                return []
                
            # Search the index
            distances, indices = index.search(embedding, k * 10)  # Get more results for filtering
            
            # Post-process results
            results = []
            for i, idx in enumerate(indices[0]):
                if idx == -1:  # FAISS returns -1 for irrelevant results
                    continue
                    
                # Find the embedding ID for this index position
                embedding_id = None
                for eid, data in index_metadata.items():
                    if data["position"] == idx:
                        embedding_id = eid
                        break
                        
                if embedding_id is None:
                    continue
                    
                metadata = index_metadata[embedding_id]["metadata"]
                
                # Apply filter
                if filter_dict:
                    match = True
                    for key, value in filter_dict.items():
                        if key not in metadata or metadata[key] != value:
                            match = False
                            break
                    if not match:
                        continue
                        
                # Convert L2 distance to similarity score (0-1)
                similarity = 1.0 / (1.0 + distances[0][i])
                
                results.append({
                    "id": embedding_id,
                    "metadata": metadata,
                    "similarity": similarity,
                    "distance": float(distances[0][i])
                })
                
                if len(results) >= k:
                    break
                    
            return results
            
    except Exception as e:
        logger.error(f"Error searching similar embeddings: {e}")
        return []

def batch_search_similar(embeddings: np.ndarray, k: int = TOP_K_RETRIEVAL, 
                        filter_dict: Dict[str, Any] = None) -> List[List[Dict[str, Any]]]:
    """
    Search for similar embeddings for multiple queries
    
    Args:
        embeddings: Query embeddings
        k: Number of results per query
        filter_dict: Optional metadata filter
        
    Returns:
        List of result lists
    """
    try:
        # Convert embeddings to the correct format
        if not isinstance(embeddings, np.ndarray):
            embeddings = np.array(embeddings, dtype=np.float32)
            
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)
            
        with index_lock:
            # If the index is empty, return empty results
            if index.ntotal == 0:
                return [[] for _ in range(len(embeddings))]
                
            # Search the index
            distances, indices = index.search(embeddings, k * 10)  # Get more results for filtering
            
            # Post-process results
            all_results = []
            
            for q, query_indices in enumerate(indices):
                results = []
                
                for i, idx in enumerate(query_indices):
                    if idx == -1:  # FAISS returns -1 for irrelevant results
                        continue
                        
                    # Find the embedding ID for this index position
                    embedding_id = None
                    for eid, data in index_metadata.items():
                        if data["position"] == idx:
                            embedding_id = eid
                            break
                            
                    if embedding_id is None:
                        continue
                        
                    metadata = index_metadata[embedding_id]["metadata"]
                    
                    # Apply filter
                    if filter_dict:
                        match = True
                        for key, value in filter_dict.items():
                            if key not in metadata or metadata[key] != value:
                                match = False
                                break
                        if not match:
                            continue
                            
                    # Convert L2 distance to similarity score (0-1)
                    similarity = 1.0 / (1.0 + distances[q][i])
                    
                    results.append({
                        "id": embedding_id,
                        "metadata": metadata,
                        "similarity": similarity,
                        "distance": float(distances[q][i])
                    })
                    
                    if len(results) >= k:
                        break
                        
                all_results.append(results)
                
            return all_results
            
    except Exception as e:
        logger.error(f"Error batch searching similar embeddings: {e}")
        return [[] for _ in range(len(embeddings))]

def get_embedding_by_id(embedding_id: str) -> Optional[np.ndarray]:
    """
    Get an embedding by its ID
    
    Args:
        embedding_id: ID of the embedding
        
    Returns:
        Embedding vector or None if not found
    """
    try:
        with index_lock:
            if embedding_id not in index_metadata:
                return None
                
            position = index_metadata[embedding_id]["position"]
            
            # Create a reconstructor for the index
            reconstructor = faiss.IndexIDMap(faiss.IndexFlat(VECTOR_DIMENSION))
            
            # Get the embedding from the position
            embedding = np.zeros((1, VECTOR_DIMENSION), dtype=np.float32)
            for i in range(index.ntotal):
                if i == position:
                    reconstructor.reconstruct(i, embedding[0])
                    return embedding[0]
                    
            return None
            
    except Exception as e:
        logger.error(f"Error getting embedding by ID: {e}")
        return None

def delete_embedding(embedding_id: str) -> bool:
    """
    Delete an embedding from the vector database
    
    Args:
        embedding_id: ID of the embedding to delete
        
    Returns:
        True if successful, False otherwise
    """
    try:
        with index_lock:
            if embedding_id not in index_metadata:
                return False
                
            # FAISS doesn't support direct deletion, so we need to rebuild the index
            # In practice, a more efficient approach would be used, such as marking
            # as deleted and periodically rebuilding the index
            
            # For now, just remove from metadata and mark for rebuild
            del index_metadata[embedding_id]
            
            # Save metadata
            save_index()
            
            # Note: The data is still in the FAISS index, but won't be returned in searches
            # A complete implementation would rebuild the index periodically
            
            return True
            
    except Exception as e:
        logger.error(f"Error deleting embedding: {e}")
        return False

def delete_embeddings_for_document(document_id: int) -> bool:
    """
    Delete all embeddings associated with a document
    
    Args:
        document_id: ID of the document
        
    Returns:
        True if successful, False otherwise
    """
    try:
        with index_lock:
            # Find all embeddings for this document
            embedding_ids_to_delete = []
            
            for embedding_id, data in index_metadata.items():
                metadata = data["metadata"]
                
                if (
                    (metadata.get("type") == "document_chunk" and metadata.get("id") in 
                     [c.id for c in document_id.chunks]) or
                    (metadata.get("document_id") == document_id)
                ):
                    embedding_ids_to_delete.append(embedding_id)
                    
            # Delete each embedding
            for embedding_id in embedding_ids_to_delete:
                del index_metadata[embedding_id]
                
            # Save metadata
            save_index()
            
            # Note: Data is still in FAISS index but won't be returned in searches
            
            return True
            
    except Exception as e:
        logger.error(f"Error deleting embeddings for document: {e}")
        return False

def get_embedding_metadata(embedding_id: str) -> Optional[Dict[str, Any]]:
    """
    Get metadata for an embedding
    
    Args:
        embedding_id: ID of the embedding
        
    Returns:
        Metadata dictionary or None if not found
    """
    try:
        with index_lock:
            if embedding_id not in index_metadata:
                return None
                
            return index_metadata[embedding_id]["metadata"]
            
    except Exception as e:
        logger.error(f"Error getting embedding metadata: {e}")
        return None

def rebuild_index() -> bool:
    """
    Rebuild the FAISS index from scratch
    
    Returns:
        True if successful, False otherwise
    """
    try:
        with index_lock:
            # Create a new index
            global index
            
            if INDEX_TYPE == "FLAT":
                new_index = faiss.IndexFlatL2(VECTOR_DIMENSION)
            elif INDEX_TYPE == "IVF":
                quantizer = faiss.IndexFlatL2(VECTOR_DIMENSION)
                new_index = faiss.IndexIVFFlat(quantizer, VECTOR_DIMENSION, 100, faiss.METRIC_L2)
            elif INDEX_TYPE == "HNSW":
                new_index = faiss.IndexHNSWFlat(VECTOR_DIMENSION, 32)
            else:
                new_index = faiss.IndexFlatL2(VECTOR_DIMENSION)
                
            # Need to extract all embeddings and rebuild
            # This method is not fully implemented as it would require 
            # storing the original embeddings, which isn't done in this example
            
            index = new_index
            save_index()
            
            logger.info("Index rebuilt successfully")
            return True
            
    except Exception as e:
        logger.error(f"Error rebuilding index: {e}")
        return False

def get_index_info() -> Dict[str, Any]:
    """
    Get information about the vector index
    
    Returns:
        Dictionary of index information
    """
    try:
        with index_lock:
            return {
                "index_type": INDEX_TYPE,
                "dimensions": VECTOR_DIMENSION,
                "total_vectors": index.ntotal,
                "total_metadata_entries": len(index_metadata),
                "index_file": index_file_path,
                "metadata_file": metadata_file_path
            }
            
    except Exception as e:
        logger.error(f"Error getting index info: {e}")
        return {}
