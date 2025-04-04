import logging
import os
from typing import List, Dict, Any, Optional, Tuple, Union
import numpy as np
import hashlib
import json

# Check for availability of packages
try:
    import torch
    from torch import Tensor
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    Tensor = Any  # Type alias for when torch is not available

try:
    from transformers import AutoTokenizer, AutoModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

# Import OpenAI for fallback
try:
    import openai
    from openai import OpenAI
    OPENAI_AVAILABLE = True
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
except ImportError:
    OPENAI_AVAILABLE = False

from app import db
from config import (
    DOCUMENT_EMBEDDING_MODEL, QUERY_EMBEDDING_MODEL, ENTITY_EMBEDDING_MODEL, 
    VECTOR_DIMENSION, BATCH_SIZE
)
from models import DocumentChunk, Entity, Query

logger = logging.getLogger(__name__)

# Initialize embedding models
document_model = None
query_model = None
entity_model = None
USE_OPENAI_EMBEDDINGS = False

def init_models():
    """Initialize the embedding models"""
    global document_model, query_model, entity_model, USE_OPENAI_EMBEDDINGS
    
    try:
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            # Load document embedding model
            document_model = SentenceTransformer(DOCUMENT_EMBEDDING_MODEL)
            logger.info(f"Loaded document embedding model: {DOCUMENT_EMBEDDING_MODEL}")
            
            # Load query embedding model (or use the same model)
            if QUERY_EMBEDDING_MODEL == DOCUMENT_EMBEDDING_MODEL:
                query_model = document_model
            else:
                query_model = SentenceTransformer(QUERY_EMBEDDING_MODEL)
            logger.info(f"Loaded query embedding model: {QUERY_EMBEDDING_MODEL}")
            
            # Load entity embedding model (or use the same model)
            if ENTITY_EMBEDDING_MODEL == DOCUMENT_EMBEDDING_MODEL:
                entity_model = document_model
            else:
                entity_model = SentenceTransformer(ENTITY_EMBEDDING_MODEL)
            logger.info(f"Loaded entity embedding model: {ENTITY_EMBEDDING_MODEL}")
        elif OPENAI_AVAILABLE:
            # Use OpenAI embeddings as fallback
            USE_OPENAI_EMBEDDINGS = True
            logger.info("Using OpenAI embeddings as fallback (sentence-transformers not available)")
        else:
            logger.error("No embedding model available. Install sentence-transformers or provide OpenAI API key.")
            raise ImportError("No embedding model available")
        
    except Exception as e:
        logger.error(f"Error loading embedding models: {e}")
        # Try OpenAI as fallback
        if OPENAI_AVAILABLE and not USE_OPENAI_EMBEDDINGS:
            USE_OPENAI_EMBEDDINGS = True
            logger.info("Using OpenAI embeddings as fallback after error with local models")
        else:
            raise

def mean_pooling(model_output: Dict[str, Tensor], attention_mask: Tensor) -> Tensor:
    """
    Perform mean pooling on model outputs using attention mask
    
    Args:
        model_output: The output from the transformer model
        attention_mask: The attention mask for the input
        
    Returns:
        Pooled embeddings tensor
    """
    token_embeddings = model_output[0]  # First element contains token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def embed_texts(texts: List[str], ids: List[int], embedding_type: str = "document") -> List[str]:
    """
    Create embeddings for a list of texts and store them in the vector database
    
    Args:
        texts: List of text strings to embed
        ids: List of corresponding IDs (document_chunk IDs, entity IDs, etc.)
        embedding_type: Type of embedding ("document", "query", or "entity")
        
    Returns:
        List of embedding IDs (from vector store)
    """
    try:
        # Initialize models if not already done
        if document_model is None and not USE_OPENAI_EMBEDDINGS:
            init_models()
        
        # Handle empty input
        if not texts or len(texts) == 0:
            return []
            
        # Create embeddings in batches
        embedding_ids = []
        
        # Get vector store module
        from services import vector_store
        
        # Use OpenAI embeddings if configured
        if USE_OPENAI_EMBEDDINGS:
            if not OPENAI_AVAILABLE:
                raise ValueError("OpenAI API not available but embeddings are configured to use it")
                
            for i in range(0, len(texts), BATCH_SIZE):
                batch_texts = texts[i:i+BATCH_SIZE]
                batch_ids = ids[i:i+BATCH_SIZE]
                
                # Process each text individually to handle potential errors
                embeddings = []
                processed_ids = []
                metadata_list = []
                
                for j, text in enumerate(batch_texts):
                    try:
                        # Truncate if too long for OpenAI
                        if len(text) > 8000:
                            logger.warning(f"Truncating text from {len(text)} to 8000 chars for OpenAI embedding")
                            text = text[:8000]
                        
                        # Generate OpenAI embedding
                        response = client.embeddings.create(
                            input=text,
                            model="text-embedding-3-small"
                        )
                        embedding = np.array(response.data[0].embedding)
                        embeddings.append(embedding)
                        
                        # Create metadata
                        text_id = batch_ids[j]
                        if embedding_type == "document":
                            metadata = {"type": "document_chunk", "id": text_id}
                        elif embedding_type == "query":
                            metadata = {"type": "query", "id": text_id}
                        elif embedding_type == "entity":
                            metadata = {"type": "entity", "id": text_id}
                        metadata_list.append(metadata)
                        processed_ids.append(text_id)
                        
                    except Exception as e:
                        logger.error(f"Error creating OpenAI embedding for text {j}: {e}")
                        # Skip this item and continue with others
                
                # Store successful embeddings
                if embeddings:
                    embeddings_array = np.array(embeddings)
                    batch_embedding_ids = vector_store.store_embeddings(embeddings_array, metadata_list)
                    embedding_ids.extend(batch_embedding_ids)
                else:
                    logger.warning(f"No successful embeddings in batch {i}")
        
        # Use local models
        else:
            # Choose appropriate model
            if embedding_type == "document":
                model = document_model
            elif embedding_type == "query":
                model = query_model
            elif embedding_type == "entity":
                model = entity_model
            else:
                raise ValueError(f"Unknown embedding type: {embedding_type}")
            
            for i in range(0, len(texts), BATCH_SIZE):
                batch_texts = texts[i:i+BATCH_SIZE]
                batch_ids = ids[i:i+BATCH_SIZE]
                
                # Generate embeddings
                embeddings = model.encode(batch_texts, convert_to_numpy=True)
                
                # Create metadata dictionaries for each text
                metadata_list = []
                for j, text_id in enumerate(batch_ids):
                    if embedding_type == "document":
                        metadata = {"type": "document_chunk", "id": text_id}
                    elif embedding_type == "query":
                        metadata = {"type": "query", "id": text_id}
                    elif embedding_type == "entity":
                        metadata = {"type": "entity", "id": text_id}
                    metadata_list.append(metadata)
                
                # Store embeddings in vector database
                batch_embedding_ids = vector_store.store_embeddings(embeddings, metadata_list)
                embedding_ids.extend(batch_embedding_ids)
        
        # Return all embedding IDs that were successfully created            
        return embedding_ids
        
    except Exception as e:
        logger.error(f"Error creating embeddings batch: {e}")
        
        # Try OpenAI as fallback if not already using it
        if not USE_OPENAI_EMBEDDINGS and OPENAI_AVAILABLE:
            logger.info("Attempting OpenAI fallback for batch embedding after error")
            try:
                # Simplified fallback approach with single texts for reliability
                fallback_ids = []
                
                for i, (text, text_id) in enumerate(zip(texts, ids)):
                    try:
                        # Create embedding with OpenAI
                        embedding, embedding_id = embed_text(text, {"id": text_id, "fallback": True}, embedding_type)
                        
                        if embedding is not None and embedding_id is not None:
                            # Store in vector database
                            if embedding_type == "document":
                                metadata = {"type": "document_chunk", "id": text_id, "fallback": True}
                            elif embedding_type == "query":
                                metadata = {"type": "query", "id": text_id, "fallback": True}
                            elif embedding_type == "entity":
                                metadata = {"type": "entity", "id": text_id, "fallback": True}
                                
                            from services import vector_store
                            vector_store.store_embedding(embedding, metadata)
                            fallback_ids.append(embedding_id)
                        else:
                            fallback_ids.append(None)
                            
                    except Exception as inner_e:
                        logger.error(f"Error in OpenAI fallback for text {i}: {inner_e}")
                        fallback_ids.append(None)
                
                return fallback_ids
                
            except Exception as fallback_e:
                logger.error(f"Error in OpenAI fallback batch processing: {fallback_e}")
                
        # Return None for each text if all attempts failed
        return [None] * len(texts)

def embed_text(text: str, metadata: Dict[str, Any] = None, embedding_type: str = "document") -> Tuple[np.ndarray, str]:
    """
    Create embedding for a single text
    
    Args:
        text: Text to embed
        metadata: Optional metadata to store with the embedding
        embedding_type: Type of embedding ("document", "query", or "entity")
        
    Returns:
        Tuple containing the embedding array and embedding ID
    """
    try:
        # Initialize models if not already done
        if document_model is None and not USE_OPENAI_EMBEDDINGS:
            init_models()
            
        # Create embedding ID
        if metadata:
            metadata_str = json.dumps(metadata, sort_keys=True)
        else:
            metadata_str = ""
        embedding_id = hashlib.sha256(f"{text}_{metadata_str}_{embedding_type}".encode()).hexdigest()
        
        # Use OpenAI if configured
        if USE_OPENAI_EMBEDDINGS:
            if not OPENAI_AVAILABLE:
                raise ValueError("OpenAI API not available but embeddings are configured to use it")
                
            # Truncate text if too long (OpenAI has token limits)
            if len(text) > 8000:
                logger.warning(f"Truncating text from {len(text)} to 8000 chars for OpenAI embedding")
                text = text[:8000]
                
            # Call OpenAI embeddings API
            response = client.embeddings.create(
                input=text,
                model="text-embedding-3-small"  # Modern, efficient embedding model
            )
            embedding = np.array(response.data[0].embedding)
            
            return embedding, embedding_id
        else:
            # Choose appropriate local model
            if embedding_type == "document":
                model = document_model
            elif embedding_type == "query":
                model = query_model
            elif embedding_type == "entity":
                model = entity_model
            else:
                raise ValueError(f"Unknown embedding type: {embedding_type}")
                
            # Generate embedding with local model
            embedding = model.encode(text, convert_to_numpy=True)
            return embedding, embedding_id
        
    except Exception as e:
        logger.error(f"Error creating single embedding: {e}")
        if USE_OPENAI_EMBEDDINGS and OPENAI_AVAILABLE:
            # Retry with OpenAI as fallback for local model errors
            try:
                if not metadata:
                    metadata = {}
                metadata["fallback"] = "openai_after_error"
                
                # Truncate text if too long
                if len(text) > 8000:
                    text = text[:8000]
                    
                # Call OpenAI embeddings API
                response = client.embeddings.create(
                    input=text,
                    model="text-embedding-3-small"
                )
                embedding = np.array(response.data[0].embedding)
                
                # Create new embedding ID for the fallback
                metadata_str = json.dumps(metadata, sort_keys=True)
                embedding_id = hashlib.sha256(f"{text}_{metadata_str}_{embedding_type}_fallback".encode()).hexdigest()
                
                logger.info(f"Successfully created fallback embedding with OpenAI")
                return embedding, embedding_id
                
            except Exception as inner_e:
                logger.error(f"Error creating fallback embedding with OpenAI: {inner_e}")
                
        return None, None

def embed_query(query_text: str, query_id: int) -> str:
    """
    Create embedding for a search query
    
    Args:
        query_text: Query text to embed
        query_id: ID of the query
        
    Returns:
        Embedding ID
    """
    try:
        # Create embedding
        embedding, embedding_id = embed_text(query_text, {"query_id": query_id}, "query")
        
        # Store in vector database
        from services import vector_store
        vector_store.store_embedding(embedding, {"type": "query", "id": query_id})
        
        # Update query record with embedding ID
        query = Query.query.get(query_id)
        if query:
            query.embedding_id = embedding_id
            db.session.commit()
            
        return embedding_id
        
    except Exception as e:
        logger.error(f"Error creating query embedding: {e}")
        return None

def embed_entity(entity_id: int) -> str:
    """
    Create embedding for an entity
    
    Args:
        entity_id: ID of the entity
        
    Returns:
        Embedding ID
    """
    try:
        # Get entity from database
        entity = Entity.query.get(entity_id)
        if not entity:
            logger.error(f"Entity not found: {entity_id}")
            return None
            
        # Create context-aware entity representation by including surrounding text
        chunk = DocumentChunk.query.get(entity.chunk_id) if entity.chunk_id else None
        if chunk:
            # Use entity name + context
            context_start = max(0, entity.start_char - chunk.start_char - 50)
            context_end = min(len(chunk.content), entity.end_char - chunk.start_char + 50)
            context = chunk.content[context_start:context_end]
            entity_text = f"{entity.name} - {entity.entity_type} - {context}"
        else:
            # Use just entity name and type
            entity_text = f"{entity.name} - {entity.entity_type}"
            
        # Create embedding
        embedding, embedding_id = embed_text(entity_text, {"entity_id": entity_id}, "entity")
        
        # Store in vector database
        from services import vector_store
        vector_store.store_embedding(embedding, {"type": "entity", "id": entity_id})
        
        # Update entity record with embedding ID
        entity.embedding_id = embedding_id
        db.session.commit()
        
        return embedding_id
        
    except Exception as e:
        logger.error(f"Error creating entity embedding: {e}")
        return None

def get_embedding_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """
    Calculate cosine similarity between two embeddings
    
    Args:
        embedding1: First embedding
        embedding2: Second embedding
        
    Returns:
        Cosine similarity score between 0 and 1
    """
    try:
        # Normalize embeddings
        embedding1_norm = embedding1 / np.linalg.norm(embedding1)
        embedding2_norm = embedding2 / np.linalg.norm(embedding2)
        
        # Calculate cosine similarity
        similarity = np.dot(embedding1_norm, embedding2_norm)
        
        return float(similarity)
        
    except Exception as e:
        logger.error(f"Error calculating embedding similarity: {e}")
        return 0.0

def list_available_models() -> Dict[str, List[str]]:
    """
    List available embedding models for different content types
    
    Returns:
        Dictionary of model types and available models
    """
    return {
        "document": [
            "sentence-transformers/all-mpnet-base-v2",
            "sentence-transformers/all-MiniLM-L6-v2",
            "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
        ],
        "query": [
            "sentence-transformers/all-mpnet-base-v2",
            "sentence-transformers/msmarco-distilbert-base-tas-b",
            "sentence-transformers/multi-qa-mpnet-base-dot-v1"
        ],
        "entity": [
            "sentence-transformers/all-mpnet-base-v2",
            "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
            "sentence-transformers/all-MiniLM-L6-v2"
        ]
    }
