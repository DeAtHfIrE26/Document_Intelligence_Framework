import logging
import time
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import os

# Check if OpenAI API key is available
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_AVAILABLE = OPENAI_API_KEY is not None

# Try importing sentence_transformers
try:
    from sentence_transformers import CrossEncoder
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

# Import OpenAI for fallback
try:
    import openai
    from openai import OpenAI
    OPENAI_CLIENT_AVAILABLE = True and OPENAI_AVAILABLE
    if OPENAI_CLIENT_AVAILABLE:
        client = OpenAI(api_key=OPENAI_API_KEY)
except ImportError:
    OPENAI_CLIENT_AVAILABLE = False

from app import app, db
from config import (
    RERANKING_MODEL, MAX_THREADS
)
from services import llm_service
from utils.helpers import get_cache, set_cache

logger = logging.getLogger(__name__)

# Initialize reranking model
reranker = None
USE_OPENAI_RERANKING = False

def init_app(app):
    """Initialize the reranking service with the app context"""
    global reranker, USE_OPENAI_RERANKING
    
    if SENTENCE_TRANSFORMERS_AVAILABLE:
        try:
            # Load cross-encoder model
            reranker = CrossEncoder(RERANKING_MODEL)
            logger.info(f"Loaded reranking model: {RERANKING_MODEL}")
        except Exception as e:
            logger.error(f"Error loading reranking model: {e}")
            reranker = None
    elif OPENAI_CLIENT_AVAILABLE:
        # Use OpenAI for reranking if sentence-transformers is not available
        USE_OPENAI_RERANKING = True
        logger.info("Using OpenAI for reranking (sentence-transformers not available)")
    else:
        logger.warning("No reranking model available - results will use default ranking only")
        reranker = None

def rerank_results(query: str, results: List[Dict[str, Any]], 
                 query_data: Dict[str, Any] = None) -> List[Dict[str, Any]]:
    """
    Rerank search results to improve relevance
    
    Args:
        query: Original search query
        results: List of search results to rerank
        query_data: Additional query context data
        
    Returns:
        Reranked results list
    """
    try:
        # If no results or only one result, no need to rerank
        if not results or len(results) <= 1:
            return results
            
        start_time = time.time()
        
        # Prepare cross-encoder inputs
        pairs = []
        for result in results:
            # Use context window if available, otherwise use content
            text = result.get("context", result.get("content", ""))
            pairs.append([query, text])
        
        # Calculate relevance scores using cross-encoder if available
        if reranker:
            # Get base cross-encoder scores using local model
            try:
                cross_encoder_scores = reranker.predict(pairs)
                
                # Normalize scores to 0-1 range
                if len(cross_encoder_scores) > 0:
                    min_score = min(cross_encoder_scores)
                    max_score = max(cross_encoder_scores)
                    score_range = max(0.00001, max_score - min_score)  # Avoid division by zero
                    
                    normalized_scores = [(score - min_score) / score_range for score in cross_encoder_scores]
                    
                    # Update scores in results
                    for i, result in enumerate(results):
                        # Blend original score with cross-encoder score (70% cross-encoder, 30% original)
                        original_score = result["score"]
                        cross_score = normalized_scores[i]
                        blended_score = (cross_score * 0.7) + (original_score * 0.3)
                        
                        # Update result with new score
                        result["original_score"] = original_score
                        result["reranker_score"] = cross_score
                        result["score"] = blended_score
                    
                    # Resort based on new scores
                    results = sorted(results, key=lambda x: x["score"], reverse=True)
                    
                    # Update ranks
                    for i, result in enumerate(results):
                        result["rank"] = i + 1
                        
            except Exception as e:
                logger.error(f"Error in cross-encoder reranking: {e}")
        
        # Use OpenAI for reranking if configured
        elif USE_OPENAI_RERANKING and OPENAI_CLIENT_AVAILABLE:
            try:
                # Only rerank up to 10 results to limit API usage
                rerank_count = min(10, len(results))
                rerank_items = results[:rerank_count]
                
                # Prepare documents for OpenAI reranking
                documents = []
                for i, result in enumerate(rerank_items):
                    # Use context window if available, otherwise use content
                    text = result.get("context", result.get("content", ""))
                    # Truncate text to avoid token limits
                    if len(text) > 1500:
                        text = text[:1500] + "..."
                    documents.append({"id": str(i), "text": text})
                
                # Call OpenAI reranking endpoint
                response = client.embeddings.create(
                    input=query,
                    model="text-embedding-3-small",
                )
                
                query_embedding = response.data[0].embedding
                
                # Calculate scores based on embedding similarity
                normalized_scores = []
                for doc in documents:
                    doc_response = client.embeddings.create(
                        input=doc["text"],
                        model="text-embedding-3-small",
                    )
                    doc_embedding = doc_response.data[0].embedding
                    
                    # Calculate cosine similarity
                    similarity = np.dot(query_embedding, doc_embedding) / (
                        np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
                    )
                    normalized_scores.append(similarity)
                
                # Update scores in results
                for i, result in enumerate(rerank_items):
                    if i < len(normalized_scores):
                        # Blend original score with OpenAI score (70% OpenAI, 30% original)
                        original_score = result["score"]
                        openai_score = normalized_scores[i]
                        blended_score = (openai_score * 0.7) + (original_score * 0.3)
                        
                        # Update result with new score
                        result["original_score"] = original_score
                        result["reranker_score"] = openai_score
                        result["score"] = blended_score
                
                # Sort all results (including non-reranked ones)
                results = sorted(results, key=lambda x: x.get("score", 0), reverse=True)
                
                # Update ranks
                for i, result in enumerate(results):
                    result["rank"] = i + 1
                    
                logger.debug(f"OpenAI reranking completed for {rerank_count} results")
                
            except Exception as e:
                logger.error(f"Error in OpenAI reranking: {e}")
        
        # Apply additional reranking strategies
        results = apply_additional_reranking(query, results, query_data)
        
        logger.debug(f"Reranking completed in {time.time() - start_time:.2f}s")
        return results
        
    except Exception as e:
        logger.error(f"Error in reranking: {e}")
        return results

def apply_additional_reranking(query: str, results: List[Dict[str, Any]], 
                            query_data: Dict[str, Any] = None) -> List[Dict[str, Any]]:
    """
    Apply additional reranking strategies
    
    Args:
        query: Original search query
        results: Results list (already reranked by cross-encoder)
        query_data: Additional query context data
        
    Returns:
        Results with additional reranking applied
    """
    try:
        # Apply diversification to promote diversity in top results
        results = apply_diversity_reranking(results)
        
        # Apply entity-aware reranking if query has entities
        if query_data and "entities" in query_data and query_data["entities"]:
            results = apply_entity_aware_reranking(results, query_data["entities"])
        
        # Apply intent-based reranking
        if query_data and "intent" in query_data:
            results = apply_intent_based_reranking(results, query_data["intent"])
        
        # Apply LLM-based reranking for ambiguous cases
        # Only apply to top results to minimize API calls
        top_results = results[:min(5, len(results))]
        if is_ambiguous_results(top_results):
            top_reranked = apply_llm_reranking(query, top_results)
            # Replace top results with reranked versions
            results = top_reranked + results[len(top_reranked):]
        
        return results
        
    except Exception as e:
        logger.error(f"Error in additional reranking: {e}")
        return results

def apply_diversity_reranking(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Apply diversity-based reranking to promote variety in results
    
    Args:
        results: Results list to diversify
        
    Returns:
        Diversified results list
    """
    try:
        if len(results) <= 5:
            return results
            
        # Extract document IDs
        doc_ids = [r["document_id"] for r in results]
        
        # Count occurrences of each document
        doc_counts = {}
        for doc_id in doc_ids:
            doc_counts[doc_id] = doc_counts.get(doc_id, 0) + 1
        
        # Apply diversity penalty for documents with multiple chunks
        for i, result in enumerate(results):
            doc_id = result["document_id"]
            if doc_counts[doc_id] > 1:
                # Apply stronger penalty to lower-ranked results from the same document
                doc_rank = [j for j, r in enumerate(results) if r["document_id"] == doc_id].index(i)
                diversity_penalty = 0.05 * doc_rank
                
                # Apply penalty (max 15% reduction)
                result["score"] = max(result["score"] * (1 - diversity_penalty), result["score"] * 0.85)
        
        # Resort based on updated scores
        results = sorted(results, key=lambda x: x["score"], reverse=True)
        
        # Update ranks
        for i, result in enumerate(results):
            result["rank"] = i + 1
            
        return results
        
    except Exception as e:
        logger.error(f"Error in diversity reranking: {e}")
        return results

def apply_entity_aware_reranking(results: List[Dict[str, Any]], 
                              query_entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Apply entity-aware reranking to boost results matching query entities
    
    Args:
        results: Results list to rerank
        query_entities: Entities extracted from the query
        
    Returns:
        Entity-aware reranked results
    """
    try:
        if not query_entities or not results:
            return results
            
        # Extract entity names from query
        query_entity_names = [e["text"].lower() for e in query_entities]
        
        # Boost scores for results containing query entities
        for result in results:
            content = result.get("content", "").lower()
            entity_matches = sum(1 for entity in query_entity_names if entity in content)
            
            if entity_matches > 0:
                # Apply boost based on number of matching entities
                entity_boost = min(0.1 * entity_matches, 0.3)  # Max 30% boost
                result["score"] = min(result["score"] * (1 + entity_boost), 1.0)
        
        # Resort based on updated scores
        results = sorted(results, key=lambda x: x["score"], reverse=True)
        
        # Update ranks
        for i, result in enumerate(results):
            result["rank"] = i + 1
            
        return results
        
    except Exception as e:
        logger.error(f"Error in entity-aware reranking: {e}")
        return results

def apply_intent_based_reranking(results: List[Dict[str, Any]], 
                              intent: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Apply intent-based reranking based on query intent
    
    Args:
        results: Results list to rerank
        intent: Query intent information
        
    Returns:
        Intent-based reranked results
    """
    try:
        if not intent or not results:
            return results
            
        # Apply different strategies based on intent
        command_type = intent.get("command_type")
        
        if command_type == "compare":
            # For comparison intent, promote diversity of content
            results = apply_diversity_reranking(results)
            
        elif command_type == "summarize":
            # For summarization intent, prioritize comprehensive results
            for result in results:
                content_length = len(result.get("content", ""))
                # Boost longer, more comprehensive content
                if content_length > 500:
                    length_boost = min(0.05 * (content_length / 500), 0.15)  # Max 15% boost
                    result["score"] = min(result["score"] * (1 + length_boost), 1.0)
                    
        elif command_type == "find" and intent.get("is_question"):
            # For factual questions, boost results with likely answers
            for result in results:
                content = result.get("content", "")
                # Simple heuristic: boost content with numbers, dates, names
                has_number = bool(re.search(r'\d+', content))
                has_date = bool(re.search(r'\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|[0-9]{4})\b', content))
                
                if has_number or has_date:
                    fact_boost = 0.1  # 10% boost
                    result["score"] = min(result["score"] * (1 + fact_boost), 1.0)
        
        # Resort based on updated scores
        results = sorted(results, key=lambda x: x["score"], reverse=True)
        
        # Update ranks
        for i, result in enumerate(results):
            result["rank"] = i + 1
            
        return results
        
    except Exception as e:
        logger.error(f"Error in intent-based reranking: {e}")
        return results

def is_ambiguous_results(results: List[Dict[str, Any]]) -> bool:
    """
    Check if results seem ambiguous or unclear
    
    Args:
        results: Top search results
        
    Returns:
        True if results seem ambiguous
    """
    try:
        if not results or len(results) < 2:
            return False
            
        # Calculate score distribution
        scores = [r["score"] for r in results]
        avg_score = sum(scores) / len(scores)
        
        # Check if all scores are very similar (suggests ambiguity)
        score_range = max(scores) - min(scores)
        if score_range < 0.1 and avg_score > 0.7:
            return True
            
        # Check for diverse content topics (suggests ambiguity)
        if len(set(r["document_id"] for r in results)) == len(results):
            return True
            
        return False
        
    except Exception as e:
        logger.error(f"Error checking result ambiguity: {e}")
        return False

def apply_llm_reranking(query: str, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Apply LLM-based reranking for challenging cases
    
    Args:
        query: Original search query
        results: Top results to rerank
        
    Returns:
        LLM-reranked results
    """
    try:
        # Skip if no results or only one result
        if not results or len(results) <= 1:
            return results
            
        # Prepare result contexts for LLM
        result_contexts = []
        for result in results:
            snippet = result.get("content", "")[:1000]  # Limit length for LLM
            result_contexts.append({
                "id": result["rank"],
                "text": snippet,
                "current_score": result["score"]
            })
            
        # Get LLM-based reranking
        reranked_order = llm_service.rerank_results(query, result_contexts)
        
        if not reranked_order:
            return results
            
        # Use LLM decision to reorder results
        reranked_results = []
        for rank in reranked_order:
            for result in results:
                if result["rank"] == rank:
                    # Update score based on new rank
                    result["llm_rank"] = reranked_order.index(rank) + 1
                    score_boost = 0.1 * (len(results) - reranked_order.index(rank)) / len(results)
                    result["score"] = min(result["score"] * (1 + score_boost), 1.0)
                    reranked_results.append(result)
                    break
        
        # If any results are missing, add them back at the end
        for result in results:
            if result not in reranked_results:
                reranked_results.append(result)
        
        # Resort based on updated scores
        reranked_results = sorted(reranked_results, key=lambda x: x["score"], reverse=True)
        
        # Update ranks
        for i, result in enumerate(reranked_results):
            result["rank"] = i + 1
            
        return reranked_results
        
    except Exception as e:
        logger.error(f"Error in LLM reranking: {e}")
        return results

import re  # Import re module for regex in intent-based reranking
