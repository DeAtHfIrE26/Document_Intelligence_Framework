import logging
import time
import json
from typing import List, Dict, Any, Optional, Tuple, Set
from concurrent.futures import ThreadPoolExecutor
import numpy as np

from app import app, db
from models import Document, DocumentChunk, Entity, Query, QueryResult
from config import (
    TOP_K_RETRIEVAL, KNOWLEDGE_GRAPH_DEPTH, MAX_THREADS,
    QUERY_EXPANSION_TECHNIQUES, ACCESS_CONTROL_ENABLED
)
from services import vector_store, knowledge_graph, embedding_service, query_processor, reranking_service
from utils.helpers import get_cache, set_cache

logger = logging.getLogger(__name__)

def retrieve_documents(query_data: Dict[str, Any], user_id: Optional[int] = None) -> Dict[str, Any]:
    """
    Retrieve documents based on processed query data
    
    Args:
        query_data: Processed query data from query_processor
        user_id: ID of the user making the query (for access control)
        
    Returns:
        Dictionary of retrieval results
    """
    try:
        # Get query details
        query_id = query_data.get("query_id")
        original_query = query_data.get("original_query", "")
        embedding_id = query_data.get("query_embedding")
        entities = query_data.get("entities", [])
        expanded_queries = query_data.get("expanded_queries", [])
        filters = query_data.get("filters", {})
        search_type = query_data.get("search_type", "hybrid")
        
        start_time = time.time()
        
        # Perform multi-strategy retrieval
        results = {}
        
        with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
            futures = []
            
            # Vector retrieval
            if search_type in ["hybrid", "vector"]:
                futures.append(executor.submit(
                    perform_vector_retrieval,
                    query_id, 
                    embedding_id, 
                    filters,
                    TOP_K_RETRIEVAL
                ))
            
            # Keyword-based retrieval
            if search_type in ["hybrid", "keyword"]:
                futures.append(executor.submit(
                    perform_keyword_retrieval,
                    query_id,
                    original_query,
                    expanded_queries,
                    filters,
                    TOP_K_RETRIEVAL
                ))
            
            # Entity-based graph retrieval
            if entities and search_type in ["hybrid", "entity"]:
                futures.append(executor.submit(
                    perform_entity_retrieval,
                    query_id,
                    entities,
                    filters,
                    TOP_K_RETRIEVAL
                ))
            
            # Collect results
            for future in futures:
                try:
                    retrieval_result = future.result()
                    if retrieval_result and "type" in retrieval_result:
                        results[retrieval_result["type"]] = retrieval_result
                except Exception as e:
                    logger.error(f"Error in retrieval task: {e}")
        
        # Merge and rank results
        merged_results = merge_retrieval_results(results, TOP_K_RETRIEVAL)
        
        # Apply access control filtering
        if ACCESS_CONTROL_ENABLED and user_id is not None:
            merged_results = filter_by_access_control(merged_results, user_id)
        
        # Apply reranking
        reranked_results = reranking_service.rerank_results(
            original_query, 
            merged_results,
            query_data
        )
        
        # Store results in database
        store_query_results(query_id, reranked_results)
        
        elapsed_time = time.time() - start_time
        
        return {
            "query_id": query_id,
            "results": reranked_results,
            "total_results": len(reranked_results),
            "elapsed_time": elapsed_time,
            "retrieval_strategies": list(results.keys())
        }
        
    except Exception as e:
        logger.error(f"Error retrieving documents: {e}")
        return {
            "query_id": query_data.get("query_id"),
            "results": [],
            "total_results": 0,
            "error": str(e)
        }

def perform_vector_retrieval(query_id: int, embedding_id: str, 
                           filters: Dict[str, Any], k: int) -> Dict[str, Any]:
    """
    Perform vector-based retrieval using query embedding
    
    Args:
        query_id: ID of the query
        embedding_id: ID of the query embedding
        filters: Filters to apply to the search
        k: Number of results to retrieve
        
    Returns:
        Dictionary of vector retrieval results
    """
    try:
        # Skip if no embedding
        if not embedding_id:
            return {"type": "vector", "results": []}
            
        # Get embedding vector from vector store
        query_embedding = vector_store.get_embedding_by_id(embedding_id)
        if query_embedding is None:
            return {"type": "vector", "results": []}
            
        # Convert query filters to vector store filters
        vector_filters = query_processor.convert_query_to_vector_filters(filters)
        
        # Search for similar chunks
        similar_chunks = vector_store.search_similar(
            query_embedding, 
            k=k*2,  # Get more results initially for filtering
            filter_dict=vector_filters
        )
        
        # Process results
        results = []
        for i, chunk in enumerate(similar_chunks):
            # Get chunk and document data
            metadata = chunk["metadata"]
            
            if metadata.get("type") != "document_chunk":
                continue
                
            chunk_id = metadata.get("id")
            db_chunk = DocumentChunk.query.get(chunk_id)
            
            if not db_chunk:
                continue
                
            document = Document.query.get(db_chunk.document_id)
            
            if not document:
                continue
                
            # Calculate context window (include neighboring chunks)
            context = get_context_window(db_chunk)
            
            # Add to results
            results.append({
                "document_id": document.id,
                "document_title": document.title,
                "chunk_id": db_chunk.id,
                "content": db_chunk.content,
                "context": context,
                "score": chunk["similarity"],
                "rank": i+1,
                "strategy": "vector"
            })
            
            # Limit results
            if len(results) >= k:
                break
                
        return {
            "type": "vector",
            "results": results,
            "count": len(results)
        }
        
    except Exception as e:
        logger.error(f"Error in vector retrieval: {e}")
        return {"type": "vector", "results": []}

def perform_keyword_retrieval(query_id: int, query_text: str, 
                            expanded_queries: List[Dict[str, Any]],
                            filters: Dict[str, Any], k: int) -> Dict[str, Any]:
    """
    Perform keyword-based retrieval
    
    Args:
        query_id: ID of the query
        query_text: Original query text
        expanded_queries: List of expanded queries
        filters: Filters to apply to the search
        k: Number of results to retrieve
        
    Returns:
        Dictionary of keyword retrieval results
    """
    try:
        # Prepare search terms
        search_texts = [query_text]
        for expanded in expanded_queries:
            if expanded["technique"] == "synonym":
                search_texts.append(expanded["text"])
                
        # Construct SQL query
        results = []
        
        # For simplicity, we'll use a basic LIKE query
        # In a production system, this would use a proper full-text search engine
        for search_text in search_texts[:3]:  # Limit to 3 expanded queries
            # Split query into keywords
            keywords = search_text.split()
            
            # Create query for each keyword
            for keyword in keywords:
                if len(keyword) < 3:
                    continue
                    
                # Search in chunks
                chunks = DocumentChunk.query.filter(
                    DocumentChunk.content.ilike(f"%{keyword}%")
                ).limit(k*2).all()
                
                for i, chunk in enumerate(chunks):
                    # Get document data
                    document = Document.query.get(chunk.document_id)
                    
                    if not document:
                        continue
                        
                    # Apply filters
                    if "document_type" in filters and document.file_type != filters["document_type"]:
                        continue
                        
                    # Calculate frequency as a simple score
                    content_lower = chunk.content.lower()
                    keyword_lower = keyword.lower()
                    frequency = content_lower.count(keyword_lower)
                    score = min(0.95, 0.5 + (frequency * 0.1))  # Scale to 0.5-0.95
                    
                    # Calculate context window
                    context = get_context_window(chunk)
                    
                    # Check if already in results
                    if any(r["chunk_id"] == chunk.id for r in results):
                        # Update score if better
                        for r in results:
                            if r["chunk_id"] == chunk.id and r["score"] < score:
                                r["score"] = score
                                if keyword != r["matched_term"]:
                                    r["matched_term"] += f", {keyword}"
                    else:
                        # Add to results
                        results.append({
                            "document_id": document.id,
                            "document_title": document.title,
                            "chunk_id": chunk.id,
                            "content": chunk.content,
                            "context": context,
                            "score": score,
                            "matched_term": keyword,
                            "strategy": "keyword"
                        })
        
        # Sort by score and apply limit
        results = sorted(results, key=lambda x: x["score"], reverse=True)[:k]
        
        # Add ranks
        for i, result in enumerate(results):
            result["rank"] = i+1
            
        return {
            "type": "keyword",
            "results": results,
            "count": len(results)
        }
        
    except Exception as e:
        logger.error(f"Error in keyword retrieval: {e}")
        return {"type": "keyword", "results": []}

def perform_entity_retrieval(query_id: int, entities: List[Dict[str, Any]],
                           filters: Dict[str, Any], k: int) -> Dict[str, Any]:
    """
    Perform entity-based retrieval using knowledge graph
    
    Args:
        query_id: ID of the query
        entities: Entities extracted from the query
        filters: Filters to apply to the search
        k: Number of results to retrieve
        
    Returns:
        Dictionary of entity retrieval results
    """
    try:
        if not entities:
            return {"type": "entity", "results": []}
            
        results = []
        entity_ids = []
        
        # Find matching entities in our database
        for entity in entities:
            db_entities = Entity.query.filter(
                Entity.name.ilike(f"%{entity['text']}%"),
                Entity.entity_type == entity["type"] if "type" in entity else True
            ).order_by(Entity.confidence.desc()).limit(3).all()
            
            entity_ids.extend([e.id for e in db_entities])
            
        if not entity_ids:
            return {"type": "entity", "results": []}
            
        # Get entity neighborhoods from knowledge graph
        neighborhoods = knowledge_graph.get_entity_neighborhoods(entity_ids, KNOWLEDGE_GRAPH_DEPTH)
        
        # Get related entities
        related_entity_ids = set()
        if "nodes" in neighborhoods:
            for node in neighborhoods["nodes"]:
                if node["id"].startswith("entity_"):
                    entity_id = int(node["id"].split("_")[1])
                    related_entity_ids.add(entity_id)
        
        # Find chunks containing these entities
        for entity_id in related_entity_ids:
            entity = Entity.query.get(entity_id)
            
            if not entity or not entity.chunk_id:
                continue
                
            chunk = DocumentChunk.query.get(entity.chunk_id)
            
            if not chunk:
                continue
                
            document = Document.query.get(chunk.document_id)
            
            if not document:
                continue
                
            # Apply filters
            if "document_type" in filters and document.file_type != filters["document_type"]:
                continue
                
            # Calculate context window
            context = get_context_window(chunk)
            
            # Calculate score based on entity confidence and relationship proximity
            score = entity.confidence * 0.9  # Scale to max 0.9
            
            # Add to results if not already present
            if not any(r["chunk_id"] == chunk.id for r in results):
                results.append({
                    "document_id": document.id,
                    "document_title": document.title,
                    "chunk_id": chunk.id,
                    "content": chunk.content,
                    "context": context,
                    "score": score,
                    "entity": {
                        "id": entity.id,
                        "name": entity.name,
                        "type": entity.entity_type
                    },
                    "strategy": "entity"
                })
        
        # Sort by score and apply limit
        results = sorted(results, key=lambda x: x["score"], reverse=True)[:k]
        
        # Add ranks
        for i, result in enumerate(results):
            result["rank"] = i+1
            
        return {
            "type": "entity",
            "results": results,
            "count": len(results)
        }
        
    except Exception as e:
        logger.error(f"Error in entity retrieval: {e}")
        return {"type": "entity", "results": []}

def merge_retrieval_results(results: Dict[str, Dict[str, Any]], k: int) -> List[Dict[str, Any]]:
    """
    Merge results from different retrieval strategies
    
    Args:
        results: Dictionary of retrieval results by strategy
        k: Number of results to return
        
    Returns:
        List of merged results
    """
    merged = []
    
    # Define strategy weights
    strategy_weights = {
        "vector": 1.0,
        "keyword": 0.8,
        "entity": 0.9
    }
    
    # Collect all results
    for strategy, result in results.items():
        for item in result.get("results", []):
            # Check if already in merged results
            existing = next((r for r in merged if r["document_id"] == item["document_id"] and r["chunk_id"] == item["chunk_id"]), None)
            
            if existing:
                # Update score if strategy has higher weight * score
                strategy_score = item["score"] * strategy_weights.get(strategy, 0.7)
                existing_strategy = existing.get("strategy", "vector")
                existing_score = existing["score"] * strategy_weights.get(existing_strategy, 0.7)
                
                if strategy_score > existing_score:
                    existing["score"] = item["score"]
                    existing["strategy"] = strategy
                    existing["rank"] = item["rank"]
                    
                # Add strategy to list of matching strategies
                if "matching_strategies" not in existing:
                    existing["matching_strategies"] = [existing["strategy"]]
                if strategy not in existing["matching_strategies"]:
                    existing["matching_strategies"].append(strategy)
            else:
                # Add new result
                item["matching_strategies"] = [strategy]
                merged.append(item)
    
    # Sort by score and apply limit
    merged = sorted(merged, key=lambda x: x["score"], reverse=True)[:k]
    
    # Recalculate ranks
    for i, item in enumerate(merged):
        item["rank"] = i+1
        
    return merged

def filter_by_access_control(results: List[Dict[str, Any]], user_id: int) -> List[Dict[str, Any]]:
    """
    Filter results based on user access permissions
    
    Args:
        results: List of results to filter
        user_id: ID of the user
        
    Returns:
        Filtered results
    """
    filtered = []
    
    for result in results:
        document_id = result["document_id"]
        document = Document.query.get(document_id)
        
        if not document:
            continue
            
        # Check if user has access to document
        has_access = False
        
        # Owner always has access
        if document.owner_id == user_id:
            has_access = True
        # Public documents are accessible to all
        elif document.is_public:
            has_access = True
        # Check permissions
        else:
            from models import Permission
            permission = Permission.query.filter_by(
                user_id=user_id,
                document_id=document_id
            ).first()
            
            if permission and permission.can_read:
                has_access = True
        
        if has_access:
            filtered.append(result)
    
    return filtered

def get_context_window(chunk: DocumentChunk) -> str:
    """
    Get context window for a chunk (includes neighboring chunks)
    
    Args:
        chunk: The document chunk
        
    Returns:
        Context text
    """
    context = chunk.content
    
    # Get previous chunk if available
    prev_chunk = DocumentChunk.query.filter_by(
        document_id=chunk.document_id,
        chunk_index=chunk.chunk_index - 1
    ).first()
    
    # Get next chunk if available
    next_chunk = DocumentChunk.query.filter_by(
        document_id=chunk.document_id,
        chunk_index=chunk.chunk_index + 1
    ).first()
    
    # Add previous and next chunks to context
    if prev_chunk:
        context = prev_chunk.content + "\n\n" + context
        
    if next_chunk:
        context = context + "\n\n" + next_chunk.content
        
    return context

def store_query_results(query_id: int, results: List[Dict[str, Any]]) -> None:
    """
    Store query results in the database
    
    Args:
        query_id: ID of the query
        results: List of results to store
    """
    try:
        # Delete any existing results for this query
        QueryResult.query.filter_by(query_id=query_id).delete()
        
        # Store new results
        for result in results:
            query_result = QueryResult(
                query_id=query_id,
                document_id=result["document_id"],
                chunk_id=result["chunk_id"],
                relevance_score=result["score"],
                rank=result["rank"],
                context_window=result.get("context", "")
            )
            db.session.add(query_result)
            
        db.session.commit()
        
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error storing query results: {e}")

def perform_multi_hop_reasoning(query_id: int, query_text: str, 
                              initial_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Perform multi-hop reasoning across documents
    
    Args:
        query_id: ID of the query
        query_text: Original query text
        initial_results: Initial retrieval results
        
    Returns:
        Enhanced results with reasoning paths
    """
    try:
        # This is a simplified version of multi-hop reasoning
        # A complete implementation would involve more sophisticated traversal strategies
        
        # Get entity IDs from initial results
        entity_ids = set()
        for result in initial_results:
            if "entity" in result and "id" in result["entity"]:
                entity_ids.add(result["entity"]["id"])
                
            # Extract entities from content
            chunk_id = result["chunk_id"]
            chunk_entities = Entity.query.filter_by(chunk_id=chunk_id).all()
            entity_ids.update([e.id for e in chunk_entities])
        
        if not entity_ids:
            return initial_results
            
        # Convert to list
        entity_ids = list(entity_ids)
        
        # Find paths between entities
        paths = []
        for i, source_id in enumerate(entity_ids):
            for target_id in entity_ids[i+1:]:
                entity_paths = knowledge_graph.find_paths_between_entities(
                    source_id, 
                    target_id, 
                    max_hops=KNOWLEDGE_GRAPH_DEPTH
                )
                
                if entity_paths:
                    paths.extend(entity_paths)
        
        # If no paths found, return initial results
        if not paths:
            return initial_results
            
        # Collect chunks containing entities in the paths
        path_entities = set()
        for path in paths:
            for step in path:
                path_entities.add(step["source"]["id"])
                path_entities.add(step["target"]["id"])
        
        # Get chunks containing these entities
        additional_chunks = set()
        for entity_id in path_entities:
            entity = Entity.query.get(entity_id)
            
            if entity and entity.chunk_id:
                additional_chunks.add(entity.chunk_id)
        
        # Get the actual chunks
        additional_results = []
        for chunk_id in additional_chunks:
            # Skip if already in results
            if any(r["chunk_id"] == chunk_id for r in initial_results):
                continue
                
            chunk = DocumentChunk.query.get(chunk_id)
            
            if not chunk:
                continue
                
            document = Document.query.get(chunk.document_id)
            
            if not document:
                continue
                
            # Context window
            context = get_context_window(chunk)
            
            # Add to additional results
            additional_results.append({
                "document_id": document.id,
                "document_title": document.title,
                "chunk_id": chunk.id,
                "content": chunk.content,
                "context": context,
                "score": 0.7,  # Default score for reasoning results
                "strategy": "multi_hop",
                "matching_strategies": ["multi_hop"]
            })
        
        # Add reasoning metadata to results
        enhanced_results = initial_results.copy()
        
        # Add reasoning paths
        for result in enhanced_results:
            result["reasoning_paths"] = []
            
            # Find paths involving this chunk
            chunk_id = result["chunk_id"]
            chunk_entities = Entity.query.filter_by(chunk_id=chunk_id).all()
            chunk_entity_ids = [e.id for e in chunk_entities]
            
            for path in paths:
                # Check if path involves this chunk
                path_entity_ids = []
                for step in path:
                    path_entity_ids.append(step["source"]["id"])
                    path_entity_ids.append(step["target"]["id"])
                    
                if any(e_id in path_entity_ids for e_id in chunk_entity_ids):
                    result["reasoning_paths"].append(path)
        
        # Add additional results and sort
        all_results = enhanced_results + additional_results
        all_results = sorted(all_results, key=lambda x: x["score"], reverse=True)
        
        # Recalculate ranks
        for i, result in enumerate(all_results):
            result["rank"] = i+1
            
        return all_results
        
    except Exception as e:
        logger.error(f"Error in multi-hop reasoning: {e}")
        return initial_results
