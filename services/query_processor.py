import logging
import re
import time
import hashlib
import json
from typing import List, Dict, Any, Optional, Tuple, Set
from concurrent.futures import ThreadPoolExecutor

from app import app, db
from models import Query, Entity, Document
from config import (
    MAX_QUERY_LENGTH, QUERY_EXPANSION_TECHNIQUES, TOP_K_RETRIEVAL,
    KNOWLEDGE_GRAPH_DEPTH, MAX_THREADS, AMBIGUOUS_QUERY_HANDLING,
    QUERY_CACHE_TTL, CACHE_TYPE
)
from services import embedding_service, knowledge_graph, vector_store, llm_service
from utils.preprocessing import clean_text, normalize_text
from utils.helpers import get_cache, set_cache

logger = logging.getLogger(__name__)

def process_query(query_text: str, user_id: Optional[int] = None, 
                filters: Dict[str, Any] = None) -> Tuple[Query, Dict[str, Any]]:
    """
    Process a search query and prepare it for retrieval
    
    Args:
        query_text: Original query text
        user_id: ID of the user making the query (optional)
        filters: Dictionary of search filters (optional)
        
    Returns:
        Query object and processed query data
    """
    try:
        # Clean and validate query
        query_text = clean_text(query_text)
        if not query_text or len(query_text) > MAX_QUERY_LENGTH:
            raise ValueError(f"Query must be between 1 and {MAX_QUERY_LENGTH} characters")
            
        # Generate cache key
        cache_key = generate_query_cache_key(query_text, user_id, filters)
        
        # Check cache
        if CACHE_TYPE:
            cached_response = get_cache(cache_key)
            if cached_response:
                logger.debug(f"Cache hit for query: {query_text}")
                # Still store the query in DB for analytics
                query = store_query(query_text, user_id)
                return query, cached_response
                
        # Create and store query
        query = store_query(query_text, user_id)
        
        # Process query in multiple ways
        processed_data = {
            "original_query": query_text,
            "processed_query": query_text,
            "intent": detect_intent(query_text),
            "expanded_queries": [],
            "entities": [],
            "query_embedding": None,
            "query_id": query.id,
            "search_type": "hybrid",
            "filters": filters or {}
        }
        
        # Extract entities
        processed_data["entities"] = extract_entities_from_query(query_text)
        
        # Expand query using selected techniques
        processed_data["expanded_queries"] = expand_query(
            query_text, 
            processed_data["entities"], 
            QUERY_EXPANSION_TECHNIQUES
        )
        
        # Generate embedding for the query
        embedding_id = embedding_service.embed_query(query_text, query.id)
        processed_data["query_embedding"] = embedding_id
        
        # Update query record with processed data
        query.processed_query = json.dumps({
            "expanded_queries": processed_data["expanded_queries"],
            "entities": processed_data["entities"],
            "intent": processed_data["intent"]
        })
        db.session.commit()
        
        # Cache the processed query
        if CACHE_TYPE:
            set_cache(cache_key, processed_data, QUERY_CACHE_TTL)
            
        return query, processed_data
        
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        # Create a minimal query object for error cases
        query = Query(
            original_query=query_text[:MAX_QUERY_LENGTH],
            user_id=user_id
        )
        db.session.add(query)
        db.session.commit()
        
        return query, {"error": str(e), "query_id": query.id}

def store_query(query_text: str, user_id: Optional[int] = None) -> Query:
    """Store query in the database"""
    query = Query(
        original_query=query_text,
        user_id=user_id,
        created_at=time.time()
    )
    db.session.add(query)
    db.session.flush()  # Get ID without committing
    return query

def detect_intent(query_text: str) -> Dict[str, Any]:
    """
    Detect the intent of a query
    
    Args:
        query_text: Query text
        
    Returns:
        Dictionary with intent information
    """
    # Detect if query is a question
    is_question = bool(re.search(r'\?|what|how|who|when|where|why|which|whose|whom', query_text.lower()))
    
    # Check for command patterns
    command_patterns = {
        "find": r'\bfind\b|\bsearch\b|\blook\s+for\b',
        "show": r'\bshow\b|\bdisplay\b|\bvisualize\b',
        "compare": r'\bcompare\b|\bdifference\b|\bvs\b|\bversus\b',
        "summarize": r'\bsummarize\b|\bsummary\b|\bbrief\b',
        "analyze": r'\banalyze\b|\banalysis\b|\bexamine\b',
        "list": r'\blist\b|\ball\b|\bevery\b'
    }
    
    command_type = None
    for cmd, pattern in command_patterns.items():
        if re.search(pattern, query_text.lower()):
            command_type = cmd
            break
    
    # Determine specificity
    specificity = "high" if len(query_text.split()) > 5 else "low"
    
    # Use LLM for more sophisticated intent detection if available
    llm_intent = None
    try:
        llm_intent = llm_service.detect_query_intent(query_text)
    except:
        pass
    
    return {
        "is_question": is_question,
        "command_type": command_type,
        "specificity": specificity,
        "llm_intent": llm_intent
    }

def extract_entities_from_query(query_text: str) -> List[Dict[str, Any]]:
    """
    Extract entities from the query
    
    Args:
        query_text: Query text
        
    Returns:
        List of extracted entities
    """
    entities = []
    
    # Simple rule-based extraction for common entities
    # In a real system, this would use a more sophisticated NER model
    
    # Look for quoted phrases
    quoted = re.findall(r'"([^"]*)"', query_text)
    for q in quoted:
        entities.append({
            "text": q,
            "type": "QUOTED_PHRASE",
            "start": query_text.find(f'"{q}"'),
            "end": query_text.find(f'"{q}"') + len(q) + 2,
            "confidence": 0.9
        })
    
    # Try using spaCy if available
    try:
        from services.document_processor import nlp
        if nlp:
            doc = nlp(query_text)
            for ent in doc.ents:
                # Skip if overlaps with quoted phrase
                if any(e["start"] <= ent.start_char and e["end"] >= ent.end_char for e in entities):
                    continue
                    
                entities.append({
                    "text": ent.text,
                    "type": ent.label_,
                    "start": ent.start_char,
                    "end": ent.end_char,
                    "confidence": 0.85  # Placeholder
                })
    except Exception as e:
        logger.warning(f"Error using spaCy for entity extraction: {e}")
    
    # Use LLM for more sophisticated entity extraction if available
    try:
        llm_entities = llm_service.extract_entities_from_query(query_text)
        if llm_entities:
            # Merge with existing entities, avoiding duplicates
            existing_texts = {e["text"].lower() for e in entities}
            for llm_entity in llm_entities:
                if llm_entity["text"].lower() not in existing_texts:
                    entities.append(llm_entity)
    except Exception as e:
        logger.warning(f"Error using LLM for entity extraction: {e}")
    
    return entities

def expand_query(query_text: str, entities: List[Dict[str, Any]], 
                techniques: List[str]) -> List[Dict[str, Any]]:
    """
    Expand query using various techniques
    
    Args:
        query_text: Original query
        entities: Extracted entities
        techniques: List of expansion techniques to use
        
    Returns:
        List of expanded queries
    """
    expanded_queries = []
    
    # Process with different techniques
    with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
        futures = []
        
        if "synonym" in techniques:
            futures.append(executor.submit(expand_with_synonyms, query_text, entities))
            
        if "entity" in techniques:
            futures.append(executor.submit(expand_with_entities, query_text, entities))
            
        if "llm" in techniques:
            futures.append(executor.submit(expand_with_llm, query_text))
            
        # Collect results
        for future in futures:
            try:
                result = future.result()
                if result:
                    expanded_queries.extend(result)
            except Exception as e:
                logger.warning(f"Error in query expansion: {e}")
    
    # Deduplicate
    seen = set()
    unique_expansions = []
    for exp in expanded_queries:
        if exp["text"] not in seen:
            seen.add(exp["text"])
            unique_expansions.append(exp)
    
    return unique_expansions

def expand_with_synonyms(query_text: str, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Expand query with synonyms for key terms
    
    Args:
        query_text: Original query
        entities: Extracted entities
        
    Returns:
        List of expanded queries using synonyms
    """
    expanded = []
    
    # Use WordNet from NLTK for synonyms
    try:
        from nltk.corpus import wordnet
        import nltk
        
        # Ensure we have the required data
        try:
            wordnet.all_synsets()
        except LookupError:
            nltk.download('wordnet')
        
        # Get POS tags
        try:
            nltk.pos_tag(['test'])
        except LookupError:
            nltk.download('averaged_perceptron_tagger')
        
        # Convert NLTK POS tags to WordNet POS tags
        def get_wordnet_pos(nltk_tag):
            if nltk_tag.startswith('J'):
                return wordnet.ADJ
            elif nltk_tag.startswith('V'):
                return wordnet.VERB
            elif nltk_tag.startswith('N'):
                return wordnet.NOUN
            elif nltk_tag.startswith('R'):
                return wordnet.ADV
            else:
                return None
        
        # Tokenize and get POS tags
        tokens = nltk.word_tokenize(query_text)
        pos_tags = nltk.pos_tag(tokens)
        
        # Skip entities and common words
        skip_words = set()
        for entity in entities:
            skip_words.update(entity["text"].lower().split())
        
        stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "with", "by"}
        
        # Find synonyms for important words
        for word, tag in pos_tags:
            # Skip if in entity or stop word
            if word.lower() in skip_words or word.lower() in stop_words:
                continue
                
            # Get WordNet POS tag
            wordnet_pos = get_wordnet_pos(tag)
            if not wordnet_pos:
                continue
                
            # Get synonyms
            synsets = wordnet.synsets(word, pos=wordnet_pos)
            for synset in synsets[:2]:  # Limit to 2 synsets per word
                for synonym in synset.lemma_names()[:3]:  # Limit to 3 synonyms per synset
                    # Skip if same as original or multi-word
                    if synonym.lower() == word.lower() or '_' in synonym:
                        continue
                        
                    # Create expanded query by replacing this word
                    expanded_text = query_text.replace(word, synonym)
                    if expanded_text != query_text:
                        expanded.append({
                            "text": expanded_text,
                            "technique": "synonym",
                            "confidence": 0.7
                        })
    
    except Exception as e:
        logger.warning(f"Error in synonym expansion: {e}")
    
    return expanded

def expand_with_entities(query_text: str, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Expand query using knowledge graph entities
    
    Args:
        query_text: Original query
        entities: Extracted entities
        
    Returns:
        List of expanded queries using related entities
    """
    expanded = []
    
    try:
        # Find related entities in knowledge graph
        for entity in entities:
            # Find entity in our database
            db_entities = Entity.query.filter(
                Entity.name.ilike(f"%{entity['text']}%"),
                Entity.entity_type == entity["type"] if "type" in entity else True
            ).limit(5).all()
            
            for db_entity in db_entities:
                # Get connections from knowledge graph
                connections = knowledge_graph.get_entity_connections(db_entity.id)
                
                for conn in connections.get("connections", [])[:3]:  # Limit to 3 connections
                    # Create expanded query by adding connected entity
                    if conn["confidence"] >= 0.7:
                        expanded_text = f"{query_text} {conn['name']}"
                        expanded.append({
                            "text": expanded_text,
                            "technique": "entity_expansion",
                            "confidence": conn["confidence"],
                            "entity": {
                                "id": conn["entity_id"],
                                "name": conn["name"],
                                "type": conn["entity_type"],
                                "relationship": conn["relationship"]
                            }
                        })
    
    except Exception as e:
        logger.warning(f"Error in entity expansion: {e}")
    
    return expanded

def expand_with_llm(query_text: str) -> List[Dict[str, Any]]:
    """
    Expand query using LLM-generated alternatives
    
    Args:
        query_text: Original query
        
    Returns:
        List of expanded queries generated by LLM
    """
    expanded = []
    
    try:
        # Use LLM service to generate alternative queries
        alternatives = llm_service.generate_query_alternatives(query_text)
        
        if alternatives:
            for alt in alternatives:
                expanded.append({
                    "text": alt["text"],
                    "technique": "llm_expansion",
                    "confidence": alt["confidence"]
                })
    
    except Exception as e:
        logger.warning(f"Error in LLM expansion: {e}")
    
    return expanded

def handle_ambiguous_query(query_text: str, 
                          possible_interpretations: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Handle ambiguous queries by generating possible interpretations
    
    Args:
        query_text: Original query
        possible_interpretations: List of possible interpretations
        
    Returns:
        Dictionary with disambiguation information
    """
    try:
        if not AMBIGUOUS_QUERY_HANDLING:
            # Just return the first interpretation if ambiguity handling is disabled
            return {"chosen_interpretation": possible_interpretations[0]}
            
        # Use LLM to disambiguate
        disambiguation = llm_service.disambiguate_query(query_text, possible_interpretations)
        
        return {
            "is_ambiguous": True,
            "possible_interpretations": possible_interpretations,
            "chosen_interpretation": disambiguation["chosen_interpretation"],
            "explanation": disambiguation["explanation"]
        }
        
    except Exception as e:
        logger.warning(f"Error handling ambiguous query: {e}")
        # Fall back to first interpretation
        return {"chosen_interpretation": possible_interpretations[0]}

def generate_query_cache_key(query_text: str, user_id: Optional[int] = None, 
                            filters: Optional[Dict[str, Any]] = None) -> str:
    """Generate a unique cache key for a query"""
    key_parts = [query_text]
    
    if user_id:
        key_parts.append(f"user_{user_id}")
        
    if filters:
        # Sort filter keys for consistency
        sorted_filters = sorted([(k, v) for k, v in filters.items()])
        filter_str = json.dumps(sorted_filters)
        key_parts.append(filter_str)
        
    key_str = "_".join(key_parts)
    return f"query_{hashlib.md5(key_str.encode()).hexdigest()}"

def get_hypothetical_documents(query: str) -> List[Dict[str, Any]]:
    """
    Generate hypothetical documents that would ideally answer the query
    
    Args:
        query: The search query
        
    Returns:
        List of hypothetical document descriptions
    """
    try:
        # Use LLM to generate hypothetical documents
        hypothetical_docs = llm_service.generate_hypothetical_documents(query)
        return hypothetical_docs
    except Exception as e:
        logger.warning(f"Error generating hypothetical documents: {e}")
        return []

def convert_query_to_vector_filters(filters: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert query filters to vector store filters
    
    Args:
        filters: Original query filters
        
    Returns:
        Filters formatted for vector store
    """
    vector_filters = {}
    
    if not filters:
        return vector_filters
        
    # Map filter keys to metadata keys
    if "document_type" in filters:
        vector_filters["file_type"] = filters["document_type"]
        
    if "author" in filters:
        vector_filters["author"] = filters["author"]
        
    if "date_range" in filters and "start_date" in filters["date_range"]:
        # Vector store would need range query support
        pass
        
    if "collection_id" in filters:
        vector_filters["collection_id"] = filters["collection_id"]
        
    if "tag_ids" in filters:
        # Handle tags differently as they're many-to-many
        pass
    
    return vector_filters
