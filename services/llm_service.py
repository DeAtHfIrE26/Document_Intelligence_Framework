import logging
import time
import os
import json
from typing import List, Dict, Any, Optional, Tuple
import traceback

from app import app
from config import DEFAULT_LLM_MODEL
from utils.helpers import get_cache, set_cache

logger = logging.getLogger(__name__)

# Initialize OpenAI client
try:
    from openai import OpenAI
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
    client = OpenAI(api_key=OPENAI_API_KEY)
    logger.info("OpenAI client initialized")
except Exception as e:
    logger.error(f"Error initializing OpenAI client: {e}")
    client = None

def init_app(app):
    """Initialize the LLM service with the app context"""
    # Nothing to initialize here since we already set up the client
    pass

def generate_response(query: str, contexts: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Generate a response to a query based on retrieved contexts
    
    Args:
        query: User query
        contexts: List of retrieved document contexts
        
    Returns:
        Dictionary with generated response and metadata
    """
    try:
        if not client:
            return {"error": "LLM service not initialized"}
            
        # Create cache key
        cache_key = f"llm_response_{hash(query)}_{hash(str(contexts))}"
        
        # Check cache
        cached = get_cache(cache_key)
        if cached:
            return cached
            
        # Prepare contexts for the prompt
        context_text = ""
        sources = []
        
        for i, ctx in enumerate(contexts[:5]):  # Limit to top 5 contexts
            document_title = ctx.get("document_title", "Document")
            content = ctx.get("content", "")
            document_id = ctx.get("document_id")
            
            # Add to context text
            context_text += f"\nSOURCE {i+1}: {document_title}\n{content}\n"
            
            # Add to sources
            sources.append({
                "id": document_id,
                "title": document_title,
                "chunk_id": ctx.get("chunk_id")
            })
            
        # Prepare prompt
        system_prompt = (
            "You are an advanced AI assistant integrated with a document retrieval system. "
            "Your task is to provide accurate, helpful responses based on the retrieved contexts. "
            "Always prioritize information from the provided contexts. "
            "If the contexts don't contain relevant information, acknowledge this limitation. "
            "If the contexts contain conflicting information, present multiple viewpoints. "
            "Never make up information or citations. Be transparent about what you know and don't know."
        )
        
        user_prompt = f"QUERY: {query}\n\nCONTEXTS:\n{context_text}\n\nBased on the provided contexts, please answer the query accurately and comprehensively."
        
        # Query LLM
        start_time = time.time()
        response = client.chat.completions.create(
            model=DEFAULT_LLM_MODEL,  # the newest OpenAI model is "gpt-4o" which was released May 13, 2024
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
            max_tokens=1024
        )
        
        # Extract response content
        answer = response.choices[0].message.content
        
        # Create response object
        result = {
            "answer": answer,
            "sources": sources,
            "model": DEFAULT_LLM_MODEL,
            "latency": time.time() - start_time
        }
        
        # Cache result
        set_cache(cache_key, result, 3600)  # Cache for 1 hour
        
        return result
        
    except Exception as e:
        logger.error(f"Error generating LLM response: {e}")
        logger.debug(traceback.format_exc())
        return {"error": str(e)}

def generate_query_alternatives(query: str) -> List[Dict[str, Any]]:
    """
    Generate alternative phrasings for a query
    
    Args:
        query: Original query
        
    Returns:
        List of alternative query phrasings with confidence scores
    """
    try:
        if not client:
            return []
            
        # Create cache key
        cache_key = f"query_alt_{hash(query)}"
        
        # Check cache
        cached = get_cache(cache_key)
        if cached:
            return cached
            
        # Prepare prompt
        system_prompt = (
            "You are an expert in information retrieval and query understanding. "
            "Your task is to generate alternative phrasings for the given query that might yield better search results. "
            "For each alternative, provide a confidence score between 0.5 and 0.9 indicating how confident you are that "
            "this alternative captures the same intent as the original query. "
            "Respond with JSON in this format: [{\"text\": \"alternative query 1\", \"confidence\": 0.8}, ...]"
        )
        
        user_prompt = f"Generate 3 alternative phrasings for this query: \"{query}\""
        
        # Query LLM
        response = client.chat.completions.create(
            model=DEFAULT_LLM_MODEL,  # the newest OpenAI model is "gpt-4o" which was released May 13, 2024
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.5,
            max_tokens=512,
            response_format={"type": "json_object"}
        )
        
        # Extract response content
        try:
            result_text = response.choices[0].message.content
            result = json.loads(result_text)
            alternatives = result if isinstance(result, list) else result.get("alternatives", [])
            
            # Validate structure
            valid_alternatives = []
            for alt in alternatives:
                if "text" in alt and "confidence" in alt:
                    # Ensure confidence is in range
                    alt["confidence"] = max(0.5, min(0.9, alt["confidence"]))
                    valid_alternatives.append(alt)
            
            # Cache result
            set_cache(cache_key, valid_alternatives, 86400)  # Cache for 24 hours
            
            return valid_alternatives
        except json.JSONDecodeError:
            logger.error(f"Error parsing LLM response as JSON: {response.choices[0].message.content}")
            return []
            
    except Exception as e:
        logger.error(f"Error generating query alternatives: {e}")
        logger.debug(traceback.format_exc())
        return []

def detect_query_intent(query: str) -> Optional[Dict[str, Any]]:
    """
    Detect the intent and structure of a query
    
    Args:
        query: User query
        
    Returns:
        Dictionary with intent classification
    """
    try:
        if not client:
            return None
            
        # Create cache key
        cache_key = f"query_intent_{hash(query)}"
        
        # Check cache
        cached = get_cache(cache_key)
        if cached:
            return cached
            
        # Prepare prompt
        system_prompt = (
            "You are an expert in query understanding and intent classification. "
            "Your task is to analyze the given query and determine its structure, intent, and type. "
            "Respond with JSON with these fields: "
            "1. query_type: one of ['factual', 'exploratory', 'navigational', 'transactional'] "
            "2. is_question: boolean "
            "3. intent: brief description of what the user is trying to achieve "
            "4. complexity: one of ['simple', 'moderate', 'complex'] "
            "5. domain: inferred subject domain or field "
            "6. temporal_context: any time-related context in the query"
        )
        
        user_prompt = f"Analyze this query: \"{query}\""
        
        # Query LLM
        response = client.chat.completions.create(
            model=DEFAULT_LLM_MODEL,  # the newest OpenAI model is "gpt-4o" which was released May 13, 2024
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
            max_tokens=512,
            response_format={"type": "json_object"}
        )
        
        # Extract response content
        try:
            result_text = response.choices[0].message.content
            result = json.loads(result_text)
            
            # Cache result
            set_cache(cache_key, result, 86400)  # Cache for 24 hours
            
            return result
        except json.JSONDecodeError:
            logger.error(f"Error parsing LLM response as JSON: {response.choices[0].message.content}")
            return None
            
    except Exception as e:
        logger.error(f"Error detecting query intent: {e}")
        logger.debug(traceback.format_exc())
        return None

def extract_entities_from_query(query: str) -> List[Dict[str, Any]]:
    """
    Extract entities from a query using LLM
    
    Args:
        query: User query
        
    Returns:
        List of extracted entities with types and positions
    """
    try:
        if not client:
            return []
            
        # Create cache key
        cache_key = f"query_entities_{hash(query)}"
        
        # Check cache
        cached = get_cache(cache_key)
        if cached:
            return cached
            
        # Prepare prompt
        system_prompt = (
            "You are an expert in named entity recognition and information extraction. "
            "Your task is to extract entities from the given query. "
            "For each entity, identify its type, and calculate its starting and ending character positions. "
            "Entity types: PERSON, ORGANIZATION, LOCATION, DATE, TIME, PRODUCT, EVENT, WORK_OF_ART, LAW, LANGUAGE, MONEY, QUANTITY, ORDINAL, CARDINAL, PERCENT, FACILITY, GPE. "
            "Respond with JSON array in this format: [{\"text\": \"entity text\", \"type\": \"ENTITY_TYPE\", \"start\": start_position, \"end\": end_position, \"confidence\": confidence_score}, ...]"
        )
        
        user_prompt = f"Extract entities from this query: \"{query}\""
        
        # Query LLM
        response = client.chat.completions.create(
            model=DEFAULT_LLM_MODEL,  # the newest OpenAI model is "gpt-4o" which was released May 13, 2024
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
            max_tokens=512,
            response_format={"type": "json_object"}
        )
        
        # Extract response content
        try:
            result_text = response.choices[0].message.content
            result = json.loads(result_text)
            entities = result if isinstance(result, list) else result.get("entities", [])
            
            # Validate positions
            for entity in entities:
                if "start" in entity and "end" in entity:
                    # Ensure positions are valid
                    start = max(0, min(len(query)-1, entity["start"]))
                    end = max(start+1, min(len(query), entity["end"]))
                    entity["start"] = start
                    entity["end"] = end
                
                # Ensure confidence is present
                if "confidence" not in entity:
                    entity["confidence"] = 0.8
            
            # Cache result
            set_cache(cache_key, entities, 86400)  # Cache for 24 hours
            
            return entities
        except json.JSONDecodeError:
            logger.error(f"Error parsing LLM response as JSON: {response.choices[0].message.content}")
            return []
            
    except Exception as e:
        logger.error(f"Error extracting entities from query: {e}")
        logger.debug(traceback.format_exc())
        return []

def generate_hypothetical_documents(query: str) -> List[Dict[str, Any]]:
    """
    Generate hypothetical documents that would ideally answer the query
    
    Args:
        query: User query
        
    Returns:
        List of hypothetical document descriptions
    """
    try:
        if not client:
            return []
            
        # Create cache key
        cache_key = f"hyp_docs_{hash(query)}"
        
        # Check cache
        cached = get_cache(cache_key)
        if cached:
            return cached
            
        # Prepare prompt
        system_prompt = (
            "You are an expert in information synthesis and document retrieval. "
            "Your task is to imagine what an ideal document that answers the given query would contain. "
            "Create 2-3 hypothetical document summaries that would perfectly address the query. "
            "Include key points, headings, and the types of information these ideal documents would include. "
            "For each document, provide a title, document type, key entities, and main topics. "
            "Respond with JSON in this format: [{\"title\": \"document title\", \"document_type\": \"type\", "
            "\"key_entities\": [\"entity1\", \"entity2\"], \"main_topics\": [\"topic1\", \"topic2\"], \"summary\": \"document summary\"}, ...]"
        )
        
        user_prompt = f"Generate hypothetical document descriptions for this query: \"{query}\""
        
        # Query LLM
        response = client.chat.completions.create(
            model=DEFAULT_LLM_MODEL,  # the newest OpenAI model is "gpt-4o" which was released May 13, 2024
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7,
            max_tokens=1024,
            response_format={"type": "json_object"}
        )
        
        # Extract response content
        try:
            result_text = response.choices[0].message.content
            result = json.loads(result_text)
            documents = result if isinstance(result, list) else result.get("documents", [])
            
            # Cache result
            set_cache(cache_key, documents, 86400)  # Cache for 24 hours
            
            return documents
        except json.JSONDecodeError:
            logger.error(f"Error parsing LLM response as JSON: {response.choices[0].message.content}")
            return []
            
    except Exception as e:
        logger.error(f"Error generating hypothetical documents: {e}")
        logger.debug(traceback.format_exc())
        return []

def disambiguate_query(query: str, 
                      possible_interpretations: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Disambiguate between multiple possible query interpretations
    
    Args:
        query: Original query
        possible_interpretations: List of possible interpretations
        
    Returns:
        Dictionary with chosen interpretation and explanation
    """
    try:
        if not client or not possible_interpretations:
            return {"chosen_interpretation": possible_interpretations[0] if possible_interpretations else None}
            
        # Prepare prompt
        system_prompt = (
            "You are an expert in query understanding and disambiguation. "
            "Your task is to analyze a potentially ambiguous query and several possible interpretations. "
            "Choose the most likely interpretation based on common usage patterns, logic, and context. "
            "Explain your reasoning for choosing this interpretation. "
            "Respond with JSON with these fields: "
            "1. chosen_interpretation_index: index of the chosen interpretation (0-based) "
            "2. explanation: explanation of why this interpretation was chosen "
            "3. confidence: confidence score between 0 and 1"
        )
        
        # Format interpretations
        interpretations_text = ""
        for i, interp in enumerate(possible_interpretations):
            interpretations_text += f"Interpretation {i}: {interp['text']}\n"
            
        user_prompt = f"Query: \"{query}\"\n\nPossible Interpretations:\n{interpretations_text}\n\nChoose the most likely interpretation."
        
        # Query LLM
        response = client.chat.completions.create(
            model=DEFAULT_LLM_MODEL,  # the newest OpenAI model is "gpt-4o" which was released May 13, 2024
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
            max_tokens=512,
            response_format={"type": "json_object"}
        )
        
        # Extract response content
        try:
            result_text = response.choices[0].message.content
            result = json.loads(result_text)
            
            # Get chosen interpretation
            chosen_index = result.get("chosen_interpretation_index", 0)
            chosen_index = max(0, min(len(possible_interpretations)-1, chosen_index))
            chosen = possible_interpretations[chosen_index]
            
            return {
                "chosen_interpretation": chosen,
                "explanation": result.get("explanation", ""),
                "confidence": result.get("confidence", 0.8)
            }
        except json.JSONDecodeError:
            logger.error(f"Error parsing LLM response as JSON: {response.choices[0].message.content}")
            return {"chosen_interpretation": possible_interpretations[0]}
            
    except Exception as e:
        logger.error(f"Error disambiguating query: {e}")
        logger.debug(traceback.format_exc())
        return {"chosen_interpretation": possible_interpretations[0] if possible_interpretations else None}

def rerank_results(query: str, results: List[Dict[str, Any]]) -> List[int]:
    """
    Use LLM to rerank search results
    
    Args:
        query: Original query
        results: Search results with text snippets
        
    Returns:
        List of result IDs in order of relevance
    """
    try:
        if not client or not results:
            return [r["id"] for r in results]
            
        # Prepare prompt
        system_prompt = (
            "You are an expert in information retrieval and relevance assessment. "
            "Your task is to rerank search results based on their relevance to the query. "
            "For each result, evaluate how well it answers the query, considering factors like: "
            "- Direct answer to the query question "
            "- Comprehensive coverage of the query topic "
            "- Authoritative and accurate information "
            "- Recency and relevance of information "
            "Return a list of result IDs in descending order of relevance (most relevant first). "
            "Respond with JSON with a single field: \"reranked_ids\": [id1, id2, ...]"
        )
        
        # Format results
        results_text = ""
        for result in results:
            results_text += f"Result ID {result['id']}:\n{result['text']}\n\n"
            
        user_prompt = f"Query: \"{query}\"\n\nSearch Results:\n{results_text}\n\nRerank these results by relevance to the query."
        
        # Query LLM
        response = client.chat.completions.create(
            model=DEFAULT_LLM_MODEL,  # the newest OpenAI model is "gpt-4o" which was released May 13, 2024
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
            max_tokens=512,
            response_format={"type": "json_object"}
        )
        
        # Extract response content
        try:
            result_text = response.choices[0].message.content
            result = json.loads(result_text)
            
            # Get reranked IDs
            reranked_ids = result.get("reranked_ids", [])
            
            # Validate IDs
            valid_ids = [r["id"] for r in results]
            validated_ids = [id for id in reranked_ids if id in valid_ids]
            
            # Add any missing IDs at the end
            missing_ids = [id for id in valid_ids if id not in validated_ids]
            validated_ids.extend(missing_ids)
            
            return validated_ids
        except json.JSONDecodeError:
            logger.error(f"Error parsing LLM response as JSON: {response.choices[0].message.content}")
            return [r["id"] for r in results]
            
    except Exception as e:
        logger.error(f"Error reranking results: {e}")
        logger.debug(traceback.format_exc())
        return [r["id"] for r in results]

def extract_relationships(text: str) -> List[Dict[str, Any]]:
    """
    Extract relationships between entities in text
    
    Args:
        text: Text to analyze
        
    Returns:
        List of relationship dictionaries
    """
    try:
        if not client:
            return []
            
        # Create cache key
        cache_key = f"extract_rel_{hash(text[:1000])}"  # Limit text length for hash
        
        # Check cache
        cached = get_cache(cache_key)
        if cached:
            return cached
            
        # Prepare prompt
        system_prompt = (
            "You are an expert in relationship extraction from text. "
            "Your task is to identify entities in the text and extract relationships between them. "
            "For each relationship, identify the source entity, target entity, and relationship type. "
            "Common relationship types: works_for, located_in, part_of, founder_of, created_by, spouse_of, etc. "
            "Respond with JSON array in this format: [{\"source\": {\"text\": \"source entity\", \"type\": \"ENTITY_TYPE\"}, "
            "\"target\": {\"text\": \"target entity\", \"type\": \"ENTITY_TYPE\"}, \"relationship\": \"relationship_type\", \"confidence\": confidence_score}, ...]"
        )
        
        # Limit text length for API
        limited_text = text[:4000] if len(text) > 4000 else text
        user_prompt = f"Extract relationships from this text:\n\n{limited_text}"
        
        # Query LLM
        response = client.chat.completions.create(
            model=DEFAULT_LLM_MODEL,  # the newest OpenAI model is "gpt-4o" which was released May 13, 2024
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
            max_tokens=1024,
            response_format={"type": "json_object"}
        )
        
        # Extract response content
        try:
            result_text = response.choices[0].message.content
            result = json.loads(result_text)
            relationships = result if isinstance(result, list) else result.get("relationships", [])
            
            # Cache result
            set_cache(cache_key, relationships, 86400)  # Cache for 24 hours
            
            return relationships
        except json.JSONDecodeError:
            logger.error(f"Error parsing LLM response as JSON: {response.choices[0].message.content}")
            return []
            
    except Exception as e:
        logger.error(f"Error extracting relationships: {e}")
        logger.debug(traceback.format_exc())
        return []
