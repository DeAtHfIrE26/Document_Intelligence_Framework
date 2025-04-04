import logging
from typing import List, Dict, Any, Optional, Set, Tuple
import networkx as nx
import json
import numpy as np
from concurrent.futures import ThreadPoolExecutor

from app import app, db
from models import Document, DocumentChunk, Entity, EntityRelationship
from config import (
    RELATION_CONFIDENCE_THRESHOLD, KNOWLEDGE_GRAPH_DEPTH,
    MAX_THREADS, ENTITY_CONFIDENCE_THRESHOLD
)
from services import embedding_service

logger = logging.getLogger(__name__)

# Global graph object 
graph = None

def init_app(app):
    """Initialize the knowledge graph service with the app context"""
    global graph
    graph = nx.DiGraph()
    
    # Load existing relationships from database
    with app.app_context():
        try:
            load_knowledge_graph()
            logger.info("Knowledge graph initialized")
        except Exception as e:
            logger.error(f"Error initializing knowledge graph: {e}")
            graph = nx.DiGraph()  # Create empty graph on failure

def load_knowledge_graph():
    """Load the knowledge graph from the database"""
    global graph
    graph = nx.DiGraph()
    
    # Load all entities
    entities = Entity.query.all()
    for entity in entities:
        node_id = f"entity_{entity.id}"
        graph.add_node(
            node_id,
            id=entity.id,
            type="entity",
            name=entity.name,
            entity_type=entity.entity_type,
            document_id=entity.document_id,
            embedding_id=entity.embedding_id
        )
    
    # Load all relationships
    relationships = EntityRelationship.query.all()
    for rel in relationships:
        source_id = f"entity_{rel.source_id}"
        target_id = f"entity_{rel.target_id}"
        
        # Skip if nodes don't exist
        if source_id not in graph or target_id not in graph:
            continue
            
        graph.add_edge(
            source_id,
            target_id,
            id=rel.id,
            type=rel.relationship_type,
            confidence=rel.confidence,
            metadata=rel.metadata
        )
    
    logger.info(f"Loaded knowledge graph with {len(graph.nodes)} nodes and {len(graph.edges)} edges")

def build_knowledge_graph_for_document(document_id: int) -> None:
    """
    Build or update the knowledge graph for a specific document
    
    Args:
        document_id: ID of the document
    """
    try:
        # Get all entities for this document
        entities = Entity.query.filter_by(document_id=document_id).all()
        logger.info(f"Building knowledge graph for document {document_id} with {len(entities)} entities")
        
        # Create entity nodes in the graph
        for entity in entities:
            node_id = f"entity_{entity.id}"
            
            # Add to graph if not already present
            if node_id not in graph:
                graph.add_node(
                    node_id,
                    id=entity.id,
                    type="entity",
                    name=entity.name,
                    entity_type=entity.entity_type,
                    document_id=entity.document_id,
                    embedding_id=entity.embedding_id
                )
            
            # Create entity embedding if not already created
            if not entity.embedding_id:
                embedding_service.embed_entity(entity.id)
        
        # Find relationships between entities
        discover_relationships(document_id, entities)
        
        # Find cross-document relationships
        discover_cross_document_relationships(document_id, entities)
        
        logger.info(f"Knowledge graph updated for document {document_id}")
        
    except Exception as e:
        logger.error(f"Error building knowledge graph for document {document_id}: {e}")

def discover_relationships(document_id: int, entities: List[Entity]) -> None:
    """
    Discover relationships between entities within the same document
    
    Args:
        document_id: ID of the document
        entities: List of entities in the document
    """
    try:
        # Group entities by chunk
        chunk_entities: Dict[int, List[Entity]] = {}
        for entity in entities:
            if entity.chunk_id:
                if entity.chunk_id not in chunk_entities:
                    chunk_entities[entity.chunk_id] = []
                chunk_entities[entity.chunk_id].append(entity)
        
        # Process each chunk's entities to find relationships
        for chunk_id, chunk_ents in chunk_entities.items():
            # Skip chunks with only one entity
            if len(chunk_ents) < 2:
                continue
                
            # Get the document chunk
            chunk = DocumentChunk.query.get(chunk_id)
            if not chunk:
                continue
                
            # Find relationships between entities in the same chunk
            discover_chunk_relationships(chunk, chunk_ents)
            
        # Find relationships across chunks but within the same document
        discover_document_level_relationships(document_id, entities)
        
    except Exception as e:
        logger.error(f"Error discovering relationships in document {document_id}: {e}")

def discover_chunk_relationships(chunk: DocumentChunk, entities: List[Entity]) -> None:
    """
    Discover relationships between entities within the same chunk
    
    Args:
        chunk: The document chunk
        entities: List of entities in the chunk
    """
    try:
        # Sort entities by position in text
        sorted_entities = sorted(entities, key=lambda e: e.start_char)
        
        # Look for co-occurring entities
        for i, source_ent in enumerate(sorted_entities):
            for j in range(i+1, len(sorted_entities)):
                target_ent = sorted_entities[j]
                
                # Skip self-relationships
                if source_ent.id == target_ent.id:
                    continue
                    
                # Determine relationship type based on entity types and proximity
                relationship_type = infer_relationship_type(source_ent, target_ent, chunk.content)
                if not relationship_type:
                    continue
                    
                # Calculate confidence based on proximity
                char_distance = target_ent.start_char - source_ent.end_char
                max_distance = 100  # Maximum distance to consider for co-occurrence
                confidence = max(0.0, 1.0 - (char_distance / max_distance)) if char_distance < max_distance else 0.0
                
                # Skip low confidence relationships
                if confidence < RELATION_CONFIDENCE_THRESHOLD:
                    continue
                    
                # Create or update relationship
                create_or_update_relationship(source_ent.id, target_ent.id, relationship_type, confidence)
                
    except Exception as e:
        logger.error(f"Error discovering chunk relationships: {e}")

def discover_document_level_relationships(document_id: int, entities: List[Entity]) -> None:
    """
    Discover relationships between entities across different chunks but in the same document
    
    Args:
        document_id: ID of the document
        entities: List of entities in the document
    """
    try:
        # Group entities by type
        entities_by_type: Dict[str, List[Entity]] = {}
        for entity in entities:
            if entity.entity_type not in entities_by_type:
                entities_by_type[entity.entity_type] = []
            entities_by_type[entity.entity_type].append(entity)
        
        # For each entity type, find relationships with other types
        for source_type, source_entities in entities_by_type.items():
            for target_type, target_entities in entities_by_type.items():
                # Skip if same type and not enough entities
                if source_type == target_type or len(source_entities) < 1 or len(target_entities) < 1:
                    continue
                    
                # Find relationships between entities of different types
                relationship_type = infer_type_relationship(source_type, target_type)
                if not relationship_type:
                    continue
                    
                # Compare embeddings of entities to find semantic relationships
                # Use ThreadPoolExecutor for parallelism
                with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
                    for source_ent in source_entities:
                        for target_ent in target_entities:
                            # Skip self-relationships
                            if source_ent.id == target_ent.id:
                                continue
                                
                            executor.submit(
                                compare_entity_embeddings,
                                source_ent,
                                target_ent,
                                relationship_type
                            )
                            
    except Exception as e:
        logger.error(f"Error discovering document-level relationships: {e}")

def compare_entity_embeddings(source_entity: Entity, target_entity: Entity, relationship_type: str) -> None:
    """
    Compare entity embeddings to find semantic relationships
    
    Args:
        source_entity: Source entity
        target_entity: Target entity
        relationship_type: Type of relationship
    """
    try:
        # Skip if either entity doesn't have an embedding
        if not source_entity.embedding_id or not target_entity.embedding_id:
            return
            
        # Get embeddings from vector store
        from services import vector_store
        source_embedding = vector_store.get_embedding_by_id(source_entity.embedding_id)
        target_embedding = vector_store.get_embedding_by_id(target_entity.embedding_id)
        
        if source_embedding is None or target_embedding is None:
            return
            
        # Calculate similarity
        similarity = embedding_service.get_embedding_similarity(source_embedding, target_embedding)
        
        # Use similarity as confidence score
        confidence = float(similarity)
        
        # Create relationship if above threshold
        if confidence >= RELATION_CONFIDENCE_THRESHOLD:
            create_or_update_relationship(source_entity.id, target_entity.id, relationship_type, confidence)
            
    except Exception as e:
        logger.error(f"Error comparing entity embeddings: {e}")

def discover_cross_document_relationships(document_id: int, new_entities: List[Entity]) -> None:
    """
    Discover relationships between entities in different documents
    
    Args:
        document_id: ID of the document being processed
        new_entities: List of entities in the document
    """
    try:
        # Get entities from other documents
        other_entities = Entity.query.filter(
            Entity.document_id != document_id,
            Entity.confidence >= ENTITY_CONFIDENCE_THRESHOLD
        ).all()
        
        logger.info(f"Discovering cross-document relationships between {len(new_entities)} new entities and {len(other_entities)} existing entities")
        
        # Skip if no other entities
        if not other_entities:
            return
            
        # Find exact matches (same name and type)
        for new_ent in new_entities:
            matches = [
                ent for ent in other_entities 
                if ent.name.lower() == new_ent.name.lower() and
                ent.entity_type == new_ent.entity_type
            ]
            
            # Create "same_as" relationships for exact matches
            for match in matches:
                create_or_update_relationship(
                    new_ent.id, match.id, "same_as", 0.95,
                    {"match_type": "exact"}
                )
                
        # Find semantic matches using embeddings
        # Limit to a reasonable number of comparisons to avoid performance issues
        max_comparisons = 1000
        if len(new_entities) * len(other_entities) > max_comparisons:
            # Sample a subset of entities to compare
            sampled_new = new_entities[:min(100, len(new_entities))]
            sampled_other = other_entities[:min(max_comparisons // len(sampled_new), len(other_entities))]
        else:
            sampled_new = new_entities
            sampled_other = other_entities
        
        # Compare embeddings in parallel
        with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
            for new_ent in sampled_new:
                for other_ent in sampled_other:
                    # Skip if same entity or exact match already found
                    if new_ent.name.lower() == other_ent.name.lower() and new_ent.entity_type == other_ent.entity_type:
                        continue
                        
                    executor.submit(
                        compare_cross_document_entities,
                        new_ent,
                        other_ent
                    )
                    
    except Exception as e:
        logger.error(f"Error discovering cross-document relationships: {e}")

def compare_cross_document_entities(entity1: Entity, entity2: Entity) -> None:
    """
    Compare entities from different documents to find semantic relationships
    
    Args:
        entity1: First entity
        entity2: Second entity
    """
    try:
        # Skip if either entity doesn't have an embedding
        if not entity1.embedding_id or not entity2.embedding_id:
            return
            
        # Get embeddings from vector store
        from services import vector_store
        embedding1 = vector_store.get_embedding_by_id(entity1.embedding_id)
        embedding2 = vector_store.get_embedding_by_id(entity2.embedding_id)
        
        if embedding1 is None or embedding2 is None:
            return
            
        # Calculate similarity
        similarity = embedding_service.get_embedding_similarity(embedding1, embedding2)
        
        # High threshold for semantic matches across documents
        if similarity >= 0.85:
            # Create "related_to" relationship
            create_or_update_relationship(
                entity1.id, entity2.id, "related_to", float(similarity),
                {"match_type": "semantic", "cross_document": True}
            )
            
    except Exception as e:
        logger.error(f"Error comparing cross-document entities: {e}")

def create_or_update_relationship(source_id: int, target_id: int, relationship_type: str, 
                                 confidence: float, metadata: Dict[str, Any] = None) -> Optional[EntityRelationship]:
    """
    Create or update a relationship between entities
    
    Args:
        source_id: ID of the source entity
        target_id: ID of the target entity
        relationship_type: Type of relationship
        confidence: Confidence score (0-1)
        metadata: Optional metadata for the relationship
        
    Returns:
        Created or updated EntityRelationship or None if error
    """
    try:
        # Check if relationship already exists
        existing = EntityRelationship.query.filter_by(
            source_id=source_id,
            target_id=target_id,
            relationship_type=relationship_type
        ).first()
        
        if existing:
            # Update confidence if new confidence is higher
            if confidence > existing.confidence:
                existing.confidence = confidence
                if metadata:
                    if existing.metadata:
                        existing.metadata.update(metadata)
                    else:
                        existing.metadata = metadata
                db.session.commit()
            return existing
        else:
            # Create new relationship
            relationship = EntityRelationship(
                source_id=source_id,
                target_id=target_id,
                relationship_type=relationship_type,
                confidence=confidence,
                metadata=metadata
            )
            db.session.add(relationship)
            
            # Add to graph
            source_node_id = f"entity_{source_id}"
            target_node_id = f"entity_{target_id}"
            graph.add_edge(
                source_node_id,
                target_node_id,
                type=relationship_type,
                confidence=confidence,
                metadata=metadata
            )
            
            db.session.commit()
            return relationship
            
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error creating relationship: {e}")
        return None

def infer_relationship_type(source_entity: Entity, target_entity: Entity, context: str = "") -> Optional[str]:
    """
    Infer relationship type between entities based on their types and context
    
    Args:
        source_entity: Source entity
        target_entity: Target entity
        context: Context text (optional)
        
    Returns:
        Inferred relationship type or None
    """
    source_type = source_entity.entity_type
    target_type = target_entity.entity_type
    
    # Define common relationship mappings based on entity types
    type_mappings = {
        ("PERSON", "PERSON"): "associated_with",
        ("PERSON", "ORG"): "affiliated_with",
        ("ORG", "PERSON"): "employs",
        ("PERSON", "GPE"): "located_in",
        ("ORG", "GPE"): "based_in",
        ("ORG", "ORG"): "related_to",
        ("WORK_OF_ART", "PERSON"): "created_by",
        ("PERSON", "WORK_OF_ART"): "created",
        ("PRODUCT", "ORG"): "produced_by",
        ("ORG", "PRODUCT"): "produces",
        ("GPE", "GPE"): "neighbor_of",
        ("DATE", "EVENT"): "time_of",
        ("EVENT", "GPE"): "located_at"
    }
    
    # Check if we have a mapping for this pair of entity types
    relationship = type_mappings.get((source_type, target_type))
    
    # Use "related_to" as default for unknown pairs
    if not relationship:
        relationship = "related_to"
        
    return relationship

def infer_type_relationship(source_type: str, target_type: str) -> Optional[str]:
    """
    Infer relationship type between entity types
    
    Args:
        source_type: Source entity type
        target_type: Target entity type
        
    Returns:
        Inferred relationship type or None
    """
    # Define common relationship mappings based on entity types
    type_mappings = {
        ("PERSON", "PERSON"): "associated_with",
        ("PERSON", "ORG"): "affiliated_with",
        ("ORG", "PERSON"): "employs",
        ("PERSON", "GPE"): "located_in",
        ("ORG", "GPE"): "based_in",
        ("ORG", "ORG"): "related_to",
        ("WORK_OF_ART", "PERSON"): "created_by",
        ("PERSON", "WORK_OF_ART"): "created",
        ("PRODUCT", "ORG"): "produced_by",
        ("ORG", "PRODUCT"): "produces",
        ("GPE", "GPE"): "neighbor_of",
        ("DATE", "EVENT"): "time_of",
        ("EVENT", "GPE"): "located_at"
    }
    
    # Check if we have a mapping for this pair of entity types
    relationship = type_mappings.get((source_type, target_type))
    
    # Use "related_to" as default for unknown pairs
    if not relationship:
        relationship = "related_to"
        
    return relationship

def traverse_graph(start_node_id: str, max_hops: int = 2) -> Dict[str, Any]:
    """
    Traverse the knowledge graph from a starting node
    
    Args:
        start_node_id: ID of the starting node
        max_hops: Maximum number of hops to traverse
        
    Returns:
        Subgraph as a dictionary
    """
    try:
        node_id = f"entity_{start_node_id}" if not start_node_id.startswith("entity_") else start_node_id
        
        # Check if the node exists
        if node_id not in graph:
            return {"nodes": [], "edges": []}
            
        # BFS traversal
        nodes = set([node_id])
        edges = set()
        current_nodes = set([node_id])
        
        for _ in range(max_hops):
            next_nodes = set()
            
            for node in current_nodes:
                # Get outgoing edges
                for neighbor in graph.successors(node):
                    if neighbor not in nodes:
                        next_nodes.add(neighbor)
                        nodes.add(neighbor)
                    edge_data = graph.get_edge_data(node, neighbor)
                    edges.add((node, neighbor, edge_data))
                
                # Get incoming edges
                for neighbor in graph.predecessors(node):
                    if neighbor not in nodes:
                        next_nodes.add(neighbor)
                        nodes.add(neighbor)
                    edge_data = graph.get_edge_data(neighbor, node)
                    edges.add((neighbor, node, edge_data))
            
            current_nodes = next_nodes
            if not current_nodes:
                break
        
        # Convert to dictionary
        nodes_list = []
        for node in nodes:
            node_data = graph.nodes[node]
            nodes_list.append({
                "id": node,
                "label": node_data.get("name", ""),
                "type": node_data.get("type", ""),
                "entity_type": node_data.get("entity_type", ""),
                "document_id": node_data.get("document_id", None)
            })
        
        edges_list = []
        for source, target, data in edges:
            edges_list.append({
                "source": source,
                "target": target,
                "label": data.get("type", ""),
                "confidence": data.get("confidence", 0.0)
            })
        
        return {
            "nodes": nodes_list,
            "edges": edges_list
        }
        
    except Exception as e:
        logger.error(f"Error traversing graph: {e}")
        return {"nodes": [], "edges": []}

def get_entity_connections(entity_id: int) -> Dict[str, Any]:
    """
    Get direct connections to an entity
    
    Args:
        entity_id: ID of the entity
        
    Returns:
        Connected entities as a dictionary
    """
    try:
        node_id = f"entity_{entity_id}"
        
        # Check if the node exists
        if node_id not in graph:
            return {"connections": []}
            
        connections = []
        
        # Get outgoing edges
        for neighbor in graph.successors(node_id):
            edge_data = graph.get_edge_data(node_id, neighbor)
            neighbor_data = graph.nodes[neighbor]
            connections.append({
                "entity_id": int(neighbor.split("_")[1]),
                "name": neighbor_data.get("name", ""),
                "entity_type": neighbor_data.get("entity_type", ""),
                "relationship": edge_data.get("type", ""),
                "confidence": edge_data.get("confidence", 0.0),
                "direction": "outgoing"
            })
            
        # Get incoming edges
        for neighbor in graph.predecessors(node_id):
            edge_data = graph.get_edge_data(neighbor, node_id)
            neighbor_data = graph.nodes[neighbor]
            connections.append({
                "entity_id": int(neighbor.split("_")[1]),
                "name": neighbor_data.get("name", ""),
                "entity_type": neighbor_data.get("entity_type", ""),
                "relationship": edge_data.get("type", ""),
                "confidence": edge_data.get("confidence", 0.0),
                "direction": "incoming"
            })
            
        return {"connections": connections}
        
    except Exception as e:
        logger.error(f"Error getting entity connections: {e}")
        return {"connections": []}

def get_entity_neighborhoods(entity_ids: List[int], max_hops: int = 2) -> Dict[str, Any]:
    """
    Get neighborhoods of multiple entities
    
    Args:
        entity_ids: List of entity IDs
        max_hops: Maximum number of hops to traverse
        
    Returns:
        Combined neighborhoods as a dictionary
    """
    try:
        combined_nodes = set()
        combined_edges = set()
        
        for entity_id in entity_ids:
            subgraph = traverse_graph(f"entity_{entity_id}", max_hops)
            
            # Extract node IDs
            node_ids = {node["id"] for node in subgraph["nodes"]}
            combined_nodes.update(node_ids)
            
            # Extract edges
            for edge in subgraph["edges"]:
                combined_edges.add((edge["source"], edge["target"], edge["label"]))
                
        # Convert to lists
        nodes_list = []
        for node_id in combined_nodes:
            node_data = graph.nodes[node_id]
            nodes_list.append({
                "id": node_id,
                "label": node_data.get("name", ""),
                "type": node_data.get("type", ""),
                "entity_type": node_data.get("entity_type", ""),
                "document_id": node_data.get("document_id", None)
            })
            
        edges_list = []
        for source, target, label in combined_edges:
            edge_data = graph.get_edge_data(source, target)
            edges_list.append({
                "source": source,
                "target": target,
                "label": label,
                "confidence": edge_data.get("confidence", 0.0)
            })
            
        return {
            "nodes": nodes_list,
            "edges": edges_list
        }
        
    except Exception as e:
        logger.error(f"Error getting entity neighborhoods: {e}")
        return {"nodes": [], "edges": []}

def find_paths_between_entities(source_id: int, target_id: int, max_hops: int = 3) -> List[Dict[str, Any]]:
    """
    Find paths between two entities
    
    Args:
        source_id: ID of the source entity
        target_id: ID of the target entity
        max_hops: Maximum number of hops to traverse
        
    Returns:
        List of paths between the entities
    """
    try:
        source_node = f"entity_{source_id}"
        target_node = f"entity_{target_id}"
        
        # Check if both nodes exist
        if source_node not in graph or target_node not in graph:
            return []
            
        # Find all simple paths
        paths = []
        for path in nx.all_simple_paths(graph, source=source_node, target=target_node, cutoff=max_hops):
            path_with_edges = []
            
            # Add nodes and edges to the path
            for i in range(len(path) - 1):
                source = path[i]
                target = path[i + 1]
                edge_data = graph.get_edge_data(source, target)
                
                source_data = graph.nodes[source]
                target_data = graph.nodes[target]
                
                path_with_edges.append({
                    "source": {
                        "id": int(source.split("_")[1]),
                        "name": source_data.get("name", ""),
                        "entity_type": source_data.get("entity_type", "")
                    },
                    "target": {
                        "id": int(target.split("_")[1]),
                        "name": target_data.get("name", ""),
                        "entity_type": target_data.get("entity_type", "")
                    },
                    "relationship": edge_data.get("type", ""),
                    "confidence": edge_data.get("confidence", 0.0)
                })
                
            paths.append(path_with_edges)
            
        return paths
        
    except Exception as e:
        logger.error(f"Error finding paths between entities: {e}")
        return []

def query_graph(query: Dict[str, Any]) -> Dict[str, Any]:
    """
    Query the knowledge graph using various filters
    
    Args:
        query: Query parameters (entity_types, relationship_types, confidence_threshold, etc.)
        
    Returns:
        Matching subgraph as a dictionary
    """
    try:
        entity_types = query.get("entity_types", [])
        relationship_types = query.get("relationship_types", [])
        confidence_threshold = query.get("confidence_threshold", 0.0)
        document_ids = query.get("document_ids", [])
        limit = query.get("limit", 100)
        
        # Filter nodes
        nodes = set()
        for node in graph.nodes:
            node_data = graph.nodes[node]
            
            # Filter by entity type
            if entity_types and node_data.get("entity_type") not in entity_types:
                continue
                
            # Filter by document ID
            if document_ids and node_data.get("document_id") not in document_ids:
                continue
                
            nodes.add(node)
            
        # Limit number of nodes
        if len(nodes) > limit:
            nodes = set(list(nodes)[:limit])
            
        # Get edges between filtered nodes
        edges = set()
        for source in nodes:
            for target in graph.successors(source):
                if target in nodes:
                    edge_data = graph.get_edge_data(source, target)
                    
                    # Filter by relationship type
                    if relationship_types and edge_data.get("type") not in relationship_types:
                        continue
                        
                    # Filter by confidence threshold
                    if edge_data.get("confidence", 0.0) < confidence_threshold:
                        continue
                        
                    edges.add((source, target, edge_data))
                    
        # Convert to lists
        nodes_list = []
        for node in nodes:
            node_data = graph.nodes[node]
            nodes_list.append({
                "id": node,
                "label": node_data.get("name", ""),
                "type": node_data.get("type", ""),
                "entity_type": node_data.get("entity_type", ""),
                "document_id": node_data.get("document_id", None)
            })
            
        edges_list = []
        for source, target, data in edges:
            edges_list.append({
                "source": source,
                "target": target,
                "label": data.get("type", ""),
                "confidence": data.get("confidence", 0.0)
            })
            
        return {
            "nodes": nodes_list,
            "edges": edges_list
        }
        
    except Exception as e:
        logger.error(f"Error querying graph: {e}")
        return {"nodes": [], "edges": []}

def get_graph_statistics() -> Dict[str, Any]:
    """
    Get statistics about the knowledge graph
    
    Returns:
        Dictionary of graph statistics
    """
    try:
        # Count nodes by entity type
        entity_types = {}
        for node in graph.nodes:
            node_data = graph.nodes[node]
            entity_type = node_data.get("entity_type")
            if entity_type:
                entity_types[entity_type] = entity_types.get(entity_type, 0) + 1
                
        # Count edges by relationship type
        relationship_types = {}
        for source, target, data in graph.edges.data():
            rel_type = data.get("type")
            if rel_type:
                relationship_types[rel_type] = relationship_types.get(rel_type, 0) + 1
                
        # Count nodes by document
        documents = {}
        for node in graph.nodes:
            node_data = graph.nodes[node]
            doc_id = node_data.get("document_id")
            if doc_id:
                documents[doc_id] = documents.get(doc_id, 0) + 1
                
        # Calculate density and other metrics
        density = nx.density(graph)
        num_connected_components = nx.number_weakly_connected_components(graph)
        
        return {
            "num_nodes": len(graph.nodes),
            "num_edges": len(graph.edges),
            "entity_types": entity_types,
            "relationship_types": relationship_types,
            "documents": documents,
            "density": density,
            "num_connected_components": num_connected_components
        }
        
    except Exception as e:
        logger.error(f"Error getting graph statistics: {e}")
        return {}

def export_graph(format: str = "networkx") -> Any:
    """
    Export the knowledge graph in various formats
    
    Args:
        format: Format to export (networkx, json, etc.)
        
    Returns:
        Exported graph
    """
    try:
        if format == "networkx":
            return graph
        elif format == "json":
            # Convert to JSON
            nodes = []
            for node, data in graph.nodes(data=True):
                node_data = {
                    "id": node,
                    "type": data.get("type", ""),
                    "name": data.get("name", ""),
                    "entity_type": data.get("entity_type", ""),
                    "document_id": data.get("document_id")
                }
                nodes.append(node_data)
                
            edges = []
            for source, target, data in graph.edges(data=True):
                edge_data = {
                    "source": source,
                    "target": target,
                    "type": data.get("type", ""),
                    "confidence": data.get("confidence", 0.0)
                }
                edges.append(edge_data)
                
            return {"nodes": nodes, "edges": edges}
        else:
            raise ValueError(f"Unsupported export format: {format}")
            
    except Exception as e:
        logger.error(f"Error exporting graph: {e}")
        return None
