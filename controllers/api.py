import logging
import traceback
import json
import time
from flask import Blueprint, request, jsonify, current_app
from flask_login import login_required, current_user
from werkzeug.utils import secure_filename
import os

from app import db
from models import (
    Document, DocumentChunk, Entity, Query, QueryResult, 
    Collection, Tag, User, Permission
)
from services import (
    document_processor, query_processor, retrieval_service, 
    knowledge_graph, vector_store, llm_service
)
from utils.helpers import get_file_size, get_mime_type, extract_filename
from utils.security import is_file_encrypted

logger = logging.getLogger(__name__)

api = Blueprint('api', __name__)

# Document Management APIs
@api.route('/documents', methods=['GET'])
@login_required
def get_documents():
    """Get documents for the current user"""
    try:
        # Parse query parameters
        page = int(request.args.get('page', 1))
        per_page = min(int(request.args.get('per_page', 20)), 100)
        query = request.args.get('query', '')
        collection_id = request.args.get('collection_id')
        tag_ids = request.args.getlist('tag_ids')
        file_types = request.args.getlist('file_types')
        sort_by = request.args.get('sort_by', 'created_at')
        sort_dir = request.args.get('sort_dir', 'desc')
        
        # Start building the query
        document_query = Document.query
        
        # Filter by owner or permissions
        document_query = document_query.filter(
            (Document.owner_id == current_user.id) |
            (Document.is_public == True) |
            (Document.id.in_(
                db.session.query(Permission.document_id)
                .filter(Permission.user_id == current_user.id, Permission.can_read == True)
            ))
        )
        
        # Apply search filter
        if query:
            document_query = document_query.filter(
                Document.title.ilike(f'%{query}%') |
                Document.description.ilike(f'%{query}%')
            )
            
        # Apply collection filter
        if collection_id:
            document_query = document_query.filter(
                Document.collections.any(id=collection_id)
            )
            
        # Apply tag filters
        if tag_ids:
            for tag_id in tag_ids:
                document_query = document_query.filter(
                    Document.tags.any(id=tag_id)
                )
                
        # Apply file type filters
        if file_types:
            document_query = document_query.filter(
                Document.file_type.in_(file_types)
            )
            
        # Apply sorting
        if sort_by == 'title':
            document_query = document_query.order_by(
                Document.title.desc() if sort_dir == 'desc' else Document.title
            )
        elif sort_by == 'file_size':
            document_query = document_query.order_by(
                Document.file_size.desc() if sort_dir == 'desc' else Document.file_size
            )
        else:  # Default to created_at
            document_query = document_query.order_by(
                Document.created_at.desc() if sort_dir == 'desc' else Document.created_at
            )
            
        # Paginate results
        paginated = document_query.paginate(page=page, per_page=per_page, error_out=False)
        
        # Format response
        documents = []
        for doc in paginated.items:
            documents.append({
                'id': doc.id,
                'title': doc.title,
                'description': doc.description,
                'file_type': doc.file_type,
                'file_size': doc.file_size,
                'mime_type': doc.mime_type,
                'owner_id': doc.owner_id,
                'is_public': doc.is_public,
                'is_encrypted': doc.is_encrypted,
                'created_at': doc.created_at.isoformat() if doc.created_at else None,
                'updated_at': doc.updated_at.isoformat() if doc.updated_at else None,
                'indexed_at': doc.indexed_at.isoformat() if doc.indexed_at else None,
                'collections': [c.id for c in doc.collections],
                'tags': [t.id for c in doc.tags]
            })
            
        return jsonify({
            'documents': documents,
            'page': page,
            'per_page': per_page,
            'total': paginated.total,
            'pages': paginated.pages
        })
        
    except Exception as e:
        logger.error(f"Error getting documents: {e}")
        logger.debug(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@api.route('/documents/<int:document_id>', methods=['GET'])
@login_required
def get_document(document_id):
    """Get a specific document"""
    try:
        # Get document
        document = Document.query.get_or_404(document_id)
        
        # Check access permissions
        if not (document.owner_id == current_user.id or 
                document.is_public or 
                Permission.query.filter_by(user_id=current_user.id, document_id=document_id, can_read=True).first()):
            return jsonify({'error': 'Access denied'}), 403
            
        # Get metadata
        metadata = None
        if document.doc_metadata:
            metadata = {
                'author': document.doc_metadata.author,
                'created_date': document.doc_metadata.created_date.isoformat() if document.doc_metadata.created_date else None,
                'modified_date': document.doc_metadata.modified_date.isoformat() if document.doc_metadata.modified_date else None,
                'page_count': document.doc_metadata.page_count,
                'word_count': document.doc_metadata.word_count,
                'source_url': document.doc_metadata.source_url,
                'source_system': document.doc_metadata.source_system,
                'additional_metadata': document.doc_metadata.additional_metadata
            }
            
        # Get collections and tags
        collections = [{'id': c.id, 'name': c.name} for c in document.collections]
        tags = [{'id': t.id, 'name': t.name} for t in document.tags]
        
        # Format response
        result = {
            'id': document.id,
            'title': document.title,
            'description': document.description,
            'file_type': document.file_type,
            'file_size': document.file_size,
            'mime_type': document.mime_type,
            'language': document.language,
            'owner_id': document.owner_id,
            'is_public': document.is_public,
            'is_encrypted': document.is_encrypted,
            'created_at': document.created_at.isoformat() if document.created_at else None,
            'updated_at': document.updated_at.isoformat() if document.updated_at else None,
            'indexed_at': document.indexed_at.isoformat() if document.indexed_at else None,
            'metadata': metadata,
            'collections': collections,
            'tags': tags,
            'chunk_count': document.chunks.count()
        }
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error getting document {document_id}: {e}")
        logger.debug(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@api.route('/documents/<int:document_id>/chunks', methods=['GET'])
@login_required
def get_document_chunks(document_id):
    """Get chunks for a specific document"""
    try:
        # Get document
        document = Document.query.get_or_404(document_id)
        
        # Check access permissions
        if not (document.owner_id == current_user.id or 
                document.is_public or 
                Permission.query.filter_by(user_id=current_user.id, document_id=document_id, can_read=True).first()):
            return jsonify({'error': 'Access denied'}), 403
            
        # Parse query parameters
        page = int(request.args.get('page', 1))
        per_page = min(int(request.args.get('per_page', 20)), 100)
        
        # Get chunks
        chunks_query = document.chunks.order_by(DocumentChunk.chunk_index)
        paginated = chunks_query.paginate(page=page, per_page=per_page, error_out=False)
        
        # Format response
        chunks = []
        for chunk in paginated.items:
            chunks.append({
                'id': chunk.id,
                'content': chunk.content,
                'chunk_index': chunk.chunk_index,
                'start_char': chunk.start_char,
                'end_char': chunk.end_char
            })
            
        return jsonify({
            'chunks': chunks,
            'page': page,
            'per_page': per_page,
            'total': paginated.total,
            'pages': paginated.pages
        })
        
    except Exception as e:
        logger.error(f"Error getting chunks for document {document_id}: {e}")
        logger.debug(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@api.route('/documents/<int:document_id>/entities', methods=['GET'])
@login_required
def get_document_entities(document_id):
    """Get entities for a specific document"""
    try:
        # Get document
        document = Document.query.get_or_404(document_id)
        
        # Check access permissions
        if not (document.owner_id == current_user.id or 
                document.is_public or 
                Permission.query.filter_by(user_id=current_user.id, document_id=document_id, can_read=True).first()):
            return jsonify({'error': 'Access denied'}), 403
            
        # Parse query parameters
        page = int(request.args.get('page', 1))
        per_page = min(int(request.args.get('per_page', 50)), 200)
        entity_type = request.args.get('entity_type')
        confidence_min = float(request.args.get('confidence_min', 0.5))
        
        # Get entities
        entities_query = document.entities
        
        if entity_type:
            entities_query = entities_query.filter(Entity.entity_type == entity_type)
            
        entities_query = entities_query.filter(Entity.confidence >= confidence_min)
        entities_query = entities_query.order_by(Entity.confidence.desc())
        
        paginated = entities_query.paginate(page=page, per_page=per_page, error_out=False)
        
        # Format response
        entities = []
        for entity in paginated.items:
            entities.append({
                'id': entity.id,
                'name': entity.name,
                'entity_type': entity.entity_type,
                'confidence': entity.confidence,
                'chunk_id': entity.chunk_id,
                'start_char': entity.start_char,
                'end_char': entity.end_char,
                'metadata': entity.entity_metadata
            })
            
        return jsonify({
            'entities': entities,
            'page': page,
            'per_page': per_page,
            'total': paginated.total,
            'pages': paginated.pages
        })
        
    except Exception as e:
        logger.error(f"Error getting entities for document {document_id}: {e}")
        logger.debug(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@api.route('/documents', methods=['POST'])
@login_required
def upload_document():
    """Upload a new document"""
    try:
        # Check if file is provided
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
            
        file = request.files['file']
        
        # Check if file is empty
        if file.filename == '':
            return jsonify({'error': 'Empty file name'}), 400
            
        # Get form data
        title = request.form.get('title', '')
        description = request.form.get('description', '')
        is_public = request.form.get('is_public', 'false').lower() == 'true'
        is_encrypted = request.form.get('is_encrypted', 'false').lower() == 'true'
        collection_ids = request.form.getlist('collection_ids')
        tag_names = request.form.getlist('tags')
        
        # If no title provided, use filename
        if not title:
            title = extract_filename(file.filename)
            
        # Save file
        filename = secure_filename(file.filename)
        file_path = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Process document
        document = document_processor.process_document(
            file_path=file_path,
            title=title,
            description=description,
            user_id=current_user.id,
            is_public=is_public,
            is_encrypted=is_encrypted
        )
        
        if not document:
            return jsonify({'error': 'Error processing document'}), 500
            
        # Add to collections
        if collection_ids:
            for collection_id in collection_ids:
                try:
                    collection = Collection.query.get(int(collection_id))
                    if collection and collection.owner_id == current_user.id:
                        document.collections.append(collection)
                except:
                    pass
        
        # Add tags
        if tag_names:
            for tag_name in tag_names:
                tag = Tag.query.filter_by(name=tag_name).first()
                if not tag:
                    tag = Tag(name=tag_name)
                    db.session.add(tag)
                document.tags.append(tag)
                
        db.session.commit()
        
        return jsonify({
            'id': document.id,
            'title': document.title,
            'message': 'Document uploaded and processed successfully'
        })
        
    except Exception as e:
        logger.error(f"Error uploading document: {e}")
        logger.debug(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@api.route('/documents/<int:document_id>', methods=['DELETE'])
@login_required
def delete_document(document_id):
    """Delete a document"""
    try:
        # Get document
        document = Document.query.get_or_404(document_id)
        
        # Check ownership
        if document.owner_id != current_user.id:
            return jsonify({'error': 'Access denied'}), 403
            
        # Delete file from disk
        try:
            if os.path.exists(document.file_path):
                os.remove(document.file_path)
        except Exception as e:
            logger.warning(f"Error deleting file {document.file_path}: {e}")
            
        # Delete from database (cascade will handle related records)
        db.session.delete(document)
        db.session.commit()
        
        return jsonify({'message': 'Document deleted successfully'})
        
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error deleting document {document_id}: {e}")
        logger.debug(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@api.route('/documents/<int:document_id>/reprocess', methods=['POST'])
@login_required
def reprocess_document(document_id):
    """Reprocess a document"""
    try:
        # Get document
        document = Document.query.get_or_404(document_id)
        
        # Check ownership
        if document.owner_id != current_user.id:
            return jsonify({'error': 'Access denied'}), 403
            
        # Reprocess document
        success = document_processor.reprocess_document(document_id)
        
        if not success:
            return jsonify({'error': 'Error reprocessing document'}), 500
            
        return jsonify({'message': 'Document reprocessed successfully'})
        
    except Exception as e:
        logger.error(f"Error reprocessing document {document_id}: {e}")
        logger.debug(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

# Collection Management APIs
@api.route('/collections', methods=['GET'])
@login_required
def get_collections():
    """Get collections for the current user"""
    try:
        # Get collections owned by the user
        collections = Collection.query.filter_by(owner_id=current_user.id).all()
        
        # Format response
        result = []
        for collection in collections:
            result.append({
                'id': collection.id,
                'name': collection.name,
                'description': collection.description,
                'is_public': collection.is_public,
                'created_at': collection.created_at.isoformat() if collection.created_at else None,
                'document_count': collection.documents.count()
            })
            
        return jsonify({'collections': result})
        
    except Exception as e:
        logger.error(f"Error getting collections: {e}")
        logger.debug(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@api.route('/collections', methods=['POST'])
@login_required
def create_collection():
    """Create a new collection"""
    try:
        data = request.get_json()
        
        # Validate input
        if not data or 'name' not in data:
            return jsonify({'error': 'Name is required'}), 400
            
        # Create collection
        collection = Collection(
            name=data['name'],
            description=data.get('description', ''),
            owner_id=current_user.id,
            is_public=data.get('is_public', False)
        )
        
        db.session.add(collection)
        db.session.commit()
        
        return jsonify({
            'id': collection.id,
            'name': collection.name,
            'message': 'Collection created successfully'
        })
        
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error creating collection: {e}")
        logger.debug(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

# Search APIs
@api.route('/search', methods=['POST'])
@login_required
def search():
    """Search documents"""
    try:
        data = request.get_json()
        
        # Validate input
        if not data or 'query' not in data:
            return jsonify({'error': 'Query is required'}), 400
            
        query_text = data['query']
        filters = data.get('filters', {})
        
        # Process query
        query, query_data = query_processor.process_query(
            query_text=query_text,
            user_id=current_user.id,
            filters=filters
        )
        
        # Check for errors
        if 'error' in query_data:
            return jsonify({'error': query_data['error']}), 400
            
        # Retrieve documents
        retrieval_result = retrieval_service.retrieve_documents(
            query_data=query_data,
            user_id=current_user.id
        )
        
        # Format response
        results = []
        for result in retrieval_result.get('results', []):
            results.append({
                'document_id': result['document_id'],
                'document_title': result['document_title'],
                'chunk_id': result['chunk_id'],
                'content': result['content'],
                'score': result['score'],
                'rank': result['rank'],
                'strategy': result.get('strategy', 'hybrid')
            })
            
        # Get multi-hop reasoning if requested
        if data.get('multi_hop', False) and results:
            enhanced_results = retrieval_service.perform_multi_hop_reasoning(
                query_id=query.id,
                query_text=query_text,
                initial_results=retrieval_result.get('results', [])
            )
            
            # Update results
            results = []
            for result in enhanced_results:
                result_data = {
                    'document_id': result['document_id'],
                    'document_title': result['document_title'],
                    'chunk_id': result['chunk_id'],
                    'content': result['content'],
                    'score': result['score'],
                    'rank': result['rank'],
                    'strategy': result.get('strategy', 'hybrid')
                }
                
                # Add reasoning paths if available
                if 'reasoning_paths' in result:
                    result_data['reasoning_paths'] = result['reasoning_paths']
                    
                results.append(result_data)
        
        # Generate LLM response if requested
        llm_response = None
        if data.get('generate_answer', False) and results:
            # Use top results as context
            contexts = retrieval_result.get('results', [])[:5]
            llm_response = llm_service.generate_response(query_text, contexts)
            
        response = {
            'query_id': query.id,
            'results': results,
            'total_results': len(results),
            'elapsed_time': retrieval_result.get('elapsed_time', 0)
        }
        
        if llm_response:
            response['answer'] = llm_response.get('answer')
            response['sources'] = llm_response.get('sources')
            
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error searching documents: {e}")
        logger.debug(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

# Knowledge Graph APIs
@api.route('/knowledge-graph/entities/<int:entity_id>', methods=['GET'])
@login_required
def get_entity_connections(entity_id):
    """Get connections for an entity"""
    try:
        # Get entity
        entity = Entity.query.get_or_404(entity_id)
        
        # Check access to document
        document = Document.query.get(entity.document_id)
        if not (document.owner_id == current_user.id or 
                document.is_public or 
                Permission.query.filter_by(user_id=current_user.id, document_id=document.id, can_read=True).first()):
            return jsonify({'error': 'Access denied'}), 403
            
        # Get connections
        connections = knowledge_graph.get_entity_connections(entity_id)
        
        return jsonify(connections)
        
    except Exception as e:
        logger.error(f"Error getting entity connections: {e}")
        logger.debug(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@api.route('/knowledge-graph/traverse', methods=['POST'])
@login_required
def traverse_graph():
    """Traverse the knowledge graph"""
    try:
        data = request.get_json()
        
        # Validate input
        if not data or 'entity_id' not in data:
            return jsonify({'error': 'Entity ID is required'}), 400
            
        entity_id = data['entity_id']
        max_hops = int(data.get('max_hops', 2))
        
        # Get entity
        entity = Entity.query.get_or_404(entity_id)
        
        # Check access to document
        document = Document.query.get(entity.document_id)
        if not (document.owner_id == current_user.id or 
                document.is_public or 
                Permission.query.filter_by(user_id=current_user.id, document_id=document.id, can_read=True).first()):
            return jsonify({'error': 'Access denied'}), 403
            
        # Traverse graph
        graph_data = knowledge_graph.traverse_graph(f"entity_{entity_id}", max_hops)
        
        return jsonify(graph_data)
        
    except Exception as e:
        logger.error(f"Error traversing graph: {e}")
        logger.debug(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@api.route('/knowledge-graph/paths', methods=['POST'])
@login_required
def find_paths():
    """Find paths between entities"""
    try:
        data = request.get_json()
        
        # Validate input
        if not data or 'source_id' not in data or 'target_id' not in data:
            return jsonify({'error': 'Source and target entity IDs are required'}), 400
            
        source_id = data['source_id']
        target_id = data['target_id']
        max_hops = int(data.get('max_hops', 3))
        
        # Get entities
        source = Entity.query.get_or_404(source_id)
        target = Entity.query.get_or_404(target_id)
        
        # Check access to documents
        source_doc = Document.query.get(source.document_id)
        target_doc = Document.query.get(target.document_id)
        
        source_access = (source_doc.owner_id == current_user.id or 
                     source_doc.is_public or 
                     Permission.query.filter_by(user_id=current_user.id, document_id=source_doc.id, can_read=True).first())
                     
        target_access = (target_doc.owner_id == current_user.id or 
                     target_doc.is_public or 
                     Permission.query.filter_by(user_id=current_user.id, document_id=target_doc.id, can_read=True).first())
                     
        if not (source_access and target_access):
            return jsonify({'error': 'Access denied'}), 403
            
        # Find paths
        paths = knowledge_graph.find_paths_between_entities(source_id, target_id, max_hops)
        
        return jsonify({'paths': paths})
        
    except Exception as e:
        logger.error(f"Error finding paths: {e}")
        logger.debug(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@api.route('/knowledge-graph/query', methods=['POST'])
@login_required
def query_graph():
    """Query the knowledge graph"""
    try:
        data = request.get_json()
        
        # Validate input
        if not data:
            return jsonify({'error': 'Query parameters are required'}), 400
            
        # Query graph
        graph_data = knowledge_graph.query_graph(data)
        
        return jsonify(graph_data)
        
    except Exception as e:
        logger.error(f"Error querying graph: {e}")
        logger.debug(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

# User Management APIs
@api.route('/users/me', methods=['GET'])
@login_required
def get_current_user():
    """Get current user information"""
    try:
        return jsonify({
            'id': current_user.id,
            'username': current_user.username,
            'email': current_user.email,
            'first_name': current_user.first_name,
            'last_name': current_user.last_name,
            'roles': [role.name for role in current_user.roles]
        })
        
    except Exception as e:
        logger.error(f"Error getting current user: {e}")
        logger.debug(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@api.route('/users/me', methods=['PUT'])
@login_required
def update_current_user():
    """Update current user information"""
    try:
        data = request.get_json()
        
        # Update fields
        if 'first_name' in data:
            current_user.first_name = data['first_name']
            
        if 'last_name' in data:
            current_user.last_name = data['last_name']
            
        if 'email' in data:
            # Check if email already exists
            existing = User.query.filter_by(email=data['email']).first()
            if existing and existing.id != current_user.id:
                return jsonify({'error': 'Email already in use'}), 400
                
            current_user.email = data['email']
            
        if 'password' in data and data['password']:
            current_user.set_password(data['password'])
            
        db.session.commit()
        
        return jsonify({'message': 'User updated successfully'})
        
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error updating user: {e}")
        logger.debug(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

# System Statistics APIs
@api.route('/stats/system', methods=['GET'])
@login_required
def get_system_stats():
    """Get system statistics"""
    try:
        # Document stats
        total_documents = Document.query.count()
        user_documents = Document.query.filter_by(owner_id=current_user.id).count()
        
        # Entity stats
        total_entities = Entity.query.count()
        
        # Graph stats
        graph_stats = knowledge_graph.get_graph_statistics()
        
        # Vector index stats
        vector_stats = vector_store.get_index_info()
        
        return jsonify({
            'documents': {
                'total': total_documents,
                'user': user_documents
            },
            'entities': {
                'total': total_entities,
                'by_type': graph_stats.get('entity_types', {})
            },
            'graph': graph_stats,
            'vector_index': vector_stats
        })
        
    except Exception as e:
        logger.error(f"Error getting system stats: {e}")
        logger.debug(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

# Handle errors
@api.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Resource not found'}), 404

@api.errorhandler(400)
def bad_request(error):
    return jsonify({'error': 'Bad request'}), 400

@api.errorhandler(401)
def unauthorized(error):
    return jsonify({'error': 'Unauthorized'}), 401

@api.errorhandler(403)
def forbidden(error):
    return jsonify({'error': 'Forbidden'}), 403

@api.errorhandler(500)
def server_error(error):
    return jsonify({'error': 'Server error'}), 500
