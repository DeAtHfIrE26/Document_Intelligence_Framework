import logging
import os
import traceback
from datetime import datetime
from flask import Blueprint, render_template, redirect, url_for, flash, request, jsonify, current_app, session
from flask_login import login_user, logout_user, login_required, current_user
from werkzeug.utils import secure_filename

from app import db
from models import User, Document, Collection, Tag, Entity, EntityRelationship
from services import document_processor, query_processor, retrieval_service, knowledge_graph, llm_service
from utils.helpers import get_file_size, get_mime_type, extract_filename

logger = logging.getLogger(__name__)

web = Blueprint('web', __name__)

@web.route('/')
def index():
    """Home page"""
    if current_user.is_authenticated:
        # Get recent documents for the user
        recent_documents = Document.query.filter(
            (Document.owner_id == current_user.id) | 
            (Document.is_public == True)
        ).order_by(Document.created_at.desc()).limit(5).all()
        
        # Get recent searches
        from models import Query
        recent_searches = Query.query.filter_by(
            user_id=current_user.id
        ).order_by(Query.created_at.desc()).limit(5).all()
        
        # Get collections
        collections = Collection.query.filter_by(
            owner_id=current_user.id
        ).order_by(Collection.name).all()
        
        return render_template(
            'index.html', 
            recent_documents=recent_documents,
            recent_searches=recent_searches,
            collections=collections
        )
    else:
        return render_template('index.html')

@web.route('/login', methods=['GET', 'POST'])
def login():
    """User login"""
    if current_user.is_authenticated:
        return redirect(url_for('web.index'))
        
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        if not username or not password:
            flash('Please enter both username and password', 'danger')
            return render_template('login.html')
            
        user = User.query.filter_by(username=username).first()
        
        if user and user.check_password(password):
            login_user(user)
            user.last_login = datetime.utcnow()
            db.session.commit()
            
            # Redirect to requested page or home
            next_page = request.args.get('next')
            if next_page:
                return redirect(next_page)
            return redirect(url_for('web.index'))
        else:
            flash('Invalid username or password', 'danger')
            
    return render_template('login.html')

@web.route('/logout')
@login_required
def logout():
    """User logout"""
    logout_user()
    return redirect(url_for('web.index'))

@web.route('/register', methods=['GET', 'POST'])
def register():
    """User registration"""
    if current_user.is_authenticated:
        return redirect(url_for('web.index'))
        
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        
        # Basic validation
        if not username or not email or not password:
            flash('All fields are required', 'danger')
            return render_template('register.html')
            
        if password != confirm_password:
            flash('Passwords do not match', 'danger')
            return render_template('register.html')
            
        # Check if username or email already exists
        if User.query.filter_by(username=username).first():
            flash('Username already exists', 'danger')
            return render_template('register.html')
            
        if User.query.filter_by(email=email).first():
            flash('Email already exists', 'danger')
            return render_template('register.html')
            
        # Create new user
        user = User(
            username=username,
            email=email,
            created_at=datetime.utcnow()
        )
        user.set_password(password)
        
        db.session.add(user)
        db.session.commit()
        
        flash('Registration successful! You can now log in.', 'success')
        return redirect(url_for('web.login'))
        
    return render_template('register.html')

@web.route('/upload', methods=['GET', 'POST'])
@login_required
def upload():
    """Upload document page"""
    if request.method == 'POST':
        # Check if file is provided
        if 'file' not in request.files:
            flash('No file selected', 'danger')
            return redirect(request.url)
            
        file = request.files['file']
        
        # Check if file is empty
        if file.filename == '':
            flash('No file selected', 'danger')
            return redirect(request.url)
            
        # Get form data
        title = request.form.get('title', '')
        description = request.form.get('description', '')
        is_public = request.form.get('is_public', 'false') == 'true'
        is_encrypted = request.form.get('is_encrypted', 'false') == 'true'
        collection_ids = request.form.getlist('collection_ids')
        tag_names = request.form.getlist('tags')
        
        # If no title provided, use filename
        if not title:
            title = extract_filename(file.filename)
            
        # Save file
        filename = secure_filename(file.filename)
        file_path = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        try:
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
                flash('Error processing document', 'danger')
                return redirect(request.url)
                
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
            
            flash('Document uploaded and processed successfully', 'success')
            return redirect(url_for('web.view_document', document_id=document.id))
            
        except Exception as e:
            logger.error(f"Error processing document: {e}")
            logger.debug(traceback.format_exc())
            flash(f'Error processing document: {str(e)}', 'danger')
            return redirect(request.url)
    
    # Get collections for the form
    collections = Collection.query.filter_by(owner_id=current_user.id).all()
    
    # Get popular tags for the form
    tags = Tag.query.join(Tag.documents).group_by(Tag.id).order_by(db.func.count().desc()).limit(20).all()
    
    return render_template('upload.html', collections=collections, tags=tags)

@web.route('/search', methods=['GET', 'POST'])
@login_required
def search():
    """Search page"""
    if request.method == 'POST':
        query_text = request.form.get('query', '')
        search_mode = request.form.get('search_mode', 'hybrid')
        filters = {}
        
        # Parse filters
        collection_id = request.form.get('collection_id')
        if collection_id and collection_id != 'all':
            filters['collection_id'] = int(collection_id)
            
        document_type = request.form.get('document_type')
        if document_type and document_type != 'all':
            filters['document_type'] = document_type
            
        # Process query
        query, query_data = query_processor.process_query(
            query_text=query_text,
            user_id=current_user.id,
            filters=filters
        )
        
        if 'error' in query_data:
            flash(f"Error processing query: {query_data['error']}", 'danger')
            return redirect(url_for('web.search'))
            
        # Override search type if specified
        if search_mode != 'hybrid':
            query_data['search_type'] = search_mode
            
        # Retrieve documents
        retrieval_result = retrieval_service.retrieve_documents(
            query_data=query_data,
            user_id=current_user.id
        )
        
        # Check for multi-hop option
        multi_hop = request.form.get('multi_hop', 'false') == 'true'
        if multi_hop and retrieval_result.get('results'):
            enhanced_results = retrieval_service.perform_multi_hop_reasoning(
                query_id=query.id,
                query_text=query_text,
                initial_results=retrieval_result.get('results', [])
            )
            retrieval_result['results'] = enhanced_results
            
        # Generate answer if requested
        generate_answer = request.form.get('generate_answer', 'false') == 'true'
        answer = None
        sources = None
        
        if generate_answer and retrieval_result.get('results'):
            contexts = retrieval_result.get('results', [])[:5]
            llm_response = llm_service.generate_response(query_text, contexts)
            
            if llm_response and 'error' not in llm_response:
                answer = llm_response.get('answer')
                sources = llm_response.get('sources')
        
        # Get collections for the form
        collections = Collection.query.filter_by(owner_id=current_user.id).all()
        
        return render_template(
            'search.html',
            query=query_text,
            query_data=query_data,
            results=retrieval_result.get('results', []),
            total_results=retrieval_result.get('total_results', 0),
            elapsed_time=retrieval_result.get('elapsed_time', 0),
            collections=collections,
            selected_collection=collection_id,
            selected_type=document_type,
            search_mode=search_mode,
            multi_hop=multi_hop,
            generate_answer=generate_answer,
            answer=answer,
            sources=sources
        )
        
    # Get collections for the form
    collections = Collection.query.filter_by(owner_id=current_user.id).all()
    
    # Handle GET request
    query = request.args.get('q', '')
    return render_template('search.html', query=query, collections=collections)

@web.route('/documents')
@login_required
def documents():
    """Document list page"""
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
            db.session.query('Permission.document_id')
            .filter_by(user_id=current_user.id, can_read=True)
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
    
    # Get collections and tags for filters
    collections = Collection.query.filter_by(owner_id=current_user.id).all()
    tags = Tag.query.join(Tag.documents).group_by(Tag.id).order_by(db.func.count().desc()).limit(20).all()
    
    # Get unique file types
    file_type_results = db.session.query(Document.file_type).distinct().all()
    available_file_types = [ft[0] for ft in file_type_results]
    
    return render_template(
        'documents.html',
        documents=paginated.items,
        pagination=paginated,
        query=query,
        collections=collections,
        tags=tags,
        selected_collection=collection_id,
        selected_tags=tag_ids,
        selected_file_types=file_types,
        available_file_types=available_file_types,
        sort_by=sort_by,
        sort_dir=sort_dir
    )

@web.route('/documents/<int:document_id>')
@login_required
def view_document(document_id):
    """Document view page"""
    # Get document
    document = Document.query.get_or_404(document_id)
    
    # Check access permissions
    if not (document.owner_id == current_user.id or 
            document.is_public or 
            db.session.query('Permission').filter_by(user_id=current_user.id, document_id=document_id, can_read=True).first()):
        flash('You do not have permission to view this document', 'danger')
        return redirect(url_for('web.documents'))
        
    # Get document chunks
    chunks = document.chunks.order_by(db.models.DocumentChunk.chunk_index).all()
    
    # Get entities
    entities = document.entities.order_by(db.models.Entity.confidence.desc()).limit(100).all()
    
    # Get document metadata
    metadata = document.doc_metadata
    
    # Get collections
    collections = document.collections
    
    # Get tags
    tags = document.tags
    
    return render_template(
        'document.html',
        document=document,
        chunks=chunks,
        entities=entities,
        metadata=metadata,
        collections=collections,
        tags=tags
    )

@web.route('/graph')
@login_required
def knowledge_graph_viewer():
    """Knowledge graph visualization page"""
    # Get initial entities with highest confidence
    top_entities = Entity.query.order_by(Entity.confidence.desc()).limit(10).all()
    
    # Get entity types for filtering
    entity_types = db.session.query(Entity.entity_type).distinct().all()
    entity_types = [et[0] for et in entity_types]
    
    # Get relationship types for filtering
    rel_types = db.session.query(EntityRelationship.relationship_type).distinct().all()
    rel_types = [rt[0] for rt in rel_types]
    
    # Get graph statistics
    graph_stats = knowledge_graph.get_graph_statistics()
    
    return render_template(
        'graph.html',
        initial_entities=top_entities,
        entity_types=entity_types,
        relationship_types=rel_types,
        graph_stats=graph_stats
    )

@web.route('/collections')
@login_required
def collections():
    """Collections management page"""
    # Get user collections
    user_collections = Collection.query.filter_by(owner_id=current_user.id).all()
    
    return render_template('collections.html', collections=user_collections)

@web.route('/collections/<int:collection_id>')
@login_required
def view_collection(collection_id):
    """Collection view page"""
    # Get collection
    collection = Collection.query.get_or_404(collection_id)
    
    # Check ownership
    if collection.owner_id != current_user.id:
        flash('You do not have permission to view this collection', 'danger')
        return redirect(url_for('web.collections'))
        
    # Get documents in collection
    documents = collection.documents.all()
    
    return render_template(
        'collection.html',
        collection=collection,
        documents=documents
    )

@web.route('/profile')
@login_required
def profile():
    """User profile page"""
    # Get user statistics
    document_count = Document.query.filter_by(owner_id=current_user.id).count()
    collection_count = Collection.query.filter_by(owner_id=current_user.id).count()
    
    from models import Query
    search_count = Query.query.filter_by(user_id=current_user.id).count()
    
    return render_template(
        'profile.html',
        user=current_user,
        document_count=document_count,
        collection_count=collection_count,
        search_count=search_count
    )

@web.route('/about')
def about():
    """About page"""
    return render_template('about.html')

# Error handlers
@web.errorhandler(404)
def page_not_found(e):
    return render_template('errors/404.html'), 404

@web.errorhandler(500)
def server_error(e):
    return render_template('errors/500.html'), 500

@web.errorhandler(403)
def forbidden(e):
    return render_template('errors/403.html'), 403
