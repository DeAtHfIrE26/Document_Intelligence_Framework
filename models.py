from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import UserMixin
from app import db

# Association tables for many-to-many relationships
document_collection = db.Table(
    'document_collection',
    db.Column('document_id', db.Integer, db.ForeignKey('document.id'), primary_key=True),
    db.Column('collection_id', db.Integer, db.ForeignKey('collection.id'), primary_key=True)
)

document_tag = db.Table(
    'document_tag',
    db.Column('document_id', db.Integer, db.ForeignKey('document.id'), primary_key=True),
    db.Column('tag_id', db.Integer, db.ForeignKey('tag.id'), primary_key=True)
)

user_roles = db.Table(
    'user_roles',
    db.Column('user_id', db.Integer, db.ForeignKey('user.id'), primary_key=True),
    db.Column('role_id', db.Integer, db.ForeignKey('role.id'), primary_key=True)
)

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), unique=True, nullable=False, index=True)
    email = db.Column(db.String(120), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(256), nullable=False)
    first_name = db.Column(db.String(50))
    last_name = db.Column(db.String(50))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    active = db.Column(db.Boolean, default=True)
    last_login = db.Column(db.DateTime)
    
    # Relationships
    documents = db.relationship('Document', backref='owner', lazy='dynamic')
    roles = db.relationship('Role', secondary=user_roles, backref=db.backref('users', lazy='dynamic'))
    permissions = db.relationship('Permission', backref='user', lazy='dynamic')
    
    def set_password(self, password):
        self.password_hash = generate_password_hash(password)
        
    def check_password(self, password):
        return check_password_hash(self.password_hash, password)
    
    def has_role(self, role_name):
        return any(role.name == role_name for role in self.roles)
    
    def __repr__(self):
        return f'<User {self.username}>'

class Role(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(64), unique=True, nullable=False)
    description = db.Column(db.String(255))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f'<Role {self.name}>'

class Permission(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    document_id = db.Column(db.Integer, db.ForeignKey('document.id'), nullable=False)
    can_read = db.Column(db.Boolean, default=True)
    can_write = db.Column(db.Boolean, default=False)
    can_delete = db.Column(db.Boolean, default=False)
    can_share = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Unique constraint to prevent duplicate permissions
    __table_args__ = (db.UniqueConstraint('user_id', 'document_id', name='unique_user_document_permission'),)
    
    def __repr__(self):
        return f'<Permission user_id={self.user_id} document_id={self.document_id}>'

class Document(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(255), nullable=False, index=True)
    description = db.Column(db.Text)
    file_path = db.Column(db.String(512), nullable=False)
    file_type = db.Column(db.String(50), nullable=False)
    file_size = db.Column(db.Integer, nullable=False)  # Size in bytes
    mime_type = db.Column(db.String(100))
    language = db.Column(db.String(10))
    owner_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    is_public = db.Column(db.Boolean, default=False)
    is_encrypted = db.Column(db.Boolean, default=False)
    encryption_metadata = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    indexed_at = db.Column(db.DateTime)
    
    # Relationships
    collections = db.relationship('Collection', secondary=document_collection, backref=db.backref('documents', lazy='dynamic'))
    tags = db.relationship('Tag', secondary=document_tag, backref=db.backref('documents', lazy='dynamic'))
    chunks = db.relationship('DocumentChunk', backref='document', lazy='dynamic', cascade='all, delete-orphan')
    permissions = db.relationship('Permission', backref='document', lazy='dynamic', cascade='all, delete-orphan')
    doc_metadata = db.relationship('DocumentMetadata', backref='document', uselist=False, cascade='all, delete-orphan')
    entities = db.relationship('Entity', backref='document', lazy='dynamic')
    
    def __repr__(self):
        return f'<Document {self.title}>'

class DocumentChunk(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    document_id = db.Column(db.Integer, db.ForeignKey('document.id'), nullable=False)
    content = db.Column(db.Text, nullable=False)
    chunk_index = db.Column(db.Integer, nullable=False)
    embedding_id = db.Column(db.String(64), index=True)
    start_char = db.Column(db.Integer, nullable=False)
    end_char = db.Column(db.Integer, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Composite index for efficient document chunk retrieval
    __table_args__ = (db.Index('idx_document_chunk_index', 'document_id', 'chunk_index'),)
    
    def __repr__(self):
        return f'<DocumentChunk doc_id={self.document_id} index={self.chunk_index}>'

class DocumentMetadata(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    document_id = db.Column(db.Integer, db.ForeignKey('document.id'), nullable=False, unique=True)
    author = db.Column(db.String(255))
    created_date = db.Column(db.DateTime)
    modified_date = db.Column(db.DateTime)
    page_count = db.Column(db.Integer)
    word_count = db.Column(db.Integer)
    source_url = db.Column(db.String(512))
    source_system = db.Column(db.String(100))
    additional_metadata = db.Column(db.JSON)
    
    def __repr__(self):
        return f'<DocumentMetadata for doc_id={self.document_id}>'

class Collection(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    description = db.Column(db.Text)
    owner_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    is_public = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationship
    owner = db.relationship('User', backref=db.backref('collections', lazy='dynamic'))
    
    def __repr__(self):
        return f'<Collection {self.name}>'

class Tag(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(50), nullable=False, unique=True, index=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f'<Tag {self.name}>'

class Entity(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(255), nullable=False, index=True)
    entity_type = db.Column(db.String(50), nullable=False, index=True)
    document_id = db.Column(db.Integer, db.ForeignKey('document.id'), nullable=False)
    chunk_id = db.Column(db.Integer, db.ForeignKey('document_chunk.id'))
    confidence = db.Column(db.Float, nullable=False)
    start_char = db.Column(db.Integer)
    end_char = db.Column(db.Integer)
    entity_metadata = db.Column(db.JSON)
    embedding_id = db.Column(db.String(64), index=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationships
    chunk = db.relationship('DocumentChunk', backref=db.backref('entities', lazy='dynamic'))
    relationships_as_source = db.relationship(
        'EntityRelationship', 
        foreign_keys='EntityRelationship.source_id',
        backref='source', 
        lazy='dynamic',
        cascade='all, delete-orphan'
    )
    relationships_as_target = db.relationship(
        'EntityRelationship', 
        foreign_keys='EntityRelationship.target_id',
        backref='target', 
        lazy='dynamic',
        cascade='all, delete-orphan'
    )
    
    def __repr__(self):
        return f'<Entity {self.name} ({self.entity_type})>'

class EntityRelationship(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    source_id = db.Column(db.Integer, db.ForeignKey('entity.id'), nullable=False)
    target_id = db.Column(db.Integer, db.ForeignKey('entity.id'), nullable=False)
    relationship_type = db.Column(db.String(100), nullable=False, index=True)
    confidence = db.Column(db.Float, nullable=False)
    relationship_metadata = db.Column(db.JSON)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Prevent self-relationships and duplicate relationships
    __table_args__ = (
        db.CheckConstraint('source_id != target_id', name='check_not_self_relation'),
        db.UniqueConstraint('source_id', 'target_id', 'relationship_type', name='unique_relationship')
    )
    
    def __repr__(self):
        return f'<EntityRelationship {self.relationship_type} from {self.source_id} to {self.target_id}>'

class Query(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    original_query = db.Column(db.String(1000), nullable=False)
    processed_query = db.Column(db.String(1000))
    embedding_id = db.Column(db.String(64), index=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationships
    user = db.relationship('User', backref=db.backref('queries', lazy='dynamic'))
    results = db.relationship('QueryResult', backref='query', lazy='dynamic', cascade='all, delete-orphan')
    
    def __repr__(self):
        return f'<Query {self.original_query[:50]}>'

class QueryResult(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    query_id = db.Column(db.Integer, db.ForeignKey('query.id'), nullable=False)
    document_id = db.Column(db.Integer, db.ForeignKey('document.id'), nullable=False)
    chunk_id = db.Column(db.Integer, db.ForeignKey('document_chunk.id'))
    relevance_score = db.Column(db.Float, nullable=False)
    rank = db.Column(db.Integer, nullable=False)
    context_window = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationships
    document = db.relationship('Document')
    chunk = db.relationship('DocumentChunk')
    
    def __repr__(self):
        return f'<QueryResult query_id={self.query_id} doc_id={self.document_id} rank={self.rank}>'

class AuditLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    action = db.Column(db.String(100), nullable=False, index=True)
    resource_type = db.Column(db.String(50), nullable=False, index=True)
    resource_id = db.Column(db.Integer)
    log_details = db.Column(db.JSON)
    ip_address = db.Column(db.String(45))
    user_agent = db.Column(db.String(512))
    created_at = db.Column(db.DateTime, default=datetime.utcnow, index=True)
    
    # Relationship
    user = db.relationship('User', backref=db.backref('audit_logs', lazy='dynamic'))
    
    def __repr__(self):
        return f'<AuditLog user_id={self.user_id} action={self.action} resource={self.resource_type} id={self.resource_id}>'
