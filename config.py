import os

# General Configuration
DEBUG = True
TESTING = False

# OpenAI API Configuration
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
DEFAULT_LLM_MODEL = "gpt-4o"  # the newest OpenAI model is "gpt-4o" which was released May 13, 2024

# Embedding Models
DOCUMENT_EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
QUERY_EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
ENTITY_EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"

# Vector Store Configuration
VECTOR_DIMENSION = 768
INDEX_TYPE = "HNSW"  # Options: FLAT, IVF, HNSW
TOP_K_RETRIEVAL = 10

# Knowledge Graph Configuration
ENTITY_CONFIDENCE_THRESHOLD = 0.5
RELATION_CONFIDENCE_THRESHOLD = 0.6
MAX_ENTITY_LENGTH = 50
KNOWLEDGE_GRAPH_DEPTH = 3  # Maximum depth for multi-hop traversal

# Text Processing Configuration
CHUNK_SIZE = 512
CHUNK_OVERLAP = 128
MAX_DOCUMENT_SIZE = 10 * 1024 * 1024  # 10MB

# Query Processing
MAX_QUERY_LENGTH = 1000
QUERY_EXPANSION_TECHNIQUES = ["synonym", "entity", "llm"]
RERANKING_MODEL = "cross-encoder/ms-marco-MiniLM-L-12-v2"

# Security
ACCESS_CONTROL_ENABLED = True
DOCUMENT_LEVEL_PERMISSIONS = True
ENCRYPTION_ENABLED = True
AUDIT_LOGGING_ENABLED = True

# Caching
CACHE_TYPE = "redis"
CACHE_REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
CACHE_DEFAULT_TIMEOUT = 3600
QUERY_CACHE_TTL = 900  # 15 minutes

# Performance
MAX_THREADS = 8
BATCH_SIZE = 32

# Edge Case Handling
CORRUPTED_DOCUMENT_RECOVERY = True
AMBIGUOUS_QUERY_HANDLING = True
MIXED_LANGUAGE_PROCESSING = True

# NLP Pipeline
SPACY_MODEL = "en_core_web_sm"

# Multi-modal Processing
ENABLE_IMAGE_PROCESSING = True
ENABLE_AUDIO_PROCESSING = True
ENABLE_VIDEO_PROCESSING = True
IMAGE_EMBEDDING_MODEL = "clip"
