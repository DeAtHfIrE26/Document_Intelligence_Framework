import os
import mimetypes
import json
import hashlib
import time
import logging
from typing import Any, Dict, Optional, Union
import redis

logger = logging.getLogger(__name__)

# Initialize Redis connection if REDIS_URL is available
redis_client = None
try:
    from app import app
    redis_url = app.config.get('CACHE_REDIS_URL')
    if redis_url:
        redis_client = redis.from_url(redis_url)
        logger.info(f"Redis connected: {redis_url}")
except Exception as e:
    logger.warning(f"Redis connection failed: {e}")

def get_file_size(file_path: str) -> int:
    """
    Get the size of a file in bytes
    
    Args:
        file_path: Path to the file
        
    Returns:
        Size in bytes
    """
    try:
        return os.path.getsize(file_path)
    except Exception as e:
        logger.error(f"Error getting file size for {file_path}: {e}")
        return 0

def get_mime_type(file_path: str) -> str:
    """
    Get MIME type of a file
    
    Args:
        file_path: Path to the file
        
    Returns:
        MIME type string
    """
    mime_type, _ = mimetypes.guess_type(file_path)
    return mime_type or 'application/octet-stream'

def extract_filename(file_path: str) -> str:
    """
    Extract filename without extension from a path
    
    Args:
        file_path: Path to the file
        
    Returns:
        Filename without extension
    """
    basename = os.path.basename(file_path)
    return os.path.splitext(basename)[0]

def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Human-readable string
    """
    # Define units and thresholds
    units = ['B', 'KB', 'MB', 'GB', 'TB']
    threshold = 1024.0
    
    # Find appropriate unit
    unit_index = 0
    size = float(size_bytes)
    while size >= threshold and unit_index < len(units) - 1:
        size /= threshold
        unit_index += 1
    
    # Format with appropriate precision
    if unit_index == 0:
        return f"{int(size)} {units[unit_index]}"
    else:
        return f"{size:.2f} {units[unit_index]}"

def sanitize_html(html_content: str) -> str:
    """
    Sanitize HTML content to prevent XSS attacks
    
    Args:
        html_content: HTML content to sanitize
        
    Returns:
        Sanitized HTML
    """
    # This is a simplified implementation
    # For production use a proper HTML sanitization library like bleach
    
    # Remove script tags
    html_content = html_content.replace('<script>', '&lt;script&gt;').replace('</script>', '&lt;/script&gt;')
    
    # Remove onclick and other event handlers
    html_content = html_content.replace('onclick=', 'data-blocked-onclick=')
    html_content = html_content.replace('onerror=', 'data-blocked-onerror=')
    html_content = html_content.replace('onload=', 'data-blocked-onload=')
    
    return html_content

def set_cache(key: str, value: Any, ttl: int = 3600) -> bool:
    """
    Set a value in the cache
    
    Args:
        key: Cache key
        value: Value to cache
        ttl: Time to live in seconds
        
    Returns:
        True if successful, False otherwise
    """
    if not redis_client:
        return False
        
    try:
        # Serialize the value if it's not a simple string
        if not isinstance(value, (str, bytes)):
            value = json.dumps(value)
            
        redis_client.setex(key, ttl, value)
        return True
    except Exception as e:
        logger.error(f"Error setting cache for key {key}: {e}")
        return False

def get_cache(key: str) -> Optional[Any]:
    """
    Get a value from the cache
    
    Args:
        key: Cache key
        
    Returns:
        Cached value or None if not found
    """
    if not redis_client:
        return None
        
    try:
        value = redis_client.get(key)
        if not value:
            return None
            
        # Try to deserialize JSON
        try:
            return json.loads(value)
        except:
            # Return as is if not JSON
            return value
    except Exception as e:
        logger.error(f"Error getting cache for key {key}: {e}")
        return None

def clear_cache(key: str) -> bool:
    """
    Clear a value from the cache
    
    Args:
        key: Cache key
        
    Returns:
        True if successful, False otherwise
    """
    if not redis_client:
        return False
        
    try:
        redis_client.delete(key)
        return True
    except Exception as e:
        logger.error(f"Error clearing cache for key {key}: {e}")
        return False

def generate_cache_key(prefix: str, *args) -> str:
    """
    Generate a deterministic cache key from arguments
    
    Args:
        prefix: Key prefix
        *args: Arguments to include in key
        
    Returns:
        Cache key string
    """
    key_parts = [prefix]
    
    for arg in args:
        if arg is None:
            continue
            
        if isinstance(arg, dict):
            # Sort dict items for consistent hashing
            arg_str = json.dumps(arg, sort_keys=True)
        else:
            arg_str = str(arg)
            
        key_parts.append(arg_str)
    
    key_str = "_".join(key_parts)
    
    # Hash if the key is too long
    if len(key_str) > 100:
        key_hash = hashlib.md5(key_str.encode()).hexdigest()
        return f"{prefix}_{key_hash}"
    
    return key_str

def timeit(func):
    """
    Decorator to measure function execution time
    
    Args:
        func: Function to time
        
    Returns:
        Wrapped function
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logger.debug(f"{func.__name__} took {end_time - start_time:.4f} seconds")
        return result
    return wrapper
