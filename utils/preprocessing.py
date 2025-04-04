import re
import logging
import unicodedata
import string
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

def clean_text(text: str) -> str:
    """
    Clean text by removing extra whitespace, fixing common issues
    
    Args:
        text: Text to clean
        
    Returns:
        Cleaned text
    """
    if not text:
        return ""
        
    # Replace multiple whitespace with single space
    text = re.sub(r'\s+', ' ', text)
    
    # Fix newlines
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)  # Single newlines to space
    text = re.sub(r'\n{2,}', '\n\n', text)  # Multiple newlines to double newline
    
    # Remove control characters except newlines and tabs
    text = ''.join(ch for ch in text if unicodedata.category(ch)[0] != 'C' 
                  or ch in ['\n', '\t'])
    
    # Replace non-breaking spaces with regular spaces
    text = text.replace('\xa0', ' ')
    
    # Fix common OCR errors
    text = text.replace('…', '...')
    text = text.replace('—', '-')
    text = text.replace(''', "'")
    text = text.replace(''', "'")
    text = text.replace('"', '"')
    text = text.replace('"', '"')
    
    # Fix encoding issues
    try:
        text = unicodedata.normalize('NFKD', text)
    except:
        pass
    
    # Trim whitespace
    text = text.strip()
    
    return text

def normalize_text(text: str) -> str:
    """
    Normalize text for comparison or search (lowercase, remove punctuation)
    
    Args:
        text: Text to normalize
        
    Returns:
        Normalized text
    """
    if not text:
        return ""
        
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Trim whitespace
    text = text.strip()
    
    return text

def split_into_sentences(text: str) -> List[str]:
    """
    Split text into sentences
    
    Args:
        text: Text to split
        
    Returns:
        List of sentences
    """
    # Clean the text first
    text = clean_text(text)
    
    # Simple sentence splitting
    # This is a basic implementation, more sophisticated NLP libraries would do better
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s', text)
    
    # Remove empty sentences
    sentences = [s.strip() for s in sentences if s.strip()]
    
    return sentences

def detect_language_hint(text: str) -> Optional[str]:
    """
    Get a hint of the text language based on character frequency
    
    Args:
        text: Text to analyze
        
    Returns:
        Language code hint or None
    """
    # This is a very simple heuristic
    # For production use a proper language detection library
    
    # Sample common character patterns
    patterns = {
        'en': ['the', 'and', 'ing', 'tion'],
        'es': ['el', 'la', 'que', 'de'],
        'fr': ['le', 'la', 'les', 'de'],
        'de': ['der', 'die', 'das', 'und'],
        'zh': ['的', '是', '了', '在'],
        'ja': ['は', 'を', 'に', 'の'],
        'ru': ['и', 'в', 'не', 'на']
    }
    
    text_lower = text.lower()
    scores = {}
    
    for lang, patterns_list in patterns.items():
        score = sum(text_lower.count(pattern) for pattern in patterns_list)
        scores[lang] = score
    
    if not scores:
        return None
        
    max_lang = max(scores, key=scores.get)
    
    # Only return if we have some confidence
    if scores[max_lang] > 2:
        return max_lang
    
    return None

def extract_keywords(text: str, max_keywords: int = 10) -> List[str]:
    """
    Extract keywords from text using simple frequency
    
    Args:
        text: Text to analyze
        max_keywords: Maximum number of keywords to extract
        
    Returns:
        List of keywords
    """
    # This is a very simple implementation
    # For production use a proper NLP library
    
    # Normalize text
    text = normalize_text(text)
    
    # Split into words
    words = text.split()
    
    # Remove common stopwords (simplified list)
    stopwords = {
        'the', 'a', 'an', 'and', 'or', 'but', 'if', 'because', 'as', 'what',
        'which', 'this', 'that', 'these', 'those', 'then', 'just', 'so', 'than',
        'such', 'both', 'through', 'about', 'for', 'is', 'of', 'while', 'during',
        'to', 'from', 'in', 'on', 'at', 'by', 'with', 'about', 'against', 'between',
        'into', 'through', 'without', 'be', 'been', 'being', 'have', 'has', 'had',
        'having', 'do', 'does', 'did', 'doing', 'would', 'should', 'could', 'ought',
        'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them',
        'my', 'your', 'his', 'its', 'our', 'their', 'mine', 'yours', 'hers', 'ours', 'theirs'
    }
    
    filtered_words = [w for w in words if w not in stopwords and len(w) > 2]
    
    # Count word frequency
    word_counts = {}
    for word in filtered_words:
        word_counts[word] = word_counts.get(word, 0) + 1
    
    # Sort by frequency
    sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
    
    # Return top keywords
    return [word for word, count in sorted_words[:max_keywords]]

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> List[Dict[str, Any]]:
    """
    Split text into overlapping chunks
    
    Args:
        text: Text to chunk
        chunk_size: Maximum size of each chunk
        overlap: Overlap between chunks
        
    Returns:
        List of chunk dictionaries with content, start and end positions
    """
    chunks = []
    
    # Clean text first
    text = clean_text(text)
    
    # Split into chunks
    start = 0
    while start < len(text):
        # Calculate end position with overlap
        end = min(start + chunk_size, len(text))
        
        # Try to end at sentence boundary if possible
        if end < len(text):
            # Look for sentence boundary within the last 20% of the chunk
            boundary_search_start = max(start, end - int(chunk_size * 0.2))
            potential_boundary = text.rfind('. ', boundary_search_start, end)
            if potential_boundary != -1:
                end = potential_boundary + 1  # Include the period
        
        # Create chunk
        chunks.append({
            'content': text[start:end],
            'start_char': start,
            'end_char': end
        })
        
        # Move to next chunk with overlap
        start = end - overlap
        
        # Avoid getting stuck at the same position
        if start >= end:
            start = end
    
    return chunks

def dedup_text(texts: List[str]) -> List[str]:
    """
    Remove duplicate or near-duplicate text snippets
    
    Args:
        texts: List of text snippets
        
    Returns:
        Deduplicated list of text snippets
    """
    unique_texts = []
    normalized_texts = set()
    
    for text in texts:
        # Create a normalized version for comparison
        normalized = normalize_text(text)
        
        # Skip if too similar to existing text
        if normalized in normalized_texts:
            continue
            
        # Add to results
        unique_texts.append(text)
        normalized_texts.add(normalized)
    
    return unique_texts
