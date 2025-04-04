import os
import logging
import base64
import hashlib
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import tempfile

from app import app
from models import Document

logger = logging.getLogger(__name__)

# Initialize encryption key
encryption_key = None
try:
    # Get secret key from environment or generate a deterministic one from app secret
    ENCRYPTION_KEY = os.environ.get("ENCRYPTION_KEY")
    if not ENCRYPTION_KEY:
        # Derive a deterministic key from app secret
        app_secret = app.secret_key or "default_secret"
        salt = b'graphrag_system_salt'  # Fixed salt for deterministic key
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000
        )
        ENCRYPTION_KEY = base64.urlsafe_b64encode(kdf.derive(app_secret.encode()))
    
    # Create Fernet cipher
    encryption_key = Fernet(ENCRYPTION_KEY)
    logger.info("Encryption initialized successfully")
except Exception as e:
    logger.error(f"Error initializing encryption: {e}")

def encrypt_file(file_path: str) -> str:
    """
    Encrypt a file and return the encrypted file path
    
    Args:
        file_path: Path to the file to encrypt
        
    Returns:
        Path to the encrypted file
    """
    if not encryption_key:
        logger.warning("Encryption not available, returning original file path")
        return file_path
        
    try:
        # Read the file
        with open(file_path, 'rb') as f:
            file_data = f.read()
            
        # Encrypt the data
        encrypted_data = encryption_key.encrypt(file_data)
        
        # Create encrypted file
        encrypted_path = f"{file_path}.encrypted"
        with open(encrypted_path, 'wb') as f:
            f.write(encrypted_data)
            
        logger.info(f"File encrypted: {file_path} -> {encrypted_path}")
        return encrypted_path
    except Exception as e:
        logger.error(f"Error encrypting file {file_path}: {e}")
        return file_path

def decrypt_file(file_path: str) -> str:
    """
    Decrypt a file and return the temporary path to the decrypted file
    
    Args:
        file_path: Path to the encrypted file
        
    Returns:
        Path to the decrypted file (temporary)
    """
    if not encryption_key or not file_path.endswith('.encrypted'):
        logger.warning("File not encrypted or encryption not available")
        return file_path
        
    try:
        # Read the encrypted file
        with open(file_path, 'rb') as f:
            encrypted_data = f.read()
            
        # Decrypt the data
        decrypted_data = encryption_key.decrypt(encrypted_data)
        
        # Create temporary file
        fd, temp_path = tempfile.mkstemp()
        os.close(fd)
        
        # Write decrypted data to temporary file
        with open(temp_path, 'wb') as f:
            f.write(decrypted_data)
            
        logger.info(f"File decrypted: {file_path} -> {temp_path}")
        return temp_path
    except Exception as e:
        logger.error(f"Error decrypting file {file_path}: {e}")
        return file_path

def is_file_encrypted(document: Document) -> bool:
    """
    Check if a document's file is encrypted
    
    Args:
        document: Document object
        
    Returns:
        True if the file is encrypted, False otherwise
    """
    return document.is_encrypted or document.file_path.endswith('.encrypted')

def hash_password(password: str) -> str:
    """
    Generate a secure password hash
    
    Args:
        password: Plain text password
        
    Returns:
        Hashed password
    """
    # This is implemented in the User model using werkzeug.security
    # This function is for compatibility with external systems if needed
    from werkzeug.security import generate_password_hash
    return generate_password_hash(password)

def verify_password(hashed_password: str, plain_password: str) -> bool:
    """
    Verify a password against a hash
    
    Args:
        hashed_password: Stored password hash
        plain_password: Plain text password to verify
        
    Returns:
        True if password matches, False otherwise
    """
    # This is implemented in the User model using werkzeug.security
    # This function is for compatibility with external systems if needed
    from werkzeug.security import check_password_hash
    return check_password_hash(hashed_password, plain_password)

def generate_secure_token() -> str:
    """
    Generate a secure random token for authentication or verification
    
    Returns:
        Secure random token
    """
    return base64.urlsafe_b64encode(os.urandom(32)).decode()

def hash_text(text: str) -> str:
    """
    Generate a hash of text
    
    Args:
        text: Text to hash
        
    Returns:
        Hashed text
    """
    return hashlib.sha256(text.encode()).hexdigest()
