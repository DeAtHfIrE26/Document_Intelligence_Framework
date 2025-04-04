import os
import logging
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm import DeclarativeBase
from flask_login import LoginManager

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Base class for SQLAlchemy models
class Base(DeclarativeBase):
    pass

# Initialize extensions
db = SQLAlchemy(model_class=Base)
login_manager = LoginManager()

# Create Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET")

# Configure database
app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get("DATABASE_URL")
app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
    "pool_recycle": 300,
    "pool_pre_ping": True,
}
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

# Configure file uploads
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024  # 100MB limit
app.config["UPLOAD_FOLDER"] = "uploads"

# Initialize extensions with the app
db.init_app(app)
login_manager.init_app(app)
login_manager.login_view = "web.login"

# Create uploads directory if it doesn't exist
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# Initialize database tables
with app.app_context():
    # Import models to ensure they're registered with SQLAlchemy
    import models
    
    # Create database tables
    db.create_all()
    logger.info("Database tables created successfully")

# Register blueprints
from controllers.web import web
from controllers.api import api

app.register_blueprint(web)
app.register_blueprint(api, url_prefix="/api")

# Setup login manager
@login_manager.user_loader
def load_user(user_id):
    from models import User
    return User.query.get(int(user_id))

# Import and initialize services
from services import document_processor, knowledge_graph, vector_store

# Initialize services
with app.app_context():
    try:
        document_processor.init_app(app)
        knowledge_graph.init_app(app)
        vector_store.init_app(app)
        logger.info("Services initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing services: {e}")

logger.info("Application initialized successfully")
