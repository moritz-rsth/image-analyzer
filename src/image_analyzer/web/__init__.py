"""
Web Application Factory

Creates and configures the Flask application and SocketIO instance.
Registers all blueprints and initializes the database.
"""

import os
import logging
from flask import Flask
from flask_socketio import SocketIO
from src.image_analyzer.web.config import get_secret_key
from src.image_analyzer.web.database import init_db
from src.image_analyzer.web.progress import set_socketio as set_progress_socketio
from src.image_analyzer.web.socketio_events import register_socketio_events
from src.image_analyzer.web.routes.auth import auth_bp
from src.image_analyzer.web.routes.admin import admin_bp
from src.image_analyzer.web.routes.config import config_bp
from src.image_analyzer.web.routes.upload import upload_bp, set_socketio as set_upload_socketio
from src.image_analyzer.web.routes.download import download_bp

# Configure logging to suppress harmless Socket.IO session errors
logging.getLogger('socketio').setLevel(logging.ERROR)
logging.getLogger('engineio').setLevel(logging.ERROR)


def create_app():
    """
    Create and configure the Flask application.
    
    :return: Flask application instance
    """
    # Get the root directory (where app.py is located)
    # Go up from src/image_analyzer/web/__init__.py to project root
    # __file__ is: .../image-analyzer/src/image_analyzer/web/__init__.py
    # We need to go up 3 levels: web -> image_analyzer -> src -> image-analyzer (root)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_path = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
    
    app = Flask(
        __name__,
        root_path=root_path,
        template_folder='templates',
        static_folder='static'
    )
    app.config['SECRET_KEY'] = get_secret_key()
    
    # Initialize database
    init_db()
    
    # Register blueprints (without prefix to maintain route names for url_for compatibility)
    app.register_blueprint(auth_bp)
    app.register_blueprint(admin_bp)
    app.register_blueprint(config_bp)
    app.register_blueprint(upload_bp)
    app.register_blueprint(download_bp)
    
    return app


def create_socketio(app):
    """
    Create and configure the SocketIO instance.
    
    :param app: Flask application instance
    :return: SocketIO instance
    """
    socketio = SocketIO(
        app,
        cors_allowed_origins="*",
        async_mode='threading',
        logger=False,
        engineio_logger=False
    )
    
    # Register Socket.IO event handlers
    register_socketio_events(socketio)
    
    # Set SocketIO instance in modules that need it
    set_progress_socketio(socketio)
    set_upload_socketio(socketio)
    
    return socketio


# Create app and socketio instances
app = create_app()
socketio = create_socketio(app)
