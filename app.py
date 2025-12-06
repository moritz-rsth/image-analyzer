"""
Main Application Entry Point

This file serves as a compatibility wrapper for Procfile/Dockerfile.
The actual application is created in src.image_analyzer.web.
"""

import os
from src.image_analyzer.web import app, socketio
from src.image_analyzer.web.config import get_port

# Export app and socketio for Procfile/Dockerfile compatibility
# Procfile uses: gunicorn --worker-class eventlet --workers 1 --bind 0.0.0.0:$PORT --timeout 120 app:app
# Dockerfile uses: CMD gunicorn --worker-class eventlet --workers 1 --bind 0.0.0.0:${PORT:-5000} --timeout 120 app:app

if __name__ == '__main__':
    # Only run with debug in development
    if os.getenv('FLASK_ENV') == 'development':
        PORT = get_port()
        socketio.run(app, debug=True, host='0.0.0.0', port=PORT)
    else:
        # In production, Gunicorn will handle this
        # Railway and GCP use Gunicorn via Procfile/startup-script
        pass
