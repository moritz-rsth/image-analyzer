"""
Download Routes

Handles file downloads and serving uploaded files.
"""

import os
from flask import Blueprint, jsonify, send_from_directory, session
from src.image_analyzer.web.auth import login_required
from src.image_analyzer.web.file_management import get_user_folder, get_user_zip_folder
from src.image_analyzer.web.config import get_upload_folder, get_output_folder

# Create blueprint with name for consistent endpoint naming
download_bp = Blueprint('download', __name__)


@download_bp.route('/download/<path:filename>', endpoint='download_file')
@login_required
def download_file(filename):
    """Download file from user's database folder (ZIP files) or output folder (other files)"""
    user_id = session.get('user_id')
    
    # If filename includes user_id prefix, remove it
    if filename.startswith(f"{user_id}/"):
        filename = filename[len(user_id) + 1:]
    
    # Check if it's a ZIP file - ZIP files are in database/user_id/
    if filename.endswith('.zip') and filename.startswith('Image-Analyzer_run_'):
        user_zip_folder = get_user_zip_folder(user_id)
        file_path = os.path.abspath(os.path.join(user_zip_folder, filename))
        
        # Security check: ensure the resolved file path is within the user's zip folder
        if not file_path.startswith(os.path.abspath(user_zip_folder)):
            return jsonify({'status': 'error', 'message': 'Invalid file path'}), 403
        
        # Check that file exists and is a file (not a directory)
        if not os.path.exists(file_path):
            return jsonify({'status': 'error', 'message': 'File not found'}), 404
        
        if not os.path.isfile(file_path):
            return jsonify({'status': 'error', 'message': 'Not a file'}), 403
        
        # Use the relative filename for send_from_directory (needs relative path from directory)
        return send_from_directory(user_zip_folder, filename, as_attachment=True)
    else:
        # Other files are still in outputs/user_id/
        user_output_folder = get_user_folder(user_id, get_output_folder())
        file_path = os.path.abspath(os.path.join(user_output_folder, filename))
        
        # Security check: ensure the resolved file path is within the user's output folder
        if not file_path.startswith(os.path.abspath(user_output_folder)):
            return jsonify({'status': 'error', 'message': 'Invalid file path'}), 403
        
        # Check that file exists and is a file (not a directory)
        if not os.path.exists(file_path):
            return jsonify({'status': 'error', 'message': 'File not found'}), 404
        
        if not os.path.isfile(file_path):
            return jsonify({'status': 'error', 'message': 'Not a file'}), 403
        
        # Use the relative filename for send_from_directory (needs relative path from directory)
        return send_from_directory(user_output_folder, filename, as_attachment=True)


@download_bp.route('/uploads/<path:filename>', endpoint='serve_upload')
def serve_upload(filename):
    """Serve uploaded images publicly so Replicate can access them via URL"""
    upload_folder = get_upload_folder()
    # Security: only serve files from uploads folder, prevent directory traversal
    safe_path = os.path.normpath(os.path.join(upload_folder, filename))
    if not safe_path.startswith(os.path.abspath(upload_folder)):
        return jsonify({'status': 'error', 'message': 'Invalid path'}), 403
    return send_from_directory(upload_folder, filename)

