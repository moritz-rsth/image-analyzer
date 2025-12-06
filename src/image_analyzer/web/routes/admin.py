"""
Admin Routes

Handles admin functionality for user management.
"""

import os
import sqlite3
import logging
from flask import Blueprint, request, jsonify, render_template
from src.image_analyzer.web.auth import login_required, admin_required
from src.image_analyzer.web.database import (
    get_db_connection,
    upsert_user_token,
    update_user_limit,
    list_users,
    delete_user
)
from src.image_analyzer.utils import user_management
from src.image_analyzer.web.config import (
    get_database_base_path,
    get_upload_folder,
    get_output_folder,
    get_default_user_config_file
)

logger = logging.getLogger(__name__)

# Create blueprint with name for consistent endpoint naming
admin_bp = Blueprint('admin', __name__)


@admin_bp.route('/admin', endpoint='admin_home')
@login_required
@admin_required
def admin_home():
    """Render admin user management page."""
    users = list_users()
    return render_template('admin.html', users=users)


@admin_bp.route('/admin/create-token', methods=['POST'], endpoint='create_token')
@login_required
@admin_required
def admin_create_token():
    """Admin endpoint to create or update a user token with an image limit."""
    try:
        data = request.get_json()
        username = (data or {}).get('username')
        token = (data or {}).get('token')
        image_limit = int((data or {}).get('image_limit', 0))

        if not username or not token:
            return jsonify({'status': 'error', 'message': 'Username and token are required'}), 400
        if image_limit < 0:
            return jsonify({'status': 'error', 'message': 'Image limit must be >= 0'}), 400
        
        # Prevent creating users with admin username
        admin_user = os.getenv('ADMIN_USERNAME', 'admin')
        if username.lower() == admin_user.lower():
            return jsonify({'status': 'error', 'message': f'Cannot create user with admin username "{admin_user}"'}), 400

        # Check if user already exists in database
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT username FROM users WHERE username = ?", (username,))
        user_exists = cur.fetchone() is not None
        conn.close()

        # Create or update user in database
        saved_token = upsert_user_token(username, token, image_limit)
        
        # If this is a new user, create folders and initialize config
        if not user_exists:
            try:
                # Create user folders
                user_management.create_user_folders(
                    username,
                    get_database_base_path(),
                    get_upload_folder(),
                    get_output_folder()
                )
                
                # Initialize user config from template
                user_management.initialize_user_config(
                    username,
                    get_database_base_path(),
                    get_default_user_config_file()
                )
                
                logger.info(f"Created folders and initialized config for new user {username}")
            except Exception as e:
                logger.error(f"Error creating folders/config for user {username}: {str(e)}")
                # Continue even if folder creation fails (user is already in DB)
        
        return jsonify({
            'status': 'success',
            'message': f'Token set for user {username}',
            'image_limit': image_limit,
            'token': saved_token  # only returned on creation/update
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@admin_bp.route('/admin/users', methods=['GET'], endpoint='list_users')
@login_required
@admin_required
def admin_list_users():
    """List users (JSON) for admin dashboard."""
    try:
        return jsonify({'status': 'success', 'users': list_users()})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@admin_bp.route('/admin/update-limit', methods=['POST'], endpoint='update_limit')
@login_required
@admin_required
def admin_update_limit():
    """Update only the image limit for an existing user (does not change token)."""
    try:
        data = request.get_json() or {}
        username = data.get('username')
        image_limit = int(data.get('image_limit', 0))

        if not username:
            return jsonify({'status': 'error', 'message': 'Username is required'}), 400
        if image_limit < 0:
            return jsonify({'status': 'error', 'message': 'Image limit must be >= 0'}), 400
        
        # Check if user exists
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT username FROM users WHERE username = ?", (username,))
        if not cur.fetchone():
            conn.close()
            return jsonify({'status': 'error', 'message': 'User not found'}), 404
        conn.close()

        update_user_limit(username, image_limit)
        return jsonify({
            'status': 'success',
            'message': f'Image limit updated for user {username}',
            'image_limit': image_limit
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@admin_bp.route('/admin/delete-user', methods=['POST'], endpoint='delete_user')
@login_required
@admin_required
def admin_delete_user():
    """Delete a user."""
    try:
        data = request.get_json() or {}
        username = data.get('username')
        if not username:
            return jsonify({'status': 'error', 'message': 'Username required'}), 400
        delete_user(username)
        return jsonify({'status': 'success', 'message': f'User {username} deleted'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

