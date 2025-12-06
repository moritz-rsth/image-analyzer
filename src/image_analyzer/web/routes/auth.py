"""
Authentication Routes

Handles login and logout functionality.
"""

import os
from flask import Blueprint, request, jsonify, session, redirect, url_for, render_template
from werkzeug.security import check_password_hash
from src.image_analyzer.web.database import verify_user_token
from src.image_analyzer.web.progress import user_processing, clear_processing_state

# Create blueprint with name for consistent endpoint naming
auth_bp = Blueprint('auth', __name__)


@auth_bp.route('/login', methods=['GET', 'POST'], endpoint='login')
def login():
    """Login: admin uses password from env hash; users use token stored in DB."""
    if request.method == 'POST':
        data = request.get_json() or {}
        username = data.get('username')
        secret = data.get('secret')  # admin password or user token

        if not username or not secret:
            return jsonify({'status': 'error', 'message': 'Username and secret required'}), 400

        admin_user = os.getenv('ADMIN_USERNAME', 'admin')
        admin_hash = os.getenv('ADMIN_HASH_BCRYPT')

        if username == admin_user:
            if not admin_hash:
                return jsonify({'status': 'error', 'message': 'Server configuration error'}), 500
            if not check_password_hash(admin_hash, secret):
                return jsonify({'status': 'error', 'message': 'Invalid credentials'}), 401
            session['user_id'] = admin_user
            session['role'] = 'admin'
            session.permanent = True
            return jsonify({'status': 'success', 'message': 'Admin login successful'})

        # regular user via token stored in DB
        if not verify_user_token(username, secret):
            return jsonify({'status': 'error', 'message': 'Invalid credentials'}), 401
        session['user_id'] = username
        session['role'] = 'user'
        session.permanent = True
        return jsonify({'status': 'success', 'message': 'User login successful'})

    # GET request - show login page
    if 'user_id' in session:
        return redirect(url_for('upload.upload_file'))
    return render_template('login.html')


@auth_bp.route('/logout', endpoint='logout')
def logout():
    """Logout and clear session"""
    user_id = session.get('user_id')
    if user_id and user_id in user_processing:
        clear_processing_state(user_id)
    session.clear()
    return redirect(url_for('auth.login'))

