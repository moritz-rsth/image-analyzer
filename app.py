from flask import Flask, request, render_template, send_from_directory, jsonify, session, redirect, url_for
from flask_socketio import SocketIO, emit, join_room
from functools import wraps
import os
import sqlite3
import pandas as pd
from datetime import datetime
from src.image_analyzer.pipeline.ia_pipeline_deployment import IA
from src.image_analyzer.utils import user_management
import yaml
import shutil
import threading
import time
import zipfile
import tempfile
import hashlib
import secrets
from werkzeug.security import generate_password_hash, check_password_hash
from dotenv import load_dotenv
import logging

load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'your-secret-key-change-in-production')

# Detect deployment platform and configure accordingly
DEPLOYMENT_PLATFORM = os.getenv('DEPLOYMENT_PLATFORM', 'local')
PORT = int(os.getenv('PORT', 5000))

# Auto-configure APP_BASE_URL (Railway or local). No manual env required.
# Used by Replicate API to access uploaded images via public URLs.
railway_domain = os.getenv('RAILWAY_PUBLIC_DOMAIN')
if railway_domain:
    DEPLOYMENT_PLATFORM = 'railway'
    os.environ['APP_BASE_URL'] = f"https://{railway_domain}"
else:
    # Local development: defaults to localhost
    os.environ['APP_BASE_URL'] = f"http://localhost:{PORT}"

# Configure logging to suppress harmless Socket.IO session errors
logging.getLogger('socketio').setLevel(logging.ERROR)
logging.getLogger('engineio').setLevel(logging.ERROR)

socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading', logger=False, engineio_logger=False)

UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
# New config structure
DEPLOYMENT_CONFIG_FILE = os.path.join('config', 'deploymentConfig.yaml')
DEFAULT_USER_CONFIG_FILE = os.path.join('config', 'userConfig.yaml')
# Legacy config files (kept for backward compatibility)
DEFAULT_CONFIG_FILE = os.path.join('config', 'configuration_deployment.yaml')
CURR_CONFIG_FILE = os.path.join('config', 'configuration.yaml')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Database path configuration (prefers explicit env, then Railway volume, else local folder)
DATABASE_BASE_PATH = os.getenv('DATABASE_PATH') or os.getenv('RAILWAY_VOLUME_MOUNT_PATH') or './database'
os.environ['DATABASE_PATH'] = DATABASE_BASE_PATH  # expose for downstream code/tools
os.makedirs(DATABASE_BASE_PATH, exist_ok=True)
DB_FILE = os.path.join(DATABASE_BASE_PATH, 'user.db')

# Per-user processing state: {user_id: {'session_id': str, 'status': str, 'progress': dict, 'started_at': timestamp}}
user_processing = {}


# --- Database helpers ---
def get_db_connection():
    """Create a SQLite connection (auto-commit)"""
    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initialize the user database (users table stores token hashes and limits)."""
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            token_hash TEXT NOT NULL,
            image_limit INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.commit()
    conn.close()


def upsert_user_token(username: str, raw_token: str, image_limit: int):
    """Create or replace a user's token hash and image limit."""
    token_hash = generate_password_hash(raw_token)
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO users (username, token_hash, image_limit)
        VALUES (?, ?, ?)
        ON CONFLICT(username) DO UPDATE SET
            token_hash=excluded.token_hash,
            image_limit=excluded.image_limit
        """,
        (username, token_hash, image_limit),
    )
    conn.commit()
    conn.close()
    return raw_token  # return raw token so caller can show it once


def verify_user_token(username: str, raw_token: str) -> bool:
    """Verify a user's token against the stored hash."""
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT token_hash FROM users WHERE username = ?", (username,))
    row = cur.fetchone()
    conn.close()
    if not row:
        return False
    return check_password_hash(row["token_hash"], raw_token)


def update_user_limit(username: str, image_limit: int):
    """Update only the image limit for an existing user (does not change token)."""
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("UPDATE users SET image_limit = ? WHERE username = ?", (image_limit, username))
    conn.commit()
    conn.close()


def get_user_quota(username: str) -> int:
    """Return remaining image quota for user (defaults to 0 if missing)."""
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT image_limit FROM users WHERE username = ?", (username,))
    row = cur.fetchone()
    conn.close()
    if not row:
        return 0
    return row["image_limit"] or 0


def consume_user_quota(username: str, count: int) -> tuple[bool, int]:
    """Attempt to deduct count from user's quota. Returns (ok, remaining)."""
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT image_limit FROM users WHERE username = ?", (username,))
    row = cur.fetchone()
    if not row:
        conn.close()
        return False, 0
    remaining = row["image_limit"] or 0
    if remaining < count:
        conn.close()
        return False, remaining
    new_remaining = remaining - count
    cur.execute("UPDATE users SET image_limit = ? WHERE username = ?", (new_remaining, username))
    conn.commit()
    conn.close()
    return True, new_remaining


def list_users():
    """Return list of users with basic info."""
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT username, image_limit, created_at FROM users ORDER BY username")
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


def delete_user(username: str):
    """
    Delete user by username.
    Deletes user from database and removes all associated folders.
    """
    # Delete user folders first (before database deletion)
    try:
        delete_results = user_management.delete_user_folders(
            username, 
            DATABASE_BASE_PATH, 
            UPLOAD_FOLDER, 
            OUTPUT_FOLDER
        )
        if delete_results['errors']:
            logging.warning(f"Some errors occurred while deleting folders for user {username}: {delete_results['errors']}")
    except Exception as e:
        logging.error(f"Error deleting folders for user {username}: {str(e)}")
        # Continue with database deletion even if folder deletion fails
    
    # Delete user from database
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("DELETE FROM users WHERE username = ?", (username,))
    conn.commit()
    conn.close()
    logging.info(f"Deleted user {username} from database")


# Initialize database on startup
init_db()

def login_required(f):
    """Decorator to require login for routes"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function


def admin_required(f):
    """Decorator to require admin role"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if session.get('role') != 'admin':
            return jsonify({'status': 'error', 'message': 'Admin privileges required'}), 403
        return f(*args, **kwargs)
    return decorated_function

def get_user_folder(user_id, base_folder):
    """Get user-specific folder path (returns absolute path)"""
    user_folder = os.path.join(base_folder, user_id)
    user_folder = os.path.abspath(user_folder)  # Always use absolute path
    os.makedirs(user_folder, exist_ok=True)
    return user_folder

def get_user_zip_folder(user_id):
    """Get user-specific folder for ZIP files in database/user_id/ (returns absolute path)"""
    user_zip_folder = os.path.join(DATABASE_BASE_PATH, user_id)
    user_zip_folder = os.path.abspath(user_zip_folder)  # Always use absolute path
    os.makedirs(user_zip_folder, exist_ok=True)
    return user_zip_folder

def cleanup_old_results(user_id, keep_count=3):
    """
    Cleanup old results, keeping only the last N runs per user.
    
    :param user_id: User ID
    :param keep_count: Number of recent runs to keep (default: 3)
    """
    try:
        # ZIP files are stored in database/user_id/
        user_zip_folder = get_user_zip_folder(user_id)
        # Pipeline outputs are still in outputs/user_id/
        user_output_folder = get_user_folder(user_id, OUTPUT_FOLDER)
        
        # Get all ZIP files for this user from database/user_id/
        zip_files = []
        if os.path.exists(user_zip_folder):
            for filename in os.listdir(user_zip_folder):
                if filename.endswith('.zip') and filename.startswith('Image-Analyzer_run_'):
                    zip_path = os.path.join(user_zip_folder, filename)
                    if os.path.isfile(zip_path):
                        try:
                            timestamp_str = filename.replace('Image-Analyzer_run_', '').replace('.zip', '')
                            mtime = os.path.getmtime(zip_path)
                            zip_files.append({
                                'filename': filename,
                                'path': zip_path,
                                'timestamp': timestamp_str,
                                'mtime': mtime
                            })
                        except Exception as e:
                            print(f"Error parsing timestamp from {filename}: {e}")
                            continue
        
        # Sort by modification time (newest first)
        zip_files.sort(key=lambda x: x['mtime'], reverse=True)
        
        # Delete old ZIP files (keep only keep_count most recent)
        deleted_zips = 0
        for zip_info in zip_files[keep_count:]:
            try:
                os.remove(zip_info['path'])
                deleted_zips += 1
                print(f"Deleted old ZIP: {zip_info['filename']}")
            except Exception as e:
                print(f"Error deleting ZIP {zip_info['filename']}: {e}")
        
        # Also delete corresponding run folders from outputs/user_id/
        deleted_folders = 0
        for zip_info in zip_files[keep_count:]:
            timestamp_str = zip_info['timestamp']
            folder_name = f"Image-Analyzer_run_{timestamp_str}"
            folder_path = os.path.join(user_output_folder, folder_name)
            
            if os.path.exists(folder_path) and os.path.isdir(folder_path):
                try:
                    shutil.rmtree(folder_path)
                    deleted_folders += 1
                    print(f"Deleted old run folder: {folder_name}")
                except Exception as e:
                    print(f"Error deleting folder {folder_name}: {e}")
        
        if deleted_zips > 0 or deleted_folders > 0:
            print(f"Cleanup completed: deleted {deleted_zips} ZIP files and {deleted_folders} folders for user {user_id}")
        
    except Exception as e:
        print(f"Error during cleanup for user {user_id}: {e}")


def delete_batch_folder(batch_folder):
    """
    Delete the batch folder and all its contents.
    
    :param batch_folder: Path to the batch folder to delete
    """
    try:
        if os.path.exists(batch_folder) and os.path.isdir(batch_folder):
            shutil.rmtree(batch_folder)
            print(f"Deleted batch folder: {batch_folder}")
        else:
            print(f"Batch folder does not exist: {batch_folder}")
    except Exception as e:
        print(f"Error deleting batch folder {batch_folder}: {e}")

def get_user_results_history(user_id, max_results=3):
    """
    Get the last N results for a user.
    
    :param user_id: User ID
    :param max_results: Maximum number of results to return (default: 3)
    :return: List of result dictionaries with filename, download_path, timestamp, and formatted_date
    """
    try:
        # ZIP files are stored in database/user_id/
        user_zip_folder = get_user_zip_folder(user_id)
        
        # Get all ZIP files for this user from database/user_id/
        zip_files = []
        if os.path.exists(user_zip_folder):
            for filename in os.listdir(user_zip_folder):
                if filename.endswith('.zip') and filename.startswith('Image-Analyzer_run_'):
                    zip_path = os.path.join(user_zip_folder, filename)
                    if os.path.isfile(zip_path):
                        try:
                            timestamp_str = filename.replace('Image-Analyzer_run_', '').replace('.zip', '')
                            mtime = os.path.getmtime(zip_path)
                            # Parse timestamp for display
                            try:
                                dt = datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S')
                                formatted_date = dt.strftime('%Y-%m-%d %H:%M:%S')
                            except:
                                formatted_date = timestamp_str
                            
                            zip_files.append({
                                'filename': filename,
                                'download_path': f"{user_id}/{filename}",
                                'timestamp': timestamp_str,
                                'formatted_date': formatted_date,
                                'mtime': mtime
                            })
                        except Exception as e:
                            print(f"Error parsing timestamp from {filename}: {e}")
                            continue
        
        # Sort by modification time (newest first) and keep only last N
        zip_files.sort(key=lambda x: x['mtime'], reverse=True)
        return zip_files[:max_results]
        
    except Exception as e:
        print(f"Error getting results history for user {user_id}: {e}")
        return []

def progress_callback(progress_data, user_session_id):
    """Callback function to emit progress updates via WebSocket"""
    if user_session_id in user_processing:
        processing_info = user_processing[user_session_id]
        room_id = processing_info.get('session_id')
        
        # Update stored progress state
        processing_info['progress'] = progress_data
        processing_info['status'] = progress_data.get('status', 'processing')
        
        # Try to emit progress update, but don't fail if session is disconnected
        try:
            socketio.emit('progress_update', progress_data, room=room_id, namespace='/')
            # Small sleep to allow the event to be processed and sent immediately
            time.sleep(0.01)
        except (KeyError, RuntimeError) as e:
            # Session disconnected - silently ignore but keep progress state
            print(f"Could not send progress update (session disconnected): {e}")

def deep_merge_dict(base_dict, override_dict):
    """
    Deep merge two dictionaries. Values from override_dict take precedence.
    
    :param base_dict: Base dictionary (deployment config)
    :param override_dict: Override dictionary (user config)
    :return: Merged dictionary
    """
    result = base_dict.copy()
    
    for key, value in override_dict.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            # Recursively merge nested dictionaries
            result[key] = deep_merge_dict(result[key], value)
        else:
            # Override with user value
            result[key] = value
    
    return result


def filter_deployment_fields(config_dict):
    """
    Filter out deployment-only fields from config, keeping only user-modifiable fields.
    
    :param config_dict: Full configuration dictionary
    :return: Filtered dictionary with only user-modifiable fields
    """
    filtered = {}
    
    if 'features' in config_dict:
        filtered['features'] = {}
        for feature_name, feature_config in config_dict['features'].items():
            filtered_feature = {}
            
            # Keep active flag
            if 'active' in feature_config:
                filtered_feature['active'] = feature_config['active']
            
            # Keep parameters, but exclude replicate_model_id
            if 'parameters' in feature_config:
                filtered_params = {}
                for param_key, param_value in feature_config['parameters'].items():
                    if param_key != 'replicate_model_id':
                        filtered_params[param_key] = param_value
                if filtered_params:
                    filtered_feature['parameters'] = filtered_params
            
            if filtered_feature:
                filtered['features'][feature_name] = filtered_feature
    
    if 'general' in config_dict:
        filtered['general'] = {}
        general = config_dict['general']
        # Keep user-modifiable general settings, exclude system paths
        for key in ['debug_image_count', 'debug_mode', 'logs', 'output_formats', 'summary_stats', 'verbose']:
            if key in general:
                filtered['general'][key] = general[key]
    
    return filtered


def get_deployment_config_path():
    """
    Get the path to the deployment configuration file.
    
    :return: Path to deploymentConfig.yaml
    """
    return DEPLOYMENT_CONFIG_FILE


def get_user_config_path(user_id=None):
    """
    Get the path to the user-specific config file.
    User-specific configs are stored in database/user_id/userConfig.yaml.
    If user_id is None, tries to get it from session.
    Returns the user-specific path if user exists, otherwise returns default user config.
    """
    if user_id is None:
        user_id = session.get('user_id')
    
    if user_id:
        # Store user-specific configs in database/user_id/userConfig.yaml
        return user_management.get_user_config_path(user_id, DATABASE_BASE_PATH)
    return DEFAULT_USER_CONFIG_FILE


def load_config(user_id=None):
    """
    Load and merge configuration for a user.
    Merges deploymentConfig.yaml (base) with userConfig.yaml (overrides).
    
    :param user_id: Optional user ID. If None, tries to get from session.
    :return: Merged configuration dictionary
    """
    # Load deployment config (base)
    deployment_config_path = get_deployment_config_path()
    if not os.path.exists(deployment_config_path):
        # Fallback to legacy config if new structure doesn't exist
        deployment_config_path = DEFAULT_CONFIG_FILE
    
    try:
        with open(deployment_config_path, 'r') as file:
            deployment_config = yaml.safe_load(file) or {}
    except Exception as e:
        logging.error(f"Error loading deployment config: {str(e)}")
        deployment_config = {}
    
    # Load user config (overrides)
    user_config_path = get_user_config_path(user_id)
    if user_id and os.path.exists(user_config_path):
        try:
            with open(user_config_path, 'r') as file:
                user_config = yaml.safe_load(file) or {}
        except Exception as e:
            logging.error(f"Error loading user config for {user_id}: {str(e)}")
            user_config = {}
    elif os.path.exists(DEFAULT_USER_CONFIG_FILE):
        # Use default user config template
        try:
            with open(DEFAULT_USER_CONFIG_FILE, 'r') as file:
                user_config = yaml.safe_load(file) or {}
        except Exception as e:
            logging.error(f"Error loading default user config: {str(e)}")
            user_config = {}
    else:
        user_config = {}
    
    # Merge: deployment config (base) + user config (overrides)
    merged_config = deep_merge_dict(deployment_config, user_config)
    return merged_config


def save_config(config_data, user_id=None):
    """
    Save user-modifiable configuration for a user.
    Only saves user-modifiable fields to database/user_id/userConfig.yaml.
    Deployment-only fields are filtered out.
    
    :param config_data: Configuration dictionary to save (may contain deployment fields)
    :param user_id: Optional user ID. If None, tries to get from session.
    """
    if user_id is None:
        user_id = session.get('user_id')
    
    if not user_id:
        logging.warning("Cannot save config: no user_id provided")
        return
    
    # Filter out deployment-only fields
    user_config_data = filter_deployment_fields(config_data)
    
    # Get user config path
    user_config_file = get_user_config_path(user_id)
    
    # Ensure user directory exists
    user_dir = os.path.dirname(user_config_file)
    os.makedirs(user_dir, exist_ok=True)
    
    # Save only user-modifiable fields
    try:
        with open(user_config_file, 'w') as file:
            yaml.dump(user_config_data, file, default_flow_style=False, sort_keys=False)
        logging.info(f"Saved user config for {user_id}")
    except Exception as e:
        logging.error(f"Error saving user config for {user_id}: {str(e)}")
        raise

@app.route('/login', methods=['GET', 'POST'])
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
        return redirect(url_for('upload_file'))
    return render_template('login.html')

@app.route('/logout')
def logout():
    """Logout and clear session"""
    user_id = session.get('user_id')
    if user_id and user_id in user_processing:
        del user_processing[user_id]
    session.clear()
    return redirect(url_for('login'))


@app.route('/admin')
@login_required
@admin_required
def admin_home():
    """Render admin user management page."""
    users = list_users()
    return render_template('admin.html', users=users)


@app.route('/admin/create-token', methods=['POST'])
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
                    DATABASE_BASE_PATH,
                    UPLOAD_FOLDER,
                    OUTPUT_FOLDER
                )
                
                # Initialize user config from template
                user_management.initialize_user_config(
                    username,
                    DATABASE_BASE_PATH,
                    DEFAULT_USER_CONFIG_FILE
                )
                
                logging.info(f"Created folders and initialized config for new user {username}")
            except Exception as e:
                logging.error(f"Error creating folders/config for user {username}: {str(e)}")
                # Continue even if folder creation fails (user is already in DB)
        
        return jsonify({
            'status': 'success',
            'message': f'Token set for user {username}',
            'image_limit': image_limit,
            'token': saved_token  # only returned on creation/update
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/admin/users', methods=['GET'])
@login_required
@admin_required
def admin_list_users():
    """List users (JSON) for admin dashboard."""
    try:
        return jsonify({'status': 'success', 'users': list_users()})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/admin/update-limit', methods=['POST'])
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


@app.route('/admin/delete-user', methods=['POST'])
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

@app.route('/config', methods=['GET', 'POST'])
@login_required
def manage_config():
    user_id = session.get('user_id')
    
    if request.method == 'POST':
        try:
            if not request.is_json:
                return jsonify({'status': 'error', 'message': 'Content-Type must be application/json'}), 400
            
            config_data = request.get_json()
            if not config_data:
                return jsonify({'status': 'error', 'message': 'No configuration data received'}), 400
            
            # Validate: Check if user is trying to save deployment-only fields
            # Filter them out and log a warning if found
            deployment_fields_found = []
            if 'features' in config_data:
                for feature_name, feature_config in config_data.get('features', {}).items():
                    if 'parameters' in feature_config:
                        if 'replicate_model_id' in feature_config['parameters']:
                            deployment_fields_found.append(f"features.{feature_name}.parameters.replicate_model_id")
                            # Remove it
                            del feature_config['parameters']['replicate_model_id']
            
            if 'general' in config_data:
                if 'input_dir' in config_data['general']:
                    deployment_fields_found.append("general.input_dir")
                    del config_data['general']['input_dir']
                if 'output_dir' in config_data['general']:
                    deployment_fields_found.append("general.output_dir")
                    del config_data['general']['output_dir']
            
            if deployment_fields_found:
                logging.warning(f"User {user_id} attempted to save deployment-only fields (ignored): {deployment_fields_found}")
            
            # Save only user-modifiable configuration
            save_config(config_data, user_id)
            
            return jsonify({'status': 'success', 'message': 'Configuration updated successfully'})
        except Exception as e:
            logging.error(f"Error saving config for user {user_id}: {str(e)}")
            return jsonify({'status': 'error', 'message': str(e)}), 400
    
    # GET request - return merged configuration, but mark which fields are user-modifiable
    try:
        # Load merged config (deployment + user)
        merged_config = load_config(user_id)
        
        # For GET, we return the full merged config so the UI can display it
        # The UI will hide/disable deployment-only fields
        return jsonify(merged_config)
    except Exception as e:
        logging.error(f"Error loading config for user {user_id}: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 400

@app.route('/', methods=['GET', 'POST'])
@login_required
def upload_file():
    user_id = session.get('user_id')
    user_upload_folder = get_user_folder(user_id, UPLOAD_FOLDER)
    
    if request.method == 'POST':
        # Get batch folder from request if provided, otherwise create new one
        batch_folder = request.form.get('batch_folder')
        if not batch_folder:
            # Create a timestamped folder for this batch in user's folder
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            batch_folder = os.path.join(user_upload_folder, f'batch_{timestamp}')
        else:
            # Ensure batch folder is within user's upload folder
            if not batch_folder.startswith(user_upload_folder):
                batch_folder = os.path.join(user_upload_folder, os.path.basename(batch_folder))
        
        os.makedirs(batch_folder, exist_ok=True)
        
        # Clear existing files in the batch folder if this is a re-upload
        if request.form.get('clear_existing') == 'true':
            for filename in os.listdir(batch_folder):
                file_path = os.path.join(batch_folder, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)
        
        # Handle file uploads
        if 'files[]' in request.files:
            files = request.files.getlist('files[]')
            for file in files:
                if file.filename:
                    # For individual files, save directly in batch folder
                    file_path = os.path.join(batch_folder, file.filename)
                    file.save(file_path)
        
        # Handle folder upload
        if 'folder' in request.files:
            folder_files = request.files.getlist('folder')
            for file in folder_files:
                if file.filename:
                    # Extract just the filename without any path
                    filename = os.path.basename(file.filename)
                    # Save directly in batch folder
                    file_path = os.path.join(batch_folder, filename)
                    file.save(file_path)
        
        return jsonify({
            'status': 'success',
            'message': f'Files uploaded successfully to {batch_folder}',
            'batch_folder': batch_folder
        })
    
    # Get user quota for display (only for regular users, not admins)
    quota = None
    if session.get('role') == 'user':
        quota = get_user_quota(user_id)
    
    # Get results history (last 3 runs)
    results_history = get_user_results_history(user_id, max_results=3)
    
    return render_template('workflow.html', quota=quota, results_history=results_history)

@app.route('/process-images', methods=['POST'])
@login_required
def process_images():
    user_id = session.get('user_id')
    user_role = session.get('role', 'user')
    
    try:
        
        data = request.get_json()
        batch_folder = data.get('batch_folder')
        socketio_session_id = data.get('session_id')
        
        if not batch_folder:
            return jsonify({'status': 'error', 'message': 'No batch folder specified'}), 400
        
        if not socketio_session_id:
            return jsonify({'status': 'error', 'message': 'Session ID required for progress updates'}), 400
        
        # Verify batch folder belongs to this user
        user_upload_folder = get_user_folder(user_id, UPLOAD_FOLDER)
        if not batch_folder.startswith(user_upload_folder):
            return jsonify({'status': 'error', 'message': 'Invalid batch folder'}), 403

        # Check if processing is already in progress
        if user_id in user_processing:
            processing_info = user_processing[user_id]
            status = processing_info.get('status', '')
            if status in ['processing', 'starting', 'finalizing']:
                return jsonify({
                    'status': 'error',
                    'message': 'Processing already in progress. Please wait for current job to complete.'
                }), 409  # Conflict status

        # Enforce user quota (admins are exempt)
        if user_role != 'admin':
            image_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp', '.tiff'}
            images_to_process = [
                f for f in os.listdir(batch_folder)
                if os.path.isfile(os.path.join(batch_folder, f))
                and os.path.splitext(f)[1].lower() in image_exts
            ]
            img_count = len(images_to_process)
            ok, remaining = consume_user_quota(user_id, img_count)
            if not ok:
                return jsonify({'status': 'error', 'message': f'Quota exceeded. Remaining: {remaining} images'}), 403
        
        # Clean up any old processing state for this user
        if user_id in user_processing:
            old_session_id = user_processing[user_id].get('session_id')
            # Try to notify old session if it still exists
            try:
                if old_session_id:
                    socketio.emit('processing_complete', {
                        'status': 'cancelled',
                        'message': 'New processing started'
                    }, room=old_session_id)
            except (KeyError, RuntimeError):
                pass  # Old session already disconnected, ignore
        
        # Store user's processing session for progress updates
        # The client should have already joined the room via join_session event
        # before calling this endpoint, so the room exists and is ready for progress updates
        user_processing[user_id] = {
            'session_id': socketio_session_id,
            'status': 'processing',
            'progress': {'percentage': 0.0, 'status': 'starting', 'message': 'Initializing...'},
            'started_at': time.time()
        }
        
        # Process images in a background thread to allow Socket.IO events to be sent immediately
        def process_images_background():
            try:
                # Load merged config (deployment + user)
                merged_config = load_config(user_id)
                
                # Create a temporary merged config file for IA (IA expects a file path)
                # Store it in the user's output folder so it's included in the run
                user_output_folder = get_user_folder(user_id, OUTPUT_FOLDER)
                temp_config_path = os.path.join(user_output_folder, f'merged_config_{int(time.time())}.yaml')
                
                # Write merged config to temporary file
                with open(temp_config_path, 'w') as f:
                    yaml.dump(merged_config, f, default_flow_style=False, sort_keys=False)
                
                # Create a fresh Image Analyzer instance for this batch
                ia = IA(config_path=temp_config_path)
                ia.input_dir = batch_folder
                # Set output directory to user's output folder
                user_output_folder = get_user_folder(user_id, OUTPUT_FOLDER)
                ia.output_dir = user_output_folder
                # Reset the pipeline for clean state (this will generate a new timestamp)
                ia.reset_pipeline()
                
                # Process the images with progress callback
                def user_progress_callback(progress_data):
                    progress_callback(progress_data, user_id)
                
                results, logs = ia.process_batch(progress_callback=user_progress_callback)
                
                # Get the timestamp from the pipeline (already created in reset_pipeline)
                timestamp = ia.timestamp
                run_folder_name = f"Image-Analyzer_run_{timestamp}"
                
                # The pipeline creates files in ia.output_dir (which includes the run folder)
                # Use that directory as the source for the ZIP
                pipeline_output_dir = ia.output_dir
                
                # Create zip file with selected output formats in database/user_id/
                zip_filename = f"{run_folder_name}.zip"
                user_zip_folder = get_user_zip_folder(user_id)
                zip_path = os.path.abspath(os.path.join(user_zip_folder, zip_filename))

                # Always use output_formats from user-specific config (or default)
                config = load_config(user_id)
                output_formats = config.get('general', {}).get('output_formats', {'excel': True, 'csv': True})
                
                # Create ZIP file from files that the pipeline already saved
                with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    # Add results files from pipeline output directory
                    if output_formats.get('excel', True):
                        excel_path = os.path.join(pipeline_output_dir, 'results.xlsx')
                        if os.path.exists(excel_path):
                            zipf.write(excel_path, f"{run_folder_name}/results.xlsx")
                        
                    if output_formats.get('csv', True):
                        csv_path = os.path.join(pipeline_output_dir, 'results.csv')
                        if os.path.exists(csv_path):
                            zipf.write(csv_path, f"{run_folder_name}/results.csv")
                    
                    # Add logs if enabled
                    if config.get('general', {}).get('logs', {}).get('active', False):
                        logs_csv_path = os.path.join(pipeline_output_dir, 'logs.csv')
                        if os.path.exists(logs_csv_path):
                            zipf.write(logs_csv_path, f"{run_folder_name}/logs.csv")
                        
                        logs_xlsx_path = os.path.join(pipeline_output_dir, 'logs.xlsx')
                        if os.path.exists(logs_xlsx_path):
                            zipf.write(logs_xlsx_path, f"{run_folder_name}/logs.xlsx")
                    
                    # Add summary stats if enabled
                    if config.get('general', {}).get('summary_stats', {}).get('active', False):
                        summary_stats_csv_path = os.path.join(pipeline_output_dir, 'summary_statistics.csv')
                        if os.path.exists(summary_stats_csv_path):
                            zipf.write(summary_stats_csv_path, f"{run_folder_name}/summary_statistics.csv")
                        
                        summary_stats_xlsx_path = os.path.join(pipeline_output_dir, 'summary_statistics.xlsx')
                        if os.path.exists(summary_stats_xlsx_path):
                            zipf.write(summary_stats_xlsx_path, f"{run_folder_name}/summary_statistics.xlsx")
                    
                    # Add configuration file used for this run (pipeline already saved it)
                    config_path = os.path.join(pipeline_output_dir, 'configuration.yaml')
                    if os.path.exists(config_path):
                        zipf.write(config_path, f"{run_folder_name}/configuration.yaml")
                        print(f"### Configuration file added to ZIP from: {config_path} ###")
                
                # Cleanup: Delete uploaded images after successful processing
                try:
                    delete_batch_folder(batch_folder)
                except Exception as e:
                    print(f"Warning: Could not delete batch folder: {e}")
                
                # Cleanup: Delete pipeline output folder after ZIP creation (only ZIP is needed)
                # pipeline_output_dir is already the run folder path (e.g., outputs/user_id/Image-Analyzer_run_timestamp)
                try:
                    if os.path.exists(pipeline_output_dir) and os.path.isdir(pipeline_output_dir):
                        shutil.rmtree(pipeline_output_dir)
                        print(f"Deleted pipeline output folder: {pipeline_output_dir}")
                except Exception as e:
                    print(f"Warning: Could not delete pipeline output folder: {e}")
                
                # Cleanup: Keep only last 3 results per user
                try:
                    cleanup_old_results(user_id, keep_count=3)
                except Exception as e:
                    print(f"Warning: Could not cleanup old results: {e}")
                
                # Update state to completed
                if user_id in user_processing:
                    user_processing[user_id]['status'] = 'completed'
                    user_processing[user_id]['progress'] = {
                        'percentage': 1.0,
                        'status': 'completed',
                        'message': 'Processing completed successfully'
                    }
                
                # Send completion notification
                try:
                    socketio.emit('processing_complete', {
                        'status': 'success',
                        'message': 'Images processed successfully',
                        'download_links': {'zip': f"{user_id}/{zip_filename}"}
                    }, room=socketio_session_id)
                except (KeyError, RuntimeError) as e:
                    # Session disconnected - silently ignore but keep state
                    print(f"Could not send completion notification (session disconnected): {e}")
                
                # Clear processing state after completion to allow new processing
                if user_id in user_processing:
                    del user_processing[user_id]
                
            except Exception as e:
                print(f"Error in background processing: {e}")
                
                # Update state to error
                if user_id in user_processing:
                    user_processing[user_id]['status'] = 'error'
                    user_processing[user_id]['progress'] = {
                        'percentage': 0.0,
                        'status': 'error',
                        'message': str(e)
                    }
                
                # Send error notification
                try:
                    socketio.emit('processing_complete', {
                        'status': 'error',
                        'message': str(e)
                    }, room=socketio_session_id)
                except (KeyError, RuntimeError) as e2:
                    # Session disconnected - silently ignore but keep state
                    print(f"Could not send error notification (session disconnected): {e2}")
                
                # Clear processing state after error to allow retry
                if user_id in user_processing:
                    del user_processing[user_id]
        
        # Start processing in background thread
        socketio.start_background_task(process_images_background)
        
        # Return immediately to client
        return jsonify({
            'status': 'processing',
            'message': 'Processing started. Progress updates will be sent via WebSocket.'
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500
    # Note: user_processing cleanup is handled in the background thread's finally block
    # We don't clean it up here because the background thread is still running

@app.route('/get-uploaded-image-count', methods=['GET'])
@login_required
def get_uploaded_image_count():
    """Get the count of uploaded images in the most recent batch folder"""
    user_id = session.get('user_id')
    user_upload_folder = get_user_folder(user_id, UPLOAD_FOLDER)
    
    uploaded_image_count = 0
    if os.path.exists(user_upload_folder):
        # Find the most recent batch folder
        batch_folders = []
        for item in os.listdir(user_upload_folder):
            item_path = os.path.join(user_upload_folder, item)
            if os.path.isdir(item_path) and item.startswith('batch_'):
                mtime = os.path.getmtime(item_path)
                batch_folders.append((item_path, mtime))
        
        if batch_folders:
            # Get the most recent batch folder
            latest_batch = max(batch_folders, key=lambda x: x[1])[0]
            # Count image files
            image_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp', '.tiff', '.tif'}
            for filename in os.listdir(latest_batch):
                if os.path.isfile(os.path.join(latest_batch, filename)):
                    if os.path.splitext(filename)[1].lower() in image_exts:
                        uploaded_image_count += 1
    
    return jsonify({'image_count': uploaded_image_count})

@app.route('/check-processing-status', methods=['GET'])
@login_required
def check_processing_status():
    """Check if there's an active processing job for the current user"""
    user_id = session.get('user_id')
    
    if user_id in user_processing:
        processing_info = user_processing[user_id]
        response_data = {
            'status': 'active',
            'processing_status': processing_info.get('status', 'processing'),
            'progress': processing_info.get('progress', {}),
            'session_id': processing_info.get('session_id')
        }
        
        # If completed or error, include download links if available
        if processing_info.get('status') == 'completed':
            # Try to find the most recent zip file from database/user_id/
            user_zip_folder = get_user_zip_folder(user_id)
            zip_files = []
            if os.path.exists(user_zip_folder):
                for filename in os.listdir(user_zip_folder):
                    if filename.endswith('.zip') and filename.startswith('Image-Analyzer_run_'):
                        zip_path = os.path.join(user_zip_folder, filename)
                        if os.path.isfile(zip_path):
                            mtime = os.path.getmtime(zip_path)
                            zip_files.append({
                                'filename': filename,
                                'mtime': mtime,
                                'download_path': f"{user_id}/{filename}"
                            })
            
            if zip_files:
                # Get the most recent zip file
                zip_files.sort(key=lambda x: x['mtime'], reverse=True)
                response_data['download_links'] = {'zip': zip_files[0]['download_path']}
        
        return jsonify(response_data)
    else:
        return jsonify({
            'status': 'inactive'
        })

@app.route('/download/<path:filename>')
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
        if not file_path.startswith(user_zip_folder):
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
        user_output_folder = get_user_folder(user_id, OUTPUT_FOLDER)
        file_path = os.path.abspath(os.path.join(user_output_folder, filename))
        
        # Security check: ensure the resolved file path is within the user's output folder
        if not file_path.startswith(user_output_folder):
            return jsonify({'status': 'error', 'message': 'Invalid file path'}), 403
        
        # Check that file exists and is a file (not a directory)
        if not os.path.exists(file_path):
            return jsonify({'status': 'error', 'message': 'File not found'}), 404
        
        if not os.path.isfile(file_path):
            return jsonify({'status': 'error', 'message': 'Not a file'}), 403
        
        # Use the relative filename for send_from_directory (needs relative path from directory)
        return send_from_directory(user_output_folder, filename, as_attachment=True)

@app.route('/uploads/<path:filename>')
def serve_upload(filename):
    """Serve uploaded images publicly so Replicate can access them via URL"""
    # Security: only serve files from uploads folder, prevent directory traversal
    safe_path = os.path.normpath(os.path.join(UPLOAD_FOLDER, filename))
    if not safe_path.startswith(os.path.abspath(UPLOAD_FOLDER)):
        return jsonify({'status': 'error', 'message': 'Invalid path'}), 403
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/config-page')
@login_required
def config_page():
    return render_template('config.html')

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    emit('connected', {'data': 'Connected'})
    
    # Check if user has active processing and send current state
    # Note: We can't access session directly in Socket.IO, so we'll rely on
    # the client to check status via HTTP endpoint after connecting

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection - silently ignore, session cleanup is automatic"""
    # Socket.IO automatically cleans up disconnected sessions
    # No need to manually clean up here as background tasks handle errors gracefully
    pass

@socketio.on('join_session')
def handle_join_session(data):
    """Handle client joining a processing session"""
    session_id = data.get('session_id')
    if session_id:
        join_room(session_id)
        emit('joined_session', {'session_id': session_id})

if __name__ == '__main__':
    # Only run with debug in development
    if os.getenv('FLASK_ENV') == 'development':
        socketio.run(app, debug=True, host='0.0.0.0', port=PORT)
    else:
        # In production, Gunicorn will handle this
        # Railway and GCP use Gunicorn via Procfile/startup-script
        pass