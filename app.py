from flask import Flask, request, render_template, send_from_directory, jsonify, session, redirect, url_for
from flask_socketio import SocketIO, emit, join_room
from functools import wraps
import os
import sqlite3
import pandas as pd
from datetime import datetime
from src.image_analyzer.pipeline.ia_pipeline_deployment import IA
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

socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
DEFAULT_CONFIG_FILE = os.path.join('config', 'configuration_deployment.yaml')
CURR_CONFIG_FILE = os.path.join('config', 'configuration.yaml')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Database path configuration (prefers explicit env, then Railway volume, else local folder)
DATABASE_BASE_PATH = os.getenv('DATABASE_PATH') or os.getenv('RAILWAY_VOLUME_MOUNT_PATH') or './database'
os.environ['DATABASE_PATH'] = DATABASE_BASE_PATH  # expose for downstream code/tools
os.makedirs(DATABASE_BASE_PATH, exist_ok=True)
DB_FILE = os.path.join(DATABASE_BASE_PATH, 'user.db')

# Per-user processing state: {user_id: socketio_room}
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
    """Delete user by username."""
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("DELETE FROM users WHERE username = ?", (username,))
    conn.commit()
    conn.close()


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

def cleanup_old_results(user_id, keep_count=3):
    """
    Cleanup old results, keeping only the last N runs per user.
    
    :param user_id: User ID
    :param keep_count: Number of recent runs to keep (default: 3)
    """
    try:
        user_output_folder = get_user_folder(user_id, OUTPUT_FOLDER)
        
        # Get all ZIP files for this user
        zip_files = []
        for filename in os.listdir(user_output_folder):
            if filename.endswith('.zip') and filename.startswith('Image-Analyzer_run_'):
                zip_path = os.path.join(user_output_folder, filename)
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
        
        # Also delete corresponding run folders
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
        user_output_folder = get_user_folder(user_id, OUTPUT_FOLDER)
        
        # Get all ZIP files for this user
        zip_files = []
        if os.path.exists(user_output_folder):
            for filename in os.listdir(user_output_folder):
                if filename.endswith('.zip') and filename.startswith('Image-Analyzer_run_'):
                    zip_path = os.path.join(user_output_folder, filename)
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
        room_id = user_processing[user_session_id]
        socketio.emit('progress_update', progress_data, room=room_id, namespace='/')
        # Small sleep to allow the event to be processed and sent immediately
        time.sleep(0.01)

def load_config():
    with open(DEFAULT_CONFIG_FILE, 'r') as file:
        return yaml.safe_load(file)

def save_config(config_data):
    with open(CURR_CONFIG_FILE, 'w') as file:
        yaml.dump(config_data, file, default_flow_style=False)

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

        saved_token = upsert_user_token(username, token, image_limit)
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
    if request.method == 'POST':
        try:
            if not request.is_json:
                return jsonify({'status': 'error', 'message': 'Content-Type must be application/json'}), 400
            
            config_data = request.get_json()
            if not config_data:
                return jsonify({'status': 'error', 'message': 'No configuration data received'}), 400
            
            # Save the configuration directly without merging
            with open(CURR_CONFIG_FILE, 'w') as file:
                yaml.dump(config_data, file, default_flow_style=False, sort_keys=False)
            
            return jsonify({'status': 'success', 'message': 'Configuration updated successfully'})
        except Exception as e:
            print("Error saving config:", str(e))
            return jsonify({'status': 'error', 'message': str(e)}), 400
    
    # GET request - return current configuration
    try:
        config_data = load_config()
        return jsonify(config_data)
    except Exception as e:
        print("Error loading config:", str(e))
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
        
        # Store user's processing session for progress updates
        # The client should have already joined the room via join_session event
        # before calling this endpoint, so the room exists and is ready for progress updates
        user_processing[user_id] = socketio_session_id
        
        # Process images in a background thread to allow Socket.IO events to be sent immediately
        def process_images_background():
            try:
                # Create a fresh Image Analyzer instance for this batch
                ia = IA(config_path=CURR_CONFIG_FILE)
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
                
                # Create zip file with selected output formats in user's output folder
                zip_filename = f"{run_folder_name}.zip"
                zip_path = os.path.abspath(os.path.join(user_output_folder, zip_filename))

                # Always use output_formats from configuration.yaml
                with open(CURR_CONFIG_FILE, 'r') as f:
                    config = yaml.safe_load(f)
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
                
                # Cleanup: Keep only last 3 results per user
                try:
                    cleanup_old_results(user_id, keep_count=3)
                except Exception as e:
                    print(f"Warning: Could not cleanup old results: {e}")
                
                # Send completion notification
                socketio.emit('processing_complete', {
                    'status': 'success',
                    'message': 'Images processed successfully',
                    'download_links': {'zip': f"{user_id}/{zip_filename}"}
                }, room=socketio_session_id)
                
            except Exception as e:
                print(f"Error in background processing: {e}")
                socketio.emit('processing_complete', {
                    'status': 'error',
                    'message': str(e)
                }, room=socketio_session_id)
            finally:
                # Clean up user_processing entry
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

@app.route('/download/<path:filename>')
@login_required
def download_file(filename):
    """Download file from user's output folder"""
    user_id = session.get('user_id')
    
    # Get user output folder (already absolute from get_user_folder)
    user_output_folder = get_user_folder(user_id, OUTPUT_FOLDER)
    
    # If filename includes user_id prefix, remove it
    if filename.startswith(f"{user_id}/"):
        filename = filename[len(user_id) + 1:]
    
    # Build the full file path using absolute paths consistently
    file_path = os.path.abspath(os.path.join(user_output_folder, filename))
    
    # Security check: ensure the resolved file path is within the user's output folder
    # Both paths are now absolute, so we can do a simple prefix check
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
def config_new_page():
    return render_template('config_new.html')

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    emit('connected', {'data': 'Connected'})

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
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