from flask import Flask, request, render_template, send_from_directory, jsonify, session, redirect, url_for
from flask_socketio import SocketIO, emit, join_room
from functools import wraps
import os
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
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'your-secret-key-change-in-production')
socketio = SocketIO(app, cors_allowed_origins="*")

UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
DEFAULT_CONFIG_FILE = os.path.join('config', 'configuration_deployment.yaml')
CURR_CONFIG_FILE = os.path.join('config', 'configuration.yaml')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Per-user processing state: {user_id: socketio_room}
user_processing = {}

def login_required(f):
    """Decorator to require login for routes"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

def get_user_folder(user_id, base_folder):
    """Get user-specific folder path"""
    user_folder = os.path.join(base_folder, user_id)
    os.makedirs(user_folder, exist_ok=True)
    return user_folder

def progress_callback(progress_data, user_session_id):
    """Callback function to emit progress updates via WebSocket"""
    if user_session_id in user_processing:
        socketio.emit('progress_update', progress_data, room=user_processing[user_session_id])

def load_config():
    with open(DEFAULT_CONFIG_FILE, 'r') as file:
        return yaml.safe_load(file)

def save_config(config_data):
    with open(CURR_CONFIG_FILE, 'w') as file:
        yaml.dump(config_data, file, default_flow_style=False)

@app.route('/login', methods=['GET', 'POST'])
def login():
    """Login page and authentication"""
    if request.method == 'POST':
        data = request.get_json()
        username = data.get('username')
        password = data.get('password')
        
        # Check username
        if username != 'admin':
            return jsonify({'status': 'error', 'message': 'Invalid credentials'}), 401
        
        # Verify password hash
        admin_password_hash = os.getenv('ADMIN_PASSWORD_HASH')
        if not admin_password_hash:
            return jsonify({'status': 'error', 'message': 'Server configuration error'}), 500
        
        # Hash the provided password and compare
        password_hash = hashlib.sha256(password.encode()).hexdigest()
        if password_hash != admin_password_hash:
            return jsonify({'status': 'error', 'message': 'Invalid credentials'}), 401
        
        # Set session
        session['user_id'] = hashlib.sha256(f"{username}{datetime.now()}".encode()).hexdigest()[:16]
        session.permanent = True
        return jsonify({'status': 'success', 'message': 'Login successful'})
    
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
    
    return render_template('workflow.html')

@app.route('/process-images', methods=['POST'])
@login_required
def process_images():
    try:
        user_id = session.get('user_id')
        if not user_id:
            return jsonify({'status': 'error', 'message': 'Session expired'}), 401
        
        data = request.get_json()
        batch_folder = data.get('batch_folder')
        socketio_session_id = data.get('session_id')
        
        if not batch_folder:
            return jsonify({'status': 'error', 'message': 'No batch folder specified'}), 400
        
        # Verify batch folder belongs to this user
        user_upload_folder = get_user_folder(user_id, UPLOAD_FOLDER)
        if not batch_folder.startswith(user_upload_folder):
            return jsonify({'status': 'error', 'message': 'Invalid batch folder'}), 403
        
        # Store user's processing session for progress updates
        if socketio_session_id:
            user_processing[user_id] = socketio_session_id
        
        # Create a fresh Image Analyzer instance for this batch
        ia = IA(config_path=CURR_CONFIG_FILE)
        ia.input_dir = batch_folder
        # Set output directory to user's output folder
        user_output_folder = get_user_folder(user_id, OUTPUT_FOLDER)
        ia.output_dir = user_output_folder
        # Reset the pipeline for clean state
        ia.reset_pipeline()
        
        # Process the images with progress callback (pass user_id for room lookup)
        def user_progress_callback(progress_data):
            progress_callback(progress_data, user_id)
        
        results, logs = ia.process_batch(progress_callback=user_progress_callback)
        
        # Create timestamp for the run
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        run_folder_name = f"Image-Analyzer_run_{timestamp}"
        
        # Create zip file with selected output formats in user's output folder
        zip_filename = f"{run_folder_name}.zip"
        zip_path = os.path.join(user_output_folder, zip_filename)

        # Always use output_formats from configuration.yaml
        with open(CURR_CONFIG_FILE, 'r') as f:
            config = yaml.safe_load(f)
        output_formats = config.get('general', {}).get('output_formats', {'excel': True, 'csv': True})
        
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Add results files based on selected formats
            if output_formats.get('excel', True):
                excel_filename = f'results_{timestamp}.xlsx'
                excel_path = os.path.join(user_output_folder, excel_filename)
                results.to_excel(excel_path, index=False)
                zipf.write(excel_path, f"{run_folder_name}/results.xlsx")
                
            if output_formats.get('csv', True):
                csv_filename = f'results_{timestamp}.csv'
                csv_path = os.path.join(user_output_folder, csv_filename)
                results.to_csv(csv_path, index=False)
                zipf.write(csv_path, f"{run_folder_name}/results.csv")
            
            # Add logs if enabled
            if config.get('general', {}).get('logs', {}).get('active', False):
                logs_csv_filename = f'logs_{timestamp}.csv'
                logs_csv_path = os.path.join(user_output_folder, logs_csv_filename)
                logs.to_csv(logs_csv_path, index=False)
                zipf.write(logs_csv_path, f"{run_folder_name}/logs.csv")
            
            # Add summary stats if enabled
            if config.get('general', {}).get('summary_stats', {}).get('active', False):
                summary_stats_filename = f'summary_stats_{timestamp}.xlsx'
                summary_stats_path = os.path.join(user_output_folder, summary_stats_filename)
                # Calculate and save summary stats
                numeric_cols = results.select_dtypes(include=['int64', 'float64']).columns
                if len(numeric_cols) > 0:
                    summary_stats = results[numeric_cols].agg(['count', 'mean', 'std', 'min', 'max'])
                    summary_stats.to_excel(summary_stats_path)
                    zipf.write(summary_stats_path, f"{run_folder_name}/summary_stats.xlsx")
            
            # Add configuration file used for this run
            config_filename = f'configuration_{timestamp}.yaml'
            config_path = os.path.join(user_output_folder, config_filename)
            
            # Copy the current configuration file to the output folder with timestamp
            if os.path.exists(CURR_CONFIG_FILE):
                shutil.copy2(CURR_CONFIG_FILE, config_path)
                zipf.write(config_path, f"{run_folder_name}/configuration.yaml")
                print(f"### Configuration file added to ZIP: {config_filename} ###")
            else:
                # If config file doesn't exist, save the current config as YAML
                with open(config_path, 'w') as f:
                    yaml.dump(config, f, default_flow_style=False, sort_keys=False)
                zipf.write(config_path, f"{run_folder_name}/configuration.yaml")
                print(f"### Configuration saved and added to ZIP: {config_filename} ###")
        
        # Clean up individual files
        for filename in [f'results_{timestamp}.xlsx', f'results_{timestamp}.csv', 
                        f'logs_{timestamp}.csv', f'configuration_{timestamp}.yaml',
                        f'summary_stats_{timestamp}.xlsx']:
            file_path = os.path.join(user_output_folder, filename)
            if os.path.exists(file_path):
                os.remove(file_path)
        
        # Clear user's processing session
        if user_id in user_processing:
            del user_processing[user_id]
        
        # Clean up Image Analyzer instance
        del ia
        
        return jsonify({
            'status': 'success',
            'message': 'Images processed successfully',
            'download_links': {'zip': f"{user_id}/{zip_filename}"}
        })
    except Exception as e:
        # Clear user's processing session on error
        if user_id in user_processing:
            del user_processing[user_id]
        # Clean up IA instance if it exists
        if 'ia' in locals():
            del ia
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/download/<path:filename>')
@login_required
def download_file(filename):
    """Download file from user's output folder"""
    user_id = session.get('user_id')
    user_output_folder = get_user_folder(user_id, OUTPUT_FOLDER)
    
    # Security: ensure filename is within user's folder
    file_path = os.path.join(user_output_folder, filename)
    if not file_path.startswith(os.path.abspath(user_output_folder)):
        return jsonify({'status': 'error', 'message': 'Invalid file path'}), 403
    
    # If filename includes user_id prefix, remove it
    if filename.startswith(f"{user_id}/"):
        filename = filename[len(user_id) + 1:]
    
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
    # Note: Flask-SocketIO doesn't have direct access to Flask session
    # Session validation happens on HTTP routes, SocketIO uses separate session
    print('Client connected')
    emit('connected', {'data': 'Connected'})

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    print('Client disconnected')

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
        socketio.run(app, debug=True, host='0.0.0.0', port=int(os.getenv('PORT', 5000)))
    else:
        # In production, Gunicorn will handle this
        pass