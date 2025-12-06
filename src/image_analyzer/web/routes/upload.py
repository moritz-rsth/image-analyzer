"""
Upload and Processing Routes

Handles file uploads and image processing.
"""

import os
import shutil
import time
import yaml
import zipfile
from datetime import datetime
from flask import Blueprint, request, jsonify, session, render_template
from src.image_analyzer.web.auth import login_required
from src.image_analyzer.web.file_management import (
    get_user_folder,
    get_user_zip_folder,
    cleanup_old_results,
    delete_batch_folder,
    get_user_results_history
)
from src.image_analyzer.web.database import get_user_quota, consume_user_quota
from src.image_analyzer.web.config_manager import load_config
from src.image_analyzer.web.progress import (
    user_processing,
    set_processing_state,
    clear_processing_state,
    get_processing_state,
    progress_callback
)
from src.image_analyzer.web.config import get_upload_folder, get_output_folder
from src.image_analyzer.pipeline.ia_pipeline_deployment import IA

# Create blueprint with name for consistent endpoint naming
upload_bp = Blueprint('upload', __name__)

# SocketIO instance will be set by app factory
_socketio = None


def set_socketio(socketio_instance):
    """Set the SocketIO instance for this module."""
    global _socketio
    _socketio = socketio_instance


@upload_bp.route('/', methods=['GET', 'POST'], endpoint='upload_file')
@login_required
def upload_file():
    user_id = session.get('user_id')
    user_upload_folder = get_user_folder(user_id, get_upload_folder())
    
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


@upload_bp.route('/process-images', methods=['POST'], endpoint='process_images')
@login_required
def process_images():
    user_id = session.get('user_id')
    user_role = session.get('role', 'user')
    
    if not _socketio:
        return jsonify({'status': 'error', 'message': 'SocketIO not initialized'}), 500
    
    try:
        data = request.get_json()
        batch_folder = data.get('batch_folder')
        socketio_session_id = data.get('session_id')
        
        if not batch_folder:
            return jsonify({'status': 'error', 'message': 'No batch folder specified'}), 400
        
        if not socketio_session_id:
            return jsonify({'status': 'error', 'message': 'Session ID required for progress updates'}), 400
        
        # Verify batch folder belongs to this user
        user_upload_folder = get_user_folder(user_id, get_upload_folder())
        if not batch_folder.startswith(user_upload_folder):
            return jsonify({'status': 'error', 'message': 'Invalid batch folder'}), 403

        # Check if processing is already in progress
        processing_state = get_processing_state(user_id)
        if processing_state:
            status = processing_state.get('status', '')
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
        if processing_state:
            old_session_id = processing_state.get('session_id')
            # Try to notify old session if it still exists
            try:
                if old_session_id:
                    _socketio.emit('processing_complete', {
                        'status': 'cancelled',
                        'message': 'New processing started'
                    }, room=old_session_id)
            except (KeyError, RuntimeError):
                pass  # Old session already disconnected, ignore
        
        # Store user's processing session for progress updates
        set_processing_state(user_id, {
            'session_id': socketio_session_id,
            'status': 'processing',
            'progress': {'percentage': 0.0, 'status': 'starting', 'message': 'Initializing...'},
            'started_at': time.time()
        })
        
        # Process images in a background thread to allow Socket.IO events to be sent immediately
        def process_images_background():
            try:
                # Load merged config (deployment + user)
                merged_config = load_config(user_id)
                
                # Create a temporary merged config file for IA (IA expects a file path)
                # Store it in the user's output folder so it's included in the run
                user_output_folder = get_user_folder(user_id, get_output_folder())
                temp_config_path = os.path.join(user_output_folder, f'merged_config_{int(time.time())}.yaml')
                
                # Write merged config to temporary file
                with open(temp_config_path, 'w') as f:
                    yaml.dump(merged_config, f, default_flow_style=False, sort_keys=False)
                
                # Create a fresh Image Analyzer instance for this batch
                ia = IA(config_path=temp_config_path)
                ia.input_dir = batch_folder
                # Set output directory to user's output folder
                user_output_folder = get_user_folder(user_id, get_output_folder())
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
                processing_state = get_processing_state(user_id)
                if processing_state:
                    processing_state['status'] = 'completed'
                    processing_state['progress'] = {
                        'percentage': 1.0,
                        'status': 'completed',
                        'message': 'Processing completed successfully'
                    }
                
                # Send completion notification
                try:
                    _socketio.emit('processing_complete', {
                        'status': 'success',
                        'message': 'Images processed successfully',
                        'download_links': {'zip': f"{user_id}/{zip_filename}"}
                    }, room=socketio_session_id)
                except (KeyError, RuntimeError) as e:
                    # Session disconnected - silently ignore but keep state
                    print(f"Could not send completion notification (session disconnected): {e}")
                
                # Clear processing state after completion to allow new processing
                clear_processing_state(user_id)
                
            except Exception as e:
                print(f"Error in background processing: {e}")
                
                # Update state to error
                processing_state = get_processing_state(user_id)
                if processing_state:
                    processing_state['status'] = 'error'
                    processing_state['progress'] = {
                        'percentage': 0.0,
                        'status': 'error',
                        'message': str(e)
                    }
                
                # Send error notification
                try:
                    _socketio.emit('processing_complete', {
                        'status': 'error',
                        'message': str(e)
                    }, room=socketio_session_id)
                except (KeyError, RuntimeError) as e2:
                    # Session disconnected - silently ignore but keep state
                    print(f"Could not send error notification (session disconnected): {e2}")
                
                # Clear processing state after error to allow retry
                clear_processing_state(user_id)
        
        # Start processing in background thread
        _socketio.start_background_task(process_images_background)
        
        # Return immediately to client
        return jsonify({
            'status': 'processing',
            'message': 'Processing started. Progress updates will be sent via WebSocket.'
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@upload_bp.route('/get-uploaded-image-count', methods=['GET'], endpoint='get_uploaded_image_count')
@login_required
def get_uploaded_image_count():
    """Get the count of uploaded images in the most recent batch folder"""
    user_id = session.get('user_id')
    user_upload_folder = get_user_folder(user_id, get_upload_folder())
    
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


@upload_bp.route('/check-processing-status', methods=['GET'], endpoint='check_processing_status')
@login_required
def check_processing_status():
    """Check if there's an active processing job for the current user"""
    user_id = session.get('user_id')
    
    processing_state = get_processing_state(user_id)
    if processing_state:
        response_data = {
            'status': 'active',
            'processing_status': processing_state.get('status', 'processing'),
            'progress': processing_state.get('progress', {}),
            'session_id': processing_state.get('session_id')
        }
        
        # If completed or error, include download links if available
        if processing_state.get('status') == 'completed':
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

