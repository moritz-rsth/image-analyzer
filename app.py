from flask import Flask, request, render_template, send_from_directory, jsonify
from flask_socketio import SocketIO, emit, join_room
import os
import pandas as pd
from datetime import datetime
from src.image_analyzer.pipeline.ia_pipeline import IA
import yaml
import shutil
import threading
import time
import zipfile
import tempfile

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
socketio = SocketIO(app, cors_allowed_origins="*")

UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
DEFAULT_CONFIG_FILE = os.path.join('config', 'configuration_default.yaml')
CURR_CONFIG_FILE = os.path.join('config', 'configuration.yaml')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Global variable to store current processing session
current_processing = None

def progress_callback(progress_data):
    """Callback function to emit progress updates via WebSocket"""
    if current_processing:
        socketio.emit('progress_update', progress_data, room=current_processing)

def load_config():
    with open(DEFAULT_CONFIG_FILE, 'r') as file:
        return yaml.safe_load(file)

def save_config(config_data):
    with open(CURR_CONFIG_FILE, 'w') as file:
        yaml.dump(config_data, file, default_flow_style=False)

@app.route('/config', methods=['GET', 'POST'])
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
def upload_file():
    if request.method == 'POST':
        # Get batch folder from request if provided, otherwise create new one
        batch_folder = request.form.get('batch_folder')
        if not batch_folder:
            # Create a timestamped folder for this batch
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            batch_folder = os.path.join(UPLOAD_FOLDER, f'batch_{timestamp}')
        
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
def process_images():
    try:
        data = request.get_json()
        batch_folder = data.get('batch_folder')
        session_id = data.get('session_id')
        
        if not batch_folder:
            return jsonify({'status': 'error', 'message': 'No batch folder specified'}), 400
        
        # Set global processing session
        global current_processing
        current_processing = session_id
        
        # Create a fresh Image Analyzer instance for this batch
        ia = IA(config_path=CURR_CONFIG_FILE)
        ia.input_dir = batch_folder
        # Set output directory to the Flask app's output folder
        ia.output_dir = OUTPUT_FOLDER
        # Reset the pipeline for clean state
        ia.reset_pipeline()
        
        # Process the images with progress callback
        results, logs = ia.process_batch(progress_callback=progress_callback)
        
        # Create timestamp for the run
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        run_folder_name = f"Image-Analyzer_run_{timestamp}"
        
        # Create zip file with selected output formats
        zip_filename = f"{run_folder_name}.zip"
        zip_path = os.path.join(OUTPUT_FOLDER, zip_filename)

        # Always use output_formats from configuration.yaml
        with open(CURR_CONFIG_FILE, 'r') as f:
            config = yaml.safe_load(f)
        output_formats = config.get('general', {}).get('output_formats', {'excel': True, 'csv': True})
        
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Add results files based on selected formats
            if output_formats.get('excel', True):
                excel_filename = f'results_{timestamp}.xlsx'
                excel_path = os.path.join(OUTPUT_FOLDER, excel_filename)
                results.to_excel(excel_path, index=False)
                zipf.write(excel_path, f"{run_folder_name}/results.xlsx")
                
            if output_formats.get('csv', True):
                csv_filename = f'results_{timestamp}.csv'
                csv_path = os.path.join(OUTPUT_FOLDER, csv_filename)
                results.to_csv(csv_path, index=False)
                zipf.write(csv_path, f"{run_folder_name}/results.csv")
            
            # Add logs if enabled
            if config.get('general', {}).get('logs', {}).get('active', False):
                logs_csv_filename = f'logs_{timestamp}.csv'
                logs_csv_path = os.path.join(OUTPUT_FOLDER, logs_csv_filename)
                logs.to_csv(logs_csv_path, index=False)
                zipf.write(logs_csv_path, f"{run_folder_name}/logs.csv")
            
            # Add summary stats if enabled
            if config.get('general', {}).get('summary_stats', {}).get('active', False):
                summary_stats_filename = f'summary_stats_{timestamp}.xlsx'
                summary_stats_path = os.path.join(OUTPUT_FOLDER, summary_stats_filename)
                # Calculate and save summary stats
                numeric_cols = results.select_dtypes(include=['int64', 'float64']).columns
                if len(numeric_cols) > 0:
                    summary_stats = results[numeric_cols].agg(['count', 'mean', 'std', 'min', 'max'])
                    summary_stats.to_excel(summary_stats_path)
                    zipf.write(summary_stats_path, f"{run_folder_name}/summary_stats.xlsx")
            
            # Add configuration file used for this run
            config_filename = f'configuration_{timestamp}.yaml'
            config_path = os.path.join(OUTPUT_FOLDER, config_filename)
            
            # Copy the current configuration file to the output folder with timestamp
            import shutil
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
            file_path = os.path.join(OUTPUT_FOLDER, filename)
            if os.path.exists(file_path):
                os.remove(file_path)
        
        # Clear processing session
        current_processing = None
        
        # Clean up Image Analyzer instance
        del ia
        
        return jsonify({
            'status': 'success',
            'message': 'Images processed successfully',
            'download_links': {'zip': zip_filename}
        })
    except Exception as e:
        # Clear processing session on error
        current_processing = None
        # Clean up IA instance if it exists
        if 'ia' in locals():
            del ia
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(OUTPUT_FOLDER, filename, as_attachment=True)

@app.route('/config-page')
def config_new_page():
    return render_template('config_new.html')

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
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
    socketio.run(app, debug=True) 