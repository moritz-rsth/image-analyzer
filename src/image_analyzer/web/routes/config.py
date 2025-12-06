"""
Configuration Routes

Handles configuration management endpoints.
"""

import logging
from flask import Blueprint, request, jsonify, session, render_template
from src.image_analyzer.web.auth import login_required
from src.image_analyzer.web.config_manager import load_config, save_config

logger = logging.getLogger(__name__)

# Create blueprint with name for consistent endpoint naming
config_bp = Blueprint('config', __name__)


@config_bp.route('/config', methods=['GET', 'POST'], endpoint='manage_config')
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
                logger.warning(f"User {user_id} attempted to save deployment-only fields (ignored): {deployment_fields_found}")
            
            # Save only user-modifiable configuration
            save_config(config_data, user_id)
            
            return jsonify({'status': 'success', 'message': 'Configuration updated successfully'})
        except Exception as e:
            logger.error(f"Error saving config for user {user_id}: {str(e)}")
            return jsonify({'status': 'error', 'message': str(e)}), 400
    
    # GET request - return merged configuration, but mark which fields are user-modifiable
    try:
        # Load merged config (deployment + user)
        merged_config = load_config(user_id)
        
        # For GET, we return the full merged config so the UI can display it
        # The UI will hide/disable deployment-only fields
        return jsonify(merged_config)
    except Exception as e:
        logger.error(f"Error loading config for user {user_id}: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 400


@config_bp.route('/config-page', endpoint='config_page')
@login_required
def config_page():
    return render_template('config.html')

