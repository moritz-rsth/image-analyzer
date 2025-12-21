"""
Configuration Routes

Handles configuration management endpoints.
"""

import logging
from flask import Blueprint, request, jsonify, session, render_template
from src.image_analyzer.web.auth import login_required
from src.image_analyzer.web.config_manager import load_config, save_config
from src.image_analyzer.web.database import is_pro_user

logger = logging.getLogger(__name__)

# Create blueprint with name for consistent endpoint naming
config_bp = Blueprint('config', __name__)

# List of Replicate API features (Pro-only)
REPLICATE_FEATURES = [
    'predict_coco_labels_yolo11',
    'predict_imagenet_classes_yolo11',
    'detect_faces',
    'detect_objects',
    'describe_blip',
    'describe_llm'
]


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
            
            # Check if user has pro features
            user_role = session.get('role', 'user')
            is_pro = (user_role == 'admin') or is_pro_user(user_id)
            
            # Validate: Check if user is trying to save deployment-only fields
            # Filter them out and log a warning if found
            deployment_fields_found = []
            replicate_features_disabled = []
            
            if 'features' in config_data:
                for feature_name, feature_config in config_data.get('features', {}).items():
                    # Remove replicate_model_id (deployment-only)
                    if 'parameters' in feature_config:
                        if 'replicate_model_id' in feature_config['parameters']:
                            deployment_fields_found.append(f"features.{feature_name}.parameters.replicate_model_id")
                            del feature_config['parameters']['replicate_model_id']
                    
                    # Disable Replicate features if user is not pro
                    if not is_pro and feature_name in REPLICATE_FEATURES:
                        if feature_config.get('active', False):
                            replicate_features_disabled.append(feature_name)
                            feature_config['active'] = False
            
            if 'general' in config_data:
                if 'input_dir' in config_data['general']:
                    deployment_fields_found.append("general.input_dir")
                    del config_data['general']['input_dir']
                if 'output_dir' in config_data['general']:
                    deployment_fields_found.append("general.output_dir")
                    del config_data['general']['output_dir']
            
            if deployment_fields_found:
                logger.warning(f"User {user_id} attempted to save deployment-only fields (ignored): {deployment_fields_found}")
            
            if replicate_features_disabled:
                logger.info(f"User {user_id} attempted to enable Replicate features (disabled - not pro user): {replicate_features_disabled}")
            
            # Save only user-modifiable configuration
            save_config(config_data, user_id)
            
            message = 'Configuration updated successfully'
            if replicate_features_disabled:
                message += '. Note: Replicate features are only available for Pro users.'
            
            return jsonify({'status': 'success', 'message': message})
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
    user_id = session.get('user_id')
    user_role = session.get('role', 'user')
    
    # Admin has all rights, regular users need pro_features
    is_pro = (user_role == 'admin') or is_pro_user(user_id)
    
    return render_template('config.html', is_pro_user=is_pro, replicate_features=REPLICATE_FEATURES)

