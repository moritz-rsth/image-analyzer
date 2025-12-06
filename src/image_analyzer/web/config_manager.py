"""
Configuration Manager Module

Handles loading, saving, and merging of configuration files.
Merges deployment config with user-specific configs.
"""

import os
import yaml
import logging
from flask import session
from src.image_analyzer.web.config import (
    get_deployment_config_file,
    get_default_user_config_file,
    get_default_config_file
)
from src.image_analyzer.utils import user_management
from src.image_analyzer.web.config import get_database_base_path

logger = logging.getLogger(__name__)


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
    return get_deployment_config_file()


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
        return user_management.get_user_config_path(user_id, get_database_base_path())
    return get_default_user_config_file()


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
        deployment_config_path = get_default_config_file()
    
    try:
        with open(deployment_config_path, 'r') as file:
            deployment_config = yaml.safe_load(file) or {}
    except Exception as e:
        logger.error(f"Error loading deployment config: {str(e)}")
        deployment_config = {}
    
    # Load user config (overrides)
    user_config_path = get_user_config_path(user_id)
    default_user_config_file = get_default_user_config_file()
    
    if user_id and os.path.exists(user_config_path):
        try:
            with open(user_config_path, 'r') as file:
                user_config = yaml.safe_load(file) or {}
        except Exception as e:
            logger.error(f"Error loading user config for {user_id}: {str(e)}")
            user_config = {}
    elif os.path.exists(default_user_config_file):
        # Use default user config template
        try:
            with open(default_user_config_file, 'r') as file:
                user_config = yaml.safe_load(file) or {}
        except Exception as e:
            logger.error(f"Error loading default user config: {str(e)}")
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
        logger.warning("Cannot save config: no user_id provided")
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
        logger.info(f"Saved user config for {user_id}")
    except Exception as e:
        logger.error(f"Error saving user config for {user_id}: {str(e)}")
        raise

