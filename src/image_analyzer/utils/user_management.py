"""
User Management Module

This module provides centralized functions for managing user folders, configurations,
and user-related file operations.
"""

import os
import shutil
import yaml
import logging

logger = logging.getLogger(__name__)


def get_user_folders(user_id, database_base_path, upload_folder, output_folder):
    """
    Get all folder paths associated with a user.
    
    :param user_id: User ID
    :param database_base_path: Base path for database directory
    :param upload_folder: Base path for uploads directory
    :param output_folder: Base path for outputs directory
    :return: Dictionary with folder paths:
        - config_dir: database/user_id/
        - upload_dir: uploads/user_id/
        - output_dir: outputs/user_id/
        - config_file: database/user_id/userConfig.yaml
    """
    config_dir = os.path.join(database_base_path, user_id)
    upload_dir = os.path.join(upload_folder, user_id)
    output_dir = os.path.join(output_folder, user_id)
    config_file = os.path.join(config_dir, 'userConfig.yaml')
    
    return {
        'config_dir': os.path.abspath(config_dir),
        'upload_dir': os.path.abspath(upload_dir),
        'output_dir': os.path.abspath(output_dir),
        'config_file': os.path.abspath(config_file)
    }


def create_user_folders(user_id, database_base_path, upload_folder, output_folder):
    """
    Create all necessary folders for a user.
    
    :param user_id: User ID
    :param database_base_path: Base path for database directory
    :param upload_folder: Base path for uploads directory
    :param output_folder: Base path for outputs directory
    :return: Dictionary of created folder paths (same as get_user_folders)
    """
    folders = get_user_folders(user_id, database_base_path, upload_folder, output_folder)
    
    try:
        # Create all user directories
        os.makedirs(folders['config_dir'], exist_ok=True)
        os.makedirs(folders['upload_dir'], exist_ok=True)
        os.makedirs(folders['output_dir'], exist_ok=True)
        
        logger.info(f"Created folders for user {user_id}")
        return folders
    except Exception as e:
        logger.error(f"Error creating folders for user {user_id}: {str(e)}")
        raise


def delete_user_folders(user_id, database_base_path, upload_folder, output_folder):
    """
    Delete all folders and files associated with a user.
    
    :param user_id: User ID
    :param database_base_path: Base path for database directory
    :param upload_folder: Base path for uploads directory
    :param output_folder: Base path for outputs directory
    :return: Dictionary with deletion results:
        - config_deleted: bool
        - upload_deleted: bool
        - output_deleted: bool
        - errors: list of error messages
    """
    folders = get_user_folders(user_id, database_base_path, upload_folder, output_folder)
    results = {
        'config_deleted': False,
        'upload_deleted': False,
        'output_deleted': False,
        'errors': []
    }
    
    # Delete config directory (contains userConfig.yaml and ZIP files)
    if os.path.exists(folders['config_dir']):
        try:
            shutil.rmtree(folders['config_dir'])
            results['config_deleted'] = True
            logger.info(f"Deleted config directory for user {user_id}: {folders['config_dir']}")
        except Exception as e:
            error_msg = f"Error deleting config directory for user {user_id}: {str(e)}"
            logger.error(error_msg)
            results['errors'].append(error_msg)
    
    # Delete upload directory
    if os.path.exists(folders['upload_dir']):
        try:
            shutil.rmtree(folders['upload_dir'])
            results['upload_deleted'] = True
            logger.info(f"Deleted upload directory for user {user_id}: {folders['upload_dir']}")
        except Exception as e:
            error_msg = f"Error deleting upload directory for user {user_id}: {str(e)}"
            logger.error(error_msg)
            results['errors'].append(error_msg)
    
    # Delete output directory
    if os.path.exists(folders['output_dir']):
        try:
            shutil.rmtree(folders['output_dir'])
            results['output_deleted'] = True
            logger.info(f"Deleted output directory for user {user_id}: {folders['output_dir']}")
        except Exception as e:
            error_msg = f"Error deleting output directory for user {user_id}: {str(e)}"
            logger.error(error_msg)
            results['errors'].append(error_msg)
    
    return results


def get_user_config_path(user_id, database_base_path):
    """
    Get the path to a user's configuration file.
    
    :param user_id: User ID
    :param database_base_path: Base path for database directory
    :return: Absolute path to userConfig.yaml
    """
    config_file = os.path.join(database_base_path, user_id, 'userConfig.yaml')
    return os.path.abspath(config_file)


def initialize_user_config(user_id, database_base_path, default_user_config_path):
    """
    Initialize a user's configuration file from the default template.
    
    :param user_id: User ID
    :param database_base_path: Base path for database directory
    :param default_user_config_path: Path to default userConfig.yaml template
    :return: Path to created config file
    """
    user_config_path = get_user_config_path(user_id, database_base_path)
    user_config_dir = os.path.dirname(user_config_path)
    
    # Create user config directory if it doesn't exist
    os.makedirs(user_config_dir, exist_ok=True)
    
    # Copy default config if user config doesn't exist
    if not os.path.exists(user_config_path):
        try:
            if os.path.exists(default_user_config_path):
                shutil.copy2(default_user_config_path, user_config_path)
                logger.info(f"Initialized user config for {user_id} from template")
            else:
                # Create empty config if template doesn't exist
                with open(user_config_path, 'w') as f:
                    yaml.dump({}, f, default_flow_style=False)
                logger.warning(f"Default user config template not found, created empty config for {user_id}")
        except Exception as e:
            logger.error(f"Error initializing user config for {user_id}: {str(e)}")
            raise
    
    return user_config_path


def user_exists(user_id, database_base_path):
    """
    Check if a user has folders and/or config file.
    
    :param user_id: User ID
    :param database_base_path: Base path for database directory
    :return: True if user has config file or folders, False otherwise
    """
    user_config_path = get_user_config_path(user_id, database_base_path)
    return os.path.exists(user_config_path)

