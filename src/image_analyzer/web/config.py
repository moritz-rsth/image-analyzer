"""
Web Application Configuration Module

Centralizes all application configuration including paths, deployment settings,
and environment variables.
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Flask configuration
SECRET_KEY = os.getenv('SECRET_KEY', 'your-secret-key-change-in-production')

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

# Folder paths
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'

# Config file paths
DEPLOYMENT_CONFIG_FILE = os.path.join('config', 'deploymentConfig.yaml')
DEFAULT_USER_CONFIG_FILE = os.path.join('config', 'userConfig.yaml')
# Legacy config files (kept for backward compatibility)
DEFAULT_CONFIG_FILE = os.path.join('config', 'configuration_deployment.yaml')
CURR_CONFIG_FILE = os.path.join('config', 'configuration.yaml')

# Database path configuration (prefers explicit env, then Railway volume, else local folder)
DATABASE_BASE_PATH = os.getenv('DATABASE_PATH') or os.getenv('RAILWAY_VOLUME_MOUNT_PATH') or './database'
os.environ['DATABASE_PATH'] = DATABASE_BASE_PATH  # expose for downstream code/tools
DB_FILE = os.path.join(DATABASE_BASE_PATH, 'user.db')

# Initialize directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(DATABASE_BASE_PATH, exist_ok=True)


# Getter functions for configuration values
def get_upload_folder():
    """Get the upload folder path."""
    return UPLOAD_FOLDER


def get_output_folder():
    """Get the output folder path."""
    return OUTPUT_FOLDER


def get_database_base_path():
    """Get the database base path."""
    return DATABASE_BASE_PATH


def get_database_file():
    """Get the database file path."""
    return DB_FILE


def get_deployment_config_file():
    """Get the deployment config file path."""
    return DEPLOYMENT_CONFIG_FILE


def get_default_user_config_file():
    """Get the default user config file path."""
    return DEFAULT_USER_CONFIG_FILE


def get_default_config_file():
    """Get the default config file path (legacy)."""
    return DEFAULT_CONFIG_FILE


def get_curr_config_file():
    """Get the current config file path (legacy)."""
    return CURR_CONFIG_FILE


def get_deployment_platform():
    """Get the deployment platform ('railway' or 'local')."""
    return DEPLOYMENT_PLATFORM


def get_port():
    """Get the port number."""
    return PORT


def get_secret_key():
    """Get the Flask secret key."""
    return SECRET_KEY

