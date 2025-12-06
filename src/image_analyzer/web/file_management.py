"""
File Management Module

Provides functions for managing user folders, batch folders, and result files.
"""

import os
import shutil
from datetime import datetime
from src.image_analyzer.web.config import (
    get_output_folder,
    get_database_base_path
)


def get_user_folder(user_id, base_folder):
    """Get user-specific folder path (returns absolute path)"""
    user_folder = os.path.join(base_folder, user_id)
    user_folder = os.path.abspath(user_folder)  # Always use absolute path
    os.makedirs(user_folder, exist_ok=True)
    return user_folder


def get_user_zip_folder(user_id):
    """Get user-specific folder for ZIP files in database/user_id/ (returns absolute path)"""
    user_zip_folder = os.path.join(get_database_base_path(), user_id)
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
        user_output_folder = get_user_folder(user_id, get_output_folder())
        
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

