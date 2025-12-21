"""
Database Module

Provides database operations for user management including
token storage, quota management, and user CRUD operations.
"""

import sqlite3
import logging
from werkzeug.security import generate_password_hash, check_password_hash
from src.image_analyzer.web.config import get_database_file
from src.image_analyzer.utils import user_management
from src.image_analyzer.web.config import (
    get_database_base_path,
    get_upload_folder,
    get_output_folder
)

logger = logging.getLogger(__name__)


def get_db_connection():
    """Create a SQLite connection (auto-commit)"""
    db_file = get_database_file()
    conn = sqlite3.connect(db_file, check_same_thread=False)
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
            pro_features INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.commit()
    conn.close()


def upsert_user_token(username: str, raw_token: str, image_limit: int, pro_features: bool = False):
    """Create or replace a user's token hash, image limit, and pro_features status."""
    token_hash = generate_password_hash(raw_token)
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO users (username, token_hash, image_limit, pro_features)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(username) DO UPDATE SET
            token_hash=excluded.token_hash,
            image_limit=excluded.image_limit,
            pro_features=excluded.pro_features
        """,
        (username, token_hash, image_limit, 1 if pro_features else 0),
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


def update_user_pro_features(username: str, pro_features: bool):
    """Update only the pro_features status for an existing user."""
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        "UPDATE users SET pro_features = ? WHERE username = ?", 
        (1 if pro_features else 0, username)
    )
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
    cur.execute("SELECT username, image_limit, pro_features, created_at FROM users ORDER BY username")
    rows = [dict(r) for r in cur.fetchall()]
    # Convert pro_features from INTEGER (0/1) to boolean
    for row in rows:
        row['pro_features'] = bool(row.get('pro_features', 0))
    conn.close()
    return rows


def is_pro_user(username: str) -> bool:
    """Check if a user has pro features enabled."""
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT pro_features FROM users WHERE username = ?", (username,))
    row = cur.fetchone()
    conn.close()
    if not row:
        return False
    # sqlite3.Row supports dictionary-like access but not .get()
    return bool(row['pro_features'] or 0)


def delete_user(username: str):
    """
    Delete user by username.
    Deletes user from database and removes all associated folders.
    """
    # Delete user folders first (before database deletion)
    try:
        delete_results = user_management.delete_user_folders(
            username, 
            get_database_base_path(), 
            get_upload_folder(), 
            get_output_folder()
        )
        if delete_results['errors']:
            logger.warning(f"Some errors occurred while deleting folders for user {username}: {delete_results['errors']}")
    except Exception as e:
        logger.error(f"Error deleting folders for user {username}: {str(e)}")
        # Continue with database deletion even if folder deletion fails
    
    # Delete user from database
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("DELETE FROM users WHERE username = ?", (username,))
    conn.commit()
    conn.close()
    logger.info(f"Deleted user {username} from database")

