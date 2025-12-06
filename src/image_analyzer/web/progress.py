"""
Progress Management Module

Handles progress tracking and WebSocket updates for image processing.
"""

import time

# Per-user processing state: {user_id: {'session_id': str, 'status': str, 'progress': dict, 'started_at': timestamp}}
user_processing = {}

# SocketIO instance (will be set by app factory)
_socketio = None


def set_socketio(socketio_instance):
    """Set the SocketIO instance for progress updates."""
    global _socketio
    _socketio = socketio_instance


def get_socketio():
    """Get the SocketIO instance."""
    return _socketio


def progress_callback(progress_data, user_session_id):
    """Callback function to emit progress updates via WebSocket"""
    if user_session_id in user_processing:
        processing_info = user_processing[user_session_id]
        room_id = processing_info.get('session_id')
        
        # Update stored progress state
        processing_info['progress'] = progress_data
        processing_info['status'] = progress_data.get('status', 'processing')
        
        # Try to emit progress update, but don't fail if session is disconnected
        if _socketio:
            try:
                _socketio.emit('progress_update', progress_data, room=room_id, namespace='/')
                # Small sleep to allow the event to be processed and sent immediately
                time.sleep(0.01)
            except (KeyError, RuntimeError) as e:
                # Session disconnected - silently ignore but keep progress state
                print(f"Could not send progress update (session disconnected): {e}")


def get_processing_state(user_id):
    """Get the processing state for a user."""
    return user_processing.get(user_id)


def set_processing_state(user_id, state):
    """Set the processing state for a user."""
    user_processing[user_id] = state


def clear_processing_state(user_id):
    """Clear the processing state for a user."""
    if user_id in user_processing:
        del user_processing[user_id]

