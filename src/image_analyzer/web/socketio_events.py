"""
Socket.IO Events Module

Handles Socket.IO event handlers for WebSocket connections.
"""

from flask_socketio import emit, join_room


def register_socketio_events(socketio):
    """
    Register Socket.IO event handlers with the given SocketIO instance.
    
    :param socketio: SocketIO instance
    """
    
    @socketio.on('connect')
    def handle_connect():
        """Handle client connection"""
        emit('connected', {'data': 'Connected'})
        
        # Check if user has active processing and send current state
        # Note: We can't access session directly in Socket.IO, so we'll rely on
        # the client to check status via HTTP endpoint after connecting
    
    @socketio.on('disconnect')
    def handle_disconnect():
        """Handle client disconnection - silently ignore, session cleanup is automatic"""
        # Socket.IO automatically cleans up disconnected sessions
        # No need to manually clean up here as background tasks handle errors gracefully
        pass
    
    @socketio.on('join_session')
    def handle_join_session(data):
        """Handle client joining a processing session"""
        session_id = data.get('session_id')
        if session_id:
            join_room(session_id)
            emit('joined_session', {'session_id': session_id})

