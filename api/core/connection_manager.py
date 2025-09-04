# ABOUTME: This file implements WebSocket connection management for streaming audio generation.
# ABOUTME: Provides connection tracking, graceful disconnection, and error broadcasting capabilities.

import asyncio
import logging
import uuid
from typing import Dict, Set, Optional, Callable
from fastapi import WebSocket
from fastapi.websockets import WebSocketState

logger = logging.getLogger(__name__)


class WebSocketConnectionManager:
    """Manages WebSocket connections for streaming audio generation.
    
    This class tracks active connections, handles graceful disconnections,
    and provides error broadcasting capabilities for the WebSocket streaming
    endpoint.
    """
    
    def __init__(self):
        # Active connections mapped by connection ID
        self._connections: Dict[str, WebSocket] = {}
        
        # Set of connection IDs currently generating audio
        self._active_generations: Set[str] = set()
        
        # Cancellation callbacks for each active generation
        self._cancellation_callbacks: Dict[str, Callable[[], None]] = {}
    
    async def connect(self, websocket: WebSocket) -> str:
        """Accept a new WebSocket connection and assign it an ID.
        
        Args:
            websocket: The WebSocket connection to accept
            
        Returns:
            The assigned connection ID
        """
        connection_id = str(uuid.uuid4())
        
        await websocket.accept()
        self._connections[connection_id] = websocket
        
        logger.info(f"WebSocket connection accepted: {connection_id}")
        return connection_id
    
    async def disconnect(self, connection_id: str, code: int = 1000):
        """Disconnect a WebSocket connection gracefully.
        
        Args:
            connection_id: The connection ID to disconnect
            code: WebSocket close code (1000=normal, 1011=internal error, 1013=try again)
        """
        if connection_id in self._connections:
            websocket = self._connections[connection_id]
            
            # Cancel any active generation
            await self.cancel_generation(connection_id)
            
            # Close the WebSocket if still connected
            if websocket.client_state == WebSocketState.CONNECTED:
                try:
                    await websocket.close(code=code)
                except Exception as e:
                    logger.warning(f"Error closing WebSocket {connection_id}: {e}")
            
            # Remove from tracking
            del self._connections[connection_id]
            logger.info(f"WebSocket connection closed: {connection_id} (code: {code})")
    
    def is_connected(self, connection_id: str) -> bool:
        """Check if a connection is still active.
        
        Args:
            connection_id: The connection ID to check
            
        Returns:
            True if the connection is active and connected
        """
        if connection_id not in self._connections:
            return False
        
        websocket = self._connections[connection_id]
        return websocket.client_state == WebSocketState.CONNECTED
    
    async def send_text(self, connection_id: str, message: str):
        """Send a text message to a specific connection.
        
        Args:
            connection_id: The connection ID to send to
            message: The text message to send
        """
        if not self.is_connected(connection_id):
            logger.warning(f"Attempted to send text to disconnected WebSocket: {connection_id}")
            return
        
        websocket = self._connections[connection_id]
        try:
            await websocket.send_text(message)
        except Exception as e:
            logger.error(f"Error sending text to WebSocket {connection_id}: {e}")
            await self.disconnect(connection_id, code=1011)  # Internal error
    
    async def send_bytes(self, connection_id: str, data: bytes):
        """Send binary data to a specific connection.
        
        Args:
            connection_id: The connection ID to send to
            data: The binary data to send
        """
        if not self.is_connected(connection_id):
            logger.warning(f"Attempted to send bytes to disconnected WebSocket: {connection_id}")
            return
        
        websocket = self._connections[connection_id]
        try:
            await websocket.send_bytes(data)
        except Exception as e:
            logger.error(f"Error sending bytes to WebSocket {connection_id}: {e}")
            await self.disconnect(connection_id, code=1011)  # Internal error
    
    def start_generation(self, connection_id: str, cancel_callback: Optional[Callable[[], None]] = None):
        """Mark a connection as having an active generation.
        
        Args:
            connection_id: The connection ID starting generation
            cancel_callback: Optional callback to cancel the generation
        """
        self._active_generations.add(connection_id)
        
        if cancel_callback:
            self._cancellation_callbacks[connection_id] = cancel_callback
        
        logger.debug(f"Started generation for WebSocket: {connection_id}")
    
    async def cancel_generation(self, connection_id: str):
        """Cancel an active generation for a connection.
        
        Args:
            connection_id: The connection ID to cancel generation for
        """
        if connection_id in self._active_generations:
            # Call the cancellation callback if available
            if connection_id in self._cancellation_callbacks:
                callback = self._cancellation_callbacks[connection_id]
                try:
                    callback()
                except Exception as e:
                    logger.error(f"Error calling cancellation callback for {connection_id}: {e}")
                
                del self._cancellation_callbacks[connection_id]
            
            self._active_generations.remove(connection_id)
            logger.debug(f"Cancelled generation for WebSocket: {connection_id}")
    
    def finish_generation(self, connection_id: str):
        """Mark a generation as finished for a connection.
        
        Args:
            connection_id: The connection ID that finished generation
        """
        if connection_id in self._active_generations:
            self._active_generations.remove(connection_id)
        
        if connection_id in self._cancellation_callbacks:
            del self._cancellation_callbacks[connection_id]
        
        logger.debug(f"Finished generation for WebSocket: {connection_id}")
    
    def is_generating(self, connection_id: str) -> bool:
        """Check if a connection has an active generation.
        
        Args:
            connection_id: The connection ID to check
            
        Returns:
            True if the connection is currently generating audio
        """
        return connection_id in self._active_generations
    
    async def broadcast_error(self, error_message: str, code: int = 1011):
        """Broadcast an error message to all active connections.
        
        Args:
            error_message: The error message to broadcast
            code: WebSocket close code to use when disconnecting
        """
        connection_ids = list(self._connections.keys())
        
        for connection_id in connection_ids:
            try:
                await self.send_text(connection_id, error_message)
                await asyncio.sleep(0.1)  # Brief delay before disconnect
                await self.disconnect(connection_id, code=code)
            except Exception as e:
                logger.error(f"Error broadcasting to WebSocket {connection_id}: {e}")
    
    async def cleanup_disconnected(self):
        """Remove connections that have been disconnected by clients.
        
        This should be called periodically to clean up stale connections.
        """
        disconnected_ids = []
        
        for connection_id, websocket in self._connections.items():
            if websocket.client_state != WebSocketState.CONNECTED:
                disconnected_ids.append(connection_id)
        
        for connection_id in disconnected_ids:
            logger.info(f"Cleaning up disconnected WebSocket: {connection_id}")
            await self.cancel_generation(connection_id)
            del self._connections[connection_id]
    
    def get_connection_count(self) -> int:
        """Get the number of active connections.
        
        Returns:
            The number of currently active WebSocket connections
        """
        return len(self._connections)
    
    def get_active_generations_count(self) -> int:
        """Get the number of active generations.
        
        Returns:
            The number of connections currently generating audio
        """
        return len(self._active_generations)


# Global connection manager instance
connection_manager = WebSocketConnectionManager()