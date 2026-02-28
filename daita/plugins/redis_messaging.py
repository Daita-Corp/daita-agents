"""
Redis Messaging Plugin for Daita Agents.

Provides distributed messaging and persistent storage using Redis Streams and Pub/Sub.
Solves single-process limitation and memory-bound storage while maintaining the
existing RelayManager API.

Features:
- Redis Streams for message persistence and ordering
- Redis Pub/Sub for real-time message delivery
- Automatic TTL-based message cleanup
- Connection pooling and error handling
- Compatible with existing RelayManager interface

Example:
    ```python
    from daita.plugins import redis_messaging
    from daita.core.workflow import Workflow
    
    # Create Redis messaging plugin
    redis_msg = redis_messaging(url="redis://localhost:6379")
    
    # Use with workflow
    workflow = Workflow("Distributed Pipeline", messaging_plugin=redis_msg)
    workflow.connect("agent_a", "data_channel", "agent_b")
    ```
"""

import asyncio
import json
import logging
import time
import uuid
from typing import Dict, Any, Optional, List, Callable, Union
import weakref

logger = logging.getLogger(__name__)

class RedisMessagingPlugin:
    """
    Redis messaging plugin for distributed agent communication.
    
    Uses Redis Streams for message persistence and Pub/Sub for real-time delivery.
    Maintains compatibility with existing RelayManager interface while adding
    distributed capabilities.
    """
    
    def __init__(
        self,
        url: str = "redis://localhost:6379",
        max_connections: int = 10,
        message_ttl: int = 86400,  # 24 hours
        max_stream_length: int = 10000,
        connection_timeout: int = 30,
        **kwargs
    ):
        """
        Initialize Redis messaging plugin.
        
        Args:
            url: Redis connection URL
            max_connections: Maximum Redis connections in pool
            message_ttl: Message TTL in seconds (24 hours default)
            max_stream_length: Maximum messages per stream (10k default)
            connection_timeout: Connection timeout in seconds
            **kwargs: Additional Redis parameters
        """
        self.url = url
        self.max_connections = max_connections
        self.message_ttl = message_ttl
        self.max_stream_length = max_stream_length
        self.connection_timeout = connection_timeout
        self.extra_params = kwargs
        
        # Redis connections (lazy initialization)
        self._redis_pool = None
        self._pubsub_connections: Dict[str, Any] = {}
        
        # Subscribers: channel_name -> weak set of callbacks
        self._subscribers: Dict[str, weakref.WeakSet] = {}
        
        # Running state
        self._running = False
        
        logger.debug(f"RedisMessagingPlugin initialized (url: {url})")
    
    async def start(self) -> None:
        """Start the Redis messaging plugin."""
        if self._running:
            return
        
        try:
            import redis.asyncio as redis
        except ImportError:
            raise ImportError(
                "redis package required for RedisMessagingPlugin. "
                "Install with: pip install redis"
            )
        
        # Create connection pool
        self._redis_pool = redis.ConnectionPool.from_url(
            self.url,
            max_connections=self.max_connections,
            socket_connect_timeout=self.connection_timeout,
            socket_keepalive=True,
            socket_keepalive_options={},
            health_check_interval=30,
            **self.extra_params
        )
        
        self._running = True
        logger.info("RedisMessagingPlugin started")
    
    async def stop(self) -> None:
        """Stop the Redis messaging plugin and cleanup connections."""
        if not self._running:
            return
        
        self._running = False
        
        # Close all pubsub connections
        for channel, pubsub in self._pubsub_connections.items():
            try:
                await pubsub.unsubscribe(f"pubsub:{channel}")
                await pubsub.close()
            except Exception as e:
                logger.warning(f"Error closing pubsub connection for {channel}: {e}")
        
        self._pubsub_connections.clear()
        
        # Close connection pool
        if self._redis_pool:
            await self._redis_pool.disconnect()
            self._redis_pool = None
        
        logger.info("RedisMessagingPlugin stopped")
    
    async def _get_redis(self):
        """Get Redis client from pool."""
        if not self._running:
            await self.start()
        
        import redis.asyncio as redis
        return redis.Redis(connection_pool=self._redis_pool)
    
    async def publish(self, channel: str, message: Dict[str, Any], 
                     publisher: Optional[str] = None) -> str:
        """
        Publish message to Redis stream and notify via pub/sub.
        
        Args:
            channel: Channel name
            message: Message data  
            publisher: Optional publisher identifier
            
        Returns:
            Message ID
        """
        if not self._running:
            await self.start()
        
        redis_client = await self._get_redis()
        message_id = uuid.uuid4().hex
        
        try:
            # Prepare message for storage
            stored_message = {
                "id": message_id,
                "data": message,
                "publisher": publisher,
                "timestamp": time.time()
            }
            
            # Store in Redis Stream for persistence
            stream_key = f"stream:{channel}"
            await redis_client.xadd(
                stream_key,
                stored_message,
                id=message_id,
                maxlen=self.max_stream_length,
                approximate=True
            )
            
            # Set TTL on stream
            await redis_client.expire(stream_key, self.message_ttl)
            
            # Notify via pub/sub for real-time delivery
            pubsub_key = f"pubsub:{channel}"
            notification = {
                "message_id": message_id,
                "data": message,
                "publisher": publisher,
                "timestamp": time.time()
            }
            
            await redis_client.publish(pubsub_key, json.dumps(notification))
            
            logger.debug(f"Published message {message_id} to channel '{channel}'")
            return message_id
            
        except Exception as e:
            logger.error(f"Error publishing message to Redis: {e}")
            raise
        finally:
            await redis_client.close()
    
    async def subscribe(self, channel: str, callback: Callable) -> None:
        """
        Subscribe to channel with callback.
        
        Args:
            channel: Channel name
            callback: Callback function to receive messages
        """
        if not self._running:
            await self.start()
        
        # Add callback to subscribers
        if channel not in self._subscribers:
            self._subscribers[channel] = weakref.WeakSet()
        self._subscribers[channel].add(callback)
        
        # Start pubsub listener if not already running
        if channel not in self._pubsub_connections:
            await self._start_pubsub_listener(channel)
        
        logger.debug(f"Subscribed to channel '{channel}'")
    
    async def _start_pubsub_listener(self, channel: str) -> None:
        """Start pub/sub listener for a channel."""
        try:
            redis_client = await self._get_redis()
            pubsub = redis_client.pubsub()
            
            await pubsub.subscribe(f"pubsub:{channel}")
            self._pubsub_connections[channel] = pubsub
            
            # Start listener task
            task = asyncio.create_task(self._pubsub_message_handler(channel, pubsub))
            task.add_done_callback(lambda t: t.exception() if not t.cancelled() else None)
            
        except Exception as e:
            logger.error(f"Error starting pubsub listener for {channel}: {e}")
    
    async def _pubsub_message_handler(self, channel: str, pubsub) -> None:
        """Handle messages from pub/sub subscription."""
        try:
            async for message in pubsub.listen():
                if message['type'] == 'message':
                    try:
                        # Parse message data
                        data = json.loads(message['data'].decode())
                        message_data = data['data']
                        
                        # Notify all subscribers
                        await self._notify_subscribers(channel, message_data)
                        
                    except Exception as e:
                        logger.error(f"Error processing pubsub message: {e}")
        
        except asyncio.CancelledError:
            # Clean shutdown
            pass
        except Exception as e:
            logger.error(f"Error in pubsub message handler for {channel}: {e}")
        finally:
            # Cleanup
            try:
                await pubsub.unsubscribe(f"pubsub:{channel}")
                await pubsub.close()
            except Exception:
                pass
            
            self._pubsub_connections.pop(channel, None)
    
    async def _notify_subscribers(self, channel: str, message_data: Any) -> None:
        """Notify all subscribers of a channel."""
        if channel not in self._subscribers:
            return
        
        # Get snapshot of subscribers
        subscriber_list = list(self._subscribers[channel])
        
        if not subscriber_list:
            return
        
        # Notify all subscribers concurrently
        tasks = []
        for subscriber in subscriber_list:
            task = asyncio.create_task(self._call_subscriber(subscriber, message_data))
            task.add_done_callback(lambda t: t.exception() if not t.cancelled() else None)
            tasks.append(task)
        
        # Wait for all notifications
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _call_subscriber(self, callback: Callable, message_data: Any) -> None:
        """Safely call a subscriber callback."""
        try:
            if asyncio.iscoroutinefunction(callback):
                await callback(message_data)
            else:
                # Run sync callback in thread pool
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, callback, message_data)
        except Exception as e:
            logger.error(f"Error in subscriber callback: {str(e)}")
    
    async def get_latest(self, channel: str, count: int = 1) -> List[Any]:
        """
        Get latest messages from Redis stream.
        
        Args:
            channel: Channel name
            count: Number of messages to retrieve
            
        Returns:
            List of message data (newest first)
        """
        if not self._running:
            await self.start()
        
        redis_client = await self._get_redis()
        
        try:
            stream_key = f"stream:{channel}"
            
            # Get latest messages from stream (XREVRANGE for newest first)
            messages = await redis_client.xrevrange(stream_key, count=count)
            
            # Extract message data
            result = []
            for message_id, fields in messages:
                try:
                    # Parse the stored message data
                    data = fields.get(b'data', b'{}').decode()
                    message_data = json.loads(data) if data != '{}' else {}
                    result.append(message_data)
                except Exception as e:
                    logger.warning(f"Error parsing message {message_id}: {e}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error retrieving latest messages from Redis: {e}")
            return []
        finally:
            await redis_client.close()
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Check Redis connection health.
        
        Returns:
            Health status information
        """
        try:
            if not self._running:
                return {"status": "stopped", "connected": False}
            
            redis_client = await self._get_redis()
            
            # Test connection with ping
            start_time = time.time()
            await redis_client.ping()
            ping_time = (time.time() - start_time) * 1000  # ms
            
            # Get Redis info
            info = await redis_client.info()
            
            await redis_client.close()
            
            return {
                "status": "healthy",
                "connected": True,
                "ping_time_ms": round(ping_time, 2),
                "redis_version": info.get("redis_version", "unknown"),
                "used_memory_human": info.get("used_memory_human", "unknown"),
                "connected_clients": info.get("connected_clients", 0),
                "total_commands_processed": info.get("total_commands_processed", 0)
            }
            
        except Exception as e:
            return {
                "status": "unhealthy", 
                "connected": False,
                "error": str(e)
            }
    
    async def clear_channel(self, channel: str) -> bool:
        """
        Clear all messages from a channel.
        
        Args:
            channel: Channel name
            
        Returns:
            True if channel was cleared
        """
        if not self._running:
            await self.start()
        
        redis_client = await self._get_redis()
        
        try:
            stream_key = f"stream:{channel}"
            deleted = await redis_client.delete(stream_key)
            
            logger.debug(f"Cleared channel '{channel}' (deleted: {deleted})")
            return deleted > 0
            
        except Exception as e:
            logger.error(f"Error clearing Redis channel {channel}: {e}")
            return False
        finally:
            await redis_client.close()
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get Redis messaging statistics."""
        if not self._running:
            return {"status": "stopped"}
        
        try:
            redis_client = await self._get_redis()
            info = await redis_client.info()
            await redis_client.close()
            
            return {
                "status": "running",
                "redis_version": info.get("redis_version", "unknown"),
                "used_memory": info.get("used_memory", 0),
                "used_memory_human": info.get("used_memory_human", "0B"),
                "connected_clients": info.get("connected_clients", 0),
                "total_commands_processed": info.get("total_commands_processed", 0),
                "pubsub_channels": len(self._pubsub_connections),
                "subscriber_channels": len(self._subscribers)
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    # Context manager support
    async def __aenter__(self) -> "RedisMessagingPlugin":
        """Async context manager entry."""
        await self.start()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.stop()

# Factory function for easy instantiation
def redis_messaging(
    url: str = "redis://localhost:6379",
    max_connections: int = 10,
    message_ttl: int = 86400,
    max_stream_length: int = 10000,
    connection_timeout: int = 30,
    **kwargs
) -> RedisMessagingPlugin:
    """
    Create a Redis messaging plugin instance.
    
    Args:
        url: Redis connection URL (default: redis://localhost:6379)
        max_connections: Maximum Redis connections in pool (default: 10)
        message_ttl: Message TTL in seconds (default: 24 hours)
        max_stream_length: Maximum messages per stream (default: 10k)
        connection_timeout: Connection timeout in seconds (default: 30)
        **kwargs: Additional Redis parameters
        
    Returns:
        RedisMessagingPlugin instance
        
    Example:
        ```python
        # Basic usage
        redis_msg = redis_messaging()
        
        # Production configuration
        redis_msg = redis_messaging(
            url="redis://redis-cluster:6379",
            max_connections=20,
            message_ttl=7*24*3600,  # 7 days
            max_stream_length=50000
        )
        ```
    """
    return RedisMessagingPlugin(
        url=url,
        max_connections=max_connections,
        message_ttl=message_ttl,
        max_stream_length=max_stream_length,
        connection_timeout=connection_timeout,
        **kwargs
    )