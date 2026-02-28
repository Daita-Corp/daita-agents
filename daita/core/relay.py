"""
Simplified Relay System for Daita Agents.

Provides simple message passing between agents in a workflow.
Focus on essential communication functionality without complex features.

Updated to pass only result data between agents for cleaner interfaces.

Example:
    ```python
    from daita.core.relay import RelayManager
    
    # Initialize relay manager
    relay = RelayManager()
    
    # Agent publishes data (full response gets stored, only result gets forwarded)
    await relay.publish("data_channel", {
        "status": "success", 
        "result": {"processed": True},
        "agent_id": "agent_123"
    })
    
    # Another agent subscribes and receives only: {"processed": True}
    async def handle_data(result_data):
        print(f"Got result: {result_data}")
    
    await relay.subscribe("data_channel", handle_data)
    ```
"""
import asyncio
import logging
import random
import time
import uuid
from typing import Dict, Any, Optional, List, Callable, Union
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
import weakref

try:
    from ..core.exceptions import DaitaError, AcknowledgmentTimeoutError
except ImportError:
    # Fallback for direct execution or testing
    from core.exceptions import DaitaError, AcknowledgmentTimeoutError

logger = logging.getLogger(__name__)

class RelayError(DaitaError):
    """Exception raised for relay-related errors."""
    pass

class MessageStatus(str, Enum):
    """Status of a message in the relay system."""
    PENDING = "pending"
    ACKNOWLEDGED = "acknowledged"
    FAILED = "failed"
    TIMEOUT = "timeout"

@dataclass
class ReliableMessage:
    """A message with acknowledgment tracking and metadata."""
    id: str
    channel: str
    data: Any
    metadata: Dict[str, Any] = field(default_factory=dict)
    publisher: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    status: MessageStatus = MessageStatus.PENDING
    ack_timeout: float = 30.0  # Default 30 second timeout
    error: Optional[str] = None
    attempts: int = 0
    max_attempts: int = 3

class RelayManager:
    """
    Simple relay manager for agent-to-agent communication.

    Handles message publishing and subscription with minimal complexity.
    Now passes only result data to maintain clean interfaces.
    """

    def __init__(
        self,
        max_messages_per_channel: int = 10,
        enable_reliability: bool = False,
        default_ack_timeout: float = 30.0
    ):
        """
        Initialize the relay manager.

        Args:
            max_messages_per_channel: Maximum messages to keep per channel (reduced to 10 for memory efficiency)
            enable_reliability: Enable message acknowledgments and reliability features
            default_ack_timeout: Default acknowledgment timeout in seconds
        """
        self.max_messages_per_channel = max_messages_per_channel
        self.enable_reliability = enable_reliability
        self.default_ack_timeout = default_ack_timeout

        # Channel storage: channel_name -> deque of result data only
        self.channels: Dict[str, deque] = {}

        # Subscribers: channel_name -> set of callbacks
        # Using regular set instead of WeakSet to prevent premature garbage collection
        self.subscribers: Dict[str, set] = {}

        # Per-channel locks to prevent race conditions between publish and subscribe
        self._channel_locks: Dict[str, asyncio.Lock] = {}

        # Subscriber error tracking
        self.subscriber_errors: deque = deque(maxlen=100)

        # Reliability features (when enabled)
        self.pending_messages: Dict[str, ReliableMessage] = {} if enable_reliability else {}
        self.message_timeouts: Dict[str, asyncio.Task] = {} if enable_reliability else {}

        # Running state
        self._running = False

        logger.debug(f"RelayManager initialized (reliability: {enable_reliability})")
    
    async def start(self) -> None:
        """Start the relay manager."""
        if self._running:
            return

        self._running = True
        logger.info("RelayManager started")

    async def stop(self) -> None:
        """Stop the relay manager and cleanup."""
        if not self._running:
            return

        self._running = False

        # Clear all channels and subscribers
        self.channels.clear()
        self.subscribers.clear()

        logger.info("RelayManager stopped")
    
    def _ensure_channel(self, channel: str) -> None:
        """Ensure channel exists."""
        if channel not in self.channels:
            self.channels[channel] = deque(maxlen=self.max_messages_per_channel)
            self.subscribers[channel] = set()  # Regular set to prevent garbage collection
            self._channel_locks[channel] = asyncio.Lock()

    def _extract_metadata(self, agent_response: Dict[str, Any], publisher: Optional[str]) -> Dict[str, Any]:
        """
        Extract metadata from agent response for context propagation.

        Automatically captures metrics, token usage, confidence scores, and other
        context that downstream agents need to make intelligent decisions.
        """
        metadata = {
            'upstream_agent': publisher or agent_response.get('agent_name'),
            'upstream_agent_id': agent_response.get('agent_id'),
            'timestamp': agent_response.get('timestamp', time.time()),
        }

        # Extract processing metrics if available
        if 'processing_time_ms' in agent_response:
            metadata['processing_time_ms'] = agent_response['processing_time_ms']

        # Extract token usage if available
        if 'token_usage' in agent_response:
            metadata['token_usage'] = agent_response['token_usage']

        # Extract confidence score if available
        if 'confidence_score' in agent_response:
            metadata['confidence_score'] = agent_response['confidence_score']
        elif 'confidence' in agent_response:
            metadata['confidence_score'] = agent_response['confidence']

        # Extract record count if available
        if 'record_count' in agent_response:
            metadata['record_count'] = agent_response['record_count']

        # Extract error information if available
        if 'error_count' in agent_response:
            metadata['error_count'] = agent_response['error_count']

        # Extract correlation ID for distributed tracing
        if 'correlation_id' in agent_response:
            metadata['correlation_id'] = agent_response['correlation_id']
        elif 'context' in agent_response and isinstance(agent_response['context'], dict):
            metadata['correlation_id'] = agent_response['context'].get('correlation_id', str(uuid.uuid4()))
        else:
            metadata['correlation_id'] = str(uuid.uuid4())

        # Extract retry information if available
        if 'retry_info' in agent_response:
            metadata['retry_info'] = agent_response['retry_info']

        # Extract status (success/error)
        metadata['status'] = agent_response.get('status', 'unknown')

        return metadata
    
    async def publish(
        self,
        channel: str,
        data: Any,
        publisher: Optional[str] = None,
        require_ack: Optional[bool] = None
    ) -> Optional[str]:
        """
        Publish data to a channel with automatic metadata propagation.

        Args:
            channel: Channel name
            data: Data to publish (can be any type - dict, list, str, etc.)
                  If dict with 'result' field, extracts result and metadata (backward compat)
                  Otherwise, publishes data as-is
            publisher: Optional publisher identifier
            require_ack: Whether this message requires acknowledgment (overrides global setting)

        Returns:
            Message ID if reliability is enabled, None otherwise
        """
        if not self._running:
            await self.start()

        # Handle both new API (raw data) and old API (agent response dict)
        if isinstance(data, dict) and 'result' in data:
            # Old API: Extract result and metadata from agent response
            result_data = data['result']
            metadata = self._extract_metadata(data, publisher)
        else:
            # New API: Use data directly, create minimal metadata
            result_data = data
            metadata = {
                'publisher': publisher,
                'timestamp': time.time(),
                'correlation_id': str(uuid.uuid4()),
                'status': 'success'
            }

        self._ensure_channel(channel)

        # Determine if we need reliability
        needs_reliability = require_ack if require_ack is not None else self.enable_reliability

        if needs_reliability:
            return await self._publish_reliable(channel, result_data, metadata, publisher)
        else:
            return await self._publish_fire_and_forget(channel, result_data, metadata, publisher)
    
    async def _publish_fire_and_forget(
        self,
        channel: str,
        result_data: Any,
        metadata: Dict[str, Any],
        publisher: Optional[str]
    ) -> None:
        """Publish message without reliability features with metadata propagation."""
        # Use per-channel lock to make publish atomic
        async with self._channel_locks[channel]:
            # Create message with result data and metadata
            message = {
                'data': result_data,
                'metadata': metadata,
                'publisher': publisher,
                'timestamp': time.time()
            }

            # Store result message
            self.channels[channel].append(message)

            # Notify subscribers with result data and metadata (while holding lock)
            await self._notify_subscribers(channel, result_data, metadata)

        logger.debug(f"Published result to channel '{channel}' from {publisher}")
        return None
    
    async def _publish_reliable(
        self,
        channel: str,
        result_data: Any,
        metadata: Dict[str, Any],
        publisher: Optional[str]
    ) -> str:
        """Publish message with reliability features and metadata propagation."""
        # Create reliable message
        message_id = uuid.uuid4().hex
        reliable_message = ReliableMessage(
            id=message_id,
            channel=channel,
            data=result_data,
            metadata=metadata,
            publisher=publisher,
            ack_timeout=self.default_ack_timeout
        )

        # Store pending message
        self.pending_messages[message_id] = reliable_message

        # Use per-channel lock to make publish atomic
        async with self._channel_locks[channel]:
            # Create message for channel storage with metadata
            message = {
                'id': message_id,
                'data': result_data,
                'metadata': metadata,
                'publisher': publisher,
                'timestamp': time.time(),
                'requires_ack': True
            }

            # Store result message
            self.channels[channel].append(message)

            # Set up timeout task
            timeout_task = asyncio.create_task(
                self._handle_message_timeout(message_id, reliable_message.ack_timeout)
            )
            timeout_task.add_done_callback(lambda t: t.exception() if not t.cancelled() else None)
            self.message_timeouts[message_id] = timeout_task

            # Notify subscribers with message ID for acknowledgment (while holding lock)
            await self._notify_subscribers_reliable(channel, result_data, metadata, message_id)

        logger.debug(f"Published reliable message {message_id} to channel '{channel}' from {publisher}")
        return message_id

    async def _notify_subscribers(self, channel: str, result_data: Any, metadata: Dict[str, Any]) -> None:
        """Notify all subscribers of a channel with result data and metadata."""
        if channel not in self.subscribers:
            return

        # Get snapshot of subscribers to avoid modification during iteration
        subscriber_list = list(self.subscribers[channel])

        if not subscriber_list:
            return

        # Notify all subscribers concurrently
        tasks = []
        for subscriber in subscriber_list:
            task = asyncio.create_task(self._call_subscriber(subscriber, result_data, metadata))
            task.add_done_callback(lambda t: t.exception() if not t.cancelled() else None)
            tasks.append(task)

        # Wait for all notifications to complete
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _notify_subscribers_reliable(self, channel: str, result_data: Any, metadata: Dict[str, Any], message_id: str) -> None:
        """Notify all subscribers of a reliable message with metadata."""
        if channel not in self.subscribers:
            return

        # Get snapshot of subscribers to avoid modification during iteration
        subscriber_list = list(self.subscribers[channel])

        if not subscriber_list:
            return

        # Notify all subscribers concurrently with message ID and metadata
        tasks = []
        for subscriber in subscriber_list:
            task = asyncio.create_task(
                self._call_subscriber_reliable(subscriber, result_data, metadata, message_id)
            )
            task.add_done_callback(lambda t: t.exception() if not t.cancelled() else None)
            tasks.append(task)

        # Wait for all notifications to complete
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _call_subscriber_reliable(self, callback: Callable, result_data: Any, metadata: Dict[str, Any], message_id: str) -> None:
        """Safely call a subscriber callback for reliable message with metadata."""
        try:
            if asyncio.iscoroutinefunction(callback):
                # Check callback signature to determine what parameters to pass
                import inspect
                sig = inspect.signature(callback)
                params = list(sig.parameters.keys())

                # Call with appropriate parameters based on signature
                if 'message_id' in params and 'metadata' in params:
                    await callback(result_data, metadata=metadata, message_id=message_id)
                elif 'message_id' in params:
                    await callback(result_data, message_id=message_id)
                elif 'metadata' in params:
                    await callback(result_data, metadata=metadata)
                else:
                    await callback(result_data)
            else:
                # Run sync callback in thread pool
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, callback, result_data)
        except Exception as e:
            logger.error(f"Error in reliable subscriber callback: {str(e)}")
            # NACK the message on callback error
            await self.nack_message(message_id, str(e))
    
    async def _call_subscriber(self, callback: Callable, result_data: Any, metadata: Dict[str, Any]) -> None:
        """Safely call a subscriber callback with result data and metadata."""
        try:
            if asyncio.iscoroutinefunction(callback):
                # Check callback signature to determine what parameters to pass
                import inspect
                sig = inspect.signature(callback)
                params = list(sig.parameters.keys())

                # Call with metadata if callback accepts it
                if 'metadata' in params:
                    await callback(result_data, metadata=metadata)
                else:
                    await callback(result_data)
            else:
                # Run sync callback in thread pool
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, callback, result_data)
        except Exception as e:
            error_info = {
                'callback': str(callback),
                'error': str(e),
                'error_type': type(e).__name__,
                'timestamp': time.time(),
                'data_preview': str(result_data)[:100]
            }
            self.subscriber_errors.append(error_info)
            logger.error(f"Subscriber callback failed: {e}")
            # Don't re-raise - we want other subscribers to continue
    
    async def subscribe(self, channel: str, callback: Callable) -> None:
        """
        Subscribe to a channel.

        Args:
            channel: Channel name
            callback: Callback function to receive result data
        """
        self._ensure_channel(channel)
        self.subscribers[channel].add(callback)

        logger.debug(f"Subscribed to channel '{channel}'")
    
    def unsubscribe(self, channel: str, callback: Callable) -> bool:
        """
        Unsubscribe from a channel.
        
        Args:
            channel: Channel name
            callback: Callback to remove
            
        Returns:
            True if callback was removed
        """
        if channel not in self.subscribers:
            return False
        
        try:
            self.subscribers[channel].remove(callback)
            logger.debug(f"Unsubscribed from channel '{channel}'")
            return True
        except KeyError:
            return False
    
    async def get_latest(self, channel: str, count: int = 1) -> List[Any]:
        """
        Get latest result data from a channel.

        Args:
            channel: Channel name
            count: Number of messages to retrieve

        Returns:
            List of result data (newest first)
        """
        if channel not in self.channels:
            return []

        # Get latest messages
        messages = list(self.channels[channel])
        latest = messages[-count:] if count < len(messages) else messages
        latest.reverse()  # Newest first

        # Return just the result data
        return [msg['data'] for msg in latest]

    def list_channels(self) -> List[str]:
        """List all channels."""
        return list(self.channels.keys())
    
    def clear_channel(self, channel: str) -> bool:
        """
        Clear all messages from a channel.
        
        Args:
            channel: Channel name
            
        Returns:
            True if channel was cleared
        """
        if channel not in self.channels:
            return False

        self.channels[channel].clear()

        logger.debug(f"Cleared channel '{channel}'")
        return True
    
    def get_stats(self) -> Dict[str, Any]:
        """Get relay manager statistics."""
        total_messages = sum(len(ch) for ch in self.channels.values())
        total_subscribers = sum(len(subs) for subs in self.subscribers.values())

        stats = {
            'running': self._running,
            'total_channels': len(self.channels),
            'total_messages': total_messages,
            'total_subscribers': total_subscribers,
            'channels': list(self.channels.keys()),
            'subscriber_errors_count': len(self.subscriber_errors)
        }

        # Add per-channel error breakdown
        if self.subscriber_errors:
            stats['errors_by_channel'] = {}
            for channel in self.channels.keys():
                channel_errors = sum(
                    1 for err in self.subscriber_errors
                    if channel in err.get('callback', '')
                )
                if channel_errors > 0:
                    stats['errors_by_channel'][channel] = channel_errors

        # Add reliability stats if enabled
        if self.enable_reliability:
            stats.update({
                'reliability_enabled': True,
                'pending_messages': len(self.pending_messages),
                'active_timeouts': len(self.message_timeouts)
            })
        else:
            stats['reliability_enabled'] = False

        return stats
    
    # Reliability methods (acknowledgments and timeouts)
    
    async def ack_message(self, message_id: str) -> bool:
        """
        Acknowledge a message as successfully processed.
        
        Args:
            message_id: ID of the message to acknowledge
            
        Returns:
            True if message was acknowledged, False if not found
        """
        if not self.enable_reliability or message_id not in self.pending_messages:
            return False
        
        # Update message status
        message = self.pending_messages[message_id]
        message.status = MessageStatus.ACKNOWLEDGED
        
        # Cancel timeout task
        if message_id in self.message_timeouts:
            timeout_task = self.message_timeouts.pop(message_id)
            if not timeout_task.done():
                timeout_task.cancel()
        
        # Remove from pending
        del self.pending_messages[message_id]
        
        logger.debug(f"Acknowledged message {message_id}")
        return True
    
    async def nack_message(self, message_id: str, error: str) -> bool:
        """
        Negative acknowledge a message (processing failed).
        
        Args:
            message_id: ID of the message to NACK
            error: Error message
            
        Returns:
            True if message was NACKed, False if not found
        """
        if not self.enable_reliability or message_id not in self.pending_messages:
            return False
        
        # Update message status
        message = self.pending_messages[message_id]
        message.status = MessageStatus.FAILED
        message.error = error
        message.attempts += 1
        
        # Cancel timeout task
        if message_id in self.message_timeouts:
            timeout_task = self.message_timeouts.pop(message_id)
            if not timeout_task.done():
                timeout_task.cancel()
        
        # Check if we should retry
        if message.attempts < message.max_attempts:
            logger.warning(f"Message {message_id} failed (attempt {message.attempts}/{message.max_attempts}): {error}")
            await self._schedule_retry(message)
        else:
            logger.error(f"Message {message_id} failed permanently after {message.attempts} attempts: {error}")
            # Remove from pending messages
            del self.pending_messages[message_id]
        
        return True
    
    async def _handle_message_timeout(self, message_id: str, timeout_duration: float) -> None:
        """Handle message acknowledgment timeout."""
        try:
            await asyncio.sleep(timeout_duration)
            
            # Check if message is still pending
            if message_id in self.pending_messages:
                message = self.pending_messages[message_id]
                message.status = MessageStatus.TIMEOUT
                message.attempts += 1
                
                logger.warning(f"Message {message_id} timed out after {timeout_duration}s")
                
                # Check if we should retry
                if message.attempts < message.max_attempts:
                    await self._schedule_retry(message)
                else:
                    logger.error(f"Message {message_id} timed out permanently after {message.attempts} attempts")
                    # Remove from pending messages
                    del self.pending_messages[message_id]
                
                # Remove timeout task
                self.message_timeouts.pop(message_id, None)
        
        except asyncio.CancelledError:
            # Timeout was cancelled (message was acknowledged)
            pass
        except Exception as e:
            logger.error(f"Error handling timeout for message {message_id}: {e}")
    
    def _calculate_retry_delay(self, attempt: int, base_delay: float = 1.0) -> float:
        """Calculate retry delay with exponential backoff and jitter."""
        # Exponential backoff: 1s, 2s, 4s, 8s, ...
        delay = base_delay * (2 ** attempt)

        # Cap at 60 seconds
        delay = min(delay, 60.0)

        # Add ±50% jitter to prevent thundering herd
        jitter_factor = 0.5 + random.random()  # Random between 0.5 and 1.5
        delay *= jitter_factor

        return delay

    async def _schedule_retry(self, message: ReliableMessage) -> None:
        """Schedule a retry for a failed or timed out message."""
        # Calculate exponential backoff delay with jitter
        delay = self._calculate_retry_delay(message.attempts - 1)

        logger.debug(f"Scheduling retry for message {message.id} in {delay:.1f}s (attempt {message.attempts + 1})")
        
        # Reset message status for retry
        message.status = MessageStatus.PENDING
        message.timestamp = time.time()  # Update timestamp for retry
        
        # Schedule the retry task
        retry_task = asyncio.create_task(self._execute_retry(message, delay))
        retry_task.add_done_callback(lambda t: t.exception() if not t.cancelled() else None)
        # Use a different key to avoid conflicts with timeout tasks
        self.message_timeouts[f"{message.id}_retry"] = retry_task
    
    async def _execute_retry(self, message: ReliableMessage, delay: float) -> None:
        """Execute a message retry after the specified delay."""
        try:
            await asyncio.sleep(delay)
            
            # Check if message is still in pending (not cancelled)
            if message.id in self.pending_messages:
                logger.debug(f"Retrying message {message.id} (attempt {message.attempts + 1})")
                
                # Republish the message by re-adding it to the channel
                self._ensure_channel(message.channel)
                
                # Create new message entry in channel
                retry_message = {
                    'id': message.id,
                    'data': message.data,
                    'metadata': message.metadata,
                    'publisher': message.publisher,
                    'timestamp': message.timestamp,
                    'requires_ack': True
                }

                # Add to channel
                self.channels[message.channel].append(retry_message)

                # Set up new timeout task for the retry
                timeout_task = asyncio.create_task(
                    self._handle_message_timeout(message.id, message.ack_timeout)
                )
                timeout_task.add_done_callback(lambda t: t.exception() if not t.cancelled() else None)
                self.message_timeouts[message.id] = timeout_task

                # Notify subscribers again
                await self._notify_subscribers_reliable(message.channel, message.data, message.metadata, message.id)
            
            # Clean up retry task reference
            self.message_timeouts.pop(f"{message.id}_retry", None)
        
        except asyncio.CancelledError:
            # Retry was cancelled
            pass
        except Exception as e:
            logger.error(f"Error executing retry for message {message.id}: {e}")
    
    def get_pending_messages(self) -> List[Dict[str, Any]]:
        """Get list of pending messages waiting for acknowledgment."""
        if not self.enable_reliability:
            return []
        
        return [
            {
                'id': msg.id,
                'channel': msg.channel,
                'publisher': msg.publisher,
                'status': msg.status.value,
                'timestamp': msg.timestamp,
                'attempts': msg.attempts,
                'max_attempts': msg.max_attempts,
                'error': msg.error,
                'age': time.time() - msg.timestamp
            }
            for msg in self.pending_messages.values()
        ]

    def get_subscriber_errors(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent subscriber errors for debugging."""
        errors = list(self.subscriber_errors)
        return errors[-limit:] if limit < len(errors) else errors

    # Context manager support
    async def __aenter__(self) -> "RelayManager":
        """Async context manager entry."""
        await self.start()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.stop()

# Global relay manager instance
_global_relay = None

def get_global_relay() -> RelayManager:
    """Get the global relay manager instance."""
    global _global_relay
    if _global_relay is None:
        _global_relay = RelayManager()
    return _global_relay

# Convenience functions
async def publish(channel: str, agent_response: Dict[str, Any], publisher: Optional[str] = None) -> None:
    """Publish using the global relay manager."""
    relay = get_global_relay()
    await relay.publish(channel, agent_response, publisher)

async def subscribe(channel: str, callback: Callable) -> None:
    """Subscribe using the global relay manager."""
    relay = get_global_relay()
    await relay.subscribe(channel, callback)

async def get_latest(channel: str, count: int = 1) -> List[Any]:
    """Get latest result data using the global relay manager."""
    relay = get_global_relay()
    return await relay.get_latest(channel, count)