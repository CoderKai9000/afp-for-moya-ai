"""
Delivery guarantee module for Agent Flow Protocol.

Provides mechanisms for ensuring reliable message delivery,
including acknowledgments, retries, and dead letter handling.
"""

import time
import threading
import queue
import uuid
from typing import Dict, List, Set, Optional, Callable, Any, Tuple
from enum import Enum, auto

from ..exceptions import AFPReliabilityError
from ..message import AFPMessage, ContentType


class DeliveryStatus(Enum):
    """Enumeration of message delivery statuses."""
    PENDING = auto()      # Message is waiting to be sent
    SENT = auto()         # Message has been sent but not acknowledged
    DELIVERED = auto()    # Message has been delivered and acknowledged
    FAILED = auto()       # Message delivery has failed
    EXPIRED = auto()      # Message has expired before delivery


class DeliveryGuarantee(Enum):
    """Enumeration of delivery guarantee levels."""
    BEST_EFFORT = auto()  # No guarantees, just try once
    AT_LEAST_ONCE = auto() # Message will be delivered at least once (may be duplicated)
    EXACTLY_ONCE = auto()  # Message will be delivered exactly once


class MessageTracker:
    """
    Tracks message delivery status and handles retries.
    
    Provides mechanisms for ensuring reliable message delivery through
    acknowledgments and automatic retries.
    """
    
    def __init__(self, 
                 retry_interval: float = 5.0,
                 max_retries: int = 3,
                 expiry_time: float = 60.0):
        """
        Initialize the message tracker.
        
        Args:
            retry_interval: Time between retry attempts (seconds)
            max_retries: Maximum number of retry attempts
            expiry_time: Time after which messages expire (seconds)
        """
        # Message tracking
        self._messages: Dict[str, Dict[str, Any]] = {}
        
        # Acknowledgments received
        self._acks: Set[str] = set()
        
        # Configuration
        self.retry_interval = retry_interval
        self.max_retries = max_retries
        self.expiry_time = expiry_time
        
        # Callbacks
        self._on_delivery: Optional[Callable[[str, DeliveryStatus], None]] = None
        self._on_retry: Optional[Callable[[str, int], None]] = None
        self._on_expire: Optional[Callable[[str], None]] = None
        
        # Retry thread
        self._stop_event = threading.Event()
        self._retry_thread = threading.Thread(target=self._retry_loop, daemon=True)
        self._retry_thread.start()
        
        # Lock for thread safety
        self._lock = threading.RLock()
    
    def track_message(self, 
                      message: AFPMessage, 
                      guarantee: DeliveryGuarantee = DeliveryGuarantee.AT_LEAST_ONCE,
                      on_send: Optional[Callable[[AFPMessage], bool]] = None) -> str:
        """
        Start tracking a message for reliable delivery.
        
        Args:
            message: The message to track
            guarantee: The delivery guarantee level
            on_send: Callback function to send the message
            
        Returns:
            The message ID
            
        Raises:
            AFPReliabilityError: If the message is invalid
        """
        if not message.message_id:
            raise AFPReliabilityError("Message must have an ID for tracking")
        
        with self._lock:
            # Check if message is already being tracked
            if message.message_id in self._messages:
                raise AFPReliabilityError(f"Message {message.message_id} is already being tracked")
            
            # Create tracking entry
            self._messages[message.message_id] = {
                'message': message,
                'status': DeliveryStatus.PENDING,
                'guarantee': guarantee,
                'on_send': on_send,
                'retries': 0,
                'last_attempt': 0,
                'created': time.time()
            }
            
            # If we have a send callback, attempt to send immediately
            if on_send:
                self._attempt_send(message.message_id)
            
            return message.message_id
    
    def acknowledge(self, message_id: str) -> bool:
        """
        Acknowledge message delivery.
        
        Args:
            message_id: The ID of the message to acknowledge
            
        Returns:
            True if the message was acknowledged, False if not found
        """
        with self._lock:
            # Check if message is being tracked
            if message_id not in self._messages:
                return False
            
            # Update status
            self._messages[message_id]['status'] = DeliveryStatus.DELIVERED
            self._acks.add(message_id)
            
            # Notify delivery
            if self._on_delivery:
                self._on_delivery(message_id, DeliveryStatus.DELIVERED)
            
            return True
    
    def fail_message(self, message_id: str, permanent: bool = False) -> bool:
        """
        Mark a message as failed.
        
        Args:
            message_id: The ID of the message to fail
            permanent: If True, the message will not be retried
            
        Returns:
            True if the message was marked as failed, False if not found
        """
        with self._lock:
            # Check if message is being tracked
            if message_id not in self._messages:
                return False
            
            # Update status
            if permanent:
                self._messages[message_id]['status'] = DeliveryStatus.FAILED
                
                # Notify delivery
                if self._on_delivery:
                    self._on_delivery(message_id, DeliveryStatus.FAILED)
            else:
                # If not permanent, it will be retried
                self._messages[message_id]['last_attempt'] = time.time()
                self._messages[message_id]['retries'] += 1
                
                # If max retries reached, mark as failed
                if self._messages[message_id]['retries'] > self.max_retries:
                    self._messages[message_id]['status'] = DeliveryStatus.FAILED
                    
                    # Notify delivery
                    if self._on_delivery:
                        self._on_delivery(message_id, DeliveryStatus.FAILED)
                else:
                    # Notify retry
                    if self._on_retry:
                        self._on_retry(message_id, self._messages[message_id]['retries'])
            
            return True
    
    def get_status(self, message_id: str) -> Optional[DeliveryStatus]:
        """
        Get the delivery status of a message.
        
        Args:
            message_id: The ID of the message
            
        Returns:
            The delivery status, or None if the message is not being tracked
        """
        with self._lock:
            if message_id not in self._messages:
                return None
            
            return self._messages[message_id]['status']
    
    def get_message(self, message_id: str) -> Optional[AFPMessage]:
        """
        Get a tracked message.
        
        Args:
            message_id: The ID of the message
            
        Returns:
            The message, or None if not found
        """
        with self._lock:
            if message_id not in self._messages:
                return None
            
            return self._messages[message_id]['message']
    
    def set_delivery_callback(self, callback: Callable[[str, DeliveryStatus], None]):
        """
        Set a callback for delivery status changes.
        
        Args:
            callback: Function to call when delivery status changes
        """
        self._on_delivery = callback
    
    def set_retry_callback(self, callback: Callable[[str, int], None]):
        """
        Set a callback for retry attempts.
        
        Args:
            callback: Function to call when a retry is attempted
        """
        self._on_retry = callback
    
    def set_expire_callback(self, callback: Callable[[str], None]):
        """
        Set a callback for message expiration.
        
        Args:
            callback: Function to call when a message expires
        """
        self._on_expire = callback
    
    def _attempt_send(self, message_id: str) -> bool:
        """
        Attempt to send a message.
        
        Args:
            message_id: The ID of the message to send
            
        Returns:
            True if the send attempt was successful, False otherwise
        """
        # Get message info
        message_info = self._messages[message_id]
        message = message_info['message']
        on_send = message_info['on_send']
        
        # Update status and attempt time
        message_info['status'] = DeliveryStatus.SENT
        message_info['last_attempt'] = time.time()
        
        # Attempt to send
        success = False
        if on_send:
            try:
                success = on_send(message)
            except Exception:
                success = False
        
        # If send failed, increment retry count
        if not success:
            message_info['retries'] += 1
            
            # If max retries reached, mark as failed
            if message_info['retries'] > self.max_retries:
                message_info['status'] = DeliveryStatus.FAILED
                
                # Notify delivery
                if self._on_delivery:
                    self._on_delivery(message_id, DeliveryStatus.FAILED)
        
        return success
    
    def _retry_loop(self):
        """Background thread for retrying failed messages."""
        while not self._stop_event.is_set():
            try:
                # Sleep for a short interval
                time.sleep(0.1)
                
                # Check for messages to retry
                with self._lock:
                    now = time.time()
                    
                    # Find messages that need to be retried
                    for message_id, info in list(self._messages.items()):
                        # Skip messages that are already delivered or permanently failed
                        if info['status'] in (DeliveryStatus.DELIVERED, DeliveryStatus.FAILED):
                            continue
                        
                        # Check if message has expired
                        if now - info['created'] > self.expiry_time:
                            info['status'] = DeliveryStatus.EXPIRED
                            
                            # Notify expiration
                            if self._on_expire:
                                self._on_expire(message_id)
                            
                            # Notify delivery
                            if self._on_delivery:
                                self._on_delivery(message_id, DeliveryStatus.EXPIRED)
                            
                            continue
                        
                        # Check if it's time to retry
                        if (info['status'] == DeliveryStatus.SENT and 
                            now - info['last_attempt'] > self.retry_interval and
                            info['retries'] < self.max_retries):
                            
                            # Attempt to send
                            self._attempt_send(message_id)
                            
                            # Notify retry
                            if self._on_retry:
                                self._on_retry(message_id, info['retries'])
            
            except Exception as e:
                # In a production system, this would be logged
                print(f"Error in retry loop: {e}")
    
    def shutdown(self):
        """Shutdown the message tracker and release resources."""
        self._stop_event.set()
        self._retry_thread.join(timeout=1.0)
        
        with self._lock:
            self._messages.clear()
            self._acks.clear()


class DeadLetterQueue:
    """
    Queue for messages that could not be delivered.
    
    Stores messages that failed delivery for later processing or analysis.
    """
    
    def __init__(self, max_size: int = 1000):
        """
        Initialize the dead letter queue.
        
        Args:
            max_size: Maximum number of messages to store
        """
        self._queue: List[Tuple[AFPMessage, str, float]] = []
        self.max_size = max_size
        self._lock = threading.RLock()
    
    def add_message(self, message: AFPMessage, reason: str):
        """
        Add a message to the dead letter queue.
        
        Args:
            message: The message that failed delivery
            reason: The reason for the failure
        """
        with self._lock:
            # Add message with timestamp
            self._queue.append((message, reason, time.time()))
            
            # Trim queue if it exceeds max size
            if len(self._queue) > self.max_size:
                self._queue = self._queue[-self.max_size:]
    
    def get_messages(self, count: Optional[int] = None) -> List[Tuple[AFPMessage, str, float]]:
        """
        Get messages from the dead letter queue.
        
        Args:
            count: Maximum number of messages to return (None for all)
            
        Returns:
            List of (message, reason, timestamp) tuples
        """
        with self._lock:
            if count is None:
                return list(self._queue)
            else:
                return list(self._queue[:count])
    
    def clear(self):
        """Clear all messages from the queue."""
        with self._lock:
            self._queue.clear()
    
    def size(self) -> int:
        """Get the number of messages in the queue."""
        with self._lock:
            return len(self._queue) 