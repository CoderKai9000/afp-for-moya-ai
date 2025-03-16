"""
Communication bus for Agent Flow Protocol.

Implements the core message routing and delivery system for AFP,
managing subscriptions, message queues, and agent registrations.
"""

import time
import threading
import queue
import uuid
import functools
import heapq
from typing import Dict, List, Any, Optional, Set, Callable, Tuple, Union
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict, OrderedDict

from .message import AFPMessage, ContentType
from .subscription import AFPSubscription
from .exceptions import (
    AFPError, AFPMessageError, AFPRoutingError, 
    AFPSubscriptionError, AFPTimeoutError
)


# Simple LRU cache implementation if not available in functools
class LRUCache:
    """Simple LRU (Least Recently Used) cache implementation."""
    
    def __init__(self, capacity: int):
        """Initialize with fixed capacity."""
        self.cache = OrderedDict()
        self.capacity = capacity
    
    def get(self, key: Any) -> Any:
        """Get item from cache, moving it to the end (most recently used)."""
        if key not in self.cache:
            return None
        # Move to end (most recently used)
        value = self.cache.pop(key)
        self.cache[key] = value
        return value
    
    def put(self, key: Any, value: Any) -> None:
        """Add item to cache, evicting least recently used if at capacity."""
        if key in self.cache:
            # Remove and re-add to move to end
            self.cache.pop(key)
        elif len(self.cache) >= self.capacity:
            # Remove least recently used
            self.cache.popitem(last=False)
        # Add to end (most recently used)
        self.cache[key] = value


class AFPCommunicationBus:
    """
    Core communication bus for Agent Flow Protocol.
    
    Manages message routing, delivery, subscriptions, and agent registrations.
    Provides both synchronous and asynchronous messaging patterns.
    """
    
    def __init__(self, max_workers: int = 10, cache_size: int = 100):
        """
        Initialize the communication bus.
        
        Args:
            max_workers: Maximum number of worker threads for message delivery
            cache_size: Size of the recipient cache for frequent message targets
        """
        # Agent registrations
        self._agents: Set[str] = set()
        
        # Subscriptions
        self._subscriptions: Dict[str, List[AFPSubscription]] = {}
        
        # Response queues for synchronous requests
        self._response_queues: Dict[str, queue.Queue] = {}
        
        # Thread pool for asynchronous message delivery
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        
        # Fine-grained locks for reduced contention
        self._agent_lock = threading.RLock()  # Lock for agent operations
        self._subscription_lock = threading.RLock()  # Lock for subscription operations
        self._queue_lock = threading.RLock()  # Lock for queue operations
        
        # LRU cache for frequent recipients
        self._recipient_cache = LRUCache(cache_size)
        
        # Fast paths for workflows
        self._direct_routes = {}  # Cache for direct agent-to-agent routes
        self._direct_routes_lock = threading.RLock()
        
        # Priority queue for message processing
        self._priority_queue = []
        self._priority_lock = threading.RLock()
        
        # Batch processing variables
        self._batch_size = 10  # Process up to 10 messages at once
        self._batch_timeout = 0.01  # Wait up to 10ms to accumulate messages
        self._batch_enabled = True
        
        # Create a subscription for handling responses
        self._setup_response_handler()
        
        # Start the batch message processor thread if enabled
        if self._batch_enabled:
            self._batch_processor_running = True
            self._batch_processor_thread = threading.Thread(
                target=self._batch_message_processor,
                daemon=True
            )
            self._batch_processor_thread.start()
    
    def _setup_response_handler(self):
        """Set up internal subscription for handling response messages."""
        def response_handler(message: AFPMessage):
            if message.parent_message_id in self._response_queues:
                self._response_queues[message.parent_message_id].put(message)
        
        # Create a subscription that matches any message with a parent_message_id
        # that corresponds to a waiting response queue
        self.subscribe(
            subscriber="__internal_response_handler__",
            callback=response_handler,
            pattern={},  # No specific pattern - we'll check in the callback
            subscription_id="__response_handler__"
        )
    
    def register_agent(self, agent_id: str) -> bool:
        """
        Register an agent with the communication bus.
        
        Args:
            agent_id: Unique identifier for the agent
            
        Returns:
            True if registration was successful, False if already registered
        """
        with self._agent_lock:
            if agent_id in self._agents:
                return False
            
            self._agents.add(agent_id)
            self._subscriptions[agent_id] = []
            
            return True
    
    def unregister_agent(self, agent_id: str) -> bool:
        """
        Unregister an agent from the communication bus.
        
        Args:
            agent_id: Unique identifier for the agent
            
        Returns:
            True if unregistration was successful, False if not registered
        """
        with self._agent_lock:
            if agent_id not in self._agents:
                return False
            
            self._agents.remove(agent_id)
            
            # Remove agent's subscriptions
            if agent_id in self._subscriptions:
                del self._subscriptions[agent_id]
            
            # Clear direct routes involving this agent
            with self._direct_routes_lock:
                keys_to_remove = []
                for key in self._direct_routes:
                    if key[0] == agent_id or key[1] == agent_id:
                        keys_to_remove.append(key)
                
                for key in keys_to_remove:
                    self._direct_routes.pop(key, None)
            
            return True
    
    def create_direct_route(self, sender: str, recipient: str) -> bool:
        """
        Create a direct route between sender and recipient for fast message delivery.
        This is especially useful for workflow patterns where the same agents 
        communicate frequently in a fixed pattern.
        
        Args:
            sender: Agent sending messages
            recipient: Agent receiving messages
            
        Returns:
            True if route was created, False otherwise
        """
        with self._agent_lock:
            if sender not in self._agents or recipient not in self._agents:
                return False
            
            with self._direct_routes_lock:
                self._direct_routes[(sender, recipient)] = True
                return True
    
    def subscribe(self, 
                  subscriber: str, 
                  callback: Callable[[AFPMessage], None],
                  pattern: Optional[Dict[str, Any]] = None,
                  regex_patterns: Optional[Dict[str, Any]] = None,
                  subscription_id: Optional[str] = None) -> str:
        """
        Subscribe to messages matching a pattern.
        
        Args:
            subscriber: Identifier of the subscribing agent
            callback: Function to call when a matching message is received
            pattern: Dictionary of message properties to match exactly
            regex_patterns: Dictionary of regular expressions to match against message properties
            subscription_id: Optional custom ID (auto-generated if not provided)
            
        Returns:
            Subscription ID that can be used to unsubscribe
            
        Raises:
            AFPSubscriptionError: If the subscription is invalid
        """
        with self._agent_lock:
            # Verify agent is registered (except for internal subscriptions)
            if not subscriber.startswith("__") and subscriber not in self._agents:
                raise AFPSubscriptionError(f"Agent {subscriber} is not registered")
            
            # Create subscription
            subscription = AFPSubscription(
                subscriber=subscriber,
                callback=callback,
                pattern=pattern,
                regex_patterns=regex_patterns,
                subscription_id=subscription_id
            )
            
            # Store subscription
            if subscription.subscription_id not in self._subscriptions:
                self._subscriptions[subscription.subscription_id] = []
            
            self._subscriptions[subscription.subscription_id].append(subscription)
            
            return subscription.subscription_id
    
    def unsubscribe(self, subscription_id: str, subscriber: Optional[str] = None) -> bool:
        """
        Unsubscribe from messages.
        
        Args:
            subscription_id: ID of the subscription to remove
            subscriber: Optional subscriber ID to only remove subscriptions for this subscriber
            
        Returns:
            True if unsubscription was successful, False otherwise
        """
        with self._subscription_lock:
            if subscription_id not in self._subscriptions:
                return False
            
            if subscriber:
                # Remove only subscriptions for this subscriber
                original_count = len(self._subscriptions[subscription_id])
                self._subscriptions[subscription_id] = [
                    s for s in self._subscriptions[subscription_id] 
                    if s.subscriber != subscriber
                ]
                
                # If no subscriptions left, remove the key
                if not self._subscriptions[subscription_id]:
                    del self._subscriptions[subscription_id]
                
                return original_count > len(self._subscriptions.get(subscription_id, []))
            else:
                # Remove all subscriptions with this ID
                del self._subscriptions[subscription_id]
                return True
    
    def send_message(self, message: AFPMessage) -> bool:
        """
        Send a message asynchronously.
        
        Args:
            message: The message to send
            
        Returns:
            True if the message was accepted for delivery, False otherwise
            
        Raises:
            AFPMessageError: If the message is invalid
            AFPRoutingError: If the message cannot be routed
        """
        # Validate message
        if not message.sender:
            raise AFPMessageError("Message sender cannot be empty")
        
        if not message.recipients:
            raise AFPMessageError("Message must have at least one recipient")
        
        # Check if sender is registered
        with self._agent_lock:
            if message.sender not in self._agents and not message.sender.startswith("__"):
                raise AFPRoutingError(f"Sender {message.sender} is not registered")
            
            # For non-broadcast messages, check if recipients are registered
            if not message.is_broadcast():
                unknown_recipients = [r for r in message.recipients if r not in self._agents]
                if unknown_recipients:
                    raise AFPRoutingError(f"Unknown recipients: {unknown_recipients}")
        
        # Submit delivery task to thread pool
        self._executor.submit(self._deliver_message, message)
        return True
    
    def send_message_sync(self, message: AFPMessage, timeout: float = 5.0) -> Optional[AFPMessage]:
        """
        Send a message and wait for a response.
        
        This method has been optimized for complex workflows by implementing:
        1. More granular locking to reduce contention
        2. Priority-based message processing
        3. Direct routing for workflow messages
        4. Intelligent timeout management
        
        Args:
            message: The message to send
            timeout: Maximum time to wait for a response (seconds)
            
        Returns:
            Response message or None if no response received within timeout
            
        Raises:
            AFPMessageError: If message validation fails
            AFPRoutingError: If no route available for the message
            AFPTimeoutError: If no response received within timeout
        """
        # Validate the message
        if not message.sender:
            raise AFPMessageError("Message must have a sender")
        
        if not message.recipients:
            raise AFPMessageError("Message must have at least one recipient")
        
        # Check if all recipients are registered
        with self._agent_lock:
            for recipient in message.recipients:
                if recipient != "*" and recipient not in self._agents:
                    raise AFPRoutingError(f"Recipient '{recipient}' is not registered")
        
        # Fast path for workflow messages with direct routes
        if len(message.recipients) == 1 and message.metadata.get("workflow_id"):
            recipient = message.recipients[0]
            route_key = (message.sender, recipient)
            
            with self._direct_routes_lock:
                if route_key not in self._direct_routes:
                    # Create direct route for future messages
                    self._direct_routes[route_key] = True
        
        # Create a response queue for this message
        response_queue = queue.Queue()
        
        # Register the response queue with a shorter timeout scope
        with self._queue_lock:
            self._response_queues[message.message_id] = response_queue
        
        try:
            # Add message to the priority queue with high priority for sync messages
            priority = message.priority or 0
            if priority < 9:  # Ensure sync messages get higher priority
                priority = 9
            
            # Give workflow messages even higher priority
            if message.metadata.get("workflow_id"):
                priority = 10
            
            with self._priority_lock:
                # Add negative priority to get highest priority first (heapq is min-heap)
                heapq.heappush(self._priority_queue, (-priority, message))
                
            # Wait for response with timeout
            try:
                # Use a slightly reduced timeout to account for processing overhead
                response = response_queue.get(timeout=timeout * 0.95)
                return response
            except queue.Empty:
                raise AFPTimeoutError(f"No response received within {timeout} seconds")
        finally:
            # Clean up the response queue
            with self._queue_lock:
                self._response_queues.pop(message.message_id, None)
    
    def _deliver_message(self, message: AFPMessage) -> int:
        """
        Deliver a message to all matching subscribers.
        
        Args:
            message: The message to deliver
            
        Returns:
            Number of subscribers the message was delivered to
        """
        # Check if message has expired
        if message.has_expired():
            return 0
        
        delivered_count = 0
        
        # Collect all subscriptions
        all_subscriptions = []
        with self._subscription_lock:
            for subscriptions in self._subscriptions.values():
                all_subscriptions.extend(subscriptions)
        
        # Deliver to matching subscriptions
        for subscription in all_subscriptions:
            # Skip delivery to sender (unless it's a self-addressed message)
            if subscription.subscriber == message.sender and subscription.subscriber not in message.recipients:
                continue
                
            # For broadcast messages, deliver to everyone except sender
            if message.is_broadcast() and subscription.subscriber != message.sender:
                if subscription.deliver(message):
                    delivered_count += 1
            # For direct messages, only deliver to intended recipients
            elif subscription.subscriber in message.recipients:
                if subscription.deliver(message):
                    delivered_count += 1
            # For pattern-based subscriptions, check if message matches
            elif subscription.matches(message):
                if subscription.deliver(message):
                    delivered_count += 1
        
        return delivered_count
    
    def get_agent_count(self) -> int:
        """Get the number of registered agents."""
        with self._agent_lock:
            return len(self._agents)
    
    def get_subscription_count(self) -> int:
        """Get the total number of active subscriptions."""
        with self._subscription_lock:
            return sum(len(subs) for subs in self._subscriptions.values())
    
    def _batch_message_processor(self):
        """Process messages in batches to reduce overhead."""
        while self._batch_processor_running:
            batch = []
            
            # Collect a batch of messages from the priority queue
            with self._priority_lock:
                start_time = time.time()
                while len(batch) < self._batch_size and time.time() - start_time < self._batch_timeout:
                    if not self._priority_queue:
                        # Wait for more messages
                        time.sleep(0.001)
                        continue
                    
                    # Get the highest priority message
                    _, message = heapq.heappop(self._priority_queue)
                    batch.append(message)
            
            # Process the batch
            if batch:
                for message in batch:
                    self._process_message(message)
            else:
                # No messages, sleep a bit to reduce CPU usage
                time.sleep(0.001)
    
    def _process_message(self, message: AFPMessage):
        """
        Process a single message, delivering it to all matching subscribers.
        
        Args:
            message: The message to process
        """
        # First check if this is a workflow message with a direct route (fast path)
        if len(message.recipients) == 1 and message.metadata.get("workflow_id"):
            recipient = message.recipients[0]
            route_key = (message.sender, recipient)
            
            with self._direct_routes_lock:
                if route_key in self._direct_routes:
                    # Fast path: deliver directly to the recipient's subscriptions
                    recipient_subscriptions = self._subscriptions.get(recipient, [])
                    for subscription in recipient_subscriptions:
                        if subscription.matches(message):
                            try:
                                subscription.callback(message)
                            except Exception as e:
                                print(f"Error in direct route callback: {e}")
                    return
        
        # Regular path for all other messages
        matching_subscriptions = self._find_matching_subscriptions(message)
        
        for subscription in matching_subscriptions:
            try:
                subscription.callback(message)
            except Exception as e:
                print(f"Error in subscription callback: {e}")
    
    def _get_recipient_subscriptions(self, recipient: str) -> List[AFPSubscription]:
        """
        Get all subscriptions for a given recipient.
        This function uses a cache for performance when using frequent recipients.
        
        Args:
            recipient: The recipient agent ID
            
        Returns:
            List of subscriptions for the recipient
        """
        # Check cache first
        cached = self._recipient_cache.get(recipient)
        if cached is not None:
            return cached
        
        # Not in cache, get from main storage
        with self._subscription_lock:
            result = self._subscriptions.get(recipient, [])
            
        # Update cache
        self._recipient_cache.put(recipient, result)
        return result
    
    def _find_matching_subscriptions(self, message: AFPMessage) -> List[AFPSubscription]:
        """
        Find all subscriptions that match a given message.
        
        Args:
            message: The message to match
            
        Returns:
            List of matching subscriptions
        """
        matching_subscriptions = []
        
        # Get and check direct recipient subscriptions first (faster path)
        for recipient in message.recipients:
            if recipient == "*":  # Broadcast message
                # For broadcast, we need to check all subscriptions
                with self._subscription_lock:
                    for subscriber, subscriptions in self._subscriptions.items():
                        for subscription in subscriptions:
                            if subscription.matches(message):
                                matching_subscriptions.append(subscription)
            else:
                # Get subscriptions for this specific recipient (cached for frequent recipients)
                recipient_subscriptions = self._get_recipient_subscriptions(recipient)
                for subscription in recipient_subscriptions:
                    if subscription.matches(message):
                        matching_subscriptions.append(subscription)
        
        return matching_subscriptions
    
    def shutdown(self):
        """Shut down the communication bus, stopping all worker threads."""
        self._executor.shutdown(wait=True)
        
        # Stop the batch processor if enabled
        if self._batch_enabled:
            self._batch_processor_running = False
            if self._batch_processor_thread.is_alive():
                self._batch_processor_thread.join(timeout=1.0)
        
        with self._agent_lock:
            self._agents.clear()
            self._subscriptions.clear()
            self._response_queues.clear()
        
        with self._direct_routes_lock:
            self._direct_routes.clear()
        
        # Clear the recipient cache
        self._recipient_cache = LRUCache(100)
    
    def add_direct_route(self, sender, recipient):
        """
        Add a direct route between sender and recipient for optimized message delivery.
        
        Args:
            sender: ID of the sender agent
            recipient: ID of the recipient agent
            
        Returns:
            True if route was added, False otherwise
        """
        with self._agent_lock:
            if sender not in self._agents or recipient not in self._agents:
                return False
            
            with self._direct_routes_lock:
                route_key = (sender, recipient)
                # Store a direct subscription function that always matches for this sender-recipient pair
                self._direct_routes[route_key] = lambda msg: True
                print(f"Added direct route: {sender} -> {recipient}")
                return True
    
    def remove_direct_route(self, sender, recipient):
        """
        Remove a direct route between sender and recipient.
        
        Args:
            sender: ID of the sender agent
            recipient: ID of the recipient agent
            
        Returns:
            True if route was removed, False if route didn't exist
        """
        with self._agent_lock:
            if sender not in self._agents or recipient not in self._agents:
                return False
            
            with self._direct_routes_lock:
                route_key = (sender, recipient)
                if route_key in self._direct_routes:
                    del self._direct_routes[route_key]
                    return True
                return False
    
    def has_direct_route(self, sender, recipient):
        """
        Check if a direct route exists between sender and recipient.
        
        Args:
            sender: ID of the sender agent
            recipient: ID of the recipient agent
            
        Returns:
            True if a direct route exists, False otherwise
        """
        with self._agent_lock:
            if sender not in self._agents or recipient not in self._agents:
                return False
            
            with self._direct_routes_lock:
                return (sender, recipient) in self._direct_routes 