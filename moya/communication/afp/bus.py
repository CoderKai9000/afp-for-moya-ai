"""
Communication bus for Agent Flow Protocol.

Implements the core message routing and delivery system for AFP,
managing subscriptions, message queues, and agent registrations.
"""

import time
import threading
import queue
import uuid
from typing import Dict, List, Any, Optional, Set, Callable, Tuple, Union
from concurrent.futures import ThreadPoolExecutor

from .message import AFPMessage, ContentType
from .subscription import AFPSubscription
from .exceptions import (
    AFPError, AFPMessageError, AFPRoutingError, 
    AFPSubscriptionError, AFPTimeoutError
)


class AFPCommunicationBus:
    """
    Core communication bus for Agent Flow Protocol.
    
    Manages message routing, delivery, subscriptions, and agent registrations.
    Provides both synchronous and asynchronous messaging patterns.
    """
    
    def __init__(self, max_workers: int = 10):
        """
        Initialize the communication bus.
        
        Args:
            max_workers: Maximum number of worker threads for message delivery
        """
        # Agent registrations
        self._agents: Set[str] = set()
        
        # Subscriptions
        self._subscriptions: Dict[str, List[AFPSubscription]] = {}
        
        # Response queues for synchronous requests
        self._response_queues: Dict[str, queue.Queue] = {}
        
        # Thread pool for asynchronous message delivery
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        
        # Lock for thread safety
        self._lock = threading.RLock()
        
        # Create a subscription for handling responses
        self._setup_response_handler()
    
    def _setup_response_handler(self):
        """Set up internal subscription for handling responses."""
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
        with self._lock:
            if agent_id in self._agents:
                return False
            
            self._agents.add(agent_id)
            return True
    
    def unregister_agent(self, agent_id: str) -> bool:
        """
        Unregister an agent from the communication bus.
        
        Args:
            agent_id: Unique identifier for the agent
            
        Returns:
            True if unregistration was successful, False if not registered
        """
        with self._lock:
            if agent_id not in self._agents:
                return False
            
            # Remove agent from registered agents
            self._agents.remove(agent_id)
            
            # Remove all subscriptions for this agent
            to_remove = []
            for sub_id, subscriptions in self._subscriptions.items():
                self._subscriptions[sub_id] = [
                    s for s in subscriptions if s.subscriber != agent_id
                ]
                if not self._subscriptions[sub_id]:
                    to_remove.append(sub_id)
            
            # Clean up empty subscription lists
            for sub_id in to_remove:
                del self._subscriptions[sub_id]
            
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
        with self._lock:
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
        with self._lock:
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
        with self._lock:
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
    
    def send_message_sync(self, 
                          message: AFPMessage, 
                          timeout: Optional[float] = None) -> AFPMessage:
        """
        Send a message synchronously and wait for a response.
        
        Args:
            message: The message to send
            timeout: Maximum time to wait for a response (in seconds)
            
        Returns:
            The response message
            
        Raises:
            AFPMessageError: If the message is invalid
            AFPRoutingError: If the message cannot be routed
            AFPTimeoutError: If no response is received within the timeout
        """
        # Create a response queue
        response_queue = queue.Queue()
        
        # Register the response queue
        with self._lock:
            self._response_queues[message.message_id] = response_queue
        
        try:
            # Send the message
            self.send_message(message)
            
            # Wait for response
            try:
                response = response_queue.get(timeout=timeout)
                return response
            except queue.Empty:
                raise AFPTimeoutError(f"No response received within {timeout} seconds")
        finally:
            # Clean up the response queue
            with self._lock:
                if message.message_id in self._response_queues:
                    del self._response_queues[message.message_id]
    
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
        with self._lock:
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
        with self._lock:
            return len(self._agents)
    
    def get_subscription_count(self) -> int:
        """Get the total number of active subscriptions."""
        with self._lock:
            return sum(len(subs) for subs in self._subscriptions.values())
    
    def shutdown(self):
        """Shutdown the communication bus and release resources."""
        self._executor.shutdown(wait=True)
        with self._lock:
            self._agents.clear()
            self._subscriptions.clear()
            self._response_queues.clear() 