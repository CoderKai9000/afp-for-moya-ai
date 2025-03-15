"""
Subscription model for Agent Flow Protocol.

Defines the subscription mechanism that allows agents to register interest
in specific message types or patterns, enabling selective message delivery.
"""

import re
import uuid
from typing import Dict, Any, Callable, Optional, List, Pattern, Union
from .message import AFPMessage
from .exceptions import AFPSubscriptionError


class AFPSubscription:
    """
    Represents a subscription to a specific message pattern.
    
    Allows agents to selectively receive messages based on filtering criteria
    such as sender, content type, or custom metadata properties.
    """
    
    def __init__(self, 
                 subscriber: str,
                 callback: Callable[[AFPMessage], None],
                 pattern: Optional[Dict[str, Any]] = None,
                 regex_patterns: Optional[Dict[str, Pattern]] = None,
                 subscription_id: Optional[str] = None):
        """
        Initialize a new subscription.
        
        Args:
            subscriber: Identifier of the subscribing agent
            callback: Function to call when a matching message is received
            pattern: Dictionary of message properties to match exactly
            regex_patterns: Dictionary of regular expressions to match against message properties
            subscription_id: Optional custom ID (auto-generated if not provided)
        """
        self.subscriber = subscriber
        self.callback = callback
        self.pattern = pattern or {}
        self.regex_patterns = regex_patterns or {}
        self.subscription_id = subscription_id or str(uuid.uuid4())
        
        # Validate the subscription
        self._validate()
    
    def _validate(self):
        """Validate the subscription configuration."""
        if not self.subscriber:
            raise AFPSubscriptionError("Subscriber identifier is required")
        
        if not callable(self.callback):
            raise AFPSubscriptionError("Callback must be callable")
        
        # Check that pattern keys are valid message attributes
        valid_fields = {
            'sender', 'recipients', 'content_type', 'content', 
            'metadata', 'message_id', 'parent_message_id', 'ttl'
        }
        
        invalid_fields = set(self.pattern.keys()) - valid_fields
        if invalid_fields:
            raise AFPSubscriptionError(f"Invalid pattern fields: {invalid_fields}")
            
        invalid_regex_fields = set(self.regex_patterns.keys()) - valid_fields
        if invalid_regex_fields:
            raise AFPSubscriptionError(f"Invalid regex pattern fields: {invalid_regex_fields}")
    
    def matches(self, message: AFPMessage) -> bool:
        """
        Check if a message matches this subscription's pattern.
        
        Args:
            message: The message to check against the subscription pattern
            
        Returns:
            True if the message matches this subscription, False otherwise
        """
        # Convert message to dictionary
        message_dict = message.to_dict()
        
        # Check exact pattern matches
        for key, value in self.pattern.items():
            # Handle special case for content_type (enum vs string)
            if key == 'content_type':
                if message.content_type.name != value:
                    return False
            # Special case for metadata (nested dict)
            elif key == 'metadata' and isinstance(value, dict):
                for meta_key, meta_value in value.items():
                    if message_dict.get('metadata', {}).get(meta_key) != meta_value:
                        return False
            # Regular field matching
            elif message_dict.get(key) != value:
                return False
        
        # Check regex pattern matches
        for key, pattern in self.regex_patterns.items():
            field_value = message_dict.get(key)
            if field_value is None:
                return False
                
            if isinstance(field_value, list):
                # For list fields (like recipients), check if any element matches
                if not any(pattern.search(str(element)) for element in field_value):
                    return False
            elif not pattern.search(str(field_value)):
                return False
        
        return True
    
    def deliver(self, message: AFPMessage) -> bool:
        """
        Deliver a message to this subscription's callback if it matches.
        
        Args:
            message: The message to potentially deliver
            
        Returns:
            True if message was delivered, False otherwise
        """
        if self.matches(message):
            try:
                self.callback(message)
                return True
            except Exception as e:
                # In a production system, this would be logged
                print(f"Error delivering message to {self.subscriber}: {e}")
                return False
        return False 