"""
Message model for Agent Flow Protocol.

Defines the structure and validation for messages exchanged between agents,
with support for various content types, metadata, and tracing information.
"""

import uuid
import time
import json
from typing import Dict, List, Any, Optional, Union
from enum import Enum, auto
from dataclasses import dataclass, field, asdict


class ContentType(Enum):
    """Enumeration of supported content types for AFP messages."""
    TEXT = auto()
    JSON = auto()
    BINARY = auto()
    IMAGE = auto()
    AUDIO = auto()
    VIDEO = auto()
    STREAM = auto()


@dataclass
class AFPMessage:
    """
    Message format for Agent Flow Protocol communications.
    
    Provides a standardized structure for all messages exchanged between
    agents, with support for various content types, routing information,
    and tracing capabilities.
    """
    # Required fields
    sender: str
    recipients: List[str]
    content_type: ContentType
    content: Any
    
    # Optional fields with defaults
    metadata: Dict[str, Any] = field(default_factory=dict)
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    parent_message_id: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    ttl: Optional[int] = None  # Time-to-live in seconds
    trace_path: List[str] = field(default_factory=list)
    priority: int = 0  # Message priority (0-10, higher values indicate higher priority)
    
    def __post_init__(self):
        """Validate message after initialization."""
        # Add sender to trace path if not already present
        if not self.trace_path or self.trace_path[-1] != self.sender:
            self.trace_path.append(self.sender)
        
        # Ensure priority is within valid range
        if self.priority < 0:
            self.priority = 0
        elif self.priority > 10:
            self.priority = 10
    
    def is_broadcast(self) -> bool:
        """Check if this message is a broadcast (sent to all agents)."""
        return len(self.recipients) == 1 and self.recipients[0] == "*"
    
    def has_expired(self) -> bool:
        """Check if this message has expired based on TTL."""
        if self.ttl is None:
            return False
        return (time.time() - self.timestamp) > self.ttl
    
    def create_response(self, 
                        content: Any, 
                        content_type: Optional[ContentType] = None,
                        metadata: Optional[Dict[str, Any]] = None) -> 'AFPMessage':
        """
        Create a response message to this message.
        
        Args:
            content: The content of the response
            content_type: Content type of the response (defaults to same as original)
            metadata: Optional metadata for the response
            
        Returns:
            A new AFPMessage instance configured as a response
        """
        return AFPMessage(
            sender=self.recipients[0] if len(self.recipients) == 1 else self.recipients[0],
            recipients=[self.sender],
            content_type=content_type or self.content_type,
            content=content,
            metadata=metadata or {},
            parent_message_id=self.message_id,
            trace_path=self.trace_path.copy()
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary representation."""
        message_dict = asdict(self)
        # Convert enum to string for serialization
        message_dict['content_type'] = self.content_type.name
        return message_dict
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AFPMessage':
        """Create message from dictionary representation."""
        # Convert content_type from string to enum
        if 'content_type' in data and isinstance(data['content_type'], str):
            data = data.copy()  # Don't modify the original
            data['content_type'] = ContentType[data['content_type']]
        return cls(**data)
    
    def serialize(self) -> str:
        """Serialize message to JSON string."""
        return json.dumps(self.to_dict())
    
    @classmethod
    def deserialize(cls, data: str) -> 'AFPMessage':
        """Deserialize message from JSON string."""
        return cls.from_dict(json.loads(data)) 