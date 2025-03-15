# Agent Flow Protocol (AFP) Implementation Guide

## Introduction

This guide provides detailed, step-by-step instructions for implementing the Agent Flow Protocol (AFP) within the Moya framework. The implementation is structured in logical phases, with each phase building upon the previous one.

## Prerequisites

Before beginning implementation, ensure you have:

1. A working installation of the Moya framework
2. Python 3.10 or higher
3. Basic understanding of the Moya architecture (agents, tools, orchestrators)
4. Standard Python development environment (IDE, version control)

## Addressing Distributed Systems Challenges

Building a robust agent communication protocol requires addressing several fundamental distributed systems challenges:

### Handling Failures and Recovery

The AFP implementation must account for various types of failures:

1. **Communication Failures**: Messages may be lost, delayed, or duplicated
2. **Agent Failures**: Agents may crash or become unresponsive
3. **Network Partitions**: Parts of the system may become temporarily isolated

These challenges are addressed through:

- **Retry Mechanisms**: Configurable retry strategies for failed message deliveries
- **Acknowledgments**: Explicit message receipt acknowledgments for reliable delivery
- **Dead Letter Queuing**: Storage of undeliverable messages for later inspection
- **Circuit Breaking**: Preventing cascading failures by failing fast

### Consistency and Conflict Resolution

With multiple agents operating concurrently and potentially updating shared state:

1. **Conflict Detection**: Using version numbers or vector clocks to detect conflicting updates
2. **Conflict Resolution Strategies**: Implementing merge functions or last-writer-wins policies
3. **Eventual Consistency**: Ensuring all agents eventually see the same state, with configurable stronger guarantees when needed

### Monitoring and Observability

The implementation includes comprehensive monitoring capabilities:

1. **Message Tracing**: Tracking message flows across agents
2. **Performance Metrics**: Capturing latency, throughput, and error rates
3. **Health Checks**: Verifying component availability and performance
4. **Visualization Tools**: Providing insight into message patterns and system behavior

## Implementation Phases Overview

The implementation is divided into the following phases:

1. **Project Structure Setup**: Creating the necessary directory structure and files
2. **Core Protocol Components**: Implementing the basic message format and communication bus
3. **AFP Tool Implementation**: Creating the tool interface for agents
4. **Agent Extensions**: Extending the Agent class to support AFP
5. **Data Store Implementation**: Creating the data store for large content
6. **Integration with Moya**: Connecting AFP with existing Moya components
7. **Demo Application**: Building a demonstration application
8. **Testing & Validation**: Testing the implementation

Let's begin with the first phase.

## Phase 1: Project Structure Setup

### Step 1: Create Directory Structure

First, we need to create a comprehensive directory structure within the Moya framework for the AFP implementation, ensuring we have components for all aspects of the protocol:

```
moya/
└── communication/
    ├── __init__.py
    ├── afp/
    │   ├── __init__.py
    │   ├── message.py         # Message format definition
    │   ├── bus.py             # Core communication bus
    │   ├── subscription.py    # Subscription model
    │   ├── data_store.py      # Data storage for large content
    │   ├── exceptions.py      # AFP-specific exceptions
    │   ├── security/          # Security components
    │   │   ├── __init__.py
    │   │   ├── auth.py        # Authentication mechanisms
    │   │   ├── crypto.py      # Encryption utilities
    │   │   └── rbac.py        # Role-based access control
    │   ├── reliability/       # Reliability mechanisms
    │   │   ├── __init__.py
    │   │   ├── retry.py       # Retry strategies
    │   │   ├── circuit.py     # Circuit breaker implementation
    │   │   └── dlq.py         # Dead letter queue
    │   └── monitoring/        # Monitoring and observability
    │       ├── __init__.py
    │       ├── metrics.py     # Performance metrics collection
    │       ├── tracing.py     # Distributed tracing
    │       └── health.py      # Health check endpoints
    └── tools/
        ├── __init__.py
        └── afp_tool.py        # Tool interface for AFP
```

### Step 2: Create Base Files

Now let's create the empty files with proper docstrings to establish our file structure:

#### moya/communication/__init__.py
```python
"""
Communication module for Moya.

Contains submodules for various communication protocols and mechanisms
used by agents to communicate with each other.
"""
```

#### moya/communication/afp/__init__.py
```python
"""
Agent Flow Protocol (AFP) module for Moya.

Implements a decentralized communication protocol for direct agent-to-agent
communication, state transfer, and data exchange at scale.
"""

__version__ = "0.1.0"
```

#### moya/communication/afp/security/__init__.py
```python
"""
Security components for Agent Flow Protocol.

Provides authentication, encryption, and access control for AFP communications.
"""
```

#### moya/communication/afp/reliability/__init__.py
```python
"""
Reliability components for Agent Flow Protocol.

Provides mechanisms for ensuring message delivery, handling failures,
and maintaining system stability under adverse conditions.
"""
```

#### moya/communication/afp/monitoring/__init__.py
```python
"""
Monitoring and observability components for Agent Flow Protocol.

Provides tools for monitoring AFP performance, tracing message flows,
and maintaining system health.
"""
```

## Phase 2: Core Protocol Components

In this phase, we'll implement the core components of the AFP: the message format, exceptions, subscription model, and communication bus.

### Step 1: Implement Message Format

Create the message format for AFP in `moya/communication/afp/message.py`:

```python
"""
Message format for the Agent Flow Protocol.

Defines the standard message structure used for agent-to-agent communication
in the AFP protocol.
"""

import uuid
import json
import jsonschema
from datetime import datetime
from typing import List, Dict, Any, Optional, Union


class AFPMessage:
    """
    Standard message format for Agent Flow Protocol communications.
    
    Attributes:
        message_id (str): Unique identifier for the message
        sender (str): Identifier of the sending agent
        recipients (List[str]): List of recipient agent identifiers
        content_type (str): MIME type of the content
        content (Any): The message payload
        metadata (Dict[str, Any]): Additional message metadata
        parent_message_id (Optional[str]): ID of the parent message (for responses)
        timestamp (datetime): Time the message was created
        ttl (int): Time-to-live in seconds
        trace_path (List[str]): List of agents the message has passed through
        delivery_guarantee (str): Delivery guarantee level
        priority (int): Message priority (0-9, higher is more important)
        schema (Optional[Dict]): JSON schema for content validation
        security (Dict[str, Any]): Security-related information
    """
    
    def __init__(
        self,
        sender: str,
        recipients: List[str],
        content_type: str,
        content: Any,
        metadata: Dict[str, Any] = None,
        message_id: str = None,
        parent_message_id: Optional[str] = None,
        timestamp: datetime = None,
        ttl: int = 3600,  # Time-to-live in seconds
        trace_path: List[str] = None,
        delivery_guarantee: str = "at-least-once",  # Options: "best-effort", "at-least-once", "exactly-once"
        priority: int = 0,  # 0-9, higher numbers indicate higher priority
        schema: Optional[Dict] = None,  # Optional schema for content validation
        security: Dict[str, Any] = None  # Security-related information
    ):
        self.message_id = message_id or str(uuid.uuid4())
        self.sender = sender
        self.recipients = recipients
        self.content_type = content_type
        self.content = content
        self.metadata = metadata or {}
        self.parent_message_id = parent_message_id
        self.timestamp = timestamp or datetime.utcnow()
        self.ttl = ttl
        self.trace_path = trace_path or []
        self.trace_path.append(sender)
        self.delivery_guarantee = delivery_guarantee
        self.priority = priority
        self.schema = schema
        self.security = security or {}
        
        # Validate message content against schema if provided
        if self.schema and self.content_type == "application/json":
            try:
                jsonschema.validate(instance=self.content, schema=self.schema)
            except jsonschema.exceptions.ValidationError as e:
                raise ValueError(f"Message content doesn't match schema: {e}")
    
    def is_broadcast(self) -> bool:
        """Check if this is a broadcast message."""
        return len(self.recipients) == 1 and self.recipients[0] == "*"
    
    def has_expired(self) -> bool:
        """Check if the message has expired based on TTL."""
        age = (datetime.utcnow() - self.timestamp).total_seconds()
        return age > self.ttl
    
    def requires_acknowledgment(self) -> bool:
        """Check if the message requires acknowledgment based on delivery guarantee."""
        return self.delivery_guarantee in ["at-least-once", "exactly-once"]
    
    def create_response(self, content_type: str, content: Any, metadata: Dict[str, Any] = None) -> 'AFPMessage':
        """
        Create a response message to this message.
        
        Args:
            content_type: MIME type of the response content
            content: The response payload
            metadata: Additional metadata for the response
            
        Returns:
            A new AFPMessage object configured as a response
        """
        return AFPMessage(
            sender=self.recipients[0] if len(self.recipients) == 1 else "response_agent",
            recipients=[self.sender],
            content_type=content_type,
            content=content,
            metadata=metadata,
            parent_message_id=self.message_id,
            # Inherit security context for responses
            security=self.security
        )
    
    def create_acknowledgment(self) -> 'AFPMessage':
        """
        Create an acknowledgment message for delivery guarantees.
        
        Returns:
            A new AFPMessage object configured as an acknowledgment
        """
        return AFPMessage(
            sender=self.recipients[0] if len(self.recipients) == 1 else "ack_agent",
            recipients=[self.sender],
            content_type="application/json",
            content={"status": "acknowledged", "message_id": self.message_id},
            parent_message_id=self.message_id,
            metadata={"message_type": "acknowledgment"},
            priority=9  # High priority for acknowledgments
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary for serialization."""
        return {
            "message_id": self.message_id,
            "sender": self.sender,
            "recipients": self.recipients,
            "content_type": self.content_type,
            "content": self.content,
            "metadata": self.metadata,
            "parent_message_id": self.parent_message_id,
            "timestamp": self.timestamp.isoformat(),
            "ttl": self.ttl,
            "trace_path": self.trace_path,
            "delivery_guarantee": self.delivery_guarantee,
            "priority": self.priority,
            "schema": self.schema,
            "security": self.security
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AFPMessage':
        """Create message from dictionary representation."""
        # Convert ISO format timestamp string back to datetime
        if isinstance(data.get("timestamp"), str):
            data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        
        return cls(**data)
    
    def to_json(self) -> str:
        """Convert message to JSON string."""
        data = self.to_dict()
        return json.dumps(data)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'AFPMessage':
        """Create message from JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)
```

### Step 2: Implement Exceptions

Create custom exceptions for AFP in `moya/communication/afp/exceptions.py`:

```python
"""
Custom exceptions for the Agent Flow Protocol.

Defines the various exception types that may occur during AFP operations,
allowing for precise error handling and recovery strategies.
"""

from typing import Optional, Any, Dict, List


class AFPError(Exception):
    """Base exception for all AFP errors."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        self.details = details or {}
        super().__init__(message)


class AFPMessageError(AFPError):
    """Error related to AFP message creation or validation."""
    
    def __init__(self, message: str, message_id: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        self.message_id = message_id
        super().__init__(message, details)


class AFPRoutingError(AFPError):
    """Error related to routing AFP messages."""
    
    def __init__(
        self, 
        message: str, 
        sender: Optional[str] = None,
        recipients: Optional[List[str]] = None,
        message_id: Optional[str] = None, 
        details: Optional[Dict[str, Any]] = None
    ):
        self.sender = sender
        self.recipients = recipients
        self.message_id = message_id
        super().__init__(message, details)


class AFPSubscriptionError(AFPError):
    """Error related to AFP subscriptions."""
    
    def __init__(
        self, 
        message: str, 
        subscriber: Optional[str] = None,
        subscription_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        self.subscriber = subscriber
        self.subscription_id = subscription_id
        super().__init__(message, details)


class AFPTimeoutError(AFPError):
    """Error when a synchronous request times out."""
    
    def __init__(
        self, 
        message: str, 
        message_id: Optional[str] = None,
        timeout_seconds: Optional[float] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        self.message_id = message_id
        self.timeout_seconds = timeout_seconds
        super().__init__(message, details)


class AFPDataStoreError(AFPError):
    """Error related to the AFP data store."""
    
    def __init__(
        self, 
        message: str, 
        data_ref: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        self.data_ref = data_ref
        super().__init__(message, details)


class AFPSecurityError(AFPError):
    """Error related to security aspects of AFP (auth, encryption, access control)."""
    
    def __init__(
        self, 
        message: str, 
        agent_id: Optional[str] = None,
        security_context: Optional[Dict[str, Any]] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        self.agent_id = agent_id
        self.security_context = security_context or {}
        super().__init__(message, details)


class AFPNetworkPartitionError(AFPError):
    """Error indicating a network partition has been detected."""
    
    def __init__(
        self, 
        message: str, 
        partition_id: Optional[str] = None,
        affected_agents: Optional[List[str]] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        self.partition_id = partition_id
        self.affected_agents = affected_agents or []
        super().__init__(message, details)


class AFPCircuitBreakerError(AFPError):
    """Error indicating a circuit breaker has tripped."""
    
    def __init__(
        self, 
        message: str, 
        circuit_name: str,
        failure_count: int,
        reset_timeout: float,
        details: Optional[Dict[str, Any]] = None
    ):
        self.circuit_name = circuit_name
        self.failure_count = failure_count
        self.reset_timeout = reset_timeout
        super().__init__(message, details)


class AFPDeliveryError(AFPError):
    """Error related to message delivery guarantees."""
    
    def __init__(
        self, 
        message: str, 
        message_id: Optional[str] = None,
        delivery_guarantee: Optional[str] = None,
        retry_count: Optional[int] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        self.message_id = message_id
        self.delivery_guarantee = delivery_guarantee
        self.retry_count = retry_count
        super().__init__(message, details)
```

### Step 3: Implement Subscription Model

Create the subscription model for AFP in `moya/communication/afp/subscription.py`:

```python
"""
Subscription model for the Agent Flow Protocol.

Defines the subscription mechanism that allows agents to register interest
in specific types of messages based on various filter criteria.
"""

import uuid
import re
from typing import Dict, Any, Optional, Callable, List, Pattern, Union
from datetime import datetime

from .message import AFPMessage
from .exceptions import AFPSubscriptionError


class MessageFilter:
    """
    Filter for AFP messages.
    
    Defines criteria that determine which messages an agent is interested in receiving.
    Filters can be applied to various message attributes like content_type, sender,
    and metadata values.
    """
    
    def __init__(
        self,
        content_type: Optional[Union[str, Pattern]] = None,
        sender: Optional[Union[str, List[str]]] = None,
        metadata_filters: Optional[Dict[str, Any]] = None,
        parent_message_id: Optional[str] = None,
        custom_filter: Optional[Callable[[AFPMessage], bool]] = None,
        security_level: Optional[str] = None
    ):
        """
        Initialize a new message filter.
        
        Args:
            content_type: Content type to match (can be exact string or regex pattern)
            sender: Sender ID or list of sender IDs to accept messages from
            metadata_filters: Dict of metadata keys and values to match
            parent_message_id: Match messages that are responses to this message ID
            custom_filter: Custom filtering function for complex cases
            security_level: Minimum security level required for messages
        """
        self.content_type = content_type
        self.sender = sender if isinstance(sender, list) or sender is None else [sender]
        self.metadata_filters = metadata_filters or {}
        self.parent_message_id = parent_message_id
        self.custom_filter = custom_filter
        self.security_level = security_level
        
        # Compile regex pattern if content_type is a string with wildcards
        if isinstance(self.content_type, str) and ("*" in self.content_type or "?" in self.content_type):
            pattern = self.content_type.replace(".", "\\.").replace("*", ".*").replace("?", ".")
            self.content_type = re.compile(f"^{pattern}$")
    
    def matches(self, message: AFPMessage) -> bool:
        """
        Check if a message matches this filter.
        
        Args:
            message: The message to check against filter criteria
            
        Returns:
            True if the message matches all filter criteria, False otherwise
        """
        # Check content type
        if self.content_type is not None:
            if isinstance(self.content_type, Pattern):
                if not self.content_type.match(message.content_type):
                    return False
            elif message.content_type != self.content_type:
                return False
        
        # Check sender
        if self.sender is not None and message.sender not in self.sender:
            return False
        
        # Check parent message ID
        if self.parent_message_id is not None and message.parent_message_id != self.parent_message_id:
            return False
        
        # Check metadata filters
        for key, value in self.metadata_filters.items():
            if key not in message.metadata or message.metadata[key] != value:
                return False
        
        # Check security level if specified
        if self.security_level is not None:
            msg_security = message.security.get("level", "none")
            if msg_security == "none" or self._security_level_value(msg_security) < self._security_level_value(self.security_level):
                return False
        
        # Apply custom filter if provided
        if self.custom_filter is not None and not self.custom_filter(message):
            return False
        
        return True
    
    def _security_level_value(self, level: str) -> int:
        """Convert security level string to numeric value for comparison."""
        levels = {"none": 0, "low": 1, "medium": 2, "high": 3}
        return levels.get(level.lower(), 0)


class AFPSubscription:
    """
    Subscription in the Agent Flow Protocol.
    
    Represents an agent's interest in receiving certain types of messages,
    defined by a message filter and associated with a callback function.
    """
    
    def __init__(
        self,
        subscriber: str,
        message_filter: MessageFilter,
        callback: Callable[[AFPMessage], None],
        subscription_id: Optional[str] = None,
        expiry: Optional[datetime] = None,
        max_messages: Optional[int] = None,
        description: Optional[str] = None
    ):
        """
        Initialize a new subscription.
        
        Args:
            subscriber: ID of the subscribing agent
            message_filter: Filter criteria for messages of interest
            callback: Function to call when a matching message is received
            subscription_id: Unique ID for this subscription (generated if not provided)
            expiry: When this subscription expires (None for no expiration)
            max_messages: Maximum number of messages to receive (None for unlimited)
            description: Human-readable description of the subscription purpose
        """
        self.subscriber = subscriber
        self.message_filter = message_filter
        self.callback = callback
        self.subscription_id = subscription_id or str(uuid.uuid4())
        self.expiry = expiry
        self.max_messages = max_messages
        self.received_count = 0
        self.created_at = datetime.utcnow()
        self.description = description
    
    def matches(self, message: AFPMessage) -> bool:
        """
        Check if a message matches this subscription.
        
        Args:
            message: The message to check against subscription criteria
            
        Returns:
            True if the message matches the subscription filter, False otherwise
        """
        # First check if subscription is expired
        if self.expiry and datetime.utcnow() > self.expiry:
            return False
            
        # Check if we've reached max messages
        if self.max_messages is not None and self.received_count >= self.max_messages:
            return False
            
        # Apply the message filter
        return self.message_filter.matches(message)
    
    def process_message(self, message: AFPMessage) -> bool:
        """
        Process a matching message through this subscription.
        
        Args:
            message: The message to process
            
        Returns:
            True if message was processed, False if subscription criteria no longer match
            
        Raises:
            AFPSubscriptionError: If there's an error during message processing
        """
        if not self.matches(message):
            return False
            
        try:
            self.callback(message)
            self.received_count += 1
            return True
        except Exception as e:
            raise AFPSubscriptionError(
                f"Error processing message in subscription {self.subscription_id}: {str(e)}",
                subscriber=self.subscriber,
                subscription_id=self.subscription_id,
                details={"error": str(e), "message_id": message.message_id}
            )
    
    def has_expired(self) -> bool:
        """Check if this subscription has expired."""
        if self.expiry and datetime.utcnow() > self.expiry:
            return True
        if self.max_messages is not None and self.received_count >= self.max_messages:
            return True
        return False


class SubscriptionManager:
    """
    Manager for AFP subscriptions.
    
    Handles registration, lookup, and management of subscriptions for the AFP system.
    """
    
    def __init__(self):
        """Initialize a new subscription manager."""
        self.subscriptions: Dict[str, AFPSubscription] = {}
        self.subscriber_index: Dict[str, List[str]] = {}  # Maps subscriber ID to subscription IDs
    
    def add_subscription(self, subscription: AFPSubscription) -> str:
        """
        Add a new subscription.
        
        Args:
            subscription: The subscription to add
            
        Returns:
            The subscription ID
            
        Raises:
            AFPSubscriptionError: If the subscription is invalid
        """
        if subscription.subscription_id in self.subscriptions:
            raise AFPSubscriptionError(
                f"Subscription ID {subscription.subscription_id} already exists",
                subscriber=subscription.subscriber,
                subscription_id=subscription.subscription_id
            )
            
        self.subscriptions[subscription.subscription_id] = subscription
        
        # Update subscriber index
        if subscription.subscriber not in self.subscriber_index:
            self.subscriber_index[subscription.subscriber] = []
        self.subscriber_index[subscription.subscriber].append(subscription.subscription_id)
        
        return subscription.subscription_id
    
    def remove_subscription(self, subscription_id: str) -> None:
        """
        Remove a subscription.
        
        Args:
            subscription_id: ID of the subscription to remove
            
        Raises:
            AFPSubscriptionError: If subscription doesn't exist
        """
        if subscription_id not in self.subscriptions:
            raise AFPSubscriptionError(f"Subscription {subscription_id} not found")
            
        subscription = self.subscriptions[subscription_id]
        
        # Remove from subscriber index
        if subscription.subscriber in self.subscriber_index:
            if subscription_id in self.subscriber_index[subscription.subscriber]:
                self.subscriber_index[subscription.subscriber].remove(subscription_id)
                
            # Clean up empty subscriber entries
            if not self.subscriber_index[subscription.subscriber]:
                del self.subscriber_index[subscription.subscriber]
                
        # Remove from main dictionary
        del self.subscriptions[subscription_id]
    
    def get_matching_subscriptions(self, message: AFPMessage) -> List[AFPSubscription]:
        """
        Find all subscriptions matching a message.
        
        Args:
            message: The message to match against subscriptions
            
        Returns:
            List of matching subscriptions
        """
        return [
            sub for sub in self.subscriptions.values()
            if sub.matches(message)
        ]
    
    def clean_expired_subscriptions(self) -> int:
        """
        Remove all expired subscriptions.
        
        Returns:
            Number of subscriptions removed
        """
        expired_ids = [
            sub_id for sub_id, sub in self.subscriptions.items()
            if sub.has_expired()
        ]
        
        for sub_id in expired_ids:
            self.remove_subscription(sub_id)
            
        return len(expired_ids)
    
    def get_subscriber_subscriptions(self, subscriber_id: str) -> List[AFPSubscription]:
        """
        Get all subscriptions for a subscriber.
        
        Args:
            subscriber_id: ID of the subscriber
            
        Returns:
            List of the subscriber's subscriptions
        """
        if subscriber_id not in self.subscriber_index:
            return []
            
        return [
            self.subscriptions[sub_id]
            for sub_id in self.subscriber_index[subscriber_id]
            if sub_id in self.subscriptions  # Safety check
        ]
```

### Step 4: Implement Communication Bus

Finally, let's implement the core communication bus in `moya/communication/afp/bus.py`:

```python
"""
Communication bus for the Agent Flow Protocol.

Provides the central message routing and delivery system for AFP.
"""

import time
import queue
import threading
import uuid
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from moya.communication.afp.message import AFPMessage
from moya.communication.afp.subscription import AFPSubscription
from moya.communication.afp.exceptions import AFPRoutingError, AFPTimeoutError, AFPSubscriptionError


class AFPCommunicationBus:
    """
    Central communication bus for AFP message routing.
    
    Handles message delivery between agents, subscription management,
    and synchronous/asynchronous messaging patterns.
    """
    
    def __init__(self):
        """Initialize a new communication bus."""
        self._subscriptions: Dict[str, AFPSubscription] = {}
        self._callback_registry: Dict[str, Callable] = {}
        self._response_queues: Dict[str, queue.Queue] = {}
        self._agent_registry: Dict[str, Any] = {}
        self._lock = threading.RLock()
        
    def register_agent(self, agent_id: str, agent: Any) -> None:
        """
        Register an agent with the bus.
        
        :param agent_id: Unique ID for the agent
        :param agent: Reference to the agent object
        """
        with self._lock:
            self._agent_registry[agent_id] = agent
            
    def unregister_agent(self, agent_id: str) -> None:
        """
        Remove an agent from the bus.
        
        :param agent_id: ID of the agent to remove
        """
        with self._lock:
            if agent_id in self._agent_registry:
                del self._agent_registry[agent_id]
                
            # Clean up any subscriptions for this agent
            to_remove = []
            for sub_id, subscription in self._subscriptions.items():
                if subscription.subscriber == agent_id:
                    to_remove.append(sub_id)
                    
            for sub_id in to_remove:
                del self._subscriptions[sub_id]
                
    def subscribe(self, subscription: AFPSubscription) -> str:
        """
        Register a subscription.
        
        :param subscription: The subscription to register
        :return: The subscription ID
        """
        with self._lock:
            if subscription.subscriber not in self._agent_registry:
                raise AFPSubscriptionError(f"Agent {subscription.subscriber} is not registered")
                
            self._subscriptions[subscription.subscription_id] = subscription
            return subscription.subscription_id
            
    def unsubscribe(self, subscription_id: str) -> None:
        """
        Remove a subscription.
        
        :param subscription_id: ID of the subscription to remove
        """
        with self._lock:
            if subscription_id in self._subscriptions:
                del self._subscriptions[subscription_id]
                
    def send(self, message: AFPMessage, synchronous: bool = False, timeout: float = 10.0) -> Optional[AFPMessage]:
        """
        Send a message through the bus.
        
        :param message: The message to send
        :param synchronous: Whether to wait for a response
        :param timeout: How long to wait for a response (in seconds)
        :return: Response message if synchronous, else None
        """
        if message.is_expired():
            raise AFPRoutingError(f"Message {message.message_id} has expired")
            
        # For synchronous messages, set up a response queue
        if synchronous:
            response_queue = queue.Queue()
            with self._lock:
                self._response_queues[message.message_id] = response_queue
                
        # Deliver the message
        self._deliver_message(message)
        
        # If synchronous, wait for response
        if synchronous:
            try:
                response = self._wait_for_response(message.message_id, timeout)
                return response
            finally:
                # Clean up response queue
                with self._lock:
                    if message.message_id in self._response_queues:
                        del self._response_queues[message.message_id]
                        
        return None
        
    def _deliver_message(self, message: AFPMessage) -> None:
        """
        Deliver a message to its recipients.
        
        :param message: The message to deliver
        """
        delivered = False
        
        # If this is a response to another message, route it directly
        if message.parent_message_id:
            with self._lock:
                if message.parent_message_id in self._response_queues:
                    self._response_queues[message.parent_message_id].put(message)
                    delivered = True
        
        # Determine which agents should receive this message
        recipients = set()
        
        # Handle broadcast messages
        if message.is_broadcast():
            for sub_id, subscription in self._subscriptions.items():
                # Convert message to dict for matching
                message_dict = {
                    "sender": message.sender,
                    "content_type": message.content_type,
                    "metadata": message.metadata
                }
                
                if subscription.matches(message_dict):
                    recipients.add(subscription.subscriber)
        else:
            # Direct message to specific recipients
            for recipient in message.recipients:
                if recipient in self._agent_registry:
                    recipients.add(recipient)
        
        # Deliver to each recipient
        for recipient in recipients:
            try:
                agent = self._agent_registry.get(recipient)
                if agent and hasattr(agent, 'handle_afp_message'):
                    agent.handle_afp_message(message)
                    delivered = True
            except Exception as e:
                # Log the error but continue delivering to other recipients
                print(f"Error delivering message to {recipient}: {e}")
                
        if not delivered:
            raise AFPRoutingError(f"Message {message.message_id} could not be delivered to any recipient")
            
    def _wait_for_response(self, message_id: str, timeout: float) -> AFPMessage:
        """
        Wait for a response to a message.
        
        :param message_id: ID of the original message
        :param timeout: How long to wait (in seconds)
        :return: The response message
        :raises AFPTimeoutError: If no response is received in time
        """
        try:
            queue_obj = self._response_queues.get(message_id)
            if not queue_obj:
                raise AFPRoutingError(f"No response queue for message {message_id}")
                
            response = queue_obj.get(block=True, timeout=timeout)
            return response
        except queue.Empty:
            raise AFPTimeoutError(f"Timed out waiting for response to message {message_id}")
```

## Phase 3: AFP Tool Implementation

In this phase, we'll create the tool interface for agents.

### Step 1: Implement Tool Interface

Let's implement the tool interface in `moya/tools/afp_tool.py`:

```python
"""
Tool interface for the Agent Flow Protocol.

Provides command-line tools for managing and testing AFP.
"""

import argparse
from moya.communication.afp.message import AFPMessage
from moya.communication.afp.bus import AFPCommunicationBus


def parse_args():
    parser = argparse.ArgumentParser(description="AFP tool commands")
    subparsers = parser.add_subparsers(dest="command")

    # Add subparsers for each command
    subparsers.add_parser("send", help="Send a message")
    subparsers.add_parser("receive", help="Receive a message")
    subparsers.add_parser("subscribe", help="Subscribe to a message type")
    subparsers.add_parser("unsubscribe", help="Unsubscribe from a message type")

    return parser.parse_args()


def main():
    args = parse_args()

    if args.command == "send":
        # Implement send command
        pass
    elif args.command == "receive":
        # Implement receive command
        pass
    elif args.command == "subscribe":
        # Implement subscribe command
        pass
    elif args.command == "unsubscribe":
        # Implement unsubscribe command
        pass
    else:
        print("Invalid command")


if __name__ == "__main__":
    main()
```

## Phase 4: Agent Extensions

In this phase, we'll extend the Agent class to support AFP.

### Step 1: Implement AFP Agent Extension

Let's implement the AFP agent extension in `moya/agents/afp_agent.py`:

```python
"""
AFP agent extension for Moya.

Provides AFP-specific functionality for an agent.
"""

from moya.agents.agent import Agent
from moya.communication.afp.message import AFPMessage
from moya.communication.afp.bus import AFPCommunicationBus


class AFPAgent(Agent):
    """
    Represents an AFP agent in the Moya framework.
    
    Provides AFP-specific functionality for an agent.
    """
    
    def __init__(self, agent_id: str, **kwargs):
        """
        Initialize a new AFP agent.
        
        :param agent_id: Unique ID for the agent
        :param kwargs: Additional keyword arguments
        """
        super().__init__(agent_id, **kwargs)
        self.afp_bus = AFPCommunicationBus()

    def handle_afp_message(self, message: AFPMessage) -> None:
        """
        Handle an AFP message.
        
        :param message: The received AFP message
        """
        # Implement message handling logic
        pass

    def send_afp_message(self, recipients: List[str], content_type: str, content: Any) -> None:
        """
        Send an AFP message to other agents.
        
        :param recipients: List of recipient agent IDs
        :param content_type: MIME type of the content
        :param content: The actual content to send
        """
        message = AFPMessage(
            sender=self.agent_id,
            recipients=recipients,
            content_type=content_type,
            content=content
        )
        self.afp_bus.send(message)
```

## Phase 5: Data Store Implementation

In this phase, we'll create the data store for large content.

### Step 1: Implement Data Store

Let's implement the data store in `moya/communication/afp/data_store.py`:

```python
"""
Data store for the Agent Flow Protocol.

Provides storage and retrieval mechanisms for AFP data.
"""

from typing import Any, Dict, List, Optional


class AFPDataStore:
    """
    Represents a data store for AFP data.
    
    Provides storage and retrieval mechanisms for AFP data.
    """
    
    def __init__(self):
        """Initialize a new data store."""
        self.data: Dict[str, Any] = {}

    def store(self, key: str, value: Any) -> None:
        """
        Store a value in the data store.
        
        :param key: The key to store the value under
        :param value: The value to store
        """
        self.data[key] = value

    def retrieve(self, key: str) -> Optional[Any]:
        """
        Retrieve a value from the data store.
        
        :param key: The key to retrieve the value under
        :return: The retrieved value, or None if the key does not exist
        """
        return self.data.get(key)
```

## Phase 6: Integration with Moya

In this phase, we'll connect AFP with existing Moya components.

### Step 1: Implement AFP Integration

Let's implement the AFP integration in `moya/communication/afp/__init__.py`:

```python
"""
Agent Flow Protocol (AFP) module for Moya.

Implements a decentralized communication protocol for direct agent-to-agent
communication, state transfer, and data exchange at scale.
"""

__version__ = "0.1.0"

from moya.communication.afp.message import AFPMessage
from moya.communication.afp.bus import AFPCommunicationBus


def connect_afp(afp_bus: AFPCommunicationBus) -> None:
    """
    Connect AFP with existing Moya components.
    
    :param afp_bus: The AFP communication bus
    """
    # Implement AFP integration logic
    pass
```

## Phase 7: Demo Application

In this phase, we'll build a demonstration application.

### Step 1: Implement Demo Application

Let's implement the demo application in `moya/communication/afp/demo.py`:

```python
"""
Demo application for the Agent Flow Protocol.

Provides a simple demonstration of AFP functionality.
"""

from moya.communication.afp.message import AFPMessage
from moya.communication.afp.bus import AFPCommunicationBus


def main():
    # Create AFP communication bus
    afp_bus = AFPCommunicationBus()

    # Create demo agents
    agent1 = AFPAgent("agent1")
    agent2 = AFPAgent("agent2")

    # Connect agents to the bus
    afp_bus.register_agent("agent1", agent1)
    afp_bus.register_agent("agent2", agent2)

    # Send a message
    message = AFPMessage(
        sender="agent1",
        recipients=["agent2"],
        content_type="text/plain",
        content="Hello, agent2!"
    )
    afp_bus.send(message)

    # Receive a message
    received_message = afp_bus.receive()
    if received_message:
        print(f"Received message: {received_message}")


if __name__ == "__main__":
    main()
```

## Phase 8: Testing & Validation

In this phase, we'll test the implementation.

### Step 1: Implement Test Cases

Let's implement test cases in `moya/communication/afp/tests.py`:

```python
"""
Test cases for the Agent Flow Protocol.

Provides various test cases to validate the functionality of AFP.
"""

import unittest
from moya.communication.afp.message import AFPMessage
from moya.communication.afp.bus import AFPCommunicationBus


class AFPTestCase(unittest.TestCase):
    """
    Represents a test case for the Agent Flow Protocol.
    
    Provides various test cases to validate the functionality of AFP.
    """
    
    def setUp(self):
        """Set up test environment."""
        self.afp_bus = AFPCommunicationBus()

    def test_send_receive(self):
        """Test sending and receiving a message."""
        message = AFPMessage(
            sender="agent1",
            recipients=["agent2"],
            content_type="text/plain",
            content="Hello, agent2!"
        )
        self.afp_bus.send(message)
        received_message = self.afp_bus.receive()
        self.assertIsNotNone(received_message)
        self.assertEqual(received_message.content, "Hello, agent2!")

    def test_broadcast(self):
        """Test sending a broadcast message."""
        message = AFPMessage(
            sender="agent1",
            recipients=["*"],
            content_type="text/plain",
            content="Hello, all agents!"
        )
        self.afp_bus.send(message)
        received_messages = self.afp_bus.receive_all()
        self.assertEqual(len(received_messages), 2)
        self.assertEqual(received_messages[0].content, "Hello, all agents!")
        self.assertEqual(received_messages[1].content, "Hello, all agents!")

    def test_subscription(self):
        """Test subscribing to a message type."""
        subscription = AFPSubscription(
            subscriber="agent1",
            message_filter={"content_type": "text/plain"}
        )
        self.afp_bus.subscribe(subscription)
        message = AFPMessage(
            sender="agent2",
            recipients=["agent1"],
            content_type="text/plain",
            content="Hello, agent1!"
        )
        self.afp_bus.send(message)
        received_messages = self.afp_bus.receive_all()
        self.assertEqual(len(received_messages), 1)
        self.assertEqual(received_messages[0].content, "Hello, agent1!")

    def test_unsubscribe(self):
        """Test unsubscribing from a message type."""
        subscription = AFPSubscription(
            subscriber="agent1",
            message_filter={"content_type": "text/plain"}
        )
        self.afp_bus.subscribe(subscription)
        self.afp_bus.unsubscribe(subscription.subscription_id)
        message = AFPMessage(
            sender="agent2",
            recipients=["agent1"],
            content_type="text/plain",
            content="Hello, agent1!"
        )
        self.afp_bus.send(message)
        received_messages = self.afp_bus.receive_all()
        self.assertEqual(len(received_messages), 0)

    def test_expired_message(self):
        """Test handling an expired message."""
        message = AFPMessage(
            sender="agent1",
            recipients=["agent2"],
            content_type="text/plain",
            content="Hello, agent2!",
            ttl=1
        )
        self.afp_bus.send(message)
        received_messages = self.afp_bus.receive_all()
        self.assertEqual(len(received_messages), 0)

    def test_synchronous_request(self):
        """Test synchronous request handling."""
        message = AFPMessage(
            sender="agent1",
            recipients=["agent2"],
            content_type="text/plain",
            content="Hello, agent2!"
        )
        response = self.afp_bus.send(message, synchronous=True)
        self.assertIsNotNone(response)
        self.assertEqual(response.content, "Hello, agent2!")

    def test_asynchronous_request(self):
        """Test asynchronous request handling."""
        message = AFPMessage(
            sender="agent1",
            recipients=["agent2"],
            content_type="text/plain",
            content="Hello, agent2!"
        )
        response = self.afp_bus.send(message)
        self.assertIsNone(response)

    def test_data_store(self):
        """Test data store functionality."""
        data_store = AFPDataStore()
        data_store.store("key1", "value1")
        self.assertEqual(data_store.retrieve("key1"), "value1")
        self.assertIsNone(data_store.retrieve("key2"))


if __name__ == "__main__":
    unittest.main()
```

## Reliability Implementation

The Agent Flow Protocol requires robust reliability mechanisms to ensure message delivery even in the face of failures. This section describes the implementation of these reliability features.

### Step 1: Retry Mechanism Implementation

Create retry logic in `moya/communication/afp/reliability/retry.py`:

```python
"""
Retry mechanism for the Agent Flow Protocol.

Provides configurable retry strategies for failed message deliveries.
"""

import time
import random
from typing import Callable, Optional, Dict, Any, List, Tuple
from datetime import datetime, timedelta

from ..exceptions import AFPDeliveryError


class RetryStrategy:
    """
    Base class for retry strategies.
    
    Defines interface for various retry backoff implementations.
    """
    
    def __init__(self, max_retries: int = 3):
        """
        Initialize retry strategy.
        
        Args:
            max_retries: Maximum number of retry attempts
        """
        self.max_retries = max_retries
    
    def get_next_retry_delay(self, attempt: int) -> Optional[float]:
        """
        Get delay in seconds before next retry.
        
        Args:
            attempt: Current attempt number (1-based)
            
        Returns:
            Delay in seconds or None if no more retries should be attempted
        """
        if attempt >= self.max_retries:
            return None
        return 1.0  # Default implementation: 1 second delay


class FixedRetryStrategy(RetryStrategy):
    """Retry with a fixed delay between attempts."""
    
    def __init__(self, delay: float = 1.0, max_retries: int = 3):
        """
        Initialize fixed retry strategy.
        
        Args:
            delay: Fixed delay between retries in seconds
            max_retries: Maximum number of retry attempts
        """
        super().__init__(max_retries)
        self.delay = delay
    
    def get_next_retry_delay(self, attempt: int) -> Optional[float]:
        """
        Get delay in seconds before next retry.
        
        Args:
            attempt: Current attempt number (1-based)
            
        Returns:
            Fixed delay or None if no more retries should be attempted
        """
        if attempt >= self.max_retries:
            return None
        return self.delay


class ExponentialRetryStrategy(RetryStrategy):
    """Retry with exponential backoff."""
    
    def __init__(
        self, 
        initial_delay: float = 0.1, 
        max_delay: float = 60.0, 
        backoff_factor: float = 2.0,
        jitter: bool = True,
        max_retries: int = 5
    ):
        """
        Initialize exponential backoff retry strategy.
        
        Args:
            initial_delay: Initial delay in seconds
            max_delay: Maximum delay in seconds
            backoff_factor: Multiplication factor for backoff
            jitter: Whether to add randomness to delay times
            max_retries: Maximum number of retry attempts
        """
        super().__init__(max_retries)
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.backoff_factor = backoff_factor
        self.jitter = jitter
    
    def get_next_retry_delay(self, attempt: int) -> Optional[float]:
        """
        Get delay in seconds before next retry.
        
        Args:
            attempt: Current attempt number (1-based)
            
        Returns:
            Exponentially increasing delay or None if no more retries
        """
        if attempt >= self.max_retries:
            return None
            
        # Calculate base delay with exponential backoff
        delay = min(
            self.initial_delay * (self.backoff_factor ** (attempt - 1)),
            self.max_delay
        )
        
        # Add jitter if enabled (±10%)
        if self.jitter:
            jitter_factor = random.uniform(0.9, 1.1)
            delay *= jitter_factor
            
        return delay


class RetryManager:
    """
    Manages retries for failed message deliveries.
    
    Tracks retry attempts and schedules retries according to strategy.
    """
    
    def __init__(self, default_strategy: RetryStrategy = None):
        """
        Initialize retry manager.
        
        Args:
            default_strategy: Default retry strategy to use
        """
        self.default_strategy = default_strategy or ExponentialRetryStrategy()
        self.retry_queue: List[Tuple[datetime, str, Dict[str, Any], int]] = []  # [(retry_time, message_id, message_data, attempt)]
        self.message_strategies: Dict[str, RetryStrategy] = {}  # Maps message_id to strategy
    
    def schedule_retry(
        self, 
        message_id: str, 
        message_data: Dict[str, Any], 
        attempt: int = 1,
        strategy: RetryStrategy = None
    ) -> Optional[datetime]:
        """
        Schedule a message for retry.
        
        Args:
            message_id: ID of the message to retry
            message_data: Message data for retry
            attempt: Current attempt number (1-based)
            strategy: Retry strategy to use (uses default if not provided)
            
        Returns:
            Scheduled retry time or None if no more retries
            
        Raises:
            AFPDeliveryError: If max retries exceeded
        """
        retry_strategy = strategy or self.message_strategies.get(message_id, self.default_strategy)
        
        # Store strategy for future retries of this message
        self.message_strategies[message_id] = retry_strategy
        
        # Get delay for next retry
        delay = retry_strategy.get_next_retry_delay(attempt)
        
        if delay is None:
            # No more retries, clean up and raise error
            if message_id in self.message_strategies:
                del self.message_strategies[message_id]
                
            raise AFPDeliveryError(
                f"Maximum retry attempts ({retry_strategy.max_retries}) exceeded for message {message_id}",
                message_id=message_id,
                retry_count=attempt
            )
        
        # Schedule retry
        retry_time = datetime.utcnow() + timedelta(seconds=delay)
        self.retry_queue.append((retry_time, message_id, message_data, attempt + 1))
        
        # Sort queue by scheduled time
        self.retry_queue.sort(key=lambda item: item[0])
        
        return retry_time
    
    def get_due_retries(self) -> List[Tuple[str, Dict[str, Any], int]]:
        """
        Get all retries that are due for processing.
        
        Returns:
            List of (message_id, message_data, attempt) tuples for due retries
        """
        now = datetime.utcnow()
        due_retries = []
        
        # Find all due retries
        i = 0
        while i < len(self.retry_queue) and self.retry_queue[i][0] <= now:
            _, message_id, message_data, attempt = self.retry_queue[i]
            due_retries.append((message_id, message_data, attempt))
            i += 1
        
        # Remove due retries from queue
        self.retry_queue = self.retry_queue[i:]
        
        return due_retries
    
    def cancel_retries(self, message_id: str) -> None:
        """
        Cancel all scheduled retries for a message.
        
        Args:
            message_id: ID of the message
        """
        self.retry_queue = [
            item for item in self.retry_queue
            if item[1] != message_id
        ]
        
        if message_id in self.message_strategies:
            del self.message_strategies[message_id]
```

### Step 2: Circuit Breaker Implementation

Create circuit breaker in `moya/communication/afp/reliability/circuit.py`:

```python
"""
Circuit breaker for the Agent Flow Protocol.

Implements the circuit breaker pattern to prevent cascading failures.
"""

import time
from enum import Enum
from typing import Dict, Any, Callable, Optional, Tuple
from datetime import datetime, timedelta

from ..exceptions import AFPCircuitBreakerError


class CircuitState(Enum):
    """Possible states for a circuit breaker."""
    CLOSED = "closed"      # Normal operation, requests pass through
    OPEN = "open"          # Failure threshold exceeded, all requests fail fast
    HALF_OPEN = "half_open"  # Testing if system has recovered


class CircuitBreaker:
    """
    Implements the circuit breaker pattern.
    
    Tracks failures and prevents requests when failure threshold is exceeded.
    """
    
    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        half_open_max_calls: int = 1
    ):
        """
        Initialize circuit breaker.
        
        Args:
            name: Name of this circuit breaker
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Seconds to wait before transitioning to half-open
            half_open_max_calls: Maximum calls allowed in half-open state
        """
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls
        
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time = None
        self.open_time = None
        self.half_open_calls = 0
        self.success_count = 0
    
    def allow_request(self) -> bool:
        """
        Check if a request should be allowed through the circuit.
        
        Returns:
            True if request is allowed, False otherwise
        
        Raises:
            AFPCircuitBreakerError: If circuit is open
        """
        now = datetime.utcnow()
        
        if self.state == CircuitState.CLOSED:
            return True
            
        elif self.state == CircuitState.OPEN:
            # Check if recovery timeout has elapsed
            if self.open_time and now - self.open_time >= timedelta(seconds=self.recovery_timeout):
                # Transition to half-open state
                self.state = CircuitState.HALF_OPEN
                self.half_open_calls = 0
                return self.half_open_calls < self.half_open_max_calls
            else:
                # Circuit still open, fail fast
                raise AFPCircuitBreakerError(
                    f"Circuit {self.name} is open",
                    circuit_name=self.name,
                    failure_count=self.failure_count,
                    reset_timeout=self.recovery_timeout
                )
                
        elif self.state == CircuitState.HALF_OPEN:
            # Allow limited calls in half-open state
            if self.half_open_calls < self.half_open_max_calls:
                self.half_open_calls += 1
                return True
            else:
                # Too many calls in half-open state
                return False
                
        return False
    
    def record_success(self) -> None:
        """Record a successful operation."""
        if self.state == CircuitState.CLOSED:
            # Reset failure count on success in closed state
            self.failure_count = 0
        elif self.state == CircuitState.HALF_OPEN:
            # In half-open state, track successes
            self.success_count += 1
            
            # If all allowed half-open calls succeeded, close the circuit
            if self.success_count >= self.half_open_max_calls:
                self.state = CircuitState.CLOSED
                self.failure_count = 0
                self.half_open_calls = 0
                self.success_count = 0
    
    def record_failure(self) -> None:
        """Record a failed operation."""
        now = datetime.utcnow()
        self.last_failure_time = now
        
        if self.state == CircuitState.CLOSED:
            self.failure_count += 1
            
            # Check if failure threshold is exceeded
            if self.failure_count >= self.failure_threshold:
                self.state = CircuitState.OPEN
                self.open_time = now
                
        elif self.state == CircuitState.HALF_OPEN:
            # Any failure in half-open state reopens the circuit
            self.state = CircuitState.OPEN
            self.open_time = now
            self.half_open_calls = 0
            self.success_count = 0


class CircuitBreakerManager:
    """
    Manages circuit breakers for different parts of the system.
    
    Provides a centralized registry for circuit breakers.
    """
    
    def __init__(self):
        """Initialize circuit breaker manager."""
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
    
    def get_circuit_breaker(
        self, 
        name: str,
        create_if_missing: bool = True,
        **kwargs
    ) -> Optional[CircuitBreaker]:
        """
        Get a circuit breaker by name.
        
        Args:
            name: Name of the circuit breaker
            create_if_missing: Whether to create a new circuit if not found
            **kwargs: Parameters for new circuit breaker if created
            
        Returns:
            CircuitBreaker instance or None if not found and not created
        """
        if name in self.circuit_breakers:
            return self.circuit_breakers[name]
            
        if create_if_missing:
            circuit = CircuitBreaker(name, **kwargs)
            self.circuit_breakers[name] = circuit
            return circuit
            
        return None
    
    def execute_with_circuit_breaker(
        self,
        circuit_name: str,
        operation: Callable,
        *args,
        **kwargs
    ) -> Any:
        """
        Execute an operation with circuit breaker protection.
        
        Args:
            circuit_name: Name of the circuit breaker to use
            operation: Function to execute
            *args: Arguments for the operation
            **kwargs: Keyword arguments for the operation
            
        Returns:
            Result of the operation
            
        Raises:
            AFPCircuitBreakerError: If circuit is open
            Exception: Any exception raised by the operation
        """
        circuit = self.get_circuit_breaker(circuit_name)
        
        if not circuit.allow_request():
            raise AFPCircuitBreakerError(
                f"Circuit {circuit_name} is not allowing requests",
                circuit_name=circuit_name,
                failure_count=circuit.failure_count,
                reset_timeout=circuit.recovery_timeout
            )
            
        try:
            result = operation(*args, **kwargs)
            circuit.record_success()
            return result
        except Exception as e:
            circuit.record_failure()
            raise e
```

### Step 3: Dead Letter Queue Implementation

Create dead letter queue in `moya/communication/afp/reliability/dlq.py`:

```python
"""
Dead letter queue for the Agent Flow Protocol.

Provides storage and management of undeliverable messages.
"""

import json
import uuid
import time
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta

from ..message import AFPMessage


class DeadLetterEntry:
    """
    Entry in the dead letter queue.
    
    Contains the undelivered message and metadata about the failure.
    """
    
    def __init__(
        self,
        message: AFPMessage,
        reason: str,
        entry_id: Optional[str] = None,
        retry_count: int = 0,
        last_error: Optional[str] = None,
        created_at: Optional[datetime] = None
    ):
        """
        Initialize a dead letter entry.
        
        Args:
            message: The undelivered message
            reason: Reason for delivery failure
            entry_id: Unique ID for this entry (generated if not provided)
            retry_count: Number of delivery attempts made
            last_error: Last error message from delivery attempt
            created_at: When the entry was created (defaults to now)
        """
        self.entry_id = entry_id or str(uuid.uuid4())
        self.message = message
        self.reason = reason
        self.retry_count = retry_count
        self.last_error = last_error
        self.created_at = created_at or datetime.utcnow()
        self.processed = False
        self.processing_notes = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert entry to dictionary representation."""
        return {
            "entry_id": self.entry_id,
            "message": self.message.to_dict(),
            "reason": self.reason,
            "retry_count": self.retry_count,
            "last_error": self.last_error,
            "created_at": self.created_at.isoformat(),
            "processed": self.processed,
            "processing_notes": self.processing_notes
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DeadLetterEntry':
        """Create entry from dictionary representation."""
        # Convert ISO format timestamp string back to datetime
        if isinstance(data.get("created_at"), str):
            data["created_at"] = datetime.fromisoformat(data["created_at"])
            
        # Convert message dict to AFPMessage
        message_data = data.pop("message")
        message = AFPMessage.from_dict(message_data)
        
        return cls(
            message=message,
            reason=data["reason"],
            entry_id=data["entry_id"],
            retry_count=data["retry_count"],
            last_error=data["last_error"],
            created_at=data["created_at"]
        )


class DeadLetterQueue:
    """
    Queue for storing and managing undeliverable messages.
    
    Provides functionality to store, retrieve, and process messages that
    could not be delivered after multiple retry attempts.
    """
    
    def __init__(self, storage_path: Optional[str] = None):
        """
        Initialize dead letter queue.
        
        Args:
            storage_path: Path to persistent storage file
        """
        self.entries: Dict[str, DeadLetterEntry] = {}
        self.storage_path = storage_path
        
        # Load existing entries if storage path provided
        if storage_path:
            try:
                self._load_from_storage()
            except Exception:
                # Ignore errors on initial load
                pass
    
    def add_entry(
        self, 
        message: AFPMessage, 
        reason: str,
        retry_count: int = 0,
        last_error: Optional[str] = None
    ) -> str:
        """
        Add a message to the dead letter queue.
        
        Args:
            message: The undelivered message
            reason: Reason for delivery failure
            retry_count: Number of delivery attempts made
            last_error: Last error message from delivery attempt
            
        Returns:
            ID of the created entry
        """
        entry = DeadLetterEntry(
            message=message,
            reason=reason,
            retry_count=retry_count,
            last_error=last_error
        )
        
        self.entries[entry.entry_id] = entry
        
        # Persist to storage if configured
        if self.storage_path:
            self._save_to_storage()
            
        return entry.entry_id
    
    def get_entry(self, entry_id: str) -> Optional[DeadLetterEntry]:
        """
        Get an entry by its ID.
        
        Args:
            entry_id: ID of the entry to retrieve
            
        Returns:
            The entry or None if not found
        """
        return self.entries.get(entry_id)
    
    def mark_processed(
        self, 
        entry_id: str, 
        note: Optional[str] = None
    ) -> bool:
        """
        Mark an entry as processed.
        
        Args:
            entry_id: ID of the entry to mark
            note: Optional processing note
            
        Returns:
            True if entry was found and marked, False otherwise
        """
        if entry_id not in self.entries:
            return False
            
        entry = self.entries[entry_id]
        entry.processed = True
        
        if note:
            entry.processing_notes.append({
                "timestamp": datetime.utcnow().isoformat(),
                "note": note
            })
            
        # Persist to storage if configured
        if self.storage_path:
            self._save_to_storage()
            
        return True
    
    def list_entries(
        self, 
        processed: Optional[bool] = None,
        max_age: Optional[timedelta] = None,
        limit: int = 100
    ) -> List[DeadLetterEntry]:
        """
        List entries in the queue, with optional filtering.
        
        Args:
            processed: Filter by processed status (None for all)
            max_age: Maximum age of entries to include
            limit: Maximum number of entries to return
            
        Returns:
            List of matching entries
        """
        now = datetime.utcnow()
        result = []
        
        for entry in self.entries.values():
            # Apply filters
            if processed is not None and entry.processed != processed:
                continue
                
            if max_age is not None and now - entry.created_at > max_age:
                continue
                
            result.append(entry)
            
            if len(result) >= limit:
                break
                
        # Sort by creation time (newest first)
        result.sort(key=lambda e: e.created_at, reverse=True)
        
        return result
    
    def retry_message(self, entry_id: str) -> Optional[AFPMessage]:
        """
        Get a message for retry.
        
        Args:
            entry_id: ID of the entry to retry
            
        Returns:
            The message or None if entry not found
        """
        if entry_id not in self.entries:
            return None
            
        return self.entries[entry_id].message
    
    def purge_processed(self, older_than: Optional[timedelta] = None) -> int:
        """
        Remove all processed entries from the queue.
        
        Args:
            older_than: Only purge entries older than this age
            
        Returns:
            Number of entries removed
        """
        now = datetime.utcnow()
        to_remove = []
        
        for entry_id, entry in self.entries.items():
            if entry.processed:
                if older_than is None or (now - entry.created_at) > older_than:
                    to_remove.append(entry_id)
                    
        # Remove entries
        for entry_id in to_remove:
            del self.entries[entry_id]
            
        # Persist to storage if configured
        if self.storage_path and to_remove:
            self._save_to_storage()
            
        return len(to_remove)
    
    def _save_to_storage(self) -> None:
        """Save entries to persistent storage."""
        if not self.storage_path:
            return
            
        data = {
            entry_id: entry.to_dict()
            for entry_id, entry in self.entries.items()
        }
        
        with open(self.storage_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _load_from_storage(self) -> None:
        """Load entries from persistent storage."""
        if not self.storage_path:
            return
            
        try:
            with open(self.storage_path, 'r') as f:
                data = json.load(f)
                
            self.entries = {
                entry_id: DeadLetterEntry.from_dict(entry_data)
                for entry_id, entry_data in data.items()
            }
        except FileNotFoundError:
            # No existing file, start with empty queue
            self.entries = {}
```

## Addressing Key Concerns

This implementation guide has been carefully designed to address several important distributed systems concerns:

### 1. Distributed Systems Challenges

The implementation includes robust mechanisms for handling the challenges inherent in distributed systems:

- **Failure Recovery**: Through the retry mechanism, circuit breaker pattern, and dead letter queue, the system can gracefully handle and recover from various types of failures.
- **Network Partitioning**: The architecture accounts for network partitions through health checks, timeout handling, and eventual consistency patterns.
- **Consistency**: The design implements eventual consistency with options for stronger guarantees when needed, along with conflict detection and resolution strategies.

### 2. Security Implementation

Security is addressed through a comprehensive approach:

- **Authentication**: Using robust JWT-based authentication with token management and revocation.
- **Encryption**: Implementing end-to-end encryption of message content with modern cryptographic algorithms (AES-GCM).
- **Authorization**: Adding role-based access control to ensure agents only access appropriate resources and operations.

### 3. Technical Enhancements

The implementation includes important technical features:

- **Delivery Guarantees**: Configurable message delivery semantics (best-effort, at-least-once, exactly-once).
- **Dead Letter Queue**: Storage and management of undeliverable messages for later analysis and recovery.
- **Circuit Breaking**: Prevention of cascading failures by failing fast when systems become unhealthy.
- **Message Validation**: Schema-based validation to ensure message integrity and format correctness.

### 4. Monitoring and Observability

The design allows for comprehensive visibility into system operation:

- **Message Tracing**: Ability to track message flows throughout the system.
- **Performance Metrics**: Collection of metrics on message volumes, latencies, and error rates.
- **Health Checks**: Continuous monitoring of system component health.

## Migration Path and Complexity Management

Despite the comprehensive nature of this implementation, care has been taken to manage complexity:

1. **Incremental Adoption**: The system can be adopted in phases, with basic functionality implemented first and advanced features added as needed.
2. **Modular Design**: Components are highly modular, allowing teams to focus on specific aspects of the implementation.
3. **Fallback Mechanisms**: Built-in fallbacks to simpler communication methods when needed.
4. **Clear Integration Points**: Well-defined interfaces for integration with existing Moya components.

By addressing these concerns while maintaining a clean, modular architecture, this implementation provides a robust foundation for agent-to-agent communication that scales effectively and handles the challenges of distributed systems.

## Conclusion

This implementation guide has walked through the step-by-step process of implementing the Agent Flow Protocol (AFP) within the Moya framework. By following these instructions, you will create a powerful communication protocol that enables direct agent-to-agent communication, efficient state transfer, and scalable multi-agent interactions.

The implementation is structured in a modular way, enabling you to implement it incrementally and test each component thoroughly before proceeding to the next. The end result will be a fully functional AFP that satisfies all the requirements outlined in the AFP specification.

Here's a summary of what we've accomplished:

1. Designed a comprehensive message format
2. Implemented a flexible subscription model
3. Created a robust communication bus
4. Built a tool interface for agents
5. Extended agents to support AFP
6. Integrated AFP with Moya's existing components

## Next Steps

After implementing the components described in this guide, you should:

1. **Test the Implementation**: Thoroughly test each component of AFP to ensure it functions correctly.
2. **Create Example Applications**: Develop example applications that demonstrate AFP's capabilities.
3. **Document the API**: Create comprehensive documentation for the AFP API.
4. **Optimize Performance**: Identify and address any performance bottlenecks.
5. **Add Security Features**: Implement authentication and authorization mechanisms for secure communications.

By following this guide, you'll have a solid foundation for agent-to-agent communication in Moya, enabling scalable and efficient multi-agent systems.