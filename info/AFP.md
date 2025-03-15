# Agent Flow Protocol (AFP) Specification

## Overview

The Agent Flow Protocol (AFP) is a standardized communication protocol designed for the Moya framework to enable direct agent-to-agent communication, state transfer, and data exchange at scale. This protocol addresses the need for a decentralized communication mechanism when operating with hundreds of agents that need to collaborate efficiently without relying solely on centralized orchestrators.

## Core Objectives

1. **Direct Communication**: Enable agents to communicate directly with one another without always going through a central orchestrator
2. **State Transfer**: Provide mechanisms for transferring state and context between agents
3. **Data Exchange**: Support exchange of various data types (text, structured data, binary, streaming media)
4. **Scalability**: Scale efficiently to hundreds of agents operating simultaneously
5. **Discoverability**: Allow agents to discover each other's capabilities and services

## When to Use AFP vs. Orchestrators

AFP is designed to complement, not replace, Moya's existing orchestrator pattern. Here are guidelines for when each approach is most appropriate:

### Use AFP When:

- You need to scale to hundreds of agents with frequent inter-agent communication
- Performance is critical and removing central bottlenecks is necessary
- Agents need to form dynamic, peer-to-peer communication patterns
- Large data transfers between specific agents are common
- You need robust fault tolerance in agent communication

### Continue Using Orchestrators When:

- You have simpler use cases with few agents
- Centralized control and oversight of all message flows is required
- Your workflows follow clear hierarchical patterns
- You need simplified debugging and message tracing
- You're extending existing Moya applications without complex agent communication needs

Most sophisticated applications will benefit from a hybrid approach, using orchestrators for high-level workflow management and AFP for direct agent-to-agent communication when appropriate.

## Protocol Components

### 1. Message Format

```python
class AFPMessage:
    def __init__(
        self,
        sender: str,
        recipients: List[str],
        content_type: str,
        content: Any,
        metadata: Dict[str, Any] = None,
        message_id: str = None,
        parent_message_id: str = None,
        timestamp: datetime = None,
        ttl: int = 3600,  # Time-to-live in seconds
        trace_path: List[str] = None,
        delivery_guarantee: str = "at-least-once",  # Options: "best-effort", "at-least-once", "exactly-once"
        priority: int = 0,  # 0-9, higher numbers indicate higher priority
        schema: Dict = None  # Optional schema for message validation
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
```

### 2. Communication Bus

The AFP Communication Bus serves as the central infrastructure for routing messages between agents. It handles:

- Message routing based on recipient(s)
- Subscription management for topic-based messaging
- Message delivery guarantees
- Transport abstraction (HTTP, WebSockets, etc.)
- Recovery mechanisms for failed deliveries
- Circuit breaking for resilience
- Dead letter queuing for undeliverable messages

### 3. Protocol Tool

AFP is implemented as a tool in Moya's tool registry, providing a consistent interface for agents to interact with the protocol:

```python
class AFPProtocolTool(BaseTool):
    """
    Tool for inter-agent communication using the Agent Flow Protocol.
    """
    
    def __init__(
        self, 
        communication_bus,
        name: str = "AFPProtocolTool",
        description: str = "Tool for agent-to-agent communication using AFP protocol."
    ):
        super().__init__(name=name, description=description)
        self.communication_bus = communication_bus
```

### 4. AFP-Aware Agent

An extension of the base Agent class that adds AFP capabilities:

```python
class AFPAwareAgent(Agent):
    """
    An agent that can participate in AFP protocol communication.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._message_handlers = {}
        self._subscriptions = []
```

## Message Types and Operations

### 1. Synchronous Request-Response

Used when an agent needs an immediate response from another agent:

```python
response = agent.call_tool(
    tool_name="AFPProtocolTool",
    method_name="request",
    recipient="analysis_agent",
    content_type="application/json",
    content={"data": "to_analyze"},
    timeout=5.0  # Wait up to 5 seconds for response
)
```

### 2. Asynchronous Publish-Subscribe

Used for "fire and forget" scenarios or broadcast communications:

```python
agent.call_tool(
    tool_name="AFPProtocolTool",
    method_name="publish",
    recipients=["*"],  # Broadcast to all interested agents
    content_type="application/json",
    content={"status": "completed", "results": results},
    response_expected=False
)
```

### 3. State Transfer

Used to share state information between agents:

```python
agent.call_tool(
    tool_name="AFPProtocolTool",
    method_name="publish",
    recipients=["task_agents"],
    content_type="application/state",
    content=current_state,
    metadata={"state_type": "task_progress", "task_id": "123"}
)
```

### 4. Large Data Transfer

Used for efficient transfer of large data (images, videos, etc.):

```python
# Store large data and get reference
data_reference = agent.call_tool(
    tool_name="AFPProtocolTool",
    method_name="store_data",
    content_type="video/mp4",
    content=video_binary_data
)

# Send message with reference
agent.call_tool(
    tool_name="AFPProtocolTool",
    method_name="publish",
    recipients=["video_processing_agent"],
    content_type="application/json",
    content={"data_ref": data_reference, "action": "process"}
)
```

## Distributed Systems Challenges

AFP addresses several critical distributed systems challenges:

### Failure Recovery

- **Retry Mechanisms**: Configurable retry strategies for failed message deliveries
- **Acknowledgments**: Explicit message acknowledgments for at-least-once and exactly-once delivery
- **Dead Letter Queue**: Undeliverable messages are moved to a dead letter queue for later processing or inspection
- **Circuit Breaking**: Automatic circuit breaking for failing agent connections to prevent cascading failures

### Network Partitioning

- **Partition Detection**: Heartbeat mechanisms to detect network partitions
- **Partition Recovery**: Automatic state reconciliation when partitions heal
- **Local Operation**: Agents can continue operating within their partition with eventual consistency guarantees

### Consistency Management

- **Conflict Detection**: Vector clocks or version numbers to detect conflicting updates
- **Conflict Resolution**: Customizable conflict resolution strategies (last-writer-wins, custom merge logic)
- **Eventual Consistency**: By default, AFP provides eventual consistency with configurable stronger guarantees when needed

## Security Considerations

### Authentication and Authorization

- **Agent Identity**: Each agent has a cryptographically verifiable identity
- **Authentication Mechanisms**:
  - Mutual TLS for transport-level authentication
  - JWT tokens for application-level authentication
  - API keys for service-to-service authentication
- **Authorization Controls**:
  - Fine-grained permissions based on sender-recipient pairs
  - Content-type and operation-based access controls
  - Role-based access control for agent categories

### Data Protection

- **Message Encryption**:
  - Transport encryption (TLS) for all communications
  - Optional end-to-end encryption for sensitive content
  - Content-specific encryption for particular data types
- **Secure Storage**:
  - Encrypted storage for message content and data references
  - Automatic data expiry for sensitive information
  - Key rotation and management

### Audit and Compliance

- **Audit Logging**: Comprehensive logging of all security-relevant operations
- **Compliance Controls**: Features to support GDPR, HIPAA, and other regulatory requirements
- **Security Scanning**: Tools to identify potential security issues in message patterns

## Performance Considerations

### Efficiency

- **Reference-Based Transfers**: For large data
- **Transport Options**: Choose the most appropriate transport mechanism
- **Message Filtering**: Early filtering to reduce unnecessary processing
- **Compression**: Automatic compression for large messages based on content type

### Scalability

- **Distributed Communication**: Direct agent-to-agent communication reduces central bottlenecks
- **Subscription-Based Routing**: Efficient message delivery to interested parties
- **Horizontal Scaling**: Communication bus can be distributed
- **Resource Controls**:
  - Rate limiting to prevent any single agent from overwhelming the system
  - Backpressure mechanisms to handle overload situations
  - Priority queuing for critical messages

## Operational Features

### Monitoring and Observability

- **Telemetry**: Built-in metrics for message volumes, latencies, and error rates
- **Tracing**: Distributed tracing for message flows across agents
- **Logging**: Structured logging with correlation IDs for message tracking
- **Visualization**: Tools to visualize message flows and agent relationships

### Reliability Features

- **Delivery Guarantees**: Configurable delivery semantics
  - Best-effort: No guarantee of delivery (fastest)
  - At-least-once: Messages will be delivered at least once (possible duplicates)
  - Exactly-once: Messages will be delivered exactly once (slowest, most resource-intensive)
- **Message Validation**: Schema-based validation to prevent malformed messages
- **Health Checks**: Automatic health checking of communication paths
- **Failure Detection**: Quick detection of failed agents and communication paths

## Integration with Moya Components

### Orchestrators

AFP is designed to complement, not replace, Moya's orchestrators:

- **Orchestrator-Directed Flow**: Orchestrators can continue to manage high-level workflows
- **Direct Agent Communication**: Agents can communicate directly when appropriate
- **Hybrid Approach**: A new `AFPAwareOrchestrator` can leverage both patterns

### Tools Registry

AFP leverages Moya's existing tool registry:

- Registered as a standard tool
- Available to any agent with access to the tool registry
- Can be discovered like any other tool

### Memory System

AFP extends Moya's memory capabilities:

- **SharedStateRepository**: For sharing state between agents
- **Reference System**: For efficient large data handling
- **Persistent Storage**: Optional for reliability

## Migration and Integration

### Migration Path from Existing Moya Applications

1. **Assessment Phase**:
   - Analyze existing agent communication patterns
   - Identify bottlenecks in current orchestrator-based communication
   - Determine which agent interactions would benefit from direct communication

2. **Incremental Adoption**:
   - Start by implementing AFP in non-critical agent pairs
   - Gradually migrate high-volume communication paths to AFP
   - Keep orchestrators for workflow management while using AFP for direct communication

3. **Hybrid Operation**:
   - Use orchestrators for high-level control flow
   - Use AFP for direct communication between frequently interacting agents
   - Maintain compatibility with both communication methods during transition

### Fallback Mechanisms

- **Graceful Degradation**: Automatic fallback to orchestrator-based communication when AFP encounters issues
- **Circuit Breaking**: Prevent cascading failures by failing fast and falling back to alternative communication paths
- **Monitoring**: Alert on communication pattern changes to detect fallbacks

## Versioning and Compatibility

The AFP will follow semantic versioning principles:

- **Major Version**: Incompatible API changes
- **Minor Version**: Backwards-compatible feature additions
- **Patch Version**: Backwards-compatible bug fixes

## Future Extensions

Potential future enhancements to the protocol:

1. **Federated Communication**: Communication across different Moya instances
2. **Advanced Filtering**: Complex pattern matching for message subscriptions
3. **Quality of Service**: Different delivery guarantees for different message types
4. **Compression**: Automatic compression for large data transfers
5. **Streaming Protocols**: Native support for media streaming 