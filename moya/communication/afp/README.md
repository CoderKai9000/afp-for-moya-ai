# Agent Flow Protocol (AFP)

The Agent Flow Protocol (AFP) is a decentralized communication protocol for direct agent-to-agent communication, state transfer, and data exchange at scale. It provides a robust foundation for building multi-agent systems within the Moya framework.

## Features

- **Flexible Messaging**: Support for various content types, including text, JSON, binary data, and streaming media.
- **Subscription Model**: Agents can subscribe to specific message patterns, enabling selective message delivery.
- **Synchronous and Asynchronous Communication**: Support for both request-response and fire-and-forget messaging patterns.
- **Security**: Authentication, encryption, and authorization mechanisms to secure agent communications.
- **Reliability**: Delivery guarantees, retries, dead letter queues, and circuit breakers for robust communication.
- **Monitoring and Observability**: Metrics collection and distributed tracing for performance analysis and debugging.

## Architecture

The AFP module is organized into several components:

- **Core Components**:
  - `message.py`: Defines the message format and serialization.
  - `subscription.py`: Implements the subscription model for selective message delivery.
  - `bus.py`: Provides the communication bus for message routing and delivery.
  - `exceptions.py`: Defines custom exceptions for error handling.

- **Security Components** (in `security/` directory):
  - `auth.py`: Authentication mechanisms for verifying agent identities.
  - `encryption.py`: Encryption mechanisms for securing message content.
  - `authorization.py`: Authorization mechanisms for controlling access to resources.

- **Reliability Components** (in `reliability/` directory):
  - `delivery.py`: Delivery guarantees, acknowledgments, and retries.
  - `circuit_breaker.py`: Circuit breaker pattern for preventing cascading failures.

- **Monitoring Components** (in `monitoring/` directory):
  - `metrics.py`: Metrics collection for performance analysis.
  - `tracing.py`: Distributed tracing for message flows and operations.

## Usage

### Basic Message Sending

```python
from moya.communication.afp.message import AFPMessage, ContentType
from moya.communication.afp.bus import AFPCommunicationBus

# Create a communication bus
bus = AFPCommunicationBus()

# Register agents
bus.register_agent("agent1")
bus.register_agent("agent2")

# Create a message
message = AFPMessage(
    sender="agent1",
    recipients=["agent2"],
    content_type=ContentType.TEXT,
    content="Hello, agent2!"
)

# Send the message asynchronously
bus.send_message(message)

# Send the message synchronously and wait for a response
response = bus.send_message_sync(message, timeout=5.0)
```

### Subscribing to Messages

```python
from moya.communication.afp.message import ContentType

# Define a callback function
def handle_message(message):
    print(f"Received message: {message.content}")

# Subscribe to messages from agent1
bus.subscribe(
    subscriber="agent2",
    callback=handle_message,
    pattern={"sender": "agent1", "content_type": ContentType.TEXT.name}
)
```

### Secure Communication

```python
from moya.communication.afp.security.auth import HMACAuthenticator
from moya.communication.afp.security.encryption import AESGCMEncryptor

# Create authenticator and register agents
authenticator = HMACAuthenticator()
secret_key1 = authenticator.register_agent("agent1")
secret_key2 = authenticator.register_agent("agent2")

# Sign a message
signed_message = authenticator.sign_message(message, "agent1", secret_key1)

# Verify a message
is_authentic = authenticator.verify_message(signed_message)

# Create encryptor and generate keys
encryptor = AESGCMEncryptor()
key1, salt1 = encryptor.generate_key("passphrase1")
key2, salt2 = encryptor.generate_key("passphrase2")

# Encrypt a message for specific recipients
encrypted_message = encryptor.encrypt_for_recipient(
    message,
    recipient_keys={"agent2": key2}
)

# Decrypt a message
decrypted_message = encryptor.decrypt_for_recipient(
    encrypted_message,
    "agent2",
    key2
)
```

### Reliable Messaging

```python
from moya.communication.afp.reliability.delivery import MessageTracker, DeliveryGuarantee

# Create a message tracker
tracker = MessageTracker()

# Track a message with at-least-once delivery guarantee
tracker.track_message(
    message,
    guarantee=DeliveryGuarantee.AT_LEAST_ONCE,
    on_send=lambda msg: bus.send_message(msg)
)

# Acknowledge message delivery
tracker.acknowledge(message.message_id)
```

### Monitoring and Tracing

```python
from moya.communication.afp.monitoring.metrics import AFPMetricsCollector
from moya.communication.afp.monitoring.tracing import AFPMessageTracer

# Create a metrics collector
metrics = AFPMetricsCollector()

# Record message metrics
metrics.record_message_sent(len(message.serialize()), "agent1")
metrics.record_message_received(len(message.serialize()), "agent2")

# Create a message tracer
tracer = AFPMessageTracer()

# Trace message flow
span_id = tracer.trace_message_send(
    message.message_id,
    "agent1",
    ["agent2"],
    ContentType.TEXT.name,
    len(message.serialize())
)

# End the span
tracer.end_span(span_id)

# Get message trace
trace = tracer.get_message_trace(message.message_id)
```

## Extending AFP

The AFP module is designed to be extensible. You can:

- Implement custom authenticators by extending `AFPAuthenticator`.
- Implement custom encryptors by extending `AFPEncryptor`.
- Add custom metrics and traces for specific use cases.
- Implement custom delivery guarantees and reliability mechanisms.

## Best Practices

- **Security**: Always use authentication and encryption for sensitive communications.
- **Error Handling**: Handle exceptions appropriately to prevent cascading failures.
- **Monitoring**: Collect metrics and traces to identify performance bottlenecks and issues.
- **Resource Management**: Properly clean up resources by unregistering agents and shutting down components.
- **Message Size**: Keep message sizes reasonable to avoid performance issues.
- **Timeouts**: Use appropriate timeouts for synchronous operations to prevent blocking.

## Limitations and Considerations

- **Network Partitioning**: AFP provides mechanisms for handling network partitions, but application-level logic may be needed for specific use cases.
- **Eventual Consistency**: In distributed systems, eventual consistency is often the best guarantee available.
- **Performance**: Large messages or high message volumes may require performance tuning.
- **Security**: While AFP provides security mechanisms, they should be used appropriately for the specific threat model.

## Future Enhancements

- **Persistent Storage**: Add support for persisting messages to disk for durability.
- **Clustering**: Enhance the communication bus to support clustered deployments.
- **Protocol Bridges**: Add bridges to other communication protocols (e.g., MQTT, AMQP).
- **Advanced Routing**: Implement more sophisticated routing algorithms for complex topologies.
- **Schema Validation**: Add support for validating message content against schemas.

## Contributing

Contributions to the AFP module are welcome! Please follow the Moya project's contribution guidelines. 