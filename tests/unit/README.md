# Unit Tests

This directory contains unit tests for the individual components of the Agent Flow Protocol (AFP) implementation.

## Tests Included

- **test_message.py** - Tests for the AFP message model, ensuring message creation, validation, and serialization work correctly.
- **test_subscription.py** - Tests for the subscription system, verifying that agents can subscribe to and receive relevant messages.
- **test_bus.py** - Tests for the communication bus, validating message routing and delivery mechanics.
- **test_afp_components.py** - Tests for additional AFP components including:
  - **Circuit Breaker** - Tests for failure detection and recovery
  - **Distributed Tracing** - Tests for trace creation and propagation
  - **Metrics Collection** - Tests for performance metrics gathering

## Running the Tests

```bash
# Run all unit tests
python -m unittest discover tests/unit

# Run specific tests
python tests/unit/test_message.py
python tests/unit/test_subscription.py
python tests/unit/test_bus.py
python tests/unit/test_afp_components.py
```

## Test Coverage

These unit tests cover:
- Core functionality of each AFP component
- Edge cases and error handling
- State transitions and expected behaviors
- Component interactions

Each test file focuses on verifying the correctness of a specific part of the AFP implementation, ensuring that all components function as expected before they are combined in more complex integration tests. 