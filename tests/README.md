# AFP Test Suite

This directory contains tests for the Agent Flow Protocol (AFP) implementation.

## Directory Structure

- **performance/** - Performance benchmarking tests to compare AFP with baseline implementations.
- **integration/** - Integration tests demonstrating AFP in real-world scenarios, including Azure OpenAI integration.
- **unit/** - Unit tests for individual AFP components.

## Running Tests

### Performance Tests
To compare AFP performance with baseline:
```bash
# Run baseline tests
python tests/performance/baseline_test.py

# Run AFP performance tests
python tests/performance/afp_performance_test.py
```

### Integration Tests
To test AFP with real-world scenarios:
```bash
# Run the example application
python tests/integration/afp_example_app.py

# Run the Azure OpenAI integration tests (requires Azure credentials)
python tests/integration/afp_ai_integration_test.py
```

### Unit Tests
To verify individual AFP components:
```bash
# Run specific component tests
python tests/unit/test_message.py
python tests/unit/test_subscription.py
python tests/unit/test_bus.py
python tests/unit/test_afp_components.py
``` 