# MOYA - Meta Orchestration framework for Your Agents

MOYA is a reference implementation of our research paper titled "Engineering LLM Powered Multi-agent Framework for Autonomous CloudOps". The framework provides a flexible and extensible architecture for creating, managing, and orchestrating multiple AI agents to handle various tasks autonomously.

Preprint of the paper can be accessed at [arXiv](https://arxiv.org/abs/2501.08243).

## Features

- **Agent Management**: Create, register, and manage multiple AI agents.
- **Orchestration**: Orchestrate conversations and tasks across multiple agents.
- **Memory Tools**: Integrate memory tools to maintain conversation context and history.
- **Streaming Responses**: Support for streaming responses from agents.
- **Extensibility**: Easily extend the framework with new agents, tools, and orchestrators.

## Getting Started

### Prerequisites

- Python 3.10+
- Install required dependencies:
  ```bash
  pip install .
  ```

### Quick Start Examples

#### OpenAI Agent

Interactive chat example using OpenAI agent with conversation memory.

```python
# filepath: /Users/kannan/src/github/moya/examples/quick_start_openai.py

python -m examples.quick_start_openai

```

#### Bedrock Agent

Interactive chat example using BedrockAgent with conversation memory.

```python
# filepath: /Users/kannan/src/github/moya/examples/quick_start_bedrock.py

AWS_PROFILE=my_profile_name python -m examples.quick_start_bedrock
```

#### Multi-Agent Orchestration

Example demonstrating multi-agent orchestration with language and task classification.

```python
# filepath: /Users/kannan/src/github/moya/examples/quick_start_multiagent.py

python -m examples.quick_start_multiagent
```

#### Dynamic Agent Creation

Example demonstrating dynamic agent creation and registration during runtime.
![moya](./media/Dynamic_Agents.mov)

```python
# filepath: /Users/kannan/src/github/moya/examples/dynamic_agents.py


python -m examples.dynamic_agents
```

### Directory Structure

```
moya/
├── agents/                # Agent implementations (OpenAI, Bedrock, Ollama, Remote)
├── classifiers/           # Classifier implementations for agent selection
├── memory/                # Memory repository implementations
├── orchestrators/         # Orchestrator implementations for managing agent interactions
├── registry/              # Agent registry and repository implementations
├── tools/                 # Tool implementations (e.g., MemoryTool)
├── examples/              # Example scripts demonstrating various use cases
└── README.md              # This README file
```

### Contributing

We welcome contributions to the MOYA framework. Please follow these steps to contribute:

1. Fork the repository.
2. Create a new branch for your feature or bugfix.
3. Make your changes and commit them with descriptive messages.
4. Push your changes to your fork.
5. Create a pull request to the main repository.

### License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

### Contact

For any questions or inquiries, please contact the authors of the research paper or open an issue on the GitHub repository.

# Agent Flow Protocol (AFP) Implementation

The Agent Flow Protocol (AFP) is a protocol designed for reliable communication between AI agents. This implementation provides a robust foundation for building multi-agent systems with features for reliability, observability, and security.

## Directory Structure

```
├── moya/                        # Main Moya framework
│   └── communication/           
│       └── afp/                 # AFP implementation
├── tests/                       # Test suite
│   ├── performance/             # Performance benchmarking tests
│   ├── integration/             # Integration and real-world tests
│   ├── unit/                    # Component unit tests
│   └── results/                 # Test result data
```

## Running Tests

### Performance Comparison

To compare the performance of AFP with the baseline implementation:

```bash
# Run baseline tests
python tests/performance/baseline_test.py

# Run AFP performance tests
python tests/performance/afp_performance_test.py
```

These tests will generate performance metrics for direct communication, multi-agent orchestration, and complex workflows in JSON files under the `tests/results/` directory.

### Integration Tests

To test AFP with real-world scenarios, including AI API integration:

```bash
# Set Azure OpenAI credentials (if needed for integration tests)
# Replace with your actual credentials
$env:AZURE_OPENAI_API_KEY="your-api-key"
$env:AZURE_OPENAI_ENDPOINT="your-endpoint"

# Run the Azure OpenAI integration test
python tests/integration/afp_ai_integration_test.py

# Run the example application
python tests/integration/afp_example_app.py
```

### Unit Tests

To verify individual AFP components:

```bash
# Run message model tests
python tests/unit/test_message.py

# Run subscription system tests
python tests/unit/test_subscription.py

# Run communication bus tests
python tests/unit/test_bus.py

# Run component tests (circuit breaker, metrics, tracing)
python tests/unit/test_afp_components.py
```

## Test Results

The performance tests compare AFP against baseline implementation in three key areas:

1. **Direct Communication**: AFP shows significant improvements (-57.42% latency, +134.87% throughput)
2. **Multi-Agent Orchestration**: AFP provides massive throughput improvements (+26,863%)
3. **Complex Workflows**: AFP adds reliability with minimal overhead (+10.51% latency)

The integration tests with Azure OpenAI show AFP maintains efficiency while adding security and reliability mechanisms:
1. **API Performance**: +0.85% throughput, -2.29% response time
2. **Workflow Performance**: +0.07% throughput, -8.73% task time

## Implementation Highlights

- **Circuit Breaker Pattern**: Fault detection and recovery
- **Distributed Tracing**: Message flow tracking
- **Metrics Collection**: Performance monitoring
- **Security**: Enhanced communication security

## Next Steps

- Further performance optimization
- Additional integration with external services
- Extended workflow support
