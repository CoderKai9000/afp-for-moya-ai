# AFP Performance Tests

This directory contains performance tests for the Agent Flow Protocol (AFP) implementation. These tests measure the performance of various communication patterns, including direct communication, multi-agent orchestration, and complex workflows.

## Running the Tests

### Standard Performance Tests

To run the standard performance tests:

```bash
python afp_performance_test.py
```

This will run a series of tests with different configurations and save the results to `afp_results.json`.

### Optimized Performance Tests

To run the optimized performance tests:

```bash
python afp_optimized_test.py
```

This will run tests with the optimized AFP implementation and save the results to `afp_optimized_results.json`.

## Test Descriptions

### Direct Communication

Tests direct message sending between two agents without an orchestrator. Measures the latency and throughput of simple point-to-point communication.

### Multi-Agent Orchestration

Tests message routing through an orchestrator to multiple agents. Measures the performance of the orchestrator in distributing messages to multiple recipients.

### Complex Workflow

Tests a complex workflow with multiple message hops between agents. This simulates a business process with sequential steps, where each agent processes a message and passes it to the next agent in the workflow.

## Interpreting Results

The test results include the following metrics:

- **Total Time**: The total time taken to complete all iterations of the test.
- **Message Count**: The total number of messages sent during the test.
- **Average Latency**: The average time taken to process a single message (in milliseconds).
- **Throughput**: The number of messages processed per second.

For complex workflows, additional metrics are provided:

- **Workflow Completion Time**: The time taken to complete each workflow from start to finish.

## Comparing Results

To compare the performance of different implementations, run both the standard and optimized tests and compare the results. The key metrics to compare are:

1. **Throughput**: Higher is better. Indicates how many messages the system can process per second.
2. **Latency**: Lower is better. Indicates how quickly the system responds to messages.

For a detailed analysis of the optimizations and performance improvements, see the [Optimization Summary](OPTIMIZATION_SUMMARY.md). 