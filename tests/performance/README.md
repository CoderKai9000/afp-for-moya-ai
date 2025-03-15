# Performance Tests

This directory contains tests specifically designed to benchmark and compare the performance of the Agent Flow Protocol (AFP) implementation against the baseline approach.

## Tests Included

- **baseline_test.py** - Establishes baseline performance metrics without AFP for direct communication, multi-agent orchestration, and complex workflows.
- **afp_performance_test.py** - Measures performance of the same operations using AFP, allowing for direct comparison with baseline results.

## Running the Tests

```bash
# Run baseline performance test
python tests/performance/baseline_test.py

# Run AFP performance test
python tests/performance/afp_performance_test.py
```

## Output Files

When executed, these tests generate JSON result files:
- `baseline_results.json` - Contains performance metrics from the baseline test
- `afp_results.json` - Contains performance metrics from the AFP test
- `afp_comparison.json` - Contains a comparison between baseline and AFP performance

## Key Metrics

The tests measure the following key performance indicators:
- **Latency** - Time required to complete operations
- **Throughput** - Number of operations completed per second
- **Resource Utilization** - CPU and memory usage during operations
- **Scalability** - Performance under increasing load 