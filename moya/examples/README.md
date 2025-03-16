# Knowledge Graph Benchmarking Tools

This directory contains tools for benchmarking and evaluating the performance and accuracy of AFP multiagent systems with and without knowledge graphs.

## Overview

The `kg_vs_standard.py` script provides comprehensive benchmarking capabilities to:

1. **Measure Performance Impact**: Compare the overhead and total processing time between standard AFP and knowledge graph-enhanced AFP.
2. **Evaluate Accuracy**: Test how well knowledge is maintained across multi-turn conversations.
3. **Visualize Results**: Generate graphs showing the performance differences.

## Quick Start

### Performance Benchmark

Run a basic performance comparison:

```bash
python moya/examples/kg_vs_standard.py
```

This will run a default benchmark with 3 test queries and 3 runs per query, comparing standard AFP to knowledge graph-enhanced AFP.

### Accuracy Benchmark

Test how well knowledge is maintained across a multi-turn conversation:

```bash
python moya/examples/kg_vs_standard.py --accuracy
```

This will run a simulated 5-turn conversation and measure how well context and entities are retained in both systems.

## Command Line Options

### General Options

- `--mock`: Use mock components for testing without actual AFP implementation
- `--seed-only`: Only seed the knowledge graph with sample data and exit
- `--output FILENAME`: Specify a custom filename for the benchmark results
- `--simplified`: Show only simplified output (good for quick tests)

### Performance Benchmark Options

- `--queries N`: Number of test queries to run (default: 3)
- `--runs N`: Number of runs per query for averaging (default: 3)
- `--custom-query "Your query here"`: Run a single custom query instead of test queries
- `--visualize`: Generate visualizations of the benchmark results
- `--no-optimize`: Disable KG performance optimizations
- `--no-enhanced`: Disable enhanced entity extraction

### Accuracy Benchmark Options

- `--turns N`: Number of conversation turns for accuracy benchmark (default: 5)

## Output Formats

The benchmark provides several output formats:

1. **Simplified Comparison**: An easy-to-understand summary with clear percentage differences and visual indicators
2. **Full Results Summary**: Detailed breakdown of all metrics
3. **Advanced Analysis**: In-depth analysis of performance characteristics
4. **Accuracy Results**: For multi-turn conversation testing, shows how well knowledge is maintained

Example of simplified output:

```
================================================================================
KNOWLEDGE GRAPH IMPACT - SIMPLIFIED BENCHMARK RESULTS
================================================================================

📊 PERFORMANCE COMPARISON:
  Metric          Standard AFP    KG-Enhanced     Difference      Impact    
  --------------- --------------- --------------- --------------- ----------
  Total Time      0.6012s         0.8023s         +0.2011s (+33.4%)    ↑
  Overhead        0.1008s         0.3016s         +0.2008s (+199.1%)   ↑
  API Time        0.5004s         0.5007s         +0.0003s (+0.1%)     →

🧠 KNOWLEDGE ENHANCEMENT:
  Total entities identified: 8
  Queries enriched with knowledge: 3/3 (100.0%)

  Query-specific enhancement:
  1. "Tell me about Microsoft Azure and its main service..."
     Entities: 2, Context enriched: Yes
  2. "How do neural networks work? Explain the concept o..."
     Entities: 3, Context enriched: Yes
  3. "What are the key differences between Python and Ja..."
     Entities: 3, Context enriched: Yes

📝 OVERALL ASSESSMENT:
  Performance impact: 127.3% overhead
  KG efficiency rating: FAIR

  Key takeaways:
  1. Consider optimizing the knowledge graph for better performance.
  2. Low cache hit ratio. Consider tuning cache settings or preseeding with more domain knowledge.
  3. For best results, balance between performance and knowledge graph richness based on your specific use case.
================================================================================
```

## Interpreting Results

### Performance Metrics

- **Total Time**: The full processing time, including both API calls and overhead
- **Overhead**: Time spent on knowledge graph-specific operations
- **API Time**: Time spent waiting for API responses from language models

### Accuracy Metrics

- **Context Retention**: How well context from earlier turns is maintained in later turns
- **Entity References**: How many key entities are referenced in responses
- **Context Enrichment**: How many turns were enhanced with knowledge graph context

## Example Use Cases

1. **Performance Optimization**:
   ```bash
   python moya/examples/kg_vs_standard.py --queries 10 --runs 5 --visualize
   ```
   
2. **Quick Test with Custom Query**:
   ```bash
   python moya/examples/kg_vs_standard.py --custom-query "Compare BERT and GPT language models" --simplified
   ```

3. **Accuracy Testing with Long Conversations**:
   ```bash
   python moya/examples/kg_vs_standard.py --accuracy --turns 8
   ```

4. **Preseeding Knowledge Graph**:
   ```bash
   python moya/examples/kg_vs_standard.py --seed-only
   ```

## Results Storage

All benchmark results are saved as JSON files in the `benchmark_results` directory. Visualizations (if enabled) are saved in `kg_comparison_results/visualizations/`. 