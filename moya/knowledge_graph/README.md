# Knowledge Graph for Multi-Agent Framework

This module provides knowledge graph capabilities for the multi-agent framework, enabling context-aware conversations and persistent memory across interactions.

## Overview

The knowledge graph module extracts entities and relationships from conversations and stores them in a structured graph, allowing the multi-agent system to:

1. Remember key entities and their relationships
2. Enrich responses with relevant contextual information
3. Discover implicit connections between topics
4. Maintain conversation context across multiple turns

## Core Components

### KnowledgeGraphService

The core service that manages storage, retrieval, and querying of knowledge graphs.

Features:
- Thread-specific knowledge graphs (one graph per conversation)
- Relationship-based entity storage
- Multi-hop information retrieval
- Confidence scoring for relationships
- Automatic graph pruning for memory management
- Persistent storage of graphs across sessions

### Entity Extraction

Responsible for identifying entities and relationships in text.

Features:
- Multiple extraction methods including LLM-based extraction
- Relationship triplet extraction (subject → relation → object)
- Main entity identification
- Enhanced entity normalization (removing common variations)
- Coreference resolution for improved context tracking

### Knowledge Graph Orchestrator

Integrates the knowledge graph with the AFP orchestrator.

Features:
- Transparent integration with existing orchestrators
- Automatic context enrichment from knowledge graph
- Relationship extraction from conversation
- Thread management for multi-user deployments

### Performance Optimization

Utilities for optimizing knowledge graph performance.

Features:
- Performance monitoring and reporting
- Method timing and bottleneck identification
- Intelligent caching for repeated queries
- Graph optimization through duplicate node merging
- Background thread for cache maintenance

## Installation

No additional installation is required beyond the base requirements for the multi-agent framework. The knowledge graph components are bundled with the main package.

## Usage

### Basic Usage

```python
from moya.knowledge_graph.graph_service import KnowledgeGraphService
from moya.knowledge_graph.entity_extraction import EntityExtractor
from moya.knowledge_graph.orchestrator import KnowledgeGraphOrchestrator
from moya.afp.orchestrator import Orchestrator as AFPOrchestrator

# Create the knowledge graph service
kg_service = KnowledgeGraphService(storage_dir="knowledge_graphs")

# Create the entity extractor
entity_extractor = EntityExtractor(
    api_key="your_api_key",
    endpoint="your_endpoint",
    api_version="your_api_version",
    deployment_name="gpt-4"
)

# Create the standard AFP orchestrator
afp_orchestrator = AFPOrchestrator(agents=[...])

# Wrap with knowledge graph orchestrator
kg_orchestrator = KnowledgeGraphOrchestrator(
    orchestrator=afp_orchestrator,
    knowledge_graph_service=kg_service,
    entity_extractor=entity_extractor,
    enrich_context=True
)

# Use the knowledge graph orchestrator
response = kg_orchestrator.generate(
    "Tell me about Microsoft Azure services",
    context=conversation_context,
    thread_id="user_123"
)
```

### Performance Optimized Usage

```python
from moya.knowledge_graph.graph_service import KnowledgeGraphService
from moya.knowledge_graph.enhanced_extraction import create_enhanced_extractor
from moya.knowledge_graph.orchestrator import KnowledgeGraphOrchestrator
from moya.knowledge_graph.performance_utils import optimize_kg_service
from moya.afp.orchestrator import Orchestrator as AFPOrchestrator

# Create and optimize the knowledge graph service
kg_service = KnowledgeGraphService(storage_dir="knowledge_graphs")
kg_service = optimize_kg_service(kg_service)

# Create enhanced entity extractor
entity_extractor = create_enhanced_extractor(
    api_key="your_api_key",
    endpoint="your_endpoint",
    api_version="your_api_version",
    deployment_name="gpt-4"
)

# Create the standard AFP orchestrator
afp_orchestrator = AFPOrchestrator(agents=[...])

# Wrap with knowledge graph orchestrator
kg_orchestrator = KnowledgeGraphOrchestrator(
    orchestrator=afp_orchestrator,
    knowledge_graph_service=kg_service,
    entity_extractor=entity_extractor,
    enrich_context=True
)

# Use the knowledge graph orchestrator
response = kg_orchestrator.generate(
    "Tell me about Microsoft Azure services",
    context=conversation_context,
    thread_id="user_123"
)

# Generate performance report
kg_service.log_performance_summary()
report_path = kg_service.save_performance_report()
print(f"Performance report saved to: {report_path}")

# Get cache statistics
cache_stats = kg_service.get_cache_stats()
print(f"Cache hit ratio: {cache_stats['hit_ratio']:.2%}")

# Optimize graph for a specific thread
nodes_reduced = kg_service.optimize_graph("user_123")
print(f"Optimized graph: {nodes_reduced} nodes reduced")
```

## Configuration Options

### KnowledgeGraphService

| Parameter | Description | Default |
|-----------|-------------|---------|
| storage_dir | Directory to store graph data | "knowledge_graphs" |
| max_nodes_per_graph | Maximum number of nodes per graph | 1000 |
| default_prune_age | Age threshold in seconds for pruning | 604800 (7 days) |

### EntityExtractor

| Parameter | Description | Default |
|-----------|-------------|---------|
| api_key | Azure OpenAI API key | None (required) |
| endpoint | Azure OpenAI endpoint | None (required) |
| api_version | Azure OpenAI API version | "2023-07-01-preview" |
| deployment_name | Model deployment name | "gpt-4" |

### KnowledgeGraphOrchestrator

| Parameter | Description | Default |
|-----------|-------------|---------|
| orchestrator | Base orchestrator to wrap | None (required) |
| knowledge_graph_service | KnowledgeGraphService instance | None (required) |
| entity_extractor | EntityExtractor instance | None (required) |
| enrich_context | Whether to enrich context with KG data | True |
| log_kg_operations | Whether to log KG operations | False |

## Performance Optimization

The knowledge graph module includes several performance optimization features:

### Performance Monitoring

The `PerformanceMonitor` class tracks execution times of key operations:

```python
from moya.knowledge_graph.performance_utils import PerformanceMonitor

monitor = PerformanceMonitor(report_directory="knowledge_graphs/performance")

@monitor.time_operation("my_operation")
def my_function():
    # Function code here
    pass

# Log performance summary
monitor.log_summary()

# Save performance report
report_path = monitor.save_report("my_report.json")
```

### Graph Optimization

The `GraphOptimizer` class improves graph performance:

```python
from moya.knowledge_graph.performance_utils import GraphOptimizer

optimizer = GraphOptimizer(kg_service, cache_size=200)

# Optimize a specific graph
nodes_reduced = optimizer.optimize_graph("thread_123")

# Clear caches
optimizer.clear_cache()

# Get cache statistics
stats = optimizer.get_cache_stats()
```

### Automatic Optimization

The `optimize_kg_service` function applies performance optimizations to an existing service:

```python
from moya.knowledge_graph.performance_utils import optimize_kg_service

# Apply optimizations
optimized_kg_service = optimize_kg_service(kg_service)

# Use the optimized service with added methods
optimized_kg_service.get_performance_report()
optimized_kg_service.log_performance_summary()
optimized_kg_service.get_cache_stats()
optimized_kg_service.optimize_graph("thread_123")
```

## Benchmarking

The module includes a benchmark script for evaluating performance:

```bash
# Run a benchmark with default settings
python -m moya.examples.kg_benchmark

# Run with enhanced extraction and optimization
python -m moya.examples.kg_benchmark --enhanced --optimize

# Run specific scenarios only
python -m moya.examples.kg_benchmark --scenarios 0 2
```

## Example Code

See the `moya/examples` directory for complete examples:

- `kg_multiagent_example.py`: Integration with AFP orchestrator
- `kg_benchmark.py`: Performance benchmarking

## Extension Points

The knowledge graph module is designed to be extensible:

1. **Custom Entity Extraction**: Create custom extractors by implementing the same interface
2. **Alternative Graph Backends**: Replace NetworkX with other graph databases
3. **Custom Optimizers**: Create specialized optimizers for specific use cases
4. **Hybrid Retrieval**: Combine with vector databases for hybrid search 