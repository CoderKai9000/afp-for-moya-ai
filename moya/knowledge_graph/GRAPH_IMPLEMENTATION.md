# Knowledge Graph Implementation for Multi-Agent Systems

This document provides detailed information about the knowledge graph implementation in the Moya framework, explaining the technical architecture, benefits, and design decisions.

## Technical Architecture

The knowledge graph implementation consists of three main components:

1. **Knowledge Graph Service**: Core graph database functionality using NetworkX
2. **Entity Extractor**: Multi-method entity and relationship extraction
3. **Orchestrator Integration**: Knowledge graph integration with the AFP orchestrator

### Graph Data Structure

The knowledge graph uses a directed graph (DiGraph) from NetworkX with the following structure:

- **Nodes**: Represent entities with metadata including:
  - `type`: Entity type (default: "entity")
  - `last_updated`: Timestamp of last update
  - `mentions`: Count of times this entity has been mentioned

- **Edges**: Represent relationships between entities with metadata including:
  - `relations`: Dictionary of relation types where each relation has:
    - `count`: Number of occurrences
    - `created`: Timestamp of first occurrence
    - `last_updated`: Timestamp of last update
    - `confidence`: Confidence score (0.0 to 1.0)
  - `metadata`: Optional additional context
  - `last_updated`: Timestamp of last edge update

### Serialization Format

The graph is serialized to JSON with the following structure:

```json
{
  "nodes": [
    {"id": "entity1", "data": {"type": "entity", "last_updated": 1625097600, "mentions": 3}},
    {"id": "entity2", "data": {"type": "entity", "last_updated": 1625097800, "mentions": 1}}
  ],
  "edges": [
    {
      "source": "entity1",
      "target": "entity2",
      "data": {
        "relations": {
          "owns": {"count": 2, "created": 1625097600, "last_updated": 1625097700, "confidence": 0.95}
        },
        "metadata": {"source": "user_message"},
        "last_updated": 1625097700
      }
    }
  ],
  "metadata": {
    "thread_id": "user_123",
    "saved_at": 1625097900,
    "nodes_count": 2,
    "edges_count": 1
  }
}
```

## Entity Extraction Methods

The EntityExtractor supports multiple extraction methods that can be used individually or in combination:

### Rule-Based Extraction

- Uses regular expressions to identify potential entities
- Pattern matching for common relationship structures
- Fast and deterministic, but limited in capability

### spaCy-Based Extraction

- Uses Named Entity Recognition (NER) for entity identification
- Dependency parsing for subject-verb-object extraction
- More accurate but requires additional dependencies

### LLM-Based Extraction

- Uses prompt engineering with language models
- Highly flexible and adaptable to various text styles
- Most powerful method but requires API access

## Benefits of Knowledge Graph Integration

### 1. Enhanced Context Awareness

The knowledge graph provides a structured memory system that enables:

- **Long-term Memory**: Information persists across multiple turns
- **Relationship Discovery**: Find connections between entities mentioned at different times
- **Context Enrichment**: Add relevant facts to prompts automatically

### 2. Improved Response Quality

Knowledge graph integration improves LLM responses by:

- **Reducing Hallucination**: Providing grounded facts from the graph
- **Maintaining Consistency**: Ensuring responses align with previously mentioned information
- **Enabling Personalization**: Building user-specific knowledge over time

### 3. Efficiency Improvements

- **Reduced Token Usage**: Avoiding repetition of context in every prompt
- **Targeted Context**: Only including relevant information from the knowledge graph
- **Automatic Pruning**: Removing outdated or less relevant information

## Performance Optimizations

### Graph Pruning Strategy

The knowledge graph implements an intelligent pruning strategy:

1. **Age-Based Pruning**: Removes nodes older than a configurable threshold
2. **Score-Based Pruning**: If still over maximum size, uses a scoring function:
   - Score = 0.3 * normalized_mentions + 0.7 * normalized_recency
3. **Relation Preservation**: Maintains important relationships even when pruning

### Query Optimization

Knowledge retrieval is optimized for relevance:

1. **Distance-Limited Queries**: Constrains path length for relationship discovery
2. **Confidence Filtering**: Only returns relationships above a confidence threshold
3. **Relevance Scoring**: Uses a combination of confidence and recency

### Entity Normalization

To ensure consistent entity references:

1. **Case Normalization**: Converts to lowercase for matching
2. **Whitespace Handling**: Trims excess whitespace
3. **Punctuation Removal**: Removes trailing punctuation

## Integration with AFP

The KnowledgeGraphOrchestrator wraps the AFP orchestrator and:

1. Automatically extracts entities and relationships from user queries
2. Adds them to the knowledge graph
3. Retrieves relevant information for context enrichment
4. Passes the enriched context to the base orchestrator
5. Extracts from assistant responses to complete the feedback loop

This integration is transparent to both users and agents, as all functionality is handled in the wrapper layer.

## Comparison with Alternative Approaches

### Vector Databases

While vector databases are popular for semantic search:

- **Knowledge Graphs** excel at capturing explicit relationships
- **Vector DBs** are better at finding similar content
- Our approach can be combined with vector embeddings for hybrid retrieval

### Raw Prompt Engineering

Compared to adding all context in prompts:

- Knowledge graphs provide structured, queryable information
- Enable selective retrieval of only relevant context
- Scales better with conversation length

### Traditional Databases

Compared to SQL or NoSQL databases:

- Graphs naturally represent relationships between entities
- Support multi-hop relationship discovery
- Provide better handling of evolving schemas

## Future Extensions

The knowledge graph implementation can be extended in several ways:

1. **Graph Embeddings**: Add vector representations for semantic similarity
2. **Temporal Reasoning**: Enhanced support for time-based relationships
3. **Logical Inference**: Add rules engine for automatic relationship inference
4. **Multi-Modal Integration**: Support entities from images and other media
5. **Federation**: Connect multiple knowledge graphs across different domains

## Implementation Challenges and Solutions

### Entity Coreference Resolution

Challenge: Identifying when different terms refer to the same entity.

Solution: The current implementation uses simple normalization, but future versions could add more sophisticated coreference resolution.

### Confidence Calibration

Challenge: Assigning accurate confidence scores to extracted relationships.

Solution: Multiple extraction methods with confidence voting and reinforcement through repeated mentions.

### Performance at Scale

Challenge: Maintaining performance with very large knowledge graphs.

Solution: Automatic pruning, edge filtering, and the option to use more efficient graph backends.

## Conclusion

The knowledge graph implementation provides a powerful enhancement to multi-agent systems by enabling structured memory, relationship tracking, and context enrichment. The modular design allows for easy customization and extension, while the integration with the AFP orchestrator ensures seamless operation. 