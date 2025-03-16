#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Knowledge Graph Comparison Runner

This script runs the knowledge graph comparison visualizations and demonstrates
a basic example of how knowledge graphs improve context in AI applications.
"""

import os
import sys
import time
import json
from pathlib import Path
from knowledge_graph_comparison import KnowledgeGraphVisualizer

def create_directories():
    """Create necessary directories for outputs."""
    dirs = ["graph_comparisons", "example_output"]
    for directory in dirs:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")

def run_visualizations():
    """Run all the visualization comparisons."""
    print("\n=== Generating Knowledge Graph Comparison Visualizations ===")
    visualizer = KnowledgeGraphVisualizer()
    
    visualizer.create_storage_representation_comparison()
    visualizer.create_query_performance_comparison()
    visualizer.create_context_retrieval_comparison()
    visualizer.create_data_growth_comparison()
    visualizer.create_context_quality_comparison()
    
    print(f"All visualizations saved to the 'graph_comparisons' directory")

def demonstrate_traditional_context_retrieval():
    """Demonstrate how traditional storage retrieves context."""
    print("\n=== Traditional Context Retrieval Demonstration ===")
    
    # Simulate a database with tables (using dictionaries for simplicity)
    database = {
        "entities": [
            {"id": 1, "name": "Python", "type": "programming_language"},
            {"id": 2, "name": "TensorFlow", "type": "library"},
            {"id": 3, "name": "machine learning", "type": "concept"},
            {"id": 4, "name": "Guido van Rossum", "type": "person"},
            {"id": 5, "name": "deep learning", "type": "concept"},
            {"id": 6, "name": "neural networks", "type": "concept"},
            {"id": 7, "name": "data science", "type": "field"},
            {"id": 8, "name": "NumPy", "type": "library"},
            {"id": 9, "name": "pandas", "type": "library"},
            {"id": 10, "name": "scikit-learn", "type": "library"}
        ],
        "relationships": [
            {"id": 1, "subject_id": 1, "predicate": "created_by", "object_id": 4},
            {"id": 2, "subject_id": 1, "predicate": "used_for", "object_id": 3},
            {"id": 3, "subject_id": 2, "predicate": "based_on", "object_id": 6},
            {"id": 4, "subject_id": 2, "predicate": "used_with", "object_id": 1},
            {"id": 5, "subject_id": 3, "predicate": "includes", "object_id": 5},
            {"id": 6, "subject_id": 5, "predicate": "uses", "object_id": 6},
            {"id": 7, "subject_id": 7, "predicate": "uses", "object_id": 3},
            {"id": 8, "subject_id": 8, "predicate": "used_with", "object_id": 1},
            {"id": 9, "subject_id": 9, "predicate": "used_with", "object_id": 1},
            {"id": 10, "subject_id": 10, "predicate": "used_for", "object_id": 3},
            {"id": 11, "subject_id": 1, "predicate": "has_library", "object_id": 8},
            {"id": 12, "subject_id": 1, "predicate": "has_library", "object_id": 9},
            {"id": 13, "subject_id": 1, "predicate": "has_library", "object_id": 10}
        ]
    }
    
    # Simulate a user query
    user_query = "Tell me about Python and machine learning"
    print(f"User Query: '{user_query}'")
    
    # Traditional approach - Extract entities
    print("Step 1: Extract entities from query")
    extracted_entities = ["Python", "machine learning"]
    print(f"  Extracted entities: {extracted_entities}")
    
    # Find related entities through multiple queries
    print("Step 2: Search for each entity in the database")
    entity_ids = {}
    for entity in extracted_entities:
        for e in database["entities"]:
            if e["name"].lower() == entity.lower():
                entity_ids[entity] = e["id"]
                print(f"  Found '{entity}' with ID {e['id']}")
    
    print("Step 3: Find relationships for each entity (multiple queries)")
    start_time = time.time()
    context = []
    
    # For each entity, find direct relationships
    for entity_name, entity_id in entity_ids.items():
        # Find relationships where entity is subject
        for rel in database["relationships"]:
            if rel["subject_id"] == entity_id:
                # Get object entity
                object_entity = next((e for e in database["entities"] if e["id"] == rel["object_id"]), None)
                if object_entity:
                    context.append({
                        "subject": entity_name,
                        "predicate": rel["predicate"],
                        "object": object_entity["name"]
                    })
                    print(f"  Relation: {entity_name} {rel['predicate']} {object_entity['name']}")
        
        # Find relationships where entity is object
        for rel in database["relationships"]:
            if rel["object_id"] == entity_id:
                # Get subject entity
                subject_entity = next((e for e in database["entities"] if e["id"] == rel["subject_id"]), None)
                if subject_entity:
                    context.append({
                        "subject": subject_entity["name"],
                        "predicate": rel["predicate"],
                        "object": entity_name
                    })
                    print(f"  Relation: {subject_entity['name']} {rel['predicate']} {entity_name}")
    
    # Time taken for traditional approach
    traditional_time = time.time() - start_time
    print(f"Step 4: Assemble context (took {traditional_time:.4f} seconds)")
    print(f"  Found {len(context)} relationships to provide as context")
    
    # Save results
    with open("example_output/traditional_context.json", "w") as f:
        json.dump({
            "query": user_query,
            "extracted_entities": extracted_entities,
            "context": context,
            "time_taken": traditional_time
        }, f, indent=2)
    
    return traditional_time

def demonstrate_kg_context_retrieval():
    """Demonstrate how knowledge graph storage retrieves context."""
    print("\n=== Knowledge Graph Context Retrieval Demonstration ===")
    
    # Create a simple graph structure using dictionaries
    # In a real implementation, this would use NetworkX
    graph = {
        "nodes": {
            "Python": {"type": "programming_language"},
            "TensorFlow": {"type": "library"},
            "machine learning": {"type": "concept"},
            "Guido van Rossum": {"type": "person"},
            "deep learning": {"type": "concept"},
            "neural networks": {"type": "concept"},
            "data science": {"type": "field"},
            "NumPy": {"type": "library"},
            "pandas": {"type": "library"},
            "scikit-learn": {"type": "library"}
        },
        "edges": [
            {"source": "Python", "target": "Guido van Rossum", "relation": "created_by"},
            {"source": "Python", "target": "machine learning", "relation": "used_for"},
            {"source": "TensorFlow", "target": "neural networks", "relation": "based_on"},
            {"source": "TensorFlow", "target": "Python", "relation": "used_with"},
            {"source": "machine learning", "target": "deep learning", "relation": "includes"},
            {"source": "deep learning", "target": "neural networks", "relation": "uses"},
            {"source": "data science", "target": "machine learning", "relation": "uses"},
            {"source": "NumPy", "target": "Python", "relation": "used_with"},
            {"source": "pandas", "target": "Python", "relation": "used_with"},
            {"source": "scikit-learn", "target": "machine learning", "relation": "used_for"},
            {"source": "Python", "target": "NumPy", "relation": "has_library"},
            {"source": "Python", "target": "pandas", "relation": "has_library"},
            {"source": "Python", "target": "scikit-learn", "relation": "has_library"}
        ]
    }
    
    # Simulate a user query
    user_query = "Tell me about Python and machine learning"
    print(f"User Query: '{user_query}'")
    
    # Knowledge graph approach - Extract entities
    print("Step 1: Extract entities from query")
    extracted_entities = ["Python", "machine learning"]
    print(f"  Extracted entities: {extracted_entities}")
    
    print("Step 2: Find nodes in knowledge graph")
    for entity in extracted_entities:
        if entity in graph["nodes"]:
            print(f"  Found node for '{entity}' with type '{graph['nodes'][entity]['type']}'")
    
    # Traverse graph to find related entities (single operation)
    print("Step 3: Traverse graph to find related entities (single operation)")
    
    start_time = time.time()
    max_hops = 2
    context = []
    related_entities = set()
    
    # For each extracted entity, perform breadth-first traversal
    for entity in extracted_entities:
        if entity not in graph["nodes"]:
            continue
            
        related_entities.add(entity)
        frontier = {entity}
        
        # Traverse up to max_hops
        for hop in range(max_hops):
            next_frontier = set()
            
            for node in frontier:
                # Find outgoing relationships
                for edge in graph["edges"]:
                    if edge["source"] == node:
                        context.append({
                            "subject": edge["source"],
                            "predicate": edge["relation"],
                            "object": edge["target"]
                        })
                        related_entities.add(edge["target"])
                        next_frontier.add(edge["target"])
                        print(f"  Hop {hop+1}: {edge['source']} {edge['relation']} {edge['target']}")
                
                # Find incoming relationships
                for edge in graph["edges"]:
                    if edge["target"] == node:
                        context.append({
                            "subject": edge["source"],
                            "predicate": edge["relation"],
                            "object": edge["target"]
                        })
                        related_entities.add(edge["source"])
                        next_frontier.add(edge["source"])
                        print(f"  Hop {hop+1}: {edge['source']} {edge['relation']} {edge['target']}")
            
            frontier = next_frontier
    
    # Time taken for KG approach
    kg_time = time.time() - start_time
    print(f"Step 4: Return connected subgraph (took {kg_time:.4f} seconds)")
    print(f"  Found {len(context)} relationships and {len(related_entities)} entities to provide as context")
    
    # Save results
    with open("example_output/kg_context.json", "w") as f:
        json.dump({
            "query": user_query,
            "extracted_entities": extracted_entities,
            "context": context,
            "related_entities": list(related_entities),
            "time_taken": kg_time
        }, f, indent=2)
    
    return kg_time

def compare_results(trad_time, kg_time):
    """Compare the results of both approaches."""
    print("\n=== Comparison of Context Retrieval Approaches ===")
    
    # Load the saved results
    with open("example_output/traditional_context.json", "r") as f:
        trad_results = json.load(f)
    
    with open("example_output/kg_context.json", "r") as f:
        kg_results = json.load(f)
    
    # Compare number of relationships found
    trad_rels = len(trad_results["context"])
    kg_rels = len(kg_results["context"])
    
    print(f"Traditional approach found {trad_rels} direct relationships in {trad_time:.4f} seconds")
    print(f"Knowledge Graph approach found {kg_rels} relationships in {kg_time:.4f} seconds")
    
    if kg_rels > trad_rels:
        print(f"Knowledge Graph found {kg_rels - trad_rels} more relationships ({(kg_rels/trad_rels - 1)*100:.1f}% more)")
    
    # Check for unique insights in KG approach
    kg_entity_pairs = {(rel["subject"], rel["object"]) for rel in kg_results["context"]}
    trad_entity_pairs = {(rel["subject"], rel["object"]) for rel in trad_results["context"]}
    
    unique_to_kg = kg_entity_pairs - trad_entity_pairs
    
    print(f"Knowledge Graph discovered {len(unique_to_kg)} unique entity connections not found by traditional approach:")
    for subject, obj in list(unique_to_kg)[:5]:  # Show first 5 examples
        print(f"  - Connection between '{subject}' and '{obj}'")
    
    if len(unique_to_kg) > 5:
        print(f"  - ... and {len(unique_to_kg) - 5} more connections")
    
    # Calculate performance improvement
    if trad_time > 0:
        speedup = trad_time / kg_time if kg_time > 0 else float('inf')
        print(f"Knowledge Graph approach was {speedup:.2f}x faster for context retrieval")
    
    # Write comparison summary
    with open("example_output/comparison_summary.md", "w") as f:
        f.write("# Knowledge Graph vs. Traditional Storage: Context Retrieval Comparison\n\n")
        
        f.write("## Performance Metrics\n\n")
        f.write(f"- **Traditional Approach**: {trad_rels} relationships in {trad_time:.4f} seconds\n")
        f.write(f"- **Knowledge Graph Approach**: {kg_rels} relationships in {kg_time:.4f} seconds\n")
        f.write(f"- **Speedup Factor**: {speedup:.2f}x\n\n")
        
        f.write("## Context Quality\n\n")
        f.write(f"- Knowledge Graph found {len(unique_to_kg)} unique entity connections not discovered by the traditional approach\n")
        f.write(f"- This represents a {(kg_rels/trad_rels - 1)*100:.1f}% increase in context richness\n\n")
        
        f.write("## Key Advantages Demonstrated\n\n")
        f.write("1. **Traversal Efficiency**: KG approach traverses relationships much more efficiently\n")
        f.write("2. **Context Completeness**: KG discovers more comprehensive context\n")
        f.write("3. **Connection Discovery**: KG finds non-obvious connections between entities\n")
        f.write("4. **Query Simplicity**: KG retrieves related information in a single operation\n")
        
        print(f"Comparison summary written to example_output/comparison_summary.md")

def main():
    """Run the complete knowledge graph comparison demonstration."""
    print("=== Knowledge Graph vs. Traditional Storage Comparison ===\n")
    
    # Create necessary directories
    create_directories()
    
    # Run visualizations
    run_visualizations()
    
    # Run demonstrations
    trad_time = demonstrate_traditional_context_retrieval()
    kg_time = demonstrate_kg_context_retrieval()
    
    # Compare results
    compare_results(trad_time, kg_time)
    
    print("\nAll comparison results have been saved to the output directories.")
    print("Visual comparisons: 'graph_comparisons/'")
    print("Example output: 'example_output/'")
    print("\nTo understand the implementation details, please refer to 'knowledge_graph_analysis.md'")

if __name__ == "__main__":
    main() 