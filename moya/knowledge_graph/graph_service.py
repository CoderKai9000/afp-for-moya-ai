"""
Knowledge Graph Service for Moya

This module implements a knowledge graph service for storing and retrieving
entity relationships in a graph structure. It provides methods for adding,
querying, and managing knowledge graphs for multi-agent conversations.
"""

import os
import json
import time
import logging
from pathlib import Path
import networkx as nx
from typing import List, Dict, Tuple, Set, Any, Optional, Union

# Set up logging
logger = logging.getLogger(__name__)

class KnowledgeGraphService:
    """
    A service that maintains a knowledge graph for enhancing multi-agent conversations.
    Uses NetworkX for in-memory graph representation and provides methods for
    adding, querying, and managing entity relationships.
    
    Features:
    - Persistent storage of knowledge graphs in JSON format
    - Thread-specific graph isolation
    - Entity relationship tracking
    - Context enrichment for queries
    - Automatic pruning of outdated or less relevant information
    - Path-based relationship discovery
    """
    
    def __init__(self, storage_dir: str = "knowledge_graphs", max_nodes_per_graph: int = 2000):
        """
        Initialize the knowledge graph service.
        
        Args:
            storage_dir: Directory to store serialized graphs
            max_nodes_per_graph: Maximum number of nodes to keep in each graph before pruning
        """
        self.storage_dir = Path(storage_dir)
        self.graphs = {}  # thread_id -> graph mapping
        self.max_nodes_per_graph = max_nodes_per_graph
        self.default_prune_age = 86400  # 24 hours in seconds
        
        # Create storage directory if it doesn't exist
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized KnowledgeGraphService with storage directory: {storage_dir}")
    
    def get_or_create_graph(self, thread_id: str) -> nx.DiGraph:
        """
        Get an existing graph for a thread or create a new one.
        
        Args:
            thread_id: The conversation thread ID
            
        Returns:
            The graph for the specified thread
        """
        if thread_id not in self.graphs:
            # Try to load from disk first
            graph_path = self.storage_dir / f"{thread_id}.json"
            if graph_path.exists():
                self.graphs[thread_id] = self._load_graph(graph_path)
                logger.debug(f"Loaded existing graph for thread {thread_id} with {len(self.graphs[thread_id])} nodes")
            else:
                # Create a new directed graph
                self.graphs[thread_id] = nx.DiGraph()
                logger.debug(f"Created new graph for thread {thread_id}")
        
        return self.graphs[thread_id]
    
    def add_triplet(self, thread_id: str, entity1: str, relation: str, entity2: str, 
                    confidence: float = 1.0, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Add a single entity relationship triplet to the graph.
        
        Args:
            thread_id: The conversation thread ID
            entity1: The source entity
            relation: The relationship type
            entity2: The target entity
            confidence: Confidence score for this relationship (0.0 to 1.0)
            metadata: Optional metadata to store with the relationship
        """
        # Normalize entities and relation
        entity1 = self._normalize_entity(entity1)
        entity2 = self._normalize_entity(entity2)
        relation = relation.lower().strip()
        
        if not entity1 or not entity2 or not relation:
            logger.warning(f"Invalid triplet: ({entity1}, {relation}, {entity2}). Skipping.")
            return
            
        graph = self.get_or_create_graph(thread_id)
        
        # Create timestamp for this update
        timestamp = time.time()
        
        # Add nodes if they don't exist or update existing nodes
        node_metadata = {
            "type": "entity", 
            "last_updated": timestamp,
            "mentions": 1
        }
        
        if not graph.has_node(entity1):
            graph.add_node(entity1, **node_metadata)
        else:
            # Update existing node
            graph.nodes[entity1]["last_updated"] = timestamp
            graph.nodes[entity1]["mentions"] = graph.nodes[entity1].get("mentions", 1) + 1
            
        if not graph.has_node(entity2):
            graph.add_node(entity2, **node_metadata)
        else:
            # Update existing node
            graph.nodes[entity2]["last_updated"] = timestamp
            graph.nodes[entity2]["mentions"] = graph.nodes[entity2].get("mentions", 1) + 1
        
        # Add or update the edge with relationship data
        if graph.has_edge(entity1, entity2):
            # Update existing edge
            edge_data = graph.get_edge_data(entity1, entity2)
            
            # Handle relation
            relations = edge_data.get("relations", {})
            if relation in relations:
                # Update existing relation
                relations[relation]["count"] += 1
                relations[relation]["last_updated"] = timestamp
                relations[relation]["confidence"] = max(relations[relation]["confidence"], confidence)
            else:
                # Add new relation to existing edge
                relations[relation] = {
                    "count": 1,
                    "created": timestamp,
                    "last_updated": timestamp,
                    "confidence": confidence
                }
            
            # Update edge with new relations
            graph.add_edge(
                entity1, 
                entity2, 
                relations=relations,
                metadata=metadata or edge_data.get("metadata", {}),
                last_updated=timestamp
            )
        else:
            # Add new edge
            relations = {
                relation: {
                    "count": 1,
                    "created": timestamp,
                    "last_updated": timestamp,
                    "confidence": confidence
                }
            }
            
            graph.add_edge(
                entity1, 
                entity2, 
                relations=relations,
                metadata=metadata or {},
                last_updated=timestamp
            )
    
    def add_triplets(self, thread_id: str, triplets: List[Tuple[str, str, str]], 
                     confidence: float = 1.0, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Add multiple entity relationship triplets to the graph.
        
        Args:
            thread_id: The conversation thread ID
            triplets: List of (entity1, relation, entity2) tuples
            confidence: Confidence score for these relationships (0.0 to 1.0)
            metadata: Optional metadata to store with the relationships
        """
        added_count = 0
        for entity1, relation, entity2 in triplets:
            self.add_triplet(thread_id, entity1, relation, entity2, confidence, metadata)
            added_count += 1
        
        # Save the graph after batch updates
        self._save_graph(thread_id)
        
        # Check if we need to prune the graph
        graph = self.get_or_create_graph(thread_id)
        if len(graph) > self.max_nodes_per_graph:
            logger.info(f"Graph for thread {thread_id} exceeds max nodes ({len(graph)}). Pruning...")
            self.prune_graph(thread_id)
        
        logger.debug(f"Added {added_count} triplets to thread {thread_id}")
    
    def get_related_information(self, entities: List[str], thread_id: str, 
                              max_distance: int = 2, max_results: int = 20, 
                              min_confidence: float = 0.5) -> str:
        """
        Get related information for a list of entities from the knowledge graph.
        
        Args:
            entities: List of entity names to find relationships for
            thread_id: The conversation thread ID
            max_distance: Maximum path length to consider for relationships
            max_results: Maximum number of results to return
            min_confidence: Minimum confidence threshold for including relationships
            
        Returns:
            A string containing related information formatted for context enrichment
        """
        graph = self.get_or_create_graph(thread_id)
        
        # If graph is empty or no entities provided, return empty string
        if not entities or graph.number_of_nodes() == 0:
            return ""
        
        # Normalize input entities
        normalized_entities = [self._normalize_entity(entity) for entity in entities]
        normalized_entities = [e for e in normalized_entities if e and e in graph]
        
        if not normalized_entities:
            return ""
        
        related_info = []
        found_entities = set()
        
        # For each entity, find nodes within max_distance
        for entity in normalized_entities:
            # Get immediate relationships (1-hop)
            outgoing_edges = graph.out_edges(entity, data=True)
            incoming_edges = graph.in_edges(entity, data=True)
            
            # Process outgoing relationships
            for src, tgt, edge_data in outgoing_edges:
                relations = edge_data.get("relations", {})
                for relation, rel_data in relations.items():
                    if rel_data.get("confidence", 0) >= min_confidence:
                        related_info.append((
                            f"{src} {relation} {tgt}",
                            rel_data.get("confidence", 0),
                            rel_data.get("last_updated", 0)
                        ))
                        found_entities.add(tgt)
            
            # Process incoming relationships
            for src, tgt, edge_data in incoming_edges:
                if src != entity:  # Avoid duplicating outgoing relationships
                    relations = edge_data.get("relations", {})
                    for relation, rel_data in relations.items():
                        if rel_data.get("confidence", 0) >= min_confidence:
                            related_info.append((
                                f"{src} {relation} {tgt}",
                                rel_data.get("confidence", 0),
                                rel_data.get("last_updated", 0)
                            ))
                            found_entities.add(src)
            
            # If max_distance > 1, find multi-hop paths
            if max_distance > 1:
                # For 2-hop and beyond relationships, use all_simple_paths for each entity pair
                remaining_nodes = [n for n in graph.nodes() if n != entity and n not in found_entities]
                
                for target in remaining_nodes:
                    try:
                        paths = nx.all_simple_paths(graph, entity, target, cutoff=max_distance)
                        for path in paths:
                            if len(path) > 1:  # Ensure there's at least one relationship
                                path_confidence = 1.0
                                path_info = []
                                for i in range(len(path) - 1):
                                    src, tgt = path[i], path[i+1]
                                    edge_data = graph.get_edge_data(src, tgt)
                                    relations = edge_data.get("relations", {})
                                    
                                    for relation, rel_data in relations.items():
                                        rel_confidence = rel_data.get("confidence", 0)
                                        if rel_confidence >= min_confidence:
                                            path_confidence *= rel_confidence  # Compound confidence
                                            path_info.append((
                                                f"{src} {relation} {tgt}",
                                                rel_confidence,
                                                rel_data.get("last_updated", 0)
                                            ))
                                
                                if path_info:
                                    related_info.extend(path_info)
                                    found_entities.add(target)
                    except nx.NetworkXNoPath:
                        # No path exists between these nodes
                        continue
        
        # Sort by confidence and recency, then take top results
        if related_info:
            # Sort by combined score (0.7 * confidence + 0.3 * recency)
            current_time = time.time()
            max_age = 86400 * 7  # 7 days in seconds
            
            def score_item(item):
                text, confidence, timestamp = item
                recency = max(0, 1 - (current_time - timestamp) / max_age)  # 0 to 1, higher is more recent
                return 0.7 * confidence + 0.3 * recency
            
            related_info.sort(key=score_item, reverse=True)
            
            # Limit to max_results
            related_info = related_info[:max_results]
            
            # Format as readable text
            texts = [text for text, _, _ in related_info]
            
            return "Related information:\n- " + "\n- ".join(set(texts))
        else:
            return ""
    
    def prune_graph(self, thread_id: str, max_nodes: Optional[int] = None, 
                   age_threshold: Optional[float] = None) -> int:
        """
        Prune the graph by removing old or less relevant nodes.
        
        Args:
            thread_id: The conversation thread ID
            max_nodes: Maximum number of nodes to keep (defaults to class max_nodes_per_graph)
            age_threshold: Remove nodes older than this many seconds (defaults to 24 hours)
            
        Returns:
            Number of nodes removed
        """
        graph = self.get_or_create_graph(thread_id)
        
        if max_nodes is None:
            max_nodes = self.max_nodes_per_graph
            
        if age_threshold is None:
            age_threshold = self.default_prune_age
        
        current_time = time.time()
        nodes_to_remove = []
        
        # Find old nodes
        for node, data in graph.nodes(data=True):
            last_updated = data.get("last_updated", 0)
            if current_time - last_updated > age_threshold:
                nodes_to_remove.append((node, last_updated, data.get("mentions", 1)))
        
        # If we still have too many nodes, remove the oldest ones
        if len(graph) - len(nodes_to_remove) > max_nodes:
            # Sort nodes by a score combining recency and mention count
            # Higher mentions should be kept, and more recent nodes should be kept
            
            remaining_nodes = [
                (node, data.get("last_updated", 0), data.get("mentions", 1))
                for node, data in graph.nodes(data=True)
                if node not in [n for n, _, _ in nodes_to_remove]
            ]
            
            # Calculate a score where higher is better to keep
            # Score = 0.3 * normalized_mentions + 0.7 * normalized_recency
            max_mentions = max(1, max(mentions for _, _, mentions in remaining_nodes))
            oldest_timestamp = min(timestamp for _, timestamp, _ in remaining_nodes)
            
            def node_score(node_item):
                _, timestamp, mentions = node_item
                recency = (timestamp - oldest_timestamp) / (current_time - oldest_timestamp + 1e-10)
                mention_score = mentions / max_mentions
                return 0.3 * mention_score + 0.7 * recency
            
            # Sort by score, lowest scores will be removed
            remaining_nodes.sort(key=node_score)
            
            # Calculate how many more nodes to remove
            additional_to_remove = len(graph) - len(nodes_to_remove) - max_nodes
            if additional_to_remove > 0:
                nodes_to_remove.extend(remaining_nodes[:additional_to_remove])
        
        # Remove nodes
        remove_count = 0
        for node, _, _ in nodes_to_remove:
            graph.remove_node(node)
            remove_count += 1
        
        # Save the pruned graph
        self._save_graph(thread_id)
        
        logger.info(f"Pruned {remove_count} nodes from thread {thread_id}")
        return remove_count
    
    def visualize_graph(self, thread_id: str, output_path: Optional[str] = None) -> Optional[str]:
        """
        Generate a visualization of the knowledge graph.
        
        Args:
            thread_id: The conversation thread ID
            output_path: Path to save the visualization (None for in-memory only)
            
        Returns:
            Path to the saved visualization file if output_path is provided
        """
        try:
            import matplotlib.pyplot as plt
            from networkx.drawing.nx_pydot import graphviz_layout
        except ImportError:
            logger.warning("Visualization requires matplotlib and graphviz. Install with: pip install matplotlib pydot graphviz")
            return None
            
        graph = self.get_or_create_graph(thread_id)
        
        if len(graph) == 0:
            logger.warning(f"Cannot visualize empty graph for thread {thread_id}")
            return None
            
        # Create a visualization of the graph
        plt.figure(figsize=(12, 8))
        
        # Use a more advanced layout
        try:
            pos = graphviz_layout(graph, prog="dot")
        except Exception:
            # Fall back to spring layout if graphviz not available
            pos = nx.spring_layout(graph)
        
        # Draw nodes
        nx.draw_networkx_nodes(graph, pos, node_size=700, node_color="lightblue", alpha=0.8)
        
        # Draw edges
        nx.draw_networkx_edges(graph, pos, width=1.0, alpha=0.5, arrows=True)
        
        # Draw labels
        nx.draw_networkx_labels(graph, pos, font_size=10)
        
        # Draw edge labels (simplified to use first relation only)
        edge_labels = {}
        for u, v, data in graph.edges(data=True):
            relations = data.get("relations", {})
            if relations:
                # Get the most frequently mentioned relation
                relation = max(relations.items(), key=lambda x: x[1].get("count", 0))[0]
                edge_labels[(u, v)] = relation
        
        nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=8)
        
        plt.axis("off")
        plt.title(f"Knowledge Graph for Thread: {thread_id}")
        
        # Save or display
        if output_path:
            plt.savefig(output_path, bbox_inches="tight")
            plt.close()
            return output_path
        else:
            plt.tight_layout()
            plt.show()
            return None
    
    def export_graph(self, thread_id: str, format: str = "json") -> Dict[str, Any]:
        """
        Export the graph data in a specified format.
        
        Args:
            thread_id: The conversation thread ID
            format: Export format ('json' only for now)
            
        Returns:
            Dictionary containing the exported graph data
        """
        graph = self.get_or_create_graph(thread_id)
        
        if format.lower() == "json":
            # Return the data structure we use for saving
            data = {
                "nodes": [],
                "edges": [],
                "metadata": {
                    "thread_id": thread_id,
                    "node_count": len(graph),
                    "edge_count": graph.number_of_edges(),
                    "export_time": time.time()
                }
            }
            
            for node, node_data in graph.nodes(data=True):
                data["nodes"].append({
                    "id": node,
                    "data": node_data
                })
            
            for src, tgt, edge_data in graph.edges(data=True):
                data["edges"].append({
                    "source": src,
                    "target": tgt,
                    "data": edge_data
                })
                
            return data
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def get_entity_neighbors(self, entity: str, thread_id: str, 
                           max_distance: int = 1) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get all neighbors of an entity within a certain distance.
        
        Args:
            entity: The entity to get neighbors for
            thread_id: The conversation thread ID
            max_distance: Maximum path length
            
        Returns:
            Dictionary with incoming and outgoing relationships
        """
        graph = self.get_or_create_graph(thread_id)
        entity = self._normalize_entity(entity)
        
        if not entity or entity not in graph:
            return {"incoming": [], "outgoing": []}
            
        result = {
            "incoming": [],
            "outgoing": []
        }
        
        # Get outgoing relationships
        if max_distance == 1:
            # Direct neighbors only
            for _, tgt, edge_data in graph.out_edges(entity, data=True):
                for relation, rel_data in edge_data.get("relations", {}).items():
                    result["outgoing"].append({
                        "target": tgt,
                        "relation": relation,
                        "confidence": rel_data.get("confidence", 1.0),
                        "count": rel_data.get("count", 1),
                        "last_updated": rel_data.get("last_updated", 0)
                    })
            
            # Get incoming relationships
            for src, _, edge_data in graph.in_edges(entity, data=True):
                for relation, rel_data in edge_data.get("relations", {}).items():
                    result["incoming"].append({
                        "source": src,
                        "relation": relation,
                        "confidence": rel_data.get("confidence", 1.0),
                        "count": rel_data.get("count", 1),
                        "last_updated": rel_data.get("last_updated", 0)
                    })
        else:
            # Multi-hop relationships
            # Outgoing paths
            for node in graph.nodes():
                if node == entity:
                    continue
                    
                try:
                    paths = nx.all_simple_paths(graph, entity, node, cutoff=max_distance)
                    for path in paths:
                        if len(path) > 1:  # Ensure there's at least one hop
                            path_info = {
                                "target": node,
                                "path": [],
                                "confidence": 1.0
                            }
                            
                            for i in range(len(path) - 1):
                                src, tgt = path[i], path[i+1]
                                edge_data = graph.get_edge_data(src, tgt)
                                relations = edge_data.get("relations", {})
                                
                                # Use the most confident relation
                                best_rel = max(relations.items(), key=lambda x: x[1].get("confidence", 0)) if relations else (None, {})
                                
                                if best_rel[0]:
                                    relation, rel_data = best_rel
                                    confidence = rel_data.get("confidence", 1.0)
                                    path_info["confidence"] *= confidence
                                    
                                    path_info["path"].append({
                                        "source": src,
                                        "target": tgt,
                                        "relation": relation
                                    })
                            
                            if path_info["path"]:
                                result["outgoing"].append(path_info)
                except nx.NetworkXNoPath:
                    continue
                    
            # Incoming paths (similar logic)
            for node in graph.nodes():
                if node == entity:
                    continue
                    
                try:
                    paths = nx.all_simple_paths(graph, node, entity, cutoff=max_distance)
                    for path in paths:
                        if len(path) > 1:
                            path_info = {
                                "source": node,
                                "path": [],
                                "confidence": 1.0
                            }
                            
                            for i in range(len(path) - 1):
                                src, tgt = path[i], path[i+1]
                                edge_data = graph.get_edge_data(src, tgt)
                                relations = edge_data.get("relations", {})
                                
                                # Use the most confident relation
                                best_rel = max(relations.items(), key=lambda x: x[1].get("confidence", 0)) if relations else (None, {})
                                
                                if best_rel[0]:
                                    relation, rel_data = best_rel
                                    confidence = rel_data.get("confidence", 1.0)
                                    path_info["confidence"] *= confidence
                                    
                                    path_info["path"].append({
                                        "source": src,
                                        "target": tgt,
                                        "relation": relation
                                    })
                            
                            if path_info["path"]:
                                result["incoming"].append(path_info)
                except nx.NetworkXNoPath:
                    continue
        
        # Sort by confidence
        result["incoming"].sort(key=lambda x: x.get("confidence", 0), reverse=True)
        result["outgoing"].sort(key=lambda x: x.get("confidence", 0), reverse=True)
        
        return result
    
    def _normalize_entity(self, entity: str) -> str:
        """
        Normalize an entity name for consistency.
        
        Args:
            entity: The entity name to normalize
            
        Returns:
            Normalized entity name
        """
        if not entity:
            return ""
            
        # Convert to lowercase and strip whitespace
        return entity.lower().strip()
    
    def _save_graph(self, thread_id: str) -> None:
        """
        Save the graph for a thread to disk.
        
        Args:
            thread_id: The conversation thread ID
        """
        graph = self.graphs.get(thread_id)
        if not graph:
            return
        
        graph_path = self.storage_dir / f"{thread_id}.json"
        
        # Convert NetworkX graph to a serializable format
        data = {
            "nodes": [],
            "edges": [],
            "metadata": {
                "thread_id": thread_id,
                "saved_at": time.time(),
                "nodes_count": len(graph),
                "edges_count": graph.number_of_edges()
            }
        }
        
        for node, node_data in graph.nodes(data=True):
            data["nodes"].append({
                "id": node,
                "data": node_data
            })
        
        for src, tgt, edge_data in graph.edges(data=True):
            data["edges"].append({
                "source": src,
                "target": tgt,
                "data": edge_data
            })
        
        with open(graph_path, 'w') as f:
            json.dump(data, f, indent=2)
            
        logger.debug(f"Saved graph for thread {thread_id} with {len(graph)} nodes")
    
    def _load_graph(self, graph_path: Union[str, Path]) -> nx.DiGraph:
        """
        Load a graph from disk.
        
        Args:
            graph_path: Path to the saved graph file
            
        Returns:
            The loaded graph
        """
        with open(graph_path, 'r') as f:
            data = json.load(f)
        
        graph = nx.DiGraph()
        
        # Add nodes
        for node_info in data.get("nodes", []):
            graph.add_node(node_info["id"], **node_info.get("data", {}))
        
        # Add edges
        for edge_info in data.get("edges", []):
            graph.add_edge(
                edge_info["source"],
                edge_info["target"],
                **edge_info.get("data", {})
            )
        
        logger.debug(f"Loaded graph from {graph_path} with {len(graph)} nodes")
        return graph 