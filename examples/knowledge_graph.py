import os
import json
import time
import networkx as nx
from typing import List, Dict, Tuple, Set, Any, Optional

class KnowledgeGraphService:
    """
    A service that maintains a knowledge graph for enhancing multi-agent conversations.
    Uses NetworkX for in-memory graph representation and provides methods for
    adding, querying, and managing entity relationships.
    """
    
    def __init__(self, storage_dir: str = "./graph_storage"):
        """
        Initialize the knowledge graph service.
        
        Args:
            storage_dir: Directory to store serialized graphs
        """
        self.storage_dir = storage_dir
        self.graphs = {}  # thread_id -> graph mapping
        
        # Create storage directory if it doesn't exist
        os.makedirs(storage_dir, exist_ok=True)
    
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
            graph_path = os.path.join(self.storage_dir, f"{thread_id}.json")
            if os.path.exists(graph_path):
                self.graphs[thread_id] = self._load_graph(graph_path)
            else:
                # Create a new directed graph
                self.graphs[thread_id] = nx.DiGraph()
        
        return self.graphs[thread_id]
    
    def add_triplet(self, thread_id: str, entity1: str, relation: str, entity2: str) -> None:
        """
        Add a single entity relationship triplet to the graph.
        
        Args:
            thread_id: The conversation thread ID
            entity1: The source entity
            relation: The relationship type
            entity2: The target entity
        """
        graph = self.get_or_create_graph(thread_id)
        
        # Add nodes if they don't exist
        if not graph.has_node(entity1):
            graph.add_node(entity1, type="entity", last_updated=time.time())
        else:
            graph.nodes[entity1]["last_updated"] = time.time()
            
        if not graph.has_node(entity2):
            graph.add_node(entity2, type="entity", last_updated=time.time())
        else:
            graph.nodes[entity2]["last_updated"] = time.time()
        
        # Add or update the edge
        if graph.has_edge(entity1, entity2):
            # Update existing edge
            edge_data = graph.get_edge_data(entity1, entity2)
            if relation in edge_data.get("relations", []):
                # Relation already exists, just update timestamp
                edge_data["last_updated"] = time.time()
            else:
                # Add new relation to existing edge
                relations = edge_data.get("relations", [])
                relations.append(relation)
                graph.add_edge(
                    entity1, 
                    entity2, 
                    relations=relations,
                    last_updated=time.time()
                )
        else:
            # Add new edge
            graph.add_edge(
                entity1, 
                entity2, 
                relations=[relation],
                last_updated=time.time()
            )
    
    def add_triplets(self, thread_id: str, triplets: List[Tuple[str, str, str]]) -> None:
        """
        Add multiple entity relationship triplets to the graph.
        
        Args:
            thread_id: The conversation thread ID
            triplets: List of (entity1, relation, entity2) tuples
        """
        for entity1, relation, entity2 in triplets:
            self.add_triplet(thread_id, entity1, relation, entity2)
        
        # Save the graph after batch updates
        self._save_graph(thread_id)
    
    def get_related_information(self, entities: List[str], thread_id: str, max_distance: int = 2) -> str:
        """
        Get related information for a list of entities from the knowledge graph.
        
        Args:
            entities: List of entity names to find relationships for
            thread_id: The conversation thread ID
            max_distance: Maximum path length to consider for relationships
            
        Returns:
            A string containing related information formatted for context enrichment
        """
        graph = self.get_or_create_graph(thread_id)
        
        # If graph is empty or no entities provided, return empty string
        if not entities or graph.number_of_nodes() == 0:
            return ""
        
        related_info = []
        found_entities = set()
        
        # For each entity, find nodes within max_distance
        for entity in entities:
            if entity in graph:
                # Find neighbors within max_distance
                for target in graph.nodes():
                    if target != entity and target not in found_entities:
                        paths = nx.all_simple_paths(graph, entity, target, cutoff=max_distance)
                        for path in paths:
                            if len(path) > 1:  # Ensure there's at least one relationship
                                path_info = []
                                for i in range(len(path) - 1):
                                    src, tgt = path[i], path[i+1]
                                    edge_data = graph.get_edge_data(src, tgt)
                                    relations = edge_data.get("relations", [])
                                    for relation in relations:
                                        path_info.append(f"{src} {relation} {tgt}")
                                
                                if path_info:
                                    related_info.extend(path_info)
                                    found_entities.add(target)
        
        # Format related information into a readable string
        if related_info:
            return "Related information:\n- " + "\n- ".join(set(related_info))
        else:
            return ""
    
    def prune_graph(self, thread_id: str, max_nodes: int = 1000, age_threshold: float = 86400) -> None:
        """
        Prune the graph by removing old or less relevant nodes.
        
        Args:
            thread_id: The conversation thread ID
            max_nodes: Maximum number of nodes to keep
            age_threshold: Remove nodes older than this many seconds
        """
        graph = self.get_or_create_graph(thread_id)
        
        current_time = time.time()
        nodes_to_remove = []
        
        # Find old nodes
        for node, data in graph.nodes(data=True):
            last_updated = data.get("last_updated", 0)
            if current_time - last_updated > age_threshold:
                nodes_to_remove.append((node, last_updated))
        
        # If we still have too many nodes, remove the oldest ones
        if len(graph) - len(nodes_to_remove) > max_nodes:
            # Sort remaining nodes by age and mark oldest for removal
            remaining_nodes = [
                (node, data.get("last_updated", 0))
                for node, data in graph.nodes(data=True)
                if node not in [n for n, _ in nodes_to_remove]
            ]
            remaining_nodes.sort(key=lambda x: x[1])  # Sort by last_updated time
            
            # Calculate how many more nodes to remove
            additional_to_remove = len(graph) - len(nodes_to_remove) - max_nodes
            if additional_to_remove > 0:
                nodes_to_remove.extend(remaining_nodes[:additional_to_remove])
        
        # Remove nodes
        for node, _ in nodes_to_remove:
            graph.remove_node(node)
        
        # Save the pruned graph
        self._save_graph(thread_id)
    
    def _save_graph(self, thread_id: str) -> None:
        """
        Save the graph for a thread to disk.
        
        Args:
            thread_id: The conversation thread ID
        """
        graph = self.graphs.get(thread_id)
        if not graph:
            return
        
        graph_path = os.path.join(self.storage_dir, f"{thread_id}.json")
        
        # Convert NetworkX graph to a serializable format
        data = {
            "nodes": [],
            "edges": []
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
    
    def _load_graph(self, graph_path: str) -> nx.DiGraph:
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
        
        return graph 