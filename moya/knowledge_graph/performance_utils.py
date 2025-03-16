"""
Performance Optimization Utilities for Knowledge Graph

This module provides utilities and tools for monitoring and optimizing 
the performance of the knowledge graph components.
"""

import time
import logging
import gc
import os
import json
from typing import Dict, Any, List, Callable, Optional, Set, Tuple
from pathlib import Path
import threading
import networkx as nx

# Set up logging
logger = logging.getLogger(__name__)

class PerformanceMonitor:
    """
    Monitor and record performance metrics for knowledge graph operations.
    
    This class allows tracking execution time, memory usage, and other metrics
    for different knowledge graph operations to identify bottlenecks.
    """
    
    def __init__(self, enabled: bool = True, report_directory: str = "knowledge_graphs/performance"):
        """
        Initialize the performance monitor.
        
        Args:
            enabled: Whether monitoring is enabled
            report_directory: Directory to store performance reports
        """
        self.enabled = enabled
        self.report_directory = Path(report_directory)
        self.metrics = {
            "operations": [],
            "timings": {},
            "counts": {},
            "averages": {}
        }
        self.start_time = time.time()
        
        # Create report directory if needed
        if self.enabled:
            self.report_directory.mkdir(parents=True, exist_ok=True)
            
    def time_operation(self, operation_name: str) -> Callable:
        """
        Decorator to time an operation.
        
        Args:
            operation_name: Name of the operation to time
            
        Returns:
            Decorated function that records timing information
        """
        def decorator(func):
            def wrapper(*args, **kwargs):
                if not self.enabled:
                    return func(*args, **kwargs)
                
                # Record start time
                start = time.time()
                
                # Execute the function
                result = func(*args, **kwargs)
                
                # Record end time
                end = time.time()
                duration = end - start
                
                # Update metrics
                if operation_name not in self.metrics["operations"]:
                    self.metrics["operations"].append(operation_name)
                    self.metrics["timings"][operation_name] = []
                    self.metrics["counts"][operation_name] = 0
                
                self.metrics["timings"][operation_name].append(duration)
                self.metrics["counts"][operation_name] += 1
                
                # Calculate average
                self.metrics["averages"][operation_name] = sum(self.metrics["timings"][operation_name]) / len(self.metrics["timings"][operation_name])
                
                # Log if operation is slow
                if duration > 1.0:  # More than 1 second
                    logger.warning(f"Slow operation: {operation_name} took {duration:.4f}s")
                
                return result
            return wrapper
        return decorator
    
    def reset(self):
        """Reset all metrics."""
        self.metrics = {
            "operations": [],
            "timings": {},
            "counts": {},
            "averages": {}
        }
        self.start_time = time.time()
    
    def get_report(self) -> Dict[str, Any]:
        """
        Get a report of current metrics.
        
        Returns:
            Dictionary containing performance metrics
        """
        report = {
            "total_runtime": time.time() - self.start_time,
            "operations": {}
        }
        
        for op in self.metrics["operations"]:
            report["operations"][op] = {
                "count": self.metrics["counts"][op],
                "average_time": self.metrics["averages"][op],
                "total_time": sum(self.metrics["timings"][op]),
                "max_time": max(self.metrics["timings"][op]) if self.metrics["timings"][op] else 0,
                "min_time": min(self.metrics["timings"][op]) if self.metrics["timings"][op] else 0,
            }
        
        return report
    
    def save_report(self, filename: Optional[str] = None) -> str:
        """
        Save performance report to a file.
        
        Args:
            filename: Optional filename for the report
            
        Returns:
            Path to the saved report file
        """
        if not self.enabled:
            return ""
            
        # Generate filename if not provided
        if not filename:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filename = f"performance_report_{timestamp}.json"
        
        # Get report data
        report = self.get_report()
        
        # Save to file
        report_path = self.report_directory / filename
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
            
        logger.info(f"Performance report saved to {report_path}")
        return str(report_path)
    
    def log_summary(self):
        """Log a summary of performance metrics."""
        if not self.enabled:
            return
            
        report = self.get_report()
        total_runtime = report["total_runtime"]
        
        logger.info("=" * 40)
        logger.info(f"Performance Summary - Total Runtime: {total_runtime:.2f}s")
        logger.info("-" * 40)
        
        # Sort operations by total time
        sorted_ops = sorted(
            report["operations"].items(),
            key=lambda x: x[1]["total_time"],
            reverse=True
        )
        
        for op_name, op_data in sorted_ops:
            percent = (op_data["total_time"] / total_runtime) * 100 if total_runtime > 0 else 0
            logger.info(f"{op_name}: {op_data['count']} calls, {op_data['total_time']:.4f}s total ({percent:.1f}%), {op_data['average_time']:.4f}s avg")
        
        logger.info("=" * 40)


class GraphOptimizer:
    """
    Utilities for optimizing knowledge graph performance.
    
    This class provides methods to optimize graph operations, caching,
    and memory usage for improved performance.
    """
    
    def __init__(self, kg_service, cache_size: int = 100):
        """
        Initialize the graph optimizer.
        
        Args:
            kg_service: KnowledgeGraphService instance
            cache_size: Maximum number of items to keep in cache
        """
        self.kg_service = kg_service
        self.cache_size = cache_size
        self.entity_cache = {}  # thread_id -> {entity -> neighbors}
        self.related_info_cache = {}  # thread_id + entities_key -> related_info
        self.cache_hits = 0
        self.cache_misses = 0
        self.last_prune_time = time.time()
        self.prune_interval = 3600  # 1 hour
        
        # Start background pruning thread
        self._start_pruning_thread()
    
    def optimize_graph(self, thread_id: str, aggressive: bool = False) -> int:
        """
        Optimize a graph for better performance.
        
        Args:
            thread_id: The thread ID for the graph to optimize
            aggressive: Whether to use aggressive optimization
            
        Returns:
            Number of nodes reduced in the graph
        """
        graph = self.kg_service.get_or_create_graph(thread_id)
        original_size = len(graph)
        
        # Prune the graph first
        if aggressive:
            # More aggressive pruning for large graphs
            self.kg_service.prune_graph(
                thread_id, 
                max_nodes=self.kg_service.max_nodes_per_graph // 2,
                age_threshold=self.kg_service.default_prune_age // 2
            )
        else:
            # Normal pruning
            self.kg_service.prune_graph(thread_id)
        
        # Get the graph again after pruning
        graph = self.kg_service.get_or_create_graph(thread_id)
        
        # Find and collapse duplicate nodes (nodes that have identical relationships)
        duplicates = self._find_duplicate_nodes(graph)
        for nodes_to_merge in duplicates:
            self._merge_nodes(graph, nodes_to_merge)
        
        # Clear caches for this thread
        self._clear_thread_cache(thread_id)
        
        # Save the optimized graph
        self.kg_service._save_graph(thread_id)
        
        return original_size - len(graph)
    
    def clear_cache(self):
        """Clear all caches."""
        self.entity_cache.clear()
        self.related_info_cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0
    
    def _clear_thread_cache(self, thread_id: str):
        """Clear cache for a specific thread."""
        if thread_id in self.entity_cache:
            self.entity_cache.pop(thread_id)
        
        # Clear related info cache entries for this thread
        keys_to_remove = []
        for key in self.related_info_cache:
            if key.startswith(thread_id + ":"):
                keys_to_remove.append(key)
                
        for key in keys_to_remove:
            self.related_info_cache.pop(key)
    
    def _find_duplicate_nodes(self, graph: nx.DiGraph) -> List[Set[str]]:
        """Find duplicate nodes in the graph."""
        node_signatures = {}
        
        for node in graph.nodes():
            # Create a signature based on outgoing and incoming edges
            outgoing = frozenset((node, target, tuple(sorted(data.get("relations", {}).keys()))) 
                                for target, data in graph[node].items())
            
            incoming = frozenset((source, node, tuple(sorted(graph[source][node].get("relations", {}).keys()))) 
                                for source in graph.predecessors(node))
            
            signature = (outgoing, incoming)
            
            if signature not in node_signatures:
                node_signatures[signature] = set()
            
            node_signatures[signature].add(node)
        
        # Return sets of duplicate nodes (more than 1 node with same signature)
        return [nodes for nodes in node_signatures.values() if len(nodes) > 1]
    
    def _merge_nodes(self, graph: nx.DiGraph, nodes: Set[str]):
        """Merge duplicate nodes in the graph."""
        if not nodes or len(nodes) <= 1:
            return
            
        # Pick the oldest node as the canonical one
        canonical = min(nodes, key=lambda n: graph.nodes[n].get("last_updated", float('inf')))
        nodes_to_merge = nodes - {canonical}
        
        for node in nodes_to_merge:
            # Redirect incoming edges
            for pred in list(graph.predecessors(node)):
                edge_data = graph.get_edge_data(pred, node)
                
                # Skip if edge already exists to avoid overwriting
                if not graph.has_edge(pred, canonical):
                    graph.add_edge(pred, canonical, **edge_data)
                else:
                    # Merge relations if both edges exist
                    canonical_data = graph.get_edge_data(pred, canonical)
                    canonical_relations = canonical_data.get("relations", {})
                    incoming_relations = edge_data.get("relations", {})
                    
                    # Merge the relations dictionaries
                    for rel, rel_data in incoming_relations.items():
                        if rel in canonical_relations:
                            # Update existing relation
                            canonical_relations[rel]["count"] += rel_data.get("count", 1)
                            canonical_relations[rel]["confidence"] = max(
                                canonical_relations[rel].get("confidence", 0), 
                                rel_data.get("confidence", 0)
                            )
                            canonical_relations[rel]["last_updated"] = max(
                                canonical_relations[rel].get("last_updated", 0),
                                rel_data.get("last_updated", 0)
                            )
                        else:
                            # Add new relation
                            canonical_relations[rel] = rel_data
                    
                    # Update the edge with merged relations
                    graph.add_edge(pred, canonical, 
                                   relations=canonical_relations,
                                   last_updated=max(
                                       canonical_data.get("last_updated", 0),
                                       edge_data.get("last_updated", 0)
                                   ),
                                   metadata=canonical_data.get("metadata", {})
                                  )
            
            # Redirect outgoing edges
            for succ in list(graph.successors(node)):
                edge_data = graph.get_edge_data(node, succ)
                
                # Skip if edge already exists to avoid overwriting
                if not graph.has_edge(canonical, succ):
                    graph.add_edge(canonical, succ, **edge_data)
                else:
                    # Merge relations if both edges exist
                    canonical_data = graph.get_edge_data(canonical, succ)
                    canonical_relations = canonical_data.get("relations", {})
                    outgoing_relations = edge_data.get("relations", {})
                    
                    # Merge the relations dictionaries
                    for rel, rel_data in outgoing_relations.items():
                        if rel in canonical_relations:
                            # Update existing relation
                            canonical_relations[rel]["count"] += rel_data.get("count", 1)
                            canonical_relations[rel]["confidence"] = max(
                                canonical_relations[rel].get("confidence", 0), 
                                rel_data.get("confidence", 0)
                            )
                            canonical_relations[rel]["last_updated"] = max(
                                canonical_relations[rel].get("last_updated", 0),
                                rel_data.get("last_updated", 0)
                            )
                        else:
                            # Add new relation
                            canonical_relations[rel] = rel_data
                    
                    # Update the edge with merged relations
                    graph.add_edge(canonical, succ, 
                                   relations=canonical_relations,
                                   last_updated=max(
                                       canonical_data.get("last_updated", 0),
                                       edge_data.get("last_updated", 0)
                                   ),
                                   metadata=canonical_data.get("metadata", {})
                                  )
            
            # Remove the duplicate node
            graph.remove_node(node)
    
    def _start_pruning_thread(self):
        """Start a background thread for periodic cache pruning."""
        def _prune_cache_thread():
            while True:
                time.sleep(60)  # Check every minute
                now = time.time()
                
                # Prune cache if interval elapsed
                if now - self.last_prune_time > self.prune_interval:
                    self._prune_caches()
                    self.last_prune_time = now
                    
                    # Run garbage collection
                    gc.collect()
        
        # Start thread as daemon so it doesn't block program exit
        thread = threading.Thread(target=_prune_cache_thread, daemon=True)
        thread.start()
    
    def _prune_caches(self):
        """Prune caches to maintain size limit."""
        # Prune entity cache
        for thread_id in list(self.entity_cache.keys()):
            if len(self.entity_cache[thread_id]) > self.cache_size:
                # Sort by access time and keep most recent
                sorted_items = sorted(
                    self.entity_cache[thread_id].items(),
                    key=lambda x: x[1].get("last_access", 0),
                    reverse=True
                )
                
                # Keep only cache_size items
                self.entity_cache[thread_id] = dict(sorted_items[:self.cache_size])
        
        # Prune related info cache
        if len(self.related_info_cache) > self.cache_size:
            # Sort by access time and keep most recent
            sorted_items = sorted(
                self.related_info_cache.items(),
                key=lambda x: x[1].get("last_access", 0),
                reverse=True
            )
            
            # Keep only cache_size items
            self.related_info_cache = dict(sorted_items[:self.cache_size])
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        entity_cache_size = sum(len(cache) for cache in self.entity_cache.values())
        related_info_cache_size = len(self.related_info_cache)
        
        total_hits = self.cache_hits
        total_misses = self.cache_misses
        hit_ratio = self.cache_hits / (self.cache_hits + self.cache_misses) if (self.cache_hits + self.cache_misses) > 0 else 0
        
        return {
            "entity_cache_size": entity_cache_size,
            "related_info_cache_size": related_info_cache_size,
            "total_cache_size": entity_cache_size + related_info_cache_size,
            "cache_hits": total_hits,
            "cache_misses": total_misses,
            "hit_ratio": hit_ratio,
            "max_cache_size": self.cache_size
        }


def optimize_kg_service(original_kg_service):
    """
    Wrap a KnowledgeGraphService with performance optimizations.
    
    Args:
        original_kg_service: Original KnowledgeGraphService instance
        
    Returns:
        Optimized KnowledgeGraphService
    """
    # Create performance monitor and optimizer
    monitor = PerformanceMonitor()
    optimizer = GraphOptimizer(original_kg_service)
    
    # Monkey patch key methods with optimized versions
    original_get_related_information = original_kg_service.get_related_information
    original_add_triplets = original_kg_service.add_triplets
    original_prune_graph = original_kg_service.prune_graph
    
    @monitor.time_operation("get_related_information")
    def optimized_get_related_information(self, entities, thread_id, max_distance=2, 
                                        max_results=20, min_confidence=0.5):
        # Check if we have this query in cache
        cache_key = f"{thread_id}:{','.join(sorted(entities))}"
        if cache_key in optimizer.related_info_cache:
            # Update access time and return cached result
            optimizer.related_info_cache[cache_key]["last_access"] = time.time()
            optimizer.cache_hits += 1
            return optimizer.related_info_cache[cache_key]["result"]
        
        # Not in cache, call original method
        optimizer.cache_misses += 1
        result = original_get_related_information(entities, thread_id, max_distance, 
                                                max_results, min_confidence)
        
        # Cache the result
        optimizer.related_info_cache[cache_key] = {
            "result": result,
            "last_access": time.time()
        }
        
        return result
    
    @monitor.time_operation("add_triplets")
    def optimized_add_triplets(self, thread_id, triplets, confidence=1.0, metadata=None):
        # Clear caches for this thread as the graph is changing
        optimizer._clear_thread_cache(thread_id)
        
        # Call original method
        result = original_add_triplets(thread_id, triplets, confidence, metadata)
        
        # If the graph is getting large, schedule optimization
        graph = self.get_or_create_graph(thread_id)
        if len(graph) > self.max_nodes_per_graph * 0.8:  # 80% of max size
            logger.info(f"Graph {thread_id} approaching size limit ({len(graph)}). Scheduling optimization.")
            optimizer.optimize_graph(thread_id)
        
        return result
    
    @monitor.time_operation("prune_graph")
    def optimized_prune_graph(self, thread_id, max_nodes=None, age_threshold=None):
        # Clear caches for this thread
        optimizer._clear_thread_cache(thread_id)
        
        # Call original method
        return original_prune_graph(thread_id, max_nodes, age_threshold)
    
    # Replace the methods with optimized versions
    original_kg_service.get_related_information = lambda entities, thread_id, max_distance=2, max_results=20, min_confidence=0.5: optimized_get_related_information(original_kg_service, entities, thread_id, max_distance, max_results, min_confidence)
    original_kg_service.add_triplets = lambda thread_id, triplets, confidence=1.0, metadata=None: optimized_add_triplets(original_kg_service, thread_id, triplets, confidence, metadata)
    original_kg_service.prune_graph = lambda thread_id, max_nodes=None, age_threshold=None: optimized_prune_graph(original_kg_service, thread_id, max_nodes, age_threshold)
    
    # Add performance utilities to the service
    original_kg_service.performance_monitor = monitor
    original_kg_service.optimizer = optimizer
    
    # Add utility methods
    original_kg_service.get_performance_report = monitor.get_report
    original_kg_service.save_performance_report = monitor.save_report
    original_kg_service.log_performance_summary = monitor.log_summary
    original_kg_service.get_cache_stats = optimizer.get_cache_stats
    original_kg_service.clear_cache = optimizer.clear_cache
    original_kg_service.optimize_graph = optimizer.optimize_graph
    
    return original_kg_service 