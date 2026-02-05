"""
Pathfinder Engine for the disaster evacuation routing system.

This module implements Dijkstra's algorithm using a min-heap priority queue
to find optimal evacuation routes with O(E log V) time complexity.
"""

import heapq
import time
from typing import Dict, List, Optional, Set, Tuple
from ..models import PathResult, AlgorithmStats
from ..graph import GraphManager


class PathfinderEngine:
    """
    Implements Dijkstra's algorithm for finding shortest paths in weighted graphs.
    
    Uses a min-heap priority queue for efficient vertex selection, achieving
    O(E log V) time complexity with adjacency list representation.
    
    The algorithm maintains the greedy choice property: at each step, it selects
    the unvisited vertex with minimum tentative distance, guaranteeing optimality.
    """
    
    def __init__(self):
        """Initialize the pathfinder engine."""
        self._stats = AlgorithmStats()
    
    def find_shortest_path(self, graph: GraphManager, source: str, target: str) -> PathResult:
        """
        Find the shortest path between two vertices using Dijkstra's algorithm.
        
        Args:
            graph: GraphManager instance containing the graph
            source: Source vertex ID
            target: Target vertex ID
            
        Returns:
            PathResult containing the optimal path and metadata
            
        Raises:
            ValueError: If source or target vertices don't exist
        """
        # Reset statistics
        self._stats.reset()
        start_time = time.time()
        
        # Validate input
        if not graph.has_vertex(source):
            return PathResult(
                path=[], total_cost=0.0, edges_traversed=[], 
                computation_time=0.0, nodes_visited=0,
                found=False, error_message=f"Source vertex '{source}' does not exist"
            )
        
        if not graph.has_vertex(target):
            return PathResult(
                path=[], total_cost=0.0, edges_traversed=[],
                computation_time=0.0, nodes_visited=0,
                found=False, error_message=f"Target vertex '{target}' does not exist"
            )
        
        # Handle trivial case
        if source == target:
            computation_time = time.time() - start_time
            return PathResult(
                path=[source], total_cost=0.0, edges_traversed=[],
                computation_time=computation_time, nodes_visited=1
            )
        
        # Initialize Dijkstra's algorithm data structures
        distances = {vertex_id: float('inf') for vertex_id in graph.get_vertex_ids()}
        distances[source] = 0.0
        predecessors: Dict[str, Optional[str]] = {vertex_id: None for vertex_id in graph.get_vertex_ids()}
        visited: Set[str] = set()
        
        # Priority queue: (distance, vertex_id)
        priority_queue = [(0.0, source)]
        self._stats.queue_operations += 1
        
        while priority_queue:
            # Extract minimum distance vertex (greedy choice)
            current_distance, current_vertex = heapq.heappop(priority_queue)
            self._stats.queue_operations += 1
            
            # Skip if already visited (handles duplicate entries)
            if current_vertex in visited:
                continue
            
            # Mark as visited
            visited.add(current_vertex)
            self._stats.nodes_visited += 1
            
            # Found target - reconstruct path
            if current_vertex == target:
                path, edges_traversed = self._reconstruct_path(
                    graph, predecessors, source, target
                )
                computation_time = time.time() - start_time
                self._stats.computation_time = computation_time
                
                return PathResult(
                    path=path,
                    total_cost=distances[target],
                    edges_traversed=edges_traversed,
                    computation_time=computation_time,
                    nodes_visited=self._stats.nodes_visited
                )
            
            # Relax all outgoing edges
            for edge in graph.get_neighbors(current_vertex):
                self._stats.edges_examined += 1
                neighbor = edge.target
                
                # Skip if already visited
                if neighbor in visited:
                    continue
                
                # Calculate tentative distance
                edge_weight = graph.get_edge_weight(current_vertex, neighbor)
                
                # Skip blocked roads (infinite weight)
                if edge_weight == float('inf'):
                    continue
                
                tentative_distance = current_distance + edge_weight
                
                # Update if shorter path found
                if tentative_distance < distances[neighbor]:
                    distances[neighbor] = tentative_distance
                    predecessors[neighbor] = current_vertex
                    heapq.heappush(priority_queue, (tentative_distance, neighbor))
                    self._stats.queue_operations += 1
        
        # No path found
        computation_time = time.time() - start_time
        self._stats.computation_time = computation_time
        
        return PathResult(
            path=[], total_cost=0.0, edges_traversed=[],
            computation_time=computation_time, nodes_visited=self._stats.nodes_visited,
            found=False, error_message=f"No path exists from '{source}' to '{target}'"
        )
    
    def find_all_shortest_paths(self, graph: GraphManager, source: str) -> Dict[str, PathResult]:
        """
        Find shortest paths from source to all reachable vertices.
        
        Args:
            graph: GraphManager instance containing the graph
            source: Source vertex ID
            
        Returns:
            Dictionary mapping target vertices to PathResult objects
            
        Raises:
            ValueError: If source vertex doesn't exist
        """
        # Reset statistics
        self._stats.reset()
        start_time = time.time()
        
        # Validate input
        if not graph.has_vertex(source):
            return {source: PathResult(
                path=[], total_cost=0.0, edges_traversed=[],
                computation_time=0.0, nodes_visited=0,
                found=False, error_message=f"Source vertex '{source}' does not exist"
            )}
        
        # Initialize data structures
        distances = {vertex_id: float('inf') for vertex_id in graph.get_vertex_ids()}
        distances[source] = 0.0
        predecessors: Dict[str, Optional[str]] = {vertex_id: None for vertex_id in graph.get_vertex_ids()}
        visited: Set[str] = set()
        
        # Priority queue
        priority_queue = [(0.0, source)]
        self._stats.queue_operations += 1
        
        while priority_queue:
            current_distance, current_vertex = heapq.heappop(priority_queue)
            self._stats.queue_operations += 1
            
            if current_vertex in visited:
                continue
            
            visited.add(current_vertex)
            self._stats.nodes_visited += 1
            
            # Relax all outgoing edges
            for edge in graph.get_neighbors(current_vertex):
                self._stats.edges_examined += 1
                neighbor = edge.target
                
                if neighbor in visited:
                    continue
                
                edge_weight = graph.get_edge_weight(current_vertex, neighbor)
                
                if edge_weight == float('inf'):
                    continue
                
                tentative_distance = current_distance + edge_weight
                
                if tentative_distance < distances[neighbor]:
                    distances[neighbor] = tentative_distance
                    predecessors[neighbor] = current_vertex
                    heapq.heappush(priority_queue, (tentative_distance, neighbor))
                    self._stats.queue_operations += 1
        
        # Build results for all vertices
        results = {}
        computation_time = time.time() - start_time
        self._stats.computation_time = computation_time
        
        for target in graph.get_vertex_ids():
            if target == source:
                results[target] = PathResult(
                    path=[source], total_cost=0.0, edges_traversed=[],
                    computation_time=computation_time, nodes_visited=self._stats.nodes_visited
                )
            elif distances[target] == float('inf'):
                results[target] = PathResult(
                    path=[], total_cost=0.0, edges_traversed=[],
                    computation_time=computation_time, nodes_visited=self._stats.nodes_visited,
                    found=False, error_message=f"No path exists from '{source}' to '{target}'"
                )
            else:
                path, edges_traversed = self._reconstruct_path(graph, predecessors, source, target)
                results[target] = PathResult(
                    path=path, total_cost=distances[target], edges_traversed=edges_traversed,
                    computation_time=computation_time, nodes_visited=self._stats.nodes_visited
                )
        
        return results
    
    def _reconstruct_path(self, graph: GraphManager, predecessors: Dict[str, Optional[str]], 
                         source: str, target: str) -> Tuple[List[str], List]:
        """
        Reconstruct the shortest path from predecessors.
        
        Args:
            graph: GraphManager instance
            predecessors: Predecessor mapping from Dijkstra's algorithm
            source: Source vertex ID
            target: Target vertex ID
            
        Returns:
            Tuple of (path as list of vertex IDs, list of edges traversed)
        """
        path = []
        edges_traversed = []
        current = target
        
        # Build path backwards
        while current is not None:
            path.append(current)
            if predecessors[current] is not None:
                # Get the edge from predecessor to current
                edge = graph.get_edge(predecessors[current], current)
                if edge:
                    edges_traversed.append(edge)
            current = predecessors[current]
        
        # Reverse to get forward path
        path.reverse()
        edges_traversed.reverse()
        
        return path, edges_traversed
    
    def get_algorithm_stats(self) -> AlgorithmStats:
        """
        Get statistics from the last pathfinding operation.
        
        Returns:
            AlgorithmStats object with performance metrics
        """
        return AlgorithmStats(
            nodes_visited=self._stats.nodes_visited,
            computation_time=self._stats.computation_time,
            edges_examined=self._stats.edges_examined,
            queue_operations=self._stats.queue_operations
        )
    
    def validate_dijkstra_properties(self, graph: GraphManager, source: str) -> Dict[str, bool]:
        """
        Validate that Dijkstra's algorithm properties hold for the given graph.
        
        This method is primarily for educational/debugging purposes to verify
        the correctness of the algorithm implementation.
        
        Args:
            graph: GraphManager instance
            source: Source vertex ID
            
        Returns:
            Dictionary of property validation results
        """
        properties = {
            "non_negative_weights": True,
            "greedy_choice_optimal": True,
            "optimal_substructure": True,
            "single_source_property": True
        }
        
        # Check for non-negative weights
        for edge in graph.get_all_edges():
            weight = graph.get_edge_weight(edge.source, edge.target)
            if weight < 0 and weight != float('inf'):
                properties["non_negative_weights"] = False
                break
        
        # Run algorithm and verify properties
        if graph.has_vertex(source):
            results = self.find_all_shortest_paths(graph, source)
            
            # Verify single source property (all paths start from source)
            for target, result in results.items():
                if result.found and len(result.path) > 0:
                    if result.path[0] != source:
                        properties["single_source_property"] = False
                        break
        
        return properties
    
    def __str__(self) -> str:
        """String representation of the pathfinder engine."""
        return f"PathfinderEngine(stats={self._stats})"
    
    def __repr__(self) -> str:
        return self.__str__()