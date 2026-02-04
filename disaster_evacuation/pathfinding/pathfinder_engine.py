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
    the unvisited vertex with minimum tentative distance, which is guaranteed
    to be optimal due to non-negative edge weights.
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
        
        # Special case: source equals target
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
        heapq.heapify(priority_queue)
        
        # Dijkstra's main loop
        while priority_queue:
            self._stats.queue_operations += 1
            current_distance, current_vertex = heapq.heappop(priority_queue)
            
            # Skip if already visited (handles duplicate entries in queue)
            if current_vertex in visited:
                continue
            
            # Mark as visited
            visited.add(current_vertex)
            self._stats.nodes_visited += 1
            
            # Early termination if target reached
            if current_vertex == target:
                break
            
            # Skip if current distance is outdated (relaxation already found better path)
            if current_distance > distances[current_vertex]:
                continue
            
            # Examine all neighbors
            neighbors = graph.get_neighbors(current_vertex)
            for edge in neighbors:
                self._stats.edges_examined += 1
                neighbor = edge.target
                
                # Skip blocked edges (infinite weight)
                if edge.is_blocked:
                    continue
                
                # Calculate tentative distance
                edge_weight = graph.get_edge_weight(current_vertex, neighbor)
                tentative_distance = distances[current_vertex] + edge_weight
                
                # Relaxation: update if better path found
                if tentative_distance < distances[neighbor]:
                    distances[neighbor] = tentative_distance
                    predecessors[neighbor] = current_vertex
                    
                    # Add to priority queue (allows duplicates, handled by visited check)
                    heapq.heappush(priority_queue, (tentative_distance, neighbor))
                    self._stats.queue_operations += 1
        
        # Record computation time
        computation_time = time.time() - start_time
        self._stats.computation_time = computation_time
        
        # Check if target was reached
        if target not in visited or distances[target] == float('inf'):
            return PathResult(
                path=[], total_cost=0.0, edges_traversed=[],
                computation_time=computation_time, nodes_visited=self._stats.nodes_visited,
                found=False, error_message=f"No path exists from '{source}' to '{target}'"
            )
        
        # Reconstruct path
        path = self._reconstruct_path(predecessors, source, target)
        edges_traversed = self._get_path_edges(graph, path)
        
        return PathResult(
            path=path,
            total_cost=distances[target],
            edges_traversed=edges_traversed,
            computation_time=computation_time,
            nodes_visited=self._stats.nodes_visited
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
        
        # Initialize Dijkstra's algorithm data structures
        distances = {vertex_id: float('inf') for vertex_id in graph.get_vertex_ids()}
        distances[source] = 0.0
        predecessors: Dict[str, Optional[str]] = {vertex_id: None for vertex_id in graph.get_vertex_ids()}
        visited: Set[str] = set()
        
        # Priority queue: (distance, vertex_id)
        priority_queue = [(0.0, source)]
        heapq.heapify(priority_queue)
        
        # Dijkstra's main loop (runs until all reachable vertices processed)
        while priority_queue:
            self._stats.queue_operations += 1
            current_distance, current_vertex = heapq.heappop(priority_queue)
            
            # Skip if already visited
            if current_vertex in visited:
                continue
            
            # Mark as visited
            visited.add(current_vertex)
            self._stats.nodes_visited += 1
            
            # Skip if current distance is outdated
            if current_distance > distances[current_vertex]:
                continue
            
            # Examine all neighbors
            neighbors = graph.get_neighbors(current_vertex)
            for edge in neighbors:
                self._stats.edges_examined += 1
                neighbor = edge.target
                
                # Skip blocked edges
                if edge.is_blocked:
                    continue
                
                # Calculate tentative distance
                edge_weight = graph.get_edge_weight(current_vertex, neighbor)
                tentative_distance = distances[current_vertex] + edge_weight
                
                # Relaxation
                if tentative_distance < distances[neighbor]:
                    distances[neighbor] = tentative_distance
                    predecessors[neighbor] = current_vertex
                    heapq.heappush(priority_queue, (tentative_distance, neighbor))
                    self._stats.queue_operations += 1
        
        # Record computation time
        computation_time = time.time() - start_time
        self._stats.computation_time = computation_time
        
        # Build results for all vertices
        results = {}
        for target in graph.get_vertex_ids():
            if target == source:
                # Source to itself
                results[target] = PathResult(
                    path=[source], total_cost=0.0, edges_traversed=[],
                    computation_time=computation_time, nodes_visited=self._stats.nodes_visited
                )
            elif target in visited and distances[target] != float('inf'):
                # Reachable target
                path = self._reconstruct_path(predecessors, source, target)
                edges_traversed = self._get_path_edges(graph, path)
                
                results[target] = PathResult(
                    path=path,
                    total_cost=distances[target],
                    edges_traversed=edges_traversed,
                    computation_time=computation_time,
                    nodes_visited=self._stats.nodes_visited
                )
            else:
                # Unreachable target
                results[target] = PathResult(
                    path=[], total_cost=0.0, edges_traversed=[],
                    computation_time=computation_time, nodes_visited=self._stats.nodes_visited,
                    found=False, error_message=f"No path exists from '{source}' to '{target}'"
                )
        
        return results
    
    def _reconstruct_path(self, predecessors: Dict[str, Optional[str]], 
                         source: str, target: str) -> List[str]:
        """
        Reconstruct the shortest path from predecessors dictionary.
        
        Args:
            predecessors: Dictionary mapping vertices to their predecessors
            source: Source vertex ID
            target: Target vertex ID
            
        Returns:
            List of vertex IDs representing the path from source to target
        """
        path = []
        current = target
        
        # Trace back from target to source
        while current is not None:
            path.append(current)
            current = predecessors[current]
        
        # Reverse to get path from source to target
        path.reverse()
        
        # Verify path starts with source (sanity check)
        if path and path[0] != source:
            raise RuntimeError(f"Path reconstruction error: expected source '{source}', got '{path[0]}'")
        
        return path
    
    def _get_path_edges(self, graph: GraphManager, path: List[str]) -> List:
        """
        Get the edges traversed in a path.
        
        Args:
            graph: GraphManager instance
            path: List of vertex IDs representing the path
            
        Returns:
            List of Edge objects traversed in the path
        """
        edges = []
        for i in range(len(path) - 1):
            source = path[i]
            target = path[i + 1]
            edge = graph.get_edge(source, target)
            if edge:
                edges.append(edge)
        
        return edges
    
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
    
    def verify_path_optimality(self, graph: GraphManager, path_result: PathResult) -> bool:
        """
        Verify that a computed path cost is correct (matches edge weights).
        
        This method checks that the path cost equals the sum of edge weights.
        It does not verify global optimality, only cost correctness.
        
        Args:
            graph: GraphManager instance
            path_result: PathResult to verify
            
        Returns:
            True if path cost is correct, False otherwise
        """
        if not path_result.found or len(path_result.path) < 2:
            return True  # Trivial cases are always correct
        
        # Verify path cost equals sum of edge weights
        calculated_cost = 0.0
        for edge in path_result.edges_traversed:
            calculated_cost += graph.get_edge_weight(edge.source, edge.target)
        
        return abs(calculated_cost - path_result.total_cost) < 1e-10
    
    def __str__(self) -> str:
        """String representation of the pathfinder engine."""
        return f"PathfinderEngine(stats={self._stats})"
    
    def __repr__(self) -> str:
        return self.__str__()