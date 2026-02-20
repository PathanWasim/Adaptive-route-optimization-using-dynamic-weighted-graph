"""
Bellman-Ford Pathfinding Engine for the disaster evacuation routing system.

This module implements the Bellman-Ford algorithm for finding shortest paths.
While slower than Dijkstra O(VE) vs O(E log V), it serves as:
1. A comparative baseline for algorithmic analysis
2. Capable of handling negative edge weights (theoretical completeness)
3. Can detect negative weight cycles

Time Complexity: O(V * E)
Space Complexity: O(V)
"""

import time
from typing import Dict, List, Optional, Set, Tuple
from ..models import PathResult, AlgorithmStats
from ..graph import GraphManager


class BellmanFordEngine:
    """
    Implements the Bellman-Ford algorithm for finding shortest paths.
    
    Unlike Dijkstra's greedy approach, Bellman-Ford uses dynamic programming:
    it performs V-1 iterations, each time relaxing ALL edges. This guarantees
    correctness even with negative edge weights.
    
    Comparison with Dijkstra:
    - Dijkstra: O(E log V) — faster, requires non-negative weights
    - Bellman-Ford: O(VE) — slower, handles negative weights, detects negative cycles
    
    For disaster evacuation (all weights ≥ 0), Dijkstra is preferred.
    Bellman-Ford is included for academic comparison and completeness.
    """
    
    def __init__(self):
        """Initialize the Bellman-Ford engine."""
        self._stats = AlgorithmStats()
        self._has_negative_cycle = False
    
    def find_shortest_path(self, graph: GraphManager, source: str, target: str,
                          track_steps: bool = False) -> PathResult:
        """
        Find the shortest path using the Bellman-Ford algorithm.
        
        Args:
            graph: GraphManager instance containing the graph
            source: Source vertex ID
            target: Target vertex ID
            track_steps: If True, track algorithm steps for visualization
            
        Returns:
            PathResult containing the optimal path and metadata
        """
        self._stats.reset()
        self._has_negative_cycle = False
        start_time = time.time()
        
        algorithm_steps = [] if track_steps else None
        
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
        
        if source == target:
            computation_time = time.time() - start_time
            return PathResult(
                path=[source], total_cost=0.0, edges_traversed=[],
                computation_time=computation_time, nodes_visited=1
            )
        
        # Initialize distances
        vertex_ids = list(graph.get_vertex_ids())
        num_vertices = len(vertex_ids)
        distances = {v: float('inf') for v in vertex_ids}
        distances[source] = 0.0
        predecessors: Dict[str, Optional[str]] = {v: None for v in vertex_ids}
        
        # Collect all edges once
        all_edges_data = []
        for v in vertex_ids:
            for edge in graph.get_neighbors(v):
                weight = graph.get_edge_weight(edge.source, edge.target)
                if weight != float('inf'):  # Skip blocked roads
                    all_edges_data.append((edge.source, edge.target, weight))
        
        # Main Bellman-Ford loop: V-1 iterations
        for iteration in range(num_vertices - 1):
            updated = False
            
            if track_steps:
                algorithm_steps.append({
                    'type': 'iteration',
                    'iteration': iteration + 1,
                    'total_iterations': num_vertices - 1
                })
            
            # Relax ALL edges
            for u, v, weight in all_edges_data:
                self._stats.edges_examined += 1
                
                if distances[u] != float('inf'):
                    tentative = distances[u] + weight
                    
                    if tentative < distances[v]:
                        distances[v] = tentative
                        predecessors[v] = u
                        updated = True
                        
                        if track_steps:
                            algorithm_steps.append({
                                'type': 'relax',
                                'from': u,
                                'to': v,
                                'new_distance': tentative,
                                'iteration': iteration + 1
                            })
            
            # Early termination: no updates means convergence
            if not updated:
                if track_steps:
                    algorithm_steps.append({
                        'type': 'early_stop',
                        'iteration': iteration + 1,
                        'message': f'Converged after {iteration + 1} iterations'
                    })
                break
        
        self._stats.nodes_visited = num_vertices
        
        # Check for negative weight cycles (V-th iteration)
        for u, v, weight in all_edges_data:
            if distances[u] != float('inf') and distances[u] + weight < distances[v]:
                self._has_negative_cycle = True
                computation_time = time.time() - start_time
                self._stats.computation_time = computation_time
                return PathResult(
                    path=[], total_cost=0.0, edges_traversed=[],
                    computation_time=computation_time, nodes_visited=self._stats.nodes_visited,
                    found=False, error_message="Negative weight cycle detected"
                )
        
        # Check if target is reachable
        if distances[target] == float('inf'):
            computation_time = time.time() - start_time
            self._stats.computation_time = computation_time
            return PathResult(
                path=[], total_cost=0.0, edges_traversed=[],
                computation_time=computation_time, nodes_visited=self._stats.nodes_visited,
                found=False, error_message=f"No path exists from '{source}' to '{target}'"
            )
        
        # Reconstruct path
        path, edges_traversed = self._reconstruct_path(graph, predecessors, source, target)
        computation_time = time.time() - start_time
        self._stats.computation_time = computation_time
        
        result = PathResult(
            path=path,
            total_cost=distances[target],
            edges_traversed=edges_traversed,
            computation_time=computation_time,
            nodes_visited=self._stats.nodes_visited
        )
        
        if track_steps:
            result.algorithm_steps = algorithm_steps
        
        return result
    
    def has_negative_cycle(self) -> bool:
        """Check if the last computation detected a negative weight cycle."""
        return self._has_negative_cycle
    
    def _reconstruct_path(self, graph: GraphManager, predecessors: Dict[str, Optional[str]],
                         source: str, target: str) -> Tuple[List[str], List]:
        """Reconstruct the shortest path from predecessors."""
        path = []
        edges_traversed = []
        current = target
        
        # Safety limit to prevent infinite loops
        max_steps = len(predecessors) + 1
        steps = 0
        
        while current is not None and steps < max_steps:
            path.append(current)
            if predecessors[current] is not None:
                edge = graph.get_edge(predecessors[current], current)
                if edge:
                    edges_traversed.append(edge)
            current = predecessors[current]
            steps += 1
        
        path.reverse()
        edges_traversed.reverse()
        return path, edges_traversed
    
    def get_algorithm_stats(self) -> AlgorithmStats:
        """Get statistics from the last pathfinding operation."""
        return AlgorithmStats(
            nodes_visited=self._stats.nodes_visited,
            computation_time=self._stats.computation_time,
            edges_examined=self._stats.edges_examined,
            queue_operations=self._stats.queue_operations
        )
    
    def __str__(self) -> str:
        return f"BellmanFordEngine(stats={self._stats})"
    
    def __repr__(self) -> str:
        return self.__str__()
