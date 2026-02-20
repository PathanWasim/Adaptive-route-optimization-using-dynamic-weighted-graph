"""
A* Pathfinding Engine for the disaster evacuation routing system.

This module implements the A* algorithm using a min-heap priority queue with
Haversine distance as an admissible heuristic, achieving optimal pathfinding
with typically fewer node expansions than Dijkstra's algorithm.

Time Complexity: O(E log V) worst case (same as Dijkstra)
Space Complexity: O(V)
Advantage: Explores fewer nodes by using heuristic guidance toward the target.
"""

import heapq
import math
import time
from typing import Dict, List, Optional, Set, Tuple
from ..models import PathResult, AlgorithmStats
from ..models import GraphManager


class AStarEngine:
    """
    Implements the A* algorithm for finding shortest paths in weighted graphs.
    
    A* extends Dijkstra's algorithm by adding a heuristic function h(n) that
    estimates the remaining cost to the target. The priority becomes:
        f(n) = g(n) + h(n)
    where g(n) is the actual cost from source to n.
    
    The Haversine heuristic is admissible (never overestimates) because the
    straight-line geographic distance is always ≤ actual road distance.
    This guarantees A* finds the optimal path.
    
    Compared to Dijkstra:
    - Same optimal path and cost
    - Typically visits 30-60% fewer nodes
    - Same worst-case complexity O(E log V)
    """
    
    EARTH_RADIUS_METERS = 6_371_000  # Earth's radius in meters
    
    def __init__(self):
        """Initialize the A* engine."""
        self._stats = AlgorithmStats()
    
    @staticmethod
    def _haversine_distance(coord1: Tuple[float, float], coord2: Tuple[float, float]) -> float:
        """
        Calculate great-circle distance between two geographic coordinates.
        
        This is the admissible heuristic for A*: straight-line distance
        never overestimates the actual road distance.
        
        Args:
            coord1: (latitude, longitude) of first point
            coord2: (latitude, longitude) of second point
            
        Returns:
            Distance in meters (straight line)
        """
        lat1, lon1 = math.radians(coord1[0]), math.radians(coord1[1])
        lat2, lon2 = math.radians(coord2[0]), math.radians(coord2[1])
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
        c = 2 * math.asin(math.sqrt(a))
        
        return AStarEngine.EARTH_RADIUS_METERS * c
    
    def find_shortest_path(self, graph: GraphManager, source: str, target: str,
                          track_steps: bool = False) -> PathResult:
        """
        Find the shortest path using A* algorithm with Haversine heuristic.
        
        Args:
            graph: GraphManager instance containing the graph
            source: Source vertex ID
            target: Target vertex ID
            track_steps: If True, track algorithm steps for visualization
            
        Returns:
            PathResult containing the optimal path and metadata
        """
        # Reset statistics
        self._stats.reset()
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
        
        # Handle trivial case
        if source == target:
            computation_time = time.time() - start_time
            return PathResult(
                path=[source], total_cost=0.0, edges_traversed=[],
                computation_time=computation_time, nodes_visited=1
            )
        
        # Get target coordinates for heuristic
        target_coords = graph.get_node_coordinates(target)
        
        # Initialize A* data structures
        g_scores = {vertex_id: float('inf') for vertex_id in graph.get_vertex_ids()}
        g_scores[source] = 0.0
        predecessors: Dict[str, Optional[str]] = {vertex_id: None for vertex_id in graph.get_vertex_ids()}
        visited: Set[str] = set()
        
        # Compute initial heuristic
        source_coords = graph.get_node_coordinates(source)
        h_source = self._compute_heuristic(source_coords, target_coords)
        
        # Priority queue: (f_score, tiebreaker, vertex_id)
        # f_score = g_score + heuristic
        counter = 0  # Tiebreaker for equal f_scores
        priority_queue = [(h_source, counter, source)]
        self._stats.queue_operations += 1
        
        while priority_queue:
            f_score, _, current_vertex = heapq.heappop(priority_queue)
            self._stats.queue_operations += 1
            
            # Skip if already visited
            if current_vertex in visited:
                continue
            
            # Mark as visited
            visited.add(current_vertex)
            self._stats.nodes_visited += 1
            
            # Track step for visualization
            if track_steps:
                algorithm_steps.append({
                    'type': 'visit',
                    'node': current_vertex,
                    'distance': g_scores[current_vertex],
                    'f_score': f_score,
                    'visited': list(visited),
                    'queue_size': len(priority_queue)
                })
            
            # Found target — reconstruct path
            if current_vertex == target:
                path, edges_traversed = self._reconstruct_path(
                    graph, predecessors, source, target
                )
                computation_time = time.time() - start_time
                self._stats.computation_time = computation_time
                
                result = PathResult(
                    path=path,
                    total_cost=g_scores[target],
                    edges_traversed=edges_traversed,
                    computation_time=computation_time,
                    nodes_visited=self._stats.nodes_visited
                )
                
                if track_steps:
                    result.algorithm_steps = algorithm_steps
                
                return result
            
            # Relax all outgoing edges
            for edge in graph.get_neighbors(current_vertex):
                self._stats.edges_examined += 1
                neighbor = edge.target
                
                if neighbor in visited:
                    continue
                
                edge_weight = graph.get_edge_weight(current_vertex, neighbor)
                
                # Skip blocked roads
                if edge_weight == float('inf'):
                    continue
                
                tentative_g = g_scores[current_vertex] + edge_weight
                
                if tentative_g < g_scores[neighbor]:
                    g_scores[neighbor] = tentative_g
                    predecessors[neighbor] = current_vertex
                    
                    # Compute f_score = g + h
                    neighbor_coords = graph.get_node_coordinates(neighbor)
                    h_value = self._compute_heuristic(neighbor_coords, target_coords)
                    f_score = tentative_g + h_value
                    
                    counter += 1
                    heapq.heappush(priority_queue, (f_score, counter, neighbor))
                    self._stats.queue_operations += 1
                    
                    if track_steps:
                        algorithm_steps.append({
                            'type': 'relax',
                            'from': current_vertex,
                            'to': neighbor,
                            'new_distance': tentative_g,
                            'f_score': f_score,
                            'heuristic': h_value
                        })
        
        # No path found
        computation_time = time.time() - start_time
        self._stats.computation_time = computation_time
        
        return PathResult(
            path=[], total_cost=0.0, edges_traversed=[],
            computation_time=computation_time, nodes_visited=self._stats.nodes_visited,
            found=False, error_message=f"No path exists from '{source}' to '{target}'"
        )
    
    def _compute_heuristic(self, coords: Optional[Tuple[float, float]],
                           target_coords: Optional[Tuple[float, float]]) -> float:
        """
        Compute admissible heuristic value using Haversine distance.
        
        If coordinates are unavailable, returns 0 (degrades to Dijkstra behavior).
        
        Args:
            coords: Current node coordinates (lat, lon)
            target_coords: Target node coordinates (lat, lon)
            
        Returns:
            Estimated remaining distance in meters
        """
        if coords is None or target_coords is None:
            return 0.0  # Falls back to Dijkstra behavior
        return self._haversine_distance(coords, target_coords)
    
    def _reconstruct_path(self, graph: GraphManager, predecessors: Dict[str, Optional[str]],
                         source: str, target: str) -> Tuple[List[str], List]:
        """Reconstruct the shortest path from predecessors."""
        path = []
        edges_traversed = []
        current = target
        
        while current is not None:
            path.append(current)
            if predecessors[current] is not None:
                edge = graph.get_edge(predecessors[current], current)
                if edge:
                    edges_traversed.append(edge)
            current = predecessors[current]
        
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
        return f"AStarEngine(stats={self._stats})"
    
    def __repr__(self) -> str:
        return self.__str__()
