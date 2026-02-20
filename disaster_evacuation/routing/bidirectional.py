"""
Bidirectional Dijkstra's Algorithm Implementation.

Finds the shortest path by running two simultaneous searches:
one forward from the source and one backward from the target.
Reduces search space from O(b^d) to O(b^(d/2)).
"""

import time
import heapq
from typing import List, Dict, Set, Tuple, Optional

from ..models import PathResult, AlgorithmStats, GraphManager, Edge


class BidirectionalDijkstra:
    """Implement Bidirectional Dijkstra for finding optimal paths."""
    
    def __init__(self):
        self._stats = AlgorithmStats()
        
    def find_shortest_path(self, graph: GraphManager, source: str, target: str, 
                          track_steps: bool = False) -> PathResult:
        """
        Find shortest path using bidirectional search.
        
        Args:
            graph: GraphManager containing the network
            source: Source vertex ID
            target: Target vertex ID
            track_steps: Whether to track intermediate visualization steps
            
        Returns:
            PathResult object containing the route
        """
        self._stats.reset()
        start_time = time.time()
        
        algorithm_steps = [] if track_steps else None
        
        if not graph.has_vertex(source):
            return PathResult([], 0.0, [], 0.0, 0, False, "Source vertex does not exist")
        if not graph.has_vertex(target):
            return PathResult([], 0.0, [], 0.0, 0, False, "Target vertex does not exist")
            
        if source == target:
            return PathResult([source], 0.0, [], time.time() - start_time, 1)
            
        # Build reverse adjacency list for backward search O(E)
        incoming_edges: Dict[str, List[Edge]] = {v: [] for v in graph.get_vertex_ids()}
        for edge in graph.get_all_edges():
            incoming_edges[edge.target].append(edge)
            
        # Initialization
        dist_f = {v: float('inf') for v in graph.get_vertex_ids()}
        dist_b = {v: float('inf') for v in graph.get_vertex_ids()}
        
        dist_f[source] = 0.0
        dist_b[target] = 0.0
        
        pred_f: Dict[str, Optional[str]] = {v: None for v in graph.get_vertex_ids()}
        pred_b: Dict[str, Optional[str]] = {v: None for v in graph.get_vertex_ids()}
        
        visited_f: Set[str] = set()
        visited_b: Set[str] = set()
        
        # Priority queues: (distance, vertex_id)
        pq_f = [(0.0, source)]
        pq_b = [(0.0, target)]
        self._stats.queue_operations += 2
        
        mu = float('inf')  # Best path length found so far
        best_meet_node = None
        
        while pq_f and pq_b:
            # Stopping criterion
            if pq_f[0][0] + pq_b[0][0] >= mu:
                break
                
            # Decide which direction to expand (expand smaller queue for performance)
            if len(pq_f) <= len(pq_b):
                # Forward Step
                d, u = heapq.heappop(pq_f)
                self._stats.queue_operations += 1
                
                if u in visited_f: continue
                visited_f.add(u)
                self._stats.nodes_visited += 1
                
                if track_steps:
                    algorithm_steps.append({'type': 'visit', 'node': u, 'distance': d, 'visited': list(visited_f), 'queue_size': len(pq_f)})
                
                for edge in graph.get_neighbors(u):
                    self._stats.edges_examined += 1
                    v = edge.target
                    weight = graph.get_edge_weight(u, v)
                    
                    if weight == float('inf') or v in visited_f:
                        continue
                        
                    if dist_f[u] + weight < dist_f[v]:
                        dist_f[v] = dist_f[u] + weight
                        pred_f[v] = u
                        heapq.heappush(pq_f, (dist_f[v], v))
                        self._stats.queue_operations += 1
                        
                        if track_steps:
                            algorithm_steps.append({'type': 'relax', 'from': u, 'to': v, 'new_distance': dist_f[v]})
                            
                        # Update best path
                        if v in visited_b and dist_f[v] + dist_b[v] < mu:
                            mu = dist_f[v] + dist_b[v]
                            best_meet_node = v
            else:
                # Backward Step
                d, v = heapq.heappop(pq_b)
                self._stats.queue_operations += 1
                
                if v in visited_b: continue
                visited_b.add(v)
                self._stats.nodes_visited += 1
                
                if track_steps:
                    algorithm_steps.append({'type': 'visit', 'node': v, 'distance': d, 'visited': list(visited_b), 'queue_size': len(pq_b)})
                
                for edge in incoming_edges[v]:
                    self._stats.edges_examined += 1
                    u = edge.source
                    weight = graph.get_edge_weight(u, v)
                    
                    if weight == float('inf') or u in visited_b:
                        continue
                        
                    if dist_b[v] + weight < dist_b[u]:
                        dist_b[u] = dist_b[v] + weight
                        pred_b[u] = v
                        heapq.heappush(pq_b, (dist_b[u], u))
                        self._stats.queue_operations += 1
                        
                        if track_steps:
                            algorithm_steps.append({'type': 'relax', 'from': v, 'to': u, 'new_distance': dist_b[u]})
                            
                        # Update best path
                        if u in visited_f and dist_f[u] + dist_b[u] < mu:
                            mu = dist_f[u] + dist_b[u]
                            best_meet_node = u
                            
        comp_time = time.time() - start_time
        self._stats.computation_time = comp_time
        
        if best_meet_node is None:
            return PathResult([], 0.0, [], comp_time, self._stats.nodes_visited, False, "No path exists")
            
        # Reconstruct path
        path, edges_traversed = self._reconstruct_bidirectional(graph, pred_f, pred_b, source, target, best_meet_node)
        
        res = PathResult(path, mu, edges_traversed, comp_time, self._stats.nodes_visited, True)
        if track_steps:
            res.algorithm_steps = algorithm_steps
        return res
        
    def _reconstruct_bidirectional(self, graph: GraphManager, pred_f: Dict, pred_b: Dict, 
                                   source: str, target: str, meet: str) -> Tuple[List[str], List[Edge]]:
        """Reconstruct full optimal path from forward and backward predecessors joining at meet."""
        # Forward path part
        path_f = []
        curr = meet
        while curr is not None:
            path_f.append(curr)
            curr = pred_f[curr]
        path_f.reverse()
        
        # Backward path part
        path_b = []
        curr = pred_b[meet]
        while curr is not None:
            path_b.append(curr)
            curr = pred_b[curr]
            
        full_path = path_f + path_b
        
        edges = []
        for i in range(len(full_path)-1):
            u = full_path[i]
            v = full_path[i+1]
            e = graph.get_edge(u, v)
            if e: edges.append(e)
            
        return full_path, edges
        
    def get_algorithm_stats(self) -> AlgorithmStats:
        return self._stats
