"""
Yen's k-Shortest Paths Algorithm Implementation.

Finds the top k shortest loopless paths between a source and target node
in a weighted graph. Useful for providing alternative evacuation routes.
Time Complexity: O(k*V*(E + V log V))
"""

import time
import heapq
from typing import List, Dict, Set, Tuple

from ..models import PathResult, AlgorithmStats, GraphManager
from .dijkstra import PathfinderEngine


class YenKShortestPaths:
    """Implement Yen's algorithm for finding k-shortest loopless paths."""
    
    def __init__(self):
        """Initialize the Yen algorithm engine."""
        self._stats = AlgorithmStats()
        self._dijkstra = PathfinderEngine()
        
    def find_k_shortest_paths(self, graph: GraphManager, source: str, target: str, k: int = 3) -> List[PathResult]:
        """
        Find the k shortest loopless paths from source to target.
        
        Args:
            graph: GraphManager containing the network
            source: Source vertex ID
            target: Target vertex ID
            k: Number of paths to find
            
        Returns:
            List of PathResult objects, sorted from shortest to longest
        """
        self._stats.reset()
        start_time = time.time()
        
        # A list to store the k shortest paths
        A: List[PathResult] = []
        # A min-heap to store candidate paths: (cost, id(path), tuple(path), PathResult)
        # We use path_counter to prevent comparing tuples/PathResults if costs are equal
        B = []
        B_paths: Set[Tuple[str, ...]] = set()
        path_counter = 0  # Tie breaker
        
        # 1. Find the first shortest path
        initial_path = self._dijkstra.find_shortest_path(graph, source, target)
        self._stats.nodes_visited += self._dijkstra.get_algorithm_stats().nodes_visited
        self._stats.edges_examined += self._dijkstra.get_algorithm_stats().edges_examined
        self._stats.queue_operations += self._dijkstra.get_algorithm_stats().queue_operations
        
        if not initial_path.found:
            self._stats.computation_time = time.time() - start_time
            return []
            
        A.append(initial_path)
        
        # 2. Iteratively find the k-th shortest path
        for k_idx in range(1, k):
            prev_path_nodes = A[k_idx - 1].path
            
            # The spur node ranges from the first node to the next to last node
            for i in range(len(prev_path_nodes) - 1):
                spur_node = prev_path_nodes[i]
                root_path = prev_path_nodes[:i+1]
                
                ignored_edges: Set[Tuple[str, str]] = set()
                ignored_nodes: Set[str] = set()
                
                # Ignore edges that are part of the previous shortest paths sharing the root
                for p in A:
                    if len(p.path) > i + 1 and p.path[:i+1] == root_path:
                        ignored_edges.add((p.path[i], p.path[i+1]))
                        
                # Ignore nodes in the root path (except the spur node) to maintain strict loopless behavior
                for node in root_path[:-1]:
                    ignored_nodes.add(node)
                    
                # Calculate spur path from spur_node to target
                # We reuse Dijkstra, passing the ignored components
                spur_path_result = self._dijkstra.find_shortest_path(
                    graph, spur_node, target, 
                    ignored_nodes=ignored_nodes, 
                    ignored_edges=ignored_edges
                )
                
                # Accumulate stats from subroutine
                curr_algo_stats = self._dijkstra.get_algorithm_stats()
                self._stats.nodes_visited += curr_algo_stats.nodes_visited
                self._stats.edges_examined += curr_algo_stats.edges_examined
                self._stats.queue_operations += curr_algo_stats.queue_operations
                
                if spur_path_result.found:
                    # Construct complete path natively
                    total_path_nodes = root_path[:-1] + spur_path_result.path
                    path_tuple = tuple(total_path_nodes)
                    
                    if path_tuple not in B_paths:
                        # Compute full cost and traversed edges
                        total_cost = 0.0
                        edges_traversed = []
                        for j in range(len(total_path_nodes) - 1):
                            u = total_path_nodes[j]
                            v = total_path_nodes[j+1]
                            edge_weight = graph.get_edge_weight(u, v)
                            total_cost += edge_weight
                            
                            edge = graph.get_edge(u, v)
                            if edge:
                                edges_traversed.append(edge)
                                
                        new_path_result = PathResult(
                            path=total_path_nodes,
                            total_cost=total_cost,
                            edges_traversed=edges_traversed,
                            computation_time=0.0,  # Filled at the end
                            nodes_visited=self._stats.nodes_visited,
                            found=True
                        )
                        
                        heapq.heappush(B, (total_cost, path_counter, path_tuple, new_path_result))
                        B_paths.add(path_tuple)
                        path_counter += 1
                        self._stats.queue_operations += 1
                        
            if not B:
                break
                
            # Pop the shortest path candidate from B
            _, _, extract_tuple, shortest_candidate = heapq.heappop(B)
            B_paths.remove(extract_tuple)
            A.append(shortest_candidate)
            
        self._stats.computation_time = time.time() - start_time
        
        # Consistency: Update computation time metric for all results
        for p in A:
            p.computation_time = self._stats.computation_time
            
        return A
        
    def get_algorithm_stats(self) -> AlgorithmStats:
        """Get performance metrics for the last operation."""
        return self._stats
