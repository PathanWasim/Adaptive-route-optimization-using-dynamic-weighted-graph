"""
Benchmark Runner for empirical complexity analysis of pathfinding algorithms.

Generates random weighted graphs of increasing sizes and measures actual runtime
to empirically validate theoretical complexity bounds:
- Dijkstra: O(E log V)
- A*: O(E log V) with heuristic advantage
- Bellman-Ford: O(V * E)
"""

import time
import random
import math
import json
from typing import Dict, List, Tuple, Optional
from ..models import GraphManager
from ..models import VertexType
from ..routing import PathfinderEngine, AStarEngine, BellmanFordEngine, BidirectionalDijkstra


class BenchmarkRunner:
    """
    Runs comparative benchmarks across all pathfinding algorithms.
    
    Generates graphs of configurable sizes and measures performance metrics
    for each algorithm, producing data suitable for plotting complexity curves.
    """
    
    # Default graph sizes for benchmarking
    DEFAULT_SIZES = [50, 100, 200, 500, 1000, 1500, 2000]
    
    def __init__(self):
        """Initialize the benchmark runner."""
        self.results: List[Dict] = []
    
    @staticmethod
    def generate_random_graph(num_vertices: int, avg_degree: float = 4.0,
                              seed: int = 42) -> Tuple[GraphManager, Dict[str, Tuple[float, float]]]:
        """
        Generate a random connected graph simulating an urban road network.
        
        Creates vertices with random geographic coordinates and edges with
        distance-based weights. Ensures graph connectivity via spanning tree.
        
        Args:
            num_vertices: Number of vertices (intersections)
            avg_degree: Average connections per vertex (default 4 for urban grid)
            seed: Random seed for reproducibility
            
        Returns:
            Tuple of (GraphManager, coordinate_mapping)
        """
        rng = random.Random(seed)
        graph = GraphManager()
        coords = {}
        
        # Base coordinates (simulating a city area)
        base_lat, base_lon = 18.52, 73.85
        spread = 0.02 * math.sqrt(num_vertices / 100)
        
        # Create vertices with random coordinates
        for i in range(num_vertices):
            vid = str(i)
            lat = base_lat + rng.uniform(-spread, spread)
            lon = base_lon + rng.uniform(-spread, spread)
            graph.add_vertex(vid, VertexType.INTERSECTION, (lat, lon))
            graph.set_node_coordinates(vid, lat, lon)
            coords[vid] = (lat, lon)
        
        # Ensure connectivity: build a random spanning tree first
        vertices = list(range(num_vertices))
        rng.shuffle(vertices)
        connected = {vertices[0]}
        
        for v in vertices[1:]:
            # Connect to a random already-connected vertex
            u = rng.choice(list(connected))
            connected.add(v)
            
            # Calculate distance-based weight
            c1, c2 = coords[str(u)], coords[str(v)]
            dist = BenchmarkRunner._euclidean_distance(c1, c2) * 111000  # Approx meters
            dist = max(dist, 10.0)  # Minimum 10m
            
            risk = rng.uniform(0.0, 0.3)
            congestion = rng.uniform(0.0, 0.2)
            
            graph.add_edge(str(u), str(v), dist, risk, congestion)
            graph.add_edge(str(v), str(u), dist, risk, congestion)
        
        # Add extra edges to reach desired average degree
        target_edges = int(num_vertices * avg_degree / 2)
        current_edges = num_vertices - 1
        
        attempts = 0
        max_attempts = target_edges * 10
        
        while current_edges < target_edges and attempts < max_attempts:
            u = rng.randint(0, num_vertices - 1)
            v = rng.randint(0, num_vertices - 1)
            
            if u != v and not graph.is_connected(str(u), str(v)):
                c1, c2 = coords[str(u)], coords[str(v)]
                dist = BenchmarkRunner._euclidean_distance(c1, c2) * 111000
                dist = max(dist, 10.0)
                
                risk = rng.uniform(0.0, 0.3)
                congestion = rng.uniform(0.0, 0.2)
                
                graph.add_edge(str(u), str(v), dist, risk, congestion)
                graph.add_edge(str(v), str(u), dist, risk, congestion)
                current_edges += 1
            
            attempts += 1
        
        return graph, coords
    
    @staticmethod
    def _euclidean_distance(c1: Tuple[float, float], c2: Tuple[float, float]) -> float:
        """Simple Euclidean distance between two coordinate pairs."""
        return math.sqrt((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2)
    
    def run_benchmarks(self, sizes: Optional[List[int]] = None,
                      runs_per_size: int = 5) -> List[Dict]:
        """
        Run comprehensive benchmarks across all algorithms and graph sizes.
        
        Args:
            sizes: List of graph sizes to test (default: DEFAULT_SIZES)
            runs_per_size: Number of runs per size for averaging
            
        Returns:
            List of benchmark result dictionaries
        """
        if sizes is None:
            sizes = self.DEFAULT_SIZES
        
        self.results = []
        
        dijkstra = PathfinderEngine()
        astar = AStarEngine()
        bellman_ford = BellmanFordEngine()
        bidirectional = BidirectionalDijkstra()
        from disaster_evacuation.routing.yen_k_shortest import YenKShortestPaths
        yen = YenKShortestPaths()
        
        for size in sizes:
            print(f"  Benchmarking V={size}...")
            
            dijkstra_times = []
            astar_times = []
            bf_times = []
            bi_times = []
            dijkstra_nodes = []
            astar_nodes = []
            bf_nodes = []
            bi_nodes = []
            yen_times = []
            edge_count = 0
            
            for run in range(runs_per_size):
                # Generate graph with different seed each run
                graph, coords = self.generate_random_graph(size, seed=42 + run)
                edge_count = graph.get_edge_count()
                
                # Pick random source and target (ensure they're different)
                rng = random.Random(100 + run)
                source = str(rng.randint(0, size - 1))
                target = str(rng.randint(0, size - 1))
                while target == source:
                    target = str(rng.randint(0, size - 1))
                
                # Benchmark Dijkstra
                start = time.perf_counter()
                result_d = dijkstra.find_shortest_path(graph, source, target)
                dijkstra_times.append(time.perf_counter() - start)
                dijkstra_nodes.append(result_d.nodes_visited if result_d.found else 0)
                
                # Benchmark A*
                start = time.perf_counter()
                result_a = astar.find_shortest_path(graph, source, target)
                astar_times.append(time.perf_counter() - start)
                astar_nodes.append(result_a.nodes_visited if result_a.found else 0)
                
                # Benchmark Bidirectional Dijkstra
                start = time.perf_counter()
                result_bi = bidirectional.find_shortest_path(graph, source, target)
                bi_times.append(time.perf_counter() - start)
                bi_nodes.append(result_bi.nodes_visited if result_bi.found else 0)
                
                # Benchmark Bellman-Ford (skip for very large graphs — too slow)
                if size <= 1500:
                    start = time.perf_counter()
                    result_bf = bellman_ford.find_shortest_path(graph, source, target)
                    bf_times.append(time.perf_counter() - start)
                    bf_nodes.append(result_bf.nodes_visited if result_bf.found else 0)
                
                # Benchmark Yen's k-shortest (skip for very large graphs, k=3)
                if size <= 1500:
                    start = time.perf_counter()
                    yen.find_k_shortest_paths(graph, source, target, k=3)
                    yen_times.append(time.perf_counter() - start)
            
            # Compute theoretical complexity values
            V = size
            E = edge_count
            theoretical_e_log_v = E * math.log2(V) if V > 0 else 0
            theoretical_ve = V * E
            
            result = {
                'vertices': V,
                'edges': E,
                'dijkstra': {
                    'avg_time_ms': sum(dijkstra_times) / len(dijkstra_times) * 1000,
                    'min_time_ms': min(dijkstra_times) * 1000,
                    'max_time_ms': max(dijkstra_times) * 1000,
                    'avg_nodes_visited': sum(dijkstra_nodes) / len(dijkstra_nodes),
                    'theoretical_complexity': 'O(E log V)',
                    'theoretical_value': theoretical_e_log_v
                },
                'astar': {
                    'avg_time_ms': sum(astar_times) / len(astar_times) * 1000,
                    'min_time_ms': min(astar_times) * 1000,
                    'max_time_ms': max(astar_times) * 1000,
                    'avg_nodes_visited': sum(astar_nodes) / len(astar_nodes),
                    'theoretical_complexity': 'O(E log V)',
                    'theoretical_value': theoretical_e_log_v,
                    'node_reduction_pct': (
                        (1 - sum(astar_nodes) / max(sum(dijkstra_nodes), 1)) * 100
                    )
                },
                'bidirectional': {
                    'avg_time_ms': sum(bi_times) / len(bi_times) * 1000,
                    'min_time_ms': min(bi_times) * 1000,
                    'max_time_ms': max(bi_times) * 1000,
                    'avg_nodes_visited': sum(bi_nodes) / len(bi_nodes),
                    'theoretical_complexity': 'O(b^(d/2))',
                    'theoretical_value': theoretical_e_log_v / 2, # Approximation for plotting
                    'node_reduction_pct': (
                        (1 - sum(bi_nodes) / max(sum(dijkstra_nodes), 1)) * 100
                    )
                },
                'bellman_ford': {
                    'avg_time_ms': sum(bf_times) / len(bf_times) * 1000 if bf_times else None,
                    'min_time_ms': min(bf_times) * 1000 if bf_times else None,
                    'max_time_ms': max(bf_times) * 1000 if bf_times else None,
                    'avg_nodes_visited': sum(bf_nodes) / len(bf_nodes) if bf_nodes else None,
                    'theoretical_complexity': 'O(V * E)',
                    'theoretical_value': theoretical_ve,
                    'skipped': len(bf_times) == 0
                },
                'yen_k_paths': {
                    'avg_time_ms': sum(yen_times) / len(yen_times) * 1000 if yen_times else None,
                    'min_time_ms': min(yen_times) * 1000 if yen_times else None,
                    'max_time_ms': max(yen_times) * 1000 if yen_times else None,
                    'theoretical_complexity': 'O(k * V * (E + V log V))',
                    'theoretical_value': 3 * V * (E + V * math.log2(V)) if V > 0 else 0,
                    'skipped': len(yen_times) == 0
                }
            }
            
            self.results.append(result)
        
        return self.results
    
    def get_summary_table(self) -> str:
        """
        Generate a formatted summary table of benchmark results.
        
        Returns:
            Markdown-formatted table string
        """
        lines = []
        lines.append("| V | E | Dijkstra (ms) | A* (ms) | Bi-Dijkstra (ms) | Bellman-Ford (ms) | Yen k=3 (ms) |")
        lines.append("|---|---|--------------|---------|------------------|-------------------|--------------|")
        
        for r in self.results:
            bf_time = f"{r['bellman_ford']['avg_time_ms']:.2f}" if r['bellman_ford']['avg_time_ms'] is not None else "N/A"
            yen_time = f"{r['yen_k_paths']['avg_time_ms']:.2f}" if r['yen_k_paths']['avg_time_ms'] is not None else "N/A"
            
            lines.append(
                f"| {r['vertices']} | {r['edges']} | "
                f"{r['dijkstra']['avg_time_ms']:.2f} | "
                f"{r['astar']['avg_time_ms']:.2f} | "
                f"{r['bidirectional']['avg_time_ms']:.2f} | "
                f"{bf_time} | "
                f"{yen_time} |"
            )
        
        return "\n".join(lines)
    
    def save_results(self, filepath: str) -> None:
        """Save benchmark results to a JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2)
    
    def get_complexity_comparison(self) -> Dict:
        """
        Generate data for complexity comparison chart.
        
        Returns:
            Dictionary with data points for each algorithm
        """
        return {
            'labels': [r['vertices'] for r in self.results],
            'dijkstra_times': [r['dijkstra']['avg_time_ms'] for r in self.results],
            'astar_times': [r['astar']['avg_time_ms'] for r in self.results],
            'bidirectional_times': [r['bidirectional']['avg_time_ms'] for r in self.results],
            'bellman_ford_times': [
                r['bellman_ford']['avg_time_ms'] if r['bellman_ford']['avg_time_ms'] is not None else None
                for r in self.results
            ],
            'yen_k_times': [
                r['yen_k_paths']['avg_time_ms'] if r['yen_k_paths']['avg_time_ms'] is not None else None
                for r in self.results
            ],
            'dijkstra_nodes': [r['dijkstra']['avg_nodes_visited'] for r in self.results],
            'astar_nodes': [r['astar']['avg_nodes_visited'] for r in self.results],
            'bidirectional_nodes': [r['bidirectional']['avg_nodes_visited'] for r in self.results],
            'edges': [r['edges'] for r in self.results]
        }
