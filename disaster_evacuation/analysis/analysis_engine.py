"""
Analysis Engine for the disaster evacuation routing system.

This module implements the AnalysisEngine class that provides comparative analysis
capabilities for evaluating different routing strategies, computing trade-offs between
safety and efficiency, and generating statistical measures for academic evaluation.
"""

import statistics
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from ..models import PathResult, DisasterEvent, Edge, Vertex
from ..models import GraphManager
from ..routing import PathfinderEngine
from ..models import DisasterModel


@dataclass
class RouteAnalysis:
    """
    Comprehensive analysis of a single route.
    
    Contains detailed metrics and statistics for academic evaluation
    and comparative analysis purposes.
    """
    route_id: str
    path: List[str]
    total_cost: float
    total_distance: float
    total_risk: float
    total_congestion: float
    edge_count: int
    computation_time: float
    nodes_visited: int
    
    # Safety metrics
    safety_score: float
    risk_distribution: List[float]
    blocked_roads_avoided: int
    high_risk_roads_used: int
    
    # Efficiency metrics
    efficiency_score: float
    distance_optimality: float
    time_optimality: float
    
    # Robustness metrics
    robustness_score: float
    alternative_paths_available: int
    critical_edges: List[str]
    
    # Additional metadata
    disaster_applied: bool
    disaster_type: Optional[str] = None
    disaster_severity: Optional[float] = None


@dataclass
class ComparativeAnalysis:
    """
    Comparative analysis between multiple routing strategies.
    
    Provides quantitative metrics for evaluating trade-offs between
    different approaches to evacuation routing.
    """
    analysis_id: str
    routes: List[RouteAnalysis]
    
    # Comparative metrics
    safety_vs_efficiency_tradeoff: Dict[str, float]
    pareto_optimal_routes: List[str]
    dominated_routes: List[str]
    
    # Statistical measures
    cost_variance: float
    risk_variance: float
    efficiency_correlation: float
    
    # Academic metrics
    algorithm_performance: Dict[str, Any]
    complexity_analysis: Dict[str, Any]
    theoretical_bounds: Dict[str, float]


class AnalysisEngine:
    """
    Provides comprehensive analysis capabilities for evacuation routing strategies.
    
    The AnalysisEngine performs:
    - Quantitative comparison of routing strategies
    - Safety vs efficiency trade-off analysis
    - Statistical evaluation of algorithm performance
    - Academic-quality metrics for research purposes
    """
    
    def __init__(self):
        """Initialize the analysis engine."""
        self._pathfinder = PathfinderEngine()
        self._disaster_model = DisasterModel()
        
        # Analysis configuration
        self.safety_weight = 0.4
        self.efficiency_weight = 0.3
        self.robustness_weight = 0.3
        
        # Thresholds for classification
        self.high_risk_threshold = 0.5
        self.critical_edge_threshold = 0.8
        self.efficiency_threshold = 0.7
    
    def analyze_single_route(self, graph: GraphManager, path_result: PathResult,
                           route_id: str, disaster: Optional[DisasterEvent] = None) -> RouteAnalysis:
        """
        Perform comprehensive analysis of a single route.
        
        Args:
            graph: GraphManager instance containing the network
            path_result: PathResult from pathfinding algorithm
            route_id: Unique identifier for this route analysis
            disaster: Optional disaster event context
            
        Returns:
            RouteAnalysis with comprehensive metrics
        """
        if not path_result.found:
            # Return minimal analysis for failed routes
            return RouteAnalysis(
                route_id=route_id,
                path=[],
                total_cost=float('inf'),
                total_distance=0.0,
                total_risk=0.0,
                total_congestion=0.0,
                edge_count=0,
                computation_time=path_result.computation_time,
                nodes_visited=path_result.nodes_visited,
                safety_score=0.0,
                risk_distribution=[],
                blocked_roads_avoided=0,
                high_risk_roads_used=0,
                efficiency_score=0.0,
                distance_optimality=0.0,
                time_optimality=0.0,
                robustness_score=0.0,
                alternative_paths_available=0,
                critical_edges=[],
                disaster_applied=disaster is not None,
                disaster_type=disaster.disaster_type.value if disaster else None,
                disaster_severity=disaster.severity if disaster else None
            )
        
        # Basic metrics
        edges = path_result.edges_traversed
        total_distance = sum(edge.base_distance for edge in edges)
        total_risk = sum(edge.base_risk for edge in edges)
        total_congestion = sum(edge.base_congestion for edge in edges)
        
        # Safety analysis
        safety_metrics = self._analyze_route_safety(graph, path_result, disaster)
        
        # Efficiency analysis
        efficiency_metrics = self._analyze_route_efficiency(graph, path_result)
        
        # Robustness analysis
        robustness_metrics = self._analyze_route_robustness(graph, path_result)
        
        return RouteAnalysis(
            route_id=route_id,
            path=path_result.path,
            total_cost=path_result.total_cost,
            total_distance=total_distance,
            total_risk=total_risk,
            total_congestion=total_congestion,
            edge_count=len(edges),
            computation_time=path_result.computation_time,
            nodes_visited=path_result.nodes_visited,
            safety_score=safety_metrics["safety_score"],
            risk_distribution=safety_metrics["risk_distribution"],
            blocked_roads_avoided=safety_metrics["blocked_roads_avoided"],
            high_risk_roads_used=safety_metrics["high_risk_roads_used"],
            efficiency_score=efficiency_metrics["efficiency_score"],
            distance_optimality=efficiency_metrics["distance_optimality"],
            time_optimality=efficiency_metrics["time_optimality"],
            robustness_score=robustness_metrics["robustness_score"],
            alternative_paths_available=robustness_metrics["alternative_paths_available"],
            critical_edges=robustness_metrics["critical_edges"],
            disaster_applied=disaster is not None,
            disaster_type=disaster.disaster_type.value if disaster else None,
            disaster_severity=disaster.severity if disaster else None
        )
    
    def compare_routing_strategies(self, graph: GraphManager, source: str, destination: str,
                                 strategies: List[Dict[str, Any]]) -> ComparativeAnalysis:
        """
        Compare multiple routing strategies comprehensively.
        
        Args:
            graph: GraphManager instance containing the network
            source: Source vertex ID
            destination: Destination vertex ID
            strategies: List of strategy configurations to compare
            
        Returns:
            ComparativeAnalysis with detailed comparison metrics
        """
        route_analyses = []
        
        # Analyze each strategy
        for i, strategy in enumerate(strategies):
            strategy_id = strategy.get("id", f"strategy_{i}")
            disaster = strategy.get("disaster")
            
            # Apply disaster effects if specified
            if disaster:
                self._disaster_model.apply_disaster_effects(graph, disaster)
            
            try:
                # Compute path using the strategy
                path_result = self._pathfinder.find_shortest_path(graph, source, destination)
                
                # Analyze the resulting route
                analysis = self.analyze_single_route(graph, path_result, strategy_id, disaster)
                route_analyses.append(analysis)
                
            finally:
                # Clean up disaster effects
                if disaster:
                    self._disaster_model.remove_disaster_effects(graph, disaster)
        
        # Perform comparative analysis
        return self._perform_comparative_analysis(route_analyses)
    
    def evaluate_safety_efficiency_tradeoff(self, analyses: List[RouteAnalysis]) -> Dict[str, Any]:
        """
        Evaluate the trade-off between safety and efficiency across routes.
        
        Args:
            analyses: List of route analyses to compare
            
        Returns:
            Dictionary with trade-off analysis results
        """
        if not analyses:
            return {"error": "No analyses provided"}
        
        # Extract safety and efficiency scores
        safety_scores = [a.safety_score for a in analyses if a.path]
        efficiency_scores = [a.efficiency_score for a in analyses if a.path]
        
        if not safety_scores or not efficiency_scores:
            return {"error": "No valid routes to analyze"}
        
        # Calculate correlation
        correlation = self._calculate_correlation(safety_scores, efficiency_scores)
        
        # Find Pareto optimal solutions
        pareto_optimal = self._find_pareto_optimal_routes(analyses)
        
        # Calculate trade-off metrics
        tradeoff_analysis = {
            "safety_efficiency_correlation": correlation,
            "pareto_optimal_routes": [r.route_id for r in pareto_optimal],
            "safety_range": {
                "min": min(safety_scores),
                "max": max(safety_scores),
                "mean": statistics.mean(safety_scores),
                "std": statistics.stdev(safety_scores) if len(safety_scores) > 1 else 0.0
            },
            "efficiency_range": {
                "min": min(efficiency_scores),
                "max": max(efficiency_scores),
                "mean": statistics.mean(efficiency_scores),
                "std": statistics.stdev(efficiency_scores) if len(efficiency_scores) > 1 else 0.0
            },
            "trade_off_ratio": self._calculate_tradeoff_ratio(analyses),
            "dominated_solutions": len(analyses) - len(pareto_optimal)
        }
        
        return tradeoff_analysis
    
    def generate_performance_report(self, comparative_analysis: ComparativeAnalysis) -> Dict[str, Any]:
        """
        Generate a comprehensive performance report for academic evaluation.
        
        Args:
            comparative_analysis: Comparative analysis results
            
        Returns:
            Dictionary with detailed performance metrics
        """
        routes = comparative_analysis.routes
        valid_routes = [r for r in routes if r.path]
        
        if not valid_routes:
            return {"error": "No valid routes to analyze"}
        
        # Algorithm performance metrics
        computation_times = [r.computation_time for r in valid_routes]
        nodes_visited = [r.nodes_visited for r in valid_routes]
        
        # Route quality metrics
        costs = [r.total_cost for r in valid_routes]
        distances = [r.total_distance for r in valid_routes]
        risks = [r.total_risk for r in valid_routes]
        
        # Generate comprehensive report
        report = {
            "executive_summary": {
                "total_strategies_analyzed": len(routes),
                "successful_routes": len(valid_routes),
                "pareto_optimal_solutions": len(comparative_analysis.pareto_optimal_routes),
                "best_safety_route": max(valid_routes, key=lambda r: r.safety_score).route_id,
                "best_efficiency_route": max(valid_routes, key=lambda r: r.efficiency_score).route_id,
                "most_robust_route": max(valid_routes, key=lambda r: r.robustness_score).route_id
            },
            "algorithm_performance": {
                "computation_time": {
                    "mean": statistics.mean(computation_times),
                    "median": statistics.median(computation_times),
                    "std": statistics.stdev(computation_times) if len(computation_times) > 1 else 0.0,
                    "min": min(computation_times),
                    "max": max(computation_times)
                },
                "nodes_visited": {
                    "mean": statistics.mean(nodes_visited),
                    "median": statistics.median(nodes_visited),
                    "std": statistics.stdev(nodes_visited) if len(nodes_visited) > 1 else 0.0,
                    "min": min(nodes_visited),
                    "max": max(nodes_visited)
                },
                "complexity_analysis": comparative_analysis.complexity_analysis
            },
            "route_quality": {
                "cost_distribution": {
                    "mean": statistics.mean(costs),
                    "median": statistics.median(costs),
                    "std": statistics.stdev(costs) if len(costs) > 1 else 0.0,
                    "coefficient_of_variation": statistics.stdev(costs) / statistics.mean(costs) if len(costs) > 1 and statistics.mean(costs) > 0 else 0.0
                },
                "distance_distribution": {
                    "mean": statistics.mean(distances),
                    "median": statistics.median(distances),
                    "std": statistics.stdev(distances) if len(distances) > 1 else 0.0
                },
                "risk_distribution": {
                    "mean": statistics.mean(risks),
                    "median": statistics.median(risks),
                    "std": statistics.stdev(risks) if len(risks) > 1 else 0.0
                }
            },
            "safety_analysis": {
                "average_safety_score": statistics.mean([r.safety_score for r in valid_routes]),
                "safety_variance": comparative_analysis.cost_variance,
                "high_risk_roads_usage": sum(r.high_risk_roads_used for r in valid_routes),
                "blocked_roads_avoided": sum(r.blocked_roads_avoided for r in valid_routes)
            },
            "efficiency_analysis": {
                "average_efficiency_score": statistics.mean([r.efficiency_score for r in valid_routes]),
                "distance_optimality": statistics.mean([r.distance_optimality for r in valid_routes]),
                "time_optimality": statistics.mean([r.time_optimality for r in valid_routes])
            },
            "robustness_analysis": {
                "average_robustness_score": statistics.mean([r.robustness_score for r in valid_routes]),
                "alternative_paths_available": statistics.mean([r.alternative_paths_available for r in valid_routes]),
                "critical_edges_identified": sum(len(r.critical_edges) for r in valid_routes)
            },
            "trade_off_analysis": comparative_analysis.safety_vs_efficiency_tradeoff,
            "theoretical_bounds": comparative_analysis.theoretical_bounds
        }
        
        return report
    
    def _analyze_route_safety(self, graph: GraphManager, path_result: PathResult,
                            disaster: Optional[DisasterEvent]) -> Dict[str, Any]:
        """Analyze safety aspects of a route."""
        edges = path_result.edges_traversed
        
        # Risk distribution analysis
        risk_values = [edge.base_risk for edge in edges]
        risk_distribution = risk_values.copy()
        
        # Count high-risk roads used
        high_risk_roads_used = sum(1 for risk in risk_values if risk > self.high_risk_threshold)
        
        # Count blocked roads avoided (if disaster present)
        blocked_roads_avoided = 0
        if disaster:
            all_edges = graph.get_all_edges()
            affected_edges = self._disaster_model.get_affected_edges(
                graph, disaster.epicenter, disaster.max_effect_radius
            )
            blocked_edges = [e for e in affected_edges if e.is_blocked]
            
            # Count blocked edges not in path
            path_edge_keys = set((e.source, e.target) for e in edges)
            blocked_roads_avoided = sum(
                1 for e in blocked_edges 
                if (e.source, e.target) not in path_edge_keys
            )
        
        # Calculate overall safety score
        if risk_values:
            avg_risk = statistics.mean(risk_values)
            risk_penalty = high_risk_roads_used / len(edges) if edges else 0
            safety_score = max(0.0, 1.0 - avg_risk - risk_penalty * 0.2)
        else:
            safety_score = 0.0
        
        return {
            "safety_score": safety_score,
            "risk_distribution": risk_distribution,
            "blocked_roads_avoided": blocked_roads_avoided,
            "high_risk_roads_used": high_risk_roads_used
        }
    
    def _analyze_route_efficiency(self, graph: GraphManager, path_result: PathResult) -> Dict[str, Any]:
        """Analyze efficiency aspects of a route."""
        if not path_result.found or not path_result.path:
            return {
                "efficiency_score": 0.0,
                "distance_optimality": 0.0,
                "time_optimality": 0.0
            }
        
        # Calculate theoretical minimum distance (straight-line)
        source_vertex = graph.get_vertex(path_result.path[0])
        target_vertex = graph.get_vertex(path_result.path[-1])
        
        if source_vertex and target_vertex:
            straight_line_distance = source_vertex.distance_to(target_vertex)
            actual_distance = sum(edge.base_distance for edge in path_result.edges_traversed)
            
            distance_optimality = straight_line_distance / actual_distance if actual_distance > 0 else 0.0
        else:
            distance_optimality = 0.0
        
        # Time optimality based on computation efficiency
        time_optimality = min(1.0, 1.0 / (path_result.computation_time + 0.001))  # Avoid division by zero
        
        # Overall efficiency score (clamped to [0, 1])
        efficiency_score = min(1.0, distance_optimality * 0.6 + time_optimality * 0.4)
        
        return {
            "efficiency_score": efficiency_score,
            "distance_optimality": distance_optimality,
            "time_optimality": time_optimality
        }
    
    def _analyze_route_robustness(self, graph: GraphManager, path_result: PathResult) -> Dict[str, Any]:
        """Analyze robustness aspects of a route."""
        if not path_result.found or not path_result.path:
            return {
                "robustness_score": 0.0,
                "alternative_paths_available": 0,
                "critical_edges": []
            }
        
        # Identify critical edges (edges with high weight relative to alternatives)
        critical_edges = []
        for edge in path_result.edges_traversed:
            if edge.current_weight > self.critical_edge_threshold:
                critical_edges.append(f"{edge.source}->{edge.target}")
        
        # Estimate alternative paths (simplified heuristic)
        source = path_result.path[0]
        target = path_result.path[-1]
        
        # Count vertices with degree > 2 as potential alternative route points
        alternative_paths_available = 0
        for vertex_id in path_result.path[1:-1]:  # Exclude source and target
            neighbors = graph.get_neighbors(vertex_id)
            if len(neighbors) > 2:
                alternative_paths_available += 1
        
        # Calculate robustness score (clamped to [0, 1])
        critical_edge_penalty = len(critical_edges) / len(path_result.edges_traversed) if path_result.edges_traversed else 0
        alternative_bonus = min(1.0, alternative_paths_available / max(1, len(path_result.path) - 2))
        
        robustness_score = max(0.0, min(1.0, 1.0 - critical_edge_penalty + alternative_bonus * 0.3))
        
        return {
            "robustness_score": robustness_score,
            "alternative_paths_available": alternative_paths_available,
            "critical_edges": critical_edges
        }
    
    def _perform_comparative_analysis(self, route_analyses: List[RouteAnalysis]) -> ComparativeAnalysis:
        """Perform comprehensive comparative analysis."""
        valid_routes = [r for r in route_analyses if r.path]
        
        if not valid_routes:
            return ComparativeAnalysis(
                analysis_id="empty_analysis",
                routes=route_analyses,
                safety_vs_efficiency_tradeoff={},
                pareto_optimal_routes=[],
                dominated_routes=[],
                cost_variance=0.0,
                risk_variance=0.0,
                efficiency_correlation=0.0,
                algorithm_performance={
                    "average_computation_time": 0.0,
                    "average_nodes_visited": 0,
                    "time_complexity_estimate": "O(1)",
                    "space_complexity_estimate": "O(1)"
                },
                complexity_analysis={},
                theoretical_bounds={}
            )
        
        # Calculate trade-off analysis
        tradeoff_analysis = self.evaluate_safety_efficiency_tradeoff(route_analyses)
        
        # Find Pareto optimal routes
        pareto_optimal = self._find_pareto_optimal_routes(valid_routes)
        pareto_optimal_ids = [r.route_id for r in pareto_optimal]
        dominated_ids = [r.route_id for r in valid_routes if r.route_id not in pareto_optimal_ids]
        
        # Calculate statistical measures
        costs = [r.total_cost for r in valid_routes]
        risks = [r.total_risk for r in valid_routes]
        efficiency_scores = [r.efficiency_score for r in valid_routes]
        safety_scores = [r.safety_score for r in valid_routes]
        
        cost_variance = statistics.variance(costs) if len(costs) > 1 else 0.0
        risk_variance = statistics.variance(risks) if len(risks) > 1 else 0.0
        efficiency_correlation = self._calculate_correlation(efficiency_scores, safety_scores)
        
        # Algorithm performance analysis
        computation_times = [r.computation_time for r in valid_routes]
        nodes_visited = [r.nodes_visited for r in valid_routes]
        
        if computation_times and nodes_visited:
            algorithm_performance = {
                "average_computation_time": statistics.mean(computation_times),
                "average_nodes_visited": statistics.mean(nodes_visited),
                "time_complexity_estimate": self._estimate_time_complexity(valid_routes),
                "space_complexity_estimate": self._estimate_space_complexity(valid_routes)
            }
        else:
            algorithm_performance = {
                "average_computation_time": 0.0,
                "average_nodes_visited": 0,
                "time_complexity_estimate": "O(1)",
                "space_complexity_estimate": "O(1)"
            }
        
        # Complexity analysis
        complexity_analysis = {
            "graph_size_impact": self._analyze_graph_size_impact(valid_routes),
            "disaster_impact": self._analyze_disaster_impact(valid_routes),
            "scalability_metrics": self._analyze_scalability(valid_routes)
        }
        
        # Theoretical bounds
        theoretical_bounds = self._calculate_theoretical_bounds(valid_routes)
        
        return ComparativeAnalysis(
            analysis_id=f"analysis_{len(route_analyses)}_routes",
            routes=route_analyses,
            safety_vs_efficiency_tradeoff=tradeoff_analysis,
            pareto_optimal_routes=pareto_optimal_ids,
            dominated_routes=dominated_ids,
            cost_variance=cost_variance,
            risk_variance=risk_variance,
            efficiency_correlation=efficiency_correlation,
            algorithm_performance=algorithm_performance,
            complexity_analysis=complexity_analysis,
            theoretical_bounds=theoretical_bounds
        )
    
    def _find_pareto_optimal_routes(self, routes: List[RouteAnalysis]) -> List[RouteAnalysis]:
        """Find Pareto optimal routes based on safety and efficiency."""
        if not routes:
            return []
        
        pareto_optimal = []
        
        for route in routes:
            is_dominated = False
            
            for other_route in routes:
                if other_route.route_id == route.route_id:
                    continue
                
                # Check if other_route dominates route
                if (other_route.safety_score >= route.safety_score and 
                    other_route.efficiency_score >= route.efficiency_score and
                    (other_route.safety_score > route.safety_score or 
                     other_route.efficiency_score > route.efficiency_score)):
                    is_dominated = True
                    break
            
            if not is_dominated:
                pareto_optimal.append(route)
        
        return pareto_optimal
    
    def _calculate_correlation(self, x: List[float], y: List[float]) -> float:
        """Calculate Pearson correlation coefficient."""
        if len(x) != len(y) or len(x) < 2:
            return 0.0
        
        n = len(x)
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(x[i] * y[i] for i in range(n))
        sum_x2 = sum(xi * xi for xi in x)
        sum_y2 = sum(yi * yi for yi in y)
        
        numerator = n * sum_xy - sum_x * sum_y
        denominator_x = n * sum_x2 - sum_x * sum_x
        denominator_y = n * sum_y2 - sum_y * sum_y
        
        # Handle edge cases to prevent division by zero or complex numbers
        if denominator_x <= 0 or denominator_y <= 0:
            return 0.0
        
        denominator = (denominator_x * denominator_y) ** 0.5
        
        if denominator == 0:
            return 0.0
        
        correlation = numerator / denominator
        
        # Clamp to valid range to handle floating point precision issues
        return max(-1.0, min(1.0, correlation))
    
    def _calculate_tradeoff_ratio(self, analyses: List[RouteAnalysis]) -> float:
        """Calculate the trade-off ratio between safety and efficiency."""
        valid_routes = [r for r in analyses if r.path]
        if not valid_routes:
            return 0.0
        
        safety_scores = [r.safety_score for r in valid_routes]
        efficiency_scores = [r.efficiency_score for r in valid_routes]
        
        if not safety_scores or not efficiency_scores:
            return 0.0
        
        safety_range = max(safety_scores) - min(safety_scores)
        efficiency_range = max(efficiency_scores) - min(efficiency_scores)
        
        return safety_range / efficiency_range if efficiency_range > 0 else 0.0
    
    def _estimate_time_complexity(self, routes: List[RouteAnalysis]) -> str:
        """Estimate time complexity based on performance data."""
        if not routes:
            return "O(1)"
        
        # Simplified complexity estimation based on nodes visited
        avg_nodes = statistics.mean([r.nodes_visited for r in routes])
        avg_edges = statistics.mean([r.edge_count for r in routes])
        
        if avg_nodes > 0 and avg_edges > 0:
            ratio = avg_nodes / avg_edges
            if ratio > 2:
                return "O(E log V)"
            else:
                return "O(V + E)"
        
        return "O(V)"
    
    def _estimate_space_complexity(self, routes: List[RouteAnalysis]) -> str:
        """Estimate space complexity based on route data."""
        if not routes:
            return "O(1)"
        
        avg_path_length = statistics.mean([len(r.path) for r in routes if r.path])
        
        if avg_path_length > 10:
            return "O(V + E)"
        else:
            return "O(V)"
    
    def _analyze_graph_size_impact(self, routes: List[RouteAnalysis]) -> Dict[str, float]:
        """Analyze the impact of graph size on performance."""
        if not routes:
            return {}
        
        path_lengths = [len(r.path) for r in routes if r.path]
        computation_times = [r.computation_time for r in routes]
        
        if path_lengths and computation_times:
            correlation = self._calculate_correlation(path_lengths, computation_times)
            return {
                "size_time_correlation": correlation,
                "average_path_length": statistics.mean(path_lengths),
                "path_length_variance": statistics.variance(path_lengths) if len(path_lengths) > 1 else 0.0
            }
        
        return {}
    
    def _analyze_disaster_impact(self, routes: List[RouteAnalysis]) -> Dict[str, Any]:
        """Analyze the impact of disasters on routing performance."""
        disaster_routes = [r for r in routes if r.disaster_applied]
        normal_routes = [r for r in routes if not r.disaster_applied]
        
        if not disaster_routes or not normal_routes:
            return {"insufficient_data": True}
        
        disaster_times = [r.computation_time for r in disaster_routes]
        normal_times = [r.computation_time for r in normal_routes]
        
        disaster_costs = [r.total_cost for r in disaster_routes]
        normal_costs = [r.total_cost for r in normal_routes]
        
        return {
            "computation_time_increase": statistics.mean(disaster_times) / statistics.mean(normal_times) if normal_times and statistics.mean(normal_times) > 0 else 1.0,
            "cost_increase": statistics.mean(disaster_costs) / statistics.mean(normal_costs) if normal_costs and statistics.mean(normal_costs) > 0 else 1.0,
            "disaster_routes_analyzed": len(disaster_routes),
            "normal_routes_analyzed": len(normal_routes)
        }
    
    def _analyze_scalability(self, routes: List[RouteAnalysis]) -> Dict[str, float]:
        """Analyze scalability metrics."""
        if not routes:
            return {}
        
        computation_times = [r.computation_time for r in routes]
        nodes_visited = [r.nodes_visited for r in routes]
        
        return {
            "time_scalability_factor": max(computation_times) / min(computation_times) if computation_times and min(computation_times) > 0 else 1.0,
            "space_scalability_factor": max(nodes_visited) / min(nodes_visited) if nodes_visited and min(nodes_visited) > 0 else 1.0,
            "performance_consistency": 1.0 / (statistics.stdev(computation_times) + 0.001) if len(computation_times) > 1 else 1.0
        }
    
    def _calculate_theoretical_bounds(self, routes: List[RouteAnalysis]) -> Dict[str, float]:
        """Calculate theoretical performance bounds."""
        if not routes:
            return {}
        
        costs = [r.total_cost for r in routes if r.path]
        distances = [r.total_distance for r in routes if r.path]
        
        if not costs or not distances:
            return {}
        
        return {
            "minimum_cost_bound": min(costs),
            "maximum_cost_bound": max(costs),
            "minimum_distance_bound": min(distances),
            "maximum_distance_bound": max(distances),
            "cost_optimality_ratio": min(costs) / max(costs) if max(costs) > 0 else 1.0,
            "distance_optimality_ratio": min(distances) / max(distances) if max(distances) > 0 else 1.0
        }
    
    def __str__(self) -> str:
        """String representation of the analysis engine."""
        return f"AnalysisEngine(safety_weight={self.safety_weight}, efficiency_weight={self.efficiency_weight})"
    
    def __repr__(self) -> str:
        return self.__str__()