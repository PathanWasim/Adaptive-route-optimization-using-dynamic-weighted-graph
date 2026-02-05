"""
Unit tests for AnalysisEngine class.
"""

import pytest
from datetime import datetime
from disaster_evacuation.analysis import AnalysisEngine
from disaster_evacuation.graph import GraphManager
from disaster_evacuation.pathfinding import PathfinderEngine
from disaster_evacuation.models import (
    DisasterEvent, DisasterType, VertexType, PathResult, Edge
)


class TestAnalysisEngine:
    """Test suite for AnalysisEngine class."""
    
    @pytest.fixture
    def sample_graph(self):
        """Create a sample graph for testing."""
        graph = GraphManager()
        
        # Add vertices
        graph.add_vertex("A", VertexType.INTERSECTION, (0.0, 0.0))
        graph.add_vertex("B", VertexType.INTERSECTION, (1.0, 0.0))
        graph.add_vertex("C", VertexType.SHELTER, (2.0, 0.0), capacity=100)
        graph.add_vertex("D", VertexType.EVACUATION_POINT, (1.0, 1.0), capacity=500)
        graph.add_vertex("E", VertexType.INTERSECTION, (0.5, 0.5))
        
        # Add edges
        graph.add_edge("A", "B", 1.0, 0.1, 0.2)
        graph.add_edge("B", "C", 1.0, 0.2, 0.1)
        graph.add_edge("A", "D", 1.4, 0.3, 0.3)
        graph.add_edge("C", "D", 1.4, 0.1, 0.4)
        graph.add_edge("A", "E", 0.7, 0.1, 0.1)
        graph.add_edge("E", "B", 0.7, 0.1, 0.1)
        
        return graph
    
    @pytest.fixture
    def analysis_engine(self):
        """Create an AnalysisEngine instance."""
        return AnalysisEngine()
    
    @pytest.fixture
    def sample_disaster(self):
        """Create a sample disaster event."""
        return DisasterEvent(
            disaster_type=DisasterType.FLOOD,
            epicenter=(0.5, 0.5),
            severity=0.7,
            max_effect_radius=1.5,
            start_time=datetime.now()
        )
    
    @pytest.fixture
    def sample_path_result(self, sample_graph):
        """Create a sample PathResult."""
        pathfinder = PathfinderEngine()
        return pathfinder.find_shortest_path(sample_graph, "A", "C")
    
    def test_initialization(self, analysis_engine):
        """Test AnalysisEngine initialization."""
        assert analysis_engine.safety_weight == 0.4
        assert analysis_engine.efficiency_weight == 0.3
        assert analysis_engine.robustness_weight == 0.3
        assert analysis_engine.high_risk_threshold == 0.5
        assert analysis_engine.critical_edge_threshold == 0.8
        assert analysis_engine.efficiency_threshold == 0.7
    
    def test_analyze_single_route_success(self, analysis_engine, sample_graph, sample_path_result):
        """Test successful single route analysis."""
        analysis = analysis_engine.analyze_single_route(
            sample_graph, sample_path_result, "test_route"
        )
        
        assert analysis.route_id == "test_route"
        assert analysis.path == sample_path_result.path
        assert analysis.total_cost == sample_path_result.total_cost
        assert analysis.edge_count == len(sample_path_result.edges_traversed)
        assert analysis.computation_time == sample_path_result.computation_time
        assert analysis.nodes_visited == sample_path_result.nodes_visited
        
        # Check that scores are calculated
        assert 0.0 <= analysis.safety_score <= 1.0
        assert 0.0 <= analysis.efficiency_score <= 1.0
        assert 0.0 <= analysis.robustness_score <= 1.0
        
        # Check that metrics are present
        assert isinstance(analysis.risk_distribution, list)
        assert analysis.blocked_roads_avoided >= 0
        assert analysis.high_risk_roads_used >= 0
        assert analysis.alternative_paths_available >= 0
        assert isinstance(analysis.critical_edges, list)
        
        assert analysis.disaster_applied is False
        assert analysis.disaster_type is None
        assert analysis.disaster_severity is None
    
    def test_analyze_single_route_with_disaster(self, analysis_engine, sample_graph, 
                                              sample_path_result, sample_disaster):
        """Test single route analysis with disaster context."""
        analysis = analysis_engine.analyze_single_route(
            sample_graph, sample_path_result, "disaster_route", sample_disaster
        )
        
        assert analysis.route_id == "disaster_route"
        assert analysis.disaster_applied is True
        assert analysis.disaster_type == "flood"
        assert analysis.disaster_severity == 0.7
        
        # Should still have valid scores
        assert 0.0 <= analysis.safety_score <= 1.0
        assert 0.0 <= analysis.efficiency_score <= 1.0
        assert 0.0 <= analysis.robustness_score <= 1.0
    
    def test_analyze_single_route_failed(self, analysis_engine, sample_graph):
        """Test analysis of failed route."""
        # Create a failed PathResult
        failed_result = PathResult(
            found=False,
            path=[],
            total_cost=float('inf'),
            edges_traversed=[],
            computation_time=0.1,
            nodes_visited=5,
            error_message="No path found"
        )
        
        analysis = analysis_engine.analyze_single_route(
            sample_graph, failed_result, "failed_route"
        )
        
        assert analysis.route_id == "failed_route"
        assert analysis.path == []
        assert analysis.total_cost == float('inf')
        assert analysis.edge_count == 0
        assert analysis.safety_score == 0.0
        assert analysis.efficiency_score == 0.0
        assert analysis.robustness_score == 0.0
    
    def test_compare_routing_strategies(self, analysis_engine, sample_graph):
        """Test comparison of multiple routing strategies."""
        strategies = [
            {"id": "normal", "disaster": None},
            {"id": "flood_aware", "disaster": DisasterEvent(
                disaster_type=DisasterType.FLOOD,
                epicenter=(1.0, 0.5),
                severity=0.6,
                max_effect_radius=1.0,
                start_time=datetime.now()
            )}
        ]
        
        comparison = analysis_engine.compare_routing_strategies(
            sample_graph, "A", "C", strategies
        )
        
        assert comparison.analysis_id == "analysis_2_routes"
        assert len(comparison.routes) == 2
        assert len(comparison.pareto_optimal_routes) >= 1
        assert comparison.cost_variance >= 0.0
        assert comparison.risk_variance >= 0.0
        
        # Check that algorithm performance metrics are present
        assert "average_computation_time" in comparison.algorithm_performance
        assert "average_nodes_visited" in comparison.algorithm_performance
        assert "time_complexity_estimate" in comparison.algorithm_performance
        assert "space_complexity_estimate" in comparison.algorithm_performance
    
    def test_evaluate_safety_efficiency_tradeoff(self, analysis_engine, sample_graph):
        """Test safety vs efficiency trade-off evaluation."""
        # Create some sample analyses
        pathfinder = PathfinderEngine()
        
        # Normal route
        normal_result = pathfinder.find_shortest_path(sample_graph, "A", "C")
        normal_analysis = analysis_engine.analyze_single_route(
            sample_graph, normal_result, "normal"
        )
        
        # Alternative route
        alt_result = pathfinder.find_shortest_path(sample_graph, "A", "D")
        alt_analysis = analysis_engine.analyze_single_route(
            sample_graph, alt_result, "alternative"
        )
        
        analyses = [normal_analysis, alt_analysis]
        tradeoff = analysis_engine.evaluate_safety_efficiency_tradeoff(analyses)
        
        assert "safety_efficiency_correlation" in tradeoff
        assert "pareto_optimal_routes" in tradeoff
        assert "safety_range" in tradeoff
        assert "efficiency_range" in tradeoff
        assert "trade_off_ratio" in tradeoff
        assert "dominated_solutions" in tradeoff
        
        # Check range structures
        assert "min" in tradeoff["safety_range"]
        assert "max" in tradeoff["safety_range"]
        assert "mean" in tradeoff["safety_range"]
        assert "std" in tradeoff["safety_range"]
    
    def test_evaluate_safety_efficiency_tradeoff_empty(self, analysis_engine):
        """Test trade-off evaluation with empty analyses."""
        tradeoff = analysis_engine.evaluate_safety_efficiency_tradeoff([])
        
        assert "error" in tradeoff
        assert tradeoff["error"] == "No analyses provided"
    
    def test_generate_performance_report(self, analysis_engine, sample_graph):
        """Test performance report generation."""
        strategies = [
            {"id": "strategy1", "disaster": None},
            {"id": "strategy2", "disaster": None}
        ]
        
        comparison = analysis_engine.compare_routing_strategies(
            sample_graph, "A", "C", strategies
        )
        
        report = analysis_engine.generate_performance_report(comparison)
        
        # Check main sections
        assert "executive_summary" in report
        assert "algorithm_performance" in report
        assert "route_quality" in report
        assert "safety_analysis" in report
        assert "efficiency_analysis" in report
        assert "robustness_analysis" in report
        assert "trade_off_analysis" in report
        assert "theoretical_bounds" in report
        
        # Check executive summary
        summary = report["executive_summary"]
        assert "total_strategies_analyzed" in summary
        assert "successful_routes" in summary
        assert "pareto_optimal_solutions" in summary
        assert "best_safety_route" in summary
        assert "best_efficiency_route" in summary
        assert "most_robust_route" in summary
        
        # Check algorithm performance
        algo_perf = report["algorithm_performance"]
        assert "computation_time" in algo_perf
        assert "nodes_visited" in algo_perf
        assert "complexity_analysis" in algo_perf
        
        # Check statistical measures
        assert "mean" in algo_perf["computation_time"]
        assert "median" in algo_perf["computation_time"]
        assert "std" in algo_perf["computation_time"]
    
    def test_generate_performance_report_empty(self, analysis_engine):
        """Test performance report with no valid routes."""
        from disaster_evacuation.analysis.analysis_engine import ComparativeAnalysis
        
        empty_comparison = ComparativeAnalysis(
            analysis_id="empty",
            routes=[],
            safety_vs_efficiency_tradeoff={},
            pareto_optimal_routes=[],
            dominated_routes=[],
            cost_variance=0.0,
            risk_variance=0.0,
            efficiency_correlation=0.0,
            algorithm_performance={},
            complexity_analysis={},
            theoretical_bounds={}
        )
        
        report = analysis_engine.generate_performance_report(empty_comparison)
        
        assert "error" in report
        assert report["error"] == "No valid routes to analyze"
    
    def test_pareto_optimal_routes(self, analysis_engine):
        """Test Pareto optimal route identification."""
        from disaster_evacuation.analysis.analysis_engine import RouteAnalysis
        
        # Create sample route analyses with different trade-offs
        routes = [
            RouteAnalysis(
                route_id="high_safety", path=["A", "B"], total_cost=10.0,
                total_distance=8.0, total_risk=2.0, total_congestion=1.0,
                edge_count=1, computation_time=0.1, nodes_visited=3,
                safety_score=0.9, risk_distribution=[0.1], blocked_roads_avoided=2,
                high_risk_roads_used=0, efficiency_score=0.6, distance_optimality=0.8,
                time_optimality=0.9, robustness_score=0.7, alternative_paths_available=1,
                critical_edges=[], disaster_applied=False
            ),
            RouteAnalysis(
                route_id="high_efficiency", path=["A", "C"], total_cost=8.0,
                total_distance=6.0, total_risk=4.0, total_congestion=2.0,
                edge_count=1, computation_time=0.05, nodes_visited=2,
                safety_score=0.5, risk_distribution=[0.4], blocked_roads_avoided=0,
                high_risk_roads_used=1, efficiency_score=0.9, distance_optimality=0.9,
                time_optimality=0.95, robustness_score=0.6, alternative_paths_available=0,
                critical_edges=[], disaster_applied=False
            ),
            RouteAnalysis(
                route_id="dominated", path=["A", "D"], total_cost=12.0,
                total_distance=10.0, total_risk=3.0, total_congestion=3.0,
                edge_count=2, computation_time=0.2, nodes_visited=4,
                safety_score=0.4, risk_distribution=[0.3, 0.3], blocked_roads_avoided=1,
                high_risk_roads_used=1, efficiency_score=0.4, distance_optimality=0.5,
                time_optimality=0.5, robustness_score=0.5, alternative_paths_available=1,
                critical_edges=[], disaster_applied=False
            )
        ]
        
        pareto_optimal = analysis_engine._find_pareto_optimal_routes(routes)
        pareto_ids = [r.route_id for r in pareto_optimal]
        
        # Should include high_safety and high_efficiency, but not dominated
        assert "high_safety" in pareto_ids
        assert "high_efficiency" in pareto_ids
        assert "dominated" not in pareto_ids
    
    def test_correlation_calculation(self, analysis_engine):
        """Test correlation coefficient calculation."""
        # Perfect positive correlation
        x1 = [1, 2, 3, 4, 5]
        y1 = [2, 4, 6, 8, 10]
        corr1 = analysis_engine._calculate_correlation(x1, y1)
        assert abs(corr1 - 1.0) < 0.001
        
        # Perfect negative correlation
        x2 = [1, 2, 3, 4, 5]
        y2 = [10, 8, 6, 4, 2]
        corr2 = analysis_engine._calculate_correlation(x2, y2)
        assert abs(corr2 - (-1.0)) < 0.001
        
        # No correlation
        x3 = [1, 2, 3, 4, 5]
        y3 = [3, 3, 3, 3, 3]
        corr3 = analysis_engine._calculate_correlation(x3, y3)
        assert corr3 == 0.0
        
        # Edge cases
        assert analysis_engine._calculate_correlation([], []) == 0.0
        assert analysis_engine._calculate_correlation([1], [2]) == 0.0
        assert analysis_engine._calculate_correlation([1, 2], [3]) == 0.0
    
    def test_time_complexity_estimation(self, analysis_engine):
        """Test time complexity estimation."""
        from disaster_evacuation.analysis.analysis_engine import RouteAnalysis
        
        # High nodes to edges ratio (suggests priority queue usage)
        high_ratio_routes = [
            RouteAnalysis(
                route_id="test", path=["A", "B"], total_cost=1.0,
                total_distance=1.0, total_risk=0.1, total_congestion=0.1,
                edge_count=1, computation_time=0.1, nodes_visited=10,
                safety_score=0.8, risk_distribution=[], blocked_roads_avoided=0,
                high_risk_roads_used=0, efficiency_score=0.8, distance_optimality=0.8,
                time_optimality=0.8, robustness_score=0.8, alternative_paths_available=0,
                critical_edges=[], disaster_applied=False
            )
        ]
        
        complexity = analysis_engine._estimate_time_complexity(high_ratio_routes)
        assert complexity == "O(E log V)"
        
        # Low ratio
        low_ratio_routes = [
            RouteAnalysis(
                route_id="test", path=["A", "B"], total_cost=1.0,
                total_distance=1.0, total_risk=0.1, total_congestion=0.1,
                edge_count=5, computation_time=0.1, nodes_visited=3,
                safety_score=0.8, risk_distribution=[], blocked_roads_avoided=0,
                high_risk_roads_used=0, efficiency_score=0.8, distance_optimality=0.8,
                time_optimality=0.8, robustness_score=0.8, alternative_paths_available=0,
                critical_edges=[], disaster_applied=False
            )
        ]
        
        complexity = analysis_engine._estimate_time_complexity(low_ratio_routes)
        assert complexity == "O(V + E)"
        
        # Empty routes
        assert analysis_engine._estimate_time_complexity([]) == "O(1)"
    
    def test_string_representation(self, analysis_engine):
        """Test string representation of analysis engine."""
        str_repr = str(analysis_engine)
        
        assert "AnalysisEngine" in str_repr
        assert "safety_weight" in str_repr
        assert "efficiency_weight" in str_repr
    
    def test_disaster_impact_analysis(self, analysis_engine):
        """Test disaster impact analysis."""
        from disaster_evacuation.analysis.analysis_engine import RouteAnalysis
        
        # Create routes with and without disasters
        normal_route = RouteAnalysis(
            route_id="normal", path=["A", "B"], total_cost=5.0,
            total_distance=4.0, total_risk=1.0, total_congestion=1.0,
            edge_count=1, computation_time=0.1, nodes_visited=3,
            safety_score=0.8, risk_distribution=[], blocked_roads_avoided=0,
            high_risk_roads_used=0, efficiency_score=0.8, distance_optimality=0.8,
            time_optimality=0.8, robustness_score=0.8, alternative_paths_available=0,
            critical_edges=[], disaster_applied=False
        )
        
        disaster_route = RouteAnalysis(
            route_id="disaster", path=["A", "C"], total_cost=8.0,
            total_distance=6.0, total_risk=2.0, total_congestion=2.0,
            edge_count=1, computation_time=0.2, nodes_visited=4,
            safety_score=0.6, risk_distribution=[], blocked_roads_avoided=1,
            high_risk_roads_used=1, efficiency_score=0.6, distance_optimality=0.6,
            time_optimality=0.6, robustness_score=0.6, alternative_paths_available=0,
            critical_edges=[], disaster_applied=True, disaster_type="flood",
            disaster_severity=0.7
        )
        
        routes = [normal_route, disaster_route]
        impact = analysis_engine._analyze_disaster_impact(routes)
        
        assert "computation_time_increase" in impact
        assert "cost_increase" in impact
        assert "disaster_routes_analyzed" in impact
        assert "normal_routes_analyzed" in impact
        
        assert impact["computation_time_increase"] == 2.0  # 0.2 / 0.1
        assert impact["cost_increase"] == 1.6  # 8.0 / 5.0
        assert impact["disaster_routes_analyzed"] == 1
        assert impact["normal_routes_analyzed"] == 1
    
    def test_theoretical_bounds_calculation(self, analysis_engine):
        """Test theoretical bounds calculation."""
        from disaster_evacuation.analysis.analysis_engine import RouteAnalysis
        
        routes = [
            RouteAnalysis(
                route_id="route1", path=["A", "B"], total_cost=10.0,
                total_distance=8.0, total_risk=1.0, total_congestion=1.0,
                edge_count=1, computation_time=0.1, nodes_visited=3,
                safety_score=0.8, risk_distribution=[], blocked_roads_avoided=0,
                high_risk_roads_used=0, efficiency_score=0.8, distance_optimality=0.8,
                time_optimality=0.8, robustness_score=0.8, alternative_paths_available=0,
                critical_edges=[], disaster_applied=False
            ),
            RouteAnalysis(
                route_id="route2", path=["A", "C"], total_cost=15.0,
                total_distance=12.0, total_risk=2.0, total_congestion=2.0,
                edge_count=2, computation_time=0.2, nodes_visited=4,
                safety_score=0.6, risk_distribution=[], blocked_roads_avoided=0,
                high_risk_roads_used=1, efficiency_score=0.6, distance_optimality=0.6,
                time_optimality=0.6, robustness_score=0.6, alternative_paths_available=1,
                critical_edges=[], disaster_applied=False
            )
        ]
        
        bounds = analysis_engine._calculate_theoretical_bounds(routes)
        
        assert bounds["minimum_cost_bound"] == 10.0
        assert bounds["maximum_cost_bound"] == 15.0
        assert bounds["minimum_distance_bound"] == 8.0
        assert bounds["maximum_distance_bound"] == 12.0
        assert bounds["cost_optimality_ratio"] == 10.0 / 15.0
        assert bounds["distance_optimality_ratio"] == 8.0 / 12.0