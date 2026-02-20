"""
Unit tests for RouteController class.
"""

import pytest
from datetime import datetime
from disaster_evacuation.controller import RouteController
from disaster_evacuation.models import GraphManager
from disaster_evacuation.models import (
    Vertex, Edge, DisasterEvent, DisasterType, VertexType
)


class TestRouteController:
    """Test suite for RouteController class."""
    
    @pytest.fixture
    def simple_graph(self):
        """Create a simple graph for testing."""
        graph = GraphManager()
        
        # Add vertices
        graph.add_vertex("A", VertexType.INTERSECTION, (0.0, 0.0))
        graph.add_vertex("B", VertexType.INTERSECTION, (1.0, 0.0))
        graph.add_vertex("C", VertexType.SHELTER, (2.0, 0.0), capacity=100)
        graph.add_vertex("D", VertexType.EVACUATION_POINT, (1.0, 1.0), capacity=500)
        
        # Add edges
        graph.add_edge("A", "B", 1.0, 0.1, 0.2)
        graph.add_edge("B", "C", 1.0, 0.2, 0.1)
        graph.add_edge("A", "D", 1.4, 0.3, 0.3)
        graph.add_edge("C", "D", 1.4, 0.1, 0.4)
        
        return graph
    
    @pytest.fixture
    def controller(self, simple_graph):
        """Create a RouteController instance."""
        return RouteController(simple_graph)
    
    def test_initialization(self, simple_graph):
        """Test RouteController initialization."""
        controller = RouteController(simple_graph)
        
        assert controller._graph == simple_graph
        assert controller._disaster_model is not None
        assert controller._pathfinder is not None
        assert len(controller._active_disasters) == 0
    
    def test_compute_route_success(self, controller):
        """Test successful route computation."""
        result = controller.compute_route("A", "C")
        
        assert result["success"] is True
        assert "path" in result
        assert "total_cost" in result
        assert result["source"]["id"] == "A"
        assert result["destination"]["id"] == "C"
        assert "edges" in result
        assert "statistics" in result
    
    def test_compute_route_with_disaster(self, controller):
        """Test route computation with disaster effects."""
        disaster = DisasterEvent(
            disaster_type=DisasterType.FLOOD,
            epicenter=(0.5, 0.5),
            severity=0.7,
            max_effect_radius=2.0,
            start_time=datetime.now()
        )
        
        result = controller.compute_route("A", "C", disaster)
        
        assert result["success"] is True
        assert result["disaster_applied"] is True
        assert result["disaster_info"] is not None
        assert result["disaster_info"]["type"] == "flood"
    
    def test_compute_route_invalid_source(self, controller):
        """Test route computation with invalid source."""
        result = controller.compute_route("INVALID", "C")
        
        assert result["success"] is False
        assert result["error_type"] == "source_not_found"
        assert "does not exist" in result["error"]
    
    def test_compute_route_invalid_destination(self, controller):
        """Test route computation with invalid destination."""
        result = controller.compute_route("A", "INVALID")
        
        assert result["success"] is False
        assert result["error_type"] == "destination_not_found"
        assert "does not exist" in result["error"]
    
    def test_compute_route_empty_source(self, controller):
        """Test route computation with empty source."""
        result = controller.compute_route("", "C")
        
        assert result["success"] is False
        assert result["error_type"] == "invalid_source"
    
    def test_compute_route_same_location(self, controller):
        """Test route computation with same source and destination."""
        result = controller.compute_route("A", "A")
        
        assert result["success"] is False
        assert result["error_type"] == "same_location"
    
    def test_validate_route_request_valid(self, controller):
        """Test validation of valid route request."""
        result = controller._validate_route_request("A", "C")
        
        assert result["valid"] is True
    
    def test_validate_route_request_invalid_source(self, controller):
        """Test validation with invalid source."""
        result = controller._validate_route_request("INVALID", "C")
        
        assert result["valid"] is False
        assert result["error_type"] == "source_not_found"
    
    def test_format_successful_route(self, controller):
        """Test formatting of successful route."""
        result = controller.compute_route("A", "C")
        
        assert result["success"] is True
        assert "source" in result
        assert "destination" in result
        assert "edges" in result
        assert "statistics" in result
        assert "total_distance" in result["statistics"]
        assert "total_risk" in result["statistics"]
        assert "total_congestion" in result["statistics"]
        assert "computation_time" in result["statistics"]
    
    def test_compare_routes_without_disaster(self, controller):
        """Test route comparison without disaster."""
        result = controller.compare_routes("A", "C")
        
        assert result["success"] is True
        assert "normal_route" in result
        assert result["normal_route"]["found"] is True
    
    def test_compare_routes_with_disaster(self, controller):
        """Test route comparison with disaster."""
        disaster = DisasterEvent(
            disaster_type=DisasterType.FIRE,
            epicenter=(0.5, 0.5),
            severity=0.8,
            max_effect_radius=1.5,
            start_time=datetime.now()
        )
        
        result = controller.compare_routes("A", "C", disaster)
        
        assert result["success"] is True
        assert "normal_route" in result
        assert "disaster_aware_route" in result
        assert "disaster_info" in result
        assert "analysis" in result
    
    def test_get_evacuation_options(self, controller):
        """Test getting evacuation options."""
        result = controller.get_evacuation_options("A")
        
        assert result["success"] is True
        assert "evacuation_options" in result
        assert result["total_options"] >= 0
        assert "reachable_options" in result
    
    def test_get_evacuation_options_invalid_source(self, controller):
        """Test evacuation options with invalid source."""
        result = controller.get_evacuation_options("INVALID")
        
        assert result["success"] is False
        assert result["error_type"] == "source_not_found"
    
    def test_clear_disasters(self, controller):
        """Test clearing disaster effects."""
        disaster = DisasterEvent(
            disaster_type=DisasterType.EARTHQUAKE,
            epicenter=(1.0, 1.0),
            severity=0.6,
            max_effect_radius=2.0,
            start_time=datetime.now()
        )
        
        controller.compute_route("A", "C", disaster)
        assert len(controller.get_active_disasters()) > 0
        
        controller.clear_disasters()
        assert len(controller.get_active_disasters()) == 0
    
    def test_get_graph_summary(self, controller):
        """Test getting graph summary."""
        summary = controller.get_graph_summary()
        
        assert "total_vertices" in summary
        assert "total_edges" in summary
        assert "intersections" in summary
        assert "evacuation_points" in summary
        assert "shelters" in summary
        assert "active_disasters" in summary
        assert summary["total_vertices"] == 4
        assert summary["total_edges"] == 4
    
    def test_string_representation(self, controller):
        """Test string representation of controller."""
        str_repr = str(controller)
        
        assert "RouteController" in str_repr
        assert "graph_vertices" in str_repr
        assert "active_disasters" in str_repr


class TestRouteControllerDisconnectedGraph:
    """Test RouteController with disconnected graph."""
    
    @pytest.fixture
    def disconnected_graph(self):
        """Create a disconnected graph."""
        graph = GraphManager()
        
        # Component 1
        graph.add_vertex("A", VertexType.INTERSECTION, (0.0, 0.0))
        graph.add_vertex("B", VertexType.SHELTER, (1.0, 0.0), capacity=100)
        graph.add_edge("A", "B", 1.0, 0.1, 0.1)
        
        # Component 2 (disconnected)
        graph.add_vertex("C", VertexType.INTERSECTION, (5.0, 5.0))
        graph.add_vertex("D", VertexType.EVACUATION_POINT, (6.0, 5.0), capacity=200)
        graph.add_edge("C", "D", 1.0, 0.1, 0.1)
        
        return graph
    
    def test_compute_route_disconnected(self, disconnected_graph):
        """Test route computation on disconnected graph."""
        controller = RouteController(disconnected_graph)
        result = controller.compute_route("A", "D")
        
        assert result["success"] is False
        assert result["error_type"] == "no_path_found"
        assert "suggestion" in result


class TestRouteControllerOutputFormatting:
    """Test output formatting functionality."""
    
    @pytest.fixture
    def complex_graph(self):
        """Create a more complex graph for testing."""
        graph = GraphManager()
        
        # Create a grid-like structure
        graph.add_vertex("A", VertexType.INTERSECTION, (0.0, 0.0))
        graph.add_vertex("B", VertexType.INTERSECTION, (1.0, 0.0))
        graph.add_vertex("C", VertexType.INTERSECTION, (2.0, 0.0))
        graph.add_vertex("D", VertexType.INTERSECTION, (0.0, 1.0))
        graph.add_vertex("E", VertexType.SHELTER, (1.0, 1.0), capacity=150)
        graph.add_vertex("F", VertexType.EVACUATION_POINT, (2.0, 1.0), capacity=300)
        
        # Add edges
        graph.add_edge("A", "B", 1.0, 0.1, 0.1)
        graph.add_edge("B", "C", 1.0, 0.1, 0.1)
        graph.add_edge("A", "D", 1.0, 0.2, 0.2)
        graph.add_edge("D", "E", 1.0, 0.1, 0.1)
        graph.add_edge("B", "E", 1.4, 0.3, 0.3)
        graph.add_edge("E", "F", 1.0, 0.1, 0.1)
        graph.add_edge("C", "F", 1.0, 0.2, 0.2)
        
        return graph
    
    def test_output_contains_all_required_fields(self, complex_graph):
        """Test that output contains all required fields."""
        controller = RouteController(complex_graph)
        result = controller.compute_route("A", "F")
        
        # Check required fields
        required_fields = [
            "success", "path", "total_cost", "source", "destination",
            "edges", "avoided_dangerous_roads", "statistics", "disaster_applied"
        ]
        
        for field in required_fields:
            assert field in result, f"Missing required field: {field}"
    
    def test_source_destination_formatting(self, complex_graph):
        """Test source and destination formatting."""
        controller = RouteController(complex_graph)
        result = controller.compute_route("A", "F")
        
        # Check source formatting
        assert "id" in result["source"]
        assert "type" in result["source"]
        assert "coordinates" in result["source"]
        
        # Check destination formatting
        assert "id" in result["destination"]
        assert "type" in result["destination"]
        assert "coordinates" in result["destination"]
        assert "capacity" in result["destination"]
    
    def test_edge_formatting(self, complex_graph):
        """Test edge information formatting."""
        controller = RouteController(complex_graph)
        result = controller.compute_route("A", "F")
        
        assert len(result["edges"]) > 0
        
        for edge in result["edges"]:
            assert "from" in edge
            assert "to" in edge
            assert "distance" in edge
            assert "current_weight" in edge
            assert "is_blocked" in edge
    
    def test_statistics_formatting(self, complex_graph):
        """Test statistics formatting."""
        controller = RouteController(complex_graph)
        result = controller.compute_route("A", "F")
        
        stats = result["statistics"]
        
        assert "total_distance" in stats
        assert "total_risk" in stats
        assert "total_congestion" in stats
        assert "edge_count" in stats
        assert "computation_time" in stats
        assert "nodes_visited" in stats
        
        # Verify types
        assert isinstance(stats["total_distance"], (int, float))
        assert isinstance(stats["total_risk"], (int, float))
        assert isinstance(stats["edge_count"], int)
    
    def test_comparison_output_formatting(self, complex_graph):
        """Test comparison output formatting."""
        controller = RouteController(complex_graph)
        
        disaster = DisasterEvent(
            disaster_type=DisasterType.FLOOD,
            epicenter=(1.0, 0.5),
            severity=0.7,
            max_effect_radius=1.5,
            start_time=datetime.now()
        )
        
        result = controller.compare_routes("A", "F", disaster)
        
        assert "normal_route" in result
        assert "disaster_aware_route" in result
        assert "analysis" in result
        assert "disaster_info" in result
        
        # Check analysis fields
        if result["normal_route"]["found"] and result["disaster_aware_route"]["found"]:
            analysis = result["analysis"]
            assert "path_changed" in analysis
            assert "cost_increase" in analysis
            assert "cost_increase_percentage" in analysis
    
    def test_evacuation_options_formatting(self, complex_graph):
        """Test evacuation options formatting."""
        controller = RouteController(complex_graph)
        result = controller.get_evacuation_options("A")
        
        assert "evacuation_options" in result
        assert "total_options" in result
        assert "reachable_options" in result
        
        for option in result["evacuation_options"]:
            assert "destination" in option
            assert "type" in option
            assert "capacity" in option
            assert "reachable" in option
            
            if option["reachable"]:
                assert "cost" in option
                assert "distance" in option
                assert "path_length" in option
