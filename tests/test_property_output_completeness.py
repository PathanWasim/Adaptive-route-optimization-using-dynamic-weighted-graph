"""
Property-based tests for output completeness in RouteController.

**Property 9: Output Completeness**
**Validates: Requirements 5.2**

This module tests that the RouteController provides complete and consistent
output information for all successful route computations.
"""

import pytest
from hypothesis import given, strategies as st, assume, settings, HealthCheck
from datetime import datetime
from disaster_evacuation.controller import RouteController
from disaster_evacuation.models import GraphManager
from disaster_evacuation.models import DisasterEvent, DisasterType, VertexType


def create_sample_controller():
    """Create a sample controller for testing."""
    graph = GraphManager()
    
    # Add vertices
    graph.add_vertex("A", VertexType.INTERSECTION, (0.0, 0.0))
    graph.add_vertex("B", VertexType.INTERSECTION, (1.0, 0.0))
    graph.add_vertex("C", VertexType.SHELTER, (2.0, 0.0), capacity=100)
    graph.add_vertex("D", VertexType.EVACUATION_POINT, (1.0, 1.0), capacity=500)
    graph.add_vertex("E", VertexType.INTERSECTION, (0.5, 0.5))
    
    # Add edges to create multiple paths
    graph.add_edge("A", "B", 1.0, 0.1, 0.2)
    graph.add_edge("B", "C", 1.0, 0.2, 0.1)
    graph.add_edge("A", "D", 1.4, 0.3, 0.3)
    graph.add_edge("C", "D", 1.4, 0.1, 0.4)
    graph.add_edge("A", "E", 0.7, 0.1, 0.1)
    graph.add_edge("E", "B", 0.7, 0.1, 0.1)
    graph.add_edge("E", "C", 1.5, 0.2, 0.2)
    
    return RouteController(graph)


class TestPropertyOutputCompleteness:
    """Property-based tests for output completeness."""
    
    @given(
        st.sampled_from(["A", "B", "C", "D", "E"]),
        st.sampled_from(["A", "B", "C", "D", "E"])
    )
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_property_successful_route_completeness(self, source, destination):
        """
        Property: Successful routes should contain all required information.
        
        **Validates: Requirements 5.2**
        
        When a route is successfully computed, the output should contain
        all required fields with appropriate data types and values.
        """
        controller = create_sample_controller()
        
        # Skip same source and destination
        assume(source != destination)
        
        result = controller.compute_route(source, destination)
        
        if result["success"]:
            # Check required top-level fields
            required_fields = [
                "success", "path", "total_cost", "source", "destination",
                "edges", "avoided_dangerous_roads", "statistics", "disaster_applied"
            ]
            
            for field in required_fields:
                assert field in result, f"Missing required field: {field}"
            
            # Validate field types and content
            assert isinstance(result["success"], bool)
            assert result["success"] is True
            
            assert isinstance(result["path"], list)
            assert len(result["path"]) >= 2  # At least source and destination
            assert result["path"][0] == source
            assert result["path"][-1] == destination
            
            assert isinstance(result["total_cost"], (int, float))
            assert result["total_cost"] > 0
            
            # Validate source information
            assert isinstance(result["source"], dict)
            source_fields = ["id", "type", "coordinates"]
            for field in source_fields:
                assert field in result["source"], f"Missing source field: {field}"
            assert result["source"]["id"] == source
            
            # Validate destination information
            assert isinstance(result["destination"], dict)
            dest_fields = ["id", "type", "coordinates", "capacity"]
            for field in dest_fields:
                assert field in result["destination"], f"Missing destination field: {field}"
            assert result["destination"]["id"] == destination
            
            # Validate edges information
            assert isinstance(result["edges"], list)
            assert len(result["edges"]) == len(result["path"]) - 1
            
            for edge in result["edges"]:
                edge_fields = ["from", "to", "distance", "current_weight", "is_blocked"]
                for field in edge_fields:
                    assert field in edge, f"Missing edge field: {field}"
                assert isinstance(edge["distance"], (int, float))
                assert isinstance(edge["current_weight"], (int, float))
                assert isinstance(edge["is_blocked"], bool)
            
            # Validate statistics
            assert isinstance(result["statistics"], dict)
            stats_fields = [
                "total_distance", "total_risk", "total_congestion",
                "edge_count", "computation_time", "nodes_visited"
            ]
            for field in stats_fields:
                assert field in result["statistics"], f"Missing statistics field: {field}"
            
            # Validate avoided dangerous roads
            assert isinstance(result["avoided_dangerous_roads"], list)
            
            # Validate disaster information
            assert isinstance(result["disaster_applied"], bool)
    
    @given(
        st.sampled_from(["A", "B", "C", "D", "E"]),
        st.sampled_from(["A", "B", "C", "D", "E"]),
        st.sampled_from([DisasterType.FLOOD, DisasterType.FIRE, DisasterType.EARTHQUAKE]),
        st.floats(min_value=0.1, max_value=1.0),
        st.floats(min_value=0.5, max_value=3.0)
    )
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_property_disaster_route_completeness(self, source, destination, disaster_type, severity, radius):
        """
        Property: Routes with disasters should include disaster information.
        
        **Validates: Requirements 5.2**
        
        When a route is computed with disaster effects, the output should
        include complete disaster information and affected road details.
        """
        controller = create_sample_controller()
        
        # Skip same source and destination
        assume(source != destination)
        
        disaster = DisasterEvent(
            disaster_type=disaster_type,
            epicenter=(0.5, 0.5),
            severity=severity,
            max_effect_radius=radius,
            start_time=datetime.now()
        )
        
        result = controller.compute_route(source, destination, disaster)
        
        if result["success"]:
            # Should have disaster-specific fields
            assert result["disaster_applied"] is True
            assert "disaster_info" in result
            assert result["disaster_info"] is not None
            
            # Validate disaster info structure
            disaster_info = result["disaster_info"]
            disaster_fields = ["type", "epicenter", "severity", "effect_radius", "start_time"]
            for field in disaster_fields:
                assert field in disaster_info, f"Missing disaster info field: {field}"
            
            assert disaster_info["type"] == disaster_type.value
            assert isinstance(disaster_info["epicenter"], (list, tuple))
            assert len(disaster_info["epicenter"]) == 2
            assert isinstance(disaster_info["severity"], (int, float))
            assert 0.0 <= disaster_info["severity"] <= 1.0
            
            # Avoided dangerous roads should be properly formatted
            for avoided_road in result["avoided_dangerous_roads"]:
                road_fields = ["from", "to", "reason", "original_weight", "current_weight"]
                for field in road_fields:
                    assert field in avoided_road, f"Missing avoided road field: {field}"
                assert avoided_road["reason"] in ["blocked", "high_risk"]
    
    @given(
        st.sampled_from(["A", "B", "C", "D", "E"]),
        st.sampled_from(["A", "B", "C", "D", "E"])
    )
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_property_comparison_output_completeness(self, source, destination):
        """
        Property: Route comparisons should provide complete analysis.
        
        **Validates: Requirements 5.2**
        
        When comparing routes, the output should include both normal and
        disaster-aware routes with complete analysis information.
        """
        controller = create_sample_controller()
        
        # Skip same source and destination
        assume(source != destination)
        
        disaster = DisasterEvent(
            disaster_type=DisasterType.FLOOD,
            epicenter=(0.5, 0.5),
            severity=0.7,
            max_effect_radius=2.0,
            start_time=datetime.now()
        )
        
        result = controller.compare_routes(source, destination, disaster)
        
        if result["success"]:
            # Check required comparison fields
            comparison_fields = [
                "success", "source", "destination", "normal_route",
                "disaster_aware_route", "disaster_info"
            ]
            
            for field in comparison_fields:
                assert field in result, f"Missing comparison field: {field}"
            
            # Validate normal route structure
            normal_route = result["normal_route"]
            normal_fields = ["found", "path", "cost", "edge_count"]
            for field in normal_fields:
                assert field in normal_route, f"Missing normal route field: {field}"
            
            # Validate disaster-aware route structure
            disaster_route = result["disaster_aware_route"]
            disaster_fields = ["found", "path", "cost", "edge_count"]
            for field in disaster_fields:
                assert field in disaster_route, f"Missing disaster route field: {field}"
            
            # If both routes found, should have analysis
            if normal_route["found"] and disaster_route["found"]:
                assert "analysis" in result
                analysis = result["analysis"]
                analysis_fields = ["path_changed", "cost_increase", "cost_increase_percentage"]
                for field in analysis_fields:
                    assert field in analysis, f"Missing analysis field: {field}"
                
                assert isinstance(analysis["path_changed"], bool)
                assert isinstance(analysis["cost_increase"], (int, float))
                assert isinstance(analysis["cost_increase_percentage"], (int, float))
    
    @given(st.sampled_from(["A", "B", "C", "D", "E"]))
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_property_evacuation_options_completeness(self, source):
        """
        Property: Evacuation options should provide complete information.
        
        **Validates: Requirements 5.2**
        
        When getting evacuation options, the output should include complete
        information about all available evacuation points and shelters.
        """
        controller = create_sample_controller()
        
        result = controller.get_evacuation_options(source)
        
        if result["success"]:
            # Check required fields
            required_fields = [
                "success", "source", "evacuation_options", "total_options",
                "reachable_options", "disaster_applied"
            ]
            
            for field in required_fields:
                assert field in result, f"Missing evacuation options field: {field}"
            
            assert result["source"] == source
            assert isinstance(result["evacuation_options"], list)
            assert isinstance(result["total_options"], int)
            assert isinstance(result["reachable_options"], int)
            assert result["reachable_options"] <= result["total_options"]
            
            # Validate each evacuation option
            for option in result["evacuation_options"]:
                option_fields = ["destination", "type", "capacity", "coordinates", "reachable"]
                for field in option_fields:
                    assert field in option, f"Missing evacuation option field: {field}"
                
                assert option["type"] in ["shelter", "evacuation_point"]
                assert isinstance(option["capacity"], (int, type(None)))
                assert isinstance(option["coordinates"], (list, tuple))
                assert isinstance(option["reachable"], bool)
                
                if option["reachable"]:
                    reachable_fields = ["cost", "distance", "path_length"]
                    for field in reachable_fields:
                        assert field in option, f"Missing reachable option field: {field}"
                    assert isinstance(option["cost"], (int, float))
                    assert isinstance(option["distance"], (int, float))
                    assert isinstance(option["path_length"], int)
                else:
                    assert "reason" in option
                    assert isinstance(option["reason"], str)
    
    def test_property_error_response_completeness(self):
        """
        Property: Error responses should provide complete information.
        
        **Validates: Requirements 5.2**
        
        When operations fail, error responses should include all necessary
        information for debugging and user feedback.
        """
        controller = create_sample_controller()
        
        # Test various error scenarios
        error_scenarios = [
            ("INVALID", "C"),  # Invalid source
            ("A", "INVALID"),  # Invalid destination
            ("A", "A"),        # Same source and destination
            ("", "C"),         # Empty source
            ("A", ""),         # Empty destination
        ]
        
        for source, destination in error_scenarios:
            result = controller.compute_route(source, destination)
            
            if not result["success"]:
                # Check required error fields
                error_fields = ["success", "error", "error_type", "source", "destination"]
                for field in error_fields:
                    assert field in result, f"Missing error field: {field}"
                
                assert result["success"] is False
                assert isinstance(result["error"], str)
                assert len(result["error"]) > 0
                assert isinstance(result["error_type"], str)
                assert len(result["error_type"]) > 0
    
    @given(
        st.sampled_from(["A", "B", "C", "D", "E"]),
        st.sampled_from(["A", "B", "C", "D", "E"])
    )
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_property_path_consistency(self, source, destination):
        """
        Property: Path information should be internally consistent.
        
        **Validates: Requirements 5.2**
        
        The path, edges, and statistics should be consistent with each other
        in successful route computations.
        """
        controller = create_sample_controller()
        
        # Skip same source and destination
        assume(source != destination)
        
        result = controller.compute_route(source, destination)
        
        if result["success"]:
            path = result["path"]
            edges = result["edges"]
            stats = result["statistics"]
            
            # Path and edges consistency
            assert len(edges) == len(path) - 1
            assert stats["edge_count"] == len(edges)
            
            # Verify path connectivity through edges
            for i, edge in enumerate(edges):
                assert edge["from"] == path[i]
                assert edge["to"] == path[i + 1]
            
            # Statistics consistency
            total_distance = sum(edge["distance"] for edge in edges)
            assert abs(stats["total_distance"] - total_distance) < 0.001
            
            # Cost should be reasonable
            assert result["total_cost"] >= stats["total_distance"]
    
    @given(
        st.sampled_from(["A", "B", "C", "D", "E"]),
        st.sampled_from(["A", "B", "C", "D", "E"])
    )
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_property_coordinate_information(self, source, destination):
        """
        Property: Coordinate information should be complete and valid.
        
        **Validates: Requirements 5.2**
        
        All vertex coordinate information should be properly included
        in the output for visualization purposes.
        """
        controller = create_sample_controller()
        
        # Skip same source and destination
        assume(source != destination)
        
        result = controller.compute_route(source, destination)
        
        if result["success"]:
            # Source coordinates
            source_coords = result["source"]["coordinates"]
            assert isinstance(source_coords, (list, tuple))
            assert len(source_coords) == 2
            assert all(isinstance(coord, (int, float)) for coord in source_coords)
            
            # Destination coordinates
            dest_coords = result["destination"]["coordinates"]
            assert isinstance(dest_coords, (list, tuple))
            assert len(dest_coords) == 2
            assert all(isinstance(coord, (int, float)) for coord in dest_coords)
    
    def test_property_graph_summary_completeness(self):
        """
        Property: Graph summary should provide complete network information.
        
        **Validates: Requirements 5.2**
        
        The graph summary should include all relevant network statistics
        for system monitoring and analysis.
        """
        controller = create_sample_controller()
        
        summary = controller.get_graph_summary()
        
        # Check required summary fields
        summary_fields = [
            "total_vertices", "total_edges", "intersections",
            "evacuation_points", "shelters", "average_degree", "active_disasters"
        ]
        
        for field in summary_fields:
            assert field in summary, f"Missing summary field: {field}"
        
        # Validate field types and values
        assert isinstance(summary["total_vertices"], int)
        assert isinstance(summary["total_edges"], int)
        assert isinstance(summary["intersections"], int)
        assert isinstance(summary["evacuation_points"], int)
        assert isinstance(summary["shelters"], int)
        assert isinstance(summary["average_degree"], (int, float))
        assert isinstance(summary["active_disasters"], int)
        
        # Logical consistency
        assert summary["total_vertices"] > 0
        assert summary["total_edges"] > 0
        assert summary["intersections"] + summary["evacuation_points"] + summary["shelters"] == summary["total_vertices"]
        assert summary["active_disasters"] >= 0
    
    @given(
        st.sampled_from(["A", "B", "C", "D", "E"]),
        st.sampled_from(["A", "B", "C", "D", "E"])
    )
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_property_timing_information(self, source, destination):
        """
        Property: Timing information should be included in results.
        
        **Validates: Requirements 5.2**
        
        All route computations should include timing information
        for performance analysis.
        """
        controller = create_sample_controller()
        
        # Skip same source and destination
        assume(source != destination)
        
        result = controller.compute_route(source, destination)
        
        # Both successful and failed results should have timing info
        if result["success"]:
            assert "computation_time" in result["statistics"]
            assert isinstance(result["statistics"]["computation_time"], (int, float))
            assert result["statistics"]["computation_time"] >= 0
        else:
            # Failed results might have timing info too
            if "computation_time" in result:
                assert isinstance(result["computation_time"], (int, float))
                assert result["computation_time"] >= 0