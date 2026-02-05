"""
Property-based tests for input validation in RouteController.

**Property 8: Input Validation**
**Validates: Requirements 5.1, 5.3, 5.4**

This module tests that the RouteController properly validates all user inputs
and provides appropriate error messages for invalid inputs.
"""

import pytest
from hypothesis import given, strategies as st, assume, settings, HealthCheck
from datetime import datetime, timedelta
from disaster_evacuation.controller import RouteController
from disaster_evacuation.graph import GraphManager
from disaster_evacuation.models import DisasterEvent, DisasterType, VertexType


def create_sample_controller():
    """Create a sample controller for testing."""
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
    
    return RouteController(graph)


class TestPropertyInputValidation:
    """Property-based tests for input validation."""
    
    @given(st.text())
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_property_invalid_source_handling(self, invalid_source):
        """
        Property: Invalid source vertices should be properly rejected.
        
        **Validates: Requirements 5.1, 5.3**
        
        For any string that is not a valid vertex ID in the graph,
        the system should return a proper error response.
        """
        controller = create_sample_controller()
        
        # Skip valid vertex IDs
        assume(invalid_source not in ["A", "B", "C", "D"])
        
        result = controller.compute_route(invalid_source, "C")
        
        # Should fail with appropriate error
        assert result["success"] is False
        assert result["error_type"] in ["invalid_source", "source_not_found"]
        assert "error" in result
        assert isinstance(result["error"], str)
        assert len(result["error"]) > 0
    
    @given(st.text())
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_property_invalid_destination_handling(self, invalid_destination):
        """
        Property: Invalid destination vertices should be properly rejected.
        
        **Validates: Requirements 5.1, 5.3**
        
        For any string that is not a valid vertex ID in the graph,
        the system should return a proper error response.
        """
        controller = create_sample_controller()
        
        # Skip valid vertex IDs
        assume(invalid_destination not in ["A", "B", "C", "D"])
        
        result = controller.compute_route("A", invalid_destination)
        
        # Should fail with appropriate error
        assert result["success"] is False
        assert result["error_type"] in ["invalid_destination", "destination_not_found"]
        assert "error" in result
        assert isinstance(result["error"], str)
        assert len(result["error"]) > 0
    
    @given(st.one_of(st.none(), st.text().filter(lambda x: len(x.strip()) == 0)))
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_property_empty_source_handling(self, empty_source):
        """
        Property: Empty or None source should be properly rejected.
        
        **Validates: Requirements 5.1, 5.4**
        
        Empty strings, None values, or whitespace-only strings should
        be rejected with appropriate error messages.
        """
        controller = create_sample_controller()
        
        result = controller.compute_route(empty_source, "C")
        
        # Should fail with appropriate error
        assert result["success"] is False
        assert result["error_type"] in ["invalid_source", "source_not_found"]
        assert "error" in result
        assert "empty" in result["error"].lower() or "cannot be" in result["error"].lower() or "not exist" in result["error"].lower()
    
    @given(st.one_of(st.none(), st.text().filter(lambda x: len(x.strip()) == 0)))
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_property_empty_destination_handling(self, empty_destination):
        """
        Property: Empty or None destination should be properly rejected.
        
        **Validates: Requirements 5.1, 5.4**
        
        Empty strings, None values, or whitespace-only strings should
        be rejected with appropriate error messages.
        """
        controller = create_sample_controller()
        
        result = controller.compute_route("A", empty_destination)
        
        # Should fail with appropriate error
        assert result["success"] is False
        assert result["error_type"] in ["invalid_destination", "destination_not_found"]
        assert "error" in result
        assert "empty" in result["error"].lower() or "cannot be" in result["error"].lower() or "not exist" in result["error"].lower()
    
    @given(st.sampled_from(["A", "B", "C", "D"]))
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_property_same_source_destination_handling(self, vertex_id):
        """
        Property: Same source and destination should be properly rejected.
        
        **Validates: Requirements 5.1, 5.4**
        
        When source and destination are the same, the system should
        reject the request with an appropriate error message.
        """
        controller = create_sample_controller()
        
        result = controller.compute_route(vertex_id, vertex_id)
        
        # Should fail with appropriate error
        assert result["success"] is False
        assert result["error_type"] == "same_location"
        assert "error" in result
        assert "same" in result["error"].lower()
    
    @given(
        st.floats(min_value=-1000.0, max_value=1000.0),
        st.floats(min_value=-1000.0, max_value=1000.0),
        st.floats(min_value=0.0, max_value=1.0),
        st.floats(min_value=0.1, max_value=10.0)  # Ensure radius is positive
    )
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_property_disaster_parameter_validation(self, x, y, severity, radius):
        """
        Property: Disaster parameters should be validated properly.
        
        **Validates: Requirements 5.1, 5.3**
        
        Disaster events with valid parameters should be accepted,
        while invalid parameters should be handled gracefully.
        """
        controller = create_sample_controller()
        
        try:
            # Create disaster with generated parameters
            disaster = DisasterEvent(
                disaster_type=DisasterType.FLOOD,
                epicenter=(x, y),
                severity=severity,
                max_effect_radius=radius,
                start_time=datetime.now()
            )
            
            result = controller.compute_route("A", "C", disaster)
            
            # Should either succeed or fail gracefully
            assert "success" in result
            assert isinstance(result["success"], bool)
            
            if not result["success"]:
                # If it fails, should have proper error information
                assert "error" in result
                assert "error_type" in result
                assert isinstance(result["error"], str)
                
        except ValueError:
            # ValueError is acceptable for invalid disaster parameters
            pass
    
    @given(st.integers())
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_property_non_string_input_handling(self, non_string_input):
        """
        Property: Non-string inputs should be handled gracefully.
        
        **Validates: Requirements 5.1, 5.4**
        
        When non-string values are passed as vertex IDs,
        the system should handle them gracefully.
        """
        controller = create_sample_controller()
        
        # Test with non-string source
        result = controller.compute_route(non_string_input, "C")
        
        # Should fail with appropriate error
        assert result["success"] is False
        assert "error" in result
        assert "error_type" in result
    
    @given(st.text(min_size=1, max_size=100))
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_property_error_message_quality(self, invalid_input):
        """
        Property: Error messages should be informative and non-empty.
        
        **Validates: Requirements 5.4**
        
        All error messages should be non-empty strings that provide
        useful information to the user.
        """
        controller = create_sample_controller()
        
        # Skip valid inputs
        assume(invalid_input not in ["A", "B", "C", "D"])
        
        result = controller.compute_route(invalid_input, "C")
        
        if not result["success"]:
            error_msg = result["error"]
            
            # Error message should be informative
            assert isinstance(error_msg, str)
            assert len(error_msg) > 0
            assert len(error_msg.strip()) > 0
            
            # Should contain relevant information
            assert any(word in error_msg.lower() for word in [
                "not", "exist", "invalid", "empty", "cannot", "error", "found"
            ])
    
    @given(
        st.sampled_from(["A", "B", "C", "D"]),
        st.sampled_from(["A", "B", "C", "D"])
    )
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_property_valid_input_acceptance(self, source, destination):
        """
        Property: Valid inputs should be accepted and processed.
        
        **Validates: Requirements 5.1**
        
        When both source and destination are valid vertex IDs and different,
        the system should accept the input and attempt route computation.
        """
        controller = create_sample_controller()
        
        # Skip same source and destination
        assume(source != destination)
        
        result = controller.compute_route(source, destination)
        
        # Should have proper response structure
        assert "success" in result
        assert isinstance(result["success"], bool)
        
        if result["success"]:
            # Successful response should have required fields
            assert "path" in result
            assert "total_cost" in result
            assert "source" in result
            assert "destination" in result
        else:
            # Failed response should have error information
            assert "error" in result
            assert "error_type" in result
    
    def test_property_evacuation_options_input_validation(self):
        """
        Property: Evacuation options should validate source input.
        
        **Validates: Requirements 5.1, 5.3**
        
        The get_evacuation_options method should properly validate
        the source vertex input.
        """
        controller = create_sample_controller()
        
        # Test with invalid source
        result = controller.get_evacuation_options("INVALID")
        
        assert result["success"] is False
        assert result["error_type"] == "source_not_found"
        assert "error" in result
        
        # Test with valid source
        result = controller.get_evacuation_options("A")
        
        assert result["success"] is True
        assert "evacuation_options" in result
        assert "total_options" in result
    
    @given(st.text())
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_property_comparison_input_validation(self, invalid_input):
        """
        Property: Route comparison should validate inputs properly.
        
        **Validates: Requirements 5.1, 5.3**
        
        The compare_routes method should validate both source and
        destination inputs properly.
        """
        controller = create_sample_controller()
        
        # Skip valid vertex IDs
        assume(invalid_input not in ["A", "B", "C", "D"])
        
        # Test invalid source
        result = controller.compare_routes(invalid_input, "C")
        
        assert result["success"] is False
        assert "error" in result
        assert "error_type" in result
        
        # Test invalid destination
        result = controller.compare_routes("A", invalid_input)
        
        assert result["success"] is False
        assert "error" in result
        assert "error_type" in result
    
    @given(st.one_of(st.none(), st.integers(), st.floats(), st.booleans()))
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_property_type_safety(self, wrong_type_input):
        """
        Property: System should handle wrong input types gracefully.
        
        **Validates: Requirements 5.4**
        
        When inputs of wrong types are provided, the system should
        handle them gracefully without crashing.
        """
        controller = create_sample_controller()
        
        try:
            result = controller.compute_route(wrong_type_input, "C")
            
            # Should either succeed or fail gracefully
            assert "success" in result
            assert isinstance(result["success"], bool)
            
            if not result["success"]:
                assert "error" in result
                assert "error_type" in result
                
        except (TypeError, ValueError, AttributeError):
            # Type errors are acceptable for wrong input types
            pass
    
    def test_property_consistent_error_structure(self):
        """
        Property: Error responses should have consistent structure.
        
        **Validates: Requirements 5.4**
        
        All error responses should follow the same structure
        for consistent error handling.
        """
        controller = create_sample_controller()
        
        # Test various invalid inputs
        test_cases = [
            ("", "C"),
            ("A", ""),
            ("INVALID", "C"),
            ("A", "INVALID"),
            ("A", "A"),
        ]
        
        for source, destination in test_cases:
            result = controller.compute_route(source, destination)
            
            if not result["success"]:
                # Check consistent error structure
                assert "success" in result
                assert "error" in result
                assert "error_type" in result
                assert "source" in result
                assert "destination" in result
                
                # Check types
                assert isinstance(result["success"], bool)
                assert isinstance(result["error"], str)
                assert isinstance(result["error_type"], str)
                assert result["success"] is False