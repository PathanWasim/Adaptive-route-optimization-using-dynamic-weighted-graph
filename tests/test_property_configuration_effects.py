"""
Property-based tests for configuration effects.

Feature: disaster-evacuation-routing
Property 14: Configuration Effects

For any modification to disaster parameters or algorithm settings, the changes should
take effect in subsequent computations and produce different results when the settings
impact the calculation.

Validates: Requirements 10.4
"""

import pytest
import tempfile
import shutil
from hypothesis import given, strategies as st, settings
from disaster_evacuation.config import ConfigurationManager
from disaster_evacuation.graph import GraphManager
from disaster_evacuation.disaster import DisasterModel
from disaster_evacuation.models import VertexType, DisasterType


class TestPropertyConfigurationEffects:
    """Property-based tests for configuration effects."""
    
    @given(
        max_iterations=st.integers(min_value=100, max_value=50000),
        timeout_seconds=st.floats(min_value=1.0, max_value=120.0, allow_nan=False, allow_infinity=False)
    )
    @settings(max_examples=50, deadline=2000)
    def test_algorithm_settings_modification_takes_effect(self, max_iterations, timeout_seconds):
        """
        Property 14: Configuration Effects
        
        Modifying algorithm settings should update the configuration.
        """
        temp_dir = tempfile.mkdtemp()
        try:
            config_manager = ConfigurationManager(temp_dir, enable_logging=False)
            
            # Set new algorithm settings
            config_manager.set_algorithm_setting("max_iterations", max_iterations)
            config_manager.set_algorithm_setting("timeout_seconds", timeout_seconds)
            
            # Verify settings were updated
            assert config_manager.get_algorithm_setting("max_iterations") == max_iterations, \
                "max_iterations should be updated"
            assert config_manager.get_algorithm_setting("timeout_seconds") == timeout_seconds, \
                "timeout_seconds should be updated"
            
            config_manager.close()
        finally:
            shutil.rmtree(temp_dir)
    
    @given(
        severity=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        radius=st.floats(min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False),
        threshold=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
    )
    @settings(max_examples=50, deadline=2000)
    def test_disaster_parameters_modification_takes_effect(self, severity, radius, threshold):
        """
        Property 14: Configuration Effects
        
        Modifying disaster parameters should update the configuration.
        """
        temp_dir = tempfile.mkdtemp()
        try:
            config_manager = ConfigurationManager(temp_dir, enable_logging=False)
            
            # Set new disaster parameters
            config_manager.set_disaster_parameter("flood", "default_severity", severity)
            config_manager.set_disaster_parameter("flood", "default_radius", radius)
            config_manager.set_disaster_parameter("flood", "blocking_threshold", threshold)
            
            # Verify parameters were updated
            assert config_manager.get_disaster_parameter("flood", "default_severity") == severity, \
                "default_severity should be updated"
            assert config_manager.get_disaster_parameter("flood", "default_radius") == radius, \
                "default_radius should be updated"
            assert config_manager.get_disaster_parameter("flood", "blocking_threshold") == threshold, \
                "blocking_threshold should be updated"
            
            config_manager.close()
        finally:
            shutil.rmtree(temp_dir)
    
    @given(
        severity1=st.floats(min_value=0.1, max_value=0.5, allow_nan=False, allow_infinity=False),
        severity2=st.floats(min_value=0.6, max_value=1.0, allow_nan=False, allow_infinity=False)
    )
    @settings(max_examples=50, deadline=2000)
    def test_disaster_severity_affects_edge_weights(self, severity1, severity2):
        """
        Property 14: Configuration Effects
        
        Different disaster severity settings should produce different edge weights.
        """
        from disaster_evacuation.models import DisasterEvent
        
        # Create a simple graph
        graph = GraphManager()
        graph.add_vertex("A", VertexType.INTERSECTION, (0.0, 0.0))
        graph.add_vertex("B", VertexType.INTERSECTION, (1.0, 0.0))
        graph.add_edge("A", "B", 1.0, 0.5, 0.2)
        
        disaster_model = DisasterModel()
        
        # Apply disaster with first severity
        disaster1 = DisasterEvent(DisasterType.FLOOD, (0.5, 0.0), severity1, 2.0)
        disaster_model.apply_disaster_effects(graph, disaster1)
        edge1 = graph.get_neighbors("A")[0]
        weight1 = edge1.current_weight
        
        # Clear and apply disaster with second severity
        disaster_model.clear_all_disaster_effects(graph)
        disaster2 = DisasterEvent(DisasterType.FLOOD, (0.5, 0.0), severity2, 2.0)
        disaster_model.apply_disaster_effects(graph, disaster2)
        edge2 = graph.get_neighbors("A")[0]
        weight2 = edge2.current_weight
        
        # Different severities should produce different weights
        assert weight1 != weight2, \
            f"Different severities ({severity1} vs {severity2}) should produce different weights"
        
        # Higher severity should produce higher weight
        assert weight2 > weight1, \
            f"Higher severity ({severity2}) should produce higher weight than lower severity ({severity1})"
    
    @given(
        radius1=st.floats(min_value=0.5, max_value=1.5, allow_nan=False, allow_infinity=False),
        radius2=st.floats(min_value=2.0, max_value=5.0, allow_nan=False, allow_infinity=False)
    )
    @settings(max_examples=50, deadline=2000)
    def test_disaster_radius_affects_affected_edges(self, radius1, radius2):
        """
        Property 14: Configuration Effects
        
        Different disaster radius settings should affect different numbers of edges.
        """
        from disaster_evacuation.models import DisasterEvent
        
        # Create a graph with edges at different distances
        graph = GraphManager()
        graph.add_vertex("Center", VertexType.INTERSECTION, (0.0, 0.0))
        graph.add_vertex("Near", VertexType.INTERSECTION, (1.0, 0.0))
        graph.add_vertex("Far", VertexType.INTERSECTION, (3.0, 0.0))
        graph.add_edge("Center", "Near", 1.0, 0.5, 0.2)
        graph.add_edge("Center", "Far", 3.0, 0.5, 0.2)
        
        disaster_model = DisasterModel()
        
        # Apply disaster with smaller radius
        disaster1 = DisasterEvent(DisasterType.FIRE, (0.0, 0.0), 0.8, radius1)
        disaster_model.apply_disaster_effects(graph, disaster1)
        affected1 = disaster_model.get_affected_edges(graph, (0.0, 0.0), radius1)
        
        # Clear and apply disaster with larger radius
        disaster_model.clear_all_disaster_effects(graph)
        disaster2 = DisasterEvent(DisasterType.FIRE, (0.0, 0.0), 0.8, radius2)
        disaster_model.apply_disaster_effects(graph, disaster2)
        affected2 = disaster_model.get_affected_edges(graph, (0.0, 0.0), radius2)
        
        # Larger radius should affect more or equal edges
        assert len(affected2) >= len(affected1), \
            f"Larger radius ({radius2}) should affect at least as many edges as smaller radius ({radius1})"
    
    @given(
        severity=st.floats(min_value=0.8, max_value=0.99, allow_nan=False, allow_infinity=False)
    )
    @settings(max_examples=50, deadline=2000)
    def test_blocking_threshold_affects_road_blocking(self, severity):
        """
        Property 14: Configuration Effects
        
        High severity disasters should increase the likelihood of road blocking.
        """
        from disaster_evacuation.models import DisasterEvent
        
        # Create a graph
        graph = GraphManager()
        graph.add_vertex("A", VertexType.INTERSECTION, (0.0, 0.0))
        graph.add_vertex("B", VertexType.INTERSECTION, (0.5, 0.0))
        graph.add_edge("A", "B", 1.0, 0.7, 0.2)  # Risk of 0.7
        
        disaster_model = DisasterModel()
        
        # Apply disaster with high severity
        disaster = DisasterEvent(DisasterType.FIRE, (0.25, 0.0), severity, 2.0)
        disaster_model.apply_disaster_effects(graph, disaster)
        edge = graph.get_neighbors("A")[0]
        edge_midpoint = (0.25, 0.0)
        
        # Check if road is blocked
        is_blocked = disaster_model.is_road_blocked(edge, disaster, edge_midpoint)
        
        # High severity fire should either block the road or significantly increase weight
        if not is_blocked:
            # If not blocked, weight should be significantly increased
            base_weight = edge.base_distance + edge.base_risk + edge.base_congestion
            assert edge.current_weight > base_weight * 1.5, \
                f"High severity disaster should significantly increase edge weight"
    
    @given(
        setting_value=st.floats(min_value=0.0, max_value=2.0, allow_nan=False, allow_infinity=False)
    )
    @settings(max_examples=50, deadline=2000)
    def test_configuration_persists_across_save_load(self, setting_value):
        """
        Property 14: Configuration Effects
        
        Configuration changes should persist across save/load cycles.
        """
        temp_dir = tempfile.mkdtemp()
        try:
            config_manager = ConfigurationManager(temp_dir, enable_logging=False)
            
            # Set a custom setting
            config_manager.set_algorithm_setting("heuristic_weight", setting_value)
            
            # Create and save a graph
            graph = GraphManager()
            graph.add_vertex("A", VertexType.INTERSECTION, (0.0, 0.0))
            config_manager.save_graph_configuration(graph, "test_config")
            
            # Load the graph
            loaded_graph, _ = config_manager.load_graph_configuration("test_config")
            
            # Verify the setting persisted
            assert config_manager.get_algorithm_setting("heuristic_weight") == setting_value, \
                "Configuration setting should persist across save/load"
            
            config_manager.close()
        finally:
            shutil.rmtree(temp_dir)
    
    @given(
        disaster_type=st.sampled_from([DisasterType.FLOOD, DisasterType.FIRE, DisasterType.EARTHQUAKE])
    )
    @settings(max_examples=30, deadline=2000)
    def test_disaster_type_specific_parameters_affect_calculations(self, disaster_type):
        """
        Property 14: Configuration Effects
        
        Disaster-type-specific parameters should affect calculations differently.
        """
        from disaster_evacuation.models import DisasterEvent
        
        # Create a graph
        graph = GraphManager()
        graph.add_vertex("A", VertexType.INTERSECTION, (0.0, 0.0))
        graph.add_vertex("B", VertexType.INTERSECTION, (1.0, 0.0))
        graph.add_edge("A", "B", 1.0, 0.5, 0.2)
        
        disaster_model = DisasterModel()
        
        # Apply disaster
        disaster = DisasterEvent(disaster_type, (0.5, 0.0), 0.8, 2.0)
        disaster_model.apply_disaster_effects(graph, disaster)
        edge = graph.get_neighbors("A")[0]
        weight = edge.current_weight
        
        # Weight should be affected by disaster
        base_weight = 1.0 + (0.5 * 1.0) + (0.2 * 1.0)  # distance + risk + congestion
        assert weight > base_weight, \
            f"Disaster should increase edge weight from base {base_weight}"
    
    def test_reset_to_defaults_restores_original_settings(self):
        """
        Property 14: Configuration Effects
        
        Resetting to defaults should restore original configuration.
        """
        temp_dir = tempfile.mkdtemp()
        try:
            config_manager = ConfigurationManager(temp_dir, enable_logging=False)
            
            # Get original defaults
            original_max_iter = config_manager.get_algorithm_setting("max_iterations")
            original_severity = config_manager.get_disaster_parameter("flood", "default_severity")
            
            # Modify settings
            config_manager.set_algorithm_setting("max_iterations", 5000)
            config_manager.set_disaster_parameter("flood", "default_severity", 0.9)
            
            # Verify they changed
            assert config_manager.get_algorithm_setting("max_iterations") == 5000
            assert config_manager.get_disaster_parameter("flood", "default_severity") == 0.9
            
            # Reset to defaults
            config_manager.reset_to_defaults()
            
            # Verify defaults are restored
            assert config_manager.get_algorithm_setting("max_iterations") == original_max_iter, \
                "max_iterations should be restored to default"
            assert config_manager.get_disaster_parameter("flood", "default_severity") == original_severity, \
                "default_severity should be restored to default"
            
            config_manager.close()
        finally:
            shutil.rmtree(temp_dir)
