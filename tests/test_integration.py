"""
Integration tests for complete workflows in the disaster evacuation system.

These tests verify end-to-end functionality including component interactions,
data flow, and error propagation.
"""

import pytest
from disaster_evacuation.main import DisasterEvacuationApp
from disaster_evacuation.models import DisasterType, VertexType


class TestEndToEndWorkflows:
    """Test complete end-to-end workflows."""
    
    def test_complete_evacuation_workflow(self, tmp_path):
        """Test complete workflow from graph creation to route computation."""
        # Initialize application
        app = DisasterEvacuationApp(config_dir=str(tmp_path))
        
        # Create sample graph
        app.create_sample_graph()
        assert app.graph.get_vertex_count() == 7
        assert app.graph.get_edge_count() == 9
        
        # Compute route without disaster
        result = app.compute_route("Home", "Evac_Point", show_visualization=False)
        assert result is not None
        assert result['path'] == ['Home', 'Hospital', 'Shelter_A', 'Evac_Point']
        assert result['total_cost'] > 0
        
        # Apply disaster
        success = app.apply_disaster("fire", (3.0, 2.0), 0.8, 5.0)
        assert success is True
        
        # Compute route with disaster
        result_with_disaster = app.compute_route("Home", "Evac_Point", show_visualization=False)
        assert result_with_disaster is not None
        assert result_with_disaster['total_cost'] >= result['total_cost']
    
    def test_configuration_persistence_workflow(self, tmp_path):
        """Test workflow with configuration save and load."""
        # Initialize application
        app = DisasterEvacuationApp(config_dir=str(tmp_path))
        
        # Create and save graph
        app.create_sample_graph()
        success = app.save_graph_to_config("test_config")
        assert success is True
        
        # Create new application and load configuration
        app2 = DisasterEvacuationApp(config_dir=str(tmp_path))
        success = app2.load_graph_from_config("test_config")
        assert success is True
        assert app2.graph.get_vertex_count() == 7
        
        # Verify routes work on loaded graph
        result = app2.compute_route("Home", "Evac_Point", show_visualization=False)
        assert result is not None
    
    def test_disaster_comparison_workflow(self, tmp_path):
        """Test workflow comparing routes with and without disaster."""
        # Initialize application
        app = DisasterEvacuationApp(config_dir=str(tmp_path))
        app.create_sample_graph()
        
        # Apply disaster
        app.apply_disaster("flood", (2.0, 1.0), 0.6, 4.0)
        
        # Compare routes
        comparison = app.compare_routes("Home", "Evac_Point", show_visualization=False)
        assert comparison is not None
        assert comparison['success'] is True
        assert 'normal_route' in comparison
        assert 'disaster_aware_route' in comparison


class TestComponentInteractions:
    """Test interactions between components."""
    
    def test_graph_disaster_pathfinder_interaction(self):
        """Test interaction between graph, disaster model, and pathfinder."""
        app = DisasterEvacuationApp()
        app.create_sample_graph()
        
        # Get initial route
        initial_result = app.compute_route("Home", "Shelter_A", show_visualization=False)
        initial_cost = initial_result['total_cost']
        
        # Apply disaster
        app.apply_disaster("earthquake", (4.0, 1.0), 0.7, 3.0)
        
        # Get route after disaster
        disaster_result = app.compute_route("Home", "Shelter_A", show_visualization=False)
        disaster_cost = disaster_result['total_cost']
        
        # Cost should increase due to disaster
        assert disaster_cost >= initial_cost
    
    def test_controller_visualization_interaction(self):
        """Test interaction between controller and visualization."""
        app = DisasterEvacuationApp()
        app.create_sample_graph()
        
        # Compute route (visualization disabled for testing)
        result = app.compute_route("Home", "Evac_Point", show_visualization=False)
        assert result is not None
        
        # Verify visualization can handle the result
        # (actual visualization not tested in unit tests)
        assert 'path' in result
        assert len(result['path']) > 0


class TestErrorPropagation:
    """Test error handling and propagation."""
    
    def test_invalid_vertex_error_propagation(self):
        """Test error propagation for invalid vertices."""
        app = DisasterEvacuationApp()
        app.create_sample_graph()
        
        # Try to compute route with invalid vertex
        result = app.compute_route("InvalidStart", "Evac_Point", show_visualization=False)
        assert result is None
    
    def test_empty_graph_error_handling(self):
        """Test error handling for empty graph."""
        app = DisasterEvacuationApp()
        
        # Try to compute route on empty graph
        result = app.compute_route("Home", "Evac_Point", show_visualization=False)
        assert result is None
        
        # Try to apply disaster on empty graph
        success = app.apply_disaster("fire", (0.0, 0.0), 0.5, 5.0)
        assert success is False
    
    def test_invalid_disaster_parameters(self):
        """Test error handling for invalid disaster parameters."""
        app = DisasterEvacuationApp()
        app.create_sample_graph()
        
        # Invalid severity
        success = app.apply_disaster("fire", (0.0, 0.0), 1.5, 5.0)
        assert success is False
        
        # Invalid radius
        success = app.apply_disaster("fire", (0.0, 0.0), 0.5, -1.0)
        assert success is False
        
        # Invalid disaster type
        success = app.apply_disaster("tornado", (0.0, 0.0), 0.5, 5.0)
        assert success is False


class TestDataFlow:
    """Test data flow through the system."""
    
    def test_graph_modification_affects_routes(self):
        """Test that graph modifications affect route computation."""
        app = DisasterEvacuationApp()
        app.create_sample_graph()
        
        # Compute initial route
        initial_result = app.compute_route("Home", "Evac_Point", show_visualization=False)
        initial_path = initial_result['path']
        
        # Apply disaster that blocks roads
        app.apply_disaster("fire", (4.0, 0.0), 0.9, 2.0)
        
        # Compute new route
        new_result = app.compute_route("Home", "Evac_Point", show_visualization=False)
        
        # Route should still be found (may be different or same)
        assert new_result is not None
        assert len(new_result['path']) > 0
    
    def test_multiple_disasters_cumulative_effect(self):
        """Test that multiple disasters have cumulative effects."""
        app = DisasterEvacuationApp()
        app.create_sample_graph()
        
        # Get baseline cost
        baseline = app.compute_route("Home", "Evac_Point", show_visualization=False)
        baseline_cost = baseline['total_cost']
        
        # Apply first disaster
        app.apply_disaster("flood", (2.0, 1.0), 0.5, 3.0)
        result1 = app.compute_route("Home", "Evac_Point", show_visualization=False)
        cost1 = result1['total_cost']
        
        # Apply second disaster
        app.apply_disaster("fire", (5.0, 2.0), 0.6, 3.0)
        result2 = app.compute_route("Home", "Evac_Point", show_visualization=False)
        cost2 = result2['total_cost']
        
        # Costs should generally increase with more disasters
        assert cost1 >= baseline_cost
        assert cost2 >= cost1


class TestPerformanceMonitoring:
    """Test performance monitoring and statistics."""
    
    def test_statistics_tracking(self):
        """Test that statistics are tracked correctly."""
        app = DisasterEvacuationApp()
        app.create_sample_graph()
        
        # Initial stats
        stats = app.get_performance_stats()
        assert stats['routes_computed'] == 0
        assert stats['disasters_applied'] == 0
        assert stats['comparisons_made'] == 0
        
        # Compute route
        app.compute_route("Home", "Evac_Point", show_visualization=False)
        stats = app.get_performance_stats()
        assert stats['routes_computed'] == 1
        
        # Apply disaster
        app.apply_disaster("fire", (3.0, 2.0), 0.7, 4.0)
        stats = app.get_performance_stats()
        assert stats['disasters_applied'] == 1
        
        # Compare routes
        app.compare_routes("Home", "Evac_Point", show_visualization=False)
        stats = app.get_performance_stats()
        assert stats['comparisons_made'] == 1
    
    def test_computation_time_tracking(self):
        """Test that computation times are tracked."""
        app = DisasterEvacuationApp()
        app.create_sample_graph()
        
        # Compute multiple routes
        for _ in range(3):
            app.compute_route("Home", "Evac_Point", show_visualization=False)
        
        stats = app.get_performance_stats()
        assert stats['total_computation_time'] > 0
        assert stats['average_computation_time'] > 0
        assert stats['routes_computed'] == 3
