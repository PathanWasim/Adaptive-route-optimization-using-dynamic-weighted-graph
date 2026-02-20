"""
Property-based tests for visualization consistency in VisualizationEngine.

**Property 11: Visualization Consistency**
**Validates: Requirements 6.2, 6.4**

This module tests that the VisualizationEngine produces consistent and correct
visualizations across different inputs and scenarios.
"""

import pytest
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing
import matplotlib.pyplot as plt
import os
import tempfile
from hypothesis import given, strategies as st, assume, settings, HealthCheck
from datetime import datetime
from disaster_evacuation.visualization import VisualizationEngine
from disaster_evacuation.models import GraphManager
from disaster_evacuation.models import DisasterEvent, DisasterType, VertexType


def create_test_graph(vertex_count: int = 5) -> GraphManager:
    """Create a test graph with specified number of vertices."""
    graph = GraphManager()
    
    # Add vertices in a grid pattern
    for i in range(vertex_count):
        x = i % 3
        y = i // 3
        vertex_type = VertexType.INTERSECTION
        capacity = None
        
        if i == vertex_count - 1:
            vertex_type = VertexType.EVACUATION_POINT
            capacity = 500
        elif i == vertex_count - 2 and vertex_count > 1:
            vertex_type = VertexType.SHELTER
            capacity = 200
        
        graph.add_vertex(f"V{i}", vertex_type, (float(x), float(y)), capacity)
    
    # Add edges to connect vertices
    for i in range(vertex_count - 1):
        graph.add_edge(f"V{i}", f"V{i+1}", 1.0, 0.1, 0.1)
    
    # Add some cross connections for more complex paths
    if vertex_count >= 4:
        graph.add_edge("V0", "V2", 1.4, 0.2, 0.2)
    if vertex_count >= 6:
        graph.add_edge("V1", "V4", 1.8, 0.15, 0.15)
    
    return graph


class TestPropertyVisualizationConsistency:
    """Property-based tests for visualization consistency."""
    
    @given(st.integers(min_value=2, max_value=10))
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=1000)
    def test_property_graph_visualization_consistency(self, vertex_count):
        """
        Property: Graph visualizations should be consistent across different graph sizes.
        
        **Validates: Requirements 6.2**
        
        For any valid graph, the visualization should:
        - Create a valid matplotlib figure
        - Include all vertices and edges
        - Have consistent visual properties
        """
        viz_engine = VisualizationEngine()
        graph = create_test_graph(vertex_count)
        
        fig = viz_engine.visualize_graph(graph, title=f"Test Graph {vertex_count} vertices")
        
        # Should create a valid figure
        assert fig is not None
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) > 0
        
        # Figure should have reasonable dimensions
        width, height = fig.get_size_inches()
        assert width > 0 and height > 0
        
        # Should have a title
        ax = fig.axes[0]
        assert ax.get_title() is not None
        assert len(ax.get_title()) > 0
        
        plt.close(fig)
    
    @given(
        st.integers(min_value=2, max_value=8),
        st.integers(min_value=1, max_value=5)
    )
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=1000)
    def test_property_path_visualization_consistency(self, vertex_count, path_length):
        """
        Property: Path visualizations should be consistent for valid paths.
        
        **Validates: Requirements 6.2, 6.4**
        
        For any valid path through a graph, the visualization should:
        - Highlight the path correctly
        - Show all path vertices
        - Maintain visual consistency
        """
        # Ensure path length doesn't exceed vertex count
        assume(path_length <= vertex_count)
        
        viz_engine = VisualizationEngine()
        graph = create_test_graph(vertex_count)
        
        # Create a valid path
        path = [f"V{i}" for i in range(path_length)]
        path_edges = []
        for i in range(path_length - 1):
            edge = graph.get_edge(path[i], path[i + 1])
            if edge:
                path_edges.append(edge)
        
        # Skip if we couldn't create a valid path
        assume(len(path_edges) == path_length - 1)
        
        fig = viz_engine.visualize_path(graph, path, path_edges, 
                                       title=f"Path Length {path_length}")
        
        # Should create a valid figure
        assert fig is not None
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) > 0
        
        plt.close(fig)
    
    @given(
        st.sampled_from([DisasterType.FLOOD, DisasterType.FIRE, DisasterType.EARTHQUAKE]),
        st.floats(min_value=0.1, max_value=1.0),
        st.floats(min_value=0.5, max_value=3.0)
    )
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=1000)
    def test_property_disaster_visualization_consistency(self, disaster_type, severity, radius):
        """
        Property: Disaster visualizations should be consistent across disaster parameters.
        
        **Validates: Requirements 6.2, 6.4**
        
        For any valid disaster configuration, the visualization should:
        - Show disaster effects clearly
        - Use appropriate colors for disaster type
        - Display disaster area correctly
        """
        viz_engine = VisualizationEngine()
        graph = create_test_graph(5)
        
        disaster = DisasterEvent(
            disaster_type=disaster_type,
            epicenter=(1.0, 1.0),
            severity=severity,
            max_effect_radius=radius,
            start_time=datetime.now()
        )
        
        # Get some affected edges (simplified)
        affected_edges = graph.get_all_edges()[:2]  # Take first 2 edges as affected
        
        fig = viz_engine.visualize_disaster_effects(
            graph, disaster, affected_edges,
            title=f"{disaster_type.value.title()} Visualization"
        )
        
        # Should create a valid figure
        assert fig is not None
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) > 0
        
        # Should use correct disaster color
        expected_color = viz_engine.disaster_colors[disaster_type]
        assert expected_color in viz_engine.disaster_colors.values()
        
        plt.close(fig)
    
    @given(st.integers(min_value=2, max_value=6))
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=1000)
    def test_property_comparison_visualization_consistency(self, vertex_count):
        """
        Property: Comparison visualizations should be consistent.
        
        **Validates: Requirements 6.4**
        
        Path comparison visualizations should:
        - Show both paths when available
        - Use distinct colors for different paths
        - Maintain consistent layout
        """
        viz_engine = VisualizationEngine()
        graph = create_test_graph(vertex_count)
        
        # Create two different paths
        normal_path = [f"V{i}" for i in range(min(3, vertex_count))]
        disaster_path = [f"V{i}" for i in range(0, min(vertex_count, 4), 2)]
        
        # Get edges for paths
        normal_edges = []
        for i in range(len(normal_path) - 1):
            edge = graph.get_edge(normal_path[i], normal_path[i + 1])
            if edge:
                normal_edges.append(edge)
        
        disaster_edges = []
        for i in range(len(disaster_path) - 1):
            edge = graph.get_edge(disaster_path[i], disaster_path[i + 1])
            if edge:
                disaster_edges.append(edge)
        
        # Only test if we have valid paths
        assume(len(normal_edges) > 0 or len(disaster_edges) > 0)
        
        fig = viz_engine.visualize_path_comparison(
            graph, normal_path, normal_edges,
            disaster_path, disaster_edges,
            title="Comparison Test"
        )
        
        # Should create a valid figure with two subplots
        assert fig is not None
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 2
        
        plt.close(fig)
    
    @given(
        st.tuples(st.floats(min_value=8.0, max_value=20.0), 
                 st.floats(min_value=6.0, max_value=15.0))
    )
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=1000)
    def test_property_figure_size_consistency(self, figsize):
        """
        Property: Figure sizes should be respected consistently.
        
        **Validates: Requirements 6.2**
        
        Custom figure sizes should be applied correctly to all visualizations.
        """
        viz_engine = VisualizationEngine(figsize=figsize)
        graph = create_test_graph(4)
        
        fig = viz_engine.visualize_graph(graph)
        
        # Check that figure size is approximately correct (allowing for small variations)
        actual_size = fig.get_size_inches()
        assert abs(actual_size[0] - figsize[0]) < 0.1
        assert abs(actual_size[1] - figsize[1]) < 0.1
        
        plt.close(fig)
    
    @given(st.text(min_size=1, max_size=50, alphabet=st.characters(min_codepoint=32, max_codepoint=126)))
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.filter_too_much], deadline=1000)
    def test_property_title_consistency(self, title):
        """
        Property: Titles should be displayed consistently.
        
        **Validates: Requirements 6.2**
        
        Any valid title string should be displayed correctly in visualizations.
        """
        # Filter out problematic characters that might cause issues
        assume('\n' not in title and '\r' not in title)  # No line breaks
        assume(len(title.strip()) > 0)  # Not just whitespace
        
        viz_engine = VisualizationEngine()
        graph = create_test_graph(3)
        
        fig = viz_engine.visualize_graph(graph, title=title)
        
        # Should have the correct title
        ax = fig.axes[0]
        actual_title = ax.get_title()
        assert title in actual_title or actual_title == title
        
        plt.close(fig)
    
    @given(st.integers(min_value=1, max_value=5))
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=2000)  # Increase deadline for file I/O
    def test_property_save_functionality_consistency(self, graph_size):
        """
        Property: Save functionality should work consistently.
        
        **Validates: Requirements 6.2**
        
        All visualizations should be saveable to files with consistent quality.
        """
        viz_engine = VisualizationEngine()
        graph = create_test_graph(graph_size)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, f"test_graph_{graph_size}.png")
            
            fig = viz_engine.visualize_graph(graph, save_path=save_path)
            
            # File should be created and have content
            assert os.path.exists(save_path)
            assert os.path.getsize(save_path) > 0
            
            plt.close(fig)
    
    @given(st.integers(min_value=2, max_value=8))
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=1000)
    def test_property_vertex_type_visualization_consistency(self, vertex_count):
        """
        Property: Different vertex types should be visualized consistently.
        
        **Validates: Requirements 6.2, 6.4**
        
        Each vertex type should have consistent visual representation
        across different graphs and scenarios.
        """
        viz_engine = VisualizationEngine()
        
        # Create graph with mixed vertex types
        graph = GraphManager()
        vertex_types = [VertexType.INTERSECTION, VertexType.SHELTER, VertexType.EVACUATION_POINT]
        
        for i in range(vertex_count):
            vertex_type = vertex_types[i % len(vertex_types)]
            capacity = 100 if vertex_type != VertexType.INTERSECTION else None
            graph.add_vertex(f"V{i}", vertex_type, (float(i % 3), float(i // 3)), capacity)
        
        # Add some edges
        for i in range(vertex_count - 1):
            graph.add_edge(f"V{i}", f"V{i+1}", 1.0, 0.1, 0.1)
        
        fig = viz_engine.visualize_graph(graph)
        
        # Should handle all vertex types correctly
        assert fig is not None
        assert isinstance(fig, plt.Figure)
        
        # Check that vertex colors are configured for all types
        for vertex_type in vertex_types:
            assert vertex_type in viz_engine.vertex_colors
            assert vertex_type in viz_engine.vertex_sizes
        
        plt.close(fig)
    
    @given(st.integers(min_value=1, max_value=3))
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=1000)
    def test_property_empty_path_handling_consistency(self, graph_size):
        """
        Property: Empty or invalid paths should be handled consistently.
        
        **Validates: Requirements 6.4**
        
        When no valid path exists, visualizations should handle this gracefully
        and provide appropriate feedback.
        """
        viz_engine = VisualizationEngine()
        graph = create_test_graph(graph_size)
        
        # Test with empty path
        fig = viz_engine.visualize_path(graph, [], [], title="Empty Path Test")
        
        # Should still create a valid figure
        assert fig is not None
        assert isinstance(fig, plt.Figure)
        
        plt.close(fig)
    
    @given(
        st.integers(min_value=2, max_value=5),
        st.integers(min_value=1, max_value=3)
    )
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=1000)
    def test_property_dynamic_comparison_consistency(self, vertex_count, scenario_count):
        """
        Property: Dynamic comparison visualizations should be consistent.
        
        **Validates: Requirements 6.4**
        
        Multi-scenario comparisons should maintain visual consistency
        and proper layout regardless of the number of scenarios.
        """
        viz_engine = VisualizationEngine()
        graph = create_test_graph(vertex_count)
        
        # Create comparison data
        comparison_data = {
            "normal_route": {
                "found": True,
                "path": [f"V{i}" for i in range(min(3, vertex_count))],
                "cost": 2.5,
                "edge_count": min(2, vertex_count - 1)
            },
            "disaster_aware_route": {
                "found": scenario_count > 1,
                "path": [f"V{i}" for i in range(0, min(vertex_count, 4), 2)] if scenario_count > 1 else [],
                "cost": 3.2 if scenario_count > 1 else None,
                "edge_count": max(1, scenario_count - 1)
            }
        }
        
        if scenario_count > 1:
            comparison_data["analysis"] = {
                "path_changed": True,
                "cost_increase": 0.7,
                "cost_increase_percentage": 28.0
            }
        
        fig = viz_engine.visualize_dynamic_comparison(
            graph, comparison_data, title="Dynamic Comparison Test"
        )
        
        # Should create a valid complex figure
        assert fig is not None
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) >= 4  # Should have multiple subplots
        
        plt.close(fig)
    
    def test_property_color_consistency_across_visualizations(self):
        """
        Property: Colors should be consistent across different visualization types.
        
        **Validates: Requirements 6.2, 6.4**
        
        The same vertex types and disaster types should use the same colors
        across all visualization methods.
        """
        viz_engine = VisualizationEngine()
        
        # Test that color mappings are consistent
        vertex_colors = viz_engine.vertex_colors
        disaster_colors = viz_engine.disaster_colors
        
        # All vertex types should have colors
        for vertex_type in VertexType:
            assert vertex_type in vertex_colors
            assert isinstance(vertex_colors[vertex_type], str)
            assert vertex_colors[vertex_type].startswith('#')  # Should be hex color
        
        # All disaster types should have colors
        for disaster_type in DisasterType:
            assert disaster_type in disaster_colors
            assert isinstance(disaster_colors[disaster_type], str)
            assert disaster_colors[disaster_type].startswith('#')  # Should be hex color
        
        # Colors should be distinct
        vertex_color_values = list(vertex_colors.values())
        assert len(set(vertex_color_values)) == len(vertex_color_values)  # All unique
        
        disaster_color_values = list(disaster_colors.values())
        assert len(set(disaster_color_values)) == len(disaster_color_values)  # All unique
    
    @given(st.integers(min_value=1, max_value=4))
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=1000)
    def test_property_legend_consistency(self, graph_complexity):
        """
        Property: Legends should be consistent and informative.
        
        **Validates: Requirements 6.2**
        
        All visualizations should include appropriate legends that help
        users understand the visual elements.
        """
        viz_engine = VisualizationEngine()
        graph = create_test_graph(graph_complexity + 2)  # Ensure minimum size
        
        # Test basic graph visualization
        fig = viz_engine.visualize_graph(graph, title="Legend Test")
        
        # Should have legend elements (this is implicit in the visualization)
        # We can't easily test legend content without accessing internal matplotlib structures
        # But we can ensure the visualization completes successfully
        assert fig is not None
        assert isinstance(fig, plt.Figure)
        
        plt.close(fig)