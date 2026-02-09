"""
Unit tests for Map_Visualizer component.

These tests verify specific examples, edge cases, and error conditions.
Note: Visualization tests don't actually display plots (plt.show() is mocked).
"""

import pytest
import networkx as nx
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing
import matplotlib.pyplot as plt
from disaster_evacuation.osm.graph_converter import GraphConverter
from disaster_evacuation.visualization.map_visualizer import MapVisualizer


class TestMapVisualizer:
    """Test suite for MapVisualizer class."""
    
    def test_initialization(self):
        """Test that MapVisualizer initializes correctly."""
        # Create simple graph
        G = nx.MultiDiGraph()
        G.add_node(0, x=-122.0, y=37.8)
        G.add_node(1, x=-122.001, y=37.8)
        G.add_edge(0, 1, length=100.0)
        
        converter = GraphConverter()
        graph_manager, coord_mapping = converter.convert_osm_to_internal(G)
        
        visualizer = MapVisualizer(graph_manager, coord_mapping)
        assert visualizer is not None
        assert "MapVisualizer" in str(visualizer)
    
    def test_plot_edge(self):
        """Test plotting a single edge."""
        # Create graph
        G = nx.MultiDiGraph()
        G.add_node(0, x=-122.0, y=37.8)
        G.add_node(1, x=-122.001, y=37.801)
        G.add_edge(0, 1, length=100.0)
        
        converter = GraphConverter()
        graph_manager, coord_mapping = converter.convert_osm_to_internal(G)
        
        visualizer = MapVisualizer(graph_manager, coord_mapping)
        
        # Create figure
        fig, ax = plt.subplots()
        
        # Plot edge (should not raise error)
        visualizer._plot_edge(ax, 0, 1, color='blue', linewidth=2.0)
        
        plt.close(fig)
    
    def test_plot_network_basic(self):
        """Test plotting basic network."""
        # Create graph
        G = nx.MultiDiGraph()
        for i in range(4):
            G.add_node(i, x=-122.0 + i * 0.001, y=37.8 + i * 0.001)
        
        G.add_edge(0, 1, length=100.0)
        G.add_edge(1, 2, length=100.0)
        G.add_edge(2, 3, length=100.0)
        
        converter = GraphConverter()
        graph_manager, coord_mapping = converter.convert_osm_to_internal(G)
        
        visualizer = MapVisualizer(graph_manager, coord_mapping)
        
        # Mock plt.show() to prevent display
        original_show = plt.show
        plt.show = lambda: None
        
        try:
            # Should not raise error
            visualizer.plot_network()
        finally:
            plt.show = original_show
            plt.close('all')
    
    def test_plot_network_with_blocked_edges(self):
        """Test plotting network with blocked edges shown in red."""
        # Create graph
        G = nx.MultiDiGraph()
        for i in range(4):
            G.add_node(i, x=-122.0 + i * 0.001, y=37.8)
        
        G.add_edge(0, 1, length=100.0)
        G.add_edge(1, 2, length=100.0)
        G.add_edge(2, 3, length=100.0)
        
        converter = GraphConverter()
        graph_manager, coord_mapping = converter.convert_osm_to_internal(G)
        
        visualizer = MapVisualizer(graph_manager, coord_mapping)
        
        # Mock plt.show()
        original_show = plt.show
        plt.show = lambda: None
        
        try:
            # Plot with blocked edge
            visualizer.plot_network(blocked_edges=[(1, 2)])
        finally:
            plt.show = original_show
            plt.close('all')
    
    def test_plot_path(self):
        """Test plotting a path."""
        # Create graph
        G = nx.MultiDiGraph()
        for i in range(5):
            G.add_node(i, x=-122.0 + i * 0.001, y=37.8)
        
        for i in range(4):
            G.add_edge(i, i + 1, length=100.0)
        
        converter = GraphConverter()
        graph_manager, coord_mapping = converter.convert_osm_to_internal(G)
        
        visualizer = MapVisualizer(graph_manager, coord_mapping)
        
        # Create figure
        fig, ax = plt.subplots()
        
        # Plot path
        path = [0, 1, 2, 3, 4]
        visualizer._plot_path(ax, path, color='green', linewidth=2.0, label='Test Path')
        
        plt.close(fig)
    
    def test_plot_route_comparison(self):
        """Test plotting route comparison with both routes displayed."""
        # Create graph
        G = nx.MultiDiGraph()
        for i in range(6):
            G.add_node(i, x=-122.0 + i * 0.001, y=37.8 + (i % 2) * 0.001)
        
        # Create two possible paths
        G.add_edge(0, 1, length=100.0)
        G.add_edge(1, 2, length=100.0)
        G.add_edge(2, 5, length=100.0)
        G.add_edge(0, 3, length=150.0)
        G.add_edge(3, 4, length=150.0)
        G.add_edge(4, 5, length=150.0)
        
        converter = GraphConverter()
        graph_manager, coord_mapping = converter.convert_osm_to_internal(G)
        
        visualizer = MapVisualizer(graph_manager, coord_mapping)
        
        # Mock plt.show()
        original_show = plt.show
        plt.show = lambda: None
        
        try:
            # Plot comparison
            normal_path = [0, 1, 2, 5]
            disaster_path = [0, 3, 4, 5]
            
            visualizer.plot_route_comparison(
                source=0,
                target=5,
                normal_path=normal_path,
                normal_distance=300.0,
                disaster_path=disaster_path,
                disaster_distance=450.0,
                blocked_edges=[(1, 2)]
            )
        finally:
            plt.show = original_show
            plt.close('all')
    
    def test_empty_path_handling(self):
        """Test handling of empty paths."""
        # Create graph
        G = nx.MultiDiGraph()
        G.add_node(0, x=-122.0, y=37.8)
        G.add_node(1, x=-122.001, y=37.8)
        
        converter = GraphConverter()
        graph_manager, coord_mapping = converter.convert_osm_to_internal(G)
        
        visualizer = MapVisualizer(graph_manager, coord_mapping)
        
        # Create figure
        fig, ax = plt.subplots()
        
        # Plot empty path (should not raise error)
        visualizer._plot_path(ax, [], color='blue')
        visualizer._plot_path(ax, [0], color='blue')  # Single node
        
        plt.close(fig)
    
    def test_coordinate_accuracy(self):
        """Test that plotted coordinates match the coordinate mapping."""
        # Create graph with known coordinates
        G = nx.MultiDiGraph()
        G.add_node(0, x=-122.5, y=37.9)
        G.add_node(1, x=-122.4, y=37.85)
        G.add_edge(0, 1, length=100.0)
        
        converter = GraphConverter()
        graph_manager, coord_mapping = converter.convert_osm_to_internal(G)
        
        visualizer = MapVisualizer(graph_manager, coord_mapping)
        
        # Verify coordinates are preserved
        assert coord_mapping[0] == (37.9, -122.5)
        assert coord_mapping[1] == (37.85, -122.4)


class TestMapVisualizerEdgeCases:
    """Test suite for edge cases in MapVisualizer."""
    
    def test_empty_graph(self):
        """Test visualization of empty graph."""
        G = nx.MultiDiGraph()
        
        converter = GraphConverter()
        graph_manager, coord_mapping = converter.convert_osm_to_internal(G)
        
        visualizer = MapVisualizer(graph_manager, coord_mapping)
        
        # Mock plt.show()
        original_show = plt.show
        plt.show = lambda: None
        
        try:
            # Should not raise error
            visualizer.plot_network()
        finally:
            plt.show = original_show
            plt.close('all')
    
    def test_single_node_graph(self):
        """Test visualization of graph with single node."""
        G = nx.MultiDiGraph()
        G.add_node(0, x=-122.0, y=37.8)
        
        converter = GraphConverter()
        graph_manager, coord_mapping = converter.convert_osm_to_internal(G)
        
        visualizer = MapVisualizer(graph_manager, coord_mapping)
        
        # Mock plt.show()
        original_show = plt.show
        plt.show = lambda: None
        
        try:
            visualizer.plot_network()
        finally:
            plt.show = original_show
            plt.close('all')
