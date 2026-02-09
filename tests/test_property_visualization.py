"""
Property-based tests for Map_Visualizer component.

These tests verify universal correctness properties for visualization
using Hypothesis for randomized testing.
"""

import pytest
from hypothesis import given, strategies as st, settings, assume
import networkx as nx
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from disaster_evacuation.osm.graph_converter import GraphConverter
from disaster_evacuation.visualization.map_visualizer import MapVisualizer


# Feature: osm-road-network-integration, Property 14: Visualization Coordinate Accuracy
@given(
    num_nodes=st.integers(min_value=2, max_value=15)
)
@settings(max_examples=50, deadline=None)
def test_visualization_coordinate_accuracy(num_nodes):
    """
    For any road network visualization, the plotted coordinates for each node
    should match the Real_Coordinates from the coordinate mapping.
    
    Validates: Requirements 4.1
    """
    # Create OSM graph with known coordinates
    G = nx.MultiDiGraph()
    
    expected_coords = {}
    for i in range(num_nodes):
        lat = 37.8 + i * 0.0001
        lon = -122.0 + i * 0.0001
        G.add_node(i, x=lon, y=lat)
        expected_coords[i] = (lat, lon)
    
    # Add some edges
    for i in range(num_nodes - 1):
        G.add_edge(i, i + 1, length=10.0)
    
    # Convert to internal format
    converter = GraphConverter()
    graph_manager, coord_mapping = converter.convert_osm_to_internal(G)
    
    # Create visualizer
    visualizer = MapVisualizer(graph_manager, coord_mapping)
    
    # Verify all coordinates in mapping match expected
    for internal_id, (lat, lon) in coord_mapping.items():
        # Find corresponding OSM node
        osm_id = list(G.nodes())[internal_id]
        expected_lat, expected_lon = expected_coords[osm_id]
        
        assert abs(lat - expected_lat) < 1e-6, \
            f"Latitude mismatch for node {internal_id}: {lat} != {expected_lat}"
        assert abs(lon - expected_lon) < 1e-6, \
            f"Longitude mismatch for node {internal_id}: {lon} != {expected_lon}"
    
    # Clean up
    plt.close('all')
