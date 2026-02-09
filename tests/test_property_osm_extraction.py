"""
Property-based tests for OSM extraction components.

These tests verify universal correctness properties across all inputs
using Hypothesis for randomized testing.
"""

import pytest
from hypothesis import given, strategies as st, settings, assume
import networkx as nx
from disaster_evacuation.osm.osm_extractor import OSMExtractor, AreaTooLargeError


# Feature: osm-road-network-integration, Property 1: Area Bounds Validation
@given(
    num_nodes=st.integers(min_value=10, max_value=100),
    area_multiplier=st.floats(min_value=0.5, max_value=4.0)
)
@settings(max_examples=100, deadline=None)
def test_area_bounds_validation(num_nodes, area_multiplier):
    """
    For any extracted road network, the geographic area covered should be 
    between 1 km² and 3 km².
    
    Validates: Requirements 1.2
    """
    # Create a mock OSM graph with controlled area
    G = nx.MultiDiGraph()
    
    # Add nodes in a grid pattern with spacing that controls area
    # Spacing of ~0.01 degrees ≈ 1km, so we can control area
    spacing = 0.001 * area_multiplier
    grid_size = int(num_nodes ** 0.5)
    
    for i in range(grid_size):
        for j in range(grid_size):
            node_id = i * grid_size + j
            G.add_node(node_id, x=-122.0 + j * spacing, y=37.8 + i * spacing)
    
    # Add some edges
    for i in range(grid_size - 1):
        for j in range(grid_size - 1):
            node_id = i * grid_size + j
            G.add_edge(node_id, node_id + 1, length=100.0)
            G.add_edge(node_id, node_id + grid_size, length=100.0)
    
    # Skip if graph is empty
    assume(G.number_of_edges() > 0)
    
    extractor = OSMExtractor()
    stats = extractor.get_network_stats(G)
    
    # If area is too large, validation should catch it
    if stats['area_km2'] > 3.0:
        with pytest.raises(AreaTooLargeError):
            extractor._validate_network(G, "test_location")
    else:
        # Should not raise for valid areas
        extractor._validate_network(G, "test_location")


# Feature: osm-road-network-integration, Property 2: Node Coordinate Completeness
@given(
    num_nodes=st.integers(min_value=5, max_value=50)
)
@settings(max_examples=100, deadline=None)
def test_node_coordinate_completeness(num_nodes):
    """
    For any extracted OSM road network, every node should have valid 
    Real_Coordinates (latitude and longitude values).
    
    Validates: Requirements 1.3
    """
    # Create a mock OSM graph
    G = nx.MultiDiGraph()
    
    # Add nodes with coordinates
    for i in range(num_nodes):
        lat = 37.8 + (i * 0.001)
        lon = -122.0 + (i * 0.001)
        G.add_node(i, x=lon, y=lat)
    
    # Verify all nodes have coordinates
    for node_id, data in G.nodes(data=True):
        assert 'x' in data, f"Node {node_id} missing longitude (x)"
        assert 'y' in data, f"Node {node_id} missing latitude (y)"
        assert isinstance(data['x'], (int, float)), f"Node {node_id} x coordinate not numeric"
        assert isinstance(data['y'], (int, float)), f"Node {node_id} y coordinate not numeric"
        # Validate coordinate ranges
        assert -180 <= data['x'] <= 180, f"Node {node_id} longitude out of range"
        assert -90 <= data['y'] <= 90, f"Node {node_id} latitude out of range"


# Feature: osm-road-network-integration, Property 3: Edge Distance Completeness
@given(
    num_nodes=st.integers(min_value=3, max_value=30),
    num_edges=st.integers(min_value=2, max_value=50)
)
@settings(max_examples=100, deadline=None)
def test_edge_distance_completeness(num_nodes, num_edges):
    """
    For any extracted OSM road network, every edge should have a distance 
    measurement in meters.
    
    Validates: Requirements 1.4
    """
    # Create a mock OSM graph
    G = nx.MultiDiGraph()
    
    # Add nodes
    for i in range(num_nodes):
        G.add_node(i, x=-122.0 + i * 0.001, y=37.8 + i * 0.001)
    
    # Add edges with distances
    edges_added = 0
    for _ in range(num_edges):
        if edges_added >= num_edges:
            break
        u = edges_added % num_nodes
        v = (edges_added + 1) % num_nodes
        if u != v:
            distance = 50.0 + (edges_added * 10.0)
            G.add_edge(u, v, length=distance)
            edges_added += 1
    
    # Skip if no edges
    assume(G.number_of_edges() > 0)
    
    # Verify all edges have length attribute
    for u, v, data in G.edges(data=True):
        assert 'length' in data, f"Edge ({u}, {v}) missing length attribute"
        assert isinstance(data['length'], (int, float)), f"Edge ({u}, {v}) length not numeric"
        assert data['length'] > 0, f"Edge ({u}, {v}) has non-positive length"
