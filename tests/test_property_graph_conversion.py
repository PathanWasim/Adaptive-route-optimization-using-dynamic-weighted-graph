"""
Property-based tests for Graph_Converter component.

These tests verify universal correctness properties for OSM graph conversion
using Hypothesis for randomized testing.
"""

import pytest
from hypothesis import given, strategies as st, settings, assume
import networkx as nx
from disaster_evacuation.osm.graph_converter import GraphConverter


# Feature: osm-road-network-integration, Property 4: Graph Format Conversion Correctness
@given(
    num_nodes=st.integers(min_value=3, max_value=30),
    num_edges=st.integers(min_value=2, max_value=50)
)
@settings(max_examples=100, deadline=None)
def test_graph_format_conversion_correctness(num_nodes, num_edges):
    """
    For any OSM graph, the converted internal graph should have the same
    number of nodes and edges as the original.
    
    Validates: Requirements 2.1
    """
    # Create a mock OSM graph
    G = nx.MultiDiGraph()
    
    # Add nodes with coordinates
    for i in range(num_nodes):
        G.add_node(i, x=-122.0 + i * 0.0001, y=37.8 + i * 0.0001)
    
    # Add edges with distances (avoid parallel edges)
    edges_added = set()
    for _ in range(num_edges):
        if len(edges_added) >= num_edges:
            break
        u = len(edges_added) % num_nodes
        v = (len(edges_added) + 1) % num_nodes
        if u != v and (u, v) not in edges_added:
            distance = 50.0 + (len(edges_added) * 10.0)
            G.add_edge(u, v, length=distance)
            edges_added.add((u, v))
    
    # Skip if no edges
    assume(G.number_of_edges() > 0)
    
    # Convert to internal format
    converter = GraphConverter()
    graph_manager, coordinate_mapping = converter.convert_osm_to_internal(G)
    
    # Verify node count matches
    assert graph_manager.get_vertex_count() == G.number_of_nodes(), \
        f"Node count mismatch: {graph_manager.get_vertex_count()} != {G.number_of_nodes()}"
    
    # Verify edge count matches
    assert graph_manager.get_edge_count() == G.number_of_edges(), \
        f"Edge count mismatch: {graph_manager.get_edge_count()} != {G.number_of_edges()}"
    
    # Verify coordinate mapping has entry for each node
    assert len(coordinate_mapping) == G.number_of_nodes(), \
        f"Coordinate mapping size mismatch: {len(coordinate_mapping)} != {G.number_of_nodes()}"


# Feature: osm-road-network-integration, Property 5: Coordinate Preservation
@given(
    num_nodes=st.integers(min_value=3, max_value=20)
)
@settings(max_examples=100, deadline=None)
def test_coordinate_preservation(num_nodes):
    """
    For any OSM graph, every node's geographic coordinates should be preserved
    in the coordinate mapping after conversion.
    
    Validates: Requirements 2.2
    """
    # Create a mock OSM graph with known coordinates
    G = nx.MultiDiGraph()
    
    expected_coords = {}
    for i in range(num_nodes):
        lat = 37.8 + i * 0.0001
        lon = -122.0 + i * 0.0001
        G.add_node(i, x=lon, y=lat)
        expected_coords[i] = (lat, lon)
    
    # Add at least one edge to make it a valid graph
    if num_nodes >= 2:
        G.add_edge(0, 1, length=100.0)
    
    # Convert to internal format
    converter = GraphConverter()
    graph_manager, coordinate_mapping = converter.convert_osm_to_internal(G)
    
    # Verify all coordinates are preserved
    for internal_id, (lat, lon) in coordinate_mapping.items():
        # Find corresponding OSM node
        osm_id = list(G.nodes())[internal_id]
        expected_lat, expected_lon = expected_coords[osm_id]
        
        assert abs(lat - expected_lat) < 1e-6, \
            f"Latitude mismatch for node {internal_id}: {lat} != {expected_lat}"
        assert abs(lon - expected_lon) < 1e-6, \
            f"Longitude mismatch for node {internal_id}: {lon} != {expected_lon}"


# Feature: osm-road-network-integration, Property 6: Connectivity Preservation
@given(
    num_nodes=st.integers(min_value=3, max_value=20),
    num_edges=st.integers(min_value=2, max_value=40)
)
@settings(max_examples=100, deadline=None)
def test_connectivity_preservation(num_nodes, num_edges):
    """
    For any OSM graph, if there is an edge from node A to node B in the OSM graph,
    there should be a corresponding edge in the internal graph.
    
    Validates: Requirements 2.3
    """
    # Create a mock OSM graph
    G = nx.MultiDiGraph()
    
    # Add nodes
    for i in range(num_nodes):
        G.add_node(i, x=-122.0 + i * 0.0001, y=37.8 + i * 0.0001)
    
    # Add edges and track them
    osm_edges = set()
    edges_added = 0
    for _ in range(num_edges):
        if edges_added >= num_edges:
            break
        u = edges_added % num_nodes
        v = (edges_added + 1) % num_nodes
        if u != v:
            G.add_edge(u, v, length=100.0 + edges_added)
            osm_edges.add((u, v))
            edges_added += 1
    
    # Skip if no edges
    assume(len(osm_edges) > 0)
    
    # Convert to internal format
    converter = GraphConverter()
    graph_manager, coordinate_mapping = converter.convert_osm_to_internal(G)
    
    # Create reverse mapping (internal_id -> osm_id)
    osm_nodes = list(G.nodes())
    
    # Verify all OSM edges exist in internal graph
    for u_osm, v_osm in osm_edges:
        # Find internal IDs
        u_internal = osm_nodes.index(u_osm)
        v_internal = osm_nodes.index(v_osm)
        
        # Convert to string IDs (GraphManager uses strings)
        source = str(u_internal)
        target = str(v_internal)
        
        # Check if edge exists in internal graph
        neighbors = graph_manager.get_neighbors(source)
        neighbor_targets = [edge.target for edge in neighbors]
        
        assert target in neighbor_targets, \
            f"Edge ({u_osm}->{v_osm}) not found in internal graph as ({source}->{target})"


# Feature: osm-road-network-integration, Property 7: Distance Data Preservation
@given(
    num_nodes=st.integers(min_value=3, max_value=15),
    num_edges=st.integers(min_value=2, max_value=30)
)
@settings(max_examples=100, deadline=None)
def test_distance_data_preservation(num_nodes, num_edges):
    """
    For any OSM graph, the distance (length) of each edge should be preserved
    in the internal graph representation.
    
    Validates: Requirements 2.4
    """
    # Create a mock OSM graph with known distances
    G = nx.MultiDiGraph()
    
    # Add nodes
    for i in range(num_nodes):
        G.add_node(i, x=-122.0 + i * 0.0001, y=37.8 + i * 0.0001)
    
    # Add edges with specific distances (avoid parallel edges)
    edge_distances = {}
    edges_added = set()
    for _ in range(num_edges):
        if len(edges_added) >= num_edges:
            break
        u = len(edges_added) % num_nodes
        v = (len(edges_added) + 1) % num_nodes
        if u != v and (u, v) not in edges_added:
            distance = 100.0 + (len(edges_added) * 25.0)
            G.add_edge(u, v, length=distance)
            edge_distances[(u, v)] = distance
            edges_added.add((u, v))
    
    # Skip if no edges
    assume(len(edge_distances) > 0)
    
    # Convert to internal format
    converter = GraphConverter()
    graph_manager, coordinate_mapping = converter.convert_osm_to_internal(G)
    
    # Create node mapping
    osm_nodes = list(G.nodes())
    
    # Verify all edge distances are preserved
    for (u_osm, v_osm), expected_distance in edge_distances.items():
        # Find internal IDs
        u_internal = osm_nodes.index(u_osm)
        v_internal = osm_nodes.index(v_osm)
        
        # Convert to string IDs
        source = str(u_internal)
        target = str(v_internal)
        
        # Get edge from internal graph
        neighbors = graph_manager.get_neighbors(source)
        edge = next((e for e in neighbors if e.target == target), None)
        
        assert edge is not None, f"Edge ({source}->{target}) not found"
        assert abs(edge.base_distance - expected_distance) < 1e-6, \
            f"Distance mismatch: {edge.base_distance} != {expected_distance}"


# Feature: osm-road-network-integration, Property 8: Initial Factor Initialization
@given(
    num_nodes=st.integers(min_value=3, max_value=15),
    num_edges=st.integers(min_value=2, max_value=30)
)
@settings(max_examples=100, deadline=None)
def test_initial_factor_initialization(num_nodes, num_edges):
    """
    For any converted graph, all edges should have risk_factor = 0 and
    congestion_factor = 0 initially.
    
    Validates: Requirements 2.5, 2.6
    """
    # Create a mock OSM graph
    G = nx.MultiDiGraph()
    
    # Add nodes
    for i in range(num_nodes):
        G.add_node(i, x=-122.0 + i * 0.0001, y=37.8 + i * 0.0001)
    
    # Add edges
    edges_added = 0
    for _ in range(num_edges):
        if edges_added >= num_edges:
            break
        u = edges_added % num_nodes
        v = (edges_added + 1) % num_nodes
        if u != v:
            G.add_edge(u, v, length=100.0)
            edges_added += 1
    
    # Skip if no edges
    assume(G.number_of_edges() > 0)
    
    # Convert to internal format
    converter = GraphConverter()
    graph_manager, coordinate_mapping = converter.convert_osm_to_internal(G)
    
    # Verify all edges have zero risk and congestion factors
    for vertex_id in range(num_nodes):
        neighbors = graph_manager.get_neighbors(str(vertex_id))
        for edge in neighbors:
            assert edge.base_risk == 0.0, \
                f"Edge {edge.source}->{edge.target} has non-zero risk: {edge.base_risk}"
            assert edge.base_congestion == 0.0, \
                f"Edge {edge.source}->{edge.target} has non-zero congestion: {edge.base_congestion}"


# Feature: osm-road-network-integration, Property 9: Weight Calculation Formula
@given(
    num_nodes=st.integers(min_value=3, max_value=15),
    num_edges=st.integers(min_value=2, max_value=30)
)
@settings(max_examples=100, deadline=None)
def test_weight_calculation_formula(num_nodes, num_edges):
    """
    For any converted graph with zero risk and congestion factors,
    the edge weight should equal the base distance.
    
    Validates: Requirements 2.7
    """
    # Create a mock OSM graph
    G = nx.MultiDiGraph()
    
    # Add nodes
    for i in range(num_nodes):
        G.add_node(i, x=-122.0 + i * 0.0001, y=37.8 + i * 0.0001)
    
    # Add edges with known distances
    edge_distances = {}
    edges_added = 0
    for _ in range(num_edges):
        if edges_added >= num_edges:
            break
        u = edges_added % num_nodes
        v = (edges_added + 1) % num_nodes
        if u != v:
            distance = 100.0 + (edges_added * 25.0)
            G.add_edge(u, v, length=distance)
            edge_distances[(u, v)] = distance
            edges_added += 1
    
    # Skip if no edges
    assume(len(edge_distances) > 0)
    
    # Convert to internal format
    converter = GraphConverter()
    graph_manager, coordinate_mapping = converter.convert_osm_to_internal(G)
    
    # Create node mapping
    osm_nodes = list(G.nodes())
    
    # Verify weight = distance (since risk and congestion are 0)
    for (u_osm, v_osm), expected_distance in edge_distances.items():
        # Find internal IDs
        u_internal = osm_nodes.index(u_osm)
        v_internal = osm_nodes.index(v_osm)
        
        # Convert to string IDs
        source = str(u_internal)
        target = str(v_internal)
        
        # Get edge weight
        weight = graph_manager.get_edge_weight(source, target)
        
        # Weight should equal distance (risk=0, congestion=0)
        assert abs(weight - expected_distance) < 1e-6, \
            f"Weight mismatch: {weight} != {expected_distance}"
