"""
Property-based tests for Disaster_Modeler component.

These tests verify universal correctness properties for disaster modeling
using Hypothesis for randomized testing.
"""

import pytest
from hypothesis import given, strategies as st, settings, assume
import networkx as nx
from disaster_evacuation.osm.graph_converter import GraphConverter
from disaster_evacuation.disaster.disaster_modeler import DisasterModeler


# Feature: osm-road-network-integration, Property 10: Distance-Based Disaster Impact
@given(
    num_nodes=st.integers(min_value=4, max_value=15),
    radius=st.floats(min_value=100.0, max_value=2000.0)
)
@settings(max_examples=50, deadline=None)
def test_distance_based_disaster_impact(num_nodes, radius):
    """
    For any graph with disaster epicenter E and two edges e1 and e2,
    if e1 is closer to E than e2, then e1 should be affected at least
    as much as e2 (equal or greater weight increase).
    
    Validates: Requirements 3.2
    """
    # Create a simple OSM graph
    G = nx.MultiDiGraph()
    
    # Add nodes in a line
    for i in range(num_nodes):
        lat = 37.8 + i * 0.001
        lon = -122.0
        G.add_node(i, x=lon, y=lat)
    
    # Add edges
    for i in range(num_nodes - 1):
        G.add_edge(i, i + 1, length=100.0)
    
    # Convert to internal format
    converter = GraphConverter()
    graph_manager, coord_mapping = converter.convert_osm_to_internal(G)
    
    # Store initial weights
    initial_weights = {}
    for i in range(num_nodes - 1):
        initial_weights[i] = graph_manager.get_edge_weight(str(i), str(i + 1))
    
    # Apply flood disaster at first node
    epicenter = coord_mapping[0]
    modeler = DisasterModeler(graph_manager, coord_mapping)
    modeler.apply_flood(epicenter, radius, risk_multiplier=0.5)
    
    # Get final weights
    final_weights = {}
    for i in range(num_nodes - 1):
        final_weights[i] = graph_manager.get_edge_weight(str(i), str(i + 1))
    
    # Calculate weight increases
    weight_increases = {}
    for i in range(num_nodes - 1):
        weight_increases[i] = final_weights[i] - initial_weights[i]
    
    # Verify: closer edges should have equal or greater weight increase
    # Edge 0 is closest, edge 1 is farther, etc.
    for i in range(num_nodes - 2):
        if weight_increases[i] > 0:  # If edge i was affected
            # Edge i+1 (farther) should have equal or less increase
            assert weight_increases[i] >= weight_increases[i + 1], \
                f"Closer edge {i} has less increase ({weight_increases[i]}) than farther edge {i+1} ({weight_increases[i+1]})"


# Feature: osm-road-network-integration, Property 11: Flood Radius Effect
@given(
    num_nodes=st.integers(min_value=5, max_value=15),
    radius=st.floats(min_value=50.0, max_value=500.0)
)
@settings(max_examples=50, deadline=None)
def test_flood_radius_effect(num_nodes, radius):
    """
    For any graph and flood disaster with epicenter E and radius R,
    all edges whose midpoint is within distance R of E should have
    risk_factor > 0 after applying the flood.
    
    Validates: Requirements 3.3
    """
    # Create OSM graph
    G = nx.MultiDiGraph()
    
    # Add nodes in a grid
    grid_size = int(num_nodes ** 0.5) + 1
    for i in range(grid_size):
        for j in range(grid_size):
            if i * grid_size + j >= num_nodes:
                break
            lat = 37.8 + i * 0.0001
            lon = -122.0 + j * 0.0001
            G.add_node(i * grid_size + j, x=lon, y=lat)
    
    # Add edges
    for i in range(grid_size - 1):
        for j in range(grid_size - 1):
            node_id = i * grid_size + j
            if node_id >= num_nodes - 1:
                break
            if node_id + 1 < num_nodes:
                G.add_edge(node_id, node_id + 1, length=10.0)
            if node_id + grid_size < num_nodes:
                G.add_edge(node_id, node_id + grid_size, length=10.0)
    
    assume(G.number_of_edges() > 0)
    
    # Convert
    converter = GraphConverter()
    graph_manager, coord_mapping = converter.convert_osm_to_internal(G)
    
    # Store initial weights
    initial_weights = {}
    for i in range(num_nodes):
        vertex_id = str(i)
        if graph_manager.has_vertex(vertex_id):
            neighbors = graph_manager.get_neighbors(vertex_id)
            for edge in neighbors:
                initial_weights[(vertex_id, edge.target)] = edge.current_weight
    
    # Apply flood at center
    epicenter = coord_mapping[0]
    modeler = DisasterModeler(graph_manager, coord_mapping)
    modeler.apply_flood(epicenter, radius, risk_multiplier=0.5)
    
    # Check edges within radius
    for i in range(num_nodes):
        vertex_id = str(i)
        if not graph_manager.has_vertex(vertex_id):
            continue
        
        neighbors = graph_manager.get_neighbors(vertex_id)
        for edge in neighbors:
            target_id = int(edge.target)
            
            # Calculate edge midpoint
            midpoint = modeler._get_edge_midpoint(i, target_id)
            
            # Calculate distance from epicenter
            distance = modeler._haversine_distance(epicenter, midpoint)
            
            # Get current weight
            current_weight = graph_manager.get_edge_weight(vertex_id, edge.target)
            initial_weight = initial_weights.get((vertex_id, edge.target), current_weight)
            
            # If within radius, weight should have increased
            if distance <= radius:
                assert current_weight > initial_weight, \
                    f"Edge within radius ({distance:.1f}m <= {radius}m) was not affected"


# Feature: osm-road-network-integration, Property 12: Fire Blocking Effect
@given(
    num_nodes=st.integers(min_value=4, max_value=12),
    radius=st.floats(min_value=50.0, max_value=300.0)
)
@settings(max_examples=50, deadline=None)
def test_fire_blocking_effect(num_nodes, radius):
    """
    For any graph and fire disaster with epicenter E and radius R,
    all edges whose midpoint is within distance R of E should have
    weight >= BLOCKED_WEIGHT after applying the fire.
    
    Validates: Requirements 3.4
    """
    # Create OSM graph
    G = nx.MultiDiGraph()
    
    # Add nodes in a line
    for i in range(num_nodes):
        lat = 37.8 + i * 0.0001
        lon = -122.0
        G.add_node(i, x=lon, y=lat)
    
    # Add edges
    for i in range(num_nodes - 1):
        G.add_edge(i, i + 1, length=10.0)
    
    # Convert
    converter = GraphConverter()
    graph_manager, coord_mapping = converter.convert_osm_to_internal(G)
    
    # Apply fire at first node
    epicenter = coord_mapping[0]
    modeler = DisasterModeler(graph_manager, coord_mapping)
    modeler.apply_fire(epicenter, radius)
    
    # Check edges within radius
    for i in range(num_nodes - 1):
        vertex_id = str(i)
        target_id = str(i + 1)
        
        # Calculate edge midpoint
        midpoint = modeler._get_edge_midpoint(i, i + 1)
        
        # Calculate distance from epicenter
        distance = modeler._haversine_distance(epicenter, midpoint)
        
        # Get current weight
        current_weight = graph_manager.get_edge_weight(vertex_id, target_id)
        
        # If within radius, should be blocked
        if distance <= radius:
            assert current_weight >= DisasterModeler.BLOCKED_WEIGHT * 0.99, \
                f"Edge within radius ({distance:.1f}m <= {radius}m) is not blocked (weight={current_weight})"


# Feature: osm-road-network-integration, Property 13: Earthquake Dual Effect
@given(
    num_nodes=st.integers(min_value=5, max_value=12),
    radius=st.floats(min_value=100.0, max_value=500.0)
)
@settings(max_examples=50, deadline=None)
def test_earthquake_dual_effect(num_nodes, radius):
    """
    For any graph and earthquake disaster with epicenter E and radius R,
    all edges whose midpoint is within distance R of E should have either
    (congestion_factor > 0) OR (weight >= BLOCKED_WEIGHT).
    
    Validates: Requirements 3.5
    """
    # Create OSM graph
    G = nx.MultiDiGraph()
    
    # Add nodes
    for i in range(num_nodes):
        lat = 37.8 + i * 0.0001
        lon = -122.0 + i * 0.0001
        G.add_node(i, x=lon, y=lat)
    
    # Add edges
    for i in range(num_nodes - 1):
        G.add_edge(i, i + 1, length=100.0)
    
    # Convert
    converter = GraphConverter()
    graph_manager, coord_mapping = converter.convert_osm_to_internal(G)
    
    # Store initial weights
    initial_weights = {}
    for i in range(num_nodes - 1):
        initial_weights[i] = graph_manager.get_edge_weight(str(i), str(i + 1))
    
    # Apply earthquake at center
    epicenter = coord_mapping[num_nodes // 2]
    modeler = DisasterModeler(graph_manager, coord_mapping)
    modeler.apply_earthquake(epicenter, radius, failure_probability=0.3, congestion_multiplier=0.8)
    
    # Check edges within radius
    for i in range(num_nodes - 1):
        vertex_id = str(i)
        target_id = str(i + 1)
        
        # Calculate edge midpoint
        midpoint = modeler._get_edge_midpoint(i, i + 1)
        
        # Calculate distance from epicenter
        distance = modeler._haversine_distance(epicenter, midpoint)
        
        # Get current weight
        current_weight = graph_manager.get_edge_weight(vertex_id, target_id)
        
        # If within radius, should be either blocked OR have increased weight
        if distance <= radius:
            is_blocked = current_weight >= DisasterModeler.BLOCKED_WEIGHT * 0.99
            has_increased = current_weight > initial_weights[i]
            
            assert is_blocked or has_increased, \
                f"Edge within radius ({distance:.1f}m <= {radius}m) was not affected (weight={current_weight}, initial={initial_weights[i]})"
