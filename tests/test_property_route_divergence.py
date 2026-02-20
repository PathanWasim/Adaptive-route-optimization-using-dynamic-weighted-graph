"""
Property-based test for route divergence under disaster conditions.

This test verifies that disaster scenarios cause routes to diverge from
the optimal path when roads are blocked or have increased weights.
"""

import pytest
from hypothesis import given, strategies as st, settings, assume, HealthCheck
import networkx as nx
from disaster_evacuation.osm.graph_converter import GraphConverter
from disaster_evacuation.models.disaster_modeler import DisasterModeler
from disaster_evacuation.routing.dijkstra import PathfinderEngine


# Feature: osm-road-network-integration, Property 16: Route Divergence Under Disaster
@given(
    num_nodes=st.integers(min_value=5, max_value=10)
)
@settings(max_examples=20, deadline=None, suppress_health_check=[HealthCheck.filter_too_much])
def test_route_divergence_under_disaster(num_nodes):
    """
    For any graph where a disaster blocks or significantly increases weights
    on edges in the optimal path, the disaster-aware route should differ from
    the normal route.
    
    Validates: Requirements 6.2
    """
    # Create a graph with multiple paths
    G = nx.MultiDiGraph()
    
    # Create nodes in a line
    for i in range(num_nodes):
        lat = 37.8 + i * 0.0002
        lon = -122.0
        G.add_node(i, x=lon, y=lat)
    
    # Add main path
    for i in range(num_nodes - 1):
        G.add_edge(i, i + 1, length=100.0)
    
    # Add alternative longer path (skip connections)
    for i in range(0, num_nodes - 2, 2):
        if i + 2 < num_nodes:
            G.add_edge(i, i + 2, length=250.0)
    
    # Convert to internal format
    converter = GraphConverter()
    graph_manager, coord_mapping = converter.convert_osm_to_internal(G)
    
    # Compute normal route (without disaster)
    pathfinder = PathfinderEngine()
    source = "0"
    target = str(num_nodes - 1)
    
    result = pathfinder.find_shortest_path(graph_manager, source, target)
    normal_path = result.path
    normal_cost = result.total_cost
    
    # Apply disaster at middle of path to block it
    middle_idx = len(normal_path) // 2
    middle_node_id = int(normal_path[middle_idx])
    epicenter = coord_mapping[middle_node_id]
    
    # Apply fire disaster with radius that will block nearby edges
    modeler = DisasterModeler(graph_manager, coord_mapping)
    modeler.apply_fire(epicenter, 50.0)  # Small radius to block specific edges
    
    # Compute disaster-aware route
    result2 = pathfinder.find_shortest_path(graph_manager, source, target)
    disaster_path = result2.path
    disaster_cost = result2.total_cost
    
    # Check if any edge in normal path was blocked
    blocked_count = 0
    for i in range(len(normal_path) - 1):
        u = normal_path[i]
        v = normal_path[i + 1]
        weight = graph_manager.get_edge_weight(u, v)
        if weight >= DisasterModeler.BLOCKED_WEIGHT * 0.99:
            blocked_count += 1
    
    # If disaster blocked edges in the normal path, routes should diverge
    if blocked_count > 0:
        # Routes should be different
        assert normal_path != disaster_path, \
            f"Routes should diverge when {blocked_count} edges are blocked"
