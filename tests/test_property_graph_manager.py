"""
Property-based tests for GraphManager enhancements.

These tests verify universal correctness properties for coordinate storage
and weight updates using Hypothesis for randomized testing.
"""

import pytest
from hypothesis import given, strategies as st, settings, assume
from disaster_evacuation.graph.graph_manager import GraphManager
from disaster_evacuation.models import VertexType


# Feature: osm-road-network-integration, Property 15: Non-Negative Weight Invariant
@given(
    num_vertices=st.integers(min_value=2, max_value=20),
    num_edges=st.integers(min_value=1, max_value=30),
    weight_updates=st.integers(min_value=0, max_value=10)
)
@settings(max_examples=100, deadline=None)
def test_non_negative_weight_invariant(num_vertices, num_edges, weight_updates):
    """
    For any graph operation (conversion, disaster application, weight update),
    all edge weights should remain non-negative.
    
    Validates: Requirements 5.3
    """
    # Create graph
    graph = GraphManager()
    
    # Add vertices
    for i in range(num_vertices):
        graph.add_vertex(
            vertex_id=str(i),
            vertex_type=VertexType.INTERSECTION,
            coordinates=(37.8 + i * 0.001, -122.0 + i * 0.001),
            capacity=None
        )
    
    # Add edges with positive weights
    edges_added = set()
    for _ in range(num_edges):
        if len(edges_added) >= num_edges:
            break
        u = len(edges_added) % num_vertices
        v = (len(edges_added) + 1) % num_vertices
        if u != v and (u, v) not in edges_added:
            distance = 100.0 + (len(edges_added) * 50.0)
            graph.add_edge(
                source=str(u),
                target=str(v),
                distance=distance,
                base_risk=0.0,
                base_congestion=0.0
            )
            edges_added.add((u, v))
    
    # Skip if no edges
    assume(len(edges_added) > 0)
    
    # Verify all initial weights are non-negative
    for u, v in edges_added:
        weight = graph.get_edge_weight(str(u), str(v))
        assert weight >= 0, f"Initial weight is negative: {weight}"
    
    # Perform random weight updates
    for _ in range(weight_updates):
        if len(edges_added) == 0:
            break
        # Pick a random edge
        u, v = list(edges_added)[_ % len(edges_added)]
        
        # Update with a positive weight
        new_weight = 50.0 + (_ * 25.0)
        graph.update_edge_weight(str(u), str(v), new_weight)
        
        # Verify weight is non-negative
        updated_weight = graph.get_edge_weight(str(u), str(v))
        assert updated_weight >= 0, f"Updated weight is negative: {updated_weight}"
        assert abs(updated_weight - new_weight) < 1e-6, \
            f"Weight not updated correctly: {updated_weight} != {new_weight}"
    
    # Verify all final weights are non-negative
    for u, v in edges_added:
        weight = graph.get_edge_weight(str(u), str(v))
        assert weight >= 0, f"Final weight is negative: {weight}"


@given(
    num_vertices=st.integers(min_value=2, max_value=15)
)
@settings(max_examples=100, deadline=None)
def test_negative_weight_rejected(num_vertices):
    """
    Attempting to set a negative edge weight should raise ValueError.
    
    Validates: Requirements 5.3
    """
    # Create graph
    graph = GraphManager()
    
    # Add vertices
    for i in range(num_vertices):
        graph.add_vertex(
            vertex_id=str(i),
            vertex_type=VertexType.INTERSECTION,
            coordinates=(37.8, -122.0),
            capacity=None
        )
    
    # Add an edge
    if num_vertices >= 2:
        graph.add_edge(
            source="0",
            target="1",
            distance=100.0,
            base_risk=0.0,
            base_congestion=0.0
        )
        
        # Attempt to set negative weight
        with pytest.raises(ValueError) as exc_info:
            graph.update_edge_weight("0", "1", -50.0)
        
        assert "negative" in str(exc_info.value).lower()
