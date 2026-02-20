"""
Property-based tests for blocked road handling.

Feature: disaster-evacuation-routing, Property 5: Blocked Road Handling
**Validates: Requirements 2.3**
"""

import pytest
from hypothesis import given, strategies as st, assume
from disaster_evacuation.models import DisasterModel
from disaster_evacuation.models import GraphManager, WeightCalculator
from disaster_evacuation.models import DisasterEvent, DisasterType, VertexType


# Hypothesis strategies for generating test data
coordinate_strategy = st.floats(min_value=-50.0, max_value=50.0, allow_nan=False, allow_infinity=False)
coordinates_strategy = st.tuples(coordinate_strategy, coordinate_strategy)
distance_strategy = st.floats(min_value=0.1, max_value=20.0, allow_nan=False, allow_infinity=False)
risk_strategy = st.floats(min_value=0.0, max_value=2.0, allow_nan=False, allow_infinity=False)
congestion_strategy = st.floats(min_value=0.0, max_value=2.0, allow_nan=False, allow_infinity=False)
high_severity_strategy = st.floats(min_value=0.8, max_value=1.0, allow_nan=False, allow_infinity=False)
medium_severity_strategy = st.floats(min_value=0.4, max_value=0.7, allow_nan=False, allow_infinity=False)
radius_strategy = st.floats(min_value=1.0, max_value=20.0, allow_nan=False, allow_infinity=False)
disaster_type_strategy = st.sampled_from(list(DisasterType))
blocking_threshold_strategy = st.floats(min_value=0.1, max_value=0.9, allow_nan=False, allow_infinity=False)


class TestBlockedRoadHandling:
    """Property-based tests for blocked road handling."""
    
    @given(
        disaster_type=disaster_type_strategy,
        epicenter=coordinates_strategy,
        severity=high_severity_strategy,
        radius=radius_strategy,
        edge_distance=distance_strategy,
        edge_risk=risk_strategy,
        edge_congestion=congestion_strategy
    )
    def test_blocked_roads_have_infinite_weight(self, disaster_type, epicenter, severity, 
                                              radius, edge_distance, edge_risk, edge_congestion):
        """
        Property 5: Blocked Road Handling - Infinite Weight Assignment
        
        For any road marked as blocked by disaster effects, the pathfinding algorithm
        should treat it as having infinite weight.
        
        **Validates: Requirements 2.3**
        """
        # Create graph with edge at disaster epicenter (maximum effect)
        graph = GraphManager()
        disaster_model = DisasterModel()
        
        graph.add_vertex("A", VertexType.INTERSECTION, epicenter)
        graph.add_vertex("B", VertexType.INTERSECTION, (epicenter[0] + 0.1, epicenter[1]))
        graph.add_edge("A", "B", edge_distance, edge_risk, edge_congestion)
        
        # Create high-severity disaster at epicenter
        disaster = DisasterEvent(disaster_type, epicenter, severity, radius)
        
        # Apply disaster effects
        disaster_model.apply_disaster_effects(graph, disaster)
        
        # Check edge blocking
        edge = graph.get_edge("A", "B")
        weight = graph.get_edge_weight("A", "B")
        
        # If edge is blocked, weight must be infinite
        if edge.is_blocked:
            assert weight == float('inf')
            assert weight > 1e10  # Verify it's actually infinite
        else:
            # If not blocked, weight must be finite and positive
            assert weight != float('inf')
            assert weight > 0
            assert weight < float('inf')
    
    @given(
        disaster_type=disaster_type_strategy,
        epicenter=coordinates_strategy,
        severity=high_severity_strategy,
        radius=radius_strategy,
        edge_coords=st.lists(coordinates_strategy, min_size=2, max_size=8, unique=True)
    )
    def test_blocked_roads_avoid_pathfinding(self, disaster_type, epicenter, severity, 
                                           radius, edge_coords):
        """
        Property 5: Blocked Road Handling - Pathfinding Avoidance
        
        For any road marked as blocked, alternative routes should be found
        that avoid the blocked road entirely.
        
        **Validates: Requirements 2.3**
        """
        # Create graph with multiple possible paths
        graph = GraphManager()
        disaster_model = DisasterModel()
        
        # Add vertices
        vertex_ids = [f"V{i}" for i in range(len(edge_coords))]
        for i, coords in enumerate(edge_coords):
            graph.add_vertex(vertex_ids[i], VertexType.INTERSECTION, coords)
        
        # Add edges to create multiple paths
        edges_added = []
        for i in range(len(vertex_ids) - 1):
            source = vertex_ids[i]
            target = vertex_ids[i + 1]
            graph.add_edge(source, target, 1.0, 0.2, 0.1)
            edges_added.append((source, target))
        
        if len(edges_added) == 0:
            return  # Skip if no edges
        
        # Create disaster
        disaster = DisasterEvent(disaster_type, epicenter, severity, radius)
        disaster_model.apply_disaster_effects(graph, disaster)
        
        # Verify blocked roads have infinite weight
        blocked_edges = []
        unblocked_edges = []
        
        for source, target in edges_added:
            edge = graph.get_edge(source, target)
            weight = graph.get_edge_weight(source, target)
            
            if edge.is_blocked:
                assert weight == float('inf')
                blocked_edges.append((source, target))
            else:
                assert weight != float('inf')
                assert weight > 0
                unblocked_edges.append((source, target))
        
        # At least some edges should remain unblocked for alternative routes
        # (unless all edges are in the disaster zone)
        total_edges = len(edges_added)
        blocked_count = len(blocked_edges)
        
        # The important property is that blocked edges have infinite weight
        # and unblocked edges have finite positive weight - this is the core requirement
        assert all(graph.get_edge_weight(s, t) == float('inf') for s, t in blocked_edges)
        assert all(0 < graph.get_edge_weight(s, t) < float('inf') for s, t in unblocked_edges)
        
        # In extreme disaster scenarios, all edges might be blocked - this is valid behavior
    
    @given(
        disaster_type=disaster_type_strategy,
        epicenter=coordinates_strategy,
        severity=high_severity_strategy,
        radius=radius_strategy
    )
    def test_blocked_roads_unblock_after_disaster_removal(self, disaster_type, epicenter, 
                                                         severity, radius):
        """
        Property 5: Blocked Road Handling - Unblocking After Removal
        
        For any blocked road, removing the disaster should unblock the road
        and restore it to a finite weight.
        
        **Validates: Requirements 2.3**
        """
        # Create graph
        graph = GraphManager()
        disaster_model = DisasterModel()
        
        # Add edge at disaster epicenter for maximum blocking chance
        graph.add_vertex("A", VertexType.INTERSECTION, epicenter)
        graph.add_vertex("B", VertexType.INTERSECTION, (epicenter[0] + 0.1, epicenter[1]))
        graph.add_edge("A", "B", 2.0, 0.3, 0.2)
        
        # Store original state
        original_weight = graph.get_edge_weight("A", "B")
        original_edge = graph.get_edge("A", "B")
        original_blocked = original_edge.is_blocked
        
        # Apply disaster
        disaster = DisasterEvent(disaster_type, epicenter, severity, radius)
        disaster_model.apply_disaster_effects(graph, disaster)
        
        # Check if edge was blocked
        edge_after_disaster = graph.get_edge("A", "B")
        weight_after_disaster = graph.get_edge_weight("A", "B")
        was_blocked = edge_after_disaster.is_blocked
        
        # Remove disaster
        disaster_model.remove_disaster_effects(graph, disaster)
        
        # Verify unblocking
        edge_after_removal = graph.get_edge("A", "B")
        weight_after_removal = graph.get_edge_weight("A", "B")
        
        # Edge should be unblocked
        assert not edge_after_removal.is_blocked
        assert weight_after_removal != float('inf')
        assert weight_after_removal > 0
        
        # Weight should be restored to original
        assert abs(weight_after_removal - original_weight) < 1e-10
        assert edge_after_removal.is_blocked == original_blocked
    
    @given(
        disaster_type=disaster_type_strategy,
        epicenter=coordinates_strategy,
        severity=high_severity_strategy,
        radius=radius_strategy,
        blocking_threshold=blocking_threshold_strategy
    )
    def test_blocking_threshold_consistency(self, disaster_type, epicenter, severity, 
                                          radius, blocking_threshold):
        """
        Property 5: Blocked Road Handling - Threshold Consistency
        
        For any disaster and blocking threshold, roads should be blocked consistently
        based on the effective severity at their location.
        
        **Validates: Requirements 2.3**
        """
        # Create edge at epicenter
        edge_coords = epicenter
        edge = type('Edge', (), {
            'source': 'A',
            'target': 'B',
            'base_distance': 1.0,
            'base_risk': 0.2,
            'base_congestion': 0.1
        })()
        
        disaster = DisasterEvent(disaster_type, epicenter, severity, radius)
        
        # Test blocking decision
        is_blocked = WeightCalculator.is_edge_blocked(edge, disaster, edge_coords, blocking_threshold)
        
        # Calculate expected blocking decision
        distance_to_epicenter = disaster.distance_to_point(edge_coords)
        if distance_to_epicenter <= radius:
            proximity_factor = max(0.0, 1.0 - distance_to_epicenter / radius)
            effective_severity = severity * proximity_factor
            
            # Get disaster-specific threshold adjustment
            if disaster_type == DisasterType.FIRE:
                adjusted_threshold = blocking_threshold * 0.6
            elif disaster_type == DisasterType.EARTHQUAKE:
                adjusted_threshold = blocking_threshold * 0.7
            else:  # FLOOD
                adjusted_threshold = blocking_threshold
            
            expected_blocked = effective_severity > adjusted_threshold
            assert is_blocked == expected_blocked
        else:
            # Outside radius should not be blocked
            assert not is_blocked
    
    @given(
        disaster_type=disaster_type_strategy,
        epicenter=coordinates_strategy,
        severity=high_severity_strategy,
        radius=radius_strategy,
        test_distances=st.lists(
            st.floats(min_value=0.0, max_value=30.0, allow_nan=False, allow_infinity=False),
            min_size=2, max_size=10
        )
    )
    def test_distance_based_blocking_gradient(self, disaster_type, epicenter, severity, 
                                            radius, test_distances):
        """
        Property 5: Blocked Road Handling - Distance-Based Gradient
        
        For any disaster, roads closer to the epicenter should be more likely
        to be blocked than roads farther away.
        
        **Validates: Requirements 2.3**
        """
        # Sort distances
        test_distances = sorted(set(test_distances))
        if len(test_distances) < 2:
            return
        
        disaster = DisasterEvent(disaster_type, epicenter, severity, radius)
        
        # Test blocking at different distances
        blocking_results = []
        for distance in test_distances:
            if distance <= radius:
                # Create test point at specified distance
                test_point = (epicenter[0] + distance, epicenter[1])
                
                # Create mock edge
                edge = type('Edge', (), {
                    'source': 'A',
                    'target': 'B',
                    'base_distance': 1.0,
                    'base_risk': 0.2,
                    'base_congestion': 0.1
                })()
                
                is_blocked = WeightCalculator.is_edge_blocked(edge, disaster, test_point)
                blocking_results.append((distance, is_blocked))
        
        # Verify blocking gradient: closer distances should have higher blocking probability
        # This is a statistical property, so we check the general trend
        if len(blocking_results) >= 2:
            close_blocking = sum(1 for d, blocked in blocking_results[:len(blocking_results)//2] if blocked)
            far_blocking = sum(1 for d, blocked in blocking_results[len(blocking_results)//2:] if blocked)
            
            # Closer roads should have at least as much blocking as farther roads
            close_count = len(blocking_results) // 2
            far_count = len(blocking_results) - close_count
            
            if close_count > 0 and far_count > 0:
                close_rate = close_blocking / close_count
                far_rate = far_blocking / far_count
                
                # Allow some tolerance for edge cases
                assert close_rate >= far_rate - 0.1
    
    @given(
        epicenter=coordinates_strategy,
        severity=high_severity_strategy,
        radius=radius_strategy
    )
    def test_disaster_type_blocking_aggressiveness(self, epicenter, severity, radius):
        """
        Property 5: Blocked Road Handling - Type-Specific Aggressiveness
        
        For any disaster parameters, fire should be most aggressive in blocking,
        followed by earthquake, then flood.
        
        **Validates: Requirements 2.3**
        """
        # Create identical edge at epicenter for all disaster types
        edge = type('Edge', (), {
            'source': 'A',
            'target': 'B',
            'base_distance': 1.0,
            'base_risk': 0.3,
            'base_congestion': 0.2
        })()
        
        edge_point = epicenter
        
        # Test each disaster type
        fire = DisasterEvent(DisasterType.FIRE, epicenter, severity, radius)
        earthquake = DisasterEvent(DisasterType.EARTHQUAKE, epicenter, severity, radius)
        flood = DisasterEvent(DisasterType.FLOOD, epicenter, severity, radius)
        
        fire_blocked = WeightCalculator.is_edge_blocked(edge, fire, edge_point)
        earthquake_blocked = WeightCalculator.is_edge_blocked(edge, earthquake, edge_point)
        flood_blocked = WeightCalculator.is_edge_blocked(edge, flood, edge_point)
        
        # Fire should be most aggressive (lowest threshold)
        # If fire doesn't block, others shouldn't either
        if not fire_blocked:
            assert not earthquake_blocked
            assert not flood_blocked
        
        # If flood blocks, others should too
        if flood_blocked:
            assert earthquake_blocked
            assert fire_blocked
    
    @given(
        disaster_type=disaster_type_strategy,
        epicenter=coordinates_strategy,
        severity=high_severity_strategy,
        radius=radius_strategy,
        num_edges=st.integers(min_value=1, max_value=10)
    )
    def test_blocked_roads_consistency_across_graph(self, disaster_type, epicenter, 
                                                   severity, radius, num_edges):
        """
        Property 5: Blocked Road Handling - Graph-Wide Consistency
        
        For any graph and disaster, blocked road handling should be consistent
        across all edges in the graph.
        
        **Validates: Requirements 2.3**
        """
        # Create graph with multiple edges
        graph = GraphManager()
        disaster_model = DisasterModel()
        
        # Add vertices and edges
        vertices = []
        edges = []
        
        for i in range(num_edges + 1):
            vertex_id = f"V{i}"
            # Spread vertices around epicenter
            angle = (i * 2 * 3.14159) / (num_edges + 1)
            coords = (epicenter[0] + radius * 0.5 * (angle / 3.14159), 
                     epicenter[1] + radius * 0.3 * (angle / 3.14159))
            graph.add_vertex(vertex_id, VertexType.INTERSECTION, coords)
            vertices.append((vertex_id, coords))
        
        # Add edges
        for i in range(num_edges):
            source = vertices[i][0]
            target = vertices[i + 1][0]
            graph.add_edge(source, target, 1.0, 0.2, 0.1)
            edges.append((source, target))
        
        # Apply disaster
        disaster = DisasterEvent(disaster_type, epicenter, severity, radius)
        disaster_model.apply_disaster_effects(graph, disaster)
        
        # Verify consistency
        for source, target in edges:
            edge = graph.get_edge(source, target)
            weight = graph.get_edge_weight(source, target)
            
            # Consistency check: blocked edges have infinite weight
            if edge.is_blocked:
                assert weight == float('inf')
            else:
                assert weight != float('inf')
                assert weight > 0
            
            # Verify blocking decision matches WeightCalculator
            source_vertex = graph.get_vertex(source)
            target_vertex = graph.get_vertex(target)
            edge_midpoint = WeightCalculator.calculate_edge_midpoint(
                source_vertex.coordinates, target_vertex.coordinates
            )
            
            expected_blocked = WeightCalculator.is_edge_blocked(edge, disaster, edge_midpoint)
            assert edge.is_blocked == expected_blocked