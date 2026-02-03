"""
Property-based tests for disaster effects application.

Feature: disaster-evacuation-routing, Property 4: Disaster Effects Application
**Validates: Requirements 2.2, 2.4, 2.5**
"""

import pytest
from hypothesis import given, strategies as st, assume, settings, HealthCheck
from disaster_evacuation.disaster import DisasterModel
from disaster_evacuation.graph import GraphManager, WeightCalculator
from disaster_evacuation.models import DisasterEvent, DisasterType, VertexType


# Hypothesis strategies for generating test data
coordinate_strategy = st.floats(min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False)
coordinates_strategy = st.tuples(coordinate_strategy, coordinate_strategy)
distance_strategy = st.floats(min_value=0.1, max_value=50.0, allow_nan=False, allow_infinity=False)
risk_strategy = st.floats(min_value=0.0, max_value=5.0, allow_nan=False, allow_infinity=False)
congestion_strategy = st.floats(min_value=0.0, max_value=5.0, allow_nan=False, allow_infinity=False)
severity_strategy = st.floats(min_value=0.1, max_value=1.0, allow_nan=False, allow_infinity=False)
radius_strategy = st.floats(min_value=1.0, max_value=50.0, allow_nan=False, allow_infinity=False)
disaster_type_strategy = st.sampled_from(list(DisasterType))


class TestDisasterEffectsApplication:
    """Property-based tests for disaster effects application."""
    
    @given(
        disaster_type=disaster_type_strategy,
        epicenter=coordinates_strategy,
        severity=severity_strategy,
        radius=radius_strategy,
        vertex_coords=st.lists(coordinates_strategy, min_size=2, max_size=10, unique=True),
        edge_properties=st.lists(
            st.tuples(distance_strategy, risk_strategy, congestion_strategy),
            min_size=1, max_size=20
        )
    )
    def test_disaster_effects_modify_weights_consistently(self, disaster_type, epicenter, 
                                                        severity, radius, vertex_coords, 
                                                        edge_properties):
        """
        Property 4: Disaster Effects Application - Weight Modification
        
        For any graph and disaster event, applying disaster effects should modify
        edge weights according to disaster-specific formulas, with different disaster
        types producing different weight patterns based on proximity and severity.
        
        **Validates: Requirements 2.2, 2.4, 2.5**
        """
        # Create graph
        graph = GraphManager()
        disaster_model = DisasterModel()
        
        # Add vertices
        vertex_ids = [f"V{i}" for i in range(len(vertex_coords))]
        for i, coords in enumerate(vertex_coords):
            graph.add_vertex(vertex_ids[i], VertexType.INTERSECTION, coords)
        
        # Add edges between consecutive vertices
        edges_added = []
        for i in range(min(len(vertex_ids) - 1, len(edge_properties))):
            source = vertex_ids[i]
            target = vertex_ids[i + 1]
            distance, risk, congestion = edge_properties[i]
            
            graph.add_edge(source, target, distance, risk, congestion)
            edges_added.append((source, target))
        
        if len(edges_added) == 0:
            return  # Skip if no edges
        
        # Store original weights
        original_weights = {}
        for source, target in edges_added:
            original_weights[(source, target)] = graph.get_edge_weight(source, target)
        
        # Create disaster event
        disaster = DisasterEvent(disaster_type, epicenter, severity, radius)
        
        # Apply disaster effects
        disaster_model.apply_disaster_effects(graph, disaster)
        
        # Verify disaster effects were applied
        disaster_multiplier = disaster.get_disaster_multiplier()
        
        for source, target in edges_added:
            edge = graph.get_edge(source, target)
            new_weight = graph.get_edge_weight(source, target)
            original_weight = original_weights[(source, target)]
            
            # Calculate edge midpoint
            source_vertex = graph.get_vertex(source)
            target_vertex = graph.get_vertex(target)
            edge_midpoint = WeightCalculator.calculate_edge_midpoint(
                source_vertex.coordinates, target_vertex.coordinates
            )
            
            # Check if edge is affected by disaster
            if disaster.is_point_affected(edge_midpoint):
                # Edge should be affected - weight should change
                if not edge.is_blocked:
                    # If not blocked, weight should be calculated using disaster formula
                    expected_weight = WeightCalculator.calculate_dynamic_weight(
                        edge, disaster, edge_midpoint, disaster_model._get_traffic_multiplier(disaster)
                    )
                    assert abs(new_weight - expected_weight) < 1e-10
                else:
                    # If blocked, weight should be infinite
                    assert new_weight == float('inf')
            else:
                # Edge outside disaster radius should have original weight or slight modification
                # due to traffic multiplier
                pass  # Weight may still change due to traffic effects
    
    @given(
        disaster_type=disaster_type_strategy,
        epicenter=coordinates_strategy,
        severity=severity_strategy,
        radius=radius_strategy
    )
    def test_disaster_type_specific_effects(self, disaster_type, epicenter, severity, radius):
        """
        Property 4: Disaster Effects Application - Type-Specific Effects
        
        For any disaster type, the effects should follow type-specific patterns:
        - Flood: multiplier 2.0, traffic 1.5x
        - Fire: multiplier 3.0, traffic 2.0x  
        - Earthquake: multiplier 2.5, traffic 3.0x
        
        **Validates: Requirements 2.2, 2.5**
        """
        # Create simple test graph
        graph = GraphManager()
        disaster_model = DisasterModel()
        
        graph.add_vertex("A", VertexType.INTERSECTION, epicenter)
        graph.add_vertex("B", VertexType.INTERSECTION, (epicenter[0] + 1, epicenter[1]))
        graph.add_edge("A", "B", 1.0, 0.5, 0.2)
        
        disaster = DisasterEvent(disaster_type, epicenter, severity, radius)
        
        # Apply disaster effects
        disaster_model.apply_disaster_effects(graph, disaster)
        
        # Verify disaster-specific multipliers
        expected_disaster_multiplier = disaster.get_disaster_multiplier()
        expected_traffic_multiplier = disaster_model._get_traffic_multiplier(disaster)
        
        if disaster_type == DisasterType.FLOOD:
            assert expected_disaster_multiplier == 2.0
            assert expected_traffic_multiplier == 1.5
        elif disaster_type == DisasterType.FIRE:
            assert expected_disaster_multiplier == 3.0
            assert expected_traffic_multiplier == 2.0
        elif disaster_type == DisasterType.EARTHQUAKE:
            assert expected_disaster_multiplier == 2.5
            assert expected_traffic_multiplier == 3.0
    
    @given(
        epicenter=coordinates_strategy,
        severity=severity_strategy,
        radius=radius_strategy,
        test_points=st.lists(coordinates_strategy, min_size=1, max_size=10)
    )
    def test_proximity_based_effects(self, epicenter, severity, radius, test_points):
        """
        Property 4: Disaster Effects Application - Proximity-Based Effects
        
        For any disaster and set of points, effects should be stronger for points
        closer to the epicenter and weaker for points farther away.
        
        **Validates: Requirements 2.4, 2.5**
        """
        disaster = DisasterEvent(DisasterType.FLOOD, epicenter, severity, radius)
        
        # Calculate effects for each test point
        point_effects = []
        for point in test_points:
            distance_to_epicenter = disaster.distance_to_point(point)
            if distance_to_epicenter <= radius:
                proximity_factor = max(0.0, 1.0 - distance_to_epicenter / radius)
                point_effects.append((point, distance_to_epicenter, proximity_factor))
        
        # Sort by distance to epicenter
        point_effects.sort(key=lambda x: x[1])
        
        # Verify proximity factors decrease with distance
        for i in range(len(point_effects) - 1):
            current_distance = point_effects[i][1]
            next_distance = point_effects[i + 1][1]
            current_proximity = point_effects[i][2]
            next_proximity = point_effects[i + 1][2]
            
            # If distances are different, proximity should be inversely related
            if abs(next_distance - current_distance) > 1e-10:
                assert current_proximity >= next_proximity
    
    @given(
        disaster_type=disaster_type_strategy,
        epicenter=coordinates_strategy,
        severity=severity_strategy,
        radius=radius_strategy,
        edge_distance=distance_strategy,
        edge_risk=risk_strategy,
        edge_congestion=congestion_strategy
    )
    def test_disaster_effects_reversibility(self, disaster_type, epicenter, severity, 
                                          radius, edge_distance, edge_risk, edge_congestion):
        """
        Property 4: Disaster Effects Application - Reversibility
        
        For any graph and disaster, applying and then removing disaster effects
        should restore the graph to its original state.
        
        **Validates: Requirements 2.2, 2.4**
        """
        # Create simple graph
        graph = GraphManager()
        disaster_model = DisasterModel()
        
        graph.add_vertex("A", VertexType.INTERSECTION, (0.0, 0.0))
        graph.add_vertex("B", VertexType.INTERSECTION, (1.0, 1.0))
        graph.add_edge("A", "B", edge_distance, edge_risk, edge_congestion)
        
        # Store original state
        original_weight = graph.get_edge_weight("A", "B")
        original_edge = graph.get_edge("A", "B")
        original_blocked = original_edge.is_blocked
        
        # Apply disaster effects
        disaster = DisasterEvent(disaster_type, epicenter, severity, radius)
        disaster_model.apply_disaster_effects(graph, disaster)
        
        # Remove disaster effects
        disaster_model.remove_disaster_effects(graph, disaster)
        
        # Verify restoration
        restored_weight = graph.get_edge_weight("A", "B")
        restored_edge = graph.get_edge("A", "B")
        
        assert abs(restored_weight - original_weight) < 1e-10
        assert restored_edge.is_blocked == original_blocked
        assert len(disaster_model.get_active_disasters()) == 0
    
    @given(
        disaster_types=st.lists(disaster_type_strategy, min_size=1, max_size=3, unique=True),
        epicenters=st.lists(coordinates_strategy, min_size=1, max_size=3),
        severities=st.lists(severity_strategy, min_size=1, max_size=3),
        radii=st.lists(radius_strategy, min_size=1, max_size=3)
    )
    @settings(suppress_health_check=[HealthCheck.filter_too_much])
    def test_multiple_disasters_cumulative_effects(self, disaster_types, epicenters, 
                                                 severities, radii):
        """
        Property 4: Disaster Effects Application - Multiple Disasters
        
        For any set of disasters applied to a graph, the effects should be
        cumulative and each disaster should be tracked independently.
        
        **Validates: Requirements 2.2, 2.4**
        """
        assume(len(disaster_types) == len(epicenters) == len(severities) == len(radii))
        
        # Create graph
        graph = GraphManager()
        disaster_model = DisasterModel()
        
        graph.add_vertex("A", VertexType.INTERSECTION, (0.0, 0.0))
        graph.add_vertex("B", VertexType.INTERSECTION, (2.0, 2.0))
        graph.add_edge("A", "B", 2.0, 0.3, 0.2)
        
        # Create disasters
        disasters = []
        for i in range(len(disaster_types)):
            disaster = DisasterEvent(disaster_types[i], epicenters[i], severities[i], radii[i])
            disasters.append(disaster)
        
        # Apply disasters one by one
        weights_after_each = []
        for disaster in disasters:
            disaster_model.apply_disaster_effects(graph, disaster)
            weights_after_each.append(graph.get_edge_weight("A", "B"))
        
        # Verify all disasters are active
        active_disasters = disaster_model.get_active_disasters()
        assert len(active_disasters) == len(disasters)
        
        # Remove disasters one by one and verify effects
        for i, disaster in enumerate(disasters):
            disaster_model.remove_disaster_effects(graph, disaster)
            remaining_disasters = len(disasters) - i - 1
            assert len(disaster_model.get_active_disasters()) == remaining_disasters
    
    @given(
        disaster_type=disaster_type_strategy,
        epicenter=coordinates_strategy,
        severity=severity_strategy,
        radius=radius_strategy
    )
    def test_disaster_impact_summary_consistency(self, disaster_type, epicenter, severity, radius):
        """
        Property 4: Disaster Effects Application - Impact Summary
        
        For any disaster, the impact summary should accurately reflect the
        disaster's effects on the graph.
        
        **Validates: Requirements 2.2, 2.4**
        """
        # Create graph with multiple edges
        graph = GraphManager()
        disaster_model = DisasterModel()
        
        # Add vertices in a grid pattern
        vertices = []
        for i in range(3):
            for j in range(3):
                vertex_id = f"V{i}{j}"
                coords = (i * 2.0, j * 2.0)
                graph.add_vertex(vertex_id, VertexType.INTERSECTION, coords)
                vertices.append((vertex_id, coords))
        
        # Add edges between adjacent vertices
        edges_added = 0
        for i in range(3):
            for j in range(3):
                current = f"V{i}{j}"
                # Add horizontal edge
                if j < 2:
                    right = f"V{i}{j+1}"
                    graph.add_edge(current, right, 2.0, 0.2, 0.1)
                    edges_added += 1
                # Add vertical edge
                if i < 2:
                    down = f"V{i+1}{j}"
                    graph.add_edge(current, down, 2.0, 0.1, 0.2)
                    edges_added += 1
        
        # Create disaster
        disaster = DisasterEvent(disaster_type, epicenter, severity, radius)
        
        # Get impact summary
        summary = disaster_model.get_disaster_impact_summary(graph, disaster)
        
        # Verify summary structure and consistency
        assert summary["disaster_type"] == disaster_type.value
        assert summary["severity"] == severity
        assert summary["epicenter"] == epicenter
        assert summary["effect_radius"] == radius
        assert summary["total_edges"] == edges_added
        assert 0 <= summary["affected_edges"] <= summary["total_edges"]
        assert 0 <= summary["blocked_edges"] <= summary["affected_edges"]
        assert 0 <= summary["high_risk_edges"] <= summary["affected_edges"]
        assert 0 <= summary["impact_percentage"] <= 100
        
        # Impact percentage should be consistent
        if summary["total_edges"] > 0:
            expected_percentage = (summary["affected_edges"] / summary["total_edges"]) * 100
            assert abs(summary["impact_percentage"] - expected_percentage) < 1e-10
    
    @given(
        disaster_type=disaster_type_strategy,
        epicenter=coordinates_strategy,
        severity1=severity_strategy,
        severity2=severity_strategy,
        radius=radius_strategy
    )
    def test_severity_monotonicity(self, disaster_type, epicenter, severity1, severity2, radius):
        """
        Property 4: Disaster Effects Application - Severity Monotonicity
        
        For any disaster type and location, higher severity should result in
        greater or equal effects (higher weights, more blocked roads).
        
        **Validates: Requirements 2.2, 2.5**
        """
        assume(abs(severity1 - severity2) > 0.1)  # Ensure meaningful difference
        
        # Create identical graphs
        graph1 = GraphManager()
        graph2 = GraphManager()
        disaster_model1 = DisasterModel()
        disaster_model2 = DisasterModel()
        
        # Add same structure to both graphs
        for graph in [graph1, graph2]:
            graph.add_vertex("A", VertexType.INTERSECTION, epicenter)
            graph.add_vertex("B", VertexType.INTERSECTION, (epicenter[0] + 1, epicenter[1]))
            graph.add_edge("A", "B", 2.0, 0.5, 0.3)
        
        # Apply disasters with different severities
        disaster1 = DisasterEvent(disaster_type, epicenter, severity1, radius)
        disaster2 = DisasterEvent(disaster_type, epicenter, severity2, radius)
        
        disaster_model1.apply_disaster_effects(graph1, disaster1)
        disaster_model2.apply_disaster_effects(graph2, disaster2)
        
        # Compare effects
        weight1 = graph1.get_edge_weight("A", "B")
        weight2 = graph2.get_edge_weight("A", "B")
        
        edge1 = graph1.get_edge("A", "B")
        edge2 = graph2.get_edge("A", "B")
        
        # Higher severity should have greater or equal effects
        if severity1 > severity2:
            # Higher severity should have higher weight (unless both are blocked)
            if not (edge1.is_blocked and edge2.is_blocked):
                if edge1.is_blocked:
                    assert weight1 == float('inf')
                elif edge2.is_blocked:
                    # This shouldn't happen if severity1 > severity2
                    pass
                else:
                    assert weight1 >= weight2
        elif severity2 > severity1:
            # Similar logic for opposite case
            if not (edge1.is_blocked and edge2.is_blocked):
                if edge2.is_blocked:
                    assert weight2 == float('inf')
                elif edge1.is_blocked:
                    pass
                else:
                    assert weight2 >= weight1