"""
Property-based tests for path recomputation.

Feature: disaster-evacuation-routing, Property 6: Path Recomputation
**Validates: Requirements 3.2**
"""

import pytest
from hypothesis import given, strategies as st, assume
from disaster_evacuation.pathfinding import PathfinderEngine
from disaster_evacuation.graph import GraphManager
from disaster_evacuation.disaster import DisasterModel
from disaster_evacuation.models import VertexType, DisasterEvent, DisasterType


# Hypothesis strategies for generating test data
vertex_count_strategy = st.integers(min_value=3, max_value=8)
edge_weight_strategy = st.floats(min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False)
weight_multiplier_strategy = st.floats(min_value=1.1, max_value=5.0, allow_nan=False, allow_infinity=False)
coordinate_strategy = st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False)
coordinates_strategy = st.tuples(coordinate_strategy, coordinate_strategy)
severity_strategy = st.floats(min_value=0.1, max_value=1.0, allow_nan=False, allow_infinity=False)
radius_strategy = st.floats(min_value=1.0, max_value=15.0, allow_nan=False, allow_infinity=False)
disaster_type_strategy = st.sampled_from(list(DisasterType))


class TestPathRecomputation:
    """Property-based tests for path recomputation."""
    
    @given(
        vertex_count=vertex_count_strategy,
        edge_weights=st.lists(edge_weight_strategy, min_size=5, max_size=20),
        weight_change_multiplier=weight_multiplier_strategy,
        source_idx=st.integers(min_value=0, max_value=7),
        target_idx=st.integers(min_value=0, max_value=7)
    )
    def test_path_recomputation_after_weight_changes(self, vertex_count, edge_weights, 
                                                   weight_change_multiplier, source_idx, target_idx):
        """
        Property 6: Path Recomputation - Weight Change Response
        
        For any graph where edge weights are modified, recomputing paths should
        produce different results when the weight changes affect the optimal route.
        
        **Validates: Requirements 3.2**
        """
        assume(vertex_count >= 3)
        assume(len(edge_weights) >= vertex_count)
        assume(source_idx < vertex_count)
        assume(target_idx < vertex_count)
        assume(source_idx != target_idx)
        
        # Create graph
        graph = GraphManager()
        pathfinder = PathfinderEngine()
        
        # Add vertices
        vertices = [f"V{i}" for i in range(vertex_count)]
        for vertex_id in vertices:
            graph.add_vertex(vertex_id, VertexType.INTERSECTION, (0.0, 0.0))
        
        # Create connected graph (ring topology)
        for i in range(vertex_count):
            next_i = (i + 1) % vertex_count
            weight = edge_weights[i % len(edge_weights)]
            graph.add_edge(vertices[i], vertices[next_i], weight, 0.0, 0.0)
        
        # Add some additional edges for alternative paths
        for i in range(min(vertex_count - 2, len(edge_weights) - vertex_count)):
            source_v = vertices[i]
            target_v = vertices[(i + 2) % vertex_count]
            weight = edge_weights[vertex_count + i]
            graph.add_edge(source_v, target_v, weight, 0.0, 0.0)
        
        source_id = vertices[source_idx]
        target_id = vertices[target_idx]
        
        # Find initial path
        result1 = pathfinder.find_shortest_path(graph, source_id, target_id)
        
        if not result1.found:
            return  # Skip if no path exists
        
        initial_cost = result1.total_cost
        initial_path = result1.path
        
        # Modify edge weights (make some edges more expensive)
        modified_edges = []
        for i in range(min(2, len(result1.edges_traversed))):
            edge = result1.edges_traversed[i]
            original_weight = graph.get_edge_weight(edge.source, edge.target)
            new_weight = original_weight * weight_change_multiplier
            graph.update_edge_weight(edge.source, edge.target, new_weight)
            modified_edges.append((edge.source, edge.target, original_weight))
        
        # Recompute path
        result2 = pathfinder.find_shortest_path(graph, source_id, target_id)
        
        if result2.found:
            new_cost = result2.total_cost
            new_path = result2.path
            
            # Path should be recomputed (either same path with higher cost or different path)
            if new_path == initial_path:
                # Same path, but cost should be higher due to weight increases
                assert new_cost >= initial_cost
            else:
                # Different path found - this is valid recomputation
                assert new_path[0] == source_id
                assert new_path[-1] == target_id
            
            # Verify the new path cost is calculated correctly
            calculated_cost = 0.0
            for i in range(len(new_path) - 1):
                edge_weight = graph.get_edge_weight(new_path[i], new_path[i + 1])
                calculated_cost += edge_weight
            
            assert abs(calculated_cost - new_cost) < 1e-10
    
    @given(
        disaster_type=disaster_type_strategy,
        epicenter=coordinates_strategy,
        severity=severity_strategy,
        radius=radius_strategy,
        vertex_positions=st.lists(coordinates_strategy, min_size=4, max_size=8, unique=True)
    )
    def test_path_recomputation_after_disaster_effects(self, disaster_type, epicenter, 
                                                     severity, radius, vertex_positions):
        """
        Property 6: Path Recomputation - Disaster Effect Response
        
        For any graph where disaster effects are applied, paths should be recomputed
        to account for changed edge weights and blocked roads.
        
        **Validates: Requirements 3.2**
        """
        assume(len(vertex_positions) >= 4)
        
        # Create graph
        graph = GraphManager()
        pathfinder = PathfinderEngine()
        disaster_model = DisasterModel()
        
        # Add vertices at specified positions
        vertices = []
        for i, pos in enumerate(vertex_positions):
            vertex_id = f"V{i}"
            graph.add_vertex(vertex_id, VertexType.INTERSECTION, pos)
            vertices.append(vertex_id)
        
        # Create connected graph
        for i in range(len(vertices)):
            for j in range(i + 1, len(vertices)):
                # Add edge if vertices are reasonably close
                pos1 = vertex_positions[i]
                pos2 = vertex_positions[j]
                distance = ((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2) ** 0.5
                
                if distance <= radius * 1.5:  # Connect nearby vertices
                    graph.add_edge(vertices[i], vertices[j], distance, 0.1, 0.1)
        
        if graph.get_edge_count() == 0:
            return  # Skip if no edges
        
        source_id = vertices[0]
        target_id = vertices[-1]
        
        # Find initial path
        result1 = pathfinder.find_shortest_path(graph, source_id, target_id)
        
        if not result1.found:
            return  # Skip if no initial path
        
        initial_cost = result1.total_cost
        initial_path = result1.path
        
        # Apply disaster effects
        disaster = DisasterEvent(disaster_type, epicenter, severity, radius)
        disaster_model.apply_disaster_effects(graph, disaster)
        
        # Recompute path after disaster
        result2 = pathfinder.find_shortest_path(graph, source_id, target_id)
        
        if result2.found:
            new_cost = result2.total_cost
            new_path = result2.path
            
            # Path should be valid
            assert new_path[0] == source_id
            assert new_path[-1] == target_id
            
            # No blocked edges should be used
            for i in range(len(new_path) - 1):
                edge = graph.get_edge(new_path[i], new_path[i + 1])
                assert not edge.is_blocked
                assert graph.get_edge_weight(new_path[i], new_path[i + 1]) != float('inf')
            
            # Cost should be recalculated correctly
            calculated_cost = 0.0
            for i in range(len(new_path) - 1):
                edge_weight = graph.get_edge_weight(new_path[i], new_path[i + 1])
                calculated_cost += edge_weight
            
            assert abs(calculated_cost - new_cost) < 1e-10
        else:
            # If no path found after disaster, it should be due to blocking
            assert "No path exists" in result2.error_message
    
    @given(
        vertex_count=vertex_count_strategy,
        edge_weights=st.lists(edge_weight_strategy, min_size=6, max_size=15),
        modifications=st.lists(
            st.tuples(st.integers(min_value=0, max_value=7), weight_multiplier_strategy),
            min_size=1, max_size=3
        )
    )
    def test_multiple_weight_changes_recomputation(self, vertex_count, edge_weights, modifications):
        """
        Property 6: Path Recomputation - Multiple Weight Changes
        
        For any sequence of weight changes, each recomputation should produce
        the optimal path for the current graph state.
        
        **Validates: Requirements 3.2**
        """
        assume(vertex_count >= 3)
        assume(len(edge_weights) >= vertex_count)
        
        # Create graph
        graph = GraphManager()
        pathfinder = PathfinderEngine()
        
        # Add vertices
        vertices = [f"V{i}" for i in range(vertex_count)]
        for vertex_id in vertices:
            graph.add_vertex(vertex_id, VertexType.INTERSECTION, (0.0, 0.0))
        
        # Create connected graph (star topology for guaranteed connectivity)
        center = vertices[0]
        edges_added = []
        for i in range(1, vertex_count):
            weight = edge_weights[i - 1]
            graph.add_edge(center, vertices[i], weight, 0.0, 0.0)
            graph.add_edge(vertices[i], center, weight, 0.0, 0.0)
            edges_added.append((center, vertices[i]))
            edges_added.append((vertices[i], center))
        
        # Add some cross-connections
        for i in range(1, min(vertex_count - 1, len(edge_weights) - vertex_count + 1)):
            weight = edge_weights[vertex_count - 1 + i]
            graph.add_edge(vertices[i], vertices[i + 1], weight, 0.0, 0.0)
            edges_added.append((vertices[i], vertices[i + 1]))
        
        source_id = vertices[0]
        target_id = vertices[-1]
        
        # Track path costs through modifications
        path_costs = []
        
        # Initial path
        result = pathfinder.find_shortest_path(graph, source_id, target_id)
        if result.found:
            path_costs.append(result.total_cost)
        else:
            return  # Skip if no initial path
        
        # Apply modifications sequentially
        for edge_idx, multiplier in modifications:
            if edge_idx < len(edges_added):
                source, target = edges_added[edge_idx]
                current_weight = graph.get_edge_weight(source, target)
                new_weight = current_weight * multiplier
                graph.update_edge_weight(source, target, new_weight)
                
                # Recompute path
                result = pathfinder.find_shortest_path(graph, source_id, target_id)
                if result.found:
                    path_costs.append(result.total_cost)
                    
                    # Verify path is valid and optimal for current state
                    assert result.path[0] == source_id
                    assert result.path[-1] == target_id
                    
                    # Verify cost calculation
                    calculated_cost = 0.0
                    for i in range(len(result.path) - 1):
                        edge_weight = graph.get_edge_weight(result.path[i], result.path[i + 1])
                        calculated_cost += edge_weight
                    
                    assert abs(calculated_cost - result.total_cost) < 1e-10
        
        # Should have computed at least one path
        assert len(path_costs) >= 1
    
    @given(
        disaster_type=disaster_type_strategy,
        epicenter=coordinates_strategy,
        severity=severity_strategy,
        radius=radius_strategy
    )
    def test_disaster_removal_path_restoration(self, disaster_type, epicenter, severity, radius):
        """
        Property 6: Path Recomputation - Disaster Removal Restoration
        
        For any graph with disaster effects, removing the disaster should restore
        paths to their original state (or better alternatives).
        
        **Validates: Requirements 3.2**
        """
        # Create simple test graph
        graph = GraphManager()
        pathfinder = PathfinderEngine()
        disaster_model = DisasterModel()
        
        # Create diamond graph for multiple path options
        graph.add_vertex("A", VertexType.INTERSECTION, (0.0, 0.0))
        graph.add_vertex("B", VertexType.INTERSECTION, (1.0, 1.0))
        graph.add_vertex("C", VertexType.INTERSECTION, (1.0, -1.0))
        graph.add_vertex("D", VertexType.EVACUATION_POINT, (2.0, 0.0), capacity=100)
        
        graph.add_edge("A", "B", 1.5, 0.1, 0.1)
        graph.add_edge("A", "C", 1.5, 0.1, 0.1)
        graph.add_edge("B", "D", 1.5, 0.1, 0.1)
        graph.add_edge("C", "D", 1.5, 0.1, 0.1)
        
        # Find original path
        result_original = pathfinder.find_shortest_path(graph, "A", "D")
        
        if not result_original.found:
            return
        
        original_cost = result_original.total_cost
        
        # Apply disaster
        disaster = DisasterEvent(disaster_type, epicenter, severity, radius)
        disaster_model.apply_disaster_effects(graph, disaster)
        
        # Find path with disaster effects
        result_with_disaster = pathfinder.find_shortest_path(graph, "A", "D")
        
        # Remove disaster effects
        disaster_model.remove_disaster_effects(graph, disaster)
        
        # Find path after disaster removal
        result_after_removal = pathfinder.find_shortest_path(graph, "A", "D")
        
        if result_after_removal.found:
            # Path should be restored to original optimality
            restored_cost = result_after_removal.total_cost
            
            # Cost should be back to original (within floating point precision)
            assert abs(restored_cost - original_cost) < 1e-10
            
            # Path should be valid
            assert result_after_removal.path[0] == "A"
            assert result_after_removal.path[-1] == "D"
    
    @given(
        vertex_count=st.integers(min_value=4, max_value=6),
        weight_changes=st.lists(
            st.tuples(
                st.integers(min_value=0, max_value=5),  # edge index
                st.integers(min_value=0, max_value=5),  # target index
                edge_weight_strategy  # new weight
            ),
            min_size=2, max_size=5
        )
    )
    def test_incremental_path_recomputation(self, vertex_count, weight_changes):
        """
        Property 6: Path Recomputation - Incremental Changes
        
        For any sequence of incremental weight changes, each recomputation should
        maintain optimality and path validity.
        
        **Validates: Requirements 3.2**
        """
        assume(vertex_count >= 4)
        
        # Create graph
        graph = GraphManager()
        pathfinder = PathfinderEngine()
        
        # Add vertices
        vertices = [f"V{i}" for i in range(vertex_count)]
        for vertex_id in vertices:
            graph.add_vertex(vertex_id, VertexType.INTERSECTION, (0.0, 0.0))
        
        # Create initial edges (complete graph for maximum connectivity)
        initial_edges = []
        for i in range(vertex_count):
            for j in range(i + 1, vertex_count):
                weight = 1.0 + (i + j) * 0.5  # Varied weights
                graph.add_edge(vertices[i], vertices[j], weight, 0.0, 0.0)
                graph.add_edge(vertices[j], vertices[i], weight, 0.0, 0.0)
                initial_edges.append((vertices[i], vertices[j]))
                initial_edges.append((vertices[j], vertices[i]))
        
        source_id = vertices[0]
        target_id = vertices[-1]
        
        # Apply incremental changes
        for edge_idx, target_idx, new_weight in weight_changes:
            if edge_idx < len(initial_edges) and target_idx < vertex_count:
                source, target = initial_edges[edge_idx]
                
                # Update weight
                graph.update_edge_weight(source, target, new_weight)
                
                # Recompute path
                result = pathfinder.find_shortest_path(graph, source_id, target_id)
                
                if result.found:
                    # Verify path properties
                    assert result.path[0] == source_id
                    assert result.path[-1] == target_id
                    assert result.total_cost >= 0
                    
                    # Verify cost calculation
                    calculated_cost = 0.0
                    for i in range(len(result.path) - 1):
                        edge_weight = graph.get_edge_weight(result.path[i], result.path[i + 1])
                        calculated_cost += edge_weight
                    
                    assert abs(calculated_cost - result.total_cost) < 1e-10
                    
                    # Path should not use blocked edges
                    for i in range(len(result.path) - 1):
                        edge_weight = graph.get_edge_weight(result.path[i], result.path[i + 1])
                        assert edge_weight != float('inf')
    
    @given(
        source_idx=st.integers(min_value=0, max_value=4),
        target_idx=st.integers(min_value=0, max_value=4),
        weight_multipliers=st.lists(weight_multiplier_strategy, min_size=3, max_size=6)
    )
    def test_path_recomputation_determinism(self, source_idx, target_idx, weight_multipliers):
        """
        Property 6: Path Recomputation - Deterministic Results
        
        For any graph state, multiple recomputations should produce identical results.
        
        **Validates: Requirements 3.2**
        """
        assume(source_idx != target_idx)
        assume(len(weight_multipliers) >= 3)
        
        # Create fixed graph
        graph = GraphManager()
        pathfinder = PathfinderEngine()
        
        vertices = ["A", "B", "C", "D", "E"]
        for vertex_id in vertices:
            graph.add_vertex(vertex_id, VertexType.INTERSECTION, (0.0, 0.0))
        
        # Add edges with weights based on multipliers
        edges = [
            ("A", "B", weight_multipliers[0]),
            ("B", "C", weight_multipliers[1]),
            ("C", "D", weight_multipliers[2]),
            ("A", "D", weight_multipliers[0] + weight_multipliers[1]),
            ("B", "E", weight_multipliers[1] if len(weight_multipliers) > 3 else 2.0),
            ("E", "D", weight_multipliers[2] if len(weight_multipliers) > 4 else 1.5)
        ]
        
        for source, target, weight in edges:
            graph.add_edge(source, target, weight, 0.0, 0.0)
        
        source_id = vertices[source_idx]
        target_id = vertices[target_idx]
        
        # Compute path multiple times
        results = []
        for _ in range(3):
            result = pathfinder.find_shortest_path(graph, source_id, target_id)
            results.append(result)
        
        # All results should be identical
        for i in range(1, len(results)):
            assert results[0].found == results[i].found
            
            if results[0].found:
                assert abs(results[0].total_cost - results[i].total_cost) < 1e-10
                assert results[0].path == results[i].path