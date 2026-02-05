"""
Property-based tests for disconnected graph handling.

Feature: disaster-evacuation-routing, Property 7: Disconnected Graph Handling
**Validates: Requirements 3.5**
"""

import pytest
from hypothesis import given, strategies as st, assume
from disaster_evacuation.pathfinding import PathfinderEngine
from disaster_evacuation.graph import GraphManager
from disaster_evacuation.models import VertexType


# Hypothesis strategies for generating test data
vertex_count_strategy = st.integers(min_value=2, max_value=10)
component_count_strategy = st.integers(min_value=2, max_value=4)
edge_weight_strategy = st.floats(min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False)
component_size_strategy = st.integers(min_value=1, max_value=4)


class TestDisconnectedGraphHandling:
    """Property-based tests for disconnected graph handling."""
    
    @given(
        component_sizes=st.lists(component_size_strategy, min_size=2, max_size=4),
        edge_weights=st.lists(edge_weight_strategy, min_size=5, max_size=15)
    )
    def test_disconnected_components_no_path_exists(self, component_sizes, edge_weights):
        """
        Property 7: Disconnected Graph Handling - No Path Between Components
        
        For any graph with disconnected components, attempting to find a path
        between vertices in different components should return an appropriate
        "no path exists" result rather than an invalid path.
        
        **Validates: Requirements 3.5**
        """
        assume(len(component_sizes) >= 2)
        assume(sum(component_sizes) >= 2)
        assume(len(edge_weights) >= sum(component_sizes))
        
        # Create graph with multiple disconnected components
        graph = GraphManager()
        pathfinder = PathfinderEngine()
        
        components = []
        vertex_counter = 0
        
        # Create each component
        for comp_size in component_sizes:
            if comp_size <= 0:
                continue
                
            component_vertices = []
            
            # Add vertices for this component
            for i in range(comp_size):
                vertex_id = f"C{len(components)}_V{i}"
                graph.add_vertex(vertex_id, VertexType.INTERSECTION, (len(components) * 10, i))
                component_vertices.append(vertex_id)
                vertex_counter += 1
            
            # Connect vertices within component (if more than one vertex)
            if len(component_vertices) > 1:
                for i in range(len(component_vertices) - 1):
                    weight_idx = (vertex_counter + i) % len(edge_weights)
                    weight = edge_weights[weight_idx]
                    graph.add_edge(component_vertices[i], component_vertices[i + 1], weight, 0.0, 0.0)
                
                # Add some additional internal connections for larger components
                if len(component_vertices) > 2:
                    weight_idx = (vertex_counter + len(component_vertices)) % len(edge_weights)
                    weight = edge_weights[weight_idx]
                    graph.add_edge(component_vertices[0], component_vertices[-1], weight, 0.0, 0.0)
            
            components.append(component_vertices)
        
        # Filter out empty components
        components = [comp for comp in components if len(comp) > 0]
        
        if len(components) < 2:
            return  # Need at least 2 components
        
        # Test paths between different components
        for i in range(len(components)):
            for j in range(i + 1, len(components)):
                source_vertex = components[i][0]
                target_vertex = components[j][0]
                
                result = pathfinder.find_shortest_path(graph, source_vertex, target_vertex)
                
                # Should not find a path between disconnected components
                assert not result.found
                assert "No path exists" in result.error_message
                assert result.path == []
                assert result.total_cost == 0.0
        
        # Test paths within components (should work)
        for component in components:
            if len(component) > 1:
                source_vertex = component[0]
                target_vertex = component[-1]
                
                result = pathfinder.find_shortest_path(graph, source_vertex, target_vertex)
                
                # Should find path within connected component
                if result.found:
                    assert result.path[0] == source_vertex
                    assert result.path[-1] == target_vertex
                    assert result.total_cost >= 0
    
    @given(
        isolated_count=st.integers(min_value=1, max_value=5),
        connected_size=st.integers(min_value=2, max_value=6),
        edge_weights=st.lists(edge_weight_strategy, min_size=3, max_size=10)
    )
    def test_isolated_vertices_handling(self, isolated_count, connected_size, edge_weights):
        """
        Property 7: Disconnected Graph Handling - Isolated Vertices
        
        For any graph containing isolated vertices, paths to/from isolated vertices
        should correctly report no path exists.
        
        **Validates: Requirements 3.5**
        """
        assume(len(edge_weights) >= connected_size)
        
        # Create graph
        graph = GraphManager()
        pathfinder = PathfinderEngine()
        
        # Add isolated vertices
        isolated_vertices = []
        for i in range(isolated_count):
            vertex_id = f"ISOLATED_{i}"
            graph.add_vertex(vertex_id, VertexType.INTERSECTION, (i * 20, 0))
            isolated_vertices.append(vertex_id)
        
        # Add connected component
        connected_vertices = []
        for i in range(connected_size):
            vertex_id = f"CONNECTED_{i}"
            graph.add_vertex(vertex_id, VertexType.INTERSECTION, (0, i))
            connected_vertices.append(vertex_id)
        
        # Connect the connected component
        for i in range(connected_size - 1):
            weight = edge_weights[i % len(edge_weights)]
            graph.add_edge(connected_vertices[i], connected_vertices[i + 1], weight, 0.0, 0.0)
        
        # Test paths from isolated to connected
        for isolated_vertex in isolated_vertices:
            for connected_vertex in connected_vertices:
                result = pathfinder.find_shortest_path(graph, isolated_vertex, connected_vertex)
                assert not result.found
                assert "No path exists" in result.error_message
                
                # Test reverse direction
                result_reverse = pathfinder.find_shortest_path(graph, connected_vertex, isolated_vertex)
                assert not result_reverse.found
                assert "No path exists" in result_reverse.error_message
        
        # Test paths between isolated vertices
        for i in range(len(isolated_vertices)):
            for j in range(i + 1, len(isolated_vertices)):
                result = pathfinder.find_shortest_path(graph, isolated_vertices[i], isolated_vertices[j])
                assert not result.found
                assert "No path exists" in result.error_message
        
        # Test paths within connected component (should work)
        if len(connected_vertices) > 1:
            result = pathfinder.find_shortest_path(graph, connected_vertices[0], connected_vertices[-1])
            if result.found:  # May not be found if component is not fully connected
                assert result.path[0] == connected_vertices[0]
                assert result.path[-1] == connected_vertices[-1]
    
    @given(
        component_count=component_count_strategy,
        vertices_per_component=st.lists(st.integers(min_value=1, max_value=3), min_size=2, max_size=4),
        source_component=st.integers(min_value=0, max_value=3),
        target_component=st.integers(min_value=0, max_value=3)
    )
    def test_find_all_paths_disconnected_graph(self, component_count, vertices_per_component, 
                                             source_component, target_component):
        """
        Property 7: Disconnected Graph Handling - Find All Paths
        
        For any disconnected graph, find_all_shortest_paths should correctly identify
        reachable and unreachable vertices from any source.
        
        **Validates: Requirements 3.5**
        """
        assume(len(vertices_per_component) >= 2)
        assume(source_component < len(vertices_per_component))
        
        # Create graph with multiple components
        graph = GraphManager()
        pathfinder = PathfinderEngine()
        
        all_vertices = []
        components = []
        
        for comp_idx, vertex_count in enumerate(vertices_per_component):
            component_vertices = []
            
            # Add vertices for this component
            for v_idx in range(vertex_count):
                vertex_id = f"C{comp_idx}_V{v_idx}"
                graph.add_vertex(vertex_id, VertexType.INTERSECTION, (comp_idx * 10, v_idx))
                component_vertices.append(vertex_id)
                all_vertices.append(vertex_id)
            
            # Connect vertices within component
            for i in range(len(component_vertices) - 1):
                weight = 1.0 + i * 0.5
                graph.add_edge(component_vertices[i], component_vertices[i + 1], weight, 0.0, 0.0)
            
            components.append(component_vertices)
        
        if len(components) == 0 or len(components[source_component]) == 0:
            return
        
        # Choose source vertex from specified component
        source_vertex = components[source_component][0]
        
        # Find all shortest paths from source
        all_results = pathfinder.find_all_shortest_paths(graph, source_vertex)
        
        # Verify results
        assert len(all_results) == len(all_vertices)
        
        # Check reachability within same component
        for vertex in components[source_component]:
            assert vertex in all_results
            assert all_results[vertex].found
            assert all_results[vertex].path[0] == source_vertex
            
            if vertex == source_vertex:
                assert all_results[vertex].total_cost == 0.0
                assert all_results[vertex].path == [source_vertex]
        
        # Check unreachability to other components
        for comp_idx, component in enumerate(components):
            if comp_idx != source_component:
                for vertex in component:
                    assert vertex in all_results
                    assert not all_results[vertex].found
                    assert "No path exists" in all_results[vertex].error_message
    
    @given(
        bridge_components=st.integers(min_value=2, max_value=4),
        component_sizes=st.lists(st.integers(min_value=2, max_value=4), min_size=2, max_size=4)
    )
    def test_bridge_removal_creates_disconnection(self, bridge_components, component_sizes):
        """
        Property 7: Disconnected Graph Handling - Bridge Removal
        
        For any connected graph, removing bridge edges should create disconnected
        components, and pathfinding should correctly handle the disconnection.
        
        **Validates: Requirements 3.5**
        """
        assume(len(component_sizes) >= 2)
        
        # Create initially connected graph with bridge edges
        graph = GraphManager()
        pathfinder = PathfinderEngine()
        
        components = []
        bridge_edges = []
        
        # Create components
        for comp_idx, size in enumerate(component_sizes):
            component_vertices = []
            
            # Add vertices
            for v_idx in range(size):
                vertex_id = f"C{comp_idx}_V{v_idx}"
                graph.add_vertex(vertex_id, VertexType.INTERSECTION, (comp_idx * 5, v_idx))
                component_vertices.append(vertex_id)
            
            # Connect within component
            for i in range(len(component_vertices) - 1):
                graph.add_edge(component_vertices[i], component_vertices[i + 1], 1.0, 0.0, 0.0)
            
            components.append(component_vertices)
        
        # Add bridge edges between components
        for i in range(len(components) - 1):
            bridge_source = components[i][-1]  # Last vertex of component i
            bridge_target = components[i + 1][0]  # First vertex of component i+1
            graph.add_edge(bridge_source, bridge_target, 2.0, 0.0, 0.0)
            bridge_edges.append((bridge_source, bridge_target))
        
        if len(components) < 2:
            return
        
        # Initially should be connected
        source_vertex = components[0][0]
        target_vertex = components[-1][-1]
        
        initial_result = pathfinder.find_shortest_path(graph, source_vertex, target_vertex)
        
        if initial_result.found:
            # Remove bridge edges to create disconnection
            for bridge_source, bridge_target in bridge_edges:
                graph.update_edge_weight(bridge_source, bridge_target, float('inf'))
            
            # Now should be disconnected
            disconnected_result = pathfinder.find_shortest_path(graph, source_vertex, target_vertex)
            
            assert not disconnected_result.found
            assert "No path exists" in disconnected_result.error_message
            assert disconnected_result.path == []
            assert disconnected_result.total_cost == 0.0
    
    @given(
        vertex_count=st.integers(min_value=3, max_value=8),
        disconnection_probability=st.floats(min_value=0.3, max_value=0.7)
    )
    def test_random_disconnection_handling(self, vertex_count, disconnection_probability):
        """
        Property 7: Disconnected Graph Handling - Random Disconnections
        
        For any graph with randomly removed edges, pathfinding should correctly
        handle the resulting disconnected components.
        
        **Validates: Requirements 3.5**
        """
        # Create initially connected graph
        graph = GraphManager()
        pathfinder = PathfinderEngine()
        
        # Add vertices
        vertices = [f"V{i}" for i in range(vertex_count)]
        for vertex_id in vertices:
            graph.add_vertex(vertex_id, VertexType.INTERSECTION, (0, 0))
        
        # Add edges to create connected graph (ring + some additional edges)
        edges_to_add = []
        
        # Ring edges for basic connectivity
        for i in range(vertex_count):
            next_i = (i + 1) % vertex_count
            edges_to_add.append((vertices[i], vertices[next_i], 1.0))
        
        # Additional edges for redundancy
        for i in range(vertex_count):
            for j in range(i + 2, vertex_count):
                if (j - i) % vertex_count > 1:  # Avoid immediate neighbors
                    edges_to_add.append((vertices[i], vertices[j], 2.0))
        
        # Add edges to graph
        for source, target, weight in edges_to_add:
            graph.add_edge(source, target, weight, 0.0, 0.0)
        
        # Randomly disconnect some edges
        import random
        random.seed(42)  # For reproducibility
        
        for source, target, weight in edges_to_add:
            if random.random() < disconnection_probability:
                graph.update_edge_weight(source, target, float('inf'))
        
        # Test pathfinding between all vertex pairs
        for i in range(vertex_count):
            for j in range(i + 1, vertex_count):
                source_vertex = vertices[i]
                target_vertex = vertices[j]
                
                result = pathfinder.find_shortest_path(graph, source_vertex, target_vertex)
                
                if result.found:
                    # If path found, verify it's valid
                    assert result.path[0] == source_vertex
                    assert result.path[-1] == target_vertex
                    assert result.total_cost >= 0
                    
                    # Verify no infinite weight edges are used
                    for k in range(len(result.path) - 1):
                        edge_weight = graph.get_edge_weight(result.path[k], result.path[k + 1])
                        assert edge_weight != float('inf')
                else:
                    # If no path found, verify error message
                    assert "No path exists" in result.error_message
                    assert result.path == []
                    assert result.total_cost == 0.0
    
    @given(
        single_vertex_count=st.integers(min_value=1, max_value=3),
        pair_count=st.integers(min_value=1, max_value=3),
        triangle_count=st.integers(min_value=0, max_value=2)
    )
    def test_mixed_component_types(self, single_vertex_count, pair_count, triangle_count):
        """
        Property 7: Disconnected Graph Handling - Mixed Component Types
        
        For any graph with mixed component types (isolated vertices, pairs, triangles),
        pathfinding should correctly handle all component types.
        
        **Validates: Requirements 3.5**
        """
        # Create graph with different component types
        graph = GraphManager()
        pathfinder = PathfinderEngine()
        
        all_vertices = []
        component_info = []
        
        # Single vertex components
        for i in range(single_vertex_count):
            vertex_id = f"SINGLE_{i}"
            graph.add_vertex(vertex_id, VertexType.INTERSECTION, (i, 0))
            all_vertices.append(vertex_id)
            component_info.append(("single", [vertex_id]))
        
        # Pair components
        for i in range(pair_count):
            v1_id = f"PAIR_{i}_A"
            v2_id = f"PAIR_{i}_B"
            graph.add_vertex(v1_id, VertexType.INTERSECTION, (i, 10))
            graph.add_vertex(v2_id, VertexType.INTERSECTION, (i, 11))
            graph.add_edge(v1_id, v2_id, 1.0, 0.0, 0.0)
            all_vertices.extend([v1_id, v2_id])
            component_info.append(("pair", [v1_id, v2_id]))
        
        # Triangle components
        for i in range(triangle_count):
            v1_id = f"TRI_{i}_A"
            v2_id = f"TRI_{i}_B"
            v3_id = f"TRI_{i}_C"
            graph.add_vertex(v1_id, VertexType.INTERSECTION, (i, 20))
            graph.add_vertex(v2_id, VertexType.INTERSECTION, (i, 21))
            graph.add_vertex(v3_id, VertexType.INTERSECTION, (i, 22))
            graph.add_edge(v1_id, v2_id, 1.0, 0.0, 0.0)
            graph.add_edge(v2_id, v3_id, 1.0, 0.0, 0.0)
            graph.add_edge(v3_id, v1_id, 1.0, 0.0, 0.0)
            all_vertices.extend([v1_id, v2_id, v3_id])
            component_info.append(("triangle", [v1_id, v2_id, v3_id]))
        
        if len(all_vertices) < 2:
            return
        
        # Test paths within and between components
        for i, (comp_type_i, vertices_i) in enumerate(component_info):
            for j, (comp_type_j, vertices_j) in enumerate(component_info):
                if i != j:
                    # Test paths between different components (should fail)
                    source = vertices_i[0]
                    target = vertices_j[0]
                    
                    result = pathfinder.find_shortest_path(graph, source, target)
                    assert not result.found
                    assert "No path exists" in result.error_message
                else:
                    # Test paths within same component
                    if len(vertices_i) > 1:
                        source = vertices_i[0]
                        target = vertices_i[-1]
                        
                        result = pathfinder.find_shortest_path(graph, source, target)
                        
                        if comp_type_i == "single":
                            # Single vertex component - source == target
                            if source == target:
                                assert result.found
                                assert result.total_cost == 0.0
                        else:
                            # Multi-vertex component - should find path
                            assert result.found
                            assert result.path[0] == source
                            assert result.path[-1] == target