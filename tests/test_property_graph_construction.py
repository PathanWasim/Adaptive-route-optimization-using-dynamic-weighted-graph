"""
Property-based tests for graph construction integrity.

Feature: disaster-evacuation-routing, Property 2: Graph Construction Integrity
**Validates: Requirements 1.2, 1.3, 1.4**
"""

import pytest
from hypothesis import given, strategies as st, assume
from disaster_evacuation.models import GraphManager
from disaster_evacuation.models import VertexType


# Hypothesis strategies for generating test data
vertex_id_strategy = st.text(min_size=1, max_size=10, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd')))
coordinate_strategy = st.floats(min_value=-1000.0, max_value=1000.0, allow_nan=False, allow_infinity=False)
coordinates_strategy = st.tuples(coordinate_strategy, coordinate_strategy)
vertex_type_strategy = st.sampled_from(list(VertexType))
capacity_strategy = st.one_of(st.none(), st.integers(min_value=0, max_value=10000))
distance_strategy = st.floats(min_value=0.0, max_value=1000.0, allow_nan=False, allow_infinity=False)
risk_strategy = st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False)
congestion_strategy = st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False)


class TestGraphConstructionIntegrity:
    """Property-based tests for graph construction integrity."""
    
    @given(
        vertex_id=vertex_id_strategy,
        vertex_type=vertex_type_strategy,
        coordinates=coordinates_strategy,
        capacity=capacity_strategy
    )
    def test_vertex_creation_maintains_properties(self, vertex_id, vertex_type, coordinates, capacity):
        """
        Property 2: Graph Construction Integrity - Vertex Creation
        
        For any vertex creation with valid type and coordinates, the resulting graph
        should maintain proper structure with all vertices having assigned types.
        
        **Validates: Requirements 1.2, 1.3**
        """
        graph = GraphManager()
        
        # Add vertex to graph
        graph.add_vertex(vertex_id, vertex_type, coordinates, capacity)
        
        # Verify graph maintains integrity
        assert graph.get_vertex_count() == 1
        assert graph.has_vertex(vertex_id)
        
        # Verify vertex properties are preserved
        vertex = graph.get_vertex(vertex_id)
        assert vertex is not None
        assert vertex.id == vertex_id
        assert vertex.vertex_type == vertex_type
        assert vertex.coordinates == coordinates
        assert vertex.capacity == capacity
        
        # Verify vertex is in adjacency list
        neighbors = graph.get_neighbors(vertex_id)
        assert isinstance(neighbors, list)
        assert len(neighbors) == 0  # New vertex has no edges
        
        # Verify vertex appears in vertex collections
        all_vertices = graph.get_all_vertices()
        assert len(all_vertices) == 1
        assert all_vertices[0].id == vertex_id
        
        vertex_ids = graph.get_vertex_ids()
        assert vertex_id in vertex_ids
        assert len(vertex_ids) == 1
    
    @given(
        vertices=st.lists(
            st.tuples(vertex_id_strategy, vertex_type_strategy, coordinates_strategy, capacity_strategy),
            min_size=1,
            max_size=20,
            unique_by=lambda x: x[0]  # Unique by vertex_id
        )
    )
    def test_multiple_vertex_creation_integrity(self, vertices):
        """
        Property 2: Graph Construction Integrity - Multiple Vertices
        
        For any set of vertices with unique IDs, the graph should maintain
        proper structure and all vertices should be accessible.
        
        **Validates: Requirements 1.2, 1.3**
        """
        graph = GraphManager()
        
        # Add all vertices
        for vertex_id, vertex_type, coordinates, capacity in vertices:
            graph.add_vertex(vertex_id, vertex_type, coordinates, capacity)
        
        # Verify graph integrity
        assert graph.get_vertex_count() == len(vertices)
        
        # Verify all vertices are accessible
        for vertex_id, vertex_type, coordinates, capacity in vertices:
            assert graph.has_vertex(vertex_id)
            vertex = graph.get_vertex(vertex_id)
            assert vertex is not None
            assert vertex.id == vertex_id
            assert vertex.vertex_type == vertex_type
            assert vertex.coordinates == coordinates
            assert vertex.capacity == capacity
        
        # Verify collections are consistent
        all_vertices = graph.get_all_vertices()
        assert len(all_vertices) == len(vertices)
        
        vertex_ids = graph.get_vertex_ids()
        assert len(vertex_ids) == len(vertices)
        
        expected_ids = {v[0] for v in vertices}
        assert vertex_ids == expected_ids
    
    @given(
        source_id=vertex_id_strategy,
        target_id=vertex_id_strategy,
        source_coords=coordinates_strategy,
        target_coords=coordinates_strategy,
        distance=distance_strategy,
        risk=risk_strategy,
        congestion=congestion_strategy
    )
    def test_edge_creation_maintains_properties(self, source_id, target_id, source_coords, 
                                              target_coords, distance, risk, congestion):
        """
        Property 2: Graph Construction Integrity - Edge Creation
        
        For any edge creation with valid attributes, the resulting graph should
        maintain proper structure with all edges containing distance, risk, and congestion factors.
        
        **Validates: Requirements 1.3, 1.4**
        """
        assume(source_id != target_id)  # No self-loops
        
        graph = GraphManager()
        
        # Add vertices first
        graph.add_vertex(source_id, VertexType.INTERSECTION, source_coords)
        graph.add_vertex(target_id, VertexType.INTERSECTION, target_coords)
        
        # Add edge
        graph.add_edge(source_id, target_id, distance, risk, congestion)
        
        # Verify graph integrity
        assert graph.get_edge_count() == 1
        assert graph.is_connected(source_id, target_id)
        
        # Verify edge properties are preserved
        edge = graph.get_edge(source_id, target_id)
        assert edge is not None
        assert edge.source == source_id
        assert edge.target == target_id
        assert edge.base_distance == distance
        assert edge.base_risk == risk
        assert edge.base_congestion == congestion
        
        # Verify weight calculation
        expected_weight = distance + risk + congestion
        assert abs(edge.current_weight - expected_weight) < 1e-10
        assert abs(graph.get_edge_weight(source_id, target_id) - expected_weight) < 1e-10
        
        # Verify edge appears in adjacency list
        neighbors = graph.get_neighbors(source_id)
        assert len(neighbors) == 1
        assert neighbors[0].target == target_id
        
        # Verify edge appears in collections
        all_edges = graph.get_all_edges()
        assert len(all_edges) == 1
        assert all_edges[0].source == source_id
        assert all_edges[0].target == target_id
    
    @given(
        graph_data=st.tuples(
            # Vertices: list of (id, type, coords, capacity)
            st.lists(
                st.tuples(vertex_id_strategy, vertex_type_strategy, coordinates_strategy, capacity_strategy),
                min_size=2,
                max_size=10,
                unique_by=lambda x: x[0]
            ),
            # Edges: list of (source_idx, target_idx, distance, risk, congestion)
            st.data()
        )
    )
    def test_complete_graph_construction_integrity(self, graph_data):
        """
        Property 2: Graph Construction Integrity - Complete Graph
        
        For any valid graph construction with vertices and edges, the resulting graph
        should maintain complete structural integrity and all properties should be preserved.
        
        **Validates: Requirements 1.2, 1.3, 1.4**
        """
        vertices, data = graph_data
        
        # Generate edges between existing vertices
        edges = []
        if len(vertices) >= 2:
            for _ in range(min(20, len(vertices) * 2)):  # Limit edge count
                source_idx = data.draw(st.integers(min_value=0, max_value=len(vertices) - 1))
                target_idx = data.draw(st.integers(min_value=0, max_value=len(vertices) - 1))
                
                if source_idx != target_idx:  # No self-loops
                    distance = data.draw(distance_strategy)
                    risk = data.draw(risk_strategy)
                    congestion = data.draw(congestion_strategy)
                    
                    source_id = vertices[source_idx][0]
                    target_id = vertices[target_idx][0]
                    
                    # Avoid duplicate edges
                    edge_key = (source_id, target_id)
                    if edge_key not in [e[:2] for e in edges]:
                        edges.append((source_id, target_id, distance, risk, congestion))
        
        graph = GraphManager()
        
        # Add all vertices
        for vertex_id, vertex_type, coordinates, capacity in vertices:
            graph.add_vertex(vertex_id, vertex_type, coordinates, capacity)
        
        # Add all edges
        for source_id, target_id, distance, risk, congestion in edges:
            graph.add_edge(source_id, target_id, distance, risk, congestion)
        
        # Verify overall graph integrity
        assert graph.get_vertex_count() == len(vertices)
        assert graph.get_edge_count() == len(edges)
        
        # Verify all vertices maintain their properties
        for vertex_id, vertex_type, coordinates, capacity in vertices:
            vertex = graph.get_vertex(vertex_id)
            assert vertex is not None
            assert vertex.id == vertex_id
            assert vertex.vertex_type == vertex_type
            assert vertex.coordinates == coordinates
            assert vertex.capacity == capacity
        
        # Verify all edges maintain their properties
        for source_id, target_id, distance, risk, congestion in edges:
            assert graph.is_connected(source_id, target_id)
            edge = graph.get_edge(source_id, target_id)
            assert edge is not None
            assert edge.source == source_id
            assert edge.target == target_id
            assert edge.base_distance == distance
            assert edge.base_risk == risk
            assert edge.base_congestion == congestion
            
            # Verify weight calculation integrity
            expected_weight = distance + risk + congestion
            assert abs(edge.current_weight - expected_weight) < 1e-10
        
        # Verify graph statistics are consistent
        info = graph.get_graph_info()
        assert info["vertex_count"] == len(vertices)
        assert info["edge_count"] == len(edges)
        
        # Verify collections are consistent
        assert len(graph.get_all_vertices()) == len(vertices)
        assert len(graph.get_all_edges()) == len(edges)
        assert len(graph.get_vertex_ids()) == len(vertices)
    
    @given(
        vertex_id=vertex_id_strategy,
        coordinates=coordinates_strategy
    )
    def test_vertex_type_assignment_integrity(self, vertex_id, coordinates):
        """
        Property 2: Graph Construction Integrity - Vertex Type Assignment
        
        For any vertex creation, the vertex type should be properly assigned and
        maintained throughout the graph's lifetime.
        
        **Validates: Requirements 1.2**
        """
        graph = GraphManager()
        
        # Test each vertex type
        for vertex_type in VertexType:
            test_id = f"{vertex_id}_{vertex_type.value}"
            capacity = 100 if vertex_type in [VertexType.SHELTER, VertexType.EVACUATION_POINT] else None
            
            graph.add_vertex(test_id, vertex_type, coordinates, capacity)
            
            vertex = graph.get_vertex(test_id)
            assert vertex.vertex_type == vertex_type
            
            # Verify type appears in graph info
            info = graph.get_graph_info()
            assert vertex_type.value in info["vertex_types"]
            assert info["vertex_types"][vertex_type.value] >= 1