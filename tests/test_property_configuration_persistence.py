"""
Property-based tests for configuration persistence round-trip.

Feature: disaster-evacuation-routing
Property 12: Configuration Persistence Round-Trip

For any valid graph configuration, saving to persistent storage and then loading
should produce an equivalent graph with the same structure, connectivity, and properties.

Validates: Requirements 10.1, 10.2
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from hypothesis import given, strategies as st, settings, assume
from disaster_evacuation.config import ConfigurationManager
from disaster_evacuation.models import GraphManager
from disaster_evacuation.models import VertexType, DisasterType


# Custom strategies for generating test data
@st.composite
def vertex_id_strategy(draw):
    """Generate valid vertex IDs."""
    return draw(st.text(
        alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd')),
        min_size=1,
        max_size=10
    ))


@st.composite
def coordinates_strategy(draw):
    """Generate valid coordinate tuples."""
    x = draw(st.floats(min_value=-1000.0, max_value=1000.0, allow_nan=False, allow_infinity=False))
    y = draw(st.floats(min_value=-1000.0, max_value=1000.0, allow_nan=False, allow_infinity=False))
    return (x, y)


@st.composite
def vertex_type_strategy(draw):
    """Generate valid vertex types."""
    return draw(st.sampled_from([VertexType.INTERSECTION, VertexType.SHELTER, VertexType.EVACUATION_POINT]))


@st.composite
def capacity_strategy(draw):
    """Generate valid capacity values."""
    return draw(st.one_of(
        st.none(),
        st.integers(min_value=1, max_value=10000)
    ))


@st.composite
def edge_weight_strategy(draw):
    """Generate valid edge weight components."""
    distance = draw(st.floats(min_value=0.1, max_value=100.0, allow_nan=False, allow_infinity=False))
    risk = draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False))
    congestion = draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False))
    return distance, risk, congestion


@st.composite
def graph_strategy(draw, min_vertices=2, max_vertices=20):
    """Generate valid graph configurations."""
    # Generate vertices
    num_vertices = draw(st.integers(min_value=min_vertices, max_value=max_vertices))
    vertex_ids = draw(st.lists(
        vertex_id_strategy(),
        min_size=num_vertices,
        max_size=num_vertices,
        unique=True
    ))
    
    graph = GraphManager()
    
    # Add vertices to graph
    for vertex_id in vertex_ids:
        vertex_type = draw(vertex_type_strategy())
        coordinates = draw(coordinates_strategy())
        capacity = draw(capacity_strategy()) if vertex_type != VertexType.INTERSECTION else None
        graph.add_vertex(vertex_id, vertex_type, coordinates, capacity)
    
    # Generate edges (ensure at least some connectivity)
    num_edges = draw(st.integers(min_value=num_vertices - 1, max_value=min(num_vertices * 3, 50)))
    
    for _ in range(num_edges):
        source = draw(st.sampled_from(vertex_ids))
        target = draw(st.sampled_from(vertex_ids))
        
        # Avoid self-loops
        if source == target:
            continue
        
        # Check if edge already exists
        existing_neighbors = [edge.target for edge in graph.get_neighbors(source)]
        if target in existing_neighbors:
            continue
        
        distance, risk, congestion = draw(edge_weight_strategy())
        graph.add_edge(source, target, distance, risk, congestion)
    
    return graph


@st.composite
def metadata_strategy(draw):
    """Generate valid metadata dictionaries."""
    return draw(st.dictionaries(
        keys=st.text(min_size=1, max_size=20),
        values=st.one_of(
            st.text(max_size=100),
            st.integers(),
            st.floats(allow_nan=False, allow_infinity=False),
            st.booleans(),
            st.lists(st.text(max_size=50), max_size=5)
        ),
        max_size=10
    ))


class TestPropertyConfigurationPersistence:
    """Property-based tests for configuration persistence."""
    
    @given(graph=graph_strategy())
    @settings(max_examples=100, deadline=2000)
    def test_save_load_preserves_graph_structure(self, graph):
        """
        Property 12: Configuration Persistence Round-Trip
        
        For any valid graph, saving and loading should preserve:
        - Number of vertices
        - Number of edges
        - Vertex IDs
        """
        # Create temporary directory and config manager
        temp_dir = tempfile.mkdtemp()
        try:
            config_manager = ConfigurationManager(temp_dir, enable_logging=False)
            
            # Save the graph
            filename = "test_roundtrip"
            result = config_manager.save_graph_configuration(graph, filename)
            assert result is True, "Save operation should succeed"
            
            # Load the graph
            loaded_result = config_manager.load_graph_configuration(filename)
            assert loaded_result is not None, "Load operation should succeed"
            
            loaded_graph, _ = loaded_result
            
            # Verify structure preservation
            original_vertex_ids = set(graph.get_vertex_ids())
            loaded_vertex_ids = set(loaded_graph.get_vertex_ids())
            
            assert original_vertex_ids == loaded_vertex_ids, \
                f"Vertex IDs should be preserved. Original: {original_vertex_ids}, Loaded: {loaded_vertex_ids}"
            
            assert len(graph.get_all_edges()) == len(loaded_graph.get_all_edges()), \
                "Number of edges should be preserved"
            
            config_manager.close()
        finally:
            shutil.rmtree(temp_dir)
    
    @given(graph=graph_strategy())
    @settings(max_examples=100, deadline=2000)
    def test_save_load_preserves_vertex_properties(self, graph):
        """
        Property 12: Configuration Persistence Round-Trip
        
        For any valid graph, saving and loading should preserve vertex properties:
        - Vertex type
        - Coordinates
        - Capacity
        """
        temp_dir = tempfile.mkdtemp()
        try:
            config_manager = ConfigurationManager(temp_dir, enable_logging=False)
            
            # Save the graph
            filename = "test_vertex_props"
            config_manager.save_graph_configuration(graph, filename)
            
            # Load the graph
            loaded_graph, _ = config_manager.load_graph_configuration(filename)
            
            # Verify vertex properties
            for vertex_id in graph.get_vertex_ids():
                original_vertex = graph.get_vertex(vertex_id)
                loaded_vertex = loaded_graph.get_vertex(vertex_id)
                
                assert loaded_vertex is not None, f"Vertex {vertex_id} should exist after loading"
                assert original_vertex.vertex_type == loaded_vertex.vertex_type, \
                    f"Vertex type should be preserved for {vertex_id}"
                
                # Check coordinates (with floating point tolerance)
                assert abs(original_vertex.coordinates[0] - loaded_vertex.coordinates[0]) < 1e-6, \
                    f"X coordinate should be preserved for {vertex_id}"
                assert abs(original_vertex.coordinates[1] - loaded_vertex.coordinates[1]) < 1e-6, \
                    f"Y coordinate should be preserved for {vertex_id}"
                
                assert original_vertex.capacity == loaded_vertex.capacity, \
                    f"Capacity should be preserved for {vertex_id}"
            
            config_manager.close()
        finally:
            shutil.rmtree(temp_dir)
    
    @given(graph=graph_strategy())
    @settings(max_examples=100, deadline=2000)
    def test_save_load_preserves_edge_properties(self, graph):
        """
        Property 12: Configuration Persistence Round-Trip
        
        For any valid graph, saving and loading should preserve edge properties:
        - Source and target vertices
        - Base distance
        - Base risk
        - Base congestion
        """
        temp_dir = tempfile.mkdtemp()
        try:
            config_manager = ConfigurationManager(temp_dir, enable_logging=False)
            
            # Save the graph
            filename = "test_edge_props"
            config_manager.save_graph_configuration(graph, filename)
            
            # Load the graph
            loaded_graph, _ = config_manager.load_graph_configuration(filename)
            
            # Create edge lookup for comparison
            original_edges = {(e.source, e.target): e for e in graph.get_all_edges()}
            loaded_edges = {(e.source, e.target): e for e in loaded_graph.get_all_edges()}
            
            assert len(original_edges) == len(loaded_edges), "Number of edges should match"
            
            # Verify edge properties
            for (source, target), original_edge in original_edges.items():
                assert (source, target) in loaded_edges, \
                    f"Edge ({source}, {target}) should exist after loading"
                
                loaded_edge = loaded_edges[(source, target)]
                
                # Check edge properties with floating point tolerance
                assert abs(original_edge.base_distance - loaded_edge.base_distance) < 1e-6, \
                    f"Base distance should be preserved for edge ({source}, {target})"
                assert abs(original_edge.base_risk - loaded_edge.base_risk) < 1e-6, \
                    f"Base risk should be preserved for edge ({source}, {target})"
                assert abs(original_edge.base_congestion - loaded_edge.base_congestion) < 1e-6, \
                    f"Base congestion should be preserved for edge ({source}, {target})"
            
            config_manager.close()
        finally:
            shutil.rmtree(temp_dir)
    
    @given(graph=graph_strategy(), metadata=metadata_strategy())
    @settings(max_examples=100, deadline=2000)
    def test_save_load_preserves_metadata(self, graph, metadata):
        """
        Property 12: Configuration Persistence Round-Trip
        
        For any valid graph with metadata, saving and loading should preserve metadata.
        """
        temp_dir = tempfile.mkdtemp()
        try:
            config_manager = ConfigurationManager(temp_dir, enable_logging=False)
            
            # Save the graph with metadata
            filename = "test_metadata"
            config_manager.save_graph_configuration(graph, filename, metadata)
            
            # Load the graph
            loaded_graph, loaded_metadata = config_manager.load_graph_configuration(filename)
            
            # Verify metadata preservation (excluding auto-added fields)
            for key, value in metadata.items():
                assert key in loaded_metadata, f"Metadata key '{key}' should be preserved"
                assert loaded_metadata[key] == value, \
                    f"Metadata value for '{key}' should be preserved"
            
            config_manager.close()
        finally:
            shutil.rmtree(temp_dir)
    
    @given(graph=graph_strategy())
    @settings(max_examples=100, deadline=2000)
    def test_save_load_preserves_connectivity(self, graph):
        """
        Property 12: Configuration Persistence Round-Trip
        
        For any valid graph, saving and loading should preserve connectivity:
        - Neighbor relationships
        - Reachability between vertices
        """
        temp_dir = tempfile.mkdtemp()
        try:
            config_manager = ConfigurationManager(temp_dir, enable_logging=False)
            
            # Save the graph
            filename = "test_connectivity"
            config_manager.save_graph_configuration(graph, filename)
            
            # Load the graph
            loaded_graph, _ = config_manager.load_graph_configuration(filename)
            
            # Verify neighbor relationships
            for vertex_id in graph.get_vertex_ids():
                original_neighbors = {edge.target for edge in graph.get_neighbors(vertex_id)}
                loaded_neighbors = {edge.target for edge in loaded_graph.get_neighbors(vertex_id)}
                
                assert original_neighbors == loaded_neighbors, \
                    f"Neighbor set should be preserved for vertex {vertex_id}"
            
            config_manager.close()
        finally:
            shutil.rmtree(temp_dir)
    
    @given(graph=graph_strategy())
    @settings(max_examples=50, deadline=2000)
    def test_multiple_save_load_cycles_preserve_graph(self, graph):
        """
        Property 12: Configuration Persistence Round-Trip
        
        Multiple save/load cycles should preserve graph properties.
        """
        temp_dir = tempfile.mkdtemp()
        try:
            config_manager = ConfigurationManager(temp_dir, enable_logging=False)
            
            filename = "test_multiple_cycles"
            
            # Perform multiple save/load cycles
            current_graph = graph
            for cycle in range(3):
                # Save
                config_manager.save_graph_configuration(current_graph, filename)
                
                # Load
                loaded_graph, _ = config_manager.load_graph_configuration(filename)
                
                # Verify structure is preserved
                assert set(current_graph.get_vertex_ids()) == set(loaded_graph.get_vertex_ids()), \
                    f"Vertex IDs should be preserved in cycle {cycle}"
                assert len(current_graph.get_all_edges()) == len(loaded_graph.get_all_edges()), \
                    f"Edge count should be preserved in cycle {cycle}"
                
                current_graph = loaded_graph
            
            config_manager.close()
        finally:
            shutil.rmtree(temp_dir)
    
    @given(graph=graph_strategy())
    @settings(max_examples=100, deadline=2000)
    def test_save_load_idempotency(self, graph):
        """
        Property 12: Configuration Persistence Round-Trip
        
        Loading a saved graph twice should produce identical results.
        """
        temp_dir = tempfile.mkdtemp()
        try:
            config_manager = ConfigurationManager(temp_dir, enable_logging=False)
            
            filename = "test_idempotency"
            
            # Save once
            config_manager.save_graph_configuration(graph, filename)
            
            # Load twice
            loaded_graph1, _ = config_manager.load_graph_configuration(filename)
            loaded_graph2, _ = config_manager.load_graph_configuration(filename)
            
            # Verify both loads produce identical graphs
            assert set(loaded_graph1.get_vertex_ids()) == set(loaded_graph2.get_vertex_ids()), \
                "Multiple loads should produce identical vertex sets"
            assert len(loaded_graph1.get_all_edges()) == len(loaded_graph2.get_all_edges()), \
                "Multiple loads should produce identical edge counts"
            
            config_manager.close()
        finally:
            shutil.rmtree(temp_dir)
    
    @given(graph=graph_strategy())
    @settings(max_examples=100, deadline=2000)
    def test_empty_graph_round_trip(self, graph):
        """
        Property 12: Configuration Persistence Round-Trip
        
        Even empty graphs should round-trip correctly.
        """
        temp_dir = tempfile.mkdtemp()
        try:
            config_manager = ConfigurationManager(temp_dir, enable_logging=False)
            
            # Create an empty graph
            empty_graph = GraphManager()
            
            filename = "test_empty"
            config_manager.save_graph_configuration(empty_graph, filename)
            
            loaded_graph, _ = config_manager.load_graph_configuration(filename)
            
            assert len(loaded_graph.get_vertex_ids()) == 0, "Empty graph should have no vertices"
            assert len(loaded_graph.get_all_edges()) == 0, "Empty graph should have no edges"
            
            config_manager.close()
        finally:
            shutil.rmtree(temp_dir)
