"""
Unit tests for Graph_Converter component.

These tests verify specific examples, edge cases, and error conditions.
"""

import pytest
import networkx as nx
from disaster_evacuation.osm.graph_converter import (
    GraphConverter,
    MissingCoordinateError
)


class TestGraphConverter:
    """Test suite for GraphConverter class."""
    
    def test_initialization(self):
        """Test that GraphConverter initializes correctly."""
        converter = GraphConverter()
        assert converter is not None
        assert str(converter) == "GraphConverter()"
    
    def test_node_id_mapping(self):
        """Test that OSM node IDs are mapped to sequential internal IDs."""
        converter = GraphConverter()
        
        # Create graph with arbitrary OSM node IDs
        G = nx.MultiDiGraph()
        osm_ids = [1000, 5000, 2500, 7500]
        for osm_id in osm_ids:
            G.add_node(osm_id, x=-122.0, y=37.8)
        
        # Create mapping
        id_mapping = converter._map_osm_nodes(G)
        
        # Verify sequential mapping
        assert len(id_mapping) == len(osm_ids)
        internal_ids = sorted(id_mapping.values())
        assert internal_ids == list(range(len(osm_ids)))
    
    def test_coordinate_extraction(self):
        """Test coordinate extraction from OSM nodes."""
        converter = GraphConverter()
        
        # Create graph with known coordinates
        G = nx.MultiDiGraph()
        G.add_node(0, x=-122.5, y=37.9)
        G.add_node(1, x=-122.6, y=37.8)
        
        id_mapping = {0: 0, 1: 1}
        coord_mapping = converter._create_coordinate_mapping(G, id_mapping)
        
        # Verify coordinates (stored as lat, lon)
        assert coord_mapping[0] == (37.9, -122.5)
        assert coord_mapping[1] == (37.8, -122.6)
    
    def test_edge_distance_extraction(self):
        """Test distance extraction from OSM edges."""
        converter = GraphConverter()
        
        # Create graph with edge
        G = nx.MultiDiGraph()
        G.add_node(0, x=-122.0, y=37.8)
        G.add_node(1, x=-122.001, y=37.8)
        G.add_edge(0, 1, length=150.5)
        
        distance = converter._extract_edge_distance(G, 0, 1)
        assert distance == 150.5
    
    def test_missing_coordinate_error(self):
        """Test that missing coordinate data raises MissingCoordinateError."""
        converter = GraphConverter()
        
        # Create graph with node missing coordinates
        G = nx.MultiDiGraph()
        G.add_node(0, x=-122.0, y=37.8)
        G.add_node(1, x=-122.001)  # Missing 'y' coordinate
        
        id_mapping = {0: 0, 1: 1}
        
        with pytest.raises(MissingCoordinateError) as exc_info:
            converter._create_coordinate_mapping(G, id_mapping)
        
        assert "missing coordinate data" in str(exc_info.value).lower()
    
    def test_missing_coordinate_x(self):
        """Test error when node is missing x (longitude) coordinate."""
        converter = GraphConverter()
        
        # Create graph with node missing x coordinate
        G = nx.MultiDiGraph()
        G.add_node(0, x=-122.0, y=37.8)
        G.add_node(1, y=37.8)  # Missing 'x' coordinate
        
        id_mapping = {0: 0, 1: 1}
        
        with pytest.raises(MissingCoordinateError):
            converter._create_coordinate_mapping(G, id_mapping)
    
    def test_missing_distance_defaults_to_zero(self):
        """Test that missing distance data defaults to 0.0."""
        converter = GraphConverter()
        
        # Create graph with edge missing length attribute
        G = nx.MultiDiGraph()
        G.add_node(0, x=-122.0, y=37.8)
        G.add_node(1, x=-122.001, y=37.8)
        G.add_edge(0, 1)  # No length attribute
        
        distance = converter._extract_edge_distance(G, 0, 1)
        assert distance == 0.0
    
    def test_full_conversion_simple_graph(self):
        """Test complete conversion of a simple graph."""
        converter = GraphConverter()
        
        # Create simple OSM graph
        G = nx.MultiDiGraph()
        G.add_node(100, x=-122.0, y=37.8)
        G.add_node(200, x=-122.001, y=37.801)
        G.add_node(300, x=-122.002, y=37.802)
        G.add_edge(100, 200, length=100.0)
        G.add_edge(200, 300, length=150.0)
        
        # Convert
        graph_manager, coord_mapping = converter.convert_osm_to_internal(G)
        
        # Verify conversion
        assert graph_manager.get_vertex_count() == 3
        assert graph_manager.get_edge_count() == 2
        assert len(coord_mapping) == 3
        
        # Verify edges exist
        assert graph_manager.is_connected("0", "1")
        assert graph_manager.is_connected("1", "2")
    
    def test_conversion_preserves_edge_weights(self):
        """Test that edge weights are correctly initialized."""
        converter = GraphConverter()
        
        # Create graph
        G = nx.MultiDiGraph()
        G.add_node(0, x=-122.0, y=37.8)
        G.add_node(1, x=-122.001, y=37.8)
        G.add_edge(0, 1, length=200.0)
        
        # Convert
        graph_manager, coord_mapping = converter.convert_osm_to_internal(G)
        
        # Verify weight equals distance (since risk and congestion are 0)
        weight = graph_manager.get_edge_weight("0", "1")
        assert abs(weight - 200.0) < 1e-6
    
    def test_empty_graph_conversion(self):
        """Test conversion of graph with no edges."""
        converter = GraphConverter()
        
        # Create graph with only nodes
        G = nx.MultiDiGraph()
        G.add_node(0, x=-122.0, y=37.8)
        G.add_node(1, x=-122.001, y=37.8)
        
        # Convert
        graph_manager, coord_mapping = converter.convert_osm_to_internal(G)
        
        # Verify nodes exist but no edges
        assert graph_manager.get_vertex_count() == 2
        assert graph_manager.get_edge_count() == 0
    
    def test_multi_edge_handling(self):
        """Test handling of multiple edges between same nodes."""
        converter = GraphConverter()
        
        # Create graph with parallel edges
        G = nx.MultiDiGraph()
        G.add_node(0, x=-122.0, y=37.8)
        G.add_node(1, x=-122.001, y=37.8)
        G.add_edge(0, 1, length=100.0)
        G.add_edge(0, 1, length=120.0)  # Parallel edge
        
        # Convert
        graph_manager, coord_mapping = converter.convert_osm_to_internal(G)
        
        # GraphManager doesn't support parallel edges, so only last edge is kept
        # This is expected behavior - internal graph uses simple adjacency list
        assert graph_manager.get_edge_count() >= 1
        assert graph_manager.is_connected("0", "1")
    
    def test_coordinate_mapping_keys(self):
        """Test that coordinate mapping uses internal IDs as keys."""
        converter = GraphConverter()
        
        # Create graph with arbitrary OSM IDs
        G = nx.MultiDiGraph()
        G.add_node(9999, x=-122.0, y=37.8)
        G.add_node(5555, x=-122.001, y=37.801)
        
        # Convert
        graph_manager, coord_mapping = converter.convert_osm_to_internal(G)
        
        # Coordinate mapping should use sequential internal IDs (0, 1)
        assert set(coord_mapping.keys()) == {0, 1}
        
        # Verify coordinates are tuples of (lat, lon)
        for internal_id, coords in coord_mapping.items():
            assert isinstance(coords, tuple)
            assert len(coords) == 2
            lat, lon = coords
            assert isinstance(lat, float)
            assert isinstance(lon, float)


class TestGraphConverterErrorCases:
    """Test suite for error handling in GraphConverter."""
    
    def test_invalid_graph_structure_no_nodes(self):
        """Test conversion of completely empty graph."""
        converter = GraphConverter()
        
        # Create empty graph
        G = nx.MultiDiGraph()
        
        # Should handle gracefully
        graph_manager, coord_mapping = converter.convert_osm_to_internal(G)
        
        assert graph_manager.get_vertex_count() == 0
        assert graph_manager.get_edge_count() == 0
        assert len(coord_mapping) == 0
    
    def test_node_with_invalid_coordinate_types(self):
        """Test handling of nodes with non-numeric coordinates."""
        converter = GraphConverter()
        
        # Create graph with invalid coordinate
        G = nx.MultiDiGraph()
        G.add_node(0, x="invalid", y=37.8)
        
        id_mapping = {0: 0}
        
        # Should not raise during mapping creation
        # (coordinate validation happens at usage time)
        coord_mapping = converter._create_coordinate_mapping(G, id_mapping)
        assert 0 in coord_mapping
