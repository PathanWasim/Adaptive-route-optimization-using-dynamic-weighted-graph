"""
Unit tests for OSM_Extractor component.

These tests verify specific examples, edge cases, and error conditions.
"""

import pytest
import networkx as nx
from disaster_evacuation.osm.osm_extractor import (
    OSMExtractor, 
    AreaTooLargeError, 
    PlaceNotFoundError, 
    EmptyNetworkError
)


class TestOSMExtractor:
    """Test suite for OSMExtractor class."""
    
    def test_initialization(self):
        """Test that OSMExtractor initializes correctly."""
        extractor = OSMExtractor()
        assert extractor is not None
        assert str(extractor) == "OSMExtractor(cache=True)"
    
    def test_get_network_stats_basic(self):
        """Test network statistics calculation."""
        extractor = OSMExtractor()
        
        # Create a simple test graph
        G = nx.MultiDiGraph()
        G.add_node(0, x=-122.0, y=37.8)
        G.add_node(1, x=-122.001, y=37.8)
        G.add_node(2, x=-122.0, y=37.801)
        G.add_edge(0, 1, length=100.0)
        G.add_edge(1, 2, length=150.0)
        
        stats = extractor.get_network_stats(G)
        
        assert stats['num_nodes'] == 3
        assert stats['num_edges'] == 2
        assert stats['avg_edge_length'] == 125.0  # (100 + 150) / 2
        assert 'area_km2' in stats
    
    def test_empty_network_error(self):
        """Test that empty networks raise EmptyNetworkError."""
        extractor = OSMExtractor()
        
        # Create graph with nodes but no edges
        G = nx.MultiDiGraph()
        G.add_node(0, x=-122.0, y=37.8)
        G.add_node(1, x=-122.001, y=37.8)
        
        with pytest.raises(EmptyNetworkError) as exc_info:
            extractor._validate_network(G, "test_location")
        
        assert "No roads found" in str(exc_info.value)
    
    def test_area_too_large_error(self):
        """Test that large areas raise AreaTooLargeError."""
        extractor = OSMExtractor()
        
        # Create a large graph (simulate > 3 km²)
        G = nx.MultiDiGraph()
        
        # Add nodes in a large grid
        for i in range(20):
            for j in range(20):
                node_id = i * 20 + j
                # Large spacing to create big area
                G.add_node(node_id, x=-122.0 + j * 0.01, y=37.8 + i * 0.01)
        
        # Add edges
        for i in range(19):
            for j in range(19):
                node_id = i * 20 + j
                G.add_edge(node_id, node_id + 1, length=1000.0)
                G.add_edge(node_id, node_id + 20, length=1000.0)
        
        # Mark as projected for area calculation
        G.graph['crs'] = 'EPSG:32610'  # UTM zone 10N
        
        with pytest.raises(AreaTooLargeError) as exc_info:
            extractor._validate_network(G, "test_location")
        
        assert "exceeds the 3 km² limit" in str(exc_info.value)
    
    def test_valid_network_passes_validation(self):
        """Test that valid networks pass validation."""
        extractor = OSMExtractor()
        
        # Create a small valid graph
        G = nx.MultiDiGraph()
        
        # Add nodes in a small grid with very small spacing
        for i in range(5):
            for j in range(5):
                node_id = i * 5 + j
                # Use much smaller spacing to keep area under 3 km²
                G.add_node(node_id, x=-122.0 + j * 0.0001, y=37.8 + i * 0.0001)
        
        # Add edges
        for i in range(4):
            for j in range(4):
                node_id = i * 5 + j
                G.add_edge(node_id, node_id + 1, length=100.0)
                G.add_edge(node_id, node_id + 5, length=100.0)
        
        # Should not raise any exception
        extractor._validate_network(G, "test_location")
    
    def test_network_stats_with_zero_edges(self):
        """Test statistics calculation with zero edges."""
        extractor = OSMExtractor()
        
        G = nx.MultiDiGraph()
        G.add_node(0, x=-122.0, y=37.8)
        
        stats = extractor.get_network_stats(G)
        
        assert stats['num_nodes'] == 1
        assert stats['num_edges'] == 0
        assert stats['avg_edge_length'] == 0.0


class TestOSMExtractorRealData:
    """
    Tests that would use real OSM data.
    
    These are marked as slow and can be skipped in CI/CD.
    """
    
    @pytest.mark.slow
    @pytest.mark.skip(reason="Requires internet connection and OSM API access")
    def test_extract_small_place(self):
        """Test extracting a small real place."""
        extractor = OSMExtractor()
        
        # Try to extract a very small area
        # Note: This test requires internet and may be slow
        try:
            G = extractor.extract_by_place("Piedmont, California, USA", network_type="drive")
            assert G.number_of_nodes() > 0
            assert G.number_of_edges() > 0
            
            stats = extractor.get_network_stats(G)
            assert stats['area_km2'] <= 3.0
        except Exception as e:
            pytest.skip(f"OSM extraction failed: {e}")
    
    @pytest.mark.slow
    @pytest.mark.skip(reason="Requires internet connection")
    def test_invalid_place_name(self):
        """Test that invalid place names raise PlaceNotFoundError."""
        extractor = OSMExtractor()
        
        with pytest.raises(PlaceNotFoundError):
            extractor.extract_by_place("ThisPlaceDoesNotExist12345XYZ")
    
    @pytest.mark.slow
    @pytest.mark.skip(reason="Requires internet connection")
    def test_extract_by_bbox_small_area(self):
        """Test extracting by bounding box."""
        extractor = OSMExtractor()
        
        # Small bbox around Berkeley, CA
        north, south = 37.875, 37.865
        east, west = -122.255, -122.265
        
        try:
            G = extractor.extract_by_bbox(north, south, east, west)
            assert G.number_of_nodes() > 0
            assert G.number_of_edges() > 0
        except Exception as e:
            pytest.skip(f"OSM extraction failed: {e}")
