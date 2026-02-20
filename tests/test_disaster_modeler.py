"""
Unit tests for Disaster_Modeler component.

These tests verify specific examples, edge cases, and error conditions.
"""

import pytest
import networkx as nx
import math
from disaster_evacuation.osm.graph_converter import GraphConverter
from disaster_evacuation.models.disaster_modeler import (
    DisasterModeler,
    InvalidCoordinateError,
    InvalidRadiusError
)


class TestDisasterModeler:
    """Test suite for DisasterModeler class."""
    
    def test_initialization(self):
        """Test that DisasterModeler initializes correctly."""
        # Create simple graph
        G = nx.MultiDiGraph()
        G.add_node(0, x=-122.0, y=37.8)
        G.add_node(1, x=-122.001, y=37.8)
        G.add_edge(0, 1, length=100.0)
        
        converter = GraphConverter()
        graph_manager, coord_mapping = converter.convert_osm_to_internal(G)
        
        modeler = DisasterModeler(graph_manager, coord_mapping)
        assert modeler is not None
        assert "DisasterModeler" in str(modeler)
    
    def test_haversine_distance(self):
        """Test Haversine distance calculation."""
        G = nx.MultiDiGraph()
        G.add_node(0, x=-122.0, y=37.8)
        
        converter = GraphConverter()
        graph_manager, coord_mapping = converter.convert_osm_to_internal(G)
        
        modeler = DisasterModeler(graph_manager, coord_mapping)
        
        # Test distance between same point
        coord = (37.8, -122.0)
        distance = modeler._haversine_distance(coord, coord)
        assert distance == 0.0
        
        # Test distance between two known points
        coord1 = (37.8, -122.0)
        coord2 = (37.801, -122.0)  # ~111 meters north
        distance = modeler._haversine_distance(coord1, coord2)
        assert 100 < distance < 120  # Approximately 111 meters
    
    def test_edge_midpoint(self):
        """Test edge midpoint calculation."""
        G = nx.MultiDiGraph()
        G.add_node(0, x=-122.0, y=37.8)
        G.add_node(1, x=-122.002, y=37.802)
        G.add_edge(0, 1, length=100.0)
        
        converter = GraphConverter()
        graph_manager, coord_mapping = converter.convert_osm_to_internal(G)
        
        modeler = DisasterModeler(graph_manager, coord_mapping)
        
        midpoint = modeler._get_edge_midpoint(0, 1)
        expected_lat = (37.8 + 37.802) / 2
        expected_lon = (-122.0 + -122.002) / 2
        
        assert abs(midpoint[0] - expected_lat) < 1e-6
        assert abs(midpoint[1] - expected_lon) < 1e-6
    
    def test_invalid_epicenter_latitude(self):
        """Test that invalid latitude raises InvalidCoordinateError."""
        G = nx.MultiDiGraph()
        G.add_node(0, x=-122.0, y=37.8)
        
        converter = GraphConverter()
        graph_manager, coord_mapping = converter.convert_osm_to_internal(G)
        
        modeler = DisasterModeler(graph_manager, coord_mapping)
        
        # Test latitude > 90
        with pytest.raises(InvalidCoordinateError) as exc_info:
            modeler.apply_flood((95.0, -122.0), 100.0)
        assert "Latitude" in str(exc_info.value)
        
        # Test latitude < -90
        with pytest.raises(InvalidCoordinateError):
            modeler.apply_flood((-95.0, -122.0), 100.0)
    
    def test_invalid_epicenter_longitude(self):
        """Test that invalid longitude raises InvalidCoordinateError."""
        G = nx.MultiDiGraph()
        G.add_node(0, x=-122.0, y=37.8)
        
        converter = GraphConverter()
        graph_manager, coord_mapping = converter.convert_osm_to_internal(G)
        
        modeler = DisasterModeler(graph_manager, coord_mapping)
        
        # Test longitude > 180
        with pytest.raises(InvalidCoordinateError) as exc_info:
            modeler.apply_flood((37.8, 185.0), 100.0)
        assert "Longitude" in str(exc_info.value)
        
        # Test longitude < -180
        with pytest.raises(InvalidCoordinateError):
            modeler.apply_flood((37.8, -185.0), 100.0)
    
    def test_invalid_radius_negative(self):
        """Test that negative radius raises InvalidRadiusError."""
        G = nx.MultiDiGraph()
        G.add_node(0, x=-122.0, y=37.8)
        
        converter = GraphConverter()
        graph_manager, coord_mapping = converter.convert_osm_to_internal(G)
        
        modeler = DisasterModeler(graph_manager, coord_mapping)
        
        with pytest.raises(InvalidRadiusError) as exc_info:
            modeler.apply_flood((37.8, -122.0), -100.0)
        assert "positive" in str(exc_info.value).lower()
    
    def test_invalid_radius_zero(self):
        """Test that zero radius raises InvalidRadiusError."""
        G = nx.MultiDiGraph()
        G.add_node(0, x=-122.0, y=37.8)
        
        converter = GraphConverter()
        graph_manager, coord_mapping = converter.convert_osm_to_internal(G)
        
        modeler = DisasterModeler(graph_manager, coord_mapping)
        
        with pytest.raises(InvalidRadiusError):
            modeler.apply_flood((37.8, -122.0), 0.0)
    
    def test_flood_increases_weight(self):
        """Test that flood disaster increases edge weights."""
        G = nx.MultiDiGraph()
        G.add_node(0, x=-122.0, y=37.8)
        G.add_node(1, x=-122.0001, y=37.8)
        G.add_edge(0, 1, length=100.0)
        
        converter = GraphConverter()
        graph_manager, coord_mapping = converter.convert_osm_to_internal(G)
        
        # Get initial weight
        initial_weight = graph_manager.get_edge_weight("0", "1")
        
        # Apply flood at node 0
        modeler = DisasterModeler(graph_manager, coord_mapping)
        modeler.apply_flood(coord_mapping[0], 50.0, risk_multiplier=0.5)
        
        # Get final weight
        final_weight = graph_manager.get_edge_weight("0", "1")
        
        # Weight should have increased
        assert final_weight > initial_weight
    
    def test_fire_blocks_roads(self):
        """Test that fire disaster blocks roads."""
        G = nx.MultiDiGraph()
        G.add_node(0, x=-122.0, y=37.8)
        G.add_node(1, x=-122.0001, y=37.8)
        G.add_edge(0, 1, length=100.0)
        
        converter = GraphConverter()
        graph_manager, coord_mapping = converter.convert_osm_to_internal(G)
        
        # Apply fire at node 0
        modeler = DisasterModeler(graph_manager, coord_mapping)
        modeler.apply_fire(coord_mapping[0], 50.0)
        
        # Get final weight
        final_weight = graph_manager.get_edge_weight("0", "1")
        
        # Weight should be blocked
        assert final_weight >= DisasterModeler.BLOCKED_WEIGHT * 0.99
    
    def test_earthquake_affects_roads(self):
        """Test that earthquake disaster affects roads."""
        G = nx.MultiDiGraph()
        G.add_node(0, x=-122.0, y=37.8)
        G.add_node(1, x=-122.0001, y=37.8)
        G.add_edge(0, 1, length=100.0)
        
        converter = GraphConverter()
        graph_manager, coord_mapping = converter.convert_osm_to_internal(G)
        
        # Get initial weight
        initial_weight = graph_manager.get_edge_weight("0", "1")
        
        # Apply earthquake at node 0
        modeler = DisasterModeler(graph_manager, coord_mapping)
        modeler.apply_earthquake(coord_mapping[0], 50.0, 
                                failure_probability=1.0,  # Always block
                                congestion_multiplier=0.8)
        
        # Get final weight
        final_weight = graph_manager.get_edge_weight("0", "1")
        
        # Weight should have changed (either blocked or increased)
        assert final_weight != initial_weight
    
    def test_disaster_outside_radius_no_effect(self):
        """Test that disasters outside radius don't affect edges."""
        G = nx.MultiDiGraph()
        G.add_node(0, x=-122.0, y=37.8)
        G.add_node(1, x=-122.0001, y=37.8)
        G.add_edge(0, 1, length=100.0)
        
        converter = GraphConverter()
        graph_manager, coord_mapping = converter.convert_osm_to_internal(G)
        
        # Get initial weight
        initial_weight = graph_manager.get_edge_weight("0", "1")
        
        # Apply flood far away with small radius
        far_epicenter = (37.9, -122.0)  # ~11 km away
        modeler = DisasterModeler(graph_manager, coord_mapping)
        modeler.apply_flood(far_epicenter, 10.0, risk_multiplier=0.5)
        
        # Get final weight
        final_weight = graph_manager.get_edge_weight("0", "1")
        
        # Weight should be unchanged
        assert abs(final_weight - initial_weight) < 1e-6


class TestDisasterModelerEdgeCases:
    """Test suite for edge cases in DisasterModeler."""
    
    def test_empty_graph(self):
        """Test disaster modeling on empty graph."""
        G = nx.MultiDiGraph()
        
        converter = GraphConverter()
        graph_manager, coord_mapping = converter.convert_osm_to_internal(G)
        
        modeler = DisasterModeler(graph_manager, coord_mapping)
        
        # Should not raise error
        modeler.apply_flood((37.8, -122.0), 100.0)
    
    def test_graph_with_no_edges(self):
        """Test disaster modeling on graph with nodes but no edges."""
        G = nx.MultiDiGraph()
        G.add_node(0, x=-122.0, y=37.8)
        G.add_node(1, x=-122.001, y=37.8)
        
        converter = GraphConverter()
        graph_manager, coord_mapping = converter.convert_osm_to_internal(G)
        
        modeler = DisasterModeler(graph_manager, coord_mapping)
        
        # Should not raise error
        modeler.apply_flood((37.8, -122.0), 100.0)
