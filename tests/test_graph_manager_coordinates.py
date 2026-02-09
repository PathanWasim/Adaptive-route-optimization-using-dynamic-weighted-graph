"""
Unit tests for GraphManager coordinate storage functionality.

These tests verify specific examples, edge cases, and error conditions
for the coordinate storage methods.
"""

import pytest
from disaster_evacuation.graph.graph_manager import GraphManager
from disaster_evacuation.models import VertexType


class TestGraphManagerCoordinates:
    """Test suite for coordinate storage in GraphManager."""
    
    def test_set_and_get_coordinates(self):
        """Test setting and retrieving node coordinates."""
        graph = GraphManager()
        
        # Add vertex
        graph.add_vertex(
            vertex_id="0",
            vertex_type=VertexType.INTERSECTION,
            coordinates=(0.0, 0.0),
            capacity=None
        )
        
        # Set coordinates
        graph.set_node_coordinates("0", 37.8, -122.0)
        
        # Retrieve coordinates
        coords = graph.get_node_coordinates("0")
        assert coords == (37.8, -122.0)
    
    def test_set_coordinates_invalid_vertex(self):
        """Test that setting coordinates for non-existent vertex raises error."""
        graph = GraphManager()
        
        with pytest.raises(ValueError) as exc_info:
            graph.set_node_coordinates("999", 37.8, -122.0)
        
        assert "does not exist" in str(exc_info.value)
    
    def test_get_coordinates_invalid_vertex(self):
        """Test that getting coordinates for non-existent vertex raises error."""
        graph = GraphManager()
        
        with pytest.raises(ValueError) as exc_info:
            graph.get_node_coordinates("999")
        
        assert "does not exist" in str(exc_info.value)
    
    def test_get_coordinates_not_set(self):
        """Test that getting coordinates returns None if not set."""
        graph = GraphManager()
        
        # Add vertex without setting coordinates
        graph.add_vertex(
            vertex_id="0",
            vertex_type=VertexType.INTERSECTION,
            coordinates=(0.0, 0.0),
            capacity=None
        )
        
        # Should return None
        coords = graph.get_node_coordinates("0")
        assert coords is None
    
    def test_update_coordinates(self):
        """Test updating existing coordinates."""
        graph = GraphManager()
        
        # Add vertex
        graph.add_vertex(
            vertex_id="0",
            vertex_type=VertexType.INTERSECTION,
            coordinates=(0.0, 0.0),
            capacity=None
        )
        
        # Set initial coordinates
        graph.set_node_coordinates("0", 37.8, -122.0)
        
        # Update coordinates
        graph.set_node_coordinates("0", 37.9, -122.1)
        
        # Verify updated
        coords = graph.get_node_coordinates("0")
        assert coords == (37.9, -122.1)
    
    def test_multiple_nodes_coordinates(self):
        """Test coordinate storage for multiple nodes."""
        graph = GraphManager()
        
        # Add multiple vertices
        for i in range(5):
            graph.add_vertex(
                vertex_id=str(i),
                vertex_type=VertexType.INTERSECTION,
                coordinates=(0.0, 0.0),
                capacity=None
            )
            
            # Set unique coordinates
            graph.set_node_coordinates(str(i), 37.8 + i * 0.01, -122.0 + i * 0.01)
        
        # Verify all coordinates
        for i in range(5):
            coords = graph.get_node_coordinates(str(i))
            expected = (37.8 + i * 0.01, -122.0 + i * 0.01)
            assert coords == expected
    
    def test_coordinates_cleared_on_clear(self):
        """Test that coordinates are cleared when graph is cleared."""
        graph = GraphManager()
        
        # Add vertex with coordinates
        graph.add_vertex(
            vertex_id="0",
            vertex_type=VertexType.INTERSECTION,
            coordinates=(0.0, 0.0),
            capacity=None
        )
        graph.set_node_coordinates("0", 37.8, -122.0)
        
        # Clear graph
        graph.clear()
        
        # Verify graph is empty
        assert graph.get_vertex_count() == 0
    
    def test_coordinate_precision(self):
        """Test that coordinate precision is preserved."""
        graph = GraphManager()
        
        # Add vertex
        graph.add_vertex(
            vertex_id="0",
            vertex_type=VertexType.INTERSECTION,
            coordinates=(0.0, 0.0),
            capacity=None
        )
        
        # Set high-precision coordinates
        lat = 37.87654321
        lon = -122.12345678
        graph.set_node_coordinates("0", lat, lon)
        
        # Verify precision preserved
        coords = graph.get_node_coordinates("0")
        assert coords[0] == lat
        assert coords[1] == lon
