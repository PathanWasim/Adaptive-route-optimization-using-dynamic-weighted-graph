"""
Unit tests for GraphManager class.
"""

import pytest
from disaster_evacuation.models import GraphManager
from disaster_evacuation.models import VertexType


class TestGraphManager:
    """Test cases for GraphManager class."""
    
    def test_empty_graph_creation(self):
        """Test creating an empty graph."""
        graph = GraphManager()
        assert graph.get_vertex_count() == 0
        assert graph.get_edge_count() == 0
        assert len(graph.get_all_vertices()) == 0
        assert len(graph.get_all_edges()) == 0
    
    def test_add_vertex(self):
        """Test adding vertices to the graph."""
        graph = GraphManager()
        
        # Add intersection vertex
        graph.add_vertex("A", VertexType.INTERSECTION, (0.0, 0.0))
        assert graph.get_vertex_count() == 1
        assert graph.has_vertex("A")
        
        vertex = graph.get_vertex("A")
        assert vertex.id == "A"
        assert vertex.vertex_type == VertexType.INTERSECTION
        assert vertex.coordinates == (0.0, 0.0)
        assert vertex.capacity is None
        
        # Add shelter with capacity
        graph.add_vertex("S1", VertexType.SHELTER, (1.0, 1.0), capacity=100)
        assert graph.get_vertex_count() == 2
        
        shelter = graph.get_vertex("S1")
        assert shelter.capacity == 100
    
    def test_add_duplicate_vertex(self):
        """Test adding duplicate vertex raises error."""
        graph = GraphManager()
        graph.add_vertex("A", VertexType.INTERSECTION, (0.0, 0.0))
        
        with pytest.raises(ValueError, match="Vertex 'A' already exists"):
            graph.add_vertex("A", VertexType.SHELTER, (1.0, 1.0))
    
    def test_add_edge(self):
        """Test adding edges to the graph."""
        graph = GraphManager()
        graph.add_vertex("A", VertexType.INTERSECTION, (0.0, 0.0))
        graph.add_vertex("B", VertexType.INTERSECTION, (1.0, 0.0))
        
        # Add edge
        graph.add_edge("A", "B", 1.0, 0.1, 0.2)
        assert graph.get_edge_count() == 1
        assert graph.is_connected("A", "B")
        assert not graph.is_connected("B", "A")  # Directed graph
        
        # Check edge properties
        edge = graph.get_edge("A", "B")
        assert edge.source == "A"
        assert edge.target == "B"
        assert edge.base_distance == 1.0
        assert edge.base_risk == 0.1
        assert edge.base_congestion == 0.2
        assert edge.current_weight == 1.3  # 1.0 + 0.1 + 0.2
        
        # Check weight retrieval
        assert graph.get_edge_weight("A", "B") == 1.3
    
    def test_add_edge_nonexistent_vertex(self):
        """Test adding edge with nonexistent vertices raises error."""
        graph = GraphManager()
        graph.add_vertex("A", VertexType.INTERSECTION, (0.0, 0.0))
        
        with pytest.raises(ValueError, match="Target vertex 'B' does not exist"):
            graph.add_edge("A", "B", 1.0, 0.1, 0.2)
        
        with pytest.raises(ValueError, match="Source vertex 'C' does not exist"):
            graph.add_edge("C", "A", 1.0, 0.1, 0.2)
    
    def test_get_neighbors(self):
        """Test getting neighbors of a vertex."""
        graph = GraphManager()
        graph.add_vertex("A", VertexType.INTERSECTION, (0.0, 0.0))
        graph.add_vertex("B", VertexType.INTERSECTION, (1.0, 0.0))
        graph.add_vertex("C", VertexType.INTERSECTION, (2.0, 0.0))
        
        # Add edges from A
        graph.add_edge("A", "B", 1.0, 0.1, 0.2)
        graph.add_edge("A", "C", 2.0, 0.2, 0.3)
        
        neighbors = graph.get_neighbors("A")
        assert len(neighbors) == 2
        
        neighbor_targets = [edge.target for edge in neighbors]
        assert "B" in neighbor_targets
        assert "C" in neighbor_targets
        
        # Vertex with no outgoing edges
        assert len(graph.get_neighbors("B")) == 0
    
    def test_get_neighbors_nonexistent_vertex(self):
        """Test getting neighbors of nonexistent vertex raises error."""
        graph = GraphManager()
        
        with pytest.raises(ValueError, match="Vertex 'A' does not exist"):
            graph.get_neighbors("A")
    
    def test_update_edge_weight(self):
        """Test updating edge weights."""
        graph = GraphManager()
        graph.add_vertex("A", VertexType.INTERSECTION, (0.0, 0.0))
        graph.add_vertex("B", VertexType.INTERSECTION, (1.0, 0.0))
        graph.add_edge("A", "B", 1.0, 0.1, 0.2)
        
        # Update weight
        graph.update_edge_weight("A", "B", 5.0)
        assert graph.get_edge_weight("A", "B") == 5.0
        
        edge = graph.get_edge("A", "B")
        assert edge.current_weight == 5.0
    
    def test_update_edge_weight_invalid(self):
        """Test updating edge weight with invalid values."""
        graph = GraphManager()
        graph.add_vertex("A", VertexType.INTERSECTION, (0.0, 0.0))
        graph.add_vertex("B", VertexType.INTERSECTION, (1.0, 0.0))
        graph.add_edge("A", "B", 1.0, 0.1, 0.2)
        
        # Negative weight
        with pytest.raises(ValueError, match="Edge weight cannot be negative"):
            graph.update_edge_weight("A", "B", -1.0)
        
        # Nonexistent edge
        with pytest.raises(ValueError, match="Edge 'A' -> 'C' does not exist"):
            graph.update_edge_weight("A", "C", 2.0)
    
    def test_remove_vertex(self):
        """Test removing vertices from the graph."""
        graph = GraphManager()
        graph.add_vertex("A", VertexType.INTERSECTION, (0.0, 0.0))
        graph.add_vertex("B", VertexType.INTERSECTION, (1.0, 0.0))
        graph.add_vertex("C", VertexType.INTERSECTION, (2.0, 0.0))
        
        # Add edges
        graph.add_edge("A", "B", 1.0, 0.1, 0.2)
        graph.add_edge("B", "C", 1.0, 0.1, 0.2)
        graph.add_edge("A", "C", 2.0, 0.2, 0.3)
        
        assert graph.get_vertex_count() == 3
        assert graph.get_edge_count() == 3
        
        # Remove vertex B
        graph.remove_vertex("B")
        assert graph.get_vertex_count() == 2
        assert graph.get_edge_count() == 1  # Only A->C remains
        assert not graph.has_vertex("B")
        assert not graph.is_connected("A", "B")
        assert not graph.is_connected("B", "C")
        assert graph.is_connected("A", "C")
    
    def test_remove_nonexistent_vertex(self):
        """Test removing nonexistent vertex raises error."""
        graph = GraphManager()
        
        with pytest.raises(ValueError, match="Vertex 'A' does not exist"):
            graph.remove_vertex("A")
    
    def test_clear_graph(self):
        """Test clearing the graph."""
        graph = GraphManager()
        graph.add_vertex("A", VertexType.INTERSECTION, (0.0, 0.0))
        graph.add_vertex("B", VertexType.INTERSECTION, (1.0, 0.0))
        graph.add_edge("A", "B", 1.0, 0.1, 0.2)
        
        assert graph.get_vertex_count() == 2
        assert graph.get_edge_count() == 1
        
        graph.clear()
        assert graph.get_vertex_count() == 0
        assert graph.get_edge_count() == 0
        assert len(graph.get_all_vertices()) == 0
        assert len(graph.get_all_edges()) == 0
    
    def test_graph_info(self):
        """Test getting graph information."""
        graph = GraphManager()
        graph.add_vertex("A", VertexType.INTERSECTION, (0.0, 0.0))
        graph.add_vertex("B", VertexType.INTERSECTION, (1.0, 0.0))
        graph.add_vertex("S1", VertexType.SHELTER, (2.0, 0.0), capacity=100)
        graph.add_vertex("E1", VertexType.EVACUATION_POINT, (3.0, 0.0), capacity=500)
        
        graph.add_edge("A", "B", 1.0, 0.1, 0.2)
        graph.add_edge("B", "S1", 1.0, 0.1, 0.2)
        
        info = graph.get_graph_info()
        assert info["vertex_count"] == 4
        assert info["edge_count"] == 2
        assert info["vertex_types"]["intersection"] == 2
        assert info["vertex_types"]["shelter"] == 1
        assert info["vertex_types"]["evacuation_point"] == 1
        assert info["average_degree"] == 0.5  # 2 edges / 4 vertices
    
    def test_get_vertex_ids(self):
        """Test getting all vertex IDs."""
        graph = GraphManager()
        graph.add_vertex("A", VertexType.INTERSECTION, (0.0, 0.0))
        graph.add_vertex("B", VertexType.INTERSECTION, (1.0, 0.0))
        graph.add_vertex("C", VertexType.SHELTER, (2.0, 0.0))
        
        vertex_ids = graph.get_vertex_ids()
        assert vertex_ids == {"A", "B", "C"}
    
    def test_string_representation(self):
        """Test string representation of graph."""
        graph = GraphManager()
        graph.add_vertex("A", VertexType.INTERSECTION, (0.0, 0.0))
        graph.add_vertex("B", VertexType.INTERSECTION, (1.0, 0.0))
        graph.add_edge("A", "B", 1.0, 0.1, 0.2)
        
        graph_str = str(graph)
        assert "Graph(vertices=2, edges=1)" == graph_str