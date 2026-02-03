"""
Unit tests for core data models.
"""

import pytest
from datetime import datetime
from disaster_evacuation.models import (
    Vertex, VertexType, Edge, DisasterEvent, DisasterType, PathResult, AlgorithmStats
)


class TestVertex:
    """Test cases for Vertex model."""
    
    def test_vertex_creation(self):
        """Test basic vertex creation."""
        vertex = Vertex("A", VertexType.INTERSECTION, (0.0, 0.0))
        assert vertex.id == "A"
        assert vertex.vertex_type == VertexType.INTERSECTION
        assert vertex.coordinates == (0.0, 0.0)
        assert vertex.capacity is None
    
    def test_vertex_with_capacity(self):
        """Test vertex creation with capacity."""
        vertex = Vertex("S1", VertexType.SHELTER, (1.0, 1.0), capacity=100)
        assert vertex.capacity == 100
    
    def test_vertex_validation(self):
        """Test vertex validation."""
        with pytest.raises(ValueError, match="Vertex ID cannot be empty"):
            Vertex("", VertexType.INTERSECTION, (0.0, 0.0))
        
        with pytest.raises(ValueError, match="Coordinates must be a tuple"):
            Vertex("A", VertexType.INTERSECTION, (0.0,))
        
        with pytest.raises(ValueError, match="Capacity cannot be negative"):
            Vertex("A", VertexType.SHELTER, (0.0, 0.0), capacity=-1)
    
    def test_distance_calculation(self):
        """Test distance calculation between vertices."""
        v1 = Vertex("A", VertexType.INTERSECTION, (0.0, 0.0))
        v2 = Vertex("B", VertexType.INTERSECTION, (3.0, 4.0))
        assert v1.distance_to(v2) == 5.0  # 3-4-5 triangle


class TestEdge:
    """Test cases for Edge model."""
    
    def test_edge_creation(self):
        """Test basic edge creation."""
        edge = Edge("A", "B", 1.0, 0.1, 0.2)
        assert edge.source == "A"
        assert edge.target == "B"
        assert edge.base_distance == 1.0
        assert edge.base_risk == 0.1
        assert edge.base_congestion == 0.2
        assert edge.current_weight == 1.3  # 1.0 + 0.1 + 0.2
        assert not edge.is_blocked
    
    def test_edge_validation(self):
        """Test edge validation."""
        with pytest.raises(ValueError, match="Source and target vertex IDs cannot be empty"):
            Edge("", "B", 1.0, 0.1, 0.2)
        
        with pytest.raises(ValueError, match="Self-loops are not allowed"):
            Edge("A", "A", 1.0, 0.1, 0.2)
        
        with pytest.raises(ValueError, match="Base distance cannot be negative"):
            Edge("A", "B", -1.0, 0.1, 0.2)
    
    def test_weight_reset(self):
        """Test edge weight reset functionality."""
        edge = Edge("A", "B", 1.0, 0.1, 0.2)
        edge.current_weight = 5.0
        edge.is_blocked = True
        
        edge.reset_weight()
        assert edge.current_weight == 1.3
        assert not edge.is_blocked


class TestDisasterEvent:
    """Test cases for DisasterEvent model."""
    
    def test_disaster_creation(self):
        """Test basic disaster event creation."""
        disaster = DisasterEvent(
            disaster_type=DisasterType.FLOOD,
            epicenter=(0.0, 0.0),
            severity=0.5,
            max_effect_radius=10.0
        )
        assert disaster.disaster_type == DisasterType.FLOOD
        assert disaster.epicenter == (0.0, 0.0)
        assert disaster.severity == 0.5
        assert disaster.max_effect_radius == 10.0
        assert isinstance(disaster.start_time, datetime)
    
    def test_disaster_validation(self):
        """Test disaster event validation."""
        with pytest.raises(ValueError, match="Severity must be between 0.0 and 1.0"):
            DisasterEvent(DisasterType.FIRE, (0.0, 0.0), 1.5, 10.0)
        
        with pytest.raises(ValueError, match="Max effect radius must be positive"):
            DisasterEvent(DisasterType.EARTHQUAKE, (0.0, 0.0), 0.5, -1.0)
    
    def test_distance_calculation(self):
        """Test distance calculation from epicenter."""
        disaster = DisasterEvent(DisasterType.FLOOD, (0.0, 0.0), 0.5, 10.0)
        assert disaster.distance_to_point((3.0, 4.0)) == 5.0
    
    def test_disaster_multipliers(self):
        """Test disaster type multipliers."""
        flood = DisasterEvent(DisasterType.FLOOD, (0.0, 0.0), 0.5, 10.0)
        fire = DisasterEvent(DisasterType.FIRE, (0.0, 0.0), 0.5, 10.0)
        earthquake = DisasterEvent(DisasterType.EARTHQUAKE, (0.0, 0.0), 0.5, 10.0)
        
        assert flood.get_disaster_multiplier() == 2.0
        assert fire.get_disaster_multiplier() == 3.0
        assert earthquake.get_disaster_multiplier() == 2.5
    
    def test_point_affected(self):
        """Test point affected by disaster."""
        disaster = DisasterEvent(DisasterType.FLOOD, (0.0, 0.0), 0.5, 5.0)
        assert disaster.is_point_affected((3.0, 4.0))  # Distance = 5.0, within radius
        assert not disaster.is_point_affected((4.0, 4.0))  # Distance > 5.0


class TestPathResult:
    """Test cases for PathResult model."""
    
    def test_successful_path_result(self):
        """Test successful path result creation."""
        edges = [Edge("A", "B", 1.0, 0.1, 0.2)]
        result = PathResult(
            path=["A", "B"],
            total_cost=1.3,
            edges_traversed=edges,
            computation_time=0.001,
            nodes_visited=2
        )
        assert result.found
        assert result.path_length == 2
        assert result.edge_count == 1
        assert "A -> B" in result.get_path_summary()
    
    def test_failed_path_result(self):
        """Test failed path result creation."""
        result = PathResult(
            path=[],
            total_cost=0.0,
            edges_traversed=[],
            computation_time=0.001,
            nodes_visited=5,
            found=False,
            error_message="No path exists"
        )
        assert not result.found
        assert result.error_message == "No path exists"
        assert "No path found" in result.get_path_summary()


class TestAlgorithmStats:
    """Test cases for AlgorithmStats model."""
    
    def test_stats_creation(self):
        """Test algorithm stats creation."""
        stats = AlgorithmStats()
        assert stats.nodes_visited == 0
        assert stats.computation_time == 0.0
        assert stats.edges_examined == 0
        assert stats.queue_operations == 0
    
    def test_stats_reset(self):
        """Test stats reset functionality."""
        stats = AlgorithmStats(nodes_visited=10, computation_time=1.5)
        stats.reset()
        assert stats.nodes_visited == 0
        assert stats.computation_time == 0.0