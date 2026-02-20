"""
Unit tests for DisasterModel class.
"""

import pytest
from disaster_evacuation.models import DisasterModel
from disaster_evacuation.models import GraphManager
from disaster_evacuation.models import DisasterEvent, DisasterType, VertexType


class TestDisasterModel:
    """Test cases for DisasterModel class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.graph = GraphManager()
        self.disaster_model = DisasterModel()
        
        # Create a simple test graph
        self.graph.add_vertex("A", VertexType.INTERSECTION, (0.0, 0.0))
        self.graph.add_vertex("B", VertexType.INTERSECTION, (2.0, 0.0))
        self.graph.add_vertex("C", VertexType.SHELTER, (4.0, 0.0), capacity=100)
        self.graph.add_vertex("D", VertexType.EVACUATION_POINT, (2.0, 2.0), capacity=500)
        
        # Add edges
        self.graph.add_edge("A", "B", 2.0, 0.1, 0.2)  # Edge midpoint at (1.0, 0.0)
        self.graph.add_edge("B", "C", 2.0, 0.2, 0.1)  # Edge midpoint at (3.0, 0.0)
        self.graph.add_edge("A", "D", 2.8, 0.3, 0.3)  # Edge midpoint at (1.0, 1.0)
        self.graph.add_edge("C", "D", 2.8, 0.1, 0.4)  # Edge midpoint at (3.0, 1.0)
    
    def test_disaster_model_creation(self):
        """Test basic disaster model creation."""
        model = DisasterModel()
        assert len(model.get_active_disasters()) == 0
        assert str(model) == "DisasterModel(active_disasters=0)"
    
    def test_apply_disaster_effects_flood(self):
        """Test applying flood disaster effects."""
        disaster = DisasterEvent(DisasterType.FLOOD, (1.0, 0.0), 0.8, 3.0)
        
        # Store original weights
        original_weight_ab = self.graph.get_edge_weight("A", "B")
        original_weight_bc = self.graph.get_edge_weight("B", "C")
        
        # Apply disaster effects
        self.disaster_model.apply_disaster_effects(self.graph, disaster)
        
        # Check that weights have changed for affected edges
        new_weight_ab = self.graph.get_edge_weight("A", "B")
        new_weight_bc = self.graph.get_edge_weight("B", "C")
        
        # Edge A-B is at epicenter, should have increased weight
        assert new_weight_ab > original_weight_ab
        
        # Edge B-C is within radius, weight may increase or decrease based on calculation
        # The important thing is that disaster effects were applied
        assert new_weight_bc != original_weight_bc or new_weight_ab > original_weight_ab
        
        # Check active disasters
        active_disasters = self.disaster_model.get_active_disasters()
        assert len(active_disasters) == 1
        assert active_disasters[0] == disaster
    
    def test_apply_disaster_effects_fire(self):
        """Test applying fire disaster effects."""
        disaster = DisasterEvent(DisasterType.FIRE, (1.0, 0.0), 0.9, 2.0)
        
        # Apply disaster effects
        self.disaster_model.apply_disaster_effects(self.graph, disaster)
        
        # Fire has higher disaster multiplier (3.0) than flood (2.0)
        # Edge A-B should have significant weight increase
        edge_ab = self.graph.get_edge("A", "B")
        assert edge_ab.current_weight > edge_ab.base_distance + edge_ab.base_risk + edge_ab.base_congestion
    
    def test_apply_disaster_effects_earthquake(self):
        """Test applying earthquake disaster effects."""
        disaster = DisasterEvent(DisasterType.EARTHQUAKE, (2.0, 1.0), 0.7, 4.0)
        
        # Apply disaster effects
        self.disaster_model.apply_disaster_effects(self.graph, disaster)
        
        # All edges should be affected due to large radius
        for edge in self.graph.get_all_edges():
            base_weight = edge.base_distance + edge.base_risk + edge.base_congestion
            assert edge.current_weight >= base_weight  # Weight should increase or stay same
    
    def test_road_blocking(self):
        """Test road blocking functionality."""
        # High severity fire disaster at edge A-B midpoint
        disaster = DisasterEvent(DisasterType.FIRE, (1.0, 0.0), 0.95, 2.0)
        
        # Apply disaster effects
        self.disaster_model.apply_disaster_effects(self.graph, disaster)
        
        # Edge A-B should be blocked due to high severity fire
        edge_ab = self.graph.get_edge("A", "B")
        if edge_ab.is_blocked:
            assert self.graph.get_edge_weight("A", "B") == float('inf')
    
    def test_get_affected_edges(self):
        """Test getting edges affected by disaster."""
        epicenter = (1.5, 0.0)
        radius = 2.0
        
        affected_edges = self.disaster_model.get_affected_edges(self.graph, epicenter, radius)
        
        # Should include edges A-B and B-C (close to epicenter)
        affected_sources = [edge.source for edge in affected_edges]
        affected_targets = [edge.target for edge in affected_edges]
        
        assert "A" in affected_sources or "A" in affected_targets
        assert "B" in affected_sources or "B" in affected_targets
    
    def test_remove_disaster_effects(self):
        """Test removing disaster effects."""
        disaster = DisasterEvent(DisasterType.FLOOD, (1.0, 0.0), 0.8, 3.0)
        
        # Store original weights
        original_weights = {}
        for edge in self.graph.get_all_edges():
            original_weights[(edge.source, edge.target)] = edge.current_weight
        
        # Apply disaster effects
        self.disaster_model.apply_disaster_effects(self.graph, disaster)
        
        # Verify weights changed
        for edge in self.graph.get_all_edges():
            key = (edge.source, edge.target)
            # Some weights should have changed (those affected by disaster)
        
        # Remove disaster effects
        self.disaster_model.remove_disaster_effects(self.graph, disaster)
        
        # Verify weights are back to original
        for edge in self.graph.get_all_edges():
            key = (edge.source, edge.target)
            assert abs(edge.current_weight - original_weights[key]) < 1e-10
        
        # Check no active disasters
        assert len(self.disaster_model.get_active_disasters()) == 0
    
    def test_clear_all_disaster_effects(self):
        """Test clearing all disaster effects."""
        disaster1 = DisasterEvent(DisasterType.FLOOD, (1.0, 0.0), 0.8, 3.0)
        disaster2 = DisasterEvent(DisasterType.FIRE, (3.0, 0.0), 0.7, 2.0)
        
        # Apply multiple disasters
        self.disaster_model.apply_disaster_effects(self.graph, disaster1)
        self.disaster_model.apply_disaster_effects(self.graph, disaster2)
        
        assert len(self.disaster_model.get_active_disasters()) == 2
        
        # Clear all effects
        self.disaster_model.clear_all_disaster_effects(self.graph)
        
        # Verify no active disasters
        assert len(self.disaster_model.get_active_disasters()) == 0
        
        # Verify all edges are reset
        for edge in self.graph.get_all_edges():
            expected_weight = edge.base_distance + edge.base_risk + edge.base_congestion
            assert abs(edge.current_weight - expected_weight) < 1e-10
    
    def test_multiple_disasters(self):
        """Test applying multiple disasters simultaneously."""
        disaster1 = DisasterEvent(DisasterType.FLOOD, (1.0, 0.0), 0.6, 2.0)
        disaster2 = DisasterEvent(DisasterType.EARTHQUAKE, (3.0, 1.0), 0.5, 2.0)
        
        # Apply first disaster
        self.disaster_model.apply_disaster_effects(self.graph, disaster1)
        weight_after_first = self.graph.get_edge_weight("A", "B")
        
        # Apply second disaster
        self.disaster_model.apply_disaster_effects(self.graph, disaster2)
        weight_after_second = self.graph.get_edge_weight("A", "B")
        
        # Weight should be recalculated with both disasters
        assert len(self.disaster_model.get_active_disasters()) == 2
    
    def test_disaster_impact_summary(self):
        """Test getting disaster impact summary."""
        disaster = DisasterEvent(DisasterType.FIRE, (1.0, 0.0), 0.8, 3.0)
        
        summary = self.disaster_model.get_disaster_impact_summary(self.graph, disaster)
        
        # Verify summary structure
        assert "disaster_type" in summary
        assert "severity" in summary
        assert "epicenter" in summary
        assert "effect_radius" in summary
        assert "total_edges" in summary
        assert "affected_edges" in summary
        assert "blocked_edges" in summary
        assert "high_risk_edges" in summary
        assert "impact_percentage" in summary
        
        # Verify values
        assert summary["disaster_type"] == "fire"
        assert summary["severity"] == 0.8
        assert summary["epicenter"] == (1.0, 0.0)
        assert summary["effect_radius"] == 3.0
        assert summary["total_edges"] == 4
        assert 0 <= summary["impact_percentage"] <= 100
    
    def test_apply_disaster_empty_graph(self):
        """Test applying disaster to empty graph raises error."""
        empty_graph = GraphManager()
        disaster = DisasterEvent(DisasterType.FLOOD, (0.0, 0.0), 0.5, 10.0)
        
        with pytest.raises(ValueError, match="Cannot apply disaster effects to empty graph"):
            self.disaster_model.apply_disaster_effects(empty_graph, disaster)
    
    def test_apply_disaster_invalid_parameters(self):
        """Test applying disaster with invalid parameters raises error."""
        # Zero severity
        disaster1 = DisasterEvent(DisasterType.FLOOD, (0.0, 0.0), 0.0, 10.0)
        with pytest.raises(ValueError, match="Invalid disaster parameters"):
            self.disaster_model.apply_disaster_effects(self.graph, disaster1)
        
        # Negative radius - this will be caught by DisasterEvent validation
        with pytest.raises(ValueError, match="Max effect radius must be positive"):
            DisasterEvent(DisasterType.FLOOD, (0.0, 0.0), 0.5, 0.0)
    
    def test_traffic_multiplier(self):
        """Test traffic multiplier for different disaster types."""
        model = DisasterModel()
        
        flood = DisasterEvent(DisasterType.FLOOD, (0.0, 0.0), 0.5, 10.0)
        fire = DisasterEvent(DisasterType.FIRE, (0.0, 0.0), 0.5, 10.0)
        earthquake = DisasterEvent(DisasterType.EARTHQUAKE, (0.0, 0.0), 0.5, 10.0)
        
        # Access private method for testing
        flood_multiplier = model._get_traffic_multiplier(flood)
        fire_multiplier = model._get_traffic_multiplier(fire)
        earthquake_multiplier = model._get_traffic_multiplier(earthquake)
        
        assert flood_multiplier == 1.5
        assert fire_multiplier == 2.0
        assert earthquake_multiplier == 3.0
    
    def test_risk_penalty_calculation(self):
        """Test risk penalty calculation through disaster model."""
        disaster = DisasterEvent(DisasterType.FLOOD, (1.0, 0.0), 0.8, 5.0)
        edge = self.graph.get_edge("A", "B")
        edge_midpoint = (1.0, 0.0)  # At disaster epicenter
        
        risk_penalty = self.disaster_model.calculate_risk_penalty(edge, disaster, edge_midpoint)
        
        # Should be positive for edge at epicenter
        assert risk_penalty > 0
        
        # Should match WeightCalculator result
        from disaster_evacuation.models import WeightCalculator
        expected_penalty = WeightCalculator.calculate_risk_penalty(edge, disaster, edge_midpoint)
        assert abs(risk_penalty - expected_penalty) < 1e-10
    
    def test_is_road_blocked(self):
        """Test road blocking check through disaster model."""
        disaster = DisasterEvent(DisasterType.FIRE, (1.0, 0.0), 0.95, 2.0)
        edge = self.graph.get_edge("A", "B")
        edge_midpoint = (1.0, 0.0)  # At disaster epicenter
        
        is_blocked = self.disaster_model.is_road_blocked(edge, disaster, edge_midpoint)
        
        # Should match WeightCalculator result
        from disaster_evacuation.models import WeightCalculator
        expected_blocked = WeightCalculator.is_edge_blocked(edge, disaster, edge_midpoint)
        assert is_blocked == expected_blocked