"""
Unit tests for WeightCalculator class.
"""

import pytest
from disaster_evacuation.graph import WeightCalculator
from disaster_evacuation.models import Edge, DisasterEvent, DisasterType
from datetime import datetime


class TestWeightCalculator:
    """Test cases for WeightCalculator class."""
    
    def test_calculate_base_weight(self):
        """Test basic weight calculation without disaster effects."""
        edge = Edge("A", "B", 1.0, 0.1, 0.2)
        weight = WeightCalculator.calculate_base_weight(edge)
        assert weight == 1.3  # 1.0 + 0.1 + 0.2
    
    def test_calculate_risk_penalty_flood(self):
        """Test risk penalty calculation for flood disaster."""
        edge = Edge("A", "B", 1.0, 0.5, 0.2)
        disaster = DisasterEvent(DisasterType.FLOOD, (0.0, 0.0), 0.8, 10.0)
        edge_midpoint = (2.0, 0.0)  # Distance = 2.0 from epicenter
        
        risk_penalty = WeightCalculator.calculate_risk_penalty(edge, disaster, edge_midpoint)
        
        # Expected calculation:
        # proximity_factor = max(0, 1 - 2.0/10.0) = 0.8
        # disaster_multiplier = 2.0 (flood)
        # risk_penalty = 0.5 * 2.0 * 0.8 * 0.8 = 0.64
        expected = 0.5 * 2.0 * 0.8 * 0.8
        assert abs(risk_penalty - expected) < 1e-10
    
    def test_calculate_risk_penalty_fire(self):
        """Test risk penalty calculation for fire disaster."""
        edge = Edge("A", "B", 1.0, 0.3, 0.2)
        disaster = DisasterEvent(DisasterType.FIRE, (0.0, 0.0), 0.6, 5.0)
        edge_midpoint = (1.0, 0.0)  # Distance = 1.0 from epicenter
        
        risk_penalty = WeightCalculator.calculate_risk_penalty(edge, disaster, edge_midpoint)
        
        # Expected calculation:
        # proximity_factor = max(0, 1 - 1.0/5.0) = 0.8
        # disaster_multiplier = 3.0 (fire)
        # risk_penalty = 0.3 * 3.0 * 0.8 * 0.6 = 0.432
        expected = 0.3 * 3.0 * 0.8 * 0.6
        assert abs(risk_penalty - expected) < 1e-10
    
    def test_calculate_risk_penalty_earthquake(self):
        """Test risk penalty calculation for earthquake disaster."""
        edge = Edge("A", "B", 1.0, 0.4, 0.2)
        disaster = DisasterEvent(DisasterType.EARTHQUAKE, (0.0, 0.0), 0.9, 8.0)
        edge_midpoint = (3.0, 4.0)  # Distance = 5.0 from epicenter
        
        risk_penalty = WeightCalculator.calculate_risk_penalty(edge, disaster, edge_midpoint)
        
        # Expected calculation:
        # proximity_factor = max(0, 1 - 5.0/8.0) = 0.375
        # disaster_multiplier = 2.5 (earthquake)
        # risk_penalty = 0.4 * 2.5 * 0.375 * 0.9 = 0.3375
        expected = 0.4 * 2.5 * 0.375 * 0.9
        assert abs(risk_penalty - expected) < 1e-10
    
    def test_calculate_risk_penalty_outside_radius(self):
        """Test risk penalty when edge is outside disaster radius."""
        edge = Edge("A", "B", 1.0, 0.5, 0.2)
        disaster = DisasterEvent(DisasterType.FLOOD, (0.0, 0.0), 0.8, 5.0)
        edge_midpoint = (10.0, 0.0)  # Distance = 10.0, outside radius of 5.0
        
        risk_penalty = WeightCalculator.calculate_risk_penalty(edge, disaster, edge_midpoint)
        
        # proximity_factor = max(0, 1 - 10.0/5.0) = 0
        assert risk_penalty == 0.0
    
    def test_calculate_congestion_penalty(self):
        """Test congestion penalty calculation."""
        edge = Edge("A", "B", 1.0, 0.1, 0.3)
        
        # Default traffic multiplier
        penalty = WeightCalculator.calculate_congestion_penalty(edge)
        assert penalty == 0.3
        
        # Custom traffic multiplier
        penalty = WeightCalculator.calculate_congestion_penalty(edge, 2.5)
        assert penalty == 0.75  # 0.3 * 2.5
    
    def test_calculate_dynamic_weight_no_disaster(self):
        """Test dynamic weight calculation without disaster."""
        edge = Edge("A", "B", 1.0, 0.1, 0.2)
        
        weight = WeightCalculator.calculate_dynamic_weight(edge)
        assert weight == 1.3  # 1.0 + 0.1 + 0.2
        
        # With traffic multiplier
        weight = WeightCalculator.calculate_dynamic_weight(edge, traffic_multiplier=2.0)
        assert weight == 1.5  # 1.0 + 0.1 + (0.2 * 2.0)
    
    def test_calculate_dynamic_weight_with_disaster(self):
        """Test dynamic weight calculation with disaster effects."""
        edge = Edge("A", "B", 1.0, 0.2, 0.1)
        disaster = DisasterEvent(DisasterType.FLOOD, (0.0, 0.0), 0.5, 10.0)
        edge_midpoint = (5.0, 0.0)  # Distance = 5.0 from epicenter
        
        weight = WeightCalculator.calculate_dynamic_weight(edge, disaster, edge_midpoint)
        
        # Expected calculation:
        # base_distance = 1.0
        # proximity_factor = max(0, 1 - 5.0/10.0) = 0.5
        # risk_penalty = 0.2 * 2.0 * 0.5 * 0.5 = 0.1
        # congestion_penalty = 0.1 * 1.0 = 0.1
        # total = 1.0 + 0.1 + 0.1 = 1.2
        expected = 1.0 + (0.2 * 2.0 * 0.5 * 0.5) + 0.1
        assert abs(weight - expected) < 1e-10
    
    def test_calculate_dynamic_weight_missing_midpoint(self):
        """Test dynamic weight calculation with disaster but missing midpoint."""
        edge = Edge("A", "B", 1.0, 0.1, 0.2)
        disaster = DisasterEvent(DisasterType.FIRE, (0.0, 0.0), 0.5, 10.0)
        
        with pytest.raises(ValueError, match="Edge midpoint required when disaster is provided"):
            WeightCalculator.calculate_dynamic_weight(edge, disaster)
    
    def test_calculate_edge_midpoint(self):
        """Test edge midpoint calculation."""
        source_coords = (0.0, 0.0)
        target_coords = (4.0, 6.0)
        
        midpoint = WeightCalculator.calculate_edge_midpoint(source_coords, target_coords)
        assert midpoint == (2.0, 3.0)
        
        # Test with negative coordinates
        source_coords = (-2.0, -4.0)
        target_coords = (2.0, 4.0)
        
        midpoint = WeightCalculator.calculate_edge_midpoint(source_coords, target_coords)
        assert midpoint == (0.0, 0.0)
    
    def test_is_edge_blocked_fire(self):
        """Test edge blocking for fire disaster."""
        edge = Edge("A", "B", 1.0, 0.1, 0.2)
        disaster = DisasterEvent(DisasterType.FIRE, (0.0, 0.0), 0.9, 10.0)
        
        # Edge close to epicenter - should be blocked
        edge_midpoint = (1.0, 0.0)  # Distance = 1.0, high effective severity
        assert WeightCalculator.is_edge_blocked(edge, disaster, edge_midpoint)
        
        # Edge far from epicenter - should not be blocked
        edge_midpoint = (9.0, 0.0)  # Distance = 9.0, low effective severity
        assert not WeightCalculator.is_edge_blocked(edge, disaster, edge_midpoint)
        
        # Edge outside disaster radius - should not be blocked
        edge_midpoint = (15.0, 0.0)  # Distance = 15.0, outside radius
        assert not WeightCalculator.is_edge_blocked(edge, disaster, edge_midpoint)
    
    def test_is_edge_blocked_earthquake(self):
        """Test edge blocking for earthquake disaster."""
        edge = Edge("A", "B", 1.0, 0.1, 0.2)
        disaster = DisasterEvent(DisasterType.EARTHQUAKE, (0.0, 0.0), 0.8, 10.0)
        
        # Edge close to epicenter with high severity
        edge_midpoint = (2.0, 0.0)  # Distance = 2.0
        # effective_severity = 0.8 * (1 - 2.0/10.0) = 0.8 * 0.8 = 0.64
        # blocking_threshold * 0.7 = 0.8 * 0.7 = 0.56
        # 0.64 > 0.56, so should be blocked
        assert WeightCalculator.is_edge_blocked(edge, disaster, edge_midpoint)
    
    def test_is_edge_blocked_flood(self):
        """Test edge blocking for flood disaster."""
        edge = Edge("A", "B", 1.0, 0.1, 0.2)
        disaster = DisasterEvent(DisasterType.FLOOD, (0.0, 0.0), 0.9, 10.0)
        
        # Edge close to epicenter with high severity
        edge_midpoint = (1.0, 0.0)  # Distance = 1.0
        # effective_severity = 0.9 * (1 - 1.0/10.0) = 0.9 * 0.9 = 0.81
        # blocking_threshold = 0.8 (default)
        # 0.81 > 0.8, so should be blocked
        assert WeightCalculator.is_edge_blocked(edge, disaster, edge_midpoint)
        
        # Edge with lower effective severity
        edge_midpoint = (5.0, 0.0)  # Distance = 5.0
        # effective_severity = 0.9 * (1 - 5.0/10.0) = 0.9 * 0.5 = 0.45
        # 0.45 < 0.8, so should not be blocked
        assert not WeightCalculator.is_edge_blocked(edge, disaster, edge_midpoint)
    
    def test_is_edge_blocked_custom_threshold(self):
        """Test edge blocking with custom threshold."""
        edge = Edge("A", "B", 1.0, 0.1, 0.2)
        disaster = DisasterEvent(DisasterType.FLOOD, (0.0, 0.0), 0.6, 10.0)
        edge_midpoint = (5.0, 0.0)  # Distance = 5.0
        
        # effective_severity = 0.6 * (1 - 5.0/10.0) = 0.6 * 0.5 = 0.3
        
        # With high threshold - should not be blocked
        assert not WeightCalculator.is_edge_blocked(edge, disaster, edge_midpoint, 0.5)
        
        # With low threshold - should be blocked
        assert WeightCalculator.is_edge_blocked(edge, disaster, edge_midpoint, 0.2)