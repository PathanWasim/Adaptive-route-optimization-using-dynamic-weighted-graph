"""
Property-based tests for weight calculation consistency.

Feature: disaster-evacuation-routing, Property 3: Weight Calculation Consistency
**Validates: Requirements 1.4**
"""

import pytest
from hypothesis import given, strategies as st, assume
from disaster_evacuation.graph import WeightCalculator
from disaster_evacuation.models import Edge, DisasterEvent, DisasterType


# Hypothesis strategies for generating test data
distance_strategy = st.floats(min_value=0.0, max_value=1000.0, allow_nan=False, allow_infinity=False)
risk_strategy = st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False)
congestion_strategy = st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False)
coordinate_strategy = st.floats(min_value=-1000.0, max_value=1000.0, allow_nan=False, allow_infinity=False)
coordinates_strategy = st.tuples(coordinate_strategy, coordinate_strategy)
severity_strategy = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
radius_strategy = st.floats(min_value=0.1, max_value=100.0, allow_nan=False, allow_infinity=False)
traffic_multiplier_strategy = st.floats(min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False)
disaster_type_strategy = st.sampled_from(list(DisasterType))


class TestWeightCalculationConsistency:
    """Property-based tests for weight calculation consistency."""
    
    @given(
        distance=distance_strategy,
        risk=risk_strategy,
        congestion=congestion_strategy
    )
    def test_base_weight_calculation_consistency(self, distance, risk, congestion):
        """
        Property 3: Weight Calculation Consistency - Base Weight
        
        For any edge with base distance, risk factor, and congestion factor,
        the calculated base weight should exactly equal distance + risk + congestion.
        
        **Validates: Requirements 1.4**
        """
        edge = Edge("A", "B", distance, risk, congestion)
        
        # Calculate base weight using WeightCalculator
        calculated_weight = WeightCalculator.calculate_base_weight(edge)
        
        # Expected weight is sum of components
        expected_weight = distance + risk + congestion
        
        # Verify exact consistency (within floating point precision)
        assert abs(calculated_weight - expected_weight) < 1e-10
        
        # Verify edge's current_weight is also consistent
        assert abs(edge.current_weight - expected_weight) < 1e-10
    
    @given(
        distance=distance_strategy,
        risk=risk_strategy,
        congestion=congestion_strategy,
        disaster_type=disaster_type_strategy,
        epicenter=coordinates_strategy,
        severity=severity_strategy,
        radius=radius_strategy,
        edge_midpoint=coordinates_strategy
    )
    def test_dynamic_weight_calculation_consistency(self, distance, risk, congestion,
                                                  disaster_type, epicenter, severity,
                                                  radius, edge_midpoint):
        """
        Property 3: Weight Calculation Consistency - Dynamic Weight
        
        For any edge and disaster event, the calculated dynamic weight should
        exactly equal distance + risk_penalty + congestion_penalty according
        to the specified formula.
        
        **Validates: Requirements 1.4**
        """
        edge = Edge("A", "B", distance, risk, congestion)
        disaster = DisasterEvent(disaster_type, epicenter, severity, radius)
        
        # Calculate dynamic weight
        dynamic_weight = WeightCalculator.calculate_dynamic_weight(
            edge, disaster, edge_midpoint
        )
        
        # Calculate components separately
        risk_penalty = WeightCalculator.calculate_risk_penalty(
            edge, disaster, edge_midpoint
        )
        congestion_penalty = WeightCalculator.calculate_congestion_penalty(edge)
        
        # Expected weight is sum of components
        expected_weight = distance + risk_penalty + congestion_penalty
        
        # Verify exact consistency
        assert abs(dynamic_weight - expected_weight) < 1e-10
    
    @given(
        distance=distance_strategy,
        risk=risk_strategy,
        congestion=congestion_strategy,
        traffic_multiplier=traffic_multiplier_strategy
    )
    def test_congestion_penalty_consistency(self, distance, risk, congestion, traffic_multiplier):
        """
        Property 3: Weight Calculation Consistency - Congestion Penalty
        
        For any edge and traffic multiplier, the congestion penalty should
        exactly equal base_congestion * traffic_multiplier.
        
        **Validates: Requirements 1.4**
        """
        edge = Edge("A", "B", distance, risk, congestion)
        
        # Calculate congestion penalty
        penalty = WeightCalculator.calculate_congestion_penalty(edge, traffic_multiplier)
        
        # Expected penalty
        expected_penalty = congestion * traffic_multiplier
        
        # Verify exact consistency
        assert abs(penalty - expected_penalty) < 1e-10
    
    @given(
        distance=distance_strategy,
        risk=risk_strategy,
        congestion=congestion_strategy,
        disaster_type=disaster_type_strategy,
        epicenter=coordinates_strategy,
        severity=severity_strategy,
        radius=radius_strategy,
        edge_midpoint=coordinates_strategy
    )
    def test_risk_penalty_formula_consistency(self, distance, risk, congestion,
                                            disaster_type, epicenter, severity,
                                            radius, edge_midpoint):
        """
        Property 3: Weight Calculation Consistency - Risk Penalty Formula
        
        For any edge and disaster event, the risk penalty should follow the exact formula:
        risk_penalty = base_risk * disaster_multiplier * proximity_factor * severity
        
        **Validates: Requirements 1.4**
        """
        edge = Edge("A", "B", distance, risk, congestion)
        disaster = DisasterEvent(disaster_type, epicenter, severity, radius)
        
        # Calculate risk penalty using WeightCalculator
        calculated_penalty = WeightCalculator.calculate_risk_penalty(
            edge, disaster, edge_midpoint
        )
        
        # Calculate expected penalty using the formula
        distance_to_epicenter = disaster.distance_to_point(edge_midpoint)
        proximity_factor = max(0.0, 1.0 - distance_to_epicenter / radius)
        disaster_multiplier = disaster.get_disaster_multiplier()
        
        expected_penalty = risk * disaster_multiplier * proximity_factor * severity
        
        # Verify exact consistency
        assert abs(calculated_penalty - expected_penalty) < 1e-10
    
    @given(
        source_coords=coordinates_strategy,
        target_coords=coordinates_strategy
    )
    def test_edge_midpoint_calculation_consistency(self, source_coords, target_coords):
        """
        Property 3: Weight Calculation Consistency - Edge Midpoint
        
        For any source and target coordinates, the calculated midpoint should
        exactly equal ((x1+x2)/2, (y1+y2)/2).
        
        **Validates: Requirements 1.4**
        """
        # Calculate midpoint using WeightCalculator
        calculated_midpoint = WeightCalculator.calculate_edge_midpoint(
            source_coords, target_coords
        )
        
        # Calculate expected midpoint
        x1, y1 = source_coords
        x2, y2 = target_coords
        expected_midpoint = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
        
        # Verify exact consistency
        assert abs(calculated_midpoint[0] - expected_midpoint[0]) < 1e-10
        assert abs(calculated_midpoint[1] - expected_midpoint[1]) < 1e-10
    
    @given(
        distance=distance_strategy,
        risk=risk_strategy,
        congestion=congestion_strategy,
        traffic_multiplier=traffic_multiplier_strategy
    )
    def test_dynamic_weight_without_disaster_consistency(self, distance, risk, congestion,
                                                       traffic_multiplier):
        """
        Property 3: Weight Calculation Consistency - No Disaster Case
        
        For any edge without disaster effects, the dynamic weight should equal
        distance + base_risk + (base_congestion * traffic_multiplier).
        
        **Validates: Requirements 1.4**
        """
        edge = Edge("A", "B", distance, risk, congestion)
        
        # Calculate dynamic weight without disaster
        dynamic_weight = WeightCalculator.calculate_dynamic_weight(
            edge, traffic_multiplier=traffic_multiplier
        )
        
        # Expected weight without disaster effects
        expected_weight = distance + risk + (congestion * traffic_multiplier)
        
        # Verify exact consistency
        assert abs(dynamic_weight - expected_weight) < 1e-10
    
    @given(
        distance=distance_strategy,
        risk=risk_strategy,
        congestion=congestion_strategy,
        disaster_type=disaster_type_strategy,
        epicenter=coordinates_strategy,
        severity=severity_strategy,
        radius=radius_strategy
    )
    def test_weight_calculation_components_sum_consistency(self, distance, risk, congestion,
                                                         disaster_type, epicenter,
                                                         severity, radius):
        """
        Property 3: Weight Calculation Consistency - Component Sum
        
        For any edge and disaster, the dynamic weight should always equal the sum
        of its individual components, regardless of the specific values.
        
        **Validates: Requirements 1.4**
        """
        edge = Edge("A", "B", distance, risk, congestion)
        disaster = DisasterEvent(disaster_type, epicenter, severity, radius)
        
        # Generate a random edge midpoint within reasonable bounds
        edge_midpoint = (epicenter[0] + radius * 0.5, epicenter[1])
        
        # Calculate total dynamic weight
        total_weight = WeightCalculator.calculate_dynamic_weight(
            edge, disaster, edge_midpoint
        )
        
        # Calculate individual components
        base_distance = distance
        risk_penalty = WeightCalculator.calculate_risk_penalty(
            edge, disaster, edge_midpoint
        )
        congestion_penalty = WeightCalculator.calculate_congestion_penalty(edge)
        
        # Sum of components
        component_sum = base_distance + risk_penalty + congestion_penalty
        
        # Verify consistency
        assert abs(total_weight - component_sum) < 1e-10
    
    @given(
        distance=distance_strategy,
        risk=risk_strategy,
        congestion=congestion_strategy,
        disaster_type=disaster_type_strategy,
        severity=severity_strategy,
        radius=radius_strategy
    )
    def test_disaster_multiplier_consistency(self, distance, risk, congestion,
                                           disaster_type, severity, radius):
        """
        Property 3: Weight Calculation Consistency - Disaster Multipliers
        
        For any disaster type, the disaster multiplier should be consistent
        with the expected values: Flood=2.0, Fire=3.0, Earthquake=2.5.
        
        **Validates: Requirements 1.4**
        """
        disaster = DisasterEvent(disaster_type, (0.0, 0.0), severity, radius)
        
        multiplier = disaster.get_disaster_multiplier()
        
        # Verify expected multipliers
        if disaster_type == DisasterType.FLOOD:
            assert multiplier == 2.0
        elif disaster_type == DisasterType.FIRE:
            assert multiplier == 3.0
        elif disaster_type == DisasterType.EARTHQUAKE:
            assert multiplier == 2.5
        else:
            pytest.fail(f"Unknown disaster type: {disaster_type}")
    
    @given(
        distance=distance_strategy,
        risk=risk_strategy,
        congestion=congestion_strategy,
        epicenter=coordinates_strategy,
        radius=radius_strategy
    )
    def test_proximity_factor_consistency(self, distance, risk, congestion,
                                        epicenter, radius):
        """
        Property 3: Weight Calculation Consistency - Proximity Factor
        
        For any point and disaster epicenter, the proximity factor should follow
        the formula: max(0, 1 - distance_to_epicenter / max_effect_radius).
        
        **Validates: Requirements 1.4**
        """
        edge = Edge("A", "B", distance, risk, congestion)
        disaster = DisasterEvent(DisasterType.FLOOD, epicenter, 0.5, radius)
        
        # Test various points at different distances
        test_points = [
            epicenter,  # At epicenter
            (epicenter[0] + radius * 0.5, epicenter[1]),  # Half radius
            (epicenter[0] + radius, epicenter[1]),  # At radius boundary
            (epicenter[0] + radius * 2, epicenter[1])  # Beyond radius
        ]
        
        for point in test_points:
            # Calculate risk penalty (which uses proximity factor internally)
            risk_penalty = WeightCalculator.calculate_risk_penalty(edge, disaster, point)
            
            # Calculate expected proximity factor
            distance_to_epicenter = disaster.distance_to_point(point)
            expected_proximity = max(0.0, 1.0 - distance_to_epicenter / radius)
            
            # Calculate expected risk penalty
            expected_penalty = risk * disaster.get_disaster_multiplier() * expected_proximity * disaster.severity
            
            # Verify consistency
            assert abs(risk_penalty - expected_penalty) < 1e-10