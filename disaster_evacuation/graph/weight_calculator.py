"""
Weight Calculator for the disaster evacuation routing system.

This module implements the WeightCalculator class that computes dynamic edge weights
using the formula: weight = distance + risk_penalty + congestion_penalty
"""

from typing import Tuple
from ..models import Edge, DisasterEvent


class WeightCalculator:
    """
    Calculates dynamic edge weights based on base properties and disaster effects.
    
    Uses the formula: weight = base_distance + risk_penalty + congestion_penalty
    where risk_penalty and congestion_penalty are modified by disaster conditions.
    """
    
    @staticmethod
    def calculate_base_weight(edge: Edge) -> float:
        """
        Calculate the base weight of an edge without disaster effects.
        
        Args:
            edge: Edge to calculate weight for
            
        Returns:
            Base weight = distance + base_risk + base_congestion
        """
        return edge.base_distance + edge.base_risk + edge.base_congestion
    
    @staticmethod
    def calculate_risk_penalty(edge: Edge, disaster: DisasterEvent, 
                             edge_midpoint: Tuple[float, float]) -> float:
        """
        Calculate the risk penalty for an edge based on disaster effects.
        
        Args:
            edge: Edge to calculate penalty for
            disaster: Disaster event affecting the area
            edge_midpoint: Midpoint coordinates of the edge
            
        Returns:
            Risk penalty value
        """
        # Calculate distance from disaster epicenter to edge midpoint
        distance_to_epicenter = disaster.distance_to_point(edge_midpoint)
        
        # Calculate proximity factor (1.0 at epicenter, 0.0 at max radius)
        proximity_factor = max(0.0, 1.0 - distance_to_epicenter / disaster.max_effect_radius)
        
        # Get disaster-specific multiplier
        disaster_multiplier = disaster.get_disaster_multiplier()
        
        # Calculate risk penalty
        risk_penalty = edge.base_risk * disaster_multiplier * proximity_factor * disaster.severity
        
        return risk_penalty
    
    @staticmethod
    def calculate_congestion_penalty(edge: Edge, traffic_multiplier: float = 1.0) -> float:
        """
        Calculate the congestion penalty for an edge.
        
        Args:
            edge: Edge to calculate penalty for
            traffic_multiplier: Multiplier for traffic conditions (default 1.0)
            
        Returns:
            Congestion penalty value
        """
        return edge.base_congestion * traffic_multiplier
    
    @staticmethod
    def calculate_dynamic_weight(edge: Edge, disaster: DisasterEvent = None, 
                               edge_midpoint: Tuple[float, float] = None,
                               traffic_multiplier: float = 1.0) -> float:
        """
        Calculate the dynamic weight of an edge considering all factors.
        
        Args:
            edge: Edge to calculate weight for
            disaster: Optional disaster event affecting the area
            edge_midpoint: Midpoint coordinates of the edge (required if disaster provided)
            traffic_multiplier: Multiplier for traffic conditions
            
        Returns:
            Dynamic weight = distance + risk_penalty + congestion_penalty
            
        Raises:
            ValueError: If disaster is provided but edge_midpoint is not
        """
        # Start with base distance
        weight = edge.base_distance
        
        # Add risk penalty
        if disaster is not None:
            if edge_midpoint is None:
                raise ValueError("Edge midpoint required when disaster is provided")
            risk_penalty = WeightCalculator.calculate_risk_penalty(edge, disaster, edge_midpoint)
            weight += risk_penalty
        else:
            weight += edge.base_risk
        
        # Add congestion penalty
        congestion_penalty = WeightCalculator.calculate_congestion_penalty(edge, traffic_multiplier)
        weight += congestion_penalty
        
        return weight
    
    @staticmethod
    def calculate_edge_midpoint(source_coords: Tuple[float, float], 
                              target_coords: Tuple[float, float]) -> Tuple[float, float]:
        """
        Calculate the midpoint of an edge given source and target coordinates.
        
        Args:
            source_coords: Coordinates of source vertex
            target_coords: Coordinates of target vertex
            
        Returns:
            Midpoint coordinates (x, y)
        """
        x1, y1 = source_coords
        x2, y2 = target_coords
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
    
    @staticmethod
    def is_edge_blocked(edge: Edge, disaster: DisasterEvent, 
                       edge_midpoint: Tuple[float, float],
                       blocking_threshold: float = 0.8) -> bool:
        """
        Determine if an edge should be blocked based on disaster severity.
        
        Args:
            edge: Edge to check
            disaster: Disaster event
            edge_midpoint: Midpoint coordinates of the edge
            blocking_threshold: Severity threshold for blocking (0.0-1.0)
            
        Returns:
            True if edge should be blocked, False otherwise
        """
        if not disaster.is_point_affected(edge_midpoint):
            return False
        
        # Calculate effective severity at edge location
        distance_to_epicenter = disaster.distance_to_point(edge_midpoint)
        proximity_factor = max(0.0, 1.0 - distance_to_epicenter / disaster.max_effect_radius)
        effective_severity = disaster.severity * proximity_factor
        
        # Different disaster types have different blocking behaviors
        if disaster.disaster_type.value == "fire":
            # Fire blocks roads more aggressively
            return effective_severity > (blocking_threshold * 0.6)
        elif disaster.disaster_type.value == "earthquake":
            # Earthquake blocks based on infrastructure damage
            return effective_severity > (blocking_threshold * 0.7)
        elif disaster.disaster_type.value == "flood":
            # Flood blocks low-lying areas
            return effective_severity > blocking_threshold
        
        return False