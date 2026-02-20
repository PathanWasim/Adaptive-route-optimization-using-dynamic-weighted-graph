"""
Disaster Model for the disaster evacuation routing system.

This module implements the DisasterModel class that applies disaster-specific
transformations to edge weights based on disaster type, location, and severity.
"""

from typing import List, Tuple
from ..models import DisasterEvent, DisasterType
from ..models import GraphManager, WeightCalculator


class DisasterModel:
    """
    Applies disaster-specific effects to graph edge weights.
    
    The DisasterModel modifies edge weights based on disaster type, proximity to epicenter,
    and severity. Different disaster types have different effects on road networks:
    - Flood: Affects low-lying areas and increases risk based on water depth
    - Fire: Blocks or heavily penalizes roads in fire zones
    - Earthquake: Increases congestion and structural failure probability
    """
    
    def __init__(self):
        """Initialize the disaster model."""
        self._active_disasters: List[DisasterEvent] = []
    
    def apply_disaster_effects(self, graph: GraphManager, disaster: DisasterEvent,
                               alpha: float = 1.0, beta: float = 1.0, gamma: float = 1.0) -> None:
        """
        Apply disaster effects to all edges in the graph, with optional routing weights.
        
        This method modifies edge weights based on the disaster's type, location,
        and severity. Edges closer to the disaster epicenter are affected more severely.
        
        Args:
            graph: GraphManager instance to modify
            disaster: DisasterEvent to apply to the graph
            alpha: Distance weight
            beta: Risk weight
            gamma: Congestion weight
            
        Raises:
            ValueError: If graph is empty or disaster parameters are invalid
        """
        if graph.get_vertex_count() == 0:
            raise ValueError("Cannot apply disaster effects to empty graph")
        
        if disaster.severity <= 0 or disaster.max_effect_radius <= 0:
            raise ValueError("Invalid disaster parameters")
        
        # Store active disaster
        if disaster not in self._active_disasters:
            self._active_disasters.append(disaster)
        
        # Apply effects
        self.apply_objective_weights(graph, alpha, beta, gamma)
            
    def apply_objective_weights(self, graph: GraphManager, alpha: float = 1.0, 
                                beta: float = 1.0, gamma: float = 1.0) -> None:
        """
        Force a recalculation of all graph edge weights using multi-objective parameters.
        Applies currently active disasters automatically.
        
        Args:
            graph: GraphManager instance to modify
            alpha: Weight for distance objective
            beta: Weight for risk objective
            gamma: Weight for congestion objective
        """
        all_edges = graph.get_all_edges()
        
        # For simplicity, handle the most recent active disaster
        active_disaster = self._active_disasters[-1] if self._active_disasters else None
        
        for edge in all_edges:
            source_vertex = graph.get_vertex(edge.source)
            target_vertex = graph.get_vertex(edge.target)
            
            if source_vertex is None or target_vertex is None:
                continue
                
            edge_midpoint = WeightCalculator.calculate_edge_midpoint(
                source_vertex.coordinates, target_vertex.coordinates
            )
            
            # Reset blocked status to re-evaluate
            edge.is_blocked = False
            
            if active_disaster and active_disaster.is_point_affected(edge_midpoint):
                # Calculate new weight with disaster effects
                new_weight = self._calculate_disaster_affected_weight(
                    edge, active_disaster, edge_midpoint, alpha, beta, gamma
                )
                
                # Check if edge should be blocked
                if WeightCalculator.is_edge_blocked(edge, active_disaster, edge_midpoint):
                    edge.is_blocked = True
                    new_weight = float('inf')  # Infinite weight for blocked roads
            else:
                # No disaster effect on this edge
                new_weight = WeightCalculator.calculate_dynamic_weight(
                    edge, alpha=alpha, beta=beta, gamma=gamma
                )
            
            # Update edge weight in graph
            graph.update_edge_weight(edge.source, edge.target, new_weight)
    
    def _calculate_disaster_affected_weight(self, edge, disaster: DisasterEvent, 
                                          edge_midpoint: Tuple[float, float],
                                          alpha: float = 1.0, beta: float = 1.0, gamma: float = 1.0) -> float:
        """
        Calculate the new weight for an edge affected by disaster.
        
        Args:
            edge: Edge to calculate weight for
            disaster: DisasterEvent affecting the edge
            edge_midpoint: Midpoint coordinates of the edge
            alpha: Distance weight
            beta: Risk weight
            gamma: Congestion weight
            
        Returns:
            New weight incorporating disaster effects
        """
        # Use WeightCalculator for consistent weight calculation
        return WeightCalculator.calculate_dynamic_weight(
            edge, disaster, edge_midpoint, self._get_traffic_multiplier(disaster),
            alpha, beta, gamma
        )
    
    def _get_traffic_multiplier(self, disaster: DisasterEvent) -> float:
        """
        Get traffic multiplier based on disaster type.
        
        Different disasters affect traffic flow differently:
        - Earthquake: High congestion due to infrastructure damage
        - Fire: Moderate congestion from evacuation traffic
        - Flood: Variable congestion based on passable routes
        
        Args:
            disaster: DisasterEvent to get multiplier for
            
        Returns:
            Traffic multiplier for congestion calculations
        """
        multipliers = {
            DisasterType.EARTHQUAKE: 3.0,  # High congestion from damage
            DisasterType.FIRE: 2.0,        # Moderate evacuation congestion
            DisasterType.FLOOD: 1.5        # Some congestion from route limitations
        }
        return multipliers.get(disaster.disaster_type, 1.0)
    
    def calculate_risk_penalty(self, edge, disaster: DisasterEvent, 
                             edge_midpoint: Tuple[float, float]) -> float:
        """
        Calculate the risk penalty for an edge based on disaster effects.
        
        This method delegates to WeightCalculator for consistency but provides
        a convenient interface for disaster-specific risk calculations.
        
        Args:
            edge: Edge to calculate penalty for
            disaster: DisasterEvent affecting the area
            edge_midpoint: Midpoint coordinates of the edge
            
        Returns:
            Risk penalty value
        """
        return WeightCalculator.calculate_risk_penalty(edge, disaster, edge_midpoint)
    
    def is_road_blocked(self, edge, disaster: DisasterEvent, 
                       edge_midpoint: Tuple[float, float]) -> bool:
        """
        Determine if a road should be blocked based on disaster severity.
        
        Args:
            edge: Edge to check
            disaster: DisasterEvent to evaluate
            edge_midpoint: Midpoint coordinates of the edge
            
        Returns:
            True if road should be blocked, False otherwise
        """
        return WeightCalculator.is_edge_blocked(edge, disaster, edge_midpoint)
    
    def get_affected_edges(self, graph: GraphManager, epicenter: Tuple[float, float], 
                          radius: float) -> List:
        """
        Get all edges within a specified radius of a point.
        
        Args:
            graph: GraphManager to search
            epicenter: Center point coordinates
            radius: Search radius
            
        Returns:
            List of edges within the specified radius
        """
        affected_edges = []
        
        for edge in graph.get_all_edges():
            # Get vertex coordinates
            source_vertex = graph.get_vertex(edge.source)
            target_vertex = graph.get_vertex(edge.target)
            
            if source_vertex is None or target_vertex is None:
                continue
            
            # Calculate edge midpoint
            edge_midpoint = WeightCalculator.calculate_edge_midpoint(
                source_vertex.coordinates, target_vertex.coordinates
            )
            
            # Check if edge midpoint is within radius
            distance = ((edge_midpoint[0] - epicenter[0]) ** 2 + 
                       (edge_midpoint[1] - epicenter[1]) ** 2) ** 0.5
            
            if distance <= radius:
                affected_edges.append(edge)
        
        return affected_edges
    
    def remove_disaster_effects(self, graph: GraphManager, disaster: DisasterEvent) -> None:
        """
        Remove the effects of a specific disaster from the graph.
        
        This method resets edge weights to their base values and unblocks roads
        that were blocked by the specified disaster.
        
        Args:
            graph: GraphManager to modify
            disaster: DisasterEvent to remove effects for
        """
        if disaster in self._active_disasters:
            self._active_disasters.remove(disaster)
        
        # Reset all edges to base weights
        for edge in graph.get_all_edges():
            edge.reset_weight()
            graph.update_edge_weight(edge.source, edge.target, edge.current_weight)
        
        # Reapply remaining active disasters
        for active_disaster in self._active_disasters:
            self.apply_disaster_effects(graph, active_disaster)
    
    def clear_all_disaster_effects(self, graph: GraphManager) -> None:
        """
        Remove all disaster effects from the graph.
        
        Args:
            graph: GraphManager to reset
        """
        self._active_disasters.clear()
        
        # Reset all edges to base weights
        for edge in graph.get_all_edges():
            edge.reset_weight()
            graph.update_edge_weight(edge.source, edge.target, edge.current_weight)
    
    def get_active_disasters(self) -> List[DisasterEvent]:
        """
        Get list of currently active disasters.
        
        Returns:
            List of active DisasterEvent objects
        """
        return self._active_disasters.copy()
    
    def get_disaster_impact_summary(self, graph: GraphManager, 
                                   disaster: DisasterEvent) -> dict:
        """
        Get a summary of disaster impact on the graph.
        
        Args:
            graph: GraphManager to analyze
            disaster: DisasterEvent to analyze
            
        Returns:
            Dictionary with impact statistics
        """
        affected_edges = self.get_affected_edges(
            graph, disaster.epicenter, disaster.max_effect_radius
        )
        
        blocked_edges = []
        high_risk_edges = []
        
        for edge in affected_edges:
            source_vertex = graph.get_vertex(edge.source)
            target_vertex = graph.get_vertex(edge.target)
            
            if source_vertex and target_vertex:
                edge_midpoint = WeightCalculator.calculate_edge_midpoint(
                    source_vertex.coordinates, target_vertex.coordinates
                )
                
                if self.is_road_blocked(edge, disaster, edge_midpoint):
                    blocked_edges.append(edge)
                else:
                    risk_penalty = self.calculate_risk_penalty(edge, disaster, edge_midpoint)
                    if risk_penalty > edge.base_risk * 2:  # High risk threshold
                        high_risk_edges.append(edge)
        
        return {
            "disaster_type": disaster.disaster_type.value,
            "severity": disaster.severity,
            "epicenter": disaster.epicenter,
            "effect_radius": disaster.max_effect_radius,
            "total_edges": graph.get_edge_count(),
            "affected_edges": len(affected_edges),
            "blocked_edges": len(blocked_edges),
            "high_risk_edges": len(high_risk_edges),
            "impact_percentage": (len(affected_edges) / graph.get_edge_count() * 100) 
                               if graph.get_edge_count() > 0 else 0
        }
    
    def __str__(self) -> str:
        """String representation of the disaster model."""
        return f"DisasterModel(active_disasters={len(self._active_disasters)})"
    
    def __repr__(self) -> str:
        return self.__str__()