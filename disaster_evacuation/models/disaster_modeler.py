"""
Disaster Modeler for applying geographic disaster effects to road networks.

This module provides functionality to model disaster impacts on road networks
using real geographic coordinates and distance calculations.
"""

import math
import random
from typing import Dict, Tuple
from disaster_evacuation.models.graph import GraphManager


class InvalidCoordinateError(Exception):
    """Raised when epicenter coordinates are invalid."""
    pass


class InvalidRadiusError(Exception):
    """Raised when radius value is invalid."""
    pass


class DisasterModeler:
    """
    Models disaster effects on road networks using geographic coordinates.
    
    This component applies disaster scenarios (flood, fire, earthquake) to
    road networks by modifying edge weights based on geographic proximity
    to disaster epicenters.
    """
    
    BLOCKED_WEIGHT = 1e9  # Weight for completely blocked roads
    EARTH_RADIUS_METERS = 6371000  # Earth's radius in meters
    
    def __init__(self, graph_manager: GraphManager, coordinate_mapping: Dict[int, Tuple[float, float]]):
        """
        Initialize disaster modeler with graph and coordinate data.
        
        Args:
            graph_manager: GraphManager instance
            coordinate_mapping: Dict mapping internal node IDs to (lat, lon)
        """
        self.graph_manager = graph_manager
        self.coordinate_mapping = coordinate_mapping
    
    def _haversine_distance(self, coord1: Tuple[float, float], coord2: Tuple[float, float]) -> float:
        """
        Calculate great-circle distance between two coordinates using Haversine formula.
        
        Args:
            coord1: (latitude, longitude) of first point
            coord2: (latitude, longitude) of second point
        
        Returns:
            Distance in meters
        """
        lat1, lon1 = coord1
        lat2, lon2 = coord2
        
        # Convert to radians
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        delta_lat = math.radians(lat2 - lat1)
        delta_lon = math.radians(lon2 - lon1)
        
        # Haversine formula
        a = math.sin(delta_lat / 2) ** 2 + \
            math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon / 2) ** 2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        
        distance = self.EARTH_RADIUS_METERS * c
        return distance
    
    def _get_edge_midpoint(self, u: int, v: int) -> Tuple[float, float]:
        """
        Calculate midpoint coordinates of an edge.
        
        Args:
            u: Source node internal ID
            v: Target node internal ID
        
        Returns:
            (latitude, longitude) of edge midpoint
        """
        # Get coordinates for both nodes
        lat1, lon1 = self.coordinate_mapping[u]
        lat2, lon2 = self.coordinate_mapping[v]
        
        # Calculate midpoint (simple average for small distances)
        mid_lat = (lat1 + lat2) / 2
        mid_lon = (lon1 + lon2) / 2
        
        return (mid_lat, mid_lon)
    
    def _validate_epicenter(self, epicenter: Tuple[float, float]) -> None:
        """
        Validate epicenter coordinates.
        
        Args:
            epicenter: (latitude, longitude) tuple
        
        Raises:
            InvalidCoordinateError: If coordinates are out of valid range
        """
        lat, lon = epicenter
        
        if not (-90 <= lat <= 90):
            raise InvalidCoordinateError(
                f"Latitude {lat} is out of valid range [-90, 90]"
            )
        
        if not (-180 <= lon <= 180):
            raise InvalidCoordinateError(
                f"Longitude {lon} is out of valid range [-180, 180]"
            )
    
    def _validate_radius(self, radius_meters: float) -> None:
        """
        Validate radius value.
        
        Args:
            radius_meters: Radius in meters
        
        Raises:
            InvalidRadiusError: If radius is not positive
        """
        if radius_meters <= 0:
            raise InvalidRadiusError(
                f"Radius must be positive, got {radius_meters}"
            )
    
    def apply_flood(self, epicenter: Tuple[float, float], radius_meters: float,
                   risk_multiplier: float = 0.5) -> None:
        """
        Apply flood disaster centered at geographic coordinates.
        
        Increases risk factor for edges within the affected radius.
        
        Args:
            epicenter: (latitude, longitude) of flood center
            radius_meters: Affected radius in meters
            risk_multiplier: Risk factor to add to affected edges (default: 0.5)
        
        Raises:
            InvalidCoordinateError: If epicenter is invalid
            InvalidRadiusError: If radius is not positive
        """
        # Validate inputs
        self._validate_epicenter(epicenter)
        self._validate_radius(radius_meters)
        
        # Check each edge
        for node_id in range(len(self.coordinate_mapping)):
            vertex_id = str(node_id)
            
            if not self.graph_manager.has_vertex(vertex_id):
                continue
            
            # Get all outgoing edges
            neighbors = self.graph_manager.get_neighbors(vertex_id)
            
            for edge in neighbors:
                # Get target node ID
                target_id = int(edge.target)
                
                # Calculate edge midpoint
                midpoint = self._get_edge_midpoint(node_id, target_id)
                
                # Calculate distance from epicenter to midpoint
                distance = self._haversine_distance(epicenter, midpoint)
                
                # If within radius, increase risk
                if distance <= radius_meters:
                    # Calculate new weight with increased risk
                    # weight = distance + (risk * distance) + (congestion * distance)
                    new_risk = edge.base_risk + risk_multiplier
                    new_weight = edge.base_distance * (1 + new_risk + edge.base_congestion)
                    
                    # Update edge weight
                    self.graph_manager.update_edge_weight(vertex_id, edge.target, new_weight)
                    
                    # Update the edge object's risk factor
                    edge.base_risk = new_risk
    
    def apply_fire(self, epicenter: Tuple[float, float], radius_meters: float) -> None:
        """
        Apply fire disaster - blocks roads in affected area.
        
        Sets edge weights to BLOCKED_WEIGHT for edges within radius.
        
        Args:
            epicenter: (latitude, longitude) of fire center
            radius_meters: Fire zone radius in meters
        
        Raises:
            InvalidCoordinateError: If epicenter is invalid
            InvalidRadiusError: If radius is not positive
        """
        # Validate inputs
        self._validate_epicenter(epicenter)
        self._validate_radius(radius_meters)
        
        # Check each edge
        for node_id in range(len(self.coordinate_mapping)):
            vertex_id = str(node_id)
            
            if not self.graph_manager.has_vertex(vertex_id):
                continue
            
            # Get all outgoing edges
            neighbors = self.graph_manager.get_neighbors(vertex_id)
            
            for edge in neighbors:
                # Get target node ID
                target_id = int(edge.target)
                
                # Calculate edge midpoint
                midpoint = self._get_edge_midpoint(node_id, target_id)
                
                # Calculate distance from epicenter to midpoint
                distance = self._haversine_distance(epicenter, midpoint)
                
                # If within radius, block the road
                if distance <= radius_meters:
                    # Set weight to blocked
                    self.graph_manager.update_edge_weight(vertex_id, edge.target, self.BLOCKED_WEIGHT)
    
    def apply_earthquake(self, epicenter: Tuple[float, float], radius_meters: float,
                        failure_probability: float = 0.2,
                        congestion_multiplier: float = 0.8) -> None:
        """
        Apply earthquake disaster - random failures and congestion.
        
        Randomly blocks some edges and increases congestion on others within radius.
        
        Args:
            epicenter: (latitude, longitude) of earthquake center
            radius_meters: Affected radius in meters
            failure_probability: Probability of road becoming blocked (default: 0.2)
            congestion_multiplier: Congestion factor for non-blocked roads (default: 0.8)
        
        Raises:
            InvalidCoordinateError: If epicenter is invalid
            InvalidRadiusError: If radius is not positive
        """
        # Validate inputs
        self._validate_epicenter(epicenter)
        self._validate_radius(radius_meters)
        
        # Check each edge
        for node_id in range(len(self.coordinate_mapping)):
            vertex_id = str(node_id)
            
            if not self.graph_manager.has_vertex(vertex_id):
                continue
            
            # Get all outgoing edges
            neighbors = self.graph_manager.get_neighbors(vertex_id)
            
            for edge in neighbors:
                # Get target node ID
                target_id = int(edge.target)
                
                # Calculate edge midpoint
                midpoint = self._get_edge_midpoint(node_id, target_id)
                
                # Calculate distance from epicenter to midpoint
                distance = self._haversine_distance(epicenter, midpoint)
                
                # If within radius, apply earthquake effects
                if distance <= radius_meters:
                    # Randomly determine if road fails
                    if random.random() < failure_probability:
                        # Block the road
                        self.graph_manager.update_edge_weight(vertex_id, edge.target, self.BLOCKED_WEIGHT)
                    else:
                        # Increase congestion
                        new_congestion = edge.base_congestion + congestion_multiplier
                        new_weight = edge.base_distance * (1 + edge.base_risk + new_congestion)
                        
                        # Update edge weight
                        self.graph_manager.update_edge_weight(vertex_id, edge.target, new_weight)
                        
                        # Update the edge object's congestion factor
                        edge.base_congestion = new_congestion
    
    def __str__(self) -> str:
        """String representation of the disaster modeler."""
        return f"DisasterModeler(nodes={len(self.coordinate_mapping)})"
    
    def __repr__(self) -> str:
        return self.__str__()
