"""
Vertex data model for the disaster evacuation routing system.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Tuple, Optional


class VertexType(Enum):
    """Types of vertices in the evacuation graph."""
    INTERSECTION = "intersection"
    SHELTER = "shelter"
    EVACUATION_POINT = "evacuation_point"


@dataclass
class Vertex:
    """
    Represents a vertex in the evacuation graph.
    
    A vertex can be an intersection, shelter, or evacuation point in the city network.
    Coordinates are used for distance calculations and disaster proximity effects.
    """
    id: str
    vertex_type: VertexType
    coordinates: Tuple[float, float]  # (latitude, longitude) or (x, y)
    capacity: Optional[int] = None  # For shelters and evacuation points
    
    def __post_init__(self):
        """Validate vertex data after initialization."""
        if not self.id:
            raise ValueError("Vertex ID cannot be empty")
        
        if not isinstance(self.vertex_type, VertexType):
            raise ValueError(f"Invalid vertex type: {self.vertex_type}")
        
        if len(self.coordinates) != 2:
            raise ValueError("Coordinates must be a tuple of (x, y)")
        
        if self.capacity is not None and self.capacity < 0:
            raise ValueError("Capacity cannot be negative")
    
    def distance_to(self, other: 'Vertex') -> float:
        """
        Calculate Euclidean distance to another vertex.
        
        Args:
            other: Another vertex to calculate distance to
            
        Returns:
            Euclidean distance between vertices
        """
        x1, y1 = self.coordinates
        x2, y2 = other.coordinates
        return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
    
    def __str__(self) -> str:
        return f"Vertex({self.id}, {self.vertex_type.value}, {self.coordinates})"
    
    def __repr__(self) -> str:
        return self.__str__()