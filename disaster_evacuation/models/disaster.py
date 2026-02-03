"""
Disaster event data model for the disaster evacuation routing system.
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Tuple


class DisasterType(Enum):
    """Types of disasters that can affect evacuation routes."""
    FLOOD = "flood"
    FIRE = "fire"
    EARTHQUAKE = "earthquake"


@dataclass
class DisasterEvent:
    """
    Represents a disaster event that affects the evacuation graph.
    
    Each disaster has a type, location, severity, and area of effect that
    influences how edge weights are modified during route computation.
    """
    disaster_type: DisasterType
    epicenter: Tuple[float, float]  # (latitude, longitude) or (x, y)
    severity: float  # 0.0 to 1.0
    max_effect_radius: float
    start_time: datetime = None
    
    def __post_init__(self):
        """Validate disaster event data after initialization."""
        if not isinstance(self.disaster_type, DisasterType):
            raise ValueError(f"Invalid disaster type: {self.disaster_type}")
        
        if len(self.epicenter) != 2:
            raise ValueError("Epicenter must be a tuple of (x, y)")
        
        if not 0.0 <= self.severity <= 1.0:
            raise ValueError("Severity must be between 0.0 and 1.0")
        
        if self.max_effect_radius <= 0:
            raise ValueError("Max effect radius must be positive")
        
        if self.start_time is None:
            self.start_time = datetime.now()
    
    def distance_to_point(self, point: Tuple[float, float]) -> float:
        """
        Calculate distance from disaster epicenter to a given point.
        
        Args:
            point: Coordinates (x, y) to calculate distance to
            
        Returns:
            Euclidean distance from epicenter to point
        """
        x1, y1 = self.epicenter
        x2, y2 = point
        return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
    
    def get_disaster_multiplier(self) -> float:
        """
        Get the base disaster multiplier for this disaster type.
        
        Returns:
            Multiplier value based on disaster type
        """
        multipliers = {
            DisasterType.FLOOD: 2.0,
            DisasterType.FIRE: 3.0,
            DisasterType.EARTHQUAKE: 2.5
        }
        return multipliers[self.disaster_type]
    
    def is_point_affected(self, point: Tuple[float, float]) -> bool:
        """
        Check if a point is within the disaster's area of effect.
        
        Args:
            point: Coordinates to check
            
        Returns:
            True if point is within max_effect_radius of epicenter
        """
        return self.distance_to_point(point) <= self.max_effect_radius
    
    def __str__(self) -> str:
        return f"DisasterEvent({self.disaster_type.value}, {self.epicenter}, severity={self.severity})"
    
    def __repr__(self) -> str:
        return self.__str__()