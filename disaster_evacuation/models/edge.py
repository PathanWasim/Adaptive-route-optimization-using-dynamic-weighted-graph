"""
Edge data model for the disaster evacuation routing system.
"""

from dataclasses import dataclass


@dataclass
class Edge:
    """
    Represents an edge (road) in the evacuation graph.
    
    Each edge connects two vertices and has base properties that are modified
    by disaster conditions to compute the final traversal weight.
    """
    source: str
    target: str
    base_distance: float
    base_risk: float
    base_congestion: float
    current_weight: float = 0.0
    is_blocked: bool = False
    
    def __post_init__(self):
        """Validate edge data after initialization."""
        if not self.source or not self.target:
            raise ValueError("Source and target vertex IDs cannot be empty")
        
        if self.source == self.target:
            raise ValueError("Self-loops are not allowed")
        
        if self.base_distance < 0:
            raise ValueError("Base distance cannot be negative")
        
        if self.base_risk < 0:
            raise ValueError("Base risk cannot be negative")
        
        if self.base_congestion < 0:
            raise ValueError("Base congestion cannot be negative")
        
        # Initialize current weight if not set
        if self.current_weight == 0.0:
            self.current_weight = self.base_distance + self.base_risk + self.base_congestion
    
    def reset_weight(self):
        """Reset current weight to base values without disaster effects."""
        self.current_weight = self.base_distance + self.base_risk + self.base_congestion
        self.is_blocked = False
    
    def __str__(self) -> str:
        blocked_str = " (BLOCKED)" if self.is_blocked else ""
        return f"Edge({self.source} -> {self.target}, weight={self.current_weight:.2f}{blocked_str})"
    
    def __repr__(self) -> str:
        return self.__str__()