"""
Path result and algorithm statistics data models.
"""

from dataclasses import dataclass
from typing import List, Optional
from .edge import Edge


@dataclass
class AlgorithmStats:
    """Statistics collected during pathfinding algorithm execution."""
    nodes_visited: int = 0
    computation_time: float = 0.0
    edges_examined: int = 0
    queue_operations: int = 0
    
    def reset(self):
        """Reset all statistics to zero."""
        self.nodes_visited = 0
        self.computation_time = 0.0
        self.edges_examined = 0
        self.queue_operations = 0


@dataclass
class PathResult:
    """
    Result of a pathfinding operation.
    
    Contains the computed path, cost, and metadata about the computation.
    """
    path: List[str]  # List of vertex IDs in order
    total_cost: float
    edges_traversed: List[Edge]
    computation_time: float
    nodes_visited: int
    found: bool = True
    error_message: Optional[str] = None
    
    def __post_init__(self):
        """Validate path result data after initialization."""
        if self.found:
            if not self.path:
                raise ValueError("Found path cannot be empty")
            if len(self.path) != len(self.edges_traversed) + 1:
                raise ValueError("Path length must be edges + 1")
            if self.total_cost < 0:
                raise ValueError("Total cost cannot be negative")
        else:
            if self.error_message is None:
                self.error_message = "No path found"
    
    @property
    def path_length(self) -> int:
        """Number of vertices in the path."""
        return len(self.path)
    
    @property
    def edge_count(self) -> int:
        """Number of edges in the path."""
        return len(self.edges_traversed)
    
    def get_path_summary(self) -> str:
        """
        Get a human-readable summary of the path.
        
        Returns:
            String summary of the path result
        """
        if not self.found:
            return f"No path found: {self.error_message}"
        
        return (f"Path: {' -> '.join(self.path)}\n"
                f"Total cost: {self.total_cost:.2f}\n"
                f"Edges: {self.edge_count}\n"
                f"Computation time: {self.computation_time:.4f}s\n"
                f"Nodes visited: {self.nodes_visited}")
    
    def __str__(self) -> str:
        if self.found:
            return f"PathResult(path={' -> '.join(self.path)}, cost={self.total_cost:.2f})"
        else:
            return f"PathResult(not found: {self.error_message})"
    
    def __repr__(self) -> str:
        return self.__str__()