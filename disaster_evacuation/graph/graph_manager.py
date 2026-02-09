"""
Graph Manager for the disaster evacuation routing system.

This module implements the GraphManager class that maintains the city graph structure
using an adjacency list representation optimized for sparse graphs.
"""

from typing import Dict, List, Optional, Tuple, Set
from ..models import Vertex, Edge, VertexType


class GraphManager:
    """
    Manages the graph structure for the disaster evacuation routing system.
    
    Uses adjacency list representation for efficient sparse graph operations.
    Supports O(1) vertex lookup and O(degree) neighbor enumeration.
    Space complexity: O(V + E)
    """
    
    def __init__(self):
        """Initialize an empty graph."""
        self._vertices: Dict[str, Vertex] = {}
        self._adjacency_list: Dict[str, List[Edge]] = {}
        self._edge_weights: Dict[Tuple[str, str], float] = {}
        self._edges: Dict[Tuple[str, str], Edge] = {}
        self._node_coordinates: Dict[str, Tuple[float, float]] = {}
    
    def add_vertex(self, vertex_id: str, vertex_type: VertexType, 
                   coordinates: Tuple[float, float], capacity: Optional[int] = None) -> None:
        """
        Add a vertex to the graph.
        
        Args:
            vertex_id: Unique identifier for the vertex
            vertex_type: Type of vertex (INTERSECTION, SHELTER, EVACUATION_POINT)
            coordinates: (x, y) coordinates of the vertex
            capacity: Optional capacity for shelters and evacuation points
            
        Raises:
            ValueError: If vertex already exists or invalid parameters
        """
        if vertex_id in self._vertices:
            raise ValueError(f"Vertex '{vertex_id}' already exists")
        
        vertex = Vertex(vertex_id, vertex_type, coordinates, capacity)
        self._vertices[vertex_id] = vertex
        self._adjacency_list[vertex_id] = []
    
    def add_edge(self, source: str, target: str, distance: float, 
                 base_risk: float, base_congestion: float) -> None:
        """
        Add an edge to the graph.
        
        Args:
            source: Source vertex ID
            target: Target vertex ID
            distance: Base distance of the edge
            base_risk: Base risk factor
            base_congestion: Base congestion factor
            
        Raises:
            ValueError: If vertices don't exist or invalid parameters
        """
        if source not in self._vertices:
            raise ValueError(f"Source vertex '{source}' does not exist")
        if target not in self._vertices:
            raise ValueError(f"Target vertex '{target}' does not exist")
        
        edge = Edge(source, target, distance, base_risk, base_congestion)
        
        # Add edge to adjacency list
        self._adjacency_list[source].append(edge)
        
        # Store edge for quick lookup
        edge_key = (source, target)
        self._edges[edge_key] = edge
        self._edge_weights[edge_key] = edge.current_weight
    
    def get_neighbors(self, vertex_id: str) -> List[Edge]:
        """
        Get all outgoing edges from a vertex.
        
        Args:
            vertex_id: Vertex to get neighbors for
            
        Returns:
            List of edges from the vertex
            
        Raises:
            ValueError: If vertex doesn't exist
        """
        if vertex_id not in self._vertices:
            raise ValueError(f"Vertex '{vertex_id}' does not exist")
        
        return self._adjacency_list[vertex_id].copy()
    
    def get_edge_weight(self, source: str, target: str) -> float:
        """
        Get the current weight of an edge.
        
        Args:
            source: Source vertex ID
            target: Target vertex ID
            
        Returns:
            Current weight of the edge
            
        Raises:
            ValueError: If edge doesn't exist
        """
        edge_key = (source, target)
        if edge_key not in self._edge_weights:
            raise ValueError(f"Edge '{source}' -> '{target}' does not exist")
        
        return self._edge_weights[edge_key]
    
    def update_edge_weight(self, source: str, target: str, new_weight: float) -> None:
        """
        Update the weight of an edge.
        
        Args:
            source: Source vertex ID
            target: Target vertex ID
            new_weight: New weight for the edge
            
        Raises:
            ValueError: If edge doesn't exist or weight is negative
        """
        if new_weight < 0:
            raise ValueError("Edge weight cannot be negative")
        
        edge_key = (source, target)
        if edge_key not in self._edges:
            raise ValueError(f"Edge '{source}' -> '{target}' does not exist")
        
        # Update both the edge object and weight cache
        self._edges[edge_key].current_weight = new_weight
        self._edge_weights[edge_key] = new_weight
    
    def is_connected(self, source: str, target: str) -> bool:
        """
        Check if there is a direct edge between two vertices.
        
        Args:
            source: Source vertex ID
            target: Target vertex ID
            
        Returns:
            True if direct edge exists, False otherwise
        """
        edge_key = (source, target)
        return edge_key in self._edges
    
    def get_vertex_count(self) -> int:
        """Get the number of vertices in the graph."""
        return len(self._vertices)
    
    def get_edge_count(self) -> int:
        """Get the number of edges in the graph."""
        return len(self._edges)
    
    def get_vertex(self, vertex_id: str) -> Optional[Vertex]:
        """
        Get a vertex by ID.
        
        Args:
            vertex_id: Vertex ID to retrieve
            
        Returns:
            Vertex object or None if not found
        """
        return self._vertices.get(vertex_id)
    
    def get_edge(self, source: str, target: str) -> Optional[Edge]:
        """
        Get an edge by source and target vertices.
        
        Args:
            source: Source vertex ID
            target: Target vertex ID
            
        Returns:
            Edge object or None if not found
        """
        edge_key = (source, target)
        return self._edges.get(edge_key)
    
    def get_all_vertices(self) -> List[Vertex]:
        """Get all vertices in the graph."""
        return list(self._vertices.values())
    
    def get_all_edges(self) -> List[Edge]:
        """Get all edges in the graph."""
        return list(self._edges.values())
    
    def get_vertex_ids(self) -> Set[str]:
        """Get all vertex IDs in the graph."""
        return set(self._vertices.keys())
    
    def has_vertex(self, vertex_id: str) -> bool:
        """Check if a vertex exists in the graph."""
        return vertex_id in self._vertices
    
    def remove_vertex(self, vertex_id: str) -> None:
        """
        Remove a vertex and all its edges from the graph.
        
        Args:
            vertex_id: Vertex ID to remove
            
        Raises:
            ValueError: If vertex doesn't exist
        """
        if vertex_id not in self._vertices:
            raise ValueError(f"Vertex '{vertex_id}' does not exist")
        
        # Remove all edges involving this vertex
        edges_to_remove = []
        for edge_key, edge in self._edges.items():
            if edge.source == vertex_id or edge.target == vertex_id:
                edges_to_remove.append(edge_key)
        
        for edge_key in edges_to_remove:
            del self._edges[edge_key]
            del self._edge_weights[edge_key]
        
        # Remove from adjacency lists
        del self._adjacency_list[vertex_id]
        for adj_list in self._adjacency_list.values():
            adj_list[:] = [edge for edge in adj_list if edge.target != vertex_id]
        
        # Remove vertex
        del self._vertices[vertex_id]
    
    def clear(self) -> None:
        """Remove all vertices and edges from the graph."""
        self._vertices.clear()
        self._adjacency_list.clear()
        self._edge_weights.clear()
        self._edges.clear()
        self._node_coordinates.clear()
    
    def set_node_coordinates(self, vertex_id: str, lat: float, lon: float) -> None:
        """
        Store geographic coordinates for a node.
        
        Args:
            vertex_id: Vertex ID
            lat: Latitude
            lon: Longitude
            
        Raises:
            ValueError: If vertex doesn't exist
        """
        if vertex_id not in self._vertices:
            raise ValueError(f"Vertex '{vertex_id}' does not exist")
        
        self._node_coordinates[vertex_id] = (lat, lon)
    
    def get_node_coordinates(self, vertex_id: str) -> Optional[Tuple[float, float]]:
        """
        Retrieve geographic coordinates for a node.
        
        Args:
            vertex_id: Vertex ID
            
        Returns:
            Tuple of (latitude, longitude) or None if not set
            
        Raises:
            ValueError: If vertex doesn't exist
        """
        if vertex_id not in self._vertices:
            raise ValueError(f"Vertex '{vertex_id}' does not exist")
        
        return self._node_coordinates.get(vertex_id)
    
    def get_graph_info(self) -> Dict[str, any]:
        """
        Get summary information about the graph.
        
        Returns:
            Dictionary with graph statistics
        """
        vertex_types = {}
        for vertex in self._vertices.values():
            vertex_type = vertex.vertex_type.value
            vertex_types[vertex_type] = vertex_types.get(vertex_type, 0) + 1
        
        return {
            "vertex_count": len(self._vertices),
            "edge_count": len(self._edges),
            "vertex_types": vertex_types,
            "average_degree": len(self._edges) / len(self._vertices) if self._vertices else 0
        }
    
    def __str__(self) -> str:
        """String representation of the graph."""
        info = self.get_graph_info()
        return f"Graph(vertices={info['vertex_count']}, edges={info['edge_count']})"
    
    def __repr__(self) -> str:
        return self.__str__()