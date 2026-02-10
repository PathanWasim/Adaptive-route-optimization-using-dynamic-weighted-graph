"""
Graph Converter for transforming OSM NetworkX graphs to internal format.

This module converts OpenStreetMap road networks into the internal GraphManager
adjacency list format while preserving all necessary geographic information.
"""

import networkx as nx
from typing import Dict, Tuple
from disaster_evacuation.graph.graph_manager import GraphManager
from disaster_evacuation.models import VertexType


class MissingCoordinateError(Exception):
    """Raised when a node is missing coordinate data."""
    pass


class GraphConverter:
    """
    Converts OSM NetworkX graphs to internal GraphManager format.
    
    This component handles the transformation from OSM's arbitrary node IDs
    to sequential 0-based IDs while preserving all geographic information.
    """
    
    def __init__(self):
        """Initialize the graph converter."""
        pass
    
    def _map_osm_nodes(self, osm_graph: nx.MultiDiGraph) -> Dict[int, int]:
        """
        Create mapping from OSM node IDs to sequential internal IDs.
        
        OSM node IDs are arbitrary large integers. This method creates a
        mapping to sequential 0-based IDs for use in the internal graph.
        
        Args:
            osm_graph: NetworkX graph from OSM extraction
        
        Returns:
            Dictionary mapping {osm_node_id: internal_node_id}
        """
        # Get all unique node IDs from the OSM graph
        osm_node_ids = list(osm_graph.nodes())
        
        # Create sequential mapping: 0, 1, 2, ...
        id_mapping = {osm_id: internal_id 
                     for internal_id, osm_id in enumerate(osm_node_ids)}
        
        return id_mapping
    
    def _create_coordinate_mapping(self, osm_graph: nx.MultiDiGraph, 
                                   id_mapping: Dict[int, int]) -> Dict[int, Tuple[float, float]]:
        """
        Create mapping from internal IDs to geographic coordinates.
        
        Args:
            osm_graph: NetworkX graph from OSM extraction
            id_mapping: Mapping from OSM node IDs to internal IDs
        
        Returns:
            Dictionary mapping {internal_node_id: (latitude, longitude)}
        
        Raises:
            MissingCoordinateError: If any node lacks coordinate data
        """
        coordinate_mapping = {}
        
        for osm_id, internal_id in id_mapping.items():
            node_data = osm_graph.nodes[osm_id]
            
            # OSM nodes have 'x' (longitude) and 'y' (latitude) attributes
            if 'x' not in node_data or 'y' not in node_data:
                raise MissingCoordinateError(
                    f"Node {osm_id} (internal ID {internal_id}) missing coordinate data"
                )
            
            lon = node_data['x']
            lat = node_data['y']
            
            # Store as (latitude, longitude) tuple
            coordinate_mapping[internal_id] = (lat, lon)
        
        return coordinate_mapping
    
    def _extract_edge_distance(self, osm_graph: nx.MultiDiGraph, 
                               u: int, v: int, key: int = 0) -> float:
        """
        Extract distance from OSM edge attributes.
        
        Args:
            osm_graph: NetworkX graph from OSM extraction
            u, v: OSM node IDs for the edge
            key: Edge key (for MultiDiGraph with parallel edges)
        
        Returns:
            Distance in meters (from 'length' attribute)
        """
        # Get edge data
        edge_data = osm_graph.get_edge_data(u, v, key)
        
        if edge_data is None:
            # Edge doesn't exist, return 0
            return 0.0
        
        # Extract length attribute (in meters)
        distance = edge_data.get('length', 0.0)
        
        return float(distance)
    
    def convert_osm_to_internal(self, osm_graph: nx.MultiDiGraph) -> Tuple[GraphManager, Dict[int, Tuple[float, float]]]:
        """
        Convert OSM graph to internal GraphManager format.
        
        This method transforms an OSM NetworkX graph into the internal adjacency
        list representation while preserving all geographic information.
        
        Args:
            osm_graph: NetworkX MultiDiGraph from OSM extraction
        
        Returns:
            Tuple of (GraphManager instance, coordinate_mapping)
            coordinate_mapping: dict mapping internal node IDs to (lat, lon)
        
        Raises:
            MissingCoordinateError: If any node lacks coordinate data
        """
        # Step 1: Create node ID mapping (OSM IDs → sequential 0-based IDs)
        id_mapping = self._map_osm_nodes(osm_graph)
        
        # Step 2: Create coordinate mapping for internal IDs
        coordinate_mapping = self._create_coordinate_mapping(osm_graph, id_mapping)
        
        # Step 3: Create GraphManager instance
        graph_manager = GraphManager()
        
        # Step 4: Add all vertices to GraphManager
        for osm_id, internal_id in id_mapping.items():
            lat, lon = coordinate_mapping[internal_id]
            
            # Convert internal ID to string for GraphManager
            vertex_id = str(internal_id)
            
            # All OSM nodes are intersections
            graph_manager.add_vertex(
                vertex_id=vertex_id,
                vertex_type=VertexType.INTERSECTION,
                coordinates=(lat, lon),
                capacity=None
            )
        
        # Step 5: Add all edges to GraphManager
        for u_osm, v_osm, key, edge_data in osm_graph.edges(keys=True, data=True):
            # Skip self-loops (not allowed in our graph model)
            if u_osm == v_osm:
                continue
            
            # Convert OSM IDs to internal IDs
            u_internal = id_mapping[u_osm]
            v_internal = id_mapping[v_osm]
            
            # Convert to string IDs
            source = str(u_internal)
            target = str(v_internal)
            
            # Extract distance from edge data
            distance = self._extract_edge_distance(osm_graph, u_osm, v_osm, key)
            
            # Initialize with zero risk and congestion factors
            # (will be modified by disaster modeling later)
            risk_factor = 0.0
            congestion_factor = 0.0
            
            # Add edge to GraphManager
            graph_manager.add_edge(
                source=source,
                target=target,
                distance=distance,
                base_risk=risk_factor,
                base_congestion=congestion_factor
            )
        
        return graph_manager, coordinate_mapping
    
    def __str__(self) -> str:
        """String representation of the converter."""
        return "GraphConverter()"
    
    def __repr__(self) -> str:
        return self.__str__()
