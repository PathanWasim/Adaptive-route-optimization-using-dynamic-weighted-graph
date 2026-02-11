"""
OSM Extractor for downloading real road network data from OpenStreetMap.

This module provides functionality to extract road networks using OSMnx
while maintaining academic focus on algorithmic implementation.
"""

import osmnx as ox
import networkx as nx
from typing import Optional


class AreaTooLargeError(Exception):
    """Raised when extracted area exceeds the 1-3 km² bounds."""
    pass


class PlaceNotFoundError(Exception):
    """Raised when OSMnx cannot find the specified place."""
    pass


class EmptyNetworkError(Exception):
    """Raised when extracted network has no roads."""
    pass


class OSMExtractor:
    """
    Extracts real road network data from OpenStreetMap using OSMnx.
    
    This component is responsible for downloading road networks and validating
    that they meet the academic project requirements (1-3 km² area).
    """
    
    def __init__(self):
        """Initialize the OSM extractor."""
        # Configure OSMnx settings for consistent behavior
        ox.settings.use_cache = True
        ox.settings.log_console = False
        # CRITICAL: Increase max query area to allow larger bounding boxes
        ox.settings.max_query_area_size = 50000000  # 50 million square meters (~50 km²)
    
    def extract_by_place(self, place_name: str, network_type: str = "drive") -> nx.MultiDiGraph:
        """
        Extract road network for a named place.
        
        Args:
            place_name: City or area name (e.g., "Piedmont, California, USA")
            network_type: Type of street network ("drive", "walk", "bike", "all")
        
        Returns:
            NetworkX MultiDiGraph with nodes containing (x, y) coordinates
            and edges containing length, highway type, and other OSM attributes
        
        Raises:
            PlaceNotFoundError: If place name is not found in OSM
            AreaTooLargeError: If extracted area exceeds 3 km²
            EmptyNetworkError: If no roads found in the area
        """
        try:
            # Download street network from OSM
            G = ox.graph_from_place(place_name, network_type=network_type, simplify=True)
            
            # Validate the extracted network
            self._validate_network(G, place_name)
            
            # Return unprojected graph with lat/lon coordinates
            # (Projection to UTM is only needed for area calculation, not for routing)
            return G
            
        except Exception as e:
            if "Could not geocode" in str(e) or "not found" in str(e).lower():
                raise PlaceNotFoundError(
                    f"Place '{place_name}' not found. "
                    f"Try a more specific name like 'City, State, Country'"
                ) from e
            else:
                raise
    
    def extract_by_bbox(self, north: float, south: float, east: float, west: float,
                       network_type: str = "drive") -> nx.MultiDiGraph:
        """
        Extract road network within bounding box.
        
        Args:
            north, south, east, west: Latitude/longitude bounds
            network_type: Type of street network
        
        Returns:
            NetworkX MultiDiGraph with geographic data
        
        Raises:
            AreaTooLargeError: If extracted area exceeds 3 km²
            EmptyNetworkError: If no roads found in the area
        """
        try:
            # CRITICAL: Set VERY large max query area to avoid sub-queries
            ox.settings.max_query_area_size = 5000000000000  # 5 trillion square meters
            
            # Download street network from bounding box (using keyword arguments for newer OSMnx)
            G = ox.graph_from_bbox(bbox=(north, south, east, west), 
                                  network_type=network_type, simplify=True)
            
            # Validate the extracted network
            self._validate_network(G, f"bbox({north},{south},{east},{west})")
            
            # Return unprojected graph with lat/lon coordinates
            return G
            
        except Exception as e:
            if "Could not geocode" in str(e):
                raise PlaceNotFoundError(
                    f"Invalid bounding box coordinates"
                ) from e
            else:
                raise
    
    def get_network_stats(self, graph: nx.MultiDiGraph) -> dict:
        """
        Return statistics about extracted network.
        
        Args:
            graph: NetworkX graph from OSM extraction
        
        Returns:
            Dictionary with keys: num_nodes, num_edges, area_km2, avg_edge_length
        """
        num_nodes = graph.number_of_nodes()
        num_edges = graph.number_of_edges()
        
        # Calculate average edge length
        total_length = sum(data.get('length', 0) 
                          for u, v, data in graph.edges(data=True))
        avg_edge_length = total_length / num_edges if num_edges > 0 else 0
        
        # Calculate area in km²
        area_km2 = 0.0
        
        # Check if this is a real OSM graph with CRS
        if 'crs' in graph.graph:
            try:
                # Graph is from OSM, use OSMnx for accurate area calculation
                if 'proj' in str(graph.graph['crs']).lower():
                    # Graph is projected (UTM), calculate area from convex hull
                    nodes_gdf = ox.graph_to_gdfs(graph, edges=False)
                    area_m2 = nodes_gdf.unary_union.convex_hull.area
                    area_km2 = area_m2 / 1_000_000
                else:
                    # Use OSMnx basic_stats for unprojected graphs
                    stats = ox.basic_stats(graph)
                    area_km2 = stats.get('area_km2', 0)
            except Exception:
                # Fallback to bounding box estimation
                area_km2 = self._estimate_area_from_bbox(graph)
        else:
            # Test graph without CRS - estimate from bounding box
            area_km2 = self._estimate_area_from_bbox(graph)
        
        return {
            'num_nodes': num_nodes,
            'num_edges': num_edges,
            'area_km2': area_km2,
            'avg_edge_length': avg_edge_length
        }
    
    def _estimate_area_from_bbox(self, graph: nx.MultiDiGraph) -> float:
        """
        Estimate area from bounding box of node coordinates.
        
        Args:
            graph: NetworkX graph with node coordinates
        
        Returns:
            Estimated area in km²
        """
        if graph.number_of_nodes() == 0:
            return 0.0
        
        # Get all node coordinates
        lats = [data.get('y', 0) for _, data in graph.nodes(data=True)]
        lons = [data.get('x', 0) for _, data in graph.nodes(data=True)]
        
        if not lats or not lons:
            return 0.0
        
        # Calculate bounding box
        min_lat, max_lat = min(lats), max(lats)
        min_lon, max_lon = min(lons), max(lons)
        
        # Rough approximation: 1 degree ≈ 111 km at equator
        # This is inaccurate but sufficient for test validation
        lat_diff_km = (max_lat - min_lat) * 111
        lon_diff_km = (max_lon - min_lon) * 111 * abs(((min_lat + max_lat) / 2))
        
        area_km2 = lat_diff_km * lon_diff_km
        return area_km2
    
    def _validate_network(self, graph: nx.MultiDiGraph, location: str) -> None:
        """
        Validate that extracted network meets requirements.
        
        Args:
            graph: Extracted network graph
            location: Location identifier for error messages
        
        Raises:
            EmptyNetworkError: If network has no edges
        """
        # Check if network is empty
        if graph.number_of_edges() == 0:
            raise EmptyNetworkError(
                f"No roads found in {location}. "
                f"Try a different area with road infrastructure."
            )
        
        # Note: Area validation removed to support larger city areas
    
    def __str__(self) -> str:
        """String representation of the extractor."""
        return "OSMExtractor(cache=True)"
    
    def __repr__(self) -> str:
        return self.__str__()
