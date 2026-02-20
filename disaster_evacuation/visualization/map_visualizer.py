"""
Map Visualizer for rendering road networks with real geographic coordinates.

This module provides functionality to visualize road networks and routing results
using matplotlib with actual latitude/longitude coordinates.
"""

import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from disaster_evacuation.models.graph import GraphManager


class InvalidNodeError(Exception):
    """Raised when a node ID is invalid."""
    pass


class MapVisualizer:
    """
    Visualizes road networks and routes using real geographic coordinates.
    
    This component renders road networks and routing results on matplotlib
    plots using actual latitude/longitude coordinates from OpenStreetMap.
    """
    
    def __init__(self, graph_manager: GraphManager, coordinate_mapping: Dict[int, Tuple[float, float]]):
        """
        Initialize visualizer with graph and coordinates.
        
        Args:
            graph_manager: GraphManager instance
            coordinate_mapping: Dict mapping internal node IDs to (lat, lon)
        """
        self.graph_manager = graph_manager
        self.coordinate_mapping = coordinate_mapping
    
    def _plot_edge(self, ax, u, v, color: str = 'gray', 
                  linewidth: float = 0.5, linestyle: str = '-', alpha: float = 1.0) -> None:
        """
        Plot a single edge on the map.
        
        Args:
            ax: Matplotlib axis
            u, v: Node IDs (can be string or integer)
            color: Edge color
            linewidth: Line width
            linestyle: Line style ('-', '--', ':', etc.)
            alpha: Transparency (0.0 to 1.0)
        """
        # Convert to integers for coordinate lookup
        u_int = int(u) if isinstance(u, str) else u
        v_int = int(v) if isinstance(v, str) else v
        
        # Get coordinates
        lat1, lon1 = self.coordinate_mapping[u_int]
        lat2, lon2 = self.coordinate_mapping[v_int]
        
        # Plot edge (matplotlib expects x, y = lon, lat)
        ax.plot([lon1, lon2], [lat1, lat2], 
               color=color, linewidth=linewidth, linestyle=linestyle, alpha=alpha)
    
    def plot_network(self, blocked_edges: Optional[List[Tuple[int, int]]] = None,
                    figsize: Tuple[int, int] = (12, 10)) -> None:
        """
        Plot the entire road network.
        
        Args:
            blocked_edges: List of (u, v) tuples for blocked roads (shown in red)
            figsize: Figure size for matplotlib
        
        Effect:
            Displays road network with normal roads in gray, blocked roads in red
        """
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Convert blocked edges to set for fast lookup
        blocked_set = set(blocked_edges) if blocked_edges else set()
        
        # Plot all edges
        for node_id in range(len(self.coordinate_mapping)):
            vertex_id = str(node_id)
            
            if not self.graph_manager.has_vertex(vertex_id):
                continue
            
            # Get all outgoing edges
            neighbors = self.graph_manager.get_neighbors(vertex_id)
            
            for edge in neighbors:
                target_id = int(edge.target)
                
                # Determine color based on whether edge is blocked
                if (node_id, target_id) in blocked_set:
                    color = 'red'
                    linewidth = 1.5
                    alpha = 0.8
                else:
                    color = 'gray'
                    linewidth = 0.5
                    alpha = 0.6
                
                # Plot edge
                self._plot_edge(ax, node_id, target_id, color=color, 
                              linewidth=linewidth, alpha=alpha)
        
        # Set equal aspect ratio to prevent distortion
        ax.set_aspect('equal')
        
        # Labels
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_title('Road Network')
        
        # Grid
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def _plot_path(self, ax, path: List[int], color: str = 'blue', 
                  linewidth: float = 2.0, linestyle: str = '-', label: Optional[str] = None) -> None:
        """
        Plot a path as connected edges.
        
        Args:
            ax: Matplotlib axis
            path: List of node IDs forming the path
            color: Path color
            linewidth: Line width
            linestyle: Line style
            label: Legend label
        """
        if len(path) < 2:
            return
        
        # Plot each edge in the path
        for i in range(len(path) - 1):
            u = path[i]
            v = path[i + 1]
            
            # Convert string IDs to integers for coordinate lookup
            u_int = int(u) if isinstance(u, str) else u
            v_int = int(v) if isinstance(v, str) else v
            
            # Only add label to first edge
            if i == 0 and label:
                lat1, lon1 = self.coordinate_mapping[u_int]
                lat2, lon2 = self.coordinate_mapping[v_int]
                ax.plot([lon1, lon2], [lat1, lat2], 
                       color=color, linewidth=linewidth, linestyle=linestyle, 
                       label=label, alpha=0.9)
            else:
                self._plot_edge(ax, u, v, color=color, linewidth=linewidth, 
                              linestyle=linestyle, alpha=0.9)
    
    def plot_route_comparison(self, source: int, target: int,
                             normal_path: List[int], normal_distance: float,
                             disaster_path: List[int], disaster_distance: float,
                             blocked_edges: Optional[List[Tuple[int, int]]] = None,
                             figsize: Tuple[int, int] = (14, 12)) -> None:
        """
        Plot comparison of normal route vs disaster-aware route.
        
        Args:
            source, target: Start and end node IDs
            normal_path: Path computed without disaster
            normal_distance: Total distance of normal path
            disaster_path: Path computed with disaster
            disaster_distance: Total distance of disaster path
            blocked_edges: List of blocked edges to show in red
        
        Effect:
            Displays road network with:
            - Gray: normal roads
            - Red: blocked roads
            - Blue dashed: normal route
            - Green solid: disaster-aware route
            - Markers for source (green) and target (red)
            - Legend with distance comparison
        """
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Convert blocked edges to set
        blocked_set = set(blocked_edges) if blocked_edges else set()
        
        # Plot all edges (road network)
        for node_id in range(len(self.coordinate_mapping)):
            vertex_id = str(node_id)
            
            if not self.graph_manager.has_vertex(vertex_id):
                continue
            
            neighbors = self.graph_manager.get_neighbors(vertex_id)
            
            for edge in neighbors:
                target_id = int(edge.target)
                
                # Determine color
                if (node_id, target_id) in blocked_set:
                    color = 'red'
                    linewidth = 1.5
                    alpha = 0.7
                else:
                    color = 'lightgray'
                    linewidth = 0.5
                    alpha = 0.4
                
                self._plot_edge(ax, node_id, target_id, color=color, 
                              linewidth=linewidth, alpha=alpha)
        
        # Plot normal route (blue dashed)
        if normal_path and len(normal_path) >= 2:
            self._plot_path(ax, normal_path, color='blue', linewidth=2.5, 
                          linestyle='--', label=f'Normal Route ({normal_distance:.1f}m)')
        
        # Plot disaster-aware route (green solid)
        if disaster_path and len(disaster_path) >= 2:
            self._plot_path(ax, disaster_path, color='green', linewidth=2.5, 
                          linestyle='-', label=f'Disaster-Aware Route ({disaster_distance:.1f}m)')
        
        # Add markers for source and target
        if source in self.coordinate_mapping:
            lat_s, lon_s = self.coordinate_mapping[source]
            ax.plot(lon_s, lat_s, 'go', markersize=12, label='Source', zorder=10)
        
        if target in self.coordinate_mapping:
            lat_t, lon_t = self.coordinate_mapping[target]
            ax.plot(lon_t, lat_t, 'ro', markersize=12, label='Target', zorder=10)
        
        # Set equal aspect ratio
        ax.set_aspect('equal')
        
        # Labels and title
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        
        # Calculate distance difference
        distance_diff = disaster_distance - normal_distance
        diff_percent = (distance_diff / normal_distance * 100) if normal_distance > 0 else 0
        
        ax.set_title(f'Route Comparison\n'
                    f'Distance Increase: {distance_diff:.1f}m ({diff_percent:.1f}%)')
        
        # Legend
        ax.legend(loc='best', fontsize=10)
        
        # Grid
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def __str__(self) -> str:
        """String representation of the visualizer."""
        return f"MapVisualizer(nodes={len(self.coordinate_mapping)})"
    
    def __repr__(self) -> str:
        return self.__str__()
