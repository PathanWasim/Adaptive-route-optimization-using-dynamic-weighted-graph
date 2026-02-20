"""
Risk Heatmap Visualizer

Generates continuous risk intensity values for points in the network
based on proximity to a disaster epicenter and severity.
"""

from typing import List, Dict, Optional
from ..models import GraphManager, DisasterEvent
from ..models import WeightCalculator

class HeatmapGenerator:
    """Generates continuous heatmap data points for visual overlay."""
    
    @staticmethod
    def generate_heatmap_data(graph_manager: GraphManager, disaster: DisasterEvent) -> List[Dict]:
        """
        Generate a list of intensity points for heatmap rendering.
        
        Args:
            graph_manager: The graph containing nodes to sample.
            disaster: The disaster event causing the risk.
            
        Returns:
            List of dictionaries containing lat, lon, and intensity.
        """
        if disaster is None:
            return []
            
        heatmap_data = []
        
        # Sample points across all nodes
        for vid in graph_manager.get_vertex_ids():
            coords = graph_manager.get_node_coordinates(vid)
            if not coords:
                continue
                
            dist_to_center = disaster.distance_to_point(coords)
            
            # If outside the effect radius, intensity is 0
            if dist_to_center > disaster.max_effect_radius:
                continue
                
            # Proximity factor (squared for exponential drop-off as in weights)
            proximity = max(0.0, 1.0 - (dist_to_center / disaster.max_effect_radius))
            intensity = proximity * disaster.severity
            
            # Add some base scaling based on disaster type for visual impact
            if disaster.disaster_type.value == 'fire':
                intensity *= 1.2
            elif disaster.disaster_type.value == 'earthquake':
                intensity *= 1.1
            
            # Cap at 1.0 (Leaflet heatmaps usually scale 0.0 to 1.0)
            intensity = min(1.0, intensity)
            
            # Only send points with meaningful intensity
            if intensity > 0.05:
                heatmap_data.append({
                    "lat": coords[0],
                    "lng": coords[1],
                    "intensity": intensity
                })
                
        # Also sample along edges for more continuous visualization?
        # A bit heavier payload, but might make the map look better.
        # For performance, just keeping node points is usually enough for leaflet.heat blur.
        
        return heatmap_data
