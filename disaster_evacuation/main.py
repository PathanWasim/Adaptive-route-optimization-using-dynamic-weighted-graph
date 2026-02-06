"""
Main application class for the Disaster Evacuation Routing System.

This module provides the main entry point and integrates all system components
including graph management, disaster modeling, pathfinding, visualization, and analysis.
"""

import argparse
import sys
from typing import Optional, Tuple
from .graph import GraphManager
from .disaster import DisasterModel
from .pathfinding import PathfinderEngine
from .visualization import VisualizationEngine
from .analysis import AnalysisEngine
from .controller import RouteController
from .config import ConfigurationManager
from .models import DisasterType, VertexType, DisasterEvent


class DisasterEvacuationApp:
    """
    Main application class integrating all system components.
    
    This class provides a complete workflow from input to visualization,
    coordinating between graph management, disaster modeling, pathfinding,
    and visualization components.
    """
    
    def __init__(self, config_dir: Optional[str] = None):
        """
        Initialize the disaster evacuation application.
        
        Args:
            config_dir: Directory for configuration files (optional)
        """
        # Initialize core components
        self.graph = GraphManager()
        self.disaster_model = DisasterModel()
        self.pathfinder = PathfinderEngine()
        self.visualization = VisualizationEngine()
        self.analysis = AnalysisEngine()
        self.controller = RouteController(
            self.graph, self.disaster_model, self.pathfinder
        )
        
        # Initialize configuration manager if directory provided
        self.config_manager = None
        if config_dir:
            self.config_manager = ConfigurationManager(config_dir)
    
    def load_graph_from_config(self, config_name: str) -> bool:
        """
        Load a graph configuration from file.
        
        Args:
            config_name: Name of the configuration to load
            
        Returns:
            True if successful, False otherwise
        """
        if not self.config_manager:
            print("Error: Configuration manager not initialized")
            return False
        
        try:
            self.graph, metadata = self.config_manager.load_graph_configuration(config_name)
            print(f"Loaded graph: {metadata.get('name', config_name)}")
            print(f"Vertices: {self.graph.get_vertex_count()}, Edges: {self.graph.get_edge_count()}")
            return True
        except Exception as e:
            print(f"Error loading configuration: {e}")
            return False
    
    def save_graph_to_config(self, config_name: str, metadata: Optional[dict] = None) -> bool:
        """
        Save current graph configuration to file.
        
        Args:
            config_name: Name for the configuration
            metadata: Optional metadata dictionary
            
        Returns:
            True if successful, False otherwise
        """
        if not self.config_manager:
            print("Error: Configuration manager not initialized")
            return False
        
        try:
            self.config_manager.save_graph_configuration(self.graph, config_name, metadata)
            print(f"Saved graph configuration: {config_name}")
            return True
        except Exception as e:
            print(f"Error saving configuration: {e}")
            return False
    
    def create_sample_graph(self) -> None:
        """Create a sample graph for demonstration purposes."""
        # Add vertices representing city locations
        self.graph.add_vertex("Home", VertexType.INTERSECTION, (0.0, 0.0))
        self.graph.add_vertex("School", VertexType.INTERSECTION, (2.0, 1.0))
        self.graph.add_vertex("Hospital", VertexType.INTERSECTION, (4.0, 0.0))
        self.graph.add_vertex("Park", VertexType.INTERSECTION, (3.0, 3.0))
        self.graph.add_vertex("Shelter_A", VertexType.SHELTER, (6.0, 1.0), capacity=500)
        self.graph.add_vertex("Shelter_B", VertexType.SHELTER, (5.0, 4.0), capacity=300)
        self.graph.add_vertex("Evac_Point", VertexType.EVACUATION_POINT, (8.0, 2.0), capacity=1000)
        
        # Add edges with distance, risk, and congestion
        self.graph.add_edge("Home", "School", 2.2, 0.1, 0.3)
        self.graph.add_edge("Home", "Hospital", 4.0, 0.2, 0.2)
        self.graph.add_edge("School", "Hospital", 2.8, 0.15, 0.25)
        self.graph.add_edge("School", "Park", 2.2, 0.1, 0.2)
        self.graph.add_edge("Hospital", "Shelter_A", 2.8, 0.1, 0.3)
        self.graph.add_edge("Park", "Shelter_B", 2.8, 0.2, 0.25)
        self.graph.add_edge("Shelter_A", "Evac_Point", 2.8, 0.05, 0.15)
        self.graph.add_edge("Shelter_B", "Evac_Point", 4.2, 0.1, 0.2)
        self.graph.add_edge("Shelter_A", "Shelter_B", 3.6, 0.15, 0.2)
        
        print("Created sample graph with 7 vertices and 9 edges")
    
    def apply_disaster(self, disaster_type: str, epicenter: Tuple[float, float],
                      severity: float, radius: float) -> bool:
        """
        Apply a disaster to the current graph.
        
        Args:
            disaster_type: Type of disaster (flood, fire, earthquake)
            epicenter: Disaster epicenter coordinates
  