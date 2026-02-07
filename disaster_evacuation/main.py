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
        self.controller = RouteController(self.graph)
        
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
            severity: Disaster severity (0.0 to 1.0)
            radius: Maximum effect radius
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Convert string to DisasterType enum
            disaster_map = {
                'flood': DisasterType.FLOOD,
                'fire': DisasterType.FIRE,
                'earthquake': DisasterType.EARTHQUAKE
            }
            
            if disaster_type.lower() not in disaster_map:
                print(f"Error: Invalid disaster type '{disaster_type}'")
                print("Valid types: flood, fire, earthquake")
                return False
            
            disaster_enum = disaster_map[disaster_type.lower()]
            
            # Create disaster event
            disaster_event = DisasterEvent(
                disaster_type=disaster_enum,
                epicenter=epicenter,
                severity=severity,
                max_effect_radius=radius
            )
            
            # Apply disaster effects to graph
            self.disaster_model.apply_disaster_effects(self.graph, disaster_event)
            
            print(f"Applied {disaster_type} disaster at {epicenter} with severity {severity}")
            return True
            
        except Exception as e:
            print(f"Error applying disaster: {e}")
            return False
    
    def compute_route(self, start: str, destination: str, 
                     show_visualization: bool = True) -> Optional[dict]:
        """
        Compute evacuation route from start to destination.
        
        Args:
            start: Starting vertex ID
            destination: Destination vertex ID
            show_visualization: Whether to display visualization
            
        Returns:
            Dictionary with route information, or None if failed
        """
        try:
            # Use controller to compute route
            result = self.controller.compute_route(start, destination)
            
            if result:
                print(f"\nRoute from {start} to {destination}:")
                print(f"Path: {' -> '.join(result['path'])}")
                print(f"Total cost: {result['total_cost']:.2f}")
                
                if result.get('blocked_roads'):
                    print(f"Avoided blocked roads: {len(result['blocked_roads'])}")
                
                # Visualize if requested
                if show_visualization:
                    self.visualization.visualize_path(
                        self.graph, result['path'], 
                        title=f"Evacuation Route: {start} to {destination}"
                    )
                
                return result
            else:
                print(f"No route found from {start} to {destination}")
                return None
                
        except Exception as e:
            print(f"Error computing route: {e}")
            return None
    
    def compare_routes(self, start: str, destination: str,
                      show_visualization: bool = True) -> Optional[dict]:
        """
        Compare normal and disaster-aware routes.
        
        Args:
            start: Starting vertex ID
            destination: Destination vertex ID
            show_visualization: Whether to display visualization
            
        Returns:
            Dictionary with comparison results, or None if failed
        """
        try:
            # Use analysis engine for comparison
            comparison = self.analysis.compare_paths(
                self.graph, self.pathfinder, start, destination
            )
            
            if comparison:
                print(f"\nRoute Comparison: {start} to {destination}")
                print(f"\nNormal shortest path:")
                print(f"  Path: {' -> '.join(comparison['normal_path'])}")
                print(f"  Cost: {comparison['normal_cost']:.2f}")
                
                print(f"\nDisaster-aware path:")
                print(f"  Path: {' -> '.join(comparison['disaster_aware_path'])}")
                print(f"  Cost: {comparison['disaster_aware_cost']:.2f}")
                
                print(f"\nCost difference: {comparison['cost_difference']:.2f}")
                print(f"Safety improvement: {comparison.get('safety_improvement', 'N/A')}")
                
                # Visualize comparison if requested
                if show_visualization:
                    self.visualization.visualize_path_comparison(
                        self.graph,
                        comparison['normal_path'],
                        comparison['disaster_aware_path'],
                        title="Route Comparison: Normal vs Disaster-Aware"
                    )
                
                return comparison
            else:
                print("Unable to compare routes")
                return None
                
        except Exception as e:
            print(f"Error comparing routes: {e}")
            return None
    
    def run_interactive(self) -> None:
        """Run the application in interactive mode."""
        print("=" * 60)
        print("Disaster Evacuation Route Optimization System")
        print("=" * 60)
        
        while True:
            print("\nOptions:")
            print("1. Create sample graph")
            print("2. Load graph from configuration")
            print("3. Apply disaster")
            print("4. Compute evacuation route")
            print("5. Compare routes (normal vs disaster-aware)")
            print("6. Visualize current graph")
            print("7. Save graph configuration")
            print("8. Exit")
            
            choice = input("\nEnter choice (1-8): ").strip()
            
            if choice == '1':
                self.create_sample_graph()
                
            elif choice == '2':
                if not self.config_manager:
                    config_dir = input("Enter configuration directory: ").strip()
                    self.config_manager = ConfigurationManager(config_dir)
                
                config_name = input("Enter configuration name: ").strip()
                self.load_graph_from_config(config_name)
                
            elif choice == '3':
                disaster_type = input("Enter disaster type (flood/fire/earthquake): ").strip()
                try:
                    x = float(input("Enter epicenter X coordinate: ").strip())
                    y = float(input("Enter epicenter Y coordinate: ").strip())
                    severity = float(input("Enter severity (0.0-1.0): ").strip())
                    radius = float(input("Enter effect radius: ").strip())
                    self.apply_disaster(disaster_type, (x, y), severity, radius)
                except ValueError:
                    print("Error: Invalid numeric input")
                    
            elif choice == '4':
                start = input("Enter start location: ").strip()
                destination = input("Enter destination: ").strip()
                self.compute_route(start, destination)
                
            elif choice == '5':
                start = input("Enter start location: ").strip()
                destination = input("Enter destination: ").strip()
                self.compare_routes(start, destination)
                
            elif choice == '6':
                self.visualization.visualize_graph(
                    self.graph, title="Current Graph State"
                )
                
            elif choice == '7':
                if not self.config_manager:
                    config_dir = input("Enter configuration directory: ").strip()
                    self.config_manager = ConfigurationManager(config_dir)
                
                config_name = input("Enter configuration name: ").strip()
                self.save_graph_to_config(config_name)
                
            elif choice == '8':
                print("Exiting...")
                break
                
            else:
                print("Invalid choice. Please enter 1-8.")


def create_cli_parser() -> argparse.ArgumentParser:
    """Create command-line argument parser."""
    parser = argparse.ArgumentParser(
        description='Disaster Evacuation Route Optimization System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode
  python -m disaster_evacuation.main --interactive
  
  # Compute route with sample graph
  python -m disaster_evacuation.main --sample --route Home Evac_Point
  
  # Apply disaster and compute route
  python -m disaster_evacuation.main --sample --disaster fire 3.0 2.0 0.8 5.0 --route Home Evac_Point
  
  # Compare routes
  python -m disaster_evacuation.main --sample --disaster flood 2.0 1.0 0.6 4.0 --compare Home Evac_Point
        """
    )
    
    parser.add_argument('--interactive', '-i', action='store_true',
                       help='Run in interactive mode')
    parser.add_argument('--sample', '-s', action='store_true',
                       help='Create sample graph')
    parser.add_argument('--config', '-c', type=str,
                       help='Configuration directory')
    parser.add_argument('--load', '-l', type=str,
                       help='Load graph configuration by name')
    parser.add_argument('--disaster', '-d', nargs=5, metavar=('TYPE', 'X', 'Y', 'SEVERITY', 'RADIUS'),
                       help='Apply disaster: type x y severity radius')
    parser.add_argument('--route', '-r', nargs=2, metavar=('START', 'DEST'),
                       help='Compute route from start to destination')
    parser.add_argument('--compare', nargs=2, metavar=('START', 'DEST'),
                       help='Compare normal and disaster-aware routes')
    parser.add_argument('--no-viz', action='store_true',
                       help='Disable visualization')
    parser.add_argument('--save', type=str,
                       help='Save graph configuration with given name')
    
    return parser


def main():
    """Main entry point for the application."""
    parser = create_cli_parser()
    args = parser.parse_args()
    
    # Initialize application
    app = DisasterEvacuationApp(config_dir=args.config)
    
    # Interactive mode
    if args.interactive:
        app.run_interactive()
        return
    
    # Create sample graph if requested
    if args.sample:
        app.create_sample_graph()
    
    # Load configuration if specified
    if args.load:
        if not app.load_graph_from_config(args.load):
            print("Failed to load configuration")
            sys.exit(1)
    
    # Check if graph is initialized
    if app.graph.get_vertex_count() == 0:
        print("Error: No graph loaded. Use --sample or --load to create/load a graph.")
        sys.exit(1)
    
    # Apply disaster if specified
    if args.disaster:
        disaster_type, x, y, severity, radius = args.disaster
        try:
            x, y = float(x), float(y)
            severity, radius = float(severity), float(radius)
            if not app.apply_disaster(disaster_type, (x, y), severity, radius):
                sys.exit(1)
        except ValueError:
            print("Error: Invalid numeric values for disaster parameters")
            sys.exit(1)
    
    # Compute route if specified
    if args.route:
        start, dest = args.route
        result = app.compute_route(start, dest, show_visualization=not args.no_viz)
        if not result:
            sys.exit(1)
    
    # Compare routes if specified
    if args.compare:
        start, dest = args.compare
        result = app.compare_routes(start, dest, show_visualization=not args.no_viz)
        if not result:
            sys.exit(1)
    
    # Save configuration if specified
    if args.save:
        if not app.save_graph_to_config(args.save):
            sys.exit(1)
    
    # If no action specified, show help
    if not any([args.sample, args.load, args.route, args.compare]):
        parser.print_help()


if __name__ == '__main__':
    main()