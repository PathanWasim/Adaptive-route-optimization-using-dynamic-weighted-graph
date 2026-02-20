"""
Interactive Disaster Evacuation Routing Application

Features:
- Choose different cities from OpenStreetMap
- Select disaster types and parameters
- Pick source and destination points interactively
- Animated route finding with algorithm visualization
- Compare multiple routes
- Save visualizations to files
"""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
import time
import os
from datetime import datetime
from disaster_evacuation.osm.osm_extractor import OSMExtractor
from disaster_evacuation.osm.graph_converter import GraphConverter
from disaster_evacuation.models.disaster_modeler import DisasterModeler
from disaster_evacuation.routing.dijkstra import PathfinderEngine
from disaster_evacuation.visualization.map_visualizer import MapVisualizer


class InteractiveEvacuationApp:
    """Interactive application with animated route finding."""
    
    def __init__(self):
        self.extractor = OSMExtractor()
        self.converter = GraphConverter()
        self.pathfinder = PathfinderEngine()
        
        self.osm_graph = None
        self.graph_manager = None
        self.coord_mapping = None
        self.visualizer = None
        self.modeler = None
        
        self.current_city = None
        self.stats = None
        
        # Create output directory for saved visualizations
        self.output_dir = "evacuation_routes"
        os.makedirs(self.output_dir, exist_ok=True)
    
    def show_menu(self):
        """Display main menu."""
        print("\n" + "=" * 70)
        print("🚨 INTERACTIVE DISASTER EVACUATION ROUTING SYSTEM 🚨")
        print("=" * 70)
        print("\n📍 MAIN MENU:")
        print("  1. Load a city from OpenStreetMap")
        print("  2. Apply disaster scenario")
        print("  3. Find evacuation route (animated)")
        print("  4. Compare multiple routes")
        print("  5. Interactive node selection (click on map)")
        print("  6. View current network")
        print("  7. Save current visualization")
        print("  8. Reset disasters")
        print("  9. Exit")
        print("=" * 70)
    
    def load_city(self):
        """Load a city from OpenStreetMap."""
        print("\n📍 LOAD CITY FROM OPENSTREETMAP")
        print("-" * 70)
        print("\nPopular small cities (recommended):")
        print("  • Piedmont, California, USA")
        print("  • Berkeley, California, USA")
        print("  • Palo Alto, California, USA")
        print("  • Cambridge, Massachusetts, USA")
        print("  • Santa Monica, California, USA")
        print("\nOr enter any city name in format: 'City, State, Country'")
        print("-" * 70)
        
        city_name = input("\n🏙️  Enter city name: ").strip()
        
        if not city_name:
            print("❌ No city name provided")
            return False
        
        print(f"\n⏳ Extracting road network for {city_name}...")
        print("   (This may take 10-30 seconds depending on city size)")
        
        try:
            self.osm_graph = self.extractor.extract_by_place(city_name, network_type="drive")
            self.stats = self.extractor.get_network_stats(self.osm_graph)
            
            print(f"\n✅ Successfully loaded {city_name}!")
            print(f"   📊 Network Statistics:")
            print(f"      • Nodes (intersections): {self.stats['num_nodes']}")
            print(f"      • Edges (road segments): {self.stats['num_edges']}")
            print(f"      • Average edge length: {self.stats['avg_edge_length']:.1f} meters")
            
            # Convert to internal format
            print("\n⏳ Converting to internal format...")
            self.graph_manager, self.coord_mapping = self.converter.convert_osm_to_internal(self.osm_graph)
            self.visualizer = MapVisualizer(self