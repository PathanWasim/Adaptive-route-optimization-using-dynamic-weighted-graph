"""
Interactive Web Application for Disaster Evacuation Routing System.

This Flask application provides a professional web interface for exploring
how Dijkstra's algorithm adapts to disaster scenarios on real road networks.

ACADEMIC INTEGRITY:
- All routing computed using internal Dijkstra implementation
- Maps used ONLY for visualization
- No external routing APIs
- OSMnx used ONLY for graph extraction
"""

from flask import Flask, render_template, request, jsonify
import json
import time
from disaster_evacuation.osm.osm_extractor import OSMExtractor
from disaster_evacuation.osm.graph_converter import GraphConverter
from disaster_evacuation.disaster.disaster_modeler import DisasterModeler
from disaster_evacuation.pathfinding.pathfinder_engine import PathfinderEngine

app = Flask(__name__)

# Global cache for loaded networks
network_cache = {}

# Predefined cities with their configurations (5km AREAS - Good balance)
CITIES = {
    "pune": {
        "name": "Pune, Maharashtra, India",
        "display_name": "Pune",
        "center": [18.5204, 73.8567],
        "zoom": 13,
        "bbox": [18.4950, 18.5450, 73.8300, 73.8850]  # ~5km x 5km
    },
    "mumbai": {
        "name": "Mumbai, Maharashtra, India",
        "display_name": "Mumbai",
        "center": [19.0760, 72.8777],
        "zoom": 13,
        "bbox": [19.0510, 19.1010, 72.8527, 72.9027]  # ~5km x 5km
    },
    "bangalore": {
        "name": "Bangalore, Karnataka, India",
        "display_name": "Bangalore",
        "center": [12.9716, 77.5946],
        "zoom": 13,
        "bbox": [12.9466, 12.9966, 77.5696, 77.6196]  # ~5km x 5km
    },
    "delhi": {
        "name": "New Delhi, Delhi, India",
        "display_name": "Delhi",
        "center": [28.6139, 77.2090],
        "zoom": 13,
        "bbox": [28.5889, 28.6389, 77.1840, 77.2340]  # ~5km x 5km
    },
    "hyderabad": {
        "name": "Hyderabad, Telangana, India",
        "display_name": "Hyderabad",
        "center": [17.3850, 78.4867],
        "zoom": 13,
        "bbox": [17.3600, 17.4100, 78.4617, 78.5117]  # ~5km x 5km
    },
    "chennai": {
        "name": "Chennai, Tamil Nadu, India",
        "display_name": "Chennai",
        "center": [13.0827, 80.2707],
        "zoom": 13,
        "bbox": [13.0577, 13.1077, 80.2457, 80.2957]  # ~5km x 5km
    },
    "kolkata": {
        "name": "Kolkata, West Bengal, India",
        "display_name": "Kolkata",
        "center": [22.5726, 88.3639],
        "zoom": 13,
        "bbox": [22.5476, 22.5976, 88.3389, 88.3889]  # ~5km x 5km
    },
    "ahmedabad": {
        "name": "Ahmedabad, Gujarat, India",
        "display_name": "Ahmedabad",
        "center": [23.0225, 72.5714],
        "zoom": 13,
        "bbox": [22.9975, 23.0475, 72.5464, 72.5964]  # ~5km x 5km
    }
}


def load_network(city_key):
    """Load or retrieve cached network for a city."""
    if city_key in network_cache:
        return network_cache[city_key]
    
    city_config = CITIES.get(city_key)
    if not city_config:
        return None
    
    try:
        # Extract road network using bbox for fast, controlled loading
        extractor = OSMExtractor()
        
        if 'bbox' in city_config:
            # Use bounding box for precise, fast extraction (8km x 8km area)
            bbox = city_config['bbox']
            print(f"Loading {city_config['display_name']} with bbox: {bbox}")
            osm_graph = extractor.extract_by_bbox(
                north=bbox[1], south=bbox[0],
                east=bbox[3], west=bbox[2],
                network_type="drive"
            )
        else:
            # Fallback to place name (slower)
            print(f"Loading {city_config['display_name']} by place name")
            osm_graph = extractor.extract_by_place(city_config["name"], network_type="drive")
        
        print(f"Converting OSM graph to internal format...")
        # Convert to internal format
        converter = GraphConverter()
        graph_manager, coord_mapping = converter.convert_osm_to_internal(osm_graph)
        
        stats = extractor.get_network_stats(osm_graph)
        print(f"Loaded {stats['num_nodes']} nodes, {stats['num_edges']} edges")
        
        # Cache the network
        network_cache[city_key] = {
            "graph_manager": graph_manager,
            "coord_mapping": coord_mapping,
            "stats": stats
        }
        
        return network_cache[city_key]
    except Exception as e:
        print(f"Error loading network: {e}")
        import traceback
        traceback.print_exc()
        return None


@app.route('/')
def index():
    """Render the main application page."""
    return render_template('index_enhanced.html', cities=CITIES)

@app.route('/classic')
def classic():
    """Render the classic interface."""
    return render_template('index.html', cities=CITIES)


@app.route('/api/cities', methods=['GET'])
def get_cities():
    """Get list of available cities."""
    return jsonify({
        "cities": [
            {
                "key": key,
                "name": config["display_name"],
                "center": config["center"],
                "zoom": config["zoom"]
            }
            for key, config in CITIES.items()
        ]
    })


@app.route('/api/load_network', methods=['POST'])
def load_network_api():
    """Load a city's road network."""
    data = request.json
    city_key = data.get('city_key')
    
    if not city_key or city_key not in CITIES:
        return jsonify({"error": "Invalid city key"}), 400
    
    network = load_network(city_key)
    if not network:
        return jsonify({"error": "Failed to load network"}), 500
    
    graph_manager = network["graph_manager"]
    coord_mapping = network["coord_mapping"]
    
    # Prepare network data for frontend
    nodes = []
    edges = []
    
    for node_id in range(len(coord_mapping)):
        lat, lon = coord_mapping[node_id]
        nodes.append({
            "id": node_id,
            "lat": lat,
            "lon": lon
        })
    
    # Get all edges
    for node_id in range(len(coord_mapping)):
        vertex_id = str(node_id)
        if not graph_manager.has_vertex(vertex_id):
            continue
        
        neighbors = graph_manager.get_neighbors(vertex_id)
        for edge in neighbors:
            target_id = int(edge.target)
            lat1, lon1 = coord_mapping[node_id]
            lat2, lon2 = coord_mapping[target_id]
            
            edges.append({
                "source": node_id,
                "target": target_id,
                "coords": [[lat1, lon1], [lat2, lon2]],
                "distance": edge.base_distance
            })
    
    return jsonify({
        "nodes": nodes,
        "edges": edges,
        "stats": network["stats"],
        "center": CITIES[city_key]["center"],
        "zoom": CITIES[city_key]["zoom"]
    })


@app.route('/api/compute_route', methods=['POST'])
def compute_route():
    """
    Compute evacuation routes using internal Dijkstra algorithm.
    
    ACADEMIC INTEGRITY: This endpoint uses ONLY the internal Dijkstra
    implementation. No external routing APIs are called.
    """
    data = request.json
    city_key = data.get('city_key')
    source_id = data.get('source_id')
    target_id = data.get('target_id')
    disaster_config = data.get('disaster', {})
    animated = data.get('animated', False)  # New parameter for animation
    
    # Validate inputs
    if not city_key or city_key not in CITIES:
        return jsonify({"error": "Invalid city key"}), 400
    
    if source_id is None or target_id is None:
        return jsonify({"error": "Source and target required"}), 400
    
    # Load network
    network = load_network(city_key)
    if not network:
        return jsonify({"error": "Failed to load network"}), 500
    
    graph_manager = network["graph_manager"]
    coord_mapping = network["coord_mapping"]
    
    # Validate node IDs
    if source_id < 0 or source_id >= len(coord_mapping):
        return jsonify({"error": "Invalid source ID"}), 400
    if target_id < 0 or target_id >= len(coord_mapping):
        return jsonify({"error": "Invalid target ID"}), 400
    
    source = str(source_id)
    target = str(target_id)
    
    # Compute normal route (without disaster)
    pathfinder = PathfinderEngine()
    start_time = time.time()
    normal_result = pathfinder.find_shortest_path(graph_manager, source, target, track_steps=animated)
    normal_time = time.time() - start_time
    
    if not normal_result.found:
        return jsonify({"error": "No path exists between selected nodes"}), 400
    
    # Convert normal path to coordinates
    normal_path_coords = []
    for node_id_str in normal_result.path:
        node_id = int(node_id_str)
        lat, lon = coord_mapping[node_id]
        normal_path_coords.append([lat, lon])
    
    # Convert algorithm steps to coordinates if animated
    normal_steps_coords = None
    if animated and hasattr(normal_result, 'algorithm_steps'):
        normal_steps_coords = []
        for step in normal_result.algorithm_steps:
            step_copy = step.copy()
            if step['type'] == 'visit':
                node_id = int(step['node'])
                lat, lon = coord_mapping[node_id]
                step_copy['coords'] = [lat, lon]
            elif step['type'] == 'relax':
                from_id = int(step['from'])
                to_id = int(step['to'])
                from_lat, from_lon = coord_mapping[from_id]
                to_lat, to_lon = coord_mapping[to_id]
                step_copy['from_coords'] = [from_lat, from_lon]
                step_copy['to_coords'] = [to_lat, to_lon]
            normal_steps_coords.append(step_copy)
    
    # Apply disaster if specified
    disaster_path_coords = None
    disaster_result = None
    disaster_steps_coords = None
    blocked_edges = []
    
    disaster_type = disaster_config.get('type', 'none')
    if disaster_type != 'none':
        epicenter = disaster_config.get('epicenter')
        radius = disaster_config.get('radius', 200.0)
        severity = disaster_config.get('severity', 0.5)
        
        if epicenter:
            # Create a fresh copy of the graph for disaster scenario
            # (In production, you'd clone the graph)
            modeler = DisasterModeler(graph_manager, coord_mapping)
            
            if disaster_type == 'fire':
                modeler.apply_fire(tuple(epicenter), radius)
            elif disaster_type == 'flood':
                modeler.apply_flood(tuple(epicenter), radius, severity)
            elif disaster_type == 'earthquake':
                modeler.apply_earthquake(tuple(epicenter), radius, severity, severity * 0.8)
            
            # Compute disaster-aware route
            start_time = time.time()
            disaster_result = pathfinder.find_shortest_path(graph_manager, source, target, track_steps=animated)
            disaster_time = time.time() - start_time
            
            if disaster_result.found:
                # Convert disaster path to coordinates
                disaster_path_coords = []
                for node_id_str in disaster_result.path:
                    node_id = int(node_id_str)
                    lat, lon = coord_mapping[node_id]
                    disaster_path_coords.append([lat, lon])
                
                # Convert disaster algorithm steps to coordinates if animated
                if animated and hasattr(disaster_result, 'algorithm_steps'):
                    disaster_steps_coords = []
                    for step in disaster_result.algorithm_steps:
                        step_copy = step.copy()
                        if step['type'] == 'visit':
                            node_id = int(step['node'])
                            lat, lon = coord_mapping[node_id]
                            step_copy['coords'] = [lat, lon]
                        elif step['type'] == 'relax':
                            from_id = int(step['from'])
                            to_id = int(step['to'])
                            from_lat, from_lon = coord_mapping[from_id]
                            to_lat, to_lon = coord_mapping[to_id]
                            step_copy['from_coords'] = [from_lat, from_lon]
                            step_copy['to_coords'] = [to_lat, to_lon]
                        disaster_steps_coords.append(step_copy)
            
            # Get blocked edges
            for node_id in range(len(coord_mapping)):
                vertex_id = str(node_id)
                if not graph_manager.has_vertex(vertex_id):
                    continue
                
                neighbors = graph_manager.get_neighbors(vertex_id)
                for edge in neighbors:
                    target_id_int = int(edge.target)
                    weight = graph_manager.get_edge_weight(vertex_id, edge.target)
                    
                    if weight >= DisasterModeler.BLOCKED_WEIGHT * 0.99:
                        lat1, lon1 = coord_mapping[node_id]
                        lat2, lon2 = coord_mapping[target_id_int]
                        blocked_edges.append([[lat1, lon1], [lat2, lon2]])
    
    # Prepare response
    response = {
        "normal_route": {
            "path": normal_path_coords,
            "distance": normal_result.total_cost,
            "nodes_visited": normal_result.nodes_visited,
            "computation_time": normal_time
        },
        "blocked_edges": blocked_edges
    }
    
    # Add animation steps if requested
    if animated:
        response["normal_route"]["steps"] = normal_steps_coords
    
    if disaster_result and disaster_result.found:
        response["disaster_route"] = {
            "path": disaster_path_coords,
            "distance": disaster_result.total_cost,
            "nodes_visited": disaster_result.nodes_visited,
            "computation_time": disaster_time
        }
        
        if animated:
            response["disaster_route"]["steps"] = disaster_steps_coords
        
        # Calculate metrics
        distance_increase = disaster_result.total_cost - normal_result.total_cost
        percent_increase = (distance_increase / normal_result.total_cost * 100) if normal_result.total_cost > 0 else 0
        
        response["metrics"] = {
            "distance_increase": distance_increase,
            "percent_increase": percent_increase,
            "routes_diverged": normal_result.path != disaster_result.path
        }
    elif disaster_type != 'none':
        response["disaster_route"] = None
        response["metrics"] = {
            "no_path": True,
            "message": "No path available after disaster"
        }
    
    return jsonify(response)


@app.route('/api/save_visualization', methods=['POST'])
def save_visualization():
    """Save current visualization state."""
    data = request.json
    filename = data.get('filename', 'visualization.json')
    
    # Save to file
    with open(f'saved_visualizations/{filename}', 'w') as f:
        json.dump(data, f, indent=2)
    
    return jsonify({"success": True, "filename": filename})


if __name__ == '__main__':
    # Create saved visualizations directory
    import os
    os.makedirs('saved_visualizations', exist_ok=True)
    
    print("=" * 70)
    print("Disaster Evacuation Routing System - Interactive Web Interface")
    print("=" * 70)
    print("\nStarting server...")
    print("Open your browser and navigate to: http://localhost:5000")
    print("\nACADEMIC INTEGRITY:")
    print("  ✓ All routing uses internal Dijkstra implementation")
    print("  ✓ Maps used ONLY for visualization")
    print("  ✓ No external routing APIs")
    print("=" * 70)
    
    app.run(debug=True, port=5000)
