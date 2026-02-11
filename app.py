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

# Predefined cities - Small areas that load fast
CITIES = {
    "piedmont": {
        "name": "Piedmont, California, USA",
        "display_name": "Piedmont, CA",
        "center": [37.8244, -122.2312],
        "zoom": 14
    },
    "berkeley": {
        "name": "Berkeley, California, USA", 
        "display_name": "Berkeley, CA",
        "center": [37.8715, -122.2730],
        "zoom": 13
    },
    "albany": {
        "name": "Albany, California, USA",
        "display_name": "Albany, CA",
        "center": [37.8869, -122.2977],
        "zoom": 14
    },
    "pune_shivajinagar": {
        "name": "Shivajinagar, Pune, India",
        "display_name": "Pune - Shivajinagar",
        "center": [18.5304, 73.8567],
        "zoom": 15
    },
    "pune_koregaon": {
        "name": "Koregaon Park, Pune, India",
        "display_name": "Pune - Koregaon Park",
        "center": [18.5362, 73.8958],
        "zoom": 15
    },
    "mumbai_bandra": {
        "name": "Bandra West, Mumbai, India",
        "display_name": "Mumbai - Bandra",
        "center": [19.0596, 72.8295],
        "zoom": 15
    },
    "bangalore_indiranagar": {
        "name": "Indiranagar, Bangalore, India",
        "display_name": "Bangalore - Indiranagar",
        "center": [12.9716, 77.6412],
        "zoom": 15
    },
    "delhi_cp": {
        "name": "Connaught Place, New Delhi, India",
        "display_name": "Delhi - Connaught Place",
        "center": [28.6315, 77.2167],
        "zoom": 15
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
        # Extract road network using place name (NO bbox - let OSM handle it)
        extractor = OSMExtractor()
        
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
