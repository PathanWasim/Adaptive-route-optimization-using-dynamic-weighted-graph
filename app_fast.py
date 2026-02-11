"""
FAST Interactive Web Application - Uses pre-generated test graphs.
No OSM download required - instant loading!
"""

from flask import Flask, render_template, request, jsonify
import json
import time
from disaster_evacuation.graph import GraphManager
from disaster_evacuation.disaster.disaster_modeler import DisasterModeler
from disaster_evacuation.pathfinding.pathfinder_engine import PathfinderEngine

app = Flask(__name__)

# Pre-generated test graphs (no OSM download needed!)
def create_test_graph(city_name):
    """Create a small test graph for demonstration."""
    from disaster_evacuation.models import VertexType
    
    graph = GraphManager()
    
    # Create a simple grid network (5x5 = 25 nodes)
    coords = {}
    node_id = 0
    
    # Base coordinates for each city
    city_bases = {
        "pune": (18.5204, 73.8567),
        "mumbai": (19.0760, 72.8777),
        "bangalore": (12.9716, 77.5946),
        "delhi": (28.6139, 77.2090)
    }
    
    base_lat, base_lon = city_bases.get(city_name, (18.5204, 73.8567))
    
    # Create 5x5 grid
    for i in range(5):
        for j in range(5):
            lat = base_lat + (i * 0.002)  # ~200m spacing
            lon = base_lon + (j * 0.002)
            coords[node_id] = (lat, lon)
            graph.add_vertex(str(node_id), VertexType.INTERSECTION, (lat, lon))
            node_id += 1
    
    # Add edges (grid connections)
    for i in range(5):
        for j in range(5):
            current = i * 5 + j
            # Right connection
            if j < 4:
                right = current + 1
                graph.add_edge(str(current), str(right), distance=200.0, 
                             base_risk=0.1, base_congestion=0.1)
                graph.add_edge(str(right), str(current), distance=200.0,
                             base_risk=0.1, base_congestion=0.1)
            # Down connection
            if i < 4:
                down = current + 5
                graph.add_edge(str(current), str(down), distance=200.0,
                             base_risk=0.1, base_congestion=0.1)
                graph.add_edge(str(down), str(current), distance=200.0,
                             base_risk=0.1, base_congestion=0.1)
    
    return graph, coords

# Cache for test graphs
network_cache = {}

CITIES = {
    "pune": {"name": "Pune (Test)", "center": [18.5204, 73.8567], "zoom": 15},
    "mumbai": {"name": "Mumbai (Test)", "center": [19.0760, 72.8777], "zoom": 15},
    "bangalore": {"name": "Bangalore (Test)", "center": [12.9716, 77.5946], "zoom": 15},
    "delhi": {"name": "Delhi (Test)", "center": [28.6139, 77.2090], "zoom": 15}
}

@app.route('/')
def index():
    return render_template('index_enhanced.html', cities=CITIES)

@app.route('/api/cities', methods=['GET'])
def get_cities():
    return jsonify({
        "cities": [
            {"key": key, "name": config["name"], "center": config["center"], "zoom": config["zoom"]}
            for key, config in CITIES.items()
        ]
    })

@app.route('/api/load_network', methods=['POST'])
def load_network_api():
    data = request.json
    city_key = data.get('city_key')
    
    if city_key not in CITIES:
        return jsonify({"error": "Invalid city"}), 400
    
    # Create test graph instantly
    if city_key not in network_cache:
        graph, coords = create_test_graph(city_key)
        network_cache[city_key] = {"graph_manager": graph, "coord_mapping": coords}
    
    network = network_cache[city_key]
    graph_manager = network["graph_manager"]
    coord_mapping = network["coord_mapping"]
    
    # Prepare network data
    nodes = []
    edges = []
    
    for node_id, (lat, lon) in coord_mapping.items():
        nodes.append({"id": node_id, "lat": lat, "lon": lon})
    
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
        "stats": {"num_nodes": len(nodes), "num_edges": len(edges)},
        "center": CITIES[city_key]["center"],
        "zoom": CITIES[city_key]["zoom"]
    })

@app.route('/api/compute_route', methods=['POST'])
def compute_route():
    data = request.json
    city_key = data.get('city_key')
    source_id = data.get('source_id')
    target_id = data.get('target_id')
    disaster_config = data.get('disaster', {})
    animated = data.get('animated', False)
    
    if city_key not in network_cache:
        return jsonify({"error": "Network not loaded"}), 400
    
    network = network_cache[city_key]
    graph_manager = network["graph_manager"]
    coord_mapping = network["coord_mapping"]
    
    source = str(source_id)
    target = str(target_id)
    
    # Compute normal route
    pathfinder = PathfinderEngine()
    start_time = time.time()
    normal_result = pathfinder.find_shortest_path(graph_manager, source, target, track_steps=animated)
    normal_time = time.time() - start_time
    
    if not normal_result.found:
        return jsonify({"error": "No path exists"}), 400
    
    # Convert path to coordinates
    normal_path_coords = []
    for node_id_str in normal_result.path:
        node_id = int(node_id_str)
        lat, lon = coord_mapping[node_id]
        normal_path_coords.append([lat, lon])
    
    # Convert algorithm steps if animated
    normal_steps_coords = None
    if animated and hasattr(normal_result, 'algorithm_steps'):
        normal_steps_coords = []
        for step in normal_result.algorithm_steps:
            step_copy = step.copy()
            if step['type'] == 'visit':
                node_id = int(step['node'])
                lat, lon = coord_mapping[node_id]
                step_copy['coords'] = [lat, lon]
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
            modeler = DisasterModeler(graph_manager, coord_mapping)
            
            if disaster_type == 'fire':
                modeler.apply_fire(tuple(epicenter), radius)
            elif disaster_type == 'flood':
                modeler.apply_flood(tuple(epicenter), radius, severity)
            elif disaster_type == 'earthquake':
                modeler.apply_earthquake(tuple(epicenter), radius, severity, severity * 0.8)
            
            start_time = time.time()
            disaster_result = pathfinder.find_shortest_path(graph_manager, source, target, track_steps=animated)
            disaster_time = time.time() - start_time
            
            if disaster_result.found:
                disaster_path_coords = []
                for node_id_str in disaster_result.path:
                    node_id = int(node_id_str)
                    lat, lon = coord_mapping[node_id]
                    disaster_path_coords.append([lat, lon])
                
                if animated and hasattr(disaster_result, 'algorithm_steps'):
                    disaster_steps_coords = []
                    for step in disaster_result.algorithm_steps:
                        step_copy = step.copy()
                        if step['type'] == 'visit':
                            node_id = int(step['node'])
                            lat, lon = coord_mapping[node_id]
                            step_copy['coords'] = [lat, lon]
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
    
    response = {
        "normal_route": {
            "path": normal_path_coords,
            "distance": normal_result.total_cost,
            "nodes_visited": normal_result.nodes_visited,
            "computation_time": normal_time
        },
        "blocked_edges": blocked_edges
    }
    
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
        
        distance_increase = disaster_result.total_cost - normal_result.total_cost
        percent_increase = (distance_increase / normal_result.total_cost * 100) if normal_result.total_cost > 0 else 0
        
        response["metrics"] = {
            "distance_increase": distance_increase,
            "percent_increase": percent_increase,
            "routes_diverged": normal_result.path != disaster_result.path
        }
    
    return jsonify(response)

@app.route('/api/save_visualization', methods=['POST'])
def save_visualization():
    data = request.json
    filename = data.get('filename', 'visualization.json')
    with open(f'saved_visualizations/{filename}', 'w') as f:
        json.dump(data, f, indent=2)
    return jsonify({"success": True, "filename": filename})

if __name__ == '__main__':
    import os
    os.makedirs('saved_visualizations', exist_ok=True)
    
    print("=" * 70)
    print("FAST Disaster Evacuation Routing - Test Mode")
    print("=" * 70)
    print("\nUsing pre-generated test graphs - INSTANT loading!")
    print("Open your browser: http://localhost:5000")
    print("\nNote: This uses 5x5 grid test networks, not real OSM data")
    print("=" * 70)
    
    app.run(debug=True, port=5000)
