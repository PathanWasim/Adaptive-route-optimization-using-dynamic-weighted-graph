"""
Interactive Web Application for Disaster Evacuation Routing System.

This Flask application provides a professional web interface for exploring
how graph algorithms (Dijkstra, A*, Bellman-Ford) adapt to disaster scenarios
on real road networks.

ACADEMIC INTEGRITY:
- All routing computed using internal algorithm implementations
- Maps used ONLY for visualization
- No external routing APIs
- OSMnx used ONLY for graph extraction
"""

from flask import Flask, render_template, request, jsonify
import json
import time
from disaster_evacuation.osm.osm_extractor import OSMExtractor
from disaster_evacuation.osm.graph_converter import GraphConverter
from disaster_evacuation.models.disaster_modeler import DisasterModeler
from disaster_evacuation.routing.dijkstra import PathfinderEngine
from disaster_evacuation.routing.astar import AStarEngine
from disaster_evacuation.routing.bellman_ford import BellmanFordEngine
from disaster_evacuation.analysis.benchmarks import BenchmarkRunner
from disaster_evacuation.visualization.heatmap import HeatmapGenerator

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
    },
    "textbook_graph": {
        "name": "Textbook Example Graph",
        "display_name": "🎓 Textbook Example (Small)",
        "center": [0.0, 0.0],  # Synthetic coordinates
        "zoom": 13,
        "is_synthetic": True
    }
}


def create_textbook_graph():
    """Create a simple, manually defined graph for educational visualization."""
    from disaster_evacuation.models.graph import GraphManager
    from disaster_evacuation.models.vertex import VertexType
    
    graph = GraphManager()
    
    # Create 6 nodes arranged in a layout easy to visualize
    # A(0,0), B(2,1), C(2,-1), D(4,1), E(4,-1), F(6,0)
    nodes = {
        "0": (0.0, 0.0),      # Source
        "1": (0.02, 0.01),    # Up
        "2": (0.02, -0.01),   # Down
        "3": (0.04, 0.01),    # Far Up
        "4": (0.04, -0.01),   # Far Down
        "5": (0.06, 0.00)     # Target
    }
    
    coord_mapping = {}
    
    for nid, (lat, lon) in nodes.items():
        graph.add_vertex(nid, VertexType.INTERSECTION, (lat, lon))
        graph.set_node_coordinates(nid, lat, lon)
        coord_mapping[int(nid)] = (lat, lon)
        
    # Add edges with clear integer-like weights (converted to approx meters)
    # Weights: A->B(4), A->C(2), B->C(1), B->D(5), C->E(8), C->D(10), D->F(6), E->F(3), D->E(2)
    edges = [
        ("0", "1", 400), ("0", "2", 200),
        ("1", "2", 100), ("1", "3", 500),
        ("2", "4", 800), ("2", "3", 1000),
        ("3", "5", 600), ("4", "5", 300),
        ("3", "4", 200)
    ]
    
    for u, v, w in edges:
        # Add bidirectional edges
        graph.add_edge(u, v, w, 0.0, 0.0)
        graph.add_edge(v, u, w, 0.0, 0.0)
        
    stats = {
        "num_nodes": 6,
        "num_edges": len(edges) * 2,
        "network_type": "synthetic"
    }
    
    return {
        "graph_manager": graph,
        "coord_mapping": coord_mapping,
        "stats": stats
    }


def load_network(city_key):
    """Load or retrieve cached network for a city."""
    if city_key in network_cache:
        return network_cache[city_key]
    
    city_config = CITIES.get(city_key)
    if not city_config:
        return None
        
    # Handle synthetic educational graph
    if city_config.get("is_synthetic"):
        print(f"Generating synthetic {city_config['display_name']}")
        network_data = create_textbook_graph()
        network_cache[city_key] = network_data
        return network_data
    
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


def _get_pathfinder(algorithm_name):
    """Get the appropriate pathfinder engine based on algorithm name."""
    from disaster_evacuation.routing import YenKShortestPaths, BidirectionalDijkstra
    
    engines = {
        'dijkstra': PathfinderEngine,
        'astar': AStarEngine,
        'bellman_ford': BellmanFordEngine,
        'yen_k_shortest': YenKShortestPaths,
        'bidirectional': BidirectionalDijkstra
    }
    engine_class = engines.get(algorithm_name, PathfinderEngine)
    return engine_class()


def _convert_steps_to_coords(steps, coord_mapping):
    """Convert algorithm step node IDs to geographic coordinates."""
    if not steps:
        return None
    converted = []
    for step in steps:
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
        converted.append(step_copy)
    return converted


def _path_to_coords(path, coord_mapping):
    """Convert a path of node IDs to geographic coordinates."""
    return [[coord_mapping[int(nid)][0], coord_mapping[int(nid)][1]] for nid in path]


@app.route('/api/compute_route', methods=['POST'])
def compute_route():
    """
    Compute evacuation routes using internal algorithm implementations.
    
    Supports algorithm selection: dijkstra (default), astar, bellman_ford.
    
    ACADEMIC INTEGRITY: This endpoint uses ONLY internal algorithm
    implementations. No external routing APIs are called.
    """
    data = request.json
    city_key = data.get('city_key')
    source_id = data.get('source_id')
    target_id = data.get('target_id')
    disaster_config = data.get('disaster', {})
    animated = data.get('animated', False)
    algorithm = data.get('algorithm', 'dijkstra')  # Algorithm selection
    compare_algorithms = data.get('compare_algorithms', False)  # Multi-algo comparison
    
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
    weights_config = data.get('weights', {})
    alpha = float(weights_config.get('alpha', 1.0))
    beta = float(weights_config.get('beta', 1.0))
    gamma = float(weights_config.get('gamma', 1.0))

    # ---------- Initialize Disaster and Weights ----------
    from disaster_evacuation.models import DisasterModel, DisasterEvent, DisasterType
    import datetime
    
    disaster_model = DisasterModel()
    # Apply baseline weights considering objective scalars
    disaster_model.apply_objective_weights(graph_manager, alpha, beta, gamma)
    
    source = str(source_id)
    target = str(target_id)
    
    # ---------- Compute normal route (without disaster) ----------
    pathfinder = _get_pathfinder(algorithm)
    k_paths = int(data.get('k_paths', 3))
    start_time = time.time()
    
    if algorithm == 'yen_k_shortest':
        normal_results = pathfinder.find_k_shortest_paths(graph_manager, source, target, k=k_paths)
        if not normal_results:
            return jsonify({"error": "No path exists between selected nodes"}), 400
        normal_result = normal_results[0]
    else:
        normal_result = pathfinder.find_shortest_path(graph_manager, source, target, track_steps=animated)
        if not normal_result.found:
            return jsonify({"error": "No path exists between selected nodes"}), 400
        normal_results = [normal_result]
        
    normal_time = time.time() - start_time
    
    normal_routes_data = []
    for res in normal_results:
        normal_routes_data.append({
            "path": _path_to_coords(res.path, coord_mapping),
            "distance": res.total_cost,
            "nodes_visited": res.nodes_visited,
            "computation_time": res.computation_time,
            "path_edges": len(res.path) - 1
        })
        
    normal_path_coords = normal_routes_data[0]["path"]
    
    normal_steps_coords = None
    if animated and hasattr(normal_result, 'algorithm_steps'):
        normal_steps_coords = _convert_steps_to_coords(normal_result.algorithm_steps, coord_mapping)
    
    # ---------- Multi-algorithm comparison ----------
    algo_comparison = None
    if compare_algorithms:
        algo_comparison = {}
        for algo_name in ['dijkstra', 'astar', 'bidirectional', 'bellman_ford']:
            engine = _get_pathfinder(algo_name)
            t0 = time.time()
            result = engine.find_shortest_path(graph_manager, source, target)
            t1 = time.time()
            stats = engine.get_algorithm_stats()
            algo_comparison[algo_name] = {
                'time_ms': (t1 - t0) * 1000,
                'nodes_visited': result.nodes_visited if result.found else 0,
                'path_cost': result.total_cost if result.found else None,
                'path_length': len(result.path) if result.found else 0,
                'edges_examined': stats.edges_examined,
                'found': result.found
            }
    
    # ---------- Apply disaster if specified ----------
    disaster_path_coords = None
    disaster_result = None
    disaster_steps_coords = None
    blocked_edges = []
    active_event = None
    
    disaster_type = disaster_config.get('type', 'none')
    if disaster_type != 'none':
        epicenter = disaster_config.get('epicenter')
        radius = float(disaster_config.get('radius', 200.0))
        severity = float(disaster_config.get('severity', 0.5))
        
        if epicenter:
            dt_enum = DisasterType.FIRE
            if disaster_type == 'flood':
                dt_enum = DisasterType.FLOOD
            elif disaster_type == 'earthquake':
                dt_enum = DisasterType.EARTHQUAKE
                
            active_event = DisasterEvent(
                disaster_type=dt_enum,
                epicenter=tuple(epicenter),
                severity=severity,
                max_effect_radius=radius,
                start_time=datetime.datetime.now()
            )
            
            # Apply disaster weights according to multi-objective scalars
            disaster_model.apply_disaster_effects(graph_manager, active_event, alpha, beta, gamma)
            
            # Compute disaster-aware route using selected algorithm
            start_time = time.time()
            if algorithm == 'yen_k_shortest':
                disaster_results = pathfinder.find_k_shortest_paths(graph_manager, source, target, k=k_paths)
                disaster_result = disaster_results[0] if disaster_results else None
            else:
                disaster_result = pathfinder.find_shortest_path(graph_manager, source, target, track_steps=animated)
                disaster_results = [disaster_result] if disaster_result.found else []
                
            disaster_time = time.time() - start_time
            
            disaster_routes_data = []
            for res in disaster_results:
                disaster_routes_data.append({
                    "path": _path_to_coords(res.path, coord_mapping),
                    "distance": res.total_cost,
                    "nodes_visited": res.nodes_visited,
                    "computation_time": res.computation_time,
                    "path_edges": len(res.path) - 1
                })
            
            if disaster_result and disaster_result.found:
                disaster_path_coords = disaster_routes_data[0]["path"]
                if animated and hasattr(disaster_result, 'algorithm_steps'):
                    disaster_steps_coords = _convert_steps_to_coords(disaster_result.algorithm_steps, coord_mapping)
            
            # Get blocked edges for visualization
            for edge in graph_manager.get_all_edges():
                if edge.is_blocked:
                    if int(edge.source) in coord_mapping and int(edge.target) in coord_mapping:
                        lat1, lon1 = coord_mapping[int(edge.source)]
                        lat2, lon2 = coord_mapping[int(edge.target)]
                        blocked_edges.append([[lat1, lon1], [lat2, lon2]])
            
            # Reset graph to normal objective weights immediately to un-pollute the cache
            disaster_model.clear_all_disaster_effects(graph_manager)
            disaster_model.apply_objective_weights(graph_manager, alpha, beta, gamma)
    
    # ---------- Prepare response ----------
    response = {
        "algorithm": algorithm,
        "k_paths": k_paths,
        "normal_routes": normal_routes_data,
        "normal_route": normal_routes_data[0], # Maintain backwards compatibility
        "blocked_edges": blocked_edges
    }
    
    if animated:
        response["normal_route"]["steps"] = normal_steps_coords
    
    if algo_comparison:
        response["algorithm_comparison"] = algo_comparison
    
    if disaster_result and disaster_result.found:
        response["disaster_routes"] = disaster_routes_data
        response["disaster_route"] = disaster_routes_data[0] # Maintain backwards compatibility
        
        if animated:
            response["disaster_route"]["steps"] = disaster_steps_coords
        
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


@app.route('/api/heatmap', methods=['POST'])
def get_heatmap():
    """Generate continuous risk heatmap data based on disaster event."""
    data = request.json
    city_key = data.get('city_key')
    disaster_data = data.get('disaster')
    
    if not city_key or not disaster_data:
        return jsonify({"error": "Missing city_key or disaster data"}), 400
        
    graph_manager = app.graphs.get(city_key)
    if not graph_manager:
        return jsonify({"error": f"Network not loaded for {city_key}"}), 400
        
    try:
        disaster_type = DisasterType(disaster_data['type'])
        epicenter = tuple(disaster_data['epicenter'])
        severity = float(disaster_data['severity'])
        radius = float(disaster_data['radius'])
        
        disaster = DisasterEvent(disaster_type, epicenter, severity, radius)
    except (ValueError, KeyError, TypeError) as e:
        return jsonify({"error": f"Invalid disaster parameters: {str(e)}"}), 400
        
    heatmap_data = HeatmapGenerator.generate_heatmap_data(graph_manager, disaster)
    return jsonify({"heatmap_data": heatmap_data})


@app.route('/api/save_visualization', methods=['POST'])
def save_visualization():
    """Save current visualization state."""
    data = request.json
    filename = data.get('filename', 'visualization.json')
    
    with open(f'saved_visualizations/{filename}', 'w') as f:
        json.dump(data, f, indent=2)
    
    return jsonify({"success": True, "filename": filename})


@app.route('/api/benchmark', methods=['POST'])
def run_benchmark():
    """
    Run comparative benchmarks across all pathfinding algorithms.
    Returns empirical complexity data for plotting.
    """
    data = request.json or {}
    sizes = data.get('sizes', [50, 100, 200, 500, 1000])
    runs = data.get('runs_per_size', 3)
    
    runner = BenchmarkRunner()
    results = runner.run_benchmarks(sizes=sizes, runs_per_size=runs)
    
    return jsonify({
        "results": results,
        "summary_table": runner.get_summary_table(),
        "chart_data": runner.get_complexity_comparison()
    })


@app.route('/api/auto_demo', methods=['GET'])
def auto_demo():
    """
    Get pre-configured demo parameters for smooth presentation.
    Returns city, source, target, and disaster config.
    """
    return jsonify({
        "city_key": "pune_shivajinagar",
        "source_id": 5,
        "target_id": 50,
        "disaster": {
            "type": "flood",
            "epicenter": [18.5304, 73.8567],
            "radius": 250,
            "severity": 0.6
        },
        "animated": True,
        "compare_algorithms": True
    })


if __name__ == '__main__':
    import os
    os.makedirs('saved_visualizations', exist_ok=True)
    
    print("=" * 70)
    print("Disaster Evacuation Routing System - Interactive Web Interface")
    print("=" * 70)
    print("\nStarting server...")
    print("Open your browser and navigate to: http://localhost:5000")
    print("\nACADEMIC INTEGRITY:")
    print("  ✓ All routing uses internal Dijkstra/A*/Bellman-Ford implementations")
    print("  ✓ Maps used ONLY for visualization")
    print("  ✓ No external routing APIs")
    print("\nALGORITHMS AVAILABLE:")
    print("  → Dijkstra's Algorithm — O(E log V)")
    print("  → A* with Haversine Heuristic — O(E log V), fewer nodes")
    print("  → Bellman-Ford — O(VE), comparative baseline")
    print("=" * 70)
    
    app.run(debug=True, port=5000)
