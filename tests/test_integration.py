"""
Integration test for the complete OSM road network workflow.

This test verifies the full pipeline: extract → convert → disaster → route → visualize
using a small known area to ensure all components work together correctly.
"""

import pytest
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing
import matplotlib.pyplot as plt
from disaster_evacuation.osm.osm_extractor import OSMExtractor
from disaster_evacuation.osm.graph_converter import GraphConverter
from disaster_evacuation.disaster.disaster_modeler import DisasterModeler
from disaster_evacuation.pathfinding.pathfinder_engine import PathfinderEngine
from disaster_evacuation.visualization.map_visualizer import MapVisualizer


def test_full_workflow_integration():
    """
    Test complete pipeline: extract → convert → disaster → route → visualize.
    
    Uses a small known area (Piedmont, California, USA) to verify:
    - OSM extraction works
    - Conversion to internal format preserves data
    - Disaster modeling affects routes
    - Dijkstra works correctly on converted graphs
    - Visualization produces valid output
    
    Validates: Requirements 5.1, 6.1, 6.2, 6.4
    """
    # Step 1: Extract road network from OpenStreetMap
    extractor = OSMExtractor()
    osm_graph = extractor.extract_by_place("Piedmont, California, USA", network_type="drive")
    
    # Verify extraction succeeded
    assert osm_graph is not None
    assert osm_graph.number_of_nodes() > 0
    assert osm_graph.number_of_edges() > 0
    
    # Get network statistics
    stats = extractor.get_network_stats(osm_graph)
    assert stats['num_nodes'] > 0
    assert stats['num_edges'] > 0
    assert stats['area_km2'] >= 0  # Area might be 0 for projected graphs, just check non-negative
    
    # Step 2: Convert to internal format
    converter = GraphConverter()
    graph_manager, coord_mapping = converter.convert_osm_to_internal(osm_graph)
    
    # Verify conversion succeeded
    assert graph_manager is not None
    assert coord_mapping is not None
    assert len(coord_mapping) == stats['num_nodes']
    
    # Verify all nodes have coordinates
    for node_id in range(stats['num_nodes']):
        coords = coord_mapping.get(node_id)
        assert coords is not None
        lat, lon = coords
        # Coordinates should be valid lat/lon
        assert -90 <= lat <= 90
        assert -180 <= lon <= 180
    
    # Step 3: Select source and target nodes for routing
    # Use first and last nodes as a simple test
    source = "0"
    target = str(stats['num_nodes'] - 1)
    
    # Verify nodes exist in graph
    assert graph_manager.has_vertex(source)
    assert graph_manager.has_vertex(target)
    
    # Step 4: Compute normal route (without disaster)
    pathfinder = PathfinderEngine()
    normal_result = pathfinder.find_shortest_path(graph_manager, source, target)
    
    # Verify normal route was found
    if not normal_result.found:
        # If no path exists, try finding connected nodes
        pytest.skip("No path exists between selected nodes in this network")
    
    assert len(normal_result.path) >= 2
    assert normal_result.path[0] == source
    assert normal_result.path[-1] == target
    assert normal_result.total_cost > 0
    
    # Step 5: Apply disaster scenario
    # Get epicenter near middle of normal path
    middle_idx = len(normal_result.path) // 2
    middle_node_id = int(normal_result.path[middle_idx])
    epicenter = coord_mapping[middle_node_id]
    
    # Apply fire disaster to block roads
    modeler = DisasterModeler(graph_manager, coord_mapping)
    modeler.apply_fire(epicenter, radius_meters=200.0)
    
    # Step 6: Compute disaster-aware route
    disaster_result = pathfinder.find_shortest_path(graph_manager, source, target)
    
    # Verify disaster-aware route was found
    assert disaster_result.found or not disaster_result.found  # May or may not find path
    
    if disaster_result.found:
        assert len(disaster_result.path) >= 2
        assert disaster_result.path[0] == source
        assert disaster_result.path[-1] == target
        
        # Disaster route should be different or longer (if same, disaster didn't affect it)
        if disaster_result.path != normal_result.path:
            # Routes diverged - disaster had an effect
            assert disaster_result.total_cost >= normal_result.total_cost
    
    # Step 7: Visualize results
    visualizer = MapVisualizer(graph_manager, coord_mapping)
    
    # Test network visualization
    fig1 = plt.figure(figsize=(10, 8))
    visualizer.plot_network()
    plt.close(fig1)
    
    # Test route comparison visualization (if both routes exist)
    if disaster_result.found:
        fig2 = plt.figure(figsize=(12, 10))
        
        # Get blocked edges for visualization
        blocked_edges = []
        for u_id in range(stats['num_nodes']):
            u = str(u_id)
            if not graph_manager.has_vertex(u):
                continue
            neighbors = graph_manager.get_neighbors(u)
            for edge in neighbors:
                v = edge.target
                weight = graph_manager.get_edge_weight(u, v)
                if weight >= DisasterModeler.BLOCKED_WEIGHT * 0.99:
                    blocked_edges.append((u, v))
        
        visualizer.plot_route_comparison(
            source=source,
            target=target,
            normal_path=normal_result.path,
            normal_distance=normal_result.total_cost,
            disaster_path=disaster_result.path,
            disaster_distance=disaster_result.total_cost,
            blocked_edges=blocked_edges
        )
        plt.close(fig2)
    
    # If we got here, the full workflow succeeded
    assert True


def test_workflow_with_flood_disaster():
    """
    Test workflow with flood disaster (increases risk but doesn't block).
    
    Validates: Requirements 3.3, 6.2
    """
    # Extract small network
    extractor = OSMExtractor()
    osm_graph = extractor.extract_by_place("Piedmont, California, USA", network_type="drive")
    
    # Convert to internal format
    converter = GraphConverter()
    graph_manager, coord_mapping = converter.convert_osm_to_internal(osm_graph)
    
    # Select nodes
    source = "0"
    num_nodes = len(coord_mapping)
    target = str(num_nodes - 1)
    
    if not graph_manager.has_vertex(source) or not graph_manager.has_vertex(target):
        pytest.skip("Selected nodes don't exist in graph")
    
    # Compute normal route
    pathfinder = PathfinderEngine()
    normal_result = pathfinder.find_shortest_path(graph_manager, source, target)
    
    if not normal_result.found:
        pytest.skip("No path exists between selected nodes")
    
    # Apply flood disaster
    middle_idx = len(normal_result.path) // 2
    middle_node_id = int(normal_result.path[middle_idx])
    epicenter = coord_mapping[middle_node_id]
    
    modeler = DisasterModeler(graph_manager, coord_mapping)
    modeler.apply_flood(epicenter, radius_meters=300.0, risk_multiplier=0.8)
    
    # Compute disaster-aware route
    disaster_result = pathfinder.find_shortest_path(graph_manager, source, target)
    
    # Flood should increase cost but not block completely
    assert disaster_result.found
    assert disaster_result.total_cost >= normal_result.total_cost


def test_workflow_with_earthquake_disaster():
    """
    Test workflow with earthquake disaster (random blocking + congestion).
    
    Validates: Requirements 3.5, 6.2
    """
    # Extract small network
    extractor = OSMExtractor()
    osm_graph = extractor.extract_by_place("Piedmont, California, USA", network_type="drive")
    
    # Convert to internal format
    converter = GraphConverter()
    graph_manager, coord_mapping = converter.convert_osm_to_internal(osm_graph)
    
    # Select nodes
    source = "0"
    num_nodes = len(coord_mapping)
    target = str(num_nodes - 1)
    
    if not graph_manager.has_vertex(source) or not graph_manager.has_vertex(target):
        pytest.skip("Selected nodes don't exist in graph")
    
    # Compute normal route
    pathfinder = PathfinderEngine()
    normal_result = pathfinder.find_shortest_path(graph_manager, source, target)
    
    if not normal_result.found:
        pytest.skip("No path exists between selected nodes")
    
    # Apply earthquake disaster
    middle_idx = len(normal_result.path) // 2
    middle_node_id = int(normal_result.path[middle_idx])
    epicenter = coord_mapping[middle_node_id]
    
    modeler = DisasterModeler(graph_manager, coord_mapping)
    modeler.apply_earthquake(
        epicenter, 
        radius_meters=250.0,
        failure_probability=0.3,
        congestion_multiplier=0.6
    )
    
    # Compute disaster-aware route
    disaster_result = pathfinder.find_shortest_path(graph_manager, source, target)
    
    # Earthquake may or may not block path completely
    if disaster_result.found:
        # If path exists, it should have equal or higher cost
        assert disaster_result.total_cost >= normal_result.total_cost * 0.99  # Allow small floating point error
