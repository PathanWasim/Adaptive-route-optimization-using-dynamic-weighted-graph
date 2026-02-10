"""
Hypothesis strategies for generating test data for OSM integration.

These strategies generate random but valid OSM graphs, coordinates,
and disaster configurations for property-based testing.
"""

from hypothesis import strategies as st
import networkx as nx


@st.composite
def st_osm_graphs(draw, min_nodes=3, max_nodes=20, min_edges=2, max_edges=40):
    """
    Generate random OSM-like graphs for property testing.
    
    Args:
        draw: Hypothesis draw function
        min_nodes: Minimum number of nodes
        max_nodes: Maximum number of nodes
        min_edges: Minimum number of edges
        max_edges: Maximum number of edges
    
    Returns:
        NetworkX MultiDiGraph with OSM-like structure
    """
    num_nodes = draw(st.integers(min_value=min_nodes, max_value=max_nodes))
    
    # Generate random coordinates within a small area
    base_lat = draw(st.floats(min_value=37.7, max_value=37.9))
    base_lon = draw(st.floats(min_value=-122.5, max_value=-122.0))
    
    G = nx.MultiDiGraph()
    
    # Add nodes with coordinates
    for i in range(num_nodes):
        lat = base_lat + draw(st.floats(min_value=-0.01, max_value=0.01))
        lon = base_lon + draw(st.floats(min_value=-0.01, max_value=0.01))
        G.add_node(i, x=lon, y=lat)
    
    # Add random edges with distances (avoid parallel edges)
    num_edges = draw(st.integers(min_value=min_edges, max_value=min(max_edges, num_nodes * (num_nodes - 1))))
    edges_added = set()
    
    for _ in range(num_edges):
        if len(edges_added) >= num_edges:
            break
        
        u = draw(st.integers(min_value=0, max_value=num_nodes - 1))
        v = draw(st.integers(min_value=0, max_value=num_nodes - 1))
        
        if u != v and (u, v) not in edges_added:
            length = draw(st.floats(min_value=10.0, max_value=1000.0))
            G.add_edge(u, v, length=length)
            edges_added.add((u, v))
    
    return G


@st.composite
def st_coordinates(draw, lat_range=(-90.0, 90.0), lon_range=(-180.0, 180.0)):
    """
    Generate random valid geographic coordinates.
    
    Args:
        draw: Hypothesis draw function
        lat_range: Tuple of (min_lat, max_lat)
        lon_range: Tuple of (min_lon, max_lon)
    
    Returns:
        Tuple of (latitude, longitude)
    """
    lat = draw(st.floats(min_value=lat_range[0], max_value=lat_range[1]))
    lon = draw(st.floats(min_value=lon_range[0], max_value=lon_range[1]))
    return (lat, lon)


@st.composite
def st_disaster_configs(draw):
    """
    Generate random disaster configurations.
    
    Args:
        draw: Hypothesis draw function
    
    Returns:
        Dictionary with disaster configuration
    """
    disaster_type = draw(st.sampled_from(['flood', 'fire', 'earthquake']))
    
    # Generate epicenter coordinates
    epicenter = draw(st_coordinates(
        lat_range=(37.7, 37.9),
        lon_range=(-122.5, -122.0)
    ))
    
    # Generate radius
    radius = draw(st.floats(min_value=100.0, max_value=2000.0))
    
    config = {
        'type': disaster_type,
        'epicenter': epicenter,
        'radius_meters': radius
    }
    
    # Add type-specific parameters
    if disaster_type == 'flood':
        config['risk_multiplier'] = draw(st.floats(min_value=0.1, max_value=1.0))
    elif disaster_type == 'earthquake':
        config['failure_probability'] = draw(st.floats(min_value=0.0, max_value=0.5))
        config['congestion_multiplier'] = draw(st.floats(min_value=0.1, max_value=1.0))
    
    return config


@st.composite
def st_small_osm_graphs(draw):
    """
    Generate small OSM graphs suitable for quick testing.
    
    Returns:
        Small NetworkX MultiDiGraph
    """
    return draw(st_osm_graphs(min_nodes=3, max_nodes=8, min_edges=2, max_edges=12))


@st.composite
def st_connected_osm_graphs(draw, min_nodes=4, max_nodes=15):
    """
    Generate connected OSM graphs (all nodes reachable).
    
    Args:
        draw: Hypothesis draw function
        min_nodes: Minimum number of nodes
        max_nodes: Maximum number of nodes
    
    Returns:
        Connected NetworkX MultiDiGraph
    """
    num_nodes = draw(st.integers(min_value=min_nodes, max_value=max_nodes))
    
    # Generate coordinates
    base_lat = draw(st.floats(min_value=37.7, max_value=37.9))
    base_lon = draw(st.floats(min_value=-122.5, max_value=-122.0))
    
    G = nx.MultiDiGraph()
    
    # Add nodes
    for i in range(num_nodes):
        lat = base_lat + draw(st.floats(min_value=-0.005, max_value=0.005))
        lon = base_lon + draw(st.floats(min_value=-0.005, max_value=0.005))
        G.add_node(i, x=lon, y=lat)
    
    # Create a connected graph by adding edges in a chain
    for i in range(num_nodes - 1):
        length = draw(st.floats(min_value=50.0, max_value=500.0))
        G.add_edge(i, i + 1, length=length)
    
    # Add some random additional edges
    num_extra_edges = draw(st.integers(min_value=0, max_value=num_nodes))
    for _ in range(num_extra_edges):
        u = draw(st.integers(min_value=0, max_value=num_nodes - 1))
        v = draw(st.integers(min_value=0, max_value=num_nodes - 1))
        if u != v:
            length = draw(st.floats(min_value=50.0, max_value=500.0))
            G.add_edge(u, v, length=length)
    
    return G
