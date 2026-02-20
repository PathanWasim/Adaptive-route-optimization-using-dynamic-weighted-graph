# Adaptive Disaster Evacuation Route Optimization System

A graph-based pathfinding application that demonstrates advanced algorithmic concepts for a Design and Analysis of Algorithms (DAA) course. The system models urban environments as dynamic weighted graphs and computes optimal evacuation routes during disasters using Dijkstra's algorithm with priority queue optimization.

## Project Overview

Traditional navigation systems compute routes based on shortest distance, but during disasters such as floods, fires, or earthquakes, the shortest route may not be safe. This project implements an adaptive evacuation routing system that computes the safest and most time-efficient evacuation path by modeling real cities as dynamic weighted graphs using OpenStreetMap data.

## Key Features

- **Real-World Road Networks**: Integration with OpenStreetMap for authentic city road networks
- **Interactive Web Application**: Professional web interface with Leaflet.js map visualization
- **Algorithm Visualization**: Step-by-step animation of Dijkstra, A*, and Bellman-Ford execution
- **Dynamic Graph Modeling**: Cities represented as weighted graphs G(V, E) with dynamic edge weights
- **Multi-Algorithm Support**: Implementation of Dijkstra, A* (Haversine Heuristic), and Bellman-Ford
- **Disaster Simulation**: Support for three disaster types (Fire, Flood, Earthquake) with exponential risk multipliers
- **Optimal Pathfinding**: Comparative analysis of O(E log V) vs O(VE) algorithms
- **Benchmarking Suite**: Built-in empirical complexity analysis tools
- **Education Mode**: "Textbook Graph" for learning algorithm mechanics on small datasets
- **Auto-Demo**: One-click presentation mode for easy evaluation
- **Academic Rigor**: Formal correctness proofs and comprehensive property-based testing

## Available Cities

The system includes pre-configured cities optimized for fast loading:

**United States:**
- Piedmont, California
- Berkeley, California
- Albany, California

**India:**
- Pune - Shivajinagar
- Pune - Koregaon Park
- Mumbai - Bandra West
- Bangalore - Indiranagar
- Delhi - Connaught Place

## System Architecture

```
disaster_evacuation/
├── models/          # Core data structures (Vertex, Edge, DisasterEvent)
├── graph/           # Graph management and weight calculation
├── disaster/        # Disaster modeling and effects
├── pathfinding/     # Dijkstra's algorithm implementation
├── osm/             # OpenStreetMap integration (extraction & conversion)
├── visualization/   # Map and route visualization
├── analysis/        # Comparative analysis tools
└── controller/      # Route computation controller

Web Application:
├── app.py           # Flask backend with REST API
├── templates/       # Web interface (classic & enhanced UI)
└── static/          # Frontend assets
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd disaster-evacuation-routing
```

2. Install dependencies:
```bash
pip install -r requirements.txt
pip install -r requirements_web.txt
```

3. Install the package in development mode:
```bash
pip install -e .
```

## Usage

### Web Application (Recommended)

Run the interactive web application:

```bash
python app.py
```

Then open your browser to `http://localhost:5000`

**Features:**
- Select from pre-configured cities
- Click on map to select source and destination
- Configure disaster scenarios (fire, flood, earthquake)
- Watch Dijkstra's algorithm execute step-by-step
- Compare normal vs disaster-aware routes in real-time

### Command Line Demo

Run the demonstration script:

```bash
python demo.py
```

This loads a real city network, applies a disaster, and computes evacuation routes.

### Run Benchmarks

To analyze the time complexity of the implemented algorithms:

```bash
python run_benchmarks.py
```

This will run Dijkstra, A*, and Bellman-Ford on random graphs of increasing size (50-400 nodes) and save the results to `benchmark_results.json`.

### Auto-Demo Mode

1. Open the web interface.
2. Click the **"🎬 Auto-Demo"** button in the sidebar.
3. Sit back and watch the system automatically:
   - Select a city
   - Pick source/destination
   - Apply a disaster
   - Compare all algorithms
   - Animate the results

### Programmatic Usage

```python
from disaster_evacuation import GraphManager, DisasterModel, PathfinderEngine
from disaster_evacuation.osm import OSMExtractor, GraphConverter

# Extract real road network
extractor = OSMExtractor()
osm_graph = extractor.extract_by_place("Piedmont, California, USA")

# Convert to internal format
converter = GraphConverter()
graph, coords = converter.convert_osm_to_internal(osm_graph)

# Apply disaster effects
from disaster_evacuation.disaster import DisasterModeler
modeler = DisasterModeler(graph, coords)
modeler.apply_fire((37.8244, -122.2312), radius=200.0)

# Find optimal evacuation route
pathfinder = PathfinderEngine()
result = pathfinder.find_shortest_path(graph, "0", "50")
print(result.get_path_summary())
```

## Testing

The project uses comprehensive testing with both unit tests and property-based tests:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=disaster_evacuation

# Run property-based tests only
pytest -k "property"

# Run specific test file
pytest tests/test_pathfinder_engine.py
```

**Test Coverage:**
- 54 unit tests covering core functionality
- 15 property-based tests for algorithmic correctness
- Integration tests for OSM data pipeline
- Visualization and web API tests

## Academic Integrity

**CRITICAL:** This project maintains strict academic integrity:

✅ **All routing computed using internal Dijkstra implementation**
- No external routing APIs (Google Maps, Mapbox, etc.)
- No pre-computed route databases
- Pure algorithmic solution with O(E log V) complexity

✅ **Maps used ONLY for visualization**
- Leaflet.js displays routes computed by our algorithm
- OpenStreetMap provides road network data structure only
- No routing functionality from OSM libraries

✅ **Complete implementation transparency**
- Full source code available for review
- Comprehensive test suite validates correctness
- Algorithm execution can be visualized step-by-step

## Academic Requirements

This project fulfills the requirements for a 4-credit DAA course project:

- ✅ **Algorithmic Thinking**: Dijkstra's algorithm with priority queue optimization
- ✅ **Correctness Proofs**: Formal proof of greedy choice property and optimal substructure
- ✅ **Complexity Analysis**: O(E log V) time complexity with adjacency list + min-heap
- ✅ **Real-world Application**: Emergency evacuation routing with disaster modeling
- ✅ **Comprehensive Testing**: Property-based testing for algorithmic correctness
- ✅ **Practical Implementation**: Working web application with real city data

## Documentation

- [Requirements Document](.kiro/specs/disaster-evacuation-routing/requirements.md)
- [Design Document](.kiro/specs/disaster-evacuation-routing/design.md)
- [Implementation Tasks](.kiro/specs/disaster-evacuation-routing/tasks.md)
- [OSM Integration Spec](.kiro/specs/osm-road-network-integration/)
- [Technical Paper](TECHNICAL_PAPER.md) - Comprehensive research documentation

## Performance

**Loading Times:**
- Small cities (Piedmont, Albany): 20-30 seconds
- Medium neighborhoods: 30-60 seconds
- Route computation: <100ms for typical networks

**Algorithm Performance:**
- Time Complexity: O(E log V) using min-heap priority queue
- Space Complexity: O(V) for distance and predecessor arrays
- Typical networks: 500-2000 nodes, 1000-5000 edges

## Future Enhancements

- Multi-destination evacuation planning
- Real-time traffic integration
- Population density modeling
- Shelter capacity constraints
- Mobile application
- Historical disaster data analysis

## License

This project is developed for educational purposes as part of a Design and Analysis of Algorithms course.