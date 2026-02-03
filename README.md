# Adaptive Disaster Evacuation Route Optimization System

A graph-based pathfinding application that demonstrates advanced algorithmic concepts for a Design and Analysis of Algorithms (DAA) course. The system models urban environments as dynamic weighted graphs and computes optimal evacuation routes during disasters using Dijkstra's algorithm with priority queue optimization.

## Project Overview

Traditional navigation systems compute routes based on shortest distance, but during disasters such as floods, fires, or earthquakes, the shortest route may not be safe. This project aims to design an adaptive evacuation routing system that computes the safest and most time-efficient evacuation path by modeling a city as a dynamic weighted graph.

## Key Features

- **Dynamic Graph Modeling**: City represented as weighted graph G(V, E) with dynamic edge weights
- **Disaster Simulation**: Support for three disaster types (Flood, Fire, Earthquake) with different risk models
- **Optimal Pathfinding**: Dijkstra's algorithm implementation with O(E log V) complexity
- **Comparative Analysis**: Compare normal shortest-path vs disaster-aware routing
- **Visualization**: Interactive graph display showing route adaptations
- **Academic Rigor**: Formal correctness proofs and complexity analysis

## System Architecture

```
disaster_evacuation/
├── models/          # Core data structures (Vertex, Edge, DisasterEvent)
├── graph/           # Graph management and weight calculation
├── disaster/        # Disaster modeling and effects
├── pathfinding/     # Dijkstra's algorithm implementation
├── visualization/   # Graph and route visualization
└── analysis/        # Comparative analysis tools
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
```

3. Install the package in development mode:
```bash
pip install -e .
```

## Usage

```python
from disaster_evacuation import GraphManager, DisasterModel, PathfinderEngine

# Create graph
graph = GraphManager()
graph.add_vertex("A", VertexType.INTERSECTION, (0.0, 0.0))
graph.add_vertex("B", VertexType.EVACUATION_POINT, (2.0, 2.0))
graph.add_edge("A", "B", distance=2.8, risk=0.1, congestion=0.2)

# Apply disaster effects
disaster = DisasterEvent(DisasterType.FLOOD, (1.0, 1.0), severity=0.7, radius=3.0)
disaster_model = DisasterModel()
disaster_model.apply_disaster_effects(graph, disaster)

# Find optimal evacuation route
pathfinder = PathfinderEngine()
result = pathfinder.find_shortest_path(graph, "A", "B")
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
pytest -m "hypothesis"
```

## Academic Requirements

This project fulfills the requirements for a 4-credit DAA course project:

- ✅ **Algorithmic Thinking**: Dijkstra's algorithm with priority queue optimization
- ✅ **Correctness Proofs**: Formal proof of greedy choice property
- ✅ **Complexity Analysis**: O(E log V) time complexity with adjacency list + min-heap
- ✅ **Real-world Application**: Emergency evacuation routing with disaster modeling
- ✅ **Comprehensive Testing**: Property-based testing for algorithmic correctness

## Documentation

- [Requirements Document](.kiro/specs/disaster-evacuation-routing/requirements.md)
- [Design Document](.kiro/specs/disaster-evacuation-routing/design.md)
- [Implementation Tasks](.kiro/specs/disaster-evacuation-routing/tasks.md)

## License

This project is developed for educational purposes as part of a Design and Analysis of Algorithms course.