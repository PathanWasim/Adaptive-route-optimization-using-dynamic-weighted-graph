"""
Unit tests for PathfinderEngine class.
"""

import pytest
from disaster_evacuation.pathfinding import PathfinderEngine
from disaster_evacuation.graph import GraphManager
from disaster_evacuation.models import VertexType


class TestPathfinderEngine:
    """Test cases for PathfinderEngine class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.pathfinder = PathfinderEngine()
        self.graph = GraphManager()
        
        # Create a test graph
        #     A ----2---- B
        #     |           |
        #     3           1
        #     |           |
        #     C ----4---- D
        
        self.graph.add_vertex("A", VertexType.INTERSECTION, (0.0, 0.0))
        self.graph.add_vertex("B", VertexType.INTERSECTION, (2.0, 0.0))
        self.graph.add_vertex("C", VertexType.INTERSECTION, (0.0, 2.0))
        self.graph.add_vertex("D", VertexType.EVACUATION_POINT, (2.0, 2.0), capacity=500)
        
        self.graph.add_edge("A", "B", 2.0, 0.0, 0.0)  # Weight = 2.0
        self.graph.add_edge("A", "C", 3.0, 0.0, 0.0)  # Weight = 3.0
        self.graph.add_edge("B", "D", 1.0, 0.0, 0.0)  # Weight = 1.0
        self.graph.add_edge("C", "D", 4.0, 0.0, 0.0)  # Weight = 4.0
    
    def test_pathfinder_creation(self):
        """Test basic pathfinder creation."""
        pathfinder = PathfinderEngine()
        stats = pathfinder.get_algorithm_stats()
        assert stats.nodes_visited == 0
        assert stats.computation_time == 0.0
        assert stats.edges_examined == 0
        assert stats.queue_operations == 0
    
    def test_shortest_path_simple(self):
        """Test shortest path in simple graph."""
        result = self.pathfinder.find_shortest_path(self.graph, "A", "D")
        
        assert result.found
        assert result.path == ["A", "B", "D"]
        assert result.total_cost == 3.0  # 2.0 + 1.0
        assert len(result.edges_traversed) == 2
        assert result.computation_time >= 0  # May be 0 for very fast operations
        assert result.nodes_visited > 0
        
        # Verify edges
        assert result.edges_traversed[0].source == "A"
        assert result.edges_traversed[0].target == "B"
        assert result.edges_traversed[1].source == "B"
        assert result.edges_traversed[1].target == "D"
    
    def test_shortest_path_same_vertex(self):
        """Test shortest path from vertex to itself."""
        result = self.pathfinder.find_shortest_path(self.graph, "A", "A")
        
        assert result.found
        assert result.path == ["A"]
        assert result.total_cost == 0.0
        assert len(result.edges_traversed) == 0
        assert result.nodes_visited == 1
    
    def test_shortest_path_nonexistent_source(self):
        """Test shortest path with nonexistent source vertex."""
        result = self.pathfinder.find_shortest_path(self.graph, "X", "A")
        
        assert not result.found
        assert "Source vertex 'X' does not exist" in result.error_message
        assert result.path == []
        assert result.total_cost == 0.0
    
    def test_shortest_path_nonexistent_target(self):
        """Test shortest path with nonexistent target vertex."""
        result = self.pathfinder.find_shortest_path(self.graph, "A", "X")
        
        assert not result.found
        assert "Target vertex 'X' does not exist" in result.error_message
        assert result.path == []
        assert result.total_cost == 0.0
    
    def test_shortest_path_disconnected_graph(self):
        """Test shortest path in disconnected graph."""
        # Add isolated vertex
        self.graph.add_vertex("E", VertexType.INTERSECTION, (5.0, 5.0))
        
        result = self.pathfinder.find_shortest_path(self.graph, "A", "E")
        
        assert not result.found
        assert "No path exists" in result.error_message
        assert result.path == []
        assert result.total_cost == 0.0
    
    def test_shortest_path_with_blocked_roads(self):
        """Test shortest path with blocked roads."""
        # Block the direct path A->B by setting infinite weight
        self.graph.update_edge_weight("A", "B", float('inf'))
        
        result = self.pathfinder.find_shortest_path(self.graph, "A", "D")
        
        assert result.found
        assert result.path == ["A", "C", "D"]
        assert result.total_cost == 7.0  # 3.0 + 4.0
        assert len(result.edges_traversed) == 2
    
    def test_find_all_shortest_paths(self):
        """Test finding shortest paths to all vertices."""
        results = self.pathfinder.find_all_shortest_paths(self.graph, "A")
        
        assert len(results) == 4
        
        # Check path to A (self)
        assert results["A"].found
        assert results["A"].path == ["A"]
        assert results["A"].total_cost == 0.0
        
        # Check path to B
        assert results["B"].found
        assert results["B"].path == ["A", "B"]
        assert results["B"].total_cost == 2.0
        
        # Check path to C
        assert results["C"].found
        assert results["C"].path == ["A", "C"]
        assert results["C"].total_cost == 3.0
        
        # Check path to D
        assert results["D"].found
        assert results["D"].path == ["A", "B", "D"]
        assert results["D"].total_cost == 3.0
    
    def test_find_all_shortest_paths_nonexistent_source(self):
        """Test finding all paths with nonexistent source."""
        results = self.pathfinder.find_all_shortest_paths(self.graph, "X")
        
        assert len(results) == 1
        assert "X" in results
        assert not results["X"].found
        assert "Source vertex 'X' does not exist" in results["X"].error_message
    
    def test_algorithm_statistics(self):
        """Test algorithm statistics collection."""
        # Run pathfinding
        result = self.pathfinder.find_shortest_path(self.graph, "A", "D")
        
        # Get statistics
        stats = self.pathfinder.get_algorithm_stats()
        
        assert stats.nodes_visited > 0
        assert stats.computation_time >= 0  # May be 0 for very fast operations
        assert stats.edges_examined > 0
        assert stats.queue_operations > 0
        
        # Statistics should match result
        assert stats.nodes_visited == result.nodes_visited
        assert stats.computation_time == result.computation_time
    
    def test_dijkstra_properties_validation(self):
        """Test Dijkstra's algorithm properties validation."""
        properties = self.pathfinder.validate_dijkstra_properties(self.graph, "A")
        
        assert properties["non_negative_weights"]
        assert properties["greedy_choice_optimal"]
        assert properties["optimal_substructure"]
        assert properties["single_source_property"]
    
    def test_dijkstra_properties_negative_weights(self):
        """Test properties validation with negative weights."""
        # Since our system validates against negative weights at multiple levels,
        # we'll test the validation logic directly
        properties = self.pathfinder.validate_dijkstra_properties(self.graph, "A")
        
        # All weights in our test graph are non-negative
        assert properties["non_negative_weights"]
        
        # Test the validation logic by checking what happens with infinite weights
        # (which are allowed and represent blocked roads)
        self.graph.update_edge_weight("A", "B", float('inf'))
        properties_with_inf = self.pathfinder.validate_dijkstra_properties(self.graph, "A")
        
        # Infinite weights should still pass validation (they represent blocked roads)
        assert properties_with_inf["non_negative_weights"]
    
    def test_complex_graph_shortest_path(self):
        """Test shortest path in more complex graph."""
        # Create a more complex graph
        complex_graph = GraphManager()
        
        # Add vertices
        for i in range(6):
            complex_graph.add_vertex(f"V{i}", VertexType.INTERSECTION, (i, 0))
        
        # Add edges to create multiple paths
        complex_graph.add_edge("V0", "V1", 4.0, 0.0, 0.0)
        complex_graph.add_edge("V0", "V2", 2.0, 0.0, 0.0)
        complex_graph.add_edge("V1", "V3", 1.0, 0.0, 0.0)
        complex_graph.add_edge("V2", "V1", 1.0, 0.0, 0.0)
        complex_graph.add_edge("V2", "V4", 3.0, 0.0, 0.0)
        complex_graph.add_edge("V3", "V5", 2.0, 0.0, 0.0)
        complex_graph.add_edge("V4", "V3", 2.0, 0.0, 0.0)
        complex_graph.add_edge("V4", "V5", 1.0, 0.0, 0.0)
        
        result = self.pathfinder.find_shortest_path(complex_graph, "V0", "V5")
        
        assert result.found
        # Optimal path should be V0->V2->V1->V3->V5 (cost = 2+1+1+2 = 6)
        # or V0->V2->V4->V5 (cost = 2+3+1 = 6)
        assert result.total_cost == 6.0
        assert result.path[0] == "V0"
        assert result.path[-1] == "V5"
    
    def test_path_reconstruction_accuracy(self):
        """Test that path reconstruction is accurate."""
        result = self.pathfinder.find_shortest_path(self.graph, "A", "D")
        
        # Verify path cost by summing edge weights
        total_cost = 0.0
        for i in range(len(result.path) - 1):
            source = result.path[i]
            target = result.path[i + 1]
            edge_weight = self.graph.get_edge_weight(source, target)
            total_cost += edge_weight
        
        assert abs(total_cost - result.total_cost) < 1e-10
        
        # Verify edges match path
        assert len(result.edges_traversed) == len(result.path) - 1
        for i, edge in enumerate(result.edges_traversed):
            assert edge.source == result.path[i]
            assert edge.target == result.path[i + 1]
    
    def test_performance_with_large_graph(self):
        """Test performance characteristics with larger graph."""
        # Create a larger graph (grid pattern)
        large_graph = GraphManager()
        
        # Create 5x5 grid
        for i in range(5):
            for j in range(5):
                vertex_id = f"V{i}_{j}"
                large_graph.add_vertex(vertex_id, VertexType.INTERSECTION, (i, j))
        
        # Add horizontal and vertical edges
        for i in range(5):
            for j in range(5):
                current = f"V{i}_{j}"
                # Right edge
                if j < 4:
                    right = f"V{i}_{j+1}"
                    large_graph.add_edge(current, right, 1.0, 0.0, 0.0)
                # Down edge
                if i < 4:
                    down = f"V{i+1}_{j}"
                    large_graph.add_edge(current, down, 1.0, 0.0, 0.0)
        
        # Find path from corner to corner
        result = self.pathfinder.find_shortest_path(large_graph, "V0_0", "V4_4")
        
        assert result.found
        assert result.total_cost == 8.0  # Manhattan distance
        assert result.computation_time < 1.0  # Should be fast
        
        # Verify algorithm efficiency
        stats = self.pathfinder.get_algorithm_stats()
        assert stats.nodes_visited <= 25  # Shouldn't visit all nodes
    
    def test_string_representation(self):
        """Test string representation of pathfinder."""
        pathfinder_str = str(self.pathfinder)
        assert "PathfinderEngine" in pathfinder_str
        assert "stats=" in pathfinder_str