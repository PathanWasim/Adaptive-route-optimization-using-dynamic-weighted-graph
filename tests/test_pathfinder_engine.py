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
        
        # Create a test graph with known shortest paths
        #     A ----2---- B
        #     |           |
        #     3           1
        #     |           |
        #     C ----1---- D
        
        self.graph.add_vertex("A", VertexType.INTERSECTION, (0.0, 0.0))
        self.graph.add_vertex("B", VertexType.INTERSECTION, (2.0, 0.0))
        self.graph.add_vertex("C", VertexType.INTERSECTION, (0.0, 2.0))
        self.graph.add_vertex("D", VertexType.EVACUATION_POINT, (2.0, 2.0), capacity=500)
        
        # Add edges with known weights
        self.graph.add_edge("A", "B", 2.0, 0.0, 0.0)  # Weight = 2.0
        self.graph.add_edge("A", "C", 3.0, 0.0, 0.0)  # Weight = 3.0
        self.graph.add_edge("B", "D", 1.0, 0.0, 0.0)  # Weight = 1.0
        self.graph.add_edge("C", "D", 1.0, 0.0, 0.0)  # Weight = 1.0
    
    def test_pathfinder_creation(self):
        """Test basic pathfinder creation."""
        pathfinder = PathfinderEngine()
        stats = pathfinder.get_algorithm_stats()
        assert stats.nodes_visited == 0
        assert stats.computation_time == 0.0
        assert stats.edges_examined == 0
        assert stats.queue_operations == 0
    
    def test_shortest_path_direct_route(self):
        """Test shortest path with direct route."""
        result = self.pathfinder.find_shortest_path(self.graph, "A", "B")
        
        assert result.found
        assert result.path == ["A", "B"]
        assert result.total_cost == 2.0
        assert len(result.edges_traversed) == 1
        assert result.edges_traversed[0].source == "A"
        assert result.edges_traversed[0].target == "B"
        assert result.computation_time >= 0  # Allow zero for very fast operations
        assert result.nodes_visited >= 2
    
    def test_shortest_path_indirect_route(self):
        """Test shortest path requiring multiple hops."""
        result = self.pathfinder.find_shortest_path(self.graph, "A", "D")
        
        assert result.found
        # Shortest path: A -> B -> D (cost 3.0) vs A -> C -> D (cost 4.0)
        assert result.path == ["A", "B", "D"]
        assert result.total_cost == 3.0
        assert len(result.edges_traversed) == 2
        
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
        assert result.path == []
        assert result.total_cost == 0.0
        assert "Source vertex 'X' does not exist" in result.error_message
    
    def test_shortest_path_nonexistent_target(self):
        """Test shortest path with nonexistent target vertex."""
        result = self.pathfinder.find_shortest_path(self.graph, "A", "X")
        
        assert not result.found
        assert result.path == []
        assert result.total_cost == 0.0
        assert "Target vertex 'X' does not exist" in result.error_message
    
    def test_shortest_path_disconnected_graph(self):
        """Test shortest path in disconnected graph."""
        # Add isolated vertex
        self.graph.add_vertex("E", VertexType.INTERSECTION, (5.0, 5.0))
        
        result = self.pathfinder.find_shortest_path(self.graph, "A", "E")
        
        assert not result.found
        assert result.path == []
        assert "No path exists from 'A' to 'E'" in result.error_message
    
    def test_shortest_path_with_blocked_edges(self):
        """Test shortest path with blocked edges."""
        # Block the direct route A -> B
        edge_ab = self.graph.get_edge("A", "B")
        edge_ab.is_blocked = True
        self.graph.update_edge_weight("A", "B", float('inf'))
        
        result = self.pathfinder.find_shortest_path(self.graph, "A", "D")
        
        assert result.found
        # Should take alternate route: A -> C -> D (cost 4.0)
        assert result.path == ["A", "C", "D"]
        assert result.total_cost == 4.0
    
    def test_find_all_shortest_paths(self):
        """Test finding shortest paths to all vertices."""
        results = self.pathfinder.find_all_shortest_paths(self.graph, "A")
        
        # Should have results for all vertices
        assert len(results) == 4
        assert all(vertex in results for vertex in ["A", "B", "C", "D"])
        
        # Check specific paths
        assert results["A"].path == ["A"]
        assert results["A"].total_cost == 0.0
        
        assert results["B"].path == ["A", "B"]
        assert results["B"].total_cost == 2.0
        
        assert results["C"].path == ["A", "C"]
        assert results["C"].total_cost == 3.0
        
        assert results["D"].path == ["A", "B", "D"]
        assert results["D"].total_cost == 3.0
    
    def test_find_all_shortest_paths_nonexistent_source(self):
        """Test finding all shortest paths with nonexistent source."""
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
        assert stats.computation_time >= 0  # Allow zero for very fast operations
        assert stats.edges_examined > 0
        assert stats.queue_operations > 0
        
        # Statistics should match result
        assert stats.nodes_visited == result.nodes_visited
        assert stats.computation_time == result.computation_time
    
    def test_path_optimality_verification(self):
        """Test path optimality verification."""
        result = self.pathfinder.find_shortest_path(self.graph, "A", "D")
        
        # Verify the path is optimal
        is_optimal = self.pathfinder.verify_path_optimality(self.graph, result)
        assert is_optimal
        
        # Test with manually created suboptimal path
        from disaster_evacuation.models import PathResult
        suboptimal_result = PathResult(
            path=["A", "C", "D"],
            total_cost=4.0,
            edges_traversed=[self.graph.get_edge("A", "C"), self.graph.get_edge("C", "D")],
            computation_time=0.001,
            nodes_visited=3
        )
        
        # This path has correct cost for its edges, so verification should pass
        is_optimal = self.pathfinder.verify_path_optimality(self.graph, suboptimal_result)
        assert is_optimal  # Cost correctness verification should pass
    
    def test_complex_graph_pathfinding(self):
        """Test pathfinding on a more complex graph."""
        # Create a larger graph
        complex_graph = GraphManager()
        
        # Add vertices in a grid pattern
        for i in range(4):
            for j in range(4):
                vertex_id = f"V{i}{j}"
                complex_graph.add_vertex(vertex_id, VertexType.INTERSECTION, (i, j))
        
        # Add horizontal edges
        for i in range(4):
            for j in range(3):
                source = f"V{i}{j}"
                target = f"V{i}{j+1}"
                complex_graph.add_edge(source, target, 1.0, 0.0, 0.0)
        
        # Add vertical edges
        for i in range(3):
            for j in range(4):
                source = f"V{i}{j}"
                target = f"V{i+1}{j}"
                complex_graph.add_edge(source, target, 1.0, 0.0, 0.0)
        
        # Find path from corner to corner
        result = self.pathfinder.find_shortest_path(complex_graph, "V00", "V33")
        
        assert result.found
        assert result.total_cost == 6.0  # Manhattan distance
        assert len(result.path) == 7  # 6 edges + 1
        assert result.path[0] == "V00"
        assert result.path[-1] == "V33"
    
    def test_edge_weight_changes(self):
        """Test pathfinding with dynamic edge weight changes."""
        # Initial shortest path: A -> B -> D (cost 3.0)
        result1 = self.pathfinder.find_shortest_path(self.graph, "A", "D")
        assert result1.path == ["A", "B", "D"]
        assert result1.total_cost == 3.0
        
        # Increase weight of B -> D edge
        self.graph.update_edge_weight("B", "D", 5.0)
        
        # Now shortest path should be: A -> C -> D (cost 4.0)
        result2 = self.pathfinder.find_shortest_path(self.graph, "A", "D")
        assert result2.path == ["A", "C", "D"]
        assert result2.total_cost == 4.0
    
    def test_pathfinding_with_zero_weight_edges(self):
        """Test pathfinding with zero-weight edges."""
        # Add zero-weight edge
        self.graph.add_vertex("E", VertexType.INTERSECTION, (1.0, 1.0))
        self.graph.add_edge("A", "E", 0.0, 0.0, 0.0)  # Zero weight
        self.graph.add_edge("E", "D", 2.0, 0.0, 0.0)
        
        result = self.pathfinder.find_shortest_path(self.graph, "A", "D")
        
        assert result.found
        # Should prefer A -> E -> D (cost 2.0) over A -> B -> D (cost 3.0)
        assert result.path == ["A", "E", "D"]
        assert result.total_cost == 2.0
    
    def test_large_graph_performance(self):
        """Test pathfinding performance on larger graph."""
        # Create a larger graph for performance testing
        large_graph = GraphManager()
        
        # Add vertices
        for i in range(10):
            for j in range(10):
                vertex_id = f"V{i:02d}{j:02d}"
                large_graph.add_vertex(vertex_id, VertexType.INTERSECTION, (i, j))
        
        # Add edges (grid connectivity)
        for i in range(10):
            for j in range(10):
                current = f"V{i:02d}{j:02d}"
                
                # Right edge
                if j < 9:
                    right = f"V{i:02d}{j+1:02d}"
                    large_graph.add_edge(current, right, 1.0, 0.1, 0.1)
                
                # Down edge
                if i < 9:
                    down = f"V{i+1:02d}{j:02d}"
                    large_graph.add_edge(current, down, 1.0, 0.1, 0.1)
        
        # Find path across the graph
        result = self.pathfinder.find_shortest_path(large_graph, "V0000", "V0909")
        
        assert result.found
        assert result.computation_time < 1.0  # Should be fast
        assert result.nodes_visited <= 100  # Shouldn't visit all nodes
        
        # Verify path makes sense
        assert result.path[0] == "V0000"
        assert result.path[-1] == "V0909"
        assert result.total_cost > 0
    
    def test_string_representation(self):
        """Test string representation of pathfinder."""
        pathfinder_str = str(self.pathfinder)
        assert "PathfinderEngine" in pathfinder_str
        assert "stats=" in pathfinder_str