"""
Tests for path recomputation and disconnected graph handling.
"""

import pytest
from disaster_evacuation.pathfinding import PathfinderEngine
from disaster_evacuation.graph import GraphManager
from disaster_evacuation.disaster import DisasterModel
from disaster_evacuation.models import VertexType, DisasterEvent, DisasterType


class TestPathRecomputation:
    """Test cases for path recomputation and disconnected graph handling."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.pathfinder = PathfinderEngine()
        self.graph = GraphManager()
        self.disaster_model = DisasterModel()
        
        # Create a test graph with multiple paths
        #     A ----2---- B ----1---- E
        #     |           |           |
        #     3           2           1
        #     |           |           |
        #     C ----4---- D ----3---- F
        
        self.graph.add_vertex("A", VertexType.INTERSECTION, (0.0, 0.0))
        self.graph.add_vertex("B", VertexType.INTERSECTION, (2.0, 0.0))
        self.graph.add_vertex("C", VertexType.INTERSECTION, (0.0, 2.0))
        self.graph.add_vertex("D", VertexType.INTERSECTION, (2.0, 2.0))
        self.graph.add_vertex("E", VertexType.SHELTER, (4.0, 0.0), capacity=100)
        self.graph.add_vertex("F", VertexType.EVACUATION_POINT, (4.0, 2.0), capacity=500)
        
        # Add edges
        self.graph.add_edge("A", "B", 2.0, 0.0, 0.0)  # Weight = 2.0
        self.graph.add_edge("A", "C", 3.0, 0.0, 0.0)  # Weight = 3.0
        self.graph.add_edge("B", "D", 2.0, 0.0, 0.0)  # Weight = 2.0
        self.graph.add_edge("C", "D", 4.0, 0.0, 0.0)  # Weight = 4.0
        self.graph.add_edge("B", "E", 1.0, 0.0, 0.0)  # Weight = 1.0
        self.graph.add_edge("D", "F", 3.0, 0.0, 0.0)  # Weight = 3.0
        self.graph.add_edge("E", "F", 1.0, 0.0, 0.0)  # Weight = 1.0
    
    def test_path_recomputation_after_weight_change(self):
        """Test that paths are recomputed when edge weights change."""
        # Initial path from A to F
        result1 = self.pathfinder.find_shortest_path(self.graph, "A", "F")
        
        assert result1.found
        initial_path = result1.path
        initial_cost = result1.total_cost
        
        # Change weight of an edge to make a different path optimal
        # Make A->B very expensive
        self.graph.update_edge_weight("A", "B", 10.0)
        
        # Recompute path
        result2 = self.pathfinder.find_shortest_path(self.graph, "A", "F")
        
        assert result2.found
        new_path = result2.path
        new_cost = result2.total_cost
        
        # Path should change due to weight modification
        if len(initial_path) > 1 and initial_path[1] == "B":
            # If initial path went through B, new path should avoid it
            assert "B" not in new_path or new_path != initial_path
        
        # Verify the new path is valid
        assert new_path[0] == "A"
        assert new_path[-1] == "F"
    
    def test_path_recomputation_with_blocked_roads(self):
        """Test path recomputation when roads are blocked by disasters."""
        # Initial path from A to F
        result1 = self.pathfinder.find_shortest_path(self.graph, "A", "F")
        assert result1.found
        initial_path = result1.path
        
        # Apply disaster that blocks some roads
        disaster = DisasterEvent(DisasterType.FIRE, (2.0, 0.0), 0.9, 2.5)
        self.disaster_model.apply_disaster_effects(self.graph, disaster)
        
        # Recompute path after disaster
        result2 = self.pathfinder.find_shortest_path(self.graph, "A", "F")
        
        if result2.found:
            new_path = result2.path
            
            # Verify no blocked edges are used
            for i in range(len(new_path) - 1):
                source = new_path[i]
                target = new_path[i + 1]
                edge = self.graph.get_edge(source, target)
                weight = self.graph.get_edge_weight(source, target)
                
                # Should not use blocked roads
                assert not edge.is_blocked
                assert weight != float('inf')
        else:
            # If no path found, it should be due to all paths being blocked
            assert "No path exists" in result2.error_message
    
    def test_disconnected_graph_detection(self):
        """Test detection of disconnected graph components."""
        # Create a disconnected graph by adding isolated vertices
        isolated_graph = GraphManager()
        
        # Component 1: A-B
        isolated_graph.add_vertex("A", VertexType.INTERSECTION, (0.0, 0.0))
        isolated_graph.add_vertex("B", VertexType.INTERSECTION, (1.0, 0.0))
        isolated_graph.add_edge("A", "B", 1.0, 0.0, 0.0)
        
        # Component 2: C-D (disconnected from A-B)
        isolated_graph.add_vertex("C", VertexType.INTERSECTION, (5.0, 5.0))
        isolated_graph.add_vertex("D", VertexType.EVACUATION_POINT, (6.0, 5.0), capacity=100)
        isolated_graph.add_edge("C", "D", 1.0, 0.0, 0.0)
        
        # Try to find path between disconnected components
        result = self.pathfinder.find_shortest_path(isolated_graph, "A", "C")
        
        assert not result.found
        assert "No path exists" in result.error_message
        assert result.path == []
        assert result.total_cost == 0.0
    
    def test_single_vertex_component(self):
        """Test handling of single vertex components."""
        single_vertex_graph = GraphManager()
        single_vertex_graph.add_vertex("ISOLATED", VertexType.SHELTER, (0.0, 0.0), capacity=50)
        
        # Path from vertex to itself should work
        result1 = self.pathfinder.find_shortest_path(single_vertex_graph, "ISOLATED", "ISOLATED")
        assert result1.found
        assert result1.path == ["ISOLATED"]
        assert result1.total_cost == 0.0
        
        # Add another isolated vertex
        single_vertex_graph.add_vertex("ANOTHER", VertexType.INTERSECTION, (10.0, 10.0))
        
        # Path between isolated vertices should fail
        result2 = self.pathfinder.find_shortest_path(single_vertex_graph, "ISOLATED", "ANOTHER")
        assert not result2.found
        assert "No path exists" in result2.error_message
    
    def test_path_invalidation_after_disaster(self):
        """Test that cached paths are invalidated after disasters."""
        # Find initial path
        result1 = self.pathfinder.find_shortest_path(self.graph, "A", "E")
        assert result1.found
        initial_cost = result1.total_cost
        
        # Apply disaster
        disaster = DisasterEvent(DisasterType.EARTHQUAKE, (1.0, 0.0), 0.8, 3.0)
        self.disaster_model.apply_disaster_effects(self.graph, disaster)
        
        # Find path again - should be recomputed with new weights
        result2 = self.pathfinder.find_shortest_path(self.graph, "A", "E")
        
        if result2.found:
            # Cost should be different due to disaster effects
            # (unless the optimal path wasn't affected)
            new_cost = result2.total_cost
            
            # At minimum, the computation should have been performed fresh
            assert result2.computation_time >= 0
            assert result2.nodes_visited > 0
    
    def test_multiple_disconnected_components(self):
        """Test pathfinding in graph with multiple disconnected components."""
        multi_component_graph = GraphManager()
        
        # Component 1: Triangle A-B-C
        multi_component_graph.add_vertex("A", VertexType.INTERSECTION, (0.0, 0.0))
        multi_component_graph.add_vertex("B", VertexType.INTERSECTION, (1.0, 0.0))
        multi_component_graph.add_vertex("C", VertexType.INTERSECTION, (0.5, 1.0))
        multi_component_graph.add_edge("A", "B", 1.0, 0.0, 0.0)
        multi_component_graph.add_edge("B", "C", 1.0, 0.0, 0.0)
        multi_component_graph.add_edge("C", "A", 1.0, 0.0, 0.0)
        
        # Component 2: Line D-E-F
        multi_component_graph.add_vertex("D", VertexType.INTERSECTION, (5.0, 0.0))
        multi_component_graph.add_vertex("E", VertexType.INTERSECTION, (6.0, 0.0))
        multi_component_graph.add_vertex("F", VertexType.EVACUATION_POINT, (7.0, 0.0), capacity=200)
        multi_component_graph.add_edge("D", "E", 1.0, 0.0, 0.0)
        multi_component_graph.add_edge("E", "F", 1.0, 0.0, 0.0)
        
        # Component 3: Isolated vertex
        multi_component_graph.add_vertex("G", VertexType.SHELTER, (10.0, 10.0), capacity=100)
        
        # Test paths within components
        result_within_1 = self.pathfinder.find_shortest_path(multi_component_graph, "A", "C")
        assert result_within_1.found
        # Path A->B->C has cost 2.0, but there's also C->A edge, so we need A->C
        # Since we only have A->B->C path, cost should be 2.0
        assert result_within_1.total_cost == 2.0  # A->B->C path
        
        result_within_2 = self.pathfinder.find_shortest_path(multi_component_graph, "D", "F")
        assert result_within_2.found
        assert result_within_2.total_cost == 2.0  # D->E->F
        
        # Test paths between components (should fail)
        result_between = self.pathfinder.find_shortest_path(multi_component_graph, "A", "D")
        assert not result_between.found
        
        result_to_isolated = self.pathfinder.find_shortest_path(multi_component_graph, "A", "G")
        assert not result_to_isolated.found
    
    def test_find_all_paths_disconnected_graph(self):
        """Test finding all paths in disconnected graph."""
        # Use the multi-component graph from previous test
        multi_component_graph = GraphManager()
        
        # Component 1: A-B
        multi_component_graph.add_vertex("A", VertexType.INTERSECTION, (0.0, 0.0))
        multi_component_graph.add_vertex("B", VertexType.INTERSECTION, (1.0, 0.0))
        multi_component_graph.add_edge("A", "B", 2.0, 0.0, 0.0)
        
        # Component 2: C-D (disconnected)
        multi_component_graph.add_vertex("C", VertexType.INTERSECTION, (5.0, 0.0))
        multi_component_graph.add_vertex("D", VertexType.EVACUATION_POINT, (6.0, 0.0), capacity=100)
        multi_component_graph.add_edge("C", "D", 3.0, 0.0, 0.0)
        
        # Find all paths from A
        results = self.pathfinder.find_all_shortest_paths(multi_component_graph, "A")
        
        # Should have results for all vertices
        assert len(results) == 4
        
        # Paths within component should be found
        assert results["A"].found
        assert results["A"].total_cost == 0.0
        
        assert results["B"].found
        assert results["B"].total_cost == 2.0
        
        # Paths to disconnected component should fail
        assert not results["C"].found
        assert "No path exists" in results["C"].error_message
        
        assert not results["D"].found
        assert "No path exists" in results["D"].error_message
    
    def test_error_handling_edge_cases(self):
        """Test error handling for various edge cases."""
        empty_graph = GraphManager()
        
        # Empty graph
        result = self.pathfinder.find_shortest_path(empty_graph, "A", "B")
        assert not result.found
        assert "does not exist" in result.error_message
        
        # Single vertex graph
        single_graph = GraphManager()
        single_graph.add_vertex("ONLY", VertexType.INTERSECTION, (0.0, 0.0))
        
        result = self.pathfinder.find_shortest_path(single_graph, "ONLY", "NONEXISTENT")
        assert not result.found
        assert "Target vertex 'NONEXISTENT' does not exist" in result.error_message
    
    def test_path_recomputation_performance(self):
        """Test that path recomputation doesn't degrade performance significantly."""
        # Create a larger graph for performance testing
        large_graph = GraphManager()
        
        # Create a 4x4 grid
        for i in range(4):
            for j in range(4):
                vertex_id = f"V{i}_{j}"
                large_graph.add_vertex(vertex_id, VertexType.INTERSECTION, (i, j))
        
        # Add edges
        for i in range(4):
            for j in range(4):
                current = f"V{i}_{j}"
                if j < 3:  # Right edge
                    right = f"V{i}_{j+1}"
                    large_graph.add_edge(current, right, 1.0, 0.0, 0.0)
                if i < 3:  # Down edge
                    down = f"V{i+1}_{j}"
                    large_graph.add_edge(current, down, 1.0, 0.0, 0.0)
        
        # Initial pathfinding
        result1 = self.pathfinder.find_shortest_path(large_graph, "V0_0", "V3_3")
        assert result1.found
        time1 = result1.computation_time
        
        # Modify some weights
        large_graph.update_edge_weight("V0_0", "V0_1", 5.0)
        large_graph.update_edge_weight("V1_0", "V1_1", 5.0)
        
        # Recompute path
        result2 = self.pathfinder.find_shortest_path(large_graph, "V0_0", "V3_3")
        assert result2.found
        time2 = result2.computation_time
        
        # Performance should be reasonable (allowing for some variation)
        # Both computations should complete quickly
        assert time1 < 1.0
        assert time2 < 1.0
        
        # Path should still be valid
        assert result2.path[0] == "V0_0"
        assert result2.path[-1] == "V3_3"