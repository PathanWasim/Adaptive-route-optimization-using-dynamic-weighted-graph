"""
Specific tests for path recomputation and disconnected graph handling.
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
        #     C ----1---- D ----3---- F
        
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
        self.graph.add_edge("C", "D", 1.0, 0.0, 0.0)  # Weight = 1.0
        self.graph.add_edge("B", "E", 1.0, 0.0, 0.0)  # Weight = 1.0
        self.graph.add_edge("D", "F", 3.0, 0.0, 0.0)  # Weight = 3.0
        self.graph.add_edge("E", "F", 1.0, 0.0, 0.0)  # Weight = 1.0
    
    def test_path_recomputation_after_weight_change(self):
        """Test that paths are recomputed when edge weights change."""
        # Initial shortest path A -> F
        result1 = self.pathfinder.find_shortest_path(self.graph, "A", "F")
        
        # Should be A -> B -> E -> F (cost 4.0) vs A -> C -> D -> F (cost 7.0)
        assert result1.found
        assert result1.path == ["A", "B", "E", "F"]
        assert result1.total_cost == 4.0
        
        # Change weight of B -> E to make it expensive
        self.graph.update_edge_weight("B", "E", 10.0)
        
        # Recompute path
        result2 = self.pathfinder.find_shortest_path(self.graph, "A", "F")
        
        # Should now prefer a path that avoids the expensive B -> E edge
        # A -> B -> D -> F (cost 2+2+3=7) vs A -> C -> D -> F (cost 3+1+3=7)
        # Both have same cost, so either is acceptable
        assert result2.found
        assert result2.total_cost == 7.0  # Should be 7.0 regardless of specific path
        
        # Verify the expensive B -> E edge is avoided
        if "B" in result2.path and "E" in result2.path:
            b_index = result2.path.index("B")
            e_index = result2.path.index("E")
            assert abs(b_index - e_index) != 1  # B and E should not be adjacent
        
        # Verify paths are different
        assert result1.path != result2.path
        assert result1.total_cost != result2.total_cost
    
    def test_path_recomputation_after_disaster_effects(self):
        """Test path recomputation after disaster effects are applied."""
        # Initial path
        result1 = self.pathfinder.find_shortest_path(self.graph, "A", "F")
        original_path = result1.path
        original_cost = result1.total_cost
        
        # Apply disaster that affects the optimal route
        disaster = DisasterEvent(DisasterType.FIRE, (2.0, 0.0), 0.8, 3.0)  # Near vertex B
        self.disaster_model.apply_disaster_effects(self.graph, disaster)
        
        # Recompute path after disaster
        result2 = self.pathfinder.find_shortest_path(self.graph, "A", "F")
        
        # Path should change due to disaster effects
        assert result2.found
        
        # Cost should be different (likely higher due to disaster penalties)
        if not any(edge.is_blocked for edge in self.graph.get_all_edges()):
            # If no edges are blocked, cost should increase due to risk penalties
            assert result2.total_cost >= original_cost
        
        # Remove disaster effects
        self.disaster_model.remove_disaster_effects(self.graph, disaster)
        
        # Path should return to original or similar
        result3 = self.pathfinder.find_shortest_path(self.graph, "A", "F")
        assert result3.found
        assert abs(result3.total_cost - original_cost) < 1e-10
    
    def test_disconnected_graph_detection(self):
        """Test detection of disconnected graph components."""
        # Add isolated vertices
        self.graph.add_vertex("G", VertexType.INTERSECTION, (10.0, 10.0))
        self.graph.add_vertex("H", VertexType.SHELTER, (12.0, 10.0), capacity=50)
        self.graph.add_edge("G", "H", 2.0, 0.0, 0.0)
        
        # Try to find path from main component to isolated component
        result = self.pathfinder.find_shortest_path(self.graph, "A", "G")
        
        assert not result.found
        assert result.path == []
        assert "No path exists from 'A' to 'G'" in result.error_message
        
        # Reverse direction should also fail
        result_reverse = self.pathfinder.find_shortest_path(self.graph, "G", "A")
        
        assert not result_reverse.found
        assert "No path exists from 'G' to 'A'" in result_reverse.error_message
        
        # Path within isolated component should work
        result_isolated = self.pathfinder.find_shortest_path(self.graph, "G", "H")
        
        assert result_isolated.found
        assert result_isolated.path == ["G", "H"]
        assert result_isolated.total_cost == 2.0
    
    def test_path_invalidation_with_blocked_edges(self):
        """Test path invalidation when edges become blocked."""
        # Find initial path
        result1 = self.pathfinder.find_shortest_path(self.graph, "A", "E")
        
        # Should be A -> B -> E (cost 3.0)
        assert result1.found
        assert result1.path == ["A", "B", "E"]
        assert result1.total_cost == 3.0
        
        # Block the B -> E edge
        edge_be = self.graph.get_edge("B", "E")
        edge_be.is_blocked = True
        self.graph.update_edge_weight("B", "E", float('inf'))
        
        # Recompute path
        result2 = self.pathfinder.find_shortest_path(self.graph, "A", "E")
        
        # Should find alternative path through F: A -> C -> D -> F -> E
        # But wait, there's no F -> E edge, so let's add one for this test
        self.graph.add_edge("F", "E", 2.0, 0.0, 0.0)
        
        result2 = self.pathfinder.find_shortest_path(self.graph, "A", "E")
        
        # Should find alternative path
        assert result2.found
        assert "B" not in result2.path or result2.path.index("E") != result2.path.index("B") + 1
        assert result2.total_cost > result1.total_cost  # Alternative path is longer
    
    def test_multiple_disconnected_components(self):
        """Test handling of multiple disconnected components."""
        # Create multiple isolated components
        
        # Component 1: I - J
        self.graph.add_vertex("I", VertexType.INTERSECTION, (20.0, 0.0))
        self.graph.add_vertex("J", VertexType.INTERSECTION, (22.0, 0.0))
        self.graph.add_edge("I", "J", 1.0, 0.0, 0.0)
        
        # Component 2: K - L - M
        self.graph.add_vertex("K", VertexType.INTERSECTION, (30.0, 0.0))
        self.graph.add_vertex("L", VertexType.INTERSECTION, (32.0, 0.0))
        self.graph.add_vertex("M", VertexType.SHELTER, (34.0, 0.0), capacity=75)
        self.graph.add_edge("K", "L", 1.5, 0.0, 0.0)
        self.graph.add_edge("L", "M", 1.5, 0.0, 0.0)
        
        # Test paths within each component work
        result_main = self.pathfinder.find_shortest_path(self.graph, "A", "F")
        assert result_main.found
        
        result_comp1 = self.pathfinder.find_shortest_path(self.graph, "I", "J")
        assert result_comp1.found
        assert result_comp1.total_cost == 1.0
        
        result_comp2 = self.pathfinder.find_shortest_path(self.graph, "K", "M")
        assert result_comp2.found
        assert result_comp2.total_cost == 3.0
        
        # Test paths between components fail
        result_cross1 = self.pathfinder.find_shortest_path(self.graph, "A", "I")
        assert not result_cross1.found
        
        result_cross2 = self.pathfinder.find_shortest_path(self.graph, "I", "K")
        assert not result_cross2.found
        
        result_cross3 = self.pathfinder.find_shortest_path(self.graph, "A", "M")
        assert not result_cross3.found
    
    def test_find_all_paths_with_disconnected_components(self):
        """Test find_all_shortest_paths with disconnected components."""
        # Add isolated component
        self.graph.add_vertex("X", VertexType.INTERSECTION, (50.0, 50.0))
        self.graph.add_vertex("Y", VertexType.INTERSECTION, (52.0, 50.0))
        self.graph.add_edge("X", "Y", 1.0, 0.0, 0.0)
        
        # Find all paths from A
        results = self.pathfinder.find_all_shortest_paths(self.graph, "A")
        
        # Should have results for all vertices
        all_vertices = self.graph.get_vertex_ids()
        assert len(results) == len(all_vertices)
        
        # Paths to main component should be found
        assert results["A"].found
        assert results["B"].found
        assert results["C"].found
        assert results["D"].found
        assert results["E"].found
        assert results["F"].found
        
        # Paths to isolated component should not be found
        assert not results["X"].found
        assert not results["Y"].found
        assert "No path exists" in results["X"].error_message
        assert "No path exists" in results["Y"].error_message
    
    def test_dynamic_graph_modification(self):
        """Test pathfinding with dynamic graph modifications."""
        # Initial path
        result1 = self.pathfinder.find_shortest_path(self.graph, "A", "F")
        initial_cost = result1.total_cost
        
        # Add new vertex and edges to create shortcut
        self.graph.add_vertex("SHORTCUT", VertexType.INTERSECTION, (1.0, 1.0))
        self.graph.add_edge("A", "SHORTCUT", 0.5, 0.0, 0.0)
        self.graph.add_edge("SHORTCUT", "F", 1.0, 0.0, 0.0)
        
        # Recompute path
        result2 = self.pathfinder.find_shortest_path(self.graph, "A", "F")
        
        # Should use the new shortcut
        assert result2.found
        assert result2.total_cost == 1.5  # 0.5 + 1.0
        assert result2.total_cost < initial_cost
        assert "SHORTCUT" in result2.path
        
        # Remove the shortcut
        self.graph.remove_vertex("SHORTCUT")
        
        # Path should revert
        result3 = self.pathfinder.find_shortest_path(self.graph, "A", "F")
        assert result3.found
        assert abs(result3.total_cost - initial_cost) < 1e-10
    
    def test_edge_weight_invalidation(self):
        """Test that cached paths are invalidated when edge weights change."""
        # This test verifies that the pathfinder doesn't cache results inappropriately
        
        # Find path multiple times with same weights
        result1 = self.pathfinder.find_shortest_path(self.graph, "A", "F")
        result2 = self.pathfinder.find_shortest_path(self.graph, "A", "F")
        
        # Results should be identical
        assert result1.path == result2.path
        assert result1.total_cost == result2.total_cost
        
        # Change edge weight
        self.graph.update_edge_weight("A", "B", 10.0)
        
        # New computation should reflect the change
        result3 = self.pathfinder.find_shortest_path(self.graph, "A", "F")
        
        # Path should be different if the weight change affects optimal route
        if "B" in result1.path:
            assert result3.path != result1.path or result3.total_cost != result1.total_cost
    
    def test_error_handling_edge_cases(self):
        """Test error handling for various edge cases."""
        # Empty graph
        empty_graph = GraphManager()
        result = self.pathfinder.find_shortest_path(empty_graph, "A", "B")
        assert not result.found
        assert "does not exist" in result.error_message
        
        # Single vertex graph
        single_graph = GraphManager()
        single_graph.add_vertex("ONLY", VertexType.INTERSECTION, (0.0, 0.0))
        
        result_self = self.pathfinder.find_shortest_path(single_graph, "ONLY", "ONLY")
        assert result_self.found
        assert result_self.path == ["ONLY"]
        assert result_self.total_cost == 0.0
        
        result_missing = self.pathfinder.find_shortest_path(single_graph, "ONLY", "MISSING")
        assert not result_missing.found
        
        # Graph with only blocked edges
        blocked_graph = GraphManager()
        blocked_graph.add_vertex("START", VertexType.INTERSECTION, (0.0, 0.0))
        blocked_graph.add_vertex("END", VertexType.INTERSECTION, (1.0, 0.0))
        blocked_graph.add_edge("START", "END", 1.0, 0.0, 0.0)
        
        # Block the only edge
        edge = blocked_graph.get_edge("START", "END")
        edge.is_blocked = True
        blocked_graph.update_edge_weight("START", "END", float('inf'))
        
        result_blocked = self.pathfinder.find_shortest_path(blocked_graph, "START", "END")
        assert not result_blocked.found
        assert "No path exists" in result_blocked.error_message