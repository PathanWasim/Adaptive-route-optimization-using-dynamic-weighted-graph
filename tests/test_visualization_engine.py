"""
Unit tests for VisualizationEngine class.
"""

import pytest
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing
import matplotlib.pyplot as plt
import os
import tempfile
from datetime import datetime
from disaster_evacuation.visualization import VisualizationEngine
from disaster_evacuation.models import GraphManager
from disaster_evacuation.models import DisasterEvent, DisasterType, VertexType


class TestVisualizationEngine:
    """Test suite for VisualizationEngine class."""
    
    @pytest.fixture
    def sample_graph(self):
        """Create a sample graph for testing."""
        graph = GraphManager()
        
        # Add vertices
        graph.add_vertex("A", VertexType.INTERSECTION, (0.0, 0.0))
        graph.add_vertex("B", VertexType.INTERSECTION, (1.0, 0.0))
        graph.add_vertex("C", VertexType.SHELTER, (2.0, 0.0), capacity=100)
        graph.add_vertex("D", VertexType.EVACUATION_POINT, (1.0, 1.0), capacity=500)
        graph.add_vertex("E", VertexType.INTERSECTION, (0.5, 0.5))
        
        # Add edges
        graph.add_edge("A", "B", 1.0, 0.1, 0.2)
        graph.add_edge("B", "C", 1.0, 0.2, 0.1)
        graph.add_edge("A", "D", 1.4, 0.3, 0.3)
        graph.add_edge("C", "D", 1.4, 0.1, 0.4)
        graph.add_edge("A", "E", 0.7, 0.1, 0.1)
        graph.add_edge("E", "B", 0.7, 0.1, 0.1)
        
        return graph
    
    @pytest.fixture
    def viz_engine(self):
        """Create a VisualizationEngine instance."""
        return VisualizationEngine()
    
    @pytest.fixture
    def sample_disaster(self):
        """Create a sample disaster event."""
        return DisasterEvent(
            disaster_type=DisasterType.FLOOD,
            epicenter=(0.5, 0.5),
            severity=0.7,
            max_effect_radius=1.5,
            start_time=datetime.now()
        )
    
    def test_initialization(self, viz_engine):
        """Test VisualizationEngine initialization."""
        assert viz_engine.figsize == (12, 8)
        assert VertexType.INTERSECTION in viz_engine.vertex_colors
        assert VertexType.SHELTER in viz_engine.vertex_colors
        assert VertexType.EVACUATION_POINT in viz_engine.vertex_colors
        assert DisasterType.FLOOD in viz_engine.disaster_colors
        assert DisasterType.FIRE in viz_engine.disaster_colors
        assert DisasterType.EARTHQUAKE in viz_engine.disaster_colors
    
    def test_custom_figsize(self):
        """Test initialization with custom figure size."""
        viz_engine = VisualizationEngine(figsize=(10, 6))
        assert viz_engine.figsize == (10, 6)
    
    def test_visualize_graph(self, viz_engine, sample_graph):
        """Test basic graph visualization."""
        fig = viz_engine.visualize_graph(sample_graph, title="Test Graph")
        
        assert fig is not None
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) > 0
        
        plt.close(fig)
    
    def test_visualize_empty_graph(self, viz_engine):
        """Test visualization of empty graph."""
        empty_graph = GraphManager()
        fig = viz_engine.visualize_graph(empty_graph)
        
        assert fig is not None
        assert isinstance(fig, plt.Figure)
        
        plt.close(fig)
    
    def test_visualize_path(self, viz_engine, sample_graph):
        """Test path visualization."""
        path = ["A", "E", "B", "C"]
        path_edges = [
            sample_graph.get_edge("A", "E"),
            sample_graph.get_edge("E", "B"),
            sample_graph.get_edge("B", "C")
        ]
        
        fig = viz_engine.visualize_path(sample_graph, path, path_edges, 
                                       title="Test Path")
        
        assert fig is not None
        assert isinstance(fig, plt.Figure)
        
        plt.close(fig)
    
    def test_visualize_empty_path(self, viz_engine, sample_graph):
        """Test visualization with empty path."""
        fig = viz_engine.visualize_path(sample_graph, [], [], title="Empty Path")
        
        assert fig is not None
        assert isinstance(fig, plt.Figure)
        
        plt.close(fig)
    
    def test_visualize_disaster_effects(self, viz_engine, sample_graph, sample_disaster):
        """Test disaster effects visualization."""
        # Get affected edges
        affected_edges = [
            sample_graph.get_edge("A", "E"),
            sample_graph.get_edge("E", "B")
        ]
        
        fig = viz_engine.visualize_disaster_effects(
            sample_graph, sample_disaster, affected_edges,
            title="Test Disaster"
        )
        
        assert fig is not None
        assert isinstance(fig, plt.Figure)
        
        plt.close(fig)
    
    def test_visualize_path_comparison(self, viz_engine, sample_graph):
        """Test path comparison visualization."""
        normal_path = ["A", "B", "C"]
        normal_edges = [
            sample_graph.get_edge("A", "B"),
            sample_graph.get_edge("B", "C")
        ]
        
        disaster_path = ["A", "E", "B", "C"]
        disaster_edges = [
            sample_graph.get_edge("A", "E"),
            sample_graph.get_edge("E", "B"),
            sample_graph.get_edge("B", "C")
        ]
        
        fig = viz_engine.visualize_path_comparison(
            sample_graph, normal_path, normal_edges,
            disaster_path, disaster_edges,
            title="Test Comparison"
        )
        
        assert fig is not None
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 2  # Should have two subplots
        
        plt.close(fig)
    
    def test_visualize_path_comparison_with_disaster(self, viz_engine, sample_graph, sample_disaster):
        """Test path comparison with disaster visualization."""
        normal_path = ["A", "B", "C"]
        normal_edges = [
            sample_graph.get_edge("A", "B"),
            sample_graph.get_edge("B", "C")
        ]
        
        disaster_path = ["A", "D"]
        disaster_edges = [sample_graph.get_edge("A", "D")]
        
        fig = viz_engine.visualize_path_comparison(
            sample_graph, normal_path, normal_edges,
            disaster_path, disaster_edges,
            disaster=sample_disaster,
            title="Test Comparison with Disaster"
        )
        
        assert fig is not None
        assert isinstance(fig, plt.Figure)
        
        plt.close(fig)
    
    def test_save_graph_visualization(self, viz_engine, sample_graph):
        """Test saving graph visualization to file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "test_graph.png")
            
            fig = viz_engine.visualize_graph(sample_graph, save_path=save_path)
            
            assert os.path.exists(save_path)
            assert os.path.getsize(save_path) > 0
            
            plt.close(fig)
    
    def test_save_path_visualization(self, viz_engine, sample_graph):
        """Test saving path visualization to file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "test_path.png")
            
            path = ["A", "B", "C"]
            path_edges = [
                sample_graph.get_edge("A", "B"),
                sample_graph.get_edge("B", "C")
            ]
            
            fig = viz_engine.visualize_path(sample_graph, path, path_edges, 
                                           save_path=save_path)
            
            assert os.path.exists(save_path)
            assert os.path.getsize(save_path) > 0
            
            plt.close(fig)
    
    def test_save_disaster_visualization(self, viz_engine, sample_graph, sample_disaster):
        """Test saving disaster visualization to file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "test_disaster.png")
            
            affected_edges = [sample_graph.get_edge("A", "E")]
            
            fig = viz_engine.visualize_disaster_effects(
                sample_graph, sample_disaster, affected_edges,
                save_path=save_path
            )
            
            assert os.path.exists(save_path)
            assert os.path.getsize(save_path) > 0
            
            plt.close(fig)
    
    def test_save_comparison_visualization(self, viz_engine, sample_graph):
        """Test saving comparison visualization to file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "test_comparison.png")
            
            normal_path = ["A", "B"]
            normal_edges = [sample_graph.get_edge("A", "B")]
            
            disaster_path = ["A", "E", "B"]
            disaster_edges = [
                sample_graph.get_edge("A", "E"),
                sample_graph.get_edge("E", "B")
            ]
            
            fig = viz_engine.visualize_path_comparison(
                sample_graph, normal_path, normal_edges,
                disaster_path, disaster_edges,
                save_path=save_path
            )
            
            assert os.path.exists(save_path)
            assert os.path.getsize(save_path) > 0
            
            plt.close(fig)
    
    def test_vertex_colors_configuration(self, viz_engine):
        """Test vertex color configuration."""
        assert viz_engine.vertex_colors[VertexType.INTERSECTION] == '#4CAF50'
        assert viz_engine.vertex_colors[VertexType.SHELTER] == '#2196F3'
        assert viz_engine.vertex_colors[VertexType.EVACUATION_POINT] == '#FF9800'
    
    def test_vertex_sizes_configuration(self, viz_engine):
        """Test vertex size configuration."""
        assert viz_engine.vertex_sizes[VertexType.INTERSECTION] == 300
        assert viz_engine.vertex_sizes[VertexType.SHELTER] == 500
        assert viz_engine.vertex_sizes[VertexType.EVACUATION_POINT] == 600
    
    def test_disaster_colors_configuration(self, viz_engine):
        """Test disaster color configuration."""
        assert viz_engine.disaster_colors[DisasterType.FLOOD] == '#1976D2'
        assert viz_engine.disaster_colors[DisasterType.FIRE] == '#D32F2F'
        assert viz_engine.disaster_colors[DisasterType.EARTHQUAKE] == '#7B1FA2'
    
    def test_string_representation(self, viz_engine):
        """Test string representation of visualization engine."""
        str_repr = str(viz_engine)
        
        assert "VisualizationEngine" in str_repr
        assert "figsize" in str_repr
    
    def test_multiple_visualizations(self, viz_engine, sample_graph):
        """Test creating multiple visualizations."""
        # Create multiple figures
        fig1 = viz_engine.visualize_graph(sample_graph, title="Graph 1")
        fig2 = viz_engine.visualize_graph(sample_graph, title="Graph 2")
        
        assert fig1 is not None
        assert fig2 is not None
        assert fig1 != fig2
        
        plt.close(fig1)
        plt.close(fig2)
    
    def test_visualization_with_different_vertex_types(self, viz_engine):
        """Test visualization with all vertex types."""
        graph = GraphManager()
        
        graph.add_vertex("I1", VertexType.INTERSECTION, (0.0, 0.0))
        graph.add_vertex("S1", VertexType.SHELTER, (1.0, 0.0), capacity=100)
        graph.add_vertex("E1", VertexType.EVACUATION_POINT, (2.0, 0.0), capacity=500)
        
        graph.add_edge("I1", "S1", 1.0, 0.1, 0.1)
        graph.add_edge("S1", "E1", 1.0, 0.1, 0.1)
        
        fig = viz_engine.visualize_graph(graph)
        
        assert fig is not None
        
        plt.close(fig)
    
    def test_visualization_with_different_disaster_types(self, viz_engine, sample_graph):
        """Test visualization with different disaster types."""
        disaster_types = [DisasterType.FLOOD, DisasterType.FIRE, DisasterType.EARTHQUAKE]
        
        for disaster_type in disaster_types:
            disaster = DisasterEvent(
                disaster_type=disaster_type,
                epicenter=(1.0, 0.5),
                severity=0.6,
                max_effect_radius=1.0,
                start_time=datetime.now()
            )
            
            affected_edges = [sample_graph.get_edge("A", "B")]
            
            fig = viz_engine.visualize_disaster_effects(
                sample_graph, disaster, affected_edges,
                title=f"{disaster_type.value.title()} Test"
            )
            
            assert fig is not None
            
            plt.close(fig)
    
    def test_save_visualization_summary(self, viz_engine, sample_graph):
        """Test saving complete visualization summary."""
        with tempfile.TemporaryDirectory() as tmpdir:
            route_result = {
                "success": True,
                "path": ["A", "B", "C"],
                "edges": [
                    sample_graph.get_edge("A", "B"),
                    sample_graph.get_edge("B", "C")
                ],
                "source": {"id": "A"},
                "destination": {"id": "C"},
                "disaster_applied": False,
                "disaster_info": None
            }
            
            saved_files = viz_engine.save_visualization_summary(
                sample_graph, route_result, save_dir=tmpdir
            )
            
            assert "network" in saved_files
            assert "route" in saved_files
            assert os.path.exists(saved_files["network"])
            assert os.path.exists(saved_files["route"])
    
    def test_save_visualization_summary_failed_route(self, viz_engine, sample_graph):
        """Test saving visualization summary for failed route."""
        with tempfile.TemporaryDirectory() as tmpdir:
            route_result = {
                "success": False,
                "error": "No path found"
            }
            
            saved_files = viz_engine.save_visualization_summary(
                sample_graph, route_result, save_dir=tmpdir
            )
            
            # Should still save network graph
            assert "network" in saved_files
            assert os.path.exists(saved_files["network"])
            
            # Should not save route
            assert "route" not in saved_files
    
    def test_visualize_dynamic_comparison(self, viz_engine, sample_graph):
        """Test dynamic comparison visualization."""
        comparison_data = {
            "success": True,
            "normal_route": {
                "found": True,
                "path": ["A", "B", "C"],
                "cost": 2.3,
                "edge_count": 2
            },
            "disaster_aware_route": {
                "found": True,
                "path": ["A", "E", "B", "C"],
                "cost": 2.8,
                "edge_count": 3
            },
            "analysis": {
                "path_changed": True,
                "cost_increase": 0.5,
                "cost_increase_percentage": 21.7
            },
            "disaster_info": {
                "type": "flood",
                "epicenter": [0.5, 0.5],
                "severity": 0.7,
                "effect_radius": 1.5
            }
        }
        
        fig = viz_engine.visualize_dynamic_comparison(
            sample_graph, comparison_data, title="Test Dynamic Comparison"
        )
        
        assert fig is not None
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) >= 4  # Should have multiple subplots
        
        plt.close(fig)
    
    def test_visualize_real_time_updates(self, viz_engine, sample_graph):
        """Test real-time updates visualization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            update_sequence = [
                {
                    "success": True,
                    "path": ["A", "B", "C"],
                    "edges": [sample_graph.get_edge("A", "B"), sample_graph.get_edge("B", "C")],
                    "total_cost": 2.3,
                    "timestamp": "10:00:00",
                    "disaster_applied": False
                },
                {
                    "success": True,
                    "path": ["A", "E", "B", "C"],
                    "edges": [
                        sample_graph.get_edge("A", "E"),
                        sample_graph.get_edge("E", "B"),
                        sample_graph.get_edge("B", "C")
                    ],
                    "total_cost": 2.8,
                    "timestamp": "10:05:00",
                    "disaster_applied": True,
                    "disaster_info": {
                        "type": "flood",
                        "epicenter": [0.5, 0.5],
                        "severity": 0.7,
                        "effect_radius": 1.5
                    }
                }
            ]
            
            saved_files = viz_engine.visualize_real_time_updates(
                sample_graph, update_sequence, save_dir=tmpdir
            )
            
            assert len(saved_files) == 2
            for file_path in saved_files:
                assert os.path.exists(file_path)
                assert os.path.getsize(file_path) > 0
    
    def test_create_interactive_comparison_dashboard(self, viz_engine, sample_graph):
        """Test interactive dashboard creation (if plotly is available)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            scenarios = [
                {
                    "success": True,
                    "path": ["A", "B", "C"],
                    "total_cost": 2.3,
                    "statistics": {"total_risk": 0.3}
                },
                {
                    "success": True,
                    "path": ["A", "E", "B", "C"],
                    "total_cost": 2.8,
                    "statistics": {"total_risk": 0.2}
                }
            ]
            
            save_path = os.path.join(tmpdir, "dashboard.html")
            
            # This test will pass even if plotly is not available
            try:
                viz_engine.create_interactive_comparison_dashboard(
                    sample_graph, scenarios, save_path
                )
                # If plotly is available, file should be created
                if os.path.exists(save_path):
                    assert os.path.getsize(save_path) > 0
            except ImportError:
                # Plotly not available, which is fine for testing
                pass
    
    def test_dynamic_comparison_with_failed_routes(self, viz_engine, sample_graph):
        """Test dynamic comparison with failed routes."""
        comparison_data = {
            "success": False,
            "normal_route": {
                "found": True,
                "path": ["A", "B", "C"],
                "cost": 2.3,
                "edge_count": 2
            },
            "disaster_aware_route": {
                "found": False,
                "path": [],
                "cost": None,
                "edge_count": 0
            }
        }
        
        fig = viz_engine.visualize_dynamic_comparison(
            sample_graph, comparison_data, title="Test Failed Route Comparison"
        )
        
        assert fig is not None
        assert isinstance(fig, plt.Figure)
        
        plt.close(fig)
    
    def test_real_time_updates_with_failed_routes(self, viz_engine, sample_graph):
        """Test real-time updates with some failed routes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            update_sequence = [
                {
                    "success": True,
                    "path": ["A", "B", "C"],
                    "edges": [sample_graph.get_edge("A", "B"), sample_graph.get_edge("B", "C")],
                    "total_cost": 2.3,
                    "timestamp": "10:00:00"
                },
                {
                    "success": False,
                    "timestamp": "10:05:00",
                    "disaster_applied": True
                }
            ]
            
            saved_files = viz_engine.visualize_real_time_updates(
                sample_graph, update_sequence, save_dir=tmpdir
            )
            
            assert len(saved_files) == 2
            for file_path in saved_files:
                assert os.path.exists(file_path)
    
    def test_comparison_data_edge_cases(self, viz_engine, sample_graph):
        """Test comparison visualization with edge cases."""
        # Empty comparison data
        empty_data = {
            "normal_route": {"found": False},
            "disaster_aware_route": {"found": False}
        }
        
        fig = viz_engine.visualize_dynamic_comparison(
            sample_graph, empty_data, title="Empty Comparison"
        )
        
        assert fig is not None
        plt.close(fig)
        
        # Partial data
        partial_data = {
            "normal_route": {
                "found": True,
                "path": ["A", "B"],
                "cost": 1.0
            },
            "disaster_aware_route": {"found": False}
        }
        
        fig = viz_engine.visualize_dynamic_comparison(
            sample_graph, partial_data, title="Partial Comparison"
        )
        
        assert fig is not None
        plt.close(fig)