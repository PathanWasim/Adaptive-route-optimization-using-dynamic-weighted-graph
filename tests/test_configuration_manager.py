"""
Unit tests for ConfigurationManager class.
"""

import pytest
import json
import tempfile
import shutil
from pathlib import Path
from disaster_evacuation.config import ConfigurationManager
from disaster_evacuation.graph import GraphManager
from disaster_evacuation.models import VertexType


class TestConfigurationManager:
    """Test suite for ConfigurationManager class."""
    
    @pytest.fixture
    def temp_config_dir(self):
        """Create a temporary directory for configuration files."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        # Cleanup happens after tests complete
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def config_manager(self, temp_config_dir):
        """Create a ConfigurationManager instance with temporary directory."""
        manager = ConfigurationManager(temp_config_dir)
        yield manager
        # Explicitly close to release file handles
        manager.close()
    
    @pytest.fixture
    def sample_graph(self):
        """Create a sample graph for testing."""
        graph = GraphManager()
        graph.add_vertex("A", VertexType.INTERSECTION, (0.0, 0.0))
        graph.add_vertex("B", VertexType.SHELTER, (1.0, 0.0), capacity=100)
        graph.add_vertex("C", VertexType.EVACUATION_POINT, (2.0, 0.0), capacity=500)
        
        graph.add_edge("A", "B", 1.0, 0.1, 0.2)
        graph.add_edge("B", "C", 1.5, 0.2, 0.1)
        graph.add_edge("A", "C", 2.0, 0.15, 0.15)
        
        return graph
    
    def test_initialization(self, config_manager, temp_config_dir):
        """Test ConfigurationManager initialization."""
        assert config_manager.config_dir == Path(temp_config_dir)
        assert config_manager.config_dir.exists()
        assert isinstance(config_manager.algorithm_settings, dict)
        assert isinstance(config_manager.disaster_parameters, dict)
    
    def test_save_graph_configuration(self, config_manager, sample_graph):
        """Test saving graph configuration to JSON."""
        metadata = {"description": "Test graph", "author": "Test"}
        
        result = config_manager.save_graph_configuration(sample_graph, "test_config", metadata)
        
        assert result is True
        config_file = config_manager.config_dir / "test_config.json"
        assert config_file.exists()
        
        # Verify file content
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        assert "version" in config
        assert "created_at" in config
        assert "metadata" in config
        assert config["metadata"]["description"] == "Test graph"
        assert "graph" in config
        assert len(config["graph"]["vertices"]) == 3
        assert len(config["graph"]["edges"]) == 3
    
    def test_load_graph_configuration(self, config_manager, sample_graph):
        """Test loading graph configuration from JSON."""
        # Save first
        config_manager.save_graph_configuration(sample_graph, "test_load")
        
        # Load
        result = config_manager.load_graph_configuration("test_load")
        
        assert result is not None
        loaded_graph, metadata = result
        
        # Verify graph structure
        assert len(loaded_graph.get_vertex_ids()) == 3
        assert len(loaded_graph.get_all_edges()) == 3
        
        # Verify vertices
        vertex_a = loaded_graph.get_vertex("A")
        assert vertex_a is not None
        assert vertex_a.vertex_type == VertexType.INTERSECTION
        
        vertex_b = loaded_graph.get_vertex("B")
        assert vertex_b is not None
        assert vertex_b.vertex_type == VertexType.SHELTER
        assert vertex_b.capacity == 100
        
        # Verify metadata
        assert "loaded_at" in metadata
        assert "source_file" in metadata
    
    def test_load_nonexistent_configuration(self, config_manager):
        """Test loading a nonexistent configuration file."""
        result = config_manager.load_graph_configuration("nonexistent")
        assert result is None
    
    def test_save_load_round_trip(self, config_manager, sample_graph):
        """Test that save and load preserve graph structure."""
        # Save
        config_manager.save_graph_configuration(sample_graph, "roundtrip")
        
        # Load
        loaded_graph, _ = config_manager.load_graph_configuration("roundtrip")
        
        # Compare original and loaded graphs
        original_vertices = set(sample_graph.get_vertex_ids())
        loaded_vertices = set(loaded_graph.get_vertex_ids())
        assert original_vertices == loaded_vertices
        
        original_edges = len(sample_graph.get_all_edges())
        loaded_edges = len(loaded_graph.get_all_edges())
        assert original_edges == loaded_edges
    
    def test_list_configurations(self, config_manager, sample_graph):
        """Test listing available configuration files."""
        # Initially empty
        configs = config_manager.list_configurations()
        initial_count = len(configs)
        
        # Save some configurations
        config_manager.save_graph_configuration(sample_graph, "config1")
        config_manager.save_graph_configuration(sample_graph, "config2")
        
        # List again
        configs = config_manager.list_configurations()
        assert len(configs) == initial_count + 2
        assert "config1.json" in configs
        assert "config2.json" in configs
    
    def test_delete_configuration(self, config_manager, sample_graph):
        """Test deleting a configuration file."""
        # Save a configuration
        config_manager.save_graph_configuration(sample_graph, "to_delete")
        
        # Verify it exists
        configs = config_manager.list_configurations()
        assert "to_delete.json" in configs
        
        # Delete it
        result = config_manager.delete_configuration("to_delete")
        assert result is True
        
        # Verify it's gone
        configs = config_manager.list_configurations()
        assert "to_delete.json" not in configs
    
    def test_delete_nonexistent_configuration(self, config_manager):
        """Test deleting a nonexistent configuration."""
        result = config_manager.delete_configuration("nonexistent")
        assert result is False
    
    def test_algorithm_settings(self, config_manager):
        """Test getting and setting algorithm settings."""
        # Get default setting
        algorithm = config_manager.get_algorithm_setting("pathfinding_algorithm")
        assert algorithm == "dijkstra"
        
        # Set new value
        config_manager.set_algorithm_setting("max_iterations", 5000)
        assert config_manager.get_algorithm_setting("max_iterations") == 5000
        
        # Get nonexistent setting with default
        value = config_manager.get_algorithm_setting("nonexistent", "default_value")
        assert value == "default_value"
    
    def test_disaster_parameters(self, config_manager):
        """Test getting and setting disaster parameters."""
        # Get default parameter
        severity = config_manager.get_disaster_parameter("flood", "default_severity")
        assert severity == 0.7
        
        # Set new value
        config_manager.set_disaster_parameter("flood", "default_severity", 0.8)
        assert config_manager.get_disaster_parameter("flood", "default_severity") == 0.8
        
        # Set parameter for new disaster type
        config_manager.set_disaster_parameter("tsunami", "default_severity", 0.9)
        assert config_manager.get_disaster_parameter("tsunami", "default_severity") == 0.9
        
        # Get nonexistent parameter with default
        value = config_manager.get_disaster_parameter("flood", "nonexistent", 0.5)
        assert value == 0.5
    
    def test_export_to_geojson(self, config_manager, sample_graph, temp_config_dir):
        """Test exporting graph to GeoJSON format."""
        output_path = Path(temp_config_dir) / "test_export.geojson"
        
        result = config_manager.export_to_geojson(sample_graph, str(output_path))
        
        assert result is True
        assert output_path.exists()
        
        # Verify GeoJSON structure
        with open(output_path, 'r') as f:
            geojson = json.load(f)
        
        assert geojson["type"] == "FeatureCollection"
        assert "features" in geojson
        
        # Count features (3 vertices + 3 edges = 6 features)
        assert len(geojson["features"]) == 6
        
        # Verify vertex features
        point_features = [f for f in geojson["features"] if f["geometry"]["type"] == "Point"]
        assert len(point_features) == 3
        
        # Verify edge features
        line_features = [f for f in geojson["features"] if f["geometry"]["type"] == "LineString"]
        assert len(line_features) == 3
    
    def test_import_from_geojson(self, config_manager, temp_config_dir):
        """Test importing graph from GeoJSON format."""
        # Create a test GeoJSON file
        geojson_path = Path(temp_config_dir) / "test_import.geojson"
        geojson_data = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [0.0, 0.0]},
                    "properties": {"id": "V1", "type": "intersection"}
                },
                {
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [1.0, 1.0]},
                    "properties": {"id": "V2", "type": "shelter", "capacity": 100}
                },
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "LineString",
                        "coordinates": [[0.0, 0.0], [1.0, 1.0]]
                    },
                    "properties": {
                        "source": "V1",
                        "target": "V2",
                        "distance": 1.4,
                        "risk": 0.1,
                        "congestion": 0.2
                    }
                }
            ]
        }
        
        with open(geojson_path, 'w') as f:
            json.dump(geojson_data, f)
        
        # Import
        graph = config_manager.import_from_geojson(str(geojson_path))
        
        assert graph is not None
        assert len(graph.get_vertex_ids()) == 2
        assert len(graph.get_all_edges()) == 1
        
        # Verify vertices
        v1 = graph.get_vertex("V1")
        assert v1 is not None
        assert v1.vertex_type == VertexType.INTERSECTION
        
        v2 = graph.get_vertex("V2")
        assert v2 is not None
        assert v2.vertex_type == VertexType.SHELTER
        assert v2.capacity == 100
    
    def test_import_invalid_geojson(self, config_manager, temp_config_dir):
        """Test importing invalid GeoJSON."""
        geojson_path = Path(temp_config_dir) / "invalid.geojson"
        
        # Create invalid GeoJSON (not a FeatureCollection)
        with open(geojson_path, 'w') as f:
            json.dump({"type": "Feature"}, f)
        
        graph = config_manager.import_from_geojson(str(geojson_path))
        assert graph is None
    
    def test_graph_integrity_validation(self, config_manager):
        """Test graph integrity validation."""
        # Valid graph
        valid_graph = GraphManager()
        valid_graph.add_vertex("A", VertexType.INTERSECTION, (0.0, 0.0))
        valid_graph.add_vertex("B", VertexType.INTERSECTION, (1.0, 0.0))
        valid_graph.add_edge("A", "B", 1.0, 0.1, 0.2)
        
        assert config_manager._validate_graph_integrity(valid_graph) is True
    
    def test_configuration_with_metadata(self, config_manager, sample_graph):
        """Test saving and loading configuration with metadata."""
        metadata = {
            "name": "City Network",
            "description": "Urban evacuation network",
            "version": "1.0",
            "tags": ["urban", "evacuation"]
        }
        
        config_manager.save_graph_configuration(sample_graph, "with_metadata", metadata)
        loaded_graph, loaded_metadata = config_manager.load_graph_configuration("with_metadata")
        
        assert loaded_metadata["name"] == "City Network"
        assert loaded_metadata["description"] == "Urban evacuation network"
        assert "tags" in loaded_metadata
    
    def test_string_representation(self, config_manager):
        """Test string representation of ConfigurationManager."""
        str_repr = str(config_manager)
        assert "ConfigurationManager" in str_repr
        assert "config_dir" in str_repr
    
    def test_save_with_json_extension(self, config_manager, sample_graph):
        """Test saving with .json extension already in filename."""
        result = config_manager.save_graph_configuration(sample_graph, "test.json")
        assert result is True
        
        # Should not create test.json.json
        configs = config_manager.list_configurations()
        assert "test.json" in configs
        assert "test.json.json" not in configs
    
    def test_load_with_json_extension(self, config_manager, sample_graph):
        """Test loading with .json extension in filename."""
        config_manager.save_graph_configuration(sample_graph, "test_ext")
        
        # Load with extension
        result = config_manager.load_graph_configuration("test_ext.json")
        assert result is not None
    
    def test_empty_graph_save_load(self, config_manager):
        """Test saving and loading an empty graph."""
        empty_graph = GraphManager()
        
        result = config_manager.save_graph_configuration(empty_graph, "empty")
        assert result is True
        
        loaded_graph, _ = config_manager.load_graph_configuration("empty")
        assert loaded_graph is not None
        assert len(loaded_graph.get_vertex_ids()) == 0
        assert len(loaded_graph.get_all_edges()) == 0
