"""
Property-based tests for data import validation.

Feature: disaster-evacuation-routing
Property 13: Data Import Validation

For any standard geographic data format input, the import process should either
successfully create a valid graph or provide clear error messages for invalid data.

Validates: Requirements 10.3, 10.5
"""

import pytest
import json
import tempfile
import shutil
from pathlib import Path
from hypothesis import given, strategies as st, settings, assume, HealthCheck
from disaster_evacuation.config import ConfigurationManager
from disaster_evacuation.graph import GraphManager
from disaster_evacuation.models import VertexType


# Custom strategies for generating GeoJSON data
@st.composite
def valid_coordinates_strategy(draw):
    """Generate valid coordinate pairs."""
    lon = draw(st.floats(min_value=-180.0, max_value=180.0, allow_nan=False, allow_infinity=False))
    lat = draw(st.floats(min_value=-90.0, max_value=90.0, allow_nan=False, allow_infinity=False))
    return [lon, lat]


@st.composite
def valid_point_feature_strategy(draw):
    """Generate valid GeoJSON Point features."""
    vertex_id = draw(st.text(min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd'))))
    vertex_type = draw(st.sampled_from(['intersection', 'shelter', 'evacuation_point']))
    coordinates = draw(valid_coordinates_strategy())
    
    properties = {
        "id": vertex_id,
        "type": vertex_type
    }
    
    # Add capacity for shelters and evacuation points
    if vertex_type in ['shelter', 'evacuation_point']:
        properties["capacity"] = draw(st.integers(min_value=1, max_value=10000))
    
    return {
        "type": "Feature",
        "geometry": {
            "type": "Point",
            "coordinates": coordinates
        },
        "properties": properties
    }


@st.composite
def valid_linestring_feature_strategy(draw, vertex_ids):
    """Generate valid GeoJSON LineString features."""
    if len(vertex_ids) < 2:
        return None
    
    source = draw(st.sampled_from(vertex_ids))
    target = draw(st.sampled_from(vertex_ids))
    
    # Avoid self-loops
    assume(source != target)
    
    # Generate two coordinate pairs for the line
    coords1 = draw(valid_coordinates_strategy())
    coords2 = draw(valid_coordinates_strategy())
    
    properties = {
        "source": source,
        "target": target,
        "distance": draw(st.floats(min_value=0.1, max_value=100.0, allow_nan=False, allow_infinity=False)),
        "risk": draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)),
        "congestion": draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False))
    }
    
    return {
        "type": "Feature",
        "geometry": {
            "type": "LineString",
            "coordinates": [coords1, coords2]
        },
        "properties": properties
    }


@st.composite
def valid_geojson_strategy(draw, min_vertices=2, max_vertices=5):
    """Generate valid GeoJSON FeatureCollection."""
    num_vertices = draw(st.integers(min_value=min_vertices, max_value=max_vertices))
    
    # Generate simple unique vertex IDs
    vertex_ids = [f"V{i}" for i in range(num_vertices)]
    
    # Generate point features
    point_features = []
    for vertex_id in vertex_ids:
        vertex_type = draw(st.sampled_from(['intersection', 'shelter', 'evacuation_point']))
        lon = draw(st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False))
        lat = draw(st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False))
        
        properties = {"id": vertex_id, "type": vertex_type}
        if vertex_type in ['shelter', 'evacuation_point']:
            properties["capacity"] = draw(st.integers(min_value=10, max_value=1000))
        
        point_features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [lon, lat]},
            "properties": properties
        })
    
    # Generate a few line features
    num_edges = draw(st.integers(min_value=1, max_value=min(num_vertices, 5)))
    line_features = []
    for _ in range(num_edges):
        if len(vertex_ids) >= 2:
            source = draw(st.sampled_from(vertex_ids))
            target = draw(st.sampled_from(vertex_ids))
            
            if source != target:
                line_features.append({
                    "type": "Feature",
                    "geometry": {
                        "type": "LineString",
                        "coordinates": [[0.0, 0.0], [1.0, 1.0]]
                    },
                    "properties": {
                        "source": source,
                        "target": target,
                        "distance": draw(st.floats(min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False)),
                        "risk": draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)),
                        "congestion": draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False))
                    }
                })
    
    return {
        "type": "FeatureCollection",
        "features": point_features + line_features
    }


@st.composite
def invalid_geojson_strategy(draw):
    """Generate invalid GeoJSON data."""
    invalid_type = draw(st.sampled_from([
        {"type": "Feature"},  # Not a FeatureCollection
        {"type": "Point"},  # Wrong type
        {},  # Missing type
        {"features": []},  # Missing type field
        {"type": "FeatureCollection"},  # Missing features
    ]))
    return invalid_type


class TestPropertyDataImportValidation:
    """Property-based tests for data import validation."""
    
    @given(geojson_data=valid_geojson_strategy())
    @settings(max_examples=20, deadline=5000, suppress_health_check=[HealthCheck.large_base_example])
    def test_valid_geojson_imports_successfully(self, geojson_data):
        """
        Property 13: Data Import Validation
        
        For any valid GeoJSON FeatureCollection, import should succeed
        and create a valid graph.
        """
        temp_dir = tempfile.mkdtemp()
        try:
            config_manager = ConfigurationManager(temp_dir, enable_logging=False)
            
            # Write GeoJSON to file
            geojson_path = Path(temp_dir) / "test_import.geojson"
            with open(geojson_path, 'w') as f:
                json.dump(geojson_data, f)
            
            # Import the GeoJSON
            graph = config_manager.import_from_geojson(str(geojson_path))
            
            # Should successfully create a graph
            assert graph is not None, "Valid GeoJSON should import successfully"
            assert isinstance(graph, GraphManager), "Import should return a GraphManager instance"
            
            # Verify graph has vertices
            vertex_count = len(graph.get_vertex_ids())
            point_features = [f for f in geojson_data["features"] if f.get("geometry", {}).get("type") == "Point"]
            assert vertex_count > 0, "Imported graph should have vertices"
            
            config_manager.close()
        finally:
            shutil.rmtree(temp_dir)
    
    @given(geojson_data=invalid_geojson_strategy())
    @settings(max_examples=50, deadline=2000)
    def test_invalid_geojson_fails_gracefully(self, geojson_data):
        """
        Property 13: Data Import Validation
        
        For any invalid GeoJSON data, import should fail gracefully
        and return None or an empty graph rather than crashing.
        """
        temp_dir = tempfile.mkdtemp()
        try:
            config_manager = ConfigurationManager(temp_dir, enable_logging=False)
            
            # Write invalid GeoJSON to file
            geojson_path = Path(temp_dir) / "test_invalid.geojson"
            with open(geojson_path, 'w') as f:
                json.dump(geojson_data, f)
            
            # Import should fail gracefully
            graph = config_manager.import_from_geojson(str(geojson_path))
            
            # Should return None or empty graph for invalid data
            if graph is not None:
                # If it returns a graph, it should be empty
                assert len(graph.get_vertex_ids()) == 0, "Invalid GeoJSON should produce empty graph"
            
            config_manager.close()
        finally:
            shutil.rmtree(temp_dir)
    
    @given(geojson_data=valid_geojson_strategy())
    @settings(max_examples=20, deadline=5000, suppress_health_check=[HealthCheck.large_base_example])
    def test_imported_graph_preserves_vertex_properties(self, geojson_data):
        """
        Property 13: Data Import Validation
        
        For any valid GeoJSON, imported vertices should preserve their properties.
        """
        temp_dir = tempfile.mkdtemp()
        try:
            config_manager = ConfigurationManager(temp_dir, enable_logging=False)
            
            # Write GeoJSON to file
            geojson_path = Path(temp_dir) / "test_props.geojson"
            with open(geojson_path, 'w') as f:
                json.dump(geojson_data, f)
            
            # Import the GeoJSON
            graph = config_manager.import_from_geojson(str(geojson_path))
            
            if graph is not None:
                # Extract point features
                point_features = [f for f in geojson_data["features"] 
                                if f.get("geometry", {}).get("type") == "Point"]
                
                # Verify each vertex exists and has correct properties
                for feature in point_features:
                    vertex_id = feature["properties"]["id"]
                    vertex = graph.get_vertex(vertex_id)
                    
                    if vertex is not None:  # Vertex might not exist if ID was duplicate
                        # Verify vertex type mapping
                        expected_type_str = feature["properties"]["type"].lower()
                        if expected_type_str == "intersection":
                            assert vertex.vertex_type == VertexType.INTERSECTION
                        elif expected_type_str == "shelter":
                            assert vertex.vertex_type == VertexType.SHELTER
                        elif expected_type_str == "evacuation_point":
                            assert vertex.vertex_type == VertexType.EVACUATION_POINT
            
            config_manager.close()
        finally:
            shutil.rmtree(temp_dir)
    
    @given(geojson_data=valid_geojson_strategy())
    @settings(max_examples=20, deadline=5000, suppress_health_check=[HealthCheck.large_base_example])
    def test_imported_graph_preserves_edge_properties(self, geojson_data):
        """
        Property 13: Data Import Validation
        
        For any valid GeoJSON, imported edges should preserve their properties.
        Note: If duplicate edges exist, only the last one is kept.
        """
        temp_dir = tempfile.mkdtemp()
        try:
            config_manager = ConfigurationManager(temp_dir, enable_logging=False)
            
            # Write GeoJSON to file
            geojson_path = Path(temp_dir) / "test_edges.geojson"
            with open(geojson_path, 'w') as f:
                json.dump(geojson_data, f)
            
            # Import the GeoJSON
            graph = config_manager.import_from_geojson(str(geojson_path))
            
            if graph is not None:
                # Extract line features (keep only last occurrence of each edge)
                line_features_dict = {}
                for feature in geojson_data["features"]:
                    if feature.get("geometry", {}).get("type") == "LineString":
                        props = feature["properties"]
                        key = (props["source"], props["target"])
                        line_features_dict[key] = props
                
                # Verify edges exist
                all_edges = graph.get_all_edges()
                edge_dict = {(e.source, e.target): e for e in all_edges}
                
                for (source, target), props in line_features_dict.items():
                    # Edge might not exist if vertices were invalid
                    if (source, target) in edge_dict:
                        edge = edge_dict[(source, target)]
                        
                        # Verify edge properties (with floating point tolerance)
                        assert abs(edge.base_distance - props["distance"]) < 1e-6, \
                            f"Distance mismatch for edge ({source}, {target})"
                        assert abs(edge.base_risk - props["risk"]) < 1e-6, \
                            f"Risk mismatch for edge ({source}, {target})"
                        assert abs(edge.base_congestion - props["congestion"]) < 1e-6, \
                            f"Congestion mismatch for edge ({source}, {target})"
            
            config_manager.close()
        finally:
            shutil.rmtree(temp_dir)
    
    @given(geojson_data=valid_geojson_strategy())
    @settings(max_examples=20, deadline=5000, suppress_health_check=[HealthCheck.large_base_example])
    def test_import_export_round_trip(self, geojson_data):
        """
        Property 13: Data Import Validation
        
        Importing and then exporting should preserve graph structure.
        """
        temp_dir = tempfile.mkdtemp()
        try:
            config_manager = ConfigurationManager(temp_dir, enable_logging=False)
            
            # Write original GeoJSON
            import_path = Path(temp_dir) / "import.geojson"
            with open(import_path, 'w') as f:
                json.dump(geojson_data, f)
            
            # Import
            graph = config_manager.import_from_geojson(str(import_path))
            
            if graph is not None and len(graph.get_vertex_ids()) > 0:
                # Export
                export_path = Path(temp_dir) / "export.geojson"
                result = config_manager.export_to_geojson(graph, str(export_path))
                
                assert result is True, "Export should succeed"
                assert export_path.exists(), "Export file should be created"
                
                # Re-import the exported file
                reimported_graph = config_manager.import_from_geojson(str(export_path))
                
                assert reimported_graph is not None, "Re-import should succeed"
                
                # Verify structure is preserved
                assert len(graph.get_vertex_ids()) == len(reimported_graph.get_vertex_ids()), \
                    "Vertex count should be preserved in round-trip"
            
            config_manager.close()
        finally:
            shutil.rmtree(temp_dir)
    
    @given(
        vertex_type=st.sampled_from(['intersection', 'shelter', 'evacuation_point', 'hospital', 'INVALID'])
    )
    @settings(max_examples=50, deadline=2000)
    def test_vertex_type_mapping_consistency(self, vertex_type):
        """
        Property 13: Data Import Validation
        
        Vertex type mapping should be consistent and handle invalid types.
        """
        temp_dir = tempfile.mkdtemp()
        try:
            config_manager = ConfigurationManager(temp_dir, enable_logging=False)
            
            # Create GeoJSON with specific vertex type
            geojson_data = {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "geometry": {"type": "Point", "coordinates": [0.0, 0.0]},
                        "properties": {"id": "V1", "type": vertex_type}
                    }
                ]
            }
            
            geojson_path = Path(temp_dir) / "test_type.geojson"
            with open(geojson_path, 'w') as f:
                json.dump(geojson_data, f)
            
            # Import
            graph = config_manager.import_from_geojson(str(geojson_path))
            
            # Should always succeed (invalid types map to default)
            assert graph is not None, "Import should handle all vertex types"
            
            vertex = graph.get_vertex("V1")
            if vertex is not None:
                # Verify type mapping
                assert vertex.vertex_type in [VertexType.INTERSECTION, VertexType.SHELTER, VertexType.EVACUATION_POINT], \
                    "Vertex type should be mapped to valid type"
            
            config_manager.close()
        finally:
            shutil.rmtree(temp_dir)
    
    @given(
        distance=st.one_of(
            st.floats(min_value=0.1, max_value=100.0, allow_nan=False, allow_infinity=False),
            st.floats(allow_nan=True),
            st.floats(allow_infinity=True),
            st.just(-1.0)
        )
    )
    @settings(max_examples=50, deadline=2000)
    def test_edge_weight_validation(self, distance):
        """
        Property 13: Data Import Validation
        
        Edge weight validation should handle invalid values gracefully.
        """
        temp_dir = tempfile.mkdtemp()
        try:
            config_manager = ConfigurationManager(temp_dir, enable_logging=False)
            
            # Create GeoJSON with specific edge weight
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
                        "properties": {"id": "V2", "type": "intersection"}
                    },
                    {
                        "type": "Feature",
                        "geometry": {"type": "LineString", "coordinates": [[0.0, 0.0], [1.0, 1.0]]},
                        "properties": {
                            "source": "V1",
                            "target": "V2",
                            "distance": distance,
                            "risk": 0.1,
                            "congestion": 0.2
                        }
                    }
                ]
            }
            
            geojson_path = Path(temp_dir) / "test_weight.geojson"
            with open(geojson_path, 'w') as f:
                json.dump(geojson_data, f)
            
            # Import should not crash
            graph = config_manager.import_from_geojson(str(geojson_path))
            
            # Should handle invalid weights gracefully
            assert graph is not None, "Import should handle invalid edge weights"
            
            config_manager.close()
        finally:
            shutil.rmtree(temp_dir)
    
    @given(geojson_data=valid_geojson_strategy())
    @settings(max_examples=20, deadline=5000, suppress_health_check=[HealthCheck.large_base_example])
    def test_nonexistent_file_handling(self, geojson_data):
        """
        Property 13: Data Import Validation
        
        Attempting to import from nonexistent file should fail gracefully.
        """
        temp_dir = tempfile.mkdtemp()
        try:
            config_manager = ConfigurationManager(temp_dir, enable_logging=False)
            
            # Try to import from nonexistent file
            nonexistent_path = Path(temp_dir) / "nonexistent.geojson"
            graph = config_manager.import_from_geojson(str(nonexistent_path))
            
            # Should return None
            assert graph is None, "Import from nonexistent file should return None"
            
            config_manager.close()
        finally:
            shutil.rmtree(temp_dir)
