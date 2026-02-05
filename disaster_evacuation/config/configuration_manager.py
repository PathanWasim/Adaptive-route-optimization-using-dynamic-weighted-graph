"""
Configuration Manager for the disaster evacuation routing system.

This module implements the ConfigurationManager class that provides:
- Save/load functionality for graph configurations using JSON
- Validation for loaded graph data integrity and connectivity
- Import functionality for standard geographic data formats
- Configuration management for disaster parameters and algorithm settings
"""

import json
import os
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from pathlib import Path
from ..models import Vertex, Edge, DisasterEvent, DisasterType, VertexType
from ..graph import GraphManager


class ConfigurationManager:
    """
    Manages configuration and data persistence for the evacuation routing system.
    
    The ConfigurationManager provides:
    - JSON-based graph configuration save/load
    - Data validation and integrity checking
    - Import from standard geographic formats
    - Configuration for disaster parameters and algorithm settings
    """
    
    def __init__(self, config_dir: Optional[str] = None):
        """
        Initialize the configuration manager.
        
        Args:
            config_dir: Directory for storing configuration files.
                       Defaults to './configs' if not specified.
        """
        self.config_dir = Path(config_dir) if config_dir else Path('./configs')
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        # Default algorithm settings
        self.algorithm_settings = {
            "pathfinding_algorithm": "dijkstra",
            "max_iterations": 10000,
            "timeout_seconds": 30,
            "enable_caching": True
        }
        
        # Default disaster parameters
        self.disaster_parameters = {
            "flood": {"default_severity": 0.7, "default_radius": 2.0, "blocking_threshold": 0.8},
            "fire": {"default_severity": 0.8, "default_radius": 1.5, "blocking_threshold": 0.7},
            "earthquake": {"default_severity": 0.75, "default_radius": 2.5, "blocking_threshold": 0.75}
        }
    
    def save_graph_configuration(self, graph: GraphManager, filename: str,
                                metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Save graph configuration to a JSON file.
        
        Args:
            graph: GraphManager instance to save
            filename: Name of the configuration file (without path)
            metadata: Optional metadata to include in the configuration
            
        Returns:
            True if save was successful, False otherwise
        """
        try:
            config_path = self.config_dir / filename
            if not filename.endswith('.json'):
                config_path = self.config_dir / f"{filename}.json"
            
            # Build configuration dictionary
            config = {
                "version": "1.0",
                "created_at": datetime.now().isoformat(),
                "metadata": metadata or {},
                "graph": self._serialize_graph(graph),
                "algorithm_settings": self.algorithm_settings.copy(),
                "disaster_parameters": self.disaster_parameters.copy()
            }
            
            # Write to file with pretty formatting
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            
            return True
            
        except Exception as e:
            print(f"Error saving configuration: {e}")
            return False
    
    def load_graph_configuration(self, filename: str) -> Optional[Tuple[GraphManager, Dict[str, Any]]]:
        """
        Load graph configuration from a JSON file.
        
        Args:
            filename: Name of the configuration file to load
            
        Returns:
            Tuple of (GraphManager, metadata) if successful, None otherwise
        """
        try:
            config_path = self.config_dir / filename
            if not filename.endswith('.json'):
                config_path = self.config_dir / f"{filename}.json"
            
            if not config_path.exists():
                print(f"Configuration file not found: {config_path}")
                return None
            
            # Read configuration file
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            # Validate configuration structure
            if not self._validate_configuration(config):
                print("Invalid configuration structure")
                return None
            
            # Deserialize graph
            graph = self._deserialize_graph(config['graph'])
            
            # Validate graph integrity
            if not self._validate_graph_integrity(graph):
                print("Graph integrity validation failed")
                return None
            
            # Update settings from configuration
            if 'algorithm_settings' in config:
                self.algorithm_settings.update(config['algorithm_settings'])
            if 'disaster_parameters' in config:
                self.disaster_parameters.update(config['disaster_parameters'])
            
            metadata = config.get('metadata', {})
            metadata['loaded_at'] = datetime.now().isoformat()
            metadata['source_file'] = str(config_path)
            
            return graph, metadata
            
        except Exception as e:
            print(f"Error loading configuration: {e}")
            return None
    
    def import_from_geojson(self, filepath: str) -> Optional[GraphManager]:
        """
        Import graph from GeoJSON format.
        
        Args:
            filepath: Path to GeoJSON file
            
        Returns:
            GraphManager instance if successful, None otherwise
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                geojson_data = json.load(f)
            
            if geojson_data.get('type') != 'FeatureCollection':
                print("Invalid GeoJSON format: expected FeatureCollection")
                return None
            
            graph = GraphManager()
            features = geojson_data.get('features', [])
            
            # First pass: create vertices
            for feature in features:
                if feature.get('geometry', {}).get('type') == 'Point':
                    self._import_vertex_from_feature(graph, feature)
            
            # Second pass: create edges
            for feature in features:
                if feature.get('geometry', {}).get('type') == 'LineString':
                    self._import_edge_from_feature(graph, feature)
            
            return graph
            
        except Exception as e:
            print(f"Error importing from GeoJSON: {e}")
            return None
    
    def export_to_geojson(self, graph: GraphManager, filepath: str) -> bool:
        """
        Export graph to GeoJSON format.
        
        Args:
            graph: GraphManager instance to export
            filepath: Path for output GeoJSON file
            
        Returns:
            True if export was successful, False otherwise
        """
        try:
            features = []
            
            # Export vertices as Point features
            for vertex_id in graph.get_vertex_ids():
                vertex = graph.get_vertex(vertex_id)
                if vertex:
                    features.append({
                        "type": "Feature",
                        "geometry": {
                            "type": "Point",
                            "coordinates": [vertex.coordinates[0], vertex.coordinates[1]]
                        },
                        "properties": {
                            "id": vertex.id,
                            "type": vertex.vertex_type.value,
                            "capacity": vertex.capacity
                        }
                    })
            
            # Export edges as LineString features
            for edge in graph.get_all_edges():
                source_vertex = graph.get_vertex(edge.source)
                target_vertex = graph.get_vertex(edge.target)
                
                if source_vertex and target_vertex:
                    features.append({
                        "type": "Feature",
                        "geometry": {
                            "type": "LineString",
                            "coordinates": [
                                [source_vertex.coordinates[0], source_vertex.coordinates[1]],
                                [target_vertex.coordinates[0], target_vertex.coordinates[1]]
                            ]
                        },
                        "properties": {
                            "source": edge.source,
                            "target": edge.target,
                            "distance": edge.base_distance,
                            "risk": edge.base_risk,
                            "congestion": edge.base_congestion
                        }
                    })
            
            geojson = {
                "type": "FeatureCollection",
                "features": features
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(geojson, f, indent=2)
            
            return True
            
        except Exception as e:
            print(f"Error exporting to GeoJSON: {e}")
            return False
    
    def get_algorithm_setting(self, key: str, default: Any = None) -> Any:
        """Get an algorithm setting value."""
        return self.algorithm_settings.get(key, default)
    
    def set_algorithm_setting(self, key: str, value: Any) -> None:
        """Set an algorithm setting value."""
        self.algorithm_settings[key] = value
    
    def get_disaster_parameter(self, disaster_type: str, parameter: str, default: Any = None) -> Any:
        """Get a disaster parameter value."""
        return self.disaster_parameters.get(disaster_type, {}).get(parameter, default)
    
    def set_disaster_parameter(self, disaster_type: str, parameter: str, value: Any) -> None:
        """Set a disaster parameter value."""
        if disaster_type not in self.disaster_parameters:
            self.disaster_parameters[disaster_type] = {}
        self.disaster_parameters[disaster_type][parameter] = value
    
    def list_configurations(self) -> List[str]:
        """List all available configuration files."""
        try:
            return [f.name for f in self.config_dir.glob('*.json')]
        except Exception:
            return []
    
    def delete_configuration(self, filename: str) -> bool:
        """Delete a configuration file."""
        try:
            config_path = self.config_dir / filename
            if not filename.endswith('.json'):
                config_path = self.config_dir / f"{filename}.json"
            
            if config_path.exists():
                config_path.unlink()
                return True
            return False
            
        except Exception as e:
            print(f"Error deleting configuration: {e}")
            return False
    
    def _serialize_graph(self, graph: GraphManager) -> Dict[str, Any]:
        """Serialize a graph to a dictionary."""
        vertices = []
        for vertex_id in graph.get_vertex_ids():
            vertex = graph.get_vertex(vertex_id)
            if vertex:
                vertices.append({
                    "id": vertex.id,
                    "type": vertex.vertex_type.value,
                    "x": vertex.coordinates[0],
                    "y": vertex.coordinates[1],
                    "capacity": vertex.capacity
                })
        
        edges = []
        for edge in graph.get_all_edges():
            edges.append({
                "source": edge.source,
                "target": edge.target,
                "distance": edge.base_distance,
                "risk": edge.base_risk,
                "congestion": edge.base_congestion
            })
        
        return {
            "vertices": vertices,
            "edges": edges,
            "vertex_count": len(vertices),
            "edge_count": len(edges)
        }
    
    def _deserialize_graph(self, graph_data: Dict[str, Any]) -> GraphManager:
        """Deserialize a graph from a dictionary."""
        graph = GraphManager()
        
        # Add vertices
        for vertex_data in graph_data.get('vertices', []):
            vertex_type = VertexType(vertex_data['type'])
            graph.add_vertex(
                vertex_data['id'],
                vertex_type,
                (vertex_data['x'], vertex_data['y']),
                vertex_data.get('capacity')
            )
        
        # Add edges
        for edge_data in graph_data.get('edges', []):
            graph.add_edge(
                edge_data['source'],
                edge_data['target'],
                edge_data['distance'],
                edge_data['risk'],
                edge_data['congestion']
            )
        
        return graph
    
    def _validate_configuration(self, config: Dict[str, Any]) -> bool:
        """Validate configuration structure."""
        required_keys = ['version', 'graph']
        if not all(key in config for key in required_keys):
            return False
        
        graph_data = config['graph']
        if 'vertices' not in graph_data or 'edges' not in graph_data:
            return False
        
        return True
    
    def _validate_graph_integrity(self, graph: GraphManager) -> bool:
        """
        Validate graph integrity and connectivity.
        
        Checks:
        - All edges reference existing vertices
        - No self-loops
        - Valid edge weights
        """
        try:
            vertex_ids = set(graph.get_vertex_ids())
            
            for edge in graph.get_all_edges():
                # Check vertices exist
                if edge.source not in vertex_ids or edge.target not in vertex_ids:
                    return False
                
                # Check no self-loops
                if edge.source == edge.target:
                    return False
                
                # Check valid weights
                if edge.base_distance <= 0:
                    return False
                if not (0 <= edge.base_risk <= 1):
                    return False
                if not (0 <= edge.base_congestion <= 1):
                    return False
            
            return True
            
        except Exception:
            return False
    
    def _import_vertex_from_feature(self, graph: GraphManager, feature: Dict[str, Any]) -> None:
        """Import a vertex from a GeoJSON feature."""
        properties = feature.get('properties', {})
        coordinates = feature.get('geometry', {}).get('coordinates', [0, 0])
        
        vertex_id = properties.get('id', f"V{len(graph.get_vertex_ids())}")
        vertex_type_str = properties.get('type', 'intersection')
        
        # Map string to VertexType
        type_mapping = {
            'intersection': VertexType.INTERSECTION,
            'shelter': VertexType.SHELTER,
            'evacuation_point': VertexType.EVACUATION_POINT,
            'hospital': VertexType.SHELTER  # Map hospital to shelter
        }
        vertex_type = type_mapping.get(vertex_type_str.lower(), VertexType.INTERSECTION)
        
        capacity = properties.get('capacity')
        
        graph.add_vertex(vertex_id, vertex_type, tuple(coordinates), capacity)
    
    def _import_edge_from_feature(self, graph: GraphManager, feature: Dict[str, Any]) -> None:
        """Import an edge from a GeoJSON feature."""
        properties = feature.get('properties', {})
        
        source = properties.get('source')
        target = properties.get('target')
        distance = properties.get('distance', 1.0)
        risk = properties.get('risk', 0.1)
        congestion = properties.get('congestion', 0.1)
        
        if source and target:
            try:
                graph.add_edge(source, target, distance, risk, congestion)
            except Exception:
                pass  # Skip invalid edges
    
    def __str__(self) -> str:
        """String representation of the configuration manager."""
        return f"ConfigurationManager(config_dir={self.config_dir})"
    
    def __repr__(self) -> str:
        return self.__str__()
