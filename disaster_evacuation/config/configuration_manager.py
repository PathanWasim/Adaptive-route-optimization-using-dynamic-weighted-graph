"""
Configuration Manager for the disaster evacuation routing system.

This module implements the ConfigurationManager class that provides:
- Save/load functionality for graph configurations using JSON
- Validation for loaded graph data integrity and connectivity
- Import functionality for standard geographic data formats
- Configuration management for disaster parameters and algorithm settings
- Comprehensive error handling and logging
- Data recovery options for corrupted configurations
"""

import json
import os
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from pathlib import Path
from ..models import Vertex, Edge, DisasterEvent, DisasterType, VertexType
from ..graph import GraphManager


# Configure logging
logger = logging.getLogger(__name__)


class ConfigurationError(Exception):
    """Custom exception for configuration-related errors."""
    pass


class ConfigurationManager:
    """
    Manages configuration and data persistence for the evacuation routing system.
    
    The ConfigurationManager provides:
    - JSON-based graph configuration save/load
    - Data validation and integrity checking
    - Import from standard geographic formats
    - Configuration for disaster parameters and algorithm settings
    """
    
    def __init__(self, config_dir: Optional[str] = None, enable_logging: bool = True):
        """
        Initialize the configuration manager.
        
        Args:
            config_dir: Directory for storing configuration files.
                       Defaults to './configs' if not specified.
            enable_logging: Whether to enable logging for configuration operations.
        """
        self.config_dir = Path(config_dir) if config_dir else Path('./configs')
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize file handler reference
        self._file_handler = None
        
        # Configure logging
        self.enable_logging = enable_logging
        if enable_logging:
            self._setup_logging()
        
        # Default algorithm settings
        self.algorithm_settings = {
            "pathfinding_algorithm": "dijkstra",
            "max_iterations": 10000,
            "timeout_seconds": 30,
            "enable_caching": True,
            "use_bidirectional_search": False,
            "heuristic_weight": 1.0
        }
        
        # Default disaster parameters
        self.disaster_parameters = {
            "flood": {"default_severity": 0.7, "default_radius": 2.0, "blocking_threshold": 0.8},
            "fire": {"default_severity": 0.8, "default_radius": 1.5, "blocking_threshold": 0.7},
            "earthquake": {"default_severity": 0.75, "default_radius": 2.5, "blocking_threshold": 0.75}
        }
        
        # Backup directory for corrupted files
        self.backup_dir = self.config_dir / 'backups'
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"ConfigurationManager initialized with config_dir: {self.config_dir}")
    
    def _setup_logging(self) -> None:
        """Set up logging configuration."""
        log_file = self.config_dir / 'configuration.log'
        
        # Create file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        
        # Add handler to logger
        logger.addHandler(file_handler)
        logger.setLevel(logging.DEBUG)
        
        # Store handler reference for cleanup
        self._file_handler = file_handler
    
    def close(self) -> None:
        """Close logging handlers and cleanup resources."""
        if hasattr(self, '_file_handler') and self._file_handler:
            self._file_handler.close()
            logger.removeHandler(self._file_handler)
            self._file_handler = None
    
    def save_graph_configuration(self, graph: GraphManager, filename: str,
                                metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Save graph configuration to a JSON file with error handling.
        
        Args:
            graph: GraphManager instance to save
            filename: Name of the configuration file (without path)
            metadata: Optional metadata to include in the configuration
            
        Returns:
            True if save was successful, False otherwise
            
        Raises:
            ConfigurationError: If graph validation fails
        """
        try:
            logger.info(f"Attempting to save configuration: {filename}")
            
            # Validate graph before saving
            if not self._validate_graph_integrity(graph):
                error_msg = "Graph integrity validation failed before save"
                logger.error(error_msg)
                raise ConfigurationError(error_msg)
            
            config_path = self.config_dir / filename
            if not filename.endswith('.json'):
                config_path = self.config_dir / f"{filename}.json"
            
            # Create backup if file already exists
            if config_path.exists():
                self._create_backup(config_path)
                logger.info(f"Created backup of existing file: {config_path}")
            
            # Build configuration dictionary
            config = {
                "version": "1.0",
                "created_at": datetime.now().isoformat(),
                "metadata": metadata or {},
                "graph": self._serialize_graph(graph),
                "algorithm_settings": self.algorithm_settings.copy(),
                "disaster_parameters": self.disaster_parameters.copy()
            }
            
            # Write to temporary file first
            temp_path = config_path.with_suffix('.tmp')
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            
            # Verify the temporary file
            with open(temp_path, 'r', encoding='utf-8') as f:
                json.load(f)  # Verify it's valid JSON
            
            # Move temporary file to final location
            temp_path.replace(config_path)
            
            logger.info(f"Successfully saved configuration: {config_path}")
            return True
            
        except ConfigurationError:
            raise
        except Exception as e:
            error_msg = f"Error saving configuration: {e}"
            logger.error(error_msg)
            print(error_msg)
            return False
    
    def load_graph_configuration(self, filename: str, 
                                attempt_recovery: bool = True) -> Optional[Tuple[GraphManager, Dict[str, Any]]]:
        """
        Load graph configuration from a JSON file with error recovery.
        
        Args:
            filename: Name of the configuration file to load
            attempt_recovery: Whether to attempt recovery from backup if loading fails
            
        Returns:
            Tuple of (GraphManager, metadata) if successful, None otherwise
            
        Raises:
            ConfigurationError: If configuration is invalid and recovery fails
        """
        try:
            logger.info(f"Attempting to load configuration: {filename}")
            
            config_path = self.config_dir / filename
            if not filename.endswith('.json'):
                config_path = self.config_dir / f"{filename}.json"
            
            if not config_path.exists():
                error_msg = f"Configuration file not found: {config_path}"
                logger.warning(error_msg)
                print(error_msg)
                return None
            
            # Read configuration file
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
            except json.JSONDecodeError as e:
                error_msg = f"Invalid JSON in configuration file: {e}"
                logger.error(error_msg)
                
                if attempt_recovery:
                    logger.info("Attempting to recover from backup...")
                    return self._attempt_recovery(filename)
                else:
                    raise ConfigurationError(error_msg)
            
            # Validate configuration structure
            if not self._validate_configuration(config):
                error_msg = "Invalid configuration structure"
                logger.error(error_msg)
                
                if attempt_recovery:
                    logger.info("Attempting to recover from backup...")
                    return self._attempt_recovery(filename)
                else:
                    raise ConfigurationError(error_msg)
            
            # Deserialize graph
            try:
                graph = self._deserialize_graph(config['graph'])
            except Exception as e:
                error_msg = f"Error deserializing graph: {e}"
                logger.error(error_msg)
                
                if attempt_recovery:
                    logger.info("Attempting to recover from backup...")
                    return self._attempt_recovery(filename)
                else:
                    raise ConfigurationError(error_msg)
            
            # Validate graph integrity
            if not self._validate_graph_integrity(graph):
                error_msg = "Graph integrity validation failed"
                logger.error(error_msg)
                
                if attempt_recovery:
                    logger.info("Attempting to recover from backup...")
                    return self._attempt_recovery(filename)
                else:
                    raise ConfigurationError(error_msg)
            
            # Update settings from configuration
            if 'algorithm_settings' in config:
                self.algorithm_settings.update(config['algorithm_settings'])
                logger.debug(f"Updated algorithm settings: {self.algorithm_settings}")
            
            if 'disaster_parameters' in config:
                self.disaster_parameters.update(config['disaster_parameters'])
                logger.debug(f"Updated disaster parameters: {self.disaster_parameters}")
            
            metadata = config.get('metadata', {})
            metadata['loaded_at'] = datetime.now().isoformat()
            metadata['source_file'] = str(config_path)
            
            logger.info(f"Successfully loaded configuration: {config_path}")
            return graph, metadata
            
        except ConfigurationError:
            raise
        except Exception as e:
            error_msg = f"Error loading configuration: {e}"
            logger.error(error_msg)
            print(error_msg)
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
        """
        Set an algorithm setting value with validation.
        
        Args:
            key: Setting key
            value: Setting value
            
        Raises:
            ValueError: If value is invalid for the setting
        """
        # Validate specific settings
        if key == "max_iterations" and (not isinstance(value, int) or value <= 0):
            raise ValueError("max_iterations must be a positive integer")
        
        if key == "timeout_seconds" and (not isinstance(value, (int, float)) or value <= 0):
            raise ValueError("timeout_seconds must be a positive number")
        
        if key == "heuristic_weight" and (not isinstance(value, (int, float)) or value < 0):
            raise ValueError("heuristic_weight must be a non-negative number")
        
        self.algorithm_settings[key] = value
        logger.debug(f"Set algorithm setting: {key} = {value}")
    
    def get_disaster_parameter(self, disaster_type: str, parameter: str, default: Any = None) -> Any:
        """Get a disaster parameter value."""
        return self.disaster_parameters.get(disaster_type, {}).get(parameter, default)
    
    def set_disaster_parameter(self, disaster_type: str, parameter: str, value: Any) -> None:
        """
        Set a disaster parameter value with validation.
        
        Args:
            disaster_type: Type of disaster
            parameter: Parameter name
            value: Parameter value
            
        Raises:
            ValueError: If value is invalid for the parameter
        """
        # Validate specific parameters
        if parameter in ["default_severity", "blocking_threshold"]:
            if not isinstance(value, (int, float)) or not (0 <= value <= 1):
                raise ValueError(f"{parameter} must be a number between 0 and 1")
        
        if parameter == "default_radius":
            if not isinstance(value, (int, float)) or value <= 0:
                raise ValueError("default_radius must be a positive number")
        
        if disaster_type not in self.disaster_parameters:
            self.disaster_parameters[disaster_type] = {}
        
        self.disaster_parameters[disaster_type][parameter] = value
        logger.debug(f"Set disaster parameter: {disaster_type}.{parameter} = {value}")
    
    def reset_to_defaults(self) -> None:
        """Reset all settings to default values."""
        self.algorithm_settings = {
            "pathfinding_algorithm": "dijkstra",
            "max_iterations": 10000,
            "timeout_seconds": 30,
            "enable_caching": True,
            "use_bidirectional_search": False,
            "heuristic_weight": 1.0
        }
        
        self.disaster_parameters = {
            "flood": {"default_severity": 0.7, "default_radius": 2.0, "blocking_threshold": 0.8},
            "fire": {"default_severity": 0.8, "default_radius": 1.5, "blocking_threshold": 0.7},
            "earthquake": {"default_severity": 0.75, "default_radius": 2.5, "blocking_threshold": 0.75}
        }
        
        logger.info("Reset all settings to defaults")
    
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
                # Create backup before deleting
                self._create_backup(config_path)
                config_path.unlink()
                logger.info(f"Deleted configuration: {config_path}")
                return True
            
            logger.warning(f"Configuration file not found for deletion: {config_path}")
            return False
            
        except Exception as e:
            error_msg = f"Error deleting configuration: {e}"
            logger.error(error_msg)
            print(error_msg)
            return False
    
    def _create_backup(self, config_path: Path) -> None:
        """Create a backup of a configuration file."""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_name = f"{config_path.stem}_{timestamp}{config_path.suffix}"
            backup_path = self.backup_dir / backup_name
            
            import shutil
            shutil.copy2(config_path, backup_path)
            logger.debug(f"Created backup: {backup_path}")
            
            # Clean up old backups (keep only last 10)
            self._cleanup_old_backups(config_path.stem)
            
        except Exception as e:
            logger.warning(f"Failed to create backup: {e}")
    
    def _cleanup_old_backups(self, config_name: str, keep_count: int = 10) -> None:
        """Clean up old backup files, keeping only the most recent ones."""
        try:
            backups = sorted(
                self.backup_dir.glob(f"{config_name}_*.json"),
                key=lambda p: p.stat().st_mtime,
                reverse=True
            )
            
            # Delete old backups beyond keep_count
            for backup in backups[keep_count:]:
                backup.unlink()
                logger.debug(f"Deleted old backup: {backup}")
                
        except Exception as e:
            logger.warning(f"Failed to cleanup old backups: {e}")
    
    def _attempt_recovery(self, filename: str) -> Optional[Tuple[GraphManager, Dict[str, Any]]]:
        """Attempt to recover configuration from backup."""
        try:
            config_name = filename.replace('.json', '')
            backups = sorted(
                self.backup_dir.glob(f"{config_name}_*.json"),
                key=lambda p: p.stat().st_mtime,
                reverse=True
            )
            
            if not backups:
                logger.error("No backups available for recovery")
                print("No backups available for recovery")
                return None
            
            # Try each backup until one works
            for backup_path in backups:
                logger.info(f"Attempting recovery from backup: {backup_path}")
                print(f"Attempting recovery from backup: {backup_path.name}")
                
                try:
                    with open(backup_path, 'r', encoding='utf-8') as f:
                        config = json.load(f)
                    
                    if self._validate_configuration(config):
                        graph = self._deserialize_graph(config['graph'])
                        
                        if self._validate_graph_integrity(graph):
                            metadata = config.get('metadata', {})
                            metadata['recovered_from'] = str(backup_path)
                            metadata['recovery_time'] = datetime.now().isoformat()
                            
                            logger.info(f"Successfully recovered from backup: {backup_path}")
                            print(f"Successfully recovered from backup: {backup_path.name}")
                            return graph, metadata
                            
                except Exception as e:
                    logger.warning(f"Backup recovery failed for {backup_path}: {e}")
                    continue
            
            logger.error("All backup recovery attempts failed")
            print("All backup recovery attempts failed")
            return None
            
        except Exception as e:
            logger.error(f"Error during recovery attempt: {e}")
            return None
    
    def list_backups(self, config_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List available backup files.
        
        Args:
            config_name: Optional config name to filter backups. If None, lists all backups.
            
        Returns:
            List of backup information dictionaries
        """
        try:
            pattern = f"{config_name}_*.json" if config_name else "*.json"
            backups = sorted(
                self.backup_dir.glob(pattern),
                key=lambda p: p.stat().st_mtime,
                reverse=True
            )
            
            backup_info = []
            for backup in backups:
                stat = backup.stat()
                backup_info.append({
                    "filename": backup.name,
                    "path": str(backup),
                    "size_bytes": stat.st_size,
                    "modified_time": datetime.fromtimestamp(stat.st_mtime).isoformat()
                })
            
            return backup_info
            
        except Exception as e:
            logger.error(f"Error listing backups: {e}")
            return []
    
    def restore_from_backup(self, backup_filename: str, target_filename: Optional[str] = None) -> bool:
        """
        Restore a configuration from a backup file.
        
        Args:
            backup_filename: Name of the backup file to restore
            target_filename: Optional target filename. If None, uses original name.
            
        Returns:
            True if restore was successful, False otherwise
        """
        try:
            backup_path = self.backup_dir / backup_filename
            
            if not backup_path.exists():
                logger.error(f"Backup file not found: {backup_path}")
                print(f"Backup file not found: {backup_path}")
                return False
            
            # Determine target filename
            if target_filename is None:
                # Extract original name from backup filename (remove timestamp)
                parts = backup_filename.rsplit('_', 2)
                target_filename = f"{parts[0]}.json"
            
            target_path = self.config_dir / target_filename
            
            # Create backup of current file if it exists
            if target_path.exists():
                self._create_backup(target_path)
            
            # Copy backup to target location
            import shutil
            shutil.copy2(backup_path, target_path)
            
            logger.info(f"Restored backup {backup_filename} to {target_filename}")
            print(f"Successfully restored backup to {target_filename}")
            return True
            
        except Exception as e:
            error_msg = f"Error restoring from backup: {e}"
            logger.error(error_msg)
            print(error_msg)
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
    
    def __del__(self) -> None:
        """Destructor to ensure logging handlers are closed."""
        self.close()
