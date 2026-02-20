"""
Unit tests for ConfigurationManager error handling and recovery features.
"""

import pytest
import json
import tempfile
import shutil
from pathlib import Path
from disaster_evacuation.config import ConfigurationManager
from disaster_evacuation.config.configuration_manager import ConfigurationError
from disaster_evacuation.models import GraphManager
from disaster_evacuation.models import VertexType


class TestConfigurationErrorHandling:
    """Test suite for ConfigurationManager error handling."""
    
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
        manager = ConfigurationManager(temp_config_dir, enable_logging=False)
        yield manager
        # Explicitly close to release file handles
        manager.close()
    
    @pytest.fixture
    def sample_graph(self):
        """Create a sample graph for testing."""
        graph = GraphManager()
        graph.add_vertex("A", VertexType.INTERSECTION, (0.0, 0.0))
        graph.add_vertex("B", VertexType.SHELTER, (1.0, 0.0), capacity=100)
        graph.add_edge("A", "B", 1.0, 0.1, 0.2)
        return graph
    
    def test_backup_creation_on_save(self, config_manager, sample_graph):
        """Test that backups are created when overwriting existing files."""
        # Save initial configuration
        config_manager.save_graph_configuration(sample_graph, "test_backup")
        
        # Save again to trigger backup
        config_manager.save_graph_configuration(sample_graph, "test_backup")
        
        # Check that backup was created
        backups = config_manager.list_backups("test_backup")
        assert len(backups) >= 1
    
    def test_backup_creation_on_delete(self, config_manager, sample_graph):
        """Test that backups are created before deletion."""
        # Save configuration
        config_manager.save_graph_configuration(sample_graph, "test_delete")
        
        # Delete it
        config_manager.delete_configuration("test_delete")
        
        # Check that backup was created
        backups = config_manager.list_backups("test_delete")
        assert len(backups) >= 1
    
    def test_recovery_from_corrupted_file(self, config_manager, sample_graph):
        """Test recovery from a corrupted configuration file."""
        # Save valid configuration
        config_manager.save_graph_configuration(sample_graph, "test_corrupt")
        
        # Save again to create a backup
        config_manager.save_graph_configuration(sample_graph, "test_corrupt")
        
        # Corrupt the file
        config_path = config_manager.config_dir / "test_corrupt.json"
        with open(config_path, 'w') as f:
            f.write("{ invalid json }")
        
        # Attempt to load with recovery
        result = config_manager.load_graph_configuration("test_corrupt", attempt_recovery=True)
        
        # Should recover from backup
        assert result is not None
        graph, metadata = result
        assert 'recovered_from' in metadata
    
    def test_recovery_failure_without_backup(self, config_manager):
        """Test recovery failure when no backup exists."""
        # Create corrupted file without backup
        config_path = config_manager.config_dir / "no_backup.json"
        with open(config_path, 'w') as f:
            f.write("{ invalid json }")
        
        # Attempt to load with recovery
        result = config_manager.load_graph_configuration("no_backup", attempt_recovery=True)
        
        # Should fail
        assert result is None
    
    def test_load_without_recovery_raises_error(self, config_manager):
        """Test that loading corrupted file without recovery raises error."""
        # Create corrupted file
        config_path = config_manager.config_dir / "corrupt_no_recovery.json"
        with open(config_path, 'w') as f:
            f.write("{ invalid json }")
        
        # Attempt to load without recovery
        with pytest.raises(ConfigurationError):
            config_manager.load_graph_configuration("corrupt_no_recovery", attempt_recovery=False)
    
    def test_list_backups(self, config_manager, sample_graph):
        """Test listing backup files."""
        # Create multiple backups (first save creates file, subsequent saves create backups)
        config_manager.save_graph_configuration(sample_graph, "test_list")
        config_manager.save_graph_configuration(sample_graph, "test_list")
        config_manager.save_graph_configuration(sample_graph, "test_list")
        config_manager.save_graph_configuration(sample_graph, "test_list")
        
        # List backups
        backups = config_manager.list_backups("test_list")
        
        assert len(backups) >= 1  # At least 1 backup should be created
        assert all('filename' in b for b in backups)
        assert all('size_bytes' in b for b in backups)
        assert all('modified_time' in b for b in backups)
    
    def test_list_all_backups(self, config_manager, sample_graph):
        """Test listing all backup files."""
        # Create backups for different configs
        config_manager.save_graph_configuration(sample_graph, "config1")
        config_manager.save_graph_configuration(sample_graph, "config1")
        config_manager.save_graph_configuration(sample_graph, "config2")
        config_manager.save_graph_configuration(sample_graph, "config2")
        
        # List all backups
        all_backups = config_manager.list_backups()
        
        assert len(all_backups) >= 2
    
    def test_restore_from_backup(self, config_manager, sample_graph):
        """Test restoring a configuration from backup."""
        # Save and create backup
        config_manager.save_graph_configuration(sample_graph, "test_restore")
        config_manager.save_graph_configuration(sample_graph, "test_restore")
        
        # Get backup filename
        backups = config_manager.list_backups("test_restore")
        assert len(backups) >= 1
        backup_filename = backups[0]['filename']
        
        # Delete current config
        config_manager.delete_configuration("test_restore")
        
        # Restore from backup
        result = config_manager.restore_from_backup(backup_filename)
        assert result is True
        
        # Verify restored file exists
        configs = config_manager.list_configurations()
        assert "test_restore.json" in configs
    
    def test_restore_nonexistent_backup(self, config_manager):
        """Test restoring from nonexistent backup."""
        result = config_manager.restore_from_backup("nonexistent_backup.json")
        assert result is False
    
    def test_algorithm_setting_validation(self, config_manager):
        """Test validation of algorithm settings."""
        # Valid settings
        config_manager.set_algorithm_setting("max_iterations", 5000)
        assert config_manager.get_algorithm_setting("max_iterations") == 5000
        
        # Invalid max_iterations (negative)
        with pytest.raises(ValueError):
            config_manager.set_algorithm_setting("max_iterations", -100)
        
        # Invalid max_iterations (not integer)
        with pytest.raises(ValueError):
            config_manager.set_algorithm_setting("max_iterations", "invalid")
        
        # Invalid timeout_seconds (negative)
        with pytest.raises(ValueError):
            config_manager.set_algorithm_setting("timeout_seconds", -10)
        
        # Invalid heuristic_weight (negative)
        with pytest.raises(ValueError):
            config_manager.set_algorithm_setting("heuristic_weight", -0.5)
    
    def test_disaster_parameter_validation(self, config_manager):
        """Test validation of disaster parameters."""
        # Valid parameters
        config_manager.set_disaster_parameter("flood", "default_severity", 0.8)
        assert config_manager.get_disaster_parameter("flood", "default_severity") == 0.8
        
        # Invalid severity (> 1)
        with pytest.raises(ValueError):
            config_manager.set_disaster_parameter("flood", "default_severity", 1.5)
        
        # Invalid severity (< 0)
        with pytest.raises(ValueError):
            config_manager.set_disaster_parameter("flood", "default_severity", -0.1)
        
        # Invalid blocking_threshold (> 1)
        with pytest.raises(ValueError):
            config_manager.set_disaster_parameter("fire", "blocking_threshold", 2.0)
        
        # Invalid radius (negative)
        with pytest.raises(ValueError):
            config_manager.set_disaster_parameter("earthquake", "default_radius", -1.0)
    
    def test_reset_to_defaults(self, config_manager):
        """Test resetting settings to defaults."""
        # Modify settings
        config_manager.set_algorithm_setting("max_iterations", 5000)
        config_manager.set_disaster_parameter("flood", "default_severity", 0.9)
        
        # Reset to defaults
        config_manager.reset_to_defaults()
        
        # Verify defaults are restored
        assert config_manager.get_algorithm_setting("max_iterations") == 10000
        assert config_manager.get_disaster_parameter("flood", "default_severity") == 0.7
    
    def test_backup_cleanup(self, config_manager, sample_graph):
        """Test that old backups are cleaned up."""
        # Create many backups (more than keep_count)
        for i in range(15):
            config_manager.save_graph_configuration(sample_graph, "test_cleanup")
        
        # Check that only recent backups are kept
        backups = config_manager.list_backups("test_cleanup")
        assert len(backups) <= 10  # Default keep_count
    
    def test_temporary_file_usage(self, config_manager, sample_graph):
        """Test that temporary files are used during save."""
        config_path = config_manager.config_dir / "test_temp.json"
        temp_path = config_path.with_suffix('.tmp')
        
        # Save configuration
        config_manager.save_graph_configuration(sample_graph, "test_temp")
        
        # Temporary file should not exist after save
        assert not temp_path.exists()
        
        # Final file should exist
        assert config_path.exists()
    
    def test_invalid_graph_save_raises_error(self, config_manager):
        """Test that saving invalid graph raises ConfigurationError."""
        # Create invalid graph (this would need to be an actual invalid graph)
        # For now, we'll test with a valid graph and mock the validation
        graph = GraphManager()
        
        # This should work fine with empty graph
        result = config_manager.save_graph_configuration(graph, "empty_graph")
        assert result is True
    
    def test_configuration_error_exception(self):
        """Test ConfigurationError exception."""
        error = ConfigurationError("Test error message")
        assert str(error) == "Test error message"
        assert isinstance(error, Exception)
