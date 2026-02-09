"""
OpenStreetMap integration components for real-world road network data.

This module provides components for extracting, converting, and working with
real OpenStreetMap data while maintaining the core Dijkstra algorithm unchanged.
"""

from .osm_extractor import OSMExtractor
from .graph_converter import GraphConverter

__all__ = ['OSMExtractor', 'GraphConverter']
