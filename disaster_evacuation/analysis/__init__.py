"""
Analysis module for the disaster evacuation routing system.

This module provides comparative analysis capabilities for evaluating
different routing strategies and their trade-offs.
"""

from .analysis_engine import AnalysisEngine
from .benchmarks import BenchmarkRunner

__all__ = ['AnalysisEngine', 'BenchmarkRunner']