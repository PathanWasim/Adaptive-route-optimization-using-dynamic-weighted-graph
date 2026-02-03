"""
Pytest configuration and shared fixtures for testing.
"""

import pytest
from hypothesis import settings, Verbosity
from disaster_evacuation.models import Vertex, Edge, DisasterEvent, VertexType, DisasterType
from datetime import datetime


# Configure Hypothesis for property-based testing
settings.register_profile("default", max_examples=100, verbosity=Verbosity.normal)
settings.load_profile("default")


@pytest.fixture
def sample_vertices():
    """Create sample vertices for testing."""
    return [
        Vertex("A", VertexType.INTERSECTION, (0.0, 0.0)),
        Vertex("B", VertexType.INTERSECTION, (1.0, 0.0)),
        Vertex("C", VertexType.SHELTER, (2.0, 0.0), capacity=100),
        Vertex("D", VertexType.EVACUATION_POINT, (1.0, 1.0), capacity=500),
    ]


@pytest.fixture
def sample_edges():
    """Create sample edges for testing."""
    return [
        Edge("A", "B", 1.0, 0.1, 0.2),
        Edge("B", "C", 1.0, 0.2, 0.1),
        Edge("A", "D", 1.4, 0.3, 0.3),
        Edge("C", "D", 1.4, 0.1, 0.4),
    ]


@pytest.fixture
def sample_disaster():
    """Create a sample disaster event for testing."""
    return DisasterEvent(
        disaster_type=DisasterType.FLOOD,
        epicenter=(0.5, 0.5),
        severity=0.7,
        max_effect_radius=2.0,
        start_time=datetime.now()
    )