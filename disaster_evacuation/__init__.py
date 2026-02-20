"""
Adaptive Disaster Evacuation Route Optimization System

A graph-based pathfinding application that demonstrates advanced algorithmic concepts
for Design and Analysis of Algorithms (DAA) course. The system models urban environments
as dynamic weighted graphs and computes optimal evacuation routes during disasters
using Dijkstra's algorithm with priority queue optimization.
"""

from . import models
from . import routing
from . import controller
from . import visualization
from . import analysis
from . import config
from . import osm

__version__ = "1.0.0"
__author__ = "DAA Course Project"
__all__ = ['models', 'routing', 'controller', 'visualization', 'analysis', 'config', 'osm']