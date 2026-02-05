"""
Adaptive Disaster Evacuation Route Optimization System

A graph-based pathfinding application that demonstrates advanced algorithmic concepts
for Design and Analysis of Algorithms (DAA) course. The system models urban environments
as dynamic weighted graphs and computes optimal evacuation routes during disasters
using Dijkstra's algorithm with priority queue optimization.
"""

from . import models
from . import graph
from . import disaster
from . import pathfinding
from . import controller
from . import visualization
from . import analysis

__version__ = "1.0.0"
__author__ = "DAA Course Project"
__all__ = ['models', 'graph', 'disaster', 'pathfinding', 'controller', 'visualization', 'analysis']