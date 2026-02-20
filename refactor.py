import os
import re

directory = r"e:\Major Projects\DAA CP"

replacements = [
    # External absolute imports
    (r"disaster_evacuation\.pathfinding\.pathfinder_engine", r"disaster_evacuation.routing.dijkstra"),
    (r"disaster_evacuation\.pathfinding\.astar_engine", r"disaster_evacuation.routing.astar"),
    (r"disaster_evacuation\.pathfinding\.bellman_ford_engine", r"disaster_evacuation.routing.bellman_ford"),
    (r"disaster_evacuation\.pathfinding", r"disaster_evacuation.routing"),
    
    (r"disaster_evacuation\.graph\.graph_manager", r"disaster_evacuation.models.graph"),
    (r"disaster_evacuation\.graph\.weight_calculator", r"disaster_evacuation.models.weight_model"),
    (r"disaster_evacuation\.graph", r"disaster_evacuation.models"),
    
    (r"disaster_evacuation\.disaster\.disaster_modeler", r"disaster_evacuation.models.disaster_modeler"),
    (r"disaster_evacuation\.disaster\.disaster_model", r"disaster_evacuation.models.disaster_model"),
    (r"disaster_evacuation\.disaster", r"disaster_evacuation.models"),
    
    (r"disaster_evacuation\.benchmarks\.benchmark_runner", r"disaster_evacuation.analysis.benchmarks"),
    (r"disaster_evacuation\.benchmarks", r"disaster_evacuation.analysis"),

    # Relative internal imports
    (r"from \.\.pathfinding\.pathfinder_engine import", r"from ..routing.dijkstra import"),
    (r"from \.\.pathfinding\.astar_engine import", r"from ..routing.astar import"),
    (r"from \.\.pathfinding\.bellman_ford_engine import", r"from ..routing.bellman_ford import"),
    (r"from \.\.pathfinding import", r"from ..routing import"),
    (r"from \.pathfinding import", r"from .routing import"),
    
    (r"from \.\.graph\.graph_manager import", r"from ..models.graph import"),
    (r"from \.\.graph\.weight_calculator import", r"from ..models.weight_model import"),
    (r"from \.\.graph import", r"from ..models import"),
    (r"from \.graph import", r"from .models import"),
    
    (r"from \.\.disaster\.disaster_modeler import", r"from ..models.disaster_modeler import"),
    (r"from \.\.disaster\.disaster_model import", r"from ..models.disaster_model import"),
    (r"from \.\.disaster import", r"from ..models import"),
    (r"from \.disaster import", r"from .models import"),
    
    (r"from \.\.benchmarks\.benchmark_runner import", r"from ..analysis.benchmarks import"),
    (r"from \.\.benchmarks import", r"from ..analysis import"),
    (r"from \.benchmarks import", r"from .analysis import")
]

for root, _, files in os.walk(directory):
    if ".git" in root or "__pycache__" in root or ".kiro" in root:
        continue
    for file in files:
        if file.endswith(".py"):
            filepath = os.path.join(root, file)
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
            
            new_content = content
            for old, new in replacements:
                new_content = re.sub(old, new, new_content)
                
            if new_content != content:
                print(f"Updated {filepath}")
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(new_content)

print("Done Refactoring!")
