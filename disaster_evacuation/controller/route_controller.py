"""
Route Controller for the disaster evacuation routing system.

This module implements the RouteController class that orchestrates user interactions,
validates inputs, and coordinates between graph management, disaster modeling, and pathfinding.
"""

from typing import Dict, List, Optional, Tuple
from ..models import PathResult, DisasterEvent, DisasterType, VertexType
from ..graph import GraphManager
from ..disaster import DisasterModel
from ..pathfinding import PathfinderEngine


class RouteController:
    """
    Orchestrates user interactions and coordinates system components.
    
    The RouteController provides a high-level interface for:
    - Route computation with input validation
    - Disaster scenario management
    - Comparative analysis of routing strategies
    - User-friendly error handling and reporting
    """
    
    def __init__(self, graph: GraphManager):
        """
        Initialize the route controller.
        
        Args:
            graph: GraphManager instance containing the city network
        """
        self._graph = graph
        self._disaster_model = DisasterModel()
        self._pathfinder = PathfinderEngine()
        self._active_disasters: List[DisasterEvent] = []
    
    def compute_route(self, source: str, destination: str, 
                     disaster: Optional[DisasterEvent] = None) -> Dict[str, any]:
        """
        Compute an evacuation route with comprehensive validation and reporting.
        
        Args:
            source: Source vertex ID (starting location)
            destination: Destination vertex ID (evacuation point)
            disaster: Optional disaster event to consider
            
        Returns:
            Dictionary containing route information and metadata
        """
        # Validate inputs
        validation_result = self._validate_route_request(source, destination)
        if not validation_result["valid"]:
            return {
                "success": False,
                "error": validation_result["error"],
                "error_type": validation_result["error_type"],
                "source": source,
                "destination": destination
            }
        
        # Apply disaster effects if provided
        if disaster is not None:
            try:
                self._disaster_model.apply_disaster_effects(self._graph, disaster)
                self._active_disasters.append(disaster)
            except Exception as e:
                return {
                    "success": False,
                    "error": f"Failed to apply disaster effects: {str(e)}",
                    "error_type": "disaster_error",
                    "source": source,
                    "destination": destination
                }
        
        # Compute path
        try:
            result = self._pathfinder.find_shortest_path(self._graph, source, destination)
        except Exception as e:
            return {
                "success": False,
                "error": f"Pathfinding error: {str(e)}",
                "error_type": "pathfinding_error",
                "source": source,
                "destination": destination
            }
        
        # Format response
        if result.found:
            return self._format_successful_route(result, source, destination, disaster)
        else:
            return self._format_failed_route(result, source, destination, disaster)
    
    def _validate_route_request(self, source: str, destination: str) -> Dict[str, any]:
        """
        Validate route computation request.
        
        Args:
            source: Source vertex ID
            destination: Destination vertex ID
            
        Returns:
            Dictionary with validation result
        """
        # Check for empty inputs
        if not source or not isinstance(source, str):
            return {
                "valid": False,
                "error": "Source location cannot be empty",
                "error_type": "invalid_source"
            }
        
        if not destination or not isinstance(destination, str):
            return {
                "valid": False,
                "error": "Destination location cannot be empty",
                "error_type": "invalid_destination"
            }
        
        # Check if vertices exist
        if not self._graph.has_vertex(source):
            return {
                "valid": False,
                "error": f"Source location '{source}' does not exist in the network",
                "error_type": "source_not_found"
            }
        
        if not self._graph.has_vertex(destination):
            return {
                "valid": False,
                "error": f"Destination location '{destination}' does not exist in the network",
                "error_type": "destination_not_found"
            }
        
        # Check if source and destination are the same
        if source == destination:
            return {
                "valid": False,
                "error": "Source and destination cannot be the same location",
                "error_type": "same_location"
            }
        
        return {"valid": True}
    
    def _format_successful_route(self, result: PathResult, source: str, 
                                destination: str, disaster: Optional[DisasterEvent]) -> Dict[str, any]:
        """
        Format successful route computation result.
        
        Args:
            result: PathResult from pathfinding
            source: Source vertex ID
            destination: Destination vertex ID
            disaster: Disaster event if applicable
            
        Returns:
            Formatted route information
        """
        # Get vertex information
        source_vertex = self._graph.get_vertex(source)
        destination_vertex = self._graph.get_vertex(destination)
        
        # Identify avoided dangerous roads
        avoided_roads = self._identify_avoided_dangerous_roads(result, disaster)
        
        # Calculate route statistics
        stats = {
            "total_distance": sum(edge.base_distance for edge in result.edges_traversed),
            "total_risk": sum(edge.base_risk for edge in result.edges_traversed),
            "total_congestion": sum(edge.base_congestion for edge in result.edges_traversed),
            "edge_count": len(result.edges_traversed),
            "computation_time": result.computation_time,
            "nodes_visited": result.nodes_visited
        }
        
        return {
            "success": True,
            "path": result.path,
            "total_cost": result.total_cost,
            "source": {
                "id": source,
                "type": source_vertex.vertex_type.value,
                "coordinates": source_vertex.coordinates
            },
            "destination": {
                "id": destination,
                "type": destination_vertex.vertex_type.value,
                "coordinates": destination_vertex.coordinates,
                "capacity": destination_vertex.capacity
            },
            "edges": [
                {
                    "from": edge.source,
                    "to": edge.target,
                    "distance": edge.base_distance,
                    "current_weight": edge.current_weight,
                    "is_blocked": edge.is_blocked
                }
                for edge in result.edges_traversed
            ],
            "avoided_dangerous_roads": avoided_roads,
            "statistics": stats,
            "disaster_applied": disaster is not None,
            "disaster_info": self._format_disaster_info(disaster) if disaster else None
        }
    
    def _format_failed_route(self, result: PathResult, source: str, 
                           destination: str, disaster: Optional[DisasterEvent]) -> Dict[str, any]:
        """
        Format failed route computation result.
        
        Args:
            result: PathResult from pathfinding
            source: Source vertex ID
            destination: Destination vertex ID
            disaster: Disaster event if applicable
            
        Returns:
            Formatted error information
        """
        return {
            "success": False,
            "error": result.error_message,
            "error_type": "no_path_found",
            "source": source,
            "destination": destination,
            "computation_time": result.computation_time,
            "nodes_visited": result.nodes_visited,
            "disaster_applied": disaster is not None,
            "disaster_info": self._format_disaster_info(disaster) if disaster else None,
            "suggestion": self._generate_route_suggestion(source, destination, disaster)
        }
    
    def _identify_avoided_dangerous_roads(self, result: PathResult, 
                                        disaster: Optional[DisasterEvent]) -> List[Dict[str, any]]:
        """
        Identify dangerous roads that were avoided in the route.
        
        Args:
            result: PathResult from pathfinding
            disaster: Disaster event if applicable
            
        Returns:
            List of avoided dangerous roads
        """
        if disaster is None:
            return []
        
        avoided_roads = []
        
        # Get all edges affected by disaster
        affected_edges = self._disaster_model.get_affected_edges(
            self._graph, disaster.epicenter, disaster.max_effect_radius
        )
        
        # Find edges that are blocked or high-risk but not used in path
        used_edges = set((edge.source, edge.target) for edge in result.edges_traversed)
        
        for edge in affected_edges:
            edge_key = (edge.source, edge.target)
            if edge_key not in used_edges:
                if edge.is_blocked or edge.current_weight > edge.base_distance * 2:
                    avoided_roads.append({
                        "from": edge.source,
                        "to": edge.target,
                        "reason": "blocked" if edge.is_blocked else "high_risk",
                        "original_weight": edge.base_distance,
                        "current_weight": edge.current_weight
                    })
        
        return avoided_roads
    
    def _format_disaster_info(self, disaster: DisasterEvent) -> Dict[str, any]:
        """
        Format disaster information for output.
        
        Args:
            disaster: Disaster event
            
        Returns:
            Formatted disaster information
        """
        return {
            "type": disaster.disaster_type.value,
            "epicenter": disaster.epicenter,
            "severity": disaster.severity,
            "effect_radius": disaster.max_effect_radius,
            "start_time": disaster.start_time.isoformat() if disaster.start_time else None
        }
    
    def _generate_route_suggestion(self, source: str, destination: str, 
                                  disaster: Optional[DisasterEvent]) -> str:
        """
        Generate helpful suggestion when no route is found.
        
        Args:
            source: Source vertex ID
            destination: Destination vertex ID
            disaster: Disaster event if applicable
            
        Returns:
            Suggestion message
        """
        if disaster is not None:
            return (f"No safe route found from {source} to {destination} under current disaster conditions. "
                   f"Consider alternative destinations or wait for conditions to improve.")
        else:
            return (f"No route exists from {source} to {destination}. "
                   f"These locations may be in disconnected parts of the network.")
    
    def compare_routes(self, source: str, destination: str, 
                      disaster: Optional[DisasterEvent] = None) -> Dict[str, any]:
        """
        Compare normal shortest-distance route with disaster-aware route.
        
        Args:
            source: Source vertex ID
            destination: Destination vertex ID
            disaster: Disaster event to consider
            
        Returns:
            Comparison of routing strategies
        """
        # Validate inputs
        validation_result = self._validate_route_request(source, destination)
        if not validation_result["valid"]:
            return {
                "success": False,
                "error": validation_result["error"],
                "error_type": validation_result["error_type"]
            }
        
        # Compute normal route (without disaster)
        normal_result = self._pathfinder.find_shortest_path(self._graph, source, destination)
        
        # Compute disaster-aware route if disaster provided
        disaster_result = None
        if disaster is not None:
            try:
                self._disaster_model.apply_disaster_effects(self._graph, disaster)
                disaster_result = self._pathfinder.find_shortest_path(self._graph, source, destination)
                self._disaster_model.remove_disaster_effects(self._graph, disaster)
            except Exception as e:
                return {
                    "success": False,
                    "error": f"Failed to compute disaster-aware route: {str(e)}",
                    "error_type": "comparison_error"
                }
        
        # Format comparison
        return self._format_route_comparison(normal_result, disaster_result, source, destination, disaster)
    
    def _format_route_comparison(self, normal_result: PathResult, 
                                disaster_result: Optional[PathResult],
                                source: str, destination: str,
                                disaster: Optional[DisasterEvent]) -> Dict[str, any]:
        """
        Format route comparison results.
        
        Args:
            normal_result: Normal route result
            disaster_result: Disaster-aware route result
            source: Source vertex ID
            destination: Destination vertex ID
            disaster: Disaster event
            
        Returns:
            Formatted comparison
        """
        comparison = {
            "success": True,
            "source": source,
            "destination": destination,
            "normal_route": {
                "found": normal_result.found,
                "path": normal_result.path if normal_result.found else [],
                "cost": normal_result.total_cost if normal_result.found else None,
                "edge_count": len(normal_result.edges_traversed) if normal_result.found else 0
            }
        }
        
        if disaster_result is not None:
            comparison["disaster_aware_route"] = {
                "found": disaster_result.found,
                "path": disaster_result.path if disaster_result.found else [],
                "cost": disaster_result.total_cost if disaster_result.found else None,
                "edge_count": len(disaster_result.edges_traversed) if disaster_result.found else 0
            }
            
            # Calculate differences
            if normal_result.found and disaster_result.found:
                comparison["analysis"] = {
                    "path_changed": normal_result.path != disaster_result.path,
                    "cost_increase": disaster_result.total_cost - normal_result.total_cost,
                    "cost_increase_percentage": ((disaster_result.total_cost - normal_result.total_cost) / 
                                                normal_result.total_cost * 100) if normal_result.total_cost > 0 else 0,
                    "safety_improvement": "Route adapted to avoid disaster-affected areas",
                    "trade_off": "Increased travel cost for improved safety"
                }
            elif normal_result.found and not disaster_result.found:
                comparison["analysis"] = {
                    "path_changed": True,
                    "reason": "Normal route blocked by disaster",
                    "recommendation": "Seek alternative destination or shelter in place"
                }
            
            comparison["disaster_info"] = self._format_disaster_info(disaster)
        
        return comparison
    
    def get_evacuation_options(self, source: str, 
                              disaster: Optional[DisasterEvent] = None) -> Dict[str, any]:
        """
        Get all evacuation options from a source location.
        
        Args:
            source: Source vertex ID
            disaster: Optional disaster event to consider
            
        Returns:
            Dictionary of evacuation options
        """
        # Validate source
        if not self._graph.has_vertex(source):
            return {
                "success": False,
                "error": f"Source location '{source}' does not exist",
                "error_type": "source_not_found"
            }
        
        # Apply disaster effects if provided
        if disaster is not None:
            try:
                self._disaster_model.apply_disaster_effects(self._graph, disaster)
            except Exception as e:
                return {
                    "success": False,
                    "error": f"Failed to apply disaster effects: {str(e)}",
                    "error_type": "disaster_error"
                }
        
        # Find all evacuation points and shelters
        evacuation_targets = []
        for vertex in self._graph.get_all_vertices():
            if vertex.vertex_type in [VertexType.EVACUATION_POINT, VertexType.SHELTER]:
                if vertex.id != source:
                    evacuation_targets.append(vertex)
        
        # Compute routes to all targets
        options = []
        for target_vertex in evacuation_targets:
            result = self._pathfinder.find_shortest_path(self._graph, source, target_vertex.id)
            
            if result.found:
                options.append({
                    "destination": target_vertex.id,
                    "type": target_vertex.vertex_type.value,
                    "capacity": target_vertex.capacity,
                    "coordinates": target_vertex.coordinates,
                    "cost": result.total_cost,
                    "distance": sum(edge.base_distance for edge in result.edges_traversed),
                    "path_length": len(result.path),
                    "reachable": True
                })
            else:
                options.append({
                    "destination": target_vertex.id,
                    "type": target_vertex.vertex_type.value,
                    "capacity": target_vertex.capacity,
                    "coordinates": target_vertex.coordinates,
                    "reachable": False,
                    "reason": result.error_message
                })
        
        # Sort by cost (reachable first, then by cost)
        options.sort(key=lambda x: (not x.get("reachable", False), x.get("cost", float('inf'))))
        
        # Remove disaster effects if applied
        if disaster is not None:
            self._disaster_model.remove_disaster_effects(self._graph, disaster)
        
        return {
            "success": True,
            "source": source,
            "evacuation_options": options,
            "total_options": len(options),
            "reachable_options": sum(1 for opt in options if opt.get("reachable", False)),
            "disaster_applied": disaster is not None
        }
    
    def clear_disasters(self) -> None:
        """Clear all active disaster effects from the graph."""
        self._disaster_model.clear_all_disaster_effects(self._graph)
        self._active_disasters.clear()
    
    def get_active_disasters(self) -> List[DisasterEvent]:
        """Get list of currently active disasters."""
        return self._active_disasters.copy()
    
    def get_graph_summary(self) -> Dict[str, any]:
        """
        Get summary information about the graph.
        
        Returns:
            Dictionary with graph statistics
        """
        info = self._graph.get_graph_info()
        
        # Count evacuation points and shelters
        evacuation_points = 0
        shelters = 0
        intersections = 0
        
        for vertex in self._graph.get_all_vertices():
            if vertex.vertex_type == VertexType.EVACUATION_POINT:
                evacuation_points += 1
            elif vertex.vertex_type == VertexType.SHELTER:
                shelters += 1
            elif vertex.vertex_type == VertexType.INTERSECTION:
                intersections += 1
        
        return {
            "total_vertices": info["vertex_count"],
            "total_edges": info["edge_count"],
            "intersections": intersections,
            "evacuation_points": evacuation_points,
            "shelters": shelters,
            "average_degree": info["average_degree"],
            "active_disasters": len(self._active_disasters)
        }
    
    def __str__(self) -> str:
        """String representation of the route controller."""
        return f"RouteController(graph_vertices={self._graph.get_vertex_count()}, active_disasters={len(self._active_disasters)})"
    
    def __repr__(self) -> str:
        return self.__str__()