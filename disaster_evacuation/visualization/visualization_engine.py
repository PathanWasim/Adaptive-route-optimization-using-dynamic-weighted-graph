"""
Visualization Engine for the disaster evacuation routing system.

This module implements the VisualizationEngine class that provides graph visualization
capabilities including vertex and edge rendering, path highlighting, and disaster effects display.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from ..models import Vertex, Edge, DisasterEvent, VertexType, DisasterType
from ..models import GraphManager


class VisualizationEngine:
    """
    Provides visualization capabilities for the disaster evacuation routing system.
    
    The VisualizationEngine creates clear, informative visualizations of:
    - Graph structure with vertices and edges
    - Optimal paths with distinct highlighting
    - Disaster effects and affected areas
    - Comparative path analysis
    - Real-time updates during route computation
    """
    
    def __init__(self, figsize: Tuple[float, float] = (12, 8)):
        """
        Initialize the visualization engine.
        
        Args:
            figsize: Figure size as (width, height) in inches
        """
        self.figsize = figsize
        self.vertex_colors = {
            VertexType.INTERSECTION: '#4CAF50',      # Green
            VertexType.SHELTER: '#2196F3',           # Blue
            VertexType.EVACUATION_POINT: '#FF9800'  # Orange
        }
        self.vertex_sizes = {
            VertexType.INTERSECTION: 300,
            VertexType.SHELTER: 500,
            VertexType.EVACUATION_POINT: 600
        }
        self.disaster_colors = {
            DisasterType.FLOOD: '#1976D2',      # Blue
            DisasterType.FIRE: '#D32F2F',       # Red
            DisasterType.EARTHQUAKE: '#7B1FA2'  # Purple
        }
    
    def visualize_graph(self, graph: GraphManager, title: str = "Evacuation Network", 
                       save_path: Optional[str] = None) -> plt.Figure:
        """
        Visualize the complete graph structure.
        
        Args:
            graph: GraphManager instance containing the network
            title: Title for the visualization
            save_path: Optional path to save the figure
            
        Returns:
            Matplotlib figure object
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Get all vertices and edges
        vertices = graph.get_all_vertices()
        edges = graph.get_all_edges()
        
        if not vertices:
            ax.text(0.5, 0.5, 'No vertices in graph', 
                   horizontalalignment='center', verticalalignment='center',
                   transform=ax.transAxes, fontsize=16)
            ax.set_title(title)
            return fig
        
        # Draw edges first (so they appear behind vertices)
        self._draw_edges(ax, edges, vertices)
        
        # Draw vertices
        self._draw_vertices(ax, vertices)
        
        # Add legend
        self._add_vertex_legend(ax)
        
        # Set title and formatting
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        
        # Adjust layout
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def visualize_path(self, graph: GraphManager, path: List[str], 
                      path_edges: List[Edge], title: str = "Evacuation Route",
                      save_path: Optional[str] = None) -> plt.Figure:
        """
        Visualize a specific path through the graph.
        
        Args:
            graph: GraphManager instance containing the network
            path: List of vertex IDs in the path
            path_edges: List of edges traversed in the path
            title: Title for the visualization
            save_path: Optional path to save the figure
            
        Returns:
            Matplotlib figure object
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        vertices = graph.get_all_vertices()
        all_edges = graph.get_all_edges()
        
        if not vertices or not path:
            ax.text(0.5, 0.5, 'No path to display', 
                   horizontalalignment='center', verticalalignment='center',
                   transform=ax.transAxes, fontsize=16)
            ax.set_title(title)
            return fig
        
        # Draw all edges in light gray
        self._draw_edges(ax, all_edges, vertices, color='lightgray', alpha=0.5, width=1)
        
        # Draw path edges in bold red
        self._draw_edges(ax, path_edges, vertices, color='red', alpha=0.8, width=3)
        
        # Draw all vertices
        self._draw_vertices(ax, vertices, highlight_path=path)
        
        # Add path information
        self._add_path_info(ax, path, path_edges)
        
        # Add legend
        self._add_path_legend(ax)
        
        # Set title and formatting
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def visualize_disaster_effects(self, graph: GraphManager, disaster: DisasterEvent,
                                 affected_edges: List[Edge], title: str = "Disaster Effects",
                                 save_path: Optional[str] = None) -> plt.Figure:
        """
        Visualize disaster effects on the graph.
        
        Args:
            graph: GraphManager instance containing the network
            disaster: DisasterEvent to visualize
            affected_edges: List of edges affected by the disaster
            title: Title for the visualization
            save_path: Optional path to save the figure
            
        Returns:
            Matplotlib figure object
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        vertices = graph.get_all_vertices()
        all_edges = graph.get_all_edges()
        
        # Draw disaster effect area
        self._draw_disaster_area(ax, disaster)
        
        # Draw normal edges
        normal_edges = [e for e in all_edges if e not in affected_edges]
        self._draw_edges(ax, normal_edges, vertices, color='gray', alpha=0.6, width=1)
        
        # Draw affected edges with color coding
        blocked_edges = [e for e in affected_edges if e.is_blocked]
        high_risk_edges = [e for e in affected_edges if not e.is_blocked]
        
        self._draw_edges(ax, high_risk_edges, vertices, color='orange', alpha=0.8, width=2)
        self._draw_edges(ax, blocked_edges, vertices, color='red', alpha=0.9, width=3, style='--')
        
        # Draw vertices
        self._draw_vertices(ax, vertices)
        
        # Add disaster information
        self._add_disaster_info(ax, disaster, affected_edges)
        
        # Add legend
        self._add_disaster_legend(ax)
        
        # Set title and formatting
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def visualize_path_comparison(self, graph: GraphManager, 
                                normal_path: List[str], normal_edges: List[Edge],
                                disaster_path: List[str], disaster_edges: List[Edge],
                                disaster: Optional[DisasterEvent] = None,
                                title: str = "Path Comparison",
                                save_path: Optional[str] = None) -> plt.Figure:
        """
        Visualize comparison between normal and disaster-aware paths.
        
        Args:
            graph: GraphManager instance containing the network
            normal_path: Normal shortest path
            normal_edges: Edges in normal path
            disaster_path: Disaster-aware path
            disaster_edges: Edges in disaster-aware path
            disaster: Optional disaster event
            title: Title for the visualization
            save_path: Optional path to save the figure
            
        Returns:
            Matplotlib figure object
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(self.figsize[0] * 1.5, self.figsize[1]))
        
        vertices = graph.get_all_vertices()
        all_edges = graph.get_all_edges()
        
        # Left subplot: Normal path
        self._draw_comparison_subplot(ax1, vertices, all_edges, normal_path, normal_edges,
                                    "Normal Shortest Path", 'blue')
        
        # Right subplot: Disaster-aware path
        self._draw_comparison_subplot(ax2, vertices, all_edges, disaster_path, disaster_edges,
                                    "Disaster-Aware Path", 'red')
        
        # Add disaster area to right subplot if provided
        if disaster:
            self._draw_disaster_area(ax2, disaster, alpha=0.2)
        
        # Add comparison legend
        self._add_comparison_legend(fig)
        
        # Set main title
        fig.suptitle(title, fontsize=18, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def _draw_vertices(self, ax: plt.Axes, vertices: List[Vertex], 
                      highlight_path: Optional[List[str]] = None) -> None:
        """Draw vertices on the plot."""
        for vertex in vertices:
            x, y = vertex.coordinates
            color = self.vertex_colors[vertex.vertex_type]
            size = self.vertex_sizes[vertex.vertex_type]
            
            # Highlight path vertices
            if highlight_path and vertex.id in highlight_path:
                # Add highlight ring
                ax.scatter(x, y, s=size * 1.5, c='yellow', alpha=0.6, edgecolors='black', linewidth=2)
            
            # Draw vertex
            ax.scatter(x, y, s=size, c=color, alpha=0.8, edgecolors='black', linewidth=1)
            
            # Add vertex label
            ax.annotate(vertex.id, (x, y), xytext=(5, 5), textcoords='offset points',
                       fontsize=10, fontweight='bold', color='white',
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.7))
    
    def _draw_edges(self, ax: plt.Axes, edges: List[Edge], vertices: List[Vertex],
                   color: str = 'gray', alpha: float = 0.7, width: float = 1,
                   style: str = '-') -> None:
        """Draw edges on the plot."""
        # Create vertex coordinate lookup
        vertex_coords = {v.id: v.coordinates for v in vertices}
        
        for edge in edges:
            if edge.source in vertex_coords and edge.target in vertex_coords:
                x1, y1 = vertex_coords[edge.source]
                x2, y2 = vertex_coords[edge.target]
                
                ax.plot([x1, x2], [y1, y2], color=color, alpha=alpha, 
                       linewidth=width, linestyle=style)
    
    def _draw_disaster_area(self, ax: plt.Axes, disaster: DisasterEvent, 
                          alpha: float = 0.3) -> None:
        """Draw disaster effect area."""
        x, y = disaster.epicenter
        radius = disaster.max_effect_radius
        color = self.disaster_colors[disaster.disaster_type]
        
        # Draw disaster circle
        circle = patches.Circle((x, y), radius, facecolor=color, alpha=alpha,
                              edgecolor=color, linewidth=2)
        ax.add_patch(circle)
        
        # Mark epicenter
        ax.scatter(x, y, s=100, c=color, marker='x', linewidth=3)
        ax.annotate(f'{disaster.disaster_type.value.title()} Epicenter', 
                   (x, y), xytext=(10, 10), textcoords='offset points',
                   fontsize=9, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.8))
    
    def _draw_comparison_subplot(self, ax: plt.Axes, vertices: List[Vertex], 
                               all_edges: List[Edge], path: List[str], 
                               path_edges: List[Edge], title: str, path_color: str) -> None:
        """Draw a single subplot for path comparison."""
        # Draw all edges in light gray
        self._draw_edges(ax, all_edges, vertices, color='lightgray', alpha=0.5, width=1)
        
        # Draw path edges
        self._draw_edges(ax, path_edges, vertices, color=path_color, alpha=0.8, width=3)
        
        # Draw vertices
        self._draw_vertices(ax, vertices, highlight_path=path)
        
        # Set formatting
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
    
    def _add_vertex_legend(self, ax: plt.Axes) -> None:
        """Add legend for vertex types."""
        legend_elements = []
        for vertex_type, color in self.vertex_colors.items():
            legend_elements.append(
                plt.scatter([], [], s=100, c=color, alpha=0.8, edgecolors='black',
                          label=vertex_type.value.replace('_', ' ').title())
            )
        ax.legend(handles=legend_elements, loc='upper right')
    
    def _add_path_legend(self, ax: plt.Axes) -> None:
        """Add legend for path visualization."""
        legend_elements = [
            plt.Line2D([0], [0], color='red', linewidth=3, label='Evacuation Route'),
            plt.Line2D([0], [0], color='lightgray', linewidth=1, label='Other Roads'),
            plt.scatter([], [], s=100, c='yellow', alpha=0.6, edgecolors='black',
                       label='Path Vertices')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
    
    def _add_disaster_legend(self, ax: plt.Axes) -> None:
        """Add legend for disaster visualization."""
        legend_elements = [
            plt.Line2D([0], [0], color='red', linewidth=3, linestyle='--', label='Blocked Roads'),
            plt.Line2D([0], [0], color='orange', linewidth=2, label='High Risk Roads'),
            plt.Line2D([0], [0], color='gray', linewidth=1, label='Normal Roads'),
            patches.Circle((0, 0), 1, facecolor='red', alpha=0.3, label='Disaster Area')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
    
    def _add_comparison_legend(self, fig: plt.Figure) -> None:
        """Add legend for path comparison."""
        legend_elements = [
            plt.Line2D([0], [0], color='blue', linewidth=3, label='Normal Path'),
            plt.Line2D([0], [0], color='red', linewidth=3, label='Disaster-Aware Path'),
            plt.scatter([], [], s=100, c='yellow', alpha=0.6, edgecolors='black',
                       label='Path Vertices')
        ]
        fig.legend(handles=legend_elements, loc='upper center', ncol=3, 
                  bbox_to_anchor=(0.5, 0.95))
    
    def _add_path_info(self, ax: plt.Axes, path: List[str], path_edges: List[Edge]) -> None:
        """Add path information text box."""
        total_distance = sum(edge.base_distance for edge in path_edges)
        total_cost = sum(edge.current_weight for edge in path_edges)
        
        info_text = f"Path: {' → '.join(path)}\n"
        info_text += f"Distance: {total_distance:.2f}\n"
        info_text += f"Total Cost: {total_cost:.2f}\n"
        info_text += f"Edges: {len(path_edges)}"
        
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    def _add_disaster_info(self, ax: plt.Axes, disaster: DisasterEvent, 
                          affected_edges: List[Edge]) -> None:
        """Add disaster information text box."""
        blocked_count = sum(1 for edge in affected_edges if edge.is_blocked)
        high_risk_count = len(affected_edges) - blocked_count
        
        info_text = f"Disaster: {disaster.disaster_type.value.title()}\n"
        info_text += f"Severity: {disaster.severity:.1f}\n"
        info_text += f"Radius: {disaster.max_effect_radius:.1f}\n"
        info_text += f"Blocked Roads: {blocked_count}\n"
        info_text += f"High Risk Roads: {high_risk_count}"
        
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
    
    def visualize_dynamic_comparison(self, graph: GraphManager,
                                   comparison_data: Dict[str, Any],
                                   title: str = "Dynamic Route Comparison",
                                   save_path: Optional[str] = None) -> plt.Figure:
        """
        Create a dynamic comparison visualization with multiple scenarios.
        
        Args:
            graph: GraphManager instance containing the network
            comparison_data: Dictionary containing comparison results from RouteController
            title: Title for the visualization
            save_path: Optional path to save the figure
            
        Returns:
            Matplotlib figure object
        """
        fig = plt.figure(figsize=(self.figsize[0] * 2, self.figsize[1] * 1.2))
        
        # Create grid layout: 2x2 with bottom spanning both columns
        gs = fig.add_gridspec(3, 2, height_ratios=[2, 2, 1], hspace=0.3, wspace=0.2)
        
        vertices = graph.get_all_vertices()
        all_edges = graph.get_all_edges()
        
        # Top left: Normal route
        ax1 = fig.add_subplot(gs[0, 0])
        if comparison_data["normal_route"]["found"]:
            normal_path = comparison_data["normal_route"]["path"]
            # Reconstruct edges from path (simplified for visualization)
            normal_edges = self._reconstruct_path_edges(graph, normal_path)
            self._draw_comparison_subplot(ax1, vertices, all_edges, normal_path, normal_edges,
                                        "Normal Shortest Path", 'blue')
        else:
            ax1.text(0.5, 0.5, 'No normal path found', ha='center', va='center', transform=ax1.transAxes)
            ax1.set_title("Normal Shortest Path", fontsize=14, fontweight='bold')
        
        # Top right: Disaster-aware route
        ax2 = fig.add_subplot(gs[0, 1])
        if comparison_data["disaster_aware_route"]["found"]:
            disaster_path = comparison_data["disaster_aware_route"]["path"]
            disaster_edges = self._reconstruct_path_edges(graph, disaster_path)
            self._draw_comparison_subplot(ax2, vertices, all_edges, disaster_path, disaster_edges,
                                        "Disaster-Aware Path", 'red')
            
            # Add disaster area if available
            if "disaster_info" in comparison_data and comparison_data["disaster_info"]:
                disaster_info = comparison_data["disaster_info"]
                disaster = DisasterEvent(
                    disaster_type=DisasterType(disaster_info["type"]),
                    epicenter=tuple(disaster_info["epicenter"]),
                    severity=disaster_info["severity"],
                    max_effect_radius=disaster_info["effect_radius"],
                    start_time=datetime.now()
                )
                self._draw_disaster_area(ax2, disaster, alpha=0.2)
        else:
            ax2.text(0.5, 0.5, 'No disaster-aware path found', ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title("Disaster-Aware Path", fontsize=14, fontweight='bold')
        
        # Bottom left: Overlay comparison
        ax3 = fig.add_subplot(gs[1, 0])
        self._draw_overlay_comparison(ax3, vertices, all_edges, comparison_data)
        
        # Bottom right: Risk analysis
        ax4 = fig.add_subplot(gs[1, 1])
        self._draw_risk_analysis(ax4, comparison_data)
        
        # Bottom: Statistics comparison
        ax5 = fig.add_subplot(gs[2, :])
        self._draw_statistics_comparison(ax5, comparison_data)
        
        # Set main title
        fig.suptitle(title, fontsize=18, fontweight='bold')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def visualize_real_time_updates(self, graph: GraphManager, 
                                  update_sequence: List[Dict[str, Any]],
                                  save_dir: str = "real_time_updates") -> List[str]:
        """
        Create a sequence of visualizations showing real-time route updates.
        
        Args:
            graph: GraphManager instance
            update_sequence: List of route computation results over time
            save_dir: Directory to save the sequence of images
            
        Returns:
            List of file paths for the generated images
        """
        import os
        
        os.makedirs(save_dir, exist_ok=True)
        saved_files = []
        
        for i, update_data in enumerate(update_sequence):
            fig, ax = plt.subplots(figsize=self.figsize)
            
            vertices = graph.get_all_vertices()
            all_edges = graph.get_all_edges()
            
            # Draw base graph
            self._draw_edges(ax, all_edges, vertices, color='lightgray', alpha=0.5, width=1)
            
            # Draw current route if available
            if update_data.get("success", False):
                path = update_data["path"]
                path_edges = update_data["edges"]
                
                self._draw_edges(ax, path_edges, vertices, color='red', alpha=0.8, width=3)
                self._draw_vertices(ax, vertices, highlight_path=path)
                
                # Add timestamp and route info
                timestamp = update_data.get("timestamp", f"Update {i+1}")
                info_text = f"Time: {timestamp}\n"
                info_text += f"Route: {' → '.join(path)}\n"
                info_text += f"Cost: {update_data.get('total_cost', 'N/A'):.2f}"
                
                ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=10,
                       verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
            else:
                self._draw_vertices(ax, vertices)
                ax.text(0.5, 0.5, 'No route available', ha='center', va='center', 
                       transform=ax.transAxes, fontsize=16, color='red')
            
            # Add disaster effects if present
            if update_data.get("disaster_applied", False) and update_data.get("disaster_info"):
                disaster_info = update_data["disaster_info"]
                disaster = DisasterEvent(
                    disaster_type=DisasterType(disaster_info["type"]),
                    epicenter=tuple(disaster_info["epicenter"]),
                    severity=disaster_info["severity"],
                    max_effect_radius=disaster_info["effect_radius"],
                    start_time=datetime.now()
                )
                self._draw_disaster_area(ax, disaster, alpha=0.3)
            
            ax.set_title(f"Real-Time Route Update #{i+1}", fontsize=16, fontweight='bold')
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
            ax.set_xlabel('X Coordinate')
            ax.set_ylabel('Y Coordinate')
            
            # Save frame
            filename = f"update_{i+1:03d}.png"
            filepath = os.path.join(save_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            saved_files.append(filepath)
            
            plt.close(fig)
        
        return saved_files
    
    def create_interactive_comparison_dashboard(self, graph: GraphManager,
                                              scenarios: List[Dict[str, Any]],
                                              save_path: str) -> None:
        """
        Create an interactive dashboard comparing multiple evacuation scenarios.
        
        Args:
            graph: GraphManager instance
            scenarios: List of different evacuation scenarios to compare
            save_path: Path to save the dashboard HTML file
        """
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            import plotly.offline as pyo
        except ImportError:
            print("Plotly not available. Install with: pip install plotly")
            return
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Network Overview', 'Route Comparison', 
                          'Cost Analysis', 'Risk Assessment'),
            specs=[[{"type": "scatter"}, {"type": "scatter"}],
                   [{"type": "bar"}, {"type": "bar"}]]
        )
        
        vertices = graph.get_all_vertices()
        all_edges = graph.get_all_edges()
        
        # Network overview (top-left)
        for vertex in vertices:
            x, y = vertex.coordinates
            color = self.vertex_colors[vertex.vertex_type]
            fig.add_trace(
                go.Scatter(x=[x], y=[y], mode='markers+text', 
                          marker=dict(size=15, color=color),
                          text=vertex.id, textposition="middle center",
                          name=vertex.vertex_type.value, showlegend=False),
                row=1, col=1
            )
        
        # Add edges to network overview
        for edge in all_edges:
            source_vertex = next(v for v in vertices if v.id == edge.source)
            target_vertex = next(v for v in vertices if v.id == edge.target)
            
            fig.add_trace(
                go.Scatter(x=[source_vertex.coordinates[0], target_vertex.coordinates[0]],
                          y=[source_vertex.coordinates[1], target_vertex.coordinates[1]],
                          mode='lines', line=dict(color='gray', width=1),
                          showlegend=False, hoverinfo='skip'),
                row=1, col=1
            )
        
        # Route comparison (top-right)
        colors = ['blue', 'red', 'green', 'orange', 'purple']
        for i, scenario in enumerate(scenarios[:5]):  # Limit to 5 scenarios
            if scenario.get("success", False):
                path = scenario["path"]
                path_coords = []
                for vertex_id in path:
                    vertex = next(v for v in vertices if v.id == vertex_id)
                    path_coords.append(vertex.coordinates)
                
                x_coords = [coord[0] for coord in path_coords]
                y_coords = [coord[1] for coord in path_coords]
                
                fig.add_trace(
                    go.Scatter(x=x_coords, y=y_coords, mode='lines+markers',
                              line=dict(color=colors[i], width=3),
                              name=f"Scenario {i+1}"),
                    row=1, col=2
                )
        
        # Cost analysis (bottom-left)
        scenario_names = [f"Scenario {i+1}" for i in range(len(scenarios))]
        costs = [s.get("total_cost", 0) for s in scenarios]
        
        fig.add_trace(
            go.Bar(x=scenario_names, y=costs, name="Total Cost",
                   marker_color='lightblue'),
            row=2, col=1
        )
        
        # Risk assessment (bottom-right)
        risks = []
        for scenario in scenarios:
            if "statistics" in scenario:
                risks.append(scenario["statistics"].get("total_risk", 0))
            else:
                risks.append(0)
        
        fig.add_trace(
            go.Bar(x=scenario_names, y=risks, name="Total Risk",
                   marker_color='lightcoral'),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title="Evacuation Route Comparison Dashboard",
            height=800,
            showlegend=True
        )
        
        # Save as HTML
        pyo.plot(fig, filename=save_path, auto_open=False)
    
    def _reconstruct_path_edges(self, graph: GraphManager, path: List[str]) -> List[Edge]:
        """Reconstruct edges from a path of vertex IDs."""
        edges = []
        for i in range(len(path) - 1):
            edge = graph.get_edge(path[i], path[i + 1])
            if edge:
                edges.append(edge)
        return edges
    
    def _draw_overlay_comparison(self, ax: plt.Axes, vertices: List[Vertex],
                               all_edges: List[Edge], comparison_data: Dict[str, Any]) -> None:
        """Draw overlay comparison of normal vs disaster-aware paths."""
        # Draw base graph
        self._draw_edges(ax, all_edges, vertices, color='lightgray', alpha=0.3, width=1)
        self._draw_vertices(ax, vertices)
        
        # For overlay comparison, we'll draw simplified paths without edge reconstruction
        # since we don't have access to the original graph in this context
        
        # Draw normal path if available
        if comparison_data["normal_route"]["found"]:
            normal_path = comparison_data["normal_route"]["path"]
            self._draw_path_as_lines(ax, normal_path, vertices, color='blue', alpha=0.7, width=2, style='--')
        
        # Draw disaster-aware path if available
        if comparison_data["disaster_aware_route"]["found"]:
            disaster_path = comparison_data["disaster_aware_route"]["path"]
            self._draw_path_as_lines(ax, disaster_path, vertices, color='red', alpha=0.8, width=3)
        
        ax.set_title("Path Overlay Comparison", fontsize=14, fontweight='bold')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        # Add legend
        legend_elements = [
            plt.Line2D([0], [0], color='blue', linewidth=2, linestyle='--', label='Normal Path'),
            plt.Line2D([0], [0], color='red', linewidth=3, label='Disaster-Aware Path')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
    
    def _draw_path_as_lines(self, ax: plt.Axes, path: List[str], vertices: List[Vertex],
                           color: str = 'red', alpha: float = 0.8, width: float = 2, style: str = '-') -> None:
        """Draw a path as connected lines between vertices."""
        # Create vertex coordinate lookup
        vertex_coords = {v.id: v.coordinates for v in vertices}
        
        # Draw lines between consecutive vertices in path
        for i in range(len(path) - 1):
            if path[i] in vertex_coords and path[i + 1] in vertex_coords:
                x1, y1 = vertex_coords[path[i]]
                x2, y2 = vertex_coords[path[i + 1]]
                ax.plot([x1, x2], [y1, y2], color=color, alpha=alpha, 
                       linewidth=width, linestyle=style)
    
    def _draw_risk_analysis(self, ax: plt.Axes, comparison_data: Dict[str, Any]) -> None:
        """Draw risk analysis comparison chart."""
        categories = ['Distance', 'Risk', 'Congestion', 'Total Cost']
        
        normal_values = []
        disaster_values = []
        
        if comparison_data["normal_route"]["found"]:
            normal_values = [
                comparison_data["normal_route"].get("distance", 0),
                comparison_data["normal_route"].get("risk", 0),
                comparison_data["normal_route"].get("congestion", 0),
                comparison_data["normal_route"].get("cost", 0)
            ]
        else:
            normal_values = [0, 0, 0, 0]
        
        if comparison_data["disaster_aware_route"]["found"]:
            disaster_values = [
                comparison_data["disaster_aware_route"].get("distance", 0),
                comparison_data["disaster_aware_route"].get("risk", 0),
                comparison_data["disaster_aware_route"].get("congestion", 0),
                comparison_data["disaster_aware_route"].get("cost", 0)
            ]
        else:
            disaster_values = [0, 0, 0, 0]
        
        x = range(len(categories))
        width = 0.35
        
        ax.bar([i - width/2 for i in x], normal_values, width, label='Normal Route', color='blue', alpha=0.7)
        ax.bar([i + width/2 for i in x], disaster_values, width, label='Disaster-Aware Route', color='red', alpha=0.7)
        
        ax.set_xlabel('Metrics')
        ax.set_ylabel('Values')
        ax.set_title('Risk Analysis Comparison', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(categories, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _draw_statistics_comparison(self, ax: plt.Axes, comparison_data: Dict[str, Any]) -> None:
        """Draw statistics comparison table."""
        ax.axis('tight')
        ax.axis('off')
        
        # Prepare data for table
        headers = ['Metric', 'Normal Route', 'Disaster-Aware Route', 'Difference']
        data = []
        
        if comparison_data["normal_route"]["found"] and comparison_data["disaster_aware_route"]["found"]:
            normal_cost = comparison_data["normal_route"].get("cost", 0)
            disaster_cost = comparison_data["disaster_aware_route"].get("cost", 0)
            
            data.append(['Total Cost', f'{normal_cost:.2f}', f'{disaster_cost:.2f}', 
                        f'{disaster_cost - normal_cost:.2f}'])
            data.append(['Edge Count', 
                        str(comparison_data["normal_route"].get("edge_count", 0)),
                        str(comparison_data["disaster_aware_route"].get("edge_count", 0)),
                        str(comparison_data["disaster_aware_route"].get("edge_count", 0) - 
                            comparison_data["normal_route"].get("edge_count", 0))])
            
            if "analysis" in comparison_data:
                analysis = comparison_data["analysis"]
                data.append(['Path Changed', '', '', 'Yes' if analysis.get("path_changed", False) else 'No'])
                data.append(['Cost Increase %', '', '', f'{analysis.get("cost_increase_percentage", 0):.1f}%'])
        else:
            data.append(['Status', 
                        'Found' if comparison_data["normal_route"]["found"] else 'Not Found',
                        'Found' if comparison_data["disaster_aware_route"]["found"] else 'Not Found',
                        'N/A'])
        
        # Create table
        table = ax.table(cellText=data, colLabels=headers, cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        
        # Style the table
        for i in range(len(headers)):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        ax.set_title('Route Comparison Statistics', fontsize=14, fontweight='bold', pad=20)
    
    def save_visualization_summary(self, graph: GraphManager, 
                                 route_result: Dict[str, Any],
                                 save_dir: str = "visualizations") -> Dict[str, str]:
        """
        Save a complete visualization summary for a route computation.
        
        Args:
            graph: GraphManager instance
            route_result: Result from RouteController.compute_route()
            save_dir: Directory to save visualizations
            
        Returns:
            Dictionary mapping visualization types to file paths
        """
        import os
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        saved_files = {}
        
        # Save graph visualization
        graph_fig = self.visualize_graph(graph, "Complete Evacuation Network")
        graph_path = os.path.join(save_dir, "network_graph.png")
        graph_fig.savefig(graph_path, dpi=300, bbox_inches='tight')
        saved_files["network"] = graph_path
        plt.close(graph_fig)
        
        if route_result["success"]:
            # Save path visualization
            path_fig = self.visualize_path(
                graph, route_result["path"], route_result["edges"],
                f"Evacuation Route: {route_result['source']['id']} → {route_result['destination']['id']}"
            )
            path_path = os.path.join(save_dir, "evacuation_route.png")
            path_fig.savefig(path_path, dpi=300, bbox_inches='tight')
            saved_files["route"] = path_path
            plt.close(path_fig)
            
            # Save disaster effects if applicable
            if route_result["disaster_applied"] and route_result["disaster_info"]:
                # Note: This would require the actual disaster object and affected edges
                # which aren't directly available in the route_result
                pass
        
        return saved_files
    
    def __str__(self) -> str:
        """String representation of the visualization engine."""
        return f"VisualizationEngine(figsize={self.figsize})"
    
    def __repr__(self) -> str:
        return self.__str__()