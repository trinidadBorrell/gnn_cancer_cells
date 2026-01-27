"""
Visualization module for sanity checks on graphs and features.

Provides:
- Feature matrix heatmap
- Adjacency matrix visualization
- Node neighborhood plots on original image
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from torch_geometric.data import Data
from PIL import Image
from pathlib import Path
from typing import Optional, Union, Tuple
import pandas as pd


def plot_feature_matrix(
    features: np.ndarray,
    output_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (12, 8),
    title: str = "Feature Matrix"
) -> plt.Figure:
    """
    Plot feature matrix as a heatmap.
    
    Args:
        features: Feature matrix (N_nodes, N_features)
        output_path: Path to save figure (optional)
        figsize: Figure size
        title: Plot title
    
    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Normalize features for better visualization
    vmin, vmax = np.percentile(features, [2, 98])
    
    im = ax.imshow(features, aspect='auto', cmap='viridis', vmin=vmin, vmax=vmax)
    ax.set_xlabel('Feature Dimension')
    ax.set_ylabel('Node Index')
    ax.set_title(title)
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Feature Value')
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved feature matrix plot to: {output_path}")
    
    return fig


def plot_adjacency_matrix(
    edge_index: torch.Tensor,
    num_nodes: int,
    output_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (8, 8),
    title: str = "Adjacency Matrix"
) -> plt.Figure:
    """
    Plot adjacency matrix from edge index.
    
    Args:
        edge_index: Edge index tensor (2, E)
        num_nodes: Number of nodes
        output_path: Path to save figure (optional)
        figsize: Figure size
        title: Plot title
    
    Returns:
        matplotlib Figure
    """
    # Build adjacency matrix
    adj = np.zeros((num_nodes, num_nodes))
    src = edge_index[0].numpy()
    dst = edge_index[1].numpy()
    adj[src, dst] = 1
    
    fig, ax = plt.subplots(figsize=figsize)
    
    im = ax.imshow(adj, cmap='Blues', interpolation='nearest')
    ax.set_xlabel('Node Index')
    ax.set_ylabel('Node Index')
    ax.set_title(title)
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Edge (0/1)')
    
    # Add grid for small matrices
    if num_nodes <= 50:
        ax.set_xticks(np.arange(-0.5, num_nodes, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, num_nodes, 1), minor=True)
        ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved adjacency matrix plot to: {output_path}")
    
    return fig


def get_node_neighbors(edge_index: torch.Tensor, node_idx: int) -> np.ndarray:
    """Get indices of neighbors for a given node."""
    mask = edge_index[0] == node_idx
    neighbors = edge_index[1][mask].numpy()
    return neighbors


def plot_node_neighborhoods(
    graph: Data,
    image: np.ndarray,
    annotations: pd.DataFrame,
    output_path: Optional[Union[str, Path]] = None,
    max_nodes: int = 20,
    figsize_per_node: Tuple[float, float] = (4, 4)
) -> plt.Figure:
    """
    Plot each node with its neighbors highlighted on the original image.
    
    Args:
        graph: PyG Data object with edge_index and pos
        image: Original RGB image as numpy array
        annotations: DataFrame with xmin, ymin, xmax, ymax
        output_path: Path to save figure
        max_nodes: Maximum number of nodes to plot (for large graphs)
        figsize_per_node: Size per subplot
    
    Returns:
        matplotlib Figure
    """
    num_nodes = min(graph.x.shape[0], max_nodes)
    
    # Create figure with 1 row x num_nodes columns
    fig, axes = plt.subplots(1, num_nodes, figsize=(figsize_per_node[0] * num_nodes, figsize_per_node[1]))
    
    if num_nodes == 1:
        axes = [axes]
    
    # Colors for visualization
    node_color = 'red'
    neighbor_color = 'cyan'
    other_color = 'gray'
    
    for node_idx, ax in enumerate(axes):
        ax.imshow(image)
        
        # Get neighbors of this node
        neighbors = get_node_neighbors(graph.edge_index, node_idx)
        neighbor_set = set(neighbors)
        
        # Draw all cells
        for i, row in annotations.iterrows():
            xmin, ymin = int(row['xmin']), int(row['ymin'])
            xmax, ymax = int(row['xmax']), int(row['ymax'])
            width = xmax - xmin
            height = ymax - ymin
            
            if i == node_idx:
                # Current node - red with thick border
                rect = mpatches.Rectangle(
                    (xmin, ymin), width, height,
                    linewidth=3, edgecolor=node_color, facecolor='none'
                )
                ax.add_patch(rect)
            elif i in neighbor_set:
                # Neighbor - cyan
                rect = mpatches.Rectangle(
                    (xmin, ymin), width, height,
                    linewidth=2, edgecolor=neighbor_color, facecolor=neighbor_color, alpha=0.3
                )
                ax.add_patch(rect)
            else:
                # Other cells - gray dashed
                rect = mpatches.Rectangle(
                    (xmin, ymin), width, height,
                    linewidth=1, edgecolor=other_color, facecolor='none', linestyle='--', alpha=0.5
                )
                ax.add_patch(rect)
        
        # Draw edges from current node to neighbors
        if hasattr(graph, 'pos') and graph.pos is not None:
            node_pos = graph.pos[node_idx].numpy()
            for neighbor_idx in neighbors:
                neighbor_pos = graph.pos[neighbor_idx].numpy()
                ax.plot(
                    [node_pos[0], neighbor_pos[0]],
                    [node_pos[1], neighbor_pos[1]],
                    color='yellow', linewidth=1.5, alpha=0.7
                )
        
        ax.set_title(f'Node {node_idx}\n({len(neighbors)} neighbors)', fontsize=9)
        ax.axis('off')
    
    # Add legend
    legend_elements = [
        mpatches.Patch(edgecolor=node_color, facecolor='none', linewidth=3, label='Current Node'),
        mpatches.Patch(facecolor=neighbor_color, alpha=0.3, label='Neighbors'),
        mpatches.Patch(edgecolor=other_color, facecolor='none', linestyle='--', label='Other Cells'),
    ]
    fig.legend(handles=legend_elements, loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.05))
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved node neighborhoods plot to: {output_path}")
    
    return fig


def visualize_graph_sanity_check(
    graph: Data,
    image_path: Union[str, Path],
    annotations: pd.DataFrame,
    output_dir: Union[str, Path],
    prefix: str = "",
    max_nodes: int = 15
) -> None:
    """
    Generate all sanity check visualizations for a graph.
    
    Args:
        graph: PyG Data object
        image_path: Path to original RGB image
        annotations: DataFrame with cell annotations
        output_dir: Directory to save plots
        prefix: Prefix for output filenames
        max_nodes: Max nodes for neighborhood plot
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load image
    image = np.array(Image.open(image_path).convert('RGB'))
    
    # 1. Feature matrix
    plot_feature_matrix(
        graph.x.numpy(),
        output_path=output_dir / f"{prefix}feature_matrix.png",
        title=f"Feature Matrix ({graph.x.shape[0]} nodes × {graph.x.shape[1]} features)"
    )
    plt.close()
    
    # 2. Adjacency matrix
    plot_adjacency_matrix(
        graph.edge_index,
        num_nodes=graph.x.shape[0],
        output_path=output_dir / f"{prefix}adjacency_matrix.png",
        title=f"Adjacency Matrix ({graph.edge_index.shape[1]} edges)"
    )
    plt.close()
    
    # 3. Node neighborhoods
    plot_node_neighborhoods(
        graph,
        image,
        annotations,
        output_path=output_dir / f"{prefix}node_neighborhoods.png",
        max_nodes=max_nodes
    )
    plt.close()
    
    print(f"All visualizations saved to: {output_dir}")


if __name__ == "__main__":
    print("Visualization module - import and use visualize_graph_sanity_check()")
