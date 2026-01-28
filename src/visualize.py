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
from typing import Optional, Union, Tuple, List
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

# =============================================================================
# Training Visualization Functions
# =============================================================================

def plot_training_curves(
    history: dict,
    output_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (14, 5)
) -> plt.Figure:
    """
    Plot training and validation curves (loss, accuracy, F1).
    
    Args:
        history: Dict with keys 'train_loss', 'train_acc', 'test_loss', 'test_acc', 'test_f1'
        output_path: Path to save figure
        figsize: Figure size
    
    Returns:
        matplotlib Figure
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Train', linewidth=2)
    axes[0].plot(epochs, history['test_loss'], 'r-', label='Test', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss Curves')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[1].plot(epochs, history['train_acc'], 'b-', label='Train', linewidth=2)
    axes[1].plot(epochs, history['test_acc'], 'r-', label='Test', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Accuracy Curves')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim([0, 1])
    
    # F1 Score
    if 'test_f1' in history:
        axes[2].plot(epochs, history['test_f1'], 'g-', label='Test F1 (macro)', linewidth=2)
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('F1 Score')
        axes[2].set_title('F1 Score (Macro)')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        axes[2].set_ylim([0, 1])
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved training curves to: {output_path}")
    
    return fig


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
    output_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (10, 8),
    normalize: bool = False
) -> plt.Figure:
    """
    Plot confusion matrix as a heatmap.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        class_names: Names for each class
        output_path: Path to save figure
        figsize: Figure size
        normalize: Whether to normalize by row (true labels)
    
    Returns:
        matplotlib Figure
    """
    from sklearn.metrics import confusion_matrix as sk_confusion_matrix
    
    cm = sk_confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-6)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.figure.colorbar(im, ax=ax)
    
    # Labels
    n_classes = cm.shape[0]
    if class_names is None:
        class_names = [str(i) for i in range(n_classes)]
    
    ax.set(
        xticks=np.arange(n_classes),
        yticks=np.arange(n_classes),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel='True Label',
        xlabel='Predicted Label',
        title='Confusion Matrix' + (' (Normalized)' if normalize else '')
    )
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    
    # Add text annotations
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(n_classes):
        for j in range(n_classes):
            ax.text(j, i, format(cm[i, j], fmt),
                   ha='center', va='center',
                   color='white' if cm[i, j] > thresh else 'black',
                   fontsize=8)
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved confusion matrix to: {output_path}")
    
    return fig


def plot_per_class_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
    output_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (12, 6)
) -> plt.Figure:
    """
    Plot per-class precision, recall, and F1 scores.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        class_names: Names for each class
        output_path: Path to save figure
        figsize: Figure size
    
    Returns:
        matplotlib Figure
    """
    from sklearn.metrics import precision_recall_fscore_support
    
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, zero_division=0
    )
    
    n_classes = len(precision)
    if class_names is None:
        class_names = [str(i) for i in range(n_classes)]
    
    x = np.arange(n_classes)
    width = 0.25
    
    fig, ax = plt.subplots(figsize=figsize)
    
    bars1 = ax.bar(x - width, precision, width, label='Precision', color='steelblue')
    bars2 = ax.bar(x, recall, width, label='Recall', color='darkorange')
    bars3 = ax.bar(x + width, f1, width, label='F1', color='forestgreen')
    
    ax.set_xlabel('Class')
    ax.set_ylabel('Score')
    ax.set_title('Per-Class Metrics')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim([0, 1.1])
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add support counts as text
    for i, s in enumerate(support):
        ax.text(i, 1.02, f'n={s}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved per-class metrics to: {output_path}")
    
    return fig


def plot_cross_validation_results(
    fold_results: List[dict],
    output_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """
    Plot cross-validation results across folds.
    
    Args:
        fold_results: List of dicts with 'fold' and 'best_f1' keys
        output_path: Path to save figure
        figsize: Figure size
    
    Returns:
        matplotlib Figure
    """
    folds = [r['fold'] for r in fold_results]
    f1_scores = [r['best_f1'] for r in fold_results]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    bars = ax.bar(folds, f1_scores, color='steelblue', edgecolor='black')
    
    # Add mean line
    mean_f1 = np.mean(f1_scores)
    ax.axhline(y=mean_f1, color='red', linestyle='--', linewidth=2, 
               label=f'Mean F1: {mean_f1:.4f}')
    
    # Add std band
    std_f1 = np.std(f1_scores)
    ax.axhspan(mean_f1 - std_f1, mean_f1 + std_f1, alpha=0.2, color='red',
               label=f'±1 Std: {std_f1:.4f}')
    
    ax.set_xlabel('Fold')
    ax.set_ylabel('F1 Score (Macro)')
    ax.set_title('Cross-Validation Results')
    ax.set_xticks(folds)
    ax.legend()
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, f1 in zip(bars, f1_scores):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
               f'{f1:.3f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved CV results to: {output_path}")
    
    return fig


def visualize_training_results(
    results_path: Union[str, Path],
    output_dir: Union[str, Path],
    class_names: Optional[List[str]] = None
) -> None:
    """
    Generate all training visualizations from saved results.
    
    Args:
        results_path: Path to training_results.json
        output_dir: Directory to save visualizations
        class_names: Optional class names for labels
    """
    import json
    
    results_path = Path(results_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    # Default class names
    if class_names is None:
        class_names = [
            "AMBIGUOUS", "lymphocyte", "macrophage", "nonTILnonMQ_stromal",
            "other_nucleus", "plasma_cell", "tumor_mitotic", "tumor_nonMitotic"
        ]
    
    # Plot training curves for each fold
    for fold_result in results['folds']:
        fold = fold_result['fold']
        history = fold_result['history']
        
        plot_training_curves(
            history,
            output_path=output_dir / f"training_curves_fold_{fold}.png"
        )
        plt.close()
    
    # Plot CV summary
    plot_cross_validation_results(
        results['folds'],
        output_path=output_dir / "cv_results_summary.png"
    )
    plt.close()
    
    print(f"All training visualizations saved to: {output_dir}")


def plot_predictions_on_image(
    image_path: Union[str, Path],
    predictions: np.ndarray,
    positions: np.ndarray,
    class_names: List[str],
    output_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (12, 12),
    title: str = "Predicted Cell Classes"
) -> plt.Figure:
    """
    Plot predicted cell classes overlaid on the original image.
    
    Args:
        image_path: Path to the original image
        predictions: Predicted class indices for each node
        positions: (N, 2) array of cell centroid positions
        class_names: Names for each class
        output_path: Path to save figure
        figsize: Figure size
        title: Plot title
    
    Returns:
        matplotlib Figure
    """
    image = np.array(Image.open(image_path).convert('RGB'))
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(image)
    
    # Color map for classes
    n_classes = len(class_names)
    cmap = plt.cm.get_cmap('tab10', n_classes)
    
    # Plot each cell with its predicted class color
    for i, (pos, pred) in enumerate(zip(positions, predictions)):
        color = cmap(pred)
        ax.scatter(pos[0], pos[1], c=[color], s=50, edgecolors='white', linewidth=0.5)
    
    # Create legend
    legend_elements = [
        mpatches.Patch(facecolor=cmap(i), label=class_names[i])
        for i in range(min(n_classes, len(np.unique(predictions)) + 2))
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=8)
    
    ax.set_title(title)
    ax.axis('off')
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved predictions plot to: {output_path}")
    
    return fig


if __name__ == "__main__":
    print("Visualization module - import functions as needed")
    print("Available functions:")
    print("  - visualize_graph_sanity_check()")
    print("  - plot_training_curves()")
    print("  - plot_confusion_matrix()")
    print("  - plot_per_class_metrics()")
    print("  - plot_cross_validation_results()")
    print("  - visualize_training_results()")
    print("  - plot_predictions_on_image()")
