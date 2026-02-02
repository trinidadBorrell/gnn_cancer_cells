"""
Training pipeline for GNN-based cell classification.

Supports:
- 5-fold cross-validation (folds 1-5)
- Held-out test set (fold_999)
- Multiple GNN architectures (with/without pooling)
- Regularization (dropout, weight decay, batch norm)
- Training visualization and model saving
"""

import sys
import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from sklearn.metrics import f1_score
from collections import Counter

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from feature_extractor import MorphologicalFeatureExtractor
from graph_builder import CellGraphBuilder


# =============================================================================
# GNN Models
# =============================================================================

class GNNNodeClassifier(nn.Module):
    """
    GNN for node-level classification (each cell gets a label).
    
    Architecture:
    - Multiple SAGEConv layers with dropout and batch norm
    - Skip connections (residual)
    - Classification head
    
    Args:
        input_dim: Input feature dimension
        hidden_dim: Hidden layer dimension
        num_classes: Number of output classes
        num_layers: Number of GNN layers
        dropout: Dropout probability
        use_batch_norm: Whether to use batch normalization
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_classes: int = 8,
        num_layers: int = 3,
        dropout: float = 0.3,
        use_batch_norm: bool = True
    ):
        super().__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_batch_norm = use_batch_norm
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # GNN layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        for i in range(num_layers):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
            if use_batch_norm:
                self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # Input projection
        h = self.input_proj(x)
        
        # GNN layers with residual connections
        for i in range(self.num_layers):
            h_prev = h
            h = self.convs[i](h, edge_index)
            if self.use_batch_norm:
                h = self.batch_norms[i](h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            # Residual connection
            h = h + h_prev
        
        # Classification
        return self.classifier(h)    


# =============================================================================
# Dataset Loading - Build Graphs On-the-Fly
# =============================================================================


def build_graph_from_files(
    image_path: Path,
    csv_path: Path,
    feature_extractor: MorphologicalFeatureExtractor,
    graph_builder: CellGraphBuilder
) -> Optional[Data]:
    """
    Build a single graph from image and annotation files.
    
    Args:
        image_path: Path to RGB image
        csv_path: Path to CSV annotations
        feature_extractor: Feature extractor instance
        graph_builder: Graph builder instance
    
    Returns:
        PyG Data object or None if failed
    """
    try:
        # Extract features and annotations
        features, annotations = feature_extractor.extract_from_image_and_annotations(
            image_path, csv_path
        )
        
        # Get coordinates (cell centroids)
        coords_x = (annotations['xmin'] + annotations['xmax']) / 2
        coords_y = (annotations['ymin'] + annotations['ymax']) / 2
        coords = np.stack([coords_x.values, coords_y.values], axis=1)
        
        # Get labels (use main_classification column)
        label_col = None
        for col in ['main_classification', 'label', 'class']:
            if col in annotations.columns:
                label_col = col
                break
        labels = annotations[label_col].values if label_col else None
        
        # Build graph
        graph = graph_builder.build_graph(
            features=features,
            coords=coords,
            labels=labels,
            compute_edge_attr=True
        )
        
        # Skip graphs with no valid labels
        if hasattr(graph, 'y') and graph.y.numel() > 0:
            valid_mask = graph.y >= 0
            if valid_mask.sum() > 0:
                # Store image path for visualization
                graph.image_path = str(image_path)
                return graph
        
        return None
        
    except Exception as e:
        print(f"Warning: Could not build graph from {image_path}: {e}")
        return None


def build_augmented_graph(
    image_path: Path,
    csv_path: Path,
    feature_extractor: MorphologicalFeatureExtractor,
    graph_builder: CellGraphBuilder
) -> Optional[Data]:
    """
    Build a graph from augmented image and annotations.
    
    Applies random geometric and photometric augmentations to the image
    and updates the bounding box coordinates accordingly.
    
    Args:
        image_path: Path to RGB image
        csv_path: Path to CSV annotations
        feature_extractor: Feature extractor instance
        graph_builder: Graph builder instance
    
    Returns:
        PyG Data object or None if failed
    """
    try:
        from augmentation import augment_fov
        
        # Apply augmentation to image and annotations
        aug_image, aug_annotations = augment_fov(image_path, csv_path)
        
        # Extract features from augmented image
        features = feature_extractor.extract_features(
            aug_image, aug_annotations
        )
        
        # Get coordinates (cell centroids from augmented annotations)
        coords_x = (aug_annotations['xmin'] + aug_annotations['xmax']) / 2
        coords_y = (aug_annotations['ymin'] + aug_annotations['ymax']) / 2
        coords = np.stack([coords_x.values, coords_y.values], axis=1)
        
        # Get labels (use main_classification column)
        label_col = None
        for col in ['main_classification', 'label', 'class']:
            if col in aug_annotations.columns:
                label_col = col
                break
        labels = aug_annotations[label_col].values if label_col else None
        
        # Build graph
        graph = graph_builder.build_graph(
            features=features,
            coords=coords,
            labels=labels,
            compute_edge_attr=True
        )
        
        # Skip graphs with no valid labels
        if hasattr(graph, 'y') and graph.y.numel() > 0:
            valid_mask = graph.y >= 0
            if valid_mask.sum() > 0:
                return graph
        
        return None
        
    except Exception as e:
        print(f"Warning: Could not build augmented graph from {image_path}: {e}")
        return None


def load_fold_data(
    data_dir: Path,
    fold: int,
    feature_extractor: MorphologicalFeatureExtractor,
    graph_builder: CellGraphBuilder,
    augment_train: bool = False,
    n_augmentations: int = 1
) -> Tuple[List[Data], List[Data]]:
    """
    Build train/test graphs on-the-fly for a specific fold.
    
    Args:
        data_dir: Base data directory (contains train_test_splits/, rgb/, csv/)
        fold: Fold number (1-5, or 999 for held-out)
        feature_extractor: Feature extractor instance
        graph_builder: Graph builder instance
        augment_train: If True, apply data augmentation to training data
        n_augmentations: Number of augmented versions per original sample
    
    Returns:
        train_graphs, test_graphs
    """
    splits_dir = data_dir / "train_test_splits"
    rgb_dir = data_dir / "rgb"
    csv_dir = data_dir / "csv"
    
    train_csv = splits_dir / f"fold_{fold}_train.csv"
    test_csv = splits_dir / f"fold_{fold}_test.csv"
    
    if not train_csv.exists() or not test_csv.exists():
        raise FileNotFoundError(f"Fold {fold} files not found in {splits_dir}")
    
    train_slides = pd.read_csv(train_csv)['slide_name'].tolist()
    test_slides = pd.read_csv(test_csv)['slide_name'].tolist()
    
    def build_graphs_for_slides(slide_names: List[str], apply_augmentation: bool = False) -> List[Data]:
        """Find all image/csv pairs for slides and build graphs."""
        graphs = []
        
        for slide in slide_names:
            # Find all image files for this slide
            pattern = f"{slide}*.png"
            image_files = list(rgb_dir.glob(pattern))
            
            for img_path in image_files:
                # Find corresponding CSV (same name, different extension)
                csv_path = csv_dir / (img_path.stem + ".csv")
                
                if not csv_path.exists():
                    continue
                
                # Build original graph
                graph = build_graph_from_files(
                    img_path, csv_path, feature_extractor, graph_builder
                )
                
                if graph is not None:
                    graphs.append(graph)
                    
                    # Apply augmentation if requested (only for training)
                    if apply_augmentation:
                        for _ in range(n_augmentations):
                            aug_graph = build_augmented_graph(
                                img_path, csv_path, feature_extractor, graph_builder
                            )
                            if aug_graph is not None:
                                graphs.append(aug_graph)
        
        return graphs
    
    print(f"  Building graphs for {len(train_slides)} train slides...")
    train_graphs = build_graphs_for_slides(train_slides, apply_augmentation=augment_train)
    if augment_train:
        print(f"  Built {len(train_graphs)} train graphs (with {n_augmentations}x augmentation)")
    else:
        print(f"  Built {len(train_graphs)} train graphs")
    
    # Never augment test/validation data
    print(f"  Building graphs for {len(test_slides)} test slides...")
    test_graphs = build_graphs_for_slides(test_slides, apply_augmentation=False)
    print(f"  Built {len(test_graphs)} test graphs")
    
    return train_graphs, test_graphs


def load_all_graphs_from_dir(
    data_dir: Path,
    feature_extractor: MorphologicalFeatureExtractor,
    graph_builder: CellGraphBuilder
) -> List[Data]:
    """Build graphs from all image/csv pairs in data directory."""
    rgb_dir = data_dir / "rgb"
    csv_dir = data_dir / "csv"
    
    graphs = []
    for img_path in rgb_dir.glob("*.png"):
        csv_path = csv_dir / (img_path.stem + ".csv")
        if csv_path.exists():
            graph = build_graph_from_files(
                img_path, csv_path, feature_extractor, graph_builder
            )
            if graph is not None:
                graphs.append(graph)
    
    return graphs


def save_graphs_to_files(
    graphs: List[Data],
    output_dir: Path,
    prefix: str = "graph"
) -> None:
    """
    Save graph features, edges, positions, and labels to files.
    
    Creates the following files for each graph:
    - {prefix}_{idx}_features.csv: Node features (num_nodes x feature_dim)
    - {prefix}_{idx}_edges.csv: Edge list (2 x num_edges)
    - {prefix}_{idx}_positions.csv: Node positions (num_nodes x 2)
    - {prefix}_{idx}_labels.csv: Node labels (num_nodes,)
    - {prefix}_{idx}_edge_attr.csv: Edge attributes if available
    
    Also creates a summary file with graph statistics.
    
    Args:
        graphs: List of PyG Data objects
        output_dir: Directory to save files
        prefix: Prefix for filenames (e.g., 'train', 'test')
    """
    output_dir = Path(output_dir)
    graphs_dir = output_dir / "graphs" / prefix
    graphs_dir.mkdir(parents=True, exist_ok=True)
    
    summary_data = []
    
    for idx, graph in enumerate(graphs):
        graph_prefix = f"{prefix}_{idx:04d}"
        
        # Save node features
        if hasattr(graph, 'x') and graph.x is not None:
            features_df = pd.DataFrame(
                graph.x.cpu().numpy(),
                columns=[f"feat_{i}" for i in range(graph.x.shape[1])]
            )
            features_df.to_csv(graphs_dir / f"{graph_prefix}_features.csv", index=False)
        
        # Save edge index
        if hasattr(graph, 'edge_index') and graph.edge_index is not None:
            edges_df = pd.DataFrame({
                'source': graph.edge_index[0].cpu().numpy(),
                'target': graph.edge_index[1].cpu().numpy()
            })
            edges_df.to_csv(graphs_dir / f"{graph_prefix}_edges.csv", index=False)
        
        # Save node positions
        if hasattr(graph, 'pos') and graph.pos is not None:
            pos_df = pd.DataFrame(
                graph.pos.cpu().numpy(),
                columns=['x', 'y']
            )
            pos_df.to_csv(graphs_dir / f"{graph_prefix}_positions.csv", index=False)
        
        # Save node labels
        if hasattr(graph, 'y') and graph.y is not None:
            labels_df = pd.DataFrame({'label': graph.y.cpu().numpy()})
            labels_df.to_csv(graphs_dir / f"{graph_prefix}_labels.csv", index=False)
        
        # Save edge attributes if available
        if hasattr(graph, 'edge_attr') and graph.edge_attr is not None:
            edge_attr_df = pd.DataFrame(
                graph.edge_attr.cpu().numpy(),
                columns=[f"edge_feat_{i}" for i in range(graph.edge_attr.shape[1])]
            )
            edge_attr_df.to_csv(graphs_dir / f"{graph_prefix}_edge_attr.csv", index=False)
        
        # Collect summary statistics
        num_nodes = graph.x.shape[0] if hasattr(graph, 'x') and graph.x is not None else 0
        num_edges = graph.edge_index.shape[1] if hasattr(graph, 'edge_index') and graph.edge_index is not None else 0
        num_features = graph.x.shape[1] if hasattr(graph, 'x') and graph.x is not None else 0
        
        summary_data.append({
            'graph_id': idx,
            'num_nodes': num_nodes,
            'num_edges': num_edges,
            'num_features': num_features,
            'avg_degree': num_edges / num_nodes if num_nodes > 0 else 0
        })
    
    # Save summary
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(graphs_dir / f"{prefix}_summary.csv", index=False)
    
    print(f"Saved {len(graphs)} graphs to {graphs_dir}")
    print(f"  Total nodes: {summary_df['num_nodes'].sum()}")
    print(f"  Total edges: {summary_df['num_edges'].sum()}")
    print(f"  Avg nodes per graph: {summary_df['num_nodes'].mean():.1f}")
    print(f"  Avg edges per graph: {summary_df['num_edges'].mean():.1f}")


def visualize_random_graphs(
    graphs: List[Data],
    output_dir: Path,
    n_samples: int = 10,
    prefix: str = "sample"
) -> None:
    """
    Visualize and save random sample of graphs for inspection.
    
    Creates plots showing:
    - Original image with graph overlay (nodes and edges)
    - Graph-only view with nodes colored by label
    - Label distribution histogram
    
    Args:
        graphs: List of PyG Data objects (with image_path attribute)
        output_dir: Directory to save visualizations
        n_samples: Number of random graphs to visualize
        prefix: Prefix for filenames
    """
    import matplotlib.pyplot as plt
    from PIL import Image
    import random as rand
    
    output_dir = Path(output_dir)
    vis_dir = output_dir / "graph_visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)
    
    # Sample random graphs
    n_samples = min(n_samples, len(graphs))
    sample_indices = rand.sample(range(len(graphs)), n_samples)
    
    # Class names for legend
    class_names = [
        "AMBIGUOUS", "lymphocyte", "macrophage", "nonTILnonMQ_stromal",
        "other_nucleus", "plasma_cell", "tumor_mitotic", "tumor_nonMitotic"
    ]
    
    # Color map - use distinct colors for each class
    class_colors = [
        '#808080',  # AMBIGUOUS - gray
        '#2ecc71',  # lymphocyte - green
        '#e74c3c',  # macrophage - red
        '#3498db',  # nonTILnonMQ_stromal - blue
        '#9b59b6',  # other_nucleus - purple
        '#f39c12',  # plasma_cell - orange
        '#e91e63',  # tumor_mitotic - pink
        '#1abc9c',  # tumor_nonMitotic - teal
    ]
    
    for i, idx in enumerate(sample_indices):
        graph = graphs[idx]
        
        # Get data
        pos = graph.pos.cpu().numpy() if hasattr(graph, 'pos') and graph.pos is not None else None
        labels = graph.y.cpu().numpy() if hasattr(graph, 'y') and graph.y is not None else None
        edge_index = graph.edge_index.cpu().numpy() if hasattr(graph, 'edge_index') else None
        image_path = graph.image_path if hasattr(graph, 'image_path') else None
        
        if pos is None:
            continue
        
        # Create figure with 3 subplots: image+graph, graph-only, histogram
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        
        # Plot 1: Original image with graph overlay
        ax1 = axes[0]
        if image_path and Path(image_path).exists():
            img = np.array(Image.open(image_path).convert("RGB"))
            ax1.imshow(img)
            
            # Draw edges on image
            if edge_index is not None:
                for j in range(edge_index.shape[1]):
                    src, tgt = edge_index[0, j], edge_index[1, j]
                    ax1.plot([pos[src, 0], pos[tgt, 0]], [pos[src, 1], pos[tgt, 1]], 
                            'w-', alpha=0.4, linewidth=1.0)
            
            # Draw nodes colored by label
            if labels is not None:
                for label_idx in range(8):
                    mask = labels == label_idx
                    if mask.sum() > 0:
                        ax1.scatter(pos[mask, 0], pos[mask, 1], 
                                   c=class_colors[label_idx], s=40, 
                                   edgecolors='white', linewidth=0.8,
                                   label=class_names[label_idx], alpha=0.9)
            else:
                ax1.scatter(pos[:, 0], pos[:, 1], s=40, c='yellow', 
                           edgecolors='white', linewidth=0.8, alpha=0.9)
            
            ax1.set_title(f'Image + Graph Overlay\n{Path(image_path).stem[:50]}...', fontsize=10)
        else:
            ax1.text(0.5, 0.5, 'Image not available', ha='center', va='center', transform=ax1.transAxes)
            ax1.set_title('Image + Graph Overlay')
        
        ax1.axis('off')
        
        # Plot 2: Graph-only view with nodes and edges
        ax2 = axes[1]
        if edge_index is not None:
            for j in range(edge_index.shape[1]):
                src, tgt = edge_index[0, j], edge_index[1, j]
                ax2.plot([pos[src, 0], pos[tgt, 0]], [pos[src, 1], pos[tgt, 1]], 
                        'k-', alpha=0.15, linewidth=0.8)
        
        if labels is not None:
            for label_idx in range(8):
                mask = labels == label_idx
                if mask.sum() > 0:
                    ax2.scatter(pos[mask, 0], pos[mask, 1], 
                               c=class_colors[label_idx], s=60, 
                               edgecolors='black', linewidth=0.5,
                               label=class_names[label_idx])
            ax2.legend(loc='upper right', fontsize=7, ncol=2)
        else:
            ax2.scatter(pos[:, 0], pos[:, 1], s=60, c='steelblue', edgecolors='black', linewidth=0.5)
        
        ax2.set_title(f'Graph Structure\n{graph.x.shape[0]} nodes, {edge_index.shape[1] if edge_index is not None else 0} edges')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.invert_yaxis()
        ax2.set_aspect('equal')
        
        # Plot 3: Label distribution histogram
        ax3 = axes[2]
        if labels is not None:
            unique, counts = np.unique(labels[labels >= 0], return_counts=True)
            colors = [class_colors[int(u)] for u in unique]
            bars = ax3.bar([class_names[int(u)] for u in unique], counts, color=colors, edgecolor='black')
            ax3.set_xlabel('Class')
            ax3.set_ylabel('Count')
            ax3.set_title('Label Distribution')
            ax3.tick_params(axis='x', rotation=45)
            
            # Add count labels on bars
            for bar, count in zip(bars, counts):
                ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                        str(count), ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(vis_dir / f"{prefix}_graph_{idx:04d}.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"Saved {n_samples} graph visualizations to {vis_dir}")


# =============================================================================
# Stratified Fold Creation
# =============================================================================

def get_slide_class_distribution(
    data_dir: Path,
    slide_names: List[str]
) -> Dict[str, Counter]:
    """
    Get class distribution for each slide based on main_classification.
    
    Args:
        data_dir: Base data directory containing csv/
        slide_names: List of slide names
    
    Returns:
        Dictionary mapping slide_name -> Counter of main_classification
    """
    csv_dir = data_dir / "csv"
    slide_distributions = {}
    
    for slide in slide_names:
        pattern = f"{slide}*.csv"
        csv_files = list(csv_dir.glob(pattern))
        
        class_counts = Counter()
        for csv_path in csv_files:
            try:
                df = pd.read_csv(csv_path)
                if 'main_classification' in df.columns:
                    class_counts.update(df['main_classification'].values)
            except Exception:
                continue
        
        slide_distributions[slide] = class_counts
    
    return slide_distributions


def get_slide_dominant_class(slide_distributions: Dict[str, Counter]) -> Dict[str, str]:
    """
    Get the dominant (most common) class for each slide.
    
    Args:
        slide_distributions: Dictionary mapping slide_name -> Counter of classes
    
    Returns:
        Dictionary mapping slide_name -> dominant class
    """
    dominant_classes = {}
    for slide, counts in slide_distributions.items():
        if counts:
            dominant_classes[slide] = counts.most_common(1)[0][0]
        else:
            dominant_classes[slide] = "unknown"
    return dominant_classes


def create_stratified_folds(
    data_dir: Path,
    n_folds: int = 5,
    random_state: int = 42
) -> Dict[int, Tuple[List[str], List[str]]]:
    """
    Create multi-label stratified folds based on main_classification distribution.
    
    Uses iterative stratification to ensure all 8 classes have similar proportions
    across train/test splits in each fold.
    
    Args:
        data_dir: Base data directory containing csv/
        n_folds: Number of folds to create
        random_state: Random seed for reproducibility
    
    Returns:
        Dictionary mapping fold_number -> (train_slides, test_slides)
    """
    csv_dir = data_dir / "csv"
    
    # Get all unique slide names from CSV files
    all_csv_files = list(csv_dir.glob("*.csv"))
    slide_names = set()
    for csv_path in all_csv_files:
        # Extract slide name (everything before _id-)
        name = csv_path.stem
        if "_id-" in name:
            slide_name = name.split("_id-")[0]
        else:
            slide_name = name
        slide_names.add(slide_name)
    
    slide_names = sorted(list(slide_names))
    print(f"Found {len(slide_names)} unique slides")
    
    # Get class distribution for each slide
    slide_distributions = get_slide_class_distribution(data_dir, slide_names)
    
    # Define all 8 classes in fixed order
    all_classes = [
        "AMBIGUOUS", "lymphocyte", "macrophage", "nonTILnonMQ_stromal",
        "other_nucleus", "plasma_cell", "tumor_mitotic", "tumor_nonMitotic"
    ]
    
    # Create multi-label matrix: each slide has proportion of each class
    # This will be used for iterative stratification
    n_slides = len(slide_names)
    n_classes = len(all_classes)
    
    # Create proportion matrix for each slide
    proportion_matrix = np.zeros((n_slides, n_classes))
    for i, slide in enumerate(slide_names):
        counts = slide_distributions[slide]
        total = sum(counts.values())
        if total > 0:
            for j, cls in enumerate(all_classes):
                proportion_matrix[i, j] = counts.get(cls, 0) / total
    
    # Convert to binary multi-label (slide has class if proportion > threshold)
    # Use a low threshold to capture presence of minority classes
    threshold = 0.01  # 1% threshold
    multilabel_matrix = (proportion_matrix > threshold).astype(int)
    
    # Ensure each slide has at least one label (use dominant class)
    for i in range(n_slides):
        if multilabel_matrix[i].sum() == 0:
            dominant_idx = proportion_matrix[i].argmax()
            multilabel_matrix[i, dominant_idx] = 1
    
    # Use iterative stratification for multi-label
    np.random.seed(random_state)
    
    # Implement iterative stratification manually
    # Sort slides by rarest class combination for better distribution
    folds = _iterative_stratification(slide_names, multilabel_matrix, n_folds, random_state)
    
    return folds


def _iterative_stratification(
    slide_names: List[str],
    multilabel_matrix: np.ndarray,
    n_folds: int,
    random_state: int
) -> Dict[int, Tuple[List[str], List[str]]]:
    """
    Perform iterative stratification for multi-label data.
    
    This algorithm iteratively assigns samples to folds while trying to
    maintain the label distribution across folds.
    """
    np.random.seed(random_state)
    n_samples = len(slide_names)
    n_labels = multilabel_matrix.shape[1]
    
    # Initialize fold assignments (-1 means unassigned)
    fold_assignments = np.full(n_samples, -1, dtype=int)
    
    # Calculate desired samples per fold
    samples_per_fold = n_samples // n_folds
    remainder = n_samples % n_folds
    fold_sizes = [samples_per_fold + (1 if i < remainder else 0) for i in range(n_folds)]
    
    # Track current counts per fold per label
    fold_label_counts = np.zeros((n_folds, n_labels))
    fold_sample_counts = np.zeros(n_folds)
    
    # Calculate total label counts
    total_label_counts = multilabel_matrix.sum(axis=0)
    
    # Process labels from rarest to most common
    label_order = np.argsort(total_label_counts)
    
    # Get indices of samples sorted by number of labels (fewest first)
    sample_label_counts = multilabel_matrix.sum(axis=1)
    sample_order = np.argsort(sample_label_counts)
    
    # Shuffle within same label count groups for randomness
    shuffled_order = []
    for count in range(int(sample_label_counts.max()) + 1):
        indices = sample_order[sample_label_counts[sample_order] == count]
        np.random.shuffle(indices)
        shuffled_order.extend(indices)
    sample_order = np.array(shuffled_order)
    
    # Assign each sample to the fold that needs it most
    for sample_idx in sample_order:
        sample_labels = multilabel_matrix[sample_idx]
        
        # Calculate score for each fold (lower is better)
        scores = np.zeros(n_folds)
        for fold_idx in range(n_folds):
            # Penalize folds that are already full
            if fold_sample_counts[fold_idx] >= fold_sizes[fold_idx]:
                scores[fold_idx] = np.inf
                continue
            
            # Calculate imbalance score based on label distribution
            for label_idx in label_order:
                if sample_labels[label_idx] == 1:
                    # Desired proportion for this label in this fold
                    desired = total_label_counts[label_idx] * fold_sizes[fold_idx] / n_samples
                    current = fold_label_counts[fold_idx, label_idx]
                    # Prefer folds that are under-represented for this label
                    scores[fold_idx] -= (desired - current)
        
        # Assign to fold with lowest score
        best_fold = np.argmin(scores)
        fold_assignments[sample_idx] = best_fold
        fold_sample_counts[best_fold] += 1
        fold_label_counts[best_fold] += sample_labels
    
    # Convert to train/test splits
    folds = {}
    slides_array = np.array(slide_names)
    for fold_idx in range(n_folds):
        test_mask = fold_assignments == fold_idx
        train_mask = ~test_mask
        train_slides = slides_array[train_mask].tolist()
        test_slides = slides_array[test_mask].tolist()
        folds[fold_idx + 1] = (train_slides, test_slides)
    
    return folds


def print_data_distribution(
    data_dir: Path,
    folds: Optional[Dict[int, Tuple[List[str], List[str]]]] = None
) -> None:
    """
    Print the general data distribution and per-fold distribution.
    
    Args:
        data_dir: Base data directory containing csv/
        folds: Optional dictionary of folds (fold_number -> (train_slides, test_slides))
               If None, loads from existing fold files
    """
    csv_dir = data_dir / "csv"
    splits_dir = data_dir / "train_test_splits"
    
    print("\n" + "="*70)
    print("DATA DISTRIBUTION ANALYSIS")
    print("="*70)
    
    # Get all slides and their distributions
    all_csv_files = list(csv_dir.glob("*.csv"))
    slide_names = set()
    for csv_path in all_csv_files:
        name = csv_path.stem
        if "_id-" in name:
            slide_name = name.split("_id-")[0]
        else:
            slide_name = name
        slide_names.add(slide_name)
    
    slide_names = sorted(list(slide_names))
    slide_distributions = get_slide_class_distribution(data_dir, slide_names)
    
    # Compute overall distribution
    overall_counts = Counter()
    for counts in slide_distributions.values():
        overall_counts.update(counts)
    
    total_cells = sum(overall_counts.values())
    
    print(f"\n{'='*40}")
    print("OVERALL DATA DISTRIBUTION")
    print(f"{'='*40}")
    print(f"Total slides: {len(slide_names)}")
    print(f"Total cells: {total_cells}")
    print("\nClass distribution:")
    print(f"{'Class':<25} {'Count':>10} {'Percentage':>12}")
    print("-" * 50)
    for cls, count in sorted(overall_counts.items(), key=lambda x: -x[1]):
        pct = 100 * count / total_cells if total_cells > 0 else 0
        print(f"{cls:<25} {count:>10} {pct:>11.2f}%")
    
    # Load or use provided folds
    if folds is None:
        folds = {}
        for fold_num in range(1, 6):
            train_csv = splits_dir / f"fold_{fold_num}_train.csv"
            test_csv = splits_dir / f"fold_{fold_num}_test.csv"
            if train_csv.exists() and test_csv.exists():
                train_slides = pd.read_csv(train_csv)['slide_name'].tolist()
                test_slides = pd.read_csv(test_csv)['slide_name'].tolist()
                folds[fold_num] = (train_slides, test_slides)
    
    if not folds:
        print("\nNo folds found to analyze.")
        return
    
    # Print per-fold distribution
    print(f"\n{'='*40}")
    print("PER-FOLD DISTRIBUTION")
    print(f"{'='*40}")
    
    for fold_num in sorted(folds.keys()):
        train_slides, test_slides = folds[fold_num]
        
        # Get distributions for train and test
        train_counts = Counter()
        test_counts = Counter()
        
        for slide in train_slides:
            if slide in slide_distributions:
                train_counts.update(slide_distributions[slide])
        
        for slide in test_slides:
            if slide in slide_distributions:
                test_counts.update(slide_distributions[slide])
        
        train_total = sum(train_counts.values())
        test_total = sum(test_counts.values())
        
        print(f"\n--- Fold {fold_num} ---")
        print(f"Train: {len(train_slides)} slides, {train_total} cells")
        print(f"Test:  {len(test_slides)} slides, {test_total} cells")
        
        # Get all classes
        all_classes = sorted(set(train_counts.keys()) | set(test_counts.keys()))
        
        print(f"\n{'Class':<25} {'Train %':>10} {'Test %':>10} {'Diff':>10}")
        print("-" * 58)
        for cls in all_classes:
            train_pct = 100 * train_counts.get(cls, 0) / train_total if train_total > 0 else 0
            test_pct = 100 * test_counts.get(cls, 0) / test_total if test_total > 0 else 0
            diff = abs(train_pct - test_pct)
            print(f"{cls:<25} {train_pct:>9.2f}% {test_pct:>9.2f}% {diff:>9.2f}%")


def save_stratified_folds(
    data_dir: Path,
    folds: Dict[int, Tuple[List[str], List[str]]]
) -> None:
    """
    Save stratified folds to CSV files.
    
    Args:
        data_dir: Base data directory
        folds: Dictionary mapping fold_number -> (train_slides, test_slides)
    """
    splits_dir = data_dir / "train_test_splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    
    for fold_num, (train_slides, test_slides) in folds.items():
        # Create train DataFrame
        train_df = pd.DataFrame({
            'slide_name': train_slides,
            'type': 'train',
            'fold': fold_num
        })
        train_df.to_csv(splits_dir / f"fold_{fold_num}_train.csv", index=True)
        
        # Create test DataFrame
        test_df = pd.DataFrame({
            'slide_name': test_slides,
            'type': 'test',
            'fold': fold_num
        })
        test_df.to_csv(splits_dir / f"fold_{fold_num}_test.csv", index=True)
    
    print(f"\nSaved {len(folds)} stratified folds to {splits_dir}")


def plot_fold_distribution(
    data_dir: Path,
    output_dir: Path,
    folds: Optional[Dict[int, Tuple[List[str], List[str]]]] = None
) -> None:
    """
    Create and save a histogram showing class distribution across all folds.
    
    Args:
        data_dir: Base data directory containing csv/
        output_dir: Directory to save the plot
        folds: Optional dictionary of folds. If None, loads from existing files.
    """
    import matplotlib.pyplot as plt
    
    csv_dir = data_dir / "csv"
    splits_dir = data_dir / "train_test_splits"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all slides and their distributions
    all_csv_files = list(csv_dir.glob("*.csv"))
    slide_names = set()
    for csv_path in all_csv_files:
        name = csv_path.stem
        if "_id-" in name:
            slide_name = name.split("_id-")[0]
        else:
            slide_name = name
        slide_names.add(slide_name)
    
    slide_names = sorted(list(slide_names))
    slide_distributions = get_slide_class_distribution(data_dir, slide_names)
    
    # Load or use provided folds
    if folds is None:
        folds = {}
        for fold_num in range(1, 6):
            train_csv = splits_dir / f"fold_{fold_num}_train.csv"
            test_csv = splits_dir / f"fold_{fold_num}_test.csv"
            if train_csv.exists() and test_csv.exists():
                train_slides = pd.read_csv(train_csv)['slide_name'].tolist()
                test_slides = pd.read_csv(test_csv)['slide_name'].tolist()
                folds[fold_num] = (train_slides, test_slides)
    
    if not folds:
        print("No folds found to plot.")
        return
    
    # Define all 8 classes
    all_classes = [
        "AMBIGUOUS", "lymphocyte", "macrophage", "nonTILnonMQ_stromal",
        "other_nucleus", "plasma_cell", "tumor_mitotic", "tumor_nonMitotic"
    ]
    
    # Compute overall distribution
    overall_counts = Counter()
    for counts in slide_distributions.values():
        overall_counts.update(counts)
    total_cells = sum(overall_counts.values())
    overall_pct = {cls: 100 * overall_counts.get(cls, 0) / total_cells for cls in all_classes}
    
    # Compute per-fold train and test distributions
    fold_data = []
    for fold_num in sorted(folds.keys()):
        train_slides, test_slides = folds[fold_num]
        
        train_counts = Counter()
        test_counts = Counter()
        
        for slide in train_slides:
            if slide in slide_distributions:
                train_counts.update(slide_distributions[slide])
        
        for slide in test_slides:
            if slide in slide_distributions:
                test_counts.update(slide_distributions[slide])
        
        train_total = sum(train_counts.values())
        test_total = sum(test_counts.values())
        
        for cls in all_classes:
            train_pct = 100 * train_counts.get(cls, 0) / train_total if train_total > 0 else 0
            test_pct = 100 * test_counts.get(cls, 0) / test_total if test_total > 0 else 0
            fold_data.append({
                'Fold': f'Fold {fold_num}',
                'Class': cls,
                'Train %': train_pct,
                'Test %': test_pct,
                'Overall %': overall_pct[cls]
            })
    
    df_folds = pd.DataFrame(fold_data)
    
    # Create figure with subplots for each class
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    colors = {'Train': '#2ecc71', 'Test': '#e74c3c', 'Overall': '#3498db'}
    
    for idx, cls in enumerate(all_classes):
        ax = axes[idx]
        cls_data = df_folds[df_folds['Class'] == cls]
        
        x = np.arange(len(folds))
        width = 0.25
        
        train_vals = cls_data['Train %'].values
        test_vals = cls_data['Test %'].values
        overall_val = cls_data['Overall %'].values[0]
        
        ax.bar(x - width, train_vals, width, label='Train', color=colors['Train'], alpha=0.8)
        ax.bar(x, test_vals, width, label='Test', color=colors['Test'], alpha=0.8)
        
        # Add overall reference line
        ax.axhline(y=overall_val, color=colors['Overall'], linestyle='--', linewidth=2, label=f'Overall ({overall_val:.1f}%)')
        
        ax.set_xlabel('Fold')
        ax.set_ylabel('Percentage (%)')
        ax.set_title(f'{cls}', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([f'{i+1}' for i in range(len(folds))])
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Set y-axis to start from 0
        ax.set_ylim(bottom=0)
    
    plt.suptitle('Class Distribution Across Folds (Multi-label Stratified)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    plot_path = output_dir / "fold_distribution_histogram.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nSaved fold distribution histogram to {plot_path}")


# =============================================================================
# Training Functions
# =============================================================================

def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    class_weights: Optional[torch.Tensor] = None
) -> Tuple[float, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_nodes = 0
    
    for data in loader:
        data = data.to(device)
        
        # Skip empty graphs
        if not hasattr(data, 'y') or data.y.numel() == 0:
            continue
        
        # Filter valid labels
        valid_mask = data.y >= 0
        if valid_mask.sum() == 0:
            continue
        
        optimizer.zero_grad()
        
        # Forward pass
        logits = model(data.x, data.edge_index)
        
        # Loss only on valid labels
        loss = F.cross_entropy(
            logits[valid_mask], 
            data.y[valid_mask],
            weight=class_weights
        )
        
        loss.backward()
        optimizer.step()
        
        # Metrics
        total_loss += loss.item() * valid_mask.sum().item()
        pred = logits[valid_mask].argmax(dim=-1)
        total_correct += (pred == data.y[valid_mask]).sum().item()
        total_nodes += valid_mask.sum().item()
    
    avg_loss = total_loss / max(1, total_nodes)
    accuracy = total_correct / max(1, total_nodes)
    
    return avg_loss, accuracy


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    class_weights: Optional[torch.Tensor] = None
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """Evaluate model on a dataset."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_nodes = 0
    
    all_preds = []
    all_labels = []
    
    for data in loader:
        data = data.to(device)
        
        if not hasattr(data, 'y') or data.y.numel() == 0:
            continue
        
        valid_mask = data.y >= 0
        if valid_mask.sum() == 0:
            continue
        
        logits = model(data.x, data.edge_index)
        
        loss = F.cross_entropy(
            logits[valid_mask],
            data.y[valid_mask],
            weight=class_weights
        )
        
        total_loss += loss.item() * valid_mask.sum().item()
        pred = logits[valid_mask].argmax(dim=-1)
        total_correct += (pred == data.y[valid_mask]).sum().item()
        total_nodes += valid_mask.sum().item()
        
        all_preds.append(pred.cpu().numpy())
        all_labels.append(data.y[valid_mask].cpu().numpy())
    
    avg_loss = total_loss / max(1, total_nodes)
    accuracy = total_correct / max(1, total_nodes)
    
    all_preds = np.concatenate(all_preds) if all_preds else np.array([])
    all_labels = np.concatenate(all_labels) if all_labels else np.array([])
    
    return avg_loss, accuracy, all_preds, all_labels


def compute_class_weights(graphs: List[Data], num_classes: int) -> torch.Tensor:
    """Compute inverse frequency class weights for imbalanced data."""
    counts = np.zeros(num_classes)
    for g in graphs:
        if hasattr(g, 'y'):
            labels = g.y.numpy()
            valid = labels[labels >= 0]
            for c in range(num_classes):
                counts[c] += (valid == c).sum()
    
    # Inverse frequency with smoothing
    weights = 1.0 / (counts + 1e-6)
    weights = weights / weights.sum() * num_classes  # Normalize
    
    return torch.tensor(weights, dtype=torch.float32)


# =============================================================================
# Main Training Pipeline - Single Model Across All Folds
# =============================================================================

def train_single_model(
    config: Dict,
    data_dir: Path,
    output_dir: Path,
    save_graphs: bool = False,
    augment_train: bool = False,
    n_augmentations: int = 1,
    visualize_graphs: bool = False
) -> Dict:
    """
    Train a SINGLE model using all folds (1-5) combined.
    
    - Training data: all train splits from folds 1-5
    - Validation data: all test splits from folds 1-5
    - Final test: held-out fold 999
    
    Args:
        config: Training configuration dictionary
        data_dir: Path to data directory
        output_dir: Path to output directory
        save_graphs: If True, save graph features and edges to CSV files
        augment_train: If True, apply data augmentation to training data
        n_augmentations: Number of augmented versions per original sample
        visualize_graphs: If True, save visualizations of 10 random graphs
    """
    import matplotlib.pyplot as plt
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir = output_dir / "models"
    models_dir.mkdir(exist_ok=True)
    vis_dir = output_dir / "visualizations"
    vis_dir.mkdir(exist_ok=True)
    
    # Initialize feature extractor and graph builder
    feature_extractor = MorphologicalFeatureExtractor(n_bins=16, max_cell_size=256)
    graph_builder = CellGraphBuilder(edge_strategy='knn', k=5)
    
    # Collect ALL training and validation graphs from folds 1-5
    all_train_graphs = []
    all_val_graphs = []
    
    print("\n" + "="*60)
    print("LOADING DATA FROM ALL FOLDS")
    print("="*60)
    
    for fold in config['folds']:
        try:
            train_graphs, val_graphs = load_fold_data(
                data_dir, fold, feature_extractor, graph_builder,
                augment_train=augment_train, n_augmentations=n_augmentations
            )
            all_train_graphs.extend(train_graphs)
            all_val_graphs.extend(val_graphs)
            print(f"  Fold {fold}: {len(train_graphs)} train, {len(val_graphs)} val graphs")
        except FileNotFoundError as e:
            print(f"  Fold {fold}: not found - {e}")
            continue
    
    if len(all_train_graphs) == 0:
        raise ValueError("No training graphs found!")
    
    print(f"\nTotal: {len(all_train_graphs)} train graphs, {len(all_val_graphs)} val graphs")
    
    # Visualize random graphs if requested
    if visualize_graphs:
        print("\n" + "="*60)
        print("VISUALIZING RANDOM GRAPHS")
        print("="*60)
        visualize_random_graphs(all_train_graphs, output_dir, n_samples=10, prefix="train")
    
    # Remove image_path from graphs (not compatible with DataLoader batching)
    for g in all_train_graphs:
        if hasattr(g, 'image_path'):
            del g.image_path
    for g in all_val_graphs:
        if hasattr(g, 'image_path'):
            del g.image_path
    
    # Save graphs if requested
    if save_graphs:
        print("\n" + "="*60)
        print("SAVING GRAPH DATA")
        print("="*60)
        save_graphs_to_files(all_train_graphs, output_dir, prefix="train")
        save_graphs_to_files(all_val_graphs, output_dir, prefix="val")
    
    # Create data loaders
    train_loader = DataLoader(all_train_graphs, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(all_val_graphs, batch_size=config['batch_size'], shuffle=False)
    
    # Get input dimension
    input_dim = all_train_graphs[0].x.shape[1]
    
    # Initialize model
    print("\n" + "="*60)
    print("TRAINING SINGLE MODEL")
    print("="*60)
    
    model = GNNNodeClassifier(
        input_dim=input_dim,
        hidden_dim=config['hidden_dim'],
        num_classes=config['num_classes'],
        num_layers=config['num_layers'],
        dropout=config['dropout'],
        use_batch_norm=config['use_batch_norm']
    ).to(device)
    
    print(f"Model: {sum(p.numel() for p in model.parameters())} parameters")
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    # Scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )
    
    # Class weights
    class_weights = None
    if config['use_class_weights']:
        class_weights = compute_class_weights(all_train_graphs, config['num_classes']).to(device)
        print(f"Class weights: {class_weights.cpu().numpy()}")
    
    # Training history
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [], 'val_f1': []
    }
    
    best_f1 = 0.0
    best_model_state = None
    
    for epoch in range(1, config['epochs'] + 1):
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, device, class_weights
        )
        
        # Validate
        val_loss, val_acc, val_preds, val_labels = evaluate(
            model, val_loader, device, class_weights
        )
        
        val_f1 = f1_score(val_labels, val_preds, average='macro', zero_division=0)
        
        # Update scheduler
        scheduler.step(val_loss)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        
        # Save best model
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_model_state = model.state_dict().copy()
        
        # Print progress
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d} | "
                  f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.3f} | "
                  f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.3f}, F1: {val_f1:.3f}")
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # Save model
    model_path = models_dir / "model_final.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'best_val_f1': best_f1
    }, model_path)
    print(f"\nSaved model to {model_path}")
    
    # Plot and save training curves
    print("\nGenerating training visualizations...")
    
    # Plot training curves
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Train', linewidth=2)
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Val', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss Curves')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[1].plot(epochs, history['train_acc'], 'b-', label='Train', linewidth=2)
    axes[1].plot(epochs, history['val_acc'], 'r-', label='Val', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Accuracy Curves')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # F1
    axes[2].plot(epochs, history['val_f1'], 'g-', label='Val F1', linewidth=2)
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('F1 (macro)')
    axes[2].set_title('Validation F1')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(vis_dir / "training_curves.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved training curves to {vis_dir / 'training_curves.png'}")
    
    return {
        'history': history,
        'best_val_f1': best_f1,
        'model': model,
        'device': device,
        'class_weights': class_weights
    }


def evaluate_held_out(
    config: Dict,
    data_dir: Path,
    output_dir: Path,
    model: Optional[nn.Module] = None,
    device: Optional[torch.device] = None
) -> Dict:
    """
    Evaluate on held-out test set (fold_999).
    
    Saves:
    - Confusion matrix (normalized)
    - Per-class metrics
    - Example graph predictions
    """
    import matplotlib.pyplot as plt
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    models_dir = output_dir / "models"
    vis_dir = output_dir / "visualizations"
    vis_dir.mkdir(exist_ok=True)
    examples_dir = output_dir / "examples"
    examples_dir.mkdir(exist_ok=True)
    
    # Initialize feature extractor and graph builder
    feature_extractor = MorphologicalFeatureExtractor(n_bins=16, max_cell_size=256)
    graph_builder = CellGraphBuilder(edge_strategy='knn', k=5)
    
    # Load held-out data
    print("\n" + "="*60)
    print("HELD-OUT TEST EVALUATION (fold 999)")
    print("="*60)
    
    try:
        _, held_out_graphs = load_fold_data(
            data_dir, 999, feature_extractor, graph_builder
        )
    except FileNotFoundError:
        print("Held-out fold (999) not found")
        return {}
    
    if len(held_out_graphs) == 0:
        print("No graphs found for held-out set")
        return {}
    
    print(f"Loaded {len(held_out_graphs)} test graphs")
    
    # Load model if not provided
    if model is None:
        model_path = models_dir / "model_final.pt"
        if not model_path.exists():
            print(f"Model not found: {model_path}")
            return {}
        
        checkpoint = torch.load(model_path, map_location=device)
        input_dim = held_out_graphs[0].x.shape[1]
        
        model = GNNNodeClassifier(
            input_dim=input_dim,
            hidden_dim=config['hidden_dim'],
            num_classes=config['num_classes'],
            num_layers=config['num_layers'],
            dropout=config['dropout'],
            use_batch_norm=config['use_batch_norm']
        ).to(device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
    
    # Evaluate
    loader = DataLoader(held_out_graphs, batch_size=config['batch_size'], shuffle=False)
    _, acc, all_preds, all_labels = evaluate(model, loader, device)
    
    test_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    
    print(f"\nTest Accuracy: {acc:.4f}")
    print(f"Test F1 (macro): {test_f1:.4f}")
    
    # Class names
    class_names = [
        "AMBIGUOUS", "lymphocyte", "macrophage", "nonTILnonMQ_stromal",
        "other_nucleus", "plasma_cell", "tumor_mitotic", "tumor_nonMitotic"
    ]
    
    # Plot confusion matrix (normalized)
    print("\nGenerating held-out visualizations...")
    
    from sklearn.metrics import confusion_matrix as sk_cm
    cm = sk_cm(all_labels, all_preds)
    cm_norm = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-6)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm_norm, interpolation='nearest', cmap='Blues')
    ax.figure.colorbar(im, ax=ax)
    
    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel='True Label',
        xlabel='Predicted Label',
        title=f'Confusion Matrix (Normalized) - Test F1: {test_f1:.3f}'
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    # Add values
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, f'{cm_norm[i, j]:.2f}',
                   ha='center', va='center',
                   color='white' if cm_norm[i, j] > 0.5 else 'black', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(vis_dir / "confusion_matrix_test.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved confusion matrix to {vis_dir / 'confusion_matrix_test.png'}")
    
    # Plot per-class metrics
    from sklearn.metrics import precision_recall_fscore_support
    precision, recall, f1_per_class, support = precision_recall_fscore_support(
        all_labels, all_preds, zero_division=0
    )
    
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(class_names))
    width = 0.25
    
    ax.bar(x - width, precision, width, label='Precision', color='steelblue')
    ax.bar(x, recall, width, label='Recall', color='darkorange')
    ax.bar(x + width, f1_per_class, width, label='F1', color='forestgreen')
    
    ax.set_xlabel('Class')
    ax.set_ylabel('Score')
    ax.set_title(f'Per-Class Metrics (Test Set) - Macro F1: {test_f1:.3f}')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim([0, 1.1])
    ax.grid(True, alpha=0.3, axis='y')
    
    for i, s in enumerate(support):
        ax.text(i, 1.02, f'n={s}', ha='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(vis_dir / "per_class_metrics_test.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved per-class metrics to {vis_dir / 'per_class_metrics_test.png'}")
    
    # Save example predictions
    print("\nSaving example predictions...")
    model.eval()
    
    num_examples = min(5, len(held_out_graphs))
    for i in range(num_examples):
        graph = held_out_graphs[i].to(device)
        
        with torch.no_grad():
            logits = model(graph.x, graph.edge_index)
            preds = logits.argmax(dim=1).cpu().numpy()
        
        labels = graph.y.cpu().numpy()
        pos = graph.pos.cpu().numpy()
        
        # Plot prediction vs ground truth
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Ground truth
        scatter_gt = axes[0].scatter(pos[:, 0], pos[:, 1], c=labels, cmap='tab10', 
                                      s=30, vmin=0, vmax=7)
        axes[0].set_title(f'Ground Truth (Graph {i+1})')
        axes[0].set_xlabel('X')
        axes[0].set_ylabel('Y')
        axes[0].invert_yaxis()
        
        # Predictions
        scatter_pred = axes[1].scatter(pos[:, 0], pos[:, 1], c=preds, cmap='tab10',
                                        s=30, vmin=0, vmax=7)
        axes[1].set_title(f'Predictions (Graph {i+1})')
        axes[1].set_xlabel('X')
        axes[1].set_ylabel('Y')
        axes[1].invert_yaxis()
        
        # Accuracy for this graph
        graph_acc = (preds == labels).mean()
        fig.suptitle(f'Example {i+1} - Accuracy: {graph_acc:.3f}', fontsize=12)
        
        plt.colorbar(scatter_pred, ax=axes, label='Class', ticks=range(8))
        plt.tight_layout()
        plt.savefig(examples_dir / f"prediction_example_{i+1}.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"  Saved {num_examples} prediction examples to {examples_dir}")
    
    return {
        'accuracy': acc,
        'f1': test_f1,
        'predictions': all_preds,
        'labels': all_labels,
        'precision': precision.tolist(),
        'recall': recall.tolist(),
        'f1_per_class': f1_per_class.tolist()
    }


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train GNN for cell classification")
    
    parser.add_argument("--data_dir", type=str, default="data/QC",
                       help="Data directory containing train_test_splits/")
    parser.add_argument("--output_dir", type=str, default="output",
                       help="Output directory for models and results")
    
    # Model config
    parser.add_argument("--hidden_dim", type=int, default=32)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--use_batch_norm", action="store_true", default=True)
    
    # Training config
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--use_class_weights", action="store_true", default=True)
    
    # Fold selection
    parser.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3, 4, 5],
                       help="Folds to use for cross-validation")
    parser.add_argument("--num_classes", type=int, default=8)
    
    # Held-out evaluation
    parser.add_argument("--eval_held_out", action="store_true",
                       help="Evaluate on held-out test set after training")
    
    # Stratified fold creation
    parser.add_argument("--create_stratified_folds", action="store_true",
                       help="Create new stratified folds based on main_classification")
    parser.add_argument("--print_distribution", action="store_true",
                       help="Print data distribution (overall and per-fold)")
    parser.add_argument("--only_create_folds", action="store_true",
                       help="Only create folds without training (use with --create_stratified_folds)")
    parser.add_argument("--random_state", type=int, default=42,
                       help="Random seed for stratified fold creation")
    parser.add_argument("--save_graphs", action="store_true",
                       help="Save graph features, edges, and labels to CSV files")
    parser.add_argument("--visualize_graphs", action="store_true",
                       help="Save visualizations of 10 random graphs")
    
    # Data augmentation
    parser.add_argument("--augment", action="store_true",
                       help="Apply data augmentation to training data")
    parser.add_argument("--n_augmentations", type=int, default=1,
                       help="Number of augmented versions per original sample")
    
    args = parser.parse_args()
    
    # Build config
    config = {
        'hidden_dim': args.hidden_dim,
        'num_layers': args.num_layers,
        'dropout': args.dropout,
        'use_batch_norm': args.use_batch_norm,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'use_class_weights': args.use_class_weights,
        'folds': args.folds,
        'num_classes': args.num_classes
    }
    
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    
    # Create stratified folds if requested
    if args.create_stratified_folds:
        print("\nCreating multi-label stratified folds based on main_classification...")
        folds = create_stratified_folds(data_dir, n_folds=5, random_state=args.random_state)
        save_stratified_folds(data_dir, folds)
        print_data_distribution(data_dir, folds)
        plot_fold_distribution(data_dir, output_dir, folds)
        
        # If only creating folds, exit here
        if args.only_create_folds:
            return
    
    # Print distribution if requested (without training)
    if args.print_distribution and not args.create_stratified_folds:
        print_data_distribution(data_dir)
        plot_fold_distribution(data_dir, output_dir)
        return
    
    # Train single model on all folds
    train_results = train_single_model(
        config, data_dir, output_dir,
        save_graphs=args.save_graphs,
        augment_train=args.augment,
        n_augmentations=args.n_augmentations,
        visualize_graphs=args.visualize_graphs
    )
    
    # Save training results
    results_path = output_dir / "training_results.json"
    serializable_results = {
        'config': config,
        'best_val_f1': train_results['best_val_f1'],
        'history': train_results['history']
    }
    with open(results_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    print(f"Results saved to {results_path}")
    
    # Evaluate on held-out test set
    if args.eval_held_out:
        test_results = evaluate_held_out(
            config, data_dir, output_dir,
            model=train_results['model'],
            device=train_results['device']
        )
        
        # Save test results
        if test_results:
            test_results_path = output_dir / "test_results.json"
            with open(test_results_path, 'w') as f:
                json.dump({
                    'accuracy': test_results['accuracy'],
                    'f1': test_results['f1'],
                    'precision': test_results['precision'],
                    'recall': test_results['recall'],
                    'f1_per_class': test_results['f1_per_class']
                }, f, indent=2)
            print(f"Test results saved to {test_results_path}")


if __name__ == "__main__":
    main()
