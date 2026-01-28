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
from sklearn.metrics import  f1_score

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
                return graph
        
        return None
        
    except Exception as e:
        print(f"Warning: Could not build graph from {image_path}: {e}")
        return None


def load_fold_data(
    data_dir: Path,
    fold: int,
    feature_extractor: MorphologicalFeatureExtractor,
    graph_builder: CellGraphBuilder
) -> Tuple[List[Data], List[Data]]:
    """
    Build train/test graphs on-the-fly for a specific fold.
    
    Args:
        data_dir: Base data directory (contains train_test_splits/, rgb/, csv/)
        fold: Fold number (1-5, or 999 for held-out)
        feature_extractor: Feature extractor instance
        graph_builder: Graph builder instance
    
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
    
    def build_graphs_for_slides(slide_names: List[str]) -> List[Data]:
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
                
                graph = build_graph_from_files(
                    img_path, csv_path, feature_extractor, graph_builder
                )
                
                if graph is not None:
                    graphs.append(graph)
        
        return graphs
    
    print(f"  Building graphs for {len(train_slides)} train slides...")
    train_graphs = build_graphs_for_slides(train_slides)
    print(f"  Built {len(train_graphs)} train graphs")
    
    print(f"  Building graphs for {len(test_slides)} test slides...")
    test_graphs = build_graphs_for_slides(test_slides)
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
    output_dir: Path
) -> Dict:
    """
    Train a SINGLE model using all folds (1-5) combined.
    
    - Training data: all train splits from folds 1-5
    - Validation data: all test splits from folds 1-5
    - Final test: held-out fold 999
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
                data_dir, fold, feature_extractor, graph_builder
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
    
    # Train single model on all folds
    train_results = train_single_model(config, data_dir, output_dir)
    
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
