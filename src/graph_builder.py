"""
Graph Construction Module for Cell Graphs

Builds graphs from cell annotations where:
- Nodes = cells with CNN features
- Edges = spatial relationships (k-NN, radius-based, or Delaunay triangulation)
"""

import numpy as np
import torch
from torch_geometric.data import Data
from scipy.spatial import Delaunay, KDTree
from pathlib import Path
from typing import Tuple, Optional, Union, Literal


class CellGraphBuilder:
    """
    Builds cell graphs from spatial coordinates and features.
    
    Supports multiple edge construction strategies:
    - 'knn': k-nearest neighbors
    - 'radius': all cells within a radius
    - 'delaunay': Delaunay triangulation
    
    Args:
        edge_strategy: How to construct edges ('knn', 'radius', 'delaunay')
        k: Number of neighbors for k-NN (only used if edge_strategy='knn')
        radius: Distance threshold for radius-based edges (only used if edge_strategy='radius')
        self_loops: Whether to include self-loops
    """
    
    LABEL_MAPPING = {
        'sTIL': 0,
        'nonTIL_stromal': 1,
        'AMBIGUOUS': 2,
    }
    
    def __init__(
        self,
        edge_strategy: Literal['knn', 'radius', 'delaunay'] = 'knn',
        k: int = 5,
        radius: float = 50.0,
        self_loops: bool = False
    ):
        self.edge_strategy = edge_strategy
        self.k = k
        self.radius = radius
        self.self_loops = self_loops
    
    def _build_knn_edges(
        self,
        coords: np.ndarray,
        k: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Build edges using k-nearest neighbors."""
        n_nodes = len(coords)
        if n_nodes <= 1:
            return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
        
        # Adjust k if we have fewer nodes
        k_actual = min(k, n_nodes - 1)
        
        tree = KDTree(coords)
        _, indices = tree.query(coords, k=k_actual + 1)  # +1 because first neighbor is self
        
        # Build edge list (excluding self)
        src_nodes = []
        dst_nodes = []
        
        for i in range(n_nodes):
            for j in indices[i, 1:]:  # Skip first (self)
                src_nodes.append(i)
                dst_nodes.append(j)
        
        return np.array(src_nodes), np.array(dst_nodes)
    
    def _build_radius_edges(
        self,
        coords: np.ndarray,
        radius: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Build edges for all pairs within radius."""
        n_nodes = len(coords)
        if n_nodes <= 1:
            return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
        
        tree = KDTree(coords)
        pairs = tree.query_pairs(radius)
        
        src_nodes = []
        dst_nodes = []
        
        for i, j in pairs:
            # Add both directions for undirected graph
            src_nodes.extend([i, j])
            dst_nodes.extend([j, i])
        
        return np.array(src_nodes), np.array(dst_nodes)
    
    def _build_delaunay_edges(
        self,
        coords: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Build edges using Delaunay triangulation."""
        n_nodes = len(coords)
        if n_nodes < 3:
            # Fall back to fully connected for < 3 nodes
            if n_nodes <= 1:
                return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
            return np.array([0, 1]), np.array([1, 0])
        
        try:
            tri = Delaunay(coords)
        except Exception:
            # Fall back to k-NN if Delaunay fails (e.g., collinear points)
            return self._build_knn_edges(coords, k=self.k)
        
        edges = set()
        for simplex in tri.simplices:
            for i in range(3):
                for j in range(i + 1, 3):
                    edges.add((simplex[i], simplex[j]))
                    edges.add((simplex[j], simplex[i]))
        
        if not edges:
            return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
        
        edges = np.array(list(edges))
        return edges[:, 0], edges[:, 1]
    
    def build_edges(
        self,
        coords: np.ndarray
    ) -> torch.Tensor:
        """
        Build edge index tensor from coordinates.
        
        Args:
            coords: Node coordinates of shape (N, 2)
        
        Returns:
            Edge index tensor of shape (2, num_edges)
        """
        if self.edge_strategy == 'knn':
            src, dst = self._build_knn_edges(coords, self.k)
        elif self.edge_strategy == 'radius':
            src, dst = self._build_radius_edges(coords, self.radius)
        elif self.edge_strategy == 'delaunay':
            src, dst = self._build_delaunay_edges(coords)
        else:
            raise ValueError(f"Unknown edge strategy: {self.edge_strategy}")
        
        # Add self-loops if requested
        if self.self_loops and len(coords) > 0:
            n = len(coords)
            self_src = np.arange(n)
            self_dst = np.arange(n)
            src = np.concatenate([src, self_src])
            dst = np.concatenate([dst, self_dst])
        
        edge_index = torch.tensor(np.stack([src, dst]), dtype=torch.long)
        return edge_index
    
    def compute_edge_features(
        self,
        coords: np.ndarray,
        edge_index: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute edge features (distance and angle).
        
        Args:
            coords: Node coordinates (N, 2)
            edge_index: Edge index (2, E)
        
        Returns:
            Edge features (E, 2) containing [distance, angle]
        """
        if edge_index.shape[1] == 0:
            return torch.empty((0, 2), dtype=torch.float)
        
        src_coords = coords[edge_index[0].numpy()]
        dst_coords = coords[edge_index[1].numpy()]
        
        # Compute distance
        diff = dst_coords - src_coords
        distance = np.linalg.norm(diff, axis=1)
        
        # Compute angle (in radians)
        angle = np.arctan2(diff[:, 1], diff[:, 0])
        
        edge_features = np.stack([distance, angle], axis=1)
        return torch.tensor(edge_features, dtype=torch.float)
    
    def labels_to_tensor(
        self,
        labels: np.ndarray,
        label_mapping: Optional[dict] = None
    ) -> torch.Tensor:
        """Convert string labels to integer tensor."""
        if label_mapping is None:
            label_mapping = self.LABEL_MAPPING
        
        int_labels = np.array([label_mapping.get(label, -1) for label in labels])
        return torch.tensor(int_labels, dtype=torch.long)
    
    def build_graph(
        self,
        features: np.ndarray,
        coords: np.ndarray,
        labels: Optional[np.ndarray] = None,
        compute_edge_attr: bool = True
    ) -> Data:
        """
        Build a PyTorch Geometric Data object.
        
        Args:
            features: Node features (N, D)
            coords: Node coordinates (N, 2)
            labels: Optional node labels (N,)
            compute_edge_attr: Whether to compute edge features
        
        Returns:
            PyG Data object with node features, edges, and optional labels
        """
        # Build edge index
        edge_index = self.build_edges(coords)
        
        # Create Data object
        data = Data(
            x=torch.tensor(features, dtype=torch.float),
            edge_index=edge_index,
            pos=torch.tensor(coords, dtype=torch.float)
        )
        
        # Add edge features
        if compute_edge_attr:
            data.edge_attr = self.compute_edge_features(coords, edge_index)
        
        # Add labels if provided
        if labels is not None:
            data.y = self.labels_to_tensor(labels)
        
        return data
    
    def build_from_npz(
        self,
        npz_path: Union[str, Path],
        compute_edge_attr: bool = True
    ) -> Data:
        """
        Build graph from a saved .npz file (output of feature extractor).
        
        Args:
            npz_path: Path to .npz file with features, coords, and labels
            compute_edge_attr: Whether to compute edge features
        
        Returns:
            PyG Data object
        """
        data_dict = np.load(npz_path, allow_pickle=True)
        
        features = data_dict['features']
        coords = np.stack([data_dict['coords_x'], data_dict['coords_y']], axis=1)
        labels = data_dict.get('labels', None)
        
        return self.build_graph(
            features=features,
            coords=coords,
            labels=labels,
            compute_edge_attr=compute_edge_attr
        )


def build_graphs_for_dataset(
    features_dir: Union[str, Path],
    output_dir: Union[str, Path],
    edge_strategy: str = 'knn',
    k: int = 5,
    radius: float = 50.0
) -> None:
    """
    Build graphs for all feature files in a directory.
    
    Args:
        features_dir: Directory containing .npz feature files
        output_dir: Directory to save graph .pt files
        edge_strategy: Edge construction strategy
        k: k for k-NN
        radius: Radius for radius-based edges
    """
    features_dir = Path(features_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    builder = CellGraphBuilder(
        edge_strategy=edge_strategy,
        k=k,
        radius=radius
    )
    
    npz_files = sorted(features_dir.glob('*.npz'))
    
    print(f"Building graphs for {len(npz_files)} feature files...")
    
    for npz_path in npz_files:
        graph = builder.build_from_npz(npz_path)
        
        # Save graph
        output_path = output_dir / f"{npz_path.stem}.pt"
        torch.save(graph, output_path)
    
    print(f"Graphs saved to {output_dir}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Build cell graphs from features")
    parser.add_argument("--features_dir", type=str, required=True, help="Directory with .npz feature files")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for graphs")
    parser.add_argument("--edge_strategy", type=str, default="knn", choices=["knn", "radius", "delaunay"])
    parser.add_argument("--k", type=int, default=5, help="k for k-NN")
    parser.add_argument("--radius", type=float, default=50.0, help="Radius for radius-based edges")
    
    args = parser.parse_args()
    
    build_graphs_for_dataset(
        features_dir=args.features_dir,
        output_dir=args.output_dir,
        edge_strategy=args.edge_strategy,
        k=args.k,
        radius=args.radius
    )
