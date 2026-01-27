"""
Test script to verify the feature extraction and graph construction pipeline.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from feature_extractor import MorphologicalFeatureExtractor
from graph_builder import CellGraphBuilder
from visualize import visualize_graph_sanity_check
import numpy as np
import torch


def test_pipeline(save_output: bool = True):
    """Test feature extraction and graph building on a sample image."""
    
    # Paths
    data_dir = Path(__file__).parent.parent / "data" / "QC"
    rgb_dir = data_dir / "rgb"
    csv_dir = data_dir / "csv"
    output_dir = Path(__file__).parent.parent / "output"
    
    if save_output:
        output_dir.mkdir(parents=True, exist_ok=True)
        features_dir = output_dir / "features"
        graphs_dir = output_dir / "graphs"
        viz_dir = output_dir / "visualizations"
        features_dir.mkdir(exist_ok=True)
        graphs_dir.mkdir(exist_ok=True)
        viz_dir.mkdir(exist_ok=True)
    
    # Get first available image
    image_files = sorted(rgb_dir.glob("*.png"))
    if not image_files:
        print("No images found in", rgb_dir)
        return
    
    img_path = image_files[0]
    csv_path = csv_dir / f"{img_path.stem}.csv"
    
    print(f"Testing with: {img_path.name}")
    print("-" * 50)
    
    # 1. Test feature extraction (morphological features)
    print("\n1. Feature Extraction (Morphological)")
    extractor = MorphologicalFeatureExtractor()
    print(f"   Feature names: {extractor.FEATURE_NAMES}")
    print(f"   Feature dim: {extractor.feature_dim}")
    
    features, annotations = extractor.extract_from_image_and_annotations(
        img_path, csv_path, padding=0
    )
    
    # Compute cell centroids
    coords = np.stack([
        (annotations['xmin'].values + annotations['xmax'].values) / 2,
        (annotations['ymin'].values + annotations['ymax'].values) / 2
    ], axis=1)
    
    print(f"   Extracted features shape: {features.shape}")
    print(f"   Number of cells: {len(annotations)}")
    print("   Label distribution:")
    for label, count in annotations['super_classification'].value_counts().items():
        print(f"      {label}: {count}")
    
    # Save features
    if save_output:
        features_path = features_dir / f"{img_path.stem}.npz"
        np.savez(
            features_path,
            features=features,
            labels=annotations['super_classification'].values,
            main_labels=annotations['main_classification'].values,
            coords_x=coords[:, 0],
            coords_y=coords[:, 1],
        )
        print(f"   Saved features to: {features_path}")
    
    # 2. Test graph construction
    print("\n2. Graph Construction")
    
    labels = annotations['super_classification'].values
    for strategy in ['knn', 'radius', 'delaunay']:
        builder = CellGraphBuilder(edge_strategy=strategy, k=5, radius=50.0)
        graph = builder.build_graph(features, coords, labels, compute_edge_attr=True)
        
        print(f"\n   Strategy: {strategy}")
        print(f"   Nodes: {graph.x.shape[0]}")
        print(f"   Edges: {graph.edge_index.shape[1]}")
        print(f"   Node features: {graph.x.shape}")
        print(f"   Edge features: {graph.edge_attr.shape}")
        print(f"   Labels shape: {graph.y.shape}")
        print(f"   Unique labels: {graph.y.unique().tolist()}")
        
        # Save graph
        if save_output:
            graph_path = graphs_dir / f"{img_path.stem}_{strategy}.pt"
            torch.save(graph, graph_path)
            print(f"   Saved graph to: {graph_path}")
        
        # Generate visualizations (only for knn strategy to avoid redundancy)
        if save_output and strategy == 'knn':
            print("\n3. Generating Visualizations...")
            visualize_graph_sanity_check(
                graph=graph,
                image_path=img_path,
                annotations=annotations,
                output_dir=viz_dir,
                prefix=f"{img_path.stem}_{strategy}_",
                max_nodes=15
            )
    
    print("\n" + "=" * 50)
    print("Pipeline test completed successfully!")
    print("=" * 50)


if __name__ == "__main__":
    test_pipeline()
