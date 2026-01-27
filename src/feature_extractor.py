"""Feature Extractors for Cell Patches

Provides two types of feature extraction:
1. CNN features: Deep features from pretrained backbones (ResNet, EfficientNet)
2. Morphological features: width, height, mean color, std color, gradient
"""

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Union
import pandas as pd
from scipy import ndimage


class MorphologicalFeatureExtractor:
    """
    Extracts morphological features from cell patches.
    
    Features extracted per cell:
    - width: bounding box width
    - height: bounding box height
    - hist_r, hist_g, hist_b: RGB histograms with fixed bins (0-255)
    - grad_r, grad_g, grad_b: mean gradient magnitude per channel
    
    Total features = 2 (size) + 3*n_bins (histograms) + 3 (gradients)
    
    Args:
        n_bins: Number of histogram bins per channel (default 16)
                Bins are fixed from 0-255 so bin positions are consistent.
    """
    
    def __init__(self, n_bins: int = 16, max_cell_size: int = 256):
        self.n_bins = n_bins
        self.max_cell_size = max_cell_size  # For normalizing width/height to [0, 1]
        
        # Fixed bin edges from 0 to 256 (256 to include 255)
        self.bin_edges = np.linspace(0, 256, n_bins + 1)
        
        # Max Sobel gradient for uint8: ~4*255 = 1020 (theoretical max)
        self.max_gradient = 1020.0
        
        # Build feature names
        self.FEATURE_NAMES = ['width_norm', 'height_norm']
        for channel in ['r', 'g', 'b']:
            for i in range(n_bins):
                bin_start = int(self.bin_edges[i])
                bin_end = int(self.bin_edges[i + 1])
                self.FEATURE_NAMES.append(f'hist_{channel}_{bin_start}-{bin_end}')
        self.FEATURE_NAMES.extend(['grad_r_norm', 'grad_g_norm', 'grad_b_norm'])
        
        self.feature_dim = len(self.FEATURE_NAMES)
    
    def extract_patch_features(self, patch: np.ndarray) -> np.ndarray:
        """
        Extract morphological features from a single patch.
        
        Args:
            patch: RGB image patch as numpy array (H, W, 3)
        
        Returns:
            Feature vector of length (2 + 3*n_bins + 3)
        """
        h, w = patch.shape[:2]
        
        # Size features - normalized to [0, 1] by max_cell_size
        width_norm = w / self.max_cell_size
        height_norm = h / self.max_cell_size
        
        # RGB histograms with fixed bins (0-255)
        # Using density=True normalizes by total pixels, making patches comparable
        # Multiply by bin_width to get probabilities that sum to 1
        bin_width = 256.0 / self.n_bins
        hist_rgb = []
        for c in range(3):
            hist, _ = np.histogram(
                patch[:, :, c].flatten(),
                bins=self.bin_edges,
                density=True
            )
            # Convert density to probability (density * bin_width = probability)
            hist = hist * bin_width  # Now sums to 1, values in [0, 1]
            hist_rgb.append(hist)
        hist_rgb = np.concatenate(hist_rgb)
        
        # Gradient magnitude per channel (using Sobel) - normalized to [0, 1]
        grad_rgb = []
        for c in range(3):
            channel = patch[:, :, c].astype(np.float32)
            grad_x = ndimage.sobel(channel, axis=1)
            grad_y = ndimage.sobel(channel, axis=0)
            grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            # Normalize by max possible gradient
            grad_rgb.append(grad_magnitude.mean() / self.max_gradient)
        grad_rgb = np.array(grad_rgb)
        
        # Combine all features - all now in similar [0, 1] range
        features = np.concatenate([
            [width_norm, height_norm],
            hist_rgb,
            grad_rgb
        ])
        
        return features
    
    def extract_features(
        self,
        image: np.ndarray,
        annotations: pd.DataFrame,
        padding: int = 0
    ) -> np.ndarray:
        """
        Extract morphological features for all annotated cells.
        
        Args:
            image: Full RGB image as numpy array (H, W, 3)
            annotations: DataFrame with xmin, ymin, xmax, ymax columns
            padding: Extra padding around bounding boxes
        
        Returns:
            Feature matrix (num_cells, 11)
        """
        h_img, w_img = image.shape[:2]
        features_list = []
        
        for _, row in annotations.iterrows():
            xmin = max(0, int(row['xmin']) - padding)
            ymin = max(0, int(row['ymin']) - padding)
            xmax = min(w_img, int(row['xmax']) + padding)
            ymax = min(h_img, int(row['ymax']) + padding)
            
            patch = image[ymin:ymax, xmin:xmax]
            
            # Handle edge case of empty patch
            if patch.size == 0:
                features_list.append(np.zeros(self.feature_dim))
            else:
                features_list.append(self.extract_patch_features(patch))
        
        return np.vstack(features_list)
    
    def extract_from_image_and_annotations(
        self,
        image_path: Union[str, Path],
        csv_path: Union[str, Path],
        padding: int = 0
    ) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Extract morphological features for all cells in an image.
        
        Args:
            image_path: Path to RGB image
            csv_path: Path to CSV with cell annotations
            padding: Padding around bounding boxes
        
        Returns:
            features: Feature matrix (num_cells, 11)
            annotations: DataFrame with cell annotations
        """
        # Load image
        image = np.array(Image.open(image_path).convert('RGB'))
        
        # Load annotations
        annotations = pd.read_csv(csv_path, index_col=0)
        
        # Extract features
        features = self.extract_features(image, annotations, padding=padding)
        
        return features, annotations


class CellFeatureExtractor:
    """
    Extracts CNN features from cell patches cropped from histopathology images.
    
    Args:
        backbone: Pretrained model to use ('resnet18', 'resnet50', 'efficientnet_b0')
        device: Device to run inference on ('cuda' or 'cpu')
        patch_size: Size to resize cell patches before feature extraction
    """
    
    SUPPORTED_BACKBONES = {
        'resnet18': (models.resnet18, models.ResNet18_Weights.IMAGENET1K_V1, 512),
        'resnet50': (models.resnet50, models.ResNet50_Weights.IMAGENET1K_V1, 2048),
        'efficientnet_b0': (models.efficientnet_b0, models.EfficientNet_B0_Weights.IMAGENET1K_V1, 1280),
    }
    
    def __init__(
        self,
        backbone: str = 'resnet50',
        device: Optional[str] = None,
        patch_size: int = 64
    ):
        if backbone not in self.SUPPORTED_BACKBONES:
            raise ValueError(f"Backbone must be one of {list(self.SUPPORTED_BACKBONES.keys())}")
        
        self.backbone_name = backbone
        self.patch_size = patch_size
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load pretrained model
        model_fn, weights, self.feature_dim = self.SUPPORTED_BACKBONES[backbone]
        self.model = self._build_feature_extractor(model_fn, weights)
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Image preprocessing transform
        self.transform = transforms.Compose([
            transforms.Resize((patch_size, patch_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def _build_feature_extractor(self, model_fn, weights) -> nn.Module:
        """Remove classification head to get feature extractor."""
        model = model_fn(weights=weights)
        
        if 'resnet' in self.backbone_name:
            # Remove final FC layer, keep avgpool
            modules = list(model.children())[:-1]
            model = nn.Sequential(*modules, nn.Flatten())
        elif 'efficientnet' in self.backbone_name:
            # Remove classifier, keep avgpool
            model.classifier = nn.Identity()
        
        return model
    
    def extract_patch(
        self,
        image: np.ndarray,
        xmin: int,
        ymin: int,
        xmax: int,
        ymax: int,
        padding: int = 5
    ) -> Image.Image:
        """
        Extract a cell patch from the image given bounding box coordinates.
        
        Args:
            image: RGB image as numpy array (H, W, 3)
            xmin, ymin, xmax, ymax: Bounding box coordinates
            padding: Extra padding around the bounding box
        
        Returns:
            PIL Image of the cropped cell patch
        """
        h, w = image.shape[:2]
        
        # Add padding and clip to image bounds
        xmin = max(0, xmin - padding)
        ymin = max(0, ymin - padding)
        xmax = min(w, xmax + padding)
        ymax = min(h, ymax + padding)
        
        patch = image[ymin:ymax, xmin:xmax]
        return Image.fromarray(patch)
    
    @torch.no_grad()
    def extract_features(
        self,
        patches: List[Image.Image],
        batch_size: int = 32
    ) -> np.ndarray:
        """
        Extract features from a list of cell patches.
        
        Args:
            patches: List of PIL Images
            batch_size: Batch size for inference
        
        Returns:
            Feature matrix of shape (num_patches, feature_dim)
        """
        if not patches:
            return np.empty((0, self.feature_dim))
        
        features_list = []
        
        for i in range(0, len(patches), batch_size):
            batch_patches = patches[i:i + batch_size]
            
            # Transform and stack
            batch_tensors = torch.stack([
                self.transform(p) for p in batch_patches
            ]).to(self.device)
            
            # Extract features
            batch_features = self.model(batch_tensors)
            features_list.append(batch_features.cpu().numpy())
        
        return np.vstack(features_list)
    
    def extract_from_image_and_annotations(
        self,
        image_path: Union[str, Path],
        csv_path: Union[str, Path],
        batch_size: int = 32,
        padding: int = 5
    ) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Extract features for all annotated cells in an image.
        
        Args:
            image_path: Path to RGB image
            csv_path: Path to CSV with cell annotations
            batch_size: Batch size for feature extraction
            padding: Padding around bounding boxes
        
        Returns:
            features: Feature matrix (num_cells, feature_dim)
            annotations: DataFrame with cell annotations
        """
        # Load image
        image = np.array(Image.open(image_path).convert('RGB'))
        
        # Load annotations
        annotations = pd.read_csv(csv_path, index_col=0)
        
        # Extract patches for each cell
        patches = []
        for _, row in annotations.iterrows():
            patch = self.extract_patch(
                image,
                int(row['xmin']),
                int(row['ymin']),
                int(row['xmax']),
                int(row['ymax']),
                padding=padding
            )
            patches.append(patch)
        
        # Extract features
        features = self.extract_features(patches, batch_size=batch_size)
        
        return features, annotations


def extract_features_for_dataset(
    rgb_dir: Union[str, Path],
    csv_dir: Union[str, Path],
    output_dir: Union[str, Path],
    backbone: str = 'resnet50',
    batch_size: int = 32,
    device: Optional[str] = None
) -> None:
    """
    Extract features for all images in a dataset and save to disk.
    
    Args:
        rgb_dir: Directory containing RGB images
        csv_dir: Directory containing CSV annotations
        output_dir: Directory to save extracted features
        backbone: CNN backbone to use
        batch_size: Batch size for inference
        device: Device to use ('cuda' or 'cpu')
    """
    rgb_dir = Path(rgb_dir)
    csv_dir = Path(csv_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    extractor = CellFeatureExtractor(backbone=backbone, device=device)
    
    # Find all RGB images
    image_files = sorted(rgb_dir.glob('*.png'))
    
    print(f"Extracting features for {len(image_files)} images using {backbone}...")
    
    for img_path in image_files:
        # Find corresponding CSV
        csv_path = csv_dir / f"{img_path.stem}.csv"
        
        if not csv_path.exists():
            print(f"Warning: No CSV found for {img_path.name}, skipping")
            continue
        
        # Extract features
        features, annotations = extractor.extract_from_image_and_annotations(
            img_path, csv_path, batch_size=batch_size
        )
        
        # Save features and metadata
        output_path = output_dir / f"{img_path.stem}.npz"
        np.savez(
            output_path,
            features=features,
            labels=annotations['super_classification'].values,
            main_labels=annotations['main_classification'].values,
            coords_x=annotations['xmin'].values + (annotations['xmax'].values - annotations['xmin'].values) / 2,
            coords_y=annotations['ymin'].values + (annotations['ymax'].values - annotations['ymin'].values) / 2,
        )
    
    print(f"Features saved to {output_dir}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract CNN features from cell patches")
    parser.add_argument("--rgb_dir", type=str, required=True, help="Directory with RGB images")
    parser.add_argument("--csv_dir", type=str, required=True, help="Directory with CSV annotations")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for features")
    parser.add_argument("--backbone", type=str, default="resnet50", help="CNN backbone")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")
    
    args = parser.parse_args()
    
    extract_features_for_dataset(
        rgb_dir=args.rgb_dir,
        csv_dir=args.csv_dir,
        output_dir=args.output_dir,
        backbone=args.backbone,
        batch_size=args.batch_size,
        device=args.device
    )
