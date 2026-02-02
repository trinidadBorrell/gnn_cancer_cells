"""
Data augmentation module using PyTorch/torchvision.
Applies geometric and photometric augmentations to images and updates bounding box coordinates.
"""
import numpy as np
import pandas as pd
from PIL import Image
import random
import torchvision.transforms.functional as TF


def flip_horizontal(image: np.ndarray, df: pd.DataFrame) -> tuple:
    """Flip image horizontally and update bounding boxes."""
    H, W = image.shape[:2]
    image = np.fliplr(image).copy()
    
    df = df.copy()
    df["xmin"], df["xmax"] = W - df["xmax"], W - df["xmin"]
    return image, df


def flip_vertical(image: np.ndarray, df: pd.DataFrame) -> tuple:
    """Flip image vertically and update bounding boxes."""
    H, W = image.shape[:2]
    image = np.flipud(image).copy()
    
    df = df.copy()
    df["ymin"], df["ymax"] = H - df["ymax"], H - df["ymin"]
    return image, df


def rotate_90(image: np.ndarray, df: pd.DataFrame, k: int = 1) -> tuple:
    """
    Rotate image by k * 90 degrees counterclockwise and update bounding boxes.
    
    Args:
        image: RGB image as numpy array (H, W, 3)
        df: DataFrame with xmin, ymin, xmax, ymax columns
        k: Number of 90-degree rotations (1=90°, 2=180°, 3=270°)
    """
    H, W = image.shape[:2]
    image = np.rot90(image, k=k).copy()
    
    df = df.copy()
    
    if k % 4 == 1:  # 90° counterclockwise
        new_xmin = df["ymin"].copy()
        new_xmax = df["ymax"].copy()
        new_ymin = W - df["xmax"]
        new_ymax = W - df["xmin"]
        df["xmin"], df["xmax"] = new_xmin, new_xmax
        df["ymin"], df["ymax"] = new_ymin, new_ymax
    elif k % 4 == 2:  # 180°
        df["xmin"], df["xmax"] = W - df["xmax"], W - df["xmin"]
        df["ymin"], df["ymax"] = H - df["ymax"], H - df["ymin"]
    elif k % 4 == 3:  # 270° counterclockwise (= 90° clockwise)
        new_xmin = H - df["ymax"]
        new_xmax = H - df["ymin"]
        new_ymin = df["xmin"].copy()
        new_ymax = df["xmax"].copy()
        df["xmin"], df["xmax"] = new_xmin, new_xmax
        df["ymin"], df["ymax"] = new_ymin, new_ymax
    
    return image, df


def apply_color_augmentation(image: np.ndarray) -> np.ndarray:
    """
    Apply random photometric augmentations using torchvision.
    
    Args:
        image: RGB image as numpy array (H, W, 3), uint8
    
    Returns:
        Augmented image as numpy array (H, W, 3), uint8
    """
    # Convert to PIL Image for torchvision transforms
    pil_image = Image.fromarray(image)
    
    # Random brightness adjustment
    brightness_factor = random.uniform(0.8, 1.2)
    pil_image = TF.adjust_brightness(pil_image, brightness_factor)
    
    # Random contrast adjustment
    contrast_factor = random.uniform(0.8, 1.2)
    pil_image = TF.adjust_contrast(pil_image, contrast_factor)
    
    # Random saturation adjustment
    saturation_factor = random.uniform(0.8, 1.2)
    pil_image = TF.adjust_saturation(pil_image, saturation_factor)
    
    # Random hue adjustment (small range to avoid drastic color shifts)
    hue_factor = random.uniform(-0.05, 0.05)
    pil_image = TF.adjust_hue(pil_image, hue_factor)
    
    return np.array(pil_image)


def augment_fov(image_path, csv_path):
    """
    Apply random augmentation to a field of view image and its annotations.
    
    Input:
        image_path: path to RGB FOV image
        csv_path: path to CSV with xmin, ymin, xmax, ymax columns
    
    Output:
        aug_image: augmented image (numpy array, uint8)
        aug_df: updated dataframe with transformed bounding boxes
    """
    # Load data
    image = np.array(Image.open(image_path).convert("RGB"))
    df = pd.read_csv(csv_path)
    
    # Choose random geometric augmentation
    geom_choice = random.choice([
        "none", "hflip", "vflip", "rot90", "rot180", "rot270"
    ])
    
    if geom_choice == "hflip":
        image, df = flip_horizontal(image, df)
    elif geom_choice == "vflip":
        image, df = flip_vertical(image, df)
    elif geom_choice == "rot90":
        image, df = rotate_90(image, df, k=1)
    elif geom_choice == "rot180":
        image, df = rotate_90(image, df, k=2)
    elif geom_choice == "rot270":
        image, df = rotate_90(image, df, k=3)
    
    # Apply photometric augmentation
    image = apply_color_augmentation(image)
    
    return image, df
