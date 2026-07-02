"""
features.py
------------
"""

from pathlib import Path
from scipy.fft import fft2, fftshift
from skimage.measure import shannon_entropy
from skimage.feature import local_binary_pattern

import cv2
import numpy as np

# Project Constants
IMAGE_SIZE = (512, 512)
CANNY_LOW_THRESHOLD = 100
CANNY_HIGH_THRESHOLD = 200
LBP_RADIUS = 2
LBP_POINTS = 8 * LBP_RADIUS
#load img
def load_image(image_path: str | Path) -> np.ndarray:
    """
    Load an image from disk.

    Parameters
    ----------
    image_path : str | Path
        Path to the image.

    Returns
    -------
    np.ndarray
        Loaded image in BGR format.

    Raises
    ------
    FileNotFoundError
        If image cannot be loaded.
    """

    image = cv2.imread(str(image_path))

    if image is None:
        raise FileNotFoundError(
            f"Unable to load image: {image_path}"
        )

    return image

#diff phones produce diff resolutions so jst fiixng the size issue
def resize_image(
    image: np.ndarray,
    size: tuple[int, int] = IMAGE_SIZE
) -> np.ndarray:
    """
    Resize image to a fixed resolution.

    Parameters
    ----------
    image : np.ndarray
        Input image.

    size : tuple
        Desired width and height.

    Returns
    -------
    np.ndarray
        Resized image.
    """

    return cv2.resize(image, size)

#convert to gratscale bcz most texture analysis work better in it
def to_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Convert a BGR image to grayscale.

    Parameters
    ----------
    image : np.ndarray

    Returns
    -------
    np.ndarray
        Grayscale image.
    """

    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#normalize
def normalize_image(gray: np.ndarray) -> np.ndarray:
    """
    Normalize grayscale image to [0,1].

    Parameters
    ----------
    gray : np.ndarray

    Returns
    -------
    np.ndarray
    """

    return gray.astype(np.float32) / 255.0
def preprocess_image(image_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Complete preprocessing pipeline.

    Parameters
    ----------
    image_path : str | Path

    Returns
    -------
    tuple
        gray_image, normalized_image
    """

    image = load_image(image_path)

    image = resize_image(image)

    gray = to_grayscale(image)

    normalized = normalize_image(gray)

    return gray, normalized
#to handle high-freq standards like pixel boundaries rendering artifacts etc
#laplacian ft
def laplacian_variance(gray: np.ndarray) -> float:
    """
    Compute the variance of the Laplacian.

    This measures the amount of high-frequency detail
    (edges and sharp transitions) in an image.

    Higher values generally indicate a sharper image.

    Parameters
    ----------
    gray : np.ndarray
        Grayscale image.

    Returns
    -------
    float
        Variance of the Laplacian.
    """

    laplacian = cv2.Laplacian(gray, cv2.CV_64F)

    return float(laplacian.var())

def edge_density(gray: np.ndarray) -> float:
    """
    Compute the proportion of edge pixels in the image.

    Screen photos often contain many sharp artificial
    edges caused by display pixels, UI elements,
    bezels, and reflections.

    Parameters
    ----------
    gray : np.ndarray
        Grayscale image.

    Returns
    -------
    float
        Fraction of pixels detected as edges.
    """

    edges = cv2.Canny(
        gray,
        CANNY_LOW_THRESHOLD,
        CANNY_HIGH_THRESHOLD,
    )

    density = np.count_nonzero(edges) / edges.size

    return float(density)


def brightness(gray: np.ndarray) -> float:
    """
    Compute the average brightness of the image.
    """

    return float(np.mean(gray))

def contrast(gray: np.ndarray) -> float:
    """
    Compute image contrast using the standard deviation
    of pixel intensities.
    """

    return float(np.std(gray))

def entropy(gray: np.ndarray) -> float:
    """
    Estimate image information content.

    Images with more texture generally have
    higher entropy.
    """

    return float(shannon_entropy(gray))

def lbp_score(gray: np.ndarray) -> float:
    """
    Compute Local Binary Pattern texture score.
    """

    lbp = local_binary_pattern(
        gray,
        P=LBP_POINTS,
        R=LBP_RADIUS,
        method="uniform"
    )

    return float(np.mean(lbp))
def fft_energy(normalized: np.ndarray) -> float:
    """
    Estimate high-frequency energy using the
    Fourier Transform.

    Screen images often contain stronger
    periodic high-frequency components.
    """

    fft = fft2(normalized)

    fft = fftshift(fft)

    magnitude = np.abs(fft)

    h, w = magnitude.shape

    center = 30

    magnitude[
        h//2-center:h//2+center,
        w//2-center:w//2+center
    ] = 0

    return float(np.mean(magnitude))
def extract_features(image_path: str | Path) -> dict:
    """
    Extract handcrafted features from an image.
    """

    # ----------------------------
    # Load & preprocess
    # ----------------------------
    image = load_image(image_path)
    image = resize_image(image)

    gray = to_grayscale(image)
    normalized = normalize_image(gray)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # ----------------------------
    # Basic Statistics
    # ----------------------------
    brightness_mean = float(np.mean(gray))
    brightness_std = float(np.std(gray))

    # ----------------------------
    # RGB Statistics
    # ----------------------------
    b, g, r = cv2.split(image)

    r_mean = float(np.mean(r))
    g_mean = float(np.mean(g))
    b_mean = float(np.mean(b))

    r_std = float(np.std(r))
    g_std = float(np.std(g))
    b_std = float(np.std(b))

    # ----------------------------
    # HSV Statistics
    # ----------------------------
    h, s, v = cv2.split(hsv)

    hue_mean = float(np.mean(h))
    saturation_mean = float(np.mean(s))
    value_mean = float(np.mean(v))

    saturation_std = float(np.std(s))

    # ----------------------------
    # Glare
    # ----------------------------
    glare_ratio = float(np.sum(gray > 240) / gray.size)

    # ----------------------------
    # Dark Pixel Ratio
    # ----------------------------
    dark_ratio = float(np.sum(gray < 30) / gray.size)

    # ----------------------------
    # Edge Map
    # ----------------------------
    edges = cv2.Canny(gray, 100, 200)

    edge_pixels = float(np.count_nonzero(edges))

    edge_ratio = edge_pixels / edges.size

    # ----------------------------
    # Sobel Gradient Energy
    # ----------------------------
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)

    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1)

    gradient = np.sqrt(sobelx ** 2 + sobely ** 2)

    gradient_energy = float(np.mean(gradient))

    # ----------------------------
    # Bright Pixel Ratio
    # ----------------------------
    bright_ratio = float(np.sum(gray > 220) / gray.size)

    # ----------------------------
    # FFT Energy
    # ----------------------------
    fft = fft2(normalized)

    fft = fftshift(fft)

    magnitude = np.abs(fft)

    h_img, w_img = magnitude.shape

    center = 30

    magnitude[
        h_img//2-center:h_img//2+center,
        w_img//2-center:w_img//2+center
    ] = 0

    fft_score = float(np.mean(magnitude))

    # ----------------------------
    # Final Feature Dictionary
    # ----------------------------

    return {

        # Existing features
        "laplacian_variance": laplacian_variance(gray),

        "edge_density": edge_density(gray),

        "brightness": brightness(gray),

        "contrast": contrast(gray),

        "entropy": entropy(gray),

        "lbp": lbp_score(gray),

        "fft_energy": fft_score,

        # RGB
        "r_mean": r_mean,
        "g_mean": g_mean,
        "b_mean": b_mean,

        "r_std": r_std,
        "g_std": g_std,
        "b_std": b_std,

        # HSV
        "hue_mean": hue_mean,
        "saturation_mean": saturation_mean,
        "value_mean": value_mean,
        "saturation_std": saturation_std,

        # Brightness
        "brightness_mean": brightness_mean,
        "brightness_std": brightness_std,

        # Screen-specific
        "glare_ratio": glare_ratio,
        "dark_ratio": dark_ratio,
        "bright_ratio": bright_ratio,

        # Gradient
        "edge_ratio": edge_ratio,
        "gradient_energy": gradient_energy,
    }
if __name__ == "__main__":

    features = extract_features(
        "data/raw/real/cllgid.jpg"
    )

    print("-" * 40)

    for name, value in features.items():
        print(f"{name:<22}: {value:.4f}")