"""
Mathematical corruption implementations based on the ImageNet-C benchmark
(Hendrycks & Dietterich, 2019).

Includes:
- Seen corruptions (training / in-distribution test):
    Gaussian Blur, Gaussian Noise, JPEG Compression, Contrast Reduction, Speckle Noise
- Unseen corruptions (zero-shot out-of-distribution evaluation):
    Pixelate (aliasing), Impulse Noise (salt-and-pepper)
"""
import math
import cv2
import numpy as np

ALL_CORRUPTIONS = ["blur", "noise", "jpeg", "contrast", "speckle"]
UNSEEN_CORRUPTIONS = ["pixelate", "impulse_noise"]

def _clip_uint8(img):
    """Clip values to [0, 255] and cast to uint8."""
    return np.clip(img, 0, 255).astype(np.uint8)

# -----------------------------------------------------------------------------
# 1. Seen Corruptions
# -----------------------------------------------------------------------------
_BLUR_PARAMS = {"clean": None, "moderate": (11, 5.0), "severe": (21, 10.0)}
def apply_gaussian_blur(img: np.ndarray, severity: str) -> np.ndarray:
    """Apply spatial Gaussian blur."""
    p = _BLUR_PARAMS.get(severity, None)
    return img if p is None else cv2.GaussianBlur(img, (p[0], p[0]), p[1])

_NOISE_STD = {"clean": 0.0, "moderate": 0.20, "severe": 0.50}
def apply_gaussian_noise(img: np.ndarray, severity: str) -> np.ndarray:
    """Apply additive Gaussian white noise."""
    std = _NOISE_STD.get(severity, 0.0) * 255.0
    if std == 0.0:
        return img
    return _clip_uint8(img.astype(np.float32) + np.random.normal(0, std, img.shape))

_JPEG_QUALITY = {"clean": 100, "moderate": 5, "severe": 1}
def apply_jpeg_compression(img: np.ndarray, severity: str) -> np.ndarray:
    """Apply lossy JPEG compression artifacts."""
    q = _JPEG_QUALITY.get(severity, 100)
    if q >= 100:
        return img
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    ok, enc = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, q])
    dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return cv2.cvtColor(dec, cv2.COLOR_BGR2RGB)

_CONTRAST_FACTOR = {"clean": 1.0, "moderate": 0.10, "severe": 0.02}
def apply_contrast_shift(img: np.ndarray, severity: str) -> np.ndarray:
    """Apply global dynamic range / contrast reduction."""
    f = _CONTRAST_FACTOR.get(severity, 1.0)
    if f == 1.0:
        return img
    mean = img.mean()
    return _clip_uint8((img.astype(np.float32) - mean) * f + mean)

_SPECKLE_VAR = {"clean": 0.0, "moderate": 0.5, "severe": 1.0}
def apply_speckle_noise(img: np.ndarray, severity: str) -> np.ndarray:
    """Apply multiplicative speckle noise: img * (1 + noise)."""
    v = _SPECKLE_VAR.get(severity, 0.0)
    if v == 0.0:
        return img
    noise = np.random.normal(0, v ** 0.5, img.shape)
    return _clip_uint8(img.astype(np.float32) * (1.0 + noise))

# -----------------------------------------------------------------------------
# 2. Unseen Corruptions (Zero-Shot Generalization Test)
# -----------------------------------------------------------------------------
_PIXELATE_FACTOR = {"clean": 1.0, "moderate": 0.25, "severe": 0.10}
def apply_pixelate(img: np.ndarray, severity: str) -> np.ndarray:
    """Apply pixelation aliasing by nearest-neighbor downsampling & upsampling."""
    factor = _PIXELATE_FACTOR.get(severity, 1.0)
    if factor == 1.0:
        return img
    h, w = img.shape[:2]
    small = cv2.resize(img, (max(1, int(w * factor)), max(1, int(h * factor))), interpolation=cv2.INTER_NEAREST)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

_IMPULSE_PROB = {"clean": 0.0, "moderate": 0.08, "severe": 0.20}
def apply_impulse_noise(img: np.ndarray, severity: str) -> np.ndarray:
    """Apply salt-and-pepper impulse noise."""
    p = _IMPULSE_PROB.get(severity, 0.0)
    if p == 0.0:
        return img
    res = img.copy()
    num_salt = int(np.ceil(p * img.size * 0.5))
    coords_salt = [np.random.randint(0, i - 1, num_salt) for i in img.shape[:2]]
    res[tuple(coords_salt)] = 255
    num_pepper = int(np.ceil(p * img.size * 0.5))
    coords_pepper = [np.random.randint(0, i - 1, num_pepper) for i in img.shape[:2]]
    res[tuple(coords_pepper)] = 0
    return res

# -----------------------------------------------------------------------------
# 3. Unified Dispatch
# -----------------------------------------------------------------------------
CORRUPTION_FUNCS = {
    "blur": apply_gaussian_blur,
    "noise": apply_gaussian_noise,
    "jpeg": apply_jpeg_compression,
    "contrast": apply_contrast_shift,
    "speckle": apply_speckle_noise,
    "pixelate": apply_pixelate,
    "impulse_noise": apply_impulse_noise,
}

def apply_corruption(img_array: np.ndarray, corruption: str, severity: str) -> np.ndarray:
    """Dispatch corruption by name and severity."""
    if severity == "clean" or corruption not in CORRUPTION_FUNCS:
        return img_array
    return CORRUPTION_FUNCS[corruption](img_array, severity)
