"""Environment settings for the face-verification POC."""
import math
import os
import warnings

# Simple environment settings; defaults are POC heuristics, not calibrated quality limits.
MIN_FACE_SIZE = int(os.getenv('MIN_FACE_SIZE', '40'))
MIN_BLUR_VARIANCE = float(os.getenv('MIN_BLUR_VARIANCE', '20'))
MIN_BRIGHTNESS = float(os.getenv('MIN_BRIGHTNESS', '35'))
MAX_BRIGHTNESS = float(os.getenv('MAX_BRIGHTNESS', '220'))
DETECTION_CONFIDENCE = float(os.getenv('DETECTION_CONFIDENCE', '0.5'))
DETECTION_SIZE = int(os.getenv('DETECTION_SIZE', '640'))
CROP_MARGIN = float(os.getenv('CROP_MARGIN', '0.25'))


def get_threshold():
    value = os.environ.get("VERIFICATION_THRESHOLD")
    if value is None:
        warnings.warn("Using uncalibrated demo threshold 0.40; see evaluation report.")
        value = "0.40"
    threshold = float(value)
    if not math.isfinite(threshold) or not -1 <= threshold <= 1:
        raise ValueError("VERIFICATION_THRESHOLD must be finite and between -1 and 1")
    return threshold
