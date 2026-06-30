import cv2
import pytest
import numpy as np
from pipeline import *

def test_preprocessImageValid():
    # Create a dummy 100x100 black image
    dummyImgage = np.zeros((100, 100, 3), dtype=np.uint8)
    result = preprocessImage(dummyImgage)
    assert result.shape == (100, 100), "Grayscale/Canny output should be 2D"

def test_preprocessImageNone():
    with pytest.raises(ValueError, match="Input Image is not found."):
        preprocessImage(None)

def test_findDocumentContour_NoDocument():
    # A blank image has no contours
    blank_edged = np.zeros((100, 100), dtype=np.uint8)
    dummy_original = np.zeros((100, 100, 3), dtype=np.uint8)
    result = documentContor(blank_edged, dummy_original)
    assert result is None, "Should return None if no 4-point contour is found"
