"""
Tests successful image loading.
Tests image loading failure raises ImageLoadError.
Tests preprocessing with a valid dummy image.
Tests contour finding when no document is present.
"""
import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from pipeline import *

class TestBaseImageProcessor:
    """Tests for the BaseImageProcessor class."""

    @patch('pipeline.cv2.imread')
    def test_loadImage_success(self, mock_imread):
        """Tests successful image loading."""
        mock_imread.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
        processor = BaseImageProcessor("dummy_path.jpg")
        img = processor.loadImage()
        assert img.shape == (100, 100, 3)

    @patch('pipeline.cv2.imread')
    def test_loadImage_failure(self, mock_imread):
        """Tests image loading failure raises ImageLoadError."""
        mock_imread.return_value = None
        processor = BaseImageProcessor("invalid_path.jpg")
        with pytest.raises(ImageLoaderError, match="Could not load image"):
            processor.loadImage()

    def test_preprocessImage_valid(self):
        """Tests preprocessing with a valid dummy image."""
        processor = BaseImageProcessor("dummy.jpg")
        dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)
        result = processor.preprocessImage(dummy_image)
        # Canny edge detection outputs a 2D grayscale image
        assert result.shape == (100, 100), "Output should be 2D"
        assert len(result.shape) == 2

    def test_preprocessImage_none(self):
        """Tests preprocessing with None raises ValueError."""
        processor = BaseImageProcessor("dummy.jpg")
        with pytest.raises(ValueError, match="Input Image is None."):
            processor.preprocessImage(None)


class TestDocumentScanner:
    """Tests for the DocumentScanner class (Inheritance)."""

    def test_inheritance(self):
        """Verifies that DocumentScanner inherits from BaseImageProcessor."""
        scanner = DocumentScanner("dummy.jpg")
        assert isinstance(scanner, BaseImageProcessor)

    def test_documentContor_no_document(self):
        """Tests contour finding when no document is present."""
        scanner = DocumentScanner("dummy.jpg")
        blank_edged = np.zeros((100, 100), dtype=np.uint8)
        
        with pytest.raises(DocumentNotFoundError, match="Document boundary"):
            scanner.documentContor(blank_edged)

    @patch('pipeline.cv2.findContours')
    def test_documentContor_success(self, mock_findContours):
        """Tests successful contour finding using mocked contours."""
        scanner = DocumentScanner("dummy.jpg")
        
        # Create a mock 4-point contour
        mock_contour = np.array([[[10, 10]], [[90, 10]], [[90, 90]], [[10, 90]]], dtype=np.int32)
        mock_findContours.return_value = ([mock_contour], None)
        
        dummy_edged = np.zeros((100, 100), dtype=np.uint8)
        result = scanner.documentContor(dummy_edged)
        
        assert result is not None
        assert result.shape == (4, 2)

    def test_fourPointsTransformation(self):
        """Tests perspective transformation output shape."""
        scanner = DocumentScanner("dummy.jpg")
        
        # Dummy image and 4 corner points
        dummy_image = np.zeros((200, 200, 3), dtype=np.uint8)
        points = np.array([[10, 10], [190, 10], [190, 190], [10, 190]], dtype="float32")
        
        result = scanner.fourPointsTransformation(dummy_image, points)
        
        # The transformed image should be roughly 180x180 based on the points
        assert result.shape[0] > 0 and result.shape[1] > 0
        assert len(result.shape) == 3 # Should be a 3-channel color image

    @patch('pipeline.pytesseract.image_to_string')
    def test_extractText(self, mock_tesseract):
        """Tests OCR text extraction using mocked Tesseract."""
        scanner = DocumentScanner("dummy.jpg")
        mock_tesseract.return_value = "  Hello World  "
        
        dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)
        result = scanner.extractText(dummy_image)
        
        # Assert that it strips whitespace as per the original logic
        assert result == "Hello World"
        mock_tesseract.assert_called_once()