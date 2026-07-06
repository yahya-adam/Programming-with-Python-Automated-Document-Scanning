"""
Contains core logic of document scanning, it preprocesses images,
identifies the boundary, run perspective transforamtion, and extracts text.
"""

import cv2
import numpy as np
import pytesseract
from typing import Tuple, Optional
import shutil


# user-defined exceptions
class PipelineError(Exception):
    """Exception for document scanning pipeline"""
    pass

class ImageLoaderError(PipelineError):
    """When an image can not be loaded from path, it will be raised"""
    pass

class DocumentNotFoundError(PipelineError):
    """When document contour can not be found in the image, it will be raised"""
    pass

class BaseImageProcessor():
    """
    Handles image loading and basic preprocessing of image
    """
    def __init__(self, imagePath:str):

        """ initializes the processor with an image path and set up tesseract """

        self.imagePath= imagePath
        self.original_image= None
        self._setup_tesseract()
    
    def _setup_tesseract(self):
        """In case of cross-platform, it configures Tesseract OCR automatically"""
        tesseract_path = shutil.which('tesseract')
        if tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
        else:
            #Fallback to Windows path if not found in PATH
            pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'

    def loadImage(self) ->np.ndarray:
        """
        1- Loads the image from initialized file path.
        2- Returns np.ndarray
        3- Raises ImageLoaderError if the image is not found or can not be loaded
        """
        image = cv2.imread(self.imagePath)
        if image is None:
            raise ImageLoaderError(f"Could not load image at '{self.imagePath}'. Please check the file path.")
        return image

    def preprocessImage(self, image:np.ndarray) ->np.ndarray:
        """
        pre-processes images for edge identification.
        converts them to grayscale.
        applies Gaussian blur.
        Canny edge detection.
        """
        if image is None:
            raise ValueError("Input Image is None. Please provide a valid numpy array.")
        #Convert to grayscale
        grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        #Noise reduction by applying Gaussian blur
        blurredImage = cv2.GaussianBlur(grayImage, (5,5), 0)
        #Canny edge detection
        edgedImage = cv2.Canny(blurredImage, 75, 200)
        return edgedImage
    

class DocumentScanner(BaseImageProcessor):
    """
    DocumentScanner inherits from BaseImageProcessor.
    It extends the base functionality to include document contour detection, 
    perspective transformation, and Optical Character Recognition (OCR).
    """
    def __init__(self, imagePath: str):
        """
        Initializes the DocumentScanner.
        
        Args:
            image_path (str): Path to the input image.
        """
        super().__init__(imagePath)
        self.edged_image = None
        self.contour_points = None
        self.wrapped_image = None
        self.extracted_text = ""

    def documentContor(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Finds largest four-sided contour of edgedImage.
        Return four cornors points if there, otherwise None.
        """
        contours, _ = cv2.findContours(image.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        #Sort contours by area, largest first
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

        # Sort contours by area, largest first
        for contour in contours:
            perimeter = cv2.arcLength(contour, True)
            #Approximation of contour to a polygon
            approximation = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
            # approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

            #Assume it's document, if approximated contour has four points
            if len(approximation) == 4:
                return approximation.reshape(4, 2)
            
        raise DocumentNotFoundError("Document boundary (4-point contour) not found in the image.")
    
    def fourPointsTransformation(self, image:np.ndarray, points:np.ndarray) -> np.ndarray:
        """
        Performs a perspective transform to get a top-down view of the document.
        """
        #Oder points(e.g., top-left, top-right, bottom-right, bottom-left)
        rectangular = np.zeros((4, 2), dtype="float32")
        summation = points.sum(axis=1)
        rectangular[0] = points[np.argmin(summation)]
        rectangular[2] = points[np.argmax(summation)]
        difference = np.diff(points, axis=1)
        rectangular[1] = points[np.argmin(difference)]
        rectangular[3] = points[np.argmax(difference)]

        (topLeft,topRight,bottomLeft,bottomRight) = rectangular
        #Calculate width and height of a new image
        width1 = np.linalg.norm(bottomRight - bottomLeft)
        width2 = np.linalg.norm(topRight - topLeft)
        maxWidth = max(int(width1), int(width2))

        height1 = np.linalg.norm(topRight - bottomRight)
        height2 = np.linalg.norm(topLeft - topRight)
        maxHeight = max(int(height1), int(height2))
        
        destination = np.array([
            [0,0],
            [maxWidth -1, 0],
            [maxWidth -1, maxHeight -1],
            [0, maxHeight -1]], dtype="float32"
        )
        #claculate the perspective transform matrix and apply it
        matrix = cv2.getPerspectiveTransform(rectangular, destination)
        wrapped = cv2.warpPerspective(image, matrix, (maxWidth, maxHeight))
        return wrapped

    def extractText(self, image:np.ndarray) -> str:
        """
        Applies Optical Character Recognition (OCR) to a wrapped image.
        """
        #Apply thresholding to make text pop (Otsu's method): optional
        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

        #Configure Tesseract for better accuracy
        customConfiguration = r'--oem 3 --psm 6'
        text = pytesseract.image_to_string(threshold, config=customConfiguration)
        return text.strip()
