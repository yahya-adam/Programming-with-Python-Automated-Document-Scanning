import cv2
import numpy as np
import pytesseract
from typing import Tuple, Optional

# The 'r' before the quotes is very important for Windows file paths!
#pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'


def preprocessImage(image:np.ndarray) ->np.ndarray:
    """
    pre-processes images for edge identification.
    converts them to grayscale.
    applies Gaussian blur.
    Canny edge detection.
    """
    if image is None:
        raise ValueError("Input Image is not found. Please check the File Path")
    #Convert to grayscale
    grayImage = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    #Noise reduction by applying Gaussian blur
    blurredImage = cv2.GaussianBlur(grayImage, (5,5), 0)
    #Canny edge detection
    edgedImage = cv2.Canny(blurredImage, 75, 200)
    return edgedImage

def documentContor(image:np.ndarray, originalImage:np.ndarray)-> Optional[np.ndarray]:
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
    return None # Document not found
def fourPointsTransformation(image:np.ndarray, points:np.ndarray) ->np.ndarray:
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

def extractText(image:np.ndarray) ->str:
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
