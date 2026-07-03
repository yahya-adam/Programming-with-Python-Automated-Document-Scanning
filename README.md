# 📄 Automated Document Scanner & OCR Pipeline

An intelligent computer vision pipeline that automatically detects document boundaries in images, applies perspective transformation to get a top-down view, and extracts text using Optical Character Recognition (OCR). 
Built with **OpenCV**, **NumPy**, and **Tesseract**, this project simulates the core functionality of mobile document scanning apps like CamScanner or Microsoft Lens.

## Table of Contents
- [Overview of Project](#Overview-of-Project)
  - [Features](#Features)
  - [The Pipeline Workflow](#The-Pipeline-Workflow)
  - [Prerequisites](#Prerequisites)
  - [Installation](#Installation)
- [Install Python dependencies](#Install-Python-dependencies)
- [Create and activate a virtual environment (Recommended)](#create-and-activate-a-virtual-environment-recommended)
- [Configure Tesseract Path](#Configure-Tesseract-Path)
- [Usage](#Usage)
  - [Running the Pipeline](#Running-the-Pipeline)
  - [Running Unit Tests](#Running-Unit-Tests)
- [Project Structure](#Project-Structure)
- [Troubleshooting](#Troubleshooting)
- [License](#License)
- [Contributing](#Contributing)

## Overview of Project
### Features

- **Automated Edge Detection:** Uses Gaussian Blur and Canny Edge Detection to isolate document boundaries.
- **Smart Contour Finding:** Identifies the largest 4-point polygon representing the document.
- **Perspective Transformation:** Warps the detected document into a flat, top-down "bird's-eye" view.
- **Optical Character Recognition (OCR):** Extracts text from the scanned document using Tesseract.
- **Visual Pipeline Output:** Generates a side-by-side comparison image showing the Original, Edges, and Final Scanned result.
- **Unit Tested:** Includes `pytest` suites to ensure core preprocessing and contour logic remain robust.

---
### The Pipeline Workflow

1. **Image Loading:** Reads the input image.
2. **Preprocessing:** Converts to Grayscale ➔ Applies Gaussian Blur (Noise Reduction) ➔ Canny Edge Detection.
3. **Contour Detection:** Finds all contours, sorts by area, and approximates polygons to find the 4 corners of the document.
4. **Perspective Transform:** Calculates the transformation matrix and warps the image to a flat rectangle.
5. **OCR Extraction:** Applies Otsu's Thresholding for contrast and passes the image to Tesseract for text extraction.

---
### Prerequisites

Before installing the Python packages, you **must** install the Tesseract OCR engine on your system, as `pytesseract` is just a Python wrapper.

- **Windows:** Download the installer from [UB-Mannheim Tesseract](https://github.com/UB-Mannheim/tesseract/wiki).
- **macOS:** `brew install tesseract`
- **Linux (Ubuntu/Debian):** `sudo apt install tesseract-ocr`

---
### Installation

1. **Clone the repository:**
```bash
   git clone `https://github.com/your-username/document-scanner-pipeline.git`
   cd document-scanner-pipeline
```


## Create and activate a virtual environment (Recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```
----------

## Install Python dependencies
--------
  ```bash
pip install -r requirements.txt
  ```
--------

## Configure Tesseract Path
Crucial for Windows users: Open `pipeline.py` and ensure the Tesseract executable path matches your installation:
```Python
- # Change this path if Tesseract is installed elsewhere
- pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'
```
- (Note: macOS and Linux users can usually comment out or delete this line, as Tesseract is added to the system PATH automatically).

## Usage
### Running the Pipeline
- 1- Place an image of a document (e.g., a receipt, paper, or ID) inside an `images/` folder.
- 2- Open `main.py` and update the image path at the bottom of the file:
- Python
if __name__ == "__main__":
    runPipeline("images/your_image.jpg") # Update this path
- 3- Run the script:
```bash
python main.py
```
- 4- Output: 
-   The extracted text will be printed to the console.
-   A visualization image named `PipelineVisualization.png` will be saved in the root directory showing the Original, Edge Detection, and - - Scanned/Wrapped results.

### Running Unit Tests
- The project includes tests to verify the preprocessing and contour detection logic. Run them using pytest:
```bash
pytest test_pipeline.py -v
```
## Project Structure
- ├── main.py                 # Entry point: Orchestrates the pipeline and visualization
- ├── pipeline.py             # Core logic: Preprocessing, Contours, Transform, OCR
- ├── test_pipeline.py        # Unit tests for the pipeline functions
- ├── requirements.txt        # Python dependencies
- ├── images/                 # Folder for input images (Create this yourself)
- └── PipelineVisualization.png # Auto-generated output visualization

## Troubleshooting
|Issue                        |Solution
|-----------------------------|---------------------------------------------------------------------------------------------------------|
|`TesseractNotFoundError`     |Ensure Tesseract OCR is installed on your OS and the path in `pipeline.py` is correct.                   |
|`Document boundary not found`|The Canny edge parameters `(75, 200)` might need tweaking for your specific image lighting. Ensure the document contrasts well with the background.|
|Poor OCR Text Extraction     |Ensure the final wrapped image is clear. You can adjust the Tesseract config in `extractText()` (e.g., changing `--psm 6` to `--psm 3` for fully automatic page segmentation).|

## License
This project is open-source and available under the `MIT License`.

## Contributing
Contributions, issues, and feature requests are welcome! Feel free to check the issues page or submit a pull request to improve the edge detection algorithms or add support for multi-page PDF scanning.

