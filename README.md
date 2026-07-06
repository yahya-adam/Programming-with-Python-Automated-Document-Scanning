# 📄 Automated Document Scanner & OCR Pipeline

An intelligent computer vision pipeline that automatically detects document boundaries in images, applies perspective transformation to get a top-down view, and extracts text using Optical Character Recognition (OCR). 
Built with **OpenCV**, **NumPy**, and **Tesseract**, this project simulates the core functionality of mobile document scanning apps like CamScanner or Microsoft Lens.

## Table of Contents
- [Smart Document Scanner & OCR Pipeline](#Smart-Document-Scanner-&-OCR-Pipeline)
- [Features](#Features)
- [Architecture & Design](#Architecture-&-Design)
- [Quick Start(Google Colab Setup)](#Quick-Start(Google-Colab-Setup))
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

## Smart Document Scanner & OCR Pipeline

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.5+-green.svg)](https://opencv.org/)
[![Pytest](https://img.shields.io/badge/tested%20with-pytest-0a9ed1.svg)](https://docs.pytest.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A robust, Object-Oriented Document Scanner and Optical Character Recognition (OCR) pipeline. This project takes a photograph of a document, automatically detects its boundaries, applies a perspective transform to get a top-down view, and extracts the text using Tesseract OCR.

---

## Features

* **🖼️ Automated Document Detection:** Uses Canny Edge Detection and Contour Approximation to find the 4 corners of a document.
* **📐 Perspective Transformation:** Mathematically warps the detected document into a flat, top-down "scanned" view.
* **🔤 Optical Character Recognition (OCR):** Extracts text from the scanned image using Tesseract.
* **🏗️ Object-Oriented Architecture:** Refactored from procedural scripts into a clean, extensible class hierarchy (`BaseImageProcessor` -> `DocumentScanner`).
* **🛡️ Robust Error Handling:** Implements custom exception hierarchies (`PipelineError`, `ImageLoadError`, `DocumentNotFoundError`) for graceful failure management.
* **🧪 Fully Unit-Tested:** Comprehensive `pytest` suite using `unittest.mock` to ensure reliability without needing external files or engines during testing.

---

## Architecture & Design

This project strictly adheres to modern Software Engineering principles:

1. **Inheritance & Encapsulation:** 
   * `BaseImageProcessor` handles fundamental tasks (loading, grayscale conversion, edge detection).
   * `DocumentScanner` inherits from the base class and extends it with domain-specific logic (contour finding, perspective math, OCR).
2. **Custom Exceptions:** Instead of generic crashes, the pipeline raises specific exceptions (e.g., `DocumentNotFoundError`), allowing the `main.py` execution flow to handle missing documents gracefully.
3. **State Management:** Attributes like `self.original_image` are initialized as `None` to represent "unloaded" states, while `self.extracted_text` is initialized as `""` to ensure type safety for string operations.

---

## Quick Start(Google Colab Setup)

Want to try it without installing anything locally? Google Colab is perfect for this! Since Colab runs on Linux, setting up Tesseract requires a quick system command.

**1. Open a new Colab Notebook** and run the following cell to install dependencies:
```python
# Install essential libraries
!pip install -r requirements.txt

# Run pytest
!pytest test_pipeline.py -v

# Run entry point `main.py`
!python main.py
```

---
## Installation

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
1- Place an image of a document (e.g., a receipt, paper, or ID) inside an `images/` folder.
* 2- Open `main.py` and update the image path at the bottom of the file:
```Python
if __name__ == "__main__":
    runPipeline("images/your_image.jpg") # Update this path
```
3- Run the script:
```bash
python main.py
```
4- Output: 
*   The extracted text will be printed to the console.
*   A visualization image named `PipelineVisualization.png` will be saved in the root directory showing the Original, Edge Detection, and - - Scanned/Wrapped results.

### Running Unit Tests
* The project includes tests to verify the preprocessing and contour detection logic. Run them using pytest:
```bash
pytest test_pipeline.py -v
```
#### Expected Output:
```text
test_pipeline.py::TestBaseImageProcessor::test_loadImage_success PASSED
test_pipeline.py::TestBaseImageProcessor::test_loadImage_failure PASSED
test_pipeline.py::TestBaseImageProcessor::test_preprocessImage_valid PASSED
test_pipeline.py::TestDocumentScanner::test_inheritance PASSED
test_pipeline.py::TestDocumentScanner::test_documentContour_no_document PASSED
...
```
## Project Structure
* ├── main.py                 # Entry point: Orchestrates the pipeline and visualization
* ├── pipeline.py             # Core logic: Preprocessing, Contours, Transform, OCR
* ├── test_pipeline.py        # Unit tests for the pipeline functions
* ├── requirements.txt        # Python dependencies
* ├── images/                 # Folder for input images (Create this yourself)
* └── PipelineVisualization.png # Auto-generated output visualization

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

