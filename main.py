"""
Works as application executor or entry point for document scanning
"""

import cv2
import matplotlib.pyplot as plt
from pipeline import *

def runPipeline(imagePath:str):
    """
    Performs document scanning pipeline
    Manages visualization and errors
    """
    print(f"Processing :{imagePath}")

    try:
        # Initializes Scanner
        scanner= DocumentScanner(imagePath)

        # i. Image loading
        scanner.original_image = scanner.loadImage()
        original = cv2.cvtColor(scanner.original_image, cv2.COLOR_BGR2RGB) # for plt visualization

        # ii. Preprocessing Phase
        scanner.edged_image = scanner.preprocessImage(scanner.original_image)

        # iii. Contour finding Phase
    

        try:
            scanner.contour_points = scanner.documentContor(scanner.edged_image)
        except DocumentNotFoundError:
            print("Warning: Document boundary not found. Skipping transform.")
            scanner.wrapped_image = original
            scanner.contour_points = None

        if scanner.contour_points is not None:
                # Draw contour on original for visualization
                visualImage = original.copy()
                cv2.drawContours(visualImage, [scanner.contour_points], -1, (0, 255, 0), 2)

                # iv. Transform image
                scanner.wrapped_image = scanner.fourPointsTransformation(original, scanner.contour_points)

        # v. OCR
        scanner.extracted_text = scanner.extractText(scanner.wrapped_image)
        print("\n--- EXTRACTED TEXT ---")
        print(scanner.extracted_text)
        print("----------------------\n")

        # Visualization of Results
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(original)
        axes[0].set_title("1. Original Image")
        axes[0].axis('off')
        
        axes[1].imshow(scanner.edged_image, cmap='gray')
        axes[1].set_title("2. Edge Detection")
        axes[1].axis('off')
        
        axes[2].imshow(scanner.wrapped_image)
        axes[2].set_title("3. Scanned & Wrapped")
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig("PipelineVisualization.png", dpi=300) 
        plt.show()

    except ImageLoaderError as e:
        print(f"Image Error: {e}")
    except PipelineError as e:
        print(f"Pipeline Error: {e}")
    except ValueError as e:
        print(f"Value Error: {e}")
    except Exception as e:
        print(f"An unexpected Error happened : {e}")

if __name__ == "__main__":
    # Replace with your actual image path
    runPipeline("./images/1111.jpg") 


