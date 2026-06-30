import cv2
import matplotlib.pyplot as plt
from pipeline import *

def runPipeline(imagePath:str):
    print(f"Processing :{imagePath}")

    # i. Image loading
    image = cv2.imread(imagePath)
    if image is None:
        print("Error: Could not load image. Please Check the path.")
        return
    
    original = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # plt display

    # ii. Preprocessing Phase
    edgedImage = preprocessImage(image)
    
    # iii. Contour finding Phase
    contoursPoints = documentContor(edgedImage, image)
    if contoursPoints is None:
        print("Warning: Document boundary not found. Skipping transform.")
        wrappedImage = original
    else:
        # Draw contour on original for visualization
        visualImage = original.copy()
        cv2.drawContours(visualImage, [contoursPoints], -1, (0, 255, 0), 2)
        # iv. Transform image
        wrappedImage = fourPointsTransformation(original, contoursPoints)

    # v. OCR
    text = extractText(wrappedImage)
    print("\n--- EXTRACTED TEXT ---")
    print(text)
    print("----------------------\n")

    # Visualization of Results
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(original)
    axes[0].set_title("1. Original Image")
    axes[0].axis('off')
    
    axes[1].imshow(edgedImage, cmap='gray')
    axes[1].set_title("2. Edge Detection")
    axes[1].axis('off')
    
    axes[2].imshow(wrappedImage)
    axes[2].set_title("3. Scanned & Wrapped")
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig("PipelineVisualization.png", dpi=300) 
    plt.show()

if __name__ == "__main__":
    # Replace with your actual image path
    runPipeline("./images/1111.jpg") 


