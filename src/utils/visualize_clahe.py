from pathlib import Path
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root / 'src'))

# Load the image
image_path = project_root / 'datasets' / 'glaucoma-datasets' / \
    'REFUGE' / 'val' / 'Images_Cropped' / 'V0393.jpg'
original = cv2.imread(str(image_path))

if original is None:
    print("Error: Could not load image.")
    exit(1)

# Convert to LAB color space
lab = cv2.cvtColor(original, cv2.COLOR_BGR2LAB)

# Split the LAB channels
l, a, b = cv2.split(lab)

# Apply CLAHE to the L channel
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
l_clahe = clahe.apply(l)

# Merge the CLAHE enhanced L channel with the original A and B channels
lab_clahe = cv2.merge((l_clahe, a, b))

# Convert back to BGR color space
clahe_img = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)

# Convert BGR to RGB for matplotlib
original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
clahe_rgb = cv2.cvtColor(clahe_img, cv2.COLOR_BGR2RGB)

# Create subplots for side-by-side comparison
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Original image
axes[0].imshow(original_rgb)
axes[0].set_title('Original Image', fontsize=18)
axes[0].axis('off')

# CLAHE enhanced image
axes[1].imshow(clahe_rgb)
axes[1].set_title('CLAHE Enhanced Image', fontsize=18)
axes[1].axis('off')

# Save the figure
output_path = project_root / 'results' / 'V0393_clahe_comparison.png'
plt.savefig(output_path, bbox_inches='tight')
plt.close()

print(f"Saved comparison image as {output_path}")
