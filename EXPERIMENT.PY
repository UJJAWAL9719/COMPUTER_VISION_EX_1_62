# ============================================================
# EXPERIMENT NO. 1
# Implementation of Fundamental Image Processing Operations
# using Python and OpenCV
# ============================================================

# Step 1: Import required libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


# ============================================================
# Step 2: Load the image
# ============================================================

# IMPORTANT:
# Change "image.jpg" to the exact name of your image.
# Example: "photo.png"
image_path = "image.jpeg"

# Check whether image file exists
if not os.path.exists(image_path):
    print("ERROR: Image file not found!")
    print("Current folder:", os.getcwd())
    print("Please make sure your image is in the same folder as EXPERIMENT.PY")
    print("Files available in this folder:")

    for file in os.listdir():
        print(" -", file)

    exit()

# Read image using OpenCV
image = cv2.imread(image_path)

# Check whether image was loaded successfully
if image is None:
    print("ERROR: OpenCV could not read the image.")
    exit()

print("Image loaded successfully!")
print("Image file:", image_path)


# ============================================================
# Step 3: Convert BGR to RGB
# ============================================================

# OpenCV reads images in BGR format.
# Matplotlib expects RGB format.
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


# ============================================================
# Step 4: Display Original Image
# ============================================================

# cv2.imshow() is NOT used because this experiment
# is being performed in GitHub Codespaces/browser.

plt.figure(figsize=(8, 6))
plt.imshow(image_rgb)
plt.title("Original Image")
plt.axis("off")
plt.show()


# ============================================================
# Step 5: Examine Image Properties
# ============================================================

height, width, channels = image.shape

print("\n========================================")
print("IMAGE PROPERTIES")
print("========================================")

print("Width        :", width)
print("Height       :", height)
print("Channels     :", channels)
print("Resolution   :", width, "x", height)
print("Data Type    :", image.dtype)
print("Total Pixels :", width * height)


# ============================================================
# Step 6: Save Image in JPEG and PNG Formats
# ============================================================

cv2.imwrite("output_image.jpg", image)
cv2.imwrite("output_image.png", image)

print("\n========================================")
print("IMAGE STORAGE")
print("========================================")

print("JPEG saved as : output_image.jpg")
print("PNG saved as  : output_image.png")


# ============================================================
# Step 7: Convert Image to Grayscale
# ============================================================

gray_image = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2GRAY
)

plt.figure(figsize=(8, 6))
plt.imshow(gray_image, cmap="gray")
plt.title("Grayscale Image")
plt.axis("off")
plt.show()


# ============================================================
# Step 8: Convert Image to HSV
# ============================================================

hsv_image = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2HSV
)

# Convert HSV to RGB for proper visualization
hsv_rgb = cv2.cvtColor(
    hsv_image,
    cv2.COLOR_HSV2RGB
)

plt.figure(figsize=(8, 6))
plt.imshow(hsv_rgb)
plt.title("HSV Image")
plt.axis("off")
plt.show()


# ============================================================
# Step 9: Convert Image to LAB
# ============================================================

lab_image = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2LAB
)

# Convert LAB back to RGB for visualization
lab_rgb = cv2.cvtColor(
    lab_image,
    cv2.COLOR_LAB2RGB
)

plt.figure(figsize=(8, 6))
plt.imshow(lab_rgb)
plt.title("LAB Image")
plt.axis("off")
plt.show()


# ============================================================
# Step 10: Resize Image
# ============================================================

# Resize image to 50% of original size

new_width = max(1, width // 2)
new_height = max(1, height // 2)

resized_image = cv2.resize(
    image,
    (new_width, new_height)
)

resized_rgb = cv2.cvtColor(
    resized_image,
    cv2.COLOR_BGR2RGB
)

plt.figure(figsize=(8, 6))
plt.imshow(resized_rgb)
plt.title("Resized Image - 50%")
plt.axis("off")
plt.show()


# ============================================================
# Step 11: Rotate Image
# ============================================================

# Find image center
center = (
    width // 2,
    height // 2
)

# Create rotation matrix
rotation_matrix = cv2.getRotationMatrix2D(
    center,
    45,
    1.0
)

# Rotate image by 45 degrees
rotated_image = cv2.warpAffine(
    image,
    rotation_matrix,
    (width, height)
)

rotated_rgb = cv2.cvtColor(
    rotated_image,
    cv2.COLOR_BGR2RGB
)

plt.figure(figsize=(8, 6))
plt.imshow(rotated_rgb)
plt.title("Rotated Image - 45 Degrees")
plt.axis("off")
plt.show()


# ============================================================
# Step 12: Horizontal Flip
# ============================================================

horizontal_flip = cv2.flip(
    image,
    1
)

horizontal_flip_rgb = cv2.cvtColor(
    horizontal_flip,
    cv2.COLOR_BGR2RGB
)

plt.figure(figsize=(8, 6))
plt.imshow(horizontal_flip_rgb)
plt.title("Horizontal Flip")
plt.axis("off")
plt.show()


# ============================================================
# Step 13: Vertical Flip
# ============================================================

vertical_flip = cv2.flip(
    image,
    0
)

vertical_flip_rgb = cv2.cvtColor(
    vertical_flip,
    cv2.COLOR_BGR2RGB
)

plt.figure(figsize=(8, 6))
plt.imshow(vertical_flip_rgb)
plt.title("Vertical Flip")
plt.axis("off")
plt.show()


# ============================================================
# Step 14: Image Complement / Negative Image
# ============================================================

# Negative = 255 - Original Image

negative_image = 255 - image

negative_rgb = cv2.cvtColor(
    negative_image,
    cv2.COLOR_BGR2RGB
)

plt.figure(figsize=(8, 6))
plt.imshow(negative_rgb)
plt.title("Negative / Complement Image")
plt.axis("off")
plt.show()


# ============================================================
# Step 15: Crop Region of Interest (ROI)
# ============================================================

# Define ROI coordinates
# The coordinates automatically adjust according to image size.

x1 = 0
y1 = 0

x2 = min(300, width)
y2 = min(300, height)

roi = image[
    y1:y2,
    x1:x2
]

# Check ROI
if roi.size == 0:
    print("ROI could not be created.")
    exit()

roi_rgb = cv2.cvtColor(
    roi,
    cv2.COLOR_BGR2RGB
)

plt.figure(figsize=(6, 5))
plt.imshow(roi_rgb)
plt.title("Region of Interest (ROI)")
plt.axis("off")
plt.show()


# ============================================================
# Step 16: Display Original and Processed Images Together
# ============================================================

plt.figure(figsize=(16, 10))

# Original
plt.subplot(2, 4, 1)
plt.imshow(image_rgb)
plt.title("Original")
plt.axis("off")

# Grayscale
plt.subplot(2, 4, 2)
plt.imshow(gray_image, cmap="gray")
plt.title("Grayscale")
plt.axis("off")

# Resized
plt.subplot(2, 4, 3)
plt.imshow(resized_rgb)
plt.title("Resized")
plt.axis("off")

# Rotated
plt.subplot(2, 4, 4)
plt.imshow(rotated_rgb)
plt.title("Rotated")
plt.axis("off")

# Horizontal Flip
plt.subplot(2, 4, 5)
plt.imshow(horizontal_flip_rgb)
plt.title("Horizontal Flip")
plt.axis("off")

# Vertical Flip
plt.subplot(2, 4, 6)
plt.imshow(vertical_flip_rgb)
plt.title("Vertical Flip")
plt.axis("off")

# Negative
plt.subplot(2, 4, 7)
plt.imshow(negative_rgb)
plt.title("Negative")
plt.axis("off")

# ROI
plt.subplot(2, 4, 8)
plt.imshow(roi_rgb)
plt.title("ROI")
plt.axis("off")

plt.tight_layout()

# Save comparison image
plt.savefig(
    "all_operations.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# Step 17: Save All Processed Images
# ============================================================

cv2.imwrite(
    "grayscale.jpg",
    gray_image
)

cv2.imwrite(
    "resized.jpg",
    resized_image
)

cv2.imwrite(
    "rotated.jpg",
    rotated_image
)

cv2.imwrite(
    "horizontal_flip.jpg",
    horizontal_flip
)

cv2.imwrite(
    "vertical_flip.jpg",
    vertical_flip
)

cv2.imwrite(
    "negative.jpg",
    negative_image
)

cv2.imwrite(
    "roi.jpg",
    roi
)


# ============================================================
# Step 18: Basic Analysis of Grayscale Image
# ============================================================

minimum_intensity = np.min(gray_image)
maximum_intensity = np.max(gray_image)
mean_intensity = np.mean(gray_image)

print("\n========================================")
print("GRAYSCALE IMAGE ANALYSIS")
print("========================================")

print("Shape              :", gray_image.shape)
print("Data Type          :", gray_image.dtype)
print("Minimum Intensity  :", minimum_intensity)
print("Maximum Intensity  :", maximum_intensity)
print("Mean Intensity     :", mean_intensity)


# ============================================================
# Step 19: ROI Properties
# ============================================================

roi_height, roi_width, roi_channels = roi.shape

print("\n========================================")
print("ROI PROPERTIES")
print("========================================")

print("ROI Width     :", roi_width)
print("ROI Height    :", roi_height)
print("ROI Channels  :", roi_channels)
print("ROI Shape     :", roi.shape)


# ============================================================
# Step 20: Final Message
# ============================================================

print("\n========================================")
print("EXPERIMENT COMPLETED SUCCESSFULLY!")
print("========================================")

print("\nGenerated files:")

print("1. output_image.jpg")
print("2. output_image.png")
print("3. grayscale.jpg")
print("4. resized.jpg")
print("5. rotated.jpg")
print("6. horizontal_flip.jpg")
print("7. vertical_flip.jpg")
print("8. negative.jpg")
print("9. roi.jpg")
print("10. all_operations.png")

print("\nAll fundamental image processing operations")
print("have been implemented successfully.")