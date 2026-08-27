import cv2
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# EXPERIMENT NO. 2
# Contrast Enhancement and Histogram-Based Image Processing
# using Python and OpenCV
# ============================================================


# ------------------------------------------------------------
# 1. Load Image
# ------------------------------------------------------------

image_path = "GULL.jpg"

img = cv2.imread(image_path)

if img is None:
    print("Error: GULL.jpg not found!")
    print("Please make sure GULL.jpg is in the same folder as EXPERIMENT2.PY")
    exit()

print("GULL.jpg loaded successfully!")


# ------------------------------------------------------------
# 2. Convert Image to Grayscale
# ------------------------------------------------------------

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# ------------------------------------------------------------
# 3. Display Original and Grayscale Image
# ------------------------------------------------------------

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("Original GULL Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(gray, cmap="gray")
plt.title("Grayscale Image")
plt.axis("off")

plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# 4. Original Image Histogram
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

plt.hist(
    gray.ravel(),
    bins=256,
    range=[0, 256]
)

plt.title("Histogram of Original Image")
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency")

plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# 5. Contrast Stretching
# ------------------------------------------------------------

min_val = np.min(gray)
max_val = np.max(gray)

if max_val != min_val:
    stretched = (
        (gray - min_val)
        / (max_val - min_val)
        * 255
    ).astype(np.uint8)
else:
    stretched = gray.copy()


# ------------------------------------------------------------
# 6. Histogram Equalization
# ------------------------------------------------------------

equalized = cv2.equalizeHist(gray)


# ------------------------------------------------------------
# 7. CLAHE
# ------------------------------------------------------------

clahe = cv2.createCLAHE(
    clipLimit=2.0,
    tileGridSize=(8, 8)
)

clahe_image = clahe.apply(gray)


# ------------------------------------------------------------
# 8. Display Enhanced Images
# ------------------------------------------------------------

plt.figure(figsize=(16, 5))

plt.subplot(1, 4, 1)
plt.imshow(gray, cmap="gray")
plt.title("Original")
plt.axis("off")

plt.subplot(1, 4, 2)
plt.imshow(stretched, cmap="gray")
plt.title("Contrast Stretching")
plt.axis("off")

plt.subplot(1, 4, 3)
plt.imshow(equalized, cmap="gray")
plt.title("Histogram Equalization")
plt.axis("off")

plt.subplot(1, 4, 4)
plt.imshow(clahe_image, cmap="gray")
plt.title("CLAHE")
plt.axis("off")

plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# 9. Compare Histograms
# ------------------------------------------------------------

plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
plt.hist(
    gray.ravel(),
    bins=256,
    range=[0, 256]
)
plt.title("Original Histogram")
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency")


plt.subplot(2, 2, 2)
plt.hist(
    stretched.ravel(),
    bins=256,
    range=[0, 256]
)
plt.title("Contrast Stretching Histogram")
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency")


plt.subplot(2, 2, 3)
plt.hist(
    equalized.ravel(),
    bins=256,
    range=[0, 256]
)
plt.title("Histogram Equalization")
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency")


plt.subplot(2, 2, 4)
plt.hist(
    clahe_image.ravel(),
    bins=256,
    range=[0, 256]
)
plt.title("CLAHE Histogram")
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency")

plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# 10. Save Enhanced Images
# ------------------------------------------------------------

cv2.imwrite(
    "GULL_grayscale.jpg",
    gray
)

cv2.imwrite(
    "GULL_contrast_stretched.jpg",
    stretched
)

cv2.imwrite(
    "GULL_histogram_equalized.jpg",
    equalized
)

cv2.imwrite(
    "GULL_CLAHE.jpg",
    clahe_image
)


# ------------------------------------------------------------
# 11. Final Output
# ------------------------------------------------------------

print()
print("=" * 60)
print("EXPERIMENT 2 COMPLETED SUCCESSFULLY")
print("=" * 60)

print("Input Image: GULL.jpg")
print()
print("Generated Images:")
print("1. GULL_grayscale.jpg")
print("2. GULL_contrast_stretched.jpg")
print("3. GULL_histogram_equalized.jpg")
print("4. GULL_CLAHE.jpg")
print()
print("All contrast enhancement operations completed successfully!")