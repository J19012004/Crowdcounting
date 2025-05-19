import cv2
import numpy as np

# Load the input image or video
input_path = "crowdd.jpg"  # Replace with your image path
image = cv2.imread(input_path)

if image is None:
    print("Could not load image. Check the file path.")
    exit()

# Resize for consistency
image = cv2.resize(image, (800, 600))

# Convert to grayscale and blur
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (7, 7), 0)

# Thresholding
_, thresh = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY_INV)

# Morphological operations to reduce noise
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

# Find contours (each contour may represent a person or a group)
contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Filter out small contours
min_area = 100  # Adjust based on image scale
filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]

# Draw contours and count
for cnt in filtered_contours:
    x, y, w, h = cv2.boundingRect(cnt)
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

count = len(filtered_contours)
cv2.putText(image, f"Estimated Crowd: {count}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

# Show result
cv2.imshow("Crowd Count", image)
cv2.waitKey(0)
cv2.destroyAllWindows()