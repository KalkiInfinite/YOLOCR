import os
from ultralytics import YOLO
import matplotlib.pyplot as plt
import cv2


# Load a trained model (if you want to load the model you trained earlier)
model = YOLO(os.path.join("C:/Users/Bhawesh/Desktop/TextD/runs part/train/weights/best.pt"))

# Path to the test image
test_image_path = os.path.join("E:/KJ Somaiya/Internship/road imgs/1.jpg")

# Make predictions
results = model(test_image_path)

# Print the results
print(results)

# Load the image with OpenCV
image = cv2.imread(test_image_path)

# Calculate the area of each bounding box and draw it on the image
for result in results:
    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # extract and convert coordinates to integers
        width = x2 - x1
        height = y2 - y1
        area = width * height
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)  # draw the bounding box
        cv2.putText(image, f"Area: {area}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 0, 0), 2)  # put the area text

# Display the image with matplotlib
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')  # Hide the axis
plt.show()
