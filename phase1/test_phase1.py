import cv2
import matplotlib.pyplot as plt
from phase1.detector.grounding_dino_detector import GroundingDINODetector
from phase1.utils.visualization import draw_boxes

# Load image
image_path = "test.jpg"  # tree / glass / makeup brush
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Initialize detector
detector = GroundingDINODetector()

# Detect objects
detections = detector.detect(
    image=image,
    prompt="an object",
    box_threshold=0.3
)

# Draw results
annotated = draw_boxes(image, detections)

# Show
plt.figure(figsize=(10, 10))
plt.imshow(annotated)
plt.axis("off")
plt.show()

print(f"Detected {len(detections)} objects")
