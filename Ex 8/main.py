#Object detection with YOLOv3
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# ----------------------------------------------------
# 1. Load YOLOv3 model and class labels
# ----------------------------------------------------
cfg_path = r"C:\Users\lokes\OneDrive\Desktop\MSSQL16.MSSQLSERVER\FILES\Object Detection with YOLO3\yolov3.cfg"
weights_path = r"C:\Users\lokes\OneDrive\Desktop\MSSQL16.MSSQLSERVER\FILES\Object Detection with YOLO3\yolov3.weights"
names_path = r"C:\Users\lokes\OneDrive\Desktop\MSSQL16.MSSQLSERVER\FILES\Object Detection with YOLO3\coco.names"

# Verify files exist
for path in [cfg_path, weights_path, names_path]:
    if not os.path.exists(path):
        print(f"❌ File not found: {path}")

# Load model
net = cv2.dnn.readNet(weights_path, cfg_path)

# Load class labels
with open(names_path, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# ----------------------------------------------------
# 2. Load and preprocess image
# ----------------------------------------------------
image_path = r"C:\Users\lokes\OneDrive\Desktop\MSSQL16.MSSQLSERVER\FILES\Object Detection with YOLO3\images\dogandcat.jpg"
image = cv2.imread(image_path)
height, width = image.shape[:2]

blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
net.setInput(blob)

# ----------------------------------------------------
# 3. Get output layer names (version-independent)
# ----------------------------------------------------
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

# ----------------------------------------------------
# 4. Forward pass
# ----------------------------------------------------
outs = net.forward(output_layers)

# ----------------------------------------------------
# 5. Process predictions
# ----------------------------------------------------
conf_threshold = 0.5
nms_threshold = 0.4
boxes, confidences, class_ids = [], [], []

for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > conf_threshold:
            center_x, center_y, w, h = (detection[0:4] * np.array([width, height, width, height])).astype('int')
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)
            boxes.append([x, y, int(w), int(h)])
            confidences.append(float(confidence))
            class_ids.append(class_id)

# ----------------------------------------------------
# 6. Apply Non-Max Suppression
# ----------------------------------------------------
indexes = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

# ----------------------------------------------------
# 7. Draw Bounding Boxes + Bold Blue Text
# ----------------------------------------------------
font = cv2.FONT_HERSHEY_SIMPLEX
detected_objects = []

for i in range(len(boxes)):
    if i in indexes:
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        confidence = confidences[i]

        # Draw blue box
        color = (255, 0, 0)
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)

        # Draw bold blue text
        text = f"{label.upper()} {confidence*100:.1f}%"
        (text_w, text_h), baseline = cv2.getTextSize(text, font, 0.7, 2)
        cv2.rectangle(image, (x, y - text_h - 10), (x + text_w, y), color, -1)
        cv2.putText(image, text, (x, y - 5), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        detected_objects.append({
            "label": label,
            "confidence": confidence,
            "box": [x, y, w, h]
        })

# ----------------------------------------------------
# 8. Display Detected Object Details
# ----------------------------------------------------
if detected_objects:
    print("\n🟦 Detected Objects:")
    print("=" * 50)
    for obj in detected_objects:
        print(f"Object: {obj['label'].capitalize():15s} | "
              f"Confidence: {obj['confidence']*100:.2f}% | "
              f"Box: {obj['box']}")
else:
    print("No objects detected above confidence threshold.")

# ----------------------------------------------------
# 9. Compute Simple Evaluation Metrics
# ----------------------------------------------------
TP = len(detected_objects)
FP = len(boxes) - TP
FN = 0  # (no ground truth available)

precision = TP / (TP + FP + 1e-6)
recall = TP / (TP + FN + 1e-6)
f1 = 2 * (precision * recall) / (precision + recall + 1e-6)

print("\n📊 Evaluation Metrics (Approximation)")
print(f"Precision: {precision:.3f}")
print(f"Recall:    {recall:.3f}")
print(f"F1-Score:  {f1:.3f}")

# ----------------------------------------------------
# 10. Display Image Output
# ----------------------------------------------------
plt.figure(figsize=(10, 8))
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("YOLOv3 Object Detection with Confidence & Metrics")
plt.axis("off")
plt.show()
