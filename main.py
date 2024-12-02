import cv2
import numpy as np
import tensorflow as tf
import time

# Load the pre-trained MobileNet SSD model
model = tf.saved_model.load("model/saved_model")  # Update path if needed
detect_fn = model.signatures['serving_default']

coco_labels = {1:"person",2:"bicycle", 3:"car", 4:"motorcycle", 6:"bus", 8:"truck"}

def go_forward():
    print("Go Forward")

def stop():
    print("Stop")
# Function to run object detection on a single frame
def detect_objects(frame):
    # Convert the frame to a tensor and prepare for detection
    input_tensor = tf.convert_to_tensor(frame)
    input_tensor = input_tensor[tf.newaxis, ...]
    
    # Run detection
    detections = detect_fn(input_tensor)
    
    # Extract detection results
    detection_scores = detections['detection_scores'][0].numpy()
    detection_classes = detections['detection_classes'][0].numpy().astype(np.int32)
    detection_boxes = detections['detection_boxes'][0].numpy()

    return detection_boxes, detection_classes, detection_scores

# Initialize the video capture for the webcam
cap = cv2.VideoCapture(0)  # Use 0 for default USB camera
print("Camera status", cap.isOpened())

# Start the FPS timer
fps_start_time = time.time()
frame_count = 0

person_flag = False
other_object_flag = False

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    person_flag = False
    other_object_flag = False

    frame_count += 1

    # Run object detection
    boxes, classes, scores = detect_objects(frame)
    lenght, height, person_area, other_object_area = 0, 0, 0, 0

    # Filter out low confidence scores
    for i in range(len(scores)):
        if int(classes[i]) in coco_labels and scores[i] >= 0.5:  # Adjust confidence threshold as needed
            # Draw bounding box
            print(f"class detected {coco_labels[int(classes[i])]}")
            box = boxes[i]
            h, w, _ = frame.shape
            ymin, xmin, ymax, xmax = box
            start_point = (int(xmin * w), int(ymin * h))
            end_point = (int(xmax * w), int(ymax * h))
            lenght = xmax - xmin
            height = ymax - ymin
            area_temp = lenght * height
            class_id = classes[i]
            if coco_labels[class_id] == "person" and area_temp > person_area:
                person_flag = True
                person_area = area_temp
                break
            elif coco_labels[class_id] != "person" and area_temp > other_object_area:
                other_object_flag = True
                other_object_area = area_temp
                
            cv2.rectangle(frame, start_point, end_point, (0, 255, 0), 2)
            
            # Display class label and confidence
            label = f"{coco_labels[class_id]}: {int(scores[i] * 100)}%"
            cv2.putText(frame, label, start_point, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    if person_flag and person_area > 0.5:
        stop()
    elif other_object_flag and 0.5 <= other_object_area <= 0.75:
        go_forward()
    elif other_object_flag and other_object_area > 0.75:
        stop()
    else:
        go_forward()

    # Calculate FPS
    fps_end_time = time.time()
    time_diff = fps_end_time - fps_start_time
    fps = frame_count / time_diff
    # Display the frame with detections
    fps_text = f"FPS: {fps:.2f}"
    cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv2.imshow("Object Detection", frame)

    # Break the loop on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
