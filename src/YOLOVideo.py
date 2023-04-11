# Yolo Video Classification Implementation

import cv2
import numpy as np

# Minimum confidence threshold to appear as a recognized class
THRESHOLD = 0.5
# Suppression Threshold to remove repetitive boxes
SUPPRESSION_THRESHOLD = 0.3
# Image size
YOLO_IMAGE_SIZE = 320


'''
Find objects function iterates through model outputs and extracts the prediction and 
if the confidence threshold is passed the prediction x, y, w, h values are assigned.
Bounding box, class, and confidence were also assigned and then non-max suppression
is applied to remove repetitive bounding boxes
'''
def find_objects(model_outputs):
    bounding_box_locations = []
    class_ids = []
    confidence_values = []

    for output in model_outputs:
        for prediction in output:
            class_probabilities = prediction[5:]
            class_id = np.argmax(class_probabilities)
            confidence = class_probabilities[class_id]

            if confidence > THRESHOLD:
                # predictions -> x, y, w, h, conf, classes (80)
                w, h = int(prediction[2] * YOLO_IMAGE_SIZE), int(prediction[3] * YOLO_IMAGE_SIZE)
                # the center of the bounding box (we should transform these values)
                x, y = int(prediction[0] * YOLO_IMAGE_SIZE - w / 2), int(prediction[1] * YOLO_IMAGE_SIZE - h / 2)
                bounding_box_locations.append([x, y, w, h])
                class_ids.append(class_id)
                confidence_values.append(float(confidence))

    # Non-Max Suppression to remove repetitive boxes
    box_indexes_to_keep = cv2.dnn.NMSBoxes(bounding_box_locations, confidence_values, THRESHOLD, SUPPRESSION_THRESHOLD)

    return box_indexes_to_keep, bounding_box_locations, class_ids, confidence_values


'''
The show detected objects function iterates through bounding boxes and draws the physical boxes with the label
and confidence around detected objects
'''
def show_detected_objects(img, bounding_box_ids, all_bounding_boxes, class_ids, confidence_values, width_ratio,
                         height_ratio):
    for index in bounding_box_ids:
        bounding_box = all_bounding_boxes[index[0]]
        x, y, w, h = int(bounding_box[0]), int(bounding_box[1]), int(bounding_box[2]), int(bounding_box[3])
        # we have to transform the locations and coordinates because the resized image since the model
        # works primarily on 320x320 images however bounding box coordinates are for original image size
        # and will be overlayed on the original image
        x = int(x * width_ratio)
        y = int(y * height_ratio)
        w = int(w * width_ratio)
        h = int(h * height_ratio)

        # OpenCV deals with BGR blue green red (255,0,0) then it is the blue color
        # we are not going to detect every objects just PERSON and CAR
        if class_ids[index[0]] == 2:
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
            class_with_confidence = 'CAR' + str(int(confidence_values[index[0]] * 100)) + '%'
            cv2.putText(img, class_with_confidence, (x, y-10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.0, (0, 0, 255), 1)

        if class_ids[index[0]] == 0:
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
            class_with_confidence = 'PERSON' + str(int(confidence_values[index[0]] * 100)) + '%'
            cv2.putText(img, class_with_confidence, (x, y-10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.0, (0, 0, 255), 1)


# Capture the video as opposed to reading a image
capture = cv2.VideoCapture('data/pedestrians.mp4')

# Class declaration
classes = ['person', 'car', 'bus']

# Configure DNN using Darknet and model config files
neural_network = cv2.dnn.readNetFromDarknet('yolov3.cfg', 'yolov3.weights')
# Initialize backend and CPU utilization
neural_network.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
neural_network.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)


# exit only when video ends
while True:

    # after reading a frame (so basically one image) we just have to repeat the operation for that singular image
    frame_grabbed, frame = capture.read()

    # no more frames available (video ended)
    if not frame_grabbed:
        break

    # Grab original width and height of the video
    original_width, original_height = frame.shape[1], frame.shape[0]

    # Convert image into Blob for network input
    blob = cv2.dnn.blobFromImage(frame, 1 / 255, (YOLO_IMAGE_SIZE, YOLO_IMAGE_SIZE), True, crop=False)
    neural_network.setInput(blob)

    # Retrieve output layers and names
    layer_names = neural_network.getLayerNames()
    output_names = [layer_names[index[0] - 1] for index in neural_network.getUnconnectedOutLayers()]

    # Get outputs and use find_objects() and show_detected_images() functions to retrieve image returned
    # with box and prediction label
    outputs = neural_network.forward(output_names)
    predicted_objects, bbox_locations, class_label_ids, conf_values = find_objects(outputs)
    show_detected_objects(frame, predicted_objects, bbox_locations, class_label_ids, conf_values,
                         original_width / YOLO_IMAGE_SIZE, original_height / YOLO_IMAGE_SIZE)

    # Show image and wait until key press to close image
    cv2.imshow('YOLO Algorithm', frame)
    cv2.waitKey(1)

# After the video if complete release the capture and close all windows
capture.release()
cv2.destroyWindow()
