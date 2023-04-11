# SSD Image Classification Implementation

# Instead of VGG16 I am using MobileNet which is 23x smaller and has the same accuracy

import cv2
import numpy as np

# Minimum confidence threshold to appear as a recognized class
THRESHOLD = 0.5
# Suppression Threshold to remove repetitive boxes
SUPPRESSION_THRESHOLD = 0.3
# Image size
SSD_INPUT_SIZE = 320


'''
Parse through the class_names txt file and extract the possible classes line by line
'''
def construct_class_names(file_name='data/class_names'):
    with open(file_name, 'rt') as file:
        names = file.read().rstrip('\n').split('\n')

    return names


'''
Find show detected objects iterates through bounding boxes and draws the physical boxes with the label
around detected objects
'''
def show_detected_objects(img, boxes_to_keep, all_bounding_boxes, object_names, class_ids):
    for index in boxes_to_keep:
        box = all_bounding_boxes[index[0]]
        x, y, w, h = box[0], box[1], box[2], box[3]
        cv2.rectangle(img, (x, y), (x + w, y + h), color=(0, 0, 255), thickness=2)
        cv2.putText(img, object_names[class_ids[index[0]][0] - 1].upper(), (box[0], box[1] - 10),
                    cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.7, (0, 0, 255), 1)


# Class declaration
class_names = construct_class_names()

# read image
image = cv2.imread('data/car.jpg')

# Configure DNN using Darknet and model config files
neural_network = cv2.dnn_DetectionModel('ssd_weights.pb', 'ssd_mobilenet_coco_cfg.pbtxt')
# Initialize backend and CPU utilization
neural_network.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
neural_network.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
# Configure inputs for the network
neural_network.setInputSize(SSD_INPUT_SIZE, SSD_INPUT_SIZE)
neural_network.setInputScale(1.0/127.5)
neural_network.setInputMean((127.5, 127.5, 127.5))
neural_network.setInputSwapRB(True)

# Detect objects within image and return ids, confidence and bounding box
class_label_ids, confidences, bbox = neural_network.detect(image)
bbox = list(bbox)

# convert from 2D array to 1D array with single value
confidences = np.array(confidences).reshape(1, -1).tolist()[0]

# Non-Max Suppression to remove repetitive boxes
box_to_keep = cv2.dnn.NMSBoxes(bbox, confidences, THRESHOLD, SUPPRESSION_THRESHOLD)

# Use show detected objects function to return image with bounding box and labels
show_detected_objects(image, box_to_keep, bbox, class_names, class_label_ids)

# Show image and wait until key press to close image
cv2.imshow('SSD Algorithm', image)
cv2.waitKey()
