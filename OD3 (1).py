import cv2
import numpy as np
import urllib.request
import os
import time

# Configuration
url = 'http://192.168.43.219/cam-hi.jpg'
whT = 320
confThreshold = 0.5
nmsThreshold = 0.3

# Function to get output layer names
def get_output_layers(net):
    layerNames = net.getLayerNames()
    return [layerNames[i-1] for i in net.getUnconnectedOutLayers().flatten()]

# Load class names
classesfile = r"C:\Users\amrit\Downloads\coco.names"
if not os.path.exists(classesfile):
    raise FileNotFoundError(f"{classesfile} not found! Download it first.")

with open(classesfile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

# Load YOLO model
modelConfig = r"C:\Users\amrit\Downloads\yolov3.cfg"
modelWeights = r"C:\Users\amrit\Downloads\yolov3.weights"

if not os.path.exists(modelConfig):
    raise FileNotFoundError(f"{modelConfig} not found! Download from GitHub.")
if not os.path.exists(modelWeights):
    raise FileNotFoundError(f"{modelWeights} not found! Download from pjreddie.com.")

net = cv2.dnn.readNetFromDarknet(modelConfig, modelWeights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# ===== TEST WITH LAPTOP WEBCAM =====
print("Testing with laptop webcam (press 'q' to continue to ESP32-CAM)...")
cap = cv2.VideoCapture(0)  # 0 for default camera

while True:
    success, im = cap.read()
    if not success:
        print("Failed to read from webcam")
        break
    
    blob = cv2.dnn.blobFromImage(im, 1/255, (whT,whT), [0,0,0], 1, crop=False)
    net.setInput(blob)
    outputNames = get_output_layers(net)
    outputs = net.forward(outputNames)
    
    # Process detections
    hT, wT, cT = im.shape
    bbox = []
    classIds = []
    confs = []
    
    for output in outputs:
        for det in output:
            scores = det[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                w, h = int(det[2]*wT), int(det[3]*hT)
                x, y = int((det[0]*wT)-w/2), int((det[1]*hT)-h/2)
                bbox.append([x, y, w, h])
                classIds.append(classId)
                confs.append(float(confidence))
    
    indices = cv2.dnn.NMSBoxes(bbox, confs, confThreshold, nmsThreshold)
    
    for i in indices:
        box = bbox[i]
        x, y, w, h = box[0], box[1], box[2], box[3]
        cv2.rectangle(im, (x, y), (x+w, y+h), (255, 0, 255), 2)
        cv2.putText(im, f'{classNames[classIds[i]].upper()} {int(confs[i]*100)}%',
                   (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
    
    cv2.imshow('Webcam Test - Press Q to continue', im)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Webcam test completed. Proceeding to ESP32-CAM connection...")

# ===== MAIN CAMERA LOOP (ORIGINAL ESP32-CAM CODE) =====
def get_camera_frame():
    try:
        req = urllib.request.Request(
            url,
            headers={'User-Agent': 'Mozilla/5.0'},
            timeout=5
        )
        with urllib.request.urlopen(req) as resp:
            imgnp = np.array(bytearray(resp.read()), dtype=np.uint8)
            return cv2.imdecode(imgnp, -1)
    except Exception as e:
        print(f"Camera error: {str(e)}")
        return None

while True:
    im = get_camera_frame()
    if im is None:
        print("Waiting for camera connection...")
        time.sleep(1)
        continue
    
    # Object detection
    blob = cv2.dnn.blobFromImage(im, 1/255, (whT,whT), [0,0,0], 1, crop=False)
    net.setInput(blob)
    outputNames = get_output_layers(net)
    outputs = net.forward(outputNames)
    
    # Rest of your detection processing...
    hT, wT, cT = im.shape
    bbox = []
    classIds = []
    confs = []
    
    for output in outputs:
        for det in output:
            scores = det[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                w, h = int(det[2]*wT), int(det[3]*hT)
                x, y = int((det[0]*wT)-w/2), int((det[1]*hT)-h/2)
                bbox.append([x, y, w, h])
                classIds.append(classId)
                confs.append(float(confidence))
    
    indices = cv2.dnn.NMSBoxes(bbox, confs, confThreshold, nmsThreshold)
    
    for i in indices:
        box = bbox[i]
        x, y, w, h = box[0], box[1], box[2], box[3]
        cv2.rectangle(im, (x, y), (x+w, y+h), (255, 0, 255), 2)
        cv2.putText(im, f'{classNames[classIds[i]].upper()} {int(confs[i]*100)}%',
                   (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
    
    cv2.imshow('ESP32-CAM Object Detection', im)
    if cv2.waitKey(1) == ord('q'):
        break

cv2.destroyAllWindows()