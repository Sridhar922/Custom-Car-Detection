import cv2
import torch
import numpy as np

path= 'C:/Users/sridh/PycharmProjects/custom car detection/best.pt'
model= torch.hub.load('ultralytics/yolov5', 'custom', path, force_reload=True)
cap= cv2.VideoCapture('car rally.mp4')
while True:
    ret, frame =cap.read()
    frame=cv2.resize(frame,(1020,500))
    results=model(frame)
    frame= np.squeeze(results.render())
    cv2.imshow('Sridhar', frame)
    if cv2.waitKey(0) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()