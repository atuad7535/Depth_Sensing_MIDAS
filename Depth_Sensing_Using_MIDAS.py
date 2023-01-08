import cv2
import torch
import matplotlib.pyplot as plt
import time
import numpy as np

from statsmodels.tools import transform_model

new_frame_time = 0
prev_frame_time = 0

# Downloading the MIDAS small model for CPU based systems

midas_model_small = torch.hub.load('intel-isl/MiDaS', 'MiDaS_small')
midas_model_small.to('cpu')
midas_model_small.eval()

# Transforming the video capture for depth estimation

transform_s = torch.hub.load('intel-isl/MiDaS', 'transforms')
transformation_model = transform_s.small_transform

# Using Opencv for videocapture

cap = cv2.VideoCapture(1)
while True:
    ret, frame = cap.read()

    #Tranforming the midas
    img=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_batch=transformation_model(img).to('cpu')

    #Prediction
    with torch.no_grad():
        pred=midas_model_small(img_batch)
        pred=torch.nn.functional.interpolate(
            pred.unsqueeze(1),
            size=img.shape[:2],
            mode='bicubic',
            align_corners= False
        ).squeeze()
        output_map=pred.cpu().numpy()
        output_map=cv2.normalize(output_map,None,0,1,norm_type=cv2.NORM_MINMAX,dtype=cv2.CV_64F)
        output_map=(output_map*255).astype(np.uint8)
        output_map=cv2.applyColorMap(output_map,cv2.COLORMAP_MAGMA)
        print(pred) #outputs numpy array


    #Displayong Frame Rate
    new_frame_time = time.time()
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    cv2.putText(frame, f'FPS: {int(fps)}', (40, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
    img_RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #plt.imshow(output_map)
    cv2.imshow('Frame', frame)
    cv2.imshow('Map', output_map)
    #plt.pause(0.00001)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
#plt.show()