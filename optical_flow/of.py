# -*- coding: utf-8 -*-
import cv2
import numpy as np

capture=cv2.VideoCapture('slow_traffic_small.mp4')
result,frame_old=capture.read()
grayscale_old=cv2.cvtColor(frame_old,cv2.COLOR_BGR2GRAY)
sleep=60

feature_params=dict(
    maxCorners=100,
    qualityLevel=0.3,
    minDistance=7,
    blockSize=7    
    )

points_old=cv2.goodFeaturesToTrack(grayscale_old,mask=None,**feature_params)
mask=np.zeros_like(frame_old)

flow_params=dict(
    winSize=(15,15),
    maxLevel=2,
    criteria= (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,10,0.03)
    )

colors=np.random.randint(0,255,(100,3))


while(True):
    result,frame_new=capture.read()
    grayscale_new=cv2.cvtColor(frame_new,cv2.COLOR_BGR2GRAY)
    if(result==False):
        break
    
    points_new,status,error= cv2.calcOpticalFlowPyrLK(grayscale_old,grayscale_new,points_old,None,**flow_params)
    
    good_old=points_old[status==1]
    good_new=points_new[status==1]
    
    for i,(old,new) in enumerate(zip(good_old,good_new)):
        a,b=new.ravel()
        c,d=new.ravel()
        color=colors[i % len(good_new)].tolist()
        mask=cv2.line(mask,(a,b),(c,d),color,2)
        frame_new=cv2.circle(frame_new,(a,b),5,color,-1)
        
    lines=cv2.add(frame_new,mask)
    
    
    
    
    cv2.imshow('Optical Flow',lines)

    if cv2.waitKey(sleep) & 0xFF == ord('q'):
        break
   
    grayscale_old=grayscale_new.copy()
    points_old=good_new.reshape(-1,1,2)
    

capture.release()
cv2.destroyAllWindows()