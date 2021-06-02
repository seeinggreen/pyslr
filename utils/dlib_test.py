# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 12:35:14 2021

@author: dan
"""

#https://github.com/italojs/facial-landmarks-recognition/blob/master/shape_predictor_68_face_landmarks.dat

import cv2;
import dlib;

detector = dlib.get_frontal_face_detector();
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

#img = cv2.imread('C:\\Users\\dan\\Downloads\\1_nw617u4p2mEH08NI1Rukrg.png')

cap = cv2.VideoCapture('..\\BF1n.mov');

for i in range(100):
    ret, frame = cap.read();
    grey = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY);
    faces = detector(grey);
    
    if len(faces) > 0:
        cv2.rectangle(frame, (faces[0].left(), faces[0].top()), (faces[0].right(), faces[0].bottom()), (0, 255, 0));
        landmarks = predictor(grey,faces[0])
        for l in range(68):
            p = landmarks.part(l);
            cv2.circle(frame,(p.x,p.y),3,(255,0,0))
    
    cv2.imshow('Face',frame);
    cv2.waitKey(100);
    
cap.release();
cv2.destroyAllWindows();
