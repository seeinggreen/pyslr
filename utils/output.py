# -*- coding: utf-8 -*-
"""
Created on Sat Jul 24 12:58:49 2021

@author: dan
"""

import cv2;
import numpy as np;
import config as cf;
from utils import utils as ut;
from openpose import op;

def show_results(datapath,nImg,mods,input_data=None,loop=True,fn=None,start_frame=None,frame_ids=None):
    cap = cv2.VideoCapture(datapath);
    cont = True;
    if fn is not None:
        out = cv2.VideoWriter('..\\fn',cv2.VideoWriter_fourcc(*'mjpa'), 10, (768*2,768));
        out_vid = True;
    else:
        out_vid = False;

    while cont:
        if start_frame is not None:
            cap.set(cv2.CAP_PROP_POS_FRAMES,start_frame);
        for i in range(nImg):
            if frame_ids is not None:
                cap.set(cv2.CAP_PROP_POS_FRAMES,frame_ids[i]);
            
            _,frame = cap.read();
            crop = ut.crop(frame,256,320);
            render = cv2.imread(cf.renders_path + "{:04d}.png".format(i+1));
            
            data = {'frame':frame,'crop':crop,'render':render,'i':i};
            if input_data is not None:
                data.update(input_data);
            
            frames = [];
            for m in mods:
                frames.append(m(data));
            
            # Display the resulting frame
            row1 = np.concatenate(frames[0:3],axis=1);
            row2 = np.concatenate(frames[3:6],axis=1);
            row3 = np.concatenate(frames[6:9],axis=1);          
            grid = np.concatenate([row1,row2,row3]);
            
            out_frame = np.concatenate([grid,render],axis=1);
            cv2.imshow('frame',out_frame);
            
            if out_vid:
                out.write(out_frame);
        
            if cv2.waitKey(40) & 0xFF == ord('q'):
                cont = False;
                break;
            if not loop:
                cont = False;
    cv2.destroyAllWindows();
    cap.release();
    if out_vid:
        out.release();
        
def basic_crop(data):
    return data['crop'];

def blank(data):
    return data['crop']*0;

def show_render_face(data):
    return cv2.resize(data['render'][20:180,330:490,:],(256,256));

def highlight_points(data):
    crop = np.array(data['crop']);
    for p in range(0,8):
        ut.highlightPoint(crop,data['kpss'][data['i']][p],op.parts[p]);
    return crop;

def draw_stickfigure(data):
    crop = np.array(data['crop']);
    ut.draw_stick_figure(crop, data['kpss'][data['i']]);
    return crop;

def draw_head_box(data):
    crop = np.array(data['crop']);
    tl = data['tls'][data['i']];
    br = data['brs'][data['i']];
    cv2.rectangle(crop,tl,br,(255,255,0));
    return crop;

def draw_face_box(data):
    head = extract_head(data);
    face_box = data['position_data']['face_box'][data['i']];
    sefs = data['sefs'][data['i']];
    if face_box is not None:
        cv2.rectangle(head, *face_box, (0, 255, 0));
    else:
        cv2.rectangle(head, *sefs, (0, 0, 255));
    return head;

def extract_head(data):
    tl = data['tls'][data['i']];
    br = data['brs'][data['i']];
    head = ut.extract_head(data['frame'],tl,br,data['uc'],256);
    return head;

def draw_face_landmarks(data):
    head = extract_head(data);
    landmarks = data['position_data']['face_landmarks'][data['i']];
    face_box = data['position_data']['face_box'][data['i']];
    if face_box is not None:
        for p in landmarks:
            cv2.circle(head,p,3,(255,0,0));
    else:
        for p in landmarks:
            cv2.circle(head,p,3,(255,0,200));
    return head;