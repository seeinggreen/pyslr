# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 15:06:52 2021

@author: dan
"""

import sys;
import config as cf;
import os;
sys.path.append(cf.openpose_dir + '/build/python/openpose/Release');
os.environ['PATH']  = os.environ['PATH'] + ';' + cf.openpose_dir + '/build/x64/Release;' +  cf.openpose_dir + '/build/bin;'
import pyopenpose as pop;
from tqdm import tqdm;
import numpy as np;
import cv2;
import utils.utils as ut;
import matplotlib.pyplot as plt;
from matplotlib.lines import Line2D;

def get_wrapper(wrapper=None,hand=False):
    if wrapper is None:
        wrapper = pop.WrapperPython();
    params = {"model_folder": cf.openpose_dir + "/models/"};
    if hand:
        params["hand"] = True;
        params["hand_detector"] = 2;
        params["body"] = 0;
    wrapper.configure(params);
    return wrapper;

def get_hand_kps(frame,lhb,rhb,wrapper):
    if lhb is None and rhb is None:
        return None,None,None;
    
    hrs = [];
    if lhb is not None:
        left_hand = pop.Rectangle(*lhb[0],lhb[1][0] - lhb[0][0],lhb[1][1] - lhb[0][1]);
        hrs.append(left_hand);
    else:
        hrs.append(pop.Rectangle(0.,0.,0.,0.));
        
    if rhb is not None:
        right_hand = pop.Rectangle(*rhb[0],rhb[1][0] - rhb[0][0],rhb[1][1] - rhb[0][1]);
        hrs.append(right_hand);
    else:
        hrs.append(pop.Rectangle(0.,0.,0.,0.));
        
    datum = get_datum(frame,wrapper,hrs);
    if lhb is None:
        left = None;
        right = datum.handKeypoints[1][0];
    elif rhb is None:
        left = datum.handKeypoints[0][0];
        right = None;
    else:
        left = datum.handKeypoints[0][0];
        right = datum.handKeypoints[1][0];
    return left,right,datum;

def get_datum(frame,wrapper,hrs=None):
    # Process Image
    datum = pop.Datum()
    datum.cvInputData = np.ascontiguousarray(frame);
    if hrs is not None:
        datum.handRectangles = [hrs];
    wrapper.emplaceAndPop(pop.VectorDatum([datum]))
    
    # Display Image
    return datum;

def get_kpss(frame_gen,total,wrapper):
    kpss = [];
    dats = [];
    for frame in tqdm(frame_gen,initial=1,total=total):
        datum = get_datum(frame,wrapper);
        dats.append(datum);
        kps = datum.poseKeypoints;
        if kps is None:
            kpss.append(None);
        else:
            kpss.append(np.sum(kps,axis=0));
    return kpss,dats;

def get_hand_kpss(frame_gen,total,wrapper,lhbs,rhbs):
    lhkpss = [];
    rhkpss = [];
    outs = [];
    for i,frame in enumerate(tqdm(frame_gen,initial=1,total=total)):
        lhkps,rhkps,out = get_hand_kps(frame,lhbs[i],rhbs[i],wrapper);
        lhkpss.append(lhkps);
        rhkpss.append(rhkps);
        outs.append(out);
    return lhkpss,rhkpss,outs;

def smooth_kpss(kpss):
    kpss = kpss[:,0:8,:];
    for p in range(8):
        if kpss[0][p][2] == 0:
            kpss[0][p] = kpss[1][p];
        if kpss[-1][p][2] == 0:
            kpss[-1][p] = kpss[-2][p];
    for i in range(1,len(kpss)-1):
        for p in range(8):
            if kpss[i][p][2] == 0:
                kpss[i][p][0] = (kpss[i-1][p][0] + kpss[i+1][p][0]) / 2;
                kpss[i][p][1] = (kpss[i-1][p][1] + kpss[i+1][p][1]) / 2;
                kpss[i][p][2] = (kpss[i-1][p][2] + kpss[i+1][p][2]) / 2;
    return kpss;

def mpii_to_openpose(mpii):
    order = [9,8,13,14,15,12,11,10,6,3,4,5,2,1,0];
    return mpii[:,order,:];

def get_get_kp(cap):
    _,frame0 = cap.read();
    def frame_gen(f):
        print(f);
        cap.set(cv2.CAP_PROP_POS_FRAMES,f-5);
        for j in range(6):
            _,frame = cap.read();
            crop = ut.crop(frame,256,320);
            yield crop;
    uc = ut.get_uncrop(frame0,256,320);
    def get_kp(f):
        fg = frame_gen(f);
        raw_kpss = get_kpss(fg,6);
        kps = raw_kpss[-1];
        uc_kps = [uc(kp) for kp in kps];
        kps_with_conf = [[uc_kps[i][0],uc_kps[i][1],kps[i][2]] for i in range(15)];
        return kps_with_conf;
    return get_kp;
            
parts = ['head',
         'neck',
         'rsho',
         'relb',
         'rwri',
         'lsho',
         'lelb',
         'lwri',
         'pelv',
         'rhip',
         'rkne',
         'rank',
         'lhip',
         'lkne',
         'lank'];

def get_part_ids():
    ids = {};
    for i,p in enumerate(parts):
        ids[p] = i;
    return ids;

def plot_hand_confs(lhkpss,rhkpss,fn=None):
    handconfs = {'left':[],'right':[]};
    nImg = len(lhkpss);
    nPs = len(lhkpss[0]);
    for i in range(nImg):
        handconfs['left'].append(sum(lhkpss[i][:,2])/nPs);
        handconfs['right'].append(sum(rhkpss[i][:,2])/nPs);
    avLeft = sum(handconfs['left'])/nImg;
    avRight = sum(handconfs['right'])/nImg;
    plt.figure();
    plt.scatter(range(nImg),handconfs['left'],c='g',label='Left hand');
    plt.scatter(range(nImg),handconfs['right'],c='r',label='Right hand');
    plt.xlabel('Frame');
    plt.ylabel('Average confidence over 21 keypoints');
    plt.ylim(top=0.85);
    plt.axhline(y=avLeft,c='g',linestyle='dashed',label='Average (left)');
    plt.axhline(y=avRight,c='r',linestyle='dashed',label='Average (right)');
    plt.legend(loc='upper center',ncol=2);
    if fn is None:
        plt.show();
    else:
        plt.savefig(fn);