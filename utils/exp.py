# -*- coding: utf-8 -*-
"""
Created on Thu Jul 22 15:50:38 2021

@author: dan
"""

import cv2;
import config as cf;
import csv;
import utils.utils as ut;
import utils.manual_tag as mt;
import json;
import numpy as np;
from tqdm import tqdm;

def evaluate(cap,samples,get_kp,parts,parts_to_use,frames=None):
    time_str = ut.create_eval_folder();        
    fc = cap.get(cv2.CAP_PROP_FRAME_COUNT);
    if frames is None:
        frames = np.random.choice(int(fc),samples,replace=False);
    kpss = [];
    for i,f in tqdm(enumerate(frames)):
        cap.set(cv2.CAP_PROP_POS_FRAMES,f);
        _,frame = cap.read();
        crop = ut.crop(frame,256,320);
        kps = get_kp(f);
        kpss.append(kps);
        
        ut.highlightPoint(crop,(5,30),str(i+1));
        for p in parts_to_use:
            ut.highlightPoint(crop,kps[p],parts[p]);
        
        cv2.imwrite(cf.eval_dir + "\\" + time_str + "\\imgs\\" + "{:02d}.png".format(i+1),crop);
    cap.release();

    with open(cf.eval_dir + "\\" + time_str + "\\data.csv",'w') as f:
        writer = csv.writer(f);
        for i in range(samples):
            row = [i+1,frames[i]];
            for p in parts_to_use:
                row.append(kpss[i][p][2]);
            writer.writerow(row);
    return kpss;

def get_frames(time_str):
    """Given a time string to identify a previous experiment, the randomly 
    selected frames are returned."""
    
    with open(cf.eval_dir + "\\" + time_str + "\\data.csv",'r') as f:
        reader = csv.reader(f);
        frames = [];
        for row in reader:
            if row == []: continue;
            frames.append(int(row[1]));
    return frames;

def reevaluate(time_str,get_kp):
    """Takes an existing experiment and reruns the model to get the keypoints.
    
    Existing experiments will have a data.csv file which will contain the 
    frames which were randomly selected for that experiment. There is no need
    to reproduce the data file or the labelled images as the result won't
    change."""
    frames = get_frames(time_str);
    
    kpss = [];
    for i,f in tqdm(enumerate(frames)):
        kps = get_kp(f);
        kpss.append(kps);
    return kpss;

def manual_tag(datapath,time_str,kp=""):
    """Allows an experiment to be manually tagged with correct locations.
    
    Given a datapath to a video and the ID for an experiment, the frames will 
    be retreived and shown to the user to tag manually. A label can be provided 
    to identify which keypoint is being tagged. The method will return the 
    coordinates specified by the user's clicks"""
    frames = get_frames(time_str);
    cap = cv2.VideoCapture(datapath);
    _,frame0 = cap.read();
    def frame_gen():
        for i,f in enumerate(frames):
            cap.set(cv2.CAP_PROP_POS_FRAMES,f);
            _,frame = cap.read();
            ut.highlightPoint(frame,(5,30),kp+str(i+1));
            yield frame;
        cap.release();
    return mt.manual_tag(frame0,frame_gen(),len(frames));

def store_tags(frames,head,rwri,lelb,kpss,scores,time_str,part_ids):
    data = [];
    scores = [s.split('\t') for s in scores.split('\n')];
    hs = [int(s[0]) for s in scores];
    ws = [int(s[1]) for s in scores];
    es = [int(s[2]) for s in scores];
    for i,f in enumerate(frames):
        d = {'sample':i+1,
             'frame':f,
             'head':{'conf':float(kpss[i][part_ids['head']][2]),'pred':[float(kpss[i][part_ids['head']][0]),float(kpss[i][part_ids['head']][1])],'true':head[i],'score':hs[i]},
             'rwri':{'conf':float(kpss[i][part_ids['rwri']][2]),'pred':[float(kpss[i][part_ids['rwri']][0]),float(kpss[i][part_ids['rwri']][1])],'true':rwri[i],'score':ws[i]},
             'lelb':{'conf':float(kpss[i][part_ids['lelb']][2]),'pred':[float(kpss[i][part_ids['lelb']][0]),float(kpss[i][part_ids['lelb']][1])],'true':lelb[i],'score':es[i]}
             }
        data.append(d);
    with open(cf.eval_dir + "\\" + time_str + "\\data.json",'w') as f:
        json.dump(data,f);
        
def load_exp_data(time_str):
    with open(cf.eval_dir + "\\" + time_str + "\\data.json",'r') as f:
        raw_data = json.load(f);
    data = {};
    for kp in ['head','rwri','lelb']:
        data[kp] = {};
        for v in ['conf','true','pred','score']:
            data[kp][v] = [d[kp][v] for d in raw_data];
            
    if 'neck' in raw_data[0].keys():
        data['neck'] = {};
        for v in ['conf','true','pred']:
            data['neck'][v] = [d['neck'][v] for d in raw_data];
            
    data['all'] = {};
    for v in ['conf','true','pred','score']:
        data['all'][v] = data['head'][v] + data['rwri'][v] + data['lelb'][v];
        
    return data;
    
def plot_exp(data,fn,frame=None,kps=None,target=[],**kwargs):
    if target is None:
        target = ut.crop(frame,256,320);
        target = ut.draw_target(target,kps);
    for kp in ['all']:
        ut.plot_exp(data[kp]['conf'],data[kp]['true'],data[kp]['pred'],data[kp]['score'],target,fn,**kwargs);