# -*- coding: utf-8 -*-
"""
Created on Sun Jul 25 13:26:14 2021

@author: dan
"""

import numpy as np;
import cv2;
from utils import utils as ut;

def extract_hand(elb,wri,ret=False):    
    x, y = wri[0:2];
    a, b = elb[0:2];
    
    adj = x - a;
    opp = b - y;
    
    x = (x + a) / 2;
    y = (y + b) / 2;
    
    dim = np.sqrt(opp**2 + adj**2) * 1.5;
    
    if adj == 0:
        gam = 999 * opp;
    else:
        gam = opp/adj;
    
    if opp < 0:
        if gam <= 0 and gam >= -1:
            alp = 0;
            bet = -0.5 - gam / 2;
        elif gam < -1:
            alp = 0.5+np.arctan(gam)/(np.pi/2);
            bet = 0;
        elif gam > 1:
            alp = -(np.pi/4)/np.arctan(gam);
            bet = 0;
        else:
            alp = -1;
            bet = -0.5 + gam / 2;
    elif opp > 0:
        if gam <= 0 and gam >= -1:
            alp = -1;
            bet = -0.5 + gam / 2;
        elif gam < -1:
            alp = (np.pi/4)/np.arctan(gam);
            bet = -1;
        elif gam > 1:
            alp =  0.5-np.arctan(gam)/(np.pi/2);
            bet = -1;
        else:
            alp = 0;
            bet = -0.5 - gam / 2;
    else:
        if adj > 0:
            alp = 0;
            bet = -0.5;
        else:
            alp = -1;
            bet = -0.5;
            
    tl = (int(x+alp*dim), int(y+bet*dim));
    br = (int(x+(alp+1)*dim), int(y+(bet+1)*dim));
    
    if ret:
        return tl,br,alp,bet,gam;
    else:
        return (tl,br);

def test(n,ret=False,speed=50):
    the = np.arange(0,2*np.pi,(2*np.pi)/n);
    rand = (np.random.rand(n) / 2) + 0.5;
    opp = np.sin(the) * rand;
    adj = np.cos(the) * rand;
    x = (100 * adj).astype(int) + 256;
    y = (100 * opp).astype(int) + 256;
    
    tls = [];
    brs = [];
    alps = [];
    bets = [];
    gams = [];
    for i in range(n):
        tl,br,alp,bet,gam = extract_hand([256,256],[x[i],y[i]],True);
        tls.append(tl);
        brs.append(br);
        alps.append(alp);
        bets.append(bet);
        gams.append(gam);
    
    plts = np.ones((n,512,512,3),dtype=np.uint8) * 255;
    for i in range(n):
        cv2.line(plts[i],(256,256),(x[i],y[i]),(0,0,255));
        cv2.rectangle(plts[i],tls[i],brs[i],(255,0,0));
        cv2.putText(plts[i],"a:{:.2f}, b:{:.2f}, g:{:.2f}, t:{:.0f}".format(-alps[i],-bets[i],gams[i],(n/360)*i),(5,30),cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,0,0));
    
    cont = True;
    while cont:
        for i in range(n):
            cv2.imshow('frame',plts[i])
            if cv2.waitKey(speed) & 0xFF == ord('q'):
                cont = False;
                break;
    cv2.destroyAllWindows();
    
    if ret:
        return plts;
    
def square_boxes(boxes):
    for i,box in enumerate(boxes):
        if box is None:
            continue;
        x = box[1][0] - box[0][0];
        y = box[1][1] - box[0][1];
        
        if x == y:
            continue;       
        elif x < y:
            boxes[i] = (box[0],(box[1][0] + 1,box[1][1]));
        else:
            boxes[i] = (box[0],(box[1][0],box[1][1] + 1));
    return boxes;

def translate_hand_kps(hand_kpss,kpss,wrist_ix):
    new_kpss = [];
    for i,kps in enumerate(hand_kpss):
        if kps is None:
            new_kpss.append(None);
        else:
            dif = kps[0] - kpss[i][wrist_ix];
            new_kps = [];
            for p in kps:
                new_kps.append(p - dif);
            new_kpss.append(new_kps);
    return new_kpss;

thumb = [[0,1],[1,2],[2,3],[3,4]];
index = [[0,5],[5,6],[6,7],[7,8]];
mid = [[0,9],[9,10],[10,11],[11,12]];
ring = [[0,13],[13,14],[14,15],[15,16]];
pink = [[0,17],[17,18],[18,19],[19,20]];

def draw_hand_lines(frame,ps):
    if ps is None:
        return;
    for p0,p1 in thumb:
        cv2.line(frame,ut.pt_to_tup(ps[p0]),ut.pt_to_tup(ps[p1]),(0,0,255),2);
    for p0,p1 in index:
        cv2.line(frame,ut.pt_to_tup(ps[p0]),ut.pt_to_tup(ps[p1]),(0,166,255),2);
    for p0,p1 in mid:
        cv2.line(frame,ut.pt_to_tup(ps[p0]),ut.pt_to_tup(ps[p1]),(0,255,255),2);
    for p0,p1 in ring:
        cv2.line(frame,ut.pt_to_tup(ps[p0]),ut.pt_to_tup(ps[p1]),(0,255,0),2);
    for p0,p1 in pink:
        cv2.line(frame,ut.pt_to_tup(ps[p0]),ut.pt_to_tup(ps[p1]),(255,0,0),2);