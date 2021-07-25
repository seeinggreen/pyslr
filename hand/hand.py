# -*- coding: utf-8 -*-
"""
Created on Sun Jul 25 13:26:14 2021

@author: dan
"""

import numpy as np;
import cv2;

def extract_hand(elb,wri):
    x, y = wri;
    a, b = elb;
    adj = x - a;
    opp = b - y;
    dim = np.sqrt(opp**2 + adj**2);
    
    if adj == 0:
        gam = 0;
    else:
        gam = opp/adj;
    
    if opp < 0:
        if gam == 0:
            alp = -0.5;
            bet = 0;
        elif gam < 0 and gam > -1:
            alp = 0;
            bet = -0.5 - gam / 2;
        elif gam == -1:
            alp = 0;
            bet = 0;
        elif gam < -1:
            alp = np.arctan(gam)/np.pi;
            bet = 0;
        elif gam > 1:
            alp = np.arctan(gam)/np.pi - 1;
            bet = 0;
        else:
            alp = -1;
            bet = -0.5 + gam / 2;
    elif opp > 0:
        if gam == 0:
            alp = -0.5;
            bet = -1;
        elif gam < 0 and gam > -1:
            alp = -1;
            bet = -0.5 + gam / 2;
        elif gam == -1:
            alp = -1;
            bet = -1;
        elif gam < -1:
            alp = (np.pi/4)/np.arctan(gam);
            bet = -1;
        elif gam > 1:
            alp =  np.arctan(gam)/-np.pi;
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
    br = (int(x+(alp+1)*dim),int(y+(bet+1)*dim));
    
    return tl,br,alp,bet,gam;

def test(n):
    the = np.arange(0,2*np.pi,(2*np.pi)/n);
    opp = np.sin(the);
    adj = np.cos(the);
    x = (100 * adj).astype(int) + 180;
    y = (100 * opp).astype(int) + 180;
    
    tls = [];
    brs = [];
    alps = [];
    bets = [];
    gams = [];
    for i in range(n):
        tl,br,alp,bet,gam = extract_hand([180,180],[x[i],y[i]]);
        tls.append(tl);
        brs.append(br);
        alps.append(alp);
        bets.append(bet);
        gams.append(gam);
    
    plts = np.ones((n,360,360,3),dtype=np.uint8) * 255;
    for i in range(n):
        cv2.line(plts[i],(180,180),(x[i],y[i]),(0,0,255));
        cv2.rectangle(plts[i],tls[i],brs[i],(255,0,0));
        cv2.putText(plts[i],"{:.2f},{:.2f},{:.2f}".format(alps[i],bets[i],gams[i]),(5,30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0));
    
    cont = True;
    while cont:
        for i in range(n):
            cv2.imshow('frame',plts[i])
            if cv2.waitKey(10) & 0xFF == ord('q'):
                cont = False;
                break;
    cv2.destroyAllWindows();
    
    return plts;