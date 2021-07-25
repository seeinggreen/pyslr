# -*- coding: utf-8 -*-
"""
Created on Sun May  9 16:21:30 2021

@author: dan
"""

import cv2;
import matplotlib.pyplot as plt;
from mpl_toolkits.mplot3d import Axes3D;
from matplotlib.lines import Line2D;
import numpy as np;
from datetime import datetime;
from scipy.spatial import distance;
import os;
import config as cf;
from openpose import op;
from hourglass import hg;

plt.ioff()

#left_limbs = [[3,4],[4,5],[13,14],[14,15]];
#right_limbs = [[0,1],[1,2],[10,11],[11,12]];
#body = [[2,6],[3,6],[6,7],[7,8],[8,9],[8,12],[8,13]];
left_limbs = [[5,6],[6,7]];
right_limbs = [[2,3],[3,4]];
body = [[0,1],[1,2],[1,5]];

def calc_cent_scale(img):
    imgw = img.shape[1];
    imgh = img.shape[0];
    
    c = (int(imgw / 2), int(imgh / 2));
    s = imgh / 200;
    
    return c , s;

def highlightPoint(img,point,label):
    x = int(point[0]);
    y = int(point[1]);
    
    #Add a square around the selected pixel
    cv2.rectangle(img,(x-1,y-1),(x+1,y+1),(255,255,255));
            
    #Add a text label at the selected pixel
    cv2.putText(img,label,(x,y),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255));
            
    return img;

def pt_to_tup(pt):
    return (int(pt[0]),int(pt[1]));

def draw_stick_figure(frame,ps):
    #Red for left arm/leg
    for p0,p1 in left_limbs:
        cv2.line(frame,pt_to_tup(ps[p0]),pt_to_tup(ps[p1]),(0,0,255),5);  
    
    #Green for right arm/leg
    for p0,p1 in right_limbs:
        cv2.line(frame,pt_to_tup(ps[p0]),pt_to_tup(ps[p1]),(0,255,0),5);
    
    #Blue for body
    for p0,p1 in body:
        cv2.line(frame,pt_to_tup(ps[p0]),pt_to_tup(ps[p1]),(255,0,0),5);
        
def get_3d_ps(ps,p0,p1):
    return [ps[p0][0],ps[p1][0]],[ps[p0][2],ps[p1][2]],[-ps[p0][1],-ps[p1][1]];
    #return [ps[p0][0],ps[p1][0]],[ps[p0][1],ps[p1][1]],[ps[p0][2],ps[p1][2]];
    
def get_2d_ps(ps,p0,p1):
    return [ps[p0][0],ps[p1][0]],[-ps[p0][1],-ps[p1][1]];
        
def get_3d_lines(ps,fn=None,show=False,axis=None,scale=False):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('x');
    ax.set_ylabel('y');
    ax.set_xlim3d(-2, 2);
    ax.set_ylim3d(-2, 2);
    ax.set_zlim3d(-2, 2);
    if axis:
        if axis == 'x':
            ax.azim = 0;
            ax.elev = 0;
            ax.set_xlabel('');
            ax.set_ylabel('y');
            ax.set_zlabel('z');
            ax.set_xticks([]);
        if axis == 'y':
            ax.azim = 270;
            ax.elev = 0;
            ax.set_xlabel('x');
            ax.set_ylabel('');
            ax.set_zlabel('z');
            ax.set_yticks([]);
        if axis == 'z':   
            ax.azim = 270;
            ax.elev = 90;
            ax.set_xlabel('x');
            ax.set_ylabel('y');
            ax.set_zlabel('');
            ax.set_zticks([]);
    for p in left_limbs:
        ax.plot(*get_3d_ps(ps,*p),c='r');
    for p in right_limbs:
        ax.plot(*get_3d_ps(ps,*p),c='g');
    for p in body:
        ax.plot(*get_3d_ps(ps,*p),c='b');
    if show:
        plt.show();
    elif fn is not None:
        plt.savefig(fn);
    else:
        fig.canvas.draw();
        plt.close()
        plot = np.frombuffer(fig.canvas.tostring_rgb(),dtype=np.uint8);
        plot = plot.reshape(fig.canvas.get_width_height()[::-1] + (3,));
        plot = cv2.cvtColor(plot,cv2.COLOR_RGB2BGR);
        if scale:
            return plot[16:-16,98:-78];
        else:
            return plot;

def get_2d_lines(ps,scale=False):
    """fig,axes = plt.subplots(1,3,sharey=True);
    for i in range(3):
        for p in left_limbs:
            axes[i].plot(*get_2d_ps(ps[i],*p),c='r');
        for p in right_limbs:
            axes[i].plot(*get_2d_ps(ps[i],*p),c='g');
        for p in body:
            axes[i].plot(*get_2d_ps(ps[i],*p),c='b');
        axes[i].axis('square'); 
    fig.canvas.draw();
    if show:
        plt.show();
        return;
    plt.close()
    plot = np.frombuffer(fig.canvas.tostring_rgb(),dtype=np.uint8);
    plot = plot.reshape(fig.canvas.get_width_height()[::-1] + (3,));
    plot = cv2.cvtColor(plot,cv2.COLOR_RGB2BGR);
    return cv2.resize(plot[80:224,:,:],(768,256));"""
    return get_3d_lines(ps,axis='x',scale=scale),get_3d_lines(ps,axis='y',scale=scale),get_3d_lines(ps,axis='z',scale=scale);

def remove_outliers(ps):
    for i in range(1,len(ps) - 1):
        a = ps[i-1];
        b = ps[i];
        c = ps[i+1];
        
        m = (a+c) / 2;
        ad = abs(a - m);
        bd = abs(b - m);
        cd = abs(c - m);
        
        if ad < m * 0.1 and cd < m * 0.1 and bd > m * 0.1:
            ps[i] = m;
            
    return ps;

def remove_all_outliers(ps,js):
    #Remove outliers for specified joints (js) in x and y for all points (ps)
    for j in js:
        remove_outliers(ps[:,j,0]);
        remove_outliers(ps[:,j,1]);
        
def crop(img,size,h_cent):
    scale = img.shape[0] / size;
    new_w = int(img.shape[1] / scale);
    scaled = cv2.resize(img,(new_w,size));
    hs = int(size / 2);
    h_cent = int(h_cent / scale);
    cropped = scaled[:,h_cent - hs:h_cent + hs,:];
    return cropped;

def crop_kps(img,kps,size,h_cent):
    """
    Converts keypoints calculated on full image to corresponding cropped locations.

    Parameters
    ----------
    img : numpy.ndarray
        A sample frame at full size.
    kps : numpy.ndarray
        The keypoints for the full size image.
    size : int
        The size of the cropped image.
    h_cent : int
        The horizontal centre of the frame.

    Returns
    -------
    new_kps : list of list of int
        The keypoints rescaled to the cropped image.

    """
    scale = img.shape[0] / size;
    new_w = img.shape[1] / scale;
    h_dif = int((new_w - size) / 2);
    new_kps = [];
    for kp in kps:
        x = int(kp[0] / scale) - h_dif;
        y = int(kp[1] / scale);
        new_kps.append([x,y]);
    return new_kps;

def clip_point(p,size):
    return min(max(0,p),size);

def crop_head(head,neck,size):
    hn_dis = int(neck[1] - head[1]);
    half_d = int(hn_dis / 2);
    dim = hn_dis + half_d;
    half_dim = int(dim / 2);
    top = int(head[1] - half_d);
    bottom = int(neck[1]);
    centre = int((head[0] + neck[0]) / 2);
    left = centre - half_dim;
    right = centre + half_dim;
    
    tl = (clip_point(left,size),clip_point(top,size));
    br = (clip_point(right,size),clip_point(bottom,size));
    
    return tl,br;

def get_uncrop(img,size,h_cent):
    ar = img.shape[1] / img.shape[0];
    shift = (size * (ar - 1)) / 2;
    scale = scale = img.shape[0] / size;
    def uncrop(pt):
        new_x = int((pt[0] + shift) * scale);
        new_y = int(pt[1] * scale);
        return (new_x,new_y);
    return uncrop;

def extract_head(img,tl,br,uc,size):
    tl_uc = uc(tl);
    br_uc = uc(br);

    ex = img[tl_uc[1]:br_uc[1],tl_uc[0]:br_uc[0],:];
    
    ex = cv2.resize(ex,(size,size));
    
    return ex;
    
def show(img):
    cv2.imshow('Image',img);
    cv2.waitKey(0);
    cv2.destroyAllWindows();
    
def get_datetime_string():
    time = datetime.now();
    return time.strftime('%Y-%m-%d-%H_%M_%S');

def get_points_dis(ps_a,ps_b):
    scores = [];
    for i in range(len(ps_a)):
        scores.append(distance.euclidean(ps_a[i][0:2],ps_b[i][0:2]));
    return scores;

def score_to_colour(s):
    return ['r','k','b','g'][int(s)];

def plot_exp(conf,true_ps,pred_ps,scores,target,fn,outliers=[],legend=True):
    dis = get_points_dis(true_ps,pred_ps);
    data = np.array([[dis[i],conf[i],scores[i]] for i in range(len(conf)) if i not in outliers]);
    cs = [score_to_colour(s) for s in data[:,2]];
    fig, ax = plt.subplots(dpi=300);
    ax.scatter(data[:,0],data[:,1],c=cs);
    if legend:
        legend = [Line2D([0], [0], marker='o', color='w', label='Scatter', markerfacecolor='r', markersize=15),
              Line2D([0], [0], marker='o', color='w', label='Scatter', markerfacecolor='k', markersize=15),
              Line2D([0], [0], marker='o', color='w', label='Scatter', markerfacecolor='b', markersize=15),
              Line2D([0], [0], marker='o', color='w', label='Scatter', markerfacecolor='g', markersize=15)];
        plt.legend(handles=legend,labels=['0','1','2','3'],title='Score',labelspacing=0.6);
    else:
        ax.set_xticks([0,2,4,6,8,10,12,14,16,18,20]);
        m,c = np.polyfit(data[:,0],data[:,1],1);
        ax.plot(data[:,0],m*data[:,0]+c,c='r');
    plt.xlabel('Euclidean distance between prediction and true location (pixels)');
    plt.ylabel('Confidence of prediction');
    if legend:
        newax = fig.add_axes([0.6, 0.589, 0.18, 0.27], anchor='NE');
    else:
       newax = fig.add_axes([0.7, 0.589, 0.18, 0.27], anchor='NE'); 
    newax.set_yticks([]);
    newax.set_xticks([]);
    target = cv2.cvtColor(target,cv2.COLOR_BGR2RGB);
    newax.imshow(target)
    plt.savefig(fn);
    #plt.show()
    
def create_eval_folder():
    time_str = get_datetime_string();
    os.mkdir(cf.eval_dir + "\\" + time_str);
    os.mkdir(cf.eval_dir + "\\" + time_str + "\\imgs");
    return time_str;

def demo_hms(frame,kps):
    """
    Generates and displays heatmaps for three keypoints of a given frame.

    Parameters
    ----------
    frame : numpy.ndarray
        The uncropped frame of video.
    kps : numpy.ndarray
        The keypoints for the given frame in MPII format.

    Returns
    -------
    None.
    """
    c = crop(frame,256,320);
    highlightPoint(c,kps[9],hg.parts['mpii'][9]);
    highlightPoint(c,kps[10],hg.parts['mpii'][10]);
    highlightPoint(c,kps[15],hg.parts['mpii'][15]);
    hm = hg.gen_heatmaps([kps],256,64)[0];
    hm_he = cv2.resize(hm[9],(256,256));
    hm_rw = cv2.resize(hm[10],(256,256));
    hm_lw = cv2.resize(hm[15],(256,256));
    
    fig,axes = plt.subplots(1,4,sharey=False,dpi=180);
    axes[0].set_yticks([]);
    axes[0].set_xticks([]);
    for i in range(1,4):
        axes[i].set_yticklabels([]);
        axes[i].set_xticklabels([]);
    c = cv2.cvtColor(c,cv2.COLOR_BGR2RGB);
    axes[0].imshow(c);
    axes[1].matshow(hm_he);
    axes[2].matshow(hm_rw);
    axes[3].matshow(hm_lw);
    fig.canvas.draw();
    plt.close();
    plot = np.frombuffer(fig.canvas.tostring_rgb(),dtype=np.uint8);
    plot = plot.reshape(fig.canvas.get_width_height()[::-1] + (3,));
    plot = cv2.cvtColor(plot,cv2.COLOR_RGB2BGR);
    
    return plot[250:470,120:990];

def draw_target(crop,ps):
    target = np.array(crop);
    for p in ps:
        cx = p[0];
        cy = p[1];
        for i in range(3):
            cv2.circle(target,(cx,cy),25*(i+1),(255,255,255));
            cv2.putText(target,str(25*(i+1)),((cx+28)+24*i,cy+2),cv2.FONT_HERSHEY_SIMPLEX,0.3,(255,255,255));
        cv2.line(target,(cx-5,cy-5),(cx+5,cy+5),(0,0,255));
        cv2.line(target,(cx-5,cy+5),(cx+5,cy-5),(0,0,255));
    return target;

def draw_small_target(crop,ps,scale):
    #target = ut.draw_small_target(head_crop,[133,94],320/94)
    target = np.array(crop);
    cx = ps[0];
    cy = ps[1];
    cv2.circle(target,(cx,cy),int(5*scale),(255,255,255));
    cv2.circle(target,(cx,cy),int(14*scale),(255,255,255));
    cv2.circle(target,(cx,cy),int(25*scale),(255,255,255));
    cv2.putText(target,str(5),(int((cx+18)+scale),cy+3),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255));
    cv2.putText(target,str(15),(int((cx+18)+10*scale),cy+3),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255));
    cv2.putText(target,str(25),(int((cx+24)+20*scale),cy+3),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255));
    cv2.line(target,(cx-5,cy-5),(cx+5,cy+5),(0,0,255));
    cv2.line(target,(cx-5,cy+5),(cx+5,cy-5),(0,0,255));
    return target;
    