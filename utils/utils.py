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
from scipy import stats;

#Turn off autodisplay of plots
plt.ioff()

#Connections between OpenPose joints for stick figure drawing
left_limbs = [[5,6],[6,7]];
right_limbs = [[2,3],[3,4]];
body = [[0,1],[1,2],[1,5]];

def calc_cent_scale(img):
    """
    Calculates the centre of an image and the correct scale value.

    Parameters
    ----------
    img : numpy.ndarray
        The image to process.

    Returns
    -------
    c : int
        The coordinates of the centre of the image.
    s : float
        The scale of the image in relation to a 200x200 image.
    """
    imgw = img.shape[1];
    imgh = img.shape[0];
    
    c = (int(imgw / 2), int(imgh / 2));
    s = imgh / 200;
    
    return c , s;

def highlightPoint(img,point,label):
    """
    Highlights a keypoint and gives it the given label.

    Parameters
    ----------
    img : numpy.ndarray
        The image to tag.
    point : list OR tuple
        The XY coordinates of the point on the frame (can be >2 elements,
        others are ignored).
    label : str
        The text to display alongside the keypoint.

    Returns
    -------
    img : numpy.ndarray
        Returns the tagged image (which is also modified in place).
    """
    x = int(point[0]);
    y = int(point[1]);
    
    #Add a square around the selected pixel
    cv2.rectangle(img,(x-1,y-1),(x+1,y+1),(255,255,255));
            
    #Add a text label at the selected pixel
    cv2.putText(img,label,(x,y),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255));
            
    return img;

def pt_to_tup(pt):
    """
    Convenience method to generate a pair of two ints from a tuple or list.

    Parameters
    ----------
    pt : list OR tuple
        Can be a list or a tuple of >=2 elements as floats or ints.

    Returns
    -------
    pt : tuple of int
        A pair of two ints.
    """
    return (int(pt[0]),int(pt[1]));

def draw_stick_figure(frame,ps):
    """
    Draws a stick figure connecting the keypoints.

    Parameters
    ----------
    frame : numpy.ndarray
        The image to be drawn on.
    ps : numpy.ndarray
        The keypoints for the joints in OpenPose order.

    Returns
    -------
    None.
    """
    #Green for left arm/leg
    for p0,p1 in left_limbs:
        cv2.line(frame,pt_to_tup(ps[p0]),pt_to_tup(ps[p1]),(0,255,0),5);  
    
    #Red for right arm/leg
    for p0,p1 in right_limbs:
        cv2.line(frame,pt_to_tup(ps[p0]),pt_to_tup(ps[p1]),(0,0,255),5);
    
    #Blue for body
    for p0,p1 in body:
        cv2.line(frame,pt_to_tup(ps[p0]),pt_to_tup(ps[p1]),(255,0,0),5);
        
def get_3d_ps(ps,p0,p1):
    """
    Reorders XYZ coordinate pairs, rotating as necessary to display correctly.

    Parameters
    ----------
    ps : numpy.ndarray
        A set of coordinates for each joint.
    p0 : int
        Index for the start joint.
    p1 : int
        Index for the end joint.

    Returns
    -------
    ps_3d : list of list
        The start and end points for the XYZ coordinates.
    """
    return [ps[p0][0],ps[p1][0]],[ps[p0][2],ps[p1][2]],[-ps[p0][1],-ps[p1][1]];
        
def get_3d_lines(ps,fn=None,show=False,axis=None,scale=False):
    """
    Plots a set of 3D cordinates as a 3D stick figure.

    Parameters
    ----------
    ps : numpy.ndarray
        The 3D keypoints to plot.
    fn : str, optional
        Filename to save the plot if provided. The default is None.
    show : bool, optional
        Displays the plot to the user if True. The default is False.
    axis : str, optional
        Fixes view to look along an axis if specified. The default is None.
    scale : bool, optional
        Crops the image to remove whitespace if True. The default is False.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
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
    """
    Gets the 3D stick figure plot for each axis (effectively 2D plots).

    Parameters
    ----------
    ps : numpy.ndarray
        The 3D keypoints to be plotted.
    scale : bool, optional
        Crops the images to remove whitespace if True. The default is False.

    Returns
    -------
    x : numpy.ndarray
        The stick figure seen along the X axis.
    y : numpy.ndarray
        The stick figure seen along the Y axis.
    z : numpy.ndarray
        The stick figure seen along the Z axis.
    """
    return get_3d_lines(ps,axis='x',scale=scale),get_3d_lines(ps,axis='y',scale=scale),get_3d_lines(ps,axis='z',scale=scale);

def remove_outliers(ps):
    """
    Removes points that are greater than 10% away from their surrounding two
    points.

    Parameters
    ----------
    ps : numpy.ndarray
        The points to process.

    Returns
    -------
    ps : numpy.ndarray
        The points without outliers.
    """
    for i in range(1,len(ps) - 1):
        a = ps[i-1];
        b = ps[i];
        c = ps[i+1];
        
        #Calculate mean of two surrounding points
        m = (a+c) / 2;
        ad = abs(a - m);
        bd = abs(b - m);
        cd = abs(c - m);
        
        #Replaces point with mean if it's an outlier
        if ad < m * 0.1 and cd < m * 0.1 and bd > m * 0.1:
            ps[i] = m;
            
    return ps;

def remove_all_outliers(ps,js):
    """
    Removes outliers for specified joints in x and y for all points.

    Parameters
    ----------
    ps : numpy.ndarray
        The points to be cleaned up.
    js : list of int
        The joints to process.

    Returns
    -------
    None.
    """
    for j in js:
        remove_outliers(ps[:,j,0]);
        remove_outliers(ps[:,j,1]);
        
def crop(img,size,h_cent):
    """
    Crops an image to a square of size x size, horizontally centered around
    h_cent with sides removed.

    Parameters
    ----------
    img : numpy.ndarray
        Image to be cropped.
    size : int
        Height/width of cropped image.
    h_cent : int
        The horizontal centre of the desired crop area.

    Returns
    -------
    crop : numpy.ndarray
        The image vertically scaled to size height and cropped to size width.
    """
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
    """
    Ensure 0 <= p < size.

    Parameters
    ----------
    p : int
        Point to clip.
    size : int
        Size of the image the point sits in.

    Returns
    -------
    p : int
        Clipped point.
    """
    return min(max(0,p),size);

def crop_head(head,neck,size):
    """
    Returns a bounding box around the head given the head and neck keypoints.

    Parameters
    ----------
    head : numpy.ndarray
        The keypoint for the head.
    neck : numpy.ndarray
        The keypoint for the neck.
    size : int
        The size of the image (to keep bounding box inside image).

    Returns
    -------
    tl : tuple of int
        The coordinates of the top left corner of the bounding box.
    br : tuple of int
        The coorindates of the bottom right corner of the bounding box.

    """
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
    """
    Returns a method to convert cropped coordinate to full frame coordinate.

    Parameters
    ----------
    img : numpy.ndarray
        The full framed image.
    size : int
        The size of the cropped images.
    h_cent : int
        The horizontal centre used for cropping.

    Returns
    -------
    uc : function
        A function to 'uncrop' coordinates.
    """
    ar = img.shape[1] / img.shape[0];
    shift = (size * (ar - 1)) / 2;
    scale = scale = img.shape[0] / size;
    def uncrop(pt):
        new_x = int((pt[0] + shift) * scale);
        new_y = int(pt[1] * scale);
        return (new_x,new_y);
    return uncrop;

def extract_area(img,tl,br,uc,size):
    """
    Returns the specified area of the full frame of the image in the given size.

    Parameters
    ----------
    img : numpy.ndarray
        The full frame image.
    tl : tuple of int
        The coordinates (in cropped image scale) of the top left corner of the
        bounding box.
    br : tuple of int
        The coorindates (in croppped image scale) of the bottom right corner
        of the bounding box.
    uc : function
        A method to convert cropped image scale coordinates to full frame
        scale coordinates.
    size : int
        The size of the output image.

    Returns
    -------
    ex : numpy.ndarray
        The extracted area of the image.
    """
    tl_uc = uc(tl);
    br_uc = uc(br);

    ex = img[tl_uc[1]:br_uc[1],tl_uc[0]:br_uc[0],:];
    
    ex = cv2.resize(ex,(size,size));
    
    return ex;
    
def show(img):
    """
    Convenience method for displaying an image.

    Parameters
    ----------
    img : numpy.ndarray
        The image to display.

    Returns
    -------
    None.
    """
    cv2.imshow('Image',img);
    cv2.waitKey(0);
    cv2.destroyAllWindows();
    
def get_datetime_string():
    """
    Generates a string (that can be used in filenames) based on the current 
    date and time.

    Returns
    -------
    timestr : str
        Thr generate time string.
    """
    time = datetime.now();
    return time.strftime('%Y-%m-%d-%H_%M_%S');

def get_points_dis(ps_a,ps_b):
    """
    Calculates the Euclidean distance between points in two sets.

    Parameters
    ----------
    ps_a : numpy.ndarray
        The first set of points to compare.
    ps_b : numpy.ndarray
        The second set of points to compare.

    Returns
    -------
    scores : list of double
        The Euclidean distance between each pair of points.
    """
    scores = [];
    for i in range(len(ps_a)):
        scores.append(distance.euclidean(ps_a[i][0:2],ps_b[i][0:2]));
    return scores;

def score_to_colour(s):
    """
    Take a score of 0,1,2 or 3 and converts to a colour code.

    Parameters
    ----------
    s : str
        The score as a single character string '0', '1', etc..

    Returns
    -------
    c : str
        The colour code for red, black, blue or green.
    """
    return ['r','k','b','g'][int(s)];

def plot_exp(conf,true_ps,pred_ps,scores,target,fn,outliers=[],legend=True):
    """
    Plots an experiement for the distance between true locations and
    predictions.

    Parameters
    ----------
    conf : numpy.ndarray
        The confidence values for each point.
    true_ps : numpy.ndarray
        The true location of each point.
    pred_ps : numpy.ndarray
        The predicted location of each point.
    scores : list of int
        The manually judged score for each point.
    target : numpy.ndarray
        An image visualising the scale of the experiment.
    fn : str
        The filename to save the plot.
    outliers : list of int, optional
        Any specified indexes will be excluded. The default is [].
    legend : bool, optional
        Specifies whether to show a legend. Also uses a much smaller scale if 
        legend is not needed. The default is True.

    Returns
    -------
    None.
    """
    dis = get_points_dis(true_ps,pred_ps);
    data = np.array([[dis[i],conf[i],scores[i]] for i in range(len(conf)) if i not in outliers]);
    r = stats.pearsonr(data[:,0],data[:,1]);
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
    newax.imshow(target);
    ax.text(11.8,0.91,'r = {:.3f}'.format(r[0]),c='r');
    plt.savefig(fn);
    #plt.show()
    
def create_eval_folder():
    """
    Creates a folder named using the current date and time for storing the
    results of an experiment.

    Returns
    -------
    time_str : str
        The folder name.
    """
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
    """
    Draws a series of concentric circles around the point(s) specified.

    Parameters
    ----------
    crop : numpy.ndarray
        The image to be drawn on.
    ps : list
        The point(s) to draw around.

    Returns
    -------
    target : numpy.ndarray
        The image with circles drawn.
    """
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
    """
    Draws concentric circles around a point for smaller scales.

    Parameters
    ----------
    crop : numpy.ndarray
        The image to draw on.
    ps : numpy.ndarray
        The point to draw around.
    scale : float
        The scale difference between the pixel values to display and the
        actual pixels in the image.

    Returns
    -------
    target : numpy.ndarray
        The image with circles drawn.
    """
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
    