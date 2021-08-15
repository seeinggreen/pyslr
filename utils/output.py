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
from hand import hand;

def show_results(datapath,nImg,mods,input_data=None,loop=True,fn=None,start_frame=None,frame_ids=None):
    """
    Displays the output of the system.

    Parameters
    ----------
    datapath : string
        The data[ath of the input video.
    nImg : int
        The total number of frames.
    mods : list of function
        The functions to generate the image for each module.
    input_data : dict, optional
        The input data to be accessed by the modules. The default is None.
    loop : bool, optional
        Will loop the results if True. The default is True.
    fn : string, optional
        Filename to store the output if specified. The default is None.
    start_frame : int, optional
        The start frame of the processed data. The default is None.
    frame_ids : list of int, optional
        The frame numbers of the processed frames if not sequential. The default is None.

    Returns
    -------
    None.
    """
    cap = cv2.VideoCapture(datapath);
    cont = True;
    #Write the results to video if filename is provided
    if fn is not None:
        out = cv2.VideoWriter('..\\'+fn,cv2.VideoWriter_fourcc(*'mjpa'), 10, (768*2,768));
        out_vid = True;
    else:
        out_vid = False;

    while cont:
        #Set the video to the correct start frame
        if start_frame is not None:
            cap.set(cv2.CAP_PROP_POS_FRAMES,start_frame);
        for i in range(nImg):
            #If the frames aren't sequential, jump to correct next frame
            if frame_ids is not None:
                cap.set(cv2.CAP_PROP_POS_FRAMES,frame_ids[i]);
            
            #Get the next frame and the rendered image for that frame
            _,frame = cap.read();
            crop = ut.crop(frame,256,320);
            render = cv2.imread(cf.renders_path + "{:04d}.png".format(i+1));
            
            #Add the images to the data dict or create it if needed
            data = {'frame':frame,'crop':crop,'render':render,'i':i};
            if input_data is not None:
                data.update(input_data);
            
            #Apply each module's function to the data to get the images
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
    """
    Just display the cropped frame.

    Parameters
    ----------
    data : dict
        The input data.

    Returns
    -------
    crop : numpy.ndarray
        The cropped frame.
    """
    return data['crop'];

def blank(data):
    """
    Display a black square.

    Parameters
    ----------
    data : dict
        The input data.

    Returns
    -------
    blank : numpy.ndarray
        A blank image.

    """
    return data['crop']*0;

def show_render_face(data):
    """
    Display the face of the rendered image.

    Parameters
    ----------
    data : dict
        The input data.

    Returns
    -------
    face : numpy.ndarray
        The face of the rendered image.
    """
    return cv2.resize(data['render'][20:180,330:490,:],(256,256));

def highlight_points(data):
    """
    Display the body keypoints highlighted.

    Parameters
    ----------
    data : dict
        The input data.

    Returns
    -------
    hls : numpy.ndarray
        The cropped image with the keypoints highlighted.
    """
    crop = np.array(data['crop']);
    if data['kpss'][data['i']] is not None:
        for p in range(0,8):
            ut.highlightPoint(crop,data['kpss'][data['i']][p],op.parts[p]);
    return crop;

def draw_stickfigure(data):
    """
    Display a stickfigure connecting the keypoints.

    Parameters
    ----------
    data : dict
        The input data.

    Returns
    -------
    stick : numpy.ndarray
        The cropped image with a stick figure added.
    """
    crop = np.array(data['crop']);
    if data['kpss'][data['i']] is not None:
        ut.draw_stick_figure(crop, data['kpss'][data['i']]);
    return crop;

def draw_head_box(data):
    """
    Displays a box around the head area.

    Parameters
    ----------
    data : dict
        The input data.

    Returns
    -------
    head : numpy.ndarray
        The cropped image with the head area highlighted.
    """
    crop = np.array(data['crop']);
    tl = data['tls'][data['i']];
    br = data['brs'][data['i']];
    cv2.rectangle(crop,tl,br,(255,255,0));
    return crop;

def draw_face_box(data):
    """
    Displays a box around the face area.

    Parameters
    ----------
    data : dict
        The input data.

    Returns
    -------
    face : numpy.ndarray
        The cropped image with the face area highlighted.
    """
    head = extract_head(data);
    face_box = data['position_data']['face_box'][data['i']];
    sefs = data['sefs'][data['i']];
    if face_box is not None:
        cv2.rectangle(head, *face_box, (0, 255, 0));
    else:
        cv2.rectangle(head, *sefs, (0, 0, 255));
    return head;

def extract_area(data,box):
    """
    Extracts the area from the frame using the given box.

    Parameters
    ----------
    data : dict
        The input data.
        
    box : tuple of tuple of int
        The top left and bottom right corners of the box to extract.

    Returns
    -------
    area : numpy.ndarray
        The extracted area.
    """
    if box is None or box[0] is None or box[1] is None or box[1][0] - box[0][0] == 0 or box[1][1] - box[0][1] == 0:
        box = ((0,0),(10,10));
    area = ut.extract_area(data['frame'],*box,data['uc'],256);
    return area;

def extract_head(data):
    """
    Extracts the head from the image.

    Parameters
    ----------
    data : dict
        The input data.

    Returns
    -------
    head : numpy.ndarray
        The extracred head area.
    """
    tl = data['tls'][data['i']];
    br = data['brs'][data['i']];
    head = extract_area(data,(tl,br));
    return head;

def draw_face_landmarks(data):
    """
    Displays the predicted face landmarks.

    Parameters
    ----------
    data : dict
        The input data.

    Returns
    -------
    face : numpy.ndarray
        The extracted head area with the facial landmarks highlighted.
    """
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

def draw_hand_box(data,box,c=[255,255,255]):
    """
    Displays a box around the hand in the given colour.

    Parameters
    ----------
    data : dict
        The input data.
    c : list of int
        The BGR colour to draw the box in.

    Returns
    -------
    hand : numpy.ndarray
        The cropped image with the hand boxed in the given colour.
    """
    crop = np.array(data['crop']);
    if box is not None:
        cv2.rectangle(crop, *box, c);
    return crop;

def draw_left_hand_box(data):
    """
    Displays the cropped image with a green box around the left hand.

    Parameters
    ----------
    data : dict
        The input data.

    Returns
    -------
    hand : numpy.ndarray
        The cropped image with the left hand boxed in green.
    """
    box = data['lhb'][data['i']];
    return draw_hand_box(data,box,[0,255,0]);

def draw_right_hand_box(data):
    """
    Displays the cropped image with a red box around the right hand.

    Parameters
    ----------
    data : dict
        The input data.

    Returns
    -------
    hand : numpy.ndarray
        The cropped image with the right hand boxed in red.
    """
    box = data['rhb'][data['i']];
    return draw_hand_box(data,box,[0,0,255]);

def extract_left_hand(data):
    """
    Extracts the area containing the left hand.

    Parameters
    ----------
    data : dict
        The input data.

    Returns
    -------
    hand : numpy.ndarray
        The area containing the left hand.
    """
    box = data['lhb'][data['i']];
    return extract_area(data,box);

def extract_right_hand(data):
    """
    Extracts the area containing the right hand.

    Parameters
    ----------
    data : dict
        The input data.

    Returns
    -------
    hand : numpy.ndarray
        The area containing the right hand.
    """
    box = data['rhb'][data['i']];
    return extract_area(data,box);

def draw_rh_lines(data):
    """
    Draws a stick image for the right hand.

    Parameters
    ----------
    data : dict
        The input data.

    Returns
    -------
    hand : numpy.ndarray
        The cropped image with the right hand keypoints connected with sticks.
    """
    #hnd = extract_right_hand(data);
    hnd = np.array(data['crop']);
    hand.draw_hand_lines(hnd,data['rhkpss'][data['i']]);
    return hnd;

def draw_lh_lines(data):
    """
    Draws a stick image for the left hand.

    Parameters
    ----------
    data : dict
        The input data.

    Returns
    -------
    hand : numpy.ndarray
        The cropped image with the left hand keypoints connected with sticks.
    """
    #hnd = extract_left_hand(data);
    hnd = np.array(data['crop']);
    hand.draw_hand_lines(hnd,data['lhkpss'][data['i']]);
    return hnd;

def rend_samples(data,start_frame,cap,fs=[30,51,78,90],show=False):
    """
    Outputs a sample of 4 rendered images with 3D keypoints plot and head from
    the input video.

    Parameters
    ----------
    data : dict
        The input data.
    start_frame : int
        The starting frame of the data to offset the supplied values.
    cap : cv2.VideoCapture
        A CV2 video capture object.
    fs : list of int, optional
        The frame numbers for the frames to display. The default is [30,51,78,90].
    show : bool, optional
        Displays the results on screen if True. The default is False.

    Returns
    -------
    grid : numpy.ndarray
        The grid of images.

    """
    row1 = [];
    row2 = [];
    row3 = [];
    for f in fs:
        cap.set(cv2.CAP_PROP_POS_FRAMES,start_frame+f);
        _,data['frame'] = cap.read();
        data['i'] = f;
        row1.append(extract_head(data));
        row2.append(data['plts']['y'][f]);
        render = cv2.imread(cf.renders_path + "{:04d}.png".format(f+1));
        render = cv2.resize(render,(256,256));
        row3.append(render);
    row1 = np.concatenate(row1,axis=1);
    row2 = np.concatenate(row2,axis=1);
    row3 = np.concatenate(row3,axis=1);
    grid = np.concatenate([row1,row2,row3]);
    if show:
        ut.show(grid);
    return grid;

def hand_samples(data,lfs,rfs,start_frame,cap,show=False):
    """
    Outputs a sample of 4 hand recognitions for the left and right.

    Parameters
    ----------
    data : dict
        The input data.
    lfs : list of int
        The frames to use for the left hand.
    rfs : list of int
        The frames to use for the right hand.
    start_frame : int
        The starting frame of the data to offset the supplied values.
    cap : cv2.VideoCapture
        A CV2 video capture object.N.
    show : bool, optional
        Displays the results on screen if True. The default is False.

    Returns
    -------
    grid : numpy.ndarray
        The grid of images.

    """
    data['lhkpss'] = hand.translate_hand_kps(data['lhkpss'],data['kpss'],7);
    data['rhkpss'] = hand.translate_hand_kps(data['rhkpss'],data['kpss'],4);
    row1 = [];
    row2 = [];
    for f in lfs:
        cap.set(cv2.CAP_PROP_POS_FRAMES,start_frame+f);
        _,frame = cap.read();
        data['crop'] = ut.crop(frame,256,320);
        data['i'] = f;
        row1.append(draw_lh_lines(data)[105:-65,105:-65]);
    for f in rfs:
        cap.set(cv2.CAP_PROP_POS_FRAMES,start_frame+f);
        _,frame = cap.read();
        data['crop'] = ut.crop(frame,256,320);
        data['i'] = f;
        row2.append(draw_rh_lines(data)[65:-105,65:-105]);
    row1 = np.concatenate(row1,axis=1);
    row2 = np.concatenate(row2,axis=1);
    grid = np.concatenate([row1,row2]);
    if show:
        ut.show(grid);
    return grid;