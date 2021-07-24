# -*- coding: utf-8 -*-
"""
Created on Sun May  9 14:53:40 2021

@author: dan
"""

import h5py;
import config as cf;
from hourglass import hg;
from face import face;
from blender import blender;
import cv2;
import utils.utils as ut;
import numpy as np;
from monocap import mc;
import utils.manual_tag as mt;
from tqdm import tqdm;
from openpose import op;
import json;
from utils import exp;
from utils import output;

if __name__ == "__main__":
    #Set the data source for the video frames/images
    #datapath = 'C:\\Users\\dan\\Documents\\HWU\\MSc\\Project\\testFrames\\';
    #datapath = 'C:\\Users\\dan\\Documents\\HWU\\MSc\\Project\\monocap\\data\\tennis\\';
    #datapath = 'C:\\Users\\dan\\Documents\\HWU\\MSc\\Project\\BL28n.mov';
    #datapath = 'C:\\Users\\dan\\Documents\\HWU\\MSc\\Project\\BF2l.mov';
    datapath = 'C:\\Users\\dan\\Documents\\HWU\\MSc\\Project\\BF1n.mov';
    #imagelist = [x for x in os.listdir(datapath) if x.endswith(".png")];
    
    nImg = 100; #Total number of image frames
    h_cent = 320; #Horizontal centre of the frame
    start_frame = 200; #Starting frame to cut off intro frames
    
    #Define how to get the frame image from the iterator
    def get_frame(cap):
        _,frame = cap.read();
        return frame;
        
    #Open a capture object on the video
    cap = cv2.VideoCapture(datapath);
    #Get a sample frame and a sample cropped frame
    frame_uc = get_frame(cap);
    frame_c = ut.crop(get_frame(cap),256,h_cent);
    #Set the capture object to the starting frame
    cap.set(cv2.CAP_PROP_POS_FRAMES,start_frame);
    
    print("Getting keypoints for each frame...");
    """frames = hg.get_frames('2021-07-07-14_14_09');
    def frame_gen(f):
        cap.set(cv2.CAP_PROP_POS_FRAMES,f-5);
        for j in range(6):
            yield ut.crop(get_frame(cap),256,h_cent);
    kpss = [];
    for f in tqdm(frames):
        fg = frame_gen(f);
        raw_kpss = op.get_kpss(fg,6);
        kpss.append(raw_kpss[-1]);
   
    #kpss = op.get_kpss(wrapper,(ut.crop(get_frame(cap),256,h_cent) for i in range(nImg)),nImg);
    smooth_kpss = op.smooth_kpss(kpss);
    
    #Release the capture object
    cap.release();"""
    
    with h5py.File('man_tags.h5') as f:
        #Data is stored in MPII format starting from neck
        ds = f['coords'];
        mpii = np.zeros((100,16,3),dtype=int);
        mpii[:,8:,:] = np.array(ds);
        mpii[:,7,:] = [107,135,0.5];
        mpii[:,6,:] = [108,188,0.5];
        mpii[:,2,:] = [60,208,0.5];
        mpii[:,3,:] = [152,194,0.5];
        mpii[:,4,:] = [101,201,0.5];
        mpii[:,1,:] = [130,243,0.5];
        mpii[:,5,:] = [88,255,0.5];
        mpii[:,0,:] = [129,255,0.5];
    kpss = op.mpii_to_openpose(mpii);
    
    """print("Converting keypoints to heatmaps...");
    hms = hg.gen_heatmaps(mpii, 256, 64);
        
    print("Starting MATLAB engine...");
    eng = mc.get_ml_eng();
    
    print("Converting heatmaps to MATLAB format...")
    mlhms = mc.hms_to_ml(hms);
    
    print("Loading pose dictionary...");
    bDict = mc.get_bone_data(eng);
    
    print("Calculating pose using MATLAB code...");
    #th_vis sets the threshold to ignore points under a specified confidence level
    output = eng.PoseFromVideo('heatmap',mlhms,'dict',bDict);
    
    print("Extracting 2D and 3D pose estimates...");
    preds_2d, preds_3d = mc.get_mc_pred(output,eng,frame_c,nImg);
    
    print("Scaling 3D points...");
    ps_3d = mc.get_3d_points(preds_3d);
    
    print("Calculating plots...");
    op_ps_3d = op.mpii_to_openpose(ps_3d);
    plts = {'d':[],'x':[],'y':[],'z':[]};
    for i in tqdm(range(nImg)):
        plts['d'].append(ut.get_3d_lines(op_ps_3d[i],scale=True));
        x,y,z = ut.get_2d_lines(op_ps_3d[i],scale=True);
        plts['x'].append(x);
        plts['y'].append(y);
        plts['z'].append(z);
    
    with h5py.File('mc_data.h5','w') as f:
        f.create_dataset('op_ps_3d',data=op_ps_3d);
        f.create_dataset('plts/x',data=plts['x']);
        f.create_dataset('plts/y',data=plts['y']);
        f.create_dataset('plts/z',data=plts['z']);
        f.create_dataset('plts/d',data=plts['d']);"""
      
    print('Loading saved Monocap data...');
    with h5py.File('mc_data.h5','r') as f:
        op_ps_3d = np.array(f['op_ps_3d']);
        plts = {};
        plts['x'] = np.array(f['plts/x']);
        plts['y'] = np.array(f['plts/y']);
        plts['z'] = np.array(f['plts/z']);
        plts['d'] = np.array(f['plts/d']);

    print("Isolating head...");
    tls = [];
    brs = [];
    efs = [];
    for i in tqdm(range(nImg)):
        if kpss[i] is not None and kpss[i][0][2] > 0.1 and kpss[i][1][2] > 0.1:
            tl,br = ut.crop_head(*kpss[i][0:2],256);
            fb = face.est_face_box(*kpss[i][0:2]);
        else:
            tl,br,fb = None,None,None;
        tls.append(tl);
        brs.append(br);
        efs.append(fb);
    
    sefs = face.scale_face_boxes(efs, tls, brs, 256);
        
    cap = cv2.VideoCapture(datapath);
    frame = get_frame(cap);
    uc = ut.get_uncrop(frame_uc,256,h_cent);
    cap.set(cv2.CAP_PROP_POS_FRAMES,start_frame);
    position_data = {};
    
    print("Idetifying facial features...");
    position_data['face_box'], position_data['face_landmarks'], position_data['head_angles'] = face.id_faces((get_frame(cap) for i in range(nImg)),tls,brs,uc,sefs);
    cap.release();
        
    print("Preparing render...");
    position_data['keypoints'] = [];
    for f in range(nImg):
        position_data['keypoints'].append({});
        for i,part in enumerate(op.parts):
            p = op_ps_3d[f][i];
            rp = np.array([p[0],p[2],-p[1]]);
            position_data['keypoints'][f][part] = rp;
    blender.prepare_render(position_data,face_only=False,end_frame=nImg);
    
    print("Rendering output...");
    ### UNCOMMENT TO RENDER OUTPUT ###
    #blender.render();
    ### UNCOMMENT TO RENDER OUTPUT ###
    
    mods = [output.basic_crop,output.highlight_points,output.draw_stickfigure,
            output.draw_head_box,output.extract_head,output.draw_face_box,
            output.draw_face_landmarks,output.show_render_face,output.blank];
    data = {'kpss':kpss,'position_data':position_data,'tls':tls,'brs':brs,
            'uc':uc,'sefs':sefs};
    output.show_results(datapath,nImg,mods,start_frame=start_frame,input_data=data);