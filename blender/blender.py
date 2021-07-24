# -*- coding: utf-8 -*-
"""
Created on Sat Jun  5 14:23:25 2021

@author: dan
"""

import config as cf;
from hourglass import hg;
import numpy as np;
from scipy.spatial import distance;
import bpy;
import os;

def scale_to_root(locs,l_sh,r_sh,root):
    """
    Scale a model to its shoulder width with pelvis at the orgin.
    
    Takes a set of 3D joint locations and adjusts the position such that the 
    pelvis is at the origin and then scales the model such that the width 
    between the shoulders = 1 and other values are scaled to match.
    
    Parameters
    ----------
    locs : list of numpy.ndarray
        The joint locations of the model to be scaled.
    l_sh : numpy.ndarray
        The location data for the left shoulder.
    r_sh : numpy.ndarr
        The location data for the right shoulder.
    root : numpy.ndarr
        The location data for the pelvis.

    Returns
    -------
    scaled_locs : numpy.ndarray
        The new locations based on the new position and scale.

    """
    root = np.array(root);
    
    for pb in locs:
        locs[pb] -= root;

    dis = distance.euclidean(l_sh,r_sh);

    for pb in locs:
        locs[pb] /= dis;

    return locs;

def rot_point(p):
    """
    Rotates a given point so the model is displayed correctly.

    Parameters
    ----------
    p : numpy.ndarray
        A 3D coordinate.

    Returns
    -------
    rot_p : numpy.ndarray
        The rotated coordinate.

    """
    return np.array([p[0],p[2],-p[1]]);

def load_blend_file():
    """
    Loads a new Blender file if not loaded already.

    Returns
    -------
    None.

    """
    if bpy.data.filepath == '':
        bpy.ops.wm.open_mainfile(filepath=cf.blender_model_path);

def convert_points(named_points):
    """
    Coverts the given points to the same position and scale of the reference 
    Blender model.

    Parameters
    ----------
    ps_3d : numpy.ndarray
        The 3D points for each frame of the model.

    Returns
    -------
    scaled_ps : numpy.ndarray
        The points converted to the scale/position of the reference model.

    """    
    root = named_points[0]['pelv'];
    l_sh = named_points[0]['lsho'];
    r_sh = named_points[0]['rsho'];
    
    s_ps= [];
    #Put pelvis at origin and scale to shoulder width  
    for f in named_points:
        scaled_3d_points = scale_to_root(f,l_sh,r_sh,root);
        for pb in scaled_3d_points:
            scaled_3d_points[pb] = rot_point(scaled_3d_points[pb]);
        s_ps.append(scaled_3d_points);
    
    #Get Blender model pose bone locations
    load_blend_file();
    ob = bpy.context.object;
    locs = {};
    for pb in ob.pose.bones:
        wm = ob.convert_space(pose_bone=pb,matrix=pb.matrix,from_space='POSE',to_space='WORLD');
        locs[pb.name] = np.array(wm)[:-1,3];
    
    #Calculate BM shoulder width and root offset
    l_sh = locs['upperarm01.L'];
    r_sh = locs['upperarm01.R'];
    
    dis = distance.euclidean(l_sh,r_sh);
    root = locs['root'];
    
    #Scale and transpose 3D points by BM scale and offset
    for i,f in enumerate(s_ps):
        for pb in f:
            s_ps[i][pb] *= dis;
            s_ps[i][pb] += root;
            
    return s_ps;

def prepare_render(position_data,face_only=False,all_frames=True,end_frame=100):
    """
    Creates a temporary .blend file with the data to render.

    Parameters
    ----------
    extra_data : dict
        The data to be rendered.
    face_only : bool, optional
        Sets if only the face is to be rendered or the whole model. The 
        default is False.
    all_frames : bool, optional
        Sets if every frame will be a keyframe or just some of the frames. 
        The default is True.
    end_frame : int, optional
        Sets how many frames to render. The default is 100.

    Returns
    -------
    None.

    """
    load_blend_file();
    bpy.context.scene.render.filepath = cf.base_dir + "\\blenderRenders\\";
    lw = bpy.context.object.pose.bones.get("hand.ik.L");
    rw = bpy.context.object.pose.bones.get("hand.ik.R");
    lsho = bpy.context.object.pose.bones.get("upperarm01.L");
    rsho = bpy.context.object.pose.bones.get("upperarm01.R");
    head = bpy.context.object.pose.bones.get("head");
    head.rotation_mode = 'XYZ';
    
    get_arm_vectors(position_data);
    
    for i,f in enumerate(position_data['keypoints']):
        if all_frames:
            frame_num = i;
        else:
            frame_num = position_data['frame_num'][i];
        if not face_only:            
            position_wrist(position_data['vectors'][i]['lsho_lwri'],lsho,lw);
            """wm = ob.convert_space(pose_bone=lw,matrix=lw.matrix,from_space='POSE',to_space='WORLD');
            wm[0][3] = f['lwri'][0];
            wm[1][3] = f['lwri'][1];
            wm[2][3] = f['lwri'][2];
            pm = ob.convert_space(pose_bone=lw,matrix=wm,from_space='WORLD',to_space='POSE');
            lw.matrix = pm;"""
            lw.keyframe_insert('location',frame=frame_num);
            
            position_wrist(position_data['vectors'][i]['rsho_rwri'],rsho,rw);
            """wm = ob.convert_space(pose_bone=rw,matrix=rw.matrix,from_space='POSE',to_space='WORLD');
            wm[0][3] = f['rwri'][0];
            wm[1][3] = f['rwri'][1];
            wm[2][3] = f['rwri'][2];
            pm = ob.convert_space(pose_bone=rw,matrix=wm,from_space='WORLD',to_space='POSE');
            rw.matrix = pm;"""
            rw.keyframe_insert('location',frame=frame_num);
        
        head.rotation_euler = position_data['head_angles'][i];
        head.keyframe_insert('rotation_euler',frame=frame_num);
    
    bpy.context.scene.frame_end = end_frame;
    bpy.ops.wm.save_as_mainfile(filepath=cf.base_dir + '\\blend.blend');
    
def render():
    """
    Starts a new background instance of Blender to render the frames stored 
    in the temporary file.

    Returns
    -------
    None.

    """
    os.system(cf.blend_exe + " ../blend.blend --background --python blender/bender_render.py")
    
def get_arm_vectors(position_data):
    position_data['vectors'] = [];
    for i,kps in enumerate(position_data['keypoints']):
        position_data['vectors'].append({});
        
        left_arm_points = [kps['lsho'],kps['lelb'],kps['lwri']];
        right_arm_points = [kps['rsho'],kps['relb'],kps['rwri']];
        
        bones = bpy.context.object.pose.bones;
        world_locs = {};
        for b in ['upperarm01.L','upperarm01.R','lowerarm01.L','lowerarm01.R','wrist.L','wrist.R']: 
            wm = bpy.context.object.convert_space(pose_bone=bones[b],matrix=bones[b].matrix,from_space='POSE',to_space='WORLD');
            world_locs[b] = np.array(wm)[:-1,3];
        
        left_sho_to_elb = world_locs['lowerarm01.L'] - world_locs['upperarm01.L'];
        right_sho_to_elb = world_locs['lowerarm01.R'] - world_locs['upperarm01.R'];
        
        left_elb_to_wri = world_locs['wrist.L'] - world_locs['lowerarm01.L'];
        right_elb_to_wri = world_locs['wrist.R'] - world_locs['lowerarm01.R'];
        
        new_left_elb, new_left_wri = scale_arm(left_arm_points,left_sho_to_elb,left_elb_to_wri);
        new_right_elb, new_right_wri = scale_arm(right_arm_points,right_sho_to_elb,right_elb_to_wri);
        
        left_sho_to_wri = new_left_wri - kps['lsho'];
        right_sho_to_wri = new_right_wri - kps['rsho'];
        
        position_data['vectors'][i]['lsho_lwri'] = left_sho_to_wri;
        position_data['vectors'][i]['rsho_rwri'] = right_sho_to_wri;
        position_data['vectors'][i]['new_lelb'] = new_left_elb;
        position_data['vectors'][i]['new_lwri'] = new_left_wri;
        position_data['vectors'][i]['new_relb'] = new_right_elb;
        position_data['vectors'][i]['new_rwri'] = new_right_wri;
        
    
def scale_arm(keypoints,model_sho_to_elb,model_elb_to_wri):
    #Calculate the model's arm lengths
    model_up_arm = np.linalg.norm(model_sho_to_elb);
    model_low_arm = np.linalg.norm(model_elb_to_wri);
    
    sho_to_elb = keypoints[1] - keypoints[0];
    up_arm = np.linalg.norm(sho_to_elb);
    up_arm_scale = model_up_arm / up_arm;
    up_arm_delta = (1 - up_arm_scale) * sho_to_elb;
    
    new_elb = keypoints[1] - up_arm_delta;

    elb_to_wri = keypoints[2] - keypoints[1];
    low_arm = np.linalg.norm(elb_to_wri);
    low_arm_scale = model_low_arm / low_arm;
    low_arm_delta = (1 - low_arm_scale) * elb_to_wri;
    
    new_wri = keypoints[2] - up_arm_delta - low_arm_delta;
    
    return new_elb,new_wri;

def scale_arms(preds,model):
    new_preds = {};
    #left
    pred_sh = preds['lsho'];
    pred_el = preds['lelb'];
    pred_wr = preds['lwri'];
    mod_sh_to_el = model['lowerarm01.L'] - model['upperarm01.L'];
    mod_el_to_wr = model['wrist.L'] - model['lowerarm01.L'];
    new_preds['lelb'],new_preds['lwri'] = scale_arm(pred_sh,pred_el,pred_wr,mod_sh_to_el,mod_el_to_wr);
    
    l_sh_to_new_wr = new_preds['lwri'] - preds['lsho'];
    
    #right
    pred_sh = preds['rsho'];
    pred_el = preds['relb'];
    pred_wr = preds['rwri'];
    mod_sh_to_el = model['lowerarm01.R'] - model['upperarm01.R'];
    mod_el_to_wr = model['wrist.R'] - model['lowerarm01.R'];
    new_preds['relb'],new_preds['rwri'] = scale_arm(pred_sh,pred_el,pred_wr,mod_sh_to_el,mod_el_to_wr);
    r_sh_to_new_wr = new_preds['rwri'] - preds['rsho'];
    
    return new_preds,l_sh_to_new_wr,r_sh_to_new_wr;

def get_wrist_vectors(ps,model):
    _,lv,rv = scale_arms(ps,model);
    #lv[1] *= -1;
    #rv[1] *= -1;

    return lv,rv;

def position_wrist(vec,sho,wri):
    wm_sho = bpy.context.object.convert_space(pose_bone=sho,matrix=sho.matrix,from_space='POSE',to_space='WORLD');
    wm_wri = bpy.context.object.convert_space(pose_bone=wri,matrix=wri.matrix,from_space='POSE',to_space='WORLD');
    for i in range(3):
        wm_wri[i][3] = wm_sho[i][3] + vec[i];
    pm_wri = bpy.context.object.convert_space(pose_bone=wri,matrix=wm_wri,from_space='WORLD',to_space='POSE');
    wri.matrix = pm_wri;
    
"""def scale_arm(pred_sh,pred_el,pred_wr,mod_sh_to_el,mod_el_to_wr):    
    mod_ua = np.linalg.norm(mod_sh_to_el);
    mod_la = np.linalg.norm(mod_el_to_wr);
    
    pred_sh_to_el = pred_el - pred_sh;
    pred_ua = np.linalg.norm(pred_sh_to_el);
    ua_s = mod_ua / pred_ua;
    ua_del = (1 - ua_s) * pred_sh_to_el;
    
    pred_el_new = pred_el - ua_del;

    pred_el_to_wr = pred_wr - pred_el;
    pred_la = np.linalg.norm(pred_el_to_wr);
    la_s = mod_la / pred_la;
    la_del = (1 - la_s) * pred_el_to_wr;
    
    pred_wr_new = pred_wr - ua_del - la_del;
    
    return pred_el_new,pred_wr_new;

def scale_arms(preds,model):
    new_preds = {};
    #left
    pred_sh = preds['lsho'];
    pred_el = preds['lelb'];
    pred_wr = preds['lwri'];
    mod_sh_to_el = model['lowerarm01.L'] - model['upperarm01.L'];
    mod_el_to_wr = model['wrist.L'] - model['lowerarm01.L'];
    new_preds['lelb'],new_preds['lwri'] = scale_arm(pred_sh,pred_el,pred_wr,mod_sh_to_el,mod_el_to_wr);
    
    l_sh_to_new_wr = new_preds['lwri'] - preds['lsho'];
    
    #right
    pred_sh = preds['rsho'];
    pred_el = preds['relb'];
    pred_wr = preds['rwri'];
    mod_sh_to_el = model['lowerarm01.R'] - model['upperarm01.R'];
    mod_el_to_wr = model['wrist.R'] - model['lowerarm01.R'];
    new_preds['relb'],new_preds['rwri'] = scale_arm(pred_sh,pred_el,pred_wr,mod_sh_to_el,mod_el_to_wr);
    r_sh_to_new_wr = new_preds['rwri'] - preds['rsho'];
    
    return new_preds,l_sh_to_new_wr,r_sh_to_new_wr;

def get_wrist_vectors(ps,model):
    _,lv,rv = scale_arms(ps,model);
    #lv[1] *= -1;
    #rv[1] *= -1;

    return lv,rv;

def position_wrist(vec,sho,wri):
    wm_sho = bpy.context.object.convert_space(pose_bone=sho,matrix=sho.matrix,from_space='POSE',to_space='WORLD');
    wm_wri = wm_sho;
    for i in range(3):
        wm_wri[i][3] += vec[i];
    pm_wri = bpy.context.object.convert_space(pose_bone=wri,matrix=wm_wri,from_space='WORLD',to_space='POSE');
    wri.matrix = pm_wri;"""