# -*- coding: utf-8 -*-
"""
Created on Sat Jun  5 14:23:25 2021

@author: dan
"""

import config as cf;
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
            lw.keyframe_insert('location',frame=frame_num);
            
            position_wrist(position_data['vectors'][i]['rsho_rwri'],rsho,rw);
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
    """
    Takes the inferred position data and calculates the arm positions as
    vectors from the shoulders, using the arm lengths of the generated model
    already loaded in Blender.

    Parameters
    ----------
    position_data : dict
        Dictionary containing the keypoints for the arms.

    Returns
    -------
    None (adds results to position_data dict).
    """
    position_data['vectors'] = [];
    for i,kps in enumerate(position_data['keypoints']):
        #Add dict for vectors to be stored
        position_data['vectors'].append({});
        
        #Separate the left and right arms
        left_arm_points = [kps['lsho'],kps['lelb'],kps['lwri']];
        right_arm_points = [kps['rsho'],kps['relb'],kps['rwri']];
        
        #Get the locations in world space for each of the points
        bones = bpy.context.object.pose.bones;
        world_locs = {};
        for b in ['upperarm01.L','upperarm01.R','lowerarm01.L','lowerarm01.R','wrist.L','wrist.R']: 
            wm = bpy.context.object.convert_space(pose_bone=bones[b],matrix=bones[b].matrix,from_space='POSE',to_space='WORLD');
            #Extract XYZ coordinates from 4x4 matrix
            world_locs[b] = np.array(wm)[:-1,3];
        
        #Calculate vectors for shoulder -> elbow and elbow -> wrist
        left_sho_to_elb = world_locs['lowerarm01.L'] - world_locs['upperarm01.L'];
        right_sho_to_elb = world_locs['lowerarm01.R'] - world_locs['upperarm01.R'];
        
        left_elb_to_wri = world_locs['wrist.L'] - world_locs['lowerarm01.L'];
        right_elb_to_wri = world_locs['wrist.R'] - world_locs['lowerarm01.R'];
        
        #Scale the vectors to the correct length
        new_left_elb, new_left_wri = scale_arm(left_arm_points,left_sho_to_elb,left_elb_to_wri);
        new_right_elb, new_right_wri = scale_arm(right_arm_points,right_sho_to_elb,right_elb_to_wri);
        
        #Calculate shoulder -> wrist vectors
        left_sho_to_wri = new_left_wri - kps['lsho'];
        right_sho_to_wri = new_right_wri - kps['rsho'];
        
        #Store vectors in position_data dict
        position_data['vectors'][i]['lsho_lwri'] = left_sho_to_wri;
        position_data['vectors'][i]['rsho_rwri'] = right_sho_to_wri;
        position_data['vectors'][i]['new_lelb'] = new_left_elb;
        position_data['vectors'][i]['new_lwri'] = new_left_wri;
        position_data['vectors'][i]['new_relb'] = new_right_elb;
        position_data['vectors'][i]['new_rwri'] = new_right_wri;
        
    
def scale_arm(keypoints,model_sho_to_elb,model_elb_to_wri):
    """
    Scales arm vectors to the lengths of the generated 3D model.

    Parameters
    ----------
    keypoints : list of numpy.ndarray
        Keypoints for the shoulder, elbow and wrist.
    model_sho_to_elb : numpy.ndarray
        Vector from shoulder to elbow.
    model_elb_to_wri : numpy.ndarray
        Vector from elbow to wrist.

    Returns
    -------
    new_elb : numpy.ndarray
        New elbow position.
    new_wri : numpy.ndarray
        New wrist position.
    """
    #Calculate the model's arm lengths
    model_up_arm = np.linalg.norm(model_sho_to_elb);
    model_low_arm = np.linalg.norm(model_elb_to_wri);
    
    #Calculate shoulder to elbow vector
    sho_to_elb = keypoints[1] - keypoints[0];
    
    #Calculate upper arm length and delta between the two models
    up_arm = np.linalg.norm(sho_to_elb);
    up_arm_scale = model_up_arm / up_arm;
    up_arm_delta = (1 - up_arm_scale) * sho_to_elb;
    
    #Reposition the elbow
    new_elb = keypoints[1] - up_arm_delta;

    #Calculate elbow to wrist vector
    elb_to_wri = keypoints[2] - keypoints[1];
    
    #Calculate forearm length and delta between the two models
    low_arm = np.linalg.norm(elb_to_wri);
    low_arm_scale = model_low_arm / low_arm;
    low_arm_delta = (1 - low_arm_scale) * elb_to_wri;
    
    #Reposition the wrist based on both deltas
    new_wri = keypoints[2] - up_arm_delta - low_arm_delta;
    
    return new_elb,new_wri;

def position_wrist(vec,sho,wri):
    """
    Takes the vector shoulder -> wrist vector and the shoulder/wrist bones and
    positions the wrist in the correct position.

    Parameters
    ----------
    vec : numpy.ndarray
        The vector from the shoulder to the wrist.
    sho : bpy.types.PoseBone
        Pose bone for the shoulder.
    wri : bpy.types.PoseBone
        Pose bone for the wrist.

    Returns
    -------
    None.
    """
    #Get the world-space matrices for the shoulder and wrist
    wm_sho = bpy.context.object.convert_space(pose_bone=sho,matrix=sho.matrix,from_space='POSE',to_space='WORLD');
    wm_wri = bpy.context.object.convert_space(pose_bone=wri,matrix=wri.matrix,from_space='POSE',to_space='WORLD');
    #Set the XYZ coordinates to the shoulder + the vector
    for i in range(3):
        wm_wri[i][3] = wm_sho[i][3] + vec[i];
    #Convert back to pose-space and move the wrist
    pm_wri = bpy.context.object.convert_space(pose_bone=wri,matrix=wm_wri,from_space='WORLD',to_space='POSE');
    wri.matrix = pm_wri;