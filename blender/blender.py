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
    root = np.array(root);
    
    for pb in locs:
        locs[pb] -= root;

    dis = distance.euclidean(l_sh,r_sh);

    for pb in locs:
        locs[pb] /= dis;

    return locs;

def rot_point(p):
    return np.array([p[0],p[2],-p[1]]);

def load_blend_file():
    if bpy.data.filepath == '':
        bpy.ops.wm.open_mainfile(filepath=cf.blender_model_path);

def convert_points(ps_3d):
    #Convert to dict of named bones
    named_points = [];
    
    for f in ps_3d:
        named_point = {};
        for i,name in enumerate(hg.parts['mpii']):
            named_point[name] = f[i];
        named_points.append(named_point);
    
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

def prepare_render(s_ps,tilts):
    load_blend_file();
    ob = bpy.context.object;
    bpy.context.scene.render.filepath = cf.base_dir + "\\blenderRenders\\";
    lw = bpy.context.object.pose.bones.get("hand.ik.L");
    rw = bpy.context.object.pose.bones.get("hand.ik.R");
    head = bpy.context.object.pose.bones.get("head");
    head.rotation_mode = 'XYZ';
    
    for i,f in enumerate(s_ps):
        wm = ob.convert_space(pose_bone=lw,matrix=lw.matrix,from_space='POSE',to_space='WORLD');
        wm[0][3] = f['lwri'][0];
        wm[1][3] = f['lwri'][1];
        wm[2][3] = f['lwri'][2];
        pm = ob.convert_space(pose_bone=lw,matrix=wm,from_space='WORLD',to_space='POSE');
        lw.matrix = pm;
        lw.keyframe_insert('location',frame=i);
        
        wm = ob.convert_space(pose_bone=rw,matrix=rw.matrix,from_space='POSE',to_space='WORLD');
        wm[0][3] = f['rwri'][0];
        wm[1][3] = f['rwri'][1];
        wm[2][3] = f['rwri'][2];
        pm = ob.convert_space(pose_bone=rw,matrix=wm,from_space='WORLD',to_space='POSE');
        rw.matrix = pm;
        rw.keyframe_insert('location',frame=i);
        
        head.rotation_euler[2] = tilts[i];
        head.keyframe_insert('rotation_euler',frame=i);
    
    bpy.context.scene.frame_end = 100;
    bpy.ops.wm.save_as_mainfile(filepath=cf.base_dir + '\\blend.blend');
    
def render():
    os.system(cf.blend_exe + " ../blend.blend --background --python blender/bender_render.py")
