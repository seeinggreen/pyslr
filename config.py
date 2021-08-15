##############################################################################
#                    SETUP CONFIG FOR EXTERNAL LOCATIONS                     #
##############################################################################

#Base directory - useful if all other folders are subfolders
base_dir = "C:\\Users\\dan\\Documents\\HWU\\MSc\\Project";

#Hourglass - location of Hourglass repo
hg_dir = base_dir + "\\pytorch_stacked_hourglass";

#Monocap - location on Monocap repo and pose dictionary
mc_dir = base_dir + "\\monocap";
pose_dict_path = mc_dir + "\\dict\\poseDict-all-K128.mat";

#Blender - location of Blender model file, Blender .exe and folder to store
#output for renders
blender_model_path = base_dir + "\\testMan2.blend";
blend_exe = '"C:\\Program Files\\Blender Foundation\\Blender 2.91\\blender.exe"';
renders_path = base_dir + "\\blenderRenders\\";

#Openpose - location of Openpose build
openpose_dir = base_dir + "\\openpose";

#Directory for storing output of experiments etc.
eval_dir = base_dir + "\\eval";

##############################################################################
#                    SETUP CONFIG FOR EXTERNAL LOCATIONS                     #
##############################################################################