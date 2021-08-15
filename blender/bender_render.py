#Very simple script to be run from within Blender to render the file given the
#predefined settings in the file
import bpy
bpy.ops.render.render(animation=True)