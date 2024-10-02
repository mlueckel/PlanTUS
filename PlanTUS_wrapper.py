#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Maximilian Lueckel, mlueckel@uni-mainz.de
"""
#==============================================================================
#==============================================================================
# Specify variables
#==============================================================================
#==============================================================================

#==============================================================================
# Subject-specific variables
#==============================================================================

# Path to T1 image (output for SimNIBS' charm)
#t1_filepath = '/path/to/m2m_SubjectID/T1.nii.gz'
t1_filepath = '/media/maximilian/e4713b47-344e-4ac6-85dd-b6769e0cbfa82/m2m_Suhas/T1.nii.gz'

# Path to head mesh (.msh file generated by SimNIBS' charm)
#simnibs_mesh_filepath = '/path/to/m2m_SubjectID/SubjectID.msh'
simnibs_mesh_filepath = '/media/maximilian/e4713b47-344e-4ac6-85dd-b6769e0cbfa82/m2m_Suhas/Suhas.msh'

# Path to mask of target region of interest (in same space as T1 image)
# Note: PlanTUS output folder will have same name as this file
#target_roi_filepath = '/path/to/target_roi.nii.gz'
target_roi_filepath = '/media/maximilian/e4713b47-344e-4ac6-85dd-b6769e0cbfa82/m2m_Suhas/handknob_left.nii.gz'


#==============================================================================
# Transducer-specific variables
#==============================================================================

# Maximum and minimum focal depth of transducer (in mm)
max_distance = 76.6 # note: max. focal depth of NeuroFUS CTX-545 in Mainz
min_distance = 33.1 # note: min. focal depth of NeuroFUS CTX-545 in Mainz

# Aperture diameter (in mm)
transducer_diameter = 65

# Maximum allowed angle for tilting of TUS transducer (in degrees)
max_angle = 10

# Offset between radiating surface and exit plane of transducer (in mm)
plane_offset = 10.82 # NeuroFUS CTX-500/CTX-545
#plane_offset = 0

# Additional offset between skin and exit plane of transducer (in mm;
# e.g., due to addtional gel/silicone pad)
additional_offset = 3

# Focal distance and corresponding FLHM values (both in mm) according to, e.g.,
# calibration report
focal_distance_list = [33.1, 34.7, 36.8, 39.3, 42.4, 46.2, 50.8, 56.0, 61.8, 68.5, 76.6]
flhm_list = [8.4, 9.3, 10.3, 11.6, 13.5, 16.1, 19.2, 22.8, 26.3, 30.2, 34.5]


#==============================================================================
# PlanTUS paths
#==============================================================================

# PlanTUS code folder
#plantus_main_folder = '/path/to/PlanTUS/'
plantus_main_folder = '/media/maximilian/e4713b47-344e-4ac6-85dd-b6769e0cbfa82/PlanTUS/'
# Paths to ...
# planning scene template
planning_scene_template_filepath = plantus_main_folder + '/resources/scene_templates/TUSTransducerPlacementPlanning_TEMPLATE.scene'
# placement scene template
placement_scene_template_filepath = plantus_main_folder + '/resources/scene_templates/TUSTransducerPlacement_TEMPLATE.scene'
# transducer model (leave empty to generate generic model)
transducer_surface_model_filepath = plantus_main_folder + '/resources/transducer_models/TRANSDUCER-NEUROFUS-CTX-500-4_DEVICE.surf.gii'
#transducer_surface_model_filepath = ""
# PlanTUS code
plantus_code_path =  plantus_main_folder + '/code'


#=============================================================================
# Run PlanTUS
#=============================================================================
import os
import shutil
import numpy as np
import math
import subprocess
import re
import threading
from pynput import mouse

os.chdir(plantus_code_path)
import PlanTUS

#=============================================================================
#
#=============================================================================

target_roi_filename = os.path.split(target_roi_filepath)[1]
target_roi_name = target_roi_filename.replace(".nii", "")
target_roi_name = target_roi_name.replace(".gz", "")

output_path = os.path.split(simnibs_mesh_filepath)[0]
output_path = output_path + "/PlanTUS/" + target_roi_name
os.makedirs(output_path, exist_ok=True)

shutil.copy(target_roi_filepath, output_path + "/")
target_roi_filepath = output_path + "/" + target_roi_filename


#=============================================================================
# Convert SimNIBS mesh(es) to surface file(s)
#=============================================================================

# Skin
PlanTUS.convert_simnibs_mesh_to_surface(simnibs_mesh_filepath, [1005], "skin", output_path)

PlanTUS.add_structure_information(output_path + "/skin.surf.gii", "CORTEX_LEFT")

# Skull
PlanTUS.convert_simnibs_mesh_to_surface(simnibs_mesh_filepath, [1007, 1008], "skull", output_path)

PlanTUS.add_structure_information(output_path + "/skull.surf.gii", "CORTEX_RIGHT")


#==============================================================================
#
#==============================================================================

PlanTUS.create_avoidance_mask(simnibs_mesh_filepath, output_path + "/skin.surf.gii", transducer_diameter/2)


#==============================================================================
#
#==============================================================================

# distances between skin and (center of) target
target_center = PlanTUS.roi_center_of_gravity(target_roi_filepath)
skin_target_distances = PlanTUS.distance_between_surface_and_point(output_path + "/skin.surf.gii", target_center)
PlanTUS.create_metric_from_pseudo_nifti("distances", skin_target_distances, output_path + "/skin.surf.gii")
PlanTUS.mask_metric(output_path + "/distances_skin.func.gii", output_path + "/avoidance_skin.func.gii")
PlanTUS.add_structure_information(output_path + "/distances_skin.func.gii", "CORTEX_LEFT")
PlanTUS.threshold_metric(output_path + "/distances_skin.func.gii", max_distance)
PlanTUS.mask_metric(output_path + "/distances_skin_thresholded.func.gii", output_path + "/avoidance_skin.func.gii")
PlanTUS.add_structure_information(output_path + "/distances_skin_thresholded.func.gii", "CORTEX_LEFT")


# angles between skin normal vectors and skin-target vectors
skin_target_angles = []
_, skin_normals = PlanTUS.compute_surface_metrics(output_path + "/skin.surf.gii")
skin_target_vectors = PlanTUS.vectors_between_surface_and_point(output_path + "/skin.surf.gii", target_center)
for i in np.arange(len(skin_target_vectors)):
    skin_target_angles.append((math.degrees(PlanTUS.angle_between_vectors(skin_target_vectors[i], skin_normals[i]))))
skin_target_angles = np.abs(np.asarray(skin_target_angles))
PlanTUS.create_metric_from_pseudo_nifti("angles", skin_target_angles, output_path + "/skin.surf.gii")
PlanTUS.mask_metric(output_path + "/angles_skin.func.gii", output_path + "/avoidance_skin.func.gii")
PlanTUS.add_structure_information(output_path + "/angles_skin.func.gii", "CORTEX_LEFT")
#PlanTUS.smooth_metric(output_path + "/angles_skin.func.gii", output_path + "/skin.surf.gii", transducer_diameter)
#PlanTUS.mask_metric(output_path + "/angles_skin_s" + str(transducer_diameter) + ".func.gii", output_path + "/avoidance_skin.func.gii")
#PlanTUS.add_structure_information(output_path + "/angles_skin_s" + str(transducer_diameter) + ".func.gii", "CORTEX_LEFT")


# intersection between skin normal vectors and target region
PlanTUS.stl_from_nii(target_roi_filepath, 0.25)
skin_coordinates, skin_normals = PlanTUS.compute_surface_metrics(output_path + "/skin.surf.gii")
skin_target_intersections = PlanTUS.compute_vector_mesh_intersections(skin_coordinates, skin_normals, output_path + "/" + target_roi_name + "_3Dmodel.stl", 200)

skin_target_intersection_values = []
for i in np.arange(len(skin_target_intersections)):
    if len(skin_target_intersections[i]) == 1:
        skin_target_intersection_values.append(0)
    elif len(skin_target_intersections[i]) == 2:
        d = np.linalg.norm(np.asarray(skin_target_intersections[i][1])-np.asarray(skin_target_intersections[i][0]))
        skin_target_intersection_values.append(d)
    elif len(skin_target_intersections[i]) == 3:
        d = np.linalg.norm(np.asarray(skin_target_intersections[i][1])-np.asarray(skin_target_intersections[i][0]))
        skin_target_intersection_values.append(d)
    elif len(skin_target_intersections[i]) == 4:
        d = (np.linalg.norm(np.asarray(skin_target_intersections[i][1])-np.asarray(skin_target_intersections[i][0]))+(np.linalg.norm(np.asarray(skin_target_intersections[i][3])-np.asarray(skin_target_intersections[i][2]))))
        skin_target_intersection_values.append(d)
    elif len(skin_target_intersections[i]) > 4:
        skin_target_intersection_values.append (np.nan)
    else:
        skin_target_intersection_values.append(0)
skin_target_intersection_values = np.asarray(skin_target_intersection_values)

PlanTUS.create_metric_from_pseudo_nifti("target_intersection", skin_target_intersection_values, output_path + "/skin.surf.gii")
PlanTUS.mask_metric(output_path + "/target_intersection_skin.func.gii", output_path + "/avoidance_skin.func.gii")
PlanTUS.add_structure_information(output_path + "/target_intersection_skin.func.gii", "CORTEX_LEFT")
#PlanTUS.smooth_metric(output_path + "/target_intersection_skin.func.gii", output_path + "/skin.surf.gii", transducer_diameter)
#PlanTUS.mask_metric(output_path + "/target_intersection_skin_s" + str(transducer_diameter) + ".func.gii", output_path + "/avoidance_skin.func.gii")
#PlanTUS.add_structure_information(output_path + "/target_intersection_skin_s" + str(transducer_diameter) + ".func.gii", "CORTEX_LEFT")


# angles between skin and skull normals
skin_coordinates, skin_normals = PlanTUS.compute_surface_metrics(output_path + "/skin.surf.gii")
skull_coordinates, skull_normals = PlanTUS.compute_surface_metrics(output_path + "/skull.surf.gii")
skin_skull_intersections = PlanTUS.compute_vector_mesh_intersections(skin_coordinates, skin_normals, output_path + "/skull.stl", 40)

indices_closest_skull_vertices = []
for i in np.arange(len(skin_coordinates)):
    try: 
        intersection_coordinate = skin_skull_intersections[i][0]
        ED_skull_list = np.linalg.norm((skull_coordinates - intersection_coordinate), axis=1)
        indices_closest_skull_vertices.append((np.argmin(ED_skull_list)))
    except:
        indices_closest_skull_vertices.append((np.nan))
indices_closest_skull_vertices = np.asarray(indices_closest_skull_vertices).astype(int)

skin_skull_angle_list = []
for i in np.arange(len(skin_coordinates)):
    try:
        skin_normal = skin_normals[i]
        skull_normal = skull_normals[indices_closest_skull_vertices[i]]
        skin_skull_angle = math.degrees(PlanTUS.angle_between_vectors(skin_normal, skull_normal))
        skin_skull_angle_list.append(skin_skull_angle)
    except:
        skin_skull_angle_list.append(0)
skin_skull_angles = np.asarray(skin_skull_angle_list)

PlanTUS.create_metric_from_pseudo_nifti("skin_skull_angles", skin_skull_angles, output_path + "/skin.surf.gii")
PlanTUS.mask_metric(output_path + "/skin_skull_angles_skin.func.gii", output_path + "/avoidance_skin.func.gii")
PlanTUS.add_structure_information(output_path + "/skin_skull_angles_skin.func.gii", "CORTEX_LEFT")
#PlanTUS.smooth_metric(output_path + "/skin_skull_angles_skin.func.gii", output_path + "/skin.surf.gii", transducer_diameter)
#PlanTUS.mask_metric(output_path + "/skin_skull_angles_skin_s" + str(transducer_diameter) + ".func.gii", output_path + "/avoidance_skin.func.gii")
#PlanTUS.add_structure_information(output_path + "/skin_skull_angles_skin_s" + str(transducer_diameter) + ".func.gii", "CORTEX_LEFT")




scene_variable_names = [
    'SKIN_SURFACE_FILENAME',
    'SKIN_SURFACE_FILEPATH',
    'SKULL_SURFACE_FILENAME',
    'SKULL_SURFACE_FILEPATH',
    'DISTANCES_FILENAME',
    'DISTANCES_FILEPATH',
    'INTERSECTION_FILENAME',
    'INTERSECTION_FILEPATH',
    'ANGLES_FILENAME',
    'ANGLES_FILEPATH',
    'ANGLES_SKIN_SKULL_FILENAME',
    'ANGLES_SKIN_SKULL_FILEPATH',
    'DISTANCES_MAX_FILENAME',
    'DISTANCES_MAX_FILEPATH',
    'T1_FILENAME',
    'T1_FILEPATH',
    'MASK_FILENAME',
    'MASK_FILEPATH']

scene_variable_values = [
    'skin.surf.gii',
    './skin.surf.gii',
    'skull.surf.gii',
    './skull.surf.gii',
    'distances_skin.func.gii',
    './distances_skin.func.gii',
    'target_intersection_skin.func.gii',
    './target_intersection_skin.func.gii',
    'angles_skin.func.gii',
    './angles_skin.func.gii',
    'skin_skull_angles_skin.func.gii',
    './skin_skull_angles_skin.func.gii',
    'distances_skin_thresholded.func.gii',
    './distances_skin_thresholded.func.gii',
    'T1.nii.gz',
    '../../T1.nii.gz',
    target_roi_filename,
    './' + target_roi_filename]


PlanTUS.create_scene(planning_scene_template_filepath, output_path + "/scene.scene", scene_variable_names, scene_variable_values)

# Define the command
command = "wb_view -logging FINER " + output_path + "/scene.scene"

# Regular expression pattern to match the phrase and the number
pattern = re.compile(r"Switched vertex to triangle nearest vertex\s+(\.\d+)")

# Initialize the variable to store the number and a flag to trigger processing
triangle_number = None
process_line = False

# Function to monitor mouse clicks
def on_click(x, y, button, pressed):
    global process_line
    if pressed:
        process_line = True

# Start listening for mouse clicks in a separate thread
listener = mouse.Listener(on_click=on_click)
listener.start()

# Start the process to run the command
process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, cwd=output_path, text=True)

# Function to read the process output
def read_output():
    global triangle_number, process_line

    while True:
        output = process.stderr.readline()
        if output == '' and process.poll() is not None:
            break
        
        if process_line:
            # Process the latest line only when a mouse click is detected
            match = pattern.search(output)
            if match:
                triangle_number = match.group(1)
                triangle_number = int(triangle_number.replace(".", ""))
                print(f"Switched vertex to triangle nearest vertex: {triangle_number}")
                
                # Ask the user if they want to generate the transducer placement
                response = input(f"Generate transducer placement for vertex {triangle_number}? (yes/no): ").strip().lower()
                if response == "yes":
                    print(f"Generating transducer placement for vertex {triangle_number}")
                    PlanTUS.prepare_acoustic_simulation(triangle_number,
                                                        output_path,
                                                        target_roi_filepath,
                                                        t1_filepath,
                                                        max_distance,
                                                        min_distance,
                                                        transducer_diameter,
                                                        max_angle,
                                                        plane_offset,
                                                        additional_offset,
                                                        transducer_surface_model_filepath,
                                                        focal_distance_list,
                                                        flhm_list,
                                                        placement_scene_template_filepath)
                else:
                    print("No action taken.")
                
                # Reset the flag
                process_line = False

# Start the output reading in a separate thread
output_thread = threading.Thread(target=read_output)
output_thread.start()

# Wait for the process and threads to finish
process.wait()
output_thread.join()
listener.stop()