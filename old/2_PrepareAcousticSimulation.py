#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 10:12:19 2024

@author: maximilian
"""

#=============================================================================
# Specify variables
#=============================================================================

# Path to T1 image (used for SimNIBS' charm)
t1_filepath = '/media/maximilian/e4713b47-344e-4ac6-85dd-b6769e0cbfa82/m2m_BFTUS004/T1.nii.gz'

# Path to head mesh (.msh file) generated by SimNIBS' charm
simnibs_mesh_filepath = '/media/maximilian/e4713b47-344e-4ac6-85dd-b6769e0cbfa82/m2m_BFTUS004/BFTUS004.msh'

# Name of and path to mask of target region of interest (in same space as T1
# image)
#target_roi_name = 'BF'
target_roi_filepath = '/media/maximilian/e4713b47-344e-4ac6-85dd-b6769e0cbfa82/m2m_BFTUS004/BF.nii'

# Maximum focal depth of transducer (in mm)
# Note: Set to 1000 if you want a "whole-head" map of distances
max_distance = 76.6 # note: max. focal depth of CTX-500

transducer_diameter = 65

# Maximum allowed angle for tilting of TUS transducer (in degree)
# Note: Set to 360 if you want a "whole-head" map of angles
max_angle = 10 #10

scene_template_filepath = '/media/maximilian/e4713b47-344e-4ac6-85dd-b6769e0cbfa82/PlanTUS/TUSTransducerPlacementPlanning_TEMPLATE.scene'

output_path = '/media/maximilian/e4713b47-344e-4ac6-85dd-b6769e0cbfa82/m2m_BFTUS004/PlanTUS'

vertex_list = [8286]

offset = 10.82 # mm (CTX-500/CTX-545)
additional_offset = 3 # mm

transducer_surface_model_filepath = "/media/maximilian/e4713b47-344e-4ac6-85dd-b6769e0cbfa82/TUS_vmPFC/3D_models/TRANSDUCER-NEUROFUS-CTX-500-4_DEVICE.surf.gii"

# Focal distance and corresponding FLHM values (both in mm)
focal_distance_list = [33.1, 34.7, 36.8, 39.3, 42.4, 46.2, 50.8, 56.0, 61.8, 68.5, 76.6]
flhm_list = [8.4, 9.3, 10.3, 11.6, 13.5, 16.1, 19.2, 22.8, 26.3, 30.2, 34.5]

#=============================================================================
#
#=============================================================================

cd /media/maximilian/e4713b47-344e-4ac6-85dd-b6769e0cbfa82/PlanTUS
import PlanTUS
import os
import shutil
import numpy as np
import scipy
import math
from nilearn import image
import nibabel as nib


#=============================================================================
#
#=============================================================================

target_roi_filename = os.path.split(target_roi_filepath)[1]
target_roi_name = target_roi_filename.replace(".nii", "")
target_roi_name = target_roi_name.replace(".gz", "")

output_path = output_path + "/" + target_roi_name
os.makedirs(output_path, exist_ok=True)

shutil.copy(target_roi_filepath, output_path + "/")
target_roi_filepath = output_path + "/" + target_roi_filename



#=============================================================================
#
#=============================================================================

skin_coordinates, skin_normals = PlanTUS.compute_surface_metrics(output_path + "/skin.surf.gii")

target_center = PlanTUS.roi_center_of_gravity(target_roi_filepath)
skin_target_vectors = PlanTUS.vectors_between_surface_and_point(output_path + "/skin.surf.gii", target_center)

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

for vertex_index in vertex_list:

#==============================================================================
# Get coordinates and normal vector for vetrex of interest
#==============================================================================
    
    vertex_coordinates = skin_coordinates[vertex_index]
    
    if skin_target_intersection_values[vertex_index] == 0:
        vertex_vector = -skin_target_vectors[vertex_index]
    else:
        vertex_vector = -skin_normals[vertex_index] # negative normal vector pointing "into the head"
    
    
#==============================================================================
# Create transducer position matrix in Localite coordinate system
#==============================================================================
    
    transducer_center_coordinates = vertex_coordinates - ((offset + additional_offset) * vertex_vector)
    
    position_matrix_Localite = PlanTUS.create_Localite_position_matrix(transducer_center_coordinates, vertex_vector)
    
    # Save
    scipy.io.savemat(output_path + "/position_matrix_" + target_roi_name + "_vtx" + str(vertex_index) + "_Localite.mat", {'position_matrix': position_matrix_Localite})
    np.savetxt(output_path + "/position_matrix_" + target_roi_name + "_vtx" + str(vertex_index) + "_Localite.txt", position_matrix_Localite)
    
    # Create "fake" XML structure with matrix values
    xml = PlanTUS.create_fake_XML_structure_for_Localite(position_matrix_Localite, "transducer_position_" + target_roi_name + "_vtx" + str(vertex_index), 0, 0)

    # Save/append "fake" XML structure in text file
    with open(output_path + "/position_matrix_" + target_roi_name + "_vtx" + str(vertex_index) + "_Localite_XML.txt", "a") as f:
        f.write(xml)
    
#==============================================================================
# Convert to k-Plan coordinate system
#==============================================================================
    
    position_matrix_kPlan = PlanTUS.convert_Localite_to_kPlan_position_matrix(position_matrix_Localite)
    
    # Save
    scipy.io.savemat(output_path + "/position_matrix_" + target_roi_name + "_vtx" + str(vertex_index) + "_kPlan.mat", {'position_matrix': position_matrix_kPlan})
    np.savetxt(output_path + "/position_matrix_" + target_roi_name + "_vtx" + str(vertex_index) + "_kPlan.txt", position_matrix_kPlan)


#==============================================================================
# Optional: Transform transducer file
#==============================================================================
    
    if not transducer_surface_model_filepath == "":
        transform = np.loadtxt(output_path + "/position_matrix_" + target_roi_name + "_vtx" + str(vertex_index) + "_kPlan.txt")
        transform[0:3,3] = transform[0:3,3]*1000
        
        transform_filepath = output_path + "/position_matrix_" + target_roi_name + "_vtx" + str(vertex_index) + "_transducer.txt"
        np.savetxt(transform_filepath, transform)
        
        output_filepath = output_path + "/transducer_" + target_roi_name + "_vtx" + str(vertex_index) + ".surf.gii"
        
        PlanTUS.transform_surface_model(transducer_surface_model_filepath, transform_filepath, output_filepath, "CEREBELLUM")
    
        
#==============================================================================
# Create transducer positon (.kps) file for k-Plan
#==============================================================================

    position_matrix_kPlan_filepath  = output_path + "/position_matrix_" + target_roi_name + "_vtx" + str(vertex_index) + "_kPlan.mat"
    kps_filename = target_roi_name + "_vtx" + str(vertex_index)
    
    PlanTUS.create_kps_file_for_kPlan(position_matrix_kPlan_filepath, kps_filename)


#==============================================================================
# Compute required focal distance and corresponding FLHM
#==============================================================================
    
    # Compute distance from between head surface and center of target
    # (intersection), i.e. required focal distance
    if skin_target_intersection_values[vertex_index] == 0:
        target_center = PlanTUS.roi_center_of_gravity(target_roi_filepath)
        skin_target_distances = PlanTUS.distance_between_surface_and_point(output_path + "/skin.surf.gii", target_center)
        focal_distance = skin_target_distances[vertex_index] + additional_offset
    else:
        skin_target_intersection_center = (np.asarray(skin_target_intersections[vertex_index][0]) + np.asarray(skin_target_intersections[vertex_index][1]))/2
        focal_distance = np.linalg.norm(skin_coordinates[vertex_index] - skin_target_intersection_center) + additional_offset # add additional distance
    
    if focal_distance > max_distance:
        focal_distance = max_distance
        
    # Compute expected FLHM
    FLHM = PlanTUS.compute_FLHM_for_focal_distance(focal_distance, focal_distance_list, flhm_list)   
    

#==============================================================================
# Create ellipsoid (surface)
#==============================================================================

    # create and save transform for focus
    focus_position_transform = np.loadtxt(output_path + "/position_matrix_" + target_roi_name + "_vtx" + str(vertex_index) + "_kPlan.txt")
    if skin_target_intersection_values[vertex_index] == 0:
        focus_position_transform[0:3,3] = target_center
    else:
        focus_position_transform[0:3,3] = skin_target_intersection_center
        
    focus_position_transform_filepath = output_path + "/focus_position_matrix_" + target_roi_name + "_vtx" + str(vertex_index) + ".txt"        
    np.savetxt(focus_position_transform_filepath, focus_position_transform)
    
    ellipsoid_output_filepath = output_path + "/focus_" + target_roi_name + "_vtx" + str(vertex_index) + ".surf.gii"

    PlanTUS.create_surface_ellipsoid(FLHM, 5,
                                     focus_position_transform_filepath,
                                     t1_filepath,
                                     ellipsoid_output_filepath)

    
#==============================================================================
# Create ellipsoid (volume)
#==============================================================================

    ellipsoid_volume_output_filepath = output_path + "/focus_" + target_roi_name + "_vtx" + str(vertex_index) + ".nii.gz"


    # create surface ellipsoid with smaller dimensions
    ellipsoid_small_output_filepath = output_path + "/focus_" + target_roi_name + "_vtx" + str(vertex_index) + "_small.surf.gii"

    PlanTUS.create_surface_ellipsoid(FLHM-1, 5-1,
                             focus_position_transform_filepath,
                             t1_filepath,
                             ellipsoid_small_output_filepath)

    
    # create metric from surface
    os.system("wb_command -surface-coordinates-to-metric {} {}".format(
        ellipsoid_output_filepath,
        output_path + "/focus.func.gii"))

    # map metric to volume
    os.system("wb_command -metric-to-volume-mapping {} {} {} {} -ribbon-constrained {} {}".format(
        output_path + "/focus.func.gii",
        ellipsoid_output_filepath,
        t1_filepath,
        ellipsoid_volume_output_filepath,
        ellipsoid_small_output_filepath,
        ellipsoid_output_filepath))
    
    # fill holes in volume
    os.system("wb_command -volume-fill-holes {} {}".format(
        ellipsoid_volume_output_filepath,
        ellipsoid_volume_output_filepath))

    # extract first image from volume and binarize
    ellipsoid = image.load_img(ellipsoid_volume_output_filepath)
    ellipsoid_data = ellipsoid.get_fdata()
    ellipsoid_data = ellipsoid_data[:,:,:,0]
    ellipsoid_data[ellipsoid_data > 0] = 1
    ellipsoid_nii_image = nib.Nifti1Image(ellipsoid_data, ellipsoid.affine)
    ellipsoid_nii_image.to_filename(ellipsoid_volume_output_filepath)

    # remove intermediate files
    os.remove(output_path + "/focus.func.gii")
    os.remove(ellipsoid_small_output_filepath)


#==============================================================================
# Visualize results
#==============================================================================

    os.system("wb_view {} {} {} {} {} {} &".format(
        output_path + "/skin.surf.gii",
        output_path + "/transducer_" + target_roi_name + "_vtx" + str(vertex_index) + ".surf.gii",
        ellipsoid_output_filepath,
        t1_filepath,
        target_roi_filepath,
        ellipsoid_volume_output_filepath))
    

