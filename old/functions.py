#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  4 16:40:42 2024

@author: maximilian
"""

def check_dependencies():
    
    import os
    
    # Fresurfer
    if os.system('mris_convert -version') != 0:
        print('Freesurfer not installed!')
    
    # SimNIBS/charm
    if os.system('charm --version') != 0:
        print('SimNIBS/charm not installed!')
    
    # Connectome workbench
    if os.system('wb_command -version') != 0:
        print('Connectome workbench not installed!')
    
    # FSL
    if os.system('flirt -version') != 0:
        print('FSL not installed!')



def create_scene(scene_template_filepath, output_filepath, scene_variable_names, scene_variable_values):
    
    import os
    import shutil
    import numpy as np
    
    output_path = os.path.split(output_filepath)[0]
    
    shutil.copy(scene_template_filepath, output_path + '/scene_template_0.scene')
    
    for i in np.arange(len(scene_variable_names)):
        
        os.system("cat " + output_path + "/scene_template_" + str(i) + ".scene | awk -v awkvar=" + scene_variable_values[i] + " '{ gsub(/" + scene_variable_names[i] +"/, awkvar); print }' > " + output_path + "/scene_template_" + str(i+1) + ".scene")
        
        os.remove(output_path + "/scene_template_" + str(i) + ".scene")

    shutil.copy(output_path + "/scene_template_" + str(i+1) + ".scene", output_filepath)
    os.remove(output_path + "/scene_template_" + str(i+1) + ".scene")
    


def convert_simnibs_mesh_to_surface(simnibs_mesh_filepath, tags, mesh_name, output_path):
    
    import os
    from simnibs import mesh_io
    
    simnibs_mesh = mesh_io.read_msh(simnibs_mesh_filepath)
    simnibs_mesh = simnibs_mesh.crop_mesh(tags=tags)

    mesh_io.write_freesurfer_surface(simnibs_mesh,
                                     output_path + "/" + mesh_name + "_freesurfer")

    os.system("mris_convert {} {}".format(
        output_path + "/" + mesh_name + "_freesurfer",
        output_path + "/" + mesh_name + ".stl"))
    
    os.system("mris_convert {} {}".format(
        output_path + "/" + mesh_name + "_freesurfer",
        output_path + "/" + mesh_name + ".surf.gii"))
    
    os.remove(output_path + "/" + mesh_name + "_freesurfer")



def compute_surface_metrics(surface_filepath):
    
    import os
    import numpy as np
    from nilearn import surface
    
    output_path = os.path.split(surface_filepath)[0]
    surface_filename = os.path.split(surface_filepath)[1]
    surface_name = surface_filename.replace(".surf.gii", "")
    
    os.system('wb_command -surface-coordinates-to-metric {} {}'.format(
        surface_filepath,
        output_path + "/" + surface_name + "_coordinates.func.gii"))
    
    surface_coordinates = surface.load_surf_data(output_path + "/" + surface_name + "_coordinates.func.gii")
    surface_coordinates = np.asfarray(surface_coordinates)
    

    os.system('wb_command -surface-normals {} {}'.format(
        surface_filepath,
        output_path + "/" + surface_name + "_normals.func.gii"))
        
    surface_normals = surface.load_surf_data(output_path + "/" + surface_name + "_normals.func.gii")
    surface_normals = np.asfarray(surface_normals)

    os.remove(output_path + "/" + surface_name + "_coordinates.func.gii")
    os.remove(output_path + "/" + surface_name + "_normals.func.gii")

    return surface_coordinates, surface_normals



def create_pseudo_metric_nifti_from_surface(surface_filepath):
    
    import os
    from nilearn import image
    
    output_path = os.path.split(surface_filepath)[0]
    surface_filename = os.path.split(surface_filepath)[1]
    surface_name = surface_filename.replace(".surf.gii", "")
    
    os.system('wb_command -surface-coordinates-to-metric {} {}'.format(
        surface_filepath,
        output_path + "/" + surface_name + "_coordinates.func.gii"))
    
    os.system("wb_command -metric-reduce {} MEAN {}".format(
        output_path + "/" + surface_name + "_coordinates.func.gii",
        output_path + "/" + surface_name + "_coordinates_MEAN.func.gii"))
    
    os.system("wb_command -metric-convert -to-nifti {} {}".format(
        output_path + "/" + surface_name + "_coordinates_MEAN.func.gii",
        output_path + "/" + surface_name + "_coordinates_MEAN.nii.gz"))
    
    nii = image.load_img(output_path + "/" + surface_name + "_coordinates_MEAN.nii.gz")
    nii_data = nii.get_fdata()
    
    os.remove(output_path + "/" + surface_name + "_coordinates.func.gii")
    os.remove(output_path + "/" + surface_name + "_coordinates_MEAN.func.gii")
    os.remove(output_path + "/" + surface_name + "_coordinates_MEAN.nii.gz")
    
    return nii, nii_data



def load_stl(stl_filepath):
    
    import vtk
    
    readerSTL = vtk.vtkSTLReader()
    readerSTL.SetFileName(stl_filepath)
    readerSTL.Update()
    
    polydata = readerSTL.GetOutput()
    
    if polydata.GetNumberOfPoints() == 0:
        raise ValueError("No point data could be loaded from '{}'".format(stl_filepath))
        
    return polydata



def create_metric_from_pseudo_nifti(metric_name, metric_values, surface_filepath):
    
    import os
    import numpy as np
    import nibabel as nib
    
    output_path = os.path.split(surface_filepath)[0]
    surface_filename = os.path.split(surface_filepath)[1]
    surface_name = surface_filename.replace(".surf.gii", "")
    
    nii, nii_data = create_pseudo_metric_nifti_from_surface(surface_filepath)
    
    nii_data_tmp = nii_data.copy()
    
    n = int(np.ceil(len(metric_values) / len(nii_data_tmp)))
    
    for i in range(n):
        for j in range(len(nii_data_tmp)):
            try:
                nii_data_tmp[j][i] = [metric_values[(i * len(nii_data_tmp) + j)]]
            except:
                pass
            
    nii_new = nib.nifti1.Nifti1Image(nii_data_tmp, nii.affine, header=nii.header)
    
    nii_new.to_filename(output_path + "/" + metric_name + "_" + surface_name + ".nii.gz")
    
    os.system("wb_command -metric-convert -from-nifti {} {} {}".format(
        output_path + "/" + metric_name + "_" + surface_name + ".nii.gz",
        surface_filepath,
        output_path + "/" + metric_name + "_" + surface_name + ".func.gii"))
    
    os.remove(output_path + "/" + metric_name + "_" + surface_name + ".nii.gz")
    


def erode_metric(metric_filepath, surface_filepath, erosion_factor):
    
    os.system("wb_command -metric-erode {} {} {} {}".format(
        metric_filepath,
        surface_filepath,
        str(erosion_factor), 
        metric_filepath))
        

def compute_vector_mesh_intersections(points, vectors, mesh_filepath, vector_length):
    
    import numpy as np
    import vtk
    
    # load mesh
    mesh = load_stl(mesh_filepath)

    ### find intersection between lines/rays going perpendicular to the head
    ### surface into the brain and the avoidance regions

    obbTree = vtk.vtkOBBTree()
    obbTree.SetDataSet(mesh)
    obbTree.BuildLocator()

    intersections = []
    
    
    for i in np.arange(len(points)):
        # starting point of line on skin
        pSource = points[i]
        # end point of line (4cm into the brain)
        pTarget = points[i] - vector_length * vectors[i] # note: minus sign as normal points vector away from the head/brain
        
        pointsVTKintersection = vtk.vtkPoints()
        code = obbTree.IntersectWithLine(pSource, pTarget, pointsVTKintersection, None)

        pointsVTKIntersectionData = pointsVTKintersection.GetData()
        noPointsVTKIntersection = pointsVTKIntersectionData.GetNumberOfTuples()

        pointsIntersection = []
        for idx in range(noPointsVTKIntersection):
            _tup = pointsVTKIntersectionData.GetTuple3(idx)
            pointsIntersection.append(_tup)
            
        intersections.append(pointsIntersection)
        
    return intersections
    


def create_avoidance_mask(simnibs_mesh_filepath, surface_filepath, erosion_factor):
    
    import os
    import glob
    import pandas as pd
    import numpy as np
    from scipy.cluster.vq import kmeans

    output_path = os.path.split(surface_filepath)[0]
    surface_filename = os.path.split(surface_filepath)[1]
    surface_name = surface_filename.replace(".surf.gii", "")
    
    simnibs_mesh_path = os.path.split(simnibs_mesh_filepath)[0]
    
    avoidance_mask = []
    
    # create binary mask of final tissues file
    os.system("fslmaths {} -bin {}".format(
        simnibs_mesh_path + "/final_tissues.nii.gz",
        output_path + "/final_tissues_bin.nii.gz"))
        
    # fill holes (i.e. air cavities/sinuses) in binarzed mask
    os.system("wb_command -volume-fill-holes {} {}".format(
        output_path + "/final_tissues_bin.nii.gz",
        output_path + "/final_tissues_bin_filled.nii.gz"))
    
    # create mask of holes (air cavities/sinuses) by subtracting the previously generated masks
    os.system("fslmaths {} -sub {} {}".format(
        output_path + "/final_tissues_bin_filled.nii.gz",
        output_path + "/final_tissues_bin.nii.gz",
        output_path + "/final_tissues_air.nii.gz"))

    # create 3D stl/surface/mesh file from mask
    os.system("mri_tessellate -n {} 1 {}".format(
        output_path + "/final_tissues_air.nii.gz",
        output_path + "/final_tissues_air"))
    os.system("mris_convert {} {}".format(
        output_path + "/final_tissues_air ",
        output_path + "/final_tissues_air.surf.gii"))
    os.system("wb_command -surface-smoothing {} 0.5 10 {}".format(
        output_path + "/final_tissues_air.surf.gii",
        output_path + "/final_tissues_air.surf.gii"))
    os.system("mris_convert {} {}".format(
        output_path + "/final_tissues_air.surf.gii",
        output_path + "/final_tissues_air.stl"))
    
    surface_coordinates, surface_normals = compute_surface_metrics(surface_filepath)
    
    surface_air_intersections = compute_vector_mesh_intersections(surface_coordinates, surface_normals, output_path + "/final_tissues_air.stl", 40)
    
    for i in np.arange(len(surface_air_intersections)):
        if len(surface_air_intersections[i]) > 0:
            avoidance_mask.append(0)
        else:
            avoidance_mask.append (1)
    avoidance_mask = np.asarray(avoidance_mask)
    
    # extract eyes from final tissue segmentation
    convert_simnibs_mesh_to_surface(simnibs_mesh_filepath, [1006], "eyes", output_path)
    
    # get coordinates
    eyes_coordinates, _ = compute_surface_metrics(output_path + "/eyes.surf.gii")
    eyes_coordinates = np.array(eyes_coordinates, dtype=float)

    # remove intermediate files
    for f in glob.glob(output_path + "/final_tissues*"):
        os.remove(f)
    for f in glob.glob(output_path + "/eyes*"):
        os.remove(f)

    # separate left and right eye
    x_coordinate_centers, _ = kmeans(eyes_coordinates[:,0], 2)
    left_eye_coordinates = eyes_coordinates[eyes_coordinates[:,0] < (np.sum(x_coordinate_centers)/2)]
    right_eye_coordinates = eyes_coordinates[eyes_coordinates[:,0] > (np.sum(x_coordinate_centers)/2)]

    # center coordinates of eyes
    left_eye_center = np.mean(left_eye_coordinates, axis=0)
    right_eye_center = np.mean(right_eye_coordinates, axis=0)

    # avoid vertices within 3 cm radius around the eye centers
    avoidance_mask[np.linalg.norm((surface_coordinates-left_eye_center), axis=1) < 30] = 0
    avoidance_mask[np.linalg.norm((surface_coordinates-right_eye_center), axis=1) < 30] = 0


    # get left and right tragus coordinates (-1cm) from EEG fiducials
    eeg_fiducials = pd.read_csv(simnibs_mesh_path + "/eeg_positions/Fiducials.csv", header=1)
    LPA_coordinates = np.array(eeg_fiducials.iloc[0,1:4], dtype=np.float)
    RPA_coordinates = np.array(eeg_fiducials.iloc[1,1:4], dtype=np.float)

    LPA_coordinates[1] = LPA_coordinates[1] - 15 # move coordinate 1.5 cm posterior (to center of ear)
    RPA_coordinates[1] = RPA_coordinates[1] - 15 # move coordinate 1.5 cm posterior (to center of ear)

    # avoid vertices within 1.5 cm radius around the eye centers
    avoidance_mask[np.linalg.norm((surface_coordinates-LPA_coordinates), axis=1) < 15] = 0
    avoidance_mask[np.linalg.norm((surface_coordinates-RPA_coordinates), axis=1) < 15] = 0
    
    # 
    create_metric_from_pseudo_nifti("avoidance", avoidance_mask, surface_filepath)

    erode_metric(output_path + "/avoidance_" + surface_name + ".func.gii", surface_filepath, erosion_factor)



def unit_vector(vector):
    """ Returns the unit vector of the vector. """
    import numpy as np
    return vector / np.linalg.norm(vector)



def angle_between_vectors(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2' """
    import numpy as np
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))



def roi_center_of_gravity(roi_filepath):
    
    from nilearn import image, plotting
    
    roi = image.load_img(roi_filepath)

    # get center of mass/gravity
    roi_center = plotting.find_probabilistic_atlas_cut_coords([roi])[0]
    
    return roi_center



def vectors_between_surface_and_point(surface_filepath, point_coordinates):
    
    import numpy as np
    
    surface_point_vectors = []
    
    surface_coordinates, _ = compute_surface_metrics(surface_filepath)
    
    for i in np.arange(len(surface_coordinates)):
        surface_point_vectors.append(surface_coordinates[i] - point_coordinates)
        
    surface_point_vectors = np.asarray(surface_point_vectors)
    
    return surface_point_vectors



def distance_between_surface_and_point(surface_filepath, point_coordinates):
    
    import numpy as np
    
    surface_point_vectors = vectors_between_surface_and_point(surface_filepath, point_coordinates)
    
    surface_point_distances = []
    
    for i in np.arange(len(surface_point_vectors)):
        surface_point_distances.append(np.linalg.norm(surface_point_vectors[i]))
        
    surface_point_distances = np.abs(np.asarray(surface_point_distances))

    return surface_point_distances
    
    

def stl_from_nii(nii_filepath, threshold):
    
    nii_path = os.path.split(nii_filepath)[0]
    nii_filename = os.path.split(nii_filepath)[1]
    try:
        nii_name = nii_filename.replace(".nii.gz", "")
    except:
        nii_name = nii_filename.replace(".nii", "")
        
    nii_name = nii_name + "_3Dmodel"
    
    ### create 3D model/surface/mesh (stl file) from target ROI mask
    os.system("fslmaths {} -thr {} -bin {}".format(
        nii_filepath,
        str(threshold),
        nii_path + "/" + nii_name))
    
    os.system("mri_tessellate -n {}.nii.gz 1 {}".format(
        nii_path + "/" + nii_name,
        nii_path + "/" + nii_name))
    
    os.system("mris_convert {} {}.surf.gii".format(
        nii_path + "/" + nii_name,
        nii_path + "/" + nii_name))
    
    os.system("wb_command -surface-smoothing {}.surf.gii 0.5 10 {}.surf.gii".format(
        nii_path + "/" + nii_name,
        nii_path + "/" + nii_name))
    
    os.system("mris_convert {}.surf.gii {}.stl".format(
        nii_path + "/" + nii_name,
        nii_path + "/" + nii_name))
    
    os.remove(nii_path + "/" + nii_name)
    os.remove(nii_path + "/" + nii_name + ".nii.gz")
    os.remove(nii_path + "/" + nii_name + ".surf.gii")


def smooth_metric(metric_filepath, surface_filepath, FWHM):
    
    import os
    
    metric_path = os.path.split(metric_filepath)[0]
    metric_filename = os.path.split(metric_filepath)[1]
    metric_name = metric_filename.replace(".func.gii", "")
    
    os.system("wb_command -metric-smoothing {} {} {} {} -fwhm".format(
        surface_filepath,
        metric_filepath,
        str(FWHM),
        metric_path + "/" + metric_name + "_s" + str(FWHM) + ".func.gii"))


    
def mask_metric(metric_filepath, mask_filepath):
    
    os.system("wb_command -metric-mask {} {} {}".format(
        metric_filepath,
        mask_filepath,
        metric_filepath))



def threshold_metric(metric_filepath, threshold):
    
    import os
    
    metric_path = os.path.split(metric_filepath)[0]
    metric_filename = os.path.split(metric_filepath)[1]
    metric_name = metric_filename.replace(".func.gii", "")
    
    os.system("wb_command -metric-math 'x < {}' {} -var x {}".format(
        str(threshold),
        metric_path + "/" + metric_name + "_thresholded.func.gii",
        metric_filepath))
    
    
    
def add_structure_information(filepath, structure_label):
    
    import os
    
    os.system("wb_command -set-structure {} {} -surface-type RECONSTRUCTION".format(
        filepath,
        structure_label))
    
    
    
