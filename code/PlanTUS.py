#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Maximilian Lueckel
"""

_FREESURFER_PATH = "/Applications/freesurfer/7.4.1/bin"
_CONNECTOME_PATH = "/Applications/wb_view.app/Contents/usr/bin"
_FSL_PATH = "/Users/spichardo/fsl/share/fsl/bin"

def set_paths(freesurfer_path=None, connectome_path=None, fsl_path=None):
    global _FREESURFER_PATH, _CONNECTOME_PATH, _FSL_PATH
    if freesurfer_path is not None:
        _FREESURFER_PATH = freesurfer_path
    if connectome_path is not None:
        _CONNECTOME_PATH = connectome_path
    if fsl_path is not None:
        _FSL_PATH = fsl_path

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
        
        os.system("cat " + output_path + os.sep + "scene_template_" + str(i) + ".scene | awk -v awkvar=" + scene_variable_values[i] + " '{ gsub(/" + scene_variable_names[i] +"/, awkvar); print }' > " + output_path + os.sep +"scene_template_" + str(i+1) + ".scene")
        
        os.remove(output_path + os.sep + "scene_template_" + str(i) + ".scene")

    shutil.copy(output_path + os.sep + "scene_template_" + str(i+1) + ".scene", output_filepath)
    os.remove(output_path + os.sep + "scene_template_" + str(i+1) + ".scene")
    


def convert_simnibs_mesh_to_surface(simnibs_mesh_filepath, tags, mesh_name, output_path):
    
    import os
    from simnibs import mesh_io
    
    simnibs_mesh = mesh_io.read_msh(simnibs_mesh_filepath)
    simnibs_mesh = simnibs_mesh.crop_mesh(tags=tags)

    mesh_io.write_freesurfer_surface(simnibs_mesh,
                                     output_path + os.sep + mesh_name + "_freesurfer")

    os.system(_FREESURFER_PATH + os.sep + "mris_convert '{}' '{}'".format(
        output_path + os.sep + mesh_name + "_freesurfer",
        output_path + os.sep + mesh_name + ".stl"))
    
    os.system(_FREESURFER_PATH + os.sep + "mris_convert '{}' '{}'".format(
        output_path + os.sep + mesh_name + "_freesurfer",
        output_path + os.sep + mesh_name + ".surf.gii"))
    
    os.remove(output_path + os.sep + mesh_name + "_freesurfer")



def compute_surface_metrics(surface_filepath):
    
    import os
    import numpy as np
    from nilearn import surface
    
    output_path = os.path.split(surface_filepath)[0]
    surface_filename = os.path.split(surface_filepath)[1]
    surface_name = surface_filename.replace(".surf.gii", "")
    
    os.system(_CONNECTOME_PATH + os.sep + "wb_command -logging OFF -surface-coordinates-to-metric '{}' '{}'".format(
        surface_filepath,
        output_path + os.sep + surface_name + "_coordinates.func.gii"))
    
    surface_coordinates = surface.load_surf_data(output_path + os.sep + surface_name + "_coordinates.func.gii")
    surface_coordinates = np.asfarray(surface_coordinates)
    

    os.system(_CONNECTOME_PATH + os.sep + "wb_command -logging OFF -surface-normals '{}' '{}'".format(
        surface_filepath,
        output_path + os.sep + surface_name + "_normals.func.gii"))
        
    surface_normals = surface.load_surf_data(output_path + os.sep + surface_name + "_normals.func.gii")
    surface_normals = np.asfarray(surface_normals)

    os.remove(output_path + os.sep + surface_name + "_coordinates.func.gii")
    os.remove(output_path + os.sep + surface_name + "_normals.func.gii")

    return surface_coordinates, surface_normals



def create_pseudo_metric_nifti_from_surface(surface_filepath):
    
    import os
    from nilearn import image
    
    output_path = os.path.split(surface_filepath)[0]
    surface_filename = os.path.split(surface_filepath)[1]
    surface_name = surface_filename.replace(".surf.gii", "")
    
    os.system(_CONNECTOME_PATH + os.sep + "wb_command -logging OFF -surface-coordinates-to-metric '{}' '{}'".format(
        surface_filepath,
        output_path + os.sep + surface_name + "_coordinates.func.gii"))
    
    os.system(_CONNECTOME_PATH + os.sep + "wb_command -logging OFF -metric-reduce '{}' MEAN '{}'".format(
        output_path + os.sep + surface_name + "_coordinates.func.gii",
        output_path + os.sep + surface_name + "_coordinates_MEAN.func.gii"))
    
    os.system(_CONNECTOME_PATH + os.sep + "wb_command -logging OFF -metric-convert -to-nifti '{}' '{}'".format(
        output_path + os.sep + surface_name + "_coordinates_MEAN.func.gii",
        output_path + os.sep + surface_name + "_coordinates_MEAN.nii.gz"))
    
    nii = image.load_img(output_path + os.sep + surface_name + "_coordinates_MEAN.nii.gz")
    nii_data = nii.get_fdata()
    
    os.remove(output_path + os.sep + surface_name + "_coordinates.func.gii")
    os.remove(output_path + os.sep + surface_name + "_coordinates_MEAN.func.gii")
    os.remove(output_path + os.sep + surface_name + "_coordinates_MEAN.nii.gz")
    
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
    
    nii_new.to_filename(output_path + os.sep + metric_name + "_" + surface_name + ".nii.gz")
    
    os.system(_CONNECTOME_PATH + os.sep + "wb_command -logging OFF -metric-convert -from-nifti '{}' '{}' '{}'".format(
        output_path + os.sep + metric_name + "_" + surface_name + ".nii.gz",
        surface_filepath,
        output_path + os.sep + metric_name + "_" + surface_name + ".func.gii"))
    
    os.remove(output_path + os.sep + metric_name + "_" + surface_name + ".nii.gz")
    


def erode_metric(metric_filepath, surface_filepath, erosion_factor):
    
    import os
    
    os.system(_CONNECTOME_PATH + os.sep + "wb_command -logging OFF -metric-erode '{}' '{}' '{}' '{}'".format(
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
    os.system(_FSL_PATH + os.sep + "fslmaths '{}' -bin '{}'".format(
        simnibs_mesh_path + os.sep +"final_tissues.nii.gz",
        output_path + os.sep +"final_tissues_bin.nii.gz"))
        
    # fill holes (i.e. air cavities/sinuses) in binarzed mask
    os.system(_CONNECTOME_PATH + os.sep + "wb_command -logging OFF -volume-fill-holes '{}' '{}'".format(
        output_path + os.sep +"final_tissues_bin.nii.gz",
        output_path + os.sep +"final_tissues_bin_filled.nii.gz"))
    
    # create mask of holes (air cavities/sinuses) by subtracting the previously generated masks
    os.system(_FSL_PATH + os.sep + "fslmaths '{}' -sub '{}' '{}'".format(
        output_path + os.sep +"final_tissues_bin_filled.nii.gz",
        output_path + os.sep +"final_tissues_bin.nii.gz",
        output_path + os.sep +"final_tissues_air.nii.gz"))

    # create 3D stl/surface/mesh file from mask
    os.system(_FREESURFER_PATH + os.sep + "mri_tessellate -n '{}' 1 '{}'".format(
        output_path + os.sep +"final_tissues_air.nii.gz",
        output_path + os.sep +"final_tissues_air"))
    os.system(_FREESURFER_PATH + os.sep + "mris_convert '{}' '{}'".format(
        output_path + os.sep +"final_tissues_air",
        output_path + os.sep +"final_tissues_air.surf.gii"))
    os.system(_CONNECTOME_PATH + os.sep + "wb_command -logging OFF -surface-smoothing '{}' 0.5 10 '{}'".format(
        output_path + os.sep +"final_tissues_air.surf.gii",
        output_path + os.sep +"final_tissues_air.surf.gii"))
    os.system(_FREESURFER_PATH + os.sep + "mris_convert '{}' '{}'".format(
        output_path + os.sep +"final_tissues_air.surf.gii",
        output_path + os.sep +"final_tissues_air.stl"))
    
    surface_coordinates, surface_normals = compute_surface_metrics(surface_filepath)
    
    surface_air_intersections = compute_vector_mesh_intersections(surface_coordinates, surface_normals, output_path + os.sep +"final_tissues_air.stl", 40)
    
    for i in np.arange(len(surface_air_intersections)):
        if len(surface_air_intersections[i]) > 0:
            avoidance_mask.append(0)
        else:
            avoidance_mask.append (1)
    avoidance_mask = np.asarray(avoidance_mask)
    
    # extract eyes from final tissue segmentation
    convert_simnibs_mesh_to_surface(simnibs_mesh_filepath, [1006], "eyes", output_path)
    
    # get coordinates
    eyes_coordinates, _ = compute_surface_metrics(output_path + os.sep +"eyes.surf.gii")
    eyes_coordinates = np.array(eyes_coordinates, dtype=float)

    # remove intermediate files
    for f in glob.glob(output_path + os.sep +"final_tissues*"):
        os.remove(f)
    for f in glob.glob(output_path + os.sep +"eyes*"):
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
    eeg_fiducials = pd.read_csv(simnibs_mesh_path + os.sep +"eeg_positions" +os.sep +"Fiducials.csv", header=1)
    LPA_coordinates = np.array(eeg_fiducials.iloc[0,1:4], dtype=np.float64)
    RPA_coordinates = np.array(eeg_fiducials.iloc[1,1:4], dtype=np.float64)

    LPA_coordinates[1] = LPA_coordinates[1] - 15 # move coordinate 1.5 cm posterior (to center of ear)
    RPA_coordinates[1] = RPA_coordinates[1] - 15 # move coordinate 1.5 cm posterior (to center of ear)

    # avoid vertices within 1.5 cm radius around the eye centers
    avoidance_mask[np.linalg.norm((surface_coordinates-LPA_coordinates), axis=1) < 15] = 0
    avoidance_mask[np.linalg.norm((surface_coordinates-RPA_coordinates), axis=1) < 15] = 0
    
    # 
    create_metric_from_pseudo_nifti("avoidance", avoidance_mask, surface_filepath)

    erode_metric(output_path + os.sep +"avoidance_" + surface_name + ".func.gii", surface_filepath, erosion_factor)



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
    
    import os
    
    nii_path = os.path.split(nii_filepath)[0]
    nii_filename = os.path.split(nii_filepath)[1]
    nii_name = nii_filename.replace(".nii", "")
    nii_name = nii_name.replace(".gz", "")
        
    nii_name = nii_name + "_3Dmodel"
    
    ### create 3D model/surface/mesh (stl file) from target ROI mask
    os.system(_FSL_PATH + os.sep + "fslmaths '{}' -thr '{}' -bin '{}'".format(
        nii_filepath,
        str(threshold),
        nii_path + os.sep + nii_name))
    
    os.system(_FREESURFER_PATH + os.sep + "mri_tessellate -n '{}'.nii.gz 1 '{}'".format(
        nii_path + os.sep + nii_name,
        nii_path + os.sep + nii_name))
    
    os.system(_FREESURFER_PATH + os.sep + "mris_convert '{}' '{}'.surf.gii".format(
        nii_path + os.sep + nii_name,
        nii_path + os.sep + nii_name))
    
    os.system(_CONNECTOME_PATH + os.sep + "wb_command -logging OFF -surface-smoothing '{}'.surf.gii 0.5 10 '{}'.surf.gii".format(
        nii_path + os.sep + nii_name,
        nii_path + os.sep + nii_name))
    
    os.system(_FREESURFER_PATH + os.sep + "mris_convert '{}'.surf.gii '{}'.stl".format(
        nii_path + os.sep + nii_name,
        nii_path + os.sep + nii_name))
    
    os.remove(nii_path + os.sep + nii_name)
    os.remove(nii_path + os.sep + nii_name + ".nii.gz")
    os.remove(nii_path + os.sep + nii_name + ".surf.gii")


def smooth_metric(metric_filepath, surface_filepath, FWHM):
    
    import os
    
    metric_path = os.path.split(metric_filepath)[0]
    metric_filename = os.path.split(metric_filepath)[1]
    metric_name = metric_filename.replace(".func.gii", "")
    
    os.system(_CONNECTOME_PATH + os.sep + "wb_command -logging OFF -metric-smoothing '{}' '{}' '{}' '{}' -fwhm".format(
        surface_filepath,
        metric_filepath,
        str(FWHM),
        metric_path + os.sep + metric_name + "_s" + str(FWHM) + ".func.gii"))


    
def mask_metric(metric_filepath, mask_filepath):
    
    import os
    
    os.system(_CONNECTOME_PATH + os.sep + "wb_command -logging OFF -metric-mask '{}' '{}' '{}'".format(
        metric_filepath,
        mask_filepath,
        metric_filepath))



def threshold_metric(metric_filepath, threshold):
    
    import os
    
    metric_path = os.path.split(metric_filepath)[0]
    metric_filename = os.path.split(metric_filepath)[1]
    metric_name = metric_filename.replace(".func.gii", "")
    
    os.system(_CONNECTOME_PATH + os.sep + "wb_command -logging OFF -metric-math 'x < {}' '{}' -var x '{}'".format(
        str(threshold),
        metric_path + os.sep + metric_name + "_thresholded.func.gii",
        metric_filepath))
    
    
    
def add_structure_information(filepath, structure_label):
    
    import os
    
    os.system(_CONNECTOME_PATH + os.sep + "wb_command -logging OFF -set-structure '{}' {} -surface-type RECONSTRUCTION".format(
        filepath,
        structure_label))
    
    
    
def create_Localite_position_matrix(center_coordinates, x_vector):
    
    import numpy as np    
    
    position_matrix_Localite = np.zeros([4,4]) # dummy matrix
    position_matrix_Localite[3,3] = 1
    
    position_matrix_Localite[0:3,3] = np.transpose(center_coordinates)
    
    x_vector /= np.linalg.norm(x_vector) # compute unit vector
    position_matrix_Localite[0:3,0] = x_vector 
    
    # compute vector orhtogonal to x vector
    y_vector = np.random.randn(3) # random vector
    y_vector -= y_vector.dot(x_vector) * x_vector # make it orthogonal to x vector
    y_vector /= np.linalg.norm(y_vector) # normalize
    position_matrix_Localite[0:3,1] = y_vector 
    
    # compute vector orhtogonal to x and y vector
    z_vector = np.cross(x_vector, y_vector)
    z_vector /= np.linalg.norm(z_vector) # normalize
    position_matrix_Localite[0:3,2] = z_vector
    
    return position_matrix_Localite



def create_fake_XML_structure_for_Localite(Localite_position_matrix, position_name, element_index, uid):
    
    xml = '    <Element index="' + str(element_index) + '" selected="true" type="InstrumentMarker">\n\
            <InstrumentMarker additionalInformation="" alwaysVisible="false"\n\
                color="#00ff00" description="' + position_name + '" locked="false" set="true" uid="' + str(uid) + '">\n\
                <Matrix4D data00="' + str(round(Localite_position_matrix[0,0],6)) + '" data01="' + str(round(Localite_position_matrix[0,1],6)) + '" data02="' + str(round(Localite_position_matrix[0,2],6)) + '"\n\
                    data03="' + str(round(Localite_position_matrix[0,3],6)) + '" data10="' + str(round(Localite_position_matrix[1,0],6)) + '" data11="' + str(round(Localite_position_matrix[1,1],6)) + '"\n\
                    data12="' + str(round(Localite_position_matrix[1,2],6)) + '" data13="' + str(round(Localite_position_matrix[1,3],6)) + '" data20="' + str(round(Localite_position_matrix[2,0],6)) + '"\n\
                    data21="' + str(round(Localite_position_matrix[2,1],6)) + '" data22="' + str(round(Localite_position_matrix[2,2],6)) + '" data23="' + str(round(Localite_position_matrix[2,3],6)) + '" data30="0.0"\n\
                    data31="0.0" data32="0.0" data33="1.0"/>\n\
            </InstrumentMarker>\n\
        </Element>\n\
    '

    return xml
    

def convert_Localite_to_kPlan_position_matrix(Localite_position_matrix):
    
    kPlan_position_matrix = Localite_position_matrix.copy()
    
    kPlan_position_matrix[0:3,0] = -Localite_position_matrix[0:3,1] # x(k-Plan) = -y(Localite)
    kPlan_position_matrix[0:3,1] = Localite_position_matrix[0:3,2] # y(k-Plan) = z(k-Plan)
    kPlan_position_matrix[0:3,2] = -Localite_position_matrix[0:3,0]  # z(k-Plan) = -x(Localite)
    
    # Convert center coordinates from mm (Localite) to m (k-Plan)
    kPlan_position_matrix[0:3,3] = kPlan_position_matrix[0:3,3]/1000
    
    
    return kPlan_position_matrix



def transform_surface_model(surface_model_filepath, transform_filepath, output_filepath, structure):
    
    import os
    
    # transform transducer file
    os.system(_CONNECTOME_PATH + os.sep + "wb_command -logging OFF -surface-apply-affine '{}' '{}' '{}'".format(
        surface_model_filepath,
        transform_filepath,
        output_filepath))

    os.system(_CONNECTOME_PATH + os.sep + "wb_command -logging OFF -set-structure '{}' {} -surface-type RECONSTRUCTION".format(
        output_filepath,
        structure))
    
    
    
def create_kps_file_for_kPlan(position_matrix_filepath, kps_filename):
    
    import os
    import numpy as np
    import scipy
    import h5py

    # Create output filepath
    output_path, _ = os.path.split(position_matrix_filepath)
    output_filepath = output_path + os.sep + kps_filename + ".kps"
    
    # Load position matrix
    mat_contents = scipy.io.loadmat(position_matrix_filepath)
    position_matrix = mat_contents["position_matrix"]
    position_matrix = np.transpose(position_matrix)
    position_label = "Localite transducer position"
    
    # Reshape position matrix to (1, 4, 4)
    position_matrix = position_matrix.reshape((1, 4, 4))
    
    # Save transform matrices to HDF5 file
    with h5py.File(output_filepath, "w") as f:
        dset = f.create_dataset("/1/position_transform", (1, 4, 4), dtype="float32")
        dset[:] = position_matrix.astype("float32")
        
        # Write attributes with ASCII encoding and specifying string lengths
        f["/1"].attrs.create("transform_label", np.string_(position_label))
        f.attrs.create("application_name", np.string_("k-Plan"))
        f.attrs.create("file_type", np.string_("k-Plan Transducer Position"))
        f.attrs.create("number_transforms", np.array([1], dtype=np.uint64))
        
        
        
def compute_FLHM_for_focal_distance(focal_distance, focal_distance_list, flhm_list):
    
    import numpy as np
    import pandas as pd
    from scipy.optimize import curve_fit

    data={'Focal Distance': focal_distance_list, 'FLHM': flhm_list}
    
    df = pd.DataFrame(data)
    
    # Fit function to measured values
    def cubic(x, a, b, c, d):
        return a + b*x + c*(x**2) + d*(x**3)
    
    pars, cov = curve_fit(f=cubic, xdata=df["Focal Distance"], ydata=df["FLHM"], p0=[0, 0, 0, 0], bounds=[-np.inf, np.inf])
    
    def calculate_FLHM(focus_distance):
        FLHM = pars[0] + pars[1] * focus_distance + pars[2] * focus_distance**2 + pars[3] * focus_distance**3
        return FLHM
    
    # Compute expected FLHM
    FLHM = calculate_FLHM(focal_distance) 
    
    return FLHM



def create_surface_ellipsoid(length, width, position_transform_filepath, reference_volume_filepath, output_filepath):

    import os
    import numpy as np
    import nibabel as nib
    from nibabel.gifti import GiftiImage, GiftiDataArray
    
    output_path, _ = os.path.split(output_filepath)
    
    # load reference volume
    reference_volume = nib.load(reference_volume_filepath)
    reference_volume_data = reference_volume.get_fdata()
    
    # get voxel dimensions of T1 image
    pixdim = reference_volume.header["pixdim"][1:4]
    
    # Image dimensions and grid spacing in mm
    reference_volume_shape = reference_volume_data.shape
    grid_spacing = pixdim[0]
    
    # Calculate scaling factors to fit the desired dimensions
    # We assume the major axis is along the z-axis
    scale_z = length / (reference_volume_shape[2] * grid_spacing)
    scale_xy = width / (reference_volume_shape[0] * grid_spacing)
    
    # Calculate radii based on the scaling factors
    a = reference_volume_shape[0] * grid_spacing * scale_xy / 2
    b = reference_volume_shape[1] * grid_spacing * scale_xy / 2
    c = reference_volume_shape[2] * grid_spacing * scale_z / 2
    
    # Create a grid of points
    phi = np.linspace(0, 2 * np.pi, 50)
    theta = np.linspace(0, np.pi, 50)
    phi, theta = np.meshgrid(phi, theta)
    
    # Parametric equations for the ellipsoid
    x = a * np.sin(theta) * np.cos(phi)
    y = b * np.sin(theta) * np.sin(phi)
    z = c * np.cos(theta)
    
    # Flatten the arrays and stack them as a (N, 3) array of points
    points = np.vstack((x.flatten(), y.flatten(), z.flatten())).T.astype(np.float32)
    
    # Create faces using the points
    faces = []
    num_rows, num_cols = x.shape
    
    for i in range(num_rows - 1):
        for j in range(num_cols - 1):
            faces.append([i * num_cols + j,
                          i * num_cols + (j + 1),
                          (i + 1) * num_cols + j])
            faces.append([(i + 1) * num_cols + j,
                          i * num_cols + (j + 1),
                          (i + 1) * num_cols + (j + 1)])
    
    faces = np.array(faces, dtype=np.int32)
    
    # Create GIFTI data arrays
    coord_array = GiftiDataArray(data=points, intent="NIFTI_INTENT_POINTSET")
    face_array = GiftiDataArray(data=faces, intent="NIFTI_INTENT_TRIANGLE")
    
    # Create a GIFTI image
    gii = GiftiImage(darrays=[coord_array, face_array])
    
    # Save the GIFTI image
    nib.save(gii, output_path + os.sep +"ellipsoid.gii")
    
    # tranform .gii to .surf.gii file
    os.system(_FREESURFER_PATH + os.sep + "mris_convert '" +
              output_path + os.sep +"ellipsoid.gii" + "' '" + 
              output_filepath +"'")
    
    os.remove(output_path + os.sep +"ellipsoid.gii")
    
    # apply transform to focus (.surf.gii file) and set structure
    transform_surface_model(output_filepath, position_transform_filepath, output_filepath, "CORTEX_RIGHT")


def create_volume_ellipsoid(length, width, position_transform_filepath, reference_volume_filepath, output_filepath):

    import os
    import numpy as np
    import nibabel as nib
    from nibabel.gifti import GiftiImage, GiftiDataArray
    
    output_path, _ = os.path.split(output_filepath)
    
    # load reference volume
    reference_volume = nib.load(reference_volume_filepath)
    reference_volume_data = reference_volume.get_fdata()
    
    # get voxel dimensions of T1 image
    pixdim = reference_volume.header["pixdim"][1:4]
    
    # Image dimensions and grid spacing in mm
    reference_volume_shape = reference_volume_data.shape
    grid_spacing = pixdim[0]
    
    # Calculate scaling factors to fit the desired dimensions
    # We assume the major axis is along the z-axis
    scale_z = length / (reference_volume_shape[2] * grid_spacing)
    scale_xy = width / (reference_volume_shape[0] * grid_spacing)
    
    # Calculate radii based on the scaling factors
    a = reference_volume_shape[0] * grid_spacing * scale_xy / 2
    b = reference_volume_shape[1] * grid_spacing * scale_xy / 2
    c = reference_volume_shape[2] * grid_spacing * scale_z / 2
    
    # Create a grid of points
    phi = np.linspace(0, 2 * np.pi, 50)
    theta = np.linspace(0, np.pi, 50)
    phi, theta = np.meshgrid(phi, theta)
    
    # Parametric equations for the ellipsoid
    x = a * np.sin(theta) * np.cos(phi)
    y = b * np.sin(theta) * np.sin(phi)
    z = c * np.cos(theta)
    
    # Flatten the arrays and stack them as a (N, 3) array of points
    points = np.vstack((x.flatten(), y.flatten(), z.flatten())).T.astype(np.float32)
    
    # Create faces using the points
    faces = []
    num_rows, num_cols = x.shape
    
    for i in range(num_rows - 1):
        for j in range(num_cols - 1):
            faces.append([i * num_cols + j,
                          i * num_cols + (j + 1),
                          (i + 1) * num_cols + j])
            faces.append([(i + 1) * num_cols + j,
                          i * num_cols + (j + 1),
                          (i + 1) * num_cols + (j + 1)])
    
    faces = np.array(faces, dtype=np.int32)
    
    # Create GIFTI data arrays
    coord_array = GiftiDataArray(data=points, intent="NIFTI_INTENT_POINTSET")
    face_array = GiftiDataArray(data=faces, intent="NIFTI_INTENT_TRIANGLE")
    
    # Create a GIFTI image
    gii = GiftiImage(darrays=[coord_array, face_array])
    
    # Save the GIFTI image
    nib.save(gii, output_path + os.sep +"ellipsoid.gii")
    
    # tranform .gii to .surf.gii file
    os.system(_FREESURFER_PATH + os.sep + "mris_convert '" +
              output_path + os.sep +"ellipsoid.gii" + "' '" + 
              output_filepath + "'")
    
    os.remove(output_path + os.sep +"ellipsoid.gii")
    
    # apply transform to focus (.surf.gii file) and set structure
    transform_surface_model(output_filepath, position_transform_filepath, output_filepath, "CORTEX_RIGHT")
    
    

def create_surface_transducer_model(radius, height, output_filepath):
    
    import os
    import numpy as np
    import nibabel as nib
    from nibabel.gifti import GiftiImage, GiftiDataArray
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    
    output_path, _ = os.path.split(output_filepath)

    # Number of points to approximate the circle
    num_points = 100
    
    # Create a grid of points around the disc
    theta = np.linspace(0, 2 * np.pi, num_points)
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    z = np.zeros_like(x)
    
    # Top and bottom faces of the disc, with the origin at the center of the bottom surface
    bottom_points = np.vstack((x, y, z)).T.astype(np.float32)
    top_points = np.vstack((x, y, z + height)).T.astype(np.float32)
    
    # Combine top and bottom points
    points = np.vstack((bottom_points, top_points))
    
    # Create faces for the side and top/bottom of the disc
    faces = []
    
    # Side faces
    for i in range(num_points - 1):
        faces.append([i, i + 1, num_points + i])
        faces.append([i + 1, num_points + i + 1, num_points + i])
    
    # Connect the last point to the first
    faces.append([num_points - 1, 0, 2 * num_points - 1])
    faces.append([0, num_points, 2 * num_points - 1])
    
    # Top and bottom faces
    for i in range(1, num_points - 1):
        faces.append([0, i, i + 1])
        faces.append([num_points, num_points + i, num_points + i + 1])
    
    faces = np.array(faces, dtype=np.int32)
    
    # Create GIFTI data arrays
    coord_array = GiftiDataArray(data=points, intent='NIFTI_INTENT_POINTSET')
    face_array = GiftiDataArray(data=faces, intent='NIFTI_INTENT_TRIANGLE')
    
    # Create a GIFTI image
    gii = GiftiImage(darrays=[coord_array, face_array])
    
    # Save the GIFTI image
    nib.save(gii, output_path + os.sep +"transducer.gii")
    
    # tranform .gii to .surf.gii file
    os.system(_FREESURFER_PATH + os.sep + "mris_convert '" +
              output_path + os.sep +"transducer.gii" + "' '" + 
              output_filepath +"'")
    
    os.remove(output_path + os.sep +"transducer.gii")



def kPlan_results_to_nifti(h5_filepath, CT_filepath):
    
    import os
    import numpy as np
    import h5py
    import nibabel as nib
    import ants
    
    #=============================================================================
    #=============================================================================
    # Generate output files
    #=============================================================================
    #=============================================================================
    
    # get path and name of results file
    h5_path = os.path.split(h5_filepath)[0]
    h5_name = os.path.split(h5_filepath)[1]
    h5_name = h5_name.replace('.h5', '')
    
    # load results
    results = h5py.File(h5_filepath)
    
    # get values and grid spacing of "medium mask" (i.e. tissue segmentation)
    # generated by k-Plan
    medium_mask = results["medium_properties/medium_mask"][:]
    medium_mask_grid_spacing = results["medium_properties/medium_mask"].attrs["grid_spacing"]
    
    # get values of acoustic pressure map generated by k-Plan
    sonications = results["sonications"]
    
    for i in np.arange(len(sonications)):
        
        sonication_simulated_field_pressure_amplitude = results["sonications/" + str(i+1) + os.sep +"simulated_field/pressure_amplitude"][:]
        sonication_simulated_field_thermal_dose = results["sonications/" + str(i+1) + os.sep +"simulated_field/thermal_dose"][:]
        
        # transform extracted values into 3D maps
        def get_affine(diagonal_element):
            affine = np.zeros([4,4])
            affine[0,0] = diagonal_element
            affine[1,1] = diagonal_element
            affine[2,2] = diagonal_element
            affine[3,3] = 1
            return affine
        
        affine = get_affine(medium_mask_grid_spacing[0]*1000)
        
        medium_mask = medium_mask.copy()
        medium_mask_image = nib.Nifti1Image(np.transpose(medium_mask), affine=affine)
        
        pressure_amplitude = sonication_simulated_field_pressure_amplitude.copy()
        pressure_amplitude_image = nib.Nifti1Image(np.transpose(pressure_amplitude), affine=affine)
        
        thermal_dose = sonication_simulated_field_thermal_dose.copy()
        thermal_dose_image = nib.Nifti1Image(np.transpose(thermal_dose), affine=affine)
        
        # reslice extracted 3D maps to same space and resolution as k-Plan input data
        # (i.e., pCT)
        
        CT_ants = ants.image_read(CT_filepath)
        medium_mask_image_ants = ants.from_nibabel(medium_mask_image)
        pressure_amplitude_image_ants = ants.from_nibabel(pressure_amplitude_image)
        thermal_dose_image_ants = ants.from_nibabel(thermal_dose_image)
        
        ants_registration = ants.registration(fixed=CT_ants,
                                              moving=medium_mask_image_ants,
                                              type_of_transform='TRSAA',
                                              reg_iterations = [100,3,4,5,6])
        
        
        medium_mask_image_ants_transformed = ants.apply_transforms(fixed=CT_ants,
                                                                   moving=medium_mask_image_ants,
                                                                   transformlist=ants_registration['fwdtransforms'],
                                                                   interpolator='linear')
        
        medium_mask_image_transformed = ants.to_nibabel(medium_mask_image_ants_transformed)
        
        
        pressure_amplitude_image_ants_transformed = ants.apply_transforms(fixed=CT_ants,
                                                                          moving=pressure_amplitude_image_ants,
                                                                          transformlist=ants_registration['fwdtransforms'],
                                                                          interpolator='linear')
        
        pressure_amplitude_image_transformed = ants.to_nibabel(pressure_amplitude_image_ants_transformed)
        
        
        thermal_dose_image_ants_transformed = ants.apply_transforms(fixed=CT_ants,
                                                                          moving=thermal_dose_image_ants,
                                                                          transformlist=ants_registration['fwdtransforms'],
                                                                          interpolator='linear')
        
        thermal_dose_image_transformed = ants.to_nibabel(thermal_dose_image_ants_transformed)
        
        # save final maps
        medium_mask_image_transformed.to_filename(h5_path + '/' + h5_name + '_MediumMask_Sonication' + str(i+1) + '.nii.gz')
        pressure_amplitude_image_transformed.to_filename(h5_path + '/' + h5_name + '_AcousticPressure_Sonication' + str(i+1) + '.nii.gz')
        thermal_dose_image_transformed.to_filename(h5_path + '/' + h5_name + '_ThermalDose_Sonication' + str(i+1) + '.nii.gz')
        
        

def prepare_acoustic_simulation(vertex_number,
                                output_path,
                                target_roi_filepath,
                                t1_filepath,
                                max_distance,
                                min_distance,
                                transducer_diameter,
                                max_angle,
                                offset,
                                additional_offset,
                                transducer_surface_model_filepath,
                                focal_distance_list,
                                flhm_list,
                                placement_scene_template_filepath,
                                ID="",
                                skip_wb_view=False,
                                use_internal_viewer=False):
    
    
    import os
    import shutil
    import numpy as np
    import scipy
    import math
    from nilearn import image
    import nibabel as nib
    
    if len(ID)==0:
        suffix="vtx" + str(vertex_number)
    else:
        suffix = ID
    output_path_vtx = output_path + os.sep + suffix
    os.makedirs(output_path_vtx, exist_ok=True)
    
    target_roi_filename = os.path.split(target_roi_filepath)[1]
    target_roi_name = target_roi_filename.replace(".nii", "")
    target_roi_name = target_roi_name.replace(".gz", "")


    #=============================================================================
    #
    #=============================================================================

    skin_coordinates, skin_normals = compute_surface_metrics(output_path + os.sep +"skin.surf.gii")

    target_center = roi_center_of_gravity(target_roi_filepath)
    skin_target_vectors = vectors_between_surface_and_point(output_path + os.sep +"skin.surf.gii", target_center)

    skin_target_intersections = compute_vector_mesh_intersections(skin_coordinates, skin_normals, output_path + os.sep + target_roi_name + "_3Dmodel.stl", 200)
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

    #==============================================================================
    # Get coordinates and normal vector for vetrex of interest
    #==============================================================================
    
    vertex_coordinates = skin_coordinates[vertex_number]
    
    if skin_target_intersection_values[vertex_number] == 0:
        vertex_vector = -skin_target_vectors[vertex_number]
        vertex_vector = unit_vector(vertex_vector)
    else:
        vertex_vector = -skin_normals[vertex_number] # negative normal vector pointing "into the head"
        vertex_vector = unit_vector(vertex_vector)
    
#==============================================================================
# Create transducer position matrix in Localite coordinate system
#==============================================================================
    
    transducer_center_coordinates = vertex_coordinates - ((offset + additional_offset) * vertex_vector)
    
    position_matrix_Localite = create_Localite_position_matrix(transducer_center_coordinates, vertex_vector)
    
    # Save
    scipy.io.savemat(output_path_vtx + os.sep +"position_matrix_" + target_roi_name + "_" + suffix + "_Localite.mat", {'position_matrix': position_matrix_Localite})
    np.savetxt(output_path_vtx + os.sep +"position_matrix_" + target_roi_name + "_" + suffix + "_Localite.txt", position_matrix_Localite)
    
    # Create "fake" XML structure with matrix values
    xml = create_fake_XML_structure_for_Localite(position_matrix_Localite, "transducer_position_" + target_roi_name + suffix, 0, 0)

    # Save/append "fake" XML structure in text file
    with open(output_path_vtx + os.sep +"position_matrix_" + target_roi_name + "_" + suffix + "_Localite_XML.txt", "a") as f:
        f.write(xml)
    
#==============================================================================
# Convert to k-Plan coordinate system
#==============================================================================
    
    position_matrix_kPlan = convert_Localite_to_kPlan_position_matrix(position_matrix_Localite)
    
    # Save
    scipy.io.savemat(output_path_vtx + os.sep +"position_matrix_" + target_roi_name + "_" + suffix + "_kPlan.mat", {'position_matrix': position_matrix_kPlan})
    np.savetxt(output_path_vtx + os.sep +"position_matrix_" + target_roi_name + "_" + suffix + "_kPlan.txt", position_matrix_kPlan)


#==============================================================================
# Optional: Transform transducer file
#==============================================================================

    transform = np.loadtxt(output_path_vtx + os.sep +"position_matrix_" + target_roi_name + "_" + suffix + "_kPlan.txt")
    transform[0:3,3] = transform[0:3,3]*1000
    
    transform_filepath = output_path_vtx + os.sep +"position_matrix_" + target_roi_name + "_" + suffix + "_transducer.txt"
    np.savetxt(transform_filepath, transform)
    
    output_filepath = output_path_vtx + os.sep +"transducer_" + target_roi_name + "_" + suffix + ".surf.gii"
    output_filepath_stl = output_path_vtx + os.sep +"transducer_" + target_roi_name + "_" + suffix + ".surf.stl"


    if not transducer_surface_model_filepath == "":
        transform_surface_model(transducer_surface_model_filepath, transform_filepath, output_filepath, "CEREBELLUM")
    else:
        
        create_surface_transducer_model(transducer_diameter/2, 15, output_filepath)
        transform_surface_model(output_filepath, transform_filepath, output_filepath, "CEREBELLUM")
    os.system(_FREESURFER_PATH + os.sep + "mris_convert '{}' '{}'".format(output_filepath,output_filepath_stl))

        
    
        
#==============================================================================
# Create transducer positon (.kps) file for k-Plan
#==============================================================================

    position_matrix_kPlan_filepath  = output_path_vtx + os.sep +"position_matrix_" + target_roi_name + "_" + suffix + "_kPlan.mat"
    kps_filename = target_roi_name + suffix
    
    create_kps_file_for_kPlan(position_matrix_kPlan_filepath, kps_filename)


#==============================================================================
# Compute required focal distance and corresponding FLHM
#==============================================================================
    
    # Compute distance from between head surface and center of target
    # (intersection), i.e. required focal distance
    if skin_target_intersection_values[vertex_number] == 0:
        target_center = roi_center_of_gravity(target_roi_filepath)
        skin_target_distances = distance_between_surface_and_point(output_path + os.sep +"skin.surf.gii", target_center)
        focal_distance = skin_target_distances[vertex_number] + additional_offset
    else:
        skin_target_intersection_center = (np.asarray(skin_target_intersections[vertex_number][0]) + np.asarray(skin_target_intersections[vertex_number][1]))/2
        focal_distance = np.linalg.norm(skin_coordinates[vertex_number] - skin_target_intersection_center) + additional_offset # add additional distance
    
    if focal_distance > max_distance:
        focal_distance = max_distance
    elif focal_distance < min_distance:
        focal_distance = min_distance
        
    # Compute expected FLHM
    FLHM = compute_FLHM_for_focal_distance(focal_distance, focal_distance_list, flhm_list)   
    

#==============================================================================
# Create ellipsoid (surface)
#==============================================================================

    # create and save transform for focus
    focus_position_transform = np.loadtxt(output_path_vtx + os.sep +"position_matrix_" + target_roi_name + "_" + suffix + "_kPlan.txt")
    focus_position_transform[0:3,3] = focus_position_transform[0:3,3] * 1000
    focus_position_transform[0:3,3] = focus_position_transform[0:3,3] + (vertex_vector * (offset + focal_distance))
        
    focus_position_transform_filepath = output_path_vtx + os.sep +"focus_position_matrix_" + target_roi_name + "_" + suffix + ".txt"        
    np.savetxt(focus_position_transform_filepath, focus_position_transform)
    
    ellipsoid_output_filepath = output_path_vtx + os.sep +"focus_" + target_roi_name + "_" + suffix + "_" + str(round(focal_distance,1)) + ".surf.gii"

    create_surface_ellipsoid(FLHM, 5,
                                     focus_position_transform_filepath,
                                     t1_filepath,
                                     ellipsoid_output_filepath)

    
#==============================================================================
# Create ellipsoid (volume)
#==============================================================================

    ellipsoid_volume_output_filepath = output_path_vtx + os.sep +"focus_" + target_roi_name + "_" + suffix + "_" + str(round(focal_distance,1)) + ".nii.gz"


    # create surface ellipsoid with smaller dimensions
    ellipsoid_small_output_filepath = output_path_vtx + os.sep +"focus_" + target_roi_name + "_" + suffix + "_" + str(round(focal_distance,1)) + "_small.surf.gii"

    create_surface_ellipsoid(FLHM-1, 5-1,
                             focus_position_transform_filepath,
                             t1_filepath,
                             ellipsoid_small_output_filepath)

    
    # create metric from surface
    os.system(_CONNECTOME_PATH + os.sep + "wb_command -logging OFF -surface-coordinates-to-metric '{}' '{}'".format(
        ellipsoid_output_filepath,
        output_path_vtx + os.sep +"focus.func.gii"))

    # map metric to volume
    os.system(_CONNECTOME_PATH + os.sep + "wb_command -logging OFF -metric-to-volume-mapping '{}' '{}' '{}' '{}' -ribbon-constrained '{}' '{}'".format(
        output_path_vtx + os.sep +"focus.func.gii",
        ellipsoid_output_filepath,
        t1_filepath,
        ellipsoid_volume_output_filepath,
        ellipsoid_small_output_filepath,
        ellipsoid_output_filepath))
    
    # fill holes in volume
    os.system(_CONNECTOME_PATH + os.sep + "wb_command -logging OFF -volume-fill-holes '{}' '{}'".format(
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
    os.remove(output_path_vtx + os.sep +"focus.func.gii")
    os.remove(ellipsoid_small_output_filepath)


#==============================================================================
# Visualize results
#==============================================================================
    if skip_wb_view:
        return
    if use_internal_viewer:
        from Viewer import FinalResultViewer
        from PyQt5.QtWidgets import QDialog,QVBoxLayout
        DlgResults=QDialog()
        DlgResults.setWindowTitle("Trajectory Results")

        layout = QVBoxLayout()
        DlgResults.setLayout(layout)

        gifti_files = []
        gifti_files.append(output_path+os.sep+'skin.surf.gii')
        gifti_files.append(output_filepath)
        
        widget = FinalResultViewer(gifti_files)
        layout.addWidget(widget)
        DlgResults.resize(600, 600)
        DlgResults.exec()
    else:
        scene_variable_names = [
            'SKIN_SURFACE_FILENAME',
            'SKIN_SURFACE_FILEPATH',
            'T1_FILENAME',
            'T1_FILEPATH',
            'MASK_FILENAME',
            'MASK_FILEPATH',
            'TRANSDUCER_SURFACE_FILENAME',
            'TRANSDUCER_SURFACE_FILEPATH',
            'FOCUS_VOLUME_FILENAME',
            'FOCUS_VOLUME_FILEPATH',
            'FOCUS_SURFACE_FILENAME',
            'FOCUS_SURFACE_FILEPATH'
            ]

        scene_variable_values = [
            'skin.surf.gii',
            '../skin.surf.gii',
            'T1.nii.gz',
            '../../../T1.nii.gz',
            target_roi_filename,
            '../' + target_roi_filename,
            "transducer_" + target_roi_name + "_" + suffix + ".surf.gii",
            "./transducer_" + target_roi_name + "_" + suffix + ".surf.gii",
            "focus_" + target_roi_name + "_" + suffix + "_" + str(round(focal_distance,1)) + ".nii.gz",
            "./focus_" + target_roi_name + "_" + suffix + "_" + str(round(focal_distance,1)) + ".nii.gz",
            "focus_" + target_roi_name + "_" + suffix + "_" + str(round(focal_distance,1)) + ".surf.gii",
            "./focus_" + target_roi_name + "_" + suffix + "_" + str(round(focal_distance,1)) + ".surf.gii"]


        create_scene(placement_scene_template_filepath, output_path_vtx + os.sep +"scene.scene", scene_variable_names, scene_variable_values)

        os.system(_CONNECTOME_PATH + os.sep + "wb_view -logging OFF '" + output_path_vtx + "'" + os.sep +"scene.scene")
