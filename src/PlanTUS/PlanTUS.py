#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PlanTUS core functions
======================

Dependencies
------------------
- SimNIBS
- Connectome Workbench (`wb_command`)

"""

from __future__ import annotations

import os
import glob
import shutil
from typing import List, Sequence, Tuple

import numpy as np
import nibabel as nib


# -----------------------------------------------------------------------------
# Utility: set paths to dependencies
# -----------------------------------------------------------------------------

def set_paths(connectome_wb_path=None):
    global _CONNECTOME_WB_PATH, wb_command_cmd
    _CONNECTOME_WB_PATH = connectome_wb_path
    wb_command_cmd = _CONNECTOME_WB_PATH + os.sep + "wb_command"


# -----------------------------------------------------------------------------
# Basic NIfTI ops
# -----------------------------------------------------------------------------

def _load_nii(path: str) -> Tuple[nib.Nifti1Header, np.ndarray, np.ndarray]:
    """Load a NIfTI file as float64 array.

    Returns
    -------
    header : nib.Nifti1Header
    data : (X,Y,Z[,...]) ndarray float64
    affine : (4,4) ndarray
    """
    img = nib.load(path)
    data = img.get_fdata()  # float64
    return img.header, data, img.affine


def save_like(ref_header: nib.Nifti1Header, ref_affine: np.ndarray, data: np.ndarray, out_path: str) -> None:
    """Save `data` to NIfTI using `ref_header` and `ref_affine` for geometry.

    Parameters
    ----------
    ref_header : nib.Nifti1Header
        Header from a reference image. A copy is used.
    ref_affine : ndarray, shape (4,4)
        Affine transform mapping voxel indices to world (mm).
    data : ndarray
        Array to save. Will be written as float32 unless boolean/integer.
    out_path : str
        Destination path (".nii" or ".nii.gz").
    """
    hdr = ref_header.copy()
    dtype = np.float32
    if data.dtype == bool:
        data = data.astype(np.uint8)
        dtype = np.uint8
    elif np.issubdtype(data.dtype, np.integer):
        dtype = data.dtype
    img = nib.Nifti1Image(np.asarray(data, dtype=dtype), ref_affine, header=hdr)
    nib.save(img, out_path)


def threshold_nifti(in_path: str, thr: float, out_path: str) -> None:
    """Threshold image: values >= `thr` are kept, others set to 0.

    """
    hdr, data, aff = _load_nii(in_path)
    out = np.where(data >= float(thr), data, 0.0)
    save_like(hdr, aff, out.astype(np.float32), out_path)


def binarize_nifti(in_path: str, out_path: str) -> None:
    """Binarize image: any non-zero voxel becomes 1.

    """
    hdr, data, aff = _load_nii(in_path)
    out = (data != 0).astype(np.uint8)
    save_like(hdr, aff, out, out_path)


def subtract_nifti(in_a: str, in_b: str, out_path: str) -> None:
    """Voxelwise subtraction: ``out = A - B``.

    """
    hdr_a, A, aff_a = _load_nii(in_a)
    _, B, _ = _load_nii(in_b)
    save_like(hdr_a, aff_a, (A - B).astype(np.float32), out_path)


# -----------------------------------------------------------------------------
# NIfTI → surface mesh via marching cubes (formerly mri_tessellate/mris_convert)
# -----------------------------------------------------------------------------

def _marching_cubes_from_binary_volume(vol_path: str,
                                       isovalue: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    """Extract a triangular surface from a (binary) volume using marching cubes.

    Parameters
    ----------
    vol_path : str
        Path to NIfTI volume. Typically binary (0/1). The affine is used to
        place vertices in world (mm) space.
    isovalue : float, default=0.5
        Iso-surface value.

    Returns
    -------
    vertices_mm : (N, 3) float32
        Vertex coordinates in **mm** (world) space.
    faces : (M, 3) int32
        Triangle indices.
    """

    from simnibs.segmentation.marching_cube import marching_cube  # type: ignore

    img = nib.load(vol_path)
    vol = img.get_fdata()
    if len(vol.shape) == 4:
        vol = vol[:,:,:,0]
    aff = img.affine

    mesh, faces = marching_cube(vol.astype(np.float32), affine=aff)
    verts_mm = mesh.nodes.node_coord
    tri_mask = (mesh.elm.elm_type == 2).ravel()
    faces = mesh.elm.node_number_list[tri_mask, :3].astype(np.int32) - 1  # (M, 3)

    return mesh, verts_mm, faces.astype(np.int32)


def _write_gifti_surface(vertices_mm: np.ndarray, faces: np.ndarray, out_gii: str) -> None:
    """Write a GIFTI surface (*.surf.gii) from vertices and faces.

    Uses NiBabel's GIFTI writer. Headers are minimal.
    """
    from nibabel.gifti import GiftiImage, GiftiDataArray

    # Build data arrays; let NiBabel derive datatype from data
    coords_da = GiftiDataArray(
        data=np.asarray(vertices_mm, dtype=np.float32),
        intent="NIFTI_INTENT_POINTSET",
        datatype=nib.nifti1.data_type_codes['NIFTI_TYPE_FLOAT32'],
        )

    faces_da = GiftiDataArray(
        data=np.asarray(faces, dtype=np.int32),
        intent="NIFTI_INTENT_TRIANGLE",
        datatype=nib.nifti1.data_type_codes['NIFTI_TYPE_FLOAT32'],
        )

    gii = GiftiImage(darrays=[coords_da, faces_da])

    nib.save(gii, out_gii)


# -----------------------------------------------------------------------------
# Scene templating
# -----------------------------------------------------------------------------

def create_scene(scene_template_filepath: str,
                 output_filepath: str,
                 scene_variable_names: Sequence[str],
                 scene_variable_values: Sequence[str]) -> None:
    """Create a scene file by replacing placeholders.

    Parameters
    ----------
    scene_template_filepath : str
        Path to a template *.scene file containing placeholders.
    output_filepath : str
        Destination path for the filled-in scene.
    scene_variable_names : list[str]
        Strings to be replaced in the template (regex safe if possible).
    scene_variable_values : list[str]
        Replacement values, matched by index to names.
    """
    import re

    out_dir = os.path.split(output_filepath)[0]
    os.makedirs(out_dir, exist_ok=True)
    with open(scene_template_filepath, "r", encoding="utf-8") as f:
        txt = f.read()
    for name, val in zip(scene_variable_names, scene_variable_values):
        txt = re.sub(re.escape(name), str(val), txt)
    with open(output_filepath, "w", encoding="utf-8") as f:
        f.write(txt)


# -----------------------------------------------------------------------------
# Mesh conversions
# -----------------------------------------------------------------------------

def convert_simnibs_mesh_to_surfaces(simnibs_mesh_filepath: str,
                                     tags: Sequence[int],
                                     mesh_name: str,
                                     output_path: str) -> None:
    """Extract tagged compartment from SimNIBS mesh and write as STL + GIFTI.

    Parameters
    ----------
    simnibs_mesh_filepath : str
        Path to *.msh created by SimNIBS/charm.
    tags : sequence of int
        Region tags to keep (e.g. 1005=skin, 1007=skull, ...).
    mesh_name : str
        Basename for output files.
    output_path : str
        Destination directory.

    Outputs
    -------
    * ``{mesh_name}.stl`` STL surface
    * ``{mesh_name}.surf.gii`` GIFTI surface
    """
    from simnibs import mesh_io

    os.makedirs(output_path, exist_ok=True)

    # Load mesh
    mesh = mesh_io.read_msh(simnibs_mesh_filepath).crop_mesh(tags=list(tags))

    # Write GIFTI surface -----------------------------------------------------
    gii_out = os.path.join(output_path, f"{mesh_name}.surf.gii")

    # Optional: ensure consistent triangle winding
    if hasattr(mesh, "fix_tr_node_ordering"):
        mesh.fix_tr_node_ordering()

    # Vertices: (N, 3)
    vertices = mesh.nodes.node_coord.astype(np.float32)

    # Faces: filter triangles (elm_type==2), take first 3 node indices, convert 1-based -> 0-based
    tri_mask = (mesh.elm.elm_type == 2).ravel()
    faces = mesh.elm.node_number_list[tri_mask, :3].astype(np.int32) - 1  # (M, 3)

    # write gifti
    _write_gifti_surface(vertices, faces, gii_out)

    # Write STL directly ------------------------------------------------------
    stl_out = os.path.join(output_path, f"{mesh_name}.stl")
    mesh_io.write_stl(mesh, stl_out)


def surf_gii_to_stl_with_simnibs(in_gii: str, out_stl: str, tag: int = 1):
    """
    Convert a .surf.gii surface to .stl using SimNIBS mesh_io.

    Parameters
    ----------
    in_gii : str
        Path to input GIfTI surface file (.surf.gii)
    out_stl : str
        Path to output STL file
    tag : int
        Element tag ID (default=1). Used by SimNIBS to group triangles.
    """
    from simnibs.mesh_tools import mesh_io as mio

    # Load GIfTI
    gii = nib.load(in_gii)
    coords = gii.darrays[0].data.astype(np.float64)
    faces  = gii.darrays[1].data.astype(np.int32) + 1  # SimNIBS expects 1-based indices

    # Create mesh
    msh = mio.Msh()
    msh.nodes.node_coord = coords
    msh.elm.add_triangles(faces, tag)

    # Write STL
    mio.write_stl(msh, out_stl)


# -----------------------------------------------------------------------------
# Surface metrics & helpers (Workbench + Nilearn)
# -----------------------------------------------------------------------------

def compute_surface_metrics(surface_filepath: str) -> Tuple[np.ndarray, np.ndarray]:
    """Compute per-vertex coordinates and normals for a GIFTI surface.

    Uses Connectome Workbench to export metrics and Nilearn to load them.

    Returns
    -------
    coordinates : (N, 3) float
    normals : (N, 3) float
    """
    from nilearn import surface

    out_dir, fname = os.path.split(surface_filepath)
    base = fname.replace(".surf.gii", "")

    coords_func = os.path.join(out_dir, f"{base}_coordinates.func.gii")
    norms_func = os.path.join(out_dir, f"{base}_normals.func.gii")

    os.system(f"'{wb_command_cmd}' -logging OFF -surface-coordinates-to-metric '{surface_filepath}' '{coords_func}'")
    os.system(f"'{wb_command_cmd}' -logging OFF -surface-normals '{surface_filepath}' '{norms_func}'")

    coords = np.asarray(surface.load_surf_data(coords_func),dtype=float)
    norms = np.asarray(surface.load_surf_data(norms_func),dtype=float)

    os.remove(coords_func)
    os.remove(norms_func)
    return coords, norms


def create_pseudo_metric_nifti_from_surface(surface_filepath: str) -> Tuple[nib.Nifti1Image, np.ndarray]:
    """Create an auxiliary NIfTI metric sized for the surface vertices.

    Returns
    -------
    nii : nib.Nifti1Image
    data : ndarray
        Array shaped (N, 1, 1) where N is number of vertices.
    """
    from nilearn import image

    out_dir, fname = os.path.split(surface_filepath)
    base = fname.replace(".surf.gii", "")

    coords_func = os.path.join(out_dir, f"{base}_coordinates.func.gii")
    coords_mean_func = os.path.join(out_dir, f"{base}_coordinates_MEAN.func.gii")
    coords_mean_nii = os.path.join(out_dir, f"{base}_coordinates_MEAN.nii.gz")

    os.system(f"'{wb_command_cmd}' -logging OFF -surface-coordinates-to-metric '{surface_filepath}' '{coords_func}'")
    os.system(f"'{wb_command_cmd}' -logging OFF -metric-reduce '{coords_func}' MEAN '{coords_mean_func}'")
    os.system(f"'{wb_command_cmd}' -logging OFF -metric-convert -to-nifti '{coords_mean_func}' '{coords_mean_nii}'")

    nii = image.load_img(coords_mean_nii)
    data = nii.get_fdata()

    # cleanup
    for p in [coords_func, coords_mean_func, coords_mean_nii]:
        try:
            os.remove(p)
        except OSError:
            pass
    return nii, data


def create_metric_from_pseudo_nifti(metric_name: str,
                                    metric_values: Sequence[float],
                                    surface_filepath: str) -> None:
    """Save per-vertex values to both NIfTI and GIFTI metric for a surface.

    Parameters
    ----------
    metric_name : str
        Basename for outputs.
    metric_values : sequence of float
        One value per surface vertex.
    surface_filepath : str
        Path to the reference *.surf.gii.

    Outputs
    -------
    * ``{metric_name}_{surface_name}.func.gii`` (and a transient NIfTI)
    """
    import os
    import nibabel as nib

    # out_dir, fname = os.path.split(surface_filepath)
    # surface_name = fname.replace(".surf.gii", "")

    # nii, proto = create_pseudo_metric_nifti_from_surface(surface_filepath)
    # proto = np.asarray(proto)

    # values = np.asarray(metric_values, dtype=np.float32)
    # if values.ndim == 1:
    #     values = values[:, None, None]  # (N,1,1)

    # n_verts = proto.shape[0]
    # if values.shape[0] != n_verts:
    #     raise ValueError(f"Number of metric values ({values.shape[0]}) does not match vertices ({n_verts}).")

    # nii_new = nib.Nifti1Image(values, nii.affine, header=nii.header)
    # tmp_nii = os.path.join(out_dir, f"{metric_name}_{surface_name}.nii.gz")
    # nib.save(nii_new, tmp_nii)

    # out_func = os.path.join(out_dir, f"{metric_name}_{surface_name}.func.gii")
    # os.system(
    #     f"'{wb_command_cmd}' -logging OFF -metric-convert -from-nifti {tmp_nii} '{surface_filepath}' {out_func}"
    # )
    # os.remove(tmp_nii)

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

    os.system("'{}' -logging OFF -metric-convert -from-nifti '{}' '{}' '{}'".format(
        wb_command_cmd,
        output_path + os.sep + metric_name + "_" + surface_name + ".nii.gz",
        surface_filepath,
        output_path + os.sep + metric_name + "_" + surface_name + ".func.gii"))

    os.remove(output_path + os.sep + metric_name + "_" + surface_name + ".nii.gz")


def erode_metric(metric_filepath: str, surface_filepath: str, erosion_factor: float) -> None:
    """Erode a metric on the surface using Workbench in-place."""
    os.system(
        f"'{wb_command_cmd}' -logging OFF -metric-erode '{metric_filepath}' '{surface_filepath}' {erosion_factor} '{metric_filepath}'"
    )


# -----------------------------------------------------------------------------
# Geometry intersections
# -----------------------------------------------------------------------------

def load_stl(stl_filepath: str):
    """Load an STL file as VTK PolyData.

    Raises
    ------
    ValueError
        If the STL has zero points.
    """
    import vtk  # type: ignore

    reader = vtk.vtkSTLReader()
    reader.SetFileName(stl_filepath)
    reader.Update()
    poly = reader.GetOutput()
    if poly.GetNumberOfPoints() == 0:
        raise ValueError(f"No point data could be loaded from '{stl_filepath}'")
    return poly


def compute_vector_mesh_intersections(points: np.ndarray,
                                      vectors: np.ndarray,
                                      mesh_filepath: str,
                                      vector_length: float) -> List[List[Tuple[float, float, float]]]:
    """Intersect rays with a triangular mesh.

    Parameters
    ----------
    points : (N, 3) float
        Ray origins.
    vectors : (N, 3) float
        Ray directions (will not be normalized here).
    mesh_filepath : str
        Path to STL mesh.
    vector_length : float
        Length of the rays (mm). Negative values trace in the opposite
        direction. End point is computed as ``p - L * v`` where ``v`` is the
        provided vector (for historic compatibility).

    Returns
    -------
    intersections : list[list[tuple]]
        For each ray a list of hit points (x,y,z).
    """
    import vtk  # type: ignore

    mesh = load_stl(mesh_filepath)

    obb = vtk.vtkOBBTree()
    obb.SetDataSet(mesh)
    obb.BuildLocator()

    hits: List[List[Tuple[float, float, float]]] = []
    for i in range(len(points)):
        p0 = points[i]
        p1 = points[i] - vector_length * vectors[i]
        pts = vtk.vtkPoints()
        _ = obb.IntersectWithLine(p0, p1, pts, None)
        dat = pts.GetData()
        n = dat.GetNumberOfTuples()
        ray_hits: List[Tuple[float, float, float]] = []
        for j in range(n):
            ray_hits.append(dat.GetTuple3(j))
        hits.append(ray_hits)
    return hits


# -----------------------------------------------------------------------------
# High-level helpers
# -----------------------------------------------------------------------------

def create_avoidance_mask(simnibs_mesh_filepath: str,
                          surface_filepath: str,
                          erosion_factor: float) -> np.ndarray:
    """Create a surface mask (0/1) of vertices to avoid for transducer placement.

    Steps (brief)
    -------------
    1) Load ``final_tissues.nii.gz`` → binarize (air cavities are zero)
    2) Fill holes in the binary volume using Workbench (kept)
    3) Subtract to get an *air-only* mask
    4) Tessellate the air mask to STL via marching cubes
    5) Intersect inward normals from the skin with this mesh → mark vertices
    6) Add safety regions at the eyes and around ears using EEG fiducials
    7) Erode mask on-surface for safety margin

    Returns
    -------
    avoidance_mask : (N,) float
        Per-vertex 0/1 mask for the input ``surface_filepath``.
    """
    import pandas as pd
    from scipy.cluster.vq import kmeans
    from scipy import ndimage
    from simnibs import mesh_io

    out_dir = os.path.split(surface_filepath)[0]
    surf_name = os.path.split(surface_filepath)[1].replace(".surf.gii", "")
    m2m_dir = os.path.split(simnibs_mesh_filepath)[0]

    # ---- 1) binary mask of final tissues
    final_tissues = os.path.join(m2m_dir, "final_tissues.nii.gz")
    bin_path = os.path.join(out_dir, "final_tissues_bin.nii.gz")
    binarize_nifti(final_tissues, bin_path)  # in-place ok

    # ---- 2) fill holes (workbench)
    filled_path = os.path.join(out_dir, "final_tissues_bin_filled.nii.gz")
    # se_n = ndimage.generate_binary_structure(3,1)
    # img = nib.load(bin_path)
    # vol = img.get_fdata()
    # if len(vol.shape) == 4:
    #     vol = vol[:,:,:,0]
    # aff = img.affine
    # filled = ndimage.binary_fill_holes(vol, se_n)
    # filled_out = nib.Nifti1Image(filled.astype(np.uint16), aff)
    # nib.save(filled_out, filled_path)
    os.system(f"'{wb_command_cmd}' -logging OFF -volume-fill-holes '{bin_path}' '{filled_path}'")

    # ---- 3) get air-only mask = filled - bin
    air_path = os.path.join(out_dir, "final_tissues_air.nii.gz")
    subtract_nifti(filled_path, bin_path, air_path)

    # ---- 4) tessellate air mask → STL and GIFTI (for optional smoothing)
    mesh, verts, faces = _marching_cubes_from_binary_volume(air_path, isovalue=0.5)
    air_gii = os.path.join(out_dir, "final_tissues_air.surf.gii")
    air_stl = os.path.join(out_dir, "final_tissues_air.stl")
    _write_gifti_surface(verts, faces, air_gii)
    mesh_io.write_stl(mesh, air_stl)

    # Optional smoothing using Workbench (kept)
    os.system(f"'{wb_command_cmd}' -logging OFF -surface-smoothing '{air_gii}' 0.5 10 '{air_gii}'")

    # ---- 5) intersect inward normals from skin with air cavities
    skin_coords, skin_normals = compute_surface_metrics(surface_filepath)
    air_hits = compute_vector_mesh_intersections(skin_coords, skin_normals, air_stl, 40)

    avoidance_mask = []
    for hits in air_hits:
        avoidance_mask.append(0 if len(hits) > 0 else 1)
    avoidance_mask = np.asarray(avoidance_mask, dtype=float)

    # ---- 6) eyes region via SimNIBS mesh tags
    convert_simnibs_mesh_to_surfaces(simnibs_mesh_filepath, [1006], "eyes", out_dir)
    eyes_coords, _ = compute_surface_metrics(os.path.join(out_dir, "eyes.surf.gii"))
    eyes_coords = np.asarray(eyes_coords, dtype=float)

    # separate left/right via 1D kmeans on x, then build centers
    x_centers, _ = kmeans(eyes_coords[:, 0], 2)
    left = eyes_coords[eyes_coords[:, 0] < (np.sum(x_centers) / 2)]
    right = eyes_coords[eyes_coords[:, 0] > (np.sum(x_centers) / 2)]
    left_c = np.mean(left, axis=0)
    right_c = np.mean(right, axis=0)

    # 30 mm radius around eye centers
    avoidance_mask[np.linalg.norm((skin_coords - left_c), axis=1) < 30] = 0
    avoidance_mask[np.linalg.norm((skin_coords - right_c), axis=1) < 30] = 0

    # ears via EEG fiducials (LPA/RPA) shifted 15 mm posterior
    eeg_fids = os.path.join(m2m_dir, "eeg_positions", "Fiducials.csv")
    df = pd.read_csv(eeg_fids, header=1)
    LPA = df.iloc[0, 1:4].to_numpy(np.float64)
    RPA = df.iloc[1, 1:4].to_numpy(np.float64)
    LPA[1] -= 15
    RPA[1] -= 15
    avoidance_mask[np.linalg.norm((skin_coords - LPA), axis=1) < 15] = 0
    avoidance_mask[np.linalg.norm((skin_coords - RPA), axis=1) < 15] = 0

    # everything below eye-height/2
    mean_eye_z = ((left_c + right_c) / 2)[2]
    avoidance_mask[skin_coords[:, 2] <= (mean_eye_z / 2)] = 0

    # Save as metric on surface and erode
    create_metric_from_pseudo_nifti("avoidance", avoidance_mask, surface_filepath)
    erode_metric(os.path.join(out_dir, f"avoidance_{surf_name}.func.gii"), surface_filepath, erosion_factor)

    # Reload eroded mask values
    gii = nib.load(os.path.join(out_dir, f"avoidance_{surf_name}.func.gii"))
    avoidance_mask = np.asarray(gii.darrays[0].data)

    # Cleanup intermediates
    for f in glob.glob(os.path.join(out_dir, "final_tissues_air*")):
        try:
            os.remove(f)
        except OSError:
            pass
    for f in glob.glob(os.path.join(out_dir, "eyes*")):
        try:
            os.remove(f)
        except OSError:
            pass

    return avoidance_mask


# -----------------------------------------------------------------------------
# Linear algebra helpers
# -----------------------------------------------------------------------------

def unit_vector(vector: np.ndarray) -> np.ndarray:
    """Return the unit vector of ``vector``."""
    return vector / np.linalg.norm(vector)


def angle_between_vectors(v1: np.ndarray, v2: np.ndarray) -> float:
    """Angle in radians between vectors ``v1`` and ``v2``."""
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return float(np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)))


# -----------------------------------------------------------------------------
# ROI helpers
# -----------------------------------------------------------------------------

def roi_center_of_gravity(roi_filepath: str) -> np.ndarray:
    """Return (x,y,z) center of gravity for a NIfTI ROI.

    Uses Nilearn's center-finder on a probabilistic atlas (works robustly on
    binary masks as well).
    """
    from nilearn import image, plotting

    roi = image.load_img(roi_filepath)
    center = plotting.find_probabilistic_atlas_cut_coords([roi])[0]
    return np.asarray(center)


def vectors_between_surface_and_point(surface_filepath: str, point_coordinates: Sequence[float]) -> np.ndarray:
    """Vector from each surface vertex to a target 3D point.

    Returns
    -------
    (N, 3) ndarray
    """
    coords, _ = compute_surface_metrics(surface_filepath)
    return np.asarray(coords) - np.asarray(point_coordinates)


def distance_between_surface_and_point(surface_filepath: str, point_coordinates: Sequence[float]) -> np.ndarray:
    """Euclidean distance from each surface vertex to a point."""
    vecs = vectors_between_surface_and_point(surface_filepath, point_coordinates)
    return np.linalg.norm(vecs, axis=1)


# -----------------------------------------------------------------------------
# NIfTI → STL (threshold + tessellate)
# -----------------------------------------------------------------------------

def stl_from_nii(nii_filepath: str, threshold: float) -> None:
    """Create a smoothed STL surface named ``*_3Dmodel.stl`` from a NIfTI mask.

    Steps
    -----
    1) Threshold and binarize with NumPy
    2) Marching cubes to get vertices/faces
    3) Write GIFTI and STL; optionally smooth the GIFTI via Workbench
    4) Convert smoothed GIFTI back to STL
    """
    from simnibs import mesh_io

    out_dir, in_name = os.path.split(nii_filepath)
    base = in_name.replace(".nii.gz", "").replace(".nii", "") + "_3Dmodel"

    # 1) threshold & binarize to temporary file
    thr_path = os.path.join(out_dir, base + "_thr.nii.gz")
    bin_path = os.path.join(out_dir, base + "_bin.nii.gz")
    threshold_nifti(nii_filepath, float(threshold), thr_path)
    binarize_nifti(thr_path, bin_path)

    # 2) marching cubes
    mesh, verts, faces = _marching_cubes_from_binary_volume(bin_path)

    # 3) write GIFTI, smooth, then 4) write STL
    gii_path = os.path.join(out_dir, base + ".surf.gii")
    stl_path = os.path.join(out_dir, base + ".stl")
    _write_gifti_surface(verts, faces, gii_path)
    os.system(f"'{wb_command_cmd}' -logging OFF -surface-smoothing '{gii_path}' 0.5 10 '{gii_path}'")

    surf_gii_to_stl_with_simnibs(gii_path, stl_path)


    # cleanup temps
    for p in [thr_path, bin_path]:
        try:
            os.remove(p)
        except OSError:
            pass

    # keep the smoothed *.surf.gii as in the original pipeline


# -----------------------------------------------------------------------------
# Workbench helpers
# -----------------------------------------------------------------------------

def smooth_metric(metric_filepath: str, surface_filepath: str, FWHM: float) -> None:
    """Surface-morphometric smoothing (FWHM in mm) via Workbench."""
    out_dir, fname = os.path.split(metric_filepath)
    name = fname.replace(".func.gii", "")
    os.system(
        "'{}' -logging OFF -metric-smoothing '{}' '{}' '{}' '{}' -fwhm".format(
            wb_command_cmd,
            surface_filepath,
            metric_filepath,
            str(FWHM),
            os.path.join(out_dir, f"{name}_s{FWHM}.func.gii"),
        )
    )


def mask_metric(metric_filepath: str, mask_filepath: str) -> None:
    """Apply a binary mask to a metric (in-place) via Workbench."""
    os.system(
        f"'{wb_command_cmd}' -logging OFF -metric-mask '{metric_filepath}' '{mask_filepath}' '{metric_filepath}'"
    )


def threshold_metric(metric_filepath: str, threshold: float) -> None:
    """Create a thresholded copy of a metric using Workbench."""
    out_dir, fname = os.path.split(metric_filepath)
    name = fname.replace(".func.gii", "")
    out = os.path.join(out_dir, f"{name}_thresholded.func.gii")
    os.system(
        f"'{wb_command_cmd}' -logging OFF -metric-math 'x < {threshold}' '{out}' -var x '{metric_filepath}'"
    )


def add_structure_information(filepath: str, structure_label: str) -> None:
    """Annotate a surface or metric with a Workbench structure label."""
    os.system(
        f"'{wb_command_cmd}' -logging OFF -set-structure '{filepath}' {structure_label} -surface-type RECONSTRUCTION"
    )


# -----------------------------------------------------------------------------
# Localite / k-Plan utilities
# -----------------------------------------------------------------------------

def create_Localite_position_matrix(center_coordinates: Sequence[float], x_vector: Sequence[float]) -> np.ndarray:
    """Build a 4×4 Localite-style pose matrix from a center and an x-axis.

    The y-axis is chosen orthogonal to x via a random vector, and z is x×y.
    """
    center = np.asarray(center_coordinates, dtype=float)
    x = np.asarray(x_vector, dtype=float)

    M = np.zeros((4, 4), dtype=float)
    M[3, 3] = 1.0
    M[:3, 3] = center

    x /= np.linalg.norm(x)
    M[:3, 0] = x

    y = np.random.randn(3)
    y -= y.dot(x) * x
    y /= np.linalg.norm(y)
    M[:3, 1] = y

    z = np.cross(x, y)
    z /= np.linalg.norm(z)
    M[:3, 2] = z
    return M


def create_fake_XML_structure_for_Localite(Localite_position_matrix: np.ndarray,
                                           position_name: str,
                                           element_index: int,
                                           uid: int) -> str:
    """Return an XML snippet for a Localite InstrumentMarker."""
    M = Localite_position_matrix
    def f(v: float) -> str:
        return str(round(float(v), 6))
    xml = (
        f'    <Element index="{element_index}" selected="true" type="InstrumentMarker">\n'
        f'            <InstrumentMarker additionalInformation="" alwaysVisible="false"\n'
        f'                color="#00ff00" description="{position_name}" locked="false" set="true" uid="{uid}">\n'
        f'                <Matrix4D data00="{f(M[0,0])}" data01="{f(M[0,1])}" data02="{f(M[0,2])}"\n'
        f'                    data03="{f(M[0,3])}" data10="{f(M[1,0])}" data11="{f(M[1,1])}"\n'
        f'                    data12="{f(M[1,2])}" data13="{f(M[1,3])}" data20="{f(M[2,0])}"\n'
        f'                    data21="{f(M[2,1])}" data22="{f(M[2,2])}" data23="{f(M[2,3])}" data30="0.0"\n'
        f'                    data31="0.0" data32="0.0" data33="1.0"/>\n'
        f'            </InstrumentMarker>\n'
        f'        </Element>\n'
    )
    return xml


def convert_Localite_to_kPlan_position_matrix(Localite_position_matrix: np.ndarray) -> np.ndarray:
    """Convert Localite 4×4 matrix to k-Plan coordinates.

    Also converts translation from mm (Localite) to meters (k-Plan).
    """
    M = Localite_position_matrix.copy()
    K = M.copy()
    K[:3, 0] = -M[:3, 1]
    K[:3, 1] = M[:3, 2]
    K[:3, 2] = -M[:3, 0]
    K[:3, 3] = K[:3, 3] / 1000.0
    return K


def transform_surface_model(surface_model_filepath: str,
                            transform_filepath: str,
                            output_filepath: str,
                            structure: str) -> None:
    """Apply a 4×4 affine to a GIFTI surface model using Workbench."""
    os.system(
        f"'{wb_command_cmd}' -logging OFF -surface-apply-affine '{surface_model_filepath}' '{transform_filepath}' '{output_filepath}'"
    )
    add_structure_information(output_filepath, structure)


def create_kps_file_for_kPlan(position_matrix_filepath: str, kps_filename: str) -> None:
    """Create a *.kps file (HDF5) for k-Plan from a MATLAB .mat pose matrix."""
    import numpy as np
    import scipy
    import h5py

    out_dir, _ = os.path.split(position_matrix_filepath)
    out_file = os.path.join(out_dir, f"{kps_filename}.kps")

    mat = scipy.io.loadmat(position_matrix_filepath)
    M = mat["position_matrix"].T.reshape((1, 4, 4)).astype("float32")

    with h5py.File(out_file, "w") as f:
        dset = f.create_dataset("/1/position_transform", (1, 4, 4), dtype="float32")
        dset[:] = M
        f["/1"].attrs.create("transform_label", np.string_("Localite transducer position"))
        f.attrs.create("application_name", np.string_("k-Plan"))
        f.attrs.create("file_type", np.string_("k-Plan Transducer Position"))
        f.attrs.create("number_transforms", np.array([1], dtype=np.uint64))


# -----------------------------------------------------------------------------
# Focus/ellipsoid/transducer geometry writers
# -----------------------------------------------------------------------------

def create_surface_ellipsoid(length: float, width: float,
                             position_transform_filepath: str,
                             reference_volume_filepath: str,
                             output_filepath: str) -> None:
    """Write an ellipsoidal surface (*.surf.gii) and apply an affine.

    """
    from nibabel.gifti import GiftiImage, GiftiDataArray

    out_dir, _ = os.path.split(output_filepath)

    ref = nib.load(reference_volume_filepath)
    shape = ref.shape[:3]
    pixdim = ref.header["pixdim"][1:4]

    # radii in mm
    a = float(shape[0]) * float(pixdim[0]) * (float(width) / (float(shape[0]) * float(pixdim[0]))) / 2
    b = float(shape[1]) * float(pixdim[1]) * (float(width) / (float(shape[0]) * float(pixdim[0]))) / 2
    c = float(shape[2]) * float(pixdim[2]) * (float(length) / (float(shape[2]) * float(pixdim[2]))) / 2

    phi = np.linspace(0, 2 * np.pi, 50)
    theta = np.linspace(0, np.pi, 50)
    phi, theta = np.meshgrid(phi, theta)

    x = a * np.sin(theta) * np.cos(phi)
    y = b * np.sin(theta) * np.sin(phi)
    z = c * np.cos(theta)

    points = np.vstack((x.flatten(), y.flatten(), z.flatten())).T.astype(np.float32)
    faces = []
    nr, nc = x.shape
    for i in range(nr - 1):
        for j in range(nc - 1):
            faces.append([i * nc + j, i * nc + (j + 1), (i + 1) * nc + j])
            faces.append([(i + 1) * nc + j, i * nc + (j + 1), (i + 1) * nc + (j + 1)])
    faces = np.array(faces, dtype=np.int32)

    gii = GiftiImage(darrays=[
        GiftiDataArray(data=points, intent="NIFTI_INTENT_POINTSET"),
        GiftiDataArray(data=faces, intent="NIFTI_INTENT_TRIANGLE"),
    ])
    tmp = os.path.join(out_dir, "ellipsoid_tmp.surf.gii")
    nib.save(gii, tmp)

    # apply transform + set structure
    transform_surface_model(tmp, position_transform_filepath, output_filepath, "CORTEX_RIGHT")
    try:
        os.remove(tmp)
    except OSError:
        pass


def create_volume_ellipsoid(length: float, width: float,
                             position_transform_filepath: str,
                             reference_volume_filepath: str,
                             output_filepath: str) -> None:
    """Alias to :func:`create_surface_ellipsoid` (API compatibility)."""
    create_surface_ellipsoid(length, width, position_transform_filepath, reference_volume_filepath, output_filepath)


def create_surface_transducer_model(radius: float, height: float, output_filepath: str) -> None:
    """Create a simple cylindrical transducer surface (*.surf.gii)."""
    from nibabel.gifti import GiftiImage, GiftiDataArray

    out_dir, _ = os.path.split(output_filepath)
    num = 100
    theta = np.linspace(0, 2 * np.pi, num)
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    z = np.zeros_like(x)

    bottom = np.vstack((x, y, z)).T.astype(np.float32)
    top = np.vstack((x, y, z + height)).T.astype(np.float32)
    points = np.vstack((bottom, top))

    faces = []
    for i in range(num - 1):
        faces.append([i, i + 1, num + i])
        faces.append([i + 1, num + i + 1, num + i])
    faces.append([num - 1, 0, 2 * num - 1])
    faces.append([0, num, 2 * num - 1])
    for i in range(1, num - 1):
        faces.append([0, i, i + 1])
        faces.append([num, num + i, num + i + 1])
    faces = np.array(faces, dtype=np.int32)

    gii = GiftiImage(darrays=[
        GiftiDataArray(data=points, intent='NIFTI_INTENT_POINTSET'),
        GiftiDataArray(data=faces, intent='NIFTI_INTENT_TRIANGLE'),
    ])
    nib.save(gii, os.path.join(out_dir, "transducer_tmp.surf.gii"))
    shutil.move(os.path.join(out_dir, "transducer_tmp.surf.gii"), output_filepath)


# -----------------------------------------------------------------------------
# k-Plan result export & planning helpers (extended from original file)
# -----------------------------------------------------------------------------

def compute_FLHM_for_focal_distance(focal_distance: float,
                                    focal_distance_list: Sequence[float],
                                    flhm_list: Sequence[float]) -> float:
    """Compute expected FLHM from a calibration curve (cubic fit)."""
    if len(focal_distance_list) == 1 or len(flhm_list) == 1: # in the case of single-element transducers

        FLHM = float(flhm_list[0])

    else: # in the case of steerable transducers
        import pandas as pd
        from scipy.optimize import curve_fit

        df = pd.DataFrame({'Focal Distance': focal_distance_list, 'FLHM': flhm_list})

        def cubic(x, a, b, c, d):
            return a + b * x + c * (x ** 2) + d * (x ** 3)

        pars, _ = curve_fit(f=cubic,
                            xdata=df["Focal Distance"],
                            ydata=df["FLHM"],
                            p0=[0, 0, 0, 0],
                            bounds=[-np.inf, np.inf])

        FLHM = float(pars[0] + pars[1] * focal_distance + pars[2] * focal_distance ** 2 + pars[3] * focal_distance ** 3)

    return FLHM


def kPlan_results_to_nifti(h5_filepath: str, CT_filepath: str) -> None:
    """Convert k-Plan HDF5 outputs (mask, pressure, thermal) into T1/pCT space NIfTI."""
    import h5py
    import ants

    h5_path = os.path.split(h5_filepath)[0]
    h5_name = os.path.split(h5_filepath)[1].replace('.h5', '')

    with h5py.File(h5_filepath, 'r') as results:
        medium_mask = results["medium_properties/medium_mask"][:]
        medium_mask_grid_spacing = results["medium_properties/medium_mask"].attrs["grid_spacing"]
        sonications = results["sonications"]

        def get_affine(diagonal_element: float) -> np.ndarray:
            A = np.zeros((4, 4), dtype=float)
            A[0, 0] = diagonal_element
            A[1, 1] = diagonal_element
            A[2, 2] = diagonal_element
            A[3, 3] = 1.0
            return A

        affine = get_affine(float(medium_mask_grid_spacing[0]) * 1000.0)

        # convert base mask once
        medium_mask_image = nib.Nifti1Image(np.transpose(medium_mask), affine=affine)

        for i in range(len(sonications)):
            pa = results[f"sonications/{i+1}/simulated_field/pressure_amplitude"][:]
            td = results[f"sonications/{i+1}/simulated_field/thermal_dose"][:]

            pressure_amplitude_image = nib.Nifti1Image(np.transpose(pa), affine=affine)
            thermal_dose_image = nib.Nifti1Image(np.transpose(td), affine=affine)

            # ANTs: register medium_mask to CT, then apply to all
            CT_ants = ants.image_read(CT_filepath)
            mm_ants = ants.from_nibabel(medium_mask_image)
            pa_ants = ants.from_nibabel(pressure_amplitude_image)
            td_ants = ants.from_nibabel(thermal_dose_image)

            reg = ants.registration(fixed=CT_ants,
                                    moving=mm_ants,
                                    type_of_transform='TRSAA',
                                    reg_iterations=[100, 3, 4, 5, 6])

            mm_tx = ants.apply_transforms(fixed=CT_ants, moving=mm_ants,
                                          transformlist=reg['fwdtransforms'],
                                          interpolator='linear')
            pa_tx = ants.apply_transforms(fixed=CT_ants, moving=pa_ants,
                                          transformlist=reg['fwdtransforms'],
                                          interpolator='linear')
            td_tx = ants.apply_transforms(fixed=CT_ants, moving=td_ants,
                                          transformlist=reg['fwdtransforms'],
                                          interpolator='linear')

            nib.save(ants.to_nibabel(mm_tx), os.path.join(h5_path, f"{h5_name}_MediumMask_Sonication{i+1}.nii.gz"))
            nib.save(ants.to_nibabel(pa_tx), os.path.join(h5_path, f"{h5_name}_AcousticPressure_Sonication{i+1}.nii.gz"))
            nib.save(ants.to_nibabel(td_tx), os.path.join(h5_path, f"{h5_name}_ThermalDose_Sonication{i+1}.nii.gz"))


def prepare_acoustic_simulation(vertex_number: int,
                                output_path: str,
                                target_roi_filepath: str,
                                t1_filepath: str,
                                max_distance: float,
                                min_distance: float,
                                transducer_diameter: float,
                                max_angle: float,
                                offset: float,
                                additional_offset: float,
                                transducer_surface_model_filepath: str,
                                focal_distance_list: Sequence[float],
                                flhm_list: Sequence[float],
                                placement_scene_template_filepath: str,
                                ID="",
                                skip_wb_view=False,
                                use_internal_viewer=False) -> None:
    """End-to-end preparation for one candidate vertex (simulation folder)."""
    import scipy
    from nilearn import image

    if len(ID)==0:
        suffix="vtx" + str(vertex_number)
    else:
        suffix = ID
    output_path_vtx = os.path.join(output_path, suffix)
    os.makedirs(output_path_vtx, exist_ok=True)

    target_roi_filename = os.path.split(target_roi_filepath)[1]
    target_roi_name = target_roi_filename.replace(".nii", "").replace(".gz", "")

    # --- Surfaces and vectors
    skin_coordinates, skin_normals = compute_surface_metrics(os.path.join(output_path, "skin.surf.gii"))
    target_center = roi_center_of_gravity(target_roi_filepath)
    skin_target_vectors = vectors_between_surface_and_point(os.path.join(output_path, "skin.surf.gii"), target_center)

    skin_target_intersections = compute_vector_mesh_intersections(
        skin_coordinates, skin_normals, os.path.join(output_path, f"{target_roi_name}_3Dmodel.stl"), 200
    )
    skin_target_intersection_values = []
    for inter in skin_target_intersections:
        if len(inter) == 1:
            skin_target_intersection_values.append(0)
        elif len(inter) == 2:
            d = np.linalg.norm(np.asarray(inter[1]) - np.asarray(inter[0]))
            skin_target_intersection_values.append(d)
        elif len(inter) == 3:
            d = np.linalg.norm(np.asarray(inter[1]) - np.asarray(inter[0]))
            skin_target_intersection_values.append(d)
        elif len(inter) == 4:
            d = (np.linalg.norm(np.asarray(inter[1]) - np.asarray(inter[0])) +
                 np.linalg.norm(np.asarray(inter[3]) - np.asarray(inter[2])))
            skin_target_intersection_values.append(d)
        elif len(inter) > 4:
            skin_target_intersection_values.append(np.nan)
        else:
            skin_target_intersection_values.append(0)
    skin_target_intersection_values = np.asarray(skin_target_intersection_values)

    # --- Vertex of interest
    vertex_coordinates = skin_coordinates[vertex_number]
    if skin_target_intersection_values[vertex_number] == 0:
        vertex_vector = -skin_target_vectors[vertex_number]
    else:
        vertex_vector = -skin_normals[vertex_number]  # inward
    vertex_vector = unit_vector(vertex_vector)

    # --- Localite pose for the transducer
    transducer_center_coordinates = vertex_coordinates - ((offset + additional_offset) * vertex_vector)
    position_matrix_Localite = create_Localite_position_matrix(transducer_center_coordinates, vertex_vector)

    scipy.io.savemat(os.path.join(output_path_vtx, f"position_matrix_{target_roi_name}_{suffix}_Localite.mat"),
                     {'position_matrix': position_matrix_Localite})
    np.savetxt(os.path.join(output_path_vtx, f"position_matrix_{target_roi_name}_{suffix}_Localite.txt"),
               position_matrix_Localite)

    xml = create_fake_XML_structure_for_Localite(position_matrix_Localite,
                                                 f"transducer_position_{target_roi_name}_{suffix}", 0, 0)
    with open(os.path.join(output_path_vtx, f"position_matrix_{target_roi_name}_{suffix}_Localite_XML.txt"), "a") as f:
        f.write(xml)

    # --- Convert to k-Plan
    position_matrix_kPlan = convert_Localite_to_kPlan_position_matrix(position_matrix_Localite)
    scipy.io.savemat(os.path.join(output_path_vtx, f"position_matrix_{target_roi_name}_{suffix}_kPlan.mat"),
                     {'position_matrix': position_matrix_kPlan})
    np.savetxt(os.path.join(output_path_vtx, f"position_matrix_{target_roi_name}_{suffix}_kPlan.txt"),
               position_matrix_kPlan)

    # --- Optional: transform transducer model
    transform = np.loadtxt(os.path.join(output_path_vtx, f"position_matrix_{target_roi_name}_{suffix}_kPlan.txt"))
    transform[0:3, 3] = transform[0:3, 3] * 1000  # back to mm for Workbench affine
    transform_filepath = os.path.join(output_path_vtx, f"position_matrix_{target_roi_name}_{suffix}_transducer.txt")
    np.savetxt(transform_filepath, transform)

    transducer_out = os.path.join(output_path_vtx, f"transducer_{target_roi_name}_{suffix}.surf.gii")
    if transducer_surface_model_filepath:
        transform_surface_model(transducer_surface_model_filepath, transform_filepath, transducer_out, "CEREBELLUM")
    else:
        create_surface_transducer_model(transducer_diameter / 2, 15, transducer_out)
        transform_surface_model(transducer_out, transform_filepath, transducer_out, "CEREBELLUM")

    # --- .kps for k-Plan
    create_kps_file_for_kPlan(
        os.path.join(output_path_vtx, f"position_matrix_{target_roi_name}_{suffix}_kPlan.mat"),
        f"{target_roi_name}_{suffix}"
    )

    # --- Focal distance & FLHM
    if skin_target_intersection_values[vertex_number] == 0:
        target_center = roi_center_of_gravity(target_roi_filepath)
        skin_target_distances = distance_between_surface_and_point(os.path.join(output_path, "skin.surf.gii"), target_center)
        focal_distance = float(skin_target_distances[vertex_number] + additional_offset)
    else:
        inter_center = (np.asarray(skin_target_intersections[vertex_number][0]) +
                        np.asarray(skin_target_intersections[vertex_number][1])) / 2.0
        focal_distance = float(np.linalg.norm(skin_coordinates[vertex_number] - inter_center) + additional_offset)

    focal_distance = max(min(focal_distance, max_distance), min_distance)
    FLHM = compute_FLHM_for_focal_distance(focal_distance, focal_distance_list, flhm_list)

    # --- Ellipsoid (surface)
    focus_transform = np.loadtxt(os.path.join(output_path_vtx, f"position_matrix_{target_roi_name}_{suffix}_kPlan.txt"))
    focus_transform[0:3, 3] = focus_transform[0:3, 3] * 1000
    focus_transform[0:3, 3] = focus_transform[0:3, 3] + (vertex_vector * (offset + focal_distance))

    focus_transform_path = os.path.join(output_path_vtx, f"focus_position_matrix_{target_roi_name}_{suffix}.txt")
    np.savetxt(focus_transform_path, focus_transform)

    ellipsoid_surf = os.path.join(output_path_vtx, f"focus_{target_roi_name}_{suffix}_{round(focal_distance,1)}.surf.gii")
    create_surface_ellipsoid(FLHM, 5, focus_transform_path, t1_filepath, ellipsoid_surf)

    # --- Ellipsoid (volume) via Workbench ribbon mapping
    ellipsoid_vol = os.path.join(output_path_vtx, f"focus_{target_roi_name}_{suffix}_{round(focal_distance,1)}.nii.gz")
    ellipsoid_small = os.path.join(output_path_vtx, f"focus_{target_roi_name}_{suffix}_{round(focal_distance,1)}_small.surf.gii")
    create_surface_ellipsoid(FLHM - 1, 4, focus_transform_path, t1_filepath, ellipsoid_small)

    focus_metric = os.path.join(output_path_vtx, "focus.func.gii")
    os.system(f"'{wb_command_cmd}' -logging OFF -surface-coordinates-to-metric '{ellipsoid_surf}' '{focus_metric}'")
    os.system("'{}' -logging OFF -metric-to-volume-mapping '{}' '{}' '{}' '{}' -ribbon-constrained '{}' '{}'".format(
        wb_command_cmd, focus_metric, ellipsoid_surf, t1_filepath, ellipsoid_vol, ellipsoid_small, ellipsoid_surf
    ))
    os.system(f"'{wb_command_cmd}' -logging OFF -volume-fill-holes '{ellipsoid_vol}' '{ellipsoid_vol}'")

    # binarize first frame
    ell = image.load_img(ellipsoid_vol)
    ell_d = ell.get_fdata()[:, :, :, 0]
    ell_d[ell_d > 0] = 1
    nib.save(nib.Nifti1Image(ell_d, ell.affine), ellipsoid_vol)
    try:
        os.remove(focus_metric)
        os.remove(ellipsoid_small)
    except OSError:
        pass

    # --- Visualize results
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
        gifti_files.append(transducer_out)

        widget = FinalResultViewer(gifti_files)
        layout.addWidget(widget)
        DlgResults.resize(600, 600)
        DlgResults.exec()
    else:
        scene_variable_names = [
            'SKIN_SURFACE_FILENAME', 'SKIN_SURFACE_FILEPATH',
            'T1_FILENAME', 'T1_FILEPATH',
            'MASK_FILENAME', 'MASK_FILEPATH',
            'TRANSDUCER_SURFACE_FILENAME', 'TRANSDUCER_SURFACE_FILEPATH',
            'FOCUS_VOLUME_FILENAME', 'FOCUS_VOLUME_FILEPATH',
            'FOCUS_SURFACE_FILENAME', 'FOCUS_SURFACE_FILEPATH'
        ]

        scene_variable_values = [
            'skin.surf.gii', '../skin.surf.gii',
            'T1.nii.gz', '../../../T1.nii.gz',
            target_roi_filename, f"../{target_roi_filename}",
            f"transducer_{target_roi_name}_{suffix}.surf.gii", f"./transducer_{target_roi_name}_{suffix}.surf.gii",
            f"focus_{target_roi_name}_{suffix}_{round(focal_distance,1)}.nii.gz",
            f"./focus_{target_roi_name}_{suffix}_{round(focal_distance,1)}.nii.gz",
            f"focus_{target_roi_name}_{suffix}_{round(focal_distance,1)}.surf.gii",
            f"./focus_{target_roi_name}_{suffix}_{round(focal_distance,1)}.surf.gii"
        ]

        create_scene(placement_scene_template_filepath,
                     os.path.join(output_path_vtx, "scene.scene"),
                     scene_variable_names,
                     scene_variable_values)

        # Launch viewer (optional)
        os.system(f"wb_view -logging OFF {os.path.join(output_path_vtx, 'scene.scene')} &")
