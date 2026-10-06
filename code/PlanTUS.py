#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PlanTUS core functions
======================

Dependencies
------------------
- SimNIBS
- potpourri3d (optional — exact geodesic distances for erode_metric();
  falls back to an approximate edge-graph method if not installed)

"""

from __future__ import annotations

import os
import re
import glob
import shutil
import subprocess
from typing import List, Optional, Sequence, Tuple

import numpy as np
import nibabel as nib
import simnibs


# -----------------------------------------------------------------------------
# Basic NIfTI ops
# -----------------------------------------------------------------------------

def _load_nii(path: str) -> Tuple[nib.Nifti1Header, np.ndarray, np.ndarray]:
    """Load a NIfTI file as float64 array.

    Squeezes a trailing singleton 4th dimension if present (some tools,
    including SimNIBS' charm, write ``final_tissues.nii.gz`` as
    (X, Y, Z, 1) rather than plain (X, Y, Z)) so every downstream helper
    built on this function works with a consistent 3D shape.

    Returns
    -------
    header : nib.Nifti1Header
    data : (X,Y,Z) ndarray float64
    affine : (4,4) ndarray
    """
    img = nib.load(path)
    data = img.get_fdata()  # float64
    if data.ndim == 4 and data.shape[-1] == 1:
        data = data[..., 0]
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


def _seal_cropped_fov_faces(vol: np.ndarray, threshold: float = 0.05):
    """Seal (set to True/tissue) any image face where a meaningful
    fraction of voxels are already tissue — a strong signal that the
    field of view is cropped through anatomy at that face (e.g. a T1
    that doesn't extend far past the neck), rather than ending in
    genuinely open exterior space around the head.

    Why this matters: air-filled cavities (sinuses, etc.) are
    anatomically connected to the outside via the nasal/oral airway.
    ``scipy.ndimage.binary_fill_holes`` only fills background regions
    that are NOT reachable from the image border — so if that airway
    happens to reach a cropped face (a real risk whenever the FOV ends
    partway down the neck, since the airway runs right through there),
    the entire connected cavity system escapes detection as a "hole",
    even though it's a genuine anatomical cavity that should be found.
    Confirmed directly on real data: one face showed ~49% tissue
    (a crop straight through the neck) while the other five showed
    ~0% (genuinely open exterior) — sealing just that one face let
    binary_fill_holes correctly capture the ~100,000-voxel connected
    sinus/airway system that was otherwise being missed entirely.

    A genuinely open exterior face should have ~0% tissue touching it
    (there's empty space between the head and the edge of the FOV);
    threshold=0.05 leaves comfortable margin above that while still
    catching a crop through anatomy, which tends to show a much larger
    fraction (near-symmetric cuts through a mix of tissue/air, as seen
    on the real data above, showed ~49%).

    Returns the sealed volume and the list of face index-tuples that
    were sealed (so the caller can undo the seal in that specific
    region afterward — this is only meant to influence connectivity
    for hole-detection, not to actually add tissue there).
    """
    sealed = vol.copy()
    sealed_faces = []
    for axis in range(3):
        for sl in (slice(0, 1), slice(-1, None)):
            idx = [slice(None)] * 3
            idx[axis] = sl
            idx = tuple(idx)
            if vol[idx].mean() > threshold:
                sealed[idx] = True
                sealed_faces.append(idx)
    return sealed, sealed_faces


def _fill_holes_axial(vol: np.ndarray, affine: np.ndarray) -> np.ndarray:
    """Fill holes slice by slice (2-D) along the voxel axis that runs
    closest to world superior-inferior (found from the affine, so this is
    orientation-agnostic: it works for LAS/RAS, FreeSurfer-style LIA,
    oblique volumes, ...).

    Why: the paranasal sinuses, nasal cavity and pharynx form one air
    system that is open to the outside (nostrils, and often the airway
    running out of the bottom of the segmented neck). In 3-D that makes
    them part of the exterior, so ``binary_fill_holes`` on the volume
    never treats them as holes — confirmed on a SimNIBS example subject
    (ernie), where the 3-D fill recovered only ~8 mL of air (essentially
    just the mastoids) while ~99 mL of cavity was present, leaving the
    skin over the frontal sinuses unmarked. In an axial slice, however,
    the same cavities are enclosed by tissue all around (above the
    nostril level), so a 2-D fill captures them.
    """
    from scipy import ndimage

    axis = int(np.argmax(np.abs(affine[2, :3])))
    out = np.zeros_like(vol, dtype=bool)
    for k in range(vol.shape[axis]):
        idx = [slice(None)] * 3
        idx[axis] = k
        idx = tuple(idx)
        out[idx] = ndimage.binary_fill_holes(vol[idx])
    return out


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
        datatype=nib.nifti1.data_type_codes['NIFTI_TYPE_INT32'],
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
                                     output_path: str,
                                     smoothing_iterations: int = 0) -> None:
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
    smoothing_iterations : int, optional
        Number of Taubin-smoothing iterations to apply to the surface
        before writing it out (default 0 = no smoothing, preserving
        the raw SimNIBS-extracted geometry exactly as before). Uses
        the same trimesh.smoothing.filter_taubin call (lamb=0.5,
        nu=0.5) already used elsewhere in this module (see
        stl_from_nii) — Taubin rather than plain Laplacian smoothing,
        since Laplacian shrinks/drifts small or coarsely-triangulated
        meshes, which Taubin's alternating pos/neg steps largely
        avoid. A small number of iterations (a handful, not the 10
        used for the much smaller/coarser air-cavity or ROI meshes
        elsewhere) is enough to knock down segmentation-level mesh
        noise on a surface this size without eroding real anatomical
        detail (nose, ears, eye sockets).

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

    if smoothing_iterations > 0:
        import trimesh
        tmesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        trimesh.smoothing.filter_taubin(tmesh, lamb=0.5, nu=0.5, iterations=smoothing_iterations)
        vertices = np.asarray(tmesh.vertices, dtype=np.float32)
        faces = np.asarray(tmesh.faces, dtype=np.int32)

    # write gifti
    _write_gifti_surface(vertices, faces, gii_out)

    # Write STL directly ------------------------------------------------------
    stl_out = os.path.join(output_path, f"{mesh_name}.stl")
    if smoothing_iterations > 0:
        # Keep the STL consistent with the (now smoothed) GIFTI above,
        # rather than writing the original unsmoothed mesh.
        tmesh.export(stl_out)
    else:
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

def _load_gifti_mesh(surface_filepath: str):
    """Load a .surf.gii as (vertices, faces), and as a trimesh.Trimesh.

    Pure nibabel + trimesh — no Workbench involved. Assumes the standard
    two-darray GIFTI surface layout (pointset, then triangle list), which
    is what SimNIBS/charm, PlanTUS' own writers, and Workbench all use.
    """
    import trimesh

    gii = nib.load(surface_filepath)
    vertices = np.asarray(gii.darrays[0].data, dtype=np.float64)
    faces = np.asarray(gii.darrays[1].data, dtype=np.int64)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    return vertices, faces, mesh


def fraction_of_points_inside_volume(points_mm: np.ndarray, volume_filepath: str,
                                     margin_mm: float = 5.0) -> Tuple[float, np.ndarray, np.ndarray]:
    """Fraction of `points_mm` (N,3, world mm) lying inside the world-space
    bounding box of a NIfTI volume (grown by `margin_mm`), plus that box's
    (min, max) corners. Uses nibabel's affine — the same convention the
    SimNIBS mesh, the ROI and all PlanTUS volume outputs use."""
    img = nib.load(volume_filepath)
    n = np.array(img.shape[:3]) - 1
    corners = np.array([[x, y, z] for x in (0, n[0]) for y in (0, n[1]) for z in (0, n[2])])
    world = nib.affines.apply_affine(img.affine, corners)
    lo, hi = world.min(axis=0) - margin_mm, world.max(axis=0) + margin_mm
    inside = np.all((points_mm >= lo) & (points_mm <= hi), axis=1)
    return float(inside.mean()), lo, hi


def check_t1_mesh_consistency(t1_filepath: str, skin_surf_filepath: str,
                              min_fraction: float = 0.95) -> bool:
    """Warn (never abort) if the skin surface doesn't lie inside the T1's
    world-space extent — i.e. the T1 and the SimNIBS mesh don't appear to
    share a coordinate space (typically: the T1 passed isn't the one charm
    was run on). Everything built on the T1 grid (slice views, focus
    volumes, scene files) would then be misplaced relative to the mesh.

    A skin surface derived from this T1 lies inside its FOV by construction
    (up to a voxel at a cropped edge, hence the margin); a genuine mismatch
    is typically tens of mm off, far beyond that.
    """
    vertices = np.asarray(nib.load(skin_surf_filepath).darrays[0].data, dtype=np.float64)
    frac, lo, hi = fraction_of_points_inside_volume(vertices, t1_filepath)
    if frac >= min_fraction:
        return True
    print("\n" + "!" * 78)
    print(f"WARNING: only {frac*100:.0f}% of the skin surface lies inside the T1's world-space extent.")
    print(f"  T1 extent (mm):   x[{lo[0]:.0f},{hi[0]:.0f}] y[{lo[1]:.0f},{hi[1]:.0f}] z[{lo[2]:.0f},{hi[2]:.0f}]")
    print(f"  skin extent (mm): x[{vertices[:,0].min():.0f},{vertices[:,0].max():.0f}] "
          f"y[{vertices[:,1].min():.0f},{vertices[:,1].max():.0f}] z[{vertices[:,2].min():.0f},{vertices[:,2].max():.0f}]")
    print("  The T1 and the SimNIBS mesh do not appear to share a coordinate space, so slice")
    print("  views and T1-grid outputs will be misaligned. Use the same T1 that charm was run")
    print("  on (e.g. m2m_<subject>/T1.nii.gz).")
    print("!" * 78 + "\n")
    return False


def _write_functional_gifti(values: np.ndarray, out_path: str, structure: Optional[str] = None) -> None:
    """Write a per-vertex scalar array as a .func.gii metric.

    Pure nibabel — replaces the old NIfTI-detour + ``-metric-convert``
    round trip through Workbench. One GiftiDataArray, one column per
    vertex, matching what Viewer.py / Workbench expect on read
    (``func.darrays[0].data``).
    """
    from nibabel.gifti import GiftiImage, GiftiDataArray

    values = np.ascontiguousarray(np.asarray(values, dtype=np.float32).ravel())
    da = GiftiDataArray(
        data=values,
        intent="NIFTI_INTENT_NONE",
        datatype=nib.nifti1.data_type_codes['NIFTI_TYPE_FLOAT32'],
    )
    gii = GiftiImage(darrays=[da])
    if structure is not None:
        gii.meta['AnatomicalStructurePrimary'] = structure
    nib.save(gii, out_path)


_surface_metrics_cache = {}  # (abspath, mtime) -> (coords, normals)


def compute_surface_metrics(surface_filepath: str) -> Tuple[np.ndarray, np.ndarray]:
    """Compute per-vertex coordinates and normals for a GIFTI surface.

    Pure nibabel + trimesh — replaces the old
    ``-surface-coordinates-to-metric`` / ``-surface-normals`` Workbench
    round trip. Normals are trimesh's angle-weighted vertex normals; they
    won't be bit-identical to Workbench's algorithm but are equivalent
    for the ray-casting / angle checks PlanTUS uses them for.

    Results are cached in-process keyed by (file path, file mtime) —
    this gets called repeatedly on the same skin/skull surfaces across
    a single run (multiple metrics in the wrapper, plus once more at
    trajectory-generation time), and there's no reason to re-read the
    GIFTI and recompute trimesh normals from scratch each time. Cache
    is automatically invalidated if the underlying file changes.

    Returns
    -------
    coordinates : (N, 3) float
    normals : (N, 3) float
    """
    cache_key = (os.path.abspath(surface_filepath), os.path.getmtime(surface_filepath))
    cached = _surface_metrics_cache.get(cache_key)
    if cached is not None:
        return cached

    vertices, _, mesh = _load_gifti_mesh(surface_filepath)
    coords = np.asarray(vertices, dtype=float)
    norms = np.asarray(mesh.vertex_normals, dtype=float)
    _surface_metrics_cache[cache_key] = (coords, norms)
    return coords, norms


def create_metric_from_pseudo_nifti(metric_name: str,
                                    metric_values: Sequence[float],
                                    surface_filepath: str) -> None:
    """Save per-vertex values as a ``.func.gii`` metric for a surface.

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
    * ``{metric_name}_{surface_name}.func.gii``

    Pure nibabel — replaces the old NIfTI-proxy + ``-metric-convert``
    Workbench round trip. Kept the same name/signature so every call
    site in PlanTUS_wrapper.py needs no changes.
    """
    out_dir, fname = os.path.split(surface_filepath)
    surface_name = fname.replace(".surf.gii", "")
    out_path = os.path.join(out_dir, f"{metric_name}_{surface_name}.func.gii")

    values = np.asarray(metric_values, dtype=np.float32)
    n_verts = nib.load(surface_filepath).darrays[0].data.shape[0]
    if values.shape[0] != n_verts:
        raise ValueError(f"Number of metric values ({values.shape[0]}) does not match vertices ({n_verts}).")

    _write_functional_gifti(values, out_path)


def erode_metric(metric_filepath: str, surface_filepath: str, erosion_factor: float) -> None:
    """Erode a positive-valued metric/mask across the surface, in-place.

    Replaces Workbench's ``-metric-erode``. Computes true mesh geodesic
    distance from every *non-mask* (boundary) vertex using the heat
    method (Crane, Weischedel & Wardetzky, 2013) via ``potpourri3d``,
    and zeroes out mask vertices whose distance to the nearest boundary
    vertex is smaller than ``erosion_factor`` (mm) — the surface
    equivalent of a morphological erosion by that many mm.

    Falls back to an edge-graph Dijkstra approximation (shortest paths
    constrained to mesh edges) if ``potpourri3d`` isn't installed. That
    fallback is a valid approximation but systematically under-erodes by
    a small, growing-with-distance amount, since it can't cut diagonally
    across a triangle face the way a true geodesic can.
    Install the exact solver with: ``pip install potpourri3d``.
    """
    _, faces, mesh = _load_gifti_mesh(surface_filepath)

    values = np.asarray(nib.load(metric_filepath).darrays[0].data, dtype=np.float64)
    mask = values > 0

    if not mask.any() or mask.all():
        # Nothing to erode against (empty or whole-surface mask) — leave as is.
        _write_functional_gifti(values.astype(np.float32), metric_filepath)
        return

    # Boundary seeds = vertices immediately adjacent to the mask/non-mask
    # interface, taken from BOTH sides, so the source set brackets the
    # true continuous boundary rather than sitting systematically inside
    # it by up to one mesh edge. Using *every* exterior vertex as a
    # source (thousands of points, densely covering most of the mesh)
    # is both unnecessary and numerically unstable for the heat method
    # below — it makes the injected heat field nearly uniform almost
    # everywhere, which degrades the gradient it relies on.
    neighbors = mesh.vertex_neighbors
    inside_ring = set()
    for v in np.where(mask)[0]:
        for nb in neighbors[v]:
            if not mask[nb]:
                inside_ring.add(v)
                break
    outside_ring = {nb for v in inside_ring for nb in neighbors[v] if not mask[nb]}
    boundary_sources = np.asarray(sorted(inside_ring | outside_ring), dtype=np.int64)

    try:
        import potpourri3d as pp3d
        solver = pp3d.MeshHeatMethodDistanceSolver(
            np.asarray(mesh.vertices, dtype=np.float64),
            np.asarray(mesh.faces, dtype=np.int32),
        )
        dist_to_boundary = solver.compute_distance_multisource(boundary_sources.tolist())
        # Heat method is a numerical approximation; can be slightly
        # negative right at/near source vertices — clip for safety.
        dist_to_boundary = np.clip(dist_to_boundary, 0.0, None)
    except ImportError:
        import warnings
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import dijkstra

        warnings.warn(
            "potpourri3d not installed — erode_metric() is falling back to "
            "an edge-graph Dijkstra approximation, which slightly "
            "under-erodes relative to true geodesic distance. "
            "Install with `pip install potpourri3d` for exact results.",
            stacklevel=2,
        )

        n = len(mesh.vertices)
        edges = mesh.edges_unique
        lengths = mesh.edges_unique_length
        graph = csr_matrix(
            (np.concatenate([lengths, lengths]),
             (np.concatenate([edges[:, 0], edges[:, 1]]),
              np.concatenate([edges[:, 1], edges[:, 0]]))),
            shape=(n, n),
        )
        dist_to_boundary = dijkstra(graph, indices=boundary_sources, min_only=True)

    values_out = values.copy()
    values_out[mask & (dist_to_boundary < float(erosion_factor))] = 0

    _write_functional_gifti(values_out.astype(np.float32), metric_filepath)


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
        For each ray a list of hit points (x,y,z), ordered by distance
        from the ray origin, with coincident hits (same surface point
        reported more than once, e.g. at a shared triangle edge/vertex or
        when the origin lies on the mesh) merged into one.

    Notes
    -----
    Uses vtkStaticCellLocator rather than vtkOBBTree. vtkOBBTree's
    IntersectWithLine silently drops a large share of intersections on
    VTK 9.7.x (measured on a SimNIBS skull mesh: ~27% of skin->skull rays
    reported a hit vs 72% with the other locators, and the count even
    varied from run to run on identical input), and it also misses the
    occasional hit on older versions (cross-checked against trimesh's
    ray casting). Every ray-based metric here (target intersection, skull
    thickness, skin-skull angle, the air-cavity no-go mask, focal-distance
    estimates) goes through this function, so a silent undercount would
    corrupt all of them without any error. The static cell locator gave
    identical, trimesh-consistent results on VTK 9.3.1 through 9.7.1.
    """
    import vtk  # type: ignore

    mesh = load_stl(mesh_filepath)

    locator = vtk.vtkStaticCellLocator()
    locator.SetDataSet(mesh)
    locator.BuildLocator()

    dedup_tol = 1e-3  # mm; far below any real spacing between distinct surfaces

    hits: List[List[Tuple[float, float, float]]] = []
    for i in range(len(points)):
        p0 = points[i]
        p1 = points[i] - vector_length * vectors[i]
        pts = vtk.vtkPoints()
        locator.IntersectWithLine(p0, p1, 1e-6, pts, None)
        dat = pts.GetData()
        n = dat.GetNumberOfTuples()
        raw = [dat.GetTuple3(j) for j in range(n)]
        if n > 1:
            p0a = np.asarray(p0, dtype=float)
            raw.sort(key=lambda h: float(np.linalg.norm(np.asarray(h) - p0a)))
        ray_hits: List[Tuple[float, float, float]] = []
        for h in raw:
            if not ray_hits or np.linalg.norm(np.asarray(h) - np.asarray(ray_hits[-1])) > dedup_tol:
                ray_hits.append(h)
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
    2) Fill holes in the binary volume (pure scipy, no Workbench)
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

    # ---- 2) fill holes (pure scipy, no Workbench)
    filled_path = os.path.join(out_dir, "final_tissues_bin_filled.nii.gz")
    se_n = ndimage.generate_binary_structure(3, 1)
    img = nib.load(bin_path)
    vol = img.get_fdata()
    if len(vol.shape) == 4:
        vol = vol[:, :, :, 0]
    aff = img.affine
    vol_bool = vol.astype(bool)
    vol_sealed, sealed_faces = _seal_cropped_fov_faces(vol_bool)
    filled = ndimage.binary_fill_holes(vol_sealed, se_n)
    # Cavities open to the exterior in 3-D (sinuses/nasal cavity via the
    # nostrils/airway) are enclosed in axial slices — see _fill_holes_axial.
    filled |= _fill_holes_axial(vol_bool, aff)
    for idx in sealed_faces:
        # Undo the seal in the final result — it's only meant to
        # affect connectivity for hole-detection, not to actually add
        # tissue in that region.
        filled[idx] = vol_bool[idx]
    filled_out = nib.Nifti1Image(filled.astype(np.uint16), aff)
    nib.save(filled_out, filled_path)

    # ---- 3) get air-only mask = filled - bin
    air_path = os.path.join(out_dir, "final_tissues_air.nii.gz")
    subtract_nifti(filled_path, bin_path, air_path)

    # ---- 4) tessellate air mask → STL and GIFTI (for optional smoothing)
    mesh, verts, faces = _marching_cubes_from_binary_volume(air_path, isovalue=0.5)
    air_gii = os.path.join(out_dir, "final_tissues_air.surf.gii")
    air_stl = os.path.join(out_dir, "final_tissues_air.stl")
    _write_gifti_surface(verts, faces, air_gii)
    mesh_io.write_stl(mesh, air_stl)

    # Smoothing via trimesh Laplacian filter (lamb=0.5, 10 iterations —
    # matches the strength/iteration count of the old
    # ``-surface-smoothing`` Workbench call), then re-save the GIFTI/STL
    # in place so downstream steps see the smoothed geometry.
    import trimesh
    air_trimesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    trimesh.smoothing.filter_laplacian(air_trimesh, lamb=0.5, iterations=10)
    _write_gifti_surface(np.asarray(air_trimesh.vertices, dtype=np.float32),
                          np.asarray(air_trimesh.faces, dtype=np.int32),
                          air_gii)
    air_trimesh.export(air_stl)

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

    # Everything below halfway between the eyes and the head's own
    # lowest point. Previously: mean_eye_z / 2 — an absolute halving of
    # a coordinate value, which only gives a sensible "somewhere below
    # the eyes" cutoff if Z=0 happens to sit at some fixed anatomical
    # reference (e.g. the base of the neck) — not guaranteed by any
    # NIfTI/mesh coordinate convention. Confirmed on a real dataset
    # where the coordinate origin sits well above the eyes: halving
    # actually pushed the cutoff ABOVE eye-level (since mean_eye_z was
    # negative, halving it made it less negative), excluding 29% of the
    # head including a region above the eyes, rather than the intended
    # "below the eyes" region. This version is guaranteed to never sit
    # above eye-level (0 <= 0.5*(mean_eye_z - head_zmin), since eyes
    # are anatomically above the head's own lowest skin point) and is
    # relative to the head's own geometry rather than an absolute
    # coordinate value with no guaranteed anatomical meaning.
    head_zmin = skin_coords[:, 2].min()
    mean_eye_z = ((left_c + right_c) / 2)[2]
    eye_height_cutoff = mean_eye_z - 0.5 * (mean_eye_z - head_zmin)
    avoidance_mask[skin_coords[:, 2] <= eye_height_cutoff] = 0

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

    # 3) smooth via trimesh Taubin filtering (lamb=0.5, nu=0.5, 10
    #    iterations). Plain Laplacian smoothing (the old
    #    ``-surface-smoothing``-equivalent) causes small/coarsely-
    #    voxelized meshes — like a small subcortical ROI reconstructed
    #    at ~1mm resolution — to drift substantially off-center (up to
    #    ~8mm observed for a ~5.6mm-radius target), since a fixed
    #    iteration count is a disproportionately large amount of
    #    smoothing relative to a small object's own size. Taubin's
    #    alternating shrink/inflate passes cancel that drift almost
    #    entirely (confirmed: 0.00mm centroid shift vs 7.96mm for plain
    #    Laplacian on the same test mesh) while still removing
    #    voxelization/staircase artifacts, and volume is preserved to
    #    within a fraction of a percent.
    import trimesh
    tmesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    trimesh.smoothing.filter_taubin(tmesh, lamb=0.5, nu=0.5, iterations=10)

    gii_path = os.path.join(out_dir, base + ".surf.gii")
    stl_path = os.path.join(out_dir, base + ".stl")
    _write_gifti_surface(np.asarray(tmesh.vertices, dtype=np.float32),
                          np.asarray(tmesh.faces, dtype=np.int32),
                          gii_path)
    tmesh.export(stl_path)


    # cleanup temps
    for p in [thr_path, bin_path]:
        try:
            os.remove(p)
        except OSError:
            pass

    # keep the smoothed *.surf.gii as in the original pipeline


# -----------------------------------------------------------------------------
# Pure-Python replacements for former Workbench functionality
# -----------------------------------------------------------------------------


def mask_metric(metric_filepath: str, mask_filepath: str) -> None:
    """Zero out a metric wherever a mask is <= 0 (in-place).

    Pure nibabel — replaces Workbench's ``-metric-mask``.
    """
    values = np.asarray(nib.load(metric_filepath).darrays[0].data, dtype=np.float32)
    mask = np.asarray(nib.load(mask_filepath).darrays[0].data, dtype=np.float32)
    values = values * (mask > 0)
    _write_functional_gifti(values, metric_filepath)


def threshold_metric(metric_filepath: str, threshold: float) -> None:
    """Create a boolean (x < threshold) copy of a metric.

    Pure nibabel — replaces Workbench's ``-metric-math "x < threshold"``.
    Matches the original semantics exactly: output is 1.0 where the input
    value is below ``threshold``, 0.0 otherwise (not a value clip).
    """
    out_dir, fname = os.path.split(metric_filepath)
    name = fname.replace(".func.gii", "")
    out = os.path.join(out_dir, f"{name}_thresholded.func.gii")

    values = np.asarray(nib.load(metric_filepath).darrays[0].data, dtype=np.float32)
    result = (values < float(threshold)).astype(np.float32)
    _write_functional_gifti(result, out)


_STRUCTURE_LABEL_MAP = {
    # Maps the underscore-caps CLI-style labels used throughout this
    # codebase to the CamelCase strings Workbench actually writes into
    # GIFTI "AnatomicalStructurePrimary" metadata.
    "CORTEX_LEFT": "CortexLeft",
    "CORTEX_RIGHT": "CortexRight",
    "CEREBELLUM": "Cerebellum",
    "OTHER": "Other",
}


def add_structure_information(filepath: str, structure_label: str) -> None:
    """Annotate a surface or metric GIFTI with a structure label.

    Pure nibabel — replaces Workbench's ``-set-structure``. Sets the
    standard GIFTI "AnatomicalStructurePrimary" (and, for surfaces,
    "GeometricType") metadata *both* at the file level and on every
    individual DataArray. Workbench reads the per-DataArray metadata to
    determine a surface's Structure/Orientation — file-level metadata
    alone leaves it unable to identify the surface, which shows up as
    "Unknown" orientation and an empty surface view.
    """
    gii = nib.load(filepath)
    structure = _STRUCTURE_LABEL_MAP.get(structure_label, structure_label)
    is_surface = filepath.endswith(".surf.gii")

    gii.meta['AnatomicalStructurePrimary'] = structure
    if is_surface:
        gii.meta['GeometricType'] = 'Reconstruction'

    for da in gii.darrays:
        da.meta['AnatomicalStructurePrimary'] = structure
        if is_surface:
            da.meta['GeometricType'] = 'Reconstruction'

    nib.save(gii, filepath)

# -----------------------------------------------------------------------------
# Localite / Brainsight / BabelBrain / k-Plan utilities
# -----------------------------------------------------------------------------

# Small fixed rotation added whenever a roll of exactly 0/180/360
# degrees would otherwise be used — see deterministic_perpendicular_frame.
_ROLL_SAFETY_NUDGE_DEGREES = 1.0

def deterministic_perpendicular_frame(primary_axis: Sequence[float],
                                      roll_degrees: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    """Build two vectors perpendicular to `primary_axis`, deterministically
    (previously this was a random choice — see create_Localite_position_matrix
    / create_SimNIBS_position_matrix), with `roll_degrees` controlling
    rotation around `primary_axis`.

    At roll_degrees=0, the first returned vector is
    normalize(cross(reference, primary_axis)), where `reference` is a
    fixed, non-axis-aligned vector (unless that's nearly parallel to
    primary_axis, in which case a second fixed non-axis-aligned
    fallback is used instead).

    `reference` is deliberately NOT a coordinate axis like (0,0,1): the
    cross product of any vector with an axis-aligned reference is
    always exactly perpendicular to that axis too, which forces one
    component of the result to be an exact 0.0 at roll_degrees=0 (and
    180) for essentially any primary_axis — confirmed (via a stress
    test across thousands of random directions) to disappear entirely
    once the reference isn't axis-aligned. This was a real, reproducible
    bug: BabelBrain failed to load exported trajectories at the default
    roll of 0 degrees because of exactly this exact-zero matrix entry —
    a new failure mode versus the old np.random.randn(3)-based roll
    choice, which essentially never produced an exact zero.

    On top of that: even after the above fix, roll_degrees values of
    exactly 0, 180, or 360 are nudged by a small fixed offset
    (_ROLL_SAFETY_NUDGE_DEGREES) before building the frame — a
    defensive belt-and-suspenders measure requested after a report
    that 0-degree roll was still producing a faulty BabelBrain
    trajectory. The nudge is applied here, centrally, so it
    automatically covers both the actual generated position matrices
    AND Viewer.py's live preview (which calls this same function) —
    the preview's roll slider/label are driven by the user's own
    selected value and are unaffected, so the UI still reads "0 degrees"
    even though the underlying geometry uses a tiny nonzero rotation.

    Returns
    -------
    (perp_a, perp_b) : unit vectors, both perpendicular to primary_axis
        and to each other, with cross(primary_axis, perp_a) == perp_b.
    """
    import math

    if roll_degrees % 180.0 == 0.0:
        roll_degrees = roll_degrees + _ROLL_SAFETY_NUDGE_DEGREES

    primary = unit_vector(np.asarray(primary_axis, dtype=float))
    reference = unit_vector(np.array([1.0, 1.0, 1.0]))
    if abs(np.dot(reference, primary)) > 0.95:
        reference = unit_vector(np.array([1.0, -1.0, 1.0]))
    base_a = unit_vector(np.cross(reference, primary))
    base_b = np.cross(primary, base_a)  # already unit length (primary, base_a orthonormal)

    theta = math.radians(roll_degrees)
    perp_a = math.cos(theta) * base_a + math.sin(theta) * base_b
    perp_b = np.cross(primary, perp_a)
    return perp_a, perp_b


def create_SimNIBS_position_matrix(center_coordinates: Sequence[float], z_vector: Sequence[float],
                                   roll_degrees: float = 0.0) -> np.ndarray:
    """Build a 4×4 SimNIBS-style pose matrix from a center and a z-axis.

    The x/y axes are perpendicular to z, deterministic, and rotatable via
    roll_degrees (see deterministic_perpendicular_frame) — previously
    chosen randomly.
    """
    center = np.asarray(center_coordinates, dtype=float)
    z = unit_vector(np.asarray(z_vector, dtype=float))

    M = np.zeros((4, 4), dtype=float)
    M[3, 3] = 1.0
    M[:3, 3] = center
    M[:3, 2] = z

    y, x = deterministic_perpendicular_frame(z, roll_degrees)
    M[:3, 1] = y
    M[:3, 0] = x
    return M


def create_Localite_position_matrix(center_coordinates: Sequence[float], x_vector: Sequence[float],
                                    roll_degrees: float = 0.0) -> np.ndarray:
    """Build a 4×4 Localite-style pose matrix from a center and an x-axis.

    The y/z axes are perpendicular to x, deterministic, and rotatable via
    roll_degrees (see deterministic_perpendicular_frame) — previously
    chosen randomly.
    """
    center = np.asarray(center_coordinates, dtype=float)
    x = unit_vector(np.asarray(x_vector, dtype=float))

    M = np.zeros((4, 4), dtype=float)
    M[3, 3] = 1.0
    M[:3, 3] = center
    M[:3, 0] = x

    y, z = deterministic_perpendicular_frame(x, roll_degrees)
    M[:3, 1] = y
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


def export_Brainsight_trajectory(SimNIBS_position_matrix: np.ndarray,
                                 output_filepath: str):
    """Export trajectory text file for Brainsight."""

    simnibs.brainsight().write(np.squeeze(SimNIBS_position_matrix), output_filepath, overwrite=True)


def export_BabelBrain_trajectory(SimNIBS_position_matrix: np.ndarray,
                                 anchor_coordinates: Sequence[float],
                                 output_filepath: str,
                                 roi_name: str = "PlanTUS transducer position"):
    """Export trajectory text file for BabelBrain.

    anchor_coordinates is the point BabelBrain's trajectory format
    anchors the placement at — the target ROI's center of gravity when
    the beam axis is aimed exactly there ("target_centered" mode), or
    a point actually on the trajectory line otherwise (see
    prepare_acoustic_simulation's "vertex_normal" mode) — anchoring at
    the target center while the axis points elsewhere would describe
    two different lines, which BabelBrain has no way to reconcile.

    roi_name labels the exported trajectory/placement — target ROI
    name + vertex number by default, from the caller — falls back to
    a generic label if not given.
    """

    BabelBrain_position_matrix = SimNIBS_position_matrix.copy()
    BabelBrain_position_matrix[0:3,3] = anchor_coordinates

    simnibs.brainsight().write(np.squeeze(BabelBrain_position_matrix), output_filepath, overwrite=True)

    with open(output_filepath, 'r') as file:
      filedata = file.read()
    filedata = filedata.replace('NIfTI:Aligned', 'Brainsight')
    filedata = filedata.replace('SimNIBS v4.5.0', 'PlanTUS')
    filedata = filedata.replace('# Units: millimetres, degrees, milliseconds, and microvolts',
                                '# X=right->left, Y=anterior->posterior, Z=inferior->superior\n# Units: millimetres, degrees, milliseconds, and microvolts')
    # Was a blind filedata.replace('000', ...) — corrupted any numeric
    # value containing that digit sequence anywhere (confirmed: matched
    # inside "0.0000", "1000.5", "100.000", etc.), not just the intended
    # placeholder token. Matches only a standalone "000" (not adjacent
    # to a digit or decimal point on either side) instead — verified
    # against representative matrix-value strings before shipping.
    filedata = re.sub(r'(?<![\d.])000(?![\d.])', roi_name, filedata)
    with open(output_filepath, 'w') as file:
      file.write(filedata)


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
    """Apply a 4×4 affine to a GIFTI surface model.

    Pure nibabel/NumPy — replaces Workbench's ``-surface-apply-affine``.
    Faces are untouched; only vertex coordinates are transformed
    (row-vector convention: v' = v @ A[:3,:3].T + A[:3,3], i.e. the same
    convention Workbench and nibabel affines use).
    """
    vertices, faces, _ = _load_gifti_mesh(surface_model_filepath)
    A = np.loadtxt(transform_filepath)  # plain-text 4x4 matrix, as written by np.savetxt elsewhere

    vertices_h = np.hstack([vertices, np.ones((vertices.shape[0], 1))])
    vertices_out = (vertices_h @ A.T)[:, :3]

    _write_gifti_surface(vertices_out.astype(np.float32), faces.astype(np.int32), output_filepath)
    add_structure_information(output_filepath, structure)


def create_kps_file_for_kPlan(position_matrix_filepath: str, kps_filename: str) -> None:
    """Create a *.kps file (HDF5) for k-Plan from a MATLAB .mat pose matrix."""
    import numpy as np
    import scipy
    import h5py

    if np.lib.NumpyVersion(np.__version__) < '2.0.0':
        strfunc = np.string_
    else:
        strfunc = np.bytes_

    out_dir, _ = os.path.split(position_matrix_filepath)
    out_file = os.path.join(out_dir, f"{kps_filename}.kps")

    mat = scipy.io.loadmat(position_matrix_filepath)
    M = mat["position_matrix"].T.reshape((1, 4, 4)).astype("float32")

    with h5py.File(out_file, "w") as f:
        dset = f.create_dataset("/1/position_transform", (1, 4, 4), dtype="float32")
        dset[:] = M
        f["/1"].attrs.create("transform_label", strfunc("Localite transducer position"))
        f.attrs.create("application_name", strfunc("k-Plan"))
        f.attrs.create("file_type", strfunc("k-Plan Transducer Position"))
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


def voxelize_ellipsoid_in_volume(length: float, width: float,
                                 position_transform_filepath: str,
                                 reference_volume_filepath: str,
                                 output_filepath: str) -> None:
    """Rasterize a solid ellipsoid directly into a binary NIfTI volume.

    Pure NumPy/nibabel — replaces the old Workbench pipeline of
    ``-surface-coordinates-to-metric`` + ``-metric-to-volume-mapping
    -ribbon-constrained`` (using a second, slightly smaller ellipsoid
    surface only to define a thin "ribbon" as a rasterization proxy) +
    ``-volume-fill-holes``. Since the ellipsoid's exact geometry (radii
    + placement transform) is already known — it's the same definition
    used by :func:`create_surface_ellipsoid` — we can just evaluate the
    ellipsoid equation at every voxel center directly. No inner surface,
    no ribbon, no hole-filling needed; this is exact rather than a
    surface-shell approximation, up to ordinary voxel discretization.

    Uses the same (a, b, c) radii formula as :func:`create_surface_ellipsoid`,
    including its quirk where ``b`` scales with the reference volume's
    x/y field-of-view ratio rather than always equalling ``width/2``
    (it only reduces to ``width/2`` when the volume's x and y extents
    match, e.g. a typical isotropic 256^3 SimNIBS T1). Kept identical
    on purpose for behavioral parity with the surface-based ellipsoid.

    Parameters
    ----------
    length, width : float
        Same meaning as in :func:`create_surface_ellipsoid` (mm).
    position_transform_filepath : str
        Path to a plain-text 4x4 affine (ellipsoid-local -> world mm),
        as written by ``np.savetxt`` elsewhere in this module.
    reference_volume_filepath : str
        Defines the output grid (shape, affine) — typically the T1.
    output_filepath : str
        Destination ``.nii.gz`` for the binary (0/1) ellipsoid mask.
    """
    ref = nib.load(reference_volume_filepath)
    shape = ref.shape[:3]
    pixdim = ref.header["pixdim"][1:4]
    affine = ref.affine

    a = float(shape[0]) * float(pixdim[0]) * (float(width) / (float(shape[0]) * float(pixdim[0]))) / 2
    b = float(shape[1]) * float(pixdim[1]) * (float(width) / (float(shape[0]) * float(pixdim[0]))) / 2
    c = float(shape[2]) * float(pixdim[2]) * (float(length) / (float(shape[2]) * float(pixdim[2]))) / 2

    A = np.loadtxt(position_transform_filepath)
    A_inv = np.linalg.inv(A)

    ijk = np.indices(shape).reshape(3, -1).T
    ijk_h = np.hstack([ijk, np.ones((ijk.shape[0], 1))])
    world_h = ijk_h @ affine.T
    local_h = world_h @ A_inv.T
    x, y, z = local_h[:, 0], local_h[:, 1], local_h[:, 2]

    inside = (x / a) ** 2 + (y / b) ** 2 + (z / c) ** 2 <= 1.0
    mask = inside.reshape(shape).astype(np.uint8)

    nib.save(nib.Nifti1Image(mask, affine), output_filepath)


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
                                skip_viewer=False,
                                roll_degrees: float = 0.0,
                                orientation_mode: str = "target_centered") -> None:
    """End-to-end preparation for one candidate vertex (simulation folder).

    orientation_mode controls how the transducer's beam axis (vertex_vector)
    is oriented at the chosen skin vertex:
    - "target_centered" (default): aimed exactly at the target ROI's
      center of gravity, regardless of the local skin surface normal.
      Self-consistent with BabelBrain's exported trajectory, which
      anchors the placement at the target ROI's center — since the
      beam axis passes through that exact point by construction here.
    - "vertex_normal": aimed along the (negated) skin surface normal at
      the chosen vertex instead — not necessarily aimed at the target
      center. Since BabelBrain's trajectory format anchors the
      placement at a point ON the trajectory, using the target ROI's
      center for that anchor here would be geometrically inconsistent
      (the axis and the anchor would describe two different lines) —
      so in this mode, the BabelBrain anchor point is instead: the
      midpoint of the beam's intersection with the target ROI, if it
      intersects it at all; otherwise a point along the trajectory at
      the estimated focal distance (a well-defined but, in this
      non-intersecting case, essentially arbitrary point along the
      line, since there's no principled "correct" anchor available).
    """
    import scipy

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
    if orientation_mode == "vertex_normal":
        # Aimed along the local skin normal — not necessarily at the
        # target center. See docstring above re: the BabelBrain anchor
        # point handling this requires further down.
        vertex_vector = unit_vector(-skin_normals[vertex_number])
    else:
        # "target_centered" (default): always aim at the target
        # centroid. skin_target_intersection_values is still computed
        # above and still used below for focal_distance — only this
        # orientation choice varies with orientation_mode.
        vertex_vector = unit_vector(-skin_target_vectors[vertex_number])

    # --- Focal distance & FLHM
    # Moved earlier (was after the transducer-model/.kps export block)
    # so inter_center is available below for the BabelBrain trajectory
    # anchor point too, without duplicating this computation.
    if skin_target_intersection_values[vertex_number] == 0:
        target_center = roi_center_of_gravity(target_roi_filepath)
        skin_target_distances = distance_between_surface_and_point(os.path.join(output_path, "skin.surf.gii"), target_center)
        focal_distance = float(skin_target_distances[vertex_number] + additional_offset)
        inter_center = None
    else:
        inter_center = (np.asarray(skin_target_intersections[vertex_number][0]) +
                        np.asarray(skin_target_intersections[vertex_number][1])) / 2.0
        focal_distance = float(np.linalg.norm(skin_coordinates[vertex_number] - inter_center) + additional_offset)

    focal_distance = max(min(focal_distance, max_distance), min_distance)
    FLHM = compute_FLHM_for_focal_distance(focal_distance, focal_distance_list, flhm_list)

    # --- Localite pose for the transducer
    transducer_center_coordinates = vertex_coordinates - ((offset + additional_offset) * vertex_vector)
    position_matrix_Localite = create_Localite_position_matrix(transducer_center_coordinates, vertex_vector,
                                                                roll_degrees=roll_degrees)

    scipy.io.savemat(os.path.join(output_path_vtx, f"{target_roi_name}_{suffix}_PositionMatrix_Localite.mat"),
                     {'position_matrix': position_matrix_Localite})
    np.savetxt(os.path.join(output_path_vtx, f"{target_roi_name}_{suffix}_PositionMatrix_Localite.txt"),
               position_matrix_Localite)

    xml = create_fake_XML_structure_for_Localite(position_matrix_Localite,
                                                 f"transducer_position_{target_roi_name}_{suffix}", 0, 0)
    with open(os.path.join(output_path_vtx, f"{target_roi_name}_{suffix}_TransducerPosition_Localite_dummyXML.txt"), "a") as f:
        f.write(xml)

    # --- Convert to k-Plan
    position_matrix_kPlan = convert_Localite_to_kPlan_position_matrix(position_matrix_Localite)
    scipy.io.savemat(os.path.join(output_path_vtx, f"{target_roi_name}_{suffix}_PositionMatrix_kPlan.mat"),
                     {'position_matrix': position_matrix_kPlan})
    np.savetxt(os.path.join(output_path_vtx, f"{target_roi_name}_{suffix}_PositionMatrix_kPlan.txt"),
               position_matrix_kPlan)

    # --- Brainsight & Babelbrain trajectories
    position_matrix_SimNIBS = create_SimNIBS_position_matrix(transducer_center_coordinates, vertex_vector,
                                                              roll_degrees=roll_degrees)

    export_Brainsight_trajectory(position_matrix_SimNIBS,
                                 os.path.join(output_path_vtx, f"{target_roi_name}_{suffix}_Trajectory_Brainsight.txt"))

    if orientation_mode == "vertex_normal":
        # Axis isn't necessarily aimed at the target center here, so
        # anchoring BabelBrain's placement there (as the
        # "target_centered" mode does below) would be geometrically
        # inconsistent — the axis and the anchor would describe two
        # different lines. Anchor at a point actually ON the
        # trajectory instead: the intersection midpoint if the beam
        # crosses the target, otherwise a point along the trajectory
        # at the estimated focal distance (reuses inter_center /
        # focal_distance already computed above).
        if inter_center is not None:
            babelbrain_anchor_coordinates = inter_center
        else:
            babelbrain_anchor_coordinates = vertex_coordinates + focal_distance * vertex_vector
    else:
        babelbrain_anchor_coordinates = roi_center_of_gravity(target_roi_filepath)

    export_BabelBrain_trajectory(position_matrix_SimNIBS,
                                 babelbrain_anchor_coordinates,
                                 os.path.join(output_path_vtx, f"{target_roi_name}_{suffix}_Trajectory_BabelBrain.txt"),
                                 roi_name=f"{target_roi_name}_{suffix}")

    # --- Optional: transform transducer model
    transform = np.loadtxt(os.path.join(output_path_vtx, f"{target_roi_name}_{suffix}_PositionMatrix_kPlan.txt"))
    transform[0:3, 3] = transform[0:3, 3] * 1000  # back to mm for Workbench affine
    transform_filepath = os.path.join(output_path_vtx, f"{target_roi_name}_{suffix}_PositionMatrix_Transducer.txt")
    np.savetxt(transform_filepath, transform)

    transducer_out = os.path.join(output_path_vtx, f"{target_roi_name}_{suffix}_TransducerModel.surf.gii")
    if transducer_surface_model_filepath:
        transform_surface_model(transducer_surface_model_filepath, transform_filepath, transducer_out, "CEREBELLUM")
    else:
        # Defensive fallback for direct-Python callers that don't
        # pre-resolve a model path themselves (PlanTUS_wrapper.py
        # always does, and always uses the generic model — see its
        # "Transducer model creation" block).
        create_surface_transducer_model(transducer_diameter / 2, offset + additional_offset, transducer_out)
        transform_surface_model(transducer_out, transform_filepath, transducer_out, "CEREBELLUM")

    # --- .kps for k-Plan
    create_kps_file_for_kPlan(
        os.path.join(output_path_vtx, f"{target_roi_name}_{suffix}_PositionMatrix_kPlan.mat"),
        f"{target_roi_name}_{suffix}_TransducerPosition_kPlan"
    )

    # --- Ellipsoid (surface)
    focus_transform = np.loadtxt(os.path.join(output_path_vtx, f"{target_roi_name}_{suffix}_PositionMatrix_kPlan.txt"))
    focus_transform[0:3, 3] = focus_transform[0:3, 3] * 1000
    focus_transform[0:3, 3] = focus_transform[0:3, 3] + (vertex_vector * (offset + focal_distance))

    focus_transform_path = os.path.join(output_path_vtx, f"{target_roi_name}_{suffix}_PositionMatrix_Focus.txt")
    np.savetxt(focus_transform_path, focus_transform)

    ellipsoid_surf = os.path.join(output_path_vtx, f"{target_roi_name}_{suffix}_Focus_{round(focal_distance,1)}mm.surf.gii")
    create_surface_ellipsoid(FLHM, 5, focus_transform_path, t1_filepath, ellipsoid_surf)

    # --- Ellipsoid (volume) via direct rasterization (see voxelize_ellipsoid_in_volume)
    ellipsoid_vol = os.path.join(output_path_vtx, f"{target_roi_name}_{suffix}_Focus_{round(focal_distance,1)}mm.nii.gz")
    voxelize_ellipsoid_in_volume(FLHM, 5, focus_transform_path, t1_filepath, ellipsoid_vol)

    # --- Optional Workbench scene (never opened/required by PlanTUS
    # itself — just a data file for anyone who wants to inspect this
    # saved placement in Workbench directly). Paths are absolute rather
    # than relative to the scene file's own location: fixed folder
    # depth isn't safe now that output_folder lets the base output
    # directory sit anywhere. The template's BasePathType=AUTOMATIC
    # setting means Workbench correctly resolves absolute paths as
    # absolute regardless of the scene file's own location.
    skin_surf_filepath = os.path.join(output_path, "skin.surf.gii")
    scene_variable_names = [
        'SKIN_SURFACE_FILENAME', 'SKIN_SURFACE_FILEPATH',
        'T1_FILENAME', 'T1_FILEPATH',
        'MASK_FILENAME', 'MASK_FILEPATH',
        'TRANSDUCER_SURFACE_FILENAME', 'TRANSDUCER_SURFACE_FILEPATH',
        'FOCUS_VOLUME_FILENAME', 'FOCUS_VOLUME_FILEPATH',
        'FOCUS_SURFACE_FILENAME', 'FOCUS_SURFACE_FILEPATH'
    ]
    scene_variable_values = [
        os.path.basename(skin_surf_filepath), os.path.abspath(skin_surf_filepath),
        os.path.basename(t1_filepath), os.path.abspath(t1_filepath),
        os.path.basename(target_roi_filepath), os.path.abspath(target_roi_filepath),
        os.path.basename(transducer_out), os.path.abspath(transducer_out),
        os.path.basename(ellipsoid_vol), os.path.abspath(ellipsoid_vol),
        os.path.basename(ellipsoid_surf), os.path.abspath(ellipsoid_surf),
    ]
    try:
        create_scene(placement_scene_template_filepath,
                    os.path.join(output_path_vtx, "scene.scene"),
                    scene_variable_names,
                    scene_variable_values)
    except Exception as e:
        # Purely optional output — never let a missing/unreadable
        # template (or any other issue building the scene file) fail
        # the actual placement generation over it.
        print(f"Warning: could not create Workbench scene file ({e}); skipping.")

    # --- Visualize results
    if skip_viewer:
        return
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
