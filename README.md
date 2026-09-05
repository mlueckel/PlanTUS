# **PlanTUS** - A heuristic tool for prospective planning of transcranial ultrasound transducer placements 

##
When using this tool, please cite:

*Lueckel, M., Vijayakumar, S., & Bergmann, T. O. (2025). PlanTUS: A heuristic tool for prospective planning of transcranial ultrasound transducer placements. Brain Stimulation: Basic, Translational, and Clinical Research in Neuromodulation, 18(5), 1563–1565. DOI: 10.1016/j.brs.2025.08.013*

Link to paper: https://www.brainstimjrnl.com/article/S1935-861X(25)00306-7/fulltext
##

PlanTUS helps users of transcranial ultrasonic stimulation (TUS) to interactively and heuristically select the most promising transducer placement(s) for sonication of a specific target region of interest in a given individual.

<img src="https://github.com/user-attachments/assets/ff3850a9-2ed8-43a8-b93c-f28a9d0d7c2e" width="20" /> **PlanTUS is supposed to inform acoustic simulations, but does not replace them. Transducer positions selected using PlanTUS should always be validated using proper acoustic simulations!** <img src="https://github.com/user-attachments/assets/ff3850a9-2ed8-43a8-b93c-f28a9d0d7c2e" width="20" />

> **v2.0:** PlanTUS no longer depends on Connectome Workbench — everything now runs through PlanTUS' own built-in interactive viewer. Output filenames have also changed. See [`CHANGELOG.md`](./CHANGELOG.md) for the full list of changes, including a filename migration table if you have scripts depending on the old naming.

---

# Table of contents
- [Dependencies](#dependencies)
- [Instructions](#instructions)
  - [0. Before using PlanTUS](#0-before-using-plantus)
  - [1. Configure PlanTUS](#1-configure-plantus)
  - [2. Run the PlanTUS_wrapper.py script](#2-run-the-plantus_wrapperpy-script)
  - [3. Select transducer position(s)](#3-select-transducer-positions)
  - [4. Evaluate transducer and (estimated) focus position](#4-evaluate-transducer-and-estimated-focus-position)
  - [5. Use PlanTUS outputs for acoustic simulations and neuronavigation](#5-use-plantus-outputs-for-acoustic-simulations-and-neuronavigation)
  - [6. Reviewing k-Plan simulation results](#6-reviewing-k-plan-simulation-results)
- [Output reference](#output-reference)
- [Tips & troubleshooting](#tips--troubleshooting)
- [Contact](#contact)

---

# Dependencies

PlanTUS is a Python tool that wraps [SimNIBS](https://simnibs.github.io/simnibs/) meshes. Visualization and interactive planning are handled entirely by PlanTUS' own PyQt/VTK-based viewer.
### External software
| Software | Purpose | Link |
|---|---|---|
| **SimNIBS** (≥ 4.x) | Head segmentation (`charm`) and mesh generation; also provides the `simnibs` Python package (incl. the `brainsight` export module) used internally | https://simnibs.github.io/simnibs/build/html/installation/simnibs_installer.html |
| **FSL** | Only needed for `ImageTransform_4kPlan.py`, which uses `fslswapdim` to correct image orientation before k-Plan-compatible re-registration | https://fsl.fmrib.ox.ac.uk/fsl/fslwiki |

> **Note:** SimNIBS ships its own Python environment (`simnibs_env` / the `simnibs` conda env created by the installer). It is strongly recommended to install/run PlanTUS **inside that same environment**, since it already provides a compatible Python version and several of the dependencies below (numpy, nibabel, scipy, pandas, ants, etc.), and gives PlanTUS direct access to the `simnibs` package.

### Python packages
Beyond what SimNIBS' own environment already provides, PlanTUS additionally requires:

```
numpy
nibabel
nilearn
scipy
pandas
pyyaml
h5py
trimesh        # surface reconstruction/smoothing
vtk            # 3D rendering (the interactive viewer, always used)
PyQt5          # GUI for the interactive viewer (PySide6 also supported, tried first)
```

Optional:
```
potpourri3d    # exact geodesic-distance solver used for avoidance-mask erosion;
               # falls back to an approximate method if not installed
```

You can install these into your SimNIBS conda environment with, e.g.:

```bash
conda activate simnibs_env
pip install nilearn pyyaml h5py trimesh vtk PyQt5 potpourri3d
```

---

# Instructions

## 0. Before using PlanTUS

### What you need
In order to use PlanTUS, you need the following files:
- T1-weighted (T1w) MR image of your participant's head (`.nii`/`.nii.gz`)
- *(optionally)* T2-weighted (T2w) MR image of your participant's head (`.nii`/`.nii.gz`) — improves SimNIBS' `charm` segmentation, especially of the skull
- A binary mask of your target region of interest, co-registered to/in the same space as your participant's T1w image (`.nii`/`.nii.gz`)

### Charm
Make sure to run the SimNIBS `charm` pipeline (https://simnibs.github.io/simnibs/build/html/documentation/command_line/charm.html) on your participant's T1w (and T2w) MR image before using PlanTUS. Charm segments the head into different tissue types. This segmentation is used to extract skin, skull, and air-cavity surfaces, from which we generate a 3D model of the head and skull as well as the "no-go" avoidance regions (see step 2).

**Note:** If you want to use the PlanTUS output (i.e., the planned transducer position) for acoustic simulations in **k-Plan** (https://k-plan.io/), make sure to linearly co-register your participant's T1w MR image to a suitable MNI template (to get an image with an affine matrix that has all off-diagonal elements set to 0) and set the left, posterior, inferior corner of the image to (0,0,0) *before running charm*. Use `ImageTransform_4kPlan.py`, provided in this repository, for this — it ACPC-aligns the T1 to your MNI template via ANTs and then rewrites the affine so the origin sits at the required corner (flipping axes with FSL's `fslswapdim` where needed).

---

## 1. Configure PlanTUS

Where earlier versions of PlanTUS required editing variables directly inside `PlanTUS_wrapper.py`, all transducer- and setup-specific parameters now live in a separate **YAML configuration file**, and the subject-specific paths (T1, mesh, ROI, config) are passed as command-line arguments. This makes it easy to keep one config file per transducer model and reuse it across subjects.

### Command-line usage

```bash
python PlanTUS_wrapper.py <t1> <mesh> <roi> <config> [options]
```

| Argument | Description |
|---|---|
| `t1` | Path to the T1 image (the one used as input to SimNIBS' `charm`) |
| `mesh` | Path to the head mesh (`.msh` file generated by SimNIBS' `charm`, typically `m2m_<subject>/<subject>.msh`) |
| `roi` | Path to the mask of the target region of interest (in the same space as the T1 image). Note: the PlanTUS output folder will be named after this file. |
| `config` | Path to the YAML configuration file (see below) |

| Optional flag | Description |
|---|---|
| `--placement_only <N>` | Skip the interactive selection step entirely and directly (re-)generate the transducer placement for surface vertex number `N`. Useful for scripting or re-exporting outputs for a position you already picked. |
| `--skip_viewer` | Run all calculations but don't open the interactive viewer window — useful for batch/headless processing. |
| `--overwrite` | If existing PlanTUS results are found for the same inputs (T1, mesh, ROI, and transducer settings), overwrite them without prompting. |
| `--reuse_existing` | If existing PlanTUS results are found for the same inputs, reuse them without prompting (skips recomputation, jumps straight to the viewer with previously generated data). |

If existing results are found and neither `--overwrite` nor `--reuse_existing` is given, PlanTUS asks you interactively (in the terminal) whether to overwrite.

### Config file design and contents

Each transducer typically gets its **own config file** (e.g., `PlanTUS_config_CTX-545.yaml` for a NeuroFUS CTX-545 transducer, `PlanTUS_config_DPX-300.yaml` for a NeuroFUS DPX-300 transducer — both provided in this repository as templates). This keeps calibration data (focal distance/FLHM curves, aperture size, etc.) tied to hardware rather than to a subject, so the same config file can be reused across all participants sonicated with that transducer.

The YAML file has three logical sections:

**1) Transducer-specific variables** — the physical/acoustic properties of your transducer, typically taken from the manufacturer's calibration report:
- `max_distance`, `min_distance`, `optimal_distance`: maximum, minimum, and optimal focal depth of the transducer (mm). `optimal_distance` is used to score candidate positions (closer to optimal = better; see composite score below).
- `transducer_diameter`: aperture diameter (mm)
- `max_angle`: maximum allowed tilt of the transducer relative to the skin surface (degrees)
- `plane_offset`: offset between the radiating surface and the exit plane of the transducer (mm)
- `additional_offset`: additional offset between the skin and the exit plane of the transducer (mm; e.g., due to a gel pad or silicone spacer)
- `focal_distance_list`, `flhm_list`: paired lists of focal distance and corresponding focal length at half maximum (FLHM) values (mm), from the calibration report. For single-element (fixed-focus) transducers these can be single-value lists; for steerable/multi-element transducers, PlanTUS fits a cubic curve through these points to estimate FLHM at any focal distance actually used. **Make sure this list covers the range of focal distances you'll actually encounter** — the fitted curve is not bounded, and evaluating it well outside the calibrated range can produce unrealistic FLHM values.

**2) User-defined metric weights** — how much each geometric criterion contributes to the overall "quality" score computed for every point on the head surface (see [Output reference](#output-reference)):
- `weight_skin_target_distances`
- `weight_skin_target_angles`
- `weight_skin_target_intersections`
- `weight_skin_skull_angles`
- `weight_skull_thickness`

Each weight must be in `[0, 1]` and, conventionally, the five weights sum to 1. Setting a weight to 0 removes that criterion from the composite score; increasing a weight makes PlanTUS favor placements that score well on that particular criterion.

**Optional keys:**
- `output_folder`: base directory to write PlanTUS' `PlanTUS/<ROI-name>` output folder under. If omitted, defaults to the m2m folder (next to the `.msh` file) — the same location PlanTUS has always used.
- `IDTarget`: a custom label used to name output files/folders for a given placement (falls back to `vtx<N>`, the vertex number, if omitted)

**Example (`PlanTUS_config_CTX-545.yaml`):**
```yaml
# Output ------------------------------------------------------------------------
# Optional: base directory to write PlanTUS' "PlanTUS/<ROI_name>" output
# folder under. If omitted (or left blank), defaults to the m2m folder
# (next to the .msh file), i.e. the same location PlanTUS has always used.
# output_folder: /path/to/custom/output/base

# Transducer-specific variables ------------------------------------------------
max_distance: 76.6
min_distance: 33.1
optimal_distance: 52.4
transducer_diameter: 65.0
max_angle: 10.0
plane_offset: 10.82
additional_offset: 3.0
focal_distance_list: [33.1, 34.7, 36.8, 39.3, 42.4, 46.2, 50.8, 56.0, 61.8, 68.5, 76.6]
flhm_list:           [8.4, 9.3, 10.3, 11.6, 13.5, 16.1, 19.2, 22.8, 26.3, 30.2, 34.5]
# User-defined metric weights ---------------------------------------------------
# Note: each in [0,1], should sum to 1
weight_skin_target_distances: 0.2
weight_skin_target_angles: 0.2
weight_skin_target_intersections: 0.2
weight_skin_skull_angles: 0.2
weight_skull_thickness: 0.2
```

Copy one of the provided templates, rename it after your transducer, adjust the calibration values, and point `PlanTUS_wrapper.py` at it.

### Transducer models
PlanTUS generates a simple cylindrical transducer model, sized to your config's `transducer_diameter` (aperture) and `plane_offset` + `additional_offset` (height), and applies the selected placement's pose to it before exporting. This is the only transducer model PlanTUS uses — there's no device-specific 3D model bundled or configurable, and no config key to choose one; the geometry only affects visualization/export, not the underlying trajectory/position calculations. The live preview shown while picking a placement in the viewer matches this model's real dimensions.

*(Earlier versions bundled a NeuroFUS CTX-500-4 model and let you switch to a generic cylinder via `bUseGenericTransducerModel`; both the bundled model and that config key have been removed — see [`CHANGELOG.md`](./CHANGELOG.md).)*

---

## 2. Run the *PlanTUS_wrapper.py* script

Based on the subject-specific, 3D-reconstructed skin and skull surfaces as well as the mask of the target region, PlanTUS computes several per-vertex metrics across the entire head surface to help you intuitively evaluate potential transducer positions. See [Output reference](#output-reference) below for the full list and how they combine into a single composite score. In brief:

- **Distance** between the skin surface and the target region — restricts placement to the area within your transducer's reachable focal depth.
<img src="https://github.com/user-attachments/assets/15fa5cb8-0c5b-4d34-ab14-d622c217536e" width="200" />

- **Intersection** between the target region and an idealized (straight-line) acoustic beam trajectory.
<img src="https://github.com/user-attachments/assets/d6bdfd06-3ff7-4094-9321-f8f34641d80c" width="200" />
<img src="https://github.com/user-attachments/assets/b6747049-5366-4cfd-9550-d43431a113b7" width="200" />

- **Tilt angle** of the transducer relative to the skin surface required for the beam trajectory to hit the target.
<img src="https://github.com/user-attachments/assets/5f331eaf-b1d7-48aa-bf0d-d5b1f946078b" width="200" />

- **Angle of incidence** between the skin and skull surface normals (relevant for reflections at the skull).
<img src="https://github.com/user-attachments/assets/988faa36-3083-4b00-bdf0-f97706bdaf09" width="200" />
<img src="https://github.com/user-attachments/assets/ee09ef52-18f0-4045-86f1-64d3c3aaf0c7" width="200" />

- **Skull thickness** underneath each candidate position (thinner is generally preferable).

PlanTUS also automatically identifies no-go / avoidance regions (grey areas on the head surface below) where placing a transducer is not possible or advisable — around the eyes and ears, above air-filled cavities/sinuses, and below head/neck height — and combines the individual metrics above into a single **composite quality score** per vertex.

---

## 3. Select transducer position(s)

Unless `--skip_viewer` was given, PlanTUS opens its own interactive viewer window once the metrics above have been computed. The window shows four head-surface panels (Distance, Target Intersection, Transducer Tilt, Skin-Skull Angle — switchable via the dropdown above each panel) plus two oblique volume-view panels on the right, following the trajectory into the head.

<img src="https://github.com/user-attachments/assets/7a71ff06-2d42-430c-9161-bf2b01bd4377" width="1000" />


**To select a placement, right-click anywhere on the head surface** — no separate "selection mode" toggle needed; left-click/drag still rotates the view as normal. The viewer opens with a suggested initial placement already shown (the vertex with the best composite score), which you're free to override by right-clicking elsewhere.

On picking a vertex:
- A semi-transparent transducer model (body + a handle/cable indicator) appears at the placement, oriented along the beam axis toward the target. Use the **Transducer rotation** slider to adjust roll around that axis before saving.
- The two volume-view panels update to follow the trajectory, showing the estimated intracranial focus (an ellipsoid sized from your transducer's focal-distance/FLHM calibration) and the target ROI (green outline).
- A small marker dot is dropped on the head surface at the picked vertex. Dots from earlier picks in the same session are **not** removed by picking again — "Remove Placement Markers" clears them explicitly; "Remove Transducer Model" hides just the live preview model.

Click **Save Placement** to write the full set of output files for the current vertex (see [Output reference](#output-reference)) — this does **not** close the window, so you can keep picking and saving further candidate placements in the same session. The window only closes when you close it yourself.

Preset camera buttons (Top / Front / Lateral Right / Lateral Left / Oblique Right / Oblique Left) are available in the toolbar, along with a Screenshot button that saves the current window contents as a PNG.

> If you already know which vertex/triangle you want (e.g., from a previous run, or scripted across subjects), you can skip the interactive step entirely with `--placement_only <N>`.

---

## 4. Evaluate transducer and (estimated) focus position

The transducer model and estimated acoustic focus shown live in the viewer (step 3) are the same evaluation previously done in a separate step after generation — there's no second window to open. The oblique volume-view panels let you evaluate the expected on- vs. off-target stimulation in terms of overlap between the estimated focus (ellipsoid) and the target region (green outline) before you commit to saving a placement.

---

## 5. Use PlanTUS outputs for acoustic simulations and neuronavigation

PlanTUS outputs several files for further use with different…

**acoustic simulation software** – for validation of the selected transducer placement(s).  
<img src="https://github.com/user-attachments/assets/12315d1b-24ba-42bb-ab98-0d6b4bde651f" width="200" />

**neuronavigation software** – for MR-guided navigation of the transducer to the selected position(s).  
<img src="https://github.com/user-attachments/assets/540e9535-a9a4-4ea4-ac2a-0eebd15982ac" width="200" />

See [Output reference](#output-reference) for the complete, per-format list of exported files.

**k-Plan**: The selected transducer placement can be easily imported into the k-Plan software (https://k-plan.io/), using the `.kps` output file, for validating the heuristically selected transducer placement with proper acoustic simulations.

**Localite**: The selected transducer placement can be easily imported into the Localite neuronavigation software (https://www.localite.de/en/products/tms-navigator/) as a target for transducer navigation (i.e., instrument marker), using the exported XML snippet.

**Brainsight and BabelBrain**: PlanTUS also exports ready-to-import trajectory text files for Rogue Research's Brainsight and for [BabelBrain](https://github.com/ProteusMRIgHIFU/BabelBrain), using the same underlying pose but written in each tool's expected trajectory format.

---

# Output reference

All outputs are written into a per-target folder:
```
<m2m_subject>/PlanTUS/<ROI-name>/
```
(named after your ROI file, so different targets in the same subject don't overwrite each other; override the base directory with `output_folder` in your config — see [Configure PlanTUS](#1-configure-plantus).)

### Whole-head surface outputs (generated once per ROI, before you pick a position)
| File | Description |
|---|---|
| `skin.surf.gii`, `skull.surf.gii` | Reconstructed skin and skull surfaces (GIFTI), extracted from the SimNIBS mesh |
| `avoidance_skin.func.gii` | Per-vertex 0/1 mask marking no-go regions (eyes, ears, air cavities/sinuses, below head height) |
| `distances_skin.func.gii`, `distances_skin_thresholded.func.gii` | Distance (mm) from each skin vertex to the target region's center of gravity; thresholded version restricts to vertices within `max_distance` |
| `angles_skin.func.gii` | Angle (degrees) between the skin surface normal and the skin→target vector at each vertex — shown as "Transducer Tilt" in the viewer |
| `target_intersection_skin.func.gii` | Length (mm) of the idealized straight-line beam's intersection with the target region, per vertex |
| `skin_skull_angles_skin.func.gii` | Angle (degrees) between skin and skull surface normals at each vertex (angle of incidence) |
| `skull_thickness_skin.func.gii`, `skull_thickness_skull.func.gii` | Estimated skull thickness (mm) under each vertex |
| `composite_TargetDistance<..>_TargetAngle<..>_TargetIntersection<..>_SkinSkullAngle<..>_SkullThickness<..>_skin.func.gii` | The combined, weighted composite quality score per vertex (see below), with the weights used baked into the filename for traceability |
| `<ROI-name>_3Dmodel.stl` | Triangulated 3D surface of the target ROI, used for the beam-intersection calculations |

**How the composite score is built:** each of the five raw metrics (skin–target distance, tilt angle, beam–target intersection length, skin–skull angle, skull thickness) is rescaled to a `[0, 1]` "utility" (with distance evaluated relative to your transducer's `optimal_distance`, not just `max_distance`), zeroed out for vertices beyond `max_distance` or inside an avoidance region, weighted by your config's `weight_*` values, and combined via a weighted geometric mean — so a very poor score on one criterion can't be fully compensated for by good scores elsewhere, and a vertex missing on any required criterion scores 0 overall.

### Position-specific outputs (generated after you select and save a vertex)
Written into a subfolder named after `IDTarget` (or `vtx<N>` if not set):

| File | Format / destination | Description |
|---|---|---|
| `<roi>_<ID>_PositionMatrix_Localite.mat` / `.txt` | Localite | 4×4 affine transform defining the transducer pose, in Localite convention (mm) |
| `<roi>_<ID>_TransducerPosition_Localite_dummyXML.txt` | Localite | Ready-to-paste XML `<InstrumentMarker>` snippet for Localite's target file |
| `<roi>_<ID>_PositionMatrix_kPlan.mat` / `.txt` | k-Plan | Same pose converted to k-Plan's axis convention and units (meters) |
| `<roi>_<ID>_TransducerPosition_kPlan.kps` | k-Plan | HDF5 transducer-position file that recreates the exact placement when imported into k-Plan |
| `<roi>_<ID>_Trajectory_Brainsight.txt` | Brainsight | Trajectory file in Brainsight's native format |
| `<roi>_<ID>_Trajectory_BabelBrain.txt` | BabelBrain | Trajectory file adapted for BabelBrain (target recentered on the ROI's center of gravity) |
| `<roi>_<ID>_PositionMatrix_Transducer.txt` | visualization | Pose transform (mm) used to place the transducer 3D model |
| `<roi>_<ID>_TransducerModel.surf.gii` | visualization | The transducer 3D model (device-specific or generic cylinder), transformed to the selected pose |
| `<roi>_<ID>_PositionMatrix_Focus.txt` | visualization | Pose transform (mm) used to place the estimated-focus ellipsoid |
| `<roi>_<ID>_Focus_<focal_distance>mm.surf.gii` / `.nii.gz` | visualization | Simplified ellipsoidal representation of the expected acoustic focus (surface and binary volume), sized from the FLHM at the estimated focal distance |

> **Filenames changed in v2.0** — see [`CHANGELOG.md`](./CHANGELOG.md) for the old→new mapping if you have scripts depending on the previous naming.

---

# Tips & troubleshooting
- **One config file per transducer, reused across subjects.** Keep subject-specific paths (`t1`, `mesh`, `roi`) out of the config file entirely — they're supplied on the command line instead, so the same config works for every participant scanned with that transducer.
- **Weights don't have to be equal.** If, e.g., minimizing skull thickness matters far more for your setup than tilt angle, raise `weight_skull_thickness` and lower `weight_skin_target_angles` (keep the five roughly summing to 1 for comparable composite scores across runs).
- **You can save several placements in one session.** The viewer stays open after "Save Placement" — right-click a new vertex and save again as many times as you like before closing the window.
- **Batch/headless use.** Combine `--placement_only <N>` with `--skip_viewer` to (re-)generate all export files for a known vertex without opening the viewer at all — handy for re-exporting after a config change, or for scripting across many subjects/positions.
- **Re-running on the same inputs.** Use `--overwrite` or `--reuse_existing` to skip the interactive y/n prompt when PlanTUS detects it's already been run for the same T1/mesh/ROI/config combination.
- **Make sure your FLHM calibration covers your actual working distances.** `focal_distance_list`/`flhm_list` are fit with an unconstrained cubic curve — evaluating it well outside the calibrated range can produce unrealistic FLHM/focus-size values.
- **k-Plan compatibility starts before `charm`.** The T1→MNI/ACPC alignment and origin correction (`ImageTransform_4kPlan.py`) must be done *before* running SimNIBS' `charm`, not after — charm needs to run on the already-corrected image.
- Remember: **PlanTUS is a heuristic planning aid, not a validated acoustic simulator.** Always confirm any selected placement with proper acoustic simulation software (k-Plan, k-Wave, BabelBrain, …) before sonicating.

---

# Contact
Maximilian Lueckel  
mlueckel@uni-mainz.de
