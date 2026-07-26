# **PlanTUS** - A heuristic tool for prospective planning of transcranial ultrasound transducer placements 

##
When using this tool, please cite:

*Lueckel, M., Vijayakumar, S., & Bergmann, T. O. (2025). PlanTUS: A heuristic tool for prospective planning of transcranial ultrasound transducer placements. Brain Stimulation: Basic, Translational, and Clinical Research in Neuromodulation, 18(5), 1563–1565. DOI: 10.1016/j.brs.2025.08.013*

Link to paper: https://www.brainstimjrnl.com/article/S1935-861X(25)00306-7/fulltext
##

PlanTUS helps users of transcranial ultrasonic stimulation (TUS) to interactively and heuristically select the most promising transducer placement(s) for sonication of a specific target region of interest in a given individual.

<img src="https://github.com/user-attachments/assets/ff3850a9-2ed8-43a8-b93c-f28a9d0d7c2e" width="20" /> **PlanTUS is supposed to inform acoustic simulations, but does not replace them. Transducer positions selected using PlanTUS should always be validated using proper acoustic simulations!** <img src="https://github.com/user-attachments/assets/ff3850a9-2ed8-43a8-b93c-f28a9d0d7c2e" width="20" />

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
  - [6. Review acoustic simulation results](#6-review-acoustic-simulation-results)
- [Output reference](#output-reference)
- [Tips & troubleshooting](#tips--troubleshooting)
- [Contact](#contact)

---

# Dependencies

PlanTUS is a Python tool that wraps [SimNIBS](https://simnibs.github.io/simnibs/) meshes and drives [Connectome Workbench](https://humanconnectome.org/software/get-connectome-workbench) for visualization. It needs the following pieces of software installed and on your system:

### External software
| Software | Purpose | Link |
|---|---|---|
| **SimNIBS** (≥ 4.x) | Head segmentation (`charm`) and mesh generation; also provides the `simnibs` Python package (incl. the `brainsight` export module) used internally | https://simnibs.github.io/simnibs/build/html/installation/simnibs_installer.html |
| **Connectome Workbench** | 3D visualization of the head surface, metrics, and transducer/focus models (`wb_view`, `wb_command`) | https://humanconnectome.org/software/get-connectome-workbench |
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
antspyx        # imported as `ants`; needed for k-Plan result registration and ImageTransform_4kPlan.py
pynput         # only needed for the Connectome Workbench click-tracking workflow (see step 3)
PyQt5          # only needed if you use the internal viewer (--use_internal_viewer)
vtk            # only needed if you use the internal viewer (--use_internal_viewer)
```

You can install these into your SimNIBS conda environment with, e.g.:

```bash
conda activate simnibs_env
pip install nilearn pyyaml h5py antspyx pynput PyQt5 vtk
```

`--use_internal_viewer` is optional: if you don't pass it, PlanTUS drives `wb_view` directly and you don't need `PyQt5`/`vtk`. If you do use it, PlanTUS renders the head surface and transducer/focus models in its own lightweight VTK-based viewer (`code/Viewer.py`) instead of launching Connectome Workbench.

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
| `--skip_wb_view` | Run all calculations but skip opening Connectome Workbench (`wb_view`) — useful for batch/headless processing |
| `--use_internal_viewer` | Use PlanTUS' own lightweight VTK-based viewer instead of `wb_view` |
| `--do_only_trajectory <N>` | Skip the interactive selection step entirely and directly (re-)generate the transducer placement for surface vertex/triangle number `N`. Useful for scripting or re-exporting outputs for a position you already picked. |

### Config file design and contents

Each transducer typically gets its **own config file** (e.g., `PlanTUS_config_CTX-545.yaml` for a NeuroFUS CTX-545 transducer, `PlanTUS_config_DPX-300.yaml` for a NeuroFUS DPX-300 transducer — both provided in this repository as templates). This keeps calibration data (focal-distance/FLHM curves, aperture size, etc.) tied to hardware rather than to a subject, so the same config file can be reused across all participants sonicated with that transducer.

The YAML file has three logical sections:

**1) Transducer-specific variables** — the physical/acoustic properties of your transducer, typically taken from the manufacturer's calibration report:
- `max_distance`, `min_distance`, `optimal_distance`: maximum, minimum, and optimal focal depth of the transducer (mm). `optimal_distance` is new relative to earlier versions and is used to score candidate positions (closer to optimal = better; see composite score below).
- `transducer_diameter`: aperture diameter (mm)
- `max_angle`: maximum allowed tilt of the transducer relative to the skin surface (degrees)
- `plane_offset`: offset between the radiating surface and the exit plane of the transducer (mm)
- `additional_offset`: additional offset between the skin and the exit plane of the transducer (mm; e.g., due to a gel pad or silicone spacer)
- `focal_distance_list`, `flhm_list`: paired lists of focal distance and corresponding focal length at half maximum (FLHM) values (mm), from the calibration report. For single-element (fixed-focus) transducers these can be single-value lists; for steerable/multi-element transducers, PlanTUS fits a cubic curve through these points to estimate FLHM at any focal distance actually used.

**2) User-defined metric weights** — how much each geometric criterion contributes to the overall "quality" score computed for every point on the head surface (see [Output reference](#output-reference)):
- `weight_skin_target_distances`
- `weight_skin_target_angles`
- `weight_skin_target_intersections`
- `weight_skin_skull_angles`
- `weight_skull_thickness`

Each weight must be in `[0, 1]` and, conventionally, the five weights sum to 1. Setting a weight to 0 removes that criterion from the composite score; increasing a weight makes PlanTUS favor placements that score well on that particular criterion.

**3) Setup** — local installation paths:
- `connectome_wb_path`: path to your Connectome Workbench `bin_<platform>` folder (e.g., `/usr/local/workbench/bin_linux64`)

**Optional keys:**
- `IDTarget`: a custom label used to name output files/folders for a given placement (falls back to `vtx<N>`, the vertex number, if omitted)
- `bUseGenericTransducerModel`: if `true`, PlanTUS generates a simple cylindrical placeholder transducer model instead of using a pre-made one (see [transducer models](#transducer-models) below)

**Example (`PlanTUS_config_CTX-545.yaml`):**
```yaml
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
# Setup --------------------------------------------------------------------------
connectome_wb_path: /usr/local/workbench/bin_linux64
```

Copy one of the provided templates, rename it after your transducer, adjust the calibration values, and point `PlanTUS_wrapper.py` at it.

### Transducer models
By default, PlanTUS uses the bundled 3D model of a NeuroFUS CTX-500-4 transducer (`resources/transducer_models/TRANSDUCER-NEUROFUS-CTX-500-4_DEVICE.surf.gii`) for visualization purposes, positioned and oriented according to the metrics computed for your selected placement. If you set `bUseGenericTransducerModel: true` in your config, PlanTUS instead generates a simple cylinder of the correct aperture (`transducer_diameter`) and offset (`plane_offset` + `additional_offset`) — useful if you don't have (or don't need) a device-specific 3D model. Note that this only affects the *visualization* of the transducer; it has no effect on the underlying trajectory/position calculations.

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

The aforementioned metrics will be visualized in **Connectome Workbench** on the 3D-reconstructed head surface:

<img src="https://github.com/user-attachments/assets/df3d85c4-4056-4bb6-99aa-23b82feb822d" width="800" />

To select a potential transducer placement, simply click on the head surface wherever you would like to place the transducer. A small white sphere will appear at the respective position, marking the position of the transducer center on the head surface. The volume view (right) then allows you to check the intersection between the target region and an idealized acoustic beam trajectory (blue/green straight line) going from that position into the brain (oblique viewing option).

<img src="https://github.com/user-attachments/assets/cf6c9517-e4d4-444f-97b5-d49475feafd9" width="800" />

After clicking on a position, you will be asked (in the terminal) whether you want to generate a transducer placement for the selected position. If you answer "no", nothing happens and you can continue selecting other positions. If you answer "yes", PlanTUS computes and exports the full set of placement outputs for that vertex (see [Output reference](#output-reference)) and a new Connectome Workbench window pops up (see below).

> If you already know which vertex/triangle you want (e.g., from a previous run, or scripted across subjects), you can skip the interactive step entirely with `--do_only_trajectory <N>`.

---

## 4. Evaluate transducer and (estimated) focus position

After selecting a position, a new Connectome Workbench window pops up that shows the resulting transducer placement (left) and a simplified representation of the expected acoustic focus (red outline) overlaid on the target mask (green) and anatomical MR image (volume view on the right).

<img src="https://github.com/user-attachments/assets/44b4f69a-df07-47eb-8858-e0da64af2172" width="800" />

The oblique volume view (right) will help you to evaluate the expected on- vs. off-target stimulation in terms of overlap between the simplified acoustic focus and the target region.

<img src="https://github.com/user-attachments/assets/72fd9a0a-f7dc-461f-82db-82ea601e2751" width="800" />

---

## 5. Use PlanTUS outputs for acoustic simulations and neuronavigation

PlanTUS outputs several files for further use with different…

**acoustic simulation software** – for validation of the selected transducer placement(s).  
<img src="https://github.com/user-attachments/assets/12315d1b-24ba-42bb-ab98-0d6b4bde651f" width="200" />

**neuronavigation software** – for MR-guided navigation of the transducer to the selected position(s).  
<img src="https://github.com/user-attachments/assets/540e9535-a9a4-4ea4-ac2a-0eebd15982ac" width="200" />

See [Output reference](#output-reference) for the complete, per-format list of exported files.

**k-Plan example**: The selected transducer placement can be easily imported into the k-Plan software (https://k-plan.io/), using the `.kps` output file, for validating the heuristically selected transducer placement with proper acoustic simulations.

<img src="https://github.com/user-attachments/assets/ef43e905-1466-4dc3-9e69-ccefa266d8fa" width="800" />

**Localite example**: The selected transducer placement can be easily imported into the Localite neuronavigation software (https://www.localite.de/en/products/tms-navigator/) as a target for transducer navigation (i.e., instrument marker), using the exported XML snippet.

<img src="https://github.com/user-attachments/assets/123bb35e-965e-4993-a2f6-9ec445bbe2ba" width="800" />

**Brainsight and BabelBrain**: PlanTUS also exports ready-to-import trajectory text files for Rogue Research's Brainsight and for [BabelBrain](https://github.com/ProteusMRIgHIFU/BabelBrain), using the same underlying pose but written in each tool's expected trajectory format.

---

## 6. Review acoustic simulation results

Eventually, acoustic simulation results (e.g., acoustic pressure maps, thermal dose) can be loaded and evaluated in the same environment (Connectome Workbench), overlaid on your anatomical image. White outlines in the volume view (right) indicate the borders of the target region.

<img src="https://github.com/user-attachments/assets/77ef1860-d809-4b43-a60a-c712257150ee" width="800" />

Again, the oblique volume view (right) can help you to evaluate on- vs. off-target stimulation in terms of overlap between the simulated acoustic focus and the target region (indicated by white outline).

<img src="https://github.com/user-attachments/assets/b1502848-d858-41ee-93a7-7cc3ab297e0b" width="800" />

### New: importing k-Plan simulation results (HDF5 + CT)

To bring **k-Plan** simulation results back into this environment for review, PlanTUS can now read k-Plan's native HDF5 (`.h5`) output directly — this is a new input modality alongside the T1/mesh/ROI inputs used for planning. You additionally need:

- the k-Plan simulation results file (`.h5`), containing (at minimum) the medium mask and, per sonication, the simulated pressure amplitude and thermal dose fields, and
- a **CT image** of the same participant, in whichever space you want the results resampled into (k-Plan simulations are computed on a CT-derived acoustic medium, so a CT — rather than the T1 — is the natural common reference here).

Internally, PlanTUS:
1. Reads the medium mask and, for each sonication, the pressure-amplitude and thermal-dose volumes out of the `.h5` file (using the grid spacing stored in the file to reconstruct a NIfTI-compatible affine),
2. registers the medium mask to your CT image using an ANTs rigid+scaling transform (`TRSAA`),
3. applies that same transform to the pressure-amplitude and thermal-dose volumes, and
4. writes out three CT-aligned `.nii.gz` volumes per sonication (medium mask, acoustic pressure, thermal dose), ready to load into Connectome Workbench next to your T1 and target ROI for the on- vs. off-target review shown above.

This requires the `h5py` and `antspyx` (`ants`) Python packages (see [Dependencies](#dependencies)).

---

# Output reference

All outputs are written into a per-target folder:
```
<m2m_subject>/PlanTUS/<ROI-name>/
```
(named after your ROI file, so different targets in the same subject don't overwrite each other).

### Whole-head surface outputs (generated once per ROI, before you pick a position)
| File | Description |
|---|---|
| `skin.surf.gii`, `skull.surf.gii` | Reconstructed skin and skull surfaces (GIFTI), extracted from the SimNIBS mesh |
| `avoidance_skin.func.gii` | Per-vertex 0/1 mask marking no-go regions (eyes, ears, air cavities/sinuses, below head height) |
| `distances_skin.func.gii`, `distances_skin_thresholded.func.gii` | Distance (mm) from each skin vertex to the target region's center of gravity; thresholded version restricts to vertices within `max_distance` |
| `angles_skin.func.gii` | Angle (degrees) between the skin surface normal and the skin→target vector at each vertex |
| `target_intersection_skin.func.gii` | Length (mm) of the idealized straight-line beam's intersection with the target region, per vertex |
| `skin_skull_angles_skin.func.gii` | Angle (degrees) between skin and skull surface normals at each vertex (angle of incidence) |
| `skull_thickness_skin.func.gii`, `skull_thickness_skull.func.gii` | Estimated skull thickness (mm) under each vertex |
| `composite_TargetDistance<..>_TargetAngle<..>_TargetIntersection<..>_SkinSkullAngle<..>_SkullThickness<..>_skin.func.gii` | The combined, weighted composite quality score per vertex (see below), with the weights used baked into the filename for traceability |
| `<ROI-name>_3Dmodel.stl` | Triangulated 3D surface of the target ROI, used for the beam-intersection calculations |
| `scene.scene` | Connectome Workbench scene file tying the above together for the interactive planning view |

**How the composite score is built:** each of the five raw metrics (skin–target distance, tilt angle, beam–target intersection length, skin–skull angle, skull thickness) is rescaled to a `[0, 1]` "utility" (with distance evaluated relative to your transducer's `optimal_distance`, not just `max_distance`), zeroed out for vertices beyond `max_distance` or inside an avoidance region, weighted by your config's `weight_*` values, and combined via a weighted geometric mean — so a very poor score on one criterion can't be fully compensated for by good scores elsewhere, and a vertex missing on any required criterion scores 0 overall.

### Per-position outputs (generated after you select and confirm a vertex)
Written into a subfolder named after `IDTarget` (or `vtx<N>` if not set):

| File | Format / destination | Description |
|---|---|---|
| `position_matrix_<roi>_<ID>_Localite.mat` / `.txt` | Localite | 4×4 affine transform defining the transducer pose, in Localite convention (mm) |
| `position_matrix_<roi>_<ID>_Localite_XML.txt` | Localite | Ready-to-paste XML `<InstrumentMarker>` snippet for Localite's target file |
| `position_matrix_<roi>_<ID>_kPlan.mat` / `.txt` | k-Plan | Same pose converted to k-Plan's axis convention and units (meters) |
| `<roi>_<ID>.kps` | k-Plan | HDF5 transducer-position file that recreates the exact placement when imported into k-Plan |
| `trajectory_<roi>_<ID>_Brainsight.txt` | Brainsight | Trajectory file in Brainsight's native format |
| `trajectory_<roi>_<ID>_BabelBrain.txt` | BabelBrain | Trajectory file adapted for BabelBrain (target recentered on the ROI's center of gravity) |
| `transducer_<roi>_<ID>.surf.gii` | visualization | The transducer 3D model (device-specific or generic cylinder), transformed to the selected pose |
| `focus_<roi>_<ID>_<focal_distance>.surf.gii` / `.nii.gz` | visualization | Simplified ellipsoidal representation of the expected acoustic focus (surface and binary volume), sized from the FLHM at the estimated focal distance |
| `scene.scene` | visualization | Connectome Workbench scene tying transducer + focus + T1 + ROI together for the placement-review view |

### k-Plan simulation-review outputs (step 6, new)
Per sonication `i` found in the imported `.h5` file, written alongside it:
| File | Description |
|---|---|
| `<h5-name>_MediumMask_Sonication<i>.nii.gz` | Acoustic medium mask, registered to your CT |
| `<h5-name>_AcousticPressure_Sonication<i>.nii.gz` | Simulated acoustic pressure amplitude field, registered to your CT |
| `<h5-name>_ThermalDose_Sonication<i>.nii.gz` | Simulated thermal dose field, registered to your CT |

---

# Tips & troubleshooting
- **One config file per transducer, reused across subjects.** Keep subject-specific paths (`t1`, `mesh`, `roi`) out of the config file entirely — they're supplied on the command line instead, so the same config works for every participant scanned with that transducer.
- **Weights don't have to be equal.** If, e.g., minimizing skull thickness matters far more for your setup than tilt angle, raise `weight_skull_thickness` and lower `weight_skin_target_angles` (keep the five roughly summing to 1 for comparable composite scores across runs).
- **Batch/headless use.** Combine `--do_only_trajectory <N>` with `--skip_wb_view` to (re-)generate all export files for a known vertex without ever opening Connectome Workbench — handy for re-exporting after a config change, or for scripting across many subjects/positions.
- **`--use_internal_viewer` is optional**, not required — if you don't need PlanTUS' own VTK viewer, skip installing `PyQt5`/`vtk` altogether.
- **k-Plan compatibility starts before `charm`.** The T1→MNI/ACPC alignment and origin correction (`ImageTransform_4kPlan.py`) must be done *before* running SimNIBS' `charm`, not after — charm needs to run on the already-corrected image.
- Remember: **PlanTUS is a heuristic planning aid, not a validated acoustic simulator.** Always confirm any selected placement with proper acoustic simulation software (k-Plan, k-Wave, BabelBrain, …) before sonicating.

---

# Contact
Maximilian Lueckel  
mlueckel@uni-mainz.de
