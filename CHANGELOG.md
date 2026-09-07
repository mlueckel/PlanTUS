# Changelog

## v2.0.0

### Breaking changes

**All per-vertex output filenames have changed.** The identifying `<ROI>_vtx<N>` segment now comes first, followed by a category and type, instead of a prefix-first scheme. If you have scripts that parse PlanTUS output filenames, they will need updating.

| Old filename | New filename |
|---|---|
| `<ROI>_vtx<N>.kps` | `<ROI>_vtx<N>_TransducerPosition_kPlan.kps` |
| `focus_<ROI>_vtx<N>_<dist>.nii.gz` | `<ROI>_vtx<N>_Focus_<dist>mm.nii.gz` |
| `focus_<ROI>_vtx<N>_<dist>.surf.gii` | `<ROI>_vtx<N>_Focus_<dist>mm.surf.gii` |
| `focus_position_matrix_<ROI>_vtx<N>.txt` | `<ROI>_vtx<N>_PositionMatrix_Focus.txt` |
| `position_matrix_<ROI>_vtx<N>_kPlan.mat` | `<ROI>_vtx<N>_PositionMatrix_kPlan.mat` |
| `position_matrix_<ROI>_vtx<N>_kPlan.txt` | `<ROI>_vtx<N>_PositionMatrix_kPlan.txt` |
| `position_matrix_<ROI>_vtx<N>_Localite.mat` | `<ROI>_vtx<N>_PositionMatrix_Localite.mat` |
| `position_matrix_<ROI>_vtx<N>_Localite.txt` | `<ROI>_vtx<N>_PositionMatrix_Localite.txt` |
| `position_matrix_<ROI>_vtx<N>_Localite_XML.txt` | `<ROI>_vtx<N>_TransducerPosition_Localite_dummyXML.txt` |
| `position_matrix_<ROI>_vtx<N>_transducer.txt` | `<ROI>_vtx<N>_PositionMatrix_Transducer.txt` |
| `trajectory_<ROI>_vtx<N>_BabelBrain.txt` | `<ROI>_vtx<N>_Trajectory_BabelBrain.txt` |
| `trajectory_<ROI>_vtx<N>_Brainsight.txt` | `<ROI>_vtx<N>_Trajectory_Brainsight.txt` |
| `transducer_<ROI>_vtx<N>.surf.gii` | `<ROI>_vtx<N>_TransducerModel.surf.gii` |

**Connectome Workbench is no longer a dependency.** `wb_command`/`wb_view` are not called anywhere in the pipeline. If you were pointing PlanTUS at a Workbench install via a config path, that setting is no longer read.

**CLI flags renamed; one removed.**
| Old flag | New flag |
|---|---|
| `--do_only_trajectory <N>` | `--placement_only <N>` |
| `--skip_wb_view` | `--skip_viewer` |
| `--use_internal_viewer` | *(removed — the internal viewer is always used now, this flag was a no-op)* |

**The bundled device-specific transducer model has been removed.** PlanTUS previously bundled a NeuroFUS CTX-500-4 model (`resources/transducer_models/TRANSDUCER-NEUROFUS-CTX-500-4_DEVICE.surf.gii`) and let you choose between it and a generated generic cylinder via `bUseGenericTransducerModel` in your config. That config key and the bundled model are both gone — PlanTUS now always generates the generic cylindrical model, sized from `transducer_diameter`/`plane_offset`/`additional_offset`. If your config sets `bUseGenericTransducerModel`, it's simply ignored now (harmless, but you can remove it). The interactive viewer's live preview also now matches this model's real dimensions, rather than a fixed placeholder size.

### New dependencies

- `vtk`, `trimesh`, `nibabel` — surface/volume I/O and mesh processing (replacing Workbench)
- `potpourri3d` *(optional)* — exact geodesic distances for surface-avoidance-mask erosion; falls back to an approximate method if not installed

### New: interactive internal viewer

The planning step now opens PlanTUS' own PyQt5/VTK viewer by default (no external viewer required):

- **Right-click** directly on the head surface to select a candidate transducer placement — no separate "selection mode" toggle needed.
- A live, roughly-to-scale transducer model (body + handle/cable indicator) previews the actual placement and orientation, with a **rotation slider** to adjust roll around the beam axis before generating.
- Two oblique **volume views** (aligned to the trajectory) show the intracranial path and estimated acoustic focus (an ellipsoid sized from your transducer's focal-distance/FLHM calibration) alongside the target ROI, overlaid in green.
- **"Save Placement"** writes the output files for the current selection without closing the window — pick and save multiple candidate placements in one session.
- Placement markers (dots) on the head surface accumulate across picks; "Remove Placement Markers" clears them, "Remove Transducer Model" clears just the live preview model.
- The initially-suggested placement is the vertex with the best composite metric score; you're free to pick a different one.
- Distance and Target Intersection maps auto-scale to their own data range; colors use `jet` throughout.

### New: `output_folder` config option

Optionally set `output_folder` in your config YAML to write PlanTUS' `PlanTUS/<ROI>` output folder somewhere other than the default (next to the subject's `.msh` file, in the m2m folder). Omit it to keep the previous default location.

### New: choice of beam orientation

Previously, the transducer's beam axis at a chosen skin vertex was always aimed exactly at the target ROI's center of gravity. You can now choose between two orientation modes via the viewer's **Orientation** dropdown (or the new `orientation_mode` argument to `PlanTUS.prepare_acoustic_simulation()`, for scripted/`--placement_only` use):

- **Towards target center** *(default, previous/only behavior)* — the beam axis is aimed exactly at the target ROI's center of gravity, regardless of the local skin surface normal.
- **Along surface normal** — the beam axis is aimed along the (negated) local skin surface normal instead, which is not necessarily aimed at the target center.

The live preview (transducer glyph, volume/focus view) updates immediately when you switch modes, matching what will actually be generated.

**BabelBrain export note:** BabelBrain's trajectory format anchors the placement at a single coordinate, with the beam axis defining the trajectory direction from there. In "Towards target center" mode this anchor is the target ROI's center of gravity, as before — self-consistent, since the axis passes through exactly that point by construction. In "Along surface normal" mode, anchoring at the target center would be inconsistent (axis and anchor would describe two different lines), so the anchor is instead: the midpoint of the beam's intersection with the target ROI, if it intersects at all; otherwise a point along the trajectory at the estimated focal distance.

### Fixed

- Skull-thickness computation no longer re-extracts the skull surface from the subject's `.msh` file a second time (was previously done redundantly).
- `compute_surface_metrics()` results are now cached per file (keyed by path + modification time), avoiding repeated recomputation of the same surface's normals within a single run.
- Small ROI meshes (e.g. subcortical targets) no longer drift off-center during surface reconstruction — mesh smoothing switched from plain Laplacian to Taubin smoothing, which doesn't shrink/displace small meshes the way Laplacian smoothing does.
- Viewport camera clipping range is now recomputed correctly when switching between preset views (Top/Front/Lateral/Oblique), fixing cases where the geometry would disappear after switching views.
- Screenshot capture now correctly includes the 3D-rendered panels (previously only native Qt UI elements were captured).

### Citation

If you use PlanTUS, please cite:

> Lueckel, M., Vijayakumar, S., & Bergmann, T. O. (2025). PlanTUS: A heuristic tool for prospective planning of transcranial ultrasound transducer placements. *Brain Stimulation: Basic, Translational, and Clinical Research in Neuromodulation, 18*(5), 1563–1565. https://doi.org/10.1016/j.brs.2025.08.013
