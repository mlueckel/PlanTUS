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

**Exported poses are not identical to v1 for the same vertex.** (a) The rotation around the beam axis was random in v1; it is now deterministic and set with the viewer's rotation slider. (b) In v1 the beam axis was chosen automatically per vertex (along the skin normal where that line passed through the target, otherwise towards the target's center); v2 uses the Orientation setting (default "Towards target center") for every vertex. Position matrices, `.kps`, Localite, Brainsight and BabelBrain files from v1 and v2 therefore differ even when the vertex is the same.

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

In v1 the transducer's beam axis at a chosen skin vertex was picked automatically (along the skin normal where that line passed through the target, towards the target's center of gravity otherwise). You can now choose between two orientation modes via the viewer's **Orientation** dropdown (or the new `orientation_mode` argument to `PlanTUS.prepare_acoustic_simulation()`, for scripted/`--placement_only` use):

- **Towards target center** *(default)* — the beam axis is aimed exactly at the target ROI's center of gravity, regardless of the local skin surface normal.
- **Along surface normal** — the beam axis is aimed along the (negated) local skin surface normal instead, which is not necessarily aimed at the target center.

The live preview (transducer glyph, volume/focus view) updates immediately when you switch modes, matching what will actually be generated.

**BabelBrain export note:** BabelBrain's trajectory format anchors the placement at a single coordinate, with the beam axis defining the trajectory direction from there. In "Towards target center" mode this anchor is the target ROI's center of gravity, as before — self-consistent, since the axis passes through exactly that point by construction. In "Along surface normal" mode, anchoring at the target center would be inconsistent (axis and anchor would describe two different lines), so the anchor is instead: the midpoint of the beam's intersection with the target ROI, if it intersects at all; otherwise a point along the trajectory at the estimated focal distance.

### New: skin smoothing option

Optional config key `skin_smoothing_iterations` (default `2`; `0` disables): the extracted skin surface is lightly Taubin-smoothed before any metric is computed on it, which reduces segmentation-level mesh noise in the per-vertex normals (and so in the tilt/angle maps) without shrinking the surface. The skull surface is not smoothed.

### New: viewer additions

- **Skull Thickness** and **Composite Score** maps are loaded and selectable in every panel's dropdown, but not shown by default (the four panels still open on Target Distance, Target Intersection, Transducer Tilt and Skin-Skull Angle).
- The Transducer Tilt color scale now runs from 0° to the `max_angle` of your config; Skull Thickness is fixed to 0–10 mm and Composite Score to 0–1.
- "Remove Transducer Model" no longer resets the Orientation dropdown (it still resets the rotation slider).
- Toolbar layout: equal spacing between control groups; the Views buttons and the Screenshot button are right-aligned.
- The viewer prints one startup line with the VTK version and where the T1 sits in world space.
- The BabelBrain trajectory is named `<ROI name>_<vertex>` inside the file.

### New: optional Workbench scene files

PlanTUS still writes Connectome Workbench `scene.scene` files for anyone who wants to browse results there: one for the whole-head metric maps (in the ROI output folder) and one per saved placement (in the placement folder). Nothing in PlanTUS opens or requires Workbench, and files are referenced by absolute path, so regenerate the scene if you move the output folder.

### New: T1/mesh consistency check

PlanTUS now warns if the skin surface does not lie inside the T1's world-space extent, i.e. the T1 passed is probably not the one `charm` was run on. It catches gross mismatches (tens of mm), not subtle ones, and never stops the run.

### Fixed

- Skull-thickness computation no longer re-extracts the skull surface from the subject's `.msh` file a second time (was previously done redundantly).
- `compute_surface_metrics()` results are now cached per file (keyed by path + modification time), avoiding repeated recomputation of the same surface's normals within a single run.
- Small ROI meshes (e.g. subcortical targets) no longer drift off-center during surface reconstruction — mesh smoothing switched from plain Laplacian to Taubin smoothing, which doesn't shrink/displace small meshes the way Laplacian smoothing does.
- Viewport camera clipping range is now recomputed correctly when switching between preset views (Top/Front/Lateral/Oblique), fixing cases where the geometry would disappear after switching views.
- Screenshot capture now correctly includes the 3D-rendered panels (previously only native Qt UI elements were captured).
- **Volume slice views were misplaced relative to the surfaces.** `vtkNIFTIImageReader` returns image data in raw voxel-index space and reports the file's orientation separately; the viewer ignored it, so the T1/ROI slices sat far from the mesh-space markers and trajectory (171 mm on one example T1), the oblique planes missed the volume or cut the wrong tissue, and the camera framed the wrong region. The T1 and ROI are now positioned with their NIfTI orientation matrix (sform, else qform) and the camera uses world-space bounds. Verified at voxel level against nibabel's affine for flipped, axis-permuted, oblique and sform≠qform geometries on VTK 9.3–9.7; the renderer itself could not be tested headless.
- **Air cavities (and the no-go areas above them) were missed.** Three causes: (1) when the segmentation's field of view is cropped through the neck, the airway connects to the image edge, so faces cropped through tissue are now sealed before hole filling; (2) the sinuses and nasal cavity are open to the outside through the nostrils and airway, so they are not holes in 3-D. A slice-wise 2-D fill along the axial axis (found from the affine, so any orientation works) is now added. On the SimNIBS example subject *ernie* the detected air rose from 7,876 to 98,983 voxels, and on a second subject it added ~14.6k voxels at the location of the mastoid air cells; (3) the "below head height" cutoff was `mean_eye_z / 2`, which depends on where the coordinate origin sits (on one subject it landed above the eyes and excluded 29% of the head). It is now halfway between eye level and the lowest point of the skin surface. How much of the forehead ends up excluded in the final mask also depends on the safety margin (`transducer_diameter / 2.5` mm); with a small margin the old detection allowed placements directly over the frontal sinus.
- **Ray casting gave wrong intersections on VTK 9.7.0 and 9.7.1.** In those releases `vtkOBBTree.IntersectWithLine` indexes a per-cell point buffer with global point ids (an out-of-bounds read; upstream `master` no longer has it), which produced missed *and* spurious hits that varied between runs, plus occasional crashes. `compute_vector_mesh_intersections` (behind target intersection, skull thickness, skin-skull angle, the air-cavity mask and focal-distance estimates) now uses `vtkStaticCellLocator`, with hits sorted along the ray and coincident hits merged. Results are identical on VTK 9.3.1, 9.4.2, 9.5.2 and 9.7.1 and match trimesh's ray casting. Older copies of PlanTUS on VTK 9.7.0/9.7.1 give wrong results; update PlanTUS or use VTK ≤ 9.6.2.
- **BabelBrain trajectory files at 0° rotation.** The in-plane reference used for the rotation was axis-aligned, which forces an exactly-zero entry in the exported matrix at 0°/180° (BabelBrain failed to load such trajectories). The reference is no longer axis-aligned, and exact 0°/180° rotations are exported with a 1° offset as a precaution (the slider still reads 0°). The placeholder-name replacement in the BabelBrain file (`000`) also rewrote any number containing that digit sequence; it now only replaces the standalone token.
- A target ROI that no skull ray reaches no longer crashes the wrapper with `ValueError: need at least one array to concatenate`; the auxiliary `skin_skull_target_intersection` map is left empty and a warning explains why.

### Citation

If you use PlanTUS, please cite:

> Lueckel, M., Vijayakumar, S., & Bergmann, T. O. (2025). PlanTUS: A heuristic tool for prospective planning of transcranial ultrasound transducer placements. *Brain Stimulation: Basic, Translational, and Clinical Research in Neuromodulation, 18*(5), 1563–1565. https://doi.org/10.1016/j.brs.2025.08.013
