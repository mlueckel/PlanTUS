# PlanTUS - A tool for heuristic planning of TUS transducer placement.

<img src="https://github.com/user-attachments/assets/ff3850a9-2ed8-43a8-b93c-f28a9d0d7c2e" width="20" /> **PlanTUS does not replace proper acoustic simulations. All transducer positions planned using PlanTUS should should always be validated using acoustic simulations!** <img src="https://github.com/user-attachments/assets/ff3850a9-2ed8-43a8-b93c-f28a9d0d7c2e" width="20" />



# Software dependencies:
- FSL (https://fsl.fmrib.ox.ac.uk/fsl/docs/#/install/index)
- Freesurfer (https://surfer.nmr.mgh.harvard.edu/fswiki/DownloadAndInstall)
- Connectome Workbench (https://humanconnectome.org/software/get-connectome-workbench)
- SimNIBS (https://simnibs.github.io/simnibs/build/html/installation/simnibs_installer.html)


# Instructions

## 0. Before using PlanTUS

### What you need
In order to use PlanTUS, you need the following files:
- T1-weighted (T1w) MR image of your participant's head
- (optinally: T2-weighted (T2w) MR image of your participant's head)
- Mask of your region of interest (co-registered to, i.e., in the same space as, your participant's T1 image)
  
### Charm
Make sure to run the SimNIBS charm pipeline (https://simnibs.github.io/simnibs/build/html/documentation/command_line/charm.html) on your participant's T1w (and T2w) MR image.

**Note:** If you want to use the PlanTUS output (i.e., the planned transducer position) in k-Plan, make sure to linearly co-register your participant's T1w MR image to a suitable MNI template and set the left, posterior, inferior corner of the image to (0,0,0). You can use the Python script that is provided in this repository.

## 1. Specify variables
Specify the variables in the *PlanTUS_wrapper.py* script (see script for example values).

**Subject-specific variables**
- *t1_filepath*: Path to T1 image (output of SimNIBS' charm).
- *simnibs_mesh_filepath*: Path to head mesh (.msh file generated by SimNIBS' charm).
- *target_roi_filepath*: Path to mask of target region of interest (in same space as T1 image; note: PlanTUS output folder will have same name as this file).

**Transducer-specific variables**
- *max_distance*: Maximum focal depth of transducer (in mm).
- *min_distance*: Minimum focal depth of transducer (in mm).
- *transducer_diameter*: Transducer aperture diameter (in mm).
- *max_angle*: Maximum allowed angle for tilting of TUS transducer (in degrees).
- *plane_offset*: Offset between radiating surface and exit plane of transducer (in mm).
- *additional_offset*: Additional offset between skin and exit plane of transducer (in mm; e.g., due to addtional gel/silicone pad).
- *focal_distance_list*, *flhm_list*: Focal distance and corresponding FLHM values (both in mm) according to, e.g., calibration report.


## 2. Run the *PlanTUS_wrapper.py* script

## 3. Select transducer position(s)

Based on a subject-specific anatomical MR image and mask of the target region, PlanTUS generates useful metrics (see image for details), visualized in Connectome Workbench on the 3D-reconstructed head surface, that help you to intuitively evaluate potential transducer positions.

<img src="https://github.com/user-attachments/assets/df3d85c4-4056-4bb6-99aa-23b82feb822d" width="600" />


Simply click on the head surface where you would like to place the transducer (white dot). The volume view (right) allows you to check the intersection between the target region and an idealized acoustic beam trajectory (straight line) going from that position into the brain.

<img src="https://github.com/user-attachments/assets/cf6c9517-e4d4-444f-97b5-d49475feafd9" width="600" />



## 4. Evaluate transducer and (estimated) focus position

After selecting a position, a new window pops up that shows the resulting transducer placement (left) and a simplified representation of the expected acoustic focus (red outline) overlaid on the target mask (green) and anatomical MR image (volume view on the right).

<img src="https://github.com/user-attachments/assets/44b4f69a-df07-47eb-8858-e0da64af2172" width="600" />


The oblique volume view (right) will help you to evaluate the expected on- vs. off-target stimulation in terms of overlap between the simplified acoustic focus and the target region – which is also quantified and reported by planTUS.

<img src="https://github.com/user-attachments/assets/72fd9a0a-f7dc-461f-82db-82ea601e2751" width="600" />



## 5. Use PlanTUS outputs with acoustic simulation software

planTUS outputs several files for further use with different…

acoustic simulation software – for validation of the selected transducer placement(s).

neuronavigation software – for MR-guided navigation of the transducer to the selected position(s).

<img src="https://github.com/user-attachments/assets/12315d1b-24ba-42bb-ab98-0d6b4bde651f" width="200" />


k-Plan example: Using the planTUS output files, the selected transducer placement can be easily imported into the k-Plan software for validating the heuristically selected transducer placement with proper acoustic simulations.

<img src="https://github.com/user-attachments/assets/ef43e905-1466-4dc3-9e69-ccefa266d8fa" width="600" />



## 6. Review simulation results

Eventually, acoustic simulation results (e.g., acoustic pressure maps) can be loaded and evaluated in the same environment. White outlines in the volume view (right) indicate the borders of the target region.

<img src="https://github.com/user-attachments/assets/77ef1860-d809-4b43-a60a-c712257150ee" width="600" />


Again, the oblique volume view (right) can help you to evaluate on- vs. off-target stimulation in terms of overlap between the simulated acoustic focus and the target region (indicated by white outline) – which can also be quantified by planTUS.

<img src="https://github.com/user-attachments/assets/b1502848-d858-41ee-93a7-7cc3ab297e0b" width="600" />


