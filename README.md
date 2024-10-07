# PlanTUS - A tool for heuristic planning of TUS transducer placement.

# Software dependencies:
- FSL
- Freesurfer
- Connectome Workbench
- SimNIBS


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

![image](https://github.com/user-attachments/assets/df3d85c4-4056-4bb6-99aa-23b82feb822d)


## 4. Evaluate transducer and (estimated) focus position

## 5. Use PlanTUS outputs with acoustic simulation software

