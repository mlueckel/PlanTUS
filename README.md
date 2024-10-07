# PlanTUS - Tool for heuristic planning of TUS transducer placement.

# Software dependencies:
- FSL
- Freesurfer
- Connectome Workbench
- SimNIBS


# Instructions

## Before using PlanTUS

### What you need
In order to use PlanTUS, you need the following files:
- T1-weighted (T1w) MR image of your participant's head
- (optinally: T2-weighted (T2w) MR image of your participant's head)
- Mask of your region of interest (co-registered to, i.e., in the same space as, your participant's T1 image)
  
### Charm
Make sure to run the SimNIBS charm pipeline (https://simnibs.github.io/simnibs/build/html/documentation/command_line/charm.html) on your participant's T1w (and T2w) MR image.

**Note:** If you want to use the PlanTUS output (i.e., the planned transducer position) in k-Plan, make sure to linearly co-register your participant's T1w MR image to a suitable MNI template and set the left, posterior, inferior corner of the image to (0,0,0). You can use the Python script that is provided in this repository.
