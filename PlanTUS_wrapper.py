#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PlanTUS wrapper script
======================
This script is just a wrapper script around PlanTUS_cli to keep backwards
compatibility and for BabelBrain usage.

(The parameters from PlanTUS_cli will change in the future).

"""
import argparse
from PlanTUS.PlanTUS_cli import run

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PlanTUS wrapper")
    parser.add_argument("t1", type=str, help="Path to T1 image")
    parser.add_argument("mesh", type=str,  help="Path to head mesh")
    parser.add_argument("roi", type=str, help="Path to target ROI")
    parser.add_argument("config", type=str, help="Path to configuration file")
    parser.add_argument("--skip_wb_view",action="store_true",help="Run calculations but skip wb_view")
    parser.add_argument("--use_internal_viewer",action="store_true",help="Use own viewer instead of wb_view")
    parser.add_argument("--do_only_trajectory",type=int,default=-1,help="Optional integer to run only the generation of trajectory (default: -1). Specify number of triangles to generate.")

    args = parser.parse_args()

    run(args)

