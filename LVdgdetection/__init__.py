"""
This package implements PhotCalib to calibrate CaHK photometric data from the Prisitne survey.

Content
-------

The package mainly contains:
  dwarf galaxy datection                    return the cut image of candidates 
  dwarf galaxy classification               return ViT result
"""
from .dg_finder_header import deform
from .training import TrainingModule
from .make_plots import make_diagnostic_plots
from .apply_calib import generate_newcat
from .set_argparse import argparse_train_model, argparse_apply_model
