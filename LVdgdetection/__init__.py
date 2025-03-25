"""
This package implements PhotCalib to calibrate CaHK photometric data from the Prisitne survey.

Content
-------

The package mainly contains:
  dwarf galaxy datection                    return the cut image of candidates 
  dwarf galaxy classification               return ViT result
"""
__version__ = "0.1" 
from .dg_finder_header import divide_sky,pool_sky,show_mask_,n1writedg_reg,_mask_,bkgwrite_candi,load_reg
