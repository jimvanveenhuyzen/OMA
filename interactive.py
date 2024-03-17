import numpy as np
import matplotlib.pyplot as plt

from IPython.display import clear_output
from ipywidgets import HBox, VBox
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets
from collections import OrderedDict

from astropy.io import fits

import sys

try:
    filename = sys.argv[1]
except IndexError as err: 
    pass
# Load the data from data.astropy.org
filename = 'interactive_examples/gc_msx_e.fits'

if filename[-5:] != '.fits':
    filename = 'interactive_examples/gc_msx_e.fits'
    hdu = fits.open(filename)[0]
elif filename[:20] != 'interactive_examples':
    try:
        filename = 'interactive_examples/{}'.format(filename)
        hdu = fits.open(filename)[0]
    except:
        hdu = fits.open(filename)[0]
else:
    try:
        hdu = fits.open(filename)[0]
    except FileNotFoundError as err:
        pass
   
# For some of the plots:
from mpl_toolkits.axes_grid1 import make_axes_locatable

