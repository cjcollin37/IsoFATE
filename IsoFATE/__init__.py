import os

__version__ = "0.0.1"

# set Python env variable to keep track of example data dir
orbitize_dir = os.path.dirname(__file__)
# DATADIR = os.path.join(orbitize_dir, "/Users/collin/Documents/Harvard/Research/atm_escape/RotationXUVTracks/TrackGrid_MstarPercentile/")
DATADIR = os.path.join(orbitize_dir, "data/") #need to put the 0p files here