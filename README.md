# IsoFATE

## About

***Please note: I am currrently working to package this code into a more usable format and expect to release the full package by the end of the summer 2025.***

Isotopic Fractionation via ATmospheric Escape (IsoFATE) is a Python code that models mass fractionation resulting from diffusive separation in escaping planetary atmospheres and numerically computes atmospheric species abundance over time. The model is currently tuned to sub-Neptune sized planets with rocky cores of Earth-like bulk composition and primordial H/He atmospheres. F, G, K, and M type stellar fluxes are readily implemented. This code is not yet a Python package, so no installation is required. Simply download the source files to use.

Version 1 can simulate a ternary mixture of H, He, and D (deuterium). Version 2 is coupled to the magma ocean-atmosphere equilibrium chemistry model Atmodeller.

Author: Collin Cherubim

## Citation

If you use IsoFATE, please cite:

- Cherubim, C.; Wordsworth, R., Hu, R.; Shkolnik, E. (2024). Strong fractionation of deuterium and helium in sub-Neptune atmospheres along the the radius valley, ApJ (https://ui.adsabs.harvard.edu/abs/2024arXiv240210690C/abstract)

## Tutorial

There are no formal tutorials for IsoFATE at this time. Please contact me directly at ccherubim@g.harvard.edu for support.

Use "isofate_binary.py" for binary atmospheric mixture of deuterium and helium or hydrogen and helium. Use "isofate_ternary.py" for a ternary mixture of hydrogen, deuterium, and helium. "constants.py," "isofunks.py," and "orbit_params.py" contain physical constants and supporting functions required to run the main isofate_.py scripts. 
