# IsoFATE

## About

Isotopic Fractionation via ATmospheric Escape (IsoFATE) is a Python code that models mass fractionation resulting from diffusive separation in escaping planetary atmospheres and numerically computes atmospheric species abundance over time. The model includes core-powered mass loss and photoevaporation in the EUV-limited Lyman alpha radiative/recombination-limited regimes. IsoFATE is now coupled to [Atmodeller]([url](https://github.com/ExPlanetology/atmodeller)), capturing the fully time-resolved transition of sub-Neptunes to terrestrial super-Earths, including equilibrium chemistry (Gibbs free energy minimization) and interior-atmosphere exchange through a time-evolving magma ocean. Sub-Neptunes are assumed to initially have rocky cores of Earth-like bulk composition and primordial H/He atmospheres, though this is easily changeable. F, G, K, and M type stellar fluxes are readily implemented.

Author: Collin Cherubim

## Installation

IsoFATE is pip installable, though I strongly recommend using the package manager, uv, to avoid dependency issues. You can pipx install uv into an environemnt, or pip install uv or brew install uv:

```pip install uv```

Once you installed uv, run:

```uv pip install isofate```

That should do it!

## Tutorial

There are no formal tutorials for IsoFATE yet, but there will be soon! Please contact me directly at ccherubim@g.harvard.edu for support.

Run sim.py to get things up and running quickly.

The engine of the code is in the ```isocalc``` function in isofate_coupler.py. Files "constants.py," "isofunks.py," and "orbit_params.py" contain physical constants and supporting functions. Atmodeller is called with atmodeller_coupler.py.

## Citation

If you use IsoFATE, please cite:

- Cherubim, C.; Wordsworth, R.; Hu, R.; Shkolnik, E. (2024). Strong fractionation of deuterium and helium in sub-Neptune atmospheres along the the radius valley, ApJ (https://ui.adsabs.harvard.edu/abs/2024arXiv240210690C/abstract)

and

- Cherubim, C.; Wordsworth, R.; Bower, D.; Sossi, P.; Adams, D.; Hu, R. (2025) An Oxidation Gradient Straddling the Small Planet Radius Valley, ApJ (https://ui.adsabs.harvard.edu/abs/2025ApJ...983...97C/abstract)
