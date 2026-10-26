# IsoFATE

## About

Isotopic Fractionation via ATmospheric Escape (IsoFATE) is a Python code that models mass fractionation resulting from diffusive separation in escaping planetary atmospheres and numerically computes atmospheric species abundance over time. The model includes core-powered mass loss and photoevaporation in the EUV-limited Lyman alpha radiative/recombination-limited regimes. IsoFATE is now coupled to [Atmodeller](https://github.com/ExPlanetology/atmodeller), capturing the fully time-resolved transition of sub-Neptunes to terrestrial super-Earths, including equilibrium chemistry (Gibbs free energy minimization) and interior-atmosphere exchange through a time-evolving magma ocean. Sub-Neptunes are assumed to initially have rocky cores of Earth-like bulk composition and primordial H/He atmospheres, though this is easily changeable. F, G, K, and M type stellar fluxes are readily implemented.

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

The engine of the code is in the ```isocalc``` function in isofate_coupler.py. Files "constants.py," "isofunks.py," and "orbit_params.py" contain physical constants and supporting functions. Atmodeller is called with atmodeller_coupler.py. Each timestep of ```isocalc``` runs in escape_kernel.py, which Numba compiles on the first call in each Python session (a few seconds); its physics must stay in step with isofunks.py, which tests/test_escape_kernel.py checks.

Each Atmodeller call is one compiled solve that starts from the previous call's solution, and falls back to Atmodeller's own solve (cold start and multistart) when it does not converge. It relies on Atmodeller 0.9.1 internals; tests/test_atmodeller_coupler.py checks it against Atmodeller.

## Monte Carlo runs

```python -m isofate.mc_timeseries OUT.npz --n-planets N --seed S --n-shards K --shard I --workers W --pin```

runs the M-dwarf XUV+RR population of Cherubim et al. 2025 and saves each planet's time series at the Atmodeller steps. Every shard draws the same N planets from the seed and runs every K-th one, so an AWS Batch array job of K children covers the population (```--shard``` defaults to ```AWS_BATCH_JOB_ARRAY_INDEX```).

- A planet takes about 0.3-0.5 s per CPU with ```--pin``` (one CPU per worker), and about twice that without it. Pin only when the job has its CPUs to itself.
- Each process spends about 10 s compiling on its first planet, so give each one many planets. Pointing ```JAX_COMPILATION_CACHE_DIR``` at a directory that persists between runs saves the Atmodeller part of it.
- ```benchmarks/bench_isocalc.py mc_timeseries``` times one planet of this population.

## Citation

If you use IsoFATE, please cite:

- Cherubim, C.; Wordsworth, R.; Hu, R.; Shkolnik, E. (2024). Strong fractionation of deuterium and helium in sub-Neptune atmospheres along the the radius valley, ApJ (https://ui.adsabs.harvard.edu/abs/2024arXiv240210690C/abstract)

and

- Cherubim, C.; Wordsworth, R.; Bower, D.; Sossi, P.; Adams, D.; Hu, R. (2025) An Oxidation Gradient Straddling the Small Planet Radius Valley, ApJ (https://ui.adsabs.harvard.edu/abs/2025ApJ...983...97C/abstract)
