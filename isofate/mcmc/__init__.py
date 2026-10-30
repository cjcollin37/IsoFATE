'''
isofate.mcmc: config-driven emcee fits (Foreman-Mackey et al 2013) of coupled IsoFATE-Atmodeller evolutionary
tracks to the observables of one planet. Run `python -m isofate.mcmc --help` for the command line tool; README.md
in this folder is the manual.

Importing this package is light on purpose (no jax, atmodeller or isofate_coupler), so the parent process of a
worker pool stays small. The heavy model is imported lazily inside the workers (see model.py).

Modules:
    - config: TOML loading and validation, the PARAMS and OBSERVABLES registries, the resolved config.json
    - params, likelihood: priors and theta <-> parameters; the log-likelihood terms of the observables
    - model: one evaluation (isocalc inputs, observables of the final state, blobs, status codes), worker functions
    - env, pool: single-thread environment and compile caches; the spawn worker pool
    - sampler: the MCMC run (initialization, sampling, checkpoints, resume, convergence, summary)
    - search, solve: the root finder of the bisection initializer, and solve mode for thin ridges
    - scan, resolution, tracks, plots: grid scans, the resolution study, posterior tracks, figures
    - toy: a fast fake isocalc for testing the tool
'''

__version__ = '0.1.0' # version of the tool, recorded in every run's provenance

# Submodules that `from isofate.mcmc import *` imports
__all__ = ['env', 'config', 'params', 'likelihood', 'model', 'toy', 'pool', 'sampler', 'tracks', 'resolution',
           'plots']
