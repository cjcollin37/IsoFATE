'''
isofate.mcmc: config-driven emcee fits of coupled IsoFATE-Atmodeller evolutionary tracks to the observables of
one planet. Run `python -m isofate.mcmc --help` for the command line tool.

Importing this package is light on purpose (no jax, atmodeller or isofate_coupler), so the parent process of a
worker pool stays small. The heavy model is imported lazily inside the workers (see model.py).
'''

__version__ = '0.1.0'

__all__ = ['env', 'config', 'params', 'likelihood', 'model', 'toy', 'pool', 'sampler', 'tracks', 'resolution',
           'plots']
