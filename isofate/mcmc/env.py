'''
Process environment for isocalc: single-thread settings and on-disk compile caches.

set_single_thread_env() must run before isofate or jax is imported in any process that runs isocalc. With the
defaults, XLA spreads each worker over several threads, which costs about a third of the throughput of a 6-worker
pool on this machine (report_parallel.md section 2). Spawned workers inherit the parent's environment.
'''

import glob
import hashlib
import importlib.metadata
import os
import sys

SINGLE_THREAD_ENV = {'NPROC': '1', 'OMP_NUM_THREADS': '1', 'NUMBA_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
                     'MKL_NUM_THREADS': '1', 'VECLIB_MAXIMUM_THREADS': '1'}
XLA_SINGLE_THREAD = '--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1'


def set_single_thread_env():
    '''
    Sets NPROC=1 (the XLA CPU thread pool size, the variable that matters) and the other single-thread variables.
    Has no effect on a jax that is already imported in this process.
    '''
    os.environ.update(SINGLE_THREAD_ENV)
    xla_flags = os.environ.get('XLA_FLAGS', '')
    if '--xla_cpu_multi_thread_eigen' not in xla_flags:
        os.environ['XLA_FLAGS'] = (xla_flags + ' ' + XLA_SINGLE_THREAD).strip()


def cache_dir(cfg):
    '''
    Root of the compile caches, <output_root>/.cache.
    '''
    return os.path.join(cfg['output_root'], '.cache')


def jax_cache_dir(cfg):
    return os.path.join(cache_dir(cfg), 'jax')


def numba_cache_dir(cfg):
    '''
    Numba cache directory keyed on the contents of isofate/*.py and the numba version. Numba only invalidates a
    cache when the file defining a function changes, but escape_kernel.run inlines functions from isofunks.py,
    so the key has to cover every module.
    '''
    import isofate
    sha = hashlib.sha1()
    for path in sorted(glob.glob(os.path.join(os.path.dirname(isofate.__file__), '*.py'))):
        with open(path, 'rb') as f:
            sha.update(f.read())
    sha.update(importlib.metadata.version('numba').encode())
    return os.path.join(cache_dir(cfg), 'numba-' + sha.hexdigest()[:12])


def enable_compile_caches(cfg):
    '''
    Turns on the JAX persistent compilation cache and Numba on-disk caching of the isocalc kernels. Call it before
    the first isocalc call in the process. Imports jax and the coupled model, so use it in workers only.
    Together the caches cut a worker's warm-up from about 6 s to 2.4 s (about 1 MB on disk).
    '''
    jax_dir, numba_dir = jax_cache_dir(cfg), numba_cache_dir(cfg)
    os.makedirs(jax_dir, exist_ok = True)
    os.makedirs(numba_dir, exist_ok = True)
    os.environ['NUMBA_CACHE_DIR'] = numba_dir
    import numba
    numba.config.CACHE_DIR = numba_dir # in case numba was imported before the variable was set
    import jax
    jax.config.update('jax_compilation_cache_dir', jax_dir)
    jax.config.update('jax_persistent_cache_min_compile_time_secs', 0.0)
    jax.config.update('jax_persistent_cache_min_entry_size_bytes', -1)
    from isofate import escape_kernel
    from isofate import atmodeller_coupler
    for dispatcher in (escape_kernel.run, escape_kernel.fluxes, escape_kernel.advance,
                       atmodeller_coupler._atmosphere_descent, atmodeller_coupler._interpolate_melt_fraction):
        dispatcher.enable_caching()


def heavy_modules_loaded():
    '''
    Names of the heavy modules (jax, atmodeller, the coupled model) imported in this process so far.
    '''
    return [name for name in ('jax', 'atmodeller', 'isofate.isofate_coupler') if name in sys.modules]
