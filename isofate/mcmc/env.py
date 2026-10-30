'''
Process environment for isocalc: single-thread settings and on-disk compile caches.

The MCMC gets its parallelism from the worker processes, each running one model evaluation at a time, so every
thread pool inside a process is cut to one thread. set_single_thread_env() must run before isofate or jax is
imported in any process that runs isocalc. With the defaults, XLA spreads each worker over several threads, which
costs about a third of the throughput of a 6-worker pool on the 8-core M1 laptop. Spawned workers inherit the
parent's environment.

The compile caches keep the Numba kernels of the escape step and the JAX executables of the Atmodeller solve in
<output_root>/.cache/, so a new worker loads them from disk instead of compiling them again.
'''

import glob
import hashlib
import importlib.metadata
import os
import sys

# Thread-pool sizes of XLA's CPU backend (NPROC), OpenMP, Numba, OpenBLAS, MKL and Apple's Accelerate (vecLib)
SINGLE_THREAD_ENV = {'NPROC': '1', 'OMP_NUM_THREADS': '1', 'NUMBA_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
                     'MKL_NUM_THREADS': '1', 'VECLIB_MAXIMUM_THREADS': '1'}
# XLA flags: single-threaded Eigen on the CPU, one thread inside each operation
XLA_SINGLE_THREAD = '--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1'


def set_single_thread_env():
    '''
    Sets NPROC=1 (the XLA CPU thread pool size, the variable that matters) and the other single-thread variables
    Also adds the XLA flags to XLA_FLAGS, unless XLA_FLAGS already sets --xla_cpu_multi_thread_eigen (a value set
    by hand wins). Has no effect on a jax that is already imported in this process; processes started afterwards
    (the spawned workers) inherit the settings.
    '''
    os.environ.update(SINGLE_THREAD_ENV)
    xla_flags = os.environ.get('XLA_FLAGS', '')
    if '--xla_cpu_multi_thread_eigen' not in xla_flags:
        os.environ['XLA_FLAGS'] = (xla_flags + ' ' + XLA_SINGLE_THREAD).strip()


def cache_dir(cfg):
    '''
    Root of the compile caches, <output_root>/.cache
    '''
    return os.path.join(cfg['output_root'], '.cache')


def jax_cache_dir(cfg):
    '''
    JAX persistent compilation cache, <output_root>/.cache/jax
    '''
    return os.path.join(cache_dir(cfg), 'jax')


def numba_cache_dir(cfg):
    '''
    Numba cache directory, keyed on the contents of isofate/*.py and the numba version
    Numba only invalidates a cache when the file defining a function changes, but escape_kernel.run compiles in
    functions from isofunks.py, and Numba freezes global constants (those of constants.py) into the compiled code,
    so the key has to cover every module. Editing any isofate module therefore costs one recompile, into a new
    directory; the old one stays behind (about 0.4 MB).

    Inputs:
        - cfg: resolved config; only output_root is used [dict]

    Output: <output_root>/.cache/numba-<first 12 hex digits of the SHA-1 key> [str]
    '''
    import isofate
    sha = hashlib.sha1()
    # Sorted, so the key does not depend on the order in which glob lists the files
    for path in sorted(glob.glob(os.path.join(os.path.dirname(isofate.__file__), '*.py'))):
        with open(path, 'rb') as f:
            sha.update(f.read())
    sha.update(importlib.metadata.version('numba').encode()) # a numba upgrade also starts a fresh cache
    return os.path.join(cache_dir(cfg), 'numba-' + sha.hexdigest()[:12])


def enable_compile_caches(cfg):
    '''
    Turns on the JAX persistent compilation cache and Numba on-disk caching of the isocalc kernels
    Call it before the first isocalc call in the process. It imports jax and the coupled model, so use it only
    where the model runs: in a worker, or in check and synth, which run the model in their own process. Together
    the caches cut a worker's warm-up from about 6 s to 2.4 s, for about 1.5 MB on disk.

    Inputs:
        - cfg: resolved config; only output_root is used [dict]
    '''
    jax_dir, numba_dir = jax_cache_dir(cfg), numba_cache_dir(cfg)
    os.makedirs(jax_dir, exist_ok = True)
    os.makedirs(numba_dir, exist_ok = True)
    os.environ['NUMBA_CACHE_DIR'] = numba_dir # numba reads it when it is imported
    import numba
    numba.config.CACHE_DIR = numba_dir # in case numba was imported before the variable was set
    import jax
    # Cache every executable, however quick to compile or small (by default JAX skips those)
    jax.config.update('jax_compilation_cache_dir', jax_dir)
    jax.config.update('jax_persistent_cache_min_compile_time_secs', 0.0)
    jax.config.update('jax_persistent_cache_min_entry_size_bytes', -1)
    # The kernels are plain @njit functions (no cache = True), so caching is switched on dispatcher by dispatcher
    from isofate import escape_kernel
    from isofate import atmodeller_coupler
    for dispatcher in (escape_kernel.run, escape_kernel.fluxes, escape_kernel.advance,
                       atmodeller_coupler._atmosphere_descent, atmodeller_coupler._interpolate_melt_fraction):
        dispatcher.enable_caching()


def heavy_modules_loaded():
    '''
    Names of the heavy modules (jax, atmodeller, the coupled model) imported in this process so far
    A quick check that the parent of a worker pool stays light: there it should be [].

    Output: module names [list of str]
    '''
    return [name for name in ('jax', 'atmodeller', 'isofate.isofate_coupler') if name in sys.modules]
