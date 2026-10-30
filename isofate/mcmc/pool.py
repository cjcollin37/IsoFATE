'''
Spawn worker pool for isocalc: warm-up behind a barrier, a memory-aware worker count, and chunksize=1 maps.

Measured on this machine (report_parallel.md): a worker holds 0.65-0.86 GB at n_steps = 1e5 and 1.2-1.9 GB at
1e6; warm-up takes 6-8 s per worker without the compile caches and about 2.4 s with them. Fork deadlocks once
the parent has imported JAX, so the pool always uses spawn; the parent should never import the coupled model.

A worker that dies during a task (killed by the OS for memory, or a crash in compiled code) is replaced by
multiprocessing.Pool, but its task never returns, so a plain pool.map would wait forever. map and imap_unordered
therefore wait in short slices and raise WorkerDied as soon as a worker process has gone. A worker that dies while
idle is just as bad: it dies holding the lock of the task queue (an idle worker waits for its next task with that
lock held), so its replacement and the other workers can never take a task again. Hence any change of the worker
set since the pool started counts as a death (the pool never replaces a live worker: maxtasksperchild is unset),
also between two maps.

The same lost lock makes multiprocessing.Pool.terminate wait forever (it takes that lock before it stops the
workers), e.g. after a SIGTERM to the whole process group, which kills the workers too. WorkerPool.terminate
therefore runs it in a thread and kills (SIGKILL) whatever is left after TEARDOWN_S. Workers ignore SIGINT and
SIGHUP, so Ctrl-C and closing the terminal reach the parent alone (it checkpoints, then stops them); SIGTERM keeps
its default, because Pool.terminate stops the workers with it.
'''

import math
import multiprocessing as mp
import signal
import threading
import time

from isofate.mcmc import env, model

POLL = 2.0 # [s] how often a waiting map checks that its workers are alive
TEARDOWN_S = 10.0 # [s] terminate() waits this long for multiprocessing's own teardown, then kills the workers


class WorkerDied(RuntimeError):
    '''
    A worker process died (during an evaluation, or while idle between two maps): its task, or the queue's lock it
    held, is lost.
    '''


def est_worker_gb(n_steps, backend = 'isocalc'):
    '''
    Estimated peak memory of one worker [GB].
    '''
    if backend == 'toy':
        return 0.15
    return 0.75 + 1.1e-6*n_steps


def choose_workers(requested, n_steps, cfg):
    '''
    (n, reason): the requested worker count (default parallel.workers), reduced to fit the memory budget
    min(max_memory_gb, 70% of the available memory), or max_memory_gb alone when parallel.check_available is
    false. Prints the reason when it reduces.

    psutil's 'available' on macOS is free + inactive pages only: it leaves out what the system frees by
    compressing idle memory, and reads 2-4 GB on a 16 GB Mac with a browser open. Measured there (2026-10-28):
    6 workers at n_steps = 1e5 held 2.5 GB in total and pushed 0.8 GB to swap, while this rule allowed 2.
    '''
    import psutil
    par = cfg['parallel']
    requested = par['workers'] if requested is None else int(requested)
    est = est_worker_gb(n_steps, cfg['model']['backend'])
    available = psutil.virtual_memory().available/1e9 # [GB]
    check_available = par.get('check_available', True) # absent in configs saved before the option existed
    budget = min(par['max_memory_gb'], 0.7*available) if check_available else par['max_memory_gb'] # [GB]
    n = max(1, min(requested, math.floor(budget/est)))
    if check_available:
        source = f'min(max_memory_gb = {par["max_memory_gb"]:g}, 70% of {available:.1f} GB available)'
    else:
        source = f'max_memory_gb ({available:.1f} GB available, not checked: parallel.check_available = false)'
    reason = (f'{n} of {requested} workers: about {est:.2f} GB each at n_steps = {n_steps:.3g}, budget '
              f'{budget:.1f} GB = {source}')
    if n < requested:
        hint = ''
        if check_available and 0.7*available < par['max_memory_gb']:
            hint = ('\n  (macOS counts compressible memory as used; --set parallel.check_available=false budgets '
                    'max_memory_gb alone)')
        print(f'reducing workers: {reason}{hint}', flush = True)
    return n, reason


def _start_worker(cfg, run_dir, barrier):
    '''
    Pool initializer: workers ignore SIGINT and SIGHUP, then model.init_worker. Ctrl-C in a terminal, and the
    hang-up when the terminal closes, reach every process of the group; the parent alone handles them (checkpoint,
    then terminate the pool). Workers that raised KeyboardInterrupt too would only print tracebacks and emcee's
    "Exception while calling your likelihood function" for each running evaluation, and workers killed by SIGHUP
    could die holding the task queue's lock (module docstring). SIGTERM keeps its default: Pool.terminate uses it.
    '''
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    model.init_worker(cfg, run_dir, barrier)


class WorkerPool:
    '''
    Context manager around a spawn multiprocessing.Pool whose workers ran model.init_worker (env, caches,
    warm-up) before the constructor returns; they ignore SIGINT (Ctrl-C is the parent's). Usable as an emcee
    pool (.map).

        with WorkerPool(cfg, workers, n_steps_max, run_dir) as pool:
            results = pool.map(model.log_prob, thetas)

    The worker count is capped by choose_workers(workers, n_steps_max, cfg); use pool.n for the actual count.
    verbose = False skips the 'workers ready' line (the caller logs it).
    '''

    def __init__(self, cfg, workers, n_steps_max = None, run_dir = None, verbose = True):
        n_steps_max = cfg['model']['n_steps'] if n_steps_max is None else n_steps_max
        self.n, self.reason = choose_workers(workers, n_steps_max, cfg)
        env.set_single_thread_env() # spawned workers inherit the environment
        context = mp.get_context('spawn')
        barrier = context.Barrier(self.n + 1)
        start = time.perf_counter()
        self._pool = context.Pool(self.n, initializer = _start_worker, initargs = (cfg, run_dir, barrier))
        try:
            barrier.wait(timeout = 900)
        except threading.BrokenBarrierError:
            self.terminate()
            raise RuntimeError('worker start-up failed (see the traceback above) or took longer than 900 s') from None
        except BaseException:
            self.terminate()
            raise
        barrier.abort() # a worker the Pool respawns later (after a crash) must not wait for the others
        self._known = frozenset(self._pids()) # the workers that started: any change later is a death (_check)
        self.warmup_s = time.perf_counter() - start # [s]
        if verbose:
            print(f'{self.n} worker{"s" if self.n > 1 else ""} ready in {self.warmup_s:.1f} s', flush = True)

    def _pids(self):
        return {p.pid for p in self._pool._pool}

    def _check(self):
        '''
        Raises WorkerDied if a worker has exited since the pool started: the Pool has replaced it, so its pid has
        gone, or it is still listed with an exit code. Worker sets are compared with the start, not with the last
        map: a worker that died while idle leaves the others waiting for its lock (module docstring), and its
        replacement would otherwise look like a healthy worker.
        '''
        dead = (self._known - self._pids()) | {p.pid for p in self._pool._pool if p.exitcode is not None}
        if dead:
            raise WorkerDied(f'worker process(es) {sorted(dead)} died (killed by the OS, e.g. for memory, killed by a '
                             'signal, or crashed), so their tasks are lost and the pool cannot go on. An MCMC run '
                             'saves its chain up to the last step: python -m isofate.mcmc run --resume RUN_DIR (with '
                             'fewer --workers if memory ran out)')

    def map(self, f, iterable):
        '''
        Ordered results of f over iterable, one task at a time per worker (evaluation times vary 50x). Raises
        WorkerDied when a worker has died, before or while the map runs.
        '''
        self._check() # a death between two maps would leave this map waiting forever
        result = self._pool.map_async(f, list(iterable), chunksize = 1)
        while True:
            try:
                return result.get(timeout = POLL)
            except mp.TimeoutError:
                self._check()

    def imap_unordered(self, f, iterable):
        '''
        Results of f over iterable as they finish (a generator). Raises WorkerDied when a worker has died, before or
        while it runs.
        '''
        self._check()
        it = self._pool.imap_unordered(f, iterable, chunksize = 1)
        while True:
            try:
                yield it.next(timeout = POLL)
            except StopIteration:
                return
            except mp.TimeoutError:
                self._check()

    def close(self):
        '''
        Lets the workers finish their tasks, then joins them.
        '''
        self._pool.close()
        self._pool.join()

    def terminate(self, timeout = None):
        '''
        Stops the workers and joins them within about timeout [s] (default TEARDOWN_S) + 1 s.
        multiprocessing.Pool.terminate runs in a daemon thread, because it can wait forever for the task queue's
        lock when a worker died holding it (module docstring); workers still alive after timeout are killed
        (SIGKILL). Returns True when Pool.terminate finished by itself, False when the workers had to be killed
        (the thread is then left behind, blocked).
        '''
        pool, done = self._pool, threading.Event()
        timeout = TEARDOWN_S if timeout is None else timeout

        def stop():
            try:
                pool.terminate()
                pool.join()
            except Exception: # the workers are killed below; the caller has more important things to report
                pass
            finally:
                done.set()

        threading.Thread(target = stop, name = 'WorkerPool.terminate', daemon = True).start()
        if done.wait(timeout):
            return True
        # Pool.terminate has stopped the worker handler first thing, so no worker is replaced from here on
        for p in list(pool._pool):
            if p.exitcode is None:
                try:
                    p.kill()
                except (OSError, ValueError, AttributeError):
                    pass
        for p in list(pool._pool):
            try:
                p.join(timeout = 1.0)
            except (OSError, ValueError, AssertionError):
                pass
        return False

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.terminate()
        return False


def worker_rss_gb(pool):
    '''
    Total resident memory of the pool's worker processes [GB] (WorkerPool or multiprocessing.Pool).
    '''
    import psutil
    processes = pool._pool._pool if isinstance(pool, WorkerPool) else pool._pool
    total = 0
    for p in processes:
        try:
            total += psutil.Process(p.pid).memory_info().rss
        except psutil.Error:
            pass
    return total/1e9
