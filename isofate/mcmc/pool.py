'''
Spawn worker pool for isocalc: warm-up behind a barrier, a memory-aware worker count, chunksize=1 maps,
dead-worker detection and a bounded teardown.

The pool always uses spawn: fork deadlocks once the parent has imported JAX (JAX is multithreaded, and a forked
child inherits locks held by threads it does not have). The parent should never import the coupled model anyway,
which would add about 0.6 GB of JAX to it. Measured on the 16 GB M1 laptop: a worker holds 0.65-0.86 GB at
n_steps = 1e5 and 1.2-1.9 GB at 1e6; its warm-up (imports, Numba and JAX compiles, the Atmodeller fallback paths)
takes 6-8 s without the compile caches and about 2.4 s with them. The constructor returns once every worker is
warm, so the compiles are paid by all workers at once, before the first map. Tasks go out one at a time
(chunksize=1) because evaluation times vary 50x, from 0.01 s for a stripped track to 0.4-0.6 s for an O-rich one.

A worker that dies during a task (killed by the OS for memory, or a crash in compiled code) is replaced by
multiprocessing.Pool, but its task never returns, so a plain pool.map would wait forever. map and imap_unordered
therefore wait in slices of POLL s and raise WorkerDied as soon as a worker process has gone. A worker that dies
while idle is just as bad: an idle worker waits for its next task with the lock of the task queue held, so it dies
holding that lock, and its replacement and the other workers can never take a task again. Hence any change of the
worker set since the pool started counts as a death, also between two maps (the pool never replaces a live worker:
maxtasksperchild is unset).

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
    A worker process died, during an evaluation or while idle between two maps
    Its task, or the lock of the task queue it held, is lost, so the pool cannot go on. A RuntimeError, so the
    command line prints the message (with the resume command) and exits with code 1.
    '''


###_____Worker count_____###

def est_worker_gb(n_steps, backend = 'isocalc'):
    '''
    Estimates the peak memory of one worker
    0.75 GB plus 1.1 GB per 1e6 time steps: 0.86 GB at n_steps = 1e5 and 1.85 GB at 1e6, the top of the measured
    ranges. The memory grows with n_steps because isocalc allocates its arrays over every time step, the molecule
    arrays too, even with save_molecules = False (a solution dict is 31 MB at 1e5 steps and 312-480 MB at 1e6).

    Inputs:
        - n_steps: time steps of the largest run the worker will do [int]
        - backend: model.backend, 'isocalc', or 'toy' (0.15 GB) [str]

    Output: peak memory of one worker [GB]
    '''
    if backend == 'toy':
        return 0.15
    return 0.75 + 1.1e-6*n_steps


def choose_workers(requested, n_steps, cfg):
    '''
    Chooses the number of workers: the requested count, reduced to fit the memory budget
    The budget is min(max_memory_gb, 70% of the available memory), or max_memory_gb alone when
    parallel.check_available is false; there is always at least 1 worker. Prints the reason when it reduces the
    count, with a hint when the available memory is the limit.

    psutil's 'available' on macOS is free + inactive pages only: it leaves out what the system frees by
    compressing idle memory, and reads 2-4 GB on a 16 GB Mac with a browser open. Measured in that state: 6 workers
    at n_steps = 1e5 held 2.5 GB in total and pushed 0.8 GB to swap, while this rule allowed 2.

    Inputs:
        - requested: requested workers; None: parallel.workers [int]
        - n_steps: time steps of the largest run, for est_worker_gb [int]
        - cfg: resolved config (parallel.workers, max_memory_gb, check_available; model.backend) [dict]

    Output: (n, reason)
        - n: number of workers, at least 1 [int]
        - reason: one line on how n was chosen [str]
    '''
    import psutil
    par = cfg['parallel']
    requested = par['workers'] if requested is None else int(requested)
    est = est_worker_gb(n_steps, cfg['model']['backend'])
    available = psutil.virtual_memory().available/1e9 # [GB]
    check_available = par.get('check_available', True) # absent in configs saved before the option existed
    budget = min(par['max_memory_gb'], 0.7*available) if check_available else par['max_memory_gb'] # [GB]
    n = max(1, min(requested, math.floor(budget/est))) # one worker even when it does not fit the budget
    if check_available:
        source = f'min(max_memory_gb = {par["max_memory_gb"]:g}, 70% of {available:.1f} GB available)'
    else:
        source = f'max_memory_gb ({available:.1f} GB available, not checked: parallel.check_available = false)'
    reason = (f'{n} of {requested} workers: about {est:.2f} GB each at n_steps = {n_steps:.3g}, budget '
              f'{budget:.1f} GB = {source}')
    if n < requested:
        hint = ''
        if check_available and 0.7*available < par['max_memory_gb']: # the OS figure set the budget
            hint = ('\n  (macOS counts compressible memory as used; --set parallel.check_available=false budgets '
                    'max_memory_gb alone)')
        print(f'reducing workers: {reason}{hint}', flush = True)
    return n, reason


###_____The pool_____###

def _start_worker(cfg, run_dir, barrier):
    '''
    Pool initializer: the worker ignores SIGINT and SIGHUP, then runs model.init_worker
    model.init_worker sets the single-thread env, the compile caches and the worker globals, warms up the model
    and waits at the barrier. Ctrl-C in a terminal, and the hang-up when the terminal closes, reach every process
    of the group; the parent alone handles them (checkpoint, then terminate the pool). Workers that raised
    KeyboardInterrupt too would only print tracebacks and emcee's "Exception while calling your likelihood
    function" for each running evaluation, and workers killed by SIGHUP could die holding the task queue's lock
    (see the module docstring). SIGTERM keeps its default: Pool.terminate uses it to stop the workers.

    Inputs:
        - cfg: resolved config [dict]
        - run_dir: directory for the tracebacks of unexpected errors (worker_errors/); None: no files [str]
        - barrier: the start-up barrier, shared with the parent [multiprocessing Barrier]
    '''
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    model.init_worker(cfg, run_dir, barrier)


class WorkerPool:
    '''
    Context manager around a spawn multiprocessing.Pool whose workers ran model.init_worker (single-thread env,
    compile caches, warm-up) before the constructor returns. The workers ignore SIGINT and SIGHUP (Ctrl-C is the
    parent's). Usable as an emcee pool (.map):

        with WorkerPool(cfg, workers, n_steps_max, run_dir) as pool:
            results = pool.map(model.log_prob, thetas)

    The worker count is capped by choose_workers(workers, n_steps_max, cfg); use pool.n for the actual count.
    Leaving the with block terminates the workers.

    Attributes:
        - n: number of workers [int]
        - reason: how n was chosen (choose_workers) [str]
        - warmup_s: time from the start of the pool until every worker was warm [s]
    '''

    def __init__(self, cfg, workers, n_steps_max = None, run_dir = None, verbose = True):
        '''
        Starts the workers and returns once every one is warm (they meet at a barrier, at most 900 s)
        A worker that fails its start-up aborts the barrier: the pool is then terminated and RuntimeError raised,
        below the worker's own traceback.

        Inputs:
            - cfg: resolved config [dict]
            - workers: requested workers; None: parallel.workers [int]
            - n_steps_max: largest n_steps the pool will run, for the memory budget; None: model.n_steps [int]
            - run_dir: directory for the tracebacks of unexpected errors (worker_errors/); None: no files [str]
            - verbose: print the 'N workers ready in X s' line; False when the caller logs it [Bool]
        '''
        n_steps_max = cfg['model']['n_steps'] if n_steps_max is None else n_steps_max
        self.n, self.reason = choose_workers(workers, n_steps_max, cfg)
        env.set_single_thread_env() # spawned workers inherit the environment
        context = mp.get_context('spawn') # never fork: it deadlocks once JAX is imported
        barrier = context.Barrier(self.n + 1) # the n workers and this process
        start = time.perf_counter()
        self._pool = context.Pool(self.n, initializer = _start_worker, initargs = (cfg, run_dir, barrier))
        try:
            barrier.wait(timeout = 900) # each worker reaches it at the end of its warm-up
        except threading.BrokenBarrierError: # a worker failed (it aborts the barrier) or a wait timed out
            self.terminate()
            raise RuntimeError('worker start-up failed (see the traceback above) or took longer than 900 s') from None
        except BaseException: # e.g. Ctrl-C during the warm-up: leave no worker behind
            self.terminate()
            raise
        barrier.abort() # a worker the Pool respawns later (after a crash) must not wait for the others
        self._known = frozenset(self._pids()) # the workers that started: any change later is a death (_check)
        self.warmup_s = time.perf_counter() - start # [s]
        if verbose:
            print(f'{self.n} worker{"s" if self.n > 1 else ""} ready in {self.warmup_s:.1f} s', flush = True)

    def _pids(self):
        '''
        Process ids of the Pool's current workers (Pool._pool, multiprocessing's own list of them)
        '''
        return {p.pid for p in self._pool._pool}

    def _check(self):
        '''
        Raises WorkerDied if a worker has exited since the pool started
        The Pool has replaced it, so its pid has gone, or it is still listed with an exit code. Worker sets are
        compared with the start, not with the last map: a worker that died while idle leaves the others waiting
        for its lock (see the module docstring), and its replacement would otherwise look like a healthy worker.
        '''
        dead = (self._known - self._pids()) | {p.pid for p in self._pool._pool if p.exitcode is not None}
        if dead:
            raise WorkerDied(f'worker process(es) {sorted(dead)} died (killed by the OS, e.g. for memory, killed by a '
                             'signal, or crashed), so their tasks are lost and the pool cannot go on. An MCMC run '
                             'saves its chain up to the last step: python -m isofate.mcmc run --resume RUN_DIR (with '
                             'fewer --workers if memory ran out)')

    def map(self, f, iterable):
        '''
        Maps f over iterable, one task at a time per worker
        chunksize=1 because evaluation times vary 50x: a chunk of slow tasks would keep one worker busy while the
        others idle. Waits in slices of POLL s and raises WorkerDied when a worker has died, before or while the map
        runs.

        Inputs:
            - f: picklable, module-level function, e.g. model.log_prob [function]
            - iterable: the arguments of f, one per task [iterable]

        Output: results of f, in the order of iterable [list]
        '''
        self._check() # a death between two maps would leave this map waiting forever
        result = self._pool.map_async(f, list(iterable), chunksize = 1)
        while True:
            try:
                return result.get(timeout = POLL)
            except mp.TimeoutError: # not done yet: are all workers still alive?
                self._check()

    def imap_unordered(self, f, iterable):
        '''
        Yields the results of f over iterable as they finish, one task at a time per worker
        Raises WorkerDied when a worker has died, before or while it runs.

        Inputs:
            - f: picklable, module-level function, e.g. model.eval_task [function]
            - iterable: the arguments of f, one per task [iterable]

        Output: results of f, in the order they finish [generator]
        '''
        self._check()
        it = self._pool.imap_unordered(f, iterable, chunksize = 1)
        while True:
            try:
                yield it.next(timeout = POLL)
            except StopIteration:
                return
            except mp.TimeoutError: # no result in this slice: are all workers still alive?
                self._check()

    def close(self):
        '''
        Lets the workers finish their tasks, then joins them
        '''
        self._pool.close()
        self._pool.join()

    def terminate(self, timeout = None):
        '''
        Stops the workers and joins them within about timeout + 1 s
        multiprocessing.Pool.terminate runs in a daemon thread, because it can wait forever for the task queue's
        lock when a worker died holding it (see the module docstring); workers still alive after timeout are killed
        (SIGKILL).

        Inputs:
            - timeout: how long Pool.terminate may take by itself; None: TEARDOWN_S [s]

        Output: True when Pool.terminate finished by itself; False when the workers had to be killed, which leaves
            the thread behind, blocked [Bool]
        '''
        pool, done = self._pool, threading.Event()
        timeout = TEARDOWN_S if timeout is None else timeout

        # Pool.terminate, then join; a daemon thread never holds up the exit of the process
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
                except (OSError, ValueError, AttributeError): # gone or closed in the meantime
                    pass
        for p in list(pool._pool): # reap the killed workers, at most 1 s each
            try:
                p.join(timeout = 1.0)
            except (OSError, ValueError, AssertionError):
                pass
        return False

    def __enter__(self):
        '''
        Returns the pool itself, for the with statement
        '''
        return self

    def __exit__(self, *exc):
        '''
        Terminates the workers when the with block ends, normally or by an exception (which then propagates)
        '''
        self.terminate()
        return False


def worker_rss_gb(pool):
    '''
    Total resident memory of the pool's worker processes; workers that have gone are skipped
    On macOS the RSS leaves out compressed pages, so it can read less than the physical footprint.

    Inputs:
        - pool: a WorkerPool or a multiprocessing.Pool [object]

    Output: total resident memory of the workers [GB]
    '''
    import psutil
    processes = pool._pool._pool if isinstance(pool, WorkerPool) else pool._pool # the workers' Process objects
    total = 0
    for p in processes:
        try:
            total += psutil.Process(p.pid).memory_info().rss # [bytes]
        except psutil.Error: # the worker exited in the meantime
            pass
    return total/1e9
