'''
Monte Carlo time series of IsoFATE-Atmodeller coupled planets around an M1 star.

Usage:
    python -m isofate.mc_timeseries OUT.npz [--n-planets N] [--seed S] [--shard I] [--n-shards K] [--workers W]
                                    [--pin]

Every shard draws the same N planets from the seed and runs planets I, I + K, I + 2K, ...; --shard defaults to
AWS_BATCH_JOB_ARRAY_INDEX. OUT.npz holds one row per planet: its index, its parameters, and each time series
sampled at the Atmodeller steps, with NaN rows for planets that failed. A shard keeps its rows in memory (about
0.2 MB per planet) and writes them when it finishes, so size shards to a few thousand planets.

--pin (Linux) runs each worker on a CPU of its own, which cuts the time per planet by about 40%. Use it only when
the job has its CPUs to itself: workers of jobs sharing a machine would pin to the same CPUs.
'''

import os

os.environ.setdefault('NPROC', '1') # XLA CPU thread pool size

import argparse
import multiprocessing as mp
import time

import numpy as np

from isofate.constants import *
from isofate.orbit_params import *

T_STAR = 3600 # stellar effective temperature [K]
M_STAR = 0.5*Ms # stellar mass [kg]
R_STAR = 0.5*Rs # stellar radius [m]
N_STEPS = int(1e5)
N_ATMODELLER = int(1e2)
SAMPLES = np.arange(0, N_STEPS, N_ATMODELLER) # sampled timesteps, the Atmodeller steps

# output name: (isocalc solution key, scale)
SERIES = {'time': ('time', 1), 'Rp': ('Rp', 1), 'f_atm': ('fatm', 1), 'T_surf': ('T_surf_atmod', 1)}
for _side in ('', '_int'):
    for _molecule in ('H2', 'He', 'H2O', 'O2', 'CO2', 'CO', 'CH4', 'N2', 'S2', 'H2O4S', 'SO2'):
        if _molecule == 'He':
            SERIES[f'n_He{_side}'] = (f'N_He{_side}', 1/avogadro)
        else:
            SERIES[f'n_{_molecule}{_side}'] = (f'n_{_molecule}_a{_side}', 1)
SERIES['O2_fugacity'] = ('fO2_a', 1)


def draw_planets(n, seed):
    rng = np.random.default_rng(seed)
    return {'P': 10**rng.uniform(np.log10(1), np.log10(300), n)/s2day, # orbital period [s]
            'Mp': 10**rng.uniform(np.log10(1), np.log10(20), n)*Me, # planet mass [kg]
            'f_atm': 10**rng.uniform(np.log10(0.001), np.log10(0.10), n), # atm mass fraction [ndim]
            'OtoH_enhancement': np.ones(n)} # O/H relative to protosolar


def isocalc_args(P, Mp, f_atm, OtoH_enhancement):
    '''
    isocalc arguments and keyword arguments for one planet.
    '''
    L = Luminosity(R_STAR, T_STAR) # stellar luminosity [W]
    d = SemiMajor(M_STAR, P) # orbital distance [m]
    Fp = Insolation(L, d) # planetary insolation [W/m2]
    Teq = EqTemp(Fp, A = 0) # planetary eq temp [K]
    F0 = Fp*1e-3 # incident XUV flux [W/m2]

    OtoH_enhanced_mass = OtoH_protosolar*OtoH_enhancement*(mu_O/mu_H)
    N_H = (1 - DtoH_solar_mass - OtoH_enhanced_mass - CtoH_protosolar_mass - NtoH_protosolar_mass
           - StoH_protosolar_mass)*Mp*f_atm/(1 + HetoH_protosolar_mass)/mu_H # initial H number [atoms]
    N_He = (HetoH_protosolar_mass/(1 + HetoH_protosolar_mass))*Mp*f_atm/mu_He
    N_D = DtoH_solar_mass*Mp*f_atm/(1 + HetoH_protosolar_mass)/mu_D
    N_O = OtoH_enhanced_mass*Mp*f_atm/(1 + HetoH_protosolar_mass)/mu_O
    N_C = CtoH_protosolar_mass*Mp*f_atm/(1 + HetoH_protosolar_mass)/mu_C
    N_N = NtoH_protosolar_mass*Mp*f_atm/(1 + HetoH_protosolar_mass)/mu_N
    N_S = StoH_protosolar_mass*Mp*f_atm/(1 + HetoH_protosolar_mass)/mu_S
    mu_avg = (N_H*mu_H + N_He*mu_He + N_D*mu_D + N_O*mu_O + N_C*mu_C + N_N*mu_N + N_S*mu_S)/(
        N_H + N_He + N_D + N_O + N_C + N_N + N_S)

    args = (f_atm, Mp, M_STAR, F0, Fp, Teq, d, 5e9, 'XUV', True, N_H, N_He, N_D, N_O, N_C, N_N, N_S)
    kwargs = dict(melt_fraction_override = False, mu = mu_avg, eps = 0.15, activity = 'medium',
                  flux_model = 'power law', stellar_type = 'M1', Rp_override = False, t_sat = 5e8, step_fn = False,
                  F_final = 0, t_pms = 0, pms_factor = 1e2, n_steps = N_STEPS, t0 = 1e6, rho_rcb = 1.0, RR = True,
                  thermal = True, beta = -1.23, n_atmodeller = N_ATMODELLER, save_molecules = True,
                  mantle_iron_dict = False)
    return args, kwargs


def simulate(**planet):
    '''
    Sampled time series of one planet, keyed as SERIES.
    '''
    from isofate.isofate_coupler import isocalc

    args, kwargs = isocalc_args(**planet)
    sol = isocalc(*args, **kwargs)
    return {name: sol[key][SAMPLES]*scale for name, (key, scale) in SERIES.items()}


def _pin(cpus, counter):
    with counter.get_lock():
        os.sched_setaffinity(0, {cpus[counter.value % len(cpus)]})
        counter.value += 1


def _run(task):
    index, params = task
    try:
        return index, simulate(**params)
    except Exception as e:
        print(f'planet {index} failed: {e!r} {params}', flush = True)
        return index, None


def main():
    parser = argparse.ArgumentParser(description = __doc__, formatter_class = argparse.RawDescriptionHelpFormatter)
    parser.add_argument('out')
    parser.add_argument('--n-planets', type = int, default = int(50e3))
    parser.add_argument('--seed', type = int, default = 0)
    parser.add_argument('--shard', type = int, default = int(os.environ.get('AWS_BATCH_JOB_ARRAY_INDEX', 0)))
    parser.add_argument('--n-shards', type = int, default = 1)
    parser.add_argument('--workers', type = int, default = 1)
    parser.add_argument('--pin', action = 'store_true')
    options = parser.parse_args()
    if not 0 <= options.shard < options.n_shards:
        parser.error(f'--shard must be in [0, {options.n_shards})')

    planets = draw_planets(options.n_planets, options.seed)
    indices = np.arange(options.shard, options.n_planets, options.n_shards)
    tasks = [(i, {key: float(value[i]) for key, value in planets.items()}) for i in indices]
    row = {i: r for r, i in enumerate(indices)}
    series = {name: np.full((len(indices), len(SAMPLES)), np.nan) for name in SERIES}

    start = time.perf_counter()
    context = mp.get_context('spawn')
    pin = (sorted(os.sched_getaffinity(0)), context.Value('i', 0)) if options.pin else None
    if options.workers > 1:
        pool = context.Pool(options.workers, _pin if pin else None, pin or ())
        results = pool.imap_unordered(_run, tasks)
    else:
        if pin:
            _pin(*pin)
        pool = None
        results = map(_run, tasks)
    for done, (i, result) in enumerate(results, 1):
        if result is not None:
            for name, values in result.items():
                series[name][row[i]] = values
        print(f'{done}/{len(tasks)} planets, {time.perf_counter() - start:.1f} s', flush = True)
    if pool is not None:
        pool.close()
        pool.join()

    np.savez(options.out, index = indices, ok = ~np.isnan(series['time'][:, 0]), Mp = planets['Mp'][indices],
             P = planets['P'][indices], f_atm0 = planets['f_atm'][indices],
             OtoH_enhancement = planets['OtoH_enhancement'][indices], **series)
    print(f'saved {options.out}', flush = True)


if __name__ == '__main__':
    main()
