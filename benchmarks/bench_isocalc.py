'''
Wall-clock benchmark for isocalc on representative planets.

Usage:
    uv run python benchmarks/bench_isocalc.py [scenario ...] [--repeat N] [--save DIR | --compare DIR]

Each scenario's first call (which includes compilation in a fresh process) is reported separately from the N
repeats that follow. --save writes each scenario's solution to DIR; --compare reports the largest difference
from solutions saved in DIR. To check a change, run with --save on the base commit and with --compare on the change.
'''

import argparse
import os
import time

import numpy as np

from isofate.constants import *
from isofate.isofate_coupler import isocalc
from isofate.mc_timeseries import isocalc_args
from isofate.orbit_params import Luminosity, EqTemp, Insolation, SemiMajor


def planet(M_star, L, Mp, P, f_atm, t_sat, step_fn, n_steps=int(1e5), n_atmodeller=0, dynamic_phi=False):
    '''
    Arguments for isocalc with proto-solar initial abundances, as in sim.py.
    '''
    a = SemiMajor(M_star, P)
    Fp = Insolation(L, a)
    T = EqTemp(Fp, A = 0)
    M_atm = Mp*f_atm
    N_He = (HetoH_protosolar_mass/(1 + HetoH_protosolar_mass))*M_atm/mu_He
    N_H = (1 - DtoH_solar_mass - OtoH_protosolar_mass - CtoH_protosolar_mass - StoH_protosolar_mass - NtoH_protosolar_mass)*M_atm/(1 + HetoH_protosolar_mass)/mu_H
    N_D = DtoH_solar_mass*M_atm/(1 + HetoH_protosolar_mass)/mu_D
    N_O = OtoH_protosolar_mass*M_atm/(1 + HetoH_protosolar_mass)/mu_O
    N_C = CtoH_protosolar_mass*M_atm/(1 + HetoH_protosolar_mass)/mu_C
    N_N = NtoH_protosolar_mass*M_atm/mu_N
    N_S = StoH_protosolar_mass*M_atm/(1 + HetoH_protosolar_mass)/mu_S
    mu_avg = (N_H*mu_H + N_He*mu_He + N_D*mu_D + N_O*mu_O + N_C*mu_C + N_N*mu_N + N_S*mu_S)/(N_H+N_He+N_D+N_O+N_C+N_N+N_S)
    args = (f_atm, Mp, M_star, Fp*1e-3, Fp, T, a, 5e9, 'XUV', True)
    kwargs = dict(N_H = N_H, N_He = N_He, N_D = N_D, N_O = N_O, N_C = N_C, N_N = N_N, N_S = N_S, mu = mu_avg,
                  t_sat = t_sat, step_fn = step_fn, F_final = 0.17, n_steps = n_steps, n_atmodeller = n_atmodeller,
                  dynamic_phi = dynamic_phi)
    return args, kwargs


def LHS1140b(**kwargs):
    M_star = 0.18*Ms
    t_jump = 5.9 - 15.4*(M_star/Ms) # [Gyr]
    return planet(M_star, 0.0038*Ls, 5.6*Me, 24.74/s2day, 0.00085, t_jump*1e9, True, **kwargs)


def GJ3090b(**kwargs):
    M_star = 0.519*Ms
    return planet(M_star, Luminosity(0.516*Rs, 3556), 3.34*Me, 2.9/s2day, 0.03, 5e8, False, **kwargs)


SCENARIOS = {
    'escape': lambda: LHS1140b(),
    'escape_dynamic_phi': lambda: LHS1140b(dynamic_phi = True),
    'escape_atm_lost': lambda: GJ3090b(),
    'coupled_1e4': lambda: LHS1140b(n_atmodeller = int(1e4), dynamic_phi = True),
    'coupled_1e3': lambda: LHS1140b(n_atmodeller = int(1e3), dynamic_phi = True),
    'mc_timeseries': lambda: isocalc_args(P = 20/s2day, Mp = 5*Me, f_atm = 0.02, OtoH_enhancement = 1),
}


def flatten(solution):
    flat = {}
    for key, value in solution.items():
        if isinstance(value, dict):
            for key2, value2 in value.items():
                flat[f'{key}.{key2}'] = np.asarray(value2, dtype = float)
        else:
            flat[key] = np.asarray(value, dtype = float)
    return flat


def difference(reference, solution):
    '''
    Largest relative difference, and largest difference relative to each output's scale.
    '''
    max_rel, max_scaled = 0.0, 0.0
    for key in reference.files:
        a, b = reference[key], solution[key]
        valid = ~np.isnan(a)
        if not np.array_equal(valid, ~np.isnan(b)):
            return np.inf, np.inf
        a, b = a[valid], b[valid]
        if a.size == 0:
            continue
        diff = np.abs(a - b)
        max_rel = max(max_rel, np.max(np.where(diff == 0, 0, diff/np.maximum(np.abs(a), 1e-300))))
        max_scaled = max(max_scaled, np.max(diff)/max(np.max(np.abs(a)), 1e-300))
    return max_rel, max_scaled


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('scenarios', nargs = '*', help = ', '.join(SCENARIOS))
    parser.add_argument('--repeat', type = int, default = 3)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--save')
    group.add_argument('--compare')
    options = parser.parse_args()
    unknown = set(options.scenarios) - set(SCENARIOS)
    if unknown:
        parser.error(f'unknown scenarios {sorted(unknown)}; choose from {list(SCENARIOS)}')
    if options.repeat < 1:
        parser.error('--repeat must be at least 1')

    for name in options.scenarios or SCENARIOS:
        args, kwargs = SCENARIOS[name]()
        start = time.perf_counter()
        solution = isocalc(*args, **kwargs)
        first = time.perf_counter() - start
        times = []
        for _ in range(options.repeat):
            start = time.perf_counter()
            isocalc(*args, **kwargs)
            times.append(time.perf_counter() - start)
        line = f'{name:20s} first call {first:8.3f} s   median {np.median(times):8.3f} s   min {np.min(times):8.3f} s'
        if options.save:
            os.makedirs(options.save, exist_ok = True)
            np.savez(os.path.join(options.save, f'{name}.npz'), **flatten(solution))
        if options.compare:
            reference = np.load(os.path.join(options.compare, f'{name}.npz'))
            max_rel, max_scaled = difference(reference, flatten(solution))
            line += f'   max rel diff {max_rel:.1e}   max diff/scale {max_scaled:.1e}'
        print(line, flush = True)


if __name__ == '__main__':
    main()
