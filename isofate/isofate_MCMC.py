'''
Collin Cherubim
July 25, 2025

This script performs an MCMC for individual planets to compare 
measured escape rates to predictions from IsoFATE/Atmodeller.
'''

import multiprocessing
import os
import pickle
import sys

import emcee
import numpy as np
from scipy import stats
import time
import datetime as dt
from zoneinfo import ZoneInfo

from isofate.constants import *
from isofate.isofate_coupler import *
from isofate.isofunks import *
from isofate.orbit_params import *

# PARAMETER CONFIGURATION - easily modify which parameters to vary
PARAM_CONFIG = {
    'log_f_atm': {'vary': True, 'fixed_value': -3.0},
    't_pms': {'vary': False, 'fixed_value': 1e8},  # Set to False to keep constant
    'time': {'vary': True, 'fixed_value': 5e9}, 
    'log_OtoH_enhancement': {'vary': False, 'fixed_value': 0.0}  # Set to False to keep constant
}

# Automatically generate lists of varying and fixed parameters
VARYING_PARAMS = [(name, config) for name, config in PARAM_CONFIG.items() if config['vary']]
FIXED_PARAMS = [(name, config) for name, config in PARAM_CONFIG.items() if not config['vary']]

print("Varying parameters:", [name for name, _ in VARYING_PARAMS])
print("Fixed parameters:", [(name, config['fixed_value']) for name, config in FIXED_PARAMS])

# Fixed stellar parameters 
R_star = 0.22*Rs # [m]
M_star = 0.18*Ms # [kg]
T_star = 3096 # [K]
t_jump = 5.9 - 15.4*(M_star/Ms)
L = 0.0038*Ls

# Fixed planet parameters
Mp = 5.6*Me
P = 24.74/s2day
a = SemiMajor(M_star, P)
Fp = Insolation(L, a)
T = EqTemp(Fp, A=0)
F0 = Fp * 1e-3
d = a # orbital distance [m]
flux_model = 'power law'
stellar_type = 'M1'
t_sat = t_jump*1e9 # XUV saturation time [yr]
step_fn = True
F_final = 0.17
mechanism = 'XUV'
RR = True
rad_evol = True
Rp_override = False
n_steps = int(1e5)
n_atmodeller = int(0)
thermal = True
melt_fraction_override = False
save_molecules = False
mantle_iron_dict = False
dynamic_phi = False
t0 = 1e6

# True observed values
true_Rp_final = 1.73*Re  # True final planet radius
sigma_Rp = 0.025*Re
true_flux_ratio = 0.0011    # True H/He escape flux ratio
sigma_flux_ratio = 0.00065
true_mdot = 2e5 # True total mass loss rate [kg/s]
sigma_mdot = 6e4 # [kg/s]

def log_prior(theta):
    """
    Prior function for MCMC parameters (only for varying parameters)
    """
    # Create full parameter dictionary
    param_dict = {}
    
    # Fill in varying parameters from theta
    for i, (param_name, param_config) in enumerate(VARYING_PARAMS):
        param_dict[param_name] = theta[i]
    
    # Fill in fixed parameters
    for param_name, param_config in FIXED_PARAMS:
        param_dict[param_name] = param_config['fixed_value']
    
    # Extract values for prior checking
    log_f_atm = param_dict['log_f_atm']
    t_pms = param_dict['t_pms']
    time = param_dict['time']
    log_OtoH_enhancement = param_dict['log_OtoH_enhancement']
    
    if (-5 <= log_f_atm <= -1.3 and 
        0 <= t_pms <= 3e8 and 
        3.5e9 <= time <= 10e9 and 
        -1 <= log_OtoH_enhancement <= 2):
        return 0.0
    return -np.inf

def run_isocalc_simulation(theta):
    """
    Run the isocalc simulation for given parameters
    Returns: (log_likelihood, model_outputs_dict or None)
    """
    
    try:
        # Create full parameter dictionary
        param_dict = {}
        
        # Fill in varying parameters from theta
        for i, (param_name, param_config) in enumerate(VARYING_PARAMS):
            param_dict[param_name] = theta[i]
        
        # Fill in fixed parameters
        for param_name, param_config in FIXED_PARAMS:
            param_dict[param_name] = param_config['fixed_value']
        
        # Extract parameters
        log_f_atm = param_dict['log_f_atm']
        t_pms = param_dict['t_pms']
        time = param_dict['time']
        log_OtoH_enhancement = param_dict['log_OtoH_enhancement']
        
        # Convert free parameters
        f_atm = 10**log_f_atm
        OtoH_enhancement = 10**log_OtoH_enhancement
        
        # ... rest of the function remains exactly the same ...
        M_atm = Mp*f_atm
        OtoH_enhanced = OtoH_protosolar*OtoH_enhancement
        OtoH_enhanced_mass = OtoH_enhanced*(mu_O/mu_H)
        N_He = (HetoH_protosolar_mass/(1 + HetoH_protosolar_mass))*M_atm/mu_He
        N_H = (1 - DtoH_solar_mass - OtoH_enhanced_mass - CtoH_protosolar_mass - StoH_protosolar_mass - NtoH_protosolar_mass)*M_atm/(1 + HetoH_protosolar_mass)/mu_H
        N_D = DtoH_solar_mass*M_atm/(1 + HetoH_protosolar_mass)/mu_D
        N_O = OtoH_enhanced_mass*M_atm/(1 + HetoH_protosolar_mass)/mu_O
        N_C = CtoH_protosolar_mass*M_atm/(1 + HetoH_protosolar_mass)/mu_C
        N_N = NtoH_protosolar_mass*M_atm/(1 + HetoH_protosolar_mass)/mu_N
        N_S = StoH_protosolar_mass*M_atm/(1 + HetoH_protosolar_mass)/mu_S
        mu_avg = (N_H*mu_H + N_He*mu_He + N_D*mu_D + N_O*mu_O + N_C*mu_C + N_N*mu_N + N_S*mu_S)/(N_H+N_He+N_D+N_O+N_C+N_N+N_S)
        sol = isocalc(f_atm, Mp, M_star, F0, Fp, T, d, time, mechanism, rad_evol,
                     N_H=N_H, N_He=N_He, N_D=N_D, N_O=N_O, N_C=N_C, 
                     N_N=N_N, N_S=N_S,
                     melt_fraction_override=melt_fraction_override,
                     mu=mu_avg, eps=0.15, activity='medium', flux_model=flux_model, 
                     stellar_type=stellar_type, Rp_override=Rp_override, t_sat=t_sat, 
                     step_fn=step_fn, F_final=F_final, t_pms=t_pms, pms_factor=1e2,
                     n_steps=n_steps, t0=t0, rho_rcb=1.0, RR=RR,
                     thermal=thermal, beta=-1.23, n_atmodeller=n_atmodeller, 
                     save_molecules=save_molecules, mantle_iron_dict=mantle_iron_dict, dynamic_phi=dynamic_phi)
        model_Rp_final = sol['Rp'][-1]
        phi_he_min = 1e-20
        phi_he_safe = max(sol['Phi_He'][-1], phi_he_min)
        model_flux_ratio = sol['Phi_H'][-1] / phi_he_safe
        model_mdot = sol['phi'][-1] * 4*np.pi*sol['Rp'][-1]**2
        chi2_Rp = (model_Rp_final - true_Rp_final)**2 / sigma_Rp**2
        chi2_flux = (model_flux_ratio - true_flux_ratio)**2 / sigma_flux_ratio**2
        chi2_mdot = (model_mdot - true_mdot)**2 / sigma_mdot**2
        log_likelihood = -0.5 * (chi2_Rp + chi2_flux + chi2_mdot)
        
        # Store the full parameter set (varying + fixed) for record keeping
        full_theta = [param_dict[name] for name in ['log_f_atm', 't_pms', 'time', 'log_OtoH_enhancement']]
        
        if n_atmodeller != 0:
            model_output = {
                'parameters': full_theta,  # Store all 4 parameters
                'varying_parameters': theta.copy(),  # Store only the varying ones
                'Rp_final': model_Rp_final,
                'flux_ratio': model_flux_ratio,
                'mdot': model_mdot,
                'log_likelihood': log_likelihood,
                'T_surf_atmod': sol['T_surf_atmod'][-1],
                'n_H2_atm': sol['n_H2_a'][-1],
                'n_He_atm': sol['N_He'][-1]/avogadro,
                'n_H2O_atm': sol['n_H2O_a'][-1],
                'n_O2_atm': sol['n_O2_a'][-1],
                'n_CO2_atm': sol['n_CO2_a'][-1],
                'n_CO_atm': sol['n_CO_a'][-1],
                'n_CH4_atm': sol['n_CH4_a'][-1],
                'n_N2_atm': sol['n_N2_a'][-1],
                'n_S2_atm': sol['n_S2_a'][-1],
                'n_H2_mantle': sol['n_H2_a_int'][-1],
                'n_H2O_mantle': sol['n_H2O_a_int'][-1],
                'n_O2_mantle': sol['n_O2_a_int'][-1],
                'n_CO2_mantle': sol['n_CO2_a_int'][-1],
                'n_CO_mantle': sol['n_CO_a_int'][-1],
                'n_CH4_mantle': sol['n_CH4_a_int'][-1],
                'n_N2_mantle': sol['n_N2_a_int'][-1],
                'n_S2_mantle': sol['n_S2_a_int'][-1],
                'n_He_mantle': sol['N_He_int'][-1]/avogadro,
                'DtoH': sol['N_D'][-1]/sol['N_H'][-1],
                'fO2_a': sol['fO2_a'][-1],
            }
        else:
            model_output = {
                'parameters': full_theta,  # Store all 4 parameters
                'varying_parameters': theta.copy(),  # Store only the varying ones
                'Rp_final': model_Rp_final,
                'flux_ratio': model_flux_ratio,
                'mdot': model_mdot,
                'log_likelihood': log_likelihood,
                'N_H': sol['N_H'][-1],
                'N_He': sol['N_He'][-1],
                'N_D': sol['N_D'][-1],
                'N_O': sol['N_O'][-1],
                'N_C': sol['N_C'][-1],
                'N_N': sol['N_N'][-1],
                'N_S': sol['N_S'][-1],
            }
        return log_likelihood, model_output
    except Exception as e:
        print(f"Error in simulation: {e}")
        return -np.inf, None

# <<< CHANGED: This function now returns the detailed model output as a "blob"
def log_probability(theta):
    """Log posterior probability"""
    lp = log_prior(theta)
    if not np.isfinite(lp):
        # Return -inf for the probability and None for the blob
        return -np.inf, None

    # Run the full simulation to get both the likelihood and the detailed output
    ll, model_output = run_isocalc_simulation(theta)
    
    if not np.isfinite(ll):
        # If the simulation fails, return -inf and None for the blob
        return -np.inf, None
        
    # Return the final probability and the detailed output dictionary as the blob
    return lp + ll, model_output

def main():
    start = time.time()
    print('start time (EST):', dt.datetime.now(ZoneInfo('America/New_York')))

    # MCMC parameters (dynamic based on configuration)
    ndim = len(VARYING_PARAMS)  # Number of varying parameters
    nwalkers = 12
    nsteps = 1000
    
    print(f"MCMC setup: {ndim} varying parameters")
    
    # Initialize walkers around reasonable starting points (only for varying parameters)
    starting_point = []
    param_names = []
    
    for param_name, param_config in VARYING_PARAMS:
        starting_point.append(param_config['fixed_value'])  # Use fixed_value as starting point
        param_names.append(param_name)
    
    pos = np.array(starting_point) + 1e-4 * np.random.randn(nwalkers, ndim)
    
    for i in range(nwalkers):
        while log_prior(pos[i]) == -np.inf:
            pos[i] = np.array(starting_point) + 1e-4 * np.random.randn(ndim)
    
    print("Starting MCMC...")
    print(f"Number of walkers: {nwalkers}")
    print(f"Number of steps: {nsteps}")
    print(f"Number of dimensions: {ndim}")
    
    with multiprocessing.Pool() as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, pool=pool, blobs_dtype=object)
        sampler.run_mcmc(pos, nsteps, progress=False)
    
    print("MCMC complete!")
    
    model_outputs = sampler.get_blobs()
    samples = sampler.get_chain()
    
    # Save results
    results = {
        'samples': samples,
        'log_prob': sampler.get_log_prob(),
        'parameter_names': param_names,  # Dynamic parameter names
        'parameter_config': PARAM_CONFIG,  # Save the configuration
        'true_values': {'Rp_final': true_Rp_final, 'flux_ratio': true_flux_ratio, 'mdot': true_mdot},
        'model_outputs': model_outputs
    }
    
    output_filename = 'IsoFATE_LHS1140b_MCMC_results_n1e5_natmod0_eps15_tjump_Ffinal17_constant_tpms_OH_12w_1000s.pickle'
    with open(output_filename, 'wb') as file:
        pickle.dump(results, file)
    
    print(f"Results saved to {output_filename}")
    
    flat_samples = sampler.get_chain(discard=100, thin=5, flat=True)
    print("\nParameter estimates:")
    for i, param in enumerate(param_names):
        mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
        q = np.diff(mcmc)
        print(f"{param}: {mcmc[1]:.6f} +{q[1]:.6f} -{q[0]:.6f}")

    print('done (', round((time.time() - start)/60, 2), 'mins )')

if __name__ == "__main__":
    main()