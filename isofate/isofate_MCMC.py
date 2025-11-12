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

from isofate.constants import *
from isofate.isofate_coupler import *
from isofate.isofunks import *
from isofate.orbit_params import *

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
n_steps = int(1e4)
n_atmodeller = int(1e3)
thermal = True
melt_fraction_override = False
save_molecules = True
mantle_iron_dict = False
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
    Prior function for MCMC parameters
    """
    log_f_atm, t_pms, time, log_OtoH_enhancement = theta
    
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
    log_f_atm, t_pms, time, log_OtoH_enhancement = theta
    
    try:
        # Convert free parameters
        f_atm = 10**log_f_atm
        OtoH_enhancement = 10**log_OtoH_enhancement
        
        # Calculate initial atmospheric composition
        M_atm = Mp*f_atm
        
        OtoH_enhanced = OtoH_protosolar*OtoH_enhancement
        OtoH_enhanced_mass = OtoH_enhanced*(mu_O/mu_H)
        
        N_He = (HetoH_protosolar_mass/(1 + HetoH_protosolar_mass))*M_atm/mu_He
        N_H = (1 - DtoH_solar_mass - OtoH_enhanced_mass - CtoH_protosolar_mass)*M_atm/(1 + HetoH_protosolar_mass)/mu_H
        N_D = DtoH_solar_mass*M_atm/(1 + HetoH_protosolar_mass)/mu_D
        N_O = OtoH_enhanced_mass*M_atm/(1 + HetoH_protosolar_mass)/mu_O
        N_C = CtoH_protosolar_mass*M_atm/(1 + HetoH_protosolar_mass)/mu_C
        mu_avg = (N_H*mu_H + N_He*mu_He + N_D*mu_D + N_O*mu_O + N_C*mu_C)/(N_H+N_He+N_D+N_O+N_C)
        
        # Run simulation
        sol = isocalc(f_atm, Mp, M_star, F0, Fp, T, d, time, mechanism, rad_evol,
                     N_H=N_H, N_He=N_He, N_D=N_D, N_O=N_O, N_C=N_C, 
                     melt_fraction_override=melt_fraction_override,
                     mu=mu_avg, eps=0.15, activity='medium', flux_model=flux_model, 
                     stellar_type=stellar_type, Rp_override=Rp_override, t_sat=t_sat, 
                     step_fn=step_fn, F_final=F_final, t_pms=t_pms, pms_factor=1e2,
                     n_steps=n_steps, t0=t0, rho_rcb=1.0, RR=RR,
                     thermal=thermal, beta=-1.23, n_atmodeller=n_atmodeller, 
                     save_molecules=save_molecules, mantle_iron_dict=mantle_iron_dict)
        
        # Extract model predictions
        model_Rp_final = sol['Rp'][-1]
        
        # Safe flux ratio calculation
        phi_he_min = 1e-20
        phi_he_safe = max(sol['Phi_He'][-1], phi_he_min)
        model_flux_ratio = sol['Phi_H'][-1] / phi_he_safe
        model_mdot = sol['phi'][-1] * 4*np.pi*sol['Rp'][-1]**2
        
        # Calculate chi-squared
        chi2_Rp = (model_Rp_final - true_Rp_final)**2 / sigma_Rp**2
        chi2_flux = (model_flux_ratio - true_flux_ratio)**2 / sigma_flux_ratio**2
        chi2_mdot = (model_mdot - true_mdot)**2 / sigma_mdot**2
        
        log_likelihood = -0.5 * (chi2_Rp + chi2_flux + chi2_mdot)
        
        # Create model output dict
        model_output = {
            'parameters': theta.copy(),
            'Rp_final': model_Rp_final,
            'flux_ratio': model_flux_ratio,
            'mdot': model_mdot,
            'log_likelihood': log_likelihood
        }
        
        # Add atmodeller results if available
        # if 'atmodeller_final' in sol:
        #     model_output['atmodeller_final'] = sol['atmodeller_final'].copy()']
        model_output['T_surf_atmod'] = sol['T_surf_atmod'][-1]
        model_output['n_H2_atm'] = sol['n_H2_a'][-1]
        model_output['n_He_atm'] = sol['N_He'][-1]/avogadro
        model_output['n_H2O_atm'] = sol['n_H2O_a'][-1]
        model_output['n_O2_atm'] = sol['n_O2_a'][-1]
        model_output['n_CO2_atm'] = sol['n_CO2_a'][-1]
        model_output['n_CO_atm'] = sol['n_CO_a'][-1]
        model_output['n_CH4_atm'] = sol['n_CH4_a'][-1]
        model_output['n_N2_atm'] = sol['n_N2_a'][-1]
        model_output['n_S2_atm'] = sol['n_S2_a'][-1]
        model_output['n_H2_mantle'] = sol['n_H2_a_int'][-1]
        model_output['n_H2O_mantle'] = sol['n_H2O_a_int'][-1]
        model_output['n_O2_mantle'] = sol['n_O2_a_int'][-1]
        model_output['n_CO2_mantle'] = sol['n_CO2_a_int'][-1]
        model_output['n_CO_mantle'] = sol['n_CO_a_int'][-1]
        model_output['n_CH4_mantle'] = sol['n_CH4_a_int'][-1]
        model_output['n_N2_mantle'] = sol['n_N2_a_int'][-1]
        model_output['n_S2_mantle'] = sol['n_S2_a_int'][-1]
        model_output['n_He_mantle'] = sol['N_He_int'][-1]/avogadro
        model_output['DtoH'] = sol['N_D'][-1]/sol['N_H'][-1]
        model_output['fO2_a'] = sol['fO2_a'][-1]
        
        return log_likelihood, model_output
        
    except Exception as e:
        print(f"Error in simulation: {e}")
        return -np.inf, None

def log_likelihood(theta):
    """
    Likelihood function for MCMC (no model output saving)
    """
    log_likelihood_val, _ = run_isocalc_simulation(theta)
    return log_likelihood_val

def log_probability(theta):
    """Log posterior probability"""
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta)

def collect_model_outputs(sampler, n_samples=1000):
    """
    Collect model outputs by re-running the best samples
    """
    print(f"Collecting model outputs from best {n_samples} samples...")
    
    # Get flat samples and log probabilities
    flat_samples = sampler.get_chain(discard=5, thin=1, flat=True)
    flat_logprob = sampler.get_log_prob(discard=5, thin=1, flat=True)
    
    # Select best samples
    best_indices = np.argsort(flat_logprob.flatten())[-n_samples:]
    best_samples = flat_samples[best_indices]
    
    model_outputs = []
    for i, theta in enumerate(best_samples):
        try:
            _, model_output = run_isocalc_simulation(theta)
            if model_output is not None:
                model_outputs.append(model_output)
            
            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1}/{len(best_samples)} samples")
                
        except Exception as e:
            print(f"Error processing sample {i}: {e}")
            continue
    
    print(f"Successfully collected {len(model_outputs)} model outputs")
    return model_outputs

def main():
    # MCMC parameters
    ndim = 4  # [log_f_atm, t_pms, time, log_OtoH_enhancement]
    nwalkers = 8
    nsteps = 20
    
    # Initialize walkers around reasonable starting points
    starting_point = [-3.0, 1e8, 5e9, 0]  # (log_f_atm, t_pms, time, log_OtoH_enhancement)
    pos = starting_point + 1e-4 * np.random.randn(nwalkers, ndim)
    
    # Ensure initial positions are within priors
    for i in range(nwalkers):
        while log_prior(pos[i]) == -np.inf:
            pos[i] = starting_point + 1e-4 * np.random.randn(ndim)
    
    # Run MCMC
    print("Starting MCMC...")
    print(f"Number of walkers: {nwalkers}")
    print(f"Number of steps: {nsteps}")
    print(f"Number of dimensions: {ndim}")
    
    # For testing without multiprocessing
    # sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability)
    # sampler.run_mcmc(pos, nsteps, progress=True)
    
    # With multiprocessing
    with multiprocessing.Pool() as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, pool=pool)
        sampler.run_mcmc(pos, nsteps, progress=False)
    
    print("MCMC complete!")
    
    # Collect model outputs from best samples
    # model_outputs = collect_model_outputs(sampler, n_samples=min(1000, nwalkers * nsteps // 2))
    model_outputs = collect_model_outputs(sampler, n_samples=20)
    
    print(f"Collected {len(model_outputs)} model outputs")
    
    # Get the samples
    samples = sampler.get_chain()
    
    # Save results
    results = {
        'samples': samples,
        'log_prob': sampler.get_log_prob(),
        'parameter_names': ['log_f_atm', 't_pms', 'time', 'log_OtoH_enhancement'],
        'true_values': {'Rp_final': true_Rp_final, 'flux_ratio': true_flux_ratio, 'mdot': true_mdot},
        'model_outputs': model_outputs
    }
    
    output_filename = 'IsoFATE_LHS1140b_MCMC_results.pickle'
    with open(output_filename, 'wb') as file:
        pickle.dump(results, file)
    
    print(f"Results saved to {output_filename}")
    
    # Print some basic statistics
    flat_samples = sampler.get_chain(discard=5, thin=1, flat=True)
    print("\nParameter estimates:")
    for i, param in enumerate(['log_f_atm', 't_pms', 'time', 'log_OtoH_enhancement']):
        mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
        q = np.diff(mcmc)
        print(f"{param}: {mcmc[1]:.6f} +{q[1]:.6f} -{q[0]:.6f}")

if __name__ == "__main__":
    main()