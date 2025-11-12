'''
Collin Cherubim

Plotting script for isofate_MCMC.py
'''
import pickle

import corner
import matplotlib.pyplot as plt
import numpy as np


def plot_mcmc_results(filename='IsoFATE_LHS1140b_MCMC_results.pickle'):
    """
    Load MCMC results and create corner plot
    
    Parameters:
    -----------
    filename : str
        Path to the pickle file containing MCMC results
    """
    
    # Load the results
    print(f"Loading results from {filename}...")
    with open(filename, 'rb') as file:
        results = pickle.load(file)
    
    # Extract data
    samples = results['samples']
    parameter_names = results['parameter_names']
    
    # Print basic info
    print(f"Chain shape: {samples.shape}")
    print(f"Parameters: {parameter_names}")
    
    # Flatten the chain (discard burn-in and thin if needed)
    # Adjust discard and thin values based on your chain
    discard = 5  # Discard first step as burn-in
    thin = 1      # Take every sample
    flat_samples = samples[discard::thin, :, :].reshape((-1, samples.shape[-1]))
    
    print(f"Flattened samples shape: {flat_samples.shape}")
    
    # Create parameter labels with units/descriptions
    labels = [
        r'$\log f_{\rm atm}$',
        r'$t_{\rm PMS}$ [yr]',
        r'$t$ [yr]',
        r'$\log$ O/H enhancement'
    ]
    
    # Create corner plot
    fig = corner.corner(
        flat_samples,
        labels=labels,
        quantiles=[0.16, 0.5, 0.84],  # Show 1-sigma bounds
        show_titles=True,
        title_kwargs={"fontsize": 12},
        label_kwargs={"fontsize": 14},
        title_fmt='.4f'  # Format for displayed values
    )
    
    plt.suptitle('MCMC Results: Initial Conditions', 
                 fontsize=16, y=0.98)
    
    # Save the plot
    plt.savefig('mcmc_corner_plot_initial_conditions.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print parameter estimates
    print("\nParameter estimates (median with 1-sigma bounds):")
    print("-" * 60)
    for i, param in enumerate(parameter_names):
        mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
        q = np.diff(mcmc)
        print(f"{param:20s}: {mcmc[1]:8.4f} +{q[1]:7.4f} -{q[0]:7.4f}")
    
    # Print acceptance fraction if available
    if 'log_prob' in results:
        log_prob = results['log_prob']
        print(f"\nLog probability shape: {log_prob.shape}")
        print(f"Final log probability range: {np.min(log_prob[-100:]):.2f} to {np.max(log_prob[-100:]):.2f}")
    
    return flat_samples, fig

def plot_model_outputs_corner(filename='IsoFATE_LHS1140b_MCMC_results.pickle'):
    """
    Create corner plots for model outputs: species moles and surface temperature
    """
    
    # Load the results
    with open(filename, 'rb') as file:
        results = pickle.load(file)
    
    model_outputs = results['model_outputs']
    
    if not model_outputs:
        print("No model outputs found in results!")
        return None, None
    
    print(f"Processing {len(model_outputs)} model outputs...")
    
    # Extract species moles and temperature data
    species_data = []
    
    # Define species names for atmosphere and mantle (using new keys)
    atm_species = ['n_H2_atm', 'n_He_atm', 'n_H2O_atm', 'n_O2_atm', 'n_CO2_atm', 'n_CO_atm', 'n_CH4_atm', 'n_N2_atm', 'n_S2_atm']
    mantle_species = ['n_H2_mantle', 'n_He_mantle', 'n_H2O_mantle', 'n_O2_mantle', 'n_CO2_mantle', 'n_CO_mantle', 'n_CH4_mantle', 'n_N2_mantle', 'n_S2_mantle']
    
    for output in model_outputs:
        # Extract species moles and temperature
        row_data = []
        
        # Atmosphere species
        for species in atm_species:
            if species in output and output[species] is not None:
                row_data.append(output[species])
            else:
                row_data.append(np.nan)
        
        # Mantle species  
        for species in mantle_species:
            if species in output and output[species] is not None:
                row_data.append(output[species])
            else:
                row_data.append(np.nan)
        
        # Surface temperature
        if 'T_surf_atmod' in output and output['T_surf_atmod'] is not None:
            row_data.append(output['T_surf_atmod'])
        else:
            row_data.append(np.nan)
        
        species_data.append(row_data)
    
    if not species_data:
        print("No valid species data found!")
        return None, None
    
    species_array = np.array(species_data)
    
    # Remove rows with any NaN values
    valid_rows = ~np.isnan(species_array).any(axis=1)
    species_array = species_array[valid_rows]
    
    if len(species_array) == 0:
        print("No valid data after removing NaNs!")
        return None, None
    
    print(f"Valid samples for species plot: {len(species_array)}")
    
    # Create labels for species corner plot
    species_labels = []
    
    # Atmosphere species labels
    for species in atm_species:
        species_name = species.replace('n_', '').replace('_atm', '')
        species_labels.append(f'{species_name} (atm)')
    
    # Mantle species labels
    for species in mantle_species:
        species_name = species.replace('n_', '').replace('_mantle', '')
        species_labels.append(f'{species_name} (mantle)')
    
    # Temperature label
    species_labels.append(r'$T_{\rm surf}$ [K]')
    
    # Create corner plot for species and temperature
    fig1 = corner.corner(
        species_array,
        labels=species_labels,
        quantiles=[0.16, 0.5, 0.84],
        show_titles=True,
        title_kwargs={"fontsize": 10},
        label_kwargs={"fontsize": 12},
        title_fmt='.2e'
    )
    
    plt.suptitle('MCMC Results: Species Moles & Surface Temperature', 
                 fontsize=14, y=0.98)
    
    plt.savefig('mcmc_corner_plot_species_temperature.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return species_array, fig1

def plot_molar_abundances_corner(filename='IsoFATE_LHS1140b_MCMC_results.pickle'):
    """
    Create corner plot for atmospheric species molar abundances
    """
    
    # Load the results
    with open(filename, 'rb') as file:
        results = pickle.load(file)
    
    model_outputs = results['model_outputs']
    
    if not model_outputs:
        print("No model outputs found in results!")
        return None, None
    
    print(f"Processing {len(model_outputs)} model outputs for molar abundances...")
    
    # Extract molar abundance data
    abundance_data = []
    
    # Define atmospheric species for molar abundance calculation
    atm_species = ['n_H2_atm', 'n_He_atm', 'n_H2O_atm', 'n_O2_atm', 'n_CO2_atm', 'n_CO_atm', 'n_CH4_atm', 'n_N2_atm', 'n_S2_atm']
    
    for output in model_outputs:
        # Calculate total moles and molar fractions
        n_values = []
        valid_sample = True
        
        # Extract mole numbers for each species
        for species in atm_species:
            if species in output and output[species] is not None:
                n_values.append(output[species])
            else:
                valid_sample = False
                break
        
        if not valid_sample:
            continue
            
        # Calculate total moles
        n_tot = sum(n_values)
        
        if n_tot <= 0:
            continue
            
        # Calculate molar fractions
        molar_fractions = [n / n_tot for n in n_values]
        abundance_data.append(molar_fractions)
    
    if not abundance_data:
        print("No valid molar abundance data found!")
        return None, None
    
    abundance_array = np.array(abundance_data)
    
    # Remove rows with any NaN values
    valid_rows = ~np.isnan(abundance_array).any(axis=1)
    abundance_array = abundance_array[valid_rows]
    
    if len(abundance_array) == 0:
        print("No valid abundance data after removing NaNs!")
        return None, None
    
    print(f"Valid samples for molar abundances plot: {len(abundance_array)}")
    
    # Create labels for molar abundances
    species_names = ['H₂', 'He', 'H₂O', 'O₂', 'CO₂', 'CO', 'CH₄', 'N₂', 'S₂']
    abundance_labels = [f'$x_{{{name}}}$' for name in species_names]
    
    # Create corner plot for molar abundances
    fig2 = corner.corner(
        abundance_array,
        labels=abundance_labels,
        quantiles=[0.16, 0.5, 0.84],
        show_titles=True,
        title_kwargs={"fontsize": 12},
        label_kwargs={"fontsize": 14},
        title_fmt='.4f'
    )
    
    plt.suptitle('MCMC Results: Atmospheric Species Molar Fractions', 
                 fontsize=16, y=0.98)
    
    plt.savefig('mcmc_corner_plot_molar_abundances.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return abundance_array, fig2

def plot_chains(filename='IsoFATE_LHS1140b_MCMC_results.pickle'):
    """
    Plot the MCMC chains to check for convergence
    """
    
    # Load the results
    with open(filename, 'rb') as file:
        results = pickle.load(file)
    
    samples = results['samples']
    parameter_names = results['parameter_names']
    
    # Create chain plots
    fig, axes = plt.subplots(len(parameter_names), figsize=(10, 8), sharex=True)
    
    labels = [
        r'$\log f_{\rm atm}$',
        r'$t_{\rm PMS}$ [yr]',
        r'$t$ [yr]',
        r'$\log$ O/H enhancement'
    ]
    
    for i in range(len(parameter_names)):
        ax = axes[i]
        ax.plot(samples[:, :, i], "k", alpha=0.3)
        ax.set_ylabel(labels[i])
        ax.yaxis.set_label_coords(-0.1, 0.5)
    
    axes[-1].set_xlabel("Step number")
    plt.tight_layout()
    plt.savefig('mcmc_chains.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """
    Main function to create plots
    """
    # Create corner plot for initial conditions
    flat_samples, corner_fig = plot_mcmc_results()
    
    # Create corner plot for species moles and surface temperature
    species_data, species_fig = plot_model_outputs_corner()
    
    # Create corner plot for molar abundances
    abundance_data, abundance_fig = plot_molar_abundances_corner()
    
    # Create chain plots to check convergence
    plot_chains()
    
    print("\nPlots saved as:")
    print("- mcmc_corner_plot_initial_conditions.png")
    print("- mcmc_corner_plot_species_temperature.png")  
    print("- mcmc_corner_plot_molar_abundances.png")
    print("- mcmc_chains.png")

if __name__ == "__main__":
    main()