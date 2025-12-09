'''
Collin Cherubim

Plotting script for isofate_MCMC.py
Modified to handle both n_atmodeller = 0 and n_atmodeller != 0 cases
'''
import pickle

import corner
import matplotlib.pyplot as plt
import numpy as np


filename = '/Users/collin/Documents/Harvard/Research/isofate_mcmc/IsoFATE_LHS1140b_MCMC_results_n1e5_natmod0_eps15_tjump_Ffinal17_constant_tpms_OH_12w_50s.pickle'

def detect_mode(model_outputs):
    """
    Detect whether the MCMC was run with n_atmodeller = 0 or != 0
    
    Returns:
    --------
    mode : str
        'basic' for n_atmodeller = 0, 'atmodeller' for n_atmodeller != 0
    """
    # Check first non-None output
    for output in model_outputs.flat:
        if output is not None:
            if 'N_H' in output:
                return 'basic'
            elif 'n_H2_atm' in output:
                return 'atmodeller'
    return None

def plot_mcmc_results(filename=filename):
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
    plt.savefig(filename.replace('.pickle', '_mcmc_corner_plot_initial_conditions.png'), dpi=300, bbox_inches='tight')
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

def plot_model_outputs_corner(filename=filename):
    """
    Create corner plots for model outputs
    Handles both n_atmodeller = 0 (basic species) and n_atmodeller != 0 (atmodeller) cases
    """
    
    # Load the results
    with open(filename, 'rb') as file:
        results = pickle.load(file)
    
    model_outputs_raw = results['model_outputs']
    
    # Handle the 2D array structure from sampler.get_blobs()
    print(f"Model outputs shape: {model_outputs_raw.shape}")
    
    # Detect which mode we're in
    mode = detect_mode(model_outputs_raw)
    print(f"Detected mode: {mode}")
    
    # Flatten the 2D array and filter out None values
    model_outputs = []
    total_samples = model_outputs_raw.size
    successful_samples = 0
    
    for output in model_outputs_raw.flat:
        if output is not None:
            model_outputs.append(output)
            successful_samples += 1
    
    print(f"Total samples: {total_samples}, Successful: {successful_samples}")
    
    if len(model_outputs) == 0:
        print("No valid model outputs found!")
        return None, None
    
    print(f"Processing {len(model_outputs)} model outputs...")
    
    # Extract data based on mode
    if mode == 'basic':
        # n_atmodeller = 0: Plot N_H, N_He, N_D, N_O, N_C, N_N, N_S
        species_data = []
        species_names = ['N_H', 'N_He', 'N_D', 'N_O', 'N_C', 'N_N', 'N_S']
        
        for output in model_outputs:
            row_data = []
            for species in species_names:
                if species in output and output[species] is not None:
                    row_data.append(output[species])
                else:
                    row_data.append(np.nan)
            species_data.append(row_data)
        
        species_labels = [f'{s} [mol]' for s in species_names]
        plot_title = 'MCMC Results: Species Moles (Basic Mode)'
        save_suffix = '_mcmc_corner_plot_species_basic.png'
        
    elif mode == 'atmodeller':
        # n_atmodeller != 0: Plot atmospheric/mantle species and temperature
        species_data = []
        atm_species = ['n_H2_atm', 'n_He_atm', 'n_H2O_atm', 'n_O2_atm', 'n_CO2_atm', 'n_CO_atm', 'n_CH4_atm', 'n_N2_atm', 'n_S2_atm']
        mantle_species = ['n_H2_mantle', 'n_He_mantle', 'n_H2O_mantle', 'n_O2_mantle', 'n_CO2_mantle', 'n_CO_mantle', 'n_CH4_mantle', 'n_N2_mantle', 'n_S2_mantle']
        
        for output in model_outputs:
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
        
        # Create labels
        species_labels = []
        for species in atm_species:
            species_name = species.replace('n_', '').replace('_atm', '')
            species_labels.append(f'{species_name} (atm)')
        for species in mantle_species:
            species_name = species.replace('n_', '').replace('_mantle', '')
            species_labels.append(f'{species_name} (mantle)')
        species_labels.append(r'$T_{\rm surf}$ [K]')
        
        plot_title = 'MCMC Results: Species Moles & Surface Temperature (Atmodeller)'
        save_suffix = '_mcmc_corner_plot_species_atmodeller.png'
        
    else:
        print("Could not detect mode!")
        return None, None
    
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
    
    # Check for columns with no variation and remove them
    print("Checking for columns with no variation...")
    valid_columns = []
    valid_labels = []
    
    for i in range(species_array.shape[1]):
        col_min = np.min(species_array[:, i])
        col_max = np.max(species_array[:, i])
        col_range = col_max - col_min
        
        print(f"Column {i} ({species_labels[i]}): min={col_min:.2e}, max={col_max:.2e}, range={col_range:.2e}")
        
        if col_range > 1e-15:  # Keep columns with some variation
            valid_columns.append(i)
            valid_labels.append(species_labels[i])
        else:
            print(f"  -> Removing column {i} ({species_labels[i]}) - no variation")
    
    if len(valid_columns) == 0:
        print("No columns with variation found!")
        return None, None
    
    # Keep only columns with variation
    filtered_array = species_array[:, valid_columns]
    
    print(f"Plotting {len(valid_columns)} parameters with variation")
    
    # Create corner plot
    fig1 = corner.corner(
        filtered_array,
        labels=valid_labels,
        quantiles=[0.16, 0.5, 0.84],
        show_titles=True,
        title_kwargs={"fontsize": 10},
        label_kwargs={"fontsize": 12},
        title_fmt='.2e'
    )
    
    plt.suptitle(plot_title, fontsize=14, y=0.98)
    
    plt.savefig(filename.replace('.pickle', save_suffix), dpi=300, bbox_inches='tight')
    plt.show()
    
    return filtered_array, fig1

def plot_molar_abundances_corner(filename=filename):
    """
    Create corner plot for atmospheric species molar abundances
    Only works for n_atmodeller != 0 (atmodeller mode)
    """
    
    # Load the results
    with open(filename, 'rb') as file:
        results = pickle.load(file)
    
    model_outputs_raw = results['model_outputs']
    
    # Detect which mode we're in
    mode = detect_mode(model_outputs_raw)
    
    if mode != 'atmodeller':
        print(f"Molar abundance plotting only available for atmodeller mode (n_atmodeller != 0).")
        print(f"Current mode is: {mode}")
        return None, None
    
    # Handle the 2D array structure from sampler.get_blobs()
    # Flatten the 2D array and filter out None values
    model_outputs = []
    for output in model_outputs_raw.flat:
        if output is not None:
            model_outputs.append(output)
    
    if len(model_outputs) == 0:
        print("No valid model outputs found!")
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
    
    # Check for columns with no variation and remove them
    print("Checking molar abundance columns for variation...")
    species_names = ['H₂', 'He', 'H₂O', 'O₂', 'CO₂', 'CO', 'CH₄', 'N₂', 'S₂']
    
    valid_columns = []
    valid_labels = []
    
    for i in range(abundance_array.shape[1]):
        col_min = np.min(abundance_array[:, i])
        col_max = np.max(abundance_array[:, i])
        col_range = col_max - col_min
        
        print(f"Column {i} ({species_names[i]}): min={col_min:.4f}, max={col_max:.4f}, range={col_range:.4f}")
        
        if col_range > 1e-10:  # Keep columns with some variation (smaller threshold for fractions)
            valid_columns.append(i)
            valid_labels.append(f'$\\log x_{{{species_names[i]}}}$')
        else:
            print(f"  -> Removing column {i} ({species_names[i]}) - no variation")
    
    if len(valid_columns) == 0:
        print("No molar abundance columns with variation found!")
        return None, None
    
    # Keep only columns with variation
    filtered_abundance_array = abundance_array[:, valid_columns]
    
    # Print parameter estimates in linear scale (not log scale)
    print("\nMolar abundance estimates (median with 1-sigma bounds):")
    print("-" * 60)
    for i, col_idx in enumerate(valid_columns):
        species_name = species_names[col_idx]
        mcmc = np.percentile(filtered_abundance_array[:, i], [16, 50, 84])
        q = np.diff(mcmc)
        print(f"x_{species_name:8s}: {mcmc[1]:8.4f} +{q[1]:7.4f} -{q[0]:7.4f}")
    
    print(f"Plotting {len(valid_columns)} molar abundances with variation")
    
    # Convert to log scale for plotting (handle zeros and very small values)
    log_abundance_array = np.log10(filtered_abundance_array + 1e-15)
    
    # Create corner plot for molar abundances (in log scale)
    fig2 = corner.corner(
        log_abundance_array,
        labels=valid_labels,
        quantiles=[0.16, 0.5, 0.84],
        show_titles=True,
        title_kwargs={"fontsize": 12},
        label_kwargs={"fontsize": 14},
        title_fmt='.2f'
    )
    
    plt.suptitle('MCMC Results: Atmospheric Species Molar Fractions (Log Scale)', 
                 fontsize=16, y=0.98)
    
    plt.savefig(filename.replace('.pickle', '_mcmc_corner_plot_molar_abundances.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    return filtered_abundance_array, fig2

def plot_chains(filename=filename):
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
    
    # Handle case where there's only one parameter
    if len(parameter_names) == 1:
        axes = [axes]
    
    for i in range(len(parameter_names)):
        ax = axes[i]
        ax.plot(samples[:, :, i], "k", alpha=0.3)
        ax.set_ylabel(labels[i])
        ax.yaxis.set_label_coords(-0.1, 0.5)
    
    axes[-1].set_xlabel("Step number")
    plt.tight_layout()
    plt.savefig(filename.replace('.pickle', '_mcmc_chains.png'), dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """
    Main function to create plots
    Automatically detects mode and creates appropriate plots
    """
    # Load results to detect mode
    with open(filename, 'rb') as file:
        results = pickle.load(file)
    
    mode = detect_mode(results['model_outputs'])
    print(f"\n{'='*60}")
    print(f"Running plotting script in {mode.upper()} mode")
    print(f"{'='*60}\n")
    
    # Create corner plot for initial conditions
    flat_samples, corner_fig = plot_mcmc_results()
    
    # Create corner plot for species (works for both modes)
    species_data, species_fig = plot_model_outputs_corner()
    
    # Create corner plot for molar abundances (only for atmodeller mode)
    if mode == 'atmodeller':
        abundance_data, abundance_fig = plot_molar_abundances_corner()
    else:
        print("\nSkipping molar abundance plot (only available for atmodeller mode)")
    
    # Create chain plots to check convergence
    plot_chains()
    
    print("\n" + "="*60)
    print("Plots saved as:")
    print("-", filename.replace('.pickle', '_mcmc_corner_plot_initial_conditions.png'))
    if mode == 'basic':
        print("-", filename.replace('.pickle', '_mcmc_corner_plot_species_basic.png'))
    elif mode == 'atmodeller':
        print("-", filename.replace('.pickle', '_mcmc_corner_plot_species_atmodeller.png'))
        print("-", filename.replace('.pickle', '_mcmc_corner_plot_molar_abundances.png'))
    print("-", filename.replace('.pickle', '_mcmc_chains.png'))
    print("="*60)

if __name__ == "__main__":
    main()