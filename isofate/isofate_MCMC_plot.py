'''
Collin Cherubim

Plotting script for isofate_MCMC.py
Modified to handle both n_atmodeller = 0 and n_atmodeller != 0 cases
'''
import pickle

import corner
import matplotlib.pyplot as plt
import numpy as np
from isofate.constants import avogadro
from isofate.constants import Re


# filename = '/Users/collin/Documents/Harvard/Research/isofate_mcmc/IsoFATE_LHS1140b_MCMC_results_n1e5_natmod0_eps15_tjump_Ffinal17_constant_tpms_OH_48w_7500s.pickle'
filename = '/Users/collin/Documents/Harvard/Research/isofate_mcmc/IsoFATE_LHS1140b_MCMC_results_n1e5_natmod1e3_eps15_tjump_Ffinal17_64w_500s.pickle'

# GLOBAL PARAMETERS FOR BURN-IN AND THINNING
DISCARD = 350  # Number of steps to discard as burn-in
THIN = 1        # Take every Nth sample

# PLOT RANGE LIMITS (set to None for auto-range)
# Format: [min, max] for each parameter, or None for auto
SPECIES_PLOT_RANGE = None  # Set to None for automatic ranging
# SPECIES_PLOT_RANGE = [(8, 24), (20, 30), (12, 24), (20, 24), (20, 24), (20, 24), (20, 24)] 

# Mapping from parameter names to their latex labels
PARAMETER_LABELS = {
    'log_f_atm': r'$\log f_{\rm atm}$',
    't_pms': r'$t_{\rm PMS}$ [yr]',
    'time': r'$t$ [yr]',
    'log_OtoH_enhancement': r'$\log$ O/H enhancement'
}

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
    model_outputs_raw = results['model_outputs']
    true_values = results['true_values']
    
    # Print basic info
    print(f"Chain shape: {samples.shape}")
    print(f"Parameters: {parameter_names}")
    
    # Flatten the chain (discard burn-in and thin if needed)
    flat_samples = samples[DISCARD::THIN, :, :].reshape((-1, samples.shape[-1]))
    
    print(f"Flattened samples shape: {flat_samples.shape}")
    
    # Process model outputs with same discard and thin
    model_outputs_processed = model_outputs_raw[DISCARD::THIN, :]
    
    # Extract model output values
    model_output_data = []
    for output in model_outputs_processed.flat:
        if output is not None:
            # Extract and convert values
            Rp_final = output['Rp_final'] / Re  # Convert to Earth radii
            flux_ratio = output['flux_ratio']
            mdot = output['mdot'] * 1e3  # Convert to g/s
            model_output_data.append([Rp_final, flux_ratio, mdot])
        else:
            model_output_data.append([np.nan, np.nan, np.nan])
    
    model_output_array = np.array(model_output_data)
    
    # Check that shapes match
    print(f"Flat samples shape: {flat_samples.shape}")
    print(f"Model output array shape: {model_output_array.shape}")
    
    # Combine parameter samples with model outputs
    combined_samples = np.hstack([flat_samples, model_output_array])
    
    # Remove rows with any NaN values
    valid_rows = ~np.isnan(combined_samples).any(axis=1)
    combined_samples = combined_samples[valid_rows]
    
    print(f"Combined samples shape after removing NaNs: {combined_samples.shape}")
    
    # Create parameter labels dynamically based on actual parameters
    labels = [PARAMETER_LABELS[param] for param in parameter_names]
    
    # Add model output labels
    labels.extend([
        r'$R_p$ [$R_{\oplus}$]',
        r'H/He flux ratio',
        r'$\dot{M}$ [g/s]'
    ])
    
    # Create truth values array (convert to same units as plotted data)
    truths = [None] * len(parameter_names)  # No truth values for parameters
    truths.extend([
        true_values['Rp_final'] / Re,  # Convert to Earth radii
        true_values['flux_ratio'],
        true_values['mdot'] * 1e3  # Convert to g/s
    ])
    
    # Create corner plot
    fig = corner.corner(
        combined_samples,
        labels=labels,
        quantiles=[0.16, 0.5, 0.84],  # Show 1-sigma bounds
        show_titles=True,
        title_kwargs={"fontsize": 12},
        label_kwargs={"fontsize": 14},
        title_fmt='.4f',  # Format for displayed values
        truths=truths,  # Add vertical lines for true values
        truth_color='red'  # Color for truth lines
    )
    
    plt.suptitle('MCMC Results: Initial Conditions & Model Outputs', 
                 fontsize=16, y=0.98)
    
    # Save the plot
    plt.savefig(filename.replace('.pickle', '_mcmc_corner_plot_initial_conditions.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print parameter estimates
    print("\nParameter estimates (median with 1-sigma bounds):")
    print("-" * 60)
    for i, param in enumerate(parameter_names):
        mcmc = np.percentile(combined_samples[:, i], [16, 50, 84])
        q = np.diff(mcmc)
        if param == 'time' or param == 'mdot':
            print(f"{param:20s}: {mcmc[1]:8.4e} +{q[1]:7.4e} -{q[0]:7.4e}")
        else:
            print(f"{param:20s}: {mcmc[1]:8.4f} +{q[1]:7.4f} -{q[0]:7.4f}")
    
    # Print model output estimates
    print("\nModel output estimates (median with 1-sigma bounds):")
    print("-" * 60)
    model_output_names = ['Rp [R_Earth]', 'flux_ratio', 'mdot [g/s]']
    for i, name in enumerate(model_output_names):
        idx = len(parameter_names) + i
        mcmc = np.percentile(combined_samples[:, idx], [16, 50, 84])
        q = np.diff(mcmc)
        print(f"{name:20s}: {mcmc[1]:8.4f} +{q[1]:7.4f} -{q[0]:7.4f}")
    
    # Print true values for comparison
    print("\nTrue values:")
    print("-" * 60)
    print(f"{'Rp [R_Earth]':20s}: {true_values['Rp_final']/Re:8.4f}")
    print(f"{'flux_ratio':20s}: {true_values['flux_ratio']:8.4f}")
    print(f"{'mdot [g/s]':20s}: {true_values['mdot']*1e3:8.4f}")
    
    # Print acceptance fraction if available
    if 'log_prob' in results:
        log_prob = results['log_prob']
        print(f"\nLog probability shape: {log_prob.shape}")
        print(f"Final log probability range: {np.min(log_prob[-100:]):.2f} to {np.max(log_prob[-100:]):.2f}")
    
    return combined_samples, fig

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
    
    # Apply discard and thin BEFORE processing
    model_outputs_processed = model_outputs_raw[DISCARD::THIN, :]
    print(f"Model outputs shape after discard/thin: {model_outputs_processed.shape}")
    
    # Detect which mode we're in
    mode = detect_mode(model_outputs_processed)
    print(f"Detected mode: {mode}")
    
    # Flatten the 2D array and filter out None values
    model_outputs = []
    total_samples = model_outputs_processed.size
    successful_samples = 0
    
    for output in model_outputs_processed.flat:
        if output is not None:
            model_outputs.append(output)
            successful_samples += 1
    
    print(f"Total samples (after discard/thin): {total_samples}, Successful: {successful_samples}")
    
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
                    row_data.append(output[species]/avogadro)
                else:
                    row_data.append(np.nan)
            species_data.append(row_data)
        
        species_labels = [f'$\\log$ {s} [mol]' for s in species_names]
        plot_title = 'MCMC Results: Species Moles (Basic Mode, Log Scale)'
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
            
            # Surface temperature (keep linear, not log)
            if 'T_surf_atmod' in output and output['T_surf_atmod'] is not None:
                row_data.append(output['T_surf_atmod'])
            else:
                row_data.append(np.nan)
            
            species_data.append(row_data)
        
        # Create labels
        species_labels = []
        for species in atm_species:
            species_name = species.replace('n_', '').replace('_atm', '')
            species_labels.append(f'$\\log$ {species_name} (atm) [mol]')
        for species in mantle_species:
            species_name = species.replace('n_', '').replace('_mantle', '')
            species_labels.append(f'$\\log$ {species_name} (mantle) [mol]')
        species_labels.append(r'$T_{\rm surf}$ [K]')  # Temperature stays linear
        
        plot_title = 'MCMC Results: Species Moles & Surface Temperature (Atmodeller, Log Scale)'
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
    
    # Convert species to log scale (all columns except temperature if present)
    # Check if last valid column is temperature (only for atmodeller mode)
    if mode == 'atmodeller' and 'T_{\rm surf}' in valid_labels[-1]:
        # Last column is temperature, keep it linear
        log_species_array = filtered_array.copy()
        log_species_array[:, :-1] = np.log10(filtered_array[:, :-1] + 1e-15)  # Log for species
        # Temperature column stays as-is
    else:
        # All columns are species, convert all to log
        log_species_array = np.log10(filtered_array + 1e-15)
    
    # Create corner plot
# Create corner plot
    fig1 = corner.corner(
        log_species_array,
        labels=valid_labels,
        quantiles=[0.16, 0.5, 0.84],
        show_titles=True,
        title_kwargs={"fontsize": 10},
        label_kwargs={"fontsize": 12},
        title_fmt='.2f',  # Changed format for log values
        range=SPECIES_PLOT_RANGE  # Add this line
    )
    
    plt.suptitle(plot_title, fontsize=14, y=0.98)
    
    plt.savefig(filename.replace('.pickle', save_suffix), dpi=300, bbox_inches='tight')
    plt.show()
    
    return filtered_array, fig1  # Return original array, not log

def plot_molar_abundances_corner(filename=filename):
    """
    Create corner plot for atmospheric species molar abundances
    Only works for n_atmodeller != 0 (atmodeller mode)
    """
    
    # Load the results
    with open(filename, 'rb') as file:
        results = pickle.load(file)
    
    model_outputs_raw = results['model_outputs']
    
    # Apply discard and thin BEFORE processing
    model_outputs_processed = model_outputs_raw[DISCARD::THIN, :]
    print(f"Model outputs shape after discard/thin: {model_outputs_processed.shape}")
    
    # Detect which mode we're in
    mode = detect_mode(model_outputs_processed)
    
    if mode != 'atmodeller':
        print(f"Molar abundance plotting only available for atmodeller mode (n_atmodeller != 0).")
        print(f"Current mode is: {mode}")
        return None, None
    
    # Handle the 2D array structure from sampler.get_blobs()
    # Flatten the 2D array and filter out None values
    model_outputs = []
    for output in model_outputs_processed.flat:
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
    
    # Create parameter labels dynamically based on actual parameters
    labels = [PARAMETER_LABELS[param] for param in parameter_names]
    
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
    print(f"DISCARD: {DISCARD} steps, THIN: every {THIN} steps")
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