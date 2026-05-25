# -*- coding: utf-8 -*-
"""
Radial Density Analysis Module for Prime Number Distributions on Spirals
========================================================================
This script analyzes the radial density of prime numbers mapped onto a 2D spiral 
space (such as an Archimedean or Vogel spiral). It computes the empirical radial 
density using concentric shell binning, compares it with the theoretical density 
derived from the Prime Number Theorem, and generates high-quality visualizations.

Formulas applied:
- Spatial mapping: r = sqrt(n)
- Empirical Density: rho_emp = Count(primes in shell) / Area(shell)
- Theoretical Density: rho_theo = 1 / (pi * ln(n)) = 1 / (2 * pi * ln(r))
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

def generate_primes_sieve(limit):
    """Generates a boolean array where index represents primality up to 'limit'."""
    print(f"[1/4] Generating prime numbers up to N = {limit:,}...")
    sieve = np.ones(limit + 1, dtype=bool)
    sieve[0] = sieve[1] = False
    for i in range(2, int(np.sqrt(limit)) + 1):
        if sieve[i]:
            sieve[i*i::i] = False
    return sieve

def calculate_radial_density(limit, bin_width=5.0):
    """
    Computes empirical and theoretical radial densities.
    
    Parameters:
        limit (int): Maximum integer to evaluate.
        bin_width (float): Width of concentric radial shells (delta_r).
    """
    print("[2/4] Mapping numbers to spiral coordinates and computing density bins...")
    sieve = generate_primes_sieve(limit)
    numbers = np.arange(1, limit + 1)
    primes_mask = sieve[1:]
    
    # Radius in the spiral is proportional to the square root of the number: r = sqrt(n)
    radii_all = np.sqrt(numbers)
    radii_primes = radii_all[primes_mask]
    
    max_r = np.sqrt(limit)
    bins = np.arange(10, max_r, bin_width) # Start from r=10 to avoid singularity at ln(1)
    
    empirical_densities = []
    theoretical_densities = []
    bin_centers = []
    prime_counts = []
    total_counts = []
    
    for i in range(len(bins) - 1):
        r_start = bins[i]
        r_end = bins[i+1]
        r_mid = (r_start + r_end) / 2
        
        # Count primes and total points in this radial shell
        p_count = np.sum((radii_primes >= r_start) & (radii_primes < r_end))
        t_count = np.sum((radii_all >= r_start) & (radii_all < r_end))
        
        # Shell area: pi * (r_end^2 - r_start^2)
        shell_area = np.pi * (r_end**2 - r_start**2)
        
        # Empirical density of primes per unit area
        rho_emp = p_count / shell_area if shell_area > 0 else 0
        
        # Theoretical density from Prime Number Theorem: 1 / (pi * ln(n)) where n = r_mid^2
        # rho(r) = 1 / (2 * pi * ln(r))
        rho_theo = 1.0 / (2.0 * np.pi * np.log(r_mid)) if r_mid > 1 else 0
        
        bin_centers.append(r_mid)
        empirical_densities.append(rho_emp)
        theoretical_densities.append(rho_theo)
        prime_counts.append(p_count)
        total_counts.append(t_count)
        
    # Compile into a structured Pandas DataFrame
    df = pd.DataFrame({
        'Radius_Center': bin_centers,
        'Equivalent_N': np.square(bin_centers),
        'Prime_Count': prime_counts,
        'Total_Points': total_counts,
        'Empirical_Density': empirical_densities,
        'Theoretical_Density': theoretical_densities
    })
    return df

def plot_and_save_results(df, output_img='radial_density_plot.png', output_csv='radial_density_results.csv'):
    """Generates professional plots and saves data files."""
    print(f"[3/4] Exporting data to {output_csv}...")
    df.to_csv(output_csv, index=False)
    
    print(f"[4/4] Creating professional visualization in {output_img}...")
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # Plot 1: Densities comparison
    ax1.plot(df['Radius_Center'], df['Empirical_Density'], label='Empirical Density (Data)', 
             color='#1f77b4', alpha=0.6, linewidth=1.5)
    ax1.plot(df['Radius_Center'], df['Theoretical_Density'], label='Theoretical Density (1 / [2π ln(r)])', 
             color='#d62728', linestyle='--', linewidth=2.0)
    
    # Adding a moving average to smooth the noisy empirical data
    if len(df) > 10:
        df['Empirical_Smooth'] = df['Empirical_Density'].rolling(window=10, center=True).mean()
        ax1.plot(df['Radius_Center'], df['Empirical_Smooth'], label='Empirical (10-bin Moving Avg)', 
                 color='#2ca02c', linewidth=2.0)
                 
    ax1.set_title('Radial Density Profile of Prime Numbers on a Spiral Space', fontsize=14, fontweight='bold', pad=15)
    ax1.set_ylabel('Density (Primes per Unit Area)', fontsize=12)
    ax1.legend(frameon=True, facecolor='white', framealpha=0.9, fontsize=10)
    ax1.tick_params(axis='both', labelsize=10)
    
    # Plot 2: Relative Error / Residuals
    relative_error = ((df['Empirical_Density'] - df['Theoretical_Density']) / df['Theoretical_Density']) * 100
    if len(df) > 10:
        smooth_error = pd.Series(relative_error).rolling(window=10, center=True).mean()
        ax2.plot(df['Radius_Center'], smooth_error, color='#7f7f7f', linewidth=2.0, label='Smoothed Relative Error')
    
    ax2.scatter(df['Radius_Center'], relative_error, color='#7f7f7f', alpha=0.3, s=10, label='Raw Bin Deviation')
    ax2.axhline(0, color='black', linestyle=':', linewidth=1)
    ax2.set_title('Relative Deviation from Prime Number Theorem (%)', fontsize=12, fontweight='semibold')
    ax2.set_xlabel('Radial Distance from Center (r = √n)', fontsize=12)
    ax2.set_ylabel('Deviation (%)', fontsize=12)
    ax2.legend(frameon=True, facecolor='white', framealpha=0.9, fontsize=10)
    ax2.tick_params(axis='both', labelsize=10)
    
    plt.tight_layout()
    plt.savefig(output_img, dpi=300)
    plt.close()
    print("Analysis successfully finished and files saved!")

if __name__ == '__main__':
    # Default parameters for quick evaluation (can be scaled up)
    MAX_NUMBER = 2_000_000
    BIN_SIZE = 4.0
    
    df_results = calculate_radial_density(MAX_NUMBER, bin_width=BIN_SIZE)
    plot_and_save_results(df_results)