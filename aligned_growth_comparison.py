import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d
import os

# Import the black hole growth simulation function from the previous script
from blackhole_growth import simulate_blackhole_growth, create_example_black_holes

def universe_scale_factor(time_gyr):
    """
    Calculate the scale factor of the universe at different cosmic times
    based on the Lambda-CDM model and observed expansion history.
    
    Parameters:
    -----------
    time_gyr : array
        Cosmic time in billions of years since the Big Bang
    
    Returns:
    --------
    scale_factor : array
        Scale factor of the universe (a=1 at present day)
    """
    # Age of the universe in Gyr
    universe_age = 13.8
    
    # Present time is defined as scale factor = 1.0
    # Very early universe had tiny scale factor
    
    # Convert time to normalized time (t/t_0)
    normalized_time = time_gyr / universe_age
    
    # Different epochs:
    # - Early universe: a ~ t^(1/2) (radiation dominated)
    # - Middle period: a ~ t^(2/3) (matter dominated)
    # - Late universe: a ~ exp(t) (dark energy dominated)
    
    # Early universe (radiation dominated)
    early_mask = normalized_time < 0.05  # First ~700 million years
    
    # Middle universe (matter dominated)
    middle_mask = (normalized_time >= 0.05) & (normalized_time < 0.7)  # ~700 million to ~9.6 billion years
    
    # Late universe (dark energy accelerated expansion)
    late_mask = normalized_time >= 0.7  # Last ~4.2 billion years
    
    # Initialize scale factor array
    scale_factor = np.zeros_like(time_gyr)
    
    # Radiation dominated era: a(t) ~ t^(1/2)
    scale_factor[early_mask] = 0.005 * (normalized_time[early_mask] / 0.05)**0.5
    
    # Matter dominated era: a(t) ~ t^(2/3)
    # Match the previous era at the transition
    early_end_scale = 0.005
    scale_factor[middle_mask] = early_end_scale * (normalized_time[middle_mask] / 0.05)**(2/3)
    
    # Dark energy era: accelerated expansion
    # Using an approximate exponential acceleration
    # Match the previous era at the transition
    middle_end_time = 0.7
    middle_end_scale = early_end_scale * (middle_end_time / 0.05)**(2/3)
    
    # Approximating the acceleration as a stretched exponential
    acceleration_factor = 1.5
    scale_factor[late_mask] = middle_end_scale * np.exp(
        acceleration_factor * (normalized_time[late_mask] - middle_end_time))
    
    # Normalize to get scale factor = 1 at present time
    current_scale = scale_factor[np.abs(normalized_time - 1.0).argmin()]
    scale_factor = scale_factor / current_scale
    
    return scale_factor

def aligned_growth_comparison():
    """
    Create a visualization with all growth curves starting at the origin (time=0)
    """
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Get black hole data
    black_holes = create_example_black_holes()
    
    # Color map for different entities
    colors = plt.cm.viridis(np.linspace(0, 1, len(black_holes) + 1))
    
    # Plot universe expansion curve
    # Create time array from 0 to 13.8 Gyr (Big Bang to now)
    universe_time = np.linspace(0.001, 13.8, 1000)  # Gyr since Big Bang
    universe_scale = universe_scale_factor(universe_time)
    
    # Plot universe scale factor (normalized to 1 at present)
    plt.plot(universe_time, universe_scale, 'k-', linewidth=3, 
             label='Universe Scale Factor')
    
    # Store curves for correlation analysis
    all_curves = [universe_scale]
    all_names = ['Universe']
    
    # Plot each black hole's growth, but shift time to start at origin
    for i, (name, initial_mass, formation_time, growth_model) in enumerate(black_holes):
        # Only include representative black holes
        if name in ["Sagittarius A*", "M87*", "TON 618", "Primordial BH"]:
            # Get black hole growth data
            bh_time, bh_mass = simulate_blackhole_growth(initial_mass, formation_time, growth_model)
            
            # Shift time to start at 0 (formation time becomes 0)
            shifted_time = bh_time - formation_time
            
            # Normalize mass to maximum mass
            normalized_mass = bh_mass / bh_mass[-1]
            
            # Create consistent time points for all curves (0 to 13.8 Gyr)
            aligned_time = np.linspace(0, 13.8 - formation_time, 1000)
            
            # Interpolate the mass data to the aligned time points
            mass_interp = interp1d(shifted_time, normalized_mass, 
                                  bounds_error=False, fill_value=(0, normalized_mass[-1]))
            aligned_mass = mass_interp(aligned_time)
            
            # Plot the aligned growth curve
            plt.plot(aligned_time, aligned_mass, '-', color=colors[i+1], linewidth=2, 
                    label=f"{name} Growth (aligned)")
            
            # Store for correlation
            # Truncate to match universe timespan
            valid_indices = aligned_time <= 13.8
            all_curves.append(aligned_mass[valid_indices])
            all_names.append(name)
    
    # Calculate correlation metrics
    correlations = {}
    # Use the universe scale as reference
    reference_curve = all_curves[0]
    
    for i, curve in enumerate(all_curves[1:]):
        # Find the minimum length between the two arrays
        min_length = min(len(reference_curve), len(curve))
        
        # Compute correlation
        corr = np.corrcoef(reference_curve[:min_length], curve[:min_length])[0, 1]
        correlations[all_names[i+1]] = corr
    
    # Add correlation text
    correlation_text = "Correlation with Universe Scale Factor:\n"
    for name, corr in correlations.items():
        correlation_text += f"{name}: {corr:.3f}\n"
    
    plt.text(0.02, 0.4, correlation_text, transform=plt.gca().transAxes, fontsize=10,
            bbox=dict(facecolor='white', alpha=0.8))
    
    # Add labels and title
    plt.xlabel('Time (billion years)', fontsize=12)
    plt.ylabel('Normalized Scale/Mass', fontsize=12)
    plt.title('Aligned Growth Comparison: Universe Expansion vs. Black Hole Growth', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='lower right')
    plt.tight_layout()
    
    # Save the figure
    plt.savefig('aligned_growth_comparison.png', dpi=300)
    print("Aligned growth comparison saved as 'aligned_growth_comparison.png'")
    
    # Create a second visualization to compare the shapes directly
    plt.figure(figsize=(14, 10))
    
    # Create a grid of subplots for detailed comparisons
    plt.subplot(2, 2, 1)
    
    # Plot universe expansion
    plt.plot(universe_time/13.8, universe_scale, 'k-', linewidth=3, 
             label='Universe Scale Factor')
    
    # Plot a specific black hole for direct comparison
    for name, initial_mass, formation_time, growth_model in black_holes:
        if name == "Sagittarius A*":
            # Get black hole growth data
            bh_time, bh_mass = simulate_blackhole_growth(initial_mass, formation_time, growth_model)
            
            # Shift time to start at 0 (formation time becomes 0)
            shifted_time = bh_time - formation_time
            
            # Normalize time to total growth time
            max_time = shifted_time[-1]
            normalized_time = shifted_time / max_time
            
            # Normalize mass to maximum mass
            normalized_mass = bh_mass / bh_mass[-1]
            
            # Plot the normalized growth curve
            plt.plot(normalized_time, normalized_mass, 'r-', linewidth=3, 
                    label=f"{name} (normalized time & mass)")
            
            # Calculate correlation on normalized time axis
            # Create common time points for correlation
            common_time = np.linspace(0, 1, 1000)
            
            # Interpolate both curves to common time
            univ_interp = interp1d(universe_time/13.8, universe_scale, 
                                bounds_error=False, fill_value=(universe_scale[0], universe_scale[-1]))
            bh_interp = interp1d(normalized_time, normalized_mass,
                               bounds_error=False, fill_value=(0, normalized_mass[-1]))
            
            univ_common = univ_interp(common_time)
            bh_common = bh_interp(common_time)
            
            # Calculate correlation
            shape_correlation = np.corrcoef(univ_common, bh_common)[0, 1]
            
            # Add correlation text
            plt.text(0.05, 0.15, f"Shape correlation: {shape_correlation:.3f}",
                   transform=plt.gca().transAxes, fontsize=10,
                   bbox=dict(facecolor='white', alpha=0.8))
    
    plt.xlabel('Normalized Time', fontsize=12)
    plt.ylabel('Normalized Scale/Mass', fontsize=12)
    plt.title('Normalized Growth Patterns', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='lower right')
    
    # Add a subplot with all black holes using normalized time
    plt.subplot(2, 2, 2)
    
    # Plot universe expansion as reference
    plt.plot(universe_time/13.8, universe_scale, 'k-', linewidth=3, 
             label='Universe Scale Factor')
    
    # Plot each black hole with normalized time
    for i, (name, initial_mass, formation_time, growth_model) in enumerate(black_holes):
        if name in ["Sagittarius A*", "M87*", "TON 618", "Primordial BH"]:
            # Get black hole growth data
            bh_time, bh_mass = simulate_blackhole_growth(initial_mass, formation_time, growth_model)
            
            # Shift time to start at 0 (formation time becomes 0)
            shifted_time = bh_time - formation_time
            
            # Normalize time to total growth time
            max_time = shifted_time[-1]
            normalized_time = shifted_time / max_time
            
            # Normalize mass to maximum mass
            normalized_mass = bh_mass / bh_mass[-1]
            
            # Plot the normalized growth curve
            plt.plot(normalized_time, normalized_mass, '-', color=colors[i+1], linewidth=2, 
                    label=f"{name} (norm. time)")
    
    plt.xlabel('Normalized Time', fontsize=12)
    plt.ylabel('Normalized Scale/Mass', fontsize=12)
    plt.title('All Growth Patterns (Normalized Time)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='lower right')
    
    # Add a subplot focusing on the rate of change
    plt.subplot(2, 2, 3)
    
    # Calculate rate of change for universe expansion
    universe_rate = np.gradient(universe_scale, universe_time/13.8)
    plt.plot(universe_time/13.8, universe_rate, 'k-', linewidth=3, 
             label='Universe Expansion Rate')
    
    # Calculate rate of change for a specific black hole
    for name, initial_mass, formation_time, growth_model in black_holes:
        if name == "Sagittarius A*":
            # Get black hole growth data
            bh_time, bh_mass = simulate_blackhole_growth(initial_mass, formation_time, growth_model)
            
            # Shift time to start at 0 (formation time becomes 0)
            shifted_time = bh_time - formation_time
            
            # Normalize time to total growth time
            max_time = shifted_time[-1]
            normalized_time = shifted_time / max_time
            
            # Normalize mass to maximum mass
            normalized_mass = bh_mass / bh_mass[-1]
            
            # Calculate rate of change
            bh_rate = np.gradient(normalized_mass, normalized_time)
            
            # Plot the rate of change
            plt.plot(normalized_time, bh_rate, 'r-', linewidth=3, 
                    label=f"{name} Growth Rate")
    
    plt.xlabel('Normalized Time', fontsize=12)
    plt.ylabel('Rate of Change', fontsize=12)
    plt.title('Growth Rate Comparison', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right')
    
    # Add a subplot showing all systems on a single absolute time axis
    plt.subplot(2, 2, 4)
    
    # Plot universe expansion
    plt.plot(universe_time, universe_scale, 'k-', linewidth=3, 
             label='Universe Scale Factor')
    
    # Calculate total mass of all black holes as function of cosmic time
    # Create a time array spanning the entire cosmic history
    cosmic_time = np.linspace(0.001, 13.8, 1000)
    
    # Initialize array to store combined black hole mass
    combined_mass = np.zeros_like(cosmic_time)
    
    # For each black hole, add its mass to the combined mass at each time point
    for name, initial_mass, formation_time, growth_model in black_holes:
        # Get black hole growth data
        bh_time, bh_mass = simulate_blackhole_growth(initial_mass, formation_time, growth_model)
        
        # Create an interpolation function for the black hole mass
        mass_interp = interp1d(bh_time, bh_mass, bounds_error=False, fill_value=0)
        
        # Add the mass contribution of this black hole to the combined mass
        combined_mass += mass_interp(cosmic_time)
    
    # Normalize the combined mass
    normalized_combined_mass = combined_mass / combined_mass[-1]
    
    # Plot the combined black hole mass
    plt.plot(cosmic_time, normalized_combined_mass, 'g-', linewidth=3, 
             label='Combined BH Mass (normalized)')
    
    # Calculate correlation between combined mass and universe scale
    corr = np.corrcoef(universe_scale, normalized_combined_mass)[0, 1]
    
    # Add correlation text
    plt.text(0.05, 0.15, f"Combined mass correlation: {corr:.3f}",
           transform=plt.gca().transAxes, fontsize=10,
           bbox=dict(facecolor='white', alpha=0.8))
    
    plt.xlabel('Cosmic Time (billion years)', fontsize=12)
    plt.ylabel('Normalized Scale/Mass', fontsize=12)
    plt.title('Combined Black Hole Mass Growth', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='lower right')
    
    plt.tight_layout()
    plt.savefig('detailed_growth_comparisons.png', dpi=300)
    print("Detailed growth comparisons saved as 'detailed_growth_comparisons.png'")
    
    plt.show()

def main():
    print("Aligned Growth Comparison: Universe and Black Holes")
    print("=================================================")
    
    # Create aligned growth visualization
    aligned_growth_comparison()
    
    print("\nComparison complete.")

if __name__ == "__main__":
    main() 