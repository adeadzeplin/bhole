import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
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
    
    # We'll use a parametric model that approximates the Lambda-CDM model
    # This includes the transitions between radiation, matter and dark energy dominance
    
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

def compare_universe_and_blackholes(save_path="universe_blackhole_comparison.png"):
    """
    Create a visualization comparing the universe expansion with black hole growth
    """
    # Create cosmic time array from Big Bang to present
    cosmic_time = np.linspace(0.001, 13.8, 1000)  # Gyr since Big Bang
    
    # Calculate universe scale factor
    universe_scale = universe_scale_factor(cosmic_time)
    
    # Normalized radius of the observable universe 
    # (defined as 1 at present, scaled by the scale factor)
    universe_radius = universe_scale
    
    # Get black hole data
    black_holes = create_example_black_holes()
    
    # Set up the figure with GridSpec for custom layout
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 2, height_ratios=[2, 1], width_ratios=[1, 1])
    
    # Main plot: Universe expansion and black hole growth
    ax1 = fig.add_subplot(gs[0, :])
    
    # Plot universe radius (normalized to 1 at present)
    universe_line, = ax1.plot(cosmic_time, universe_radius, 'k-', linewidth=3, 
                          label='Universe Scale Factor')
    
    # Secondary axis for black hole masses
    ax2 = ax1.twinx()
    
    # Color map for different black holes
    colors = plt.cm.plasma(np.linspace(0, 1, len(black_holes)))
    
    # Store lines for legend
    bh_lines = []
    
    # Plot selected black holes - normalize masses for comparison with universe
    for i, (name, initial_mass, formation_time, growth_model) in enumerate(black_holes):
        # Only include representative black holes
        if name in ["Sagittarius A*", "M87*", "TON 618", "Primordial BH"]:
            time_points, mass_points = simulate_blackhole_growth(initial_mass, formation_time, growth_model)
            
            # Normalize mass to final mass for visual comparison
            # This allows us to see the "shape" of growth compared to universe expansion
            norm_mass = mass_points / mass_points[-1]
            
            # Connect from origin to formation time with dotted line
            if formation_time > 0.001:
                # Create connecting line from Big Bang to formation
                connect_time = np.array([0.001, formation_time])
                connect_mass = np.array([0, norm_mass[0]])
                ax2.plot(connect_time, connect_mass, '--', color=colors[i], alpha=0.5, linewidth=1)
            
            # Plot the black hole growth
            bh_line, = ax2.plot(time_points, norm_mass, '-', color=colors[i], 
                            linewidth=2, label=f"{name} (Normalized Growth)")
            bh_lines.append(bh_line)
            
            # Add a marker at formation time
            ax2.scatter(formation_time, norm_mass[0], color=colors[i], s=50, zorder=10)
    
    # Mark key cosmic epochs
    epoch_times = [0.38, 0.47, 4.3, 9.0]  # Gyr
    epoch_labels = ['Reionization', 'End of Dark Ages', 'Formation of Solar System', 'Accelerated Expansion']
    
    for time, label in zip(epoch_times, epoch_labels):
        ax1.axvline(x=time, color='gray', linestyle=':', alpha=0.7)
        ax1.text(time+0.2, 0.1, label, rotation=90, alpha=0.7, transform=ax1.get_xaxis_transform())
    
    # Set up axis labels and title
    ax1.set_xlabel('Cosmic Time (billion years since Big Bang)')
    ax1.set_ylabel('Universe Scale Factor', color='k')
    ax2.set_ylabel('Black Hole Mass (normalized)', color='b')
    
    # Set title
    ax1.set_title('Comparison: Universe Expansion vs. Black Hole Growth', fontsize=14)
    
    # Add combined legend
    lines = [universe_line] + bh_lines
    ax1.legend(lines, [l.get_label() for l in lines], loc='center left')
    
    # Second plot: Growth rates
    ax3 = fig.add_subplot(gs[1, 0])
    
    # Calculate expansion rate of universe (dR/dt)/R
    universe_expansion_rate = np.gradient(universe_radius, cosmic_time) / universe_radius
    ax3.plot(cosmic_time, universe_expansion_rate, 'k-', linewidth=2, 
            label='Universe Expansion Rate')
    
    # Calculate growth rates for selected black holes
    for i, (name, initial_mass, formation_time, growth_model) in enumerate(black_holes):
        if name in ["Sagittarius A*", "Primordial BH"]:
            time_points, mass_points = simulate_blackhole_growth(initial_mass, formation_time, growth_model)
            
            # Calculate relative growth rate (dM/dt)/M
            growth_rate = np.gradient(mass_points, time_points) / mass_points
            
            # Only plot from formation onward
            ax3.plot(time_points, growth_rate, '-', color=colors[i], 
                    linewidth=2, label=f"{name} Growth Rate")
    
    # Set up axis labels
    ax3.set_xlabel('Cosmic Time (billion years since Big Bang)')
    ax3.set_ylabel('Relative Growth Rate (per Gyr)')
    ax3.set_title('Expansion/Growth Rates')
    ax3.set_yscale('log')
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc='upper right')
    
    # Third plot: Log-scaled comparison 
    ax4 = fig.add_subplot(gs[1, 1])
    
    # Plot universe radius on log scale
    ax4.plot(cosmic_time, universe_radius, 'k-', linewidth=2, 
            label='Universe Scale Factor')
    
    # Plot black hole masses on separate axis
    ax5 = ax4.twinx()
    
    # Plot selected black holes with actual masses
    for i, (name, initial_mass, formation_time, growth_model) in enumerate(black_holes):
        if name in ["Sagittarius A*", "M87*", "TON 618"]:
            time_points, mass_points = simulate_blackhole_growth(initial_mass, formation_time, growth_model)
            ax5.plot(time_points, mass_points, '-', color=colors[i], 
                    linewidth=2, label=f"{name} Mass")
    
    # Set up axis labels and scales
    ax4.set_xlabel('Cosmic Time (billion years since Big Bang)')
    ax4.set_ylabel('Universe Scale Factor', color='k')
    ax5.set_ylabel('Black Hole Mass (solar masses)', color='b')
    ax5.set_yscale('log')
    ax4.set_title('Log-Scaled Comparison')
    ax4.grid(True, alpha=0.3)
    
    # Add legend for universe scale
    ax4.legend(loc='upper left')
    # Add legend for black hole masses
    ax5.legend(loc='lower right')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Comparison visualization saved as '{save_path}'")
    
    # Create a more detailed figure focused on the correlation
    plt.figure(figsize=(12, 9))
    
    # Plot normalized values for clearer shape comparison
    plt.plot(cosmic_time, universe_radius, 'k-', linewidth=3, 
            label='Universe Scale Factor')
    
    # Focus on Sagittarius A* for the clearest comparison
    for name, initial_mass, formation_time, growth_model in black_holes:
        if name == "Sagittarius A*":
            time_points, mass_points = simulate_blackhole_growth(initial_mass, formation_time, growth_model)
            
            # Normalize to better compare the growth patterns
            norm_mass = mass_points / mass_points[-1]
            
            # Adjust the curve to align better with universe expansion for visual comparison
            # Start the curve from the beginning of time for better comparison
            combined_time = np.linspace(0.001, 13.8, 1000)
            mass_interp = interp1d(time_points, norm_mass, bounds_error=False, fill_value=(0, norm_mass[-1]))
            extended_mass = mass_interp(combined_time)
            extended_mass[combined_time < formation_time] = 0
            
            # Plot rescaled black hole growth
            plt.plot(combined_time, extended_mass, 'r-', linewidth=3, 
                    label=f"{name} Normalized Growth")
            
            # Mark formation time
            plt.axvline(x=formation_time, color='red', linestyle=':', alpha=0.7)
            plt.text(formation_time+0.2, 0.5, f"{name} Formation", 
                    rotation=90, color='red', alpha=0.7)
            
            # Add correlation text
            corr = np.corrcoef(universe_radius, extended_mass)[0, 1]
            plt.text(0.5, 0.15, f"Correlation coefficient: {corr:.3f}",
                   transform=plt.gca().transAxes, fontsize=12,
                   bbox=dict(facecolor='white', alpha=0.8))
    
    # Mark key cosmic epochs
    for time, label in zip(epoch_times, epoch_labels):
        plt.axvline(x=time, color='gray', linestyle=':', alpha=0.5)
        plt.text(time+0.2, 0.1, label, rotation=90, alpha=0.7)
    
    plt.xlabel('Cosmic Time (billion years since Big Bang)')
    plt.ylabel('Normalized Scale/Mass')
    plt.title('Pattern Comparison: Universe Expansion vs. Black Hole Growth', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig('pattern_correlation.png', dpi=300)
    print("Pattern correlation visualization saved as 'pattern_correlation.png'")
    
    plt.show()

def main():
    print("Universe Expansion vs. Black Hole Growth Comparison")
    print("=================================================")
    
    # Create comparison visualization
    compare_universe_and_blackholes()
    
    print("\nComparison complete.")

if __name__ == "__main__":
    main() 