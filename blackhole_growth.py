import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d
import os

def simulate_blackhole_growth(initial_mass, formation_time, growth_model="exponential-then-linear", seed_mass=10):
    """
    Simulate the growth of a black hole from formation to present day.
    
    Parameters:
    -----------
    initial_mass : float
        Initial mass of the black hole at formation time (solar masses)
    formation_time : float
        When the black hole formed (billion years since Big Bang)
    growth_model : str
        Model to use for black hole growth ("exponential", "power-law", "exponential-then-linear")
    seed_mass : float
        Seed mass for primordial black holes
        
    Returns:
    --------
    time_points : array
        Time points from formation to present (billions of years)
    mass_points : array
        Mass of black hole at each time point (solar masses)
    """
    # Universe age (in billions of years)
    universe_age = 13.8
    
    # Create time points from formation to present
    time_points = np.linspace(formation_time, universe_age, 1000)
    
    # Time since formation (in billions of years)
    time_since_formation = time_points - formation_time
    
    # Different growth models
    if growth_model == "exponential":
        # Exponential growth model - rapid early growth then slowdown
        # M(t) = M_0 * exp(growth_rate * t)
        # Calibrate growth_rate to reach approximate final mass
        
        # Different rates based on black hole type
        if initial_mass < 1000:  # Stellar black hole
            growth_rate = 0.15  # slower growth
        else:  # Supermassive
            growth_rate = 0.35  # faster growth
            
        mass_points = initial_mass * np.exp(growth_rate * time_since_formation)
        
    elif growth_model == "power-law":
        # Power-law growth: M(t) = M_0 * (1 + t/τ)^α
        # τ: characteristic timescale, α: power-law index
        
        tau = 0.5  # characteristic timescale (Gyr)
        
        if initial_mass < 1000:  # Stellar black hole
            alpha = 1.0  # linear growth
        else:  # Supermassive
            alpha = 1.5  # faster than linear
            
        mass_points = initial_mass * (1 + time_since_formation/tau)**alpha
        
    elif growth_model == "exponential-then-linear":
        # Exponential growth in early universe, then transition to more linear growth
        # This aligns with theories of rapid early accretion followed by regulated growth
        
        # Transition point
        transition_time = formation_time + 2.0  # 2 Gyr after formation
        
        # Early phase (exponential)
        early_mask = time_points <= transition_time
        early_times = time_points[early_mask]
        early_growth_rate = 1.2 if initial_mass > 1000 else 0.6
        early_masses = initial_mass * np.exp(early_growth_rate * (early_times - formation_time))
        
        # Late phase (power law)
        late_mask = time_points > transition_time
        late_times = time_points[late_mask]
        
        # Use the last early mass as the initial mass for late growth
        late_initial_mass = early_masses[-1] if len(early_masses) > 0 else initial_mass
        
        # Power law growth for late phase
        alpha = 0.5 if initial_mass > 1000 else 0.3
        late_masses = late_initial_mass * (1 + (late_times - transition_time))**alpha
        
        # Combine the masses
        mass_points = np.concatenate([early_masses, late_masses]) if len(early_masses) > 0 and len(late_masses) > 0 else (early_masses if len(early_masses) > 0 else late_masses)
    
    elif growth_model == "primordial":
        # Model for primordial black holes formed in the very early universe
        # Start with much smaller mass and grow through accretion
        
        # Use seed mass
        mass_points = seed_mass * np.ones_like(time_points)
        
        # First growth phase: rapid accretion during radiation-dominated era
        early_phase = time_points < 0.5  # First 0.5 Gyr
        accretion_rate = 0.8  # High accretion rate in early universe
        mass_points[early_phase] = seed_mass * np.exp(accretion_rate * time_points[early_phase])
        
        # Second phase: slower growth
        mid_phase = (time_points >= 0.5) & (time_points < 3.0)
        mid_initial = mass_points[early_phase][-1] if np.any(early_phase) else seed_mass
        mass_points[mid_phase] = mid_initial * (1 + 0.5*(time_points[mid_phase] - 0.5))**1.2
        
        # Final phase: very slow growth
        late_phase = time_points >= 3.0
        late_initial = mass_points[mid_phase][-1] if np.any(mid_phase) else mid_initial
        mass_points[late_phase] = late_initial * (1 + 0.1*(time_points[late_phase] - 3.0))**0.7
    
    else:
        raise ValueError(f"Unsupported growth model: {growth_model}")
    
    return time_points, mass_points

def create_example_black_holes():
    """
    Create a set of example black holes with different characteristics
    
    Returns:
    --------
    list of tuples
        Each tuple contains (name, initial_mass, formation_time, growth_model)
    """
    black_holes = [
        # Supermassive black holes
        ("Sagittarius A*", 1e5, 0.8, "exponential-then-linear"),  # Milky Way SMBH
        ("M87*", 5e6, 0.5, "exponential-then-linear"),  # M87 galaxy SMBH
        ("TON 618", 8e6, 0.3, "exponential-then-linear"),  # One of the most massive known BHs
        
        # Stellar black holes
        ("Cygnus X-1", 10, 9.5, "power-law"),  # Famous stellar BH in the Milky Way
        ("GRS 1915+105", 12, 10.0, "power-law"),  # Another well-known stellar BH
        
        # Theoretical models
        ("Primordial BH", 1e-5, 0.001, "primordial"),  # Hypothetical primordial BH
        ("Early Universe SMBH", 1e4, 0.2, "exponential"),  # Early formed SMBH
        ("Merger-grown SMBH", 1e3, 2.0, "power-law"),  # SMBH grown primarily through mergers
    ]
    
    return black_holes

def visualize_blackhole_growth(save_path="blackhole_growth.png"):
    """
    Create visualizations of black hole growth over cosmic time
    """
    # Create example black holes
    black_holes = create_example_black_holes()
    
    # Set up subplots
    fig = plt.figure(figsize=(16, 12))
    
    # Plot 1: All black holes on log scale
    ax1 = fig.add_subplot(2, 2, 1)
    
    # Color map for different types
    colors = plt.cm.viridis(np.linspace(0, 1, len(black_holes)))
    
    # Plot each black hole's growth
    for i, (name, initial_mass, formation_time, growth_model) in enumerate(black_holes):
        time_points, mass_points = simulate_blackhole_growth(initial_mass, formation_time, growth_model)
        ax1.plot(time_points, mass_points, label=name, color=colors[i], linewidth=2)
        
        # Add a dot at formation time
        ax1.scatter(formation_time, initial_mass, color=colors[i], s=50, zorder=10)
    
    ax1.set_yscale('log')
    ax1.set_xlabel('Cosmic Time (billion years since Big Bang)')
    ax1.set_ylabel('Black Hole Mass (solar masses)')
    ax1.set_title('Growth of Different Black Holes Over Cosmic Time')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left')
    
    # Mark key cosmic epochs
    ax1.axvline(x=0.4, color='gray', linestyle='--', alpha=0.7)
    ax1.text(0.42, ax1.get_ylim()[0]*10, 'Reionization', rotation=90, alpha=0.7)
    
    ax1.axvline(x=1.5, color='gray', linestyle='--', alpha=0.7)
    ax1.text(1.52, ax1.get_ylim()[0]*10, 'Peak Star Formation', rotation=90, alpha=0.7)
    
    ax1.axvline(x=6.0, color='gray', linestyle='--', alpha=0.7)
    ax1.text(6.02, ax1.get_ylim()[0]*10, 'Solar System Formation', rotation=90, alpha=0.7)
    
    # Plot 2: Supermassive black holes only
    ax2 = fig.add_subplot(2, 2, 2)
    smbh_colors = plt.cm.plasma(np.linspace(0, 1, 4))
    
    # Plot only the supermassive black holes
    smbh_counter = 0
    for i, (name, initial_mass, formation_time, growth_model) in enumerate(black_holes):
        if initial_mass > 1000:  # Only SMBHs
            time_points, mass_points = simulate_blackhole_growth(initial_mass, formation_time, growth_model)
            ax2.plot(time_points, mass_points, label=name, color=smbh_colors[smbh_counter], linewidth=2)
            smbh_counter += 1
    
    ax2.set_yscale('log')
    ax2.set_xlabel('Cosmic Time (billion years since Big Bang)')
    ax2.set_ylabel('Black Hole Mass (solar masses)')
    ax2.set_title('Growth of Supermassive Black Holes')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper left')
    
    # Plot 3: Growth rates
    ax3 = fig.add_subplot(2, 2, 3)
    
    for i, (name, initial_mass, formation_time, growth_model) in enumerate(black_holes):
        # Only show select black holes for clarity
        if name in ["Sagittarius A*", "Cygnus X-1", "Primordial BH"]:
            time_points, mass_points = simulate_blackhole_growth(initial_mass, formation_time, growth_model)
            
            # Calculate growth rate (fractional mass increase per Gyr)
            # Use finite difference to estimate derivative
            growth_rates = np.gradient(mass_points, time_points)
            growth_rates_fraction = growth_rates / mass_points
            
            ax3.plot(time_points, growth_rates_fraction, label=name, linewidth=2)
    
    ax3.set_yscale('log')
    ax3.set_xlabel('Cosmic Time (billion years since Big Bang)')
    ax3.set_ylabel('Fractional Growth Rate (per Gyr)')
    ax3.set_title('Black Hole Growth Rates Over Time')
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc='upper right')
    
    # Plot 4: Normalized growth
    ax4 = fig.add_subplot(2, 2, 4)
    
    for i, (name, initial_mass, formation_time, growth_model) in enumerate(black_holes):
        time_points, mass_points = simulate_blackhole_growth(initial_mass, formation_time, growth_model)
        
        # Normalize mass to show relative growth
        normalized_mass = mass_points / initial_mass
        
        # Only plot from formation time onward
        ax4.plot(time_points, normalized_mass, label=name, color=colors[i], linewidth=2)
    
    ax4.set_yscale('log')
    ax4.set_xlabel('Cosmic Time (billion years since Big Bang)')
    ax4.set_ylabel('Mass Relative to Initial Mass')
    ax4.set_title('Normalized Black Hole Growth')
    ax4.grid(True, alpha=0.3)
    ax4.legend(loc='upper left')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Black hole growth visualization saved as '{save_path}'")
    
    # Create an additional figure showing just a single black hole in detail
    plt.figure(figsize=(12, 8))
    
    # Select Sagittarius A* for detailed analysis
    for name, initial_mass, formation_time, growth_model in black_holes:
        if name == "Sagittarius A*":
            time_points, mass_points = simulate_blackhole_growth(initial_mass, formation_time, growth_model)
            
            plt.plot(time_points, mass_points, 'b-', linewidth=3)
            
            # Add point at formation
            plt.scatter(formation_time, initial_mass, s=100, color='red', zorder=10, 
                       label=f'Formation: {formation_time:.1f} Gyr, {initial_mass:.1e} M☉')
            
            # Add point at present day
            plt.scatter(time_points[-1], mass_points[-1], s=100, color='green', zorder=10,
                       label=f'Present: 13.8 Gyr, {mass_points[-1]:.1e} M☉')
            
            # Add total growth text
            growth_factor = mass_points[-1] / initial_mass
            plt.text(0.5, 0.95, f"Total Growth Factor: {growth_factor:.1e}x",
                    transform=plt.gca().transAxes, fontsize=12,
                    bbox=dict(facecolor='white', alpha=0.8))
            
            # Add key growth phases
            plt.axvspan(formation_time, formation_time + 2.0, alpha=0.2, color='red',
                       label='Rapid Accretion Phase')
            plt.axvspan(formation_time + 2.0, 13.8, alpha=0.1, color='blue',
                       label='Steady Growth Phase')
            
            break
    
    plt.yscale('log')
    plt.xlabel('Cosmic Time (billion years since Big Bang)')
    plt.ylabel('Black Hole Mass (solar masses)')
    plt.title('Detailed Growth History of Sagittarius A*')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig('sagittarius_a_growth.png', dpi=300)
    print("Detailed Sagittarius A* growth visualization saved as 'sagittarius_a_growth.png'")
    
    plt.show()

def export_growth_data_to_csv():
    """
    Export black hole growth data to CSV files for further analysis
    """
    black_holes = create_example_black_holes()
    
    # Create a directory for the data if it doesn't exist
    os.makedirs("black_hole_data", exist_ok=True)
    
    # Export growth curves for each black hole
    for name, initial_mass, formation_time, growth_model in black_holes:
        time_points, mass_points = simulate_blackhole_growth(initial_mass, formation_time, growth_model)
        
        # Calculate additional metrics
        growth_rates = np.gradient(mass_points, time_points)  # Mass increase per Gyr
        relative_growth = growth_rates / mass_points  # Fractional growth rate
        
        # Create a DataFrame
        df = pd.DataFrame({
            'cosmic_time_gyr': time_points,
            'mass_solar': mass_points,
            'growth_rate_solar_per_gyr': growth_rates,
            'relative_growth_rate_per_gyr': relative_growth
        })
        
        # Export to CSV
        safe_name = name.replace(" ", "_").replace("+", "_plus_").replace("*", "")
        file_path = os.path.join("black_hole_data", f"{safe_name}_growth.csv")
        df.to_csv(file_path, index=False)
        print(f"Exported growth data for {name} to {file_path}")

def main():
    print("Black Hole Growth Simulation")
    print("===========================")
    
    # Visualize black hole growth
    visualize_blackhole_growth()
    
    # Export data to CSV for further analysis
    export_growth_data_to_csv()
    
    print("\nSimulation complete.")

if __name__ == "__main__":
    main() 