import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from astroquery.simbad import Simbad
from astropy.io import fits
import requests
from io import BytesIO
import os

def download_bh_catalog():
    """Download black hole data from an online catalog if available"""
    # Check if data file already exists
    if os.path.exists('blackhole_data.csv'):
        print("Using existing black hole data file...")
        df = pd.read_csv('blackhole_data.csv')
        # Convert to cosmic time (Big Bang at 0)
        if 'cosmic_time_byr' not in df.columns:
            # Calculate cosmic time (time since Big Bang)
            # In this model: cosmic_time = 13.8 - age_estimate
            df['cosmic_time_byr'] = 13.8 - df['age_estimate_byr']
            df.to_csv('blackhole_data.csv', index=False)
        return df
    
    # If we need to generate data, we'll use a combination of real data and simulated data
    print("Fetching black hole data...")
    
    # Start with some real black hole mass estimates 
    # Data based on known measurements from scientific literature
    bh_data = {
        'name': [
            'Sagittarius A*', 'M87*', 'TON 618', 'IC 1101', 'NGC 4889',
            'S5 0014+81', 'SDSS J010013.02+280225.8', 'NGC 1600', 'Holmberg 15A',
            'Cygnus X-1', 'V404 Cygni', 'LB-1', 'GRO J1655-40', 'XTE J1118+480',
            'GRO J0422+32', 'GRS 1915+105', 'A0620-00', 'Swift J1753.5-0127'
        ],
        'mass_solar': [
            4.3e6, 6.5e9, 6.6e10, 4.0e10, 2.1e10,
            4.0e10, 3.3e10, 1.7e10, 4.0e10,
            15, 9, 70, 7, 8,
            4, 12, 6.6, 9
        ],
        'type': [
            'SMBH', 'SMBH', 'SMBH', 'SMBH', 'SMBH',
            'SMBH', 'SMBH', 'SMBH', 'SMBH',
            'Stellar', 'Stellar', 'Stellar', 'Stellar', 'Stellar',
            'Stellar', 'Stellar', 'Stellar', 'Stellar'
        ],
        'distance_kpc': [
            8.2, 16800, 3100000, 330000, 94000,
            3500000, 3400000, 64000, 220000,
            2.2, 2.4, 2.3, 3.2, 1.7,
            2.5, 8.6, 1.1, 6
        ]
    }
    
    # Create DataFrame
    df = pd.DataFrame(bh_data)
    
    # Estimate age based on cosmological principles and distance
    # More distant objects are generally older (due to light travel time)
    # For SMBHs, also consider their mass - larger masses suggest older black holes
    # This is a simplified model for demonstration purposes
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Calculate estimated age in billion years
    df['age_estimate_byr'] = np.zeros(len(df))
    
    # For SMBHs: estimate age based on both distance and mass
    smbh_mask = df['type'] == 'SMBH'
    # Convert distance to estimated cosmological age (simplified model)
    df.loc[smbh_mask, 'age_estimate_byr'] = df.loc[smbh_mask, 'distance_kpc'] / 30000000 * 13.8
    # Add some noise and constraints
    df.loc[smbh_mask, 'age_estimate_byr'] = np.minimum(df.loc[smbh_mask, 'age_estimate_byr'] + 
                                                      np.random.normal(0, 0.5, sum(smbh_mask)), 13.7)
    
    # For stellar black holes: typically much younger (formed from stars in our galaxy)
    stellar_mask = df['type'] == 'Stellar'
    df.loc[stellar_mask, 'age_estimate_byr'] = np.random.uniform(0.01, 0.2, sum(stellar_mask))
    
    # Add more synthetic black holes to increase the dataset size
    # These are simulated values based on theoretical distributions
    n_synthetic = 200
    
    # Names for synthetic black holes
    synthetic_names = [f"Synthetic BH-{i}" for i in range(1, n_synthetic + 1)]
    
    # Random selection of types (80% SMBHs, 20% Stellar)
    synthetic_types = np.random.choice(['SMBH', 'Stellar'], n_synthetic, p=[0.8, 0.2])
    
    # Mass distribution depends on type
    synthetic_masses = np.zeros(n_synthetic)
    
    # For SMBHs: log-normal distribution centered around 10^9 solar masses
    smbh_indices = np.where(synthetic_types == 'SMBH')[0]
    synthetic_masses[smbh_indices] = 10**(np.random.normal(9, 1.5, len(smbh_indices)))
    
    # For stellar black holes: typically between 5-100 solar masses
    stellar_indices = np.where(synthetic_types == 'Stellar')[0]
    synthetic_masses[stellar_indices] = np.random.lognormal(2, 0.7, len(stellar_indices))
    
    # Distances (in kpc) - further for SMBHs
    synthetic_distances = np.zeros(n_synthetic)
    synthetic_distances[smbh_indices] = np.random.lognormal(12, 2, len(smbh_indices)) * 1000  # Convert to kpc
    synthetic_distances[stellar_indices] = np.random.lognormal(1, 1, len(stellar_indices)) * 5  # Convert to kpc
    
    # Age estimates
    synthetic_ages = np.zeros(n_synthetic)
    
    # For SMBHs: older ages, correlated with distance and mass
    synthetic_ages[smbh_indices] = (synthetic_distances[smbh_indices] / 30000000 * 13.8 + 
                                   np.log10(synthetic_masses[smbh_indices]) / 12 * 13.7 +
                                   np.random.normal(0, 1, len(smbh_indices)))
    # Enforce age constraints (can't be older than universe)
    synthetic_ages[smbh_indices] = np.minimum(np.maximum(synthetic_ages[smbh_indices], 0.1), 13.7)
    
    # For stellar black holes: much younger
    synthetic_ages[stellar_indices] = np.random.beta(2, 5, len(stellar_indices)) * 0.5
    
    # Create synthetic DataFrame
    synthetic_df = pd.DataFrame({
        'name': synthetic_names,
        'mass_solar': synthetic_masses,
        'type': synthetic_types,
        'distance_kpc': synthetic_distances,
        'age_estimate_byr': synthetic_ages
    })
    
    # Combine real and synthetic data
    combined_df = pd.concat([df, synthetic_df], ignore_index=True)
    
    # Calculate cosmic time (time since Big Bang)
    # In this model: cosmic_time = 13.8 - age_estimate
    combined_df['cosmic_time_byr'] = 13.8 - combined_df['age_estimate_byr']
    
    # Save data to CSV
    combined_df.to_csv('blackhole_data.csv', index=False)
    print(f"Black hole dataset created with {len(combined_df)} entries")
    
    return combined_df

def analyze_bh_data(df):
    """Analyze black hole data for mass vs age relationships"""
    
    print("\nBlack Hole Data Summary:")
    print(f"Total black holes: {len(df)}")
    print(f"SMBHs: {sum(df['type'] == 'SMBH')}")
    print(f"Stellar black holes: {sum(df['type'] == 'Stellar')}")
    
    print("\nMass statistics (solar masses):")
    print(f"Min: {df['mass_solar'].min():.2f}")
    print(f"Max: {df['mass_solar'].max():.2e}")
    print(f"Mean: {df['mass_solar'].mean():.2e}")
    print(f"Median: {df['mass_solar'].median():.2e}")
    
    print("\nCosmic time statistics (billion years since Big Bang):")
    print(f"Min: {df['cosmic_time_byr'].min():.2f}")
    print(f"Max: {df['cosmic_time_byr'].max():.2f}")
    print(f"Mean: {df['cosmic_time_byr'].mean():.2f}")
    
    # Calculate correlation
    correlation = df['mass_solar'].corr(df['cosmic_time_byr'])
    print(f"\nCorrelation between mass and cosmic time: {correlation:.3f}")
    
    # For SMBHs only
    smbh_df = df[df['type'] == 'SMBH']
    smbh_corr = smbh_df['mass_solar'].corr(smbh_df['cosmic_time_byr'])
    print(f"Correlation for SMBHs only: {smbh_corr:.3f}")
    
    # For stellar black holes only
    stellar_df = df[df['type'] == 'Stellar']
    stellar_corr = stellar_df['mass_solar'].corr(stellar_df['cosmic_time_byr'])
    print(f"Correlation for stellar black holes only: {stellar_corr:.3f}")
    
    return {
        'correlation': correlation,
        'smbh_corr': smbh_corr,
        'stellar_corr': stellar_corr
    }

def visualize_bh_data(df):
    """Create visualizations for black hole mass vs age"""
    
    # Set up the figure with a larger size
    plt.figure(figsize=(14, 10))
    
    # Create a scatter plot with logarithmic scale for mass
    plt.subplot(2, 2, 1)
    plt.scatter(df['cosmic_time_byr'], df['mass_solar'], 
                c=df['type'].map({'SMBH': 'red', 'Stellar': 'blue'}),
                alpha=0.6, s=50)
    plt.yscale('log')
    plt.xlabel('Cosmic Time (billion years since Big Bang)')
    plt.ylabel('Black Hole Mass (solar masses)')
    plt.title('Black Hole Mass vs. Cosmic Time')
    plt.colorbar(plt.cm.ScalarMappable(cmap='coolwarm'), 
                 label='Type', ticks=[0.25, 0.75], 
                 boundaries=[0, 0.5, 1])
    plt.clim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Add annotations for some notable black holes
    for i, row in df[df['name'].isin(['Sagittarius A*', 'M87*', 'TON 618', 'Cygnus X-1'])].iterrows():
        plt.annotate(row['name'], 
                    (row['cosmic_time_byr'], row['mass_solar']),
                    xytext=(10, 10),
                    textcoords='offset points',
                    arrowprops=dict(arrowstyle='->', color='black', alpha=0.7))
    
    # Add a subplot with separate trends for SMBHs and stellar black holes
    plt.subplot(2, 2, 2)
    smbh_df = df[df['type'] == 'SMBH']
    stellar_df = df[df['type'] == 'Stellar']
    
    plt.scatter(smbh_df['cosmic_time_byr'], smbh_df['mass_solar'], 
                color='red', alpha=0.6, label='Supermassive BH')
    plt.scatter(stellar_df['cosmic_time_byr'], stellar_df['mass_solar'], 
                color='blue', alpha=0.6, label='Stellar BH')
    
    # Add trend lines
    from scipy import stats
    
    # For SMBHs
    if len(smbh_df) > 1:
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            smbh_df['cosmic_time_byr'], np.log10(smbh_df['mass_solar']))
        x_range = np.linspace(smbh_df['cosmic_time_byr'].min(), smbh_df['cosmic_time_byr'].max(), 100)
        plt.plot(x_range, 10**(slope * x_range + intercept), 'r--', 
                 label=f'SMBH trend (r={r_value:.2f})')
    
    # For stellar black holes
    if len(stellar_df) > 1:
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            stellar_df['cosmic_time_byr'], np.log10(stellar_df['mass_solar']))
        x_range = np.linspace(stellar_df['cosmic_time_byr'].min(), stellar_df['cosmic_time_byr'].max(), 100)
        plt.plot(x_range, 10**(slope * x_range + intercept), 'b--', 
                 label=f'Stellar BH trend (r={r_value:.2f})')
    
    plt.yscale('log')
    plt.xlabel('Cosmic Time (billion years since Big Bang)')
    plt.ylabel('Black Hole Mass (solar masses)')
    plt.title('Black Hole Mass vs. Cosmic Time by Type')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add a histogram of masses
    plt.subplot(2, 2, 3)
    plt.hist([np.log10(stellar_df['mass_solar']), np.log10(smbh_df['mass_solar'])], 
             bins=20, color=['blue', 'red'], alpha=0.7, 
             label=['Stellar BH', 'Supermassive BH'])
    plt.xlabel('Log10(Black Hole Mass) [solar masses]')
    plt.ylabel('Count')
    plt.title('Distribution of Black Hole Masses')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add a histogram of cosmic times
    plt.subplot(2, 2, 4)
    plt.hist([stellar_df['cosmic_time_byr'], smbh_df['cosmic_time_byr']], 
             bins=20, color=['blue', 'red'], alpha=0.7, 
             label=['Stellar BH', 'Supermassive BH'])
    plt.xlabel('Cosmic Time (billion years since Big Bang)')
    plt.ylabel('Count')
    plt.title('Distribution of Black Holes Through Cosmic Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('blackhole_mass_vs_cosmic_time.png', dpi=300)
    print("Visualization saved as 'blackhole_mass_vs_cosmic_time.png'")
    plt.show()

def black_hole_cosmology_analysis(df):
    """Perform specific analysis relevant to black hole cosmology theories"""
    print("\n=== Black Hole Cosmology Analysis ===")
    
    # Calculate mass ratios and distributions that might be relevant to cosmological models
    total_bh_mass = df['mass_solar'].sum()
    print(f"Total observed black hole mass (solar masses): {total_bh_mass:.2e}")
    
    # Estimate mass density
    # Very rough approximation of observable universe volume in cubic megaparsecs
    # This is a vast oversimplification for illustration purposes
    approx_volume = 4/3 * np.pi * (13.8e9 * 306000)**3  # r = age * speed of light
    mass_density = total_bh_mass * 2e30 / approx_volume  # Convert solar masses to kg
    print(f"Estimated black hole mass density: {mass_density:.2e} kg/m³")
    
    # Look for patterns in mass distribution that might support specific models
    # Some black hole cosmology models suggest specific ratios or relationships
    
    # Mass distribution by cosmic time brackets
    cosmic_time_brackets = [0, 3, 7, 11, 13.8]
    bracket_labels = [f"{cosmic_time_brackets[i]}-{cosmic_time_brackets[i+1]} Gyr" for i in range(len(cosmic_time_brackets)-1)]
    
    bracket_masses = []
    for i in range(len(cosmic_time_brackets)-1):
        mask = (df['cosmic_time_byr'] >= cosmic_time_brackets[i]) & (df['cosmic_time_byr'] < cosmic_time_brackets[i+1])
        bracket_masses.append(df.loc[mask, 'mass_solar'].sum())
    
    print("\nMass distribution by cosmic time bracket (since Big Bang):")
    for label, mass in zip(bracket_labels, bracket_masses):
        print(f"  {label}: {mass:.2e} solar masses")
    
    # Calculate mass doubling rate (if universe is expanding, how quickly does BH mass accumulate)
    # This is a very simplified calculation
    total_mass_first_half = df[df['cosmic_time_byr'] <= 6.9]['mass_solar'].sum()
    total_mass_second_half = df[df['cosmic_time_byr'] > 6.9]['mass_solar'].sum()
    
    print(f"\nMass in first half of universe history (0-6.9 Gyr): {total_mass_first_half:.2e} solar masses")
    print(f"Mass in second half of universe history (6.9-13.8 Gyr): {total_mass_second_half:.2e} solar masses")
    print(f"Ratio (second/first): {total_mass_second_half/total_mass_first_half:.2f}")
    
    plt.figure(figsize=(10, 6))
    plt.bar(bracket_labels, bracket_masses)
    plt.xlabel('Cosmic Time (billion years since Big Bang)')
    plt.ylabel('Total Black Hole Mass (solar masses)')
    plt.title('Black Hole Mass Distribution Through Cosmic Time')
    plt.yscale('log')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('blackhole_cosmology_time_distribution.png', dpi=300)
    print("Cosmic time distribution visualization saved as 'blackhole_cosmology_time_distribution.png'")
    
    # Create an additional plot showing mass growth through cosmic time
    plt.figure(figsize=(12, 7))
    
    # Sort data by cosmic time
    df_sorted = df.sort_values('cosmic_time_byr')
    
    # Calculate cumulative mass through cosmic time
    df_sorted['cumulative_mass'] = df_sorted['mass_solar'].cumsum()
    
    # Plot cumulative mass
    plt.plot(df_sorted['cosmic_time_byr'], df_sorted['cumulative_mass'], 'g-', linewidth=2)
    
    # Add reference lines
    plt.axvline(x=6.9, color='gray', linestyle='--', alpha=0.7, label='Universe Half-age (6.9 Gyr)')
    
    plt.xlabel('Cosmic Time (billion years since Big Bang)')
    plt.ylabel('Cumulative Black Hole Mass (solar masses)')
    plt.title('Black Hole Mass Accumulation Through Cosmic Time')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('blackhole_mass_accumulation.png', dpi=300)
    print("Mass accumulation visualization saved as 'blackhole_mass_accumulation.png'")
    
    return {
        'total_mass': total_bh_mass,
        'mass_density': mass_density,
        'time_distribution': dict(zip(bracket_labels, bracket_masses)),
        'mass_ratio': total_mass_second_half/total_mass_first_half
    }

def main():
    print("Black Hole Mass vs. Cosmic Time Analysis")
    print("=======================================")
    
    # Download or load black hole data
    df = download_bh_catalog()
    
    # Analyze the data
    results = analyze_bh_data(df)
    
    # Visualize the data
    visualize_bh_data(df)
    
    # Perform black hole cosmology analysis
    cosmology_results = black_hole_cosmology_analysis(df)
    
    print("\nAnalysis complete.")
    print("Check the generated visualization files for detailed views of the data.")

if __name__ == "__main__":
    main() 