import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tax_simulation import Simulation

# Set style
sns.set_theme(style="whitegrid")

def plot_simulation_results(sim: Simulation):
    """
    Plots the results of the simulation:
    1. Histogram/KDE of Pre vs Post Tax vs Redistributed
    2. Lorenz Curve comparison
    """
    if sim.pre_tax_income is None:
        print("Simulation not run yet!")
        return

    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    stats = sim.get_stats()
    
    # 1. Distribution Plot
    max_val = np.percentile(sim.pre_tax_income, 98) 
    
    # Pre-tax
    sns.histplot(sim.pre_tax_income, ax=axes[0], color="blue", alpha=0.2, 
                 label=f"Pre-Tax (Gini: {stats['pre_gini']:.2f})", 
                 kde=True, element="step", stat="density", common_norm=False)
    
    # Post-tax
    sns.histplot(sim.post_tax_income, ax=axes[0], color="green", alpha=0.3, 
                 label=f"Post-Tax (Gini: {stats['post_gini']:.2f})", 
                 kde=True, element="step", stat="density", common_norm=False)

    # Redistributed
    sns.histplot(sim.redistributed_income, ax=axes[0], color="orange", alpha=0.4, 
                 label=f"With UBI (Gini: {stats['redist_gini']:.2f})", 
                 kde=True, element="step", stat="density", common_norm=False)
    
    axes[0].set_title("Income Distribution (The 'Who Gets What' View)")
    axes[0].set_xlabel("Income (USD)")
    axes[0].set_ylabel("Density (Population %)")
    axes[0].set_xlim(0, max_val)
    axes[0].legend()
    
    # 2. Lorenz Curve
    lorenz_curve(sim.pre_tax_income, ax=axes[1], label="Pre-Tax", color="blue")
    lorenz_curve(sim.post_tax_income, ax=axes[1], label="Post-Tax", color="green")
    lorenz_curve(sim.redistributed_income, ax=axes[1], label="With UBI (Standardized)", color="orange")
    
    # Line of equality
    axes[1].plot([0, 1], [0, 1], color='black', linestyle='--', label="Perfect Equality")
    
    axes[1].set_title("Inequality (Lorenz Curve)")
    axes[1].legend()
    
    plt.tight_layout()
    plt.show()

def lorenz_curve(data, ax, label, color):
    """Plot Lorenz curve for array data on ax."""
    # Ensure positive
    data = np.maximum(0, data)
    sorted_data = np.sort(data)
    n = len(data)
    
    # Cumulative population %
    x = np.linspace(0, 1, n)
    
    # Cumulative income %
    # cumsum / total sum
    y = np.cumsum(sorted_data) / np.sum(sorted_data)
    # Insert (0,0)
    x = np.insert(x, 0, 0)
    y = np.insert(y, 0, 0)
    
    ax.plot(x, y, label=label, color=color, linewidth=2)
    ax.set_xlabel("Cumulative Share of Population")
    ax.set_ylabel("Cumulative Share of Income")

