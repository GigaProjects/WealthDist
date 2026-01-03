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

    fig, ax = plt.subplots(figsize=(12, 7))
    stats = sim.get_stats()
    
    # Calculate Total Percentage Taken
    total_income = np.sum(sim.pre_tax_income)
    total_tax = np.sum(sim.taxes)
    tax_pct = (total_tax / total_income) * 100 if total_income > 0 else 0
    
    # 1. Decide on Scale (Log for Pareto/Extreme, Linear for Normal/LogNormal)
    # Use percentiles to avoid being distracted by a single billionaire
    p1, p99 = np.percentile(sim.pre_tax_income, [1, 99.5])
    use_log = (p99 / np.maximum(1, p1)) > 50
    
    # Pre-calculate data
    plot_pre = sim.pre_tax_income if not use_log else sim.pre_tax_income[sim.pre_tax_income > 1]
    plot_post = sim.post_tax_income if not use_log else sim.post_tax_income[sim.post_tax_income > 1]
    plot_redist = sim.redistributed_income if not use_log else sim.redistributed_income[sim.redistributed_income > 1]

    # Plotting
    sns.kdeplot(plot_pre, ax=ax, color="blue", fill=True, alpha=0.1, 
                bw_adjust=1.2, log_scale=use_log, gridsize=1000,
                label=f"Pre-Tax (Gini: {stats['pre_gini']:.2f})")
    
    sns.kdeplot(plot_post, ax=ax, color="green", fill=True, alpha=0.2, 
                bw_adjust=1.2, log_scale=use_log, gridsize=1000,
                label=f"Post-Tax (Gini: {stats['post_gini']:.2f})")

    sns.kdeplot(plot_redist, ax=ax, color="orange", fill=True, alpha=0.3, 
                bw_adjust=1.2, log_scale=use_log, gridsize=1000,
                label=f"With UBI (Gini: {stats['redist_gini']:.2f})")
    
    scale_name = "Log Scale" if use_log else "Linear Scale"
    ax.set_title(f"Income Distribution ({scale_name})\nCollected from income tax: {tax_pct:.1f}% of all income")
    ax.set_xlabel("Income (USD)")
    ax.set_ylabel("Density")
    
    # Smart X-Limits
    if use_log:
        ax.set_xlim(np.maximum(1, p1 * 0.5), p99 * 2)
        from matplotlib.ticker import ScalarFormatter
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.ticklabel_format(style='plain', axis='x')
    else:
        ax.set_xlim(0, p99 * 1.1)

    ax.legend(loc='upper right')
    
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

