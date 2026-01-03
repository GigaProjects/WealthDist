import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.ticker import FuncFormatter
from tax_simulation import Simulation

# Set style
sns.set_theme(style="whitegrid")

def format_currency(value, pos=None):
    """Format large numbers as $50k, $1.2M, etc."""
    if value >= 1_000_000_000:
        return f"${value/1_000_000_000:.1f}B"
    elif value >= 1_000_000:
        return f"${value/1_000_000:.1f}M"
    elif value >= 1_000:
        return f"${value/1_000:.0f}k"
    else:
        return f"${value:.0f}"

def plot_simulation_results(sim: Simulation):
    """
    Plots the results of the simulation:
    1. Histogram/KDE of Pre vs Post Tax vs Redistributed
    2. Lorenz Curve comparison
    """
    if sim.pre_tax_income is None:
        print("Simulation not run yet!")
        return

    # Toggle this to enable/disable the right graph
    SHOW_TAX_RATE_CHART = False
    
    if SHOW_TAX_RATE_CHART:
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        ax = axes[0]
    else:
        fig, ax = plt.subplots(figsize=(12, 7))
        axes = None
    
    stats = sim.get_stats()
    
    # Calculate Total Percentage Taken
    total_income = np.sum(sim.pre_tax_income)
    total_tax = np.sum(sim.taxes)
    tax_pct = (total_tax / total_income) * 100 if total_income > 0 else 0
    
    # ========== LEFT PLOT: Income Distribution ==========
    
    # Decide on Scale (Log for Pareto/Extreme, Linear for Normal/LogNormal)
    p1, p99 = np.percentile(sim.pre_tax_income, [1, 99.5])
    use_log = (p99 / np.maximum(1, p1)) > 50
    
    # Pre-calculate data
    plot_pre = sim.pre_tax_income if not use_log else sim.pre_tax_income[sim.pre_tax_income > 1]
    plot_post = sim.post_tax_income if not use_log else sim.post_tax_income[sim.post_tax_income > 1]
    plot_redist = sim.redistributed_income if not use_log else sim.redistributed_income[sim.redistributed_income > 1]

    # Plotting
    sns.kdeplot(plot_pre, ax=ax, color="blue", fill=True, alpha=0.1, 
                bw_adjust=1.2, log_scale=use_log, gridsize=500,
                label=f"Pre-Tax (Gini: {stats['pre_gini']:.2f})")
    
    sns.kdeplot(plot_post, ax=ax, color="green", fill=True, alpha=0.2, 
                bw_adjust=1.2, log_scale=use_log, gridsize=500,
                label=f"Post-Tax (Gini: {stats['post_gini']:.2f})")

    sns.kdeplot(plot_redist, ax=ax, color="orange", fill=True, alpha=0.3, 
                bw_adjust=1.2, log_scale=use_log, gridsize=500,
                label=f"With UBI (Gini: {stats['redist_gini']:.2f})")
    
    scale_name = "Log Scale" if use_log else "Linear Scale"
    ax.set_title(f"Income Distribution ({scale_name})\nCollected from income tax: {tax_pct:.1f}% of all income")
    ax.set_xlabel("Income")
    ax.set_ylabel("Density")
    ax.xaxis.set_major_formatter(FuncFormatter(format_currency))
    
    if use_log:
        ax.set_xlim(np.maximum(1, p1 * 0.5), p99 * 2)
    else:
        # For linear scale, start at 0 and focus on 95th percentile
        p95 = np.percentile(sim.pre_tax_income, 95)
        ax.set_xlim(0, p95 * 1.3)

    ax.legend(loc='upper right')
    
    # ========== RIGHT PLOT: Effective Tax Rate (disabled by SHOW_TAX_RATE_CHART flag) ==========
    if SHOW_TAX_RATE_CHART and axes is not None:
        ax2 = axes[1]
        
        # Calculate theoretical effective tax rate curve (not from simulation, but from the tax formula)
        # This gives us an exact, clean line
        max_income = p99 * 1.5
        income_range = np.linspace(1, max_income, 1000)
        theoretical_taxes = sim.tax_system.calculate_tax(income_range)
        theoretical_rate = (theoretical_taxes / income_range) * 100
        
        # Plot the main curve
        ax2.plot(income_range, theoretical_rate, color='darkred', linewidth=2.5, label='Effective Tax Rate')
        
        # Add bracket indicators if it's a progressive tax
        from tax_simulation import ProgressiveTax, FlatTax
        
        if isinstance(sim.tax_system, ProgressiveTax):
            brackets = sim.tax_system.brackets
            colors = plt.cm.Greens(np.linspace(0.4, 0.9, len(brackets)))
            
            for i, bracket in enumerate(brackets):
                threshold = bracket.threshold
                
                if threshold > 0 and threshold < max_income:
                    # Calculate the ACTUAL effective rate at this income level
                    tax_at_threshold = sim.tax_system.calculate_tax(np.array([float(threshold)]))[0]
                    actual_effective_rate = (tax_at_threshold / threshold) * 100
                    
                    # Vertical line at bracket threshold
                    ax2.axvline(x=threshold, color='gray', linestyle=':', alpha=0.4, linewidth=1)
                    
                    # Horizontal line from curve to y-axis
                    ax2.hlines(y=actual_effective_rate, xmin=0, xmax=threshold, 
                              color=colors[i], linestyle='--', alpha=0.8, linewidth=1.5)
                    
                    # Small dot on the curve
                    ax2.scatter([threshold], [actual_effective_rate], color=colors[i], s=40, zorder=5)
                    
                    # Label on the y-axis side
                    ax2.annotate(f'{actual_effective_rate:.1f}%', 
                               xy=(0, actual_effective_rate), 
                               xytext=(-5, 0), textcoords='offset points',
                               fontsize=8, color=colors[i], ha='right', va='center',
                               fontweight='bold')
        
        elif isinstance(sim.tax_system, FlatTax):
            flat_rate = sim.tax_system.rate * 100
            deduction = sim.tax_system.deduction
            
            # Show deduction threshold
            if deduction > 0:
                ax2.axvline(x=deduction, color='blue', linestyle='--', alpha=0.7, linewidth=1)
                ax2.annotate(f'Deduction\n${deduction:,.0f}', xy=(deduction, flat_rate/2), 
                            fontsize=8, color='blue', ha='left')
            
            # Horizontal line at the flat rate
            ax2.axhline(y=flat_rate, color='gray', linestyle=':', alpha=0.5, linewidth=1)
            ax2.annotate(f'Cap: {flat_rate:.0f}%', xy=(max_income * 0.8, flat_rate + 1), 
                        fontsize=9, color='gray')
        
        ax2.set_title("Effective Tax Rate by Income")
        ax2.set_xlabel("Pre-Tax Income")
        ax2.set_ylabel("Effective Tax Rate (%)")
        ax2.xaxis.set_major_formatter(FuncFormatter(format_currency))
        
        # Set Y limits based on max effective rate
        max_rate = np.max(theoretical_rate)
        ax2.set_ylim(0, min(100, max_rate * 1.15))
        
        # X limits
        if use_log:
            ax2.set_xscale('log')
            ax2.set_xlim(np.maximum(1, p1 * 0.5), p99 * 2)
        else:
            ax2.set_xlim(0, max_income)
        
        ax2.legend(loc='lower right')
        ax2.grid(True, alpha=0.3)
    
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

