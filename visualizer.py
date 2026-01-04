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
    
    # We use a Linear Scale as requested by the user.
    # Focus on 99th percentile to keep the "Long Tail" visible without squashing the main curve.
    p99 = np.percentile(sim.pre_tax_income, 99.5)
    
    # Pre-calculate data
    plot_pre = sim.pre_tax_income
    plot_post = sim.post_tax_income
    plot_redist = sim.redistributed_income

    # Plotting
    sns.kdeplot(plot_pre, ax=ax, color="blue", fill=True, alpha=0.1, 
                bw_adjust=1.2, log_scale=False, gridsize=500,
                label=f"Pre-Tax (Gini: {stats['pre_gini']:.2f})")
    
    # Capture the natural peak height of the pre-tax distribution
    pre_max_y = ax.get_ylim()[1]
    
    sns.kdeplot(plot_post, ax=ax, color="green", fill=True, alpha=0.2, 
                bw_adjust=1.2, log_scale=False, gridsize=500,
                label=f"Post-Tax (Gini: {stats['post_gini']:.2f})")

    sns.kdeplot(plot_redist, ax=ax, color="orange", fill=True, alpha=0.3, 
                bw_adjust=1.2, log_scale=False, gridsize=500,
                label=f"With UBI (Gini: {stats['redist_gini']:.2f})")
    
    # Force y-axis back to a readable range based on the pre-tax curve.
    ax.set_ylim(0, pre_max_y * 1.5)
    
    ax.set_title(f"Income Distribution (Linear Scale)\nCollected from income tax: {tax_pct:.1f}% of all income")
    ax.set_xlabel("Income")
    ax.set_ylabel("Density")
    ax.xaxis.set_major_formatter(FuncFormatter(format_currency))
    
    # Set x-limit to show most of the distribution
    ax.set_xlim(0, p99 * 1.1)

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


##### Interactive Plotting #####

    ax.set_ylabel("Cumulative Share of Income")


##### Interactive Plotting #####

def plot_wealth_history(sim):
    """
    Plot animated wealth distribution over time.
    Single plot with Play/Pause button and detailed stats.
    """
    from matplotlib.widgets import Slider, Button
    import matplotlib.animation as animation
    
    # Setup Figure (Single Plot)
    fig, ax = plt.subplots(figsize=(12, 8))
    # Pre-calculate stats for all years
    years = range(len(sim.history['post']))
    
    # Calculate constant annual stats (for info only)
    total_income = np.sum(sim.annual_income)
    total_annual_tax = np.sum(sim.annual_taxes)
    avg_tax_rate = (total_annual_tax / total_income) * 100
    
    # Determine global limits and scaling (STABLE LINEAR RANGE)
    # We use 'pre' wealth at YEAR 50 to find the max spread
    final_pre = sim.history['pre'][-1]
    
    # Linear limits starting from 0 to 110% of max wealth
    p99 = np.percentile(final_pre, 99.5)
    min_w = 0
    max_w = p99 * 1.1
    
    # State container for animation
    state = {
        'running': False,
        'obj': None
    }
    
    def update_plot(val):
        year_idx = int(slider.val)
        
        # Get data for all 3 scenarios
        w_pre = sim.history['pre'][year_idx]
        w_post = sim.history['post'][year_idx]
        w_ubi = sim.history['ubi'][year_idx]
        
        # Get pre-calculated Ginis
        g_pre = sim.gini_history['pre'][year_idx]
        g_post = sim.gini_history['post'][year_idx]
        g_ubi = sim.gini_history['ubi'][year_idx]
        
        # Clear and Redraw
        ax.clear()
        
        # Plot 3 Overlaid Distributions (Forced Linear)
        sns.kdeplot(w_pre, ax=ax, color="blue", fill=True, alpha=0.1,
                   log_scale=False, gridsize=200, label=f"No Tax Forever (Gini: {g_pre:.3f})")
                   
        sns.kdeplot(w_post, ax=ax, color="green", fill=True, alpha=0.2,
                   log_scale=False, gridsize=200, label=f"Taxed, No UBI (Gini: {g_post:.3f})")
                   
        sns.kdeplot(w_ubi, ax=ax, color="orange", fill=True, alpha=0.3,
                   log_scale=False, gridsize=200, label=f"Taxed + UBI (Gini: {g_ubi:.3f})")
        
        # Enhanced Title
        title_text = (
            f"Wealth Accumulation - Year {year_idx} (Linear Scale)\n"
            f"Scenario Comparison after {year_idx} years of policy | Avg Tax: {avg_tax_rate:.1f}%"
        )
        ax.set_title(title_text, fontsize=14, pad=15)
        ax.set_xlabel("Total Wealth ($)")
        ax.set_ylabel("Density")
        ax.legend(loc='upper right')
        
        ax.set_xlim(min_w, max_w)
        ax.xaxis.set_major_formatter(FuncFormatter(format_currency))
        ax.grid(True, alpha=0.3)
        
        fig.canvas.draw_idle()

    # --- Slider ---
    ax_slider = plt.axes([0.2, 0.1, 0.6, 0.03], facecolor='lightgoldenrodyellow')
    slider = Slider(
        ax=ax_slider,
        label='Year ',
        valmin=0,
        valmax=len(sim.history['pre']) - 1,
        valinit=0,
        valstep=1
    )
    
    slider.on_changed(update_plot)
    
    # --- Play Button Logic ---
    def animate(frame):
        # Move slider forward, loop back at end
        val = slider.val + 1
        if val > slider.valmax:
            val = 0
        slider.set_val(val)
        return val

    def toggle_animation(event):
        if state['running']:
            if state['obj']:
                state['obj'].event_source.stop()
            state['running'] = False
            btn.label.set_text('▶ Play')
        else:
            # interval=50ms (20 frames per second)
            state['obj'] = animation.FuncAnimation(fig, animate, interval=50, save_count=100)
            state['running'] = True
            btn.label.set_text('⏸ Pause')
        fig.canvas.draw_idle()

    # Add Play Button
    ax_btn = plt.axes([0.82, 0.02, 0.1, 0.05])
    btn = Button(ax_btn, '▶ Play')
    btn.on_clicked(toggle_animation)

    # Initial Plot
    update_plot(0)
    plt.show()
    print("\n[Interactive Mode] Use slider or Play button to view wealth evolution.")

