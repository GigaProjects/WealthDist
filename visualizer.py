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

    # Calculate Mean Wealth
    m_pre = np.mean(plot_pre)
    m_post = np.mean(plot_post)
    m_redist = np.mean(plot_redist)

    # Plotting
    sns.kdeplot(plot_pre, ax=ax, color="blue", fill=True, alpha=0.1, 
                bw_adjust=1.2, log_scale=False, gridsize=500,
                label=f"Pre-Tax (Gini: {stats['pre_gini']:.2f}, Mean: {format_currency(m_pre)})")
    
    # Capture the natural peak height of the pre-tax distribution
    pre_max_y = ax.get_ylim()[1]
    
    sns.kdeplot(plot_post, ax=ax, color="green", fill=True, alpha=0.2, 
                bw_adjust=1.2, log_scale=False, gridsize=500,
                label=f"Post-Tax (Gini: {stats['post_gini']:.2f}, Mean: {format_currency(m_post)})")

    sns.kdeplot(plot_redist, ax=ax, color="orange", fill=True, alpha=0.3, 
                bw_adjust=1.2, log_scale=False, gridsize=500,
                label=f"With UBI (Gini: {stats['redist_gini']:.2f}, Mean: {format_currency(m_redist)})")
    
    # Add Mean Lines for Single Year Plot (Average lines removed per user request)
    # ax.axvline(m_pre, color="darkblue", linestyle="--", alpha=0.9, linewidth=2.5, zorder=10, label=f"Mean (Pre-Tax): {format_currency(m_pre)}")
    # ax.axvline(m_post, color="green", linestyle="--", alpha=0.7, linewidth=1.5, zorder=9, label=f"Mean (Post-Tax): {format_currency(m_post)}")
    # ax.axvline(m_redist, color="orange", linestyle="--", alpha=0.7, linewidth=1.5, zorder=9, label=f"Mean (With UBI): {format_currency(m_redist)}")
    
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

def plot_wealth_history(sim):
    """
    Plot animated wealth distribution over time.
    Dual subplot version (Side-by-Side):
    1. Wealth Distribution (KDE) - Dynamic
    2. Gini Coefficient History - Static with Indicator
    """
    from matplotlib.widgets import Slider, Button
    import matplotlib.animation as animation
    
    # Setup Figure (Side-by-side square-ish plots)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    fig.subplots_adjust(bottom=0.2, wspace=0.25) # Room for slider/button and horizontal spacing
    
    # Pre-calculate constant global stats
    total_income = np.sum(sim.annual_income)
    total_annual_tax = np.sum(sim.annual_taxes)
    avg_tax_rate = (total_annual_tax / total_income) * 100 if total_income > 0 else 0
    
    # Global limits for stable KDE
    final_pre = sim.history['pre'][-1]
    p99 = np.percentile(final_pre, 99.5)
    min_w = 0
    max_w = max(1.0, p99 * 1.1) if not np.isnan(p99) else 1.0
    
    # --- AX2: Static Gini History Setup (Drawn Once) ---
    all_years = range(sim.n_years + 1)
    ax2.plot(all_years, sim.gini_history['pre'], color="blue", alpha=0.9, linewidth=2, label="No Tax")
    ax2.plot(all_years, sim.gini_history['post'], color="green", alpha=0.9, linewidth=2, label="Taxed")
    ax2.plot(all_years, sim.gini_history['ubi'], color="orange", alpha=0.9, linewidth=2, label="UBI")
    
    # Create the movable indicator dots and line for ax2
    curr_line = ax2.axvline(0, color='red', linestyle='--', alpha=0.4, label='Current Year')
    dot_pre, = ax2.plot([0], [sim.gini_history['pre'][0]], 'o', color="blue", markersize=6)
    dot_post, = ax2.plot([0], [sim.gini_history['post'][0]], 'o', color="green", markersize=6)
    dot_ubi, = ax2.plot([0], [sim.gini_history['ubi'][0]], 'o', color="orange", markersize=6)

    ax2.set_title("Inequality Evolution (Static View)", fontsize=13)
    ax2.set_xlabel("Year")
    ax2.set_ylabel("Gini Coefficient")
    ax2.set_xlim(0, sim.n_years)
    ax2.set_ylim(0, 1.0)
    ax2.legend(loc='lower right', fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # State container
    state = {'running': False, 'obj': None}
    
    def update_plot(val):
        year_idx = int(slider.val)
        
        # Get data for current year
        w_pre = sim.history['pre'][year_idx]
        w_post = sim.history['post'][year_idx]
        w_ubi = sim.history['ubi'][year_idx]
        
        g_pre = sim.gini_history['pre'][year_idx]
        g_post = sim.gini_history['post'][year_idx]
        g_ubi = sim.gini_history['ubi'][year_idx]
        
        grow_pre = sim.gdp_growth_history['pre'][year_idx]
        grow_post = sim.gdp_growth_history['post'][year_idx]
        grow_ubi = sim.gdp_growth_history['ubi'][year_idx]
        
        # Calculate Means
        m_pre = np.mean(w_pre)
        m_post = np.mean(w_post)
        m_ubi = np.mean(w_ubi)

        # --- AX1: Dynamic Distribution Plot ---
        ax1.clear()
        sns.kdeplot(w_pre, ax=ax1, color="blue", fill=True, alpha=0.1, label=f"No Tax (Gini: {g_pre:.3f}, Mean: {format_currency(m_pre)})")
        sns.kdeplot(w_post, ax=ax1, color="green", fill=True, alpha=0.2, label=f"Taxed (Gini: {g_post:.3f}, Mean: {format_currency(m_post)})")
        sns.kdeplot(w_ubi, ax=ax1, color="orange", fill=True, alpha=0.3, label=f"UBI (Gini: {g_ubi:.3f}, Mean: {format_currency(m_ubi)})")

        title_text = (
            f"Wealth Accumulation - Year {year_idx}\n"
            f"Annual GDP Growth:  Pre: {grow_pre:+.2f}% | Tax: {grow_post:+.2f}% | UBI: {grow_ubi:+.2f}%\n"
            f"Avg Tax Rate: {avg_tax_rate:.1f}%"
        )
        ax1.set_title(title_text, fontsize=12, pad=10)
        ax1.set_xlabel("Total Wealth ($)")
        ax1.set_ylabel("Density")
        ax1.set_xlim(min_w, max_w)
        ax1.xaxis.set_major_formatter(FuncFormatter(format_currency))
        ax1.legend(loc='upper right', fontsize=9)
        ax1.grid(True, alpha=0.2)

        # --- Update AX2 Indicator ---
        curr_line.set_xdata([year_idx, year_idx])
        dot_pre.set_data([year_idx], [g_pre])
        dot_post.set_data([year_idx], [g_post])
        dot_ubi.set_data([year_idx], [g_ubi])

        fig.canvas.draw_idle()

    # Controls Setup
    ax_slider = plt.axes([0.15, 0.05, 0.55, 0.03], facecolor='lightgoldenrodyellow')
    slider = Slider(ax_slider, 'Year ', 0, sim.n_years, valinit=0, valstep=1)
    slider.on_changed(update_plot)
    
    def animate(frame):
        val = slider.val + 1
        if val > sim.n_years: val = 0
        slider.set_val(val)
        return val

    def toggle_animation(event):
        if state['running']:
            if state['obj']: state['obj'].event_source.stop()
            state['running'] = False
            btn.label.set_text('▶ Play')
        else:
            state['obj'] = animation.FuncAnimation(fig, animate, interval=100, save_count=100)
            state['running'] = True
            btn.label.set_text('⏸ Pause')
        fig.canvas.draw_idle()

    ax_btn = plt.axes([0.8, 0.04, 0.1, 0.05])
    btn = Button(ax_btn, '▶ Play')
    btn.on_clicked(toggle_animation)

    update_plot(0)
    plt.show()
