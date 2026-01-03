import sys
import subprocess

def check_dependencies():
    required = {'numpy', 'matplotlib', 'seaborn', 'scipy'}
    installed = set()
    
    try:
        import numpy
        installed.add('numpy')
        import matplotlib
        installed.add('matplotlib')
        import seaborn
        installed.add('seaborn')
        import scipy
        installed.add('scipy')
    except ImportError:
        pass
        
    missing = required - installed
    if missing:
        print(f"Missing libraries: {missing}")
        print("Installing now...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', *missing])
        print("Installed!")

# Run check before imports
check_dependencies()

from tax_simulation import (
    Simulation, 
    NormalDistribution, LogNormalDistribution, ParetoDistribution,
    FlatTax, ProgressiveTax, TaxBracket
)
from visualizer import plot_simulation_results

def get_float_input(prompt, default=None):
    while True:
        try:
            val_str = input(f"{prompt} (default {default}): " if default is not None else f"{prompt}: ")
            if not val_str and default is not None:
                return default
            return float(val_str)
        except ValueError:
            print("Please enter a valid number.")

def main():
    print("=== Income Tax Distribution Simulator ===")
    
    while True:
        # 1. Select Distribution
        print("\n--- Step 1: Select Initial Distribution ---")
        print("1. LogNormal (Good for realistic income)")
        print("2. Pareto (Good for 'rich get richer' tail)")
        print("3. Normal (Bell curve, educational)")
        
        choice = input("Choice (1-3): ")
        
        dist = None
        if choice == '1':
            mean = get_float_input("Enter Mean Income", 50000)
            sigma = get_float_input("Enter Sigma (inequality param, ~0.5-1.0)", 0.75)
            dist = LogNormalDistribution(mean, sigma)
        elif choice == '2':
            print("Note: Pareto is usually for top incomes. Be careful with parameters.")
            alpha = get_float_input("Enter Shape (Alpha, ~1.16 for 80/20)", 1.16)
            scale = get_float_input("Enter Scale (Min Income)", 10000)
            dist = ParetoDistribution(alpha, scale)
        elif choice == '3':
            mean = get_float_input("Enter Mean Income", 50000)
            std = get_float_input("Enter Std Dev", 15000)
            dist = NormalDistribution(mean, std)
        else:
            print("Invalid choice.")
            continue
            
        # 2. Select Tax System
        print("\n--- Step 2: Configure Tax System ---")
        print("1. Flat Tax")
        print("2. Progressive Tax (Brackets)")
        
        tax_choice = input("Choice (1-2): ")
        tax_system = None
        
        if tax_choice == '1':
            rate = get_float_input("Enter Tax Rate (0.0 - 1.0)", 0.20)
            deduction = get_float_input("Enter Standard Deduction (0 for none)", 10000)
            tax_system = FlatTax(rate, deduction)
        elif tax_choice == '2':
            brackets = []
            print("Enter brackets. Type 'done' when finished.")
            print("Format: threshold rate")
            print("Example: 0 0.10 (0% to 10%)")
            
            # Helper for user
            print("Let's add the first bracket (starting at 0)")
            rate0 = get_float_input("Rate for income starting at 0", 0.0)
            brackets.append(TaxBracket(0, rate0))
            
            while True:
                more = input("Add another bracket? (y/n): ")
                if more.lower() != 'y':
                    break
                thresh = get_float_input("Enter Threshold (income amount)")
                rate = get_float_input("Enter Rate (0.0 - 1.0) above this threshold")
                brackets.append(TaxBracket(thresh, rate))
                
            tax_system = ProgressiveTax(brackets)
        else:
            print("Invalid choice.")
            continue
            
        # 3. Simulate
        print("\n--- Running Simulation ---")
        n_people = 100000
        sim = Simulation(dist, tax_system, n_people)
        sim.run()
        
        stats = sim.get_stats()
        print("\n=== Results ===")
        print(f"Pre-Tax Gini:           {stats['pre_gini']:.4f}")
        print(f"Post-Tax Gini:          {stats['post_gini']:.4f}")
        print(f"Post-Redistribution Gini: {stats['redist_gini']:.4f}")
        print("-" * 30)
        print(f"Total Revenue collected: ${stats['total_tax_revenue']:,.2f}")
        print(f"UBI Payout per person:   ${stats['ubi_per_person']:,.2f}")
        
        print("\nShowing plots...")
        plot_simulation_results(sim)
        
        again = input("\nRun another simulation? (y/n): ")
        if again.lower() != 'y':
            break

if __name__ == "__main__":
    main()
