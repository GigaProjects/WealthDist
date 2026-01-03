
import numpy as np
from tax_simulation import MultiYearSimulation, LogNormalDistribution, ProgressiveTax, TaxBracket, Simulation

# Setup minimal simulation
dist = LogNormalDistribution(mean=50000, sigma=0.75)
brackets = [
    TaxBracket(0, 0.10),
    TaxBracket(50000, 0.30),
    TaxBracket(100000, 0.50)
]
tax = ProgressiveTax(brackets)

sim = MultiYearSimulation(dist, tax, n_people=10000, n_years=10, savings_rate=0.20, initial_wealth_multiplier=2.0)
sim.run()

print("Year | Pre-Tax Gini (Blue) | Actual/UBI Gini (Orange) | Wealth Base Mean")
print("-" * 70)

for year in range(11):
    w_pre = sim.history['pre'][year]
    w_ubi = sim.history['ubi'][year]
    
    gini_pre = Simulation.gini(w_pre)
    gini_ubi = Simulation.gini(w_ubi)
    mean_base = np.mean(w_ubi)
    
    print(f"{year:4d} | {gini_pre:.4f}             | {gini_ubi:.4f}               | ${mean_base:,.0f}")

print("\nCheck: Is UBI Gini decreasing? Is Pre Gini > UBI Gini?")
