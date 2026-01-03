import numpy as np
from tax_simulation import Simulation, NormalDistribution, FlatTax, ProgressiveTax, TaxBracket

def test_flat_tax():
    print("Testing Flat Tax...")
    # 10k income, 10% tax. Should be 1k tax, 9k post-tax.
    dist = NormalDistribution(10000, 0) # All 10000
    tax = FlatTax(0.10)
    sim = Simulation(dist, tax, n_people=10)
    sim.run()
    
    assert np.allclose(sim.pre_tax_income, 10000), "Pre tax income incorrect"
    assert np.allclose(sim.taxes, 1000), "Tax incorrect"
    assert np.allclose(sim.post_tax_income, 9000), "Post tax income incorrect"
    print("Flat Tax Passed!")

def test_progressive_tax():
    print("Testing Progressive Tax...")
    # Brackets: 0-10k @ 0%, 10k+ @ 50%
    # Income: 20k
    # Tax: 0 on first 10k, 50% on next 10k = 5000. Total 5000.
    brackets = [TaxBracket(0, 0.0), TaxBracket(10000, 0.5)]
    tax = ProgressiveTax(brackets)
    
    dist = NormalDistribution(20000, 0)
    sim = Simulation(dist, tax, n_people=10)
    sim.run()
    
    expected_tax = 5000
    print(f"Income: {sim.pre_tax_income[0]}, Tax: {sim.taxes[0]}")
    assert np.allclose(sim.taxes, expected_tax), f"Expected {expected_tax}, got {sim.taxes[0]}"
    print("Progressive Tax Passed!")

def test_gini():
    print("Testing Gini...")
    # Perfect equality = 0
    dist = NormalDistribution(10000, 0)
    sim = Simulation(dist, FlatTax(0), 100)
    sim.run()
    gini = sim.get_stats()['pre_gini']
    assert gini < 0.01, f"Gini for equality should be 0, got {gini}"
    print("Gini Passed!")

if __name__ == "__main__":
    try:
        test_flat_tax()
        test_progressive_tax()
        test_gini()
        print("\nALL TESTS PASSED")
    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        exit(1)
