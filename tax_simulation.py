import numpy as np
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict
from dataclasses import dataclass

# --- Distribution Classes ---

class Distribution(ABC):
    """Abstract base class for income/wealth distributions."""
    
    @abstractmethod
    def generate(self, n: int) -> np.ndarray:
        """Generate n samples from the distribution."""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        pass

class NormalDistribution(Distribution):
    def __init__(self, mean: float, std_dev: float):
        self.mean = mean
        self.std_dev = std_dev
        
    def generate(self, n: int) -> np.ndarray:
        return np.maximum(0, np.random.normal(self.mean, self.std_dev, n))
    
    def get_name(self) -> str:
        return f"Normal(μ={self.mean}, σ={self.std_dev})"

class LogNormalDistribution(Distribution):
    def __init__(self, mean: float, sigma: float):
        """
        mean: Real mean of the distribution (not the underlying normal)
        sigma: Standard deviation of the underlying normal distribution (shape parameter)
        """
        # Convert real mean to underlying mu
        # E[X] = exp(mu + sigma^2/2) => mu = ln(E[X]) - sigma^2/2
        self.real_mean = mean
        self.sigma = sigma
        self.mu = np.log(mean) - (sigma**2 / 2)
        
    def generate(self, n: int) -> np.ndarray:
        return np.random.lognormal(self.mu, self.sigma, n)
    
    def get_name(self) -> str:
        return f"LogNormal(Mean={self.real_mean}, σ={self.sigma})"

class ParetoDistribution(Distribution):
    def __init__(self, shape: float, scale: float):
        """
        shape (alpha): Tail index. Lower = more inequality (typically 1.16 used for 80-20 rule).
        scale (xm): Minimum value.
        """
        self.shape = shape
        self.scale = scale
        
    def generate(self, n: int) -> np.ndarray:
        # Pareto II (Lomax) or Type I? classic Pareto Type I: x >= xm
        return (np.random.pareto(self.shape, n) + 1) * self.scale
    
    def get_name(self) -> str:
        return f"Pareto(α={self.shape}, x_min={self.scale})"


# --- Tax System Classes ---

class TaxSystem(ABC):
    @abstractmethod
    def calculate_tax(self, incomes: np.ndarray) -> np.ndarray:
        """Return the tax amount for each income."""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        pass

class FlatTax(TaxSystem):
    def __init__(self, rate: float, deduction: float = 0.0):
        self.rate = rate
        self.deduction = deduction  # Standard deduction
        
    def calculate_tax(self, incomes: np.ndarray) -> np.ndarray:
        taxable_income = np.maximum(0, incomes - self.deduction)
        return taxable_income * self.rate

    @property
    def name(self) -> str:
        if self.deduction > 0:
            return f"Flat Tax ({self.rate*100:.1f}%) on everything above ${self.deduction:,.0f}"
        else:
            return f"Flat Tax ({self.rate*100:.1f}%) on all income"

@dataclass
class TaxBracket:
    threshold: float
    rate: float

class ProgressiveTax(TaxSystem):
    def __init__(self, brackets: List[TaxBracket]):
        """
        brackets: List of TaxBracket. 
        Example: [TaxBracket(0, 0.10), TaxBracket(10000, 0.20)]
        Means: 10% on 0-10k, 20% on everything above 10k.
        Logic: Sort by threshold.
        """
        self.brackets = sorted(brackets, key=lambda b: b.threshold)
        
    def calculate_tax(self, incomes: np.ndarray) -> np.ndarray:
        taxes = np.zeros_like(incomes)
        
        # We need to calculate marginal tax for each segment
        # Iterate through brackets. 
        # For bracket i, income covered is min(income, next_threshold) - current_threshold
        
        # Example: 
        # 0     -> 10%
        # 10000 -> 20%
        # 50000 -> 40%
        
        # If income is 60000:
        # 0-10000 (10k) * 0.10 = 1000
        # 10000-50000 (40k) * 0.20 = 8000
        # 50000-60000 (10k) * 0.40 = 4000
        # Total = 13000
        
        for i, bracket in enumerate(self.brackets):
            # Start of this bracket
            start = bracket.threshold
            rate = bracket.rate
            
            # End of this bracket (next bracket's start, or infinity)
            if i + 1 < len(self.brackets):
                end = self.brackets[i+1].threshold
                # Amount of income in this bracket
                # It is income above start, capped at end-start
                income_in_bracket = np.clip(incomes - start, 0, end - start)
            else:
                # Last bracket, goes to infinity
                income_in_bracket = np.maximum(0, incomes - start)
            
            taxes += income_in_bracket * rate
            
        return taxes

    @property
    def name(self) -> str:
        s = "Progressive Tax Brackets:\n"
        for i, b in enumerate(self.brackets):
            rate_pct = f"{b.rate*100:.1f}%"
            if i + 1 < len(self.brackets):
                next_t = self.brackets[i+1].threshold
                s += f"  ${b.threshold:,.0f} - ${next_t:,.0f}: {rate_pct}\n"
            else:
                s += f"  Everything over ${b.threshold:,.0f}: {rate_pct}\n"
        return s.strip()

# --- Simulation Engine ---

class Simulation:
    def __init__(self, distribution: Distribution, tax_system: TaxSystem, n_people: int = 10000):
        self.distribution = distribution
        self.tax_system = tax_system
        self.n = n_people
        self.pre_tax_income = None
        self.taxes = None
        self.post_tax_income = None
        self.redistributed_income = None
        
    def run(self):
        self.pre_tax_income = self.distribution.generate(self.n)
        self.taxes = self.tax_system.calculate_tax(self.pre_tax_income)
        self.post_tax_income = self.pre_tax_income - self.taxes
        
        # Calculate Redistribution (UBI)
        total_revenue = np.sum(self.taxes)
        ubi_amount = total_revenue / self.n
        self.redistributed_income = self.post_tax_income + ubi_amount
        
    def get_stats(self) -> Dict:
        if self.pre_tax_income is None:
            return {}
            
        return {
            "pre_mean": np.mean(self.pre_tax_income),
            "post_mean": np.mean(self.post_tax_income),
            "pre_gini": self.gini(self.pre_tax_income),
            "post_gini": self.gini(self.post_tax_income),
            "redist_gini": self.gini(self.redistributed_income),
            "total_tax_revenue": np.sum(self.taxes),
            "ubi_per_person": np.sum(self.taxes) / self.n,
            "avg_tax_rate": np.mean(self.taxes / np.maximum(1, self.pre_tax_income))
        }
    
    @staticmethod
    def gini(array):
        """Calculate the Gini coefficient of a numpy array."""
        # from https://github.com/oliviaguest/gini
        if len(array) == 0: return 0
        array = array.flatten()
        if np.amin(array) < 0:
            array -= np.amin(array)  # Values cannot be negative
        array += 0.0000001  # Values cannot be 0
        array = np.sort(array)
        index = np.arange(1, array.shape[0] + 1)
        n = array.shape[0]
        return ((np.sum((2 * index - n - 1) * array)) / (n * np.sum(array)))


@dataclass
class MultiYearSimulation:
    distribution: Distribution
    tax_system: TaxSystem
    n_people: int = 10000
    n_years: int = 50
    savings_rate: float = 0.20
    initial_wealth_multiplier: float = 6.0
    
    def __post_init__(self):
        self.annual_income = None
        
        # Parallel Realities (Independent Timelines)
        self.wealth_pre = None
        self.wealth_post = None
        self.wealth_ubi = None
        
        # History tracks
        self.history = {
            'pre': [],
            'post': [],
            'ubi': []
        }
        # Gini histories
        self.gini_history = {
            'pre': [],
            'post': [],
            'ubi': []
        }
        
    def run(self):
        # 1. Generate constant annual income for all scenarios
        self.annual_income = self.distribution.generate(self.n_people)
        
        # 2. Calculate constant annual flows for each scenario
        # Pre-tax scenario
        annual_income_pre = self.annual_income
        annual_taxes_pre = np.zeros_like(self.annual_income) # No taxes in pre-tax scenario
        annual_post_tax_pre = annual_income_pre
        
        # Post-tax scenario (without UBI)
        self.annual_taxes = self.tax_system.calculate_tax(self.annual_income)
        annual_post_tax_post = self.annual_income - self.annual_taxes
        
        # UBI scenario (post-tax with redistribution)
        total_revenue_ubi = np.sum(self.annual_taxes) # UBI revenue comes from the same tax system
        ubi_per_person = total_revenue_ubi / self.n_people
        annual_redist_ubi = annual_post_tax_post + ubi_per_person
        
        # 3. Initialize Wealth for all scenarios (Year 0)
        initial_wealth = self.annual_income * self.initial_wealth_multiplier
        self.wealth_pre = initial_wealth.copy()
        self.wealth_post = initial_wealth.copy()
        self.wealth_ubi = initial_wealth.copy()
        
        # 4. Calculate annual savings for each scenario (DIVERGENT timelines)
        save_pre = annual_post_tax_pre * self.savings_rate
        save_post = annual_post_tax_post * self.savings_rate
        save_ubi = annual_redist_ubi * self.savings_rate
        
        # 5. Simulation Loop (Including Year 0)
        for year in range(self.n_years + 1):
            # Record current state of all 3 universes
            self._record_state()
            
            # Step forward (except on the last year)
            if year < self.n_years:
                self.wealth_pre += save_pre
                self.wealth_post += save_post
                self.wealth_ubi += save_ubi
            
    def _record_state(self):
        # Store full history
        self.history['pre'].append(self.wealth_pre.copy())
        self.history['post'].append(self.wealth_post.copy())
        self.history['ubi'].append(self.wealth_ubi.copy())
        
        # Track Ginis
        self.gini_history['pre'].append(Simulation.gini(self.wealth_pre))
        self.gini_history['post'].append(Simulation.gini(self.wealth_post))
        self.gini_history['ubi'].append(Simulation.gini(self.wealth_ubi))
