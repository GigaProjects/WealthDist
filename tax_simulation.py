import numpy as np
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict
from dataclasses import dataclass
from scipy import stats

# --- Distribution Classes ---

class Distribution(ABC):
    """Abstract base class for income/wealth distributions."""
    
    @abstractmethod
    def generate(self, n: int) -> np.ndarray:
        """Generate n representative quantiles from the distribution."""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        pass

class NormalDistribution(Distribution):
    def __init__(self, mean: float, std_dev: float):
        self.mean = mean
        self.std_dev = std_dev
        self.dist = stats.norm(loc=mean, scale=std_dev)
        
    def generate(self, n: int) -> np.ndarray:
        # Use mid-point quantiles for smooth representation
        p = np.linspace(0.5/n, 1-0.5/n, n)
        return np.maximum(0, self.dist.ppf(p))
    
    def get_name(self) -> str:
        return f"Normal(μ={self.mean}, σ={self.std_dev})"

class LogNormalDistribution(Distribution):
    def __init__(self, mean: float, sigma: float):
        self.real_mean = mean
        self.sigma = sigma
        # E[X] = exp(mu + sigma^2/2) => exp(mu) = E[X] / exp(sigma^2/2)
        scale = mean / np.exp(sigma**2 / 2)
        self.dist = stats.lognorm(s=sigma, scale=scale)
        
    def generate(self, n: int) -> np.ndarray:
        p = np.linspace(0.5/n, 1-0.5/n, n)
        return self.dist.ppf(p)
    
    def get_name(self) -> str:
        return f"LogNormal(Mean={self.real_mean}, σ={self.sigma})"

class ParetoDistribution(Distribution):
    def __init__(self, shape: float, scale: float):
        self.shape = shape
        self.scale = scale
        self.dist = stats.pareto(b=shape, scale=scale)
        
    def generate(self, n: int) -> np.ndarray:
        # Avoid the very extreme tail for Pareto to keep plots readable
        p = np.linspace(0.5/n, 1-1.0/n, n)
        return self.dist.ppf(p)
    
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
        self.deduction = deduction
        
    def calculate_tax(self, incomes: np.ndarray) -> np.ndarray:
        taxable_income = np.maximum(0, incomes - self.deduction)
        return taxable_income * self.rate

    @property
    def name(self) -> str:
        if self.deduction > 0:
            return f"Flat Tax ({self.rate*100:.1f}%) above ${self.deduction:,.0f}"
        else:
            return f"Flat Tax ({self.rate*100:.1f}%)"

@dataclass
class TaxBracket:
    threshold: float
    rate: float

class ProgressiveTax(TaxSystem):
    def __init__(self, brackets: List[TaxBracket]):
        self.brackets = sorted(brackets, key=lambda b: b.threshold)
        
    def calculate_tax(self, incomes: np.ndarray) -> np.ndarray:
        taxes = np.zeros_like(incomes)
        for i, bracket in enumerate(self.brackets):
            start = bracket.threshold
            rate = bracket.rate
            if i + 1 < len(self.brackets):
                end = self.brackets[i+1].threshold
                income_in_bracket = np.clip(incomes - start, 0, end - start)
            else:
                income_in_bracket = np.maximum(0, incomes - start)
            taxes += income_in_bracket * rate
        return taxes

    @property
    def name(self) -> str:
        return "Progressive Tax System"

# --- Simulation Engine ---

class Simulation:
    def __init__(self, distribution: Distribution, tax_system: TaxSystem, n: int = 1000):
        self.distribution = distribution
        self.tax_system = tax_system
        self.n = n
        self.pre_tax_income = None
        self.taxes = None
        self.post_tax_income = None
        self.redistributed_income = None
        
    def run(self):
        # Generate representative quantiles (already sorted)
        self.pre_tax_income = self.distribution.generate(self.n)
        self.taxes = self.tax_system.calculate_tax(self.pre_tax_income)
        self.post_tax_income = self.pre_tax_income - self.taxes
        
        # Redistribution (UBI)
        total_revenue = np.sum(self.taxes)
        self.ubi_amount = total_revenue / self.n
        self.redistributed_income = self.post_tax_income + self.ubi_amount
        
    def get_stats(self) -> Dict:
        if self.pre_tax_income is None: return {}
        return {
            "pre_gini": self.gini(self.pre_tax_income),
            "post_gini": self.gini(self.post_tax_income),
            "redist_gini": self.gini(self.redistributed_income),
            "total_tax_revenue": np.sum(self.taxes),
            "ubi_per_person": self.ubi_amount,
        }
    
    @staticmethod
    def gini(array):
        """Calculate Gini coefficient. Data is assumed to be PRE-SORTED for speed."""
        if len(array) == 0: return 0
        n = len(array)
        # Using the simplified formula for sorted arrays
        index = np.arange(1, n + 1)
        return (np.sum((2 * index - n - 1) * array)) / (n * np.sum(array))


@dataclass
class MultiYearSimulation:
    distribution: Distribution
    tax_system: TaxSystem
    n_people: int = 1000
    n_years: int = 50
    initial_wealth_multiplier: float = 6.0
    depreciation_rate: float = 0.05 # Annual cost/decay (inflation, maintenance, etc.)
    return_on_capital: float = 0.05 # Annual return on existing wealth (r)
    green_line_ubi_pct: float = 0.0 # Percentage of UBI redistributed in the Green line (0.0 to 1.0)
    
    def run(self):
        # 1. Generate core income quantiles (sorted)
        self.annual_income = self.distribution.generate(self.n_people)
        
        # 3. Disposable Labor Incomes (Calculated once)
        self.annual_taxes = self.tax_system.calculate_tax(self.annual_income)
        net_labor_pre = self.annual_income.copy() # Pre-tax labor
        net_labor_post = self.annual_income - self.annual_taxes + (np.sum(self.annual_taxes) / self.n_people * self.green_line_ubi_pct)
        net_labor_ubi  = self.annual_income - self.annual_taxes + (np.sum(self.annual_taxes) / self.n_people)
        
        # 4. Savings Propensity Scale
        # We use the initial labor income distribution as a "standard" to map income to percentile.
        # This ensures that if someone's inflow matches the top labor earners, they save at the top rate.
        income_sorted = np.sort(self.annual_income)
        def get_savings_rate(total_inflow):
            ranks = np.searchsorted(income_sorted, total_inflow) / self.n_people
            ranks = np.clip(ranks, 0, 1)
            return 0.02 + (ranks ** 3) * 0.48

        # 5. Initialization
        wealth_init = self.annual_income * self.initial_wealth_multiplier
        self.w_pre = wealth_init.copy()
        self.w_post = wealth_init.copy()
        self.w_ubi = wealth_init.copy()
        
        # History
        self.history = {'pre': [], 'post': [], 'ubi': []}
        self.gini_history = {'pre': [], 'post': [], 'ubi': []}
        self.gdp_growth_history = {'pre': [0.0], 'post': [0.0], 'ubi': [0.0]}
        
        # 6. Simulation Loop
        for year in range(self.n_years + 1):
            self.history['pre'].append(self.w_pre.copy())
            self.history['post'].append(self.w_post.copy())
            self.history['ubi'].append(self.w_ubi.copy())
            
            self.gini_history['pre'].append(Simulation.gini(self.w_pre))
            self.gini_history['post'].append(Simulation.gini(self.w_post))
            self.gini_history['ubi'].append(Simulation.gini(self.w_ubi))
            
            # Step forward
            if year < self.n_years:
                prev_pre = np.sum(self.w_pre)
                prev_post = np.sum(self.w_post)
                prev_ubi = np.sum(self.w_ubi)
                
                # Dynamic Logic:
                # 1. Calculate Total Annual Inflow (Net Labor + Capital Returns)
                inflow_pre = net_labor_pre + (self.w_pre * self.return_on_capital)
                inflow_post = net_labor_post + (self.w_post * self.return_on_capital)
                inflow_ubi = net_labor_ubi + (self.w_ubi * self.return_on_capital)
                
                # 2. Determine Savings Propensity based on Total Inflow
                s_rate_pre = get_savings_rate(inflow_pre)
                s_rate_post = get_savings_rate(inflow_post)
                s_rate_ubi = get_savings_rate(inflow_ubi)
                
                # 3. Update Wealth: (Maintenance) + (Saved Inflow)
                # Formula: CurrentWealth * (1 - decay) + (TotalInflow * SavingsRate)
                self.w_pre = (self.w_pre * (1 - self.depreciation_rate)) + (inflow_pre * s_rate_pre)
                self.w_post = (self.w_post * (1 - self.depreciation_rate)) + (inflow_post * s_rate_post)
                self.w_ubi = (self.w_ubi * (1 - self.depreciation_rate)) + (inflow_ubi * s_rate_ubi)
                
                # Record GDP Growth
                self.gdp_growth_history['pre'].append((np.sum(self.w_pre)/prev_pre - 1) * 100)
                self.gdp_growth_history['post'].append((np.sum(self.w_post)/prev_post - 1) * 100)
                self.gdp_growth_history['ubi'].append((np.sum(self.w_ubi)/prev_ubi - 1) * 100)
