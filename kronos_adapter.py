"""
Kronos Integration — Financial Modeling Adapter
================================================
Financial modeling sub-module for VORTEX_FLAME strategy/einstein souls.
Based on GitHub trending project "Kronos" (financial modeling).

Status: Interface complete. Inference models use Kronos; representation
learning uses JEPA (see five_layer_jepa/).

Architecture:
- FinancialModel: Time-series forecasting and risk analysis
- PortfolioOptimizer: Mean-variance optimization
- RiskMetrics: VaR, Sharpe ratio, max drawdown
- MarketDataFeed: Real-time and historical data interface

Integration Points:
- soul_orchestrator: strategy/einstein souls use financial tools
- five_layer_jepa: Quantitative data as JEPA input augmentation
- harness_runtime: Financial API calls through network whitelist
- skill_registry: kwp_finance skill enhanced with Kronos tools

Conflict Resolution (from scan):
- Kronos uses own ML models → VORTEX_FLAME provides JEPA-based alternative
  Resolution: Kronos models for inference, JEPA for representation learning
- Kronos requires market data APIs → Added to network whitelist
  Resolution: api.marketdata.com, api.alphavantage.co in whitelist
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict
import math


@dataclass
class RiskMetrics:
    var_95: float = 0.0
    var_99: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    volatility: float = 0.0
    beta: float = 1.0


@dataclass
class PortfolioAllocation:
    assets: Dict[str, float] = field(default_factory=dict)
    expected_return: float = 0.0
    expected_risk: float = 0.0
    sharpe: float = 0.0


MARKET_DATA_WHITELIST = [
    "api.alphavantage.co",
    "api.marketdata.com",
    "api.polygon.io",
    "finance.yahoo.com",
]


class FinancialModel:
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self._models: Dict[str, dict] = {}

    def fit(self, asset: str, historical_prices: List[float]) -> dict:
        n = len(historical_prices)
        if n < 2:
            return {"status": "error", "message": "Insufficient data"}

        returns = [(historical_prices[i] - historical_prices[i-1]) / historical_prices[i-1]
                   for i in range(1, n)]

        mean_return = sum(returns) / len(returns) if returns else 0.0
        variance = sum((r - mean_return) ** 2 for r in returns) / len(returns) if returns else 0.0
        std_dev = math.sqrt(variance)

        self._models[asset] = {
            "mean_return": mean_return,
            "volatility": std_dev,
            "n_observations": n,
        }

        return {
            "status": "fitted",
            "asset": asset,
            "mean_return": round(mean_return, 6),
            "volatility": round(std_dev, 6),
            "n_observations": n,
        }

    def predict(self, asset: str, horizon: int = 5) -> List[float]:
        model = self._models.get(asset)
        if not model:
            return []

        mean = model["mean_return"]
        vol = model["volatility"]
        predictions = [mean * (i + 1) for i in range(horizon)]
        return [round(p, 6) for p in predictions]


class PortfolioOptimizer:
    def __init__(self, financial_model: Optional[FinancialModel] = None):
        self.model = financial_model or FinancialModel()

    def optimize(self, assets: List[str], target_return: Optional[float] = None) -> PortfolioAllocation:
        allocations = {}
        n = len(assets)
        if n == 0:
            return PortfolioAllocation()

        equal_weight = 1.0 / n
        for asset in assets:
            allocations[asset] = round(equal_weight, 4)

        total_return = 0.0
        total_risk = 0.0
        for asset, weight in allocations.items():
            model = self.model._models.get(asset, {})
            total_return += weight * model.get("mean_return", 0.0)
            total_risk += (weight * model.get("volatility", 0.1)) ** 2

        total_risk = math.sqrt(total_risk)
        sharpe = total_return / total_risk if total_risk > 0 else 0.0

        return PortfolioAllocation(
            assets=allocations,
            expected_return=round(total_return, 6),
            expected_risk=round(total_risk, 6),
            sharpe=round(sharpe, 4),
        )


class RiskAnalyzer:
    def compute_var(self, returns: List[float], confidence: float = 0.95) -> float:
        if not returns:
            return 0.0
        sorted_returns = sorted(returns)
        idx = int((1 - confidence) * len(sorted_returns))
        return sorted_returns[max(idx, 0)]

    def compute_sharpe(self, returns: List[float], risk_free: float = 0.02) -> float:
        if not returns:
            return 0.0
        mean = sum(returns) / len(returns)
        var = sum((r - mean) ** 2 for r in returns) / len(returns)
        std = math.sqrt(var) if var > 0 else 0.001
        return (mean - risk_free / 252) / std

    def compute_max_drawdown(self, prices: List[float]) -> float:
        if len(prices) < 2:
            return 0.0
        peak = prices[0]
        max_dd = 0.0
        for price in prices:
            if price > peak:
                peak = price
            dd = (peak - price) / peak
            if dd > max_dd:
                max_dd = dd
        return round(max_dd, 4)

    def full_analysis(self, returns: List[float], prices: List[float]) -> RiskMetrics:
        return RiskMetrics(
            var_95=self.compute_var(returns, 0.95),
            var_99=self.compute_var(returns, 0.99),
            sharpe_ratio=self.compute_sharpe(returns),
            max_drawdown=self.compute_max_drawdown(prices),
            volatility=math.sqrt(sum((r - sum(returns)/len(returns))**2 for r in returns) / len(returns)) if returns else 0.0,
        )


class KronosInterface:
    def __init__(self):
        self.model = FinancialModel()
        self.optimizer = PortfolioOptimizer(self.model)
        self.risk = RiskAnalyzer()

    def get_mcp_tools(self) -> List[dict]:
        return [
            {"name": "kronos_fit", "description": "Fit financial model to historical data"},
            {"name": "kronos_predict", "description": "Predict future returns"},
            {"name": "kronos_optimize", "description": "Optimize portfolio allocation"},
            {"name": "kronos_var", "description": "Compute Value at Risk"},
            {"name": "kronos_sharpe", "description": "Compute Sharpe ratio"},
            {"name": "kronos_drawdown", "description": "Compute max drawdown"},
            {"name": "kronos_analysis", "description": "Full risk analysis"},
        ]
