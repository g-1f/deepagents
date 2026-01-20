"""
Tool Definitions for Deep Agent
===============================
Tools are the atomic operations that skills orchestrate.
Each tool is a LangChain BaseTool that can be dynamically bound to subagents.
"""

import json
import random
from datetime import datetime, timedelta
from typing import Optional, Type
from pydantic import BaseModel, Field

from langchain_core.tools import BaseTool, tool
from deep_agent import DeepAgent


# ============================================================================
# TOOL INPUT SCHEMAS
# ============================================================================

class SymbolInput(BaseModel):
    """Input for single symbol tools."""
    symbol: str = Field(description="Stock ticker symbol (e.g., AAPL, MSFT)")


class OHLCVInput(BaseModel):
    """Input for OHLCV data retrieval."""
    symbol: str = Field(description="Stock ticker symbol")
    period: str = Field(default="1M", description="Time period: 1D, 1W, 1M, 3M, 1Y, 5Y")
    interval: str = Field(default="1d", description="Data interval: 1m, 5m, 1h, 1d")


class OptionsChainInput(BaseModel):
    """Input for options chain retrieval."""
    symbol: str = Field(description="Stock ticker symbol")
    expiry: Optional[str] = Field(default=None, description="Expiration date (YYYY-MM-DD)")


class OptionsPricerInput(BaseModel):
    """Input for options pricing."""
    spot: float = Field(description="Current spot price")
    strike: float = Field(description="Strike price")
    expiry_days: int = Field(description="Days to expiration")
    volatility: float = Field(description="Implied volatility (as decimal, e.g., 0.25)")
    rate: float = Field(default=0.05, description="Risk-free rate")
    option_type: str = Field(default="call", description="Option type: call or put")


class KellyInput(BaseModel):
    """Input for Kelly criterion calculation."""
    win_prob: float = Field(description="Probability of winning (0-1)")
    win_loss_ratio: float = Field(description="Average win / average loss ratio")
    max_fraction: float = Field(default=0.25, description="Maximum fraction of capital to risk")


class SentimentInput(BaseModel):
    """Input for sentiment analysis."""
    text: str = Field(description="Text to analyze for sentiment")


class CodeInput(BaseModel):
    """Input for code execution."""
    code: str = Field(description="Python code to execute")


class RiskInput(BaseModel):
    """Input for risk calculations."""
    returns: list[float] = Field(description="List of historical returns")
    confidence: float = Field(default=0.95, description="VaR confidence level")


class PortfolioInput(BaseModel):
    """Input for portfolio operations."""
    portfolio_id: str = Field(default="default", description="Portfolio identifier")


# ============================================================================
# MARKET DATA TOOLS
# ============================================================================

class FetchOHLCVTool(BaseTool):
    """Fetch OHLCV (Open, High, Low, Close, Volume) data."""
    
    name: str = "fetch_ohlcv"
    description: str = """Fetch historical OHLCV price data for a symbol.
    Returns: date, open, high, low, close, volume, and derived metrics."""
    args_schema: Type[BaseModel] = OHLCVInput
    
    def _run(self, symbol: str, period: str = "1M", interval: str = "1d") -> str:
        """Fetch OHLCV data (mock implementation)."""
        # In production, replace with actual data provider (yfinance, polygon, etc.)
        
        # Generate mock data
        end_date = datetime.now()
        periods_map = {"1D": 1, "1W": 7, "1M": 30, "3M": 90, "1Y": 252, "5Y": 1260}
        num_periods = periods_map.get(period, 30)
        
        base_price = random.uniform(50, 500)
        data = []
        
        for i in range(num_periods):
            date = end_date - timedelta(days=num_periods - i)
            daily_return = random.gauss(0.0005, 0.02)
            base_price *= (1 + daily_return)
            
            open_price = base_price * random.uniform(0.99, 1.01)
            high_price = max(open_price, base_price) * random.uniform(1.0, 1.02)
            low_price = min(open_price, base_price) * random.uniform(0.98, 1.0)
            volume = int(random.uniform(1e6, 1e8))
            
            data.append({
                "date": date.strftime("%Y-%m-%d"),
                "open": round(open_price, 2),
                "high": round(high_price, 2),
                "low": round(low_price, 2),
                "close": round(base_price, 2),
                "volume": volume
            })
        
        # Calculate summary statistics
        closes = [d["close"] for d in data]
        returns = [(closes[i] - closes[i-1]) / closes[i-1] for i in range(1, len(closes))]
        
        summary = {
            "symbol": symbol,
            "period": period,
            "interval": interval,
            "data_points": len(data),
            "latest_close": closes[-1],
            "period_return": round((closes[-1] - closes[0]) / closes[0] * 100, 2),
            "volatility": round((sum(r**2 for r in returns) / len(returns)) ** 0.5 * (252**0.5) * 100, 2),
            "high_52w": max(closes),
            "low_52w": min(closes),
            "avg_volume": int(sum(d["volume"] for d in data) / len(data)),
            "recent_data": data[-5:]  # Last 5 data points
        }
        
        return json.dumps(summary, indent=2)


class FetchQuoteTool(BaseTool):
    """Fetch current quote for a symbol."""
    
    name: str = "fetch_quote"
    description: str = """Fetch current quote including price, bid/ask, and basic stats."""
    args_schema: Type[BaseModel] = SymbolInput
    
    def _run(self, symbol: str) -> str:
        """Fetch current quote (mock implementation)."""
        price = random.uniform(50, 500)
        change = random.uniform(-5, 5)
        
        quote = {
            "symbol": symbol,
            "timestamp": datetime.now().isoformat(),
            "price": round(price, 2),
            "change": round(change, 2),
            "change_pct": round(change / price * 100, 2),
            "bid": round(price - 0.01, 2),
            "ask": round(price + 0.01, 2),
            "bid_size": random.randint(100, 1000),
            "ask_size": random.randint(100, 1000),
            "volume": random.randint(1000000, 100000000),
            "avg_volume": random.randint(5000000, 50000000),
            "market_cap": round(price * random.uniform(1e8, 1e11), 0)
        }
        
        return json.dumps(quote, indent=2)


class FetchOptionsChainTool(BaseTool):
    """Fetch options chain for a symbol."""
    
    name: str = "fetch_options_chain"
    description: str = """Fetch options chain with strikes, premiums, and Greeks."""
    args_schema: Type[BaseModel] = OptionsChainInput
    
    def _run(self, symbol: str, expiry: Optional[str] = None) -> str:
        """Fetch options chain (mock implementation)."""
        spot = random.uniform(100, 200)
        
        # Generate strikes around ATM
        strikes = [round(spot * m, 0) for m in [0.85, 0.90, 0.95, 1.0, 1.05, 1.10, 1.15]]
        
        chain = {
            "symbol": symbol,
            "spot_price": round(spot, 2),
            "expiry": expiry or (datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d"),
            "iv_rank": random.randint(20, 80),
            "iv_percentile": random.randint(25, 75),
            "calls": [],
            "puts": []
        }
        
        for strike in strikes:
            moneyness = spot / strike
            base_iv = 0.25 + random.uniform(-0.05, 0.05)
            
            # Call option
            call_premium = max(0, spot - strike) + random.uniform(1, 10)
            chain["calls"].append({
                "strike": strike,
                "bid": round(call_premium * 0.98, 2),
                "ask": round(call_premium * 1.02, 2),
                "iv": round(base_iv + (0.1 if moneyness < 1 else 0), 3),
                "delta": round(max(0, min(1, 0.5 + (spot - strike) / (spot * 0.3))), 2),
                "gamma": round(random.uniform(0.01, 0.05), 3),
                "theta": round(-random.uniform(0.02, 0.1), 3),
                "vega": round(random.uniform(0.1, 0.5), 3),
                "volume": random.randint(100, 10000),
                "oi": random.randint(1000, 100000)
            })
            
            # Put option
            put_premium = max(0, strike - spot) + random.uniform(1, 10)
            chain["puts"].append({
                "strike": strike,
                "bid": round(put_premium * 0.98, 2),
                "ask": round(put_premium * 1.02, 2),
                "iv": round(base_iv + (0.1 if moneyness > 1 else 0), 3),
                "delta": round(max(-1, min(0, -0.5 + (spot - strike) / (spot * 0.3))), 2),
                "gamma": round(random.uniform(0.01, 0.05), 3),
                "theta": round(-random.uniform(0.02, 0.1), 3),
                "vega": round(random.uniform(0.1, 0.5), 3),
                "volume": random.randint(100, 10000),
                "oi": random.randint(1000, 100000)
            })
        
        return json.dumps(chain, indent=2)


class FetchFundamentalsTool(BaseTool):
    """Fetch fundamental data for a symbol."""
    
    name: str = "fetch_fundamentals"
    description: str = """Fetch fundamental data including financials, ratios, and estimates."""
    args_schema: Type[BaseModel] = SymbolInput
    
    def _run(self, symbol: str) -> str:
        """Fetch fundamentals (mock implementation)."""
        price = random.uniform(50, 300)
        eps = random.uniform(2, 15)
        
        fundamentals = {
            "symbol": symbol,
            "company_name": f"{symbol} Corporation",
            "sector": random.choice(["Technology", "Healthcare", "Finance", "Consumer"]),
            "market_cap": round(price * random.uniform(1e8, 1e11), 0),
            "valuation": {
                "pe_ratio": round(price / eps, 2),
                "forward_pe": round(price / (eps * 1.1), 2),
                "peg_ratio": round(random.uniform(0.8, 2.5), 2),
                "pb_ratio": round(random.uniform(1, 10), 2),
                "ps_ratio": round(random.uniform(1, 15), 2),
                "ev_ebitda": round(random.uniform(8, 25), 2)
            },
            "profitability": {
                "gross_margin": round(random.uniform(0.3, 0.7) * 100, 1),
                "operating_margin": round(random.uniform(0.1, 0.4) * 100, 1),
                "net_margin": round(random.uniform(0.05, 0.3) * 100, 1),
                "roe": round(random.uniform(0.1, 0.4) * 100, 1),
                "roa": round(random.uniform(0.05, 0.2) * 100, 1)
            },
            "growth": {
                "revenue_growth_yoy": round(random.uniform(-0.1, 0.4) * 100, 1),
                "earnings_growth_yoy": round(random.uniform(-0.2, 0.5) * 100, 1),
                "revenue_growth_5y": round(random.uniform(0.05, 0.25) * 100, 1)
            },
            "estimates": {
                "current_eps": round(eps, 2),
                "next_year_eps": round(eps * random.uniform(1.05, 1.2), 2),
                "analyst_target": round(price * random.uniform(0.9, 1.3), 2),
                "num_analysts": random.randint(5, 30),
                "buy_ratings": random.randint(5, 20),
                "hold_ratings": random.randint(2, 10),
                "sell_ratings": random.randint(0, 5)
            },
            "dividends": {
                "dividend_yield": round(random.uniform(0, 0.04) * 100, 2),
                "payout_ratio": round(random.uniform(0.1, 0.6) * 100, 1)
            }
        }
        
        return json.dumps(fundamentals, indent=2)


# ============================================================================
# NEWS & SENTIMENT TOOLS
# ============================================================================

class FetchNewsTool(BaseTool):
    """Fetch recent news for a symbol or market."""
    
    name: str = "fetch_news"
    description: str = """Fetch recent news headlines and summaries. Can filter by symbol."""
    args_schema: Type[BaseModel] = SymbolInput
    
    def _run(self, symbol: str) -> str:
        """Fetch news (mock implementation)."""
        headlines = [
            f"{symbol} Reports Strong Q4 Earnings, Beats Estimates",
            f"Analysts Upgrade {symbol} on Growth Prospects",
            f"{symbol} Announces New Product Launch",
            f"Market Watch: {symbol} Faces Regulatory Scrutiny",
            f"{symbol} CEO Discusses AI Strategy in Interview",
            f"Institutional Investors Increase {symbol} Holdings",
            f"{symbol} Partners with Industry Leader on New Initiative"
        ]
        
        news = {
            "symbol": symbol,
            "timestamp": datetime.now().isoformat(),
            "articles": []
        }
        
        for i, headline in enumerate(random.sample(headlines, 5)):
            pub_time = datetime.now() - timedelta(hours=random.randint(1, 72))
            news["articles"].append({
                "headline": headline,
                "source": random.choice(["Reuters", "Bloomberg", "CNBC", "WSJ", "FT"]),
                "published": pub_time.isoformat(),
                "sentiment": random.choice(["positive", "negative", "neutral"]),
                "relevance_score": round(random.uniform(0.7, 1.0), 2)
            })
        
        return json.dumps(news, indent=2)


class FetchSECFilingsTool(BaseTool):
    """Fetch recent SEC filings."""
    
    name: str = "fetch_sec_filings"
    description: str = """Fetch recent SEC filings (10-K, 10-Q, 8-K, etc.)."""
    args_schema: Type[BaseModel] = SymbolInput
    
    def _run(self, symbol: str) -> str:
        """Fetch SEC filings (mock implementation)."""
        filings = {
            "symbol": symbol,
            "filings": [
                {
                    "type": "10-Q",
                    "date": (datetime.now() - timedelta(days=45)).strftime("%Y-%m-%d"),
                    "description": "Quarterly Report",
                    "url": f"https://sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={symbol}"
                },
                {
                    "type": "8-K",
                    "date": (datetime.now() - timedelta(days=10)).strftime("%Y-%m-%d"),
                    "description": "Current Report - Material Events",
                    "url": f"https://sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={symbol}"
                },
                {
                    "type": "4",
                    "date": (datetime.now() - timedelta(days=5)).strftime("%Y-%m-%d"),
                    "description": "Insider Trading Report",
                    "url": f"https://sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={symbol}"
                }
            ]
        }
        
        return json.dumps(filings, indent=2)


class AnalyzeSentimentTool(BaseTool):
    """Analyze sentiment of text."""
    
    name: str = "analyze_sentiment"
    description: str = """Analyze sentiment of text, returning score and classification."""
    args_schema: Type[BaseModel] = SentimentInput
    
    def _run(self, text: str) -> str:
        """Analyze sentiment (mock implementation)."""
        # In production, use actual NLP model
        positive_words = ["strong", "growth", "beat", "upgrade", "bullish", "gains"]
        negative_words = ["weak", "decline", "miss", "downgrade", "bearish", "loss"]
        
        text_lower = text.lower()
        pos_count = sum(1 for w in positive_words if w in text_lower)
        neg_count = sum(1 for w in negative_words if w in text_lower)
        
        if pos_count > neg_count:
            sentiment = "positive"
            score = min(1.0, 0.5 + pos_count * 0.1)
        elif neg_count > pos_count:
            sentiment = "negative"
            score = max(-1.0, -0.5 - neg_count * 0.1)
        else:
            sentiment = "neutral"
            score = 0.0
        
        result = {
            "text_preview": text[:100] + "..." if len(text) > 100 else text,
            "sentiment": sentiment,
            "score": round(score, 2),
            "confidence": round(random.uniform(0.7, 0.95), 2),
            "key_phrases": random.sample(positive_words + negative_words, 3)
        }
        
        return json.dumps(result, indent=2)


# ============================================================================
# ANALYSIS TOOLS
# ============================================================================

class ComputeIndicatorsTool(BaseTool):
    """Compute technical indicators."""
    
    name: str = "compute_indicators"
    description: str = """Compute technical indicators (MA, RSI, MACD, Bollinger, etc.)."""
    args_schema: Type[BaseModel] = SymbolInput
    
    def _run(self, symbol: str) -> str:
        """Compute indicators (mock implementation)."""
        price = random.uniform(100, 200)
        
        indicators = {
            "symbol": symbol,
            "price": round(price, 2),
            "moving_averages": {
                "sma_20": round(price * random.uniform(0.95, 1.05), 2),
                "sma_50": round(price * random.uniform(0.90, 1.10), 2),
                "sma_200": round(price * random.uniform(0.85, 1.15), 2),
                "ema_12": round(price * random.uniform(0.97, 1.03), 2),
                "ema_26": round(price * random.uniform(0.95, 1.05), 2)
            },
            "momentum": {
                "rsi_14": round(random.uniform(30, 70), 1),
                "macd": round(random.uniform(-2, 2), 2),
                "macd_signal": round(random.uniform(-2, 2), 2),
                "macd_histogram": round(random.uniform(-1, 1), 2),
                "stoch_k": round(random.uniform(20, 80), 1),
                "stoch_d": round(random.uniform(20, 80), 1)
            },
            "volatility": {
                "atr_14": round(price * random.uniform(0.01, 0.03), 2),
                "bb_upper": round(price * 1.05, 2),
                "bb_middle": round(price, 2),
                "bb_lower": round(price * 0.95, 2),
                "bb_width": round(random.uniform(0.05, 0.15), 3)
            },
            "trend": {
                "adx": round(random.uniform(15, 40), 1),
                "plus_di": round(random.uniform(10, 35), 1),
                "minus_di": round(random.uniform(10, 35), 1)
            },
            "signals": {
                "ma_crossover": random.choice(["bullish", "bearish", "neutral"]),
                "rsi_signal": "oversold" if random.random() < 0.2 else ("overbought" if random.random() > 0.8 else "neutral"),
                "macd_signal": random.choice(["bullish", "bearish", "neutral"]),
                "overall": random.choice(["strong_buy", "buy", "neutral", "sell", "strong_sell"])
            }
        }
        
        return json.dumps(indicators, indent=2)


class OptionsPricerTool(BaseTool):
    """Price options using Black-Scholes."""
    
    name: str = "options_pricer"
    description: str = """Price options using Black-Scholes model. Returns price and Greeks."""
    args_schema: Type[BaseModel] = OptionsPricerInput
    
    def _run(
        self,
        spot: float,
        strike: float,
        expiry_days: int,
        volatility: float,
        rate: float = 0.05,
        option_type: str = "call"
    ) -> str:
        """Price option (simplified mock implementation)."""
        import math
        
        T = expiry_days / 365
        
        # Simplified Black-Scholes
        d1 = (math.log(spot / strike) + (rate + 0.5 * volatility**2) * T) / (volatility * math.sqrt(T))
        d2 = d1 - volatility * math.sqrt(T)
        
        # Normal CDF approximation
        def norm_cdf(x):
            return 0.5 * (1 + math.erf(x / math.sqrt(2)))
        
        if option_type == "call":
            price = spot * norm_cdf(d1) - strike * math.exp(-rate * T) * norm_cdf(d2)
            delta = norm_cdf(d1)
        else:
            price = strike * math.exp(-rate * T) * norm_cdf(-d2) - spot * norm_cdf(-d1)
            delta = norm_cdf(d1) - 1
        
        # Greeks
        gamma = math.exp(-d1**2 / 2) / (spot * volatility * math.sqrt(2 * math.pi * T))
        theta = -(spot * volatility * math.exp(-d1**2 / 2)) / (2 * math.sqrt(2 * math.pi * T))
        vega = spot * math.sqrt(T) * math.exp(-d1**2 / 2) / math.sqrt(2 * math.pi)
        
        result = {
            "inputs": {
                "spot": spot,
                "strike": strike,
                "expiry_days": expiry_days,
                "volatility": volatility,
                "rate": rate,
                "option_type": option_type
            },
            "price": round(price, 2),
            "intrinsic": round(max(0, spot - strike) if option_type == "call" else max(0, strike - spot), 2),
            "time_value": round(price - max(0, spot - strike if option_type == "call" else strike - spot), 2),
            "greeks": {
                "delta": round(delta, 4),
                "gamma": round(gamma, 4),
                "theta": round(theta / 365, 4),  # Daily theta
                "vega": round(vega / 100, 4)  # Per 1% vol change
            },
            "breakeven": round(strike + price if option_type == "call" else strike - price, 2)
        }
        
        return json.dumps(result, indent=2)


class GreeksCalculatorTool(BaseTool):
    """Calculate portfolio Greeks."""
    
    name: str = "greeks_calculator"
    description: str = """Calculate aggregate Greeks for a portfolio of options."""
    args_schema: Type[BaseModel] = PortfolioInput
    
    def _run(self, portfolio_id: str = "default") -> str:
        """Calculate Greeks (mock implementation)."""
        result = {
            "portfolio_id": portfolio_id,
            "aggregate_greeks": {
                "delta": round(random.uniform(-100, 100), 2),
                "gamma": round(random.uniform(0, 10), 2),
                "theta": round(random.uniform(-50, 0), 2),
                "vega": round(random.uniform(0, 100), 2),
                "rho": round(random.uniform(-20, 20), 2)
            },
            "dollar_greeks": {
                "delta_dollars": round(random.uniform(-10000, 10000), 2),
                "gamma_dollars": round(random.uniform(0, 5000), 2),
                "theta_dollars": round(random.uniform(-500, 0), 2),
                "vega_dollars": round(random.uniform(0, 3000), 2)
            },
            "risk_metrics": {
                "max_loss": round(random.uniform(-5000, -500), 2),
                "max_gain": round(random.uniform(1000, 10000), 2),
                "probability_of_profit": round(random.uniform(0.4, 0.7), 2)
            }
        }
        
        return json.dumps(result, indent=2)


class KellyCalculatorTool(BaseTool):
    """Calculate Kelly criterion position sizing."""
    
    name: str = "kelly_calculator"
    description: str = """Calculate optimal position size using Kelly criterion."""
    args_schema: Type[BaseModel] = KellyInput
    
    def _run(
        self,
        win_prob: float,
        win_loss_ratio: float,
        max_fraction: float = 0.25
    ) -> str:
        """Calculate Kelly fraction."""
        # Kelly formula: f* = (bp - q) / b
        # where b = win/loss ratio, p = win prob, q = 1-p
        
        q = 1 - win_prob
        kelly_full = (win_prob * win_loss_ratio - q) / win_loss_ratio
        
        # Apply fractional Kelly and max constraint
        kelly_half = kelly_full * 0.5
        kelly_quarter = kelly_full * 0.25
        recommended = min(max_fraction, max(0, kelly_half))
        
        result = {
            "inputs": {
                "win_probability": win_prob,
                "win_loss_ratio": win_loss_ratio,
                "max_fraction": max_fraction
            },
            "kelly_full": round(kelly_full, 4),
            "kelly_half": round(kelly_half, 4),
            "kelly_quarter": round(kelly_quarter, 4),
            "recommended_fraction": round(recommended, 4),
            "expected_value": round(win_prob * win_loss_ratio - q, 4),
            "edge": round((win_prob * win_loss_ratio - q) / win_loss_ratio * 100, 2),
            "warning": "Negative Kelly - do not take this bet" if kelly_full < 0 else None
        }
        
        return json.dumps(result, indent=2)


# ============================================================================
# RISK TOOLS
# ============================================================================

class RiskCalculatorTool(BaseTool):
    """Calculate risk metrics from returns."""
    
    name: str = "risk_calculator"
    description: str = """Calculate VaR, CVaR, volatility, and other risk metrics."""
    args_schema: Type[BaseModel] = RiskInput
    
    def _run(self, returns: list[float], confidence: float = 0.95) -> str:
        """Calculate risk metrics."""
        import statistics
        
        n = len(returns)
        if n < 10:
            return json.dumps({"error": "Need at least 10 return observations"})
        
        mean_return = statistics.mean(returns)
        std_dev = statistics.stdev(returns)
        
        # Sort returns for VaR calculation
        sorted_returns = sorted(returns)
        var_index = int((1 - confidence) * n)
        var = -sorted_returns[var_index]
        
        # CVaR (Expected Shortfall)
        cvar = -statistics.mean(sorted_returns[:var_index + 1])
        
        # Max drawdown (simplified)
        cumulative = 1
        peak = 1
        max_dd = 0
        for r in returns:
            cumulative *= (1 + r)
            peak = max(peak, cumulative)
            dd = (peak - cumulative) / peak
            max_dd = max(max_dd, dd)
        
        # Sharpe ratio (assuming 0 risk-free rate)
        sharpe = mean_return / std_dev * (252 ** 0.5) if std_dev > 0 else 0
        
        # Sortino ratio
        downside_returns = [r for r in returns if r < 0]
        downside_std = statistics.stdev(downside_returns) if len(downside_returns) > 1 else std_dev
        sortino = mean_return / downside_std * (252 ** 0.5) if downside_std > 0 else 0
        
        result = {
            "observations": n,
            "confidence_level": confidence,
            "returns": {
                "mean_daily": round(mean_return * 100, 4),
                "mean_annual": round(mean_return * 252 * 100, 2),
                "std_daily": round(std_dev * 100, 4),
                "std_annual": round(std_dev * (252 ** 0.5) * 100, 2)
            },
            "var": {
                "daily": round(var * 100, 4),
                "annual": round(var * (252 ** 0.5) * 100, 2)
            },
            "cvar": {
                "daily": round(cvar * 100, 4),
                "annual": round(cvar * (252 ** 0.5) * 100, 2)
            },
            "max_drawdown": round(max_dd * 100, 2),
            "ratios": {
                "sharpe": round(sharpe, 2),
                "sortino": round(sortino, 2)
            }
        }
        
        return json.dumps(result, indent=2)


class CorrelationAnalyzerTool(BaseTool):
    """Analyze correlations between assets."""
    
    name: str = "correlation_analyzer"
    description: str = """Analyze correlation matrix and identify clusters."""
    args_schema: Type[BaseModel] = PortfolioInput
    
    def _run(self, portfolio_id: str = "default") -> str:
        """Analyze correlations (mock implementation)."""
        assets = ["SPY", "QQQ", "IWM", "TLT", "GLD"]
        
        # Generate mock correlation matrix
        corr_matrix = {}
        for i, a1 in enumerate(assets):
            corr_matrix[a1] = {}
            for j, a2 in enumerate(assets):
                if i == j:
                    corr_matrix[a1][a2] = 1.0
                elif i < j:
                    corr = random.uniform(0.3, 0.9) if a2 != "TLT" else random.uniform(-0.3, 0.3)
                    corr_matrix[a1][a2] = round(corr, 2)
                else:
                    corr_matrix[a1][a2] = corr_matrix[a2][a1]
        
        result = {
            "portfolio_id": portfolio_id,
            "assets": assets,
            "correlation_matrix": corr_matrix,
            "clusters": [
                {"name": "Equities", "assets": ["SPY", "QQQ", "IWM"], "avg_correlation": 0.85},
                {"name": "Safe Haven", "assets": ["TLT", "GLD"], "avg_correlation": 0.2}
            ],
            "diversification_ratio": round(random.uniform(1.2, 1.8), 2),
            "effective_assets": round(random.uniform(2.5, 4.0), 1)
        }
        
        return json.dumps(result, indent=2)


class FactorModelTool(BaseTool):
    """Analyze factor exposures."""
    
    name: str = "factor_model"
    description: str = """Analyze portfolio factor exposures (market, size, value, momentum)."""
    args_schema: Type[BaseModel] = PortfolioInput
    
    def _run(self, portfolio_id: str = "default") -> str:
        """Analyze factors (mock implementation)."""
        result = {
            "portfolio_id": portfolio_id,
            "factor_exposures": {
                "market": {"beta": round(random.uniform(0.8, 1.2), 2), "t_stat": round(random.uniform(5, 15), 2)},
                "size": {"beta": round(random.uniform(-0.3, 0.3), 2), "t_stat": round(random.uniform(-2, 2), 2)},
                "value": {"beta": round(random.uniform(-0.2, 0.4), 2), "t_stat": round(random.uniform(-1, 3), 2)},
                "momentum": {"beta": round(random.uniform(-0.1, 0.3), 2), "t_stat": round(random.uniform(-1, 2), 2)},
                "quality": {"beta": round(random.uniform(0, 0.3), 2), "t_stat": round(random.uniform(0, 3), 2)}
            },
            "r_squared": round(random.uniform(0.85, 0.95), 3),
            "alpha_annual": round(random.uniform(-0.02, 0.05), 4),
            "tracking_error": round(random.uniform(0.02, 0.08), 4),
            "information_ratio": round(random.uniform(-0.5, 1.0), 2)
        }
        
        return json.dumps(result, indent=2)


class StressTesterTool(BaseTool):
    """Run stress test scenarios."""
    
    name: str = "stress_tester"
    description: str = """Run historical and hypothetical stress test scenarios."""
    args_schema: Type[BaseModel] = PortfolioInput
    
    def _run(self, portfolio_id: str = "default") -> str:
        """Run stress tests (mock implementation)."""
        result = {
            "portfolio_id": portfolio_id,
            "historical_scenarios": [
                {"name": "2008 Financial Crisis", "impact": round(random.uniform(-30, -15), 1)},
                {"name": "2020 COVID Crash", "impact": round(random.uniform(-25, -10), 1)},
                {"name": "2022 Rate Shock", "impact": round(random.uniform(-20, -5), 1)},
                {"name": "2018 Q4 Selloff", "impact": round(random.uniform(-15, -5), 1)}
            ],
            "hypothetical_scenarios": [
                {"name": "Rates +100bps", "impact": round(random.uniform(-10, 5), 1)},
                {"name": "Rates -100bps", "impact": round(random.uniform(-5, 10), 1)},
                {"name": "VIX Spike to 40", "impact": round(random.uniform(-15, -5), 1)},
                {"name": "USD +10%", "impact": round(random.uniform(-8, 2), 1)},
                {"name": "Oil +50%", "impact": round(random.uniform(-5, 5), 1)}
            ],
            "worst_case": {
                "scenario": "Stagflation",
                "impact": round(random.uniform(-35, -20), 1),
                "recovery_time_months": random.randint(12, 36)
            }
        }
        
        return json.dumps(result, indent=2)


# ============================================================================
# PORTFOLIO TOOLS
# ============================================================================

class FetchPortfolioTool(BaseTool):
    """Fetch portfolio holdings."""
    
    name: str = "fetch_portfolio"
    description: str = """Fetch current portfolio holdings and positions."""
    args_schema: Type[BaseModel] = PortfolioInput
    
    def _run(self, portfolio_id: str = "default") -> str:
        """Fetch portfolio (mock implementation)."""
        holdings = [
            {"symbol": "SPY", "shares": 100, "avg_cost": 420.50, "current_price": 445.20},
            {"symbol": "QQQ", "shares": 50, "avg_cost": 350.00, "current_price": 375.80},
            {"symbol": "AAPL", "shares": 200, "avg_cost": 150.25, "current_price": 175.50},
            {"symbol": "MSFT", "shares": 75, "avg_cost": 320.00, "current_price": 380.00},
            {"symbol": "TLT", "shares": 150, "avg_cost": 105.00, "current_price": 95.50}
        ]
        
        total_value = sum(h["shares"] * h["current_price"] for h in holdings)
        total_cost = sum(h["shares"] * h["avg_cost"] for h in holdings)
        
        for h in holdings:
            h["market_value"] = round(h["shares"] * h["current_price"], 2)
            h["weight"] = round(h["market_value"] / total_value * 100, 2)
            h["gain_loss"] = round((h["current_price"] - h["avg_cost"]) / h["avg_cost"] * 100, 2)
        
        result = {
            "portfolio_id": portfolio_id,
            "total_value": round(total_value, 2),
            "total_cost": round(total_cost, 2),
            "total_gain_loss": round((total_value - total_cost) / total_cost * 100, 2),
            "cash": round(random.uniform(5000, 50000), 2),
            "holdings": holdings
        }
        
        return json.dumps(result, indent=2)


# ============================================================================
# CODE EXECUTION TOOL
# ============================================================================

class CodeExecutorTool(BaseTool):
    """Execute Python code."""
    
    name: str = "code_executor"
    description: str = """Execute Python code and return results. Has access to pandas, numpy, matplotlib."""
    args_schema: Type[BaseModel] = CodeInput
    
    def _run(self, code: str) -> str:
        """Execute code (sandboxed mock implementation)."""
        # In production, use a proper sandbox (Docker, AWS Lambda, etc.)
        
        # For safety, just return a mock response
        result = {
            "status": "success",
            "code_preview": code[:200] + "..." if len(code) > 200 else code,
            "output": "Code executed successfully. [Mock output - in production, actual execution results would appear here]",
            "execution_time_ms": random.randint(10, 500)
        }
        
        return json.dumps(result, indent=2)


# ============================================================================
# REGISTRATION FUNCTION
# ============================================================================

def register_default_tools(agent: DeepAgent):
    """Register all default tools with the agent."""
    
    tools = [
        # Market Data
        FetchOHLCVTool(),
        FetchQuoteTool(),
        FetchOptionsChainTool(),
        FetchFundamentalsTool(),
        
        # News & Sentiment
        FetchNewsTool(),
        FetchSECFilingsTool(),
        AnalyzeSentimentTool(),
        
        # Analysis
        ComputeIndicatorsTool(),
        OptionsPricerTool(),
        GreeksCalculatorTool(),
        KellyCalculatorTool(),
        
        # Risk
        RiskCalculatorTool(),
        CorrelationAnalyzerTool(),
        FactorModelTool(),
        StressTesterTool(),
        
        # Portfolio
        FetchPortfolioTool(),
        
        # Code
        CodeExecutorTool(),
    ]
    
    for tool in tools:
        agent.register_tool(tool)
    
    print(f"Registered {len(tools)} tools:")
    for tool in tools:
        print(f"  - {tool.name}")
