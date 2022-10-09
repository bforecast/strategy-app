from .MOM import MOMStrategy
from .PairTrade import PairTradeStrategy
from .MA import MAStrategy
from .RSI import RSIStrategy
from .MACD import MACDStrategy
from .MOM_RSI import MOM_RSIStrategy
from .SuperTrend import SuperTrendStrategy
from .CSPR import CSPRStrategy

__all__ = ["MOMStrategy", "PairTradeStrategy", "MAStrategy", "RSIStrategy", "MACDStrategy", "MOM_RSIStrategy", "SuperTrendStrategy", "CSPRStrategy"]
strategy_list = ["MA", "MACD", "MOM", "PairTrade", "RSI", "MOM_RSI", "SuperTrend", "CSPR"]