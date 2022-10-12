from .MOM import MOMStrategy
from .PairTrade import PairTradeStrategy
from .MA import MAStrategy
from .RSI import RSIStrategy
from .MACD import MACDStrategy
from .MOM_RSI import MOM_RSIStrategy
from .SuperTrend import SuperTrendStrategy
from .CSPR import CSPRStrategy
from .RSI3 import RSI3Strategy
from .PETOR import PETORStrategy

__all__ = ["MOMStrategy", "PairTradeStrategy", "MAStrategy", "RSIStrategy", "MACDStrategy", "MOM_RSIStrategy", "SuperTrendStrategy", "CSPRStrategy", "RSI3Strategy", "PETORStrategy"]
strategy_list = ["MA", "MACD", "MOM", "PairTrade", "RSI", "MOM_RSI", "SuperTrend", "CSPR", "RSI3", "PETOR"]