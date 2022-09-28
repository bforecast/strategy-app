from .MOM import MOMStrategy
from .PairTrade import PairTradeStrategy
from .MA import MAStrategy
from .RSI import RSIStrategy
from .MACD import MACDStrategy

__all__ = ["MOMStrategy", "PairTradeStrategy", "MAStrategy", "RSIStrategy", "MACDStrategy"]
strategy_list = ["MOM", "PairTrade", "MA", "RSI", "MACD"]