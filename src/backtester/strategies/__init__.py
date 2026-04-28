from .advanced_trend import AdvancedTrendFollowing
from .ma_crossover import MaCrossover

REGISTRY = {
    "ma_crossover": MaCrossover,
    "advanced_trend": AdvancedTrendFollowing,
}

__all__ = ["MaCrossover", "AdvancedTrendFollowing", "REGISTRY"]
