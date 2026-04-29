"""Trading labels: triple-barrier + sample weighting (López de Prado, AFML §3-4)."""

from .triple_barrier import (
    avg_uniqueness_weights,
    triple_barrier_events,
    triple_barrier_labels,
)

__all__ = [
    "avg_uniqueness_weights",
    "triple_barrier_events",
    "triple_barrier_labels",
]
