"""
Backend analytical modules for TaxiLens AI.
Provides modular data cleaning, statistical aggregation, and narrative synthesis.
"""

from .preprocessor import clean_and_preprocess
from .analytics import compute_aggregates
from .narrator import generate_insights, answer_query, generate_stakeholder_insights

__all__ = [
    "clean_and_preprocess",
    "compute_aggregates",
    "generate_insights",
    "answer_query",
    "generate_stakeholder_insights"
]
