"""EVCC-A: grouping of soil test results under variability and borehole-continuity constraints."""

from .candidates import exact_columns, exact_geometric_columns
from .gates import group_audit
from .ordered import best_ordered
from .partition import Partition, exact_oracle, solve_master
from .plans import build_plan, gate_diagnoses, revision_operations
from .policy import Policy, nu_critical
from .solve import candidate_family, solve_context, summarize

__version__ = "1.0.0"

__all__ = [
    "Partition",
    "Policy",
    "best_ordered",
    "build_plan",
    "candidate_family",
    "exact_columns",
    "exact_geometric_columns",
    "exact_oracle",
    "gate_diagnoses",
    "group_audit",
    "nu_critical",
    "revision_operations",
    "solve_context",
    "solve_master",
    "summarize",
]
