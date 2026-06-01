# Licensed under the MIT License.

"""
Public diagnostics API for estimator-agnostic relative well-posedness checks.

This module re-exports user-facing helpers from :mod:`nnpiv.diagnostics.core`
in four families:

1. Pre-estimation diagnostics.
2. Pre-estimation sieve-path diagnostics.
3. Post-estimation effective-direction diagnostics.
4. Post-estimation effective-direction sieve-path diagnostics.
"""

from .core import (
    relative_wellposedness_diagnostic,
    relative_wellposedness_sieve_diagnostic,
    relative_wellposedness_effective_diagnostic,
    relative_wellposedness_effective_sieve_diagnostic,
    relative_wellposedness_from_data,
    relative_wellposedness_effective_from_data,
    relative_wellposedness_sieve_from_data,
    relative_wellposedness_effective_sieve_from_data,
    relative_wellposedness_from_nested_npiv,
    relative_wellposedness_effective_from_nested_npiv,
    relative_wellposedness_sieve_from_nested_npiv,
    relative_wellposedness_effective_sieve_from_nested_npiv,
)

__all__ = [
    "relative_wellposedness_diagnostic",
    "relative_wellposedness_sieve_diagnostic",
    "relative_wellposedness_effective_diagnostic",
    "relative_wellposedness_effective_sieve_diagnostic",
    "relative_wellposedness_from_data",
    "relative_wellposedness_effective_from_data",
    "relative_wellposedness_sieve_from_data",
    "relative_wellposedness_effective_sieve_from_data",
    "relative_wellposedness_from_nested_npiv",
    "relative_wellposedness_effective_from_nested_npiv",
    "relative_wellposedness_sieve_from_nested_npiv",
    "relative_wellposedness_effective_sieve_from_nested_npiv",
]
