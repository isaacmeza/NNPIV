Estimation diagnostics
==========================

NNPIV includes estimator-agnostic diagnostics under :mod:`nnpiv.diagnostics`.
These diagnostics are optional and non-invasive: they do not modify estimator
objectives, but help characterize whether a DGP appears favorable for nested
sequential vs simultaneous estimation before fitting models, and (optionally)
how severe relative conditioning is on the realized first-stage error
direction after estimation.

.. toctree::
   :maxdepth: 2

   diagnostics/Universal
