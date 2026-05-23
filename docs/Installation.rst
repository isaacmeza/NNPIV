Installation & Replication
====================

.. _installation:


A Python package for nested NPIV estimation with RKHS, neural networks
(AGMM/AGMM2), linear/ensemble baselines, and DML-based semiparametric
procedures. The repository also contains scripts to reproduce all simulation
tables and empirical figures.

- Package source: ``nnpiv/``
- Simulation drivers: ``simulations/``
- Notebooks (usage & empirical replications): ``local_notebooks/``

.. contents::
   :local:
   :depth: 2


1. Installation
---------------

The project is PEP 517/518 compliant (``pyproject.toml``).

1.1. Create and activate an environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # From repository root
   python3.14 -m venv .venv
   source .venv/bin/activate
   python -m pip install --upgrade pip

For cluster jobs, the Slurm runner keeps using:

.. code-block:: bash

   module load python/3.13
   mamba activate nnpiv_venv

1.2. Install dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Base requirements (CPU-only friendly, Python 3.14)
   pip install -r requirements.txt

   # (Optional) If you are on a cluster, you can also use the cluster pin file
   # pip install -r requirements_cluster.txt

If you want GPU acceleration for PyTorch, install the wheel that matches your
CUDA runtime from PyTorch’s index.




1.3. Install the package
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # From the repository root
   pip install -e .

This installs ``nnpiv`` in editable mode for development and replication.

Alternatively, you can use the following command (deprecated):

.. code-block:: console

   python setup.py develop

2. What’s in the box?
---------------------

- **Core estimators** (``nnpiv``):
  RKHS (exact & Nyström-approximate), AGMM/AGMM2, linear & ensemble baselines,
  and semiparametric DML engines (long-term + mediated variants).
- **Simulations** (``simulations/``):
  *Nonparametric* experiments (Table 1) and *Semiparametric* coverage experiments
  (Table 2), with config files to switch DGP/estimators and Slurm/local runners.
- **Notebooks** (``local_notebooks/``):
  Usage examples and replication of empirical figures (Project STAR; Job Corps).


3. Quick start (library)
------------------------

Example: long-term effects via DML + RKHS.

.. code-block:: python

   import numpy as np
   from sklearn.linear_model import LogisticRegression
   from nnpiv.rkhs import ApproxRKHSIVCV
   from nnpiv.semiparametrics import DML_longterm

   # Toy shapes: Y,D,S,G are (n,1)
   Y, D, S, G = [np.random.randn(1000,1) for _ in range(4)]

   m1 = ApproxRKHSIVCV(kernel_approx='nystrom', n_components=400,
                       kernel='rbf', gamma=.001, delta_scale='auto',
                       delta_exp=.4, alpha_scales=np.geomspace(1, 10000, 10), cv=10)
   m2 = ApproxRKHSIVCV(kernel_approx='nystrom', n_components=400,
                       kernel='rbf', gamma=.001, delta_scale='auto',
                       delta_exp=.4, alpha_scales=np.geomspace(1, 10000, 10), cv=10)

   dml = DML_longterm(Y, D, S, G,
                      longterm_model='latent_unconfounded',
                      model1=[m1, m2],
                      n_folds=5, n_rep=1, CHIM=False,
                      prop_score=LogisticRegression(max_iter=2000))
   theta, var, ci = dml.dml()
   print(theta, var, ci)


4. Reproducing the simulations
------------------------------

4.1. Folder layout
~~~~~~~~~~~~~~~~~~

- ``simulations/``
  - ``config_*.py`` — configuration files (DGP, estimators, seeds, output paths)
  - ``run_simulations_local.sh`` — canonical local execution script
  - ``run_simulations.sbatch`` — canonical Slurm execution script
  - ``submit_simulations.sh`` — submission helper with resource profiles
  - ``sweep_np.py`` / ``sweep_sp.py`` — experiment drivers
  - ``./nonparametric_fit/`` — results
  - ``./semiparametric_cov/`` — results

4.2. Nonparametric simulations (Table 1)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Run **locally**:

.. code-block:: bash

   cd simulations
   ./run_simulations_local.sh --config config_np_benchmark

Smoke test locally:

.. code-block:: bash

   cd simulations
   ./run_simulations_local.sh --config config_np_benchmark --smoke-test

Run on **Slurm**:

.. code-block:: bash

   cd simulations
   ./submit_simulations.sh --profile sapphire --config config_np_benchmark

(Config input can be ``config_x``, ``config_x.py``, or a path like
``simulations/config_x.py``.)

4.3. Semiparametric coverage simulations (Table 2)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Run **locally**:

.. code-block:: bash

   cd simulations
   ./run_simulations_local.sh --config config_sp_benchmark

Smoke test locally:

.. code-block:: bash

   cd simulations
   ./run_simulations_local.sh --config config_sp_benchmark --smoke-test

Run on **Slurm**:

.. code-block:: bash

   cd simulations
   ./submit_simulations.sh --profile sapphire --config config_sp_benchmark

4.4. Unified Slurm options
~~~~~~~~~~~~~~~~~~~~~~~~~~

Run all configs as a Slurm array:

.. code-block:: bash

   cd simulations
   ./submit_simulations.sh --profile test --all-configs --smoke-test

Run all configs locally:

.. code-block:: bash

   cd simulations
   ./run_simulations_local.sh --all-configs --smoke-test

Track per-config runtime locally (printed summary + CSV log):

.. code-block:: bash

   cd simulations
   ./run_simulations_local.sh --all-configs --smoke-test --timing-log ./timings_smoke.csv

Override replication count and seed:

.. code-block:: bash

   cd simulations
   ./submit_simulations.sh --profile shared --config config_np_nn --n-experiments 50 --seed 999

4.5. Notes on parallelism & threads
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To avoid oversubscription with joblib/NumPy/BLAS/OpenMP, we cap native threads
to 1.  The Slurm scripts already export:

.. code-block:: bash

   export OMP_NUM_THREADS=1
   export OPENBLAS_NUM_THREADS=1
   export MKL_NUM_THREADS=1
   export VECLIB_MAXIMUM_THREADS=1
   export NUMEXPR_NUM_THREADS=1

Internally, the Python drivers also set ``threadpoolctl(1)`` when appropriate.



5. Empirical replications (notebooks)
-------------------------------------

.. toctree::
   :maxdepth: 2

   notebooks/longitudinal_notebook_agmm.rst
   notebooks/semiparametrics_notebook_agmm.rst


Replication notebooks are located in ``local_notebooks/``:

- **STAR long-term outcomes** — reproduces paper figures (RKHS + NN).
- **Job Corps mediation** — DML(mediated) with neural nets and RKHS.

.. note::

   The repository includes paths expecting CSVs under ``data/``.
   Data might not be redistributed for license reasons.



6. Repository structure
-----------------------

.. code-block:: text

   NNPIV/
   ├─ nnpiv/                    # package
   ├─ simulations/              # simulation configs + runners
   │  ├─ run_simulations_local.sh # Local execution runner
   │  ├─ run_simulations.sbatch # Slurm execution runner
   │  ├─ submit_simulations.sh  # Slurm submission helper
   │  ├─ sweep_np.py            # driver (NP)
   │  ├─ sweep_sp.py            # driver (SP)
   │  └─ config_*.py            # experiment configs
   ├─ local_notebooks/          # usage + empirical replications
   ├─ data/                     # (data; not always distributed)
   ├─ output/                   # results (created on run)
   ├─ pyproject.toml
   ├─ requirements.txt
   └─ README.rst



7. Citing 
----------------------

If you use this package, please cite the associated paper and code artifact::

    Meza, I., & Singh, R. (2025). Nested Nonparametric Instrumental Variable Regression.
    https://doi.org/10.48550/arXiv.2112.14249


8. License
-----------

MIT License (see ``LICENSE.txt``).
