Semiparametrics : DML (mediated) with Neural Nets — AGMM (sequential) & AGMM2L2 (simultaneous)
=============================

.. code:: ipython3

  # ---- Limit BLAS/OpenMP threads BEFORE importing heavy libs ----
  import os as os
  os.environ.setdefault("OMP_NUM_THREADS", "1")
  os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
  os.environ.setdefault("MKL_NUM_THREADS", "1")
  os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
  os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

  # ---- Standard libs ----
  import sys
  import time
  import platform
  from pathlib import Path

  # ---- Third-party ----
  import numpy as np
  from threadpoolctl import threadpool_limits

  # Keep native libraries (BLAS/OpenMP) to 1 thread
  threadpool_limits(1)

  # ---- Local repo imports (adjust path if needed) ----
  sys.path.append(str(Path.cwd() / "../../simulations"))
  import dgps_mediated as dgps  

  import torch 
  import torch.nn as nn  
  from nnpiv.neuralnet.agmm import AGMM  
  from nnpiv.neuralnet.agmm2 import AGMM2L2 
  from nnpiv.semiparametrics import DML_mediated  


  # -----------------------
  # Reproducibility helpers
  # -----------------------
  def seed_everything(seed: int = 123) -> None:
      """Set seeds for reproducibility."""
      import random
      random.seed(seed)
      np.random.seed(seed)
      torch.manual_seed(seed)
      if torch.cuda.is_available():
          torch.cuda.manual_seed_all(seed)
      # Keep torch to 1 thread too
      try:
          torch.set_num_threads(1)
          torch.set_num_interop_threads(1)
      except Exception:
          pass

  seed_everything(123)

  # -----------------------
  # Resource print utility
  # -----------------------
  def print_resources():
      """Print basic compute resource info (CPU, GPU, library versions)."""
      cpu_cores = os.cpu_count()
      pyver = sys.version.split()[0]
      npver = np.__version__
      torchver = torch.__version__
      if torch.cuda.is_available():
          try:
              gpu_name = torch.cuda.get_device_name(0)
          except Exception:
              gpu_name = "Unknown GPU"
          gpu_info = f"CUDA: available — {gpu_name}"
      else:
          gpu_info = "CUDA: not available"
      print("=== Compute resources ===")
      print(f"Python: {pyver}")
      print(f"NumPy: {npver}")
      print(f"PyTorch: {torchver}")
      print(f"CPU cores: {cpu_cores}")
      print(gpu_info)
      print("Thread caps (env):")
      for k in ["OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS","NUMEXPR_NUM_THREADS"]:
          print(f"  {k}={os.environ.get(k, 'unset')}")
      print(f"Platform: {platform.platform()}")
      print("=========================\n")


  # -----------------------
  # Result formatter
  # -----------------------
  def summarize_dml_result(name: str, result, elapsed: float):
      """
      Accepts result from .dml() and prints θ, SE, 95% CI when available.
      Compatible with returns like (theta, var, ci) or (theta, var, ci, cov).
      """
      if isinstance(result, tuple):
          if len(result) == 3:
              theta, var, ci = result
              cov = None
          elif len(result) == 4:
              theta, var, ci, cov = result
          else:
              print(f"[{name}] time={elapsed:.2f}s — result={result}")
              return
      else:
          print(f"[{name}] time={elapsed:.2f}s — result={result}")
          return

      theta = np.atleast_1d(theta).astype(float)
      var = np.atleast_1d(var).astype(float)
      se = np.sqrt(var)
      ci = np.array(ci, dtype=float) if ci is not None else None

      def fmt_arr(a):
          return f"{float(a[0]):.4f}" if a.size == 1 else np.array2string(a, precision=4)

      print(f"[{name}] time={elapsed:.2f}s")
      print(f"  theta: {fmt_arr(theta)}")
      print(f"  SE   : {fmt_arr(se)}")
      if ci is not None:
          if ci.ndim == 1 and ci.size == 2:
              print(f"  95% CI: [{ci[0]:.4f}, {ci[1]:.4f}]")
          else:
              print(f"  95% CI: {np.array2string(ci, precision=4)}")
      if 'cov' in locals() and cov is not None:
          print(f"  (cov shape: {cov.shape})")
      print("")


.. code:: ipython3

  # -----------------------
  # Print resources 
  # -----------------------
  print_resources()
  DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


.. parsed-literal::

  === Compute resources ===
  Python: 3.10.18
  NumPy: 2.2.6
  PyTorch: 2.5.0
  CPU cores: 112
  CUDA: not available
  Thread caps (env):
    OMP_NUM_THREADS=1
    OPENBLAS_NUM_THREADS=1
    MKL_NUM_THREADS=1
    VECLIB_MAXIMUM_THREADS=1
    NUMEXPR_NUM_THREADS=1
  Platform: Linux-4.18.0-553.44.1.el8_10.x86_64-x86_64-with-glibc2.28
  =========================    


.. code:: ipython3

  # =========================================================
  # Data generation
  # =========================================================
  # Function dictionary (for reference):
  # {'abs': 0, '2dpoly': 1, 'sigmoid': 2,
  #  'sin': 3, 'frequent_sin': 4, 'abs_sqrt': 5, 'step': 6, '3dpoly': 7,
  #  'linear': 8, 'rand_pw': 9, 'abspos': 10, 'sqrpos': 11, 'band': 12,
  #  'invband': 13, 'steplinear': 14, 'pwlinear': 15, 'exponential': 16}

  fn_number = 0
  tau_fn = dgps.get_tau_fn(fn_number)
  tauinv_fn = dgps.get_tauinv_fn(fn_number)  # kept for parity with your code
  W, Z, X, M, D, Y, tau_fn = dgps.get_data(2000, tau_fn)

  # Ground-truth value for the target estimand (for log reference)
  TRUE_PARAM = 4.05
  print(f"=== Ground truth (for log reference) ===\nTrue parameter for E[Y(1,M(0))] ≈ {TRUE_PARAM:.2f}\n")


  # =========================================================
  # NN architecture helpers (dropout & width are configurable)
  # =========================================================
  p = 0.10
  n_hidden = 100

  def _get_learner(n_t: int) -> nn.Module:
      return nn.Sequential(
          nn.Dropout(p=p), nn.Linear(n_t, n_hidden), nn.LeakyReLU(),
          nn.Dropout(p=p), nn.Linear(n_hidden, 1)
      )

  def _get_adversary(n_z: int) -> nn.Module:
      return nn.Sequential(
          nn.Dropout(p=p), nn.Linear(n_z, n_hidden), nn.LeakyReLU(),
          nn.Dropout(p=p), nn.Linear(n_hidden, 1)
      )


  # =========================================================
  # Model builders (dimensions inferred from data)
  # =========================================================
  def build_agmm_pair_for_mediated(M, X, W, Z):
      """
      Build two AGMM models with correct input dims for the mediated setup.
      Stage 1 (bridge on treated arm):
          T = [M, X, W], Z = [M, X, Z]
      Stage 2:
          T = [X, W],     Z = [X, Z]
      """
      T1_dim = M.shape[1] + X.shape[1] + W.shape[1]
      Z1_dim = M.shape[1] + X.shape[1] + Z.shape[1]
      T2_dim = X.shape[1] + W.shape[1]
      Z2_dim = X.shape[1] + Z.shape[1]
      m1 = AGMM(_get_learner(T1_dim), _get_adversary(Z1_dim))
      m2 = AGMM(_get_learner(T2_dim), _get_adversary(Z2_dim))
      return m1, m2

  def build_agmm2_for_mediated(M, X, W, Z):
      """For model1 (outcome bridge)."""
      A_dim = M.shape[1] + X.shape[1] + W.shape[1]   
      B_dim = X.shape[1] + W.shape[1]                 
      E_dim = M.shape[1] + X.shape[1] + Z.shape[1]   
      C_dim = X.shape[1] + Z.shape[1]                 
      return AGMM2L2(
          learnerh=_get_learner(B_dim),
          learnerg=_get_learner(A_dim),
          adversary1=_get_adversary(E_dim),
          adversary2=_get_adversary(C_dim),
      )

  def build_agmm2_for_mediated_q1(M, X, W, Z):
      """For model_q1 (q-bridge)."""
      A_prime_dim = X.shape[1] + W.shape[1]                 #  (this goes to learnerg)
      B_prime_dim = M.shape[1] + X.shape[1] + W.shape[1]    #  (this goes to learnerh)
      D_prime_dim = X.shape[1] + Z.shape[1]                 #  (this goes to adversary1)
      C_prime_dim = M.shape[1] + X.shape[1] + Z.shape[1]    #  (this goes to adversary2)
      return AGMM2L2(
          learnerh=_get_learner(B_prime_dim),   
          learnerg=_get_learner(A_prime_dim),   
          adversary1=_get_adversary(D_prime_dim),  
          adversary2=_get_adversary(C_prime_dim),  
      )

.. parsed-literal::

  === Ground truth (for log reference) ===
  True parameter for E[Y(1,M(0))] ≈ 4.05

.. code:: ipython3

  # =========================================================
  # 1) Sequential estimator (MR) with AGMM
  # =========================================================
  m1, m2 = build_agmm_pair_for_mediated(M, X, W, Z)
  fitargs_seq = {
      "n_epochs": 300, "bs": 100,
      "learner_lr": 1e-4, "adversary_lr": 1e-4,
      "learner_l2": 1e-3, "adversary_l2": 1e-4,
      "adversary_norm_reg": 1e-3,
      "device": DEVICE,
  }
  dml_agmm = DML_mediated(
      Y, D, M, W, Z, X,
      estimator="MR",
      estimand="E[Y(1,M(0))]",
      nn_1=[True, True],         # use torch path for both bridge stages
      nn_q1=[True, True],        # and for q-models
      model1=[m1, m2],
      modelq1=[m2, m1],          # your original ordering
      n_folds=5, n_rep=1,
      fitargs1=[fitargs_seq, fitargs_seq],
      fitargsq1=[fitargs_seq, fitargs_seq],
      opts={"lin_degree": 1, "burnin": 200},
  )
  t0 = time.perf_counter()
  res_seq = dml_agmm.dml()
  t1 = time.perf_counter()
  summarize_dml_result("Sequential (MR) with AGMM", res_seq, t1 - t0)


  # =========================================================
  # 2) Simultaneous estimator (MR) with AGMM2L2
  # =========================================================
  agmm2_model_1  = build_agmm2_for_mediated(M, X, W, Z)
  agmm2_model_q1 = build_agmm2_for_mediated_q1(M, X, W, Z)

  fitargs_sim = {
      "n_epochs": 600, "bs": 100,
      "learner_lr": 1e-4, "adversary_lr": 1e-4,
      "learner_l2": 1e-3, "adversary_l2": 1e-4,
      "device": DEVICE,
  }
  opts_sim = {"burnin": 400}


  dml2_agmm = DML_mediated(
      Y, D, M, W, Z, X,
      estimator="MR",
      estimand="E[Y(1,M(0))]",
      model1=agmm2_model_1, nn_1=True,
      modelq1=agmm2_model_q1, nn_q1=True,
      fitargs1=fitargs_sim,
      fitargsq1=fitargs_sim,
      n_folds=5, n_rep=1, opts=opts_sim,
  )
  t0 = time.perf_counter()
  res_sim = dml2_agmm.dml()
  t1 = time.perf_counter()
  summarize_dml_result("Simultaneous (MR) with AGMM2L2", res_sim, t1 - t0)

.. parsed-literal::

  Rep: 1
  100%|██████████| 5/5 [03:17<00:00, 39.59s/it] 
  [Sequential (MR) with AGMM] time=197.93s
    theta: 4.0745
    SE   : 5.2253
    95% CI: [3.8455, 4.3035]

  Rep: 1
  100%|██████████| 5/5 [11:24<00:00, 136.81s/it]
  [Simultaneous (MR) with AGMM2L2] time=684.06s
    theta: 4.1246
    SE   : 5.2737
    95% CI: [3.8935, 4.3557]
    
