Semiparametrics Notebook AGMM
=============================

.. code:: ipython3

    import os
    import numpy as np
    import pandas as pd
    
    import mliv.dgps_nested as dgps
    import mliv.dgps_mediated as dgps
    import matplotlib.pyplot as plt
    
    import torch
    import torch.nn as nn
    from sklearn.cluster import KMeans
    #from mliv.neuralnet.deepiv_fit import deep_iv_fit
    from mliv.neuralnet.rbflayer import gaussian, inverse_multiquadric
    from mliv.neuralnet import AGMM, AGMM2, AGMM2L2
    from mliv.tsls import tsls, regtsls
    
    # Now you can import the module
    from dml_mediated import DML_mediated
    
    p = 0.1  # dropout prob of dropout layers throughout notebook
    n_hidden = 100  # width of hidden layers throughout notebook
    
    
    device = torch.cuda.current_device() if torch.cuda.is_available() else None

.. code:: ipython3

    
    fn_number = 0
    tau_fn = dgps.get_tau_fn(fn_number)
    tauinv_fn = dgps.get_tauinv_fn(fn_number)
    W, Z, X, M, D, Y, tau_fn = dgps.get_data(2000, tau_fn)
    
    V = np.random.rand(Y.shape[0])
    V = V.reshape(-1, 1)
    
    print(np.column_stack((W,X,Z)).shape)
    ind = np.where(D==0)[0]
    W0 = W[ind]
    X0 = X[ind,:]
    W0_test = np.zeros((1000, 1+X.shape[1]))
    W0_test += np.median(np.column_stack((X0,W0)), axis=0, keepdims=True)
    W0_test[:, 2] = np.linspace(np.percentile(
                W0[:, 0], 5), np.percentile(W0[:, 0], 95), 1000)
    
    # True parameters
    b_yd = 2.0; b_ym = 1.0; b_yx = np.array([[-1.0],[-1.0]]); b_yu = -1.0; b_yw = 2.0; b_y0 = 2.0
    b_wx = np.array([[0.2],[0.2]]); b_wu = -0.6; b_w0 = 0.3
    b_md = -0.3; b_mx = np.array([[-0.5],[-0.5]]); b_mu = 0.4; b_m0 = 0.0
        
    gamma_1w = (b_yw*b_wu + b_yu)/b_wu
    gamma_1x = b_yw*b_wx + b_yx - gamma_1w*b_wx
    gamma_1m = b_ym
    gamma_10 = b_y0 + b_yd + b_yw*b_w0 - gamma_1w*b_w0
    
    gamma_0w = (gamma_1m*b_mu + gamma_1w*b_wu)/b_wu
    gamma_0x = gamma_1m*b_mx + gamma_1w*b_wx + gamma_1x - gamma_0w*b_wx
    gamma_00 = gamma_10 + gamma_1m*b_m0 + gamma_1w*b_w0 - gamma_0w*b_w0
    
        # True nuisance function
    expected_te = gamma_00 + tauinv_fn(W0_test)@np.row_stack((gamma_0x, gamma_0w))
    D_ = D.copy()
    


.. parsed-literal::

    (2000, 4)
    

.. code:: ipython3

    def _get_learner(n_t):
        return nn.Sequential(nn.Dropout(p=p), nn.Linear(n_t, n_hidden), nn.LeakyReLU(),
                             nn.Dropout(p=p), nn.Linear(n_hidden, 1))
    
    
    def _get_adversary(n_z):
        return nn.Sequential(nn.Dropout(p=p), nn.Linear(n_z, n_hidden), nn.LeakyReLU(),
                             nn.Dropout(p=p), nn.Linear(n_hidden, 1))
    
    
    agmm_1 = AGMM(_get_learner(4),_get_adversary(4))
    agmm_2 = AGMM(_get_learner(3),_get_adversary(3))

.. code:: ipython3

    dml_agmm = DML_mediated(Y, D, M, W, Z, X,
                            estimator='OR',
                            estimand='E[Y(1,M(0))]',
                            model1 = agmm_1,
                            model2 = agmm_2,
                            modelq1 = agmm_2,
                            modelq2 = agmm_1,
                            n_folds=5, n_rep=1,
                            CHIM = False,
                            nn_1 = True,
                            nn_2 = True,
                            nn_q1 = True,
                            nn_q2 = True,
                            fitargs1 = {'n_epochs': 300, 'bs': 100, 'learner_lr': 1e-4, 'adversary_lr': 1e-4, 'learner_l2': 1e-3, 'adversary_l2': 1e-4, 'adversary_norm_reg' : 1e-3},
                            fitargs2 = {'n_epochs': 300, 'bs': 100, 'learner_lr': 1e-4, 'adversary_lr': 1e-4, 'learner_l2': 1e-3, 'adversary_l2': 1e-4},
                            fitargsq1 = {'n_epochs': 300, 'bs': 100, 'learner_lr': 1e-4, 'adversary_lr': 1e-4, 'learner_l2': 1e-3, 'adversary_l2': 1e-4},
                            fitargsq2 = {'n_epochs': 300, 'bs': 100, 'learner_lr': 1e-4, 'adversary_lr': 1e-4, 'learner_l2': 1e-3, 'adversary_l2': 1e-4},
                            opts = {'lin_degree': 1, 'burnin': 200})
    
    
    print(dml_agmm.dml())


.. parsed-literal::

    Rep: 1
    

.. parsed-literal::

     80%|████████  | 4/5 [05:21<00:47, 47.79s/it] Exception ignored in: <finalize object at 0x227ec129d00; dead>
    Traceback (most recent call last):
      File "C:\Program Files\WindowsApps\PythonSoftwareFoundation.Python.3.9_3.9.3568.0_x64__qbz5n2kfra8p0\lib\weakref.py", line 591, in __call__
        return info.func(*info.args, **(info.kwargs or {}))
      File "C:\Program Files\WindowsApps\PythonSoftwareFoundation.Python.3.9_3.9.3568.0_x64__qbz5n2kfra8p0\lib\tempfile.py", line 820, in _cleanup
        cls._rmtree(name)
      File "C:\Program Files\WindowsApps\PythonSoftwareFoundation.Python.3.9_3.9.3568.0_x64__qbz5n2kfra8p0\lib\tempfile.py", line 816, in _rmtree
        _shutil.rmtree(name, onerror=onerror)
      File "C:\Program Files\WindowsApps\PythonSoftwareFoundation.Python.3.9_3.9.3568.0_x64__qbz5n2kfra8p0\lib\shutil.py", line 759, in rmtree
        return _rmtree_unsafe(path, onerror)
      File "C:\Program Files\WindowsApps\PythonSoftwareFoundation.Python.3.9_3.9.3568.0_x64__qbz5n2kfra8p0\lib\shutil.py", line 629, in _rmtree_unsafe
        onerror(os.unlink, fullname, sys.exc_info())
      File "C:\Program Files\WindowsApps\PythonSoftwareFoundation.Python.3.9_3.9.3568.0_x64__qbz5n2kfra8p0\lib\tempfile.py", line 808, in onerror
        cls._rmtree(path)
      File "C:\Program Files\WindowsApps\PythonSoftwareFoundation.Python.3.9_3.9.3568.0_x64__qbz5n2kfra8p0\lib\tempfile.py", line 816, in _rmtree
        _shutil.rmtree(name, onerror=onerror)
      File "C:\Program Files\WindowsApps\PythonSoftwareFoundation.Python.3.9_3.9.3568.0_x64__qbz5n2kfra8p0\lib\shutil.py", line 759, in rmtree
        return _rmtree_unsafe(path, onerror)
      File "C:\Program Files\WindowsApps\PythonSoftwareFoundation.Python.3.9_3.9.3568.0_x64__qbz5n2kfra8p0\lib\shutil.py", line 610, in _rmtree_unsafe
        onerror(os.scandir, path, sys.exc_info())
      File "C:\Program Files\WindowsApps\PythonSoftwareFoundation.Python.3.9_3.9.3568.0_x64__qbz5n2kfra8p0\lib\shutil.py", line 607, in _rmtree_unsafe
        with os.scandir(path) as scandir_it:
    NotADirectoryError: [WinError 267] The directory name is invalid: '.\\tmp3q0ij066\\epoch292'
    100%|██████████| 5/5 [05:23<00:00, 64.69s/it]

.. parsed-literal::

    (4.1756363, 2.2139876, array([4.1104255, 4.240847 ], dtype=float32))
    

.. parsed-literal::

    
    

.. code:: ipython3

    fitargs1 = {
        'n_epochs': 300, 
        'bs': 100, 
        'learner_lr': 1e-4, 
        'adversary_lr': 1e-4, 
        'learner_l2': 1e-3, 
        'adversary_l2': 1e-4, 
        'adversary_norm_reg': 1e-3
    }
    
    fitargs2 = {
        'n_epochs': 300, 
        'bs': 100, 
        'learner_lr': 1e-4, 
        'adversary_lr': 1e-4, 
        'learner_l2': 1e-3, 
        'adversary_l2': 1e-4
    }
    
    Y, D, M, W, X, Z = map(lambda x: torch.Tensor(x), [Y, D, M, W, X, Z])
    
    ind = np.where(D == 1)[0]
    M1 = M[ind]
    W1 = W[ind]
    X1 = X[ind, :]
    Z1 = Z[ind]
    Y1 = Y[ind]
    
    A2 = torch.cat((M1, X1, Z1), 1)
    A1 = torch.cat((M1, X1, W1), 1)
    
    bridge_1 = agmm_1.fit(A2, A1, Y1, **fitargs1)
    
    A1 = torch.cat((M, X, W), 1)
    bridge_1_hat = torch.Tensor(bridge_1.predict(A1.to(device), model='avg', burn_in=200))
    
    D, W, X, Z, bridge_1_hat = map(lambda x: torch.Tensor(x), [D, W, X, Z, bridge_1_hat])
    
    ind = np.where(D == 0)[0]
    W0 = W[ind]
    X0 = X[ind, :]
    Z0 = Z[ind]
    bridge_1_hat = bridge_1_hat[ind]
    
    B2 = torch.cat((X0, Z0), 1)
    B1 = torch.cat((X0, W0), 1)
    
    bridge_2 = agmm_2.fit(B2, B1, bridge_1_hat, **fitargs2)
    
    gamma_0_hat = bridge_2.predict(torch.cat((X, W), 1).to(device), model='avg', burn_in=200)
    print(np.mean(gamma_0_hat))
    print(np.var(gamma_0_hat))
    


.. parsed-literal::

    4.175998
    5.155775
    

.. code:: ipython3

    A = torch.tensor(np.column_stack((M,X,W)), dtype=torch.float32)
    D = torch.tensor(D_, dtype=torch.float32)
    E = torch.tensor(np.column_stack((M,X,Z)), dtype=torch.float32)
    B = torch.tensor(np.column_stack((X,W)), dtype=torch.float32)
    C = torch.tensor(np.column_stack((X,Z)), dtype=torch.float32)
    
    

.. code:: ipython3

    fitargs = {
        'n_epochs': 100, 
        'bs': 100, 
        'learner_lr': 0.001, 
        'adversary_lr': 0.001, 
        'learner_l2': 1e-3, 
        'adversary_l2': 1e-4
    }
    
    agmm2_model = AGMM2(learnerh = _get_learner(B.shape[1]), learnerg = _get_learner(A.shape[1]),
                         adversary1 = _get_adversary(E.shape[1]), adversary2 = _get_adversary(C.shape[1]))
    
    
    agmm2_pred_b, agmm2_pred_a = agmm2_model.fit(A, B, C, E, Y, subsetted=True, subset_ind1=D, **fitargs).predict(B.to(device), A.to(device), model='avg', burn_in=10)

.. code:: ipython3

    print(np.mean(agmm2_pred_b))
    print(np.var(agmm2_pred_b))


.. parsed-literal::

    3.7640064
    10.023633
    
