from simulations.dgps_mediated import fn_dict, fn_dict_test

import numpy as np
from nnpiv.rkhs import (
    RKHS2IV,
    RKHS2IVCV,
    RKHS2IVL2,
    RKHS2IVL2CV,
    ApproxRKHS2IV,
    ApproxRKHS2IVCV,
    ApproxRKHS2IVL2,
    ApproxRKHS2IVL2CV,
)


alpha_scales_cv = np.geomspace(0.5, 200, 8)


CONFIG = {
    "target_dir": "semiparametric_cov",
    "reload_results": True,
    "dgp_opts": {
        "dgp_name": "rkhs_simultaneous_all",
        "fn": [0, 1, 2, 4],
        "n_samples": 2000,
    },
    "methods": {
        "RKHS2IV": [
            RKHS2IV(kernel="rbf", gamma=0.01, delta_scale="auto", delta_exp=0.4),
            RKHS2IV(kernel="rbf", gamma=0.01, delta_scale="auto", delta_exp=0.4),
        ],
        "RKHS2IVL2": [
            RKHS2IVL2(kernel="rbf", gamma=0.01, delta_scale="auto", delta_exp=0.4),
            RKHS2IVL2(kernel="rbf", gamma=0.01, delta_scale="auto", delta_exp=0.4),
        ],
        "ApproxRKHS2IV": [
            ApproxRKHS2IV(
                kernel_approx="nystrom",
                n_components=0.2,
                kernel="rbf",
                gamma=0.01,
                delta_scale="auto",
                delta_exp=0.4,
                alpha_scale=1,
            ),
            ApproxRKHS2IV(
                kernel_approx="nystrom",
                n_components=0.2,
                kernel="rbf",
                gamma=0.01,
                delta_scale="auto",
                delta_exp=0.4,
                alpha_scale=1,
            ),
        ],
        "ApproxRKHS2IVL2": [
            ApproxRKHS2IVL2(
                kernel_approx="nystrom",
                n_components=0.2,
                kernel="rbf",
                gamma=0.01,
                delta_scale="auto",
                delta_exp=0.4,
                alpha_scale=1,
            ),
            ApproxRKHS2IVL2(
                kernel_approx="nystrom",
                n_components=0.2,
                kernel="rbf",
                gamma=0.01,
                delta_scale="auto",
                delta_exp=0.4,
                alpha_scale=1,
            ),
        ],
        "RKHS2IVCV": [
            RKHS2IVCV(
                kernel="rbf",
                gamma=0.01,
                delta_scale="auto",
                delta_exp=0.4,
                alpha_scales=alpha_scales_cv,
                cv=3,
                n_alphas=len(alpha_scales_cv),
            ),
            RKHS2IVCV(
                kernel="rbf",
                gamma=0.01,
                delta_scale="auto",
                delta_exp=0.4,
                alpha_scales=alpha_scales_cv,
                cv=3,
                n_alphas=len(alpha_scales_cv),
            ),
        ],
        "RKHS2IVL2CV": [
            RKHS2IVL2CV(
                kernel="rbf",
                gamma=0.01,
                delta_scale="auto",
                delta_exp=0.4,
                alpha_scales=alpha_scales_cv,
                cv=3,
                n_alphas=len(alpha_scales_cv),
            ),
            RKHS2IVL2CV(
                kernel="rbf",
                gamma=0.01,
                delta_scale="auto",
                delta_exp=0.4,
                alpha_scales=alpha_scales_cv,
                cv=3,
                n_alphas=len(alpha_scales_cv),
            ),
        ],
        "ApproxRKHS2IVCV": [
            ApproxRKHS2IVCV(
                kernel_approx="nystrom",
                n_components=0.2,
                kernel="rbf",
                gamma=0.01,
                delta_scale="auto",
                delta_exp=0.4,
                alpha_scales=alpha_scales_cv,
                cv=3,
                n_alphas=len(alpha_scales_cv),
            ),
            ApproxRKHS2IVCV(
                kernel_approx="nystrom",
                n_components=0.2,
                kernel="rbf",
                gamma=0.01,
                delta_scale="auto",
                delta_exp=0.4,
                alpha_scales=alpha_scales_cv,
                cv=3,
                n_alphas=len(alpha_scales_cv),
            ),
        ],
        "ApproxRKHS2IVL2CV": [
            ApproxRKHS2IVL2CV(
                kernel_approx="nystrom",
                n_components=0.2,
                kernel="rbf",
                gamma=0.01,
                delta_scale="auto",
                delta_exp=0.4,
                alpha_scales=alpha_scales_cv,
                cv=3,
                n_alphas=len(alpha_scales_cv),
            ),
            ApproxRKHS2IVL2CV(
                kernel_approx="nystrom",
                n_components=0.2,
                kernel="rbf",
                gamma=0.01,
                delta_scale="auto",
                delta_exp=0.4,
                alpha_scales=alpha_scales_cv,
                cv=3,
                n_alphas=len(alpha_scales_cv),
            ),
        ],
    },
    "method_opts": {
        "nn_1": False,
        "nn_q1": False,
        "fitargs": None,
        "fitargsq1": None,
        "CHIM": False,
        "opts": {"lin_degree": 1},
    },
    "estimator": "joint",
    "mc_opts": {
        "n_experiments": 2500,
        "seed": 123,
    },
}
