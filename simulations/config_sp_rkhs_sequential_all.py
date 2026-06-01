from simulations.dgps_mediated import fn_dict, fn_dict_test

import numpy as np
from nnpiv.rkhs import (
    RKHSIV,
    RKHSIVCV,
    RKHSIVL2,
    RKHSIVL2CV,
    ApproxRKHSIV,
    ApproxRKHSIVCV,
    ApproxRKHSIVL2,
    ApproxRKHSIVL2CV,
)


alpha_scales_cv = np.geomspace(0.5, 200, 8)


def _seq_models(base_model):
    # DML_mediated sequential path expects:
    # model1=[stage1, stage2], modelq1=[stage1, stage2]
    return [[base_model, base_model], [base_model, base_model]]


CONFIG = {
    "target_dir": "semiparametric_cov",
    "reload_results": True,
    "dgp_opts": {
        "dgp_name": "rkhs_sequential_all",
        "fn": [0, 1, 2, 4],
        "n_samples": 2000,
    },
    "methods": {
        "RKHSIV": _seq_models(
            RKHSIV(kernel="rbf", gamma=0.01, delta_scale="auto", delta_exp=0.4)
        ),
        "RKHSIVL2": _seq_models(
            RKHSIVL2(kernel="rbf", gamma=0.01, delta_scale="auto", delta_exp=0.4)
        ),
        "ApproxRKHSIV": _seq_models(
            ApproxRKHSIV(
                kernel_approx="nystrom",
                n_components=0.2,
                kernel="rbf",
                gamma=0.01,
                delta_scale="auto",
                delta_exp=0.4,
                alpha_scale=60,
            )
        ),
        "ApproxRKHSIVL2": _seq_models(
            ApproxRKHSIVL2(
                kernel_approx="nystrom",
                n_components=0.2,
                kernel="rbf",
                gamma=0.01,
                delta_scale="auto",
                delta_exp=0.4,
                alpha_scale=60,
            )
        ),
        "RKHSIVCV": _seq_models(
            RKHSIVCV(
                kernel="rbf",
                gamma=0.01,
                delta_scale="auto",
                delta_exp=0.4,
                alpha_scales=alpha_scales_cv,
                cv=3,
                n_alphas=len(alpha_scales_cv),
            )
        ),
        "RKHSIVL2CV": _seq_models(
            RKHSIVL2CV(
                kernel="rbf",
                gamma=0.01,
                delta_scale="auto",
                delta_exp=0.4,
                alpha_scales=alpha_scales_cv,
                cv=3,
                n_alphas=len(alpha_scales_cv),
            )
        ),
        "ApproxRKHSIVCV": _seq_models(
            ApproxRKHSIVCV(
                kernel_approx="nystrom",
                n_components=0.2,
                kernel="rbf",
                gamma=0.01,
                delta_scale="auto",
                delta_exp=0.4,
                alpha_scales=alpha_scales_cv,
                cv=3,
                n_alphas=len(alpha_scales_cv),
            )
        ),
        "ApproxRKHSIVL2CV": _seq_models(
            ApproxRKHSIVL2CV(
                kernel_approx="nystrom",
                n_components=0.2,
                kernel="rbf",
                gamma=0.01,
                delta_scale="auto",
                delta_exp=0.4,
                alpha_scales=alpha_scales_cv,
                cv=3,
                n_alphas=len(alpha_scales_cv),
            )
        ),
    },
    "method_opts": {
        "nn_1": [False, False],
        "nn_q1": [False, False],
        "fitargs": [None, None],
        "fitargsq1": [None, None],
        "CHIM": False,
        "opts": {"lin_degree": 1},
    },
    "estimator": "sequential",
    "mc_opts": {
        "n_experiments": 2500,
        "seed": 123,
    },
}
