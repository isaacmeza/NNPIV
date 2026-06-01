from simulations.dgps_mediated import fn_dict, fn_dict_test

from nnpiv.rkhs import ApproxRKHS2IV


model = ApproxRKHS2IV(
    kernel_approx="nystrom",
    n_components=0.2,
    kernel="rbf",
    gamma=0.01,
    delta_scale="auto",
    delta_exp=0.4,
    alpha_scale=1,
)


CONFIG = {
    "target_dir": "semiparametric_cov",
    "reload_results": True,
    "dgp_opts": {
        "dgp_name": "approxrkhs2iv",
        "fn": [0, 1, 2, 4],
        "n_samples": 2000,
    },
    "methods": {
        "ApproxRKHS2IV": [model, model],
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
        "skip_failed_runs": True,
    },
}
