import numpy as np

# continuously differentiable
fn_dict_cdiff = {'2dpoly': 1, 'sigmoid': 2,
                 'sin': 3, 'frequent_sin': 4,
                 '3dpoly': 7, 'linear': 8, 'exponential': 16}
                 
# continuous but not differentiable
fn_dict_cont = {'abs': 0, 'abs_sqrt': 5, 'rand_pw': 9,
                'abspos': 10, 'sqrpos': 11, 'pwlinear': 15}

# discontinuous
fn_dict_disc = {'step': 6, 'band': 12, 'invband': 13,
                'steplinear': 14}

# monotone
fn_dict_monotone = {'sigmoid': 2,
                    'step': 6, 'linear': 8,
                    'abspos': 10, 'sqrpos': 11, 'pwlinear': 15, 'exponential': 16}

# convex
fn_dict_convex = {'abs': 0, '2dpoly': 1, 'linear': 8,
                  'abspos': 10, 'sqrpos': 11, 'exponential': 16}

# for sparse
fn_dict_sparse = {'abs': 0, 'linear': 8}

# paper
fn_dict_paper = {'linear': 8, 'pwlinear': 15, 'sigmoid': 2, 'exponential': 16}
                 
# all functions
fn_dict = {'abs': 0, '2dpoly': 1, 'sigmoid': 2,
           'sin': 3, 'frequent_sin': 4, 'abs_sqrt': 5,
           'step': 6, '3dpoly': 7, 'linear': 8, 'rand_pw': 9,
           'abspos': 10, 'sqrpos': 11, 'band': 12, 'invband': 13,
           'steplinear': 14, 'pwlinear': 15, 'exponential': 16}



def generate_random_pw_linear(lb=-2, ub=2, n_pieces=5):
    splits = np.random.choice(np.arange(lb, ub, 0.1),
                              n_pieces - 1, replace=False)
    splits.sort()
    slopes = np.random.uniform(-4, 4, size=n_pieces)
    start = []
    start.append(np.random.uniform(-1, 1))
    for t in range(n_pieces - 1):
        start.append(start[t] + slopes[t] * (splits[t] -
                                             (lb if t == 0 else splits[t - 1])))
    return lambda x: [start[ind] + slopes[ind] * (x - (lb if ind == 0 else splits[ind - 1])) for ind in [np.searchsorted(splits, x)]][0]


def get_tau_fn(func):
    def first(x):
        return x[:, [0]] if len(x.shape) == 2 else x
    # func describes the relation between response and treatment
    if func == fn_dict['abs']:
        def tau_fn(x): return np.abs(first(x))
    elif func == fn_dict['2dpoly']:
        def tau_fn(x): return -1.5 * first(x) + .9 * (first(x)**2)
    elif func == fn_dict['sigmoid']:
        def tau_fn(x): return 2 / (1 + np.exp(-2 * first(x)))
    elif func == fn_dict['sin']:
        def tau_fn(x): return np.sin(first(x))
    elif func == fn_dict['frequent_sin']:
        def tau_fn(x): return np.sin(3 * first(x))
    elif func == fn_dict['abs_sqrt']:
        def tau_fn(x): return np.sqrt(np.abs(first(x)))
    elif func == fn_dict['step']:
        def tau_fn(x): return 1. * (first(x) < 0) + 2.5 * (first(x) >= 0)
    elif func == fn_dict['3dpoly']:
        def tau_fn(x): return -1.5 * first(x) + .9 * \
            (first(x)**2) + first(x)**3
    elif func == fn_dict['linear']:
        def tau_fn(x): return first(x)
    elif func == fn_dict['rand_pw']:
        pw_linear = generate_random_pw_linear()

        def tau_fn(x):
            return np.array([pw_linear(x_i) for x_i in first(x).flatten()]).reshape(-1, 1)
    elif func == fn_dict['abspos']:
        def tau_fn(x): return np.abs(first(x)) * (first(x) >= 0)
    elif func == fn_dict['sqrpos']:
        def tau_fn(x): return (first(x)**2) * (first(x) >= 0)
    elif func == fn_dict['band']:
        def tau_fn(x): return 1.0 * (first(x) >= -.75) * (first(x) <= .75)
    elif func == fn_dict['invband']:
        def tau_fn(x): return 1. - 1. * (first(x) >= -.75) * (first(x) <= .75)
    elif func == fn_dict['steplinear']:
        def tau_fn(x): return 2. * (first(x) >= 0) - first(x)
    elif func == fn_dict['pwlinear']:
        def tau_fn(x):
            q = first(x)
            return (q + 1) * (q <= -1) + (q - 1) * (q >= 1)
    elif func == fn_dict['exponential']:
        def tau_fn(x): return np.exp(first(x))
    else:
        raise NotImplementedError()

    return tau_fn


def standardize(A1, A2, B1, B2, y, fn):
    ym = y.mean()
    ystd = y.std()
    y = (y - ym) / ystd

    def newfn(x): return (fn(x) - ym) / ystd
    return A1, A2, B1, B2, y, newfn


def get_data(n_samples, n_a, n_b, tau_fn, dgp_num):
    # Construct dataset
    # A1 :- endogeneous treatment in first stage
    # A2 :- instrument in first stage
    # B1 :- endogeneous treatment in second stage
    # B2 :- instrument in second stage
    # Y :- response (is a scalar always)

    fn = tau_fn

    U = np.random.normal(0, 1, size=(n_samples, 1))
    mean = np.zeros(n_a+n_b) 
    cov = np.eye(n_a + n_b)
    #Variance of A2
    cov[0:n_a, 0:n_a] = 0.5*cov[0:n_a, 0:n_a]
    #Variance of B2
    cov[n_a:, n_a:] = 1*cov[n_a:, n_a:]
    AB = np.random.multivariate_normal(mean, cov, n_samples)
    A2 = AB[:, 0:n_a]
    B2 = AB[:, n_a:]
    B1 = B2+U+np.random.normal(0, 0.1, size=(n_samples, 1))
    A1 = fn(B1)+U+A2
    if dgp_num == 1:
        # Independence
        epsilon = np.random.normal(0, .1, size=(n_samples, 1))
        A1 = np.cbrt(A1 + epsilon)
        Y = np.power(A1[:,0],3)
        Y = Y.reshape(n_samples,1)+U+np.random.normal(0, 0.1, size=(n_samples, 1))
    elif dgp_num == 2:
        # Mean Independence
        epsilon = (np.minimum(np.abs(B2[:,0]),1)*np.random.normal(0,1, size=(n_samples, 1)).T).T
        A1 = np.cbrt(A1 + epsilon)
        epsilon2 = (np.minimum(np.abs(A2[:,0]),1)*np.random.normal(0,1, size=(n_samples, 1)).T).T
        Y = np.power(A1[:,0],3)
        Y = Y.reshape(n_samples,1)+U+epsilon2
    else:
        # Uncorrelated
        epsilon = B2[:,0]**2-1
        A1 = np.cbrt(A1 + epsilon.reshape(n_samples,1))
        epsilon2 = A2[:,0]**2-1
        Y = np.power(A1[:,0],3)
        Y = Y.reshape(n_samples,1)+U+epsilon2.reshape(n_samples,1)

    return standardize(A1, A2, B1, B2, Y, fn)
