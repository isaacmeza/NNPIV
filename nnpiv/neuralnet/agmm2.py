'''
Joint neural estimators for nested nonparametric instrumental variables.

Classes:
    _BaseAGMM2: Base class for joint estimation of nested NPIV models.
    _BaseSupLossAGMM2: Base class for joint estimation of nested NPIV models with supervised loss.
    _BaseSupLossAGMM2L2: Base class for empirical-L2-regularized joint estimation.
    AGMM2L2: Empirical-L2-regularized adversarial estimator for nested NPIV.
'''

# Licensed under the MIT License.

import os
import numpy as np
import tempfile
import torch
from torch.utils.data import DataLoader, TensorDataset
from nnpiv.neuralnet.oadam import OAdam

# TODO. This epsilon is used only because pytorch 1.5 has an instability in torch.cdist
# when the input distance is close to zero, due to instability of the square root in
# automatic differentiation. Should be removed once pytorch fixes the instability.
# It can be set to 0 if using pytorch 1.4.0
EPSILON = 1e-2


def _as_cpu_matrix(x, name, *, one_column=False):
    """Convert an array-like input to a finite two-dimensional tensor."""
    try:
        if isinstance(x, torch.Tensor):
            out = x.detach().to(device="cpu", dtype=torch.float32)
        else:
            out = torch.as_tensor(x, dtype=torch.float32)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ValueError(f"{name} must be numeric array-like") from exc

    if out.ndim == 1:
        out = out.reshape(-1, 1)
    if out.ndim != 2:
        raise ValueError(f"{name} must be one- or two-dimensional")
    if one_column and out.shape[1] != 1:
        raise ValueError(f"{name} must contain one value per observation")
    if not torch.isfinite(out).all().item():
        raise ValueError(f"{name} must contain only finite values")
    return out.contiguous()


def _learner_losses(hat_g, hat_h, hat_f1, hat_f2, Y, W,
                    stage1_scale, stage2_scale, learner_norm_reg):
    """Return the two learner losses for the joint empirical game."""
    residual1 = hat_g - Y
    residual2 = hat_h - W * hat_g
    g_loss = (
        stage1_scale * torch.mean(2 * residual1 * hat_f1)
        + stage2_scale * torch.mean(2 * residual2 * hat_f2)
        + learner_norm_reg * torch.mean(hat_g**2)
    )
    h_loss = (
        stage2_scale * torch.mean(2 * residual2 * hat_f2)
        + learner_norm_reg * torch.mean(hat_h**2)
    )
    return g_loss, h_loss


def _adversary_losses(hat_g, hat_h, hat_f1, hat_f2, Y, W,
                      stage1_scale, stage2_scale):
    """Return the two adversary losses for the joint empirical game."""
    residual1 = hat_g - Y
    residual2 = hat_h - W * hat_g
    f2_loss = stage2_scale * (
        -torch.mean(2 * residual2 * hat_f2) + torch.mean(hat_f2**2)
    )
    f1_loss = stage1_scale * (
        -torch.mean(2 * residual1 * hat_f1) + torch.mean(hat_f1**2)
    )
    return f2_loss, f1_loss


def add_weight_decay(net, l2_value, skip_list=()):
    """
    Add weight decay.

    Parameters:
        net (torch.nn.Module): Network whose parameters are grouped.
        l2_value (object): Value for `l2_value`.
        skip_list (object): Value for `skip_list`.
    """
    decay, no_decay = [], []
    for name, param in net.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)
    return [{'params': no_decay, 'weight_decay': 0.},
            {'params': decay,    'weight_decay': l2_value}]


class _BaseAGMM2:
    """
    Base class for joint estimation of nested NPIV models.

    Methods:
        _pretrain: Prepares the variables required to begin training.
        predict: Predicts outcomes using the fitted AGMM model.
    """

    def _pretrain(self, A, B, C, D, Y, W,
                  learner_l2, adversary_l2, learner_norm_reg,
                  learner_lr, adversary_lr,
                  n_epochs, bs, train_learner_every, train_adversary_every,
                  warm_start, model_dir, device, verbose,
                  add_sample_inds=False, subsetted=False,
                  subset_ind1=None, subset_ind2=None):
        """ Prepares the variables required to begin training. """
        self.verbose = verbose

        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        self.tempdir = tempfile.TemporaryDirectory(dir=model_dir)
        self.model_dir = self.tempdir.name
        self.n_epochs = n_epochs

        if device is None:
            # prefer the device of any tensor input, otherwise fall back to CUDA/CPU
            for t in (A, B, C, D, Y, W, subset_ind1, subset_ind2):
                if isinstance(t, torch.Tensor):
                    device = t.device
                    break
            if device is None:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device)

        A = _as_cpu_matrix(A, "A")
        B = _as_cpu_matrix(B, "B")
        C = _as_cpu_matrix(C, "C")
        D = _as_cpu_matrix(D, "D")
        Y = _as_cpu_matrix(Y, "Y", one_column=True)
        W = (torch.ones_like(Y) if W is None
             else _as_cpu_matrix(W, "W", one_column=True))

        n = Y.shape[0]
        if n == 0:
            raise ValueError("Y must contain at least one observation")
        for name, value in (("A", A), ("B", B), ("C", C), ("D", D), ("W", W)):
            if value.shape[0] != n:
                raise ValueError(f"{name} must have the same number of observations as Y")

        if subsetted:
            if subset_ind1 is None:
                raise ValueError("subset_ind1 must be provided when subsetted=True")
            subset_ind1 = _as_cpu_matrix(
                subset_ind1, "subset_ind1", one_column=True
            )
            if subset_ind1.shape[0] != n:
                raise ValueError("subset_ind1 must have the same length as Y")
            if not torch.all((subset_ind1 == 0) | (subset_ind1 == 1)).item():
                raise ValueError("subset_ind1 must be a binary indicator")

            if subset_ind2 is None:
                subset_ind2 = 1 - subset_ind1
            else:
                subset_ind2 = _as_cpu_matrix(
                    subset_ind2, "subset_ind2", one_column=True
                )
                if subset_ind2.shape[0] != n:
                    raise ValueError("subset_ind2 must have the same length as Y")
                if not torch.all((subset_ind2 == 0) | (subset_ind2 == 1)).item():
                    raise ValueError("subset_ind2 must be a binary indicator")

            p = float(subset_ind1.sum().item())
            q = float(subset_ind2.sum().item())
            if p == 0 or q == 0:
                raise ValueError("each stage subset must be nonempty")
            self.stage1_scale_ = n / p
            self.stage2_scale_ = n / q
        else:
            self.stage1_scale_ = 1.0
            self.stage2_scale_ = 1.0

        if add_sample_inds:
            sample_inds = torch.arange(Y.shape[0]).clone().detach()
            self.train_ds = TensorDataset(A, B, C, D, Y, W, sample_inds) if not subsetted else \
                TensorDataset(A, B, C, D, Y, W, sample_inds, subset_ind1, subset_ind2)
        else:
            self.train_ds = TensorDataset(A, B, C, D, Y, W) if not subsetted else \
                TensorDataset(A, B, C, D, Y, W, subset_ind1, subset_ind2)

        # Pin memory only when training on CUDA
        pin = self.device.type == "cuda"
        self.train_dl = DataLoader(self.train_ds, batch_size=bs, shuffle=True, pin_memory=pin)

        # Move networks to device
        self.learnerh  = self.learnerh.to(self.device)
        self.learnerg  = self.learnerg.to(self.device)
        self.adversary1 = self.adversary1.to(self.device)
        self.adversary2 = self.adversary2.to(self.device)

        # Optional warm start
        if not warm_start:
            for net in (self.learnerh, self.learnerg, self.adversary1, self.adversary2):
                net.apply(lambda m: m.reset_parameters() if hasattr(m, 'reset_parameters') else None)

        beta1 = 0.0
        # Optimizers with weight decay L2 on parameters
        self.optimizerg = OAdam(
            add_weight_decay(self.learnerg, learner_l2),
            lr=learner_lr, betas=(beta1, .01))
        self.optimizerh = OAdam(
            add_weight_decay(self.learnerh, learner_l2),
            lr=learner_lr, betas=(beta1, .01))
        self.optimizerf = OAdam(
            add_weight_decay(self.adversary2, adversary_l2, skip_list=self.skip_list),
            lr=adversary_lr, betas=(beta1, .01))
        self.optimizerf_ = OAdam(
            add_weight_decay(self.adversary1, adversary_l2, skip_list=self.skip_list),
            lr=adversary_lr, betas=(beta1, .01))

        return (A, B, C, D, Y, W) if not subsetted else (A, B, C, D, Y, W, subset_ind1, subset_ind2)

    def predict(self, B, A, model='avg', burn_in=0, alpha=None):
        """
        Predict both fitted bridge functions from saved epoch models.

        Parameters:
            B (array-like): Covariates at which to evaluate the second bridge
                ``h``.
            A (array-like): Covariates at which to evaluate the first bridge
                ``g``.
            model (str or int): ``'avg'`` averages saved models after
                ``burn_in``; ``'final'`` uses the last epoch; an integer uses
                that epoch's model.
            burn_in (int): Number of initial epochs excluded when
                ``model='avg'``.
            alpha (float or None): If supplied with ``model='avg'``, also
                return the ``alpha / 2`` and ``1 - alpha / 2`` quantiles of
                predictions across retained epochs. These are heuristic
                epoch-stability bands, not sampling confidence intervals.

        Returns:
            tuple: Predictions ``(h(B), g(A))``. With averaged predictions
            and non-``None`` ``alpha``, the tuple additionally contains the
            lower and upper epoch quantiles for ``h`` followed by those for
            ``g``.
        """
        # real device object
        DEVICE = self.device

        # ensure inputs are tensors on the same device as loaded models
        B_dev = _as_cpu_matrix(B, "B").to(DEVICE)
        A_dev = _as_cpu_matrix(A, "A").to(DEVICE)

        if model == 'avg':
            pred_h = np.array([
                torch.load(os.path.join(self.model_dir, f"h_epoch{i}"),
                        map_location=DEVICE, weights_only=False
                ).to(DEVICE).eval()(B_dev).detach().cpu().numpy()
                for i in range(burn_in, self.n_epochs)
            ])
            pred_g = np.array([
                torch.load(os.path.join(self.model_dir, f"g_epoch{i}"),
                        map_location=DEVICE, weights_only=False
                ).to(DEVICE).eval()(A_dev).detach().cpu().numpy()
                for i in range(burn_in, self.n_epochs)
            ])
            mean_h = np.mean(pred_h, axis=0)
            mean_g = np.mean(pred_g, axis=0)
            if alpha is None:
                return mean_h, mean_g
            return (
                mean_h, mean_g,
                np.percentile(pred_h, 100 * alpha / 2, axis=0),
                np.percentile(pred_h, 100 * (1 - alpha / 2), axis=0),
                np.percentile(pred_g, 100 * alpha / 2, axis=0),
                np.percentile(pred_g, 100 * (1 - alpha / 2), axis=0),
            )

        if model == 'final':
            return (
                torch.load(os.path.join(self.model_dir, f"h_epoch{self.n_epochs-1}"),
                        map_location=DEVICE, weights_only=False
                ).to(DEVICE).eval()(B_dev).detach().cpu().numpy(),
                torch.load(os.path.join(self.model_dir, f"g_epoch{self.n_epochs-1}"),
                        map_location=DEVICE, weights_only=False
                ).to(DEVICE).eval()(A_dev).detach().cpu().numpy(),
            )

        if isinstance(model, int):
            return (
                torch.load(os.path.join(self.model_dir, f"h_epoch{model}"),
                        map_location=DEVICE, weights_only=False
                ).to(DEVICE).eval()(B_dev).detach().cpu().numpy(),
                torch.load(os.path.join(self.model_dir, f"g_epoch{model}"),
                        map_location=DEVICE, weights_only=False
                ).to(DEVICE).eval()(A_dev).detach().cpu().numpy(),
            )

        raise ValueError(f"Unknown model option: {model!r}")



class _BaseSupLossAGMM2(_BaseAGMM2):
    """
    Base class for joint estimation of nested NPIV models with supervised loss.
    """
    def fit(self, A, B, C, D, Y, W=None,
            learner_l2=1e-3, adversary_l2=1e-4, learner_norm_reg=1e-12,
            learner_lr=1e-3, adversary_lr=1e-3,
            n_epochs=100, bs=100,
            train_learner_every=1, train_adversary_every=1,
            warm_start=False, model_dir='.', device=None,
            verbose=0, subsetted=False,
            subset_ind1=None, subset_ind2=None):
        """
        Fit the joint adversarial estimator.

        Parameters:
            A (array-like): Covariates for the first bridge ``g``.
            B (array-like): Covariates for the second bridge ``h``.
            C (array-like): Instruments for the moment with residual
                ``h(B) - W * g(A)``.
            D (array-like): Instruments for the moment with residual
                ``g(A) - Y``.
            Y (array-like): One outcome value per observation.
            W (array-like or None): Observation-level multiplier in
                ``h(B) - W * g(A)``. ``None`` uses a vector of ones.
            learner_l2 (float): Weight decay for multidimensional learner
                network parameters; biases and one-dimensional parameters are
                excluded.
            adversary_l2 (float): Weight decay for multidimensional adversary
                network parameters; biases and one-dimensional parameters are
                excluded.
            learner_norm_reg (float): Common empirical-L2 penalty on the
                learner outputs ``g(A)`` and ``h(B)``.
            learner_lr (float): Learner learning rate.
            adversary_lr (float): Adversary learning rate.
            n_epochs (int): Number of training epochs.
            bs (int): Batch size.
            train_learner_every (int): Frequency for learner updates.
            train_adversary_every (int): Frequency for adversary updates.
            warm_start (bool): Whether to keep current network weights before training.
            model_dir (str): Directory for saved model checkpoints.
            device (torch.device or str or None): Device used for tensor computation.
            verbose (int): Verbosity level.
            subsetted (bool): Whether to estimate the two moment equations on
                stage-specific subsets.
            subset_ind1 (array-like or None): Nonempty binary indicator for
                the ``g(A) - Y`` moment. Required when ``subsetted=True``.
            subset_ind2 (array-like or None): Nonempty binary indicator for
                the ``h(B) - W * g(A)`` moment. If omitted, the complement of
                ``subset_ind1`` is used. Explicit indicators need not form a
                partition.

        Returns:
            AGMM2L2: The fitted estimator.
        """
        if subsetted:
            A, B, C, D, Y, W, subset_ind1, subset_ind2 = \
                self._pretrain(A, B, C, D, Y, W,
                                learner_l2, adversary_l2, learner_norm_reg,
                                learner_lr, adversary_lr,
                                n_epochs, bs,
                                train_learner_every, train_adversary_every,
                                warm_start, model_dir, device, verbose,
                                subsetted=True,
                                subset_ind1=subset_ind1,
                                subset_ind2=subset_ind2)
        else:
            A, B, C, D, Y, W = self._pretrain(
                A, B, C, D, Y, W,
                learner_l2, adversary_l2, learner_norm_reg,
                learner_lr, adversary_lr,
                n_epochs, bs,
                train_learner_every, train_adversary_every,
                warm_start, model_dir, device, verbose
            )

        for epoch in range(n_epochs):

            if verbose:
                print(f"Epoch # {epoch}")

            for it, batch in enumerate(self.train_dl):

                data = tuple(x.to(self.device, non_blocking=(self.device.type == "cuda")) for x in batch)
                if subsetted:
                    Ab, Bb, Cb, Db, Yb, Wb, subset_ind1b, subset_ind2b = data
                else:
                    Ab, Bb, Cb, Db, Yb, Wb = data

                # Learner update
                if it % train_learner_every == 0:
                    # Set models to training mode
                    self.learnerh.train(); self.learnerg.train()

                    # Forward passes
                    hat_g = self.learnerg(Ab); hat_h = self.learnerh(Bb)
                    hat_f_ = self.adversary1(Db) * (subset_ind1b if subsetted else 1)
                    hat_f  = self.adversary2(Cb) * (subset_ind2b if subsetted else 1)

                    # Calculate losses for each learner
                    G_loss, H_loss = _learner_losses(
                        hat_g, hat_h, hat_f_, hat_f, Yb, Wb,
                        self.stage1_scale_, self.stage2_scale_,
                        learner_norm_reg
                    )

                    # Backpropagate and update for learnerg
                    self.optimizerg.zero_grad(); G_loss.backward(retain_graph=True); # Retain graph for subsequent use in H_loss
                    self.optimizerg.step(); self.learnerg.eval()
                    # Backpropagate and update for learnerh
                    self.optimizerh.zero_grad(); H_loss.backward();
                    self.optimizerh.step(); self.learnerh.eval()

                # Adversary update
                if it % train_adversary_every == 0:
                    # Set models to training mode
                    self.adversary1.train(); self.adversary2.train()

                    # Since models are being reused, ensure data is consistent or re-compute if necessary
                    hat_g = self.learnerg(Ab); hat_h = self.learnerh(Bb)
                    hat_f_ = self.adversary1(Db) * (subset_ind1b if subsetted else 1)
                    hat_f  = self.adversary2(Cb) * (subset_ind2b if subsetted else 1)

                    # Calculate losses for each adversary
                    F_loss, F__loss = _adversary_losses(
                        hat_g, hat_h, hat_f_, hat_f, Yb, Wb,
                        self.stage1_scale_, self.stage2_scale_
                    )

                     # Update adversary
                    self.optimizerf.zero_grad();  F_loss.backward(retain_graph=True);
                    self.optimizerf.step();  self.adversary2 .eval()
                    self.optimizerf_.zero_grad(); F__loss.backward();
                    self.optimizerf_.step(); self.adversary1 .eval()

            # save epoch models
            torch.save(self.learnerg, os.path.join(self.model_dir, f"g_epoch{epoch}"))
            torch.save(self.learnerh, os.path.join(self.model_dir, f"h_epoch{epoch}"))
        return self


class _BaseSupLossAGMM2L2(_BaseAGMM2):
    """
    Base class for joint estimation with empirical-L2 output regularization.
    """
    fit = _BaseSupLossAGMM2.fit  # identical training loop


class AGMM2L2(_BaseSupLossAGMM2L2):
    """
    Joint adversarial estimator for nested NPIV with empirical-L2 regularization.

    The four networks approximately solve the empirical minimax problem

    ``E_n[2(g(A)-Y)f1(D)-f1(D)^2]``
    ``+ E_n[2(h(B)-Wg(A))f2(C)-f2(C)^2]``
    ``+ mu E_n[g(A)^2+h(B)^2]``,

    where ``mu`` is ``learner_norm_reg`` in :meth:`fit`. The parameter
    weight-decay arguments in :meth:`fit` are separate optimization
    regularizers. When stage subsets are used, each moment term is normalized
    by its own subset size while both learner-output penalties remain
    full-sample averages.

    Parameters:
        learnerh (torch.nn.Module): Network mapping ``B`` to the second bridge
            ``h(B)``.
        learnerg (torch.nn.Module): Network mapping ``A`` to the first bridge
            ``g(A)``.
        adversary1 (torch.nn.Module): Critic network mapping the first-moment
            instruments ``D`` to ``f1(D)``.
        adversary2 (torch.nn.Module): Critic network mapping the second-moment
            instruments ``C`` to ``f2(C)``.
    """
    def __init__(self, learnerh, learnerg, adversary1, adversary2):
        self.learnerh = learnerh
        self.learnerg = learnerg
        self.adversary1 = adversary1
        self.adversary2 = adversary2
        self.skip_list = []  # which adversary parameters to exclude from weight decay
