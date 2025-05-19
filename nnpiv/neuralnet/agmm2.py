'''
This module provides implementations of joint estimation for nested nonparametric instrumental variables (NPIV) using neural networks.

Classes:
    _BaseAGMM2: Base class for joint estimation of nested NPIV models.
    _BaseSupLossAGMM2: Base class for joint estimation of nested NPIV models with supervised loss.
    _BaseSupLossAGMM2L2: Base class for joint estimation of nested NPIV models with L2 regularization.
    AGMM2L2: Adversarial Generalized Method of Moments estimator for nested NPIV with L2 regularization.
'''
# Copyright (c) Microsoft Corporation.
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


def add_weight_decay(net, l2_value, skip_list=()):
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

        if add_sample_inds:
            sample_inds = torch.arange(Y.shape[0]).clone().detach()
            self.train_ds = TensorDataset(A, B, C, D, Y, W, sample_inds) if not subsetted else \
                TensorDataset(A, B, C, D, Y, W, sample_inds, subset_ind1, subset_ind2)
        else:
            self.train_ds = TensorDataset(A, B, C, D, Y, W) if not subsetted else \
                TensorDataset(A, B, C, D, Y, W, subset_ind1, subset_ind2)
        self.train_dl = DataLoader(self.train_ds, batch_size=bs, shuffle=True)

        # Move networks to device
        self.learnerh = self.learnerh.to(device)
        self.learnerg = self.learnerg.to(device)
        self.adversary1 = self.adversary1.to(device)
        self.adversary2 = self.adversary2.to(device)

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
        Parameters
        ----------
        B, A : endogenous vars for second and first stage
        model : one of ('avg', 'final') or an integer epoch index
        burn_in : discard the first burn_in epochs when averaging
        alpha : confidence interval level (if not None)
        """
        if model == 'avg':
            pred_h = np.array([
                torch.load(os.path.join(self.model_dir, f"h_epoch{i}"))(B)
                     .cpu().data.numpy()
                for i in range(burn_in, self.n_epochs)
            ])
            pred_g = np.array([
                torch.load(os.path.join(self.model_dir, f"g_epoch{i}"))(A)
                     .cpu().data.numpy()
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
                np.percentile(pred_g, 100 * (1 - alpha / 2), axis=0)
            )
        if model == 'final':
            return (
                torch.load(os.path.join(self.model_dir, f"h_epoch{self.n_epochs-1}"))(B)
                     .cpu().data.numpy(),
                torch.load(os.path.join(self.model_dir, f"g_epoch{self.n_epochs-1}"))(A)
                     .cpu().data.numpy()
            )
        if isinstance(model, int):
            return (
                torch.load(os.path.join(self.model_dir, f"h_epoch{model}"))(B)
                     .cpu().data.numpy(),
                torch.load(os.path.join(self.model_dir, f"g_epoch{model}"))(A)
                     .cpu().data.numpy()
            )


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
        Fit AGMM model with supervised loss.

        Parameters
        ----------
        learner_l2 : L2 on parameters of learners
        adversary_l2 : L2 on parameters of adversaries
        learner_norm_reg : ridge penalty on learner outputs
        (others as in base)
        """
        W = torch.ones(Y.shape[0]) if W is None else W
        if subsetted:
            if subset_ind1 is None:
                raise ValueError("subset_ind1 must be provided when subsetted is True")
            if len(subset_ind1) != len(Y):
                raise ValueError("subset_ind1 must have same length as Y")
            subset_ind2 = 1 - subset_ind1 if subset_ind2 is None else subset_ind2

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

                data = tuple(x.to(device) for x in batch)
                if subsetted:
                    Ab, Bb, Cb, Db, Yb, Wb, subset_ind1, subset_ind2 = data
                else:
                    Ab, Bb, Cb, Db, Yb, Wb = data

                # Learner update
                if it % train_learner_every == 0:
                    # Set models to training mode
                    self.learnerh.train(); self.learnerg.train()

                    # Forward passes
                    hat_g = self.learnerg(Ab); hat_h = self.learnerh(Bb)
                    hat_f_ = self.adversary1(Db) * (subset_ind1 if subsetted else 1)
                    hat_f  = self.adversary2(Cb) * (subset_ind2 if subsetted else 1)

                    # Calculate losses for each learner
                    G_loss = (torch.mean(2*(hat_g-Yb)*hat_f_) +
                              torch.mean(2*(hat_h-hat_g)*hat_f) +
                              learner_norm_reg * torch.mean(hat_g**2))
                    H_loss = (torch.mean(2*(hat_h-hat_g*Wb)*hat_f) +
                              learner_norm_reg * torch.mean(hat_h**2))

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
                    hat_f_ = self.adversary1(Db) * (subset_ind1 if subsetted else 1)
                    hat_f  = self.adversary2(Cb) * (subset_ind2 if subsetted else 1)

                    # Calculate losses for each adversary
                    F_loss  = -torch.mean(2*(hat_h-hat_g*Wb)*hat_f) + torch.mean(hat_f**2)
                    F__loss = -torch.mean(2*(hat_g-Yb)*hat_f_) + torch.mean(hat_f_**2)

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
    Base class for joint estimation of nested NPIV models with L2 regularization on outputs.
    """
    fit = _BaseSupLossAGMM2.fit  # identical training loop


class AGMM2L2(_BaseSupLossAGMM2L2):
    """
    Adversarial Generalized Method of Moments estimator for nested NPIV with L2 regularization.

    Parameters:
        learnerh : torch.nn.Module for second-stage learner
        learnerg : torch.nn.Module for first-stage learner
        adversary1 : torch.nn.Module for first-stage adversary
        adversary2 : torch.nn.Module for second-stage adversary
    """
    def __init__(self, learnerh, learnerg, adversary1, adversary2):
        self.learnerh = learnerh
        self.learnerg = learnerg
        self.adversary1 = adversary1
        self.adversary2 = adversary2
        self.skip_list = []  # which adversary parameters to exclude from weight decay
