# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import numpy as np
import tempfile
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from .oadam import OAdam
from .rbflayer import RBF

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
    return [{'params': no_decay, 'weight_decay': 0.}, {'params': decay, 'weight_decay': l2_value}]


class _BaseAGMM2:

    def _pretrain(self, A, B, C, D, Y,
                  learner_l2, adversary_l2, adversary_norm_reg, learner_norm_reg,
                  learner_lr, adversary_lr, n_epochs, bs, train_learner_every, train_adversary_every,
                  warm_start, model_dir, device, verbose, add_sample_inds=False):
        """ Prepares the variables required to begin training.
        """
        self.verbose = verbose

        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        self.tempdir = tempfile.TemporaryDirectory(dir=model_dir)
        self.model_dir = self.tempdir.name

        self.n_epochs = n_epochs

        if add_sample_inds:
            sample_inds = torch.tensor(np.arange(Y.shape[0]))
            self.train_ds = TensorDataset(A, B, C, D, Y, sample_inds)
        else:
            self.train_ds = TensorDataset(A, B, C, D, Y)
        self.train_dl = DataLoader(self.train_ds, batch_size=bs, shuffle=True)

        self.learnerh = self.learnerh.to(device)
        self.learnerg = self.learnerg.to(device)
        self.adversary1 = self.adversary1.to(device)
        self.adversary2 = self.adversary2.to(device)

        if not warm_start:
            self.learnerh.apply(lambda m: (
                m.reset_parameters() if hasattr(m, 'reset_parameters') else None))
            self.learnerg.apply(lambda m: (
                m.reset_parameters() if hasattr(m, 'reset_parameters') else None))
            self.adversary1.apply(lambda m: (
                m.reset_parameters() if hasattr(m, 'reset_parameters') else None))
            self.adversary2.apply(lambda m: (
                m.reset_parameters() if hasattr(m, 'reset_parameters') else None))

        beta1 = 0.
        self.optimizerg = OAdam(add_weight_decay(self.learnerg, learner_l2),
                                lr=learner_lr, betas=(beta1, .01))
        self.optimizerh = OAdam(add_weight_decay(self.learnerh, learner_l2),
                                lr=learner_lr, betas=(beta1, .01))
        self.optimizerf = OAdam(add_weight_decay(
            self.adversary2, adversary_l2, skip_list=self.skip_list), lr=adversary_lr, betas=(beta1, .01))
        self.optimizerf_ = OAdam(add_weight_decay(
            self.adversary1, adversary_l2, skip_list=self.skip_list), lr=adversary_lr, betas=(beta1, .01))

        return A, B, C, D, Y

    def predict(self, B, A, model='avg', burn_in=0, alpha=None):
        """
        Parameters
        ----------
        B, A : endogenous vars for second and first stage
        model : one of ('avg', 'final'), whether to use an average of models or the final
        burn_in : discard the first "burn_in" epochs when doing averaging
        alpha : if not None but a float, then it also returns the a/2 and 1-a/2, percentile of
            the predictions across different epochs (proxy for a confidence interval)
        """
        if model == 'avg':
            pred_h = np.array([torch.load(os.path.join(self.model_dir,
                                                      "h_epoch{}".format(i)))(B).cpu().data.numpy()
                              for i in np.arange(burn_in, self.n_epochs)])
            pred_g = np.array([torch.load(os.path.join(self.model_dir,
                                                      "g_epoch{}".format(i)))(A).cpu().data.numpy()
                              for i in np.arange(burn_in, self.n_epochs)])
            if alpha is None:
                return np.mean(pred_h, axis=0), np.mean(pred_g, axis=0)
            else:
                return np.mean(pred_h, axis=0), np.mean(pred_g, axis=0), \
                    np.percentile(
                        pred_h, 100 * alpha / 2, axis=0), np.percentile(pred_h, 100 * (1 - alpha / 2), axis=0), \
                    np.percentile(
                        pred_g, 100 * alpha / 2, axis=0), np.percentile(pred_g, 100 * (1 - alpha / 2), axis=0)
        if model == 'final':
            return torch.load(os.path.join(self.model_dir,
                                           "h_epoch{}".format(self.n_epochs - 1)))(B).cpu().data.numpy(), \
                torch.load(os.path.join(self.model_dir,
                                        "g_epoch{}".format(self.n_epochs - 1)))(A).cpu().data.numpy()
        if isinstance(model, int):
            return torch.load(os.path.join(self.model_dir,
                                           "h_epoch{}".format(model)))(B).cpu().data.numpy(), \
                torch.load(os.path.join(self.model_dir,
                                        "g_epoch{}".format(model)))(A).cpu().data.numpy()


class _BaseSupLossAGMM2(_BaseAGMM2):

    def fit(self, A, B, C, D, Y,
            learner_l2=1e-3, adversary_l2=1e-4, adversary_norm_reg=1e-3, learner_norm_reg=1e-3,
            learner_lr=0.001, adversary_lr=0.001, n_epochs=100, bs=100, train_learner_every=1, train_adversary_every=1,
            warm_start=False, model_dir='.', device=None, verbose=0):
        """
        Parameters
        ----------
        A : endogenous vars for first stage
        B : endogenous vars for second stage
        C : instrument vars for second stage
        D : instrument vars for first stage
        Y : outcome
        learner_l2, adversary_l2 : l2_regularization of parameters of learner and adversary
        adversary_norm_reg : adversary norm regularization weight
        learner_norm_reg : learner norm regularization weight
        learner_lr : learning rate of the Adam optimizer for learner
        adversary_lr : learning rate of the Adam optimizer for adversary
        n_epochs : how many passes over the data
        bs : batch size
        train_learner_every : after how many training iterations of the adversary should we train the learner
        warm_start : if False then network parameters are initialized at the beginning, otherwise we start
            from their current weights
        model_dir : folder where to store the learned models after every epoch
        """

        A, B, C, D, Y = self._pretrain(A, B, C, D, Y,
                                 learner_l2, adversary_l2, adversary_norm_reg, learner_norm_reg,
                                 learner_lr, adversary_lr, n_epochs, bs, train_learner_every, train_adversary_every,
                                 warm_start, model_dir, device, verbose)

        for epoch in range(n_epochs):

            if self.verbose > 0:
                print("Epoch #", epoch, sep="")

            for it, (Ab, Bb, Cb, Db, Yb) in enumerate(self.train_dl):

                Ab, Bb, Cb, Db, Yb = map(lambda x: x.to(device), (Ab, Bb, Cb, Db, Yb))

                if (it % train_learner_every == 0):
                    # Set models to training mode
                    self.learnerh.train()
                    self.learnerg.train()

                    # Forward passes
                    hat_g = self.learnerg(Ab)
                    hat_h = self.learnerh(Bb)
                    hat_f_ = self.adversary1(Db)
                    hat_f = self.adversary2(Cb)

                    # Calculate losses for each learner
                    G_loss = torch.mean(2 * (hat_g - Yb) * hat_f_) + torch.mean(2 * (hat_h - hat_g) * hat_f)
                    G_loss += learner_norm_reg * 0
                    H_loss = torch.mean(2 * (hat_h - hat_g) * hat_f)
                    H_loss += learner_norm_reg * 0

                    # Backpropagate and update for learnerg
                    self.optimizerg.zero_grad()
                    G_loss.backward(retain_graph=True)  # Retain graph for subsequent use in H_loss
                    self.optimizerg.step()
                    self.learnerg.eval()

                    # Backpropagate and update for learnerh
                    self.optimizerh.zero_grad()
                    H_loss.backward()
                    self.optimizerh.step()
                    self.learnerh.eval()

                if (it % train_adversary_every == 0):
                    # Set models to training mode
                    self.adversary1.train()
                    self.adversary2.train()

                    # Since models are being reused, ensure data is consistent or re-compute if necessary
                    hat_g = self.learnerg(Ab)
                    hat_h = self.learnerh(Bb)
                    hat_f_ = self.adversary1(Db)
                    hat_f = self.adversary2(Cb)

                    # Calculate losses for each adversary
                    F_loss = - torch.mean(2 * (hat_h - hat_g) * hat_f) + torch.mean(hat_f**2)
                    F__loss = - torch.mean(2 * (hat_g - Yb) * hat_f_) + torch.mean(hat_f_**2)

                    # Update adversary2
                    self.optimizerf.zero_grad()
                    F_loss.backward(retain_graph=True)
                    self.optimizerf.step()
                    self.adversary2.eval()

                    # Update adversary1
                    self.optimizerf_.zero_grad()
                    F__loss.backward()
                    self.optimizerf_.step()
                    self.adversary1.eval()

            torch.save(self.learnerg, os.path.join(
                self.model_dir, "g_epoch{}".format(epoch)))
            torch.save(self.learnerh, os.path.join(
                self.model_dir, "h_epoch{}".format(epoch)))

        return self


class AGMM2(_BaseSupLossAGMM2):

    def __init__(self, learnerh, learnerg, adversary1, adversary2):
        """
        Parameters
        ----------
        learner : a pytorch neural net module
        adversary : a pytorch neural net module
        """
        self.learnerh = learnerh
        self.learnerg = learnerg
        self.adversary1 = adversary1
        self.adversary2 = adversary2

        # which adversary parameters to not ell2 penalize
        self.skip_list = []


class _BaseSupLossAGMM2L2(_BaseAGMM2):

    def fit(self, A, B, C, D, Y,
            learner_l2=1e-3, adversary_l2=1e-4, adversary_norm_reg=1e-3, learner_norm_reg=1e-3,
            learner_lr=0.001, adversary_lr=0.001, n_epochs=100, bs=100, train_learner_every=1, train_adversary_every=1,
            warm_start=False, model_dir='.', device=None, verbose=0):
        """
        Parameters
        ----------
        A : endogenous vars for first stage
        B : endogenous vars for second stage
        C : instrument vars for second stage
        D : instrument vars for first stage
        Y : outcome
        learner_l2, adversary_l2 : l2_regularization of parameters of learner and adversary
        adversary_norm_reg : adversary norm regularization weight
        learner_norm_reg : learner norm regularization weight
        learner_lr : learning rate of the Adam optimizer for learner
        adversary_lr : learning rate of the Adam optimizer for adversary
        n_epochs : how many passes over the data
        bs : batch size
        train_learner_every : after how many training iterations of the adversary should we train the learner
        warm_start : if False then network parameters are initialized at the beginning, otherwise we start
            from their current weights
        model_dir : folder where to store the learned models after every epoch
        """

        A, B, C, D, Y = self._pretrain(A, B, C, D, Y,
                                 learner_l2, adversary_l2, adversary_norm_reg, learner_norm_reg,
                                 learner_lr, adversary_lr, n_epochs, bs, train_learner_every, train_adversary_every,
                                 warm_start, model_dir, device, verbose)

        for epoch in range(n_epochs):

            if self.verbose > 0:
                print("Epoch #", epoch, sep="")

            for it, (Ab, Bb, Cb, Db, Yb) in enumerate(self.train_dl):

                Ab, Bb, Cb, Db, Yb = map(lambda x: x.to(device), (Ab, Bb, Cb, Db, Yb))

                if (it % train_learner_every == 0):
                    # Set models to training mode
                    self.learnerh.train()
                    self.learnerg.train()

                    # Forward passes
                    hat_g = self.learnerg(Ab)
                    hat_h = self.learnerh(Bb)
                    hat_f_ = self.adversary1(Db)
                    hat_f = self.adversary2(Cb)

                    # Calculate losses for each learner
                    G_loss = torch.mean(2 * (hat_g - Yb) * hat_f_) + torch.mean(2 * (hat_h - hat_g) * hat_f)
                    G_loss += learner_norm_reg * torch.mean(hat_g**2)
                    H_loss = torch.mean(2 * (hat_h - hat_g) * hat_f)
                    H_loss += learner_norm_reg * torch.mean(hat_h**2)

                    # Backpropagate and update for learnerg
                    self.optimizerg.zero_grad()
                    G_loss.backward(retain_graph=True)  # Retain graph for subsequent use in H_loss
                    self.optimizerg.step()
                    self.learnerg.eval()

                    # Backpropagate and update for learnerh
                    self.optimizerh.zero_grad()
                    H_loss.backward()
                    self.optimizerh.step()
                    self.learnerh.eval()

                if (it % train_adversary_every == 0):
                    # Set models to training mode
                    self.adversary1.train()
                    self.adversary2.train()

                    # Since models are being reused, ensure data is consistent or re-compute if necessary
                    hat_g = self.learnerg(Ab)
                    hat_h = self.learnerh(Bb)
                    hat_f_ = self.adversary1(Db)
                    hat_f = self.adversary2(Cb)

                    # Calculate losses for each adversary
                    F_loss = - torch.mean(2 * (hat_h - hat_g) * hat_f) + torch.mean(hat_f**2)
                    F__loss = - torch.mean(2 * (hat_g - Yb) * hat_f_) + torch.mean(hat_f_**2)

                    # Update adversary2
                    self.optimizerf.zero_grad()
                    F_loss.backward(retain_graph=True)
                    self.optimizerf.step()
                    self.adversary2.eval()

                    # Update adversary1
                    self.optimizerf_.zero_grad()
                    F__loss.backward()
                    self.optimizerf_.step()
                    self.adversary1.eval()

            torch.save(self.learnerg, os.path.join(
                self.model_dir, "g_epoch{}".format(epoch)))
            torch.save(self.learnerh, os.path.join(
                self.model_dir, "h_epoch{}".format(epoch)))

        return self
    

class AGMM2L2(_BaseSupLossAGMM2L2):

    def __init__(self, learnerh, learnerg, adversary1, adversary2):
        """
        Parameters
        ----------
        learner : a pytorch neural net module
        adversary : a pytorch neural net module
        """
        self.learnerh = learnerh
        self.learnerg = learnerg
        self.adversary1 = adversary1
        self.adversary2 = adversary2

        # which adversary parameters to not ell2 penalize
        self.skip_list = []