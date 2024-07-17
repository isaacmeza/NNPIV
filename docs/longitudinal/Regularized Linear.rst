.. _regularized-linear-function-spaces:

Regularized Linear Function Spaces (:math:`\ell_2-\ell_2`)
==========================================================

We continue to work with linear function classes, but in contrast with the previous section :ref:`sparse-linear-function-spaces`, the learner and adversary function spaces are equipped with the :math:`\ell_2`-norm. This difference will translate to modifying :math:`R_{\min}` and :math:`R_{\max}` in Proposition 17, given that the dual norm for the spaces :math:`\Theta` and :math:`W` in this setting is again the :math:`\ell_2`-norm. In particular, for the sequential estimators we will take

.. math::

    R_{\min}(\alpha) = \frac{1}{2}\|\alpha\|_2^2 \;,\quad R_{\max}(\omega_1) = \frac{1}{2}\|\omega_1\|_2^2

and

.. math::

    R_{\min}(\alpha, \beta) = \frac{1}{2}\|\alpha\|_2^2 + \frac{1}{2}\|\beta\|_2^2 \;,\quad R_{\max}(\omega_1, \omega_2) = \frac{1}{2}\|\omega_1\|_2^2 + \frac{1}{2}\|\omega_2\|_2^2

for the joint estimator, since these regularizers are 1-strongly convex in their respective domains. In these cases, the updates will be essentially optimistic gradient descent. In the following subsections we only state the corresponding Lemmas analogous to section :ref:`sparse-linear-function-spaces`.

.. _estimator-1-l2:

Estimator 1 
-----------

.. admonition:: FTRL Iterates for Estimator 1
    :class: lemma
    :name: regularized-l2-est1

    Consider the iterates for :math:`t=1,\ldots, T`:

    .. math::
        :nowrap:

        \begin{aligned}
        \tilde{\alpha}_{t+1} &= \tilde{\alpha}_{t}-2\eta\left(-2\mathbb{E}_n[ac'^{\top}]\theta_{1,t} + 2\mu'\tilde{\alpha}_{t}\right) + \eta\left(-2\mathbb{E}_n[ac'^{\top}]\theta_{1,t-1} + 2\mu'\tilde{\alpha}_{t-1}\right) \\
        \alpha_{t+1} &= \tilde{\alpha}_{t+1}\min\left\{1, \frac{V_1}{\| \tilde{\alpha}_{t+1}\|_2}\right\},
        \end{aligned}

    .. math::
        :nowrap:

        \begin{aligned}
        \tilde{\theta}_{1,t+1} &= \tilde{\theta}_{1,t}+2\eta\left(2\mathbb{E}_n[c'y]-2\mathbb{E}_n[c'a^\top]\alpha_{t} - 2\mathbb{E}_n[c'c'^{\top}]\tilde\theta_{1t}\right) \\
        &\qquad -\eta\left(2\mathbb{E}_n[c'y]-2\mathbb{E}_n[c'a^\top]\alpha_{t-1} - 2\mathbb{E}_n[c'c'^{\top}]\tilde\theta_{1,t-1}\right) \\
        \theta_{1,t+1} &= \tilde{\theta_1}_{t+1}\min\left\{1, \frac{U_1}{\| \tilde{\theta_1}_{t+1}\|_2}\right\}
        \end{aligned}

    with :math:`\tilde{\alpha}_{-1} = \tilde{\alpha}_{0}=\tilde{\theta}_{1,-1}=\tilde{\theta}_{1,0} = 0`, and :math:`\eta = \frac{1}{8\|\mathbb{E}_n[aa^\top]\|_2}`.

    Then, :math:`\bar{\alpha} = \frac{1}{T}\sum_{t=1}^{T}\alpha_t`, is a :math:`O(T^{-1})`-approximate solution to

    .. math::

        \operatorname{argmin}_{\|\alpha\|_2 \leq V_1} \max _{\|\theta_1\|_1 \leq U_1} 2\langle\mathbb{E}_n[(y-\langle\alpha, a\rangle)c'],\theta_1\rangle -\mathbb{E}_n[\langle c',\theta_1\rangle^2]+\mu'\|\alpha\|_2^2

.. autosummary::
   :toctree: _autosummary
   :template: class.rst

   sparse_l2_l2.sparse_l2vsl2

.. _estimator-2-l2:

Estimator 2 
-----------

.. admonition:: FTRL Iterates for Estimator 2
    :class: lemma
    :name: regularized-l2-est2

    Consider the iterates for :math:`t=1,\ldots, T`:

    .. math::
        :nowrap:

        \begin{aligned}
        \tilde{\alpha}_{t+1} &= \tilde{\alpha}_{t}-2\eta\left(-2\mathbb{E}_n[ac'^{\top}]\theta_{1,t} + 2\mu'\mathbb{E}_n[aa^\top]\tilde{\alpha}_{t}\right) + \eta\left(-2\mathbb{E}_n[ac'^{\top}]\theta_{1,t-1} + 2\mu'\mathbb{E}_n[aa^\top]\tilde{\alpha}_{t-1}\right) \\
        \alpha_{t+1} &= \tilde{\alpha}_{t+1}\min\left\{1, \frac{V_1}{\| \tilde{\alpha}_{t+1}\|_2}\right\},
        \end{aligned}

    .. math::
        :nowrap:

        \begin{aligned}
        \tilde{\theta}_{1,t+1} &= \tilde{\theta}_{1,t}+2\eta\left(2\mathbb{E}_n[c'y]-2\mathbb{E}_n[c'a^\top]\alpha_{t} - 2\mathbb{E}_n[c'c'^{\top}]\tilde\theta_{1t}\right) \\
        &\qquad -\eta\left(2\mathbb{E}_n[c'y]-2\mathbb{E}_n[c'a^\top]\alpha_{t-1} - 2\mathbb{E}_n[c'c'^{\top}]\tilde\theta_{1,t-1}\right)\\
        \theta_{1,t+1} &= \tilde{\theta_1}_{t+1}\min\left\{1, \frac{U_1}{\| \tilde{\theta_1}_{t+1}\|_2}\right\}
        \end{aligned}

    with :math:`\tilde{\alpha}_{-1} = \tilde{\alpha}_{0}=\tilde{\theta}_{1,-1}=\tilde{\theta}_{1,0} = 0`, and :math:`\eta = \frac{1}{8\|\mathbb{E}_n[aa^\top]\|_2}`.

    Then, :math:`\bar{\alpha} = \frac{1}{T}\sum_{t=1}^{T}\alpha_t`, is a :math:`O(T^{-1})`-approximate solution to

    .. math::

        \operatorname{argmin}_{\|\alpha\|_2 \leq V_1} \max _{\|\theta_1\|_1 \leq U_1} 2\langle\mathbb{E}_n[(y-\langle\alpha, a\rangle)c'],\theta_1\rangle -\mathbb{E}_n[\langle c',\theta_1\rangle^2]+\mu'\mathbb{E}_n[\langle a,\alpha\rangle^2]

.. autosummary::
   :toctree: _autosummary
   :template: class.rst

   sparse_l2_l2.sparse_ridge_l2vsl2

.. _estimator-3-ridge-l2:


Estimator 3 - (Ridge)
-------------------

.. admonition:: FTRL Iterates for Estimator 3 (Ridge)
    :class: lemma
    :name: regularized-l2-est3-ridge

    Consider the iterates for :math:`t=1,\ldots, T`:

    .. math::
        :nowrap:

        \begin{aligned}
        \tilde{\alpha}_{t+1} &= \tilde{\alpha}_{t} -2\eta\left(-2\mathbb{E}_n[ac'^{\top}]\theta_{1,t} + 2\mu'\mathbb{E}_n[aa^\top]\tilde\alpha_{t}+ 2\mathbb{E}_n[ac^\top]\theta_{2,t}\right) \\
        &\qquad +\eta\left(-2\mathbb{E}_n[ac'^{\top}]\theta_{1,t-1} + 2\mu'\mathbb{E}_n[aa^\top]\tilde\alpha_{t-1}+ 2\mathbb{E}_n[ac^\top]\theta_{2,t-1}\right) \\
        \alpha_{t+1} &= \tilde{\alpha}_{t+1}\min\left\{1, \frac{V_1}{\| \tilde{\alpha}_{t+1}\|_2}\right\}, \\
        \tilde{\beta}_{t+1} &= \tilde{\beta}_{t}-2\eta\left(-2\mathbb{E}_n[bc^\top]\theta_{2,t}+2\mu\mathbb{E}_n[bb^\top]\tilde\beta_{t}\right)+\eta\left(-2\mathbb{E}_n[bc^\top]\theta_{2,t-1}+2\mu\mathbb{E}_n[bb^\top]\tilde\beta_{t-1}\right) \\
        \beta_{t+1} &= \tilde{\beta}_{t+1}\min\left\{1, \frac{V_2}{\| \tilde{\beta}_{t+1}\|_2}\right\},
        \end{aligned}

    .. math::
        :nowrap:

        \begin{aligned}
        \tilde{\theta}_{1,t+1} &= \tilde{\theta}_{1,t}+2\eta\left(2\mathbb{E}_n[c'y]-2\mathbb{E}_n[c'a^\top]\alpha_{t} - 2\mathbb{E}_n[c'c'^{\top}]\tilde\theta_{1,t}\right) \\
        &\qquad -\eta\left(2\mathbb{E}_n[c'y]-2\mathbb{E}_n[c'a^\top]\alpha_{t-1} - 2\mathbb{E}_n[c'c'^{\top}]\tilde\theta_{1,t-1}\right)\\
        \tilde\theta_{1,t+1} &= \tilde{\theta}_{1,t+1}\min\left\{1, \frac{U_1}{\| \tilde{\theta}_{1,t+1}\|_2}\right\}, \\
        \tilde{\theta}_{2,t+1} &= \tilde{\theta}_{2,t}+2\eta\left(2\mathbb{E}_n[ca^\top]\alpha_{t}-2\mathbb{E}_n[cb^\top]\beta_{t} - 2\mathbb{E}_n[cc^{\top}]\tilde\theta_{2,t}\right) \\
        &\qquad -\eta\left(2\mathbb{E}_n[ca^\top]\alpha_{t-1}-2\mathbb{E}_n[cb^\top]\beta_{t-1} - 2\mathbb{E}_n[cc^{\top}]\tilde\theta_{2,t-1}\right)\\
        \tilde\theta_{2,t+1} &= \tilde{\theta}_{2,t+1}\min\left\{1, \frac{U_2}{\| \tilde{\theta}_{2,t+1}\|_2}\right\}
        \end{aligned}

    with :math:`\tilde{\alpha}_{-1} = \tilde{\alpha}_{0} = \tilde{\beta}_{-1} = \tilde{\beta}_{0}= \theta_{1,-1}=\theta_{1,0} = \theta_{2,-1}=\theta_{2,0}= 0`, and :math:`\eta = [16\max\left\{\left\|\mathbb{E}_n[ac'^\top]\right\|_2, \left\|\mathbb{E}_n[ac^\top]\right\|_2, \left\| \mathbb{E}_n[bc^\top]\right\|_2\right\}]^{-1}`.

    Then,

    .. math::
        :nowrap:

        \begin{aligned}
        \bar{\alpha} = \frac{1}{T}\sum_{t=1}^{T}\alpha_{t}\,,\quad \bar{\beta} = \frac{1}{T}\sum_{t=1}^{T}\beta_{t}
        \end{aligned}

    are a :math:`O(T^{-1})`-approximate solution for

    .. math::

        \underset{\|\beta\|_2 \leq V_2}{\operatorname{argmin}_{\|\alpha\|_2 \leq V_1}} \underset{\|\theta_2\|_2\leq U_2}{\max _{\|\theta_1\|_2\leq U_1}} \left( 2\langle\mathbb{E}_n[(y-\langle\alpha, a\rangle)c'],\theta_1\rangle -\mathbb{E}_n[\langle c',\theta_1\rangle^2]+\mu'\mathbb{E}_n[\langle a,\alpha\rangle^2] \right. \\
        \left. + 2\langle\mathbb{E}_n[(\langle\alpha, a\rangle-\langle\beta, b\rangle)c],\theta_2\rangle -\mathbb{E}_n[\langle c,\theta_2\rangle^2]+\mu\mathbb{E}_n[\langle b,\beta\rangle^2] \right)

.. autosummary::
   :toctree: _autosummary
   :template: class.rst

   sparse2_l2_l2.sparse2_ridge_l2vsl2

.. _estimator-3-l2:

Estimator 3 - (:math:`\ell_2`-norm)
-----------------------------------

.. admonition:: FTRL Iterates for Estimator 3 - (:math:`\ell_2`-norm)
    :class: lemma
    :name: regularized-l2-est3

    Consider the iterates for :math:`t=1,\ldots, T`:

    .. math::
        :nowrap:
    
        \begin{aligned}
        \tilde{\alpha}_{t+1} &= \tilde{\alpha}_{t} -2\eta\left(-2\mathbb{E}_n[ac'^{\top}]\theta_{1,t} + 2\mu'\tilde\alpha_{t}+ 2\mathbb{E}_n[ac^\top]\theta_{2,t}\right) \\
        &\qquad +\eta\left(-2\mathbb{E}_n[ac'^{\top}]\theta_{1,t-1} + 2\mu'\tilde\alpha_{t-1}+ 2\mathbb{E}_n[ac^\top]\theta_{2,t-1}\right) \\
        \alpha_{t+1} &= \tilde{\alpha}_{t+1}\min\left\{1, \frac{V_1}{\| \tilde{\alpha}_{t+1}\|_2}\right\},\\
        \tilde{\beta}_{t+1} &= \tilde{\beta}_{t}-2\eta\left(-2\mathbb{E}_n[bc^\top]\theta_{2,t}+2\mu\tilde\beta_{t}\right)+\eta\left(-2\mathbb{E}_n[bc^\top]\theta_{2,t-1}+2\mu\tilde\beta_{t-1}\right) \\
        \beta_{t+1} &= \tilde{\beta}_{t+1}\min\left\{1, \frac{V_2}{\| \tilde{\beta}_{t+1}\|_2}\right\},
        \end{aligned}


    .. math::
        :nowrap:

        \begin{aligned}
        \tilde{\theta}_{1,t+1} &= \tilde{\theta}_{1,t}+2\eta\left(2\mathbb{E}_n[c'y]-2\mathbb{E}_n[c'a^\top]\alpha_{t} - 2\mathbb{E}_n[c'c'^{\top}]\tilde\theta_{1,t}\right) \\
        &\qquad -\eta\left(2\mathbb{E}_n[c'y]-2\mathbb{E}_n[c'a^\top]\alpha_{t-1} - 2\mathbb{E}_n[c'c'^{\top}]\tilde\theta_{1,t-1}\right)\\
        \tilde\theta_{1,t+1} &= \tilde{\theta}_{1,t+1}\min\left\{1, \frac{U_1}{\| \tilde{\theta}_{1,t+1}\|_2}\right\}\\
        \tilde{\theta}_{2,t+1} &= \tilde{\theta}_{2,t}+2\eta\left(2\mathbb{E}_n[ca^\top]\alpha_{t}-2\mathbb{E}_n[cb^\top]\beta_{t} - 2\mathbb{E}_n[cc^{\top}]\tilde\theta_{2,t}\right) \\
        &\qquad -\eta\left(2\mathbb{E}_n[ca^\top]\alpha_{t-1}-2\mathbb{E}_n[cb^\top]\beta_{t-1} - 2\mathbb{E}_n[cc^{\top}]\tilde\theta_{2,t-1}\right)\\
        \tilde\theta_{2,t+1} &= \tilde{\theta}_{2,t+1}\min\left\{1, \frac{U_2}{\| \tilde{\theta}_{2,t+1}\|_2}\right\}
        \end{aligned}

    with :math:`\tilde{\alpha}_{-1} = \tilde{\alpha}_{0} = \tilde{\beta}_{-1} = \tilde{\beta}_{0}= \theta_{1,-1}=\theta_{1,0} = \theta_{2,-1}=\theta_{2,0}= 0`, and :math:`\eta = [16\max\left\{\left\|\mathbb{E}_n[ac'^\top]\right\|_2, \left\|\mathbb{E}_n[ac^\top]\right\|_2, \left\| \mathbb{E}_n[bc^\top]\right\|_2\right\}]^{-1}`.

    Then,

    .. math::
        :nowrap:

        \begin{aligned}
        \bar{\alpha} = \frac{1}{T}\sum_{t=1}^{T}\alpha_{t}\,,\quad \bar{\beta} = \frac{1}{T}\sum_{t=1}^{T}\beta_{t}
        \end{aligned}

    are a :math:`O(T^{-1})`-approximate solution for

    .. math::

        \underset{\|\beta\|_2 \leq V_2}{\operatorname{argmin}_{\|\alpha\|_2 \leq V_1}} \underset{\|\theta_2\|_2\leq U_2}{\max _{\|\theta_1\|_2\leq U_1}} \left( 2\langle\mathbb{E}_n[(y-\langle\alpha, a\rangle)c'],\theta_1\rangle -\mathbb{E}_n[\langle c',\theta_1\rangle^2]+\mu'\|\alpha\|_2^2 \right. \\
        \left. + 2\langle\mathbb{E}_n[(\langle\alpha, a\rangle-\langle\beta, b\rangle)c],\theta_2\rangle -\mathbb{E}_n[\langle c,\theta_2\rangle^2]+\mu\|\beta\|_2^2 \right)

.. autosummary::
   :toctree: _autosummary
   :template: class.rst

   sparse2_l2_l2.sparse2_l2vsl2
