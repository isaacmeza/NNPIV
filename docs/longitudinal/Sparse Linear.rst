.. _sparse-linear-function-spaces:

Sparse Linear Function Spaces (:math:`\ell_1-\ell_1`)
=====================================================

In this section we address the high-dimensional case, where the function class is sparse linear, i.e. :math:`g(a) = \langle \alpha, a\rangle`, where :math:`\|\alpha\|_0 := \{j\in [p]\,|\,|\alpha_j|>0\} \leq s`. We will consider :math:`\ell_1` relaxations for the minimax optimization problem with :math:`\ell_1`-balls for the adversary. We remove the non-smoothness of the :math:`\ell_1` regularization by lifting the parameter :math:`\alpha` to a :math:`2p`-dimensional positive orthant. Consider two vectors :math:`\rho^{+}, \rho^{-} \geq 0` and then setting :math:`\alpha = \rho^{+} - \rho^{-}`, with :math:`\rho = \left(\rho^{+}; \rho^{-}\right)`. Observe that for any feasible :math:`\bar{\alpha}`, the solution :math:`\rho_i^{+} = \alpha_i 1\left\{\alpha_i > 0\right\}` and :math:`\rho_i^{-} = \alpha_i 1\left\{\alpha_i \leq 0\right\}` is still feasible and achieves the same objective, by the linearity of the loss function. Moreover, any solution :math:`\rho`, maps to a feasible solution :math:`\alpha` and thus the two optimization programs have the same optimal solutions.

Thus we will be solving an optimization problem over the :math:`2p`-dimensional simplex, and we will be using *Optimistic-Follow-the-Regularized-Leader* to find an :math:`\epsilon`-approximate solution. The approximate solutions of the minimax problems for all of our estimator will rely on the following proposition:


.. admonition:: Proposition 17 in `Dikkala et al. (2020) <https://arxiv.org/abs/2006.07201>`_
    :class: lemma

    Consider a minimax objective: :math:`\min _{\theta \in \Theta} \max _{w \in W} \ell(\theta, w)`. Suppose that :math:`\Theta, W` are convex sets and that :math:`\ell(\theta, w)` is convex in :math:`\theta` for every :math:`w` and concave in :math:`w` for any :math:`\theta`. Let :math:`\|\cdot\|_{\Theta}` and :math:`\|\cdot\|_W` be arbitrary norms in the corresponding spaces. Moreover, suppose that the following Lipschitzness properties are satisfied:

    .. math::

        \begin{aligned}
        & \forall \theta \in \Theta, w, w^{\prime} \in W: \left\|\nabla_\theta \ell(\theta, w) - \nabla_\theta \ell\left(\theta, w^{\prime}\right)\right\|_{\Theta, *} \leq L\left\|w - w^{\prime}\right\|_W \\
        & \forall w \in W, \theta, \theta^{\prime} \in \Theta: \left\|\nabla_w \ell(\theta, w) - \nabla_w \ell\left(\theta^{\prime}, w\right)\right\|_{W, *} \leq L\left\|\theta - \theta^{\prime}\right\|_W
        \end{aligned}

    where :math:`\|\cdot\|_{\Theta, *}` and :math:`\|\cdot\|_{W, *}` correspond to the dual norms of :math:`\|\cdot\|_{\Theta}` and :math:`\|\cdot\|_W`. Consider the algorithm where at each iteration each player updates their strategy based on:

    .. math::

        \begin{aligned}
        & \theta_{t+1} = \underset{\theta \in \Theta}{\arg \min } \theta^{\top}\left(\sum_{\tau \leq t} \nabla_\theta \ell\left(\theta_\tau, w_\tau\right) + \nabla_\theta \ell\left(\theta_t, w_t\right)\right) + \frac{1}{\eta} R_{\min }(\theta) \\
        & w_{t+1} = \underset{w \in W}{\arg \max } w^{\top}\left(\sum_{\tau \leq t} \nabla_w \ell\left(\theta_\tau, w_\tau\right) + \nabla_w \ell\left(\theta_t, w_t\right)\right) - \frac{1}{\eta} R_{\max }(w)
        \end{aligned}

    such that :math:`R_{\min }` is 1-strongly convex in the set :math:`\Theta` with respect to norm :math:`\|\cdot\|_{\Theta}` and :math:`R_{\max }` is 1-strongly convex in the set :math:`W` with respect to norm :math:`\|\cdot\|_W` and with any step-size :math:`\eta \leq \frac{1}{4 L}`. Then the parameters :math:`\bar{\theta} = \frac{1}{T} \sum_{t=1}^T \theta_t` and :math:`\bar{w} = \frac{1}{T} \sum_{t=1}^T w_t` correspond to an :math:`\frac{2 R_*}{\eta \cdot T}`-approximate equilibrium and hence :math:`\bar{\theta}` is a :math:`\frac{4 R_*}{\eta T}`-approximate solution to the minimax objective, where :math:`R` is defined as:

    .. math::

        R_* := \max \left\{\sup _{\theta \in \Theta} R_{\min }(\theta) - \inf _{\theta \in \Theta} R_{\min }(\theta), \sup _{w \in W} R_{\max }(w) - \inf _{w \in W} R_{\max }(w)\right\}


.. _estimator-1:

Estimator 1
-----------

The minimax problem is:

.. math::
    :label: minimax-sparse-est1

    \min_{\|\alpha\|_1 \leq V_1} \max _{\|\theta_1\|_1 \leq 1} L(\alpha, \theta) := \min_{\|\alpha\|_1 \leq V_1} \max _{\|\theta_1\|_1 \leq 1} 2\langle \mathbb{E}_n [(y - \langle \alpha, a \rangle)c'], \theta_1 \rangle - \mathbb{E}_n [\langle c', \theta_1 \rangle^2] + \mu' \|\alpha\|_1

which can be written as:

.. math::

    \min _{\rho \geq 0, \|\rho\|_1 \leq V_1} \max _{\omega_1 \geq 0, \|\omega_1\|_1 = 1} \ell(\rho, \omega_1)

where 

.. math::

    \ell(\rho, \omega_1) := 2 \omega_1^{\top} \mathbb{E}_n [u_1 y] - 2 \omega_1^{\top} \mathbb{E}_n [u_1 v_1^{\top}] \rho - \omega_1^{\top} \mathbb{E}_n [u_1 u_1^{\top}] \omega_1 + \mu' \sum_{i=1}^{2 p} \rho_i.

Moreover, :math:`v_1 = (a, -a)`, :math:`u_1 = (c', -c')`; and :math:`\theta_1 = \omega_1^{+} - \omega_1^{-}`, :math:`\alpha = \rho^+ - \rho^{-}`.


.. admonition:: FTRL iterates for Estimator 1
    :class: lemma
    :name: sparse-l1-l1-est1

    Consider the iterates for :math:`t=1,\ldots, T`:

    .. math::
        :nowrap:

        \begin{aligned}
        \tilde{\rho}_{t+1} &= \exp\left(-\frac{\eta}{V_1} \left\{\sum_{\tau \leq t} -2 \mathbb{E}_n [v_1 u_1^{\top}] \omega_{1\tau} -2 \mathbb{E}_n [v_1 u_1^{\top}] \omega_{1t} + (t+1)\mu' \right\} - 1\right) \\
        \rho_{t+1} &=  \tilde{\rho}_{t+1} \min\left\{1, \frac{V_1}{\| \tilde{\rho}_{t+1} \|_1}\right\},
        \end{aligned}

    .. math::
        :nowrap:

        \begin{aligned}
        \tilde{\omega}_{1,t+1} &= \tilde{\omega}_{1,t} \exp\bigg(2\eta\left\{2 \mathbb{E}_n [u_1 y] - 2 \mathbb{E}_n [u_1 v_1^{\top}] \rho_{t} - 2 \mathbb{E}_n [u_1 u_1^{\top}] \tilde{\omega}_{1,t}\right\} \\
        &\qquad -\eta\left\{2 \mathbb{E}_n [u_1 y] - 2 \mathbb{E}_n [u_1 v_1^{\top}] \rho_{t-1} - 2 \mathbb{E}_n [u_1 u_1^{\top}] \tilde{\omega}_{1,t-1}\right\}\bigg) \\
        \omega_{1,t+1} &= \frac{\tilde{\omega}_{1,t+1}}{\|\tilde{\omega}_{1,t+1}\|_1}
        \end{aligned}

    with :math:`\tilde{\rho}_{-1} = \tilde{\rho}_{0} = \frac{1}{e}`, :math:`\tilde{\omega}_{1,-1} = \tilde{\omega}_{1,0} = \frac{1}{2p}`, and :math:`\eta = \frac{1}{8 \|\mathbb{E}_n [v_1 u_1^{\top}]\|_\infty}`.
    
    Then, :math:`\bar{\rho} = \frac{1}{T}\sum_{t=1}^{T} \rho_t`, :math:`\bar{\alpha} = \bar{\rho}^{+} - \bar{\rho}^{-}` is a :math:`O(T^{-1})`-approximate solution for :eq:`minimax-sparse-est1`.
    
.. autosummary::
   :toctree: _autosummary
   :template: class.rst

   sparse_l1_l1.sparse_l1vsl1

**Proof**

The proof will match symbols with Proposition 17. Let 

.. math::

    \Theta = \{\rho \;|\; \rho \geq 0,\, \|\rho\|_1 \leq V_1\}\;,\quad W = \{\omega_1 \;|\; \omega_1 \geq 0, \|\omega_1\|_1 = 1\}

be the convex feasibility sets. Note that :math:`\ell` is convex in :math:`\rho` and concave in :math:`\omega_1`. Since

.. math::

    \begin{aligned}
    \nabla_{\rho} \ell(\rho, \omega_1) &= -2 \mathbb{E}_n [v_1 u_1^{\top}] \omega_1 + \mu' \\
    \nabla_{\omega_1} \ell(\rho, \omega_1) &= 2 \mathbb{E}_n [u_1 y] - 2 \mathbb{E}_n [u_1 v_1^{\top}] \rho - 2 \mathbb{E}_n [u_1 u_1^{\top}] \omega_1 
    \end{aligned}

the Lipschitzness property is satisfied with :math:`L = 2 \|\mathbb{E}_n [v_1 u_1^{\top}]\|_\infty`:

.. math::

    \begin{aligned}
    \left\|\nabla_\rho \ell(\rho, \omega_1) - \nabla_\rho \ell(\rho, \omega_1^{\prime})\right\|_{\infty} &= \left\|2 \mathbb{E}_n [v u^{\top}] (\omega_1 - \omega_1^{\prime})\right\|_{\infty} \leq 2 \|\mathbb{E}_n [v u^{\top}]\|_{\infty} \left\|\omega_1 - \omega_1^{\prime}\right\|_1 \\
    \left\|\nabla_{\omega_{1}} \ell(\rho, \omega_{1}) - \nabla_{\omega_{1}} \ell(\rho^{\prime}, \omega_{1})\right\|_{\infty} &= \left\|2 \mathbb{E}_n [u v^{\top}] (\rho - \rho^{\prime})\right\|_{\infty} \leq 2 \|\mathbb{E}_n [v u^{\top}]\|_{\infty} \left\|\rho - \rho^{\prime}\right\|_1
    \end{aligned}

Consider the entropic regularizers :math:`R_{min}(\rho) = V_1 \sum_{i=1}^{2p} \rho_i \log (\rho_i)`, and :math:`R_{max}(\omega_1) = \sum_{i=1}^{2p} \omega_{1i} \log (\omega_{1i})` which are :math:`1`-strongly convex in the spaces :math:`\Theta`, and :math:`W` respectively. Then, the iterates satisfy:

.. math::
    :nowrap:

    \begin{aligned}
    \rho_{t+1} &= \underset{\rho \geq 0, \|\rho\|_1 \leq V_1}{\operatorname{argmin}} \rho^{\top} \left(\sum_{\tau \leq t} \left\{-2 \mathbb{E}_n [v_1 u_1^{\top}] \omega_{1\tau} + \mu'\right\} - 2 \mathbb{E}_n [v_1 u_1^{\top}] \omega_{1t} + \mu'\right) + \frac{V_1}{\eta} \sum_{i=1}^{2p} \rho_i \log (\rho_i) \\
    \tilde{\rho}_{t+1} &= \exp\left(-\frac{\eta}{V_1} \left\{\sum_{\tau \leq t} -2 \mathbb{E}_n [v_1 u_1^{\top}] \omega_{1\tau} -2 \mathbb{E}_n [v_1 u_1^{\top}] \omega_{1t} + (t+1)\mu' \right\} - 1\right) \\
    \rho_{t+1} &=  \tilde{\rho}_{t+1} \min\left\{1, \frac{V_1}{\| \tilde{\rho}_{t+1} \|_1}\right\},
    \end{aligned}

.. math::
    :nowrap:

    \begin{aligned}
    \omega_{1,t+1} &= \underset{\|\omega_1\|_1 \leq 1}{\operatorname{argmax}} \omega_1^{\top} \left(\sum_{\tau \leq t} \left\{2 \mathbb{E}_n [u_1 y] - 2 \mathbb{E}_n [u_1 v_1^{\top}] \rho_{\tau} - 2 \mathbb{E}_n [u_1 u_1^{\top}] \omega_{1\tau} \right\} \right. \\
    &\qquad \left. + 2 \mathbb{E}_n [u_1 y] - 2 \mathbb{E}_n [u_1 v_1^{\top}] \rho_{t} - 2 \mathbb{E}_n [u_1 u_1^{\top}] \omega_{1t} \right) - \frac{1}{\eta} \sum_{i=1}^{2p} \omega_{1i} \log (\omega_{1i}) \\
    \tilde{\omega}_{1,t+1} &= \tilde{\omega}_{1,t} \exp\left(2\eta \left\{2 \mathbb{E}_n [u_1 y] - 2 \mathbb{E}_n [u_1 v_1^{\top}] \rho_{t} - 2 \mathbb{E}_n [u_1 u_1^{\top}] \tilde{\omega}_{1,t}\right\} \right. \\
    &\qquad \left. -\eta \left\{2 \mathbb{E}_n [u_1 y] - 2 \mathbb{E}_n [u_1 v_1^{\top}] \rho_{t-1} - 2 \mathbb{E}_n [u_1 u_1^{\top}] \tilde{\omega}_{1,t-1}\right\}\right) \\
    \omega_{1,t+1} &= \frac{\tilde{\omega}_{1,t+1}}{\|\tilde{\omega}_{1,t+1}\|_1}
    \end{aligned}


with :math:`\omega_{1,-1} = \omega_{1,0} = \frac{1}{2p}`. Therefore, by Proposition 17, the ensemble

.. math::

    \bar{\rho} = \frac{1}{T} \sum_{t=1}^T \rho_t

is :math:`O\left(\frac{1}{T}\right)`-approximate solution for the minimax objective.

.. admonition:: Duality Gap

   The ensembles :math:`\bar{\alpha}`, :math:`\bar{\theta_1}` can be thought of as primal and dual solutions and we can use the duality gap as a certificate for convergence of the algorithm.

    .. math::
        :nowrap:
    
        \begin{aligned}
        \text { Duality Gap } &:= \max _{\|\theta_1\|_1 \leq 1 } L(\bar{\alpha}, \theta_1) - \min _{\|\alpha\|_1 \leq V_1} L(\alpha, \bar{\theta_1}) \\
        &\leq \left(\mathbb{E}_n [(y - \langle \bar{\alpha}, a \rangle)c']\right)^{\top} \mathbb{E}_n [c' c'^{\top}]^{\dagger} \left(\mathbb{E}_n [(y - \langle \bar{\alpha}, a \rangle)c']\right) + \mu' \|\bar{\alpha}\|_1 \\
        &\quad - \left(\bar{\theta_1}^{\top} \mathbb{E}_n [c'y] + V_1 \left\{\mu' - 2 \|\mathbb{E}_n [a c'^{\top}] \bar{\theta_1}\|_\infty \right\}^{-} - \bar{\theta_1}^{\top} \mathbb{E}_n [c' c'^{\top}] \bar{\theta_1}\right) := \text{ tol}
        \end{aligned}

.. _estimator-2:

Estimator 2
-----------

The ridge estimator takes the form:

.. math::
    :label: minimax-sparse-est2

    \hat{\alpha} := \operatorname{argmin}_{\|\alpha\|_1 \leq V_1} \max _{\|\theta_1\|_1 \leq 1} 2 \langle \mathbb{E}_n [(y - \langle \alpha, a \rangle)c'], \theta_1 \rangle - \mathbb{E}_n [\langle c', \theta_1 \rangle^2] + \mu' \mathbb{E}_n [\langle a, \alpha \rangle^2]

This estimator can be shown to solve the problem:

.. math::

    \min _{\rho \geq 0, \|\rho\|_1 \leq V_1} \max _{\omega_1 \geq 0, \|\omega_1\|_1 = 1} \ell(\rho, \omega_1)

where 

.. math::

    \ell(\rho, \omega_1) := 2 \omega_1^{\top} \mathbb{E}_n [u_1 y] - 2 \omega_1^{\top} \mathbb{E}_n [u_1 v_1^{\top}] \rho - \omega_1^{\top} \mathbb{E}_n [u_1 u_1^{\top}] \omega_1 + \mu' \rho^{\top} \mathbb{E}_n [v_1 v_1^{\top}] \rho

Moreover, :math:`v_1 = (a, -a)`, :math:`u_1 = (c', -c')`; and :math:`\theta_1 = \omega_1^{+} - \omega_1^{-}`, :math:`\alpha = \rho^+ - \rho^{-}`.

.. admonition:: FTRL iterates for Estimator 2
    :class: lemma

    Consider the iterates for :math:`t = 1, \ldots, T`:

    .. math::
        :nowrap:

        \begin{aligned}
        \tilde{\rho}_{t+1} &= \exp\left(-\frac{\eta}{V_1} \left\{\sum_{\tau \leq t} -2 \mathbb{E}_n [v_1 u_1^{\top}] \omega_{1\tau} + 2 \mu' \mathbb{E}_n [v_1 v_1^{\top}] \tilde{\rho}_{\tau} - 2 \mathbb{E}_n [v_1 u_1^{\top}] \omega_{1t} + 2 \mu' \mathbb{E}_n [v_1 v_1^{\top}] \tilde{\rho}_{t} \right\} - 1\right) \\
        \rho_{t+1} &= \tilde{\rho}_{t+1} \min\left\{1, \frac{V_1}{\| \tilde{\rho}_{t+1} \|_1}\right\},
        \end{aligned}

    .. math::
        :nowrap:

        \begin{aligned}
        \tilde{\omega}_{1,t+1} &= \tilde{\omega}_{1,t} \exp\bigg(2\eta\left\{2 \mathbb{E}_n [u_1 y] - 2 \mathbb{E}_n [u_1 v_1^{\top}] \rho_{t} - 2 \mathbb{E}_n [u_1 u_1^{\top}] \tilde{\omega}_{1,t}\right\} \\
        &\qquad -\eta\left\{2 \mathbb{E}_n [u_1 y] - 2 \mathbb{E}_n [u_1 v_1^{\top}] \rho_{t-1} - 2 \mathbb{E}_n [u_1 u_1^{\top}] \tilde{\omega}_{1,t-1}\right\}\bigg) \\
        \omega_{1,t+1} &= \frac{\tilde{\omega}_{1,t+1}}{\|\tilde{\omega}_{1,t+1}\|_1}
        \end{aligned}

    with :math:`\tilde{\rho}_{-1} = \tilde{\rho}_{0} = \frac{1}{e}`, :math:`\tilde{\omega}_{1,-1} = \tilde{\omega}_{1,0} = \frac{1}{2p}`, and :math:`\eta = \frac{1}{8 \|\mathbb{E}_n [v_1 u_1^{\top}]\|_\infty}`.

    Then, :math:`\bar{\rho} = \frac{1}{T} \sum_{t=1}^{T} \rho_t`, :math:`\bar{\alpha} = \bar{\rho}^{+} - \bar{\rho}^{-}` is a :math:`O(T^{-1})`-approximate solution for :eq:`minimax-sparse-est2`.

.. autosummary::
   :toctree: _autosummary
   :template: class.rst

   sparse_l1_l1.sparse_ridge_l1vsl1

**Proof**

The proof is analogous to :ref:`estimator-1`.

.. admonition:: Duality gap
    :class: remark

    The upper bound for the duality gap as a certificate for convergence of the algorithm is given by:

    .. math::
        :nowrap:

        \begin{aligned}
        \text { tol } &= \left(\mathbb{E}_n [(y - \langle \bar{\alpha}, a \rangle)c']\right)^{\top} \mathbb{E}_n [c' c'^{\top}]^{\dagger} \left(\mathbb{E}_n [(y - \langle \bar{\alpha}, a \rangle)c']\right) + \mu' \bar{\alpha}^{\top} \mathbb{E}_n [aa^{\top}] \bar{\alpha} \\
        &\quad - \left(2 \bar{\theta_1}^{\top} \mathbb{E}_n [c'y] - \bar{\theta_1}^{\top} \mathbb{E}_n [c'a^{\top}] \frac{\mathbb{E}_n [aa^{\top}]^{\dagger}}{\mu'} \mathbb{E}_n [ac'^{\top}] \bar{\theta_1} - \bar{\theta_1}^{\top} \mathbb{E}_n [c' c'^{\top}] \bar{\theta_1} \right)
        \end{aligned}


Estimator 3 - (Ridge)
---------------------

The joint estimator is:

.. math::
    :label: minimax-sparse-est3

    \hat\alpha, \hat\beta := \underset{\|\beta\|_1 \leq V_2}{\operatorname{argmin}_{\|\alpha\|_1 \leq V_1}} \underset{\|\theta_2\|_1 \leq 1}{\max_{\|\theta_1\|_1 \leq 1}} & 2\langle \mathbb{E}_n [(y - \langle \alpha, a \rangle)c'], \theta_1 \rangle - \mathbb{E}_n [\langle c', \theta_1 \rangle^2] + \mu' \mathbb{E}_n [\langle a, \alpha \rangle^2] \\
    & + 2\langle \mathbb{E}_n [(\langle \alpha, a \rangle - \langle \beta, b \rangle)c], \theta_2 \rangle - \mathbb{E}_n [\langle c, \theta_2 \rangle^2] + \mu \mathbb{E}_n [\langle b, \beta \rangle^2]

and the problem is equivalent to:

.. math::

    \underset{\rho_2 \geq 0, \|\rho_2\|_1 \leq V_2}{\min_{\rho_1 \geq 0, \|\rho_1\|_1 \leq V_1}} \underset{\omega_2 \geq 0, \|\omega_2\|_1 = 1}{\max_{\omega_1 \geq 0, \|\omega_1\|_1 = 1}} \ell(\{\rho_1, \rho_2\}, \{\omega_1, \omega_2\})

.. math::

    \begin{aligned}
    \ell(\{\rho_1, \rho_2\}, \{\omega_1, \omega_2\}) := & 2 \omega_1^{\top} \mathbb{E}_n [u_1 y] - 2 \omega_1^{\top} \mathbb{E}_n \left[u_1 v_1^{\top}\right] \rho_1 - \omega_1^{\top} \mathbb{E}_n \left[u_1 u_1^{\top}\right] \omega_1 + \mu' \rho_1^\top \mathbb{E}_n [v_1 v_1^{\top}] \rho_1 \\
    & + 2 \omega_2^{\top} \mathbb{E}_n [u_2 v_1^{\top}] \rho_1 - 2 \omega_2^{\top} \mathbb{E}_n \left[u_2 v_2^{\top}\right] \rho_2 - \omega_2^{\top} \mathbb{E}_n \left[u_2 u_2^{\top}\right] \omega_2 + \mu \rho_2^\top \mathbb{E}_n [v_2 v_2^{\top}] \rho_2
    \end{aligned}

and :math:`v_1 = (a, -a)`, :math:`v_2 = (b, -b)`, :math:`u_1 = (c', -c')`, :math:`u_2 = (c, -c)`.

.. admonition:: FTRL iterates for Estimator 3 (Ridge)
    :class: lemma
    :name: sparse-l1-l1-est3

    Consider the iterates for :math:`t = 1, \ldots, T`:

    .. math::
        :nowrap:

        \begin{aligned}
        \tilde{\rho}_{1,t+1} &= \exp\left(-\frac{\eta}{V_1}\left\{\sum_{\tau = 1}^{t} \left(-2 \mathbb{E}_n [v_1 u_1^{\top}] \omega_{1,\tau} + 2 \mu' \mathbb{E}_n [v_1 v_1^\top] \tilde{\rho}_{1,\tau} + 2 \mathbb{E}_n [v_1 u_2^\top] \omega_{2,\tau}\right)\right. \right. \\
        & \left. \left. -2 \mathbb{E}_n [v_1 u_1^{\top}] \omega_{1,t} + 2 \mu' \mathbb{E}_n [v_1 v_1^\top] \tilde{\rho}_{1,t} + 2 \mathbb{E}_n [v_1 u_2^\top] \omega_{2,t} \right\} - 1\right) \\
        \rho_{1,t+1} &= \tilde{\rho}_{1,t+1} \min\left\{1, \frac{V_1}{\| \tilde{\rho}_{1,t+1} \|_1}\right\}, \\
        \tilde{\rho}_{2,t+1} &= \exp\left(-\frac{\eta}{V_2}\left\{\sum_{\tau = 1}^{t} \left(-2 \mathbb{E}_n [v_2 u_2^\top] \omega_{2,\tau} + 2 \mu \mathbb{E}_n [v_2 v_2^\top] \tilde{\rho}_{2,\tau}\right)\right. \right. \\
        & \left. \left. -2 \mathbb{E}_n [v_2 u_2^\top] \omega_{2,t} + 2 \mu \mathbb{E}_n [v_2 v_2^\top] \tilde{\rho}_{2,t} \right\} - 1\right) \\
        \rho_{2,t+1} &= \tilde{\rho}_{2,t+1} \min\left\{1, \frac{V_2}{\| \tilde{\rho}_{2,t+1} \|_1}\right\},
        \end{aligned}

    .. math::
        :nowrap:

        \begin{aligned}
        \tilde{\omega}_{1,t+1} &= \tilde{\omega}_{1,t} \exp\bigg(2\eta\left\{2 \mathbb{E}_n [u_1 y] - 2 \mathbb{E}_n [u_1 v_1^\top] \rho_{1,t} - 2 \mathbb{E}_n [u_1 u_1^{\top}] \tilde{\omega}_{1,t}\right\} \\
        &\qquad - \eta\left\{2 \mathbb{E}_n [u_1 y] - 2 \mathbb{E}_n [u_1 v_1^\top] \rho_{1,t-1} - 2 \mathbb{E}_n [u_1 u_1^{\top}] \tilde{\omega}_{1,t-1}\right\}\bigg) \\
        \omega_{1,t+1} &= \frac{\tilde{\omega}_{1,t+1}}{\|\tilde{\omega}_{1,t+1}\|_1} \\
        \tilde{\omega}_{2,t+1} &= \tilde{\omega}_{2,t} \exp\bigg(2\eta\left\{2 \mathbb{E}_n [u_2 v_1^\top] \rho_{1,t} - 2 \mathbb{E}_n [u_2 v_2^\top] \rho_{2,t} - 2 \mathbb{E}_n [u_2 u_2^{\top}] \tilde{\omega}_{2,t}\right\} \\
        &\qquad - \eta\left\{2 \mathbb{E}_n [u_2 v_1^\top] \rho_{1,t-1} - 2 \mathbb{E}_n [u_2 v_2^\top] \rho_{2,t-1} - 2 \mathbb{E}_n [u_2 u_2^{\top}] \tilde{\omega}_{2,t-1}\right\}\bigg) \\
        \omega_{2,t+1} &= \frac{\tilde{\omega}_{2,t+1}}{\|\tilde{\omega}_{2,t+1}\|_1}
        \end{aligned}

    with :math:`\tilde{\rho}_{1,-1} = \tilde{\rho}_{1,0} = \tilde{\rho}_{2,-1} = \tilde{\rho}_{2,0} = \frac{1}{e}` and :math:`\omega_{1,-1} = \omega_{1,0} = \omega_{2,-1} = \omega_{2,0} = \frac{1}{2p}`, and :math:`\eta = [16\max\left\{\left\|\mathbb{E}_n [v_1 u_1^\top]\right\|_\infty, \left\|\mathbb{E}_n [v_1 u_2^\top]\right\|_\infty, \left\|\mathbb{E}_n [v_2 u_2^\top]\right\|_\infty\right\}]^{-1}`.

    Then,

    .. math::

        \bar{\rho_1} = \frac{1}{T} \sum_{t=1}^{T} \rho_{1,t}, \quad \bar\alpha = \bar\rho_1^{+} - \bar\rho_1^{-} \\
        \bar{\rho_2} = \frac{1}{T} \sum_{t=1}^{T} \rho_{2,t}, \quad \bar\beta = \bar\rho_2^{+} - \bar\rho_2^{-}

    are a :math:`O(T^{-1})`-approximate solution for :eq:`minimax-sparse-est3`.

.. autosummary::
   :toctree: _autosummary
   :template: class.rst

   sparse2_l1_l1.sparse2_ridge_l1vsl1

**Proof**

We will match symbols with Proposition 17. Let 

.. math::

    \Theta = \{\rho_1 \;|\; \rho_1 \geq 0,\, \|\rho_1\|_1 \leq V_1\} \times \{\rho_2 \;|\; \rho_2 \geq 0,\, \|\rho_2\|_1 \leq V_2\} \\
    W = \{\omega_1 \;|\; \omega_1 \geq 0, \|\omega_1\|_1 = 1\} \times \{\omega_2 \;|\; \omega_2 \geq 0, \|\omega_2\|_1 = 1\}

be the convex feasibility sets. Note that :math:`\ell` is convex in :math:`(\rho_1, \rho_2)` and concave in :math:`(\omega_1, \omega_2)`. Equip the spaces :math:`\Theta` and :math:`W` with the direct sum :math:`1`-norm:

.. math::

    \|(\rho_1, \rho_2)\|_1 = \|\rho_1\|_1 + \|\rho_2\|_1

with dual norm:

.. math::

    \|(\rho_1, \rho_2)\|_\infty = \max\{\|\rho_1\|_\infty, \|\rho_2\|_\infty\}

Now, the derivatives are given by:

.. math::

    \begin{aligned}
    \nabla_{(\rho_1, \rho_2)} \ell(\{\rho_1, \rho_2\}, \{\omega_1, \omega_2\}) &= \begin{pmatrix}
        -2 \mathbb{E}_n [v_1 u_1^{\top}] \omega_1 + 2 \mu' \mathbb{E}_n [v_1 v_1^\top] \rho_1 + 2 \mathbb{E}_n [v_1 u_2^\top] \omega_2 \\
        -2 \mathbb{E}_n [v_2 u_2^{\top}] \omega_2 + 2 \mu \mathbb{E}_n [v_2 v_2^\top] \rho_2
    \end{pmatrix}^\top \\
    \nabla_{(\omega_1, \omega_2)} \ell(\{\rho_1, \rho_2\}, \{\omega_1, \omega_2\}) &= \begin{pmatrix}
        2 \mathbb{E}_n [u_1 y] - 2 \mathbb{E}_n [u_1 v_1^{\top}] \rho_1 - 2 \mathbb{E}_n [u_1 u_1^{\top}] \omega_1 \\
        2 \mathbb{E}_n [u_2 v_1^\top] \rho_1 - 2 \mathbb{E}_n [u_2 v_2^{\top}] \rho_2 - 2 \mathbb{E}_n [u_2 u_2^{\top}] \omega_2
    \end{pmatrix}^\top
    \end{aligned}

The Lipschitzness property is satisfied with :math:`L = 2 \max\left\{\left\|2 \mathbb{E}_n [v_1 u_1^\top]\right\|_\infty, \left\|2 \mathbb{E}_n [v_1 u_2^\top]\right\|_\infty, \left\|2 \mathbb{E}_n [v_2 u_2^\top]\right\|_\infty\right\}`:

.. math::

    \begin{aligned}
    &\left\|\nabla_{(\rho_1, \rho_2)} \ell(\{\rho_1, \rho_2\}, \{\omega_1, \omega_2\}) - \nabla_{(\rho_1, \rho_2)} \ell(\{\rho_1, \rho_2\}, \{\omega_1', \omega_2'\})\right\|_{\infty} \\
    &= \left\|\left(-2 \mathbb{E}_n [v_1 u_1^\top](\omega_1 - \omega_1') + 2 \mathbb{E}_n [v_1 u_2^\top](\omega_2 - \omega_2'), -2 \mathbb{E}_n [v_2 u_2^\top](\omega_2 - \omega_2')\right)\right\|_{\infty} \\
    &= \max\left\{\left\|-2 \mathbb{E}_n [v_1 u_1^\top](\omega_1 - \omega_1') + 2 \mathbb{E}_n [v_1 u_2^\top](\omega_2 - \omega_2')\right\|_\infty, \left\|-2 \mathbb{E}_n [v_2 u_2^\top](\omega_2 - \omega_2')\right\|_\infty\right\} \\
    &\leq \max\left\{\left\|2 \mathbb{E}_n [v_1 u_1^\top]\right\|_\infty \left\|(\omega_1 - \omega_1')\right\|_{1} + \left\|2 \mathbb{E}_n [v_1 u_2^\top]\right\|_\infty \left\|(\omega_2 - \omega_2')\right\|_{1}, \left\|2 \mathbb{E}_n [v_2 u_2^\top]\right\|_\infty \left\|(\omega_2 - \omega_2')\right\|_{1}\right\} \\
    &\leq 2 \max\left\{\left\|2 \mathbb{E}_n [v_1 u_1^\top]\right\|_\infty, \left\|2 \mathbb{E}_n [v_1 u_2^\top]\right\|_\infty, \left\|2 \mathbb{E}_n [v_2 u_2^\top]\right\|_\infty\right\} \left[\left\|(\omega_1 - \omega_1')\right\|_{1} + \left\|(\omega_2 - \omega_2')\right\|_{1}\right]
    \end{aligned}

and similarly for the Lipschitzness of :math:`\nabla_{(\omega_1, \omega_2)} \ell(\{\rho_1, \rho_2\}, \{\omega_1, \omega_2\})`.

Consider the following entropic regularizers:

.. math::

    \begin{aligned}
    R_{min}(\rho_1, \rho_2) &= V_1 \sum_{i=1}^{2p} \rho_{1i} \log (\rho_{1i}) + V_2 \sum_{i=1}^{2p} \rho_{2i} \log (\rho_{2i}) \\
    R_{max}(\omega_1, \omega_2) &= \sum_{i=1}^{2p} \omega_{1i} \log (\omega_{1i}) + \sum_{i=1}^{2p} \omega_{2i} \log (\omega_{2i})
    \end{aligned}

which are :math:`1`-strongly convex in the spaces :math:`\Theta`, and :math:`W` respectively.

To find the iterates it remains to solve:

.. math::

    \begin{aligned}
    (\rho_{1,t+1}, \rho_{2,t+1}) &= \operatorname{argmin}_{\rho_1, \rho_2} \big(\rho_1, \rho_2\big)^\top \left(\sum_{\tau = 1}^{t} \left\{\nabla_{(\rho_1, \rho_2)} \ell(\{\rho_{1,\tau}, \rho_{2,\tau}\}, \{\omega_{1,\tau}, \omega_{2,\tau}\})\right\} \right. \\
    &+\left. \nabla_{(\rho_1, \rho_2)} \ell(\{\rho_{1,t}, \rho_{2,t}\}, \{\omega_{1,t}, \omega_{2,t}\})\right) + R_{min}(\rho_1, \rho_2)
    \end{aligned}


Given the derivatives computed above, and that the problem is separable in :math:`\rho_1`, :math:`\rho_2`, the iterates are:

.. math::

    \begin{aligned}
    \tilde{\rho}_{1,t+1} &= \exp\left(-\frac{\eta}{V_1}\left\{\sum_{\tau = 1}^{t} \left(-2 \mathbb{E}_n [v_1 u_1^{\top}] \omega_{1,\tau} + 2 \mu' \mathbb{E}_n [v_1 v_1^\top] \tilde{\rho}_{1,\tau} + 2 \mathbb{E}_n [v_1 u_2^\top] \omega_{2,\tau}\right)\right.\right. \\
    &\left.\left.-2 \mathbb{E}_n [v_1 u_1^{\top}] \omega_{1,t} + 2 \mu' \mathbb{E}_n [v_1 v_1^\top] \tilde{\rho}_{1,t} + 2 \mathbb{E}_n [v_1 u_2^\top] \omega_{2,t}\right\} - 1\right) \\
    \rho_{1,t+1} &= \tilde{\rho}_{1,t+1} \min\left\{1, \frac{V_1}{\| \tilde{\rho}_{1,t+1} \|_1}\right\}, \\
    \tilde{\rho}_{2,t+1} &= \exp\left(-\frac{\eta}{V_2}\left\{\sum_{\tau = 1}^{t} \left(-2 \mathbb{E}_n [v_2 u_2^\top] \omega_{2,\tau} + 2 \mu \mathbb{E}_n [v_2 v_2^\top] \tilde{\rho}_{2,\tau}\right)\right.\right. \\
    &\left.\left.-2 \mathbb{E}_n [v_2 u_2^\top] \omega_{2,t} + 2 \mu \mathbb{E}_n [v_2 v_2^\top] \tilde{\rho}_{2,t}\right\} - 1\right) \\
    \rho_{2,t+1} &= \tilde{\rho}_{2,t+1} \min\left\{1, \frac{V_2}{\| \tilde{\rho}_{2,t+1} \|_1}\right\},
    \end{aligned}

.. math::
   :nowrap:

   \begin{aligned}
   \tilde{\omega}_{1,t+1} &= \tilde{\omega}_{1,t} \exp\bigg(2\eta\left\{2 \mathbb{E}_n [u_1 y] - 2 \mathbb{E}_n [u_1 v_1^\top] \rho_{1,t} - 2 \mathbb{E}_n [u_1 u_1^{\top}] \tilde{\omega}_{1,t}\right\} \\
   &\qquad - \eta\left\{2 \mathbb{E}_n [u_1 y] - 2 \mathbb{E}_n [u_1 v_1^\top] \rho_{1,t-1} - 2 \mathbb{E}_n [u_1 u_1^{\top}] \tilde{\omega}_{1,t-1}\right\}\bigg) \\
   \omega_{1,t+1} &= \frac{\tilde{\omega}_{1,t+1}}{\|\tilde{\omega}_{1,t+1}\|_1} \\
   \tilde{\omega}_{2,t+1} &= \tilde{\omega}_{2,t} \exp\bigg(2\eta\left\{2 \mathbb{E}_n [u_2 v_1^\top] \rho_{1,t} - 2 \mathbb{E}_n [u_2 v_2^\top] \rho_{2,t} - 2 \mathbb{E}_n [u_2 u_2^{\top}] \tilde{\omega}_{2,t}\right\} \\
   &\qquad - \eta\left\{2 \mathbb{E}_n [u_2 v_1^\top] \rho_{1,t-1} - 2 \mathbb{E}_n [u_2 v_2^\top] \rho_{2,t-1} - 2 \mathbb{E}_n [u_2 u_2^{\top}] \tilde{\omega}_{2,t-1}\right\}\bigg) \\
   \omega_{2,t+1} &= \frac{\tilde{\omega}_{2,t+1}}{\|\tilde{\omega}_{2,t+1}\|_1}
   \end{aligned}


with :math:`\tilde{\rho}_{1,-1} = \tilde{\rho}_{1,0} = \tilde{\rho}_{2,-1} = \tilde{\rho}_{2,0} = \frac{1}{e}` and :math:`\omega_{1,-1} = \omega_{1,0} = \omega_{2,-1} = \omega_{2,0} = \frac{1}{2p}`.


Putting everything together, by Proposition 17 the ensembles:

.. math::

    \bar{\rho_1} = \frac{1}{T} \sum_{t=1}^T \rho_{1,t}, \quad \bar{\rho_2} = \frac{1}{T} \sum_{t=1}^T \rho_{2,t}

are a :math:`O\left(\frac{1}{T}\right)`-approximate solution for the minimax objective.

.. admonition:: Duality gap
    :class: remark

    The upper bound for the duality gap to the minimax problem in :eq:`minimax-sparse-est3` is: 

    .. math::
        :nowrap:

        \begin{aligned}
        &\text { tol }=\left\|\mathbb{E}_n [(y - \langle \bar\alpha, a \rangle)c']\right\|^2_{\mathbb{E}_n [c'c'^\top]^{\dagger}} + \mu' \|\bar\alpha\|^2_{\mathbb{E}_n [aa^\top]} + \left\|\mathbb{E}_n [(\langle\bar\alpha, a\rangle - \langle\bar\beta, b\rangle)c]\right\|^2_{\mathbb{E}_n [c'c'^\top]^{\dagger}} + \mu \|\bar\beta\|^2_{\mathbb{E}_n [bb^\top]} \\
        &- \left(2 \bar\theta_1^\top \mathbb{E}_n [c'y] - \frac{1}{\mu'}\left\|\mathbb{E}_n [ac'^\top] \bar\theta_1 - \mathbb{E}_n [ac^\top] \bar\theta_2\right\|^2_{\mathbb{E}_n [aa^\top]^\dagger} - \frac{1}{\mu}\left\|\mathbb{E}_n [bc^\top] \bar\theta_2\right\|^2_{\mathbb{E}_n [bb^\top]^\dagger} - \left\|\bar\theta_1\right\|^2_{\mathbb{E}_n [c'c'^\top]} - \left\|\bar\theta_2\right\|^2_{\mathbb{E}_n [cc^\top]}\right)
        \end{aligned}

    where  :math:`\|x\|_{M} = x^\top M x` is the ellipsoid norm.

Remark: Subsetted estimator
^^^^^^^^^^^^^^^^^^^^^^^^^^^

For the subsetted estimators it suffices to replace the empirical mean :math:`\mathbb{E}_n` with either :math:`\mathbb{E}_p` or :math:`\mathbb{E}_q` accordingly in the iterates given by the FTRL algorithm. In concrete, for the implementation, we compute :math:`\mathbb{E}_p` as a weighted average, where the weights are set to zero for the indices outside :math:`[p]`, and analogous for :math:`\mathbb{E}_q`.

Estimator 3 - (:math:`\ell_1`-norm)
----------------------------------

The joint estimator is

.. math::
   :label: minimax-sparse_est3_l1

   \begin{aligned}
   \hat\alpha, \hat\beta &:= \underset{\|\beta\|_1 \leq V_2}{\operatorname{argmin}_{\|\alpha\|_1 \leq V_1}} \underset{\|\theta_2\|_1\leq 1}{\max_{\|\theta_1\|_1\leq 1}} \left(
   2\langle\mathbb{E}_n[(y-\langle\alpha, a\rangle)c'],\theta_1\rangle -\mathbb{E}_n[\langle c',\theta_1\rangle^2]+\mu'\|\alpha\|_1 \right.\\
   &\left. + 2\langle\mathbb{E}_n[(\langle\alpha, a\rangle-\langle\beta, b\rangle)c],\theta_2\rangle -\mathbb{E}_n[\langle c,\theta_2\rangle^2]+\mu\|\beta\|_1 
   \right)
   \end{aligned}

This minimax problem can be reformulated as

.. math::

   \underset{\rho_2 \geq 0,\|\rho_2\|_1 \leq V_2}{\min_{\rho_1 \geq 0,\|\rho_1\|_1 \leq V_1}} \underset{\omega_2\geq 0, \|\omega_2\|_1\leq 1}{\max_{\omega_1\geq 0, \|\omega_1\|_1= 1}} \ell(\{\rho_1,\rho_2\}, \{\omega_1,\omega_2\})

where

.. math::

   \ell(\{\rho_1,\rho_2\}, \{\omega_1,\omega_2\}) := 
   2\omega_1^{\top} \mathbb{E}_n[u_1 y] - 2\omega_1^{\top} \mathbb{E}_n\left[u_1 v_1^{\top}\right] \rho_1 - \omega_1^{\top} \mathbb{E}_n\left[u_1 u_1^{\top}\right]\omega_1 + \mu' \sum_{i=1}^{2 p} \rho_{1i} \\
   + 2\omega_2^{\top} \mathbb{E}_n[u_2 v_1^{\top}] \rho_1 - 2\omega_2^{\top} \mathbb{E}_n\left[u_2 v_2^{\top}\right] \rho_2 - \omega_2^{\top} \mathbb{E}_n\left[u_2 u_2^{\top}\right]\omega_2 + \mu \sum_{i=1}^{2 p} \rho_{2i}

and :math:`v_1 = (a, -a)`, :math:`v_2 = (b, -b)`, :math:`u_1 = (c',-c')`, :math:`u_2 = (c,-c)`.

We state without proof, the algorithm for an approximate solution:

.. admonition:: FTRL iterates for Estimator 3 (:math:`\ell_1`-norm)
   :name: ftrl-iterates-estimator3-l1

   Consider the iterates for :math:`t=1,\ldots, T`:

   .. math::
        :nowrap:

        \begin{aligned}
        \tilde{\rho}_{1,t+1} &= \exp\left(-\frac{\eta}{V_1}\left\{\sum_{\tau=1}^{t} \bigg(-2\mathbb{E}_n[v_1u_1^{\top}]\omega_{1,\tau} + 2\mathbb{E}_n[v_1u_2^\top]\omega_{2,\tau}\bigg)\right.\right. \\
        &\left.\left.-2\mathbb{E}_n[v_1u_1^{\top}]\omega_{1,t} + 2\mathbb{E}_n[v_1u_2^\top]\omega_{2,t} + (t+1)\mu'\right\}-1\right) \\
        \rho_{1,t+1} &=  \tilde{\rho}_{1,t+1}\min\left\{1, \frac{V_1}{\| \tilde{\rho}_{1,t+1}\|_1}\right\},\\
        \tilde{\rho}_{2,t+1} &= \exp\left(-\frac{\eta}{V_2}\left\{\sum_{\tau=1}^{t} \bigg(-2\mathbb{E}_n[v_2u_2^\top]\omega_{2,\tau}\bigg)-2\mathbb{E}_n[v_2u_2^\top]\omega_{2,t}+(t+1)\mu\right\}-1\right)\\
        \rho_{2,t+1} &=  \tilde{\rho}_{2,t+1}\min\left\{1, \frac{V_2}{\| \tilde{\rho}_{2,t+1}\|_1}\right\},
        \end{aligned}

   .. math::
        :nowrap:

        \begin{aligned}
        \tilde{\omega}_{1,t+1} &= \tilde\omega_{1,t}\exp\bigg(2\eta\left\{2\mathbb{E}_n[u_1y]-2\mathbb{E}_n[u_1v_1^\top]\rho_{1,t} - 2\mathbb{E}_n[u_1u_1^{\top}]\tilde\omega_{1,t}\right\} \\
        &\qquad -\eta\left\{2\mathbb{E}_n[u_1y]-2\mathbb{E}_n[u_1v_1^\top]\rho_{1,t-1} - 2\mathbb{E}_n[u_1u_1^{\top}]\tilde\omega_{1,t-1}\right\}\bigg)\\
        \omega_{1,t+1} &= \frac{\tilde{\omega}_{1,t+1}}{\|\tilde{\omega}_{1,t+1}\|_1}\\
        \tilde{\omega}_{2,t+1} &= \tilde\omega_{2,t}\exp\bigg(2\eta\left\{2\mathbb{E}_n[u_2v_1^\top]\rho_{1,t}-2\mathbb{E}_n[u_2v_2^\top]\rho_{2,t} - 2\mathbb{E}_n[u_2u_2^{\top}]\tilde\omega_{2,t}\right\} \\
        &\qquad -\eta\left\{2\mathbb{E}_n[u_2v_1^\top]\rho_{1,t-1}-2\mathbb{E}_n[u_2v_2^\top]\rho_{2,t-1} - 2\mathbb{E}_n[u_2u_2^{\top}]\tilde\omega_{2,t-1}\right\}\bigg)\\
        \omega_{2,t+1} &= \frac{\tilde{\omega}_{2,t+1}}{\|\tilde{\omega}_{2,t+1}\|_1}
        \end{aligned}

   with :math:`\tilde\rho_{1,-1} = \tilde\rho_{1,0} = \tilde\rho_{2,-1} = \tilde\rho_{2,0}= \frac{1}{e}` and :math:`\omega_{1,-1}=\omega_{1,0} = \omega_{2,-1}=\omega_{2,0}= \frac{1}{2p}`, and :math:`\eta =[16\max\left\{\left\|\mathbb{E}_n[v_1u_1^\top]\right\|_\infty, \left\|\mathbb{E}_n[v_1u_2^\top]\right\|_\infty, \left\| \mathbb{E}_n[v_2u_2^\top]\right\|_\infty\right\}]^{-1}`.

   Then,

   .. math::
        :nowrap:

        \begin{aligned}
        \bar{\rho_1} = \frac{1}{T}\sum_{t=1}^{T}\rho_{1,t}\,,\quad \bar\alpha = \bar\rho_1^{+}-\bar\rho_1^{-} \\
        \bar{\rho_2} = \frac{1}{T}\sum_{t=1}^{T}\rho_{2,t}\,,\quad \bar\beta = \bar\rho_2^{+}-\bar\rho_2^{-} 
        \end{aligned}

   are a :math:`O(T^{-1})`-approximate solution for :eq:`minimax-sparse_est3_l1`.

.. autosummary::
   :toctree: _autosummary
   :template: class.rst

   sparse2_l1_l1.sparse2_l1vsl1

.. admonition:: Duality gap
    :class: remark
    
    The tolerance for the duality gap (to  :eq:`minimax-sparse_est3_l1`) is given by
    
    .. math::
        :nowrap:

        \begin{aligned}
        \text{tol} &= \left\|\mathbb{E}_n[(y-\langle \bar\alpha, a \rangle)c']\right\|^2_{\mathbb{E}_n[c'c'^\top]^{\dagger}}+\mu'\|\bar\alpha\|_1+\left\|\mathbb{E}_n[(\langle\bar\alpha, a\rangle-\langle\bar\beta, b\rangle)c]\right\|^2_{\mathbb{E}_n[c'c'^\top]^{\dagger}}+\mu\|\bar\beta\|_1 \\
        &-\bigg(\bar\theta_1^\top\mathbb{E}_n[c'y] + V_1\left\{\mu'-2\|\mathbb{E}_n[ac'^\top]\bar\theta_1\|_\infty+2\|\mathbb{E}_n[ac^\top]\bar\theta_2\|_\infty\right\}^{-}+V_2\left\{\mu-2\|\mathbb{E}_n[bc^\top]\bar\theta_2\|_\infty\right\}^{-} \\
        &\qquad\qquad\qquad -\left\|\bar\theta_1\right\|_{\mathbb{E}_n[c'c'^\top]}-\left\|\bar\theta_2\right\|_{\mathbb{E}_n[cc^\top]}\bigg)
        \end{aligned}

    
