Regularized Kernel Hilbert Space
================================
.. _rkhs_estimators:

In this section we assume that the function classes 
whenever :math:`\mathcal{G}`, :math:`\mathcal{H}`, :math:`\mathcal{F}`, :math:`\mathcal{F}^\prime` are RKHS.  Let :math:`\Phi_A:\mathcal{G}\rightarrow\mathbb{R}^n` be an operator with :math:`i` th row :math:`\langle \phi(A_i), \cdot \rangle_{\mathcal{G}}` with corresponding kernel matrix :math:`K_A`.  Define analogously :math:`\Phi_B, \ldots` for the rest of the function classes.


Closed form - Estimator 1
-------------------------

We study the estimator

.. math::

    \hat{g} = \arg \min_{g \in \mathcal{G}} 
    \max_{f' \in \mathcal{F'}} \mathbb{E}_n \left[ 2 \left\{ g(A) - Y \right\} f'(C') - f'(C')^2 \right] - \lambda \| f \|_{\mathcal{F}}^2
     + \mu' \| g \|_{\mathcal{G}}^2

.. admonition:: Formula of minimizers

    The minimizer takes the form :math:`\hat{g} = \Phi_A^* \hat{\alpha}` where,

    .. math::

        \hat{\alpha} &= \left(K_A P_C' K_A + \mu K_A \right)^{\dagger} K_A P_C' Y \\
        P_{C'} &= \left(K_{C'} + \lambda \right)^{\dagger} K_{C'}

.. autosummary::
   :toctree: _autosummary
   :template: class.rst

   rkhsiv.RKHSIV
   rkhsiv.RKHSIVCV

**Remark (Nystrom approximation)**
A low-rank approximation using Nystrom method is also implemented.

.. autosummary::
   :toctree: _autosummary
   :template: class.rst

   rkhsiv.ApproxRKHSIV
   rkhsiv.ApproxRKHSIVCV

   

Closed form - Estimator 2
-------------------------

We study the estimator

.. math::

    \hat{g} = \arg \min_{g \in \mathcal{G}} 
    \max_{f' \in \mathcal{F'}} \mathbb{E}_n \left[ 2 \left\{ g(A) - Y \right\} f'(C') - f'(C')^2 \right]
     + \mu' \mathbb{E}_n \{ g(A)^2 \}

.. admonition:: Formula of minimizers

    The minimizer takes the form :math:`\hat{g} = \Phi_A^* \hat{\alpha}` where,

    .. math::

        \hat{\alpha} &= \left( K_A P_C' K_A + \mu K_A^2 \right)^{\dagger} K_A P_C' Y \\
        P_{C'} &= K_{C'}^{\dagger} K_{C'}

.. autosummary::
   :toctree: _autosummary
   :template: class.rst

   rkhsiv.RKHSIVL2
   rkhsiv.RKHSIVL2CV

Closed form - Estimator 3
-------------------------

We study the ridge regularized *joint* estimator:

.. math::

    (\hat{g}, \hat{h}) = \arg \min_{g \in \mathcal{G}, h \in \mathcal{H}} 
    \max_{f' \in \mathcal{F}} \mathbb{E}_n \left[ 2 \left\{ g(A) - Y \right\} f'(C') - f'(C')^2 \right]
    + \mu' \mathbb{E}_n \{ g(A)^2 \} \\
    \quad + \max_{f \in \mathcal{F}} \mathbb{E}_n \left[ 2 \left\{ h(B) - g(A) \right\} f(C) - f(C)^2 \right]
    + \mu \mathbb{E}_n \{ h(B)^2 \}

Let :math:`V_{g,h}' = g(A) - Y` and :math:`V_{g,h} = h(B) - g(A)`. Let :math:`\Phi_C : \mathcal{F} \rightarrow \mathbb{R}^n` be an operator with :math:`i` th row :math:`\langle \phi(C_i), \cdot \rangle_{\mathcal{F}}`. Define :math:`\Phi_{C'}` analogously, replacing :math:`C_i` with :math:`C_i'`. Let :math:`K_C` and :math:`K_{C'}` be the corresponding kernel matrices.

In remarks below, we also study the following modification, which we call the "subsetted" estimator:

.. math::

    (\hat{g}, \hat{h}) = \arg \min_{g \in \mathcal{G}, h \in \mathcal{H}} 
    \max_{f' \in \mathcal{F}} \mathbb{E}_p \left[ 2 \left\{ g(A) - Y \right\} f'(C') - f'(C')^2 \right]
    + \mu' \mathbb{E}_n \{ g(A)^2 \} \\
    \quad + \max_{f \in \mathcal{F}} \mathbb{E}_q \left[ 2 \left\{ h(B) - g(A) \right\} f(C) - f(C)^2 \right]
    + \mu \mathbb{E}_n \{ h(B)^2 \}

where :math:`[p]` and :math:`[q]` partition :math:`[n] = (1, \ldots, n)`, so :math:`p + q = n`.

For the index set :math:`[p]`, let :math:`I_{[p]} \in \mathbb{R}^{p \times n}` be the matrix of ones and zeros such that :math:`V_{[p]} = I_{[p]} V` gives the elements of :math:`V` whose indices are in :math:`[p]`.


Maximizers
^^^^^^^^^^


**Existence of maximizers**

There exist coefficients :math:`\hat{\gamma}_{g,h}, \hat{\gamma}'_{g,h} \in \mathbb{R}^n` such that maximizers take the form :math:`\hat{f}_{g,h} = \Phi_C^* \hat{\gamma}_{g,h}` and :math:`\hat{f}'_{g,h} = \Phi_{C'}^* \hat{\gamma}'_{g,h}`.

**Remark (Subsetted estimator)**

For the subsetted estimator, the same results hold but with :math:`\hat{\gamma}_{g,h;[q]} \in \mathbb{R}^q` and :math:`\hat{\gamma}'_{g,h;[p]} \in \mathbb{R}^p`, acting on appropriately modified feature operators :math:`\Phi^*_{C;[q]}` and :math:`\Phi^*_{C';[p]}`.

**Proof**

Write the objectives for the maximizers as

.. math::

    \mathcal{E}'(f') = \mathbb{E}_n \left\{ 2 V'_{g,h} f'(C') - f'(C')^2 \right\} \\
    \mathcal{E}(f) = \mathbb{E}_n \left\{ 2 V_{g,h} f(C) - f(C)^2 \right\}

We prove the former result; the latter is similar. By the Riesz representation theorem,

.. math::

    \mathcal{E}(f) = \mathbb{E}_n \left\{ 2 V_{g,h} \langle f, \phi(C) \rangle_{\mathcal{F}} - \langle f, \phi(C) \rangle_{\mathcal{F}}^2 \right\}

For an RKHS, evaluation is a continuous functional represented as the inner product with the feature map. Due to the ridge penalty, the stated objective has a maximizer :math:`\hat{f}_{g,h}` that obtains the maximum.

To lighten notation, we suppress the indexing of :math:`\hat{f}_{g,h}` by :math:`(g,h)` for the rest of this argument. Write :math:`\hat{f} = \hat{f}_n + \hat{f}^{\perp}_n` where :math:`\hat{f}_n \in \text{row}(\Phi_C)` and :math:`\hat{f}_n^{\perp} \in \text{null}(\Phi_C)`. Substituting this decomposition of :math:`\hat{f}` into the objective, we see that

.. math::

    \mathcal{E}(\hat{f}) = \mathcal{E}(\hat{f}_n)

Hence if :math:`\hat{f}` is a maximizer, then there exists :math:`\hat{f}_n` that is also a maximizer.

**Formula of maximizers**

The explicit formula for the coefficients is :math:`\hat{\gamma}_{g,h} = K_C^{\dagger} \vec{V}_{g,h}` and :math:`\hat{\gamma}'_{g,h} = K_{C'}^{\dagger} \vec{V}'_{g,h}`.

**Remark (Subsetted estimator)**

For the subsetted estimator, the same results hold but with :math:`\hat{\gamma}_{g,h;[q]} = K_{C;[q,q]}^{\dagger} \vec{V}_{g,h;[q]}` and :math:`\hat{\gamma}'_{g,h;[p]} = K_{C';[p,p]}^{\dagger} \vec{V}'_{g,h;[p]}`.

**Proof**

We prove the former result; the latter is similar. Write the objective as

.. math::

    \mathcal{E}(f) = 2 \langle f, \hat{\mu}_{g,h} \rangle_{\mathcal{F}} - \langle f, \hat{T}_C f \rangle_{\mathcal{F}}

where :math:`\hat{\mu}_{g,h} = \mathbb{E}_n \{ V_{g,h} \phi(C) \} = \frac{1}{n} \Phi_C^* \vec{V}_{g,h}` and :math:`\hat{T}_C = \mathbb{E}_n \{ \phi(C) \otimes \phi(C)^* \} = \frac{1}{n} \Phi_C^* \Phi_C`. Hence by the existence of maximizers,

.. math::

    \mathcal{E}(\gamma) = 2 \langle \Phi_C^* \gamma_{g,h}, \hat{\mu}_{g,h} \rangle_{\mathcal{F}} - \langle \Phi_C^* \gamma_{g,h}, \hat{T}_C \Phi_C^* \gamma_{g,h} \rangle_{\mathcal{F}}
    = \frac{2}{n} \gamma_{g,h}^{\top} \Phi_C \Phi_C^* \vec{V}_{g,h} - \frac{1}{n} \gamma_{g_h}^{\top} \Phi_C \Phi_C^* \Phi_C \Phi_C^* \gamma_{g,h}

Since :math:`K_C = \Phi_C \Phi_C^*`, the first order condition yields :math:`K_C \vec{V}_{g,h} = K_C^2 \hat{\gamma}_{g,h}`, i.e. :math:`\hat{\gamma}_{g,h} = K_C^{\dagger} \vec{V}_{g,h}` where :math:`K_C^{\dagger}` is the pseudoinverse of :math:`K_C`.

Minimizers
^^^^^^^^^^

Let :math:`\Phi_A : \mathcal{H} \rightarrow \mathbb{R}^n` be an operator with :math:`i` th row :math:`\langle \phi(A_i), \cdot \rangle_{\mathcal{H}}`. Define :math:`\Phi_B` analogously, replacing :math:`A_i` with :math:`B_i`. Let :math:`K_A` and :math:`K_B` be the corresponding kernel matrices.

**Existence of minimizers**

There exist coefficients :math:`\alpha, \beta \in \mathbb{R}^n` such that minimizers take the form :math:`\hat{g} = \Phi_A^* \hat{\alpha}` and :math:`\hat{h} = \Phi_B^* \hat{\beta}`.

**Remark (Subsetted estimator)**

The result remains true for the subsetted estimator.

**Proof**

To begin, write the objective :math:`\mathcal{E}(g,h)` as

.. math::

    \mathbb{E}_n \left\{ 2 V'_{g,h} \hat{f}_{g,f}'(C') - \hat{f}_{g,h}'(C')^2 \right\}
    + \mu' \mathbb{E}_n \{ g(A)^2 \} \\
    + \mathbb{E}_n \left\{ 2 V_{g,h} \hat{f}_{g,h}(C) - \hat{f}_{g,h}(C)^2 \right\}
    + \mu \mathbb{E}_n \{ h(B)^2 \}

By the existence and formula of maximizers,

.. math::

    \hat{f}_{g,f}'(C') = \langle \hat{f}_{g,f}', \phi(C') \rangle_{\mathcal{F}}
    = \langle \Phi_{C'}^* K_{C'}^{\dagger} \vec{V}'_{g,h}, \phi(C') \rangle_{\mathcal{F}} \\
    \hat{f}_{g,h}(C) = \langle \hat{f}_{g,f}, \phi(C) \rangle_{\mathcal{F}}
    = \langle \Phi_{C}^* K_{C}^{\dagger} \vec{V}_{g,h}, \phi(C) \rangle_{\mathcal{F}}

Hence :math:`(g,h)` only appear via :math:`V'_{g,h} = g(A) - Y`, :math:`V_{g,h} = h(B) - g(A)`, and directly as :math:`g(A)` and :math:`h(B)`. In all of these expressions, they can be further expressed as :math:`g(A) = \langle g, \phi(A) \rangle_{\mathcal{G}}` and :math:`h(B) = \langle h, \phi(B) \rangle_{\mathcal{H}}`, which is a linear functional. The overall objective is quadratic in such terms, so the stated objective has maximizers :math:`(\hat{g}, \hat{h})` that obtain the maximum.

By a similar argument to the existence of maximizers, for any :math:`(\hat{g}, \hat{h})` attaining the maximum, :math:`\mathcal{E}(\hat{g}, \hat{h}) = \mathcal{E}(\hat{g}_n, \hat{h}_n)` where :math:`\hat{g}_n \in \text{row}(\Phi_A)` and :math:`\hat{h}_n \in \text{row}(\Phi_B)`.

**Properties of pseudo-inverse**

For any square symmetric matrix :math:`K \in \mathbb{R}^{n \times n}`, its eigendecomposition is :math:`K = U \Sigma U^{\top}` where :math:`\Sigma \in \mathbb{R}^{r \times r}` and :math:`r \leq n`. Its pseudo-inverse is :math:`K^- = U \Sigma^{\dagger} U^{\top}`. Moreover, :math:`K^{\dagger} K = KK^{\dagger} = UU^{\top}`, which is a projection.

To lighten notation, let :math:`K_C^{\dagger} K_C = P_C`.

.. admonition:: Formula of minimizers
    
    The explicit formula for the coefficients is
    
    .. math::
    
        \hat{\beta} = \left[ K_A \left\{ - P_C + \left( P_{C'} + P_C + \mu' \right) K_A \left( K_B P_C K_A \right)^{\dagger} K_B \left( P_C + \mu \right) \right\} K_B \right]^{\dagger} K_A P_{C'} Y \\
        \hat{\alpha} = \left( K_B P_C K_A \right)^{\dagger} K_B \left( P_C + \mu \right) K_B \hat{\beta}

.. autosummary::
   :toctree: _autosummary
   :template: class.rst

   rkhs2iv.RKHS2IVL2
   rkhs2iv.RKHS2IVL2CV

**Proof**

We proceed in steps.

1. Write the objective :math:`\mathcal{E}(g,h)` as

.. math::

    2 \langle \hat{f}'_{g,h}, \hat{\mu}'_{g,h} \rangle_{\mathcal{F}} - \langle \hat{f}'_{g,h}, \hat{T}_{C'} \hat{f}'_{g,h} \rangle_{\mathcal{F}}  
    + \mu' \langle g, \hat{T}_A g \rangle_{\mathcal{G}} \\
    + 2 \langle \hat{f}_{g,h}, \hat{\mu}_{g,h} \rangle_{\mathcal{F}} - \langle \hat{f}_{g,h}, \hat{T}_C \hat{f}_{g,h} \rangle_{\mathcal{F}}  
    + \mu \langle h, \hat{T}_B h \rangle_{\mathcal{H}}

where

.. math::

    \hat{\mu}'_{g,h} = \frac{1}{n} \Phi_{C'}^* \vec{V}'_{g,h}, \quad
    \hat{\mu}_{g,h} = \frac{1}{n} \Phi_C^* \vec{V}_{g,h}

and the covariance operators are defined analogously to the formula of maximizers. Hence,

.. math::

    \mathcal{E}(g,h) =
    2 \langle \Phi_{C'}^* K_{C'}^{\dagger} \vec{V}'_{g,h}, \hat{\mu}'_{g,h} \rangle_{\mathcal{F}}
    - \langle \Phi_{C'}^* K_{C'}^{\dagger} \vec{V}'_{g,h}, \hat{T}_{C'} \Phi_{C'}^* K_{C'}^{\dagger} \vec{V}'_{g,h} \rangle_{\mathcal{F}} \\
    + \mu' \langle g, \hat{T}_A g \rangle_{\mathcal{G}} \\
    + 2 \langle \Phi_C^* K_C^{\dagger} \vec{V}_{g,h}, \hat{\mu}_{g,h} \rangle_{\mathcal{F}}
    - \langle \Phi_C^* K_C^{\dagger} \vec{V}_{g,h}, \hat{T}_C \Phi_C^* K_C^{\dagger} \vec{V}_{g,h} \rangle_{\mathcal{F}} \\
    + \mu \langle h, \hat{T}_B h \rangle_{\mathcal{H}}

.. math::

    = \frac{2}{n} (\vec{V}'_{g,h})^{\top} K_{C'}^{\dagger} \Phi_{C'} \Phi_{C'}^* \vec{V}'_{g,h}
    - \frac{1}{n} (\vec{V}'_{g,h})^{\top} K_{C'}^{\dagger} \Phi_{C'} \Phi_{C'}^* \Phi_{C'} \Phi_{C'}^* K_{C'}^{\dagger} \vec{V}'_{g,h} \\
    + \mu' \langle g, \hat{T}_A g \rangle_{\mathcal{G}} \\
    + \frac{2}{n} \vec{V}_{g,h}^{\top} K_C^{\dagger} \Phi_C \Phi_C^* \vec{V}_{g,h}
    - \frac{1}{n} \vec{V}_{g,h}^{\top} K_C^{\dagger} \Phi_C \Phi_C^* \Phi_C \Phi_C^* K_C^{\dagger} \vec{V}_{g,h} \\
    + \mu \langle h, \hat{T}_B h \rangle_{\mathcal{H}}

.. math::

    = \frac{1}{n} (\vec{V}'_{g,h})^{\top} P_{C'} \vec{V}'_{g,h}
    + \mu' \langle g, \hat{T}_A g \rangle_{\mathcal{G}} \\
    + \frac{1}{n} \vec{V}_{g,h}^{\top} P_C \vec{V}_{g,h}
    + \mu \langle h, \hat{T}_B h \rangle_{\mathcal{H}}

2. Let :math:`Y, G, H \in \mathbb{R}^n` be defined with :math:`G_i = g(A_i)` and :math:`H_i = h(B_i)`. In this notation,

.. math::

    \frac{1}{n} (\vec{V}'_{g,h})^{\top} P_{C'} \vec{V}'_{g,h}
    = \frac{1}{n} (Y^{\top} P_{C'} Y - 2 G^{\top} P_{C'} Y + G^{\top} P_{C'} G), \quad
    \mu' \langle g, \hat{T}_A g \rangle_{\mathcal{G}} = \frac{\mu'}{n} G^{\top} G \\
    \frac{1}{n} \vec{V}_{g,h}^{\top} P_C \vec{V}_{g,h}
    = \frac{1}{n} (H^{\top} P_C H - 2 G^{\top} P_C H + G^{\top} P_C G), \quad
    \mu \langle h, \hat{T}_B h \rangle_{\mathcal{H}} = \frac{\mu}{n} H^{\top} H

Combining with :math:`G = \Phi_A g = K_A \alpha` and :math:`H = \Phi_B h = K_B \beta` from the existence of minimizers,

.. math::

    n \mathcal{E}(\alpha, \beta) = Y^{\top} P_{C'} Y - 2 G^{\top} (P_{C'} Y + P_C H)
    + G^{\top} (P_{C'} + P_C + \mu') G + H^{\top} (P_C + \mu) H \\
    = Y^{\top} P_{C'} Y - 2 \alpha^{\top} K_A (P_{C'} Y + P_C K_B \beta)
    + \alpha^{\top} K_A (P_{C'} + P_C + \mu') K_A \alpha \\
    \quad + \beta^{\top} K_B (P_C + \mu) K_B \beta

3. The first order conditions yield

.. math::

    0 = -2 K_A (P_{C'} Y + P_C K_B \hat{\beta}) + 2 K_A (P_{C'} + P_C + \mu') K_A \hat{\alpha} \\
    0 = -2 K_B P_C K_A \hat{\alpha} + 2 K_B (P_C + \mu) K_B \hat{\beta}
    \Longrightarrow \hat{\alpha} = \left( K_B P_C K_A \right)^{\dagger} K_B \left( P_C + \mu \right) K_B \hat{\beta}

4. Substituting the latter into the former,

.. math::

    K_A P_{C'} Y + K_A P_C K_B \hat{\beta} = K_A (P_{C'} + P_C + \mu') K_A \left( K_B P_C K_A \right)^{\dagger} K_B \left( P_C + \mu \right) K_B \hat{\beta}

and solving for :math:`\hat{\beta}`,

.. math::

    \hat{\beta} = \left[ K_A \left\{ - P_C + \left( P_{C'} + P_C + \mu' \right) K_A \left( K_B P_C K_A \right)^{\dagger} K_B \left( P_C + \mu \right) \right\} K_B \right]^{\dagger} K_A P_{C'} Y

Remark (Subsetted estimator)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. admonition:: Formula of minimizers (Subsetted estimator)

    The explicit formula for the coefficients is
    
    .. math::
    
        \hat{\beta} = \left[ K_A \left\{ - \tilde{P}_C + \left( \tilde{P}_{C'} + \tilde{P}_C + \mu' \right) K_A \left( K_B \tilde{P}_C K_A \right)^{\dagger} K_B \left( \tilde{P}_C + \mu \right) \right\} K_B \right]^{\dagger} K_A \tilde{P}_{C'} Y \\
        \hat{\alpha} = \left( K_B \tilde{P}_C K_A \right)^{\dagger} K_B \left( \tilde{P}_C + \mu \right) K_B \hat{\beta}
    
    where :math:`\tilde{P}_{C'} = \frac{n}{p} I_{[p]}^{\top} P_{C';[p,p]} I_{[p]}` and :math:`\tilde{P}_C = \frac{n}{q} I_{[q]}^{\top} P_{C;[q,q]} I_{[q]}`. Note that :math:`P_{C';[p,p]} = (K_{C';[p,p]})^- K_{C';[p,p]}` and :math:`K_{C';[p,p]} = I_{[p]} K_{C'} I_{[p]}^{\top}`.

**Proof**

We proceed in steps.

1. Write the objective :math:`\mathcal{E}(g,h)` as

.. math::

    2 \langle \hat{f}'_{g,h}, \hat{\mu}'_{g,h;[p]} \rangle_{\mathcal{F}} - \langle \hat{f}'_{g,h}, \hat{T}_{C';[p,p]} \hat{f}'_{g,h} \rangle_{\mathcal{F}}
    + \mu' \langle g, \hat{T}_A g \rangle_{\mathcal{G}} \\
    + 2 \langle \hat{f}_{g,h}, \hat{\mu}_{g,h;[q]} \rangle_{\mathcal{F}} - \langle \hat{f}_{g,h}, \hat{T}_{C;[q,q]} \hat{f}_{g,h} \rangle_{\mathcal{F}}
    + \mu \langle h, \hat{T}_B h \rangle_{\mathcal{H}}

where

.. math::

    \hat{\mu}'_{g,h;[p]} = \frac{1}{p} \Phi_{C';[p]}^* \vec{V}'_{g,h;[p]}, \quad
    \hat{\mu}_{g,h;[q]} = \frac{1}{q} \Phi_{C;[q]}^* \vec{V}_{g,h;[q]}

and the covariance operators are defined analogously to the subsetted estimator. Hence,

.. math::

    \mathcal{E}(g,h) = \frac{1}{p} (\vec{V}'_{g,h;[p]})^{\top} P_{C';[p,p]} \vec{V}'_{g,h;[p]}
    + \mu' \langle g, \hat{T}_A g \rangle_{\mathcal{G}} \\
    + \frac{1}{q} \vec{V}_{g,h;[q]}^{\top} P_{C;[q,q]} \vec{V}_{g,h;[q]}
    + \mu \langle h, \hat{T}_B h \rangle_{\mathcal{H}}

2. Let :math:`Y, G, H \in \mathbb{R}^n` be defined with :math:`G_i = g(A_i)` and :math:`H_i = h(B_i)` as before. Now, let :math:`\tilde{P}_{C'} = \frac{n}{p} I_{[p]}^{\top} P_{C';[p,p]} I_{[p]}` and :math:`\tilde{P}_C = \frac{n}{q} I_{[q]}^{\top} P_{C;[q,q]} I_{[q]}`. Then

.. math::

    \frac{1}{p} (\vec{V}'_{g,h;[p]})^{\top} P_{C';[p,p]} \vec{V}'_{g,h;[p]}
    = \frac{1}{n} (Y^{\top} \tilde{P}_{C'} Y - 2 G^{\top} \tilde{P}_{C'} Y + G^{\top} \tilde{P}_{C'} G) \\
    \mu' \langle g, \hat{T}_A g \rangle_{\mathcal{G}} = \frac{\mu'}{n} G^{\top} G \\
    \frac{1}{q} \vec{V}_{g,h;[q]}^{\top} P_{C;[q,q]} \vec{V}_{g,h;[q]}
    = \frac{1}{n} (H^{\top} \tilde{P}_C H - 2 G^{\top} \tilde{P}_C H + G^{\top} \tilde{P}_C G) \\
    \mu \langle h, \hat{T}_B h \rangle_{\mathcal{H}} = \frac{\mu}{n} H^{\top} H

Hereafter we use the same argument as in the formula of minimizers.


Nyström approximation
^^^^^^^^^^^^^^^^^^^^^^

Computation of kernel methods may be demanding due to the inversions of matrices  that scale with :math:`n` such as :math:`K_B \in \mathbb{R}^{n \times n}`. One solution is Nyström approximation. We now provide alternative expressions for the minimizers :math:`(\hat{g}, \hat{h})` that lend themselves to Nyström approximation, then describe the procedure.

.. admonition:: Minimizer sufficient statistics

    The minimizers may be expressed as
    
    .. math::
        \hat{g} = \left(\Phi_B^* P_C \Phi_A\right)^{\dagger} \Phi_B^* (P_C + \mu) \Phi_B \hat{h},
    
    .. math::
        \hat{h} = \left[ \Phi_A^* \left\{ -P_C + \left( P_{C'} + P_C + \mu' \right) \Phi_A \left( \Phi_B^* P_C \Phi_A \right)^{\dagger} \Phi_B^* \left( P_C + \mu \right) \right\} \Phi_B \right]^{\dagger} \Phi_A^* P_{C'} Y.

**Proof**

We proceed in steps.

1. By the proof of the Formula of minimizers of Estimator 3, with :math:`G = \Phi_A g` and :math:`H = \Phi_B h`,

.. math::
    \begin{align*}
        n \mathcal{E}(g, h) &= Y^{\top} P_{C'} Y - 2 G^{\top} (P_{C'} Y + P_C H) \\
        & \quad + G^{\top} (P_{C'} + P_C + \mu') G + H^{\top} (P_C + \mu) H, \\
        &= Y^{\top} P_{C'} Y - 2 g^* \Phi_A^* (P_{C'} Y + P_C \Phi_B h) \\
        & \quad + g^* \Phi_A^* (P_{C'} + P_C + \mu') \Phi_A g + h^* \Phi_B^* (P_C + \mu) \Phi_B h.
    \end{align*}

2. Informally, the first order conditions yield

.. math::
    \begin{align*}
        0 &= -2 \Phi_A^* (P_{C'} Y + P_C \Phi_B \hat{h}) + 2 \Phi_A^* (P_{C'} + P_C + \mu') \Phi_A \hat{g}, \\
        0 &= -2 \Phi_B^* P_C \Phi_A \hat{g} + 2 \Phi_B^* (P_C + \mu) \Phi_B \hat{h}.
    \end{align*}


See `De Vito and Caponnetto (2005) <https://apps.dtic.mil/sti/tr/pdf/ADA466779.pdf>`_ (Proof of Proposition 2) for the formal way of deriving the first order condition, which incurs additional notation.

Rearranging and taking pseudo-inverses, we arrive at two equations:
    
    .. math::
        \Phi_A^* (P_{C'} + P_C + \mu') \Phi_A \hat{g} = \Phi_A^* (P_{C'} Y + P_C \Phi_B \hat{h}),
    
    .. math::
        \Phi_B^* P_C \Phi_A \hat{g} = \Phi_B^* (P_C + \mu) \Phi_B \hat{h} 
        \Longrightarrow \hat{g} = \left(\Phi_B^* P_C \Phi_A \right)^{\dagger} \Phi_B^* (P_C + \mu) \Phi_B \hat{h}.

3. Substituting the latter into the former,

    .. math::
        \Phi_A^* P_{C'} Y + \Phi_A^* P_C \Phi_B \hat{h} = \Phi_A^* (P_{C'} + P_C + \mu') \Phi_A \left(\Phi_B^* P_C \Phi_A \right)^{\dagger} \Phi_B^* (P_C + \mu) \Phi_B \hat{h},
    
    and solving for :math:`\hat{h}`,

    .. math::
        \hat{h} = \left[ \Phi_A^* \left\{ -P_C + \left( P_{C'} + P_C + \mu' \right) \Phi_A \left( \Phi_B^* P_C \Phi_A \right)^{\dagger} \Phi_B^* \left( P_C + \mu \right) \right\} \Phi_B \right]^{\dagger} \Phi_A^* P_{C'} Y. 


Remark (Nyström subsetted estimator)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. admonition:: Formula of minimizers (Subsetted estimator)

    The subsetted minimizers may be expressed as
        
    .. math::
        \hat{g} = \left(\Phi_B^* \tilde{P}_C \Phi_A \right)^{\dagger} \Phi_B^* (\tilde{P}_C + \mu) \Phi_B \hat{h},
        
    .. math::
        \hat{h} = \left[ \Phi_A^* \left\{ -\tilde{P}_C + \left( \tilde{P}_{C'} + \tilde{P}_C + \mu' \right) \Phi_A \left( \Phi_B^* \tilde{P}_C \Phi_A \right)^{\dagger} \Phi_B^* \left( \tilde{P}_C + \mu \right) \right\} \Phi_B \right]^{\dagger} \Phi_A^* \tilde{P}_{C'} Y.

**Proof**

The argument is analogous to the Remark of the properties of pseudo-inverse above.

.. admonition:: Properties of pseudo-inverse

    Continuing the notation of the (Properties of pseudo-inverse), if :math:`\Phi = U \Sigma^{1/2} V^{\top}` and 
    :math:`K = \Phi \Phi^*`, then :math:`P = UU^{\top} = K^{\dagger} K = \Phi \Phi^{\dagger}`. 
   

Combining (Minimizer sufficient statistics) and (Properties of pseudo-inverse), we conclude that sufficient statistics for 
:math:`(\hat{g}, \hat{h})` are feature operators. Within the feature operator :math:`\Phi`, the :math:`i` th row 
:math:`\langle \phi(X_i), \cdot \rangle` may be viewed as an infinite dimensional vector.

Nyström approximation is a way to approximate infinite dimensional vectors with finite dimensional ones. It uses the substitution
:math:`\phi(x) \mapsto \check{\phi}(x) = (K_{\mathcal{S} \mathcal{S}})^{-\frac{1}{2}} K_{\mathcal{S} x}`, where
:math:`\mathcal{S}` is a subset of :math:`s = |\mathcal{S}| \ll n` observations called landmarks. 
:math:`K_{\mathcal{S} \mathcal{S}} \in \mathbb{R}^{s \times s}` is defined such that 
:math:`(K_{\mathcal{S} \mathcal{S}})_{ij} = k(X_i, X_j)` for :math:`i, j \in \mathcal{S}`. Similarly, 
:math:`K_{\mathcal{S} x} \in \mathbb{R}^s` is defined such that :math:`(K_{\mathcal{S} x})_i = k(X_i, x)` 
for :math:`i \in \mathcal{S}`.

In summary, the approximate sufficient statistics are of the form :math:`\check{\Phi} \in \mathbb{R}^{n \times s}`, 
i.e. a matrix whose :math:`i` th row :math:`\langle \check{\phi}(X_i), \cdot \rangle` may be viewed as a vector 
in :math:`\mathbb{R}^s`.


Closed form - Estimator 3 (RKHS norm)
-------------------------------------

We study the RKHS-norm regularized *joint* estimator:

.. math::

    (\hat{g}, \hat{h}) = \arg \min_{g \in \mathcal{G}, h \in \mathcal{H}} 
    \max_{f' \in \mathcal{F}} \mathbb{E}_n \left[ 2 \left\{ g(A) - Y \right\} f'(C') - f'(C')^2 \right] -\lambda'\|f'\|_\mathcal{F'}^2
    + \mu'  \| g \|_{\mathcal{G}}^2 \\
    \quad + \max_{f \in \mathcal{F}} \mathbb{E}_n \left[ 2 \left\{ h(B) - g(A) \right\} f(C) - f(C)^2 \right] -\lambda\|f\|_\mathcal{F}^2
    + \mu \| h \|_{\mathcal{H}}^2


.. admonition:: Formula of minimizers 

    
    The minimizer takes the form :math:`\hat{g} = \Phi_A^*\hat\alpha`, :math:`\hat{h} = \Phi_B^*\hat\beta` where,
    
    .. math::
    
        \hat{\beta} &= \left[ K_A \left\{ - P_C + \left(P_{C'} K_A + P_C K_A + \mu'\right) \left( K_B P_C K_A \right)^{\dagger} \left( K_B P_C + \mu  \right)\right\} K_B \right]^{\dagger} K_A P_{C'} Y \\
        \hat{\alpha} &= \left( K_B P_C K_A \right)^{\dagger} \left( K_B P_C + \mu \right) K_B \hat{\beta}
    
    and
    
    .. math::
    
        P_C &= \left(K_C+\lambda\right)^{\dagger}K_C \\
        P_{C'} &= \left(K_{C'}+\lambda'\right)^{\dagger}K_{C'}


.. autosummary::
   :toctree: _autosummary
   :template: class.rst

   rkhs2iv.RKHS2IV
   rkhs2iv.RKHS2IVCV

Remark: Subsetted estimator
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. admonition:: Formula of minimizers (Subsetted estimator)
    
    The subsetted estimator satisfies:
    
    .. math::
    
        \hat{\beta} &= \left[ K_A \left\{ - \tilde{P}_C + \left(\tilde{P}_{C'} K_A + \tilde{P}_C K_A + \mu'\right) \left( K_B \tilde{P}_C K_A \right)^{\dagger} \left( K_B \tilde{P}_C + \mu  \right)\right\} K_B \right]^{\dagger} K_A \tilde{P}_{C'} Y \\
        \hat{\alpha} &= \left( K_B \tilde{P}_C K_A \right)^{\dagger} \left( K_B \tilde{P}_C + \mu \right) K_B \hat{\beta}
    
    with :math:`\tilde{P}_{C'}=\frac{n}{p}I_{[p]}^{\top}P_{C';[p,p]}I_{[p]}` and :math:`\tilde{P}_{C}=\frac{n}{q}I_{[q]}^{\top}P_{C;[q,q]}I_{[q]}`. And
    
    .. math::
    
        P_{C';[p,p]}&=(K_{C';[p,p]}+\lambda I_{[p]}I_{[p]}^\top)^-K_{C';[p,p]}\;, \qquad K_{C';[p,p]}=I_{[p]}K_{C'}I_{[p]}^{\top} \\
        P_{C;[q,q]}&=(K_{C;[q,q]}+\lambda I_{[q]}I_{[q]}^\top)^-K_{C;[q,q]}\;, \qquad K_{C;[q,q]}=I_{[q]}K_{C}I_{[q]}^{\top}
    
