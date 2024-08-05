Random Forest
=============

Let

.. math::

    \operatorname{Oracle}_{\mathcal{F},\text{reg}}\left(\{z_i\},\{u_i\}\right) &= \operatorname{argmin}_{f\in\mathcal{F}}\frac{1}{n}\sum^n_{i=1}\left(u_i-f(z_i)\right)^2 \\
    \operatorname{Oracle}_{\mathcal{F},\text{class}}\left(\{x_i\},\{v_i\}, \{w_i\}\right) &= \operatorname{argmax}_{f\in\mathcal{F}}\frac{1}{n}\sum^n_{i=1} w_i \Pr_{Z_i\sim\operatorname{Ber}\left(\frac{1+f(x_i)}{2}\right)}\left(Z_i = v_i \right)

be oracles for the regression and (weighted) classification problems. For data :math:`A = \{a_1,\ldots, a_n\}`, define :math:`\mathcal{F}_A = \left\{\left(f\left(a_1\right), \ldots, f\left(a_n\right)\right): f \in \mathcal{F}\right\}`.

Estimator 1
-----------

Whenever the function classes :math:`\mathcal{G}`, :math:`\mathcal{F'}` are already norm constrained, the estimator can be reduced to

.. math::

    \hat{g} = \arg \min_{g\in\mathcal{G}} 
    \max_{f' \in \mathcal{F'}} \mathbb{E}_n\left[2\left\{g(A)-Y\right\} f'(C')-f'(C')^2\right]

.. admonition:: Ensemble solution

    Consider the algorithm where for :math:`t=1, \ldots, T`:

    .. math::

        \begin{aligned}
        & u_i^t=\left(y_i-\frac{1}{t-1} \sum_{\tau=1}^{t-1} g_\tau\left(a_i\right)\right), \quad f'_t=\operatorname{Oracle}_{\mathcal{F'}, \text{reg}}\left(\{c_i'\}, \{u_i^t\}\right) \\
        & v_i^t=1\left\{f'_t\left(c_i'\right)>0\right\} , w_i^t=\left|f'_t\left(c_i'\right)\right| \quad g_t=\operatorname{Oracle}_{\mathcal{G}, \text{class}}\left(\{a_i\}, \{v_i^t\}, \{w_i^t\}\right) \\
        &
        \end{aligned}

    Suppose that the set :math:`\mathcal{F'}_{C'}` is a convex set. Then the ensemble :math:`\bar{g}=\frac{1}{T} \sum_{t=1}^T g_t`, is a :math:`O\left(\frac{\log (T)+1}{T}\right)`-approximate solution to the minimax problem.

.. autosummary::
   :toctree: _autosummary
   :template: class.rst

   ensemble.EnsembleIV
   ensemble.EnsembleIVStar

Estimator 2
-----------

For the estimator

.. math::

    \hat{g} = \arg \min_{g\in\mathcal{G}} 
    \max_{f' \in \mathcal{F'}} \mathbb{E}_n\left[2\left\{g(A)-Y\right\} f'(C')-f'(C')^2\right]+\mu'\mathbb{E}_n\{g(A)^2\}

.. admonition:: Ensemble solution

    Consider the algorithm where for :math:`t=1, \ldots, T`:

    .. math::

        \begin{aligned}
        & u_i^t=\left(y_i-\frac{1}{t-1} \sum_{\tau=1}^{t-1} g_\tau\left(a_i\right)\right), \quad f'_t=\operatorname{Oracle}_{\mathcal{F'}, \text{reg}}\left(\{c_i'\}, \{u_i^t\}\right) \\
        & v_i^t=\frac{1}{\mu' t}\sum_{\tau=1}^{t}f'_\tau(c_i'), \qquad \qquad \qquad g_t=\operatorname{Oracle}_{\mathcal{G}, \text{reg}}\left(\{a_i\}, \{v_i^t\}\right) \\
        &
        \end{aligned}

    Suppose that the sets :math:`\mathcal{F'}_{C'}`, :math:`\mathcal{G}_{A}` are convex. Then the ensemble: :math:`\bar{g}=\frac{1}{T} \sum_{t=1}^T g_t`, is a :math:`O\left(\frac{\log (T)+1}{T}\right)`-approximate solution to the minimax problem.

.. autosummary::
   :toctree: _autosummary
   :template: class.rst

   ensemble.EnsembleIVL2

Estimator 3
-----------

For the joint estimator with ridge regularization

.. math::

    (\hat{g},\hat{h}) = \arg \min_{g\in\mathcal{G}, h \in \mathcal{H}} 
    \max_{f' \in \mathcal{F}} \mathbb{E}_n \left[ 2 \left\{ g(A) - Y \right\} f'(C') - f'(C')^2 \right]
     + \mu' \mathbb{E}_n \left\{ g(A)^2 \right\} \\
    + \max_{f \in \mathcal{F}} \mathbb{E}_n \left[ 2 \left\{ h(B) - g(A) \right\} f(C) - f(C)^2 \right] 
     + \mu \mathbb{E}_n \left\{ h(B)^2 \right\}


.. admonition:: Ensemble solution

    Consider the algorithm where for :math:`t=1, \ldots, T`:

    .. math::

        \begin{aligned}
        & u_i'^{t}=\left(y_i-\frac{1}{t-1} \sum_{\tau=1}^{t-1} g_\tau\left(a_i\right)\right), \quad u_i^t=\frac{1}{t-1} \sum_{\tau=1}^{t-1} \bigg(g_\tau\left(a_i\right)-h_\tau\left(b_i\right)\bigg)\\
        & f'_t=\operatorname{Oracle}_{\mathcal{F'}, \text{reg}}\left(\{c_i'\}, \{u_i'^t\}\right),\quad f_t=\operatorname{Oracle}_{\mathcal{F}, \text{reg}}\left(\{c_i\}, \{u_i^t\}\right) \\
        & v_i'^t=\frac{1}{\mu't}\sum_{\tau=1}^{t}\bigg(f'_\tau(c'_i)-f_\tau(c_i)\bigg), \quad  \qquad  v_i^t=\frac{1}{\mu t}\sum_{\tau=1}^{t}f_\tau(c_i)\\
        &g_t=\operatorname{Oracle}_{\mathcal{G}, \text{reg}}\left(\{a_i\}, \{v_i'^t\}\right),  \qquad   h_t=\operatorname{Oracle}_{\mathcal{H}, \text{reg}}\left(\{b_i\}, \{v_i^t\}\right) \\
        &
        \end{aligned}

    Suppose that the sets :math:`\mathcal{F'}_{C'}`, :math:`\mathcal{F}_{C}`, :math:`\mathcal{G}_{A}`, :math:`\mathcal{H}_{B}` are all convex sets. Then the ensembles: :math:`\bar{g}=\frac{1}{T} \sum_{t=1}^T g_t`, :math:`\bar{h}=\frac{1}{T} \sum_{t=1}^T h_t`, are a :math:`O\left(\frac{\log (T)+1}{T}\right)`-approximate solution to the minimax problem.

.. autosummary::
   :toctree: _autosummary
   :template: class.rst

   ensemble2.Ensemble2IVL2

**Proof**

We can write the minimax problem as a convex-concave zero-sum game:

.. math::

    \min_{b\in \mathcal{H}_B, a\in \mathcal{G}_A}\max_{c\in \mathcal{F}_C, c'\in \mathcal{F'}_{C'}}
    \frac{1}{n}\sum_{i=1}^{n}2(Y_i-a_i)c'_i-c_i'^2+2(a_i-b_i)c_i-c_i^2+\mu'a_i^2+\mu b_i^2

.. math::

    = \max_{b\in \mathcal{H}_B, a\in \mathcal{G}_A}\min_{c\in \mathcal{F}_C, c'\in \mathcal{F'}_{C'}}
    \frac{1}{n} \underbrace{\sum_{i=1}^{n}c_i'^2-2(Y_i-a_i)c'_i+c_i^2-2(a_i-b_i)c_i-\mu'a_i^2-\mu b_i^2}_{:=\ell(\{c,c'\},\{a,b\})}

where the loss :math:`\ell(\{c,c'\},\{a,b\})` is convex in :math:`\{c,c'\}` and concave in :math:`\{a,b\}`.

The adversary chooses vector :math:`(c_i,c_i')` based on *follow-the-leader* (FTL):

.. math::

    \{c_t,c_t'\} = \operatorname{argmin}_{c\in \mathcal{F}_C, c'\in \mathcal{F'}_{C'}}
    \frac{1}{t-1}\sum_{\tau=1}^{t-1}\ell(\{c,c'\},\{a_\tau,b_\tau\})

by separating the minimization and completing the square, we have that

.. math::

    c_t &= \operatorname{argmin}_{c \in \mathcal{F}_C} \frac{1}{n} \sum_{i=1}^{n} \left( c_i - \frac{1}{t-1} \sum_{\tau=1}^{t-1} \left\{ a_{i\tau} - b_{i\tau} \right\} \right)^2 \\
        &= \operatorname{argmin}_{c \in \mathcal{F}_C} \frac{1}{n} \sum_{i=1}^{n} \left( c_i - u_i^{t} \right)^2 \\
        &= \operatorname{Oracle}_{\mathcal{F}, \text{reg}} \left( \{c_i\}, \{u_i^t\} \right)

.. math::

    c'_t &= \operatorname{argmin}_{c' \in \mathcal{F'}_{C'}} \frac{1}{n} \sum_{i=1}^{n} \left( c_i' - \frac{1}{t-1} \sum_{\tau=1}^{t-1} \left\{ y_i - a_{i\tau} \right\} \right)^2 \\
         &= \operatorname{argmin}_{c' \in \mathcal{F'}_{C'}} \frac{1}{n} \sum_{i=1}^{n} \left( c_i' - u_i'^{t} \right)^2 \\
         &= \operatorname{Oracle}_{\mathcal{F'}, \text{reg}} \left( \{c_i'\}, \{u_i'^t\} \right)


Now, the learner plays *be-the-leader* (BTL) which involves choosing :math:`(a_t,b_t)` that best responds

.. math::

    \{a_t,b_t\} = \operatorname{argmax}_{a\in \mathcal{G}_A, b\in \mathcal{H}_{B}}
    \frac{1}{t}\sum_{\tau=1}^{t}\ell(\{c_\tau,c'_\tau\},\{a,b\})

which after separating the minimization problem and completing the square we get:

.. math::

    a_t &= \operatorname{argmin}_{a \in \mathcal{G}_A} \frac{1}{n} \sum_{i=1}^{n} 
    \left( a_i - \frac{1}{\mu' t} \sum_{\tau=1}^{t} \left\{ c'_{i\tau} - c_{i\tau} \right\} \right)^2 \\
    &= \operatorname{argmin}_{a \in \mathcal{G}_A} \frac{1}{n} \sum_{i=1}^{n} \left( a_i - v_i'^{t} \right)^2 \\
    &= \operatorname{Oracle}_{\mathcal{G}, \text{reg}} \left( \{a_i\}, \{v_i'^t\} \right)

.. math::

    b_t &= \operatorname{argmin}_{b \in \mathcal{H}_{B}} \frac{1}{n} \sum_{i=1}^{n} 
    \left( b_i - \frac{1}{\mu t} \sum_{\tau=1}^{t} c_{i\tau} \right)^2 \\
    &= \operatorname{argmin}_{b \in \mathcal{H}_{B}} \frac{1}{n} \sum_{i=1}^{n} \left( b_i - v_i^{t} \right)^2 \\
    &= \operatorname{Oracle}_{\mathcal{H}, \text{reg}} \left( \{b_i\}, \{v_i^t\} \right)


Thus it remains to show that the ensembles

.. math::

    \bar{a} = \frac{1}{T}\sum_{t=1}^{T}a_t\,,\qquad \bar{b} = \frac{1}{T}\sum_{t=1}^{T}b_t

are also a solution to the empirical minimax problem.

Observe that the learner has zero regret, since it is playing the BTL algorithm. Thus if we show that the FTL algorithm has :math:`\operatorname{Regret}(T)`-regret after :math:`T` periods, then :math:`(\bar{a},\bar{b})` is an :math:`\epsilon(T)` approximate solution to the minimax problem, invoking the results of `Freund and Schapire (1999) <https://www.sciencedirect.com/science/article/pii/S0899825699907388>`_.

Hence, we now focus on the online learning problem that the adversary is facing and show that FTL is a no-regret algorithm with regret rate :math:`\operatorname{Regret}(T)=O\left(\frac{\log (T)}{T}\right)`. We can upper bound the regret of the FTL algorithm by:

.. math::

    \operatorname{Regret}(T) \leq \frac{1}{T} \sum_{t=1}^T\bigg(\ell\left(\{c_t, c'_t\}, \{a_t, b_t\}\right)-\ell\left(\{c_{t+1}, c'_{t+1}\}, \{a_t, b_t\}\right)\bigg)

The loss :math:`\ell\left(\cdot, \{a,b\}\right)` is :math:`\frac{2}{n}`-strongly convex with respect to the :math:`\|\cdot\|_2`-norm on :math:`C\times C'`. Moreover the loss is :math:`O\left(\frac{1}{\sqrt{n}}\right)`-Lipschitz, since

.. math::

    \nabla_{\{c,c'\}}\ell\left(\{c,c'\}, \{a,b\}\right) = \frac{2}{n}\left(\{c,c'\} - \{y-a,a-b\}\right)

so 

.. math::

    \|\nabla_{\{c,c'\}}\ell\left(\{c,c'\}, \{a,b\}\right)\|_2 
    &= \frac{2}{n}\sqrt{\sum_{i=1}^{n}\left(c_i-(y_i-a_i)+c_i'-(a_i-b_i)\right)^2}  \\ 
    &\leq \frac{2}{n}\left(\|c\|_2+\|y\|_2+\|a\|_2+\|c'\|_2+\|a\|_2+\|b\|_2\right) \\
    &\leq O\left(\frac{1}{\sqrt{n}}\right)

Then :math:`L_t = \sum_{\tau=1}^t \ell(\cdot, \{a_\tau, b_\tau\})` is :math:`\frac{2t}{n}`-strongly convex. Since :math:`\{c_{t+1}, c'_{t+1}\}` is a minimizer of :math:`L_t` and the set :math:`C\times C'` is convex, we have by the strong convexity and the first order condition that 

.. math::

    L_t(\{c_t,c'_t\}) &\geq L_t(\{c_{t+1},c'_{t+1}\}) + \left\langle \{c_{t},c'_{t}\}-\{c_{t+1},c'_{t+1}\}, \nabla_{\{c,c'\}}L_t(\{c_{t+1},c'_{t+1}\})\right\rangle + \frac{t}{n}\left\|\{c_{t},c'_{t}\},\{c_{t+1},c'_{t+1}\}\right\|_2^2 \\
    &\geq  L_t(\{c_{t+1},c'_{t+1}\})+ \frac{t}{n}\left\|\{c_{t},c'_{t}\},\{c_{t+1},c'_{t+1}\}\right\|_2^2

and

.. math::

    L_{t-1}(\{c_{t+1},c'_{t+1}\}) \geq L_{t-1}(\{c_{t},c'_{t}\})+ \frac{t}{n}\left\|\{c_{t},c'_{t}\},\{c_{t+1},c'_{t+1}\}\right\|_2^2

Adding the two previous equations and re-arranging gives:

.. math::

    \ell(\{c_t, c'_t\}, \{a_t, b_t\})-\ell(\{c_{t+1}, c'_{t+1}\}, \{a_{t}, b_{t}\})\geq \frac{2t}{n}\left\|\{c_{t},c'_{t}\},\{c_{t+1},c'_{t+1}\}\right\|_2^2

Invoking the Lipschitzness of :math:`\ell_t`:

.. math::

    \frac{K}{\sqrt{n}}\left\|\{c_{t},c'_{t}\},\{c_{t+1},c'_{t+1}\}\right\|_2\geq \frac{2t}{n}\left\|\{c_{t},c'_{t}\},\{c_{t+1},c'_{t+1}\}\right\|_2^2

so that

.. math::

    \left\|\{c_{t},c'_{t}\},\{c_{t+1},c'_{t+1}\}\right\|_2\leq \frac{K}{2}\frac{\sqrt{n}}{t}

Finally,

.. math::

    \operatorname{Regret}(T) &\leq \frac{1}{T} \sum_{t=1}^T \left( \ell\left(\{c_t, c'_t\}, \{a_t, b_t\}\right) 
    - \ell\left(\{c_{t+1}, c'_{t+1}\}, \{a_t, b_t\}\right) \right) \\
   & \leq \frac{1}{T} \sum_{t=1}^T \frac{K}{\sqrt{n}} \left\|\{c_{t}, c'_{t}\}, \{c_{t+1}, c'_{t+1}\}\right\|_2 \\
   & \leq \frac{1}{T} \sum_{t=1}^T \frac{K^2}{2} \frac{1}{t} \\
   & \leq K^2 \frac{\log T + 1}{T}



Subsetted estimator
^^^^^^^^^^^^^^^^^^^

For the subsetted estimator

.. math::

    (\hat{g},\hat{h}) = \arg \min _{g\in\mathcal{G}, h \in \mathcal{H}} 
    \max_{f' \in \mathcal{F}} \mathbb{E}_p\left[2\left\{g(A)-Y\right\} f'(C')-f'(C')^2\right]
    + \mu'\mathbb{E}_n\{g(A)^2\} \\
    + \max_{f \in \mathcal{F}} \mathbb{E}_q\left[2\left\{h(B)-g(A)\right\} f(C)-f(C)^2\right]   
    + \mu\mathbb{E}_n\{h(B)^2\}

We simply modify the updates for :math:`v_t', v_t` as

.. math::

    v_i'^t = \frac{1}{\mu't}\sum_{\tau=1}^{t}\bigg(f'_\tau(c'_i)1\big(i\in[p]\big)-f_\tau(c_i)1\big(i\in[q]\big)\bigg), \quad
    v_i^t = \frac{1}{\mu t}\sum_{\tau=1}^{t}f_\tau(c_i)1\big(i\in[q]\big)




Estimator 3 - (Function class bounded)
--------------------------------------

For the joint estimator:

.. math::

    (\hat{g}, \hat{h}) = \arg \min _{g \in \mathcal{G}, h \in \mathcal{H}} 
    \max_{f' \in \mathcal{F}} \mathbb{E}_n \left[ 2 \left\{ g(A) - Y \right\} f'(C') - f'(C')^2 \right] \\
    + \max_{f \in \mathcal{F}} \mathbb{E}_n \left[ 2 \left\{ h(B) - g(A) \right\} f(C) - f(C)^2 \right]

.. admonition:: Ensemble solution
    :class: lemma
    :name: lemma-ensemble-3-norm

    Consider the algorithm where for :math:`t=1, \ldots, T`:

    .. math::

        \begin{aligned}
        & u_i'^{t}=\left(y_i-\frac{1}{t-1} \sum_{\tau=1}^{t-1} g_\tau\left(a_i\right)\right), \quad u_i^t=\frac{1}{t-1} \sum_{\tau=1}^{t-1} \bigg(g_\tau\left(a_i\right)-h_\tau\left(b_i\right)\bigg)\\
        & f'_t=\operatorname{Oracle}_{\mathcal{F'}, \text{reg}}\left(\{c_i'\}, \{u_i'^t\}\right),\quad f_t=\operatorname{Oracle}_{\mathcal{F}, \text{reg}}\left(\{c_i\}, \{u_i^t\}\right) \\
        & v_i'^t=1\big(f'_t(c'_i)-f_t(c_i)>0\big),   \qquad  v_i^t=1\big(f_t(c_i)>0\big)\\
        & w_i'^t=\big|f'_t(c'_i)-f_t(c_i)\big|, \qquad  \qquad  w_i^t=|f_t(c_i)|\\
        &g_t=\operatorname{Oracle}_{\mathcal{G}, \text{class}}\left(\{a_i\}, \{v_i'^t\}, \{w_i'^t\}\right),  \quad   h_t=\operatorname{Oracle}_{\mathcal{H}, \text{class}}\left(\{b_i\}, \{v_i^t\}, \{w_i^t\}\right) \\
        &
        \end{aligned}

    Suppose that the sets :math:`\mathcal{F'}_{C'}`, :math:`\mathcal{F}_{C}` are convex sets. Then the ensembles: :math:`\bar{g} = \frac{1}{T} \sum_{t=1}^T g_t`, :math:`\bar{h} = \frac{1}{T} \sum_{t=1}^T h_t`, are a :math:`O \left( \frac{\log (T) + 1}{T} \right)`-approximate solution to the minimax problem.

.. autosummary::
   :toctree: _autosummary
   :template: class.rst

   ensemble2.Ensemble2IV

**Proof**

The proof is analogous to Lemma :ref:`lemma-ensemble-3`, except that the learner best-responds to the current test function:

.. math::

    \begin{aligned}
    \{a_t, b_t\} &= \operatorname{argmax}_{a \in \mathcal{G}_A, b \in \mathcal{H}_{B}} \ell(\{c_t, c'_t\}, \{a, b\}) \\
    &= \operatorname{argmax}_{a \in \mathcal{G}_A, b \in \mathcal{H}_{B}} \sum_{i=1}^{n} c_{it}'^2 - 2(Y_i - a_i) c'_{it} + c_{it}^2 - 2(a_i - b_i) c_{it}
    \end{aligned}

which gives:

.. math::

    \begin{aligned}
    a_t &= \operatorname{argmax}_{a \in \mathcal{G}_{A}} \sum_{i=1}^{n} a_i (c'_{it} - c_{it}) \\
    &= \operatorname{argmax}_{a \in \mathcal{G}_{A}} \frac{1}{n} \sum_i a_i \left| c'_{it} - c_{it} \right| \operatorname{sign} \left( c'_{it} - c_{it} \right) \\
    &= \operatorname{argmax}_{a \in \mathcal{G}_{A}} \frac{1}{n} \sum_i \left| c'_{it} - c_{it} \right| \mathbb{E}_{z \sim \operatorname{Bernoulli} \left( \frac{a_i + 1}{2} \right)} \left[ \left( 2 z_i - 1 \right) \operatorname{sign} \left( c'_{it} - c_{it} \right) \right] \\
    &= \operatorname{argmax}_{a \in \mathcal{G}_{A}} \frac{1}{n} \sum_i \left| c'_{it} - c_{it} \right| \left( \operatorname{Pr}_{z \sim \operatorname{Bernoulli} \left( \frac{a_i + 1}{2} \right)} \left[ \left( 2 z_i - 1 \right) = \operatorname{sign} \left( c'_{it} - c_{it} \right) \right] \right. \\
    & \qquad \qquad \qquad \left. - \operatorname{Pr}_{z \sim \operatorname{Bernoulli} \left( \frac{a_i + 1}{2} \right)} \left[ \left( 2 z_i - 1 \right) \neq \operatorname{sign} \left( c'_{it} - c_{it} \right) \right] \right) \\
    &= \operatorname{argmax}_{a \in \mathcal{G}_{A}} \frac{1}{n} \sum_i \left| c'_{it} - c_{it} \right| \left( 2 \operatorname{Pr}_{z \sim \operatorname{Bernoulli} \left( \frac{a_i + 1}{2} \right)} \left[ \left( 2 z_i - 1 \right) = \operatorname{sign} \left( c'_{it} - c_{it} \right) \right] - 1 \right) \\
    &= \operatorname{argmax}_{a \in \mathcal{G}_{A}} \frac{1}{n} \sum_i \left| c'_{it} - c_{it} \right| \operatorname{Pr}_{z \sim \operatorname{Bernoulli} \left( \frac{a_i + 1}{2} \right)} \left[ \left( 2 z_i - 1 \right) = \operatorname{sign} \left( c'_{it} - c_{it} \right) \right] \\
    &= \operatorname{argmax}_{a \in \mathcal{G}_{A}} \frac{1}{n} \sum_i \left| c'_{it} - c_{it} \right| \operatorname{Pr}_{z \sim \operatorname{Bernoulli} \left( \frac{a_i + 1}{2} \right)} \left[ z_i = \frac{\operatorname{sign} \left( c'_{it} - c_{it} \right) + 1}{2} \right] \\
    &= \operatorname{argmax}_{a \in \mathcal{G}_{A}} \frac{1}{n} \sum_i \left| c'_{it} - c_{it} \right| \operatorname{Pr}_{z \sim \operatorname{Bernoulli} \left( \frac{a_i + 1}{2} \right)} \left[ z_i = 1 \left\{ c'_{it} - c_{it} > 0 \right\} \right] \\
    &= \operatorname{argmax}_{a \in \mathcal{G}_{A}} \frac{1}{n} \sum_i w_{it} \operatorname{Pr}_{z \sim \operatorname{Bernoulli} \left( \frac{a_i + 1}{2} \right)} \left[ z_i = v_{it} \right] \\
    &= \operatorname{Oracle}_{\mathcal{G}, \text{class}} \left( \{a_i\}, \{v_i'^t\}, \{w_i'^t\} \right)
    \end{aligned}

and

.. math::

    \begin{aligned}
    b_t &= \operatorname{argmax}_{b \in \mathcal{H}_{B}} \sum_{i=1}^{n} b_i c_{it} \\
    &= \operatorname{argmax}_{b \in \mathcal{H}_{B}} \frac{1}{n} \sum_i b_i \left| c_{it} \right| \operatorname{sign} \left( c_{it} \right) \\
    &= \operatorname{argmax}_{b \in \mathcal{H}_{B}} \frac{1}{n} \sum_i \left| c_{it} \right| \mathbb{E}_{z \sim \operatorname{Bernoulli} \left( \frac{b_i + 1}{2} \right)} \left[ \left( 2 z_i - 1 \right) \operatorname{sign} \left( c_{it} \right) \right] \\
    &= \operatorname{argmax}_{b \in \mathcal{H}_{B}} \frac{1}{n} \sum_i \left| c_{it} \right| \left( \operatorname{Pr}_{z \sim \operatorname{Bernoulli} \left( \frac{b_i + 1}{2} \right)} \left[ \left( 2 z_i - 1 \right) = \operatorname{sign} \left( c_{it} \right) \right] \right. \\
    & \qquad \qquad \qquad \left. - \operatorname{Pr}_{z \sim \operatorname{Bernoulli} \left( \frac{b_i + 1}{2} \right)} \left[ \left( 2 z_i - 1 \right) \neq \operatorname{sign} \left( c_{it} \right) \right] \right) \\
    &= \operatorname{argmax}_{b \in \mathcal{H}_{B}} \frac{1}{n} \sum_i \left| c_{it} \right| \left( 2 \operatorname{Pr}_{z \sim \operatorname{Bernoulli} \left( \frac{b_i + 1}{2} \right)} \left[ \left( 2 z_i - 1 \right) = \operatorname{sign} \left( c_{it} \right) \right] - 1 \right) \\
    &= \operatorname{argmax}_{b \in \mathcal{H}_{B}} \frac{1}{n} \sum_i \left| c_{it} \right| \operatorname{Pr}_{z \sim \operatorname{Bernoulli} \left( \frac{b_i + 1}{2} \right)} \left[ \left( 2 z_i - 1 \right) = \operatorname{sign} \left( c_{it} \right) \right] \\
    &= \operatorname{argmax}_{b \in \mathcal{H}_{B}} \frac{1}{n} \sum_i \left| c_{it} \right| \operatorname{Pr}_{z \sim \operatorname{Bernoulli} \left( \frac{b_i + 1}{2} \right)} \left[ z_i = \frac{\operatorname{sign} \left( c_{it} \right) + 1}{2} \right] \\
    &= \operatorname{argmax}_{b \in \mathcal{H}_{B}} \frac{1}{n} \sum_i \left| c_{it} \right| \operatorname{Pr}_{z \sim \operatorname{Bernoulli} \left( \frac{b_i + 1}{2} \right)} \left[ z_i = 1 \left\{ c_{it} > 0 \right\} \right] \\
    &= \operatorname{argmax}_{b \in \mathcal{H}_{B}} \frac{1}{n} \sum_i w_{it} \operatorname{Pr}_{z \sim \operatorname{Bernoulli} \left( \frac{b_i + 1}{2} \right)} \left[ z_i = v_{it} \right] \\
    &= \operatorname{Oracle}_{\mathcal{H}, \text{class}} \left( \{b_i\}, \{v_i^t\}, \{w_i^t\} \right)
    \end{aligned}

Subsetted Estimator
^^^^^^^^^^^^^^^^^^^

For the subsetted estimator:

.. math::

    \begin{aligned}
    (\hat{g}, \hat{h}) = \arg \min _{g \in \mathcal{G}, h \in \mathcal{H}} 
    \max_{f' \in \mathcal{F}} \mathbb{E}_p \left[ 2 \left\{ g(A) - Y \right\} f'(C') - f'(C')^2 \right] \\
    + \max_{f \in \mathcal{F}} \mathbb{E}_q \left[ 2 \left\{ h(B) - g(A) \right\} f(C) - f(C)^2 \right]
    \end{aligned}

We simply modify the updates for :math:`(w_t', w_t)` as:

.. math::

    \begin{aligned}
    w_i'^t &= \left| f'_t(c'_i) 1\left( i \in [p] \right) - f_t(c_i) 1\left( i \in [q] \right) \right|, \\
    w_i^t &= \left| f_t(c_i) 1\left( i \in [q] \right) \right|
    \end{aligned}


