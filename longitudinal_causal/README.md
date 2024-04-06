# Simulations for Nested NPIV



Recall the NPIV conditional mean model \cite{newey_powell} of the form
\[Y = g_0(A_1)+\epsilon\;,\quad \E[\epsilon\;|\;A_2] = 0\]
The natural extension to the nested NPIV can be generically written as
\begin{align}
\label{npiv_1}
    Y =& g_0(A_1)+\epsilon_1\;,\quad \E[\epsilon_1\;|\;A_2] = 0 \\
\label{npiv_2}    
    g_0(A_1)=&h_0(B_1)+\epsilon_2\;,\quad \E[\epsilon_2\;|\;B_2] = 0 
\end{align}




In nonparametric instrumental variable regression, $A_1, B_1$ corresponds to the endogenous variables and $A_2, B_2$ corresponds to the instruments. In proximal causal inference, $A_2$ includes the negative control outcome while $B_2$ includes the negative control treatment. In the case of proximal mediation analysis, $A_2 = (D,M,X,Z)$, $A_1=(D,M,X,W)$, $B_2 = (X,Z)$, $B_1 = (X,W)$.

\subsection{Estimation}

We consider estimation of $g_0$ and $h_0$ sequentially using the adversarial estimation of \cite{dikkala}. We forked the AdversarialGMM repository and modify it accordingly for estimation of nuisance functions. 

\begin{algorithm}[H]
\SetKwInOut{Input}{input}\SetKwInOut{Output}{output}

\caption{Two-step Adversarial Estimation}\label{nested_npiv_alg}

\Input{Given generic function spaces $(\mathcal{G},\mathcal{H},\mathcal{F})$, regularization parameters $(\lambda, \mu)$, and theoretical quantities $(U,\delta)$ as defined in \cite{singh2022finite}}
\Output{Estimators $\hat{g_0}$, and $\hat{h_0}$}
\BlankLine

\textbf{Step 1 :}\;
\[\hat{g_0} = \operatorname{argmin}_{g\in \mathcal{G}}\operatorname{sup}_{f\in \mathcal{F}}\E_n[(Y-g(A_1))f(A_2)]-\lambda\left(||f||^2_{\mathcal{F}}+\frac{U}{\delta^2}||f||^2_{2,n}\right)+\mu||g||^2_{g\in\mathcal{G}}\]

\textbf{Step 2 :}\;
\[\hat{h_0} = \operatorname{argmin}_{h\in \mathcal{H}}\operatorname{sup}_{f\in \mathcal{F}}\E_n[(g_0(A_1)-h(B_1))f(B_2)]-\lambda\left(||f||^2_{\mathcal{F}}+\frac{U}{\delta^2}||f||^2_{2,n}\right)+\mu||h||^2_{h\in\mathcal{H}}\]

\end{algorithm}