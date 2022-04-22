# Methods
!!! warning "Under Construction"

In this section we give an overview of the models and methods underlying the statistics and computations performed in `PyMSM`.  

## Introduction

`PyMSM` fits and predicts based on a multistate model supplied by the user.   
Some examples of multistate models:  

1) A simple competing-risk form with three states, $A,B,C$ and two possible transitions, $A \rightarrow B$ and $A \rightarrow C$ .  

``` mermaid
stateDiagram-v2
    A --> B
    A --> C
```

2) An illness-death model with the same three states but three possible transitions $A \rightarrow B$, $A \rightarrow C$ and $B \rightarrow C$ . 

``` mermaid
stateDiagram-v2
    A --> B
    A --> C
    B --> C
```

3) Or any other more involved multistate framework (add figures of these models). 


The description of the content of `PyMSM` would be easier to digest under a certain setting.  Thus, to set the stage, we adopt the multistate model of [Roimi et. al. (2021)](https://academic.oup.com/jamia/article/28/6/1188/6105188). Specifically, assume a multi-state model consists of four states $A,B,C,D$ and six possible transitions:
$$
A \rightarrow B \,\,\,\,\,\,       A \rightarrow C   \,\,\,\,\,\,     A \rightarrow D   \,\,\,\,\,\,    B \rightarrow A \,\,\,\,\,\,    B \rightarrow D \,\,\,\,\,\,   C \rightarrow A \, .
$$

``` mermaid
stateDiagram-v2
    A --> B
    A --> C
    A --> D
    B --> A
    B --> D
    C --> A
```
Each transition is characterizes by a transition-specific hazard function, also known as a cause-specific hazard function,
$$
\lambda_{A,B} (t|Z) \,\,\, \lambda_{A,C} (t|Z) \,\,\, 	\lambda_{A,D} (t|Z) \,\,\, \lambda_{B,A} (t|Z)  \,\,\, \lambda_{B,D} (t|Z) \,\,\,  \lambda_{C,A} (t|Z) \,
$$
for $t > 0$ and $Z$ vector of covariates. Although $Z$ is shared by the six models above,  it does not imply that identical covariates must be used in these models. For example, in Cox models with   transition-dependent   regression coefficient vectors,  one can set any specific coefficient to 0 for excluding  the corresponding covariate.  


Let $J_C$ and $J_N$ denote the current and next states, respectively, and $T$ denotes the transition time. Assume the journey of an observation in the system described by the multi-state model starts at state $j^*$ with vector of baseline covariates $W$. Let $Z(t)$ be a time-dependent vector of covariates, where
$$ 
Z(t) = (W^T,\widetilde{W}(t))
$$
and $\widetilde{W}(t)$ is a time-dependent vector of covariates known at the entrance to the new state. Let $K_{j^*}$ be the set of possible states that can be reached directly from state $j^*$. Then, the conditional probability of transition $j^* \rightarrow j$, $j \in K_{j^*}$, by time $t$ given $Z(0)=Z$ is given by
$$
\Pr(T \leq t, J_N=j|J_C=j^*,Z(0)=Z) = \int_0^t \lambda_{j^*,j}(u|Z)\exp\left\{-\sum_{k=1}^{|K_{j^*}|} \Lambda_{j^*,k}(u-|Z) \right\} du \, ,
$$
where $u-$ is a time  just prior to $u$, $|K_{j^*}|$ is the cardinality of $K_{j^*}$ and  $\Lambda_{j,k}(t|Z)=\int_0^t \lambda_{j,k}(u|Z) du$ is the cumulative hazard function. In our example, if the first state $j^*=A$, $K_{j^*}=\{B,C,D\}$, and 

$$
\Pr(T \leq t, J_N=j|J_C=A,Z(0)=Z) = 
$$  
$$
\int_0^t \lambda_{A,j}(u|Z)\exp\left\{- \Lambda_{A,B}(u-|Z) - \Lambda_{A,C}(u-|Z) - \Lambda_{A,D}(u-|Z)\right\} du \, ,
$$

The marginal probability of transition $j^* \rightarrow j$ is given by
$$
\Pr(J_N=j|J_C=j^*,Z(0)=Z) = \int_0^\infty \lambda_{j^*,j}(u|Z)\exp\left\{-\sum_{k=1}^{|K_{j^*}|} \Lambda_{j^*,k}(u-|Z) \right\} du \, ,
$$
and the probability of transition time less than $t$ given a transition $j^* \rightarrow j$
$$
\Pr(T \leq t | J_N=j,J_C=j^*, Z(0)=Z)
= \frac{ \int_0^t \lambda_{j^*,j}(u|Z)\exp\left\{-\sum_{k=1}^{|K_{j^*}|} \Lambda_{j^*,k}(u-|Z) \right\} du  }
{ \int_0^\infty \lambda_{j^*,j}(u|Z)\exp\left\{-\sum_{k=1}^{|K_{j^*}|} \Lambda_{j^*,k}(u-|Z) \right\} du  }  \, .
$$
Now assume an observation entered state $j'$ at time $t'>0$ with $Z(t')$. Then, the probability of $j' \rightarrow j$ by time $t$ is given by 
$$
\Pr(T \leq t, J_N=j|J_C=j',Z(t')=Z) = \int_{t'}^t \lambda_{j',j}(u|Z)\exp\left\{-\sum_{k=1}^{|K_{j'}|} \Lambda_{j',k}(u-|Z) \right\} du \, ,
$$
and
$$
\Pr(T \leq t| J_N=j,J_C=j',Z(t')=Z) = 
\frac{ \int_{t'}^t \lambda_{j',j}(u|Z)\exp\left\{-\sum_{k=1}^{|K_{j'}|} \Lambda_{j',k}(u-|Z) \right\} du }
{ \int_{t'}^\infty \lambda_{j',j}(u|Z)\exp\left\{-\sum_{k=1}^{|K_{j'}|} \Lambda_{j',k}(u-|Z) \right\} du } \, .
$$    
All the above, set the main multi-state model components required for prediction, as will be explained in the following sections.

## Estimation

### Cox transition-specific hazard models
The estimation procedure for the hazard functions that define the multi-state model can be chosen by the user. For example, if Cox models are adopted, where each transition $j \rightarrow j'$ consists of transition-specific unspecified baseline hazard function $\lambda_{0j,j'}(\cdot)$ and a 
transition-specific vector of regression coefficients $\beta_{j,j'}$, i.e.,
$$
\lambda_{j,j'}(t|Z) = \lambda_{0j,j'}(t) \exp(Z^T \beta_{j,j'}) \, ,
$$
the estimation procedure is straightforward. Specifically, under transition-specific semi-parametric Cox models, we can easily deal with: 
- Right censoring and competing events based on the approach of [@Andersen:1991]. Namely, maximization of the likelihood function in terms of all the involved  Cox models is done by maximizing the likelihood of each transition separately. Thus, we use the standard partial likelihood estimators of $\beta_{j,j'}$ [@Klein:2006] and Breslow estimator of $\Lambda_{0j,j'}(t)=\int_0^t \lambda_{0j,j'}(u)du$ (Breslow, 1972). 
- Left truncation which occurs at each transition that is not the origin state of the subject's path. Bias due to left truncation is eliminated by using the well-known risk-set correction [@Klein:2006]. 
- Recurrent events which occurs when subjects visit the same state multiple times. In such cases, the robust standard errors account for correlated outcomes within a subject [@Andersen:1982]. 	

Based on the estimates of the regression coefficients and the cumulative baseline hazard functions all the distribution functions in the introduction section can be estimated by replacing the integrals with sums over the observed failure time, and any unknown parameter is replaced by its estimator. Specifically, let $\tau_{j^*,j}$ be the largest observed event time of transition $j^* \rightarrow j$. Then, 
\begin{equation}
    \begin{aligned}
\widehat{\Pr} (J_N=j | J_C=j^*,Z(0)=Z) \\
    =   \sum_{t_m \leq \tau_{j^*,j}} \exp\left( \widehat\beta_{j^*,j}^T Z\right) \widehat\lambda_{0j^*,j}(t_m) \exp \left\{-\sum_{k=1}^{|K_{j^*}|} \widehat\Lambda_{0j^*,k}(t_{m-1})\exp\left( \widehat\beta_{j^*,k}^T Z\right) \right\} \, ,  
    \end{aligned}
\end{equation}


\begin{equation}
    \begin{aligned}
\widehat{\Pr} (T\leq t| J_N=j', J_C=j^* , Z(0)=Z)\\
    \hspace{0.5cm} = \frac{\sum_{t_m \leq t} \exp\left( \widehat\beta_{j^*,j'}^T Z\right) \widehat\lambda_{0j^*,j'}(t_m) \exp \left\{-\sum_{k=1}^{|K_{j^*}|} \widehat\Lambda_{0j^*,k}(t_{m-1})\exp\left( \widehat\beta_{j^*k}^T Z\right) \right\} }{ \sum_{t_m \leq \tau_{j^*,j'}} \exp\left( \widehat\beta_{j^*,j'}^T Z\right) \widehat\lambda_{0j^*,j'}(t_m) \exp \left\{-\sum_{k=1}^{K_{j^*}} \widehat\Lambda_{0j^*,k}(t_{m-1})\exp\left( \widehat\beta_{j^*,k}^T Z\right) \right\} } \, , 
    \end{aligned}
\end{equation}
and finally, given a new $\breve{j}$, the estimated probability of staying at state $j'$ less than or equal $t$ time unit is given by
\begin{equation}
    \begin{aligned}
\widehat{\Pr} (T\leq t| J_N=\breve{j}, J_C=j' , Z(t')=Z) \\
    \hspace{0.5cm} = \frac{\sum_{t' < t_m \leq t} \exp\left( \widehat\beta_{j',\breve{j}}^T Z\right) \widehat\lambda_{0j',\breve{j}}(t_m) \exp \left\{-\sum_{k=1}^{|K_{j'}|} \widehat\Lambda_{0j',k}(t_{m-1})\exp\left( \widehat\beta_{j',k}^T Z\right) \right\} }{ \sum_{t' < t_m \leq \tau_{j',\breve{j}}} \exp\left( \widehat\beta_{j',\breve{j}}^T Z\right) \widehat\lambda_{0j',\breve{j}}(t_m) \exp \left\{-\sum_{k=1}^{K_{j'}} \widehat\Lambda_{0j',k}(t_{m-1})\exp\left( \widehat\beta_{j',k}^T Z\right) \right\} } \, .
    \end{aligned}
\end{equation}

### Other transition-specific models
Similarly, the user can define other survival models and estimation procedure, such as accelerated failure time model, random survival forests (ref) etc, for each transition, as explained in section Costume Fitters above.

## Prediction - Monte Carlo Simulation
Based on the multi-state model, we reconstruct the complete distribution of the path for a new observation, given the observed covariates $W$. Based on the reconstructed distribution we estimate
- The probability of visiting each state.
- The total length of stay at each state.
- The total length of stay in the entire system.

The above quantities can be predicted before entering the system and also during the stay at one of the systems' states, while correctly taking into account the accumulated time already spent in the system and $Z(\cdot)$.

We reconstruct the distribution of the path for a new observation by Monte-Carlo simulation. Assume the starting state (provided by the user) is $j^*$. Then, the next state $J_N$ is sampled based on the discrete conditional probabilities
$$
p_{j|j^*,Z}= \frac{\widehat{\Pr} (J_N=j | J_C=j^*, Z(0)=Z) }{\sum_{j'=1}^{K_{j^*}} \widehat{\Pr} (J_N=j' | J_C=j^*, Z(0)=Z)}  \, .
$$
where $j \in K_{j^*}$ and the summation is over the distinct observed event times of transition $j^* \rightarrow j$. Once we sampled the next state, denoted by $j'$, the time to be spent at state $j^*$ is sampled based on 
$$
\widehat{\Pr} (T\leq t| J_N=j', J_C=j^* , Z(0)=Z) \, .
$$
This is done by sampling $U \sim Uniform[0,1]$, equating 
$$
U=\widehat{\Pr} (T\leq t| J_N=j', J_C=j^* , Z(0)=Z)
$$ 
and solving for $t$. Denote the sampled time by $t'$ and update $Z(t')$. In case $j'$, is a terminal state, the sampling path ends here. Otherwise, the current state is updated to $J_C=j'$, and the following state is sampled by $p_{j|j',Z(t')}$, $j=1 \in  K_{j'}$, 
$$
p_{`j|j',Z}= \frac{\sum_{t' < t_m \leq \tau_{j',j}} \exp\left( \widehat\beta_{j',j}^T Z\right) \widehat\lambda_{0j',j}(t_m) \exp \left\{-\sum_{k=1}^{|K_{j'}|} \widehat\Lambda_{0j',k}(t_{m-1})\exp\left( \widehat\beta_{j',k}^T Z\right) \right\} }
    {\sum_{j^{**}=1}^{K_{j'}} \sum_{t' < t_m \leq \tau_{j',j^{**}}} \exp\left( \widehat\beta_{j',j^{**}}^T Z\right) \widehat\lambda_{0j',j^{**}}(t_m) \exp \left\{-\sum_{k=1}^{|K_{j'}|} \widehat\Lambda_{0j',k}(t_{m-1})\exp\left( \widehat\beta_{j',k}^T Z\right) \right\}} \, .`
$$

## Generating Random Multistate Survival Data
`PyMSM` allows the user to predefine a model by providing an input of transition-specific baseline hazards and hazard coefficients $\beta_{j,j'}$, and a time-varying covariates update function if needed. After providing this information, the user can then simulate trajectories, thus creating a new multi-state data-set which may be valuable for a variety of purposes.


$$
\Pr(T \leq t, J_N=j|J_C=j^*,Z(0)=Z) = \int_0^t \lambda_{j^*,j}(u|Z)\exp\left\{-\sum_{k=1}^{|K_{j^*}|} \Lambda_{j^*,k}(u-|Z) \right\}du
$$  
