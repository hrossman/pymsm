---
title: 'PyMSM: Python package for Competing Risks and Multi-state models for Survival Data'
tags:
  - Python
  - multistate models
  - survival analysis
  - competing risks
authors:
  - name: Hagai Rossman^[Co-first author] # note this makes a footnote saying 'Co-first author'
    orcid: 0000-0000-0000-0000
    affiliation: "1, 2" # (Multiple affiliations must be quoted)
  - name: Ayya Keshet^[Co-first author] # note this makes a footnote saying 'Co-first author'
    orcid: 0000-0000-0000-0000
    affiliation: "1, 2" # (Multiple affiliations must be quoted)
  - name: Maka Gorfine^[Corresponding author]
    affiliation: 3
affiliations:
 - name: Weizmann XXX
   index: 1
 - name: Weizmann XXX
   index: 2
 - name: TAU XXX
   index: 3
date: 10 April 2022
bibliography: paper.bib
---

# Summary

Multi-state data are common, and could be used to describe trajectories in diverse applications; from describing a patient's health progress through disease states, to a path of a taxi driver when picking drivers. When faced with such data, a researcher might seek to characterize the possible transitions between states, their occurrence probabilities, or to predict a trajectory for a future sample. 

Fitting a multi-state model, we can learn the hazard for each specific transition, which would later be used to predict future paths. Predicting paths could be used at a single sample level, for example predict how many pick-ups the taxi driver will do, given a time of day, or at the population level, for example predicting how many patients which arrive at the emergency-room will need to be admitted, given covariates of the patients.

# Statement of need

`PyMSM` is a Python package for fitting multi-state models, with a simple API which allows user-defined model, predictions at a single or population sample level, statistical summaries and figures. As well, to the authors best knowledge, this is the first open-source multi-state model tool. The package is designed to allow usage for researchers and non-expert users. In addition to fitting a multi-state model for a given data - {PyMSM} allows the user to simulate trajectories, thus creating a multi-state data-set, from a predefined given model.

The R version of this code was previously used in [@Roimi:2021] yet this is the first Python version to be released as an open-source package.

# The PyMSM package

A brief overview of the package functionality is described below. Detailed explanations of functions, along with usage examples are available in the package documentation.


## Model fitting
Fitting a multi-state model to a data-set requires only a few simple steps;
- Preparaing a data-set in one of two formats
- Defining a function for updating time-dependent covariates
- Define covariate columns
- Define terminal states
- Define a minimum number of data transitions needed to fit a transition  
Once all the above was done, the user can fit a multi-state model to his data-set, and use it for downstream analyses.

## Path sampling
Using the previously fitted multi-state model, the user can sample paths using the Monte Carlo simulations. Providing covariates, initial state and time - next states are sequentially sampled via the model parameters, and the process concludes when the patient arrives at a terminal state or the number of transitions exceeds the specified maximum. With the sampled paths, the user can explore statistics such as the probability of being in any of the states or the time spent in each state.

## Costume fitters
`PyMSM` allows configuration of custom event-specific-fitters.
EventSpecificFitter class is an abstract class which defines the API which needs to be implemented by the user.

Some custom fitters are available off-the-shelf such as Survival trees (Ishwaran 2008).

## Simulating Multi-state Survival Data
Using a pre-loaded or a pre-defined model, {PyMSM} provides an API to generate simulated data of random trajectories using the model parameters. Creating a simulated multi-state paths data-set could serve as a useful research tool in cases where data sharing is limited due to privacy limitations, or as a generation tool for any downstream task which requires individual trajectories.

# Models and Methods
In this section we give an overview of the models and methods underlying the statistics and computations performed in `PyMSM`.

# Introduction
The description of the content of \texttt{PyMSM} would be easier to digest under a certain setting.  Thus, to set the stage, we adopt the multi-state model of [@Roimi:2021]. Specifically, assume a multi-state model consists of four states $A,B,C,D$ and six possible transitions:
$$
A \rightarrow B \,\,\,\,\,\,       A \rightarrow C   \,\,\,\,\,\,     A \rightarrow D   \,\,\,\,\,\,    B \rightarrow A \,\,\,\,\,\,    B \rightarrow D \,\,\,\,\,\,   C \rightarrow A \, .
$$
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

# Estimation

## Cox transition-specific hazard models
The estimation procedure for the hazard functions that define the multi-state model can be chosen by the user. For example, if Cox models are  adopted, where each transition $j \rightarrow j'$ consists of transition-specific unspecified baseline hazard function $\lambda_{0j,j'}(\cdot)$ and a 
transition-specific vector of regression coefficients $\beta_{j,j'}$, i.e.,
$$
\lambda_{j,j'}(t|Z) = \lambda_{0j,j'}(t) \exp(Z^T \beta_{j,j'}) \, ,
$$
the estimation procedure is straightforward. Specifically, under transition-specific semi-parametric Cox models, we can easily deal with: 
- Right censoring and competing events based on the approach of \citet{andersen1991non}. Namely, maximization of the likelihood function in terms of all the involved  Cox models is done by maximizing the likelihood of each transition separately. Thus, we use the standard partial likelihood estimators of $\beta_{j,j'}$ \citep{klein2006survival} and Breslow estimator of $\Lambda_{0j,j'}(t)=\int_0^t \lambda_{0j,j'}(u)du$ (Breslow, 1972). 
- Left truncation which occurs at each transition that is not the origin state of the subject's path. Bias due to left truncation is eliminated by using the well-known risk-set correction \citep{klein2006survival}. 
- Recurrent events which occurs when subjects visit the same state multiple times. In such cases, the robust standard errors account for correlated outcomes within a subject \citep{andersen1982cox}. 	


Based on the estimates of the regression coefficients and the cumulative baseline hazard functions all the distribution functions of Section \ref{Sec1} can be estimated by replacing the integrals with sums over the observed failure time, and any unknown parameter is replaced by its estimator. Specifically, let $\tau_{j^*,j}$ be the largest observed event time of transition $j^* \rightarrow j$. Then, 
$$
\widehat{\Pr} (J_N=j | J_C=j^*,Z(0)=Z) \\
=   \sum_{t_m \leq \tau_{j^*,j}} \exp\left( \widehat\beta_{j^*,j}^T Z\right) \widehat\lambda_{0j^*,j}(t_m) \exp \left\{-\sum_{k=1}^{|K_{j^*}|} \widehat\Lambda_{0j^*,k}(t_{m-1})\exp\left( \widehat\beta_{j^*,k}^T Z\right) \right\} \, ,  
$$

$$
\widehat{\Pr} (T\leq t| J_N=j', J_C=j^* , Z(0)=Z)\\
\hspace{0.5cm} = \frac{\sum_{t_m \leq t} \exp\left( \widehat\beta_{j^*,j'}^T Z\right) \widehat\lambda_{0j^*,j'}(t_m) \exp \left\{-\sum_{k=1}^{|K_{j^*}|} \widehat\Lambda_{0j^*,k}(t_{m-1})\exp\left( \widehat\beta_{j^*k}^T Z\right) \right\} }{ \sum_{t_m \leq \tau_{j^*,j'}} \exp\left( \widehat\beta_{j^*,j'}^T Z\right) \widehat\lambda_{0j^*,j'}(t_m) \exp \left\{-\sum_{k=1}^{K_{j^*}} \widehat\Lambda_{0j^*,k}(t_{m-1})\exp\left( \widehat\beta_{j^*,k}^T Z\right) \right\} } \, , 
$$
and finally, given a new $\breve{j}$, the estimated probability of staying at state $j'$ less than or equal $t$ time unit is given by
$$
\widehat{\Pr} (T\leq t| J_N=\breve{j}, J_C=j' , Z(t')=Z) \\
\hspace{0.5cm} = \frac{\sum_{t' < t_m \leq t} \exp\left( \widehat\beta_{j',\breve{j}}^T Z\right) \widehat\lambda_{0j',\breve{j}}(t_m) \exp \left\{-\sum_{k=1}^{|K_{j'}|} \widehat\Lambda_{0j',k}(t_{m-1})\exp\left( \widehat\beta_{j',k}^T Z\right) \right\} }{ \sum_{t' < t_m \leq \tau_{j',\breve{j}}} \exp\left( \widehat\beta_{j',\breve{j}}^T Z\right) \widehat\lambda_{0j',\breve{j}}(t_m) \exp \left\{-\sum_{k=1}^{K_{j'}} \widehat\Lambda_{0j',k}(t_{m-1})\exp\left( \widehat\beta_{j',k}^T Z\right) \right\} } \, .
$$


# Acknowledgements

We acknowledge contributions from TBD.

# References