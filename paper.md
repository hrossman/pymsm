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


# Acknowledgements

We acknowledge contributions from TBD.

# References