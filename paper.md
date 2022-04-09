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
    affiliation: 2
  - name: Maka Gorfine^[Corresponding author]
    affiliation: 3
affiliations:
 - name: Lyman Spitzer, Jr. Fellow, Princeton University, USA
   index: 1
 - name: Institution Name, Country
   index: 2
 - name: Independent Researcher, Country
   index: 3
date: 13 August 2017
bibliography: paper.bib

# Optional fields if submitting to a AAS journal too, see this blog post:
# https://blog.joss.theoj.org/2018/12/a-new-collaboration-with-aas-publishing
aas-doi: 10.3847/xxxxx <- update this with the DOI from AAS once you know it.
aas-journal: Astrophysical Journal <- The name of the AAS journal.
---

# Summary

Multi-state data are common, and could be used to describe trajectories in diverse applications; from describing a patient's health progress through disease states, to a path of a taxi driver when picking drivers. When faced with such data, a researcher might seek to characterize the possible transitions between states, their occurrence probabilities, or to predict a trajectory for a future sample. 

Fitting a multi-state model, we can learn the hazard for each specific transition, which would later be used to predict future paths. Predicting paths could be used at a single sample level, for example predict how many pick-ups the taxi driver will do, given a time of day, or at the population level, for example predicting how many patients which arrive at the emergency-room will need to be admitted, given covariates of the patients.

# Statement of need

`PyMSM` is a Python package for fitting multi-state models, with a simple API which allows user-defined model, predictions at a single or population sample level, statistical summaries and figures. As well, to the authors best knowledge, this is the first open-source multi-state model tool. The package is designed to allow usage for researchers and non-expert users. In addition to fitting a multi-state model for a given data - {PyMSM} allows the user to simulate trajectories, thus creating a multi-state data-set, from a predefined given model.

The R version of this code was previously used in Roimi et al. (2021), yet this is the first Python version to be released as an open-source package.

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

# Acknowledgements

We acknowledge contributions from Brigitta Sipocz, Syrtis Major, and Semyeong
Oh, and support from Kathryn Johnston during the genesis of this project.

# References