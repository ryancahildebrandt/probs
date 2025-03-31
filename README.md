# Probabilistic Modeling Quick Reference
## Using pymc, bambi, rstan, and rstanarm

------------------------------------------------------------------------

[![Open in gitpod](https://gitpod.io/button/open-in-gitpod.svg)](https://gitpod.io/#https://github.com/ryancahildebrandt/probs)
[![This project contains 0% LLM-generated content](https://brainmade.org/88x31-dark.png)](https://brainmade.org/)

## *Purpose*

This project serves as a consolidated quick reference for sampling common probabilistic models in python and R. Many bayesian modeling techniques can be difficult to set up if you're not doing them on a regular basis, and my hope is that having a couple templates set up in an approachable and ready to run format will make them more approachable in my day to day work

------------------------------------------------------------------------

## Dataset & Libraries

The dataset used for the current project was pulled from the following:

-   [datasets R library](https://rdrr.io/r/datasets/datasets-package.html)
-   [scikit-learn python package](https://scikit-learn.org/stable/datasets/toy_dataset.html)

And the examples provided here were adapted from the documentation of the following modeling frameworks:

-   [pymc](https://www.pymc.io/welcome.html)
-   [bambi](https://bambinos.github.io/bambi/)
-   [rstan](https://mc-stan.org/users/interfaces/rstan)
-   [rstanarm](https://mc-stan.org/rstanarm/)

------------------------------------------------------------------------

## Outputs

-   [RMarkdown notebook](./probs.rmd) with rstan and rstanarm implementations
-   [Jupyter notebook](./probs.ipynb) and [python script](./probs.py) with pymc and bambi implementations
